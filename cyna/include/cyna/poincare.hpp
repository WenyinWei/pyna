#pragma once
// poincare.hpp - self-contained Poincaré field-line tracer
// Dependencies: C++17 stdlib + BS_thread_pool.hpp only
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <vector>
#include <algorithm>
#include <thread>
#include <limits>
#include "BS_thread_pool.hpp"

namespace cyna {

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static inline double mod2pi(double x) {
    x = std::fmod(x, 2.0 * M_PI);
    if (x < 0.0) x += 2.0 * M_PI;
    return x;
}

// Ray-casting point-in-polygon (2D, R-Z plane)
static inline bool point_in_wall(double R, double Z,
                                  const double* wR, const double* wZ, int n) {
    bool inside = false;
    for (int i = 0, j = n - 1; i < n; j = i++) {
        if (((wZ[i] > Z) != (wZ[j] > Z)) &&
            (R < (wR[j] - wR[i]) * (Z - wZ[i]) / (wZ[j] - wZ[i]) + wR[i]))
            inside = !inside;
    }
    return inside;
}

// Nearest toroidal wall slice index with periodic distance in phi
static inline int nearest_phi_idx(double phi,
                                  const double* phi_centers,
                                  int n_phi_wall) {
    double phi_mod = mod2pi(phi);
    double best_d = 1e300;
    int best_i = 0;
    for (int i = 0; i < n_phi_wall; ++i) {
        double d = std::abs(phi_centers[i] - phi_mod);
        d = std::min(d, 2.0 * M_PI - d);
        if (d < best_d) {
            best_d = d;
            best_i = i;
        }
    }
    return best_i;
}

// Toroidally varying wall: use the nearest phi slice, matching topoquest.wall.WallGeometry
static inline bool point_in_toroidal_wall(double R, double Z, double phi,
                                          const double* phi_centers,
                                          const double* wall_R,
                                          const double* wall_Z,
                                          int n_phi_wall,
                                          int n_theta_wall) {
    int idx = nearest_phi_idx(phi, phi_centers, n_phi_wall);
    const double* wR = wall_R + idx * n_theta_wall;
    const double* wZ = wall_Z + idx * n_theta_wall;
    return point_in_wall(R, Z, wR, wZ, n_theta_wall);
}

// ---------------------------------------------------------------------------
// Trilinear interpolation on regular 3D grid [iR][iZ][iPhi]
//
// Phi convention - matches topoquest's scipy setup exactly:
//   Phi_grid is Phi_ext = np.append(linspace(0, 2pi, N, endpoint=False), 2pi)
//   so it has length N+1, Phi_grid[0]=0, Phi_grid[N]=2pi,
//   and data[:,:,N] is a copy of data[:,:,0]  (the _ext() wrap).
//
// We never call mod2pi on phi before the binary search: the binary search
// works directly on the extended grid [0, 2pi], and the caller wraps phi
// into [0, 2pi) before calling.  The last interval [Phi_grid[N-1], 2pi]
// is handled naturally because Phi_grid[N]=2pi is in the grid.
//
// Out-of-bounds R or Z → NaN  (matches scipy fill_value=np.nan)
// ---------------------------------------------------------------------------
inline double interp3d(
    const double* data,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,   // nPhi = N_phi_original + 1  (extended)
    double R, double Z, double Phi)
{
    if (!std::isfinite(R) || !std::isfinite(Z) || !std::isfinite(Phi))
        return std::numeric_limits<double>::quiet_NaN();
    if (R < R_grid[0] || R > R_grid[nR - 1] ||
        Z < Z_grid[0] || Z > Z_grid[nZ - 1])
        return std::numeric_limits<double>::quiet_NaN();

    // Wrap phi into [0, 2pi) then handle the seam at 2pi
    Phi = mod2pi(Phi);  // now in [0, 2pi)

    // ── find iR (uniform grid assumed → O(1)) ──────────────────────────
    double tR_raw = (R - R_grid[0]) / (R_grid[nR-1] - R_grid[0]) * (nR - 1);
    int iR = (int)tR_raw;
    if (iR < 0)       iR = 0;
    if (iR >= nR - 1) iR = nR - 2;
    double tR = tR_raw - iR;

    // ── find iZ (uniform grid assumed → O(1)) ──────────────────────────
    double tZ_raw = (Z - Z_grid[0]) / (Z_grid[nZ-1] - Z_grid[0]) * (nZ - 1);
    int iZ = (int)tZ_raw;
    if (iZ < 0)       iZ = 0;
    if (iZ >= nZ - 1) iZ = nZ - 2;
    double tZ = tZ_raw - iZ;

    // ── find iPhi via binary search on extended grid ────────────────────
    // Phi_grid has nPhi points: [0, ..., phi_{N-1}, 2pi]
    // Valid cell index range: [0, nPhi-2]
    int iPhi;
    {
        int lo = 0, hi = nPhi - 2;     // nPhi-2 is the last valid left index
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (Phi >= Phi_grid[mid + 1]) lo = mid + 1;
            else                          hi = mid;
        }
        iPhi = lo;
    }
    // iPhi1 is the right cell index in the extended grid (never needs %wrap
    // because the last valid iPhi = nPhi-2 and iPhi1 = nPhi-1 = the 2pi copy)
    int iPhi1 = iPhi + 1;   // always valid: iPhi1 <= nPhi-1

    double phiLo = Phi_grid[iPhi];
    double phiHi = Phi_grid[iPhi1];
    double tP = (phiHi > phiLo) ? (Phi - phiLo) / (phiHi - phiLo) : 0.0;
    // Clamp numerical noise
    if (tP < 0.0) tP = 0.0;
    if (tP > 1.0) tP = 1.0;

    // ── trilinear interpolation ─────────────────────────────────────────
    // data layout: [iR][iZ][iPhi_ext],  stride = nZ * nPhi
    auto val = [&](int r, int z, int p) -> double {
        return data[r * nZ * nPhi + z * nPhi + p];
    };

    double c000 = val(iR,   iZ,   iPhi);
    double c001 = val(iR,   iZ,   iPhi1);
    double c010 = val(iR,   iZ+1, iPhi);
    double c011 = val(iR,   iZ+1, iPhi1);
    double c100 = val(iR+1, iZ,   iPhi);
    double c101 = val(iR+1, iZ,   iPhi1);
    double c110 = val(iR+1, iZ+1, iPhi);
    double c111 = val(iR+1, iZ+1, iPhi1);

    // Propagate NaN from field (matches scipy's nan fill behaviour)
    if (!std::isfinite(c000) || !std::isfinite(c001) ||
        !std::isfinite(c010) || !std::isfinite(c011) ||
        !std::isfinite(c100) || !std::isfinite(c101) ||
        !std::isfinite(c110) || !std::isfinite(c111))
        return std::numeric_limits<double>::quiet_NaN();

    return
        c000*(1-tR)*(1-tZ)*(1-tP) + c001*(1-tR)*(1-tZ)*tP +
        c010*(1-tR)*   tZ *(1-tP) + c011*(1-tR)*   tZ *tP +
        c100*   tR *(1-tZ)*(1-tP) + c101*   tR *(1-tZ)*tP +
        c110*   tR *   tZ *(1-tP) + c111*   tR *   tZ *tP;
}


// ---------------------------------------------------------------------------
// Trilinear interpolation + spatial gradients for DX_pol variational equations
// ---------------------------------------------------------------------------
static inline bool interp3d_grad(
    double& val, double& dR, double& dZ,
    const double* data,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    double R, double Z, double Phi)
{
    if (!std::isfinite(R) || !std::isfinite(Z) || !std::isfinite(Phi))
        return false;
    if (R < R_grid[0] || R > R_grid[nR - 1] ||
        Z < Z_grid[0] || Z > Z_grid[nZ - 1])
        return false;
    Phi = mod2pi(Phi);
    double dRg = (R_grid[nR-1] - R_grid[0]) / (nR - 1);
    double dZg = (Z_grid[nZ-1] - Z_grid[0]) / (nZ - 1);
    double inv_dR = 1.0 / dRg, inv_dZ = 1.0 / dZg;

    double tR_raw = (R - R_grid[0]) / (R_grid[nR-1] - R_grid[0]) * (nR - 1);
    int iR = (int)tR_raw; if (iR < 0) iR = 0; if (iR >= nR-1) iR = nR-2;
    double tR = tR_raw - iR;

    double tZ_raw = (Z - Z_grid[0]) / (Z_grid[nZ-1] - Z_grid[0]) * (nZ - 1);
    int iZ = (int)tZ_raw; if (iZ < 0) iZ = 0; if (iZ >= nZ-1) iZ = nZ-2;
    double tZ = tZ_raw - iZ;

    int iPhi; { int lo=0,hi=nPhi-2; while(lo<hi){int mid=(lo+hi)/2; if(Phi>=Phi_grid[mid+1])lo=mid+1;else hi=mid;} iPhi=lo; }
    int iPhi1 = iPhi + 1;
    double phiLo=Phi_grid[iPhi], phiHi=Phi_grid[iPhi1];
    double tP = (phiHi > phiLo) ? (Phi - phiLo) / (phiHi - phiLo) : 0.0;
    if (tP < 0.0) tP = 0.0; if (tP > 1.0) tP = 1.0;

    auto v = [&](int r,int z,int p){return data[r*nZ*nPhi+z*nPhi+p];};
    double c000=v(iR,iZ,iPhi),c001=v(iR,iZ,iPhi1),c010=v(iR,iZ+1,iPhi),c011=v(iR,iZ+1,iPhi1);
    double c100=v(iR+1,iZ,iPhi),c101=v(iR+1,iZ,iPhi1),c110=v(iR+1,iZ+1,iPhi),c111=v(iR+1,iZ+1,iPhi1);
    if(!std::isfinite(c000)||!std::isfinite(c001)||!std::isfinite(c010)||!std::isfinite(c011)||
       !std::isfinite(c100)||!std::isfinite(c101)||!std::isfinite(c110)||!std::isfinite(c111))
        return false;

    double oR=1.0-tR, oZ=1.0-tZ, oP=1.0-tP;
    val = c000*oR*oZ*oP + c001*oR*oZ*tP + c010*oR*tZ*oP + c011*oR*tZ*tP
        + c100*tR*oZ*oP + c101*tR*oZ*tP + c110*tR*tZ*oP + c111*tR*tZ*tP;
    double dtR = -c000*oZ*oP - c001*oZ*tP - c010*tZ*oP - c011*tZ*tP
                + c100*oZ*oP + c101*oZ*tP + c110*tZ*oP + c111*tZ*tP;
    dR = dtR * inv_dR;
    double dtZ = -c000*oR*oP - c001*oR*tP + c010*oR*oP + c011*oR*tP
                - c100*tR*oP - c101*tR*tP + c110*tR*oP + c111*tR*tP;
    dZ = dtZ * inv_dZ;
    return true;
}


// ---------------------------------------------------------------------------
// RK4 step: advance (R,Z,Phi) by dPhi
// ---------------------------------------------------------------------------
static inline void rk4_step(
    double& R, double& Z, double phi,
    double dPhi,
    const double* BR, const double* BPhi, const double* BZ,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi)
{
    auto dRdphi = [&](double r, double z, double p) {
        double bp = interp3d(BPhi, R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, r, z, p);
        if (!std::isfinite(bp) || std::abs(bp) <= 1e-12) return 0.0;
        double br = interp3d(BR, R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, r, z, p);
        if (!std::isfinite(br)) return 0.0;
        return r * br / bp;
    };
    auto dZdphi = [&](double r, double z, double p) {
        double bp = interp3d(BPhi, R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, r, z, p);
        if (!std::isfinite(bp) || std::abs(bp) <= 1e-12) return 0.0;
        double bz = interp3d(BZ, R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, r, z, p);
        if (!std::isfinite(bz)) return 0.0;
        return r * bz / bp;
    };

    double k1R = dRdphi(R,                  Z,                  phi);
    double k1Z = dZdphi(R,                  Z,                  phi);
    double k2R = dRdphi(R + 0.5*dPhi*k1R,  Z + 0.5*dPhi*k1Z,  phi + 0.5*dPhi);
    double k2Z = dZdphi(R + 0.5*dPhi*k1R,  Z + 0.5*dPhi*k1Z,  phi + 0.5*dPhi);
    double k3R = dRdphi(R + 0.5*dPhi*k2R,  Z + 0.5*dPhi*k2Z,  phi + 0.5*dPhi);
    double k3Z = dZdphi(R + 0.5*dPhi*k2R,  Z + 0.5*dPhi*k2Z,  phi + 0.5*dPhi);
    double k4R = dRdphi(R + dPhi*k3R,       Z + dPhi*k3Z,       phi + dPhi);
    double k4Z = dZdphi(R + dPhi*k3R,       Z + dPhi*k3Z,       phi + dPhi);

    R += dPhi / 6.0 * (k1R + 2*k2R + 2*k3R + k4R);
    Z += dPhi / 6.0 * (k1Z + 2*k2Z + 2*k3Z + k4Z);
}


// ---------------------------------------------------------------------------
// RK4 step with DX_pol evolution (variational equations)
// ---------------------------------------------------------------------------
// Advances the field-line trajectory (R,Z) AND the 2×2 poloidal Jacobian
// DX_pol(φ_s, φ_e) = ∂(R(φ_e),Z(φ_e)) / ∂(R(φ_s),Z(φ_s)).
//
// Variational equation:  d(DX_pol)/dphi_e = A(phi_e) * DX_pol
// where A = d(R*B_R/B_phi, R*B_Z/B_phi)/d(R,Z) is the analytic Jacobian
// of the field-line ODE right-hand side, evaluated on the trajectory.
//
// Initial DX_pol should be identity; after integrating over m turns
// (one full island-chain period), DX_pol = DPm(φ_s), the monodromy matrix.
//
// DX_pol stored row-major: D00,D01,D10,D11.
static inline void rk4_step_DX_pol(
    double& R, double& Z,
    double& D00, double& D01, double& D10, double& D11,
    double phi, double dPhi,
    const double* BR, const double* BPhi, const double* BZ,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi)
{
    // Analytic local gradient of the field-line ODE: A = ∂f/∂(R,Z)
    // where f(R,Z) = [R*B_R/B_phi, R*B_Z/B_phi]
    // (Not a Jacobian in the DX_pol/DPm sense - those are derivatives
    // w.r.t. initial conditions; A is a local spatial gradient.)
    auto eval_local_grad = [&](double r, double z, double p,
                         double& fR, double& fZ,
                         double& A00, double& A01, double& A10, double& A11) -> bool {
        double BRv, dBR_dR, dBR_dZ;
        double BPhiv, dBPhi_dR, dBPhi_dZ;
        double BZv, dBZ_dR, dBZ_dZ;
        if (!interp3d_grad(BRv, dBR_dR, dBR_dZ, BR, R_grid,nR,Z_grid,nZ,Phi_grid,nPhi, r,z,p)) return false;
        if (!interp3d_grad(BPhiv, dBPhi_dR, dBPhi_dZ, BPhi, R_grid,nR,Z_grid,nZ,Phi_grid,nPhi, r,z,p)) return false;
        if (!interp3d_grad(BZv, dBZ_dR, dBZ_dZ, BZ, R_grid,nR,Z_grid,nZ,Phi_grid,nPhi, r,z,p)) return false;
        if (std::abs(BPhiv) <= 1e-12) { fR = fZ = 0.0; A00=A01=A10=A11=0.0; return true; }
        double invBp = 1.0 / BPhiv, invBp2 = invBp / BPhiv;
        fR = r * BRv * invBp;
        fZ = r * BZv * invBp;
        A00 = BRv*invBp + r*(dBR_dR*invBp - BRv*invBp2*dBPhi_dR);
        A01 = r*(dBR_dZ*invBp - BRv*invBp2*dBPhi_dZ);
        A10 = BZv*invBp + r*(dBZ_dR*invBp - BZv*invBp2*dBPhi_dR);
        A11 = r*(dBZ_dZ*invBp - BZv*invBp2*dBPhi_dZ);
        return true;
    };
    double fR1,fZ1,A00_1,A01_1,A10_1,A11_1;
    if(!eval_local_grad(R,Z,phi,fR1,fZ1,A00_1,A01_1,A10_1,A11_1))return;
    double k1_D00=A00_1*D00+A01_1*D10, k1_D01=A00_1*D01+A01_1*D11;
    double k1_D10=A10_1*D00+A11_1*D10, k1_D11=A10_1*D01+A11_1*D11;
    double hh=0.5*dPhi;
    double R2=R+hh*fR1,Z2=Z+hh*fZ1, D00_2=D00+hh*k1_D00,D01_2=D01+hh*k1_D01,D10_2=D10+hh*k1_D10,D11_2=D11+hh*k1_D11;
    double fR2,fZ2,A00_2,A01_2,A10_2,A11_2;
    if(!eval_local_grad(R2,Z2,phi+hh,fR2,fZ2,A00_2,A01_2,A10_2,A11_2))return;
    double k2_D00=A00_2*D00_2+A01_2*D10_2, k2_D01=A00_2*D01_2+A01_2*D11_2;
    double k2_D10=A10_2*D00_2+A11_2*D10_2, k2_D11=A10_2*D01_2+A11_2*D11_2;
    double R3=R+hh*fR2,Z3=Z+hh*fZ2, D00_3=D00+hh*k2_D00,D01_3=D01+hh*k2_D01,D10_3=D10+hh*k2_D10,D11_3=D11+hh*k2_D11;
    double fR3,fZ3,A00_3,A01_3,A10_3,A11_3;
    if(!eval_local_grad(R3,Z3,phi+hh,fR3,fZ3,A00_3,A01_3,A10_3,A11_3))return;
    double k3_D00=A00_3*D00_3+A01_3*D10_3, k3_D01=A00_3*D01_3+A01_3*D11_3;
    double k3_D10=A10_3*D00_3+A11_3*D10_3, k3_D11=A10_3*D01_3+A11_3*D11_3;
    double R4=R+dPhi*fR3,Z4=Z+dPhi*fZ3, D00_4=D00+dPhi*k3_D00,D01_4=D01+dPhi*k3_D01,D10_4=D10+dPhi*k3_D10,D11_4=D11+dPhi*k3_D11;
    double fR4,fZ4,A00_4,A01_4,A10_4,A11_4;
    if(!eval_local_grad(R4,Z4,phi+dPhi,fR4,fZ4,A00_4,A01_4,A10_4,A11_4))return;
    double k4_D00=A00_4*D00_4+A01_4*D10_4, k4_D01=A00_4*D01_4+A01_4*D11_4;
    double k4_D10=A10_4*D00_4+A11_4*D10_4, k4_D11=A10_4*D01_4+A11_4*D11_4;
    double s6=dPhi/6.0;
    R += s6*(fR1+2*fR2+2*fR3+fR4);
    Z += s6*(fZ1+2*fZ2+2*fZ3+fZ4);
    D00 += s6*(k1_D00+2*k2_D00+2*k3_D00+k4_D00);
    D01 += s6*(k1_D01+2*k2_D01+2*k3_D01+k4_D01);
    D10 += s6*(k1_D10+2*k2_D10+2*k3_D10+k4_D10);
    D11 += s6*(k1_D11+2*k2_D11+2*k3_D11+k4_D11);
}

// ---------------------------------------------------------------------------
// Single-seed Poincaré trace  (corrected)
// ---------------------------------------------------------------------------
// Output layout:
//   poi_counts[seed_idx * n_sec + s]               = number of crossings at section s
//   poi_R_flat[seed_idx * N_turns * n_sec + s*N_turns + cnt] = R at crossing cnt of section s
//   (same for poi_Z_flat)
// ---------------------------------------------------------------------------
void trace_one_seed(
    int seed_idx, int N_seeds,
    double R0, double Z0, double phi_start,
    const double* phi_sections, int n_sec,
    int N_turns, double DPhi,
    const double* BR, const double* BPhi, const double* BZ,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    const double* wall_R, const double* wall_Z, int n_wall,
    int* poi_counts,
    double* poi_R_flat,
    double* poi_Z_flat)
{
    double R = R0, Z = Z0;
    double phi = phi_start;         // unwrapped (grows monotonically)
    double phi_end = phi_start + N_turns * 2.0 * M_PI;

    int cnt_base = seed_idx * n_sec;
    int poi_base = seed_idx * N_turns * n_sec;

    while (phi < phi_end - 1e-12) {
        double step = std::min(DPhi, phi_end - phi);

        double R_old = R, Z_old = Z, phi_old = phi;

        // RK4 advance
        rk4_step(R, Z, phi, step,
                 BR, BPhi, BZ,
                 R_grid, nR, Z_grid, nZ, Phi_grid, nPhi);
        phi += step;

        // Wall check after step
        if (n_wall > 0 && !point_in_wall(R, Z, wall_R, wall_Z, n_wall))
            break;

        // Detect Poincaré crossings in [phi_old, phi)
        // A crossing of section phi_sec occurs when phi passes
        //   phi_sec + k*2pi  for some integer k
        for (int s = 0; s < n_sec; ++s) {
            int cnt = poi_counts[cnt_base + s];
            if (cnt >= N_turns) continue;

            double sec = phi_sections[s]; // in [0, 2pi)
            // Find the crossing target phi_cross = sec + k*2pi that falls in (phi_old, phi]
            // k = ceil((phi_old - sec) / 2pi)
            double k_raw = (phi_old - sec) / (2.0 * M_PI);
            int k = (int)std::ceil(k_raw);
            // guard: k_raw might be exactly an integer (on section), skip those
            if (k_raw == (double)k) k++;
            double phi_cross = sec + k * 2.0 * M_PI;

            if (phi_cross > phi_old && phi_cross <= phi) {
                // Linear interpolation for R, Z at phi_cross
                double t = (phi_cross - phi_old) / (phi - phi_old);
                double R_c = R_old + t * (R - R_old);
                double Z_c = Z_old + t * (Z - Z_old);
                poi_R_flat[poi_base + s * N_turns + cnt] = R_c;
                poi_Z_flat[poi_base + s * N_turns + cnt] = Z_c;
                poi_counts[cnt_base + s]++;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Batch Poincaré trace (single section version)
// ---------------------------------------------------------------------------
void trace_poincare_batch(
    const double* R_seeds, const double* Z_seeds, int N_seeds,
    double phi_section,
    int N_turns, double DPhi,
    const double* BR, const double* BPhi, const double* BZ,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    const double* wall_R, const double* wall_Z, int n_wall,
    int n_threads,
    int* poi_counts,
    double* poi_R_flat,
    double* poi_Z_flat)
{
    if (n_threads <= 0)
        n_threads = (int)std::thread::hardware_concurrency();

    double phi_sec[1] = { mod2pi(phi_section) };

    BS::thread_pool pool((unsigned int)n_threads);
    pool.parallelize_loop(0, N_seeds, [&](int i_start, int i_end) {
        for (int i = i_start; i < i_end; ++i) {
            trace_one_seed(
                i, N_seeds,
                R_seeds[i], Z_seeds[i], phi_sec[0],
                phi_sec, 1,
                N_turns, DPhi,
                BR, BPhi, BZ,
                R_grid, nR, Z_grid, nZ, Phi_grid, nPhi,
                wall_R, wall_Z, n_wall,
                poi_counts, poi_R_flat, poi_Z_flat);
        }
    }).wait();
}

// ---------------------------------------------------------------------------
// Toroidally varying wall version: matches topoquest.wall.WallGeometry.is_inside
// ---------------------------------------------------------------------------
void trace_poincare_batch_twall(
    const double* R_seeds, const double* Z_seeds, int N_seeds,
    double phi_section,
    int N_turns, double DPhi,
    const double* BR, const double* BPhi, const double* BZ,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    const double* wall_phi_centers, int n_phi_wall,
    const double* wall_R, const double* wall_Z, int n_theta_wall,
    int n_threads,
    int* poi_counts,
    double* poi_R_flat,
    double* poi_Z_flat)
{
    if (n_threads <= 0)
        n_threads = (int)std::thread::hardware_concurrency();

    double phi_sec[1] = { mod2pi(phi_section) };

    BS::thread_pool pool((unsigned int)n_threads);
    pool.parallelize_loop(0, N_seeds, [&](int i_start, int i_end) {
        for (int i = i_start; i < i_end; ++i) {
            double R = R_seeds[i], Z = Z_seeds[i];
            double phi = phi_sec[0];
            int cnt_base = i;
            int poi_base = i * N_turns;
            bool alive = true;

            for (int turn = 0; turn < N_turns && alive; ++turn) {
                double phi_end_turn = phi + 2.0 * M_PI;
                while (phi < phi_end_turn - 1e-12) {
                    double step = std::min(DPhi, phi_end_turn - phi);

                    rk4_step(R, Z, phi, step,
                             BR, BPhi, BZ,
                             R_grid, nR, Z_grid, nZ, Phi_grid, nPhi);
                    phi += step;

                    if (!std::isfinite(R) || !std::isfinite(Z) ||
                        R < R_grid[0] || R > R_grid[nR - 1] ||
                        Z < Z_grid[0] || Z > Z_grid[nZ - 1]) {
                        alive = false;
                        break;
                    }

                    if (!point_in_toroidal_wall(R, Z, phi,
                                                wall_phi_centers,
                                                wall_R, wall_Z,
                                                n_phi_wall, n_theta_wall)) {
                        alive = false;
                        break;
                    }
                }

                if (alive) {
                    int cnt = poi_counts[cnt_base];
                    if (cnt < N_turns) {
                        poi_R_flat[poi_base + cnt] = R;
                        poi_Z_flat[poi_base + cnt] = Z;
                        poi_counts[cnt_base]++;
                    }
                }
            }
        }
    }).wait();
}


// ---------------------------------------------------------------------------
// Connection-length trace with toroidal wall (forward + backward)
// ---------------------------------------------------------------------------
// Output arrays L_fwd, L_bwd: length N_seeds, pre-filled with sentinel value
// (large positive float = did not terminate).
// arc_length_element: dl/dphi = R * |B| / |Bphi|
// ---------------------------------------------------------------------------
void trace_connection_length_twall(
    const double* R_seeds, const double* Z_seeds, int N_seeds,
    double phi_start,
    int max_turns, double DPhi,
    const double* BR, const double* BPhi, const double* BZ,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    const double* wall_phi_centers, int n_phi_wall,
    const double* wall_R, const double* wall_Z, int n_theta_wall,
    int n_threads,
    double* L_fwd,
    double* L_bwd)
{
    if (n_threads <= 0)
        n_threads = (int)std::thread::hardware_concurrency();

    constexpr double SENTINEL = 1e30;
    for (int i = 0; i < N_seeds; ++i) { L_fwd[i] = SENTINEL; L_bwd[i] = SENTINEL; }

    // direction: +1 = forward, -1 = backward
    for (int dir : {+1, -1}) {
        double* L_out = (dir == 1) ? L_fwd : L_bwd;

        BS::thread_pool pool((unsigned int)n_threads);
        pool.parallelize_loop(0, N_seeds, [&](int i_start, int i_end) {
            for (int i = i_start; i < i_end; ++i) {
                double R = R_seeds[i], Z = Z_seeds[i];
                double phi = mod2pi(phi_start);
                double arc = 0.0;
                double phi_total = 0.0;
                double phi_limit = max_turns * 2.0 * M_PI;

                while (phi_total < phi_limit - 1e-12) {
                    double step = std::min(DPhi, phi_limit - phi_total);

                    // Arc-length contribution before step (mid-point approximation)
                    double bp_here = interp3d(BPhi, R_grid, nR, Z_grid, nZ,
                                              Phi_grid, nPhi, R, Z, phi);
                    if (std::isfinite(bp_here) && std::abs(bp_here) > 1e-12) {
                        double br_here = interp3d(BR,  R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, R, Z, phi);
                        double bz_here = interp3d(BZ,  R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, R, Z, phi);
                        double Bmag = std::sqrt(br_here*br_here + bp_here*bp_here + bz_here*bz_here);
                        if (std::isfinite(Bmag))
                            arc += R * Bmag / std::abs(bp_here) * step;
                    }

                    double phi_step_dir = dir * step;
                    rk4_step(R, Z, phi, phi_step_dir,
                             BR, BPhi, BZ,
                             R_grid, nR, Z_grid, nZ, Phi_grid, nPhi);
                    phi = mod2pi(phi + phi_step_dir);
                    phi_total += step;

                    // Terminate if out-of-grid or non-finite
                    if (!std::isfinite(R) || !std::isfinite(Z) ||
                        R < R_grid[0] || R > R_grid[nR - 1] ||
                        Z < Z_grid[0] || Z > Z_grid[nZ - 1]) {
                        L_out[i] = arc;
                        break;
                    }

                    // Terminate if outside toroidal wall
                    if (!point_in_toroidal_wall(R, Z, phi,
                                                wall_phi_centers,
                                                wall_R, wall_Z,
                                                n_phi_wall, n_theta_wall)) {
                        L_out[i] = arc;
                        break;
                    }
                }
            }
        }).wait();
    }
}

// ---------------------------------------------------------------------------
// trace_wall_hits_twall
// Same as trace_connection_length_twall but also records the (R, Z, phi) of
// the termination point for both forward and backward directions, and reports
// which termination condition was triggered.
//
// term_type output (per seed, per direction, packed as fwd then bwd):
//   0 = not terminated (NaN hit coords)
//   1 = wall polygon crossed  (hit coords = bisected wall intersection)
//   2 = field grid exited     (hit coords = grid boundary crossing; wall location uncertain)
//   3 = non-finite field      (hit coords = last finite position)
//
// Outputs (length N_seeds each):
//   L_fwd, L_bwd, R_hit_fwd, Z_hit_fwd, phi_hit_fwd,
//                 R_hit_bwd, Z_hit_bwd, phi_hit_bwd,
//   term_type_fwd, term_type_bwd  (int arrays)
// ---------------------------------------------------------------------------
void trace_wall_hits_twall(
    const double* R_seeds, const double* Z_seeds, int N_seeds,
    double phi_start,
    int max_turns, double DPhi,
    const double* BR, const double* BPhi, const double* BZ,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    const double* wall_phi_centers, int n_phi_wall,
    const double* wall_R, const double* wall_Z, int n_theta_wall,
    int n_threads,
    double* L_fwd,     double* L_bwd,
    double* R_hit_fwd, double* Z_hit_fwd, double* phi_hit_fwd,
    double* R_hit_bwd, double* Z_hit_bwd, double* phi_hit_bwd,
    int*    term_type_fwd, int* term_type_bwd)
{
    if (n_threads <= 0)
        n_threads = (int)std::thread::hardware_concurrency();

    constexpr double NAN_VAL = std::numeric_limits<double>::quiet_NaN();
    constexpr double SENTINEL = 1e30;
    for (int i = 0; i < N_seeds; ++i) {
        L_fwd[i] = SENTINEL; L_bwd[i] = SENTINEL;
        R_hit_fwd[i] = NAN_VAL; Z_hit_fwd[i] = NAN_VAL; phi_hit_fwd[i] = NAN_VAL;
        R_hit_bwd[i] = NAN_VAL; Z_hit_bwd[i] = NAN_VAL; phi_hit_bwd[i] = NAN_VAL;
        term_type_fwd[i] = 0; term_type_bwd[i] = 0;
    }

    for (int dir : {+1, -1}) {
        double* L_out       = (dir == 1) ? L_fwd       : L_bwd;
        double* R_hit_out   = (dir == 1) ? R_hit_fwd   : R_hit_bwd;
        double* Z_hit_out   = (dir == 1) ? Z_hit_fwd   : Z_hit_bwd;
        double* phi_hit_out = (dir == 1) ? phi_hit_fwd : phi_hit_bwd;
        int*    tt_out      = (dir == 1) ? term_type_fwd : term_type_bwd;

        BS::thread_pool pool((unsigned int)n_threads);
        pool.parallelize_loop(0, N_seeds, [&](int i_start, int i_end) {
            for (int i = i_start; i < i_end; ++i) {
                double R = R_seeds[i], Z = Z_seeds[i];
                double phi = mod2pi(phi_start);
                double arc = 0.0;
                double phi_total = 0.0;
                double phi_limit = max_turns * 2.0 * M_PI;

                while (phi_total < phi_limit - 1e-12) {
                    double step = std::min(DPhi, phi_limit - phi_total);

                    // Save pre-step position for bisection
                    double R_prev = R, Z_prev = Z, phi_prev = phi;

                    // Arc-length contribution (mid-point)
                    double bp_here = interp3d(BPhi, R_grid, nR, Z_grid, nZ,
                                              Phi_grid, nPhi, R, Z, phi);
                    if (std::isfinite(bp_here) && std::abs(bp_here) > 1e-12) {
                        double br_here = interp3d(BR,  R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, R, Z, phi);
                        double bz_here = interp3d(BZ,  R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, R, Z, phi);
                        double Bmag = std::sqrt(br_here*br_here + bp_here*bp_here + bz_here*bz_here);
                        if (std::isfinite(Bmag))
                            arc += R * Bmag / std::abs(bp_here) * step;
                    }

                    double phi_step_dir = dir * step;
                    rk4_step(R, Z, phi, phi_step_dir,
                             BR, BPhi, BZ,
                             R_grid, nR, Z_grid, nZ, Phi_grid, nPhi);
                    phi = mod2pi(phi + phi_step_dir);
                    phi_total += step;

                    bool hit = false;
                    int  hit_type = 0;  // 1=wall, 2=grid, 3=nonfinite

                    // Check non-finite first
                    if (!std::isfinite(R) || !std::isfinite(Z)) {
                        hit = true; hit_type = 3;
                    }
                    // Check field grid exit
                    else if (R < R_grid[0] || R > R_grid[nR - 1] ||
                             Z < Z_grid[0] || Z > Z_grid[nZ - 1]) {
                        hit = true; hit_type = 2;
                    }
                    // Check wall polygon
                    else if (!point_in_toroidal_wall(R, Z, phi,
                                                wall_phi_centers,
                                                wall_R, wall_Z,
                                                n_phi_wall, n_theta_wall)) {
                        hit = true; hit_type = 1;
                    }

                    if (hit) {
                        L_out[i]   = arc;
                        tt_out[i]  = hit_type;

                        if (hit_type == 1) {
                            // Wall polygon crossing: bisect using wall poly as criterion
                            double rA = R_prev, zA = Z_prev, pA = phi_prev;
                            double rB = R,      zB = Z,      pB = phi;
                            for (int b = 0; b < 14; ++b) {
                                double rM = 0.5*(rA+rB);
                                double zM = 0.5*(zA+zB);
                                double pM = mod2pi(0.5*(pA+pB));
                                bool inside_M = point_in_toroidal_wall(rM, zM, pM,
                                        wall_phi_centers, wall_R, wall_Z,
                                        n_phi_wall, n_theta_wall);
                                if (inside_M) { rA=rM; zA=zM; pA=pM; }
                                else          { rB=rM; zB=zM; pB=pM; }
                            }
                            R_hit_out[i]   = 0.5*(rA+rB);
                            Z_hit_out[i]   = 0.5*(zA+zB);
                            phi_hit_out[i] = mod2pi(0.5*(pA+pB));
                        } else if (hit_type == 2) {
                            // Grid boundary exit: bisect using grid bounds as criterion.
                            // The field grid ends before the physical wall (common for HFS),
                            // so extrapolate linearly to the grid edge.
                            double rA = R_prev, zA = Z_prev, pA = phi_prev;
                            double rB = (std::isfinite(R)) ? R : R_prev;
                            double zB = (std::isfinite(Z)) ? Z : Z_prev;
                            double pB = phi;
                            for (int b = 0; b < 14; ++b) {
                                double rM = 0.5*(rA+rB);
                                double zM = 0.5*(zA+zB);
                                double pM = mod2pi(0.5*(pA+pB));
                                bool in_grid = (rM >= R_grid[0] && rM <= R_grid[nR-1] &&
                                                zM >= Z_grid[0] && zM <= Z_grid[nZ-1]);
                                if (in_grid) { rA=rM; zA=zM; pA=pM; }
                                else         { rB=rM; zB=zM; pB=pM; }
                            }
                            R_hit_out[i]   = 0.5*(rA+rB);
                            Z_hit_out[i]   = 0.5*(zA+zB);
                            phi_hit_out[i] = mod2pi(0.5*(pA+pB));
                        } else {
                            // Non-finite: report last finite position
                            R_hit_out[i]   = R_prev;
                            Z_hit_out[i]   = Z_prev;
                            phi_hit_out[i] = phi_prev;
                        }
                        break;
                    }
                }
                // Replace SENTINEL with NaN for unterminated seeds
                if (L_out[i] >= SENTINEL) L_out[i] = SENTINEL; // will be replaced in binding
            }
        }).wait();
    }
    // Sentinel → NaN
    for (int i = 0; i < N_seeds; ++i) {
        if (L_fwd[i] >= SENTINEL) {
            L_fwd[i] = NAN_VAL;
            R_hit_fwd[i] = NAN_VAL; Z_hit_fwd[i] = NAN_VAL; phi_hit_fwd[i] = NAN_VAL;
        }
        if (L_bwd[i] >= SENTINEL) {
            L_bwd[i] = NAN_VAL;
            R_hit_bwd[i] = NAN_VAL; Z_hit_bwd[i] = NAN_VAL; phi_hit_bwd[i] = NAN_VAL;
        }
    }
}

// ---------------------------------------------------------------------------
// find_fixed_points_batch
//
// For each initial guess (R0, Z0), run Newton iterations on P^n(x) - x = 0
// using finite-difference Jacobian (4 extra field-line integrations per step).
// On convergence, also returns the full 2×2 DPm = DP^m at the fixed point,
// eigenvalues, and a classification: 1=X-point (|Tr|>2), 0=O-point (|Tr|<2).
//
// Outputs (length N_seeds each):
//   R_out, Z_out          - converged position (NaN if not converged)
//   residual_out          - |P^n(x)-x| at final iterate
//   converged_out         - 1 if converged, 0 otherwise
//   DPm_out               — flattened 2×2 DPm row-major (length 4*N_seeds)
//   eig_r_out, eig_i_out  - real/imag parts of eigenvalues (length 2*N_seeds)
//   point_type_out        - 1=X-point, 0=O-point, -1=not converged
// ---------------------------------------------------------------------------
struct FixedPointResult {
    double R, Z;
    double residual;
    int    converged;   // 0 or 1
    double DPm[4];      // row-major: [00,01,10,11]
    double eig_r[2], eig_i[2];
    int    point_type;  // 1=X, 0=O, -1=failed
};

// Integrate m_turns starting from (R, Z, phi_start); return final (R, Z).
// Returns false if field line exits the grid or becomes non-finite.
static inline bool pmap_m(
    double& R, double& Z,
    double phi_start, int m_turns, double DPhi,
    const double* BR, const double* BPhi, const double* BZ,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi)
{
    double phi = phi_start;
    double phi_end = phi_start + m_turns * 2.0 * M_PI;
    while (phi < phi_end - 1e-12) {
        double step = std::min(DPhi, phi_end - phi);
        rk4_step(R, Z, phi, step, BR, BPhi, BZ, R_grid, nR, Z_grid, nZ, Phi_grid, nPhi);
        phi += step;
        if (!std::isfinite(R) || !std::isfinite(Z) ||
            R < R_grid[0] || R > R_grid[nR-1] ||
            Z < Z_grid[0] || Z > Z_grid[nZ-1])
            return false;
    }
    return true;
}


// ---------------------------------------------------------------------------
// Evolve DX_pol(φ_s, φ_s+2πm) over m toroidal turns.
// ---------------------------------------------------------------------------
// Simultaneously integrates the field-line trajectory (R,Z) and the 2×2
// poloidal Jacobian DX_pol using the analytic local gradient J(φ):
//
//   d(DX_pol)/dphi = A(phi) * DX_pol,    DX_pol(phi_s, phi_s) = I
//
// Returns DX_out[4] = DX_pol(φ_s, φ_s+2πm) in row-major order.
// When the turn count m equals the island-chain poloidal mode number,
// DX_out = DPm(φ_s), the monodromy matrix.
//
// Unlike central finite differences (which require 4 separate m-turn
// integrations and accumulate O(ε2) truncation error), this uses the
// analytic local gradient of the ODE right-hand side at each RK4 substep,
// producing the variational-equation-accurate result in a single pass.
static inline bool DX_pol_m_turns(
    double& R, double& Z,
    double DX_out[4],
    double phi_start, int m_turns, double DPhi,
    const double* BR, const double* BPhi, const double* BZ,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi)
{
    double D00=1.0,D01=0.0,D10=0.0,D11=1.0;  // DX_pol(φ_s, φ_s) = I
    double phi = phi_start;
    double phi_end = phi_start + m_turns * 2.0 * M_PI;
    while (phi < phi_end - 1e-12) {
        double step = std::min(DPhi, phi_end - phi);
        rk4_step_DX_pol(R, Z, D00,D01,D10,D11, phi, step,
                        BR, BPhi, BZ, R_grid,nR,Z_grid,nZ,Phi_grid,nPhi);
        phi += step;
        if (!std::isfinite(R) || !std::isfinite(Z) ||
            R < R_grid[0] || R > R_grid[nR-1] ||
            Z < Z_grid[0] || Z > Z_grid[nZ-1])
            return false;
    }
    DX_out[0]=D00; DX_out[1]=D01; DX_out[2]=D10; DX_out[3]=D11;
    return true;
}

// ---------------------------------------------------------------------------
// Evolve DPm(phi) along a cycle orbit using the commutator ODE.
// ---------------------------------------------------------------------------
// For a period-m fixed point (P^m(x)=x), the monodromy DPm(phi) satisfies:
//
//   d(DPm)/dphi = A(phi+2pi_m)*DPm(phi) - DPm(phi)*A(phi)
//
// At the fixed point A(phi+2pi_m)=A(phi) (field-line returns to same
// spatial point after m toroidal turns), giving the commutator:
//
//   d(DPm)/dphi = A*DPm - DPm*A  =  [A, DPm]
//
// This preserves Tr(DPm) (commutator trace=0), so X vs O classification
// is invariant along the cycle -- as physically expected.
//
// Given the orbit (R,Z,phi) at discrete points and DPm(phi0) from a
// Newton solve at phi=0, this integrates DPm along the entire cycle
// WITHOUT additional field-line tracing.
static inline void evolve_DPm_along_cycle(
    const double* R_traj, const double* Z_traj, const double* phi_traj,
    int n_pts,
    const double* DPm_init,  // [D00,D01,D10,D11] at index 0
    double* DPm_out,         // n_pts * 4
    const double* BR, const double* BPhi, const double* BZ,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi)
{
    if (n_pts < 1) return;
    DPm_out[0]=DPm_init[0];DPm_out[1]=DPm_init[1];
    DPm_out[2]=DPm_init[2];DPm_out[3]=DPm_init[3];

    // RK4 for commutator ODE: dD/dphi = J*D - D*J
    auto rk4_c = [&](double R,double Z,double phi,double dPhi,
                      double& A,double& B,double& C,double& Dm){
        double fR,fZ,A00,A01,A10,A11;
        {double BRv,dBR_dR,dBR_dZ,BPhiv,dBPhi_dR,dBPhi_dZ,BZv,dBZ_dR,dBZ_dZ;
         if(!interp3d_grad(BRv,dBR_dR,dBR_dZ,BR,R_grid,nR,Z_grid,nZ,Phi_grid,nPhi,R,Z,phi))return;
         if(!interp3d_grad(BPhiv,dBPhi_dR,dBPhi_dZ,BPhi,R_grid,nR,Z_grid,nZ,Phi_grid,nPhi,R,Z,phi))return;
         if(!interp3d_grad(BZv,dBZ_dR,dBZ_dZ,BZ,R_grid,nR,Z_grid,nZ,Phi_grid,nPhi,R,Z,phi))return;
         if(std::abs(BPhiv)<=1e-12){A00=A01=A10=A11=0.0;return;}
         double iBp=1.0/BPhiv,iBp2=iBp/BPhiv;
         fR=R*BRv*iBp;fZ=R*BZv*iBp;
         A00=BRv*iBp+R*(dBR_dR*iBp-BRv*iBp2*dBPhi_dR);
         A01=R*(dBR_dZ*iBp-BRv*iBp2*dBPhi_dZ);
         A10=BZv*iBp+R*(dBZ_dR*iBp-BZv*iBp2*dBPhi_dR);
         A11=R*(dBZ_dZ*iBp-BZv*iBp2*dBPhi_dZ);}
        // Commutator: f(D)=A*D-D*A, D=[A B;C Dm], A=[A00 A01;A10 A11]
        double fA=A01*C-B*A10,fB=A00*B+A01*Dm-A*A01-B*A11;
        double fC=A10*A+A11*C-C*A00-Dm*A10,fDm=A10*B-C*A01;
        double hh=0.5*dPhi;
        double k1_A=fA,k1_B=fB,k1_C=fC,k1_Dm=fDm;
        double A2=A+hh*k1_A,B2=B+hh*k1_B,C2=C+hh*k1_C,Dm2=Dm+hh*k1_Dm;
        double k2_A=A01*C2-B2*A10,k2_B=A00*B2+A01*Dm2-A2*A01-B2*A11;
        double k2_C=A10*A2+A11*C2-C2*A00-Dm2*A10,k2_Dm=A10*B2-C2*A01;
        double A3=A+hh*k2_A,B3=B+hh*k2_B,C3=C+hh*k2_C,Dm3=Dm+hh*k2_Dm;
        double k3_A=A01*C3-B3*A10,k3_B=A00*B3+A01*Dm3-A3*A01-B3*A11;
        double k3_C=A10*A3+A11*C3-C3*A00-Dm3*A10,k3_Dm=A10*B3-C3*A01;
        double A4=A+dPhi*k3_A,B4=B+dPhi*k3_B,C4=C+dPhi*k3_C,Dm4=Dm+dPhi*k3_Dm;
        double k4_A=A01*C4-B4*A10,k4_B=A00*B4+A01*Dm4-A4*A01-B4*A11;
        double k4_C=A10*A4+A11*C4-C4*A00-Dm4*A10,k4_Dm=A10*B4-C4*A01;
        double s6=dPhi/6.0;
        A+=s6*(k1_A+2*k2_A+2*k3_A+k4_A);B+=s6*(k1_B+2*k2_B+2*k3_B+k4_B);
        C+=s6*(k1_C+2*k2_C+2*k3_C+k4_C);Dm+=s6*(k1_Dm+2*k2_Dm+2*k3_Dm+k4_Dm);
    };
    double A=DPm_init[0],B=DPm_init[1],C=DPm_init[2],Dm=DPm_init[3];
    for(int k=1;k<n_pts;++k){
        double R=R_traj[k-1],Z=Z_traj[k-1];
        double phi0=phi_traj[k-1],phi1=phi_traj[k];
        double ds=phi1-phi0;
        if(ds<=0.0){DPm_out[4*k]=A;DPm_out[4*k+1]=B;DPm_out[4*k+2]=C;DPm_out[4*k+3]=Dm;continue;}
        int ns=(int)std::ceil(ds/0.005);if(ns<1)ns=1;
        double dss=ds/ns,ph=phi0;
        for(int s=0;s<ns;++s,ph+=dss)rk4_c(R,Z,ph,dss,A,B,C,Dm);
        DPm_out[4*k]=A;DPm_out[4*k+1]=B;DPm_out[4*k+2]=C;DPm_out[4*k+3]=Dm;
    }
}



static inline FixedPointResult newton_fixed_point(
    double R0, double Z0,
    double phi_section,     // Poincare section angle [rad]
    int    m_turns,         // P^m: m toroidal turns (poloidal mode number)
    double DPhi,
    double fd_eps,          // finite-difference step [m]
    int    max_iter,
    double tol,             // convergence: |P^m(x)-x| < tol
    const double* BR, const double* BPhi, const double* BZ,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi)
{
    FixedPointResult res;
    res.R = std::numeric_limits<double>::quiet_NaN();
    res.Z = std::numeric_limits<double>::quiet_NaN();
    res.residual = 1e30;
    res.converged = 0;
    res.DPm[0]=res.DPm[1]=res.DPm[2]=res.DPm[3] = 0.0;
    res.eig_r[0]=res.eig_r[1]=res.eig_i[0]=res.eig_i[1] = 0.0;
    res.point_type = -1;

    double phi0 = mod2pi(phi_section);
    double R = R0, Z = Z0;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Evaluate F(x) = P^m(x) - x  and compute DPm = DX_pol(φ, φ+2πm)
        // via analytic DX_pol evolution (DX_pol_m_turns).
        double Rf = R, Zf = Z;
        double DP_cur[4];
        if (!DX_pol_m_turns(Rf, Zf, DP_cur, phi0, m_turns, DPhi,
                            BR, BPhi, BZ, R_grid, nR, Z_grid, nZ, Phi_grid, nPhi))
            return res;
        double F0 = Rf - R;
        double F1 = Zf - Z;

        res.residual = std::sqrt(F0*F0 + F1*F1);
        if (res.residual < tol) {
            res.converged = 1;
            // Store DPm(φ_s) = DX_pol(φ_s, φ_s+2πm) from analytic evolution
            res.DPm[0]=DP_cur[0]; res.DPm[1]=DP_cur[1];
            res.DPm[2]=DP_cur[2]; res.DPm[3]=DP_cur[3];
            break;
        }

        // DF = DPm - I  (Jacobian of F = P^m(x) - x)
        double DF00 = DP_cur[0] - 1.0, DF01 = DP_cur[1];
        double DF10 = DP_cur[2],        DF11 = DP_cur[3] - 1.0;
        double det = DF00*DF11 - DF01*DF10;
        if (std::abs(det) < 1e-20) return res;

        // Newton step: dx = -DF^{-1} * F
        double dR = -(DF11*F0 - DF01*F1) / det;
        double dZ = -(-DF10*F0 + DF00*F1) / det;

        R += dR; Z += dZ;

        if (R < R_grid[0] || R > R_grid[nR-1] ||
            Z < Z_grid[0] || Z > Z_grid[nZ-1])
            return res;
    }

    if (!res.converged) return res;

    res.R = R; res.Z = Z;

    // DPm already stored during the final convergence-check iteration

    // Eigenvalues of 2x2 matrix via characteristic polynomial
    double a = res.DPm[0], b = res.DPm[1], c2 = res.DPm[2], d = res.DPm[3];
    double tr = a + d, det2 = a*d - b*c2;
    double disc = tr*tr - 4.0*det2;
    if (disc >= 0.0) {
        res.eig_r[0] = 0.5*(tr + std::sqrt(disc));
        res.eig_r[1] = 0.5*(tr - std::sqrt(disc));
        res.eig_i[0] = res.eig_i[1] = 0.0;
    } else {
        res.eig_r[0] = res.eig_r[1] = 0.5*tr;
        res.eig_i[0] =  0.5*std::sqrt(-disc);
        res.eig_i[1] = -0.5*std::sqrt(-disc);
    }

    res.point_type = (std::abs(tr) > 2.0) ? 1 : 0;
    return res;
}

void find_fixed_points_batch(
    const double* R_seeds, const double* Z_seeds, int N_seeds,
    double phi_section,
    int    m_turns,
    double DPhi,
    double fd_eps,
    int    max_iter,
    double tol,
    const double* BR, const double* BPhi, const double* BZ,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    int n_threads,
    double* R_out, double* Z_out,
    double* residual_out,
    int*    converged_out,
    double* DPm_out,         // 4*N_seeds (row-major per seed)
    double* eig_r_out,       // 2*N_seeds
    double* eig_i_out,       // 2*N_seeds
    int*    point_type_out)  // N_seeds
{
    if (n_threads <= 0)
        n_threads = (int)std::thread::hardware_concurrency();

    BS::thread_pool pool((unsigned int)n_threads);
    pool.parallelize_loop(0, N_seeds, [&](int i_start, int i_end) {
        for (int i = i_start; i < i_end; ++i) {
            auto r = newton_fixed_point(
                R_seeds[i], Z_seeds[i],
                phi_section, m_turns, DPhi, fd_eps, max_iter, tol,
                BR, BPhi, BZ,
                R_grid, nR, Z_grid, nZ, Phi_grid, nPhi);

            R_out[i]          = r.R;
            Z_out[i]          = r.Z;
            residual_out[i]   = r.residual;
            converged_out[i]  = r.converged;
            DPm_out[4*i+0]    = r.DPm[0];
            DPm_out[4*i+1]    = r.DPm[1];
            DPm_out[4*i+2]    = r.DPm[2];
            DPm_out[4*i+3]    = r.DPm[3];
            eig_r_out[2*i+0]  = r.eig_r[0];
            eig_r_out[2*i+1]  = r.eig_r[1];
            eig_i_out[2*i+0]  = r.eig_i[0];
            eig_i_out[2*i+1]  = r.eig_i[1];
            point_type_out[i] = r.point_type;
        }
    }).wait();
}

// ---------------------------------------------------------------------------
// trace_orbit_along_phi
//
// Starting from (R0, Z0) at phi0, integrate the field line and output
// (R, Z) at evenly spaced phi values: phi0, phi0+dphi_out, ..., phi0+phi_span.
// Also computes the 2×2 DPm(φ) = DX_pol(φ, φ+2π·m_turns_DPm) at each
// output point via analytic DX_pol evolution (DX_pol_m_turns).
// output point - used for ribbon eigenvector visualization.
//
// Output arrays (length n_out = ceil(phi_span/dphi_out)+1):
//   R_traj, Z_traj          : orbit positions
//   phi_traj                : toroidal angles (unwrapped)
//   DPm_traj [n_out x 4]   : DPm at each output point (row-major 2x2)
//   alive_out [n_out]       : 1 if integration succeeded up to that point
// ---------------------------------------------------------------------------
void trace_orbit_along_phi(
    double R0, double Z0, double phi0,
    double phi_span,    // total toroidal angle to cover [rad]
    double dphi_out,    // output spacing [rad]
    int    m_turns_DPm, // m for DPm = DX_pol(φ, φ+2πm) (island chain period)
    double DPhi,        // integration step
    double fd_eps,      // unused (was FD step; DPm now via DX_pol_m_turns)
    const double* BR, const double* BPhi, const double* BZ,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    int    n_out,
    double* R_traj, double* Z_traj, double* phi_traj,
    double* DPm_traj,  // [n_out * 4]
    int*    alive_out)
{
    constexpr double NAN_V = std::numeric_limits<double>::quiet_NaN();

    // Fill outputs with NaN / 0
    for (int i = 0; i < n_out; ++i) {
        R_traj[i] = NAN_V; Z_traj[i] = NAN_V; phi_traj[i] = NAN_V;
        DPm_traj[4*i+0]=NAN_V; DPm_traj[4*i+1]=NAN_V;
        DPm_traj[4*i+2]=NAN_V; DPm_traj[4*i+3]=NAN_V;
        alive_out[i] = 0;
    }

    double R = R0, Z = Z0;
    double phi = phi0;          // unwrapped
    double phi_next_out = phi0; // next output checkpoint
    int    out_idx = 0;

    // Helper: compute DPm(φ) = DX_pol(φ, φ+2π·m_turns_DPm) at the
    // current trajectory point.  Uses DX_pol_m_turns which integrates the
    // analytic variational equation  d(DX_pol)/dφ = J·DX_pol  alongside
    // the field-line trajectory in a single pass.
    auto compute_DPm = [&](double r, double z, double phi_sec,
                            double* DPm) -> bool {
        double Rf = r, Zf = z;
        return DX_pol_m_turns(Rf, Zf, DPm, mod2pi(phi_sec), m_turns_DPm, DPhi,
                              BR, BPhi, BZ, R_grid, nR, Z_grid, nZ, Phi_grid, nPhi);
    };

    // Record initial point
    if (out_idx < n_out) {
        R_traj[out_idx] = R; Z_traj[out_idx] = Z; phi_traj[out_idx] = phi;
        double DPm[4];
        if (compute_DPm(R, Z, phi, DPm)) {
            for (int k=0;k<4;k++) DPm_traj[4*out_idx+k] = DPm[k];
        }
        alive_out[out_idx] = 1;
        out_idx++;
        phi_next_out += dphi_out;
    }

    double phi_end = phi0 + phi_span;

    while ((phi_end > phi0 ? phi < phi_end - 1e-12 : phi > phi_end + 1e-12) && out_idx < n_out) {
        double step_raw = (phi_end > phi0 ? std::min(DPhi, phi_end - phi) : std::max(-DPhi, phi_end - phi));
        double step = step_raw;
        // Integrate one step
        rk4_step(R, Z, phi, step, BR, BPhi, BZ,
                 R_grid, nR, Z_grid, nZ, Phi_grid, nPhi);
        phi += step;

        if (!std::isfinite(R) || !std::isfinite(Z) ||
            R < R_grid[0] || R > R_grid[nR-1] ||
            Z < Z_grid[0] || Z > Z_grid[nZ-1])
            break;

        // Check if we passed an output checkpoint (forward or backward)
        while (out_idx < n_out && (dphi_out>0 ? phi >= phi_next_out - 1e-12 : phi <= phi_next_out + 1e-12)) {
            R_traj[out_idx] = R; Z_traj[out_idx] = Z; phi_traj[out_idx] = phi_next_out;
            double DPm[4];
            if (compute_DPm(R, Z, mod2pi(phi_next_out), DPm)) {
                for (int k=0;k<4;k++) DPm_traj[4*out_idx+k] = DPm[k];
            }
            alive_out[out_idx] = 1;
            out_idx++;
            phi_next_out += dphi_out;
        }
    }
}

// ---------------------------------------------------------------------------
// trace_poincare_beta_sweep
//
// Like trace_one_seed / trace_poincare_batch but applies a diamagnetic +
// Pfirsch-Schlüter field correction for finite beta on-the-fly during RK4.
//
// Beta correction (evaluated at each sub-step):
//   mu0 = 4π×10-7
//   p0  = beta * B_ref2 * (alpha+1) / (2*mu0)
//   psi_n    = min(((R-R_ax)2+(Z-Z_ax)2)/a_eff2, 1)
//   profile  = (1-psi_n)^alpha
//   dp_dpsi  = -alpha*(1-psi_n)^(alpha-1)
//
//   dBR   = -(mu0*p0/BPhi) * dp_dpsi * 2*(R-R_ax)/a_eff2
//           + 2*(mu0*p0/B2) * kappa_R * profile          [PS term]
//   dBZ   = -(mu0*p0/BPhi) * dp_dpsi * 2*(Z-Z_ax)/a_eff2
//   dBPhi =  (mu0*p0/BPhi) * profile
//   kappa_R = -(Z-Z_ax) / (sqrt((R-R_ax)2+(Z-Z_ax)2)+eps * R)
//
// Output layout (same as trace_poincare_multi / trace_one_seed):
//   poi_counts[seed*n_sec + s]                          = n crossings
//   poi_R_flat[seed*N_turns*n_sec + s*N_turns + cnt]    = R
//   poi_Z_flat[...]                                     = Z
// ---------------------------------------------------------------------------

static constexpr double BETA_MU0 = 4.0e-7 * M_PI;

static inline void rk4_step_beta(
    double& R, double& Z, double phi, double dPhi,
    const double* BR, const double* BPhi, const double* BZ,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    double beta, double R_ax, double Z_ax, double a_eff,
    double alpha, double p0)
{
    // Effective dR/dphi and dZ/dphi with beta-corrected field
    auto deriv = [&](double r, double z, double p,
                     double& dR_out, double& dZ_out) {
        double bp = interp3d(BPhi, R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, r, z, p);
        double br = interp3d(BR,   R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, r, z, p);
        double bz = interp3d(BZ,   R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, r, z, p);

        if (!std::isfinite(bp) || !std::isfinite(br) || !std::isfinite(bz)
            || std::abs(bp) < 1e-12) {
            dR_out = 0.0; dZ_out = 0.0; return;
        }

        if (beta != 0.0 && p0 != 0.0) {
            double rR = r - R_ax, rZ = z - Z_ax;
            double r2 = rR*rR + rZ*rZ;
            double r_minor = std::sqrt(r2) + 1e-10;
            double psi_n = std::min(r2 / (a_eff*a_eff), 1.0);
            double one_m_psi = std::max(1.0 - psi_n, 0.0);

            double profile  = std::pow(one_m_psi, alpha);
            double dp_dpsi  = (one_m_psi > 0.0)
                              ? -alpha * std::pow(one_m_psi, alpha - 1.0)
                              : 0.0;

            double bp_safe = (std::abs(bp) < 1e-12) ? 1e-12 : bp;
            double B2 = br*br + bp*bp + bz*bz;
            double B2_safe = (B2 < 1e-12) ? 1e-12 : B2;
            double inv2a2 = 1.0 / (a_eff*a_eff);
            double mu0_p0 = BETA_MU0 * p0;

            double dBR_dia = -(mu0_p0 / bp_safe) * dp_dpsi * 2.0 * rR * inv2a2;
            double dBZ_dia = -(mu0_p0 / bp_safe) * dp_dpsi * 2.0 * rZ * inv2a2;
            double dBPhi_d =  (mu0_p0 / bp_safe) * profile;

            double kappa_R = -rZ / (r_minor * r);
            double dBR_PS  = 2.0 * (mu0_p0 / B2_safe) * kappa_R * profile;

            br += dBR_dia + dBR_PS;
            bz += dBZ_dia;
            bp += dBPhi_d;
        }

        if (std::abs(bp) < 1e-12) { dR_out = 0.0; dZ_out = 0.0; return; }
        dR_out = r * br / bp;
        dZ_out = r * bz / bp;
    };

    double k1R, k1Z, k2R, k2Z, k3R, k3Z, k4R, k4Z;
    deriv(R,                   Z,                   phi,              k1R, k1Z);
    deriv(R + 0.5*dPhi*k1R,   Z + 0.5*dPhi*k1Z,   phi + 0.5*dPhi,  k2R, k2Z);
    deriv(R + 0.5*dPhi*k2R,   Z + 0.5*dPhi*k2Z,   phi + 0.5*dPhi,  k3R, k3Z);
    deriv(R +     dPhi*k3R,   Z +     dPhi*k3Z,   phi +     dPhi,  k4R, k4Z);

    R += dPhi / 6.0 * (k1R + 2*k2R + 2*k3R + k4R);
    Z += dPhi / 6.0 * (k1Z + 2*k2Z + 2*k3Z + k4Z);
}


static void trace_one_seed_beta(
    int seed_idx, int /*N_seeds*/,
    double R0, double Z0, double phi_start,
    const double* phi_sections, int n_sec,
    int N_turns, double DPhi,
    const double* BR, const double* BPhi, const double* BZ,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    const double* wall_R, const double* wall_Z, int n_wall,
    double beta, double R_ax, double Z_ax, double a_eff,
    double alpha, double p0,
    int* poi_counts,
    double* poi_R_flat,
    double* poi_Z_flat)
{
    double R = R0, Z = Z0;
    double phi = phi_start;
    double phi_end = phi_start + N_turns * 2.0 * M_PI;

    int cnt_base = seed_idx * n_sec;
    int poi_base = seed_idx * N_turns * n_sec;

    while (phi < phi_end - 1e-12) {
        double step = std::min(DPhi, phi_end - phi);

        double R_old = R, Z_old = Z, phi_old = phi;

        rk4_step_beta(R, Z, phi, step,
                      BR, BPhi, BZ,
                      R_grid, nR, Z_grid, nZ, Phi_grid, nPhi,
                      beta, R_ax, Z_ax, a_eff, alpha, p0);
        phi += step;

        if (n_wall > 0 && !point_in_wall(R, Z, wall_R, wall_Z, n_wall))
            break;

        for (int s = 0; s < n_sec; ++s) {
            int cnt = poi_counts[cnt_base + s];
            if (cnt >= N_turns) continue;

            double sec = phi_sections[s];
            double k_raw = (phi_old - sec) / (2.0 * M_PI);
            int k = (int)std::ceil(k_raw);
            if (k_raw == (double)k) k++;
            double phi_cross = sec + k * 2.0 * M_PI;

            if (phi_cross > phi_old && phi_cross <= phi) {
                double t = (phi_cross - phi_old) / (phi - phi_old);
                double R_c = R_old + t * (R - R_old);
                double Z_c = Z_old + t * (Z - Z_old);
                poi_R_flat[poi_base + s * N_turns + cnt] = R_c;
                poi_Z_flat[poi_base + s * N_turns + cnt] = Z_c;
                poi_counts[cnt_base + s]++;
            }
        }
    }
}


void trace_poincare_beta_sweep(
    const double* R_seeds, const double* Z_seeds, int N_seeds,
    const double* phi_sections, int n_sec,
    int N_turns, double DPhi,
    const double* BR, const double* BPhi, const double* BZ,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    const double* wall_R, const double* wall_Z, int n_wall,
    double beta, double R_ax, double Z_ax, double a_eff,
    double alpha_pressure, double B_ref,
    int n_threads,
    int* poi_counts,
    double* poi_R_flat,
    double* poi_Z_flat)
{
    if (n_threads <= 0)
        n_threads = (int)std::thread::hardware_concurrency();

    // Precompute p0
    double p0 = (beta != 0.0)
                ? beta * B_ref * B_ref * (alpha_pressure + 1.0) / (2.0 * BETA_MU0)
                : 0.0;

    // Normalise sections to [0, 2pi)
    std::vector<double> secs(n_sec);
    for (int s = 0; s < n_sec; ++s)
        secs[s] = mod2pi(phi_sections[s]);

    BS::thread_pool pool((unsigned int)n_threads);
    pool.parallelize_loop(0, N_seeds, [&](int i_start, int i_end) {
        for (int i = i_start; i < i_end; ++i) {
            trace_one_seed_beta(
                i, N_seeds,
                R_seeds[i], Z_seeds[i], secs[0],
                secs.data(), n_sec,
                N_turns, DPhi,
                BR, BPhi, BZ,
                R_grid, nR, Z_grid, nZ, Phi_grid, nPhi,
                wall_R, wall_Z, n_wall,
                beta, R_ax, Z_ax, a_eff, alpha_pressure, p0,
                poi_counts, poi_R_flat, poi_Z_flat);
        }
    }).wait();
}

} // namespace cyna
