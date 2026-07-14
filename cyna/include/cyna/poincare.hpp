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

static inline double mod_period(double x, double period) {
    x = std::fmod(x, period);
    if (x < 0.0) x += period;
    return x;
}

static inline bool locate_phi_cell(
    const double* Phi_grid, int nPhi,
    double field_period,
    double& Phi,
    int& iPhi, int& iPhi1,
    double& tP)
{
    if (nPhi <= 0 || !std::isfinite(Phi))
        return false;
    if (nPhi == 1) {
        iPhi = 0;
        iPhi1 = 0;
        tP = 0.0;
        Phi = Phi_grid[0];
        return true;
    }

    const double phi0 = Phi_grid[0];
    const double phi1 = Phi_grid[nPhi - 1];
    if (!std::isfinite(field_period) || field_period <= 0.0)
        return false;
    const double tol = 1e-12 * std::max(1.0, std::abs(field_period));
    if (std::abs((phi1 - phi0) - field_period) > tol)
        return false;

    // Phi_grid is expected to be a closed periodic grid:
    // [phi0, ..., phi0 + field_period], where the last data plane is a copy
    // of the first.  field_period is explicit (2*pi/nfp), never inferred.
    Phi = phi0 + mod_period(Phi - phi0, field_period);
    if (Phi >= phi1)
        Phi = phi0;

    int lo = 0, hi = nPhi - 2;     // nPhi-2 is the last valid left index
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (Phi >= Phi_grid[mid + 1]) lo = mid + 1;
        else                          hi = mid;
    }
    iPhi = lo;
    iPhi1 = iPhi + 1;

    double phiLo = Phi_grid[iPhi];
    double phiHi = Phi_grid[iPhi1];
    tP = (phiHi > phiLo) ? (Phi - phiLo) / (phiHi - phiLo) : 0.0;
    if (tP < 0.0) tP = 0.0;
    if (tP > 1.0) tP = 1.0;
    return true;
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
    double phi0 = phi_centers[0];
    double period = 2.0 * M_PI;
    if (n_phi_wall > 1) {
        double span = phi_centers[n_phi_wall - 1] - phi_centers[0];
        double dphi = span / (double)(n_phi_wall - 1);
        double inferred = span + dphi;
        if (std::isfinite(inferred) && inferred > 0.0 && inferred <= 2.0 * M_PI * (1.0 + 1e-8))
            period = inferred;
    }
    double phi_mod = phi0 + mod_period(phi - phi0, period);
    double best_d = 1e300;
    int best_i = 0;
    for (int i = 0; i < n_phi_wall; ++i) {
        double d = std::abs(phi_centers[i] - phi_mod);
        d = std::min(d, period - d);
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
// Phi convention:
//   Phi_grid is a closed periodic grid
//     np.append(linspace(phi0, phi0 + period, N, endpoint=False),
//               phi0 + period)
//   and data[:, :, N] is a copy of data[:, :, 0].
//
// The interpolation period is explicit.  Raw legacy callers default to 2*pi;
// object-first callers pass 2*pi/nfp from VectorFieldCylind.nfp.
//
// Out-of-bounds R or Z → NaN  (matches scipy fill_value=np.nan)
// ---------------------------------------------------------------------------
inline double interp3d(
    const double* data,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,   // nPhi = N_phi_original + 1  (extended)
    double R, double Z, double Phi,
    double field_period = 2.0 * M_PI)
{
    if (!std::isfinite(R) || !std::isfinite(Z) || !std::isfinite(Phi))
        return std::numeric_limits<double>::quiet_NaN();
    if (R < R_grid[0] || R > R_grid[nR - 1] ||
        Z < Z_grid[0] || Z > Z_grid[nZ - 1])
        return std::numeric_limits<double>::quiet_NaN();

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

    int iPhi = 0, iPhi1 = 0;
    double tP = 0.0;
    if (!locate_phi_cell(Phi_grid, nPhi, field_period, Phi, iPhi, iPhi1, tP))
        return std::numeric_limits<double>::quiet_NaN();

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
    double R, double Z, double Phi,
    double field_period = 2.0 * M_PI)
{
    if (!std::isfinite(R) || !std::isfinite(Z) || !std::isfinite(Phi))
        return false;
    if (R < R_grid[0] || R > R_grid[nR - 1] ||
        Z < Z_grid[0] || Z > Z_grid[nZ - 1])
        return false;
    double dRg = (R_grid[nR-1] - R_grid[0]) / (nR - 1);
    double dZg = (Z_grid[nZ-1] - Z_grid[0]) / (nZ - 1);
    double inv_dR = 1.0 / dRg, inv_dZ = 1.0 / dZg;

    double tR_raw = (R - R_grid[0]) / (R_grid[nR-1] - R_grid[0]) * (nR - 1);
    int iR = (int)tR_raw; if (iR < 0) iR = 0; if (iR >= nR-1) iR = nR-2;
    double tR = tR_raw - iR;

    double tZ_raw = (Z - Z_grid[0]) / (Z_grid[nZ-1] - Z_grid[0]) * (nZ - 1);
    int iZ = (int)tZ_raw; if (iZ < 0) iZ = 0; if (iZ >= nZ-1) iZ = nZ-2;
    double tZ = tZ_raw - iZ;

    int iPhi = 0, iPhi1 = 0;
    double tP = 0.0;
    if (!locate_phi_cell(Phi_grid, nPhi, field_period, Phi, iPhi, iPhi1, tP))
        return false;

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
    const double* BR, const double* BZ, const double* BPhi,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    double field_period = 2.0 * M_PI)
{
    auto dRdphi = [&](double r, double z, double p) {
        double bp = interp3d(BPhi, R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, r, z, p, field_period);
        if (!std::isfinite(bp) || std::abs(bp) <= 1e-12) return 0.0;
        double br = interp3d(BR, R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, r, z, p, field_period);
        if (!std::isfinite(br)) return 0.0;
        return r * br / bp;
    };
    auto dZdphi = [&](double r, double z, double p) {
        double bp = interp3d(BPhi, R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, r, z, p, field_period);
        if (!std::isfinite(bp) || std::abs(bp) <= 1e-12) return 0.0;
        double bz = interp3d(BZ, R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, r, z, p, field_period);
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
static inline bool rk4_step_DX_pol(
    double& R, double& Z,
    double& D00, double& D01, double& D10, double& D11,
    double phi, double dPhi,
    const double* BR, const double* BZ, const double* BPhi,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    double field_period = 2.0 * M_PI)
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
        if (!interp3d_grad(BRv, dBR_dR, dBR_dZ, BR, R_grid,nR,Z_grid,nZ,Phi_grid,nPhi, r,z,p,field_period)) return false;
        if (!interp3d_grad(BPhiv, dBPhi_dR, dBPhi_dZ, BPhi, R_grid,nR,Z_grid,nZ,Phi_grid,nPhi, r,z,p,field_period)) return false;
        if (!interp3d_grad(BZv, dBZ_dR, dBZ_dZ, BZ, R_grid,nR,Z_grid,nZ,Phi_grid,nPhi, r,z,p,field_period)) return false;
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
    if(!eval_local_grad(R,Z,phi,fR1,fZ1,A00_1,A01_1,A10_1,A11_1))return false;
    double k1_D00=A00_1*D00+A01_1*D10, k1_D01=A00_1*D01+A01_1*D11;
    double k1_D10=A10_1*D00+A11_1*D10, k1_D11=A10_1*D01+A11_1*D11;
    double hh=0.5*dPhi;
    double R2=R+hh*fR1,Z2=Z+hh*fZ1, D00_2=D00+hh*k1_D00,D01_2=D01+hh*k1_D01,D10_2=D10+hh*k1_D10,D11_2=D11+hh*k1_D11;
    double fR2,fZ2,A00_2,A01_2,A10_2,A11_2;
    if(!eval_local_grad(R2,Z2,phi+hh,fR2,fZ2,A00_2,A01_2,A10_2,A11_2))return false;
    double k2_D00=A00_2*D00_2+A01_2*D10_2, k2_D01=A00_2*D01_2+A01_2*D11_2;
    double k2_D10=A10_2*D00_2+A11_2*D10_2, k2_D11=A10_2*D01_2+A11_2*D11_2;
    double R3=R+hh*fR2,Z3=Z+hh*fZ2, D00_3=D00+hh*k2_D00,D01_3=D01+hh*k2_D01,D10_3=D10+hh*k2_D10,D11_3=D11+hh*k2_D11;
    double fR3,fZ3,A00_3,A01_3,A10_3,A11_3;
    if(!eval_local_grad(R3,Z3,phi+hh,fR3,fZ3,A00_3,A01_3,A10_3,A11_3))return false;
    double k3_D00=A00_3*D00_3+A01_3*D10_3, k3_D01=A00_3*D01_3+A01_3*D11_3;
    double k3_D10=A10_3*D00_3+A11_3*D10_3, k3_D11=A10_3*D01_3+A11_3*D11_3;
    double R4=R+dPhi*fR3,Z4=Z+dPhi*fZ3, D00_4=D00+dPhi*k3_D00,D01_4=D01+dPhi*k3_D01,D10_4=D10+dPhi*k3_D10,D11_4=D11+dPhi*k3_D11;
    double fR4,fZ4,A00_4,A01_4,A10_4,A11_4;
    if(!eval_local_grad(R4,Z4,phi+dPhi,fR4,fZ4,A00_4,A01_4,A10_4,A11_4))return false;
    double k4_D00=A00_4*D00_4+A01_4*D10_4, k4_D01=A00_4*D01_4+A01_4*D11_4;
    double k4_D10=A10_4*D00_4+A11_4*D10_4, k4_D11=A10_4*D01_4+A11_4*D11_4;
    double s6=dPhi/6.0;
    R += s6*(fR1+2*fR2+2*fR3+fR4);
    Z += s6*(fZ1+2*fZ2+2*fZ3+fZ4);
    D00 += s6*(k1_D00+2*k2_D00+2*k3_D00+k4_D00);
    D01 += s6*(k1_D01+2*k2_D01+2*k3_D01+k4_D01);
    D10 += s6*(k1_D10+2*k2_D10+2*k3_D10+k4_D10);
    D11 += s6*(k1_D11+2*k2_D11+2*k3_D11+k4_D11);
    return true;
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
    const double* BR, const double* BZ, const double* BPhi,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    const double* wall_R, const double* wall_Z, int n_wall,
    int* poi_counts,
    double* poi_R_flat,
    double* poi_Z_flat,
    int direction = +1,
    double field_period = 2.0 * M_PI)
{
    double R = R0, Z = Z0;
    double phi = phi_start;         // unwrapped
    const double dir = (direction >= 0) ? 1.0 : -1.0;
    double phi_end = phi_start + dir * N_turns * 2.0 * M_PI;
    const double step_abs = std::abs(DPhi);

    int cnt_base = seed_idx * n_sec;
    int poi_base = seed_idx * N_turns * n_sec;

    while ((dir > 0.0 ? phi < phi_end - 1e-12 : phi > phi_end + 1e-12)) {
        double step = dir * std::min(step_abs, std::abs(phi_end - phi));

        double R_old = R, Z_old = Z, phi_old = phi;

        // RK4 advance
        rk4_step(R, Z, phi, step,
                 BR, BZ, BPhi,
                 R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, field_period);
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
            double phi_cross;
            if (dir > 0.0) {
                // Find section crossing in (phi_old, phi].
                double k_raw = (phi_old - sec) / (2.0 * M_PI);
                int k = (int)std::ceil(k_raw);
                if (k_raw == (double)k) k++;
                phi_cross = sec + k * 2.0 * M_PI;
            } else {
                // Find section crossing in [phi, phi_old), skipping the start section.
                double k_raw = (phi_old - sec) / (2.0 * M_PI);
                int k = (int)std::floor(k_raw);
                if (k_raw == (double)k) k--;
                phi_cross = sec + k * 2.0 * M_PI;
            }

            if ((dir > 0.0 && phi_cross > phi_old && phi_cross <= phi) ||
                (dir < 0.0 && phi_cross < phi_old && phi_cross >= phi)) {
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
    const double* BR, const double* BZ, const double* BPhi,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    const double* wall_R, const double* wall_Z, int n_wall,
    int n_threads,
    int* poi_counts,
    double* poi_R_flat,
    double* poi_Z_flat,
    int direction = +1,
    double field_period = 2.0 * M_PI)
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
                BR, BZ, BPhi,
                R_grid, nR, Z_grid, nZ, Phi_grid, nPhi,
                wall_R, wall_Z, n_wall,
                poi_counts, poi_R_flat, poi_Z_flat,
                direction, field_period);
        }
    }).wait();
}

// ---------------------------------------------------------------------------
// Batch arbitrary-span map trace.
//
// Records P_span(x), P_span^2(x), ..., P_span^N(x) for each seed.  This is
// the field-period counterpart of trace_poincare_batch and avoids launching
// one C++ single-orbit trace per seed from Python.
// ---------------------------------------------------------------------------
void trace_map_batch_span(
    const double* R_seeds, const double* Z_seeds, int N_seeds,
    double phi_section,
    double map_span,
    int N_steps, double DPhi,
    const double* BR, const double* BZ, const double* BPhi,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    const double* wall_R, const double* wall_Z, int n_wall,
    int n_threads,
    int* map_counts,
    double* map_R_flat,
    double* map_Z_flat,
    double field_period = 2.0 * M_PI)
{
    if (n_threads <= 0)
        n_threads = (int)std::thread::hardware_concurrency();
    if (N_steps <= 0 || !std::isfinite(map_span) || std::abs(map_span) <= 1e-14 ||
        !std::isfinite(DPhi) || std::abs(DPhi) <= 1e-14)
        return;

    const double step_abs = std::abs(DPhi);
    BS::thread_pool pool((unsigned int)n_threads);
    pool.parallelize_loop(0, N_seeds, [&](int i_start, int i_end) {
        for (int i = i_start; i < i_end; ++i) {
            double R = R_seeds[i];
            double Z = Z_seeds[i];
            double phi = phi_section;
            bool alive = true;
            const int base = i * N_steps;

            if (!std::isfinite(R) || !std::isfinite(Z) ||
                R < R_grid[0] || R > R_grid[nR - 1] ||
                Z < Z_grid[0] || Z > Z_grid[nZ - 1]) {
                continue;
            }
            if (n_wall > 0 && !point_in_wall(R, Z, wall_R, wall_Z, n_wall)) {
                continue;
            }

            for (int step_idx = 0; step_idx < N_steps && alive; ++step_idx) {
                const double target = phi_section + (step_idx + 1) * map_span;
                const double dir = (target >= phi) ? 1.0 : -1.0;
                while (dir > 0.0 ? phi < target - 1e-12 : phi > target + 1e-12) {
                    const double h = dir * std::min(step_abs, std::abs(target - phi));
                    rk4_step(R, Z, phi, h,
                             BR, BZ, BPhi,
                             R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, field_period);
                    phi += h;

                    if (!std::isfinite(R) || !std::isfinite(Z) ||
                        R < R_grid[0] || R > R_grid[nR - 1] ||
                        Z < Z_grid[0] || Z > Z_grid[nZ - 1]) {
                        alive = false;
                        break;
                    }
                }
                if (!alive)
                    break;
                if (n_wall > 0 && !point_in_wall(R, Z, wall_R, wall_Z, n_wall))
                    break;

                const int cnt = map_counts[i];
                if (cnt < N_steps) {
                    map_R_flat[base + cnt] = R;
                    map_Z_flat[base + cnt] = Z;
                    map_counts[i]++;
                }
            }
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
    const double* BR, const double* BZ, const double* BPhi,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    const double* wall_phi_centers, int n_phi_wall,
    const double* wall_R, const double* wall_Z, int n_theta_wall,
    int n_threads,
    int* poi_counts,
    double* poi_R_flat,
    double* poi_Z_flat,
    int direction = +1,
    double field_period = 2.0 * M_PI)
{
    if (n_threads <= 0)
        n_threads = (int)std::thread::hardware_concurrency();

    double phi_sec[1] = { mod2pi(phi_section) };
    const double dir = (direction >= 0) ? 1.0 : -1.0;
    const double step_abs = std::abs(DPhi);

    BS::thread_pool pool((unsigned int)n_threads);
    pool.parallelize_loop(0, N_seeds, [&](int i_start, int i_end) {
        for (int i = i_start; i < i_end; ++i) {
            double R = R_seeds[i], Z = Z_seeds[i];
            double phi = phi_sec[0];
            int cnt_base = i;
            int poi_base = i * N_turns;
            bool alive = true;

            for (int turn = 0; turn < N_turns && alive; ++turn) {
                double phi_end_turn = phi + dir * 2.0 * M_PI;
                while (dir > 0.0 ? phi < phi_end_turn - 1e-12 : phi > phi_end_turn + 1e-12) {
                    double step = dir * std::min(step_abs, std::abs(phi_end_turn - phi));

                    rk4_step(R, Z, phi, step,
                             BR, BZ, BPhi,
                             R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, field_period);
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
    const double* BR, const double* BZ, const double* BPhi,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    const double* wall_phi_centers, int n_phi_wall,
    const double* wall_R, const double* wall_Z, int n_theta_wall,
    int n_threads,
    double* L_fwd,
    double* L_bwd,
    double field_period = 2.0 * M_PI)
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
                                              Phi_grid, nPhi, R, Z, phi, field_period);
                    if (std::isfinite(bp_here) && std::abs(bp_here) > 1e-12) {
                        double br_here = interp3d(BR,  R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, R, Z, phi, field_period);
                        double bz_here = interp3d(BZ,  R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, R, Z, phi, field_period);
                        double Bmag = std::sqrt(br_here*br_here + bp_here*bp_here + bz_here*bz_here);
                        if (std::isfinite(Bmag))
                            arc += R * Bmag / std::abs(bp_here) * step;
                    }

                    double phi_step_dir = dir * step;
                    rk4_step(R, Z, phi, phi_step_dir,
                             BR, BZ, BPhi,
                             R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, field_period);
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
    const double* BR, const double* BZ, const double* BPhi,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    const double* wall_phi_centers, int n_phi_wall,
    const double* wall_R, const double* wall_Z, int n_theta_wall,
    int n_threads,
    double* L_fwd,     double* L_bwd,
    double* R_hit_fwd, double* Z_hit_fwd, double* phi_hit_fwd,
    double* R_hit_bwd, double* Z_hit_bwd, double* phi_hit_bwd,
    int*    term_type_fwd, int* term_type_bwd,
    double field_period = 2.0 * M_PI)
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
                                              Phi_grid, nPhi, R, Z, phi, field_period);
                    if (std::isfinite(bp_here) && std::abs(bp_here) > 1e-12) {
                        double br_here = interp3d(BR,  R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, R, Z, phi, field_period);
                        double bz_here = interp3d(BZ,  R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, R, Z, phi, field_period);
                        double Bmag = std::sqrt(br_here*br_here + bp_here*bp_here + bz_here*bz_here);
                        if (std::isfinite(Bmag))
                            arc += R * Bmag / std::abs(bp_here) * step;
                    }

                    double phi_step_dir = dir * step;
                    rk4_step(R, Z, phi, phi_step_dir,
                             BR, BZ, BPhi,
                             R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, field_period);
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
// eigenvalues, and a classification: 1=X-point with a real stable/unstable
// eigen-pair, 0=not X, -1=failed.
//
// Outputs (length N_seeds each):
//   R_out, Z_out          - converged position (NaN if not converged)
//   residual_out          - |P^n(x)-x| at final iterate
//   converged_out         - 1 if converged, 0 otherwise
//   DPm_out               — flattened 2×2 DPm row-major (length 4*N_seeds)
//   eig_r_out, eig_i_out  - real/imag parts of eigenvalues (length 2*N_seeds)
//   point_type_out        - 1=X-point, 0=not X, -1=not converged
// ---------------------------------------------------------------------------
struct FixedPointResult {
    double R, Z;
    double residual;
    int    converged;   // 0 or 1
    double DPm[4];      // row-major: [00,01,10,11]
    double eig_r[2], eig_i[2];
    int    point_type;  // 1=X, 0=not X, -1=failed
};

// Integrate m_turns starting from (R, Z, phi_start); return final (R, Z).
// Returns false if field line exits the grid or becomes non-finite.
static inline bool pmap_m(
    double& R, double& Z,
    double phi_start, int m_turns, double DPhi,
    const double* BR, const double* BZ, const double* BPhi,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    double field_period = 2.0 * M_PI)
{
    double phi = phi_start;
    double phi_end = phi_start + m_turns * 2.0 * M_PI;
    while (phi < phi_end - 1e-12) {
        double step = std::min(DPhi, phi_end - phi);
        rk4_step(R, Z, phi, step, BR, BZ, BPhi, R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, field_period);
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
    const double* BR, const double* BZ, const double* BPhi,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    double field_period = 2.0 * M_PI)
{
    double D00=1.0,D01=0.0,D10=0.0,D11=1.0;  // DX_pol(φ_s, φ_s) = I
    double phi = phi_start;
    double phi_end = phi_start + m_turns * 2.0 * M_PI;
    while (phi < phi_end - 1e-12) {
        double step = std::min(DPhi, phi_end - phi);
        if (!rk4_step_DX_pol(R, Z, D00,D01,D10,D11, phi, step,
                             BR, BZ, BPhi, R_grid,nR,Z_grid,nZ,Phi_grid,nPhi,field_period))
            return false;
        phi += step;
        if (!std::isfinite(R) || !std::isfinite(Z) ||
            R < R_grid[0] || R > R_grid[nR-1] ||
            Z < Z_grid[0] || Z > Z_grid[nZ-1])
            return false;
    }
    DX_out[0]=D00; DX_out[1]=D01; DX_out[2]=D10; DX_out[3]=D11;
    return true;
}

// Evolve DX_pol over an arbitrary toroidal span.  This is the Nfp-safe
// variant used for field-period maps P_{2π/Nfp}^m; the legacy m-turn helper
// above is the special case map_span = 2πm.
static inline bool DX_pol_span(
    double& R, double& Z,
    double DX_out[4],
    double phi_start, double phi_span, double DPhi,
    const double* BR, const double* BZ, const double* BPhi,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    double field_period = 2.0 * M_PI)
{
    if (!std::isfinite(phi_span) || std::abs(phi_span) <= 1e-14 ||
        !std::isfinite(DPhi) || std::abs(DPhi) <= 1e-14) {
        return false;
    }
    double D00=1.0,D01=0.0,D10=0.0,D11=1.0;
    double phi = phi_start;
    const double phi_end = phi_start + phi_span;
    const double dir = (phi_span >= 0.0) ? 1.0 : -1.0;
    const double step_abs = std::abs(DPhi);
    while (dir > 0.0 ? phi < phi_end - 1e-12 : phi > phi_end + 1e-12) {
        const double step = dir * std::min(step_abs, std::abs(phi_end - phi));
        if (!rk4_step_DX_pol(R, Z, D00,D01,D10,D11, phi, step,
                             BR, BZ, BPhi, R_grid,nR,Z_grid,nZ,Phi_grid,nPhi,field_period))
            return false;
        phi += step;
        if (!std::isfinite(R) || !std::isfinite(Z) ||
            R < R_grid[0] || R > R_grid[nR-1] ||
            Z < Z_grid[0] || Z > Z_grid[nZ-1])
            return false;
    }
    DX_out[0]=D00; DX_out[1]=D01; DX_out[2]=D10; DX_out[3]=D11;
    return true;
}

static inline void eig_abs_2x2(
    double a, double b, double c, double d,
    double& eig0_abs, double& eig1_abs)
{
    const double tr = a + d;
    const double det = a * d - b * c;
    const double disc = tr * tr - 4.0 * det;
    if (disc >= 0.0) {
        const double root = std::sqrt(disc);
        eig0_abs = std::abs(0.5 * (tr + root));
        eig1_abs = std::abs(0.5 * (tr - root));
    } else if (det >= 0.0) {
        eig0_abs = eig1_abs = std::sqrt(det);
    } else {
        eig0_abs = eig1_abs = std::numeric_limits<double>::quiet_NaN();
    }
}

// ---------------------------------------------------------------------------
// Trace one seed and record cumulative DP^k at Poincare returns.
// ---------------------------------------------------------------------------
// Unlike trace_orbit_along_phi(..., m_turns_DPm), this integrates the orbit
// and variational matrix only once from the initial point.  At return k it
// records DX_pol(phi0, phi0 + k * return_period), so scanning k=1..500 is
// O(k) instead of repeatedly launching k-turn integrations from Python.
void trace_poincare_dpk_growth(
    double R0, double Z0, double phi0,
    int max_returns,
    double return_period,
    int record_stride,
    double DPhi,
    const double* BR, const double* BZ, const double* BPhi,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    int n_out,
    int* k_out,
    double* R_out, double* Z_out, double* phi_out,
    double* DPk_out,
    double* eig_abs_out,
    int* alive_out,
    double field_period = 2.0 * M_PI)
{
    constexpr double NAN_V = std::numeric_limits<double>::quiet_NaN();
    for (int i = 0; i < n_out; ++i) {
        k_out[i] = 0;
        R_out[i] = NAN_V; Z_out[i] = NAN_V; phi_out[i] = NAN_V;
        DPk_out[4*i+0] = NAN_V; DPk_out[4*i+1] = NAN_V;
        DPk_out[4*i+2] = NAN_V; DPk_out[4*i+3] = NAN_V;
        eig_abs_out[2*i+0] = NAN_V; eig_abs_out[2*i+1] = NAN_V;
        alive_out[i] = 0;
    }
    if (max_returns <= 0 || record_stride <= 0 ||
        !std::isfinite(return_period) || std::abs(return_period) <= 1e-14 ||
        !std::isfinite(DPhi) || std::abs(DPhi) <= 1e-14)
        return;

    double R = R0, Z = Z0, phi = phi0;
    double D00 = 1.0, D01 = 0.0, D10 = 0.0, D11 = 1.0;
    int out_idx = 0;

    for (int ret = 1; ret <= max_returns; ++ret) {
        const double target = phi0 + ret * return_period;
        const double dir = (target >= phi) ? 1.0 : -1.0;
        while (dir > 0.0 ? phi < target - 1e-12 : phi > target + 1e-12) {
            const double step_abs = std::min(std::abs(DPhi), std::abs(target - phi));
            const double step = dir * step_abs;
            if (!rk4_step_DX_pol(R, Z, D00,D01,D10,D11, phi, step,
                                 BR, BZ, BPhi, R_grid,nR,Z_grid,nZ,Phi_grid,nPhi,field_period))
                return;
            phi += step;
            if (!std::isfinite(R) || !std::isfinite(Z) ||
                R < R_grid[0] || R > R_grid[nR-1] ||
                Z < Z_grid[0] || Z > Z_grid[nZ-1])
                return;
            if (!std::isfinite(D00) || !std::isfinite(D01) ||
                !std::isfinite(D10) || !std::isfinite(D11))
                return;
        }

        if (ret % record_stride == 0 && out_idx < n_out) {
            k_out[out_idx] = ret;
            R_out[out_idx] = R;
            Z_out[out_idx] = Z;
            phi_out[out_idx] = target;
            DPk_out[4*out_idx+0] = D00; DPk_out[4*out_idx+1] = D01;
            DPk_out[4*out_idx+2] = D10; DPk_out[4*out_idx+3] = D11;
            eig_abs_2x2(D00, D01, D10, D11,
                        eig_abs_out[2*out_idx+0],
                        eig_abs_out[2*out_idx+1]);
            alive_out[out_idx] = 1;
            out_idx++;
        }
    }
}

void trace_poincare_dpk_growth_twall(
    double R0, double Z0, double phi0,
    int max_returns,
    double return_period,
    int record_stride,
    double DPhi,
    const double* BR, const double* BZ, const double* BPhi,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    const double* wall_phi_centers,
    const double* wall_R,
    const double* wall_Z,
    int n_phi_wall,
    int n_theta_wall,
    bool stop_at_wall,
    int n_out,
    int* k_out,
    double* R_out, double* Z_out, double* phi_out,
    double* DPk_out,
    double* eig_abs_out,
    int* alive_out,
    double* hit_out,
    int* term_type_out,
    double field_period = 2.0 * M_PI)
{
    constexpr double NAN_V = std::numeric_limits<double>::quiet_NaN();
    for (int i = 0; i < n_out; ++i) {
        k_out[i] = 0;
        R_out[i] = NAN_V; Z_out[i] = NAN_V; phi_out[i] = NAN_V;
        DPk_out[4*i+0] = NAN_V; DPk_out[4*i+1] = NAN_V;
        DPk_out[4*i+2] = NAN_V; DPk_out[4*i+3] = NAN_V;
        eig_abs_out[2*i+0] = NAN_V; eig_abs_out[2*i+1] = NAN_V;
        alive_out[i] = 0;
    }
    hit_out[0] = NAN_V; hit_out[1] = NAN_V; hit_out[2] = NAN_V; hit_out[3] = NAN_V;
    *term_type_out = 0;  // 0=no termination/hit, 1=wall hit, 2=grid/nonfinite

    if (max_returns <= 0 || record_stride <= 0 ||
        !std::isfinite(return_period) || std::abs(return_period) <= 1e-14 ||
        !std::isfinite(DPhi) || std::abs(DPhi) <= 1e-14)
        return;

    double R = R0, Z = Z0, phi = phi0;
    double D00 = 1.0, D01 = 0.0, D10 = 0.0, D11 = 1.0;
    int out_idx = 0;
    bool inside_wall = point_in_toroidal_wall(
        R, Z, phi, wall_phi_centers, wall_R, wall_Z, n_phi_wall, n_theta_wall);
    if (!inside_wall) {
        hit_out[0] = R; hit_out[1] = Z; hit_out[2] = phi; hit_out[3] = 0.0;
        *term_type_out = 1;
        if (stop_at_wall) return;
    }

    for (int ret = 1; ret <= max_returns; ++ret) {
        const double target = phi0 + ret * return_period;
        const double dir = (target >= phi) ? 1.0 : -1.0;
        while (dir > 0.0 ? phi < target - 1e-12 : phi > target + 1e-12) {
            const double R_prev = R, Z_prev = Z, phi_prev = phi;
            const bool inside_prev = inside_wall;
            const double step_abs = std::min(std::abs(DPhi), std::abs(target - phi));
            const double step = dir * step_abs;
            if (!rk4_step_DX_pol(R, Z, D00,D01,D10,D11, phi, step,
                                 BR, BZ, BPhi, R_grid,nR,Z_grid,nZ,Phi_grid,nPhi,field_period)) {
                *term_type_out = 2;
                return;
            }
            phi += step;
            if (!std::isfinite(R) || !std::isfinite(Z) ||
                R < R_grid[0] || R > R_grid[nR-1] ||
                Z < Z_grid[0] || Z > Z_grid[nZ-1]) {
                *term_type_out = 2;
                return;
            }
            if (!std::isfinite(D00) || !std::isfinite(D01) ||
                !std::isfinite(D10) || !std::isfinite(D11)) {
                *term_type_out = 2;
                return;
            }

            inside_wall = point_in_toroidal_wall(
                R, Z, phi, wall_phi_centers, wall_R, wall_Z, n_phi_wall, n_theta_wall);
            if (*term_type_out == 0 && inside_prev && !inside_wall) {
                double rA = R_prev, zA = Z_prev, pA = phi_prev;
                double rB = R,      zB = Z,      pB = phi;
                for (int b = 0; b < 16; ++b) {
                    const double rM = 0.5 * (rA + rB);
                    const double zM = 0.5 * (zA + zB);
                    const double pM = 0.5 * (pA + pB);
                    const bool inside_M = point_in_toroidal_wall(
                        rM, zM, pM, wall_phi_centers, wall_R, wall_Z,
                        n_phi_wall, n_theta_wall);
                    if (inside_M) {
                        rA = rM; zA = zM; pA = pM;
                    } else {
                        rB = rM; zB = zM; pB = pM;
                    }
                }
                hit_out[0] = 0.5 * (rA + rB);
                hit_out[1] = 0.5 * (zA + zB);
                hit_out[2] = 0.5 * (pA + pB);
                hit_out[3] = (hit_out[2] - phi0) / return_period;
                *term_type_out = 1;
                if (stop_at_wall) return;
            }
        }

        if (ret % record_stride == 0 && out_idx < n_out) {
            k_out[out_idx] = ret;
            R_out[out_idx] = R;
            Z_out[out_idx] = Z;
            phi_out[out_idx] = target;
            DPk_out[4*out_idx+0] = D00; DPk_out[4*out_idx+1] = D01;
            DPk_out[4*out_idx+2] = D10; DPk_out[4*out_idx+3] = D11;
            eig_abs_2x2(D00, D01, D10, D11,
                        eig_abs_out[2*out_idx+0],
                        eig_abs_out[2*out_idx+1]);
            alive_out[out_idx] = 1;
            out_idx++;
        }
    }
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
    const double* BR, const double* BZ, const double* BPhi,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    double field_period = 2.0 * M_PI)
{
    if (n_pts < 1) return;
    DPm_out[0]=DPm_init[0];DPm_out[1]=DPm_init[1];
    DPm_out[2]=DPm_init[2];DPm_out[3]=DPm_init[3];

    // RK4 for commutator ODE: dD/dphi = J*D - D*J
    auto rk4_c = [&](double R,double Z,double phi,double dPhi,
                      double& A,double& B,double& C,double& Dm){
        double fR,fZ,A00,A01,A10,A11;
        {double BRv,dBR_dR,dBR_dZ,BPhiv,dBPhi_dR,dBPhi_dZ,BZv,dBZ_dR,dBZ_dZ;
         if(!interp3d_grad(BRv,dBR_dR,dBR_dZ,BR,R_grid,nR,Z_grid,nZ,Phi_grid,nPhi,R,Z,phi,field_period))return;
         if(!interp3d_grad(BPhiv,dBPhi_dR,dBPhi_dZ,BPhi,R_grid,nR,Z_grid,nZ,Phi_grid,nPhi,R,Z,phi,field_period))return;
         if(!interp3d_grad(BZv,dBZ_dR,dBZ_dZ,BZ,R_grid,nR,Z_grid,nZ,Phi_grid,nPhi,R,Z,phi,field_period))return;
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


// ---------------------------------------------------------------------------
// Progress DX_pol(phi_e, phi_s) along an already sampled orbit.
// ---------------------------------------------------------------------------
// This integrates only the variational equation
//
//   d(DX_pol)/dphi = A(R(phi), Z(phi), phi) * DX_pol,
//
// with DX_pol(phi_s)=I.  The orbit samples are treated as the authoritative
// path and linearly interpolated inside each output segment; no field-line
// retracing is performed here.
static inline void progress_DX_pol_along_orbit(
    const double* R_traj, const double* Z_traj, const double* phi_traj,
    int n_pts,
    double* DX_out,         // n_pts * 4, row-major 2x2 matrices
    const double* BR, const double* BZ, const double* BPhi,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    double max_step,
    double field_period = 2.0 * M_PI)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (n_pts < 1) return;
    std::fill(DX_out, DX_out + 4 * n_pts, nan);

    if (!(max_step > 0.0) || !std::isfinite(max_step)) {
        max_step = 0.005;
    }

    double D00 = 1.0, D01 = 0.0, D10 = 0.0, D11 = 1.0;
    DX_out[0] = D00; DX_out[1] = D01; DX_out[2] = D10; DX_out[3] = D11;

    auto eval_A = [&](double R, double Z, double phi,
                      double& A00, double& A01, double& A10, double& A11) -> bool {
        double BRv, dBR_dR, dBR_dZ;
        double BPhiv, dBPhi_dR, dBPhi_dZ;
        double BZv, dBZ_dR, dBZ_dZ;
        if (!interp3d_grad(BRv, dBR_dR, dBR_dZ, BR, R_grid,nR,Z_grid,nZ,Phi_grid,nPhi, R,Z,phi,field_period)) return false;
        if (!interp3d_grad(BPhiv, dBPhi_dR, dBPhi_dZ, BPhi, R_grid,nR,Z_grid,nZ,Phi_grid,nPhi, R,Z,phi,field_period)) return false;
        if (!interp3d_grad(BZv, dBZ_dR, dBZ_dZ, BZ, R_grid,nR,Z_grid,nZ,Phi_grid,nPhi, R,Z,phi,field_period)) return false;
        if (std::abs(BPhiv) <= 1e-12) {
            A00 = A01 = A10 = A11 = 0.0;
            return true;
        }
        const double invBp = 1.0 / BPhiv;
        const double invBp2 = invBp / BPhiv;
        A00 = BRv*invBp + R*(dBR_dR*invBp - BRv*invBp2*dBPhi_dR);
        A01 = R*(dBR_dZ*invBp - BRv*invBp2*dBPhi_dZ);
        A10 = BZv*invBp + R*(dBZ_dR*invBp - BZv*invBp2*dBPhi_dR);
        A11 = R*(dBZ_dZ*invBp - BZv*invBp2*dBPhi_dZ);
        return std::isfinite(A00) && std::isfinite(A01) &&
               std::isfinite(A10) && std::isfinite(A11);
    };

    auto segment_position = [](double R0, double Z0, double p0,
                               double R1, double Z1, double p1,
                               double p, double& R, double& Z) {
        const double ds = p1 - p0;
        double t = 0.0;
        if (std::abs(ds) > 1e-30) {
            t = (p - p0) / ds;
        }
        R = R0 + t * (R1 - R0);
        Z = Z0 + t * (Z1 - Z0);
    };

    auto deriv = [&](double R, double Z, double phi,
                     double a, double b, double c, double d,
                     double& da, double& db, double& dc, double& dd) -> bool {
        double A00, A01, A10, A11;
        if (!eval_A(R, Z, phi, A00, A01, A10, A11)) return false;
        da = A00*a + A01*c;
        db = A00*b + A01*d;
        dc = A10*a + A11*c;
        dd = A10*b + A11*d;
        return std::isfinite(da) && std::isfinite(db) &&
               std::isfinite(dc) && std::isfinite(dd);
    };

    for (int k = 1; k < n_pts; ++k) {
        const double R0 = R_traj[k-1], Z0 = Z_traj[k-1], p0 = phi_traj[k-1];
        const double R1 = R_traj[k],   Z1 = Z_traj[k],   p1 = phi_traj[k];
        const double ds = p1 - p0;

        if (!std::isfinite(R0) || !std::isfinite(Z0) || !std::isfinite(p0) ||
            !std::isfinite(R1) || !std::isfinite(Z1) || !std::isfinite(p1)) {
            return;
        }

        if (std::abs(ds) <= 1e-30) {
            DX_out[4*k+0] = D00; DX_out[4*k+1] = D01;
            DX_out[4*k+2] = D10; DX_out[4*k+3] = D11;
            continue;
        }

        int ns = (int)std::ceil(std::abs(ds) / max_step);
        if (ns < 1) ns = 1;
        const double h = ds / (double)ns;
        double ph = p0;

        for (int s = 0; s < ns; ++s) {
            double rA, zA, rB, zB, rC, zC;
            segment_position(R0, Z0, p0, R1, Z1, p1, ph, rA, zA);
            segment_position(R0, Z0, p0, R1, Z1, p1, ph + 0.5*h, rB, zB);
            segment_position(R0, Z0, p0, R1, Z1, p1, ph + h, rC, zC);

            double k1_00, k1_01, k1_10, k1_11;
            if (!deriv(rA, zA, ph, D00, D01, D10, D11,
                       k1_00, k1_01, k1_10, k1_11)) return;

            const double E00_2 = D00 + 0.5*h*k1_00;
            const double E01_2 = D01 + 0.5*h*k1_01;
            const double E10_2 = D10 + 0.5*h*k1_10;
            const double E11_2 = D11 + 0.5*h*k1_11;
            double k2_00, k2_01, k2_10, k2_11;
            if (!deriv(rB, zB, ph + 0.5*h, E00_2, E01_2, E10_2, E11_2,
                       k2_00, k2_01, k2_10, k2_11)) return;

            const double E00_3 = D00 + 0.5*h*k2_00;
            const double E01_3 = D01 + 0.5*h*k2_01;
            const double E10_3 = D10 + 0.5*h*k2_10;
            const double E11_3 = D11 + 0.5*h*k2_11;
            double k3_00, k3_01, k3_10, k3_11;
            if (!deriv(rB, zB, ph + 0.5*h, E00_3, E01_3, E10_3, E11_3,
                       k3_00, k3_01, k3_10, k3_11)) return;

            const double E00_4 = D00 + h*k3_00;
            const double E01_4 = D01 + h*k3_01;
            const double E10_4 = D10 + h*k3_10;
            const double E11_4 = D11 + h*k3_11;
            double k4_00, k4_01, k4_10, k4_11;
            if (!deriv(rC, zC, ph + h, E00_4, E01_4, E10_4, E11_4,
                       k4_00, k4_01, k4_10, k4_11)) return;

            const double h6 = h / 6.0;
            D00 += h6 * (k1_00 + 2.0*k2_00 + 2.0*k3_00 + k4_00);
            D01 += h6 * (k1_01 + 2.0*k2_01 + 2.0*k3_01 + k4_01);
            D10 += h6 * (k1_10 + 2.0*k2_10 + 2.0*k3_10 + k4_10);
            D11 += h6 * (k1_11 + 2.0*k2_11 + 2.0*k3_11 + k4_11);

            if (!std::isfinite(D00) || !std::isfinite(D01) ||
                !std::isfinite(D10) || !std::isfinite(D11)) {
                return;
            }
            ph += h;
        }

        DX_out[4*k+0] = D00; DX_out[4*k+1] = D01;
        DX_out[4*k+2] = D10; DX_out[4*k+3] = D11;
    }
}

// ---------------------------------------------------------------------------
// Progress delta_X along an already sampled orbit.
// ---------------------------------------------------------------------------
// Integrates the inhomogeneous first-order response equation
//
//   d(delta_X)/dphi = A(R,Z,phi) * delta_X + delta_f(R,Z,phi),
//
// where A = d(R*B_pol/B_phi)/d(R,Z) is evaluated from the base field and
// delta_f is the first-order variation of the field-line RHS at fixed (R,Z):
//
//   delta_f_R = R * (delta_BR * B_phi - BR * delta_B_phi) / B_phi^2
//   delta_f_Z = R * (delta_BZ * B_phi - BZ * delta_B_phi) / B_phi^2.
//
// The orbit samples are treated as the authoritative path and linearly
// interpolated inside each output segment; no field-line retracing is performed.
static inline void progress_delta_X_along_orbit(
    const double* R_traj, const double* Z_traj, const double* phi_traj,
    int n_pts,
    const double* delta_X0,
    double* delta_X_out,
    const double* BR, const double* BZ, const double* BPhi,
    const double* dBR, const double* dBZ, const double* dBPhi,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    double max_step,
    double field_period = 2.0 * M_PI)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (n_pts < 1) return;
    std::fill(delta_X_out, delta_X_out + 2 * n_pts, nan);

    if (!(max_step > 0.0) || !std::isfinite(max_step)) {
        max_step = 0.005;
    }

    double xR = delta_X0 ? delta_X0[0] : 0.0;
    double xZ = delta_X0 ? delta_X0[1] : 0.0;
    delta_X_out[0] = xR;
    delta_X_out[1] = xZ;

    auto eval_A_and_delta_f = [&](double R, double Z, double phi,
                                  double& A00, double& A01,
                                  double& A10, double& A11,
                                  double& dF0, double& dF1) -> bool {
        double BRv, dBR_dR, dBR_dZ;
        double BPhiv, dBPhi_dR, dBPhi_dZ;
        double BZv, dBZ_dR, dBZ_dZ;
        if (!interp3d_grad(BRv, dBR_dR, dBR_dZ, BR, R_grid,nR,Z_grid,nZ,Phi_grid,nPhi, R,Z,phi,field_period)) return false;
        if (!interp3d_grad(BPhiv, dBPhi_dR, dBPhi_dZ, BPhi, R_grid,nR,Z_grid,nZ,Phi_grid,nPhi, R,Z,phi,field_period)) return false;
        if (!interp3d_grad(BZv, dBZ_dR, dBZ_dZ, BZ, R_grid,nR,Z_grid,nZ,Phi_grid,nPhi, R,Z,phi,field_period)) return false;
        if (std::abs(BPhiv) <= 1e-12) return false;

        const double invBp = 1.0 / BPhiv;
        const double invBp2 = invBp / BPhiv;
        A00 = BRv*invBp + R*(dBR_dR*invBp - BRv*invBp2*dBPhi_dR);
        A01 = R*(dBR_dZ*invBp - BRv*invBp2*dBPhi_dZ);
        A10 = BZv*invBp + R*(dBZ_dR*invBp - BZv*invBp2*dBPhi_dR);
        A11 = R*(dBZ_dZ*invBp - BZv*invBp2*dBPhi_dZ);

        const double dBRv = interp3d(dBR, R_grid,nR,Z_grid,nZ,Phi_grid,nPhi, R,Z,phi,field_period);
        const double dBZv = interp3d(dBZ, R_grid,nR,Z_grid,nZ,Phi_grid,nPhi, R,Z,phi,field_period);
        const double dBPhiv = interp3d(dBPhi, R_grid,nR,Z_grid,nZ,Phi_grid,nPhi, R,Z,phi,field_period);
        if (!std::isfinite(dBRv) || !std::isfinite(dBZv) || !std::isfinite(dBPhiv)) return false;
        dF0 = R * (dBRv * BPhiv - BRv * dBPhiv) * invBp2;
        dF1 = R * (dBZv * BPhiv - BZv * dBPhiv) * invBp2;
        return std::isfinite(A00) && std::isfinite(A01) &&
               std::isfinite(A10) && std::isfinite(A11) &&
               std::isfinite(dF0) && std::isfinite(dF1);
    };

    auto segment_position = [](double R0, double Z0, double p0,
                               double R1, double Z1, double p1,
                               double p, double& R, double& Z) {
        const double ds = p1 - p0;
        double t = 0.0;
        if (std::abs(ds) > 1e-30) {
            t = (p - p0) / ds;
        }
        R = R0 + t * (R1 - R0);
        Z = Z0 + t * (Z1 - Z0);
    };

    auto deriv = [&](double R, double Z, double phi,
                     double xr, double xz,
                     double& dxr, double& dxz) -> bool {
        double A00, A01, A10, A11, dF0, dF1;
        if (!eval_A_and_delta_f(R, Z, phi, A00, A01, A10, A11, dF0, dF1)) return false;
        dxr = A00 * xr + A01 * xz + dF0;
        dxz = A10 * xr + A11 * xz + dF1;
        return std::isfinite(dxr) && std::isfinite(dxz);
    };

    for (int k = 1; k < n_pts; ++k) {
        const double R0 = R_traj[k-1], Z0 = Z_traj[k-1], p0 = phi_traj[k-1];
        const double R1 = R_traj[k],   Z1 = Z_traj[k],   p1 = phi_traj[k];
        const double ds = p1 - p0;

        if (!std::isfinite(R0) || !std::isfinite(Z0) || !std::isfinite(p0) ||
            !std::isfinite(R1) || !std::isfinite(Z1) || !std::isfinite(p1)) {
            return;
        }

        if (std::abs(ds) <= 1e-30) {
            delta_X_out[2*k+0] = xR;
            delta_X_out[2*k+1] = xZ;
            continue;
        }

        int ns = (int)std::ceil(std::abs(ds) / max_step);
        if (ns < 1) ns = 1;
        const double h = ds / (double)ns;
        double ph = p0;

        for (int s = 0; s < ns; ++s) {
            double rA, zA, rB, zB, rC, zC;
            segment_position(R0, Z0, p0, R1, Z1, p1, ph, rA, zA);
            segment_position(R0, Z0, p0, R1, Z1, p1, ph + 0.5*h, rB, zB);
            segment_position(R0, Z0, p0, R1, Z1, p1, ph + h, rC, zC);

            double k1R, k1Z;
            if (!deriv(rA, zA, ph, xR, xZ, k1R, k1Z)) return;
            double k2R, k2Z;
            if (!deriv(rB, zB, ph + 0.5*h, xR + 0.5*h*k1R, xZ + 0.5*h*k1Z, k2R, k2Z)) return;
            double k3R, k3Z;
            if (!deriv(rB, zB, ph + 0.5*h, xR + 0.5*h*k2R, xZ + 0.5*h*k2Z, k3R, k3Z)) return;
            double k4R, k4Z;
            if (!deriv(rC, zC, ph + h, xR + h*k3R, xZ + h*k3Z, k4R, k4Z)) return;

            const double h6 = h / 6.0;
            xR += h6 * (k1R + 2.0*k2R + 2.0*k3R + k4R);
            xZ += h6 * (k1Z + 2.0*k2Z + 2.0*k3Z + k4Z);
            if (!std::isfinite(xR) || !std::isfinite(xZ)) return;
            ph += h;
        }

        delta_X_out[2*k+0] = xR;
        delta_X_out[2*k+1] = xZ;
    }
}

// Periodic-cycle displacement evolution.
//
// Use this when delta_X0 is already the closed-cycle initial displacement
// delta_X_cyc(phi0).  The ODE is identical to progress_delta_X_along_orbit,
// but the naming records that phi_s and phi_e = phi_s + 2*pi*m move together
// along a periodic cycle.
static inline void evolve_delta_X_cycle_along_cycle(
    const double* R_traj, const double* Z_traj, const double* phi_traj,
    int n_pts,
    const double* delta_X0,
    double* delta_X_out,
    const double* BR, const double* BZ, const double* BPhi,
    const double* dBR, const double* dBZ, const double* dBPhi,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    double max_step,
    double field_period = 2.0 * M_PI)
{
	    progress_delta_X_along_orbit(
	        R_traj, Z_traj, phi_traj, n_pts, delta_X0, delta_X_out,
	        BR, BZ, BPhi, dBR, dBZ, dBPhi,
	        R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, max_step, field_period);
	}

// Compatibility alias for older callers.
static inline void evolve_delta_X_cycle_along_orbit(
    const double* R_traj, const double* Z_traj, const double* phi_traj,
    int n_pts,
    const double* delta_X0,
    double* delta_X_out,
    const double* BR, const double* BZ, const double* BPhi,
    const double* dBR, const double* dBZ, const double* dBPhi,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    double max_step,
    double field_period = 2.0 * M_PI)
{
    evolve_delta_X_cycle_along_cycle(
        R_traj, Z_traj, phi_traj, n_pts, delta_X0, delta_X_out,
        BR, BZ, BPhi, dBR, dBZ, dBPhi,
        R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, max_step, field_period);
}



static inline FixedPointResult newton_fixed_point_span(
    double R0, double Z0,
    double phi_section,     // Poincare section angle [rad]
    double map_span,        // total map span [rad], e.g. m * 2π/Nfp
    double DPhi,
    double fd_eps,          // finite-difference step [m]
    int    max_iter,
    double tol,             // convergence: |P^m(x)-x| < tol
    const double* BR, const double* BZ, const double* BPhi,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    double field_period = 2.0 * M_PI)
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
    const double seed_R = R0;
    const double seed_Z = Z0;
    const double grid_h = std::min(
        (nR > 1 ? std::abs(R_grid[1] - R_grid[0]) : 1.0),
        (nZ > 1 ? std::abs(Z_grid[1] - Z_grid[0]) : 1.0));
    const double trust_radius0 = std::max(8.0 * grid_h, 5.0e-4);
    double trust_radius = trust_radius0;
    const double min_R = R_grid[0], max_R = R_grid[nR - 1];
    const double min_Z = Z_grid[0], max_Z = Z_grid[nZ - 1];

    auto eval_residual = [&](double Rq, double Zq,
                             double& F0, double& F1,
                             double* DP_out) -> bool {
        if (!std::isfinite(Rq) || !std::isfinite(Zq) ||
            Rq < min_R || Rq > max_R || Zq < min_Z || Zq > max_Z) {
            return false;
        }
        double Rf = Rq, Zf = Zq;
        double DP_cur[4];
        if (!DX_pol_span(Rf, Zf, DP_cur, phi0, map_span, DPhi,
                         BR, BZ, BPhi, R_grid, nR, Z_grid, nZ, Phi_grid, nPhi,
                         field_period)) {
            return false;
        }
        F0 = Rf - Rq;
        F1 = Zf - Zq;
        if (!std::isfinite(F0) || !std::isfinite(F1)) {
            return false;
        }
        if (DP_out) {
            DP_out[0] = DP_cur[0]; DP_out[1] = DP_cur[1];
            DP_out[2] = DP_cur[2]; DP_out[3] = DP_cur[3];
        }
        return true;
    };

    for (int iter = 0; iter < max_iter; ++iter) {
        // Evaluate F(x) = P_span(x) - x and compute DPm via analytic DX_pol
        // evolution over the requested toroidal map span.
        double DP_cur[4];
        double F0 = 0.0, F1 = 0.0;
        if (!eval_residual(R, Z, F0, F1, DP_cur))
            return res;

        res.residual = std::sqrt(F0*F0 + F1*F1);
        if (res.residual < tol) {
            res.converged = 1;
            // Store DPm(φ_s) = DX_pol(φ_s, φ_s+map_span) from analytic evolution
            res.DPm[0]=DP_cur[0]; res.DPm[1]=DP_cur[1];
            res.DPm[2]=DP_cur[2]; res.DPm[3]=DP_cur[3];
            break;
        }

        // DF = DPm - I  (Jacobian of F = P^m(x) - x)
        double DF00 = DP_cur[0] - 1.0, DF01 = DP_cur[1];
        double DF10 = DP_cur[2],        DF11 = DP_cur[3] - 1.0;
        double det = DF00*DF11 - DF01*DF10;

        // Accept only residual-decreasing steps, with a seed-centred trust
        // region.  This prevents a near-correct O/X-cycle seed (e.g. X_old +
        // first-order delta_X_cyc) from being thrown to a different cycle by a
        // locally ill-conditioned full Newton step.
        bool accepted = false;
        double best_R = R, best_Z = Z, best_res = res.residual;
        double accepted_step_norm = grid_h;
        auto try_step = [&](double dR_in, double dZ_in, double& step_norm_out) -> bool {
            double dR = dR_in;
            double dZ = dZ_in;
            double step_norm = std::sqrt(dR*dR + dZ*dZ);
            if (!std::isfinite(step_norm) || step_norm <= 0.0) return false;
            if (step_norm > trust_radius) {
                const double scale = trust_radius / std::max(step_norm, 1e-300);
                dR *= scale;
                dZ *= scale;
                step_norm = trust_radius;
            }
            for (int ls = 0; ls < 12; ++ls) {
                const double alpha = std::ldexp(1.0, -ls);
                const double cand_R = R + alpha * dR;
                const double cand_Z = Z + alpha * dZ;
                const double seed_dist = std::sqrt(
                    (cand_R - seed_R) * (cand_R - seed_R) +
                    (cand_Z - seed_Z) * (cand_Z - seed_Z));
                if (seed_dist > trust_radius0) continue;
                double cF0 = 0.0, cF1 = 0.0;
                if (!eval_residual(cand_R, cand_Z, cF0, cF1, nullptr)) continue;
                const double cand_res = std::sqrt(cF0*cF0 + cF1*cF1);
                if (std::isfinite(cand_res) && cand_res < best_res) {
                    best_R = cand_R;
                    best_Z = cand_Z;
                    best_res = cand_res;
                    step_norm_out = alpha * step_norm;
                    return true;
                }
            }
            return false;
        };

        if (std::abs(det) >= 1e-20) {
            // Newton step: dx = -DF^{-1} * F
            const double dR = -(DF11*F0 - DF01*F1) / det;
            const double dZ = -(-DF10*F0 + DF00*F1) / det;
            accepted = try_step(dR, dZ, accepted_step_norm);
        }

        if (!accepted) {
            // Damped least-squares fallback for singular or ill-conditioned
            // DF=DPm-I.  It preserves the same line-search and seed-centred
            // trust region as Newton, so a bad local inverse cannot jump to a
            // different periodic orbit.
            auto dls_direction = [&](double lambda, double& dR, double& dZ) -> bool {
                const double g0 = DF00*F0 + DF10*F1;
                const double g1 = DF01*F0 + DF11*F1;
                const double l2 = lambda * lambda;
                const double A00 = DF00*DF00 + DF10*DF10 + l2;
                const double A01 = DF00*DF01 + DF10*DF11;
                const double A11 = DF01*DF01 + DF11*DF11 + l2;
                const double den = A00*A11 - A01*A01;
                if (!std::isfinite(den) || std::abs(den) < 1e-300) return false;
                dR = -( A11*g0 - A01*g1) / den;
                dZ = -(-A01*g0 + A00*g1) / den;
                return std::isfinite(dR) && std::isfinite(dZ);
            };
            const double jnorm = std::sqrt(
                DF00*DF00 + DF01*DF01 + DF10*DF10 + DF11*DF11);
            const double base_lambda = std::max(1.0, jnorm);
            const double scales[] = {1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2};
            for (double scale : scales) {
                double dR = 0.0, dZ = 0.0;
                if (dls_direction(scale * base_lambda, dR, dZ) &&
                    try_step(dR, dZ, accepted_step_norm)) {
                    accepted = true;
                    break;
                }
            }
        }

        if (!accepted) {
            // Picard fallback: x <- P^m(x) = x + F.  This is also damped and
            // residual-tested, so it cannot degrade a good first-order seed.
            accepted = try_step(F0, F1, accepted_step_norm);
        }

        if (!accepted) {
            return res;
        }
        R = best_R;
        Z = best_Z;
        res.residual = best_res;
        trust_radius = std::min(trust_radius0, std::max(2.0 * accepted_step_norm, 2.0 * grid_h));
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

    if (disc >= 0.0) {
        const double ev0 = std::abs(res.eig_r[0]);
        const double ev1 = std::abs(res.eig_r[1]);
        const double eps = 1.0e-8;
        const bool area_like = std::isfinite(det2) && std::abs(det2 - 1.0) <= 5.0e-2;
        res.point_type = (
            area_like && (
                (ev0 > 1.0 + eps && ev1 < 1.0 - eps) ||
                (ev1 > 1.0 + eps && ev0 < 1.0 - eps)
            )
        ) ? 1 : 0;
    } else {
        res.point_type = 0;
    }
    return res;
}

static inline FixedPointResult newton_fixed_point(
    double R0, double Z0,
    double phi_section,
    int    m_turns,
    double DPhi,
    double fd_eps,
    int    max_iter,
    double tol,
    const double* BR, const double* BZ, const double* BPhi,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    double field_period = 2.0 * M_PI)
{
    return newton_fixed_point_span(
        R0, Z0, phi_section, double(m_turns) * 2.0 * M_PI,
        DPhi, fd_eps, max_iter, tol,
        BR, BZ, BPhi,
        R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, field_period);
}

void find_fixed_points_batch(
    const double* R_seeds, const double* Z_seeds, int N_seeds,
    double phi_section,
    int    m_turns,
    double DPhi,
    double fd_eps,
    int    max_iter,
    double tol,
    const double* BR, const double* BZ, const double* BPhi,
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
                BR, BZ, BPhi,
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

void find_fixed_points_batch_span(
    const double* R_seeds, const double* Z_seeds, int N_seeds,
    double phi_section,
    double map_span,
    double DPhi,
    double fd_eps,
    int    max_iter,
    double tol,
    const double* BR, const double* BZ, const double* BPhi,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    int n_threads,
    double* R_out, double* Z_out,
    double* residual_out,
    int*    converged_out,
    double* DPm_out,
    double* eig_r_out,
    double* eig_i_out,
    int*    point_type_out,
    double field_period = 2.0 * M_PI)
{
    if (n_threads <= 0)
        n_threads = (int)std::thread::hardware_concurrency();

    BS::thread_pool pool((unsigned int)n_threads);
    pool.parallelize_loop(0, N_seeds, [&](int i_start, int i_end) {
        for (int i = i_start; i < i_end; ++i) {
            auto r = newton_fixed_point_span(
                R_seeds[i], Z_seeds[i],
                phi_section, map_span, DPhi, fd_eps, max_iter, tol,
                BR, BZ, BPhi,
                R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, field_period);

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
    const double* BR, const double* BZ, const double* BPhi,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    int    n_out,
    double* R_traj, double* Z_traj, double* phi_traj,
    double* DPm_traj,  // [n_out * 4]
    int*    alive_out,
    double field_period = 2.0 * M_PI)
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
	    double phi_end = phi0 + phi_span;
	    const double dir = (phi_span >= 0.0) ? 1.0 : -1.0;
	    const double step_abs = std::abs(DPhi);
	    const double out_abs = std::abs(dphi_out);
	    const double dphi_out_signed = dir * out_abs;

    // Helper: compute DPm(φ) = DX_pol(φ, φ+2π·m_turns_DPm) at the
    // current trajectory point.  Uses DX_pol_m_turns which integrates the
    // analytic variational equation  d(DX_pol)/dφ = J·DX_pol  alongside
    // the field-line trajectory in a single pass.
    auto compute_DPm = [&](double r, double z, double phi_sec,
                            double* DPm) -> bool {
        double Rf = r, Zf = z;
        return DX_pol_m_turns(Rf, Zf, DPm, mod2pi(phi_sec), m_turns_DPm, DPhi,
                              BR, BZ, BPhi, R_grid, nR, Z_grid, nZ, Phi_grid, nPhi,
                              field_period);
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
	        phi_next_out += dphi_out_signed;
	    }

	    while (out_idx < n_out &&
	           (dir > 0.0 ? phi < phi_end - 1e-12 : phi > phi_end + 1e-12)) {
        double target = phi_next_out;
        if (dir > 0.0) target = std::min(target, phi_end);
        else           target = std::max(target, phi_end);

        while (dir > 0.0 ? phi < target - 1e-12 : phi > target + 1e-12) {
            double step = dir * std::min(step_abs, std::abs(target - phi));
            rk4_step(R, Z, phi, step, BR, BZ, BPhi,
                     R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, field_period);
            phi += step;

            if (!std::isfinite(R) || !std::isfinite(Z) ||
                R < R_grid[0] || R > R_grid[nR-1] ||
                Z < Z_grid[0] || Z > Z_grid[nZ-1])
                return;
        }

        R_traj[out_idx] = R; Z_traj[out_idx] = Z; phi_traj[out_idx] = target;
        double DPm[4];
        if (compute_DPm(R, Z, mod2pi(target), DPm)) {
            for (int k=0;k<4;k++) DPm_traj[4*out_idx+k] = DPm[k];
        }
        alive_out[out_idx] = 1;
        out_idx++;
        if (std::abs(target - phi_end) <= 1e-12) {
            break;
        }
        phi_next_out += dphi_out_signed;
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
    const double* BR, const double* BZ, const double* BPhi,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    double beta, double R_ax, double Z_ax, double a_eff,
    double alpha, double p0,
    double field_period = 2.0 * M_PI)
{
    // Effective dR/dphi and dZ/dphi with beta-corrected field
    auto deriv = [&](double r, double z, double p,
                     double& dR_out, double& dZ_out) {
        double bp = interp3d(BPhi, R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, r, z, p, field_period);
        double br = interp3d(BR,   R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, r, z, p, field_period);
        double bz = interp3d(BZ,   R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, r, z, p, field_period);

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
    const double* BR, const double* BZ, const double* BPhi,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    const double* wall_R, const double* wall_Z, int n_wall,
    double beta, double R_ax, double Z_ax, double a_eff,
    double alpha, double p0,
    int* poi_counts,
    double* poi_R_flat,
    double* poi_Z_flat,
    double field_period = 2.0 * M_PI)
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
                      BR, BZ, BPhi,
                      R_grid, nR, Z_grid, nZ, Phi_grid, nPhi,
                      beta, R_ax, Z_ax, a_eff, alpha, p0, field_period);
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
    const double* BR, const double* BZ, const double* BPhi,
    const double* R_grid, int nR,
    const double* Z_grid, int nZ,
    const double* Phi_grid, int nPhi,
    const double* wall_R, const double* wall_Z, int n_wall,
    double beta, double R_ax, double Z_ax, double a_eff,
    double alpha_pressure, double B_ref,
    int n_threads,
    int* poi_counts,
    double* poi_R_flat,
    double* poi_Z_flat,
    double field_period = 2.0 * M_PI)
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
                BR, BZ, BPhi,
                R_grid, nR, Z_grid, nZ, Phi_grid, nPhi,
                wall_R, wall_Z, n_wall,
                beta, R_ax, Z_ax, a_eff, alpha_pressure, p0,
                poi_counts, poi_R_flat, poi_Z_flat, field_period);
        }
    }).wait();
}

} // namespace cyna
