#pragma once
// poincare.hpp — self-contained Poincaré field-line tracer
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
// Phi convention — matches topoquest's scipy setup exactly:
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

} // namespace cyna
