// cyna/coil_field_grid.hpp
// ==========================
// Coil-set -> cylindrical magnetic-field grid, computed in C++ (OpenMP CPU).
//
// This is the cyna-native replacement for the Python/JAX Biot-Savart grid
// fill that dominates the start-up cost of a coil-set field-line trace.
//
// Inputs
// ------
//   segments : (K, S, 3) double, C order.  K filamentary coils, each given as
//              a polyline of S Cartesian points (metres) in the lab frame
//              x = R cos(phi), y = R sin(phi), z = Z.  Consecutive points
//              define S-1 straight segments; the polyline is *not* implicitly
//              closed (repeat the first point in each coil to close it).
//   currents : (K,) double, signed ampere-turns per coil.
//   R_grid   : (nR,) double, strictly increasing, R > 0 (metres).
//   Z_grid   : (nZ,) double, strictly increasing (metres).
//   Phi_grid : (nPhi,) double, increasing toroidal angles (radians).  With
//              phi_closed=true the last plane must duplicate the first one at
//              the periodic endpoint, i.e. Phi_grid[-1] - Phi_grid[0] equals
//              exactly 2*pi/nfp.  With phi_closed=false the planes are
//              nPhi distinct samples of one field period and no duplicate is
//              supplied (the pivot2t layout).
//   nfp      : number of field periods used to validate the closed Phi span;
//              it is explicit and never inferred from Phi_grid.
//
// Outputs
// -------
//   BR, BZ, BPhi : (nR, nZ, nPhi) double, C order (phi contiguous), Tesla,
//                  in the global cylindrical components
//                      BR   =  Bx cos(phi) + By sin(phi)
//                      BPhi = -Bx sin(phi) + By cos(phi)
//                      BZ   =  Bz
//                  which is the same component convention used by
//                  pivot2t.jaxfield and by pyna.toroidal.flt.
//
// Physics
// -------
//   Exact finite-straight-segment Biot-Savart sum.  For a segment P0 -> P1
//   carrying current I and a field point P, with a = P - P0, b = P - P1,
//   L = P1 - P0 = a - b:
//
//       B = mu0 I / (4 pi) * (L x a) * (|a| + |b|)
//                          / ( |a| |b| (|a||b| + a.b) )
//
//   This is algebraically identical to the Nelson form used by pivot2t
//   (cross(L, a) * [a.L/|a| - b.L/|b|] / |cross(L, a)|^2) and to the float32
//   cyna::biot_savart_field_cpu kernel.
//
//   The field is evaluated for every grid node independently; OpenMP
//   parallelises over (iR, iZ) rows.  The last phi plane of a closed grid is
//   a byte-for-byte copy of the first plane, so periodic tracing never sees a
//   seam.
//
// CUDA
// ----
//   This header is CPU-only and is always compiled.  The optional runtime
//   CUDA backend (_cyna_cuda_backend.*) is not required and is never loaded by
//   the pyna wrapper's CPU path.
//
// Header-only, depends on <cmath>, <vector>, <stdexcept> and optionally
// OpenMP.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace cyna {

/// mu0 / (4 pi) in SI units (T m / A).
inline constexpr double coil_field_mu0_over_4pi = 1.0e-7;

/// Exact field of one finite straight segment at one Cartesian point.
///
/// @param x0,y0,z0   segment start (m)
/// @param x1,y1,z1   segment end   (m)
/// @param current    signed current (A)
/// @param px,py,pz   field point (m)
/// @param bx,by,bz   accumulated Cartesian field (T), added in place
/// @return false when the point is numerically on the segment axis (the
///         singular contribution is skipped, matching the float32 path)
inline bool add_finite_segment_field(
    double x0, double y0, double z0,
    double x1, double y1, double z1,
    double current,
    double px, double py, double pz,
    double& bx, double& by, double& bz) noexcept
{
    const double ax = px - x0, ay = py - y0, az = pz - z0;  // a = P - P0
    const double bxv = px - x1, byv = py - y1, bzv = pz - z1;  // b = P - P1
    const double lx = x1 - x0, ly = y1 - y0, lz = z1 - z0;  // L = P1 - P0

    const double ra = std::sqrt(ax * ax + ay * ay + az * az);
    const double rb = std::sqrt(bxv * bxv + byv * byv + bzv * bzv);

    // Exact coincidence (or sub-femtometre distance) with a segment endpoint.
    // A grid node sitting exactly on a filament is singular; skip it exactly
    // like cyna::biot_savart_field_cpu does rather than emitting inf/NaN.
    constexpr double kCoincident = 1.0e-12;  // metres
    if (!(ra > kCoincident) || !(rb > kCoincident))
        return false;

    const double dot = ax * bxv + ay * byv + az * bzv;
    const double denom = ra * rb * (ra * rb + dot);
    if (!(std::abs(denom) > 0.0) || !std::isfinite(denom))
        return false;

    // L x a == a x b for b = a - L.
    const double cx = ly * az - lz * ay;
    const double cy = lz * ax - lx * az;
    const double cz = lx * ay - ly * ax;

    const double fac =
        coil_field_mu0_over_4pi * current * (ra + rb) / denom;
    bx += fac * cx;
    by += fac * cy;
    bz += fac * cz;
    return true;
}

/// Coil-set field on a cylindrical grid (CPU, OpenMP).
///
/// The caller owns all pointers; no allocation of the output arrays happens
/// here.  Output arrays are laid out with phi as the fastest index, matching
/// cyna's (nR, nZ, nPhi) C-order field arrays.
///
/// @param segments             (K*S*3) row-major Cartesian polylines
/// @param n_coils              K
/// @param n_points_per_coil    S (>= 2)
/// @param currents             (K,)
/// @param R_grid, nR           radial grid
/// @param Z_grid, nZ           vertical grid
/// @param Phi_grid, nPhi       toroidal grid
/// @param BR, BZ, BPhi         (nR*nZ*nPhi) outputs
/// @param phi_closed           true if Phi_grid includes the periodic
///                             duplicate endpoint
/// @param n_threads            OpenMP thread count; <= 0 uses the default
inline void coil_field_grid_cpu(
    const double* segments,
    int n_coils,
    int n_points_per_coil,
    const double* currents,
    const double* R_grid,
    int nR,
    const double* Z_grid,
    int nZ,
    const double* Phi_grid,
    int nPhi,
    double* BR,
    double* BZ,
    double* BPhi,
    bool phi_closed = true,
    int n_threads = -1)
{
    if (n_coils < 1 || n_points_per_coil < 2)
        throw std::invalid_argument(
            "coil_field_grid_cpu: segments must be (K>=1, S>=2, 3)");
    if (nR < 1 || nZ < 1 || nPhi < 1)
        throw std::invalid_argument(
            "coil_field_grid_cpu: grid axes must be non-empty");

    // Flatten the polylines into (K*(S-1)) straight segments and attach each
    // segment's coil current so the hot loop never indexes currents[].
    const std::size_t n_segments =
        static_cast<std::size_t>(n_coils) *
        static_cast<std::size_t>(n_points_per_coil - 1);
    std::vector<double> sx(n_segments), sy(n_segments), sz(n_segments);
    std::vector<double> ex(n_segments), ey(n_segments), ez(n_segments);
    std::vector<double> seg_current(n_segments);

    std::size_t seg = 0;
    for (int k = 0; k < n_coils; ++k) {
        const double* coil = segments +
            static_cast<std::size_t>(k) * static_cast<std::size_t>(n_points_per_coil) * 3;
        const double current = currents[k];
        for (int s = 0; s + 1 < n_points_per_coil; ++s) {
            const double* p0 = coil + static_cast<std::size_t>(s) * 3;
            const double* p1 = p0 + 3;
            sx[seg] = p0[0]; sy[seg] = p0[1]; sz[seg] = p0[2];
            ex[seg] = p1[0]; ey[seg] = p1[1]; ez[seg] = p1[2];
            seg_current[seg] = current;
            ++seg;
        }
    }
    if (seg != n_segments)
        throw std::logic_error("coil_field_grid_cpu: internal segment count mismatch");

    // cos/sin per toroidal plane, reused by every (R, Z) row.
    std::vector<double> cos_phi(static_cast<std::size_t>(nPhi));
    std::vector<double> sin_phi(static_cast<std::size_t>(nPhi));
    for (int iP = 0; iP < nPhi; ++iP) {
        const double phi = Phi_grid[iP];
        cos_phi[static_cast<std::size_t>(iP)] = std::cos(phi);
        sin_phi[static_cast<std::size_t>(iP)] = std::sin(phi);
    }

    const std::size_t n_seg = n_segments;
    const double* sx_p = sx.data();
    const double* sy_p = sy.data();
    const double* sz_p = sz.data();
    const double* ex_p = ex.data();
    const double* ey_p = ey.data();
    const double* ez_p = ez.data();
    const double* cur_p = seg_current.data();
    const double* cp = cos_phi.data();
    const double* sp = sin_phi.data();

    auto compute_row = [&](int iR, int iZ) {
        const double R = R_grid[iR];
        const double Z = Z_grid[iZ];
        const std::size_t row =
            (static_cast<std::size_t>(iR) * static_cast<std::size_t>(nZ) +
             static_cast<std::size_t>(iZ)) * static_cast<std::size_t>(nPhi);
        for (int iP = 0; iP < nPhi; ++iP) {
            const double c = cp[iP];
            const double s = sp[iP];
            const double px = R * c;
            const double py = R * s;
            const double pz = Z;

            double bx = 0.0, by = 0.0, bz = 0.0;
            for (std::size_t is = 0; is < n_seg; ++is) {
                add_finite_segment_field(
                    sx_p[is], sy_p[is], sz_p[is],
                    ex_p[is], ey_p[is], ez_p[is],
                    cur_p[is], px, py, pz,
                    bx, by, bz);
            }

            const std::size_t idx = row + static_cast<std::size_t>(iP);
            BR[idx] = bx * c + by * s;
            BPhi[idx] = -bx * s + by * c;
            BZ[idx] = bz;
        }
    };

#ifdef _OPENMP
    const int omp_threads = (n_threads > 0) ? n_threads : omp_get_max_threads();
    #pragma omp parallel for collapse(2) schedule(static) num_threads(omp_threads)
    for (int iR = 0; iR < nR; ++iR) {
        for (int iZ = 0; iZ < nZ; ++iZ) {
            compute_row(iR, iZ);
        }
    }
#else
    // Portable fallback for a build without OpenMP: the same row
    // decomposition, executed on std::thread workers.
    const unsigned int hardware = std::thread::hardware_concurrency();
    int workers = (n_threads > 0)
        ? n_threads
        : (hardware > 0 ? static_cast<int>(hardware) : 1);
    const int total_rows = nR * nZ;
    if (workers > total_rows) workers = total_rows;
    if (workers < 1) workers = 1;
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(workers));
    for (int t = 0; t < workers; ++t) {
        const int begin = static_cast<int>(
            (static_cast<long long>(total_rows) * t) / workers);
        const int end = static_cast<int>(
            (static_cast<long long>(total_rows) * (t + 1)) / workers);
        pool.emplace_back([&, begin, end]() {
            for (int row_index = begin; row_index < end; ++row_index)
                compute_row(row_index / nZ, row_index % nZ);
        });
    }
    for (auto& worker : pool)
        worker.join();
#endif

    // The closed periodic grid duplicates its first plane at the endpoint.
    // Copy it byte-for-byte so the seam is exact (and skip re-evaluating).
    if (phi_closed && nPhi > 1) {
        const std::size_t planes = static_cast<std::size_t>(nR) *
                                   static_cast<std::size_t>(nZ);
        for (std::size_t row = 0; row < planes; ++row) {
            const std::size_t first = row * static_cast<std::size_t>(nPhi);
            const std::size_t last = first + static_cast<std::size_t>(nPhi) - 1;
            BR[last] = BR[first];
            BZ[last] = BZ[first];
            BPhi[last] = BPhi[first];
        }
    }
}

}  // namespace cyna
