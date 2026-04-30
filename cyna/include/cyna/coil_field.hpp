// cyna/coil_field.hpp
// ====================
// CPU (OpenMP) implementations of magnetic field computation for coil systems.
//
// Provides:
//   cyna::ellipk_f / ellipe_f          — float32 AGM elliptic integrals
//   cyna::circular_coil_BrhoBz_f       — analytic ring-coil field (local cylindrical)
//   cyna::circular_coil_field_cpu      — ring-coil field on a point cloud (OpenMP)
//   cyna::biot_savart_field_cpu        — Biot-Savart for filamentary coil (OpenMP)
//
// All functions are header-only and depend only on <cmath> and (optionally)
// OpenMP.  They serve as the CPU fallback when CUDA is unavailable.
//
// Coordinate convention
// ---------------------
// Field points are supplied as (N, 3) float32 arrays in Cartesian (x, y, z).
// Outputs are in global cylindrical (BR, BPhi, BZ) where
//   BR   = B_x cos φ + B_y sin φ
//   BPhi = −B_x sin φ + B_y cos φ
//   BZ   = B_z
// with φ = atan2(y, x).
//
// Circular coil convention
// ------------------------
// A circular coil is described by its centre (cx,cy,cz), unit normal (nx,ny,nz),
// radius a, and current I [A].  Positive I with normal pointing along +z produces
// B pointing along +z on the axis (right-hand rule).

#pragma once
#include <cmath>
#include <cstdint>
#include <algorithm>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace cyna {

// ---------------------------------------------------------------------------
// AGM elliptic integrals (float32)
// ---------------------------------------------------------------------------

/// Complete elliptic integral K(m), float32, 8 AGM iterations.
/// Relative error < 7e-8 for m ∈ [0, 1).
inline float ellipk_f(float m) {
    float a = 1.0f;
    float b = std::sqrt(std::fmax(1.0f - m, 0.0f));
    for (int i = 0; i < 8; ++i) {
        float an = 0.5f * (a + b);
        b = std::sqrt(a * b);
        a = an;
    }
    return (1.5707963267948966f) / a;   // π/2 / AGM
}

/// Complete elliptic integral E(m), float32, 10 AGM iterations.
inline float ellipe_f(float m) {
    float a = 1.0f;
    float b = std::sqrt(std::fmax(1.0f - m, 0.0f));
    float S = 0.5f * m;
    float power = 1.0f;
    for (int i = 0; i < 10; ++i) {
        float c  = 0.5f * (a - b);
        float an = 0.5f * (a + b);
        b = std::sqrt(a * b);
        a = an;
        S += power * c * c;
        power *= 2.0f;
    }
    return (1.5707963267948966f) / a * (1.0f - S);
}

// ---------------------------------------------------------------------------
// Analytic ring-coil field in local cylindrical frame
// ---------------------------------------------------------------------------

/// Compute (Brho, Bz) [Tesla] of a circular coil (radius a, current I) at
/// local cylindrical point (rho, z).  Uses the Smythe / Griffiths formula.
inline void circular_coil_BrhoBz_f(
        float a, float rho, float z, float I,
        float& Brho, float& Bz_out) noexcept
{
    constexpr float MU0_2PI = 2.0e-7f;   // μ₀ / (2π)
    if (rho < 1e-9f) {                   // on-axis
        float r = std::sqrt(a * a + z * z);
        Brho   = 0.0f;
        Bz_out = MU0_2PI * I * a * a / (r * r * r);
        return;
    }
    float Q  = (a + rho) * (a + rho) + z * z;
    float sQ = std::sqrt(Q);
    float k2 = 4.0f * a * rho / Q;
    if (k2 >= 1.0f) k2 = 1.0f - 1.2e-7f;
    if (k2 <= 0.0f) { Brho = 0.0f; Bz_out = 0.0f; return; }
    float K  = ellipk_f(k2);
    float E  = ellipe_f(k2);
    float dE = (a - rho) * (a - rho) + z * z;
    if (dE < 1e-20f) dE = 1e-20f;
    float c  = MU0_2PI * I;
    Brho   = c * z / (rho * sQ) * (-K + E * (a * a + rho * rho + z * z) / dE);
    Bz_out = c / sQ             * ( K + E * (a * a - rho * rho - z * z) / dE);
}

// ---------------------------------------------------------------------------
// Circular coil field on a Cartesian point cloud (OpenMP)
// ---------------------------------------------------------------------------

/// Compute the magnetic field of one circular coil at N Cartesian field points.
///
/// @param cx,cy,cz   Coil centre (m)
/// @param nx,ny,nz   Coil normal — unit vector (right-hand rule defines current sense)
/// @param a          Coil radius (m)
/// @param current    Current (A, signed)
/// @param xyz_pts    (N, 3) float32 C-contiguous array — Cartesian field points
/// @param N          Number of field points
/// @param BR         (N,) output — B_R in Tesla
/// @param BPhi       (N,) output — B_φ in Tesla
/// @param BZ         (N,) output — B_Z in Tesla
inline void circular_coil_field_cpu(
        float cx, float cy, float cz,
        float nx, float ny, float nz,
        float a,  float current,
        const float* xyz_pts, int N,
        float* BR,
        float* BPhi,
        float* BZ) noexcept
{
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < N; ++i) {
        float px = xyz_pts[3 * i];
        float py = xyz_pts[3 * i + 1];
        float pz = xyz_pts[3 * i + 2];

        // Decompose d = P - centre  into  (rho-vec, z-local)
        float dx   = px - cx, dy = py - cy, dz_d = pz - cz;
        float z_l  = nx * dx + ny * dy + nz * dz_d;
        float rx   = dx   - z_l * nx;
        float ry   = dy   - z_l * ny;
        float rz   = dz_d - z_l * nz;
        float rho  = std::sqrt(rx * rx + ry * ry + rz * rz);

        float Brho_l, Bz_l;
        circular_coil_BrhoBz_f(a, rho, z_l, current, Brho_l, Bz_l);

        // B in Cartesian lab:  Brho_l * rho_hat + Bz_l * n
        float bx, by, bz_lab;
        if (rho > 1e-9f) {
            float inv = 1.0f / rho;
            bx    = Brho_l * rx * inv + Bz_l * nx;
            by    = Brho_l * ry * inv + Bz_l * ny;
            bz_lab= Brho_l * rz * inv + Bz_l * nz;
        } else {
            bx    = Bz_l * nx;
            by    = Bz_l * ny;
            bz_lab= Bz_l * nz;
        }

        // Convert to cylindrical
        float phi  = std::atan2(py, px);
        float cp   = std::cos(phi);
        float sp   = std::sin(phi);
        BR[i]   =  bx * cp + by * sp;
        BPhi[i] = -bx * sp + by * cp;
        BZ[i]   =  bz_lab;
    }
}

// ---------------------------------------------------------------------------
// Biot-Savart for one arbitrary filamentary coil (OpenMP)
// ---------------------------------------------------------------------------

/// Compute the Biot-Savart field of one filamentary coil defined by N_seg
/// straight segments.
///
/// @param seg_starts  (N_seg, 3) float32 — segment start points (m)
/// @param seg_ends    (N_seg, 3) float32 — segment end   points (m)
/// @param N_seg       Number of segments
/// @param current     Current (A, signed)
/// @param xyz_pts     (N, 3) float32 — Cartesian field points
/// @param N           Number of field points
/// @param BR,BPhi,BZ  (N,) outputs in Tesla, cylindrical
inline void biot_savart_field_cpu(
        const float* seg_starts,
        const float* seg_ends,
        int N_seg, float current,
        const float* xyz_pts, int N,
        float* BR,
        float* BPhi,
        float* BZ) noexcept
{
    constexpr float MU0_4PI = 1.0e-7f;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < N; ++i) {
        float px = xyz_pts[3 * i];
        float py = xyz_pts[3 * i + 1];
        float pz = xyz_pts[3 * i + 2];
        float bx = 0.0f, by = 0.0f, bz = 0.0f;

        for (int s = 0; s < N_seg; ++s) {
            float x0 = seg_starts[3*s],   y0 = seg_starts[3*s+1], z0 = seg_starts[3*s+2];
            float x1 = seg_ends[3*s],     y1 = seg_ends[3*s+1],   z1 = seg_ends[3*s+2];

            float dlx = x1-x0, dly = y1-y0, dlz = z1-z0;
            float r0x = px-x0, r0y = py-y0, r0z = pz-z0;
            float r1x = px-x1, r1y = py-y1, r1z = pz-z1;

            float R0 = std::sqrt(r0x*r0x + r0y*r0y + r0z*r0z);
            float R1 = std::sqrt(r1x*r1x + r1y*r1y + r1z*r1z);
            if (R0 < 5e-4f || R1 < 5e-4f) continue;

            float dot  = r0x*r1x + r0y*r1y + r0z*r1z;
            float dnom = R0 * R1 * (R0 * R1 + dot);
            if (std::fabs(dnom) < 1e-30f) continue;

            float fac = MU0_4PI * current * (R0 + R1) / dnom;
            bx += fac * (dly * r0z - dlz * r0y);
            by += fac * (dlz * r0x - dlx * r0z);
            bz += fac * (dlx * r0y - dly * r0x);
        }

        float phi  = std::atan2(py, px);
        float cp   = std::cos(phi);
        float sp   = std::sin(phi);
        BR[i]   =  bx * cp + by * sp;
        BPhi[i] = -bx * sp + by * cp;
        BZ[i]   =  bz;
    }
}

} // namespace cyna
