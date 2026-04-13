"""
pyna.toroidal.coils.cuda_kernels
================================
CuPy RawKernel implementations for magnetic field computation.

Two kernels are provided, compiled at first use via NVRTC (no separate build
step required):

* ``circular_coil_field_gpu`` — analytic Biot-Savart for a circular (ring)
  current loop, using the Smythe/Griffiths elliptic-integral formula.
  Single-precision throughout; ~10 ms per coil at 10 M field points on an
  RTX 3060.

* ``biot_savart_field_gpu`` — numerical Biot-Savart for an arbitrary
  filamentary coil given as N_seg line segments.  ~100 ms per TF coil
  (~200-340 segments) at 10 M field points.

Both kernels launch one CUDA thread per field point.  The field points are
expected as a (N, 3) float32 C-contiguous CuPy array in Cartesian (x,y,z)
coordinates.  Results are returned as three (N,) float32 CuPy arrays in
cylindrical (BR, BPhi, BZ).

CPU fallbacks (OpenMP, if cyna is available, otherwise single-threaded NumPy)
are provided for environments without a GPU.
"""
from __future__ import annotations

import math
import warnings
import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None          # type: ignore[assignment]
    _HAS_CUPY = False

# ── CUDA kernel source strings ────────────────────────────────────────────────
# Written in plain CUDA C++ (no system headers) so NVRTC can compile them
# without a full CUDA installation at runtime.

_CIRCULAR_COIL_KERNEL_SRC = r"""
// Analytic circular-coil field kernel.
// One thread per field point.  Single precision throughout.
//
// Coil: center (cx,cy,cz), unit normal (nx,ny,nz), radius a, current I [A].
// Field point: (xyz[3*i], xyz[3*i+1], xyz[3*i+2]) in Cartesian metres.
// Output: (BR[i], BPhi[i], BZ[i]) in Tesla, cylindrical w.r.t. global z-axis.

// AGM elliptic integral K(m), float32, 8 iterations (~7e-8 relative error)
__device__ float _kf(float m) {
    float a = 1.0f, b = sqrtf(fmaxf(1.0f - m, 0.0f));
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float an = 0.5f * (a + b);
        b = sqrtf(a * b);
        a = an;
    }
    return (1.5707963267948966f) / a;   // pi/2 / AGM
}

// AGM elliptic integral E(m), float32, 10 iterations
__device__ float _ef(float m) {
    float a = 1.0f, b = sqrtf(fmaxf(1.0f - m, 0.0f));
    float S = 0.5f * m, power = 1.0f;
#pragma unroll
    for (int i = 0; i < 10; ++i) {
        float c  = 0.5f * (a - b);
        float an = 0.5f * (a + b);
        b = sqrtf(a * b);
        a = an;
        S += power * c * c;
        power *= 2.0f;
    }
    return (1.5707963267948966f) / a * (1.0f - S);
}

// (Brho, Bz) of circular coil (radius a, current I) at local cylindrical (rho, z)
__device__ void _coil_BrhoBz(float a, float rho, float z, float I,
                               float &Brho, float &Bz_) {
    const float MU0_2PI = 2.0e-7f;   // mu0/(2*pi)
    if (rho < 1e-9f) {               // on-axis
        float r = sqrtf(a*a + z*z);
        Brho = 0.0f;
        Bz_  = MU0_2PI * I * a * a / (r * r * r);  // (mu0*I)/(2) * a^2/r^3
        return;
    }
    float Q  = (a + rho) * (a + rho) + z * z;
    float sQ = sqrtf(Q);
    float k2 = 4.0f * a * rho / Q;
    if (k2 >= 1.0f) k2 = 1.0f - 1.2e-7f;
    float K  = _kf(k2), E = _ef(k2);
    float dE = (a - rho) * (a - rho) + z * z;
    if (dE < 1e-18f) dE = 1e-18f;
    float c = MU0_2PI * I;
    Brho = c * z / (rho * sQ) * (-K + E * (a*a + rho*rho + z*z) / dE);
    Bz_  = c / sQ           * ( K + E * (a*a - rho*rho - z*z) / dE);
}

extern "C" __global__ void circular_coil_kernel(
    float cx, float cy, float cz,
    float nx, float ny, float nz,
    float a,  float current,
    const float* __restrict__ xyz,   // (N, 3)
    int N,
    float* __restrict__ BR,
    float* __restrict__ BPhi,
    float* __restrict__ BZ)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float px = xyz[3*i], py = xyz[3*i+1], pz = xyz[3*i+2];

    // Vector from coil centre to field point
    float dx = px - cx, dy = py - cy, dz = pz - cz;

    // Decompose: z_local = n·d,  rho_vec = d - z_local*n
    float z_loc = nx*dx + ny*dy + nz*dz;
    float rx = dx - z_loc*nx, ry = dy - z_loc*ny, rz = dz - z_loc*nz;
    float rho = sqrtf(rx*rx + ry*ry + rz*rz);

    float Brho_loc, Bz_loc;
    _coil_BrhoBz(a, rho, z_loc, current, Brho_loc, Bz_loc);

    // B in Cartesian lab frame:
    //   Brho_loc * rho_hat  +  Bz_loc * n
    float bx, by, bz_lab;
    if (rho > 1e-9f) {
        float inv = 1.0f / rho;
        bx    = Brho_loc * rx * inv + Bz_loc * nx;
        by    = Brho_loc * ry * inv + Bz_loc * ny;
        bz_lab= Brho_loc * rz * inv + Bz_loc * nz;
    } else {
        bx    = Bz_loc * nx;
        by    = Bz_loc * ny;
        bz_lab= Bz_loc * nz;
    }

    // Cylindrical at field point
    float phi = atan2f(py, px);
    float cp_ = cosf(phi), sp_ = sinf(phi);
    BR[i]   = bx * cp_ + by * sp_;
    BPhi[i] = -bx * sp_ + by * cp_;
    BZ[i]   = bz_lab;
}
"""

_BIOT_SAVART_KERNEL_SRC = r"""
// Biot-Savart kernel for arbitrary filamentary coil.
// Coil given as N_seg line segments (start, end points).
// One thread per field point; inner loop over segments.
//
// seg_starts, seg_ends: (N_seg, 3) float32 Cartesian metres
// current: scalar Amperes
// xyz: (N_pts, 3) float32 Cartesian metres
// Output: (BR, BPhi, BZ) Tesla, cylindrical

extern "C" __global__ void biot_savart_kernel(
    const float* __restrict__ seg_starts,  // (N_seg, 3)
    const float* __restrict__ seg_ends,    // (N_seg, 3)
    int   N_seg,
    float current,
    const float* __restrict__ xyz,         // (N_pts, 3)
    int   N_pts,
    float* __restrict__ BR,
    float* __restrict__ BPhi,
    float* __restrict__ BZ)
{
    const float MU0_4PI = 1.0e-7f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_pts) return;

    float px = xyz[3*i], py = xyz[3*i+1], pz = xyz[3*i+2];
    float bx = 0.0f, by = 0.0f, bz = 0.0f;

    for (int s = 0; s < N_seg; ++s) {
        float x0 = seg_starts[3*s],   y0 = seg_starts[3*s+1], z0 = seg_starts[3*s+2];
        float x1 = seg_ends[3*s],     y1 = seg_ends[3*s+1],   z1 = seg_ends[3*s+2];

        float dlx = x1-x0, dly = y1-y0, dlz = z1-z0;
        float r0x = px-x0, r0y = py-y0, r0z = pz-z0;
        float r1x = px-x1, r1y = py-y1, r1z = pz-z1;

        float R0 = sqrtf(r0x*r0x + r0y*r0y + r0z*r0z);
        float R1 = sqrtf(r1x*r1x + r1y*r1y + r1z*r1z);
        if (R0 < 5.0e-4f || R1 < 5.0e-4f) continue;  // skip near-wire

        float dot  = r0x*r1x + r0y*r1y + r0z*r1z;
        float dnom = R0 * R1 * (R0*R1 + dot);
        if (fabsf(dnom) < 1.0e-30f) continue;

        float fac = MU0_4PI * current * (R0 + R1) / dnom;
        bx += fac * (dly * r0z - dlz * r0y);
        by += fac * (dlz * r0x - dlx * r0z);
        bz += fac * (dlx * r0y - dly * r0x);
    }

    float phi = atan2f(py, px);
    float cp_ = cosf(phi), sp_ = sinf(phi);
    BR[i]   = bx * cp_ + by * sp_;
    BPhi[i] = -bx * sp_ + by * cp_;
    BZ[i]   = bz;
}
"""

# ── Compiled kernel singletons ────────────────────────────────────────────────
_k_circular: 'cp.RawKernel | None' = None
_k_biot:     'cp.RawKernel | None' = None
_THREADS = 256


def _get_circular_kernel():
    global _k_circular
    if _k_circular is None:
        if not _HAS_CUPY:
            raise RuntimeError("CuPy not available; cannot use circular coil CUDA kernel")
        _k_circular = cp.RawKernel(_CIRCULAR_COIL_KERNEL_SRC, 'circular_coil_kernel',
                                    options=('--use_fast_math',))
    return _k_circular


def _get_biot_kernel():
    global _k_biot
    if _k_biot is None:
        if not _HAS_CUPY:
            raise RuntimeError("CuPy not available; cannot use Biot-Savart CUDA kernel")
        _k_biot = cp.RawKernel(_BIOT_SAVART_KERNEL_SRC, 'biot_savart_kernel',
                                 options=('--use_fast_math',))
    return _k_biot


# ── Public GPU functions ──────────────────────────────────────────────────────

def circular_coil_field_gpu(
    cx: float, cy: float, cz: float,
    nx: float, ny: float, nz: float,
    radius: float, current: float,
    xyz_gpu: 'cp.ndarray',          # (N, 3) float32, already on GPU
) -> tuple['cp.ndarray', 'cp.ndarray', 'cp.ndarray']:
    """Compute circular-coil field at all points via CUDA kernel.

    Parameters
    ----------
    cx, cy, cz : coil centre (metres, Cartesian)
    nx, ny, nz : coil normal (unit vector)
    radius     : coil radius (metres)
    current    : current (Amperes, signed)
    xyz_gpu    : (N, 3) float32 CuPy array — Cartesian field points

    Returns
    -------
    BR, BPhi, BZ : (N,) float32 CuPy arrays — cylindrical B in Tesla
    """
    N = xyz_gpu.shape[0]
    BR   = cp.empty(N, dtype=cp.float32)
    BPhi = cp.empty(N, dtype=cp.float32)
    BZ   = cp.empty(N, dtype=cp.float32)

    kernel = _get_circular_kernel()
    blocks = (N + _THREADS - 1) // _THREADS
    kernel(
        (blocks,), (_THREADS,),
        (cp.float32(cx), cp.float32(cy), cp.float32(cz),
         cp.float32(nx), cp.float32(ny), cp.float32(nz),
         cp.float32(radius), cp.float32(current),
         xyz_gpu, cp.int32(N),
         BR, BPhi, BZ),
    )
    return BR, BPhi, BZ


def biot_savart_field_gpu(
    seg_starts_gpu: 'cp.ndarray',   # (N_seg, 3) float32
    seg_ends_gpu:   'cp.ndarray',   # (N_seg, 3) float32
    current: float,
    xyz_gpu: 'cp.ndarray',          # (N, 3) float32
) -> tuple['cp.ndarray', 'cp.ndarray', 'cp.ndarray']:
    """Biot-Savart field for one arbitrary filamentary coil via CUDA kernel.

    Parameters
    ----------
    seg_starts_gpu : (N_seg, 3) float32 CuPy — segment start points (metres)
    seg_ends_gpu   : (N_seg, 3) float32 CuPy — segment end points (metres)
    current        : current (Amperes, signed)
    xyz_gpu        : (N, 3) float32 CuPy — Cartesian field points

    Returns
    -------
    BR, BPhi, BZ : (N,) float32 CuPy arrays — cylindrical B in Tesla
    """
    N_seg = seg_starts_gpu.shape[0]
    N_pts = xyz_gpu.shape[0]
    BR   = cp.empty(N_pts, dtype=cp.float32)
    BPhi = cp.empty(N_pts, dtype=cp.float32)
    BZ   = cp.empty(N_pts, dtype=cp.float32)

    kernel = _get_biot_kernel()
    blocks = (N_pts + _THREADS - 1) // _THREADS
    kernel(
        (blocks,), (_THREADS,),
        (seg_starts_gpu, seg_ends_gpu, cp.int32(N_seg),
         cp.float32(current),
         xyz_gpu, cp.int32(N_pts),
         BR, BPhi, BZ),
    )
    return BR, BPhi, BZ


# ── CPU fallbacks (OpenMP via cyna, or pure NumPy) ───────────────────────────

def circular_coil_field_cpu(
    cx: float, cy: float, cz: float,
    nx: float, ny: float, nz: float,
    radius: float, current: float,
    xyz: np.ndarray,                # (N, 3) float32 NumPy
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CPU reference implementation of circular coil field.

    Tries cyna C++ OpenMP first; falls back to NumPy.
    """
    try:
        from pyna._cyna import coil_circular_field as _cf
        if _cf is not None:
            return _cf(cx, cy, cz, nx, ny, nz, float(radius), float(current),
                       np.ascontiguousarray(xyz, np.float32))
    except Exception:
        pass
    # Pure NumPy fallback
    return _circular_coil_numpy(cx, cy, cz, nx, ny, nz, radius, current, xyz)


def _circular_coil_numpy(cx, cy, cz, nx, ny, nz, a, I, xyz):
    """NumPy scalar loop fallback — correct but slow."""
    MU0_2PI = 2e-7
    dx = xyz[:, 0] - cx; dy = xyz[:, 1] - cy; dz = xyz[:, 2] - cz
    z_loc = nx*dx + ny*dy + nz*dz
    rx = dx - z_loc*nx; ry = dy - z_loc*ny; rz = dz - z_loc*nz
    rho = np.sqrt(rx**2 + ry**2 + rz**2)

    from scipy.special import ellipk, ellipe
    # Vectorised over field points
    Q = (a + rho)**2 + z_loc**2
    k2 = np.clip(4*a*rho / np.where(Q > 0, Q, 1e-30), 0, 1 - 1e-7)
    K = ellipk(k2); E = ellipe(k2)
    dE = np.where(rho > 1e-9, (a - rho)**2 + z_loc**2, 1e-20)
    sQ = np.sqrt(np.where(Q > 0, Q, 1e-30))

    Brho = np.where(rho > 1e-9,
                    MU0_2PI*I * z_loc/(rho*sQ) * (-K + E*(a**2+rho**2+z_loc**2)/dE),
                    0.0)
    Bz_l = np.where(rho > 1e-9,
                    MU0_2PI*I / sQ * (K + E*(a**2-rho**2-z_loc**2)/dE),
                    MU0_2PI*I * a**2 / np.where(sQ > 0, sQ**3, 1e-30))

    # Cartesian lab field
    inv = np.where(rho > 1e-9, 1/rho, 0.0)
    bx = Brho*rx*inv + Bz_l*nx
    by = Brho*ry*inv + Bz_l*ny
    bz = Brho*rz*inv + Bz_l*nz

    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    cp_ = np.cos(phi); sp_ = np.sin(phi)
    BR   = (bx*cp_ + by*sp_).astype(np.float32)
    BPhi = (-bx*sp_ + by*cp_).astype(np.float32)
    BZ   = bz.astype(np.float32)
    return BR, BPhi, BZ


def biot_savart_field_cpu(
    seg_starts: np.ndarray,   # (N_seg, 3) float32
    seg_ends:   np.ndarray,   # (N_seg, 3) float32
    current: float,
    xyz: np.ndarray,          # (N, 3) float32
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CPU Biot-Savart (tries cyna, falls back to NumPy loop)."""
    try:
        from pyna._cyna import coil_biot_savart as _bs
        if _bs is not None:
            return _bs(np.ascontiguousarray(seg_starts, np.float32),
                       np.ascontiguousarray(seg_ends,   np.float32),
                       float(current),
                       np.ascontiguousarray(xyz, np.float32))
    except Exception:
        pass
    return _biot_savart_numpy(seg_starts, seg_ends, current, xyz)


def _biot_savart_numpy(seg_starts, seg_ends, current, xyz):
    """NumPy Biot-Savart — chunked over segments to limit RAM."""
    MU0_4PI = 1e-7
    bx = np.zeros(len(xyz), np.float64)
    by = np.zeros_like(bx)
    bz = np.zeros_like(bx)
    P = xyz.astype(np.float64)
    for s in range(len(seg_starts)):
        p0 = seg_starts[s].astype(np.float64)
        p1 = seg_ends[s].astype(np.float64)
        dl = p1 - p0
        r0 = P - p0; r1 = P - p1
        R0 = np.linalg.norm(r0, axis=1).clip(5e-4)
        R1 = np.linalg.norm(r1, axis=1).clip(5e-4)
        dot = (r0*r1).sum(1)
        denom = (R0*R1*(R0*R1 + dot)).clip(1e-30)
        fac = (MU0_4PI * current * (R0 + R1) / denom)[:, None]
        cross = np.cross(dl, r0)
        bx += (fac * cross[:, 0:1]).ravel()
        by += (fac * cross[:, 1:2]).ravel()
        bz += (fac * cross[:, 2:3]).ravel()
    phi = np.arctan2(P[:, 1], P[:, 0])
    cp_ = np.cos(phi); sp_ = np.sin(phi)
    return (
        (bx*cp_ + by*sp_).astype(np.float32),
        (-bx*sp_ + by*cp_).astype(np.float32),
        bz.astype(np.float32),
    )


# ── Convenience: auto-dispatch GPU or CPU ────────────────────────────────────

def circular_coil_field(
    cx, cy, cz, nx, ny, nz, radius, current,
    xyz,              # NumPy (N,3) float32 on CPU  OR  CuPy on GPU
    use_gpu: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch to GPU or CPU implementation; always returns NumPy arrays."""
    if use_gpu and _HAS_CUPY:
        xyz_g = cp.asarray(xyz, dtype=cp.float32)
        BR_g, BPhi_g, BZ_g = circular_coil_field_gpu(
            cx, cy, cz, nx, ny, nz, radius, current, xyz_g)
        return cp.asnumpy(BR_g), cp.asnumpy(BPhi_g), cp.asnumpy(BZ_g)
    else:
        return circular_coil_field_cpu(
            cx, cy, cz, nx, ny, nz, radius, current,
            np.asarray(xyz, np.float32))


def biot_savart_field(
    seg_starts, seg_ends, current, xyz,
    use_gpu: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch Biot-Savart to GPU or CPU; always returns NumPy arrays."""
    if use_gpu and _HAS_CUPY:
        ss_g = cp.asarray(np.ascontiguousarray(seg_starts, np.float32))
        se_g = cp.asarray(np.ascontiguousarray(seg_ends,   np.float32))
        xyz_g = cp.asarray(np.ascontiguousarray(xyz,       np.float32))
        BR_g, BPhi_g, BZ_g = biot_savart_field_gpu(ss_g, se_g, current, xyz_g)
        return cp.asnumpy(BR_g), cp.asnumpy(BPhi_g), cp.asnumpy(BZ_g)
    else:
        return biot_savart_field_cpu(
            np.ascontiguousarray(seg_starts, np.float32),
            np.ascontiguousarray(seg_ends,   np.float32),
            current,
            np.ascontiguousarray(xyz, np.float32))
