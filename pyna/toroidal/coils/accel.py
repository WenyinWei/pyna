"""pyna.toroidal.coils.accel — GPU/CPU-accelerated circular coil field computation.

This module provides fast magnetic field evaluation for arrays of circular
(ring) coils, using either GPU (CuPy) or CPU (NumPy/SciPy) backends.

Two computation strategies are offered:

1. **Direct analytic** (``analytic_coil_field_batched_gpu``):
   Exact Smythe/Schill elliptic-integral formula evaluated on GPU.
   Cost: O(N_coils × N_pts) elliptic-integral evaluations.

2. **Template fast-path** (``CircularCoilTemplate``):
   Pre-compute a single normalised 2-D table f_ρ(ρ', z'), f_z(ρ', z') for
   the unit coil (a=1, I=1) once.  For every real coil the field is obtained
   by translate → rotate → normalise → bilinear lookup → scale back.
   Cost: O(1) table build  +  O(N_coils × N_pts) bilinear interpolations.

Scaling law (exact)
-------------------
For a circular coil with radius *a* and current *I*:

    B_ρ(ρ, z; a, I) = (I / a) · f_ρ(ρ/a, z/a)
    B_z(ρ, z; a, I) = (I / a) · f_z(ρ/a, z/a)

This follows directly from the Smythe formulae, so **one template covers
all circular coils regardless of radius or current**.

Template grid design
--------------------
The normalised domain must cover the farthest coil/field-point pair.
For a machine with typical size ~1 m and smallest coil radius a_min ≈ 0.04 m,
the worst-case normalised distance is ~(2R₀ / a_min) ≈ 50-60.

Spacing: near-coil region (ρ' < ``rho_join``) uses fine uniform spacing
to resolve the 1/r singularity; far field uses log-uniform spacing since
B ~ 1/r³ and relative interpolation error scales with the fractional step.

Performance (RTX 3060, 336 dipoles × 614 400 pts)
-------------------------------------------------
* Direct analytic: ~18 s
* Template build:  ~0.5 s (once)
* Template query:  ~5 s
* Accuracy in plasma (|B| < 5 T): mean rel ≈ 5e-4, p99 ≈ 6e-3

Public API
----------
analytic_coil_field_batched_gpu(centers, radii, normals, currents, field_pts)
biot_savart_all_coils_gpu(coils, field_pts)
CircularCoilTemplate(rho_max, z_max, N_near, N_far, N_z, rho_join, use_gpu)
get_template(**kwargs) -> CircularCoilTemplate  (module singleton)
"""

from __future__ import annotations

import warnings

import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None  # type: ignore[assignment]
    _CUPY_AVAILABLE = False
    warnings.warn(
        "CuPy not available; pyna.toroidal.coils.accel will fall back to CPU.",
        stacklevel=2,
    )

MU0_OVER_4PI: float = 1e-7   # T·m/A


# ---------------------------------------------------------------------------
# AGM-based complete elliptic integrals (NumPy / CuPy compatible)
# ---------------------------------------------------------------------------

def _ellipk_agm(m, *, xp=np):
    """Complete elliptic integral K(m) via AGM iteration.

    Parameters
    ----------
    m : array-like, m ∈ [0, 1)
    xp : numpy or cupy

    Notes
    -----
    10 AGM iterations give relative error < 1e-14 for float64.
    """
    m = xp.asarray(m, dtype=xp.float64)
    a = xp.ones_like(m)
    b = xp.sqrt(xp.maximum(1.0 - m, 0.0))
    for _ in range(10):
        a_new = 0.5 * (a + b)
        b     = xp.sqrt(a * b)
        a     = a_new
    return (xp.pi / 2.0) / a


def _ellipe_agm(m, *, xp=np):
    """Complete elliptic integral E(m) via AGM (Abramowitz & Stegun §17.6).

    14 iterations give relative error < 1e-15 for float64 over m ∈ [0, 1-1e-3].
    """
    m = xp.asarray(m, dtype=xp.float64)
    a = xp.ones_like(m)
    b = xp.sqrt(xp.maximum(1.0 - m, 0.0))
    S     = 0.5 * m
    power = 1.0
    for _ in range(14):
        c     = 0.5 * (a - b)
        a_new = 0.5 * (a + b)
        b     = xp.sqrt(a * b)
        a     = a_new
        S    += power * c ** 2
        power *= 2.0
    K = (xp.pi / 2.0) / a
    return K * (1.0 - S)


# ---------------------------------------------------------------------------
# Rotation matrices: ẑ → normals[k]  (Rodrigues formula, batched)
# ---------------------------------------------------------------------------

def _batch_rotation_matrices(normals, *, xp=np):
    """Compute N rotation matrices mapping ẑ → normals[k].

    Parameters
    ----------
    normals : (N, 3) array  (need not be unit; normalised internally)
    xp : numpy or cupy

    Returns
    -------
    R : (N, 3, 3)  R[k] @ ẑ = normals[k] / ‖normals[k]‖
    """
    normals = xp.asarray(normals, dtype=xp.float64)
    N   = normals.shape[0]
    nrm = xp.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / xp.where(nrm > 1e-30, nrm, xp.ones_like(nrm))

    z_hat = xp.array([0.0, 0.0, 1.0])
    v  = xp.cross(z_hat[None, :], normals)           # (N, 3)
    s  = xp.linalg.norm(v, axis=1, keepdims=True)    # (N, 1)
    c  = normals[:, 2:3]                              # (N, 1)

    vx = xp.zeros((N, 3, 3), dtype=xp.float64)
    vx[:, 0, 1] = -v[:, 2];  vx[:, 0, 2] =  v[:, 1]
    vx[:, 1, 0] =  v[:, 2];  vx[:, 1, 2] = -v[:, 0]
    vx[:, 2, 0] = -v[:, 1];  vx[:, 2, 1] =  v[:, 0]

    eye    = xp.eye(3, dtype=xp.float64)[None] * xp.ones((N, 1, 1), dtype=xp.float64)
    s2     = (s ** 2)[:, :, None]
    safe_s2 = xp.where(s2 > 1e-24, s2, xp.ones_like(s2))
    factor  = (1.0 - c[:, :, None]) / safe_s2

    R = eye + vx + xp.einsum('nij,njk->nik', vx, vx) * factor

    # Degenerate cases: normal ≈ ±ẑ
    near_z  = (s[:, 0] < 1e-12)
    flip_z  = (c[:, 0] < 0)
    R[near_z & ~flip_z] = xp.eye(3, dtype=xp.float64)
    flip_mat = xp.diag(xp.array([1.0, -1.0, -1.0], dtype=xp.float64))
    if xp.any(near_z & flip_z):
        R[near_z & flip_z] = flip_mat[None]

    return R   # (N, 3, 3)


# ---------------------------------------------------------------------------
# Direct GPU analytic field (all coils × all points, vectorised)
# ---------------------------------------------------------------------------

def analytic_coil_field_batched_gpu(
    centers:  np.ndarray,
    radii:    np.ndarray,
    normals:  np.ndarray,
    currents: np.ndarray,
    field_pts: np.ndarray,
    coil_batch_size: int = 20,   # kept for API compatibility
    pt_batch_size:   int = 16384,
) -> np.ndarray:
    """Exact field of N circular coils at M field points via elliptic integrals.

    All coil parameters are pushed to GPU once; field points are processed in
    batches of ``pt_batch_size`` to fit VRAM.

    Falls back to a NumPy/SciPy implementation when CuPy is unavailable.

    Parameters
    ----------
    centers : (N, 3) float64
    radii   : (N,)   float64
    normals : (N, 3) float64  coil axis unit vectors (need not be normalised)
    currents: (N,)   float64  A (signed)
    field_pts : (M, 3) float64  Cartesian evaluation points
    coil_batch_size : ignored (kept for API compat)
    pt_batch_size   : int  field-point GPU batch size

    Returns
    -------
    B : (M, 3) float64  Cartesian field in Tesla (NaN at near-wire points)
    """
    if not _CUPY_AVAILABLE:
        return _analytic_coil_field_cpu(centers, radii, normals, currents, field_pts)

    xp = cp
    M, N = field_pts.shape[0], centers.shape[0]

    c_gpu = xp.asarray(centers,   dtype=xp.float64)
    a_gpu = xp.asarray(radii,     dtype=xp.float64)
    I_gpu = xp.asarray(currents,  dtype=xp.float64)
    R_all    = _batch_rotation_matrices(xp.asarray(normals, dtype=xp.float64), xp=xp)
    Rinv_all = xp.transpose(R_all, (0, 2, 1))
    fp_gpu   = xp.asarray(field_pts, dtype=xp.float64)
    B_total  = xp.zeros((M, 3), dtype=xp.float64)

    for p0 in range(0, M, pt_batch_size):
        p1  = min(p0 + pt_batch_size, M)
        pts = fp_gpu[p0:p1]                                      # (mb, 3)

        delta   = pts[None, :, :] - c_gpu[:, None, :]           # (N, mb, 3)
        p_local = xp.einsum('kij,kmj->kmi', Rinv_all, delta)    # (N, mb, 3)
        x_l, y_l, z_l = p_local[:,:,0], p_local[:,:,1], p_local[:,:,2]
        rho_l = xp.sqrt(x_l**2 + y_l**2)
        phi_l = xp.arctan2(y_l, x_l)

        a = a_gpu[:, None]
        denom_sq = (a + rho_l)**2 + z_l**2
        m_arg = xp.where(
            denom_sq > 0,
            xp.clip(4.0 * a * rho_l / denom_sq, 0.0, 1.0 - 1e-12),
            xp.zeros_like(denom_sq),
        )
        K = _ellipk_agm(m_arg, xp=xp)
        E = _ellipe_agm(m_arg, xp=xp)

        sqrt_denom = xp.sqrt(xp.maximum(denom_sq, 1e-30))
        denom2   = (a - rho_l)**2 + z_l**2
        safe_d2  = xp.where(denom2  > 1e-30, denom2,  xp.full_like(denom2,  1e-30))
        safe_rho = xp.where(rho_l   > 1e-15, rho_l,   xp.full_like(rho_l,   1e-15))

        # Near-wire singularity: set to NaN so field-line tracers stop
        near_wire   = (rho_l**2 + z_l**2) < (0.05 * a)**2
        safe_denom  = xp.where(near_wire, xp.full_like(sqrt_denom, xp.nan), sqrt_denom)

        coeff  = (MU0_OVER_4PI * 2.0) * I_gpu[:, None] / safe_denom
        Bz_l   = coeff * (K + (a**2 - rho_l**2 - z_l**2) / safe_d2 * E)
        Brho_l = coeff * z_l / safe_rho * (-K + (a**2 + rho_l**2 + z_l**2) / safe_d2 * E)
        Brho_l = xp.where(rho_l > 1e-15, Brho_l, xp.zeros_like(Brho_l))

        B_local  = xp.stack([Brho_l*xp.cos(phi_l), Brho_l*xp.sin(phi_l), Bz_l], axis=2)
        B_global = xp.einsum('kij,kmj->kmi', R_all, B_local)    # (N, mb, 3)

        B_total[p0:p1] += xp.nansum(B_global, axis=0)
        has_nan = xp.any(xp.isnan(B_global.sum(axis=2)), axis=0)
        B_total[p0:p1][has_nan] = xp.nan

    return cp.asnumpy(B_total)


def _analytic_coil_field_cpu(centers, radii, normals, currents, field_pts):
    """CPU fallback using scipy's elliptic integrals (pyna.toroidal.coils.coil)."""
    from pyna.toroidal.coils.coil import BRBZ_induced_by_current_loop

    M = field_pts.shape[0]
    B_total = np.zeros((M, 3), dtype=np.float64)

    R_all    = _batch_rotation_matrices(normals, xp=np)
    Rinv_all = np.transpose(R_all, (0, 2, 1))

    for k, (c_k, a_k, I_k) in enumerate(zip(centers, radii, currents)):
        delta   = field_pts - c_k[None, :]          # (M, 3)
        p_local = delta @ Rinv_all[k].T             # (M, 3)
        rho_l   = np.sqrt(p_local[:,0]**2 + p_local[:,1]**2)
        phi_l   = np.arctan2(p_local[:,1], p_local[:,0])
        z_l     = p_local[:,2]

        BR, BZ  = BRBZ_induced_by_current_loop(a_k, 0.0, I_k, rho_l, z_l)

        B_local  = np.column_stack([BR*np.cos(phi_l), BR*np.sin(phi_l), BZ])
        B_global = B_local @ R_all[k].T
        B_total += B_global

    return B_total


# ---------------------------------------------------------------------------
# GPU Biot-Savart: arbitrary coils (duck-typed interface)
# ---------------------------------------------------------------------------

def biot_savart_all_coils_gpu(
    coils,
    field_pts: np.ndarray,
    pt_batch: int = 8192,
) -> np.ndarray:
    """Biot-Savart field of arbitrary coils, GPU-accelerated.

    Coil interface (duck typing)
    ----------------------------
    Each coil object must expose:
      - ``.closed_points() -> np.ndarray (K+1, 3)``
        Ordered polyline that closes the loop (first == last, or wraps around).
      - ``.current : float``  current in amperes.

    All coil segments are concatenated into one GPU kernel launch.

    Parameters
    ----------
    coils : list of coil-like objects
    field_pts : (M, 3) float64
    pt_batch : int  field-point batch size (tune to VRAM)

    Returns
    -------
    B : (M, 3) float64  total field in Tesla
    """
    try:
        import cupy as xp
        _gpu = True
    except ImportError:
        xp   = np
        _gpu = False

    mu0_4pi = 1e-7

    dl_list, mid_list, I_list = [], [], []
    for coil in coils:
        pts = coil.closed_points()
        dl  = np.diff(pts, axis=0)
        mid = 0.5 * (pts[:-1] + pts[1:])
        I   = np.full(len(dl), coil.current)
        dl_list.append(dl); mid_list.append(mid); I_list.append(I)

    dl_all  = np.concatenate(dl_list,  axis=0).astype(np.float64)
    mid_all = np.concatenate(mid_list, axis=0).astype(np.float64)
    I_all   = np.concatenate(I_list,   axis=0).astype(np.float64)
    N_seg = len(dl_all)
    M     = len(field_pts)
    fp    = field_pts.astype(np.float64)

    print(f"Biot-Savart GPU: {len(coils)} coils → {N_seg} segs, {M} pts "
          f"(pt_batch={pt_batch})", flush=True)

    dl_gpu  = xp.asarray(dl_all)
    mid_gpu = xp.asarray(mid_all)
    I_gpu   = xp.asarray(I_all)
    seg_len = xp.linalg.norm(dl_gpu, axis=-1)

    B_total = np.zeros((M, 3), dtype=np.float64)

    for p0 in range(0, M, pt_batch):
        p1  = min(p0 + pt_batch, M)
        pts = xp.asarray(fp[p0:p1])                    # (nb, 3)

        r_vec = pts[:, None, :] - mid_gpu[None, :, :]  # (nb, N_seg, 3)
        r_mag = xp.linalg.norm(r_vec, axis=-1)
        r3    = r_mag ** 3
        near  = r_mag < 0.1 * seg_len[None, :]
        r3    = xp.where(near, xp.nan, r3)

        dl_cross_r = xp.cross(dl_gpu[None, :, :], r_vec)
        weight = I_gpu[None, :] / r3
        dB = xp.nansum(dl_cross_r * weight[:, :, None], axis=1)   # (nb, 3)

        B_total[p0:p1] = mu0_4pi * (cp.asnumpy(dB) if _gpu else dB)

    return B_total


# ---------------------------------------------------------------------------
# Template fast-path: CircularCoilTemplate
# ---------------------------------------------------------------------------

class CircularCoilTemplate:
    """Pre-computed unit-coil field on a 2-D normalised (ρ', z') grid.

    The scaling law for a circular coil is exact:

        B_ρ(ρ, z; a, I) = (I / a) · f_ρ(ρ/a, z/a)
        B_z(ρ, z; a, I) = (I / a) · f_z(ρ/a, z/a)

    One template therefore covers all circular coils regardless of radius or
    current — only translation + rotation of the evaluation points is needed
    before a cheap GPU bilinear lookup.

    Grid design
    -----------
    rho_max / z_max must satisfy:
        rho_max ≥ max_over_all_coils(max_dist_to_grid_corner / coil_radius)

    For a machine of size ~1 m with smallest coil a_min ≈ 0.04 m, a value
    of 60 is safe (max normalised distance ≈ 52 with 10 % margin → 57).

    The ρ' axis uses a hybrid grid:
      [0, rho_join] : uniform   (resolves near-coil 1/r singularity)
      [rho_join, rho_max] : log-uniform  (B ~ 1/r³ → relative error uniform)
    The z' axis uses a uniform symmetric grid.

    Parameters
    ----------
    rho_max  : float  maximum ρ/a to tabulate  (default 60)
    z_max    : float  maximum |z/a| to tabulate (default 60)
    N_near   : int    uniform points in [0, rho_join]
    N_far    : int    log-uniform points in [rho_join, rho_max]
    N_z      : int    total z points
    rho_join : float  uniform/log-uniform transition
    use_gpu  : bool   build and query on GPU
    """

    def __init__(
        self,
        rho_max:  float = 60.0,
        z_max:    float = 60.0,
        N_near:   int   = 1200,
        N_far:    int   = 500,
        N_z:      int   = 3200,
        rho_join: float = 2.0,
        use_gpu:  bool  = True,
    ):
        self.rho_max  = rho_max
        self.z_max    = z_max
        self.rho_join = rho_join

        rho_near      = np.linspace(0.0, rho_join, N_near + 1)[:-1]
        rho_far       = np.exp(np.linspace(np.log(rho_join), np.log(rho_max), N_far))
        self.rho_grid = np.concatenate([rho_near, rho_far])
        self.z_grid   = np.linspace(-z_max, z_max, N_z)
        self.N_rho    = len(self.rho_grid)
        self.N_z      = N_z

        import time as _time
        t0 = _time.time()
        self._build(use_gpu)
        print(
            f"[CircularCoilTemplate] built {self.N_rho}×{self.N_z} grid in "
            f"{_time.time()-t0:.2f}s  "
            f"(rho_max={rho_max}, z_max={z_max}, rho_join={rho_join})"
        )

    # ------------------------------------------------------------------
    def _build(self, use_gpu: bool) -> None:
        xp = cp if (use_gpu and _CUPY_AVAILABLE) else np

        rho  = xp.asarray(self.rho_grid, dtype=xp.float64)
        z    = xp.asarray(self.z_grid,   dtype=xp.float64)
        rho2 = rho[:, None]
        z2   = z[None, :]

        a        = 1.0
        denom_sq = (a + rho2)**2 + z2**2
        m_arg    = xp.where(
            denom_sq > 0,
            xp.clip(4.0 * a * rho2 / denom_sq, 0.0, 1.0 - 1e-12),
            xp.zeros_like(denom_sq),
        )
        K = _ellipk_agm(m_arg, xp=xp)
        E = _ellipe_agm(m_arg, xp=xp)

        sqrt_d   = xp.sqrt(xp.maximum(denom_sq, 1e-30))
        denom2   = (a - rho2)**2 + z2**2
        safe_d2  = xp.where(denom2 > 1e-30, denom2, xp.full_like(denom2, 1e-30))
        safe_rho = xp.where(rho2   > 1e-15, rho2,   xp.full_like(rho2,   1e-15))

        coeff = MU0_OVER_4PI * 2.0 / sqrt_d
        f_z   = coeff * (K + (a**2 - rho2**2 - z2**2) / safe_d2 * E)
        f_rho = coeff * z2 / safe_rho * (-K + (a**2 + rho2**2 + z2**2) / safe_d2 * E)
        f_rho = xp.where(rho2 > 1e-15, f_rho, xp.zeros_like(f_rho))

        self._use_gpu = use_gpu and _CUPY_AVAILABLE
        if self._use_gpu:
            self._f_rho_gpu = f_rho.astype(cp.float32)   # (Nr, Nz)  on GPU
            self._f_z_gpu   = f_z.astype(cp.float32)
            self._rho_gpu   = cp.asarray(self.rho_grid, dtype=cp.float32)
            self._Nr        = self.N_rho
            self._Nz        = self.N_z
            self._z0_g      = float(self.z_grid[0])
            self._dz        = float(self.z_grid[1] - self.z_grid[0])
        else:
            from scipy.interpolate import RegularGridInterpolator as _RGI
            f_rho_np = np.asarray(f_rho)
            f_z_np   = np.asarray(f_z)
            self._itp_frho = _RGI((self.rho_grid, self.z_grid), f_rho_np,
                                   method='linear', bounds_error=False, fill_value=0.0)
            self._itp_fz   = _RGI((self.rho_grid, self.z_grid), f_z_np,
                                   method='linear', bounds_error=False, fill_value=0.0)

    # ------------------------------------------------------------------
    def _gpu_bilinear(self, rho_n, z_n):
        """Bilinear interpolation on the hybrid ρ / uniform z GPU tables.

        Parameters
        ----------
        rho_n, z_n : cp.ndarray (...) float64  normalised coords (ρ/a, z/a)

        Returns
        -------
        f_rho, f_z : cp.ndarray same shape, float32
        """
        Nr, Nz = self._Nr, self._Nz
        shape  = rho_n.shape
        flat_r = rho_n.ravel().astype(cp.float32)
        flat_z = z_n.ravel().astype(cp.float32)

        # ρ: searchsorted (non-uniform grid)
        i0 = cp.searchsorted(self._rho_gpu, flat_r, side='right') - 1
        i0 = cp.clip(i0, 0, Nr - 2).astype(cp.int32)
        i1 = i0 + 1
        r0v = self._rho_gpu[i0];  r1v = self._rho_gpu[i1]
        wr  = cp.clip((flat_r - r0v) / (r1v - r0v + 1e-30), 0.0, 1.0)

        # z: direct index (uniform grid)
        iz  = (flat_z - self._z0_g) / self._dz
        j0  = cp.clip(iz.astype(cp.int32), 0, Nz - 2)
        j1  = j0 + 1
        wz  = cp.clip(iz - j0.astype(cp.float32), 0.0, 1.0)

        def _blerp(tbl):
            return (tbl[i0, j0] * (1 - wr) * (1 - wz) +
                    tbl[i1, j0] *      wr   * (1 - wz) +
                    tbl[i0, j1] * (1 - wr) *      wz   +
                    tbl[i1, j1] *      wr   *      wz)

        return _blerp(self._f_rho_gpu).reshape(shape), \
               _blerp(self._f_z_gpu).reshape(shape)

    # ------------------------------------------------------------------
    def field_all_coils(
        self,
        centers:   np.ndarray,
        radii:     np.ndarray,
        normals:   np.ndarray,
        currents:  np.ndarray,
        field_pts: np.ndarray,
        pt_batch_size: int = 32768,
    ) -> np.ndarray:
        """Summed field of N circular coils at M field points via template.

        Coordinate pipeline (GPU):
          field_pt  →  translate  →  rotate to local  →  normalise (/a)
          →  bilinear lookup  →  scale (×I/a)  →  rotate to global  →  sum

        Parameters
        ----------
        centers   : (N, 3)
        radii     : (N,)
        normals   : (N, 3)
        currents  : (N,)
        field_pts : (M, 3)
        pt_batch_size : int  field-point batch (GPU memory control)

        Returns
        -------
        B : (M, 3) float64  total field in Cartesian
        """
        xp = cp if self._use_gpu else np
        M  = len(field_pts)

        c_gpu    = xp.asarray(centers,   dtype=xp.float64)
        a_gpu    = xp.asarray(radii,     dtype=xp.float64)
        I_gpu    = xp.asarray(currents,  dtype=xp.float64)
        R_all    = _batch_rotation_matrices(xp.asarray(normals, dtype=xp.float64), xp=xp)
        Rinv_all = xp.transpose(R_all, (0, 2, 1))
        fp_gpu   = xp.asarray(field_pts, dtype=xp.float64)
        B_total  = xp.zeros((M, 3), dtype=xp.float64)

        for p0 in range(0, M, pt_batch_size):
            p1  = min(p0 + pt_batch_size, M)
            pts = fp_gpu[p0:p1]                                     # (mb, 3)

            delta   = pts[None, :, :] - c_gpu[:, None, :]          # (N, mb, 3)
            p_local = xp.einsum('kij,kmj->kmi', Rinv_all, delta)

            x_l, y_l, z_l = p_local[:,:,0], p_local[:,:,1], p_local[:,:,2]
            rho_l = xp.sqrt(x_l**2 + y_l**2)
            phi_l = xp.arctan2(y_l, x_l)
            a     = a_gpu[:, None]
            rho_n = rho_l / a
            z_n   = z_l   / a

            if self._use_gpu:
                f_rho, f_z = self._gpu_bilinear(rho_n, z_n)
                f_rho = f_rho.astype(xp.float64)
                f_z   = f_z.astype(xp.float64)
            else:
                pts_n = np.stack([rho_n.ravel(), z_n.ravel()], axis=1)
                f_rho = self._itp_frho(pts_n).reshape(rho_n.shape)
                f_z   = self._itp_fz(pts_n).reshape(rho_n.shape)

            scale  = I_gpu[:, None] / a
            Brho_l = scale * f_rho
            Bz_l   = scale * f_z

            B_local  = xp.stack([Brho_l*xp.cos(phi_l), Brho_l*xp.sin(phi_l), Bz_l], axis=2)
            B_global = xp.einsum('kij,kmj->kmi', R_all, B_local)
            B_total[p0:p1] += B_global.sum(axis=0)

        return cp.asnumpy(B_total) if self._use_gpu else B_total


# ---------------------------------------------------------------------------
# Module-level singleton + convenience accessor
# ---------------------------------------------------------------------------

_template: CircularCoilTemplate | None = None


def get_template(
    rho_max:  float = 60.0,
    z_max:    float = 60.0,
    N_near:   int   = 1200,
    N_far:    int   = 500,
    N_z:      int   = 3200,
    rho_join: float = 2.0,
    use_gpu:  bool  = True,
) -> CircularCoilTemplate:
    """Return (and lazily build) the module-level ``CircularCoilTemplate``.

    The template is built only once per process; subsequent calls return the
    cached instance regardless of the arguments.

    Default parameters are sized for machine radius ~1 m, smallest coil
    a_min = 0.04 m:

    * Build time  : ~0.5 s (GPU)
    * Query time  : ~5 s for 336 coils × 614 400 pts
    * Plasma accuracy (|B| < 5 T): mean rel ≈ 5e-4, p99 ≈ 6e-3
    * Near-wire (|B| ≫ 5 T): larger errors, but those points should be
      NaN-masked in the field-line tracer.
    """
    global _template
    if _template is None:
        _template = CircularCoilTemplate(
            rho_max, z_max, N_near, N_far, N_z, rho_join, use_gpu
        )
    return _template
