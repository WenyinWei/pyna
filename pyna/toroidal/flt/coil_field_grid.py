"""Coil-set -> cylindrical magnetic-field grid, built in cyna (CPU).

This module is the stable pyna entry point for filling a cylindrical
(nR, nZ, nPhi) field grid from a set of filamentary coils.  The heavy
Biot-Savart sum runs in the cyna C++ extension (OpenMP, float64), which is
the default backend and needs no CUDA, JAX, or GPU.  A pure-NumPy reference
implementation is kept for tests and correctness cross-checks only; it is
much slower and emits a warning whenever it is selected.

Conventions
-----------
* Coil polylines are Cartesian (K, S, 3) arrays in metres with
  x = R cos(phi), y = R sin(phi), z = Z.  Consecutive points define
  S - 1 straight segments; polyline coils are not implicitly closed
  (repeat the first point to close a loop).
* currents is (K,) signed ampere-turns.
* R_grid, Z_grid, Phi_grid are strictly increasing float64 axes.
* Phi_grid is periodic in phi with nfp field periods.  A closed grid (the
  cyna native layout) duplicates its first plane at the endpoint:
  Phi_grid[-1] - Phi_grid[0] == 2*pi/nfp.  An open grid (the pivot2t
  layout) supplies one period without the duplicate; set
  phi_closed=False or leave phi_closed=None for auto-detection.
* Output components are the global cylindrical components in Tesla::

      BR   =  Bx cos(phi) + By sin(phi)
      BPhi = -Bx sin(phi) + By cos(phi)
      BZ   =  Bz

  laid out as (nR, nZ, nPhi) C-order arrays, i.e. phi is the fastest
  index, exactly like the field arrays consumed by pyna.toroidal.flt.

The returned dictionary uses the same keys as the pivot2t adapter
(R_grid, Z_grid, Phi_grid, BR_flat, BZ_flat, BPhi_flat, nfp), plus the
3-D arrays and the backend name.
"""
from __future__ import annotations

import warnings

import numpy as np

__all__ = ["build_coil_field_grid"]

_MU0_OVER_4PI = 1.0e-7  # T m / A

_REFERENCE_WARNING = (
    "build_coil_field_grid is using the NumPy reference backend. This path is "
    "intended for tests and correctness cross-checks only: it is orders of "
    "magnitude slower than the cyna C++ CPU kernel and is not a supported "
    "production field-line-tracing backend. Install a CPU wheel or rebuild "
    "cyna so backend='auto' selects the C++ kernel."
)


def _require_axis(values, name: str, *, positive: bool = False) -> np.ndarray:
    arr = np.ascontiguousarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if arr.size < 1:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and not np.all(arr > 0.0):
        raise ValueError(f"{name} must contain only positive values")
    if arr.size > 1 and not np.all(np.diff(arr) > 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return arr


def _require_segments(segments, currents) -> tuple[np.ndarray, np.ndarray]:
    seg = np.ascontiguousarray(segments, dtype=np.float64)
    # pivot2t's coil_segments layout: (K, S, 2, 3) explicit segment endpoints.
    # Convert to the polyline layout by keeping every segment start and
    # appending the final endpoint of each coil.
    if seg.ndim == 4 and seg.shape[2] == 2 and seg.shape[3] == 3:
        seg = np.concatenate([seg[:, :, 0, :], seg[:, -1:, 1, :]], axis=1)
    if seg.ndim != 3 or seg.shape[2] != 3:
        raise ValueError(
            "segments must have shape (K, S, 3) polylines or "
            "(K, S, 2, 3) segment endpoints"
        )
    if seg.shape[0] < 1 or seg.shape[1] < 2:
        raise ValueError("segments must contain K >= 1 polylines with S >= 2 points")
    if not np.all(np.isfinite(seg)):
        raise ValueError("segments must contain only finite values")
    cur = np.ascontiguousarray(currents, dtype=np.float64)
    if cur.ndim != 1 or cur.size != seg.shape[0]:
        raise ValueError("currents must have shape (K,) matching segments")
    if not np.all(np.isfinite(cur)):
        raise ValueError("currents must contain only finite values")
    return seg, cur


def _resolve_closed(Phi_grid: np.ndarray, nfp: int, phi_closed) -> bool:
    if phi_closed is not None:
        return bool(phi_closed)
    if Phi_grid.size <= 1:
        return False
    period = 2.0 * np.pi / int(nfp)
    span = float(Phi_grid[-1] - Phi_grid[0])
    tol = 1e-12 * max(1.0, abs(period))
    return abs(span - period) <= tol


def _cyna_kernel():
    """Return the compiled cyna kernel, or None (never imports CUDA)."""
    try:
        from pyna._cyna import _cyna_ext
    except Exception:
        return None
    return getattr(_cyna_ext, "coil_field_grid", None)


def _numpy_reference(
    segments: np.ndarray,
    currents: np.ndarray,
    R_grid: np.ndarray,
    Z_grid: np.ndarray,
    Phi_grid: np.ndarray,
    *,
    phi_closed: bool,
    chunk_points: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure NumPy finite-segment Biot-Savart reference (CPU only)."""
    nR, nZ, nPhi = R_grid.size, Z_grid.size, Phi_grid.size
    p0 = segments[:, :-1, :]                       # (K, M, 3)
    p1 = segments[:, 1:, :]
    seg_vec = p1 - p0                              # L
    current = currents[:, None]                    # (K, 1)

    Rg, Zg, Pg = np.meshgrid(R_grid, Z_grid, Phi_grid, indexing="ij")
    Rf = np.ascontiguousarray(Rg).ravel()
    Zf = np.ascontiguousarray(Zg).ravel()
    Pf = np.ascontiguousarray(Pg).ravel()
    cos_phi = np.cos(Pf)
    sin_phi = np.sin(Pf)
    points = np.stack([Rf * cos_phi, Rf * sin_phi, Zf], axis=-1)  # (N, 3)

    flat = np.empty((points.shape[0], 3), dtype=np.float64)
    for i0 in range(0, points.shape[0], chunk_points):
        i1 = min(points.shape[0], i0 + chunk_points)
        chunk = points[i0:i1]                       # (n, 3)
        a = chunk[:, None, None, :] - p0[None, :, :, :]
        b = a - seg_vec[None, :, :, :]
        cross = np.cross(seg_vec[None, :, :, :], a)
        den = np.sum(cross * cross, axis=-1)        # (n, K, M)
        good = den > 1e-30
        den_safe = np.where(good, den, 1.0)
        a_dot = np.sum(a * seg_vec[None, :, :, :], axis=-1)
        b_dot = np.sum(b * seg_vec[None, :, :, :], axis=-1)
        norm_a = np.linalg.norm(a, axis=-1)
        norm_b = np.linalg.norm(b, axis=-1)
        # |a| and |b| are > 1e-12 for every contributing segment.
        term = a_dot / np.maximum(norm_a, 1e-300) - b_dot / np.maximum(norm_b, 1e-300)
        scale = _MU0_OVER_4PI * current[None, :, :] * np.where(good, term / den_safe, 0.0)
        flat[i0:i1] = np.sum(cross * scale[..., None], axis=(1, 2))

    bx = flat[:, 0]
    by = flat[:, 1]
    bz = flat[:, 2]
    BR = (bx * cos_phi + by * sin_phi).reshape(nR, nZ, nPhi)
    BPhi = (-bx * sin_phi + by * cos_phi).reshape(nR, nZ, nPhi)
    BZ = bz.reshape(nR, nZ, nPhi)
    if phi_closed and nPhi > 1:
        BR[:, :, -1] = BR[:, :, 0]
        BZ[:, :, -1] = BZ[:, :, 0]
        BPhi[:, :, -1] = BPhi[:, :, 0]
    return (np.ascontiguousarray(BR), np.ascontiguousarray(BZ),
            np.ascontiguousarray(BPhi))


def build_coil_field_grid(
    segments,
    currents,
    R_grid,
    Z_grid,
    Phi_grid,
    nfp: int = 1,
    *,
    phi_closed=None,
    backend: str = "auto",
    n_threads: int = -1,
) -> dict:
    """Fill a cylindrical coil-field grid (nR, nZ, nPhi) in cyna.

    Parameters
    ----------
    segments : array_like, shape (K, S, 3) or (K, S, 2, 3)
        Cartesian coil polylines in metres (x = R cos(phi),
        y = R sin(phi), z = Z).  Consecutive points are straight
        finite segments; close a loop by repeating the first point.
        The pivot2t coil_segments layout (K, S, 2, 3) of explicit
        segment endpoints is accepted as well.
    currents : array_like, shape (K,)
        Signed ampere-turns per coil.
    R_grid, Z_grid : array_like, shape (nR,), (nZ,)
        Strictly increasing cylindrical axes in metres (R > 0).
    Phi_grid : array_like, shape (nPhi,)
        Strictly increasing toroidal angles in radians, periodic with
        nfp field periods.  A closed grid (phi_closed=True) must satisfy
        Phi_grid[-1] - Phi_grid[0] == 2*pi/nfp; an open grid covers one
        period without the duplicate endpoint.
    nfp : int, default 1
        Number of field periods.  Explicit; never inferred from Phi_grid.
    phi_closed : bool or None, default None
        None auto-detects from the span.  True requires the closed cyna
        layout, False treats every plane as distinct (pivot2t layout).
    backend : {"auto", "cyna", "cpu", "numpy", "reference"}, default "auto"
        "auto" uses the cyna C++ kernel when the compiled extension is
        importable.  If the kernel is missing it emits a RuntimeWarning
        and falls back to the NumPy reference.  "cyna"/"cpu" require the
        cyna kernel; "numpy"/"reference" force the reference (test-only).
        Every NumPy-reference use emits a RuntimeWarning because that path
        is not a supported production backend.  No backend imports CUDA.
    n_threads : int, default -1
        OpenMP thread count for the cyna kernel (-1 = library default).

    Returns
    -------
    dict
        R_grid, Z_grid, Phi_grid, nfp, BR_flat, BZ_flat, BPhi_flat
        (flat C-order (nR*nZ*nPhi,) floats), BR, BZ, BPhi
        ((nR, nZ, nPhi) arrays), shape and backend ("cyna-cpu" or
        "numpy-reference").
    """
    nfp = int(nfp)
    if nfp < 1:
        raise ValueError("nfp must be a positive integer")

    seg, cur = _require_segments(segments, currents)
    Rg = _require_axis(R_grid, "R_grid", positive=True)
    Zg = _require_axis(Z_grid, "Z_grid")
    Pg = _require_axis(Phi_grid, "Phi_grid")

    closed = _resolve_closed(Pg, nfp, phi_closed)
    period = 2.0 * np.pi / nfp
    tol = 1e-12 * max(1.0, abs(period))
    if Pg.size > 1:
        span = float(Pg[-1] - Pg[0])
        if closed and abs(span - period) > tol:
            raise ValueError(
                "Phi_grid must be a closed periodic grid spanning exactly "
                "2*pi/nfp when phi_closed=True; pass phi_closed=False for an "
                "open grid"
            )
        if not closed and span > period + tol:
            raise ValueError("Phi_grid spans more than one field period 2*pi/nfp")

    backend_key = str(backend).lower()
    if backend_key not in ("auto", "cyna", "cpu", "numpy", "reference"):
        raise ValueError(
            "backend must be 'auto', 'cyna', 'cpu', 'numpy' or 'reference'"
        )

    kernel = _cyna_kernel() if backend_key in ("auto", "cyna", "cpu") else None
    if kernel is not None:
        BR, BZ, BPhi = kernel(
            seg, cur, Rg, Zg, Pg, nfp, closed, int(n_threads)
        )
        BR = np.ascontiguousarray(BR, dtype=np.float64)
        BZ = np.ascontiguousarray(BZ, dtype=np.float64)
        BPhi = np.ascontiguousarray(BPhi, dtype=np.float64)
        used = "cyna-cpu"
    elif backend_key in ("cyna", "cpu"):
        raise RuntimeError(
            "the compiled cyna extension does not provide coil_field_grid; "
            "install a CPU wheel or rebuild cyna, or pass backend='numpy' "
            "for a test-only reference calculation"
        )
    else:
        warnings.warn(_REFERENCE_WARNING, RuntimeWarning, stacklevel=2)
        BR, BZ, BPhi = _numpy_reference(
            seg, cur, Rg, Zg, Pg, phi_closed=closed
        )
        used = "numpy-reference"

    expected = (Rg.size, Zg.size, Pg.size)
    if BR.shape != expected or BZ.shape != expected or BPhi.shape != expected:
        raise RuntimeError(
            f"coil-field backend returned shape {BR.shape}, expected {expected}"
        )

    return {
        "R_grid": Rg,
        "Z_grid": Zg,
        "Phi_grid": Pg,
        "nfp": nfp,
        "BR_flat": BR.ravel(),
        "BZ_flat": BZ.ravel(),
        "BPhi_flat": BPhi.ravel(),
        "BR": BR,
        "BZ": BZ,
        "BPhi": BPhi,
        "shape": expected,
        "backend": used,
    }
