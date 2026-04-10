"""pyna._cyna.utils — utility helpers for the cyna C++ bridge.

These helpers reduce redundant dtype conversions and provide a convenient
bridge from raw cyna output arrays to the pyna invariant-object hierarchy.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Zero-copy buffer preparation
# ---------------------------------------------------------------------------

def ensure_c_double(arr: np.ndarray) -> np.ndarray:
    """Return *arr* as a C-contiguous float64 array, avoiding copies when possible.

    If the input is already C-contiguous float64 the original array is
    returned (zero-copy).  Otherwise ``np.ascontiguousarray`` is called.
    """
    if arr.dtype == np.float64 and arr.flags['C_CONTIGUOUS']:
        return arr
    return np.ascontiguousarray(arr, dtype=np.float64)


def prepare_field_cache(
    field_cache: dict,
    *,
    extend_phi: bool = True,
) -> dict:
    """Prepare a field_cache dict for cyna C++ calls.

    Returns a new dict with all arrays guaranteed C-contiguous float64.
    If *extend_phi* is True (default), the toroidal grid and field arrays
    are extended by one period copy so cyna's trilinear interpolation
    handles the 2π seam correctly.

    The returned dict can be passed directly to any ``pyna._cyna.*`` function
    as keyword arguments.

    Parameters
    ----------
    field_cache : dict
        Must contain keys 'BR', 'BPhi', 'BZ', 'R_grid', 'Z_grid', 'Phi_grid'.
    extend_phi : bool
        Whether to extend the phi dimension by one period copy.

    Returns
    -------
    dict with keys 'BR', 'BPhi', 'BZ', 'R_grid', 'Z_grid', 'Phi_grid',
    all C-contiguous float64.
    """
    Rg = ensure_c_double(np.asarray(field_cache['R_grid']))
    Zg = ensure_c_double(np.asarray(field_cache['Z_grid']))
    Pg = ensure_c_double(np.asarray(field_cache['Phi_grid']))

    BR   = np.asarray(field_cache['BR'])
    BPhi = np.asarray(field_cache['BPhi'])
    BZ   = np.asarray(field_cache['BZ'])

    if extend_phi:
        if abs(float(Pg[-1]) - 2 * np.pi) > 1e-6:
            Pg = np.append(Pg, 2 * np.pi)
        # Extend field arrays along the last (phi) axis
        BR   = np.concatenate([BR,   BR[:, :, :1]],   axis=2)
        BPhi = np.concatenate([BPhi, BPhi[:, :, :1]], axis=2)
        BZ   = np.concatenate([BZ,   BZ[:, :, :1]],   axis=2)

    return {
        'BR':       ensure_c_double(BR),
        'BPhi':     ensure_c_double(BPhi),
        'BZ':       ensure_c_double(BZ),
        'R_grid':   Rg,
        'Z_grid':   Zg,
        'Phi_grid': ensure_c_double(Pg),
    }


# ---------------------------------------------------------------------------
# cyna Newton results → FixedPoint objects
# ---------------------------------------------------------------------------

def build_fixed_points_from_batch(
    R_out: np.ndarray,
    Z_out: np.ndarray,
    converged: np.ndarray,
    DPm_flat: np.ndarray,
    ptype: np.ndarray,
    phi: float,
) -> list:
    """Convert raw ``find_fixed_points_batch`` output to FixedPoint objects.

    Parameters
    ----------
    R_out, Z_out : ndarray (N,)
        Converged positions.
    converged : ndarray (N,) int
        1 if converged, 0 otherwise.
    DPm_flat : ndarray (N, 4) or (N, 2, 2)
        Monodromy matrices.
    ptype : ndarray (N,) int
        1 = X-point, 0 = O-point, -1 = unconverged.
    phi : float
        Toroidal angle of the Poincaré section [rad].

    Returns
    -------
    list of FixedPoint
        Only converged points are included.
    """
    from pyna.topo.invariants import FixedPoint

    # Normalise DPm shape
    if DPm_flat.ndim == 2 and DPm_flat.shape[1] == 4:
        DPm_arr = DPm_flat.reshape(-1, 2, 2)
    elif DPm_flat.ndim == 3:
        DPm_arr = DPm_flat
    else:
        DPm_arr = DPm_flat.reshape(len(R_out), 2, 2)

    fps = []
    for i in range(len(R_out)):
        if not converged[i]:
            continue
        kind = 'X' if int(ptype[i]) == 1 else 'O'
        fps.append(FixedPoint(
            phi=phi,
            R=float(R_out[i]),
            Z=float(Z_out[i]),
            DPm=DPm_arr[i].copy(),
            kind=kind,
        ))
    return fps


def build_cycles_from_batch(
    field_cache_cyna: dict,
    o_fps: list,
    x_fps: list,
    period: int,
    known_n: int,
    section_phis: list,
    phi0: float,
    *,
    DPhi: float = 0.05,
    fd_eps: float = 1e-4,
    max_iter: int = 40,
    tol: float = 1e-9,
) -> Tuple[list, list]:
    """Build Cycle objects from seed FixedPoints by tracing to multiple sections.

    Parameters
    ----------
    field_cache_cyna : dict
        Pre-prepared cyna field cache (from :func:`prepare_field_cache`).
    o_fps, x_fps : list of FixedPoint
        O-point and X-point seeds (from :func:`build_fixed_points_from_batch`).
    period, known_n : int
        Resonance numbers m, n.
    section_phis : list of float
        Toroidal sections to trace to.
    phi0 : float
        Reference section where seeds were found.

    Returns
    -------
    o_cycles, x_cycles : list of Cycle
    """
    from pyna.topo.invariants import Cycle, FixedPoint, MonodromyData
    from pyna._cyna import find_fixed_points_batch as _ffpb

    if _ffpb is None:
        raise RuntimeError("cyna not available")

    winding = (period, known_n)

    def _trace_fps(fps: list) -> list:
        cycles = []
        for fp0 in fps:
            sections: Dict[float, list] = {}
            for phi in section_phis:
                if abs(phi - phi0) < 1e-9:
                    sections[phi] = [fp0]
                else:
                    R_o, Z_o, _, conv, DPm_flat, _, _, ptype = _ffpb(
                        np.array([fp0.R], dtype=np.float64),
                        np.array([fp0.Z], dtype=np.float64),
                        phi, period,
                        DPhi=DPhi, fd_eps=fd_eps, max_iter=max_iter, tol=tol,
                        n_threads=1, **field_cache_cyna,
                    )
                    if conv[0]:
                        DPm = DPm_flat[0].reshape(2, 2) if DPm_flat.ndim == 2 else DPm_flat[0]
                        sections[phi] = [FixedPoint(
                            phi=phi, R=float(R_o[0]), Z=float(Z_o[0]),
                            DPm=DPm.copy(),
                        )]
            if not sections:
                continue
            mono = MonodromyData(
                DPm=fp0.DPm,
                eigenvalues=np.linalg.eigvals(fp0.DPm),
            )
            cycles.append(Cycle(winding=winding, sections=sections, monodromy=mono))
        return cycles

    return _trace_fps(o_fps), _trace_fps(x_fps)
