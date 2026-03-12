"""Centralized cache management for pyna.control.

All expensive computations (Biot-Savart field grids, manifold growth,
X/O point locations, A-matrix evaluations) should use these cached wrappers.

Cache backend: joblib.Memory (disk-persistent, safe for numpy arrays)
Cache location: ~/.pyna_cache/ (user home, not repo)
"""

import os
import joblib
import numpy as np
from pathlib import Path

_CACHE_DIR = Path.home() / '.pyna_cache'
_CACHE_DIR.mkdir(exist_ok=True)

memory = joblib.Memory(location=str(_CACHE_DIR), verbose=0)


@memory.cache
def cached_biot_savart_grid(
    coil_R: np.ndarray,
    coil_Z: np.ndarray,
    coil_I: float,
    R_grid_flat: np.ndarray,
    Z_grid_flat: np.ndarray,
) -> np.ndarray:
    """Biot-Savart field on a grid of (R,Z) points. Cached to disk.

    Parameters
    ----------
    coil_R, coil_Z : ndarray
        Coil filament coordinates.
    coil_I : float
        Coil current (A).
    R_grid_flat, Z_grid_flat : ndarray, shape (N_pts,)
        Flat arrays of evaluation points.

    Returns
    -------
    B_grid : ndarray, shape (N_pts, 3)
        [BR, BZ, Bphi] at each grid point.
    """
    # Placeholder implementation: returns zeros.
    # Replace with actual Biot-Savart integration (e.g. pyna.biot_savart).
    N = len(R_grid_flat)
    return np.zeros((N, 3))


@memory.cache
def cached_A_matrix(
    eq_params_hash: str,
    R: float,
    Z: float,
    phi: float = 0.0,
    eps: float = 1e-4,
) -> np.ndarray:
    """A-matrix at (R,Z) for a given equilibrium. Cached.

    NOTE: The actual field function cannot be cached directly (not hashable).
    This function serves as a memoization stub; the caller must ensure that
    eq_params_hash uniquely identifies the equilibrium so stale cache entries
    are not returned.

    Parameters
    ----------
    eq_params_hash : str
        Unique string identifying the equilibrium (use hash_eq_params).
    R, Z : float
        Evaluation point.
    phi : float
        Toroidal angle (0 for axisymmetric).
    eps : float
        Finite-difference step for numerical derivatives.

    Returns
    -------
    A : ndarray, shape (2,2)
    """
    # Returns zeros; actual computation requires field_func which is
    # injected at runtime (not cacheable as positional arg).
    return np.zeros((2, 2))


def hash_eq_params(**kwargs) -> str:
    """Create a hashable string from equilibrium parameters.

    Parameters
    ----------
    **kwargs
        Scalar equilibrium parameters (e.g., R0=1.85, B0=3.5).

    Returns
    -------
    key : str
        Deterministic string representation, sorted by key name.

    Examples
    --------
    >>> hash_eq_params(R0=1.85, B0=3.5, Ip=500e3)
    'B0=3.5_Ip=500000_R0=1.85'
    """
    return '_'.join(f'{k}={v:.6g}' for k, v in sorted(kwargs.items()))


def invalidate_cache() -> None:
    """Clear all pyna.control disk caches (use after equilibrium change)."""
    memory.clear(warn=False)
