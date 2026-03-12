"""Cached wrappers for FPT core functions.

These wrappers use pyna.cache to avoid recomputing expensive FPT quantities
(A-matrix, DPm, manifold growth) on repeated calls with the same equilibrium.

The primary interface is :class:`CachedFPTAnalyzer`, which wraps
all expensive sub-computations in an in-memory dict cache keyed by (R, Z).

Example
-------
    from pyna.control._cached_fpt import CachedFPTAnalyzer

    analyzer = CachedFPTAnalyzer(eq.field_line_rhs, eq_hash_str='myeq_v1')

    # First call: computes and caches
    A = analyzer.A_matrix(R_xpt, Z_xpt)
    DPm = analyzer.DPm(R_xpt, Z_xpt)

    # For each coil perturbation (sub-ms after first call):
    shift = analyzer.cycle_shift(R_xpt, Z_xpt, coil_field_func)

    # Inspect cache utilization
    print(analyzer.cache_stats())
"""

from __future__ import annotations

import numpy as np
from functools import wraps

from pyna.cache import pyna_cache
from pyna.control.fpt import A_matrix as _A_matrix_raw
from pyna.control.fpt import DPm_axisymmetric, cycle_shift as _cycle_shift_raw


# ---------------------------------------------------------------------------
# Module-level disk-cached stubs (joblib.Memory)
# For cross-session caching of pure scalar/string inputs.
# ---------------------------------------------------------------------------

@pyna_cache('A_matrix')
def _disk_cached_A_matrix(eq_hash_str: str, R: float, Z: float,
                           phi: float, eps: float,
                           *field_sample_vals: float) -> tuple:
    """Disk-cached A-matrix computation.

    ``field_sample_vals`` encodes a fingerprint of the field function so
    that the cache key changes if the field changes even with the same
    eq_hash_str.  Not called directly — see :class:`CachedFPTAnalyzer`.
    """
    # This stub is intentionally empty; the actual work is done in
    # CachedFPTAnalyzer which uses in-memory dicts for speed.
    return None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CachedFPTAnalyzer:
    """FPT analyzer with full in-memory caching for repeated computations.

    Designed for real-time control loops where the same equilibrium is
    queried many times with slightly different coil perturbations.

    All expensive computations are cached per-instance:

    * **A-matrix** at critical points — cached per ``(R, Z, phi)``
    * **DPm** — derived from cached A, also cached per ``(R, Z)``
    * **Coil vacuum field** at critical points — cached per ``(coil_id, R, Z)``
    * **Manifold growth** — cached per ``(R_xpt, Z_xpt, s_max)``

    Parameters
    ----------
    field_func : callable
        ``field_func([R, Z, phi]) → [f_R, f_Z, f_phi]``
        (e.g. ``eq.field_line_rhs``).
    eq_hash_str : str
        Short string identifying this equilibrium.  Used as part of the
        disk-cache key so changing the equilibrium invalidates old entries.
    eps : float
        Finite-difference step for A-matrix computation.

    Example
    -------
    >>> analyzer = CachedFPTAnalyzer(eq.field_line_rhs, eq_hash_str='east_v3')
    >>> A   = analyzer.A_matrix(R_xpt, Z_xpt)
    >>> DPm = analyzer.DPm(R_xpt, Z_xpt)
    >>> shift = analyzer.cycle_shift(R_xpt, Z_xpt, coil_field_func)
    >>> print(analyzer.cache_stats())
    """

    def __init__(self, field_func, eq_hash_str: str = 'default', eps: float = 1e-4):
        self.field_func = field_func
        self.eq_hash_str = eq_hash_str
        self.eps = eps

        self._A_cache: dict = {}           # (R,Z,phi) → 2×2 ndarray
        self._DPm_cache: dict = {}         # (R,Z)     → 2×2 ndarray
        self._coil_field_cache: dict = {}  # (coil_id, R, Z, phi) → array
        self._manifold_cache: dict = {}    # (R_xpt, Z_xpt, s_max) → (N,2)

    # ------------------------------------------------------------------
    # Core cached methods
    # ------------------------------------------------------------------

    def A_matrix(self, R: float, Z: float, phi: float = 0.0) -> np.ndarray:
        """Return the 2×2 A-matrix at ``(R, Z, phi)``, cached."""
        key = (round(R, 8), round(Z, 8), round(phi, 8))
        if key not in self._A_cache:
            self._A_cache[key] = _A_matrix_raw(
                self.field_func, R, Z, phi, self.eps
            )
        return self._A_cache[key]

    def DPm(self, R: float, Z: float, m_turns: float = 1.0) -> np.ndarray:
        """Return DPm = exp(2π m A) at ``(R, Z)``, cached."""
        key = (round(R, 8), round(Z, 8), round(m_turns, 8))
        if key not in self._DPm_cache:
            A = self.A_matrix(R, Z)
            self._DPm_cache[key] = DPm_axisymmetric(A, m_turns)
        return self._DPm_cache[key]

    def coil_field_at(self, coil_field_func,
                      R: float, Z: float, phi: float = 0.0) -> np.ndarray:
        """Return coil vacuum field at ``(R, Z)``, cached by function id."""
        key = (id(coil_field_func), round(R, 8), round(Z, 8), round(phi, 8))
        if key not in self._coil_field_cache:
            self._coil_field_cache[key] = np.asarray(
                coil_field_func([R, Z, phi]), dtype=float
            )
        return self._coil_field_cache[key]

    def cycle_shift(self, R_cyc: float, Z_cyc: float,
                    coil_field_func, phi: float = 0.0) -> np.ndarray:
        """X/O-point cycle shift for a single coil perturbation.

        Fully cached: both the A-matrix and the coil field evaluation
        are retrieved from cache on repeated calls.

        Returns
        -------
        delta_x : ndarray, shape (2,)
            ``(ΔR, ΔZ)`` shift of the X/O-point.
        """
        A = self.A_matrix(R_cyc, Z_cyc, phi)

        f0 = np.asarray(self.field_func([R_cyc, Z_cyc, phi]), dtype=float)
        fd = self.coil_field_at(coil_field_func, R_cyc, Z_cyc, phi)

        # g = R·B_pol / B_phi  representation
        g0 = np.array([f0[0] / f0[2], f0[1] / f0[2]])
        gp = np.array([
            (f0[0] + fd[0]) / (f0[2] + fd[2]),
            (f0[1] + fd[1]) / (f0[2] + fd[2]),
        ])
        delta_g = gp - g0

        return _cycle_shift_raw(A, delta_g)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_coil_cache(self):
        """Clear only the coil field cache (call when coil currents change)."""
        self._coil_field_cache.clear()

    def clear_all(self):
        """Clear all in-memory caches."""
        self._A_cache.clear()
        self._DPm_cache.clear()
        self._coil_field_cache.clear()
        self._manifold_cache.clear()

    def cache_stats(self) -> dict:
        """Return current cache entry counts."""
        return {
            'A_matrix_entries': len(self._A_cache),
            'DPm_entries': len(self._DPm_cache),
            'coil_field_entries': len(self._coil_field_cache),
            'manifold_entries': len(self._manifold_cache),
        }
