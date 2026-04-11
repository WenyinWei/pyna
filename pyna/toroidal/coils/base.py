"""Abstract base class for vacuum coil magnetic fields.

All concrete coil-field classes inherit from CoilFieldVacuum and implement:
  - B_at(R, Z, phi): evaluate (BR, BZ, Bphi) at given coordinates
  - divergence_free(): whether the field is guaranteed divergence-free

Concrete utility classes included here:
  - CoilFieldSuperposition: linear superposition of multiple fields
  - CoilFieldScaled: single field multiplied by a scalar (for current control)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np


class CoilFieldVacuum(ABC):
    """Abstract base for vacuum magnetic fields from coils.
    
    Coordinate convention: cylindrical (R, Z, phi) in meters/radians.
    """

    @abstractmethod
    def B_at(
        self,
        R: float | np.ndarray,
        Z: float | np.ndarray,
        phi: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate (B_R, B_Z, B_phi) at given coordinates.
        
        Parameters
        ----------
        R, Z, phi : scalar or array-like, broadcast-compatible
        
        Returns
        -------
        (BR, BZ, Bphi) : tuple of ndarray, same shape as broadcast(R, Z, phi)
        """

    @abstractmethod
    def divergence_free(self) -> bool:
        """Return True if this field is guaranteed to satisfy ∇·B = 0."""

    def to_grid_field(
        self,
        R_arr: np.ndarray,
        Z_arr: np.ndarray,
        Phi_arr: np.ndarray,
        *,
        cache_path: str | None = None,
        cache_key: str | None = None,
    ):
        """Evaluate field on a 3D (R, Z, Phi) grid, with optional joblib caching.
        
        Parameters
        ----------
        R_arr, Z_arr, Phi_arr : 1D arrays
            Grid axes.
        cache_path : str or None
            If given, use joblib.Memory at this path to cache results.
        cache_key : str or None
            Unique string key for cache lookup. If None, cache is bypassed.
        
        Returns
        -------
        (BR, BZ, Bphi) : ndarray, shape (len(R), len(Z), len(Phi))
        """
        def _compute():
            R3, Z3, P3 = np.meshgrid(R_arr, Z_arr, Phi_arr, indexing='ij')
            BR, BZ, Bp = self.B_at(R3.ravel(), Z3.ravel(), P3.ravel())
            shape = (len(R_arr), len(Z_arr), len(Phi_arr))
            return (
                np.asarray(BR).reshape(shape),
                np.asarray(BZ).reshape(shape),
                np.asarray(Bp).reshape(shape),
            )

        if cache_path is not None and cache_key is not None:
            from joblib import Memory
            mem = Memory(cache_path, verbose=0)
            return mem.cache(_compute, ignore=[])()
        return _compute()


class CoilFieldSuperposition(CoilFieldVacuum):
    """Linear superposition of multiple CoilFieldVacuum objects.
    
    The resulting field is divergence-free iff all component fields are.
    """

    def __init__(self, fields: Sequence[CoilFieldVacuum]) -> None:
        self._fields = list(fields)

    def B_at(self, R, Z, phi):
        BR = BZ = Bp = None
        for f in self._fields:
            br, bz, bp = f.B_at(R, Z, phi)
            if BR is None:
                BR, BZ, Bp = np.asarray(br, float), np.asarray(bz, float), np.asarray(bp, float)
            else:
                BR = BR + br; BZ = BZ + bz; Bp = Bp + bp
        if BR is None:
            shape = np.broadcast(R, Z, phi).shape
            return np.zeros(shape), np.zeros(shape), np.zeros(shape)
        return BR, BZ, Bp

    def divergence_free(self) -> bool:
        return all(f.divergence_free() for f in self._fields)


class CoilFieldScaled(CoilFieldVacuum):
    """A CoilFieldVacuum scaled by a constant factor (e.g. coil current).
    
    Parameters
    ----------
    field : CoilFieldVacuum
        The base field (typically computed for unit current I=1 A).
    scale : float
        Scaling factor (e.g. actual current in amperes).
    """

    def __init__(self, field: CoilFieldVacuum, scale: float) -> None:
        self._field = field
        self._scale = float(scale)

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = float(value)

    def B_at(self, R, Z, phi):
        BR, BZ, Bp = self._field.B_at(R, Z, phi)
        return self._scale * BR, self._scale * BZ, self._scale * Bp

    def divergence_free(self) -> bool:
        return self._field.divergence_free()
