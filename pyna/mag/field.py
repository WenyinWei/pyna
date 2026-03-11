"""Cylindrical-grid vector field classes implementing the VectorField3D hierarchy.

These classes extend the functionality of the legacy ``pyna.field``
classes by also inheriting from the new :class:`~pyna.system.VectorField3D`
abstract base.  All existing API (R, Z, Phi, BR, BZ, BPhi properties
and interpolation helpers) is preserved.

Backward-compatibility aliases
-------------------------------
``RegualrCylindricalGridField`` (note: intentional typo preserved)
  → :class:`CylindricalGridVectorField3D`
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pyna.system import VectorField3D, AxiSymmetricVectorField3D


# ---------------------------------------------------------------------------
# 3-D non-axisymmetric cylindrical grid field
# ---------------------------------------------------------------------------

class CylindricalGridVectorField3D(VectorField3D):
    """3-D magnetic vector field stored on a regular (R, Z, Phi) grid.

    Extends :class:`~pyna.field.CylindricalGridVectorField` with the
    :class:`~pyna.system.VectorField3D` interface so that it can be
    used directly as a dynamical system whose right-hand side is the
    field itself (useful for field-line tracing / ODE integration).

    Parameters
    ----------
    R:
        1-D array of radial grid values (m).
    Z:
        1-D array of axial grid values (m).
    Phi:
        1-D array of toroidal angle grid values (rad).
    BR, BZ, BPhi:
        3-D arrays of field components with shape
        ``(len(R), len(Z), len(Phi))`` (T).
    """

    def __init__(
        self,
        R: np.ndarray,
        Z: np.ndarray,
        Phi: np.ndarray,
        BR: np.ndarray,
        BZ: np.ndarray,
        BPhi: np.ndarray,
    ) -> None:
        self._R = np.asarray(R, dtype=float)
        self._Z = np.asarray(Z, dtype=float)
        self._Phi = np.asarray(Phi, dtype=float)
        self._BR = np.asarray(BR, dtype=float)
        self._BZ = np.asarray(BZ, dtype=float)
        self._BPhi = np.asarray(BPhi, dtype=float)

        self._interp_BR = RegularGridInterpolator(
            (self._R, self._Z, self._Phi), self._BR,
            method="linear", bounds_error=False, fill_value=np.nan,
        )
        self._interp_BZ = RegularGridInterpolator(
            (self._R, self._Z, self._Phi), self._BZ,
            method="linear", bounds_error=False, fill_value=np.nan,
        )
        self._interp_BPhi = RegularGridInterpolator(
            (self._R, self._Z, self._Phi), self._BPhi,
            method="linear", bounds_error=False, fill_value=np.nan,
        )

    # ------------------------------------------------------------------
    # Grid properties
    # ------------------------------------------------------------------

    @property
    def R(self) -> np.ndarray:
        """Radial grid (m)."""
        return self._R

    @property
    def Z(self) -> np.ndarray:
        """Axial grid (m)."""
        return self._Z

    @property
    def Phi(self) -> np.ndarray:
        """Toroidal angle grid (rad)."""
        return self._Phi

    @property
    def BR(self) -> np.ndarray:
        """Radial field component array (T), shape (nR, nZ, nPhi)."""
        return self._BR

    @property
    def BZ(self) -> np.ndarray:
        """Axial field component array (T)."""
        return self._BZ

    @property
    def BPhi(self) -> np.ndarray:
        """Toroidal field component array (T)."""
        return self._BPhi

    # ------------------------------------------------------------------
    # VectorField3D interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        rzphi: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """Evaluate the field at a single point (R, Z, φ) or array thereof.

        Parameters
        ----------
        rzphi:
            Array of shape ``(..., 3)`` with columns (R, Z, φ).
        t:
            Ignored (autonomous system); kept for interface compatibility.

        Returns
        -------
        ndarray
            Field components (B_R, B_Z, B_φ), same leading shape as
            ``rzphi``, last axis of length 3.
        """
        rzphi = np.asarray(rzphi, dtype=float)
        br = self._interp_BR(rzphi)
        bz = self._interp_BZ(rzphi)
        bphi = self._interp_BPhi(rzphi)
        return np.stack((br, bz, bphi), axis=-1)

    def interpolate_at(
        self,
        R: float | np.ndarray,
        Z: float | np.ndarray,
        Phi: float | np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convenience method: evaluate (B_R, B_Z, B_φ) at given coordinates.

        Parameters
        ----------
        R, Z, Phi:
            Scalars or broadcast-compatible arrays.

        Returns
        -------
        (BR, BZ, BPhi) : tuple of ndarray
        """
        R = np.asarray(R, dtype=float)
        Z = np.asarray(Z, dtype=float)
        Phi = np.asarray(Phi, dtype=float)
        pts = np.stack((R.ravel(), Z.ravel(), Phi.ravel()), axis=-1)
        result = self(pts)
        shape = np.broadcast(R, Z, Phi).shape
        return (
            result[..., 0].reshape(shape),
            result[..., 1].reshape(shape),
            result[..., 2].reshape(shape),
        )


# ---------------------------------------------------------------------------
# Axisymmetric specialisation
# ---------------------------------------------------------------------------

class CylindricalGridAxiVectorField3D(AxiSymmetricVectorField3D):
    """Axisymmetric 3-D magnetic field stored on a regular (R, Z) grid.

    The field components do not depend on the toroidal angle φ.
    Extends :class:`~pyna.system.AxiSymmetricVectorField3D` so that
    this object can participate in the dynamical-systems hierarchy.

    Parameters
    ----------
    R:
        1-D radial grid (m).
    Z:
        1-D axial grid (m).
    BR, BZ, BPhi:
        2-D arrays of shape ``(len(R), len(Z))`` (T).
    """

    def __init__(
        self,
        R: np.ndarray,
        Z: np.ndarray,
        BR: np.ndarray,
        BZ: np.ndarray,
        BPhi: np.ndarray,
    ) -> None:
        self._R = np.asarray(R, dtype=float)
        self._Z = np.asarray(Z, dtype=float)
        self._BR = np.asarray(BR, dtype=float)
        self._BZ = np.asarray(BZ, dtype=float)
        self._BPhi = np.asarray(BPhi, dtype=float)

        self._interp_BR = RegularGridInterpolator(
            (self._R, self._Z), self._BR,
            method="linear", bounds_error=False, fill_value=np.nan,
        )
        self._interp_BZ = RegularGridInterpolator(
            (self._R, self._Z), self._BZ,
            method="linear", bounds_error=False, fill_value=np.nan,
        )
        self._interp_BPhi = RegularGridInterpolator(
            (self._R, self._Z), self._BPhi,
            method="linear", bounds_error=False, fill_value=np.nan,
        )

    # ------------------------------------------------------------------
    # Grid properties
    # ------------------------------------------------------------------

    @property
    def R(self) -> np.ndarray:
        return self._R

    @property
    def Z(self) -> np.ndarray:
        return self._Z

    @property
    def BR(self) -> np.ndarray:
        return self._BR

    @property
    def BZ(self) -> np.ndarray:
        return self._BZ

    @property
    def BPhi(self) -> np.ndarray:
        return self._BPhi

    # ------------------------------------------------------------------
    # VectorField3D / AxiSymmetricVectorField3D interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        rzphi: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """Evaluate the field at (R, Z[, φ]) — φ component is ignored.

        Parameters
        ----------
        rzphi:
            Array of shape ``(..., 2)`` or ``(..., 3)``.  Only the
            first two components (R, Z) are used.
        t:
            Ignored (autonomous).

        Returns
        -------
        ndarray
            ``(B_R, B_Z, B_φ)``, shape ``(..., 3)``.
        """
        rzphi = np.asarray(rzphi, dtype=float)
        rz = rzphi[..., :2]
        pts = rz.reshape(-1, 2)
        shape = rzphi.shape[:-1]
        br = self._interp_BR(pts).reshape(shape)
        bz = self._interp_BZ(pts).reshape(shape)
        bphi = self._interp_BPhi(pts).reshape(shape)
        return np.stack((br, bz, bphi), axis=-1)

    def interpolate_at(
        self,
        R: float | np.ndarray,
        Z: float | np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate (B_R, B_Z, B_φ) at given (R, Z) coordinates."""
        R = np.asarray(R, dtype=float)
        Z = np.asarray(Z, dtype=float)
        pts = np.stack((R.ravel(), Z.ravel()), axis=-1)
        result = self(pts)
        shape = np.broadcast(R, Z).shape
        return (
            result[..., 0].reshape(shape),
            result[..., 1].reshape(shape),
            result[..., 2].reshape(shape),
        )


# ---------------------------------------------------------------------------
# Backward-compatibility alias (typo preserved intentionally)
# ---------------------------------------------------------------------------

#: Alias for :class:`CylindricalGridVectorField3D`.
#: The misspelled name is kept for backward compatibility with existing code.
RegualrCylindricalGridField = CylindricalGridVectorField3D
