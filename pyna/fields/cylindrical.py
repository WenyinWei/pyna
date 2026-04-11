"""Concrete cylindrical-coordinate field implementations.

These replace (and are backward-compatible with):
  - pyna.field_data.CylindricalScalarField
  - pyna.field_data.CylindricalVectorField
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pyna.fields.base import ScalarField3D, VectorField3D
from pyna.fields.properties import FieldProperty
from pyna.fields.coords import Coords3DCylindrical as _CylCoords3D


class ScalarField3DCylindrical(ScalarField3D):
    """Scalar field f(R, Z, φ) on a regular cylindrical grid.

    Replaces pyna.field_data.CylindricalScalarField.

    Parameters
    ----------
    R, Z, Phi : 1D ndarray
        Grid axes.
    value : ndarray, shape (nR, nZ, nPhi)
        Field values.
    field_periods : int
        Toroidal field periods. If > 1, Phi covers [0, 2π/N_fp].
    name, units : str
    properties : FieldProperty
    """

    def __init__(
        self,
        R: np.ndarray,
        Z: np.ndarray,
        Phi: np.ndarray,
        value: np.ndarray,
        field_periods: int = 1,
        name: str = "",
        units: str = "",
        properties: FieldProperty = FieldProperty.NONE,
    ) -> None:
        super().__init__(properties=properties, name=name, units=units,
                         coords=_CylCoords3D())
        self._R = np.asarray(R, dtype=float)
        self._Z = np.asarray(Z, dtype=float)
        self._Phi = np.asarray(Phi, dtype=float)
        self._value = np.asarray(value, dtype=float)
        self.field_periods = field_periods
        assert self._value.shape == (len(self._R), len(self._Z), len(self._Phi)), \
            f"value shape {self._value.shape} mismatch"
        self._interp: Optional[RegularGridInterpolator] = None

    # Grid properties
    @property
    def R(self) -> np.ndarray: return self._R
    @property
    def Z(self) -> np.ndarray: return self._Z
    @property
    def Phi(self) -> np.ndarray: return self._Phi
    @property
    def value(self) -> np.ndarray: return self._value

    def _build_interp(self):
        if self._interp is None:
            self._interp = RegularGridInterpolator(
                (self._R, self._Z, self._Phi), self._value,
                method='linear', bounds_error=False, fill_value=np.nan)

    def __call__(self, coords: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate at coords, shape (..., 3) = (R, Z, phi)."""
        self._build_interp()
        coords = np.asarray(coords, dtype=float)
        if self.field_periods > 1:
            coords = coords.copy()
            coords[..., 2] = coords[..., 2] % (2 * np.pi / self.field_periods)
        shape = coords.shape[:-1]
        pts = coords.reshape(-1, 3)
        return self._interp(pts).reshape(shape)

    # Legacy call signature (R, Z, phi) for backward compat
    def interpolate_at(self, R, Z, phi=None):
        R = np.asarray(R, dtype=float)
        Z = np.asarray(Z, dtype=float)
        if phi is None:
            phi = np.zeros_like(R)
        phi = np.asarray(phi, dtype=float)
        coords = np.stack([R.ravel(), Z.ravel(), phi.ravel()], axis=1)
        return self(coords).reshape(R.shape)

    def to_npz(self, path: str) -> None:
        np.savez_compressed(path, R=self._R, Z=self._Z, Phi=self._Phi,
                            value=self._value, field_periods=self.field_periods,
                            name=self.name, units=self.units)

    @classmethod
    def from_npz(cls, path: str) -> "ScalarField3DCylindrical":
        d = np.load(path, allow_pickle=True)
        return cls(R=d['R'], Z=d['Z'], Phi=d['Phi'], value=d['value'],
                   field_periods=int(d.get('field_periods', 1)),
                   name=str(d.get('name', '')), units=str(d.get('units', '')))


class VectorField3DCylindrical(VectorField3D):
    """Vector field (BR, BZ, BPhi)(R, Z, φ) on a regular cylindrical grid.

    Replaces BOTH:
      - pyna.field_data.CylindricalVectorField  (VR/VZ/VPhi naming)
      - legacy toroidal coil-field wrappers (eliminated) (BR/BZ/BPhi naming)

    Both naming conventions are supported via properties.

    Parameters
    ----------
    R, Z, Phi : 1D ndarray
    VR, VZ, VPhi : ndarray, shape (nR, nZ, nPhi)
        Vector components (also accessible as BR, BZ, BPhi for magnetic fields).
    field_periods : int
    name, units : str
    properties : FieldProperty
    """

    def __init__(
        self,
        R: np.ndarray,
        Z: np.ndarray,
        Phi: np.ndarray,
        VR: np.ndarray,
        VZ: np.ndarray,
        VPhi: np.ndarray,
        field_periods: int = 1,
        name: str = "",
        units: str = "",
        properties: FieldProperty = FieldProperty.NONE,
    ) -> None:
        super().__init__(properties=properties, name=name, units=units,
                         coords=_CylCoords3D())
        self._R = np.asarray(R, dtype=float)
        self._Z = np.asarray(Z, dtype=float)
        self._Phi = np.asarray(Phi, dtype=float)
        self._VR = np.asarray(VR, dtype=float)
        self._VZ = np.asarray(VZ, dtype=float)
        self._VPhi = np.asarray(VPhi, dtype=float)
        self.field_periods = field_periods
        shape = (len(self._R), len(self._Z), len(self._Phi))
        for arr, nm in [(self._VR,'VR'),(self._VZ,'VZ'),(self._VPhi,'VPhi')]:
            assert arr.shape == shape, f"{nm} shape {arr.shape} != {shape}"
        self._interp_VR: Optional[RegularGridInterpolator] = None
        self._interp_VZ: Optional[RegularGridInterpolator] = None
        self._interp_VPhi: Optional[RegularGridInterpolator] = None

    # Grid accessors
    @property
    def R(self) -> np.ndarray: return self._R
    @property
    def Z(self) -> np.ndarray: return self._Z
    @property
    def Phi(self) -> np.ndarray: return self._Phi

    # Both naming conventions
    @property
    def VR(self) -> np.ndarray: return self._VR
    @property
    def VZ(self) -> np.ndarray: return self._VZ
    @property
    def VPhi(self) -> np.ndarray: return self._VPhi
    @property
    def BR(self) -> np.ndarray: return self._VR   # magnetic field alias
    @property
    def BZ(self) -> np.ndarray: return self._VZ
    @property
    def BPhi(self) -> np.ndarray: return self._VPhi

    def _build_interps(self):
        if self._interp_VR is None:
            kw = dict(method='linear', bounds_error=False, fill_value=np.nan)
            axes = (self._R, self._Z, self._Phi)
            self._interp_VR   = RegularGridInterpolator(axes, self._VR,   **kw)
            self._interp_VZ   = RegularGridInterpolator(axes, self._VZ,   **kw)
            self._interp_VPhi = RegularGridInterpolator(axes, self._VPhi, **kw)

    def __call__(self, coords_or_R: np.ndarray, Z=None, Phi=None, **kwargs) -> np.ndarray:
        """Evaluate field.

        Two call signatures supported:
        - New: ``field(coords)``  where coords has shape (..., 3) → returns (..., 3)
        - Legacy: ``field(R, Z, phi)`` → returns (VR, VZ, VPhi) tuple (backward compat)
        """
        if Z is not None:
            # Legacy 3-arg call: (R, Z, phi) → (VR, VZ, VPhi)
            return self.interpolate_at(coords_or_R, Z, Phi)
        # New-style single-coords call
        self._build_interps()
        coords = np.asarray(coords_or_R, dtype=float)
        if self.field_periods > 1:
            coords = coords.copy()
            coords[..., 2] = coords[..., 2] % (2 * np.pi / self.field_periods)
        shape = coords.shape[:-1]
        pts = coords.reshape(-1, 3)
        VR   = self._interp_VR(pts).reshape(shape)
        VZ   = self._interp_VZ(pts).reshape(shape)
        VPhi = self._interp_VPhi(pts).reshape(shape)
        return np.stack([VR, VZ, VPhi], axis=-1)

    def interpolate_at(self, R, Z, Phi=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Legacy convenience: returns (VR, VZ, VPhi) at given coordinates."""
        R = np.asarray(R, dtype=float)
        Z = np.asarray(Z, dtype=float)
        if Phi is None:
            Phi = np.zeros_like(R)
        Phi = np.asarray(Phi, dtype=float)
        coords = np.stack([R.ravel(), Z.ravel(), Phi.ravel()], axis=1)
        out = self(coords)
        shape = R.shape
        return out[...,0].reshape(shape), out[...,1].reshape(shape), out[...,2].reshape(shape)

    @classmethod
    def from_callable(cls, func, R, Z, Phi, n_workers=1, name="", units="", **kwargs):
        """Build from a callable.

        Supports two calling conventions:
        - New: ``func(R_arr, Z_arr, Phi_arr) -> (VR, VZ, VPhi)``  (vectorised)
        - Legacy: ``func(np.array([R, Z, phi])) -> (VR, VZ, VPhi)``  (point-by-point)

        The convention is auto-detected by trying the new signature first.
        """
        R = np.asarray(R, dtype=float)
        Z = np.asarray(Z, dtype=float)
        Phi = np.asarray(Phi, dtype=float)
        RR, ZZ, PP = np.meshgrid(R, Z, Phi, indexing='ij')
        # Try new vectorised convention first
        try:
            result = func(RR.ravel(), ZZ.ravel(), PP.ravel())
            VR, VZ, VP = result
        except TypeError:
            # Fall back to legacy point-by-point convention
            n = RR.size
            VR_flat = np.empty(n, dtype=float)
            VZ_flat = np.empty(n, dtype=float)
            VP_flat = np.empty(n, dtype=float)
            R_f, Z_f, P_f = RR.ravel(), ZZ.ravel(), PP.ravel()
            for k in range(n):
                res = func(np.array([R_f[k], Z_f[k], P_f[k]]))
                VR_flat[k] = res[0]
                VZ_flat[k] = res[1]
                VP_flat[k] = res[2]
            VR, VZ, VP = VR_flat, VZ_flat, VP_flat
        shape = (len(R), len(Z), len(Phi))
        return cls(R, Z, Phi,
                   np.asarray(VR).reshape(shape),
                   np.asarray(VZ).reshape(shape),
                   np.asarray(VP).reshape(shape),
                   name=name, units=units, **kwargs)

    def to_npz(self, path: str) -> None:
        np.savez_compressed(path, R=self._R, Z=self._Z, Phi=self._Phi,
                            VR=self._VR, VZ=self._VZ, VPhi=self._VPhi,
                            field_periods=self.field_periods,
                            name=self.name, units=self.units)

    @classmethod
    def from_npz(cls, path: str) -> "VectorField3DCylindrical":
        d = np.load(path, allow_pickle=True)
        # Support both VR/VZ/VPhi and BR/BZ/BPhi keys
        VR = d.get('VR', d.get('BR'))
        VZ = d.get('VZ', d.get('BZ'))
        VP = d.get('VPhi', d.get('BPhi'))
        return cls(R=d['R'], Z=d['Z'], Phi=d['Phi'], VR=VR, VZ=VZ, VPhi=VP,
                   field_periods=int(d.get('field_periods', 1)),
                   name=str(d.get('name', '')), units=str(d.get('units', '')))


class VectorField3DAxiSymmetric(VectorField3DCylindrical):
    """Axisymmetric vector field: components depend only on (R, Z)."""

    def __init__(self, R, Z, VR_2d, VZ_2d, VPhi_2d,
                 name="", units="", properties=FieldProperty.NONE):
        Phi = np.array([0.0])
        def _e(a): return np.asarray(a, dtype=float)[:, :, np.newaxis]
        super().__init__(R, Z, Phi, _e(VR_2d), _e(VZ_2d), _e(VPhi_2d),
                         field_periods=1, name=name, units=units, properties=properties)

    def __call__(self, coords: np.ndarray, **kwargs) -> np.ndarray:
        """Override: ignore phi (axisymmetric), always query at phi=0."""
        coords = np.asarray(coords, dtype=float)
        coords_axi = coords.copy()
        coords_axi[..., 2] = 0.0  # force phi=0
        return super().__call__(coords_axi, **kwargs)

    def interpolate_at(self, R, Z, Phi=None):
        """Override: ignore Phi."""
        R = np.asarray(R, dtype=float)
        Z = np.asarray(Z, dtype=float)
        Phi_zero = np.zeros_like(R)
        return super().interpolate_at(R, Z, Phi_zero)


class ScalarField3DAxiSymmetric(ScalarField3DCylindrical):
    """Axisymmetric scalar field: value depends only on (R, Z)."""

    def __init__(self, R, Z, value_2d, name="", units="", properties=FieldProperty.NONE):
        Phi = np.array([0.0])
        value_3d = np.asarray(value_2d, dtype=float)[:, :, np.newaxis]
        super().__init__(R, Z, Phi, value_3d, field_periods=1,
                         name=name, units=units, properties=properties)

    @property
    def value_2d(self) -> np.ndarray:
        return self._value[:, :, 0]

    def __call__(self, coords: np.ndarray, **kwargs) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        coords_axi = coords.copy()
        coords_axi[..., 2] = 0.0
        return super().__call__(coords_axi, **kwargs)
