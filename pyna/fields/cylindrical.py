"""Concrete cylindrical-coordinate field implementations.

These replace (and are backward-compatible with):
  - pyna.field_data.CylindricalScalarField
  - pyna.field_data.CylindricalVectorField
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pyna.fields.base import ScalarField3D, VectorField3D
from pyna.fields.properties import FieldProperty
from pyna.fields.coords import Coords3DCylindrical as _CylCoords3D


def _extend_endpoint_false_periodic_phi(
    phi: np.ndarray,
    values: np.ndarray,
    period: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Close a uniform endpoint=False field-period grid for interpolation."""

    phi_ext, value_ext = close_periodic_phi_grid(phi, values, period=period)
    return phi_ext, value_ext


def _span_is_twopi_fraction(span: float) -> bool:
    if not np.isfinite(span) or span <= 0.0:
        return False
    nfp = (2.0 * np.pi) / span
    nfp_round = round(nfp)
    if nfp_round < 1:
        return False
    return abs(nfp - nfp_round) <= max(1.0e-8, 1.0e-8 * abs(nfp_round))


def _periodic_endpoint_values_match(values: tuple[np.ndarray, ...]) -> bool:
    if not values:
        return False
    for arr in values:
        val = np.asarray(arr)
        if val.ndim < 3 or val.shape[2] < 2:
            return False
        if not np.allclose(val[:, :, 0], val[:, :, -1], rtol=1.0e-8, atol=1.0e-10):
            return False
    return True


def _normalize_field_periods(field_periods: int = 1, nfp: int | None = None) -> int:
    periods = int(field_periods)
    if nfp is not None:
        nfp_i = int(nfp)
        if nfp_i < 1:
            raise ValueError("nfp must be a positive integer")
        if periods not in (1, nfp_i):
            raise ValueError("field_periods and nfp disagree")
        periods = nfp_i
    if periods < 1:
        raise ValueError("field_periods must be a positive integer")
    return periods


def validate_phi_grid(
    phi: np.ndarray,
    *,
    nfp: int = 1,
    name: str = "Phi",
    allow_single: bool = True,
) -> np.ndarray:
    """Validate and return a cylindrical toroidal-angle grid.

    The grid is expected to be one-dimensional, finite, and strictly
    increasing.  For ``nfp > 1`` it must describe one field period, because
    field evaluation wraps query angles modulo ``2*pi/nfp``.
    """

    phi_arr = np.asarray(phi, dtype=np.float64)
    if phi_arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if phi_arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if phi_arr.size == 1:
        if not allow_single:
            raise ValueError(f"{name} must contain at least two points")
        if not np.isfinite(phi_arr[0]):
            raise ValueError(f"{name} must contain only finite values")
        return np.ascontiguousarray(phi_arr, dtype=np.float64)
    if not np.all(np.isfinite(phi_arr)):
        raise ValueError(f"{name} must contain only finite values")
    dphi = np.diff(phi_arr)
    if np.any(dphi <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    nfp_i = _normalize_field_periods(nfp)
    if nfp_i > 1:
        period = 2.0 * np.pi / nfp_i
        tol = max(1.0e-10, 1.0e-10 * abs(period))
        if phi_arr[0] < -tol or phi_arr[-1] > period + tol:
            raise ValueError(
                f"{name} for nfp={nfp_i} must lie within one field period [0, 2*pi/nfp]"
            )
        if (phi_arr[-1] - phi_arr[0]) > period + tol:
            raise ValueError(f"{name} span exceeds one field period for nfp={nfp_i}")
    return np.ascontiguousarray(phi_arr, dtype=np.float64)


def _validate_closed_periodic_endpoint(
    phi: np.ndarray,
    values: tuple[np.ndarray, ...],
    *,
    nfp: int,
    name: str,
) -> None:
    if phi.size < 2:
        return
    period = 2.0 * np.pi / max(int(nfp), 1)
    tol = max(1.0e-10, 1.0e-10 * abs(period))
    if abs(float(phi[-1] - phi[0]) - period) > tol:
        return
    if not _periodic_endpoint_values_match(values):
        raise ValueError(f"{name} has a closed periodic Phi endpoint but first/last field slices differ")


def close_periodic_phi_grid(
    phi: np.ndarray,
    *values: np.ndarray,
    period: Optional[float] = None,
) -> tuple[np.ndarray, ...]:
    """Return a phi grid closed by a duplicate first slice.

    If *period* is not supplied, the period is inferred as ``phi[-1] + dphi`` for
    endpoint=False grids.  A grid that already spans ``2*pi/Nfp`` and whose
    endpoint values match the first slice is treated as already closed.
    """

    phi_arr = np.asarray(phi, dtype=np.float64)
    value_arrs = tuple(np.asarray(v, dtype=np.float64) for v in values)
    if phi_arr.ndim != 1:
        raise ValueError("phi grid must be one-dimensional")
    if phi_arr.size == 0:
        return (np.ascontiguousarray(phi_arr, dtype=np.float64), *value_arrs)
    if phi_arr.size == 1:
        if period is None or not np.isfinite(period) or period <= 0.0:
            return (np.ascontiguousarray(phi_arr, dtype=np.float64), *value_arrs)
        phi_ext = np.ascontiguousarray(
            [float(phi_arr[0]), float(phi_arr[0]) + float(period)],
            dtype=np.float64,
        )
        extended = [phi_ext]
        for val in value_arrs:
            if val.ndim < 3 or val.shape[2] != 1:
                raise ValueError("periodic field values must have phi on axis 2")
            extended.append(np.concatenate([val, val], axis=2))
        return tuple(extended)

    dphi = float(phi_arr[1] - phi_arr[0])
    if not np.isfinite(dphi) or dphi <= 0.0:
        return (np.ascontiguousarray(phi_arr, dtype=np.float64), *value_arrs)

    if period is not None and np.isfinite(period) and period > 0.0:
        endpoint = float(phi_arr[0]) + float(period)
        tol = max(1.0e-10, 1.0e-10 * abs(float(period)))
        already_closed = abs(float(phi_arr[-1]) - endpoint) <= tol
    else:
        span = float(phi_arr[-1] - phi_arr[0])
        already_closed = (
            _span_is_twopi_fraction(span)
            and _periodic_endpoint_values_match(value_arrs)
        )
        endpoint = float(phi_arr[-1]) + dphi

    if already_closed:
        return (np.ascontiguousarray(phi_arr, dtype=np.float64), *value_arrs)

    phi_ext = np.ascontiguousarray(np.append(phi_arr, endpoint), dtype=np.float64)
    extended = [phi_ext]
    for val in value_arrs:
        if val.ndim < 3 or val.shape[2] != phi_arr.size:
            raise ValueError("periodic field values must have phi on axis 2")
        extended.append(np.concatenate([val, val[:, :, :1]], axis=2))
    return tuple(extended)


def close_periodic_field_cache_phi(field_cache: Mapping[str, Any]) -> dict[str, Any]:
    """Return a field-cache copy whose ``Phi_grid`` is closed periodically."""

    out = dict(field_cache)
    phi_raw = np.asarray(field_cache["Phi_grid"], dtype=np.float64)
    field_periods = int(field_cache.get("field_periods", 1))
    period = (
        2.0 * np.pi / max(field_periods, 1)
        if "field_periods" in field_cache or phi_raw.size <= 1
        else None
    )
    Phi, BR, BZ, BPhi = close_periodic_phi_grid(
        phi_raw,
        field_cache["BR"],
        field_cache["BZ"],
        field_cache["BPhi"],
        period=period,
    )
    out["Phi_grid"] = np.ascontiguousarray(Phi, dtype=np.float64)
    out["BR"] = np.ascontiguousarray(BR, dtype=np.float64)
    out["BZ"] = np.ascontiguousarray(BZ, dtype=np.float64)
    out["BPhi"] = np.ascontiguousarray(BPhi, dtype=np.float64)
    out["R_grid"] = np.ascontiguousarray(field_cache["R_grid"], dtype=np.float64)
    out["Z_grid"] = np.ascontiguousarray(field_cache["Z_grid"], dtype=np.float64)
    return out


@dataclass(frozen=True)
class CylindricalFieldArrays:
    """Contiguous cylindrical field arrays for low-level backends.

    High-level APIs should accept :class:`VectorFieldCylind`.  This container
    exists only at bridge boundaries such as cyna, where flat component arrays
    are unavoidable.
    """

    R_grid: np.ndarray
    Z_grid: np.ndarray
    Phi_grid: np.ndarray
    BR: np.ndarray
    BZ: np.ndarray
    BPhi: np.ndarray
    field_periods: int = 1

    @property
    def BR_flat(self) -> np.ndarray:
        return self.BR.ravel()

    @property
    def BZ_flat(self) -> np.ndarray:
        return self.BZ.ravel()

    @property
    def BPhi_flat(self) -> np.ndarray:
        return self.BPhi.ravel()

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.BR.shape

    def as_field_cache(self) -> dict[str, np.ndarray]:
        return {
            "BR": self.BR,
            "BZ": self.BZ,
            "BPhi": self.BPhi,
            "R_grid": self.R_grid,
            "Z_grid": self.Z_grid,
            "Phi_grid": self.Phi_grid,
            "field_periods": int(self.field_periods),
        }

    def cyna_component_args(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return component arrays in the public cyna ABI order."""

        return self.BR_flat, self.BZ_flat, self.BPhi_flat

    def cyna_grid_args(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.R_grid, self.Z_grid, self.Phi_grid


class ScalarFieldCylind(ScalarField3D):
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

    __slots__ = (
        "_R",
        "_Z",
        "_Phi",
        "_value",
        "_value_2d_view",
        "_shape",
        "_nR",
        "_nZ",
        "_nPhi",
        "_axisymmetric",
        "_interp",
        "field_periods",
    )

    def __init__(
        self,
        R: np.ndarray,
        Z: np.ndarray,
        Phi: np.ndarray,
        value: np.ndarray,
        field_periods: int = 1,
        nfp: int | None = None,
        name: str = "",
        units: str = "",
        properties: FieldProperty = FieldProperty.NONE,
        *,
        axisymmetric: bool = False,
    ) -> None:
        super().__init__(properties=properties, name=name, units=units,
                         coords=_CylCoords3D())
        field_periods = _normalize_field_periods(field_periods, nfp)
        R_arr = np.asarray(R, dtype=np.float64)
        Z_arr = np.asarray(Z, dtype=np.float64)
        Phi_arr = validate_phi_grid(Phi, nfp=field_periods)
        value_arr = np.asarray(value, dtype=np.float64)
        shape = (len(R_arr), len(Z_arr), len(Phi_arr))
        if value_arr.shape != shape:
            raise ValueError(f"value shape {value_arr.shape} mismatch; expected {shape}")
        _validate_closed_periodic_endpoint(Phi_arr, (value_arr,), nfp=field_periods, name="ScalarFieldCylind")
        self._R = R_arr
        self._Z = Z_arr
        self._Phi = Phi_arr
        self._value = value_arr
        self.field_periods = int(field_periods)
        self._shape = shape
        self._nR = int(shape[0])
        self._nZ = int(shape[1])
        self._nPhi = int(shape[2])
        self._axisymmetric = bool(axisymmetric)
        self._value_2d_view = (
            value_arr[:, :, 0] if self._axisymmetric and self._nPhi == 1 else None
        )
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
    @property
    def B(self) -> np.ndarray: return self._value
    @property
    def nfp(self) -> int: return int(self.field_periods)
    @property
    def field_period(self) -> float: return 2.0 * np.pi / max(int(self.field_periods), 1)

    def _build_interp(self):
        if self._interp is None:
            phi = self._Phi
            value = self._value
            if self.field_periods > 1:
                period = 2.0 * np.pi / max(int(self.field_periods), 1)
                phi, value = _extend_endpoint_false_periodic_phi(phi, value, period)
            self._interp = RegularGridInterpolator(
                (self._R, self._Z, phi), value,
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
    def from_npz(cls, path: str) -> "ScalarFieldCylind":
        d = np.load(path, allow_pickle=True)
        return cls(R=d['R'], Z=d['Z'], Phi=d['Phi'], value=d['value'],
                   field_periods=int(d.get('field_periods', d.get('nfp', 1))),
                   name=str(d.get('name', '')), units=str(d.get('units', '')))

    @property
    def is_axisymmetric(self) -> bool:
        return self._axisymmetric

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @property
    def nR(self) -> int:
        return self._nR

    @property
    def nZ(self) -> int:
        return self._nZ

    @property
    def nPhi(self) -> int:
        return self._nPhi

    @property
    def data(self) -> np.ndarray:
        return self._value

    def grad(self) -> "VectorFieldCylind":
        from pyna.fields.diff_ops import gradient
        return gradient(self)

    gradient = grad

    def laplacian(self) -> "ScalarFieldCylind":
        from pyna.fields.diff_ops import laplacian
        return laplacian(self)

    def hessian(self):
        from pyna.fields.diff_ops import hessian
        return hessian(self)

    def _binary(self, other: Any, op, opname: str, *, reverse: bool = False):
        return _scalar_binary_op(self, other, op, opname, reverse=reverse)

    def __add__(self, other: Any):
        return self._binary(other, np.add, "+")

    def __radd__(self, other: Any):
        return self.__add__(other)

    def __sub__(self, other: Any):
        return self._binary(other, np.subtract, "-")

    def __rsub__(self, other: Any):
        return self._binary(other, np.subtract, "-", reverse=True)

    def __mul__(self, other: Any):
        return self._binary(other, np.multiply, "*")

    def __rmul__(self, other: Any):
        return self.__mul__(other)

    def __truediv__(self, other: Any):
        return self._binary(other, np.divide, "/")

    def __rtruediv__(self, other: Any):
        return self._binary(other, np.divide, "/", reverse=True)

    def __neg__(self):
        return _make_scalar_result(
            self.R, self.Z, self.Phi, -self.value,
            axisym=False,
            name=f"-({self.name})" if self.name else "",
            units=self.units,
            properties=self.properties,
        )


class VectorFieldCylind(VectorField3D):
    """Vector field (BR, BZ, BPhi)(R, Z, φ) on a regular cylindrical grid.

    Replaces BOTH:
      - pyna.field_data.CylindricalVectorField  (VR/VZ/VPhi naming)
      - legacy toroidal coil-field wrappers (eliminated) (BR/BZ/BPhi naming)

    ``VectorFieldCylind`` is the canonical public name.  The old
    ``VectorFieldCylind`` name remains as a compatibility alias.

    Parameters
    ----------
    R, Z, Phi : 1D ndarray
        Grid axes.  ``Phi`` may be omitted for a fixed-section 2-D field.
    VR, VZ, VPhi or BR, BZ, BPhi : ndarray
        Vector components in canonical ``R, Z, Phi`` order.
    field_periods : int
    nfp : int, optional
        Alias for ``field_periods``.  If ``nfp > 1``, ``Phi`` must cover one
        field period, not a full torus.
    name, units : str
    properties : FieldProperty
    """

    __slots__ = (
        "_R",
        "_Z",
        "_Phi",
        "_VR",
        "_VZ",
        "_VPhi",
        "_shape",
        "_nR",
        "_nZ",
        "_nPhi",
        "_axisymmetric",
        "_section_mode",
        "_interp_VR",
        "_interp_VZ",
        "_interp_VPhi",
        "phi",
        "label",
        "field_periods",
    )

    component_order = ("R", "Z", "Phi")
    magnetic_component_order = ("BR", "BZ", "BPhi")

    def __init__(
        self,
        R: np.ndarray,
        Z: np.ndarray,
        Phi: Optional[np.ndarray] = None,
        VR: Optional[np.ndarray] = None,
        VZ: Optional[np.ndarray] = None,
        VPhi: Optional[np.ndarray] = None,
        *,
        BR: Optional[np.ndarray] = None,
        BZ: Optional[np.ndarray] = None,
        BPhi: Optional[np.ndarray] = None,
        phi: float = 0.0,
        label: Optional[str] = None,
        field_periods: int = 1,
        nfp: int | None = None,
        name: str = "",
        units: str = "",
        properties: FieldProperty = FieldProperty.NONE,
        section_mode: Optional[bool] = None,
        axisymmetric: bool = False,
    ) -> None:
        # New fixed-section positional form: VectorFieldCylind(R, Z, BR, BZ, BPhi)
        if VPhi is None and BR is None and BZ is None and BPhi is None and VZ is not None:
            BR, BZ, BPhi = Phi, VR, VZ
            Phi = None
            VR = VZ = VPhi = None

        if BR is None:
            BR = VR
        if BZ is None:
            BZ = VZ
        if BPhi is None:
            BPhi = VPhi
        if BR is None or BZ is None or BPhi is None:
            raise TypeError("VectorFieldCylind requires BR, BZ, BPhi components")
        field_periods = _normalize_field_periods(field_periods, nfp)

        BR_arr = np.asarray(BR, dtype=float)
        BZ_arr = np.asarray(BZ, dtype=float)
        BPhi_arr = np.asarray(BPhi, dtype=float)
        if BR_arr.shape != BZ_arr.shape or BR_arr.shape != BPhi_arr.shape:
            raise ValueError(
                "BR, BZ, BPhi must have identical shapes; "
                f"got {BR_arr.shape}, {BZ_arr.shape}, {BPhi_arr.shape}"
            )

        if BR_arr.ndim == 2:
            section = True if section_mode is None else bool(section_mode)
            Phi_arr = validate_phi_grid([phi] if Phi is None else Phi, nfp=field_periods)
            if Phi_arr.size != 1:
                raise ValueError("2-D VectorFieldCylind sections require exactly one Phi value")
            BR_3d = BR_arr[:, :, np.newaxis]
            BZ_3d = BZ_arr[:, :, np.newaxis]
            BPhi_3d = BPhi_arr[:, :, np.newaxis]
        elif BR_arr.ndim == 3:
            section = False if section_mode is None else bool(section_mode)
            if Phi is None:
                if BR_arr.shape[2] == 1:
                    Phi_arr = np.asarray([phi], dtype=float)
                else:
                    period = 2.0 * np.pi / field_periods
                    Phi_arr = np.linspace(0.0, period, BR_arr.shape[2], endpoint=False)
            else:
                Phi_arr = validate_phi_grid(Phi, nfp=field_periods)
            if Phi_arr.size != BR_arr.shape[2]:
                raise ValueError(
                    f"Phi length {Phi_arr.size} does not match component nPhi={BR_arr.shape[2]}"
                )
            BR_3d = BR_arr
            BZ_3d = BZ_arr
            BPhi_3d = BPhi_arr
        else:
            raise ValueError("BR, BZ, BPhi must be 2-D section arrays or 3-D grid arrays")

        super().__init__(properties=properties, name=name, units=units,
                         coords=_CylCoords3D())
        self._R = np.asarray(R, dtype=np.float64)
        self._Z = np.asarray(Z, dtype=np.float64)
        self._Phi = Phi_arr
        self._VR = BR_3d
        self._VZ = BZ_3d
        self._VPhi = BPhi_3d
        self.phi = float(phi)
        self.label = label
        self._section_mode = section
        self.field_periods = int(field_periods)
        self._axisymmetric = bool(axisymmetric)
        shape = (len(self._R), len(self._Z), len(self._Phi))
        for arr, nm in [(self._VR, 'VR'), (self._VZ, 'VZ'), (self._VPhi, 'VPhi')]:
            if arr.shape != shape:
                raise ValueError(f"{nm} shape {arr.shape} != {shape}")
        _validate_closed_periodic_endpoint(
            self._Phi,
            (self._VR, self._VZ, self._VPhi),
            nfp=self.field_periods,
            name="VectorFieldCylind",
        )
        self._shape = shape[:2] if section else shape
        self._nR = int(shape[0])
        self._nZ = int(shape[1])
        self._nPhi = int(shape[2])
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
    @property
    def R_arr(self) -> np.ndarray: return self._R
    @property
    def Z_arr(self) -> np.ndarray: return self._Z
    @property
    def coordinate_names(self) -> Tuple[str, str, str]: return ("R", "Z", "phi")
    @property
    def is_section(self) -> bool: return self._section_mode
    @property
    def nPhi(self) -> int: return self._nPhi
    @property
    def nfp(self) -> int: return int(self.field_periods)
    @property
    def field_period(self) -> float: return 2.0 * np.pi / max(int(self.field_periods), 1)

    # Both naming conventions
    @property
    def VR(self) -> np.ndarray: return self.BR
    @property
    def VZ(self) -> np.ndarray: return self.BZ
    @property
    def VPhi(self) -> np.ndarray: return self.BPhi
    @property
    def BR(self) -> np.ndarray: return self._VR[:, :, 0] if self._section_mode else self._VR
    @property
    def BZ(self) -> np.ndarray: return self._VZ[:, :, 0] if self._section_mode else self._VZ
    @property
    def BPhi(self) -> np.ndarray: return self._VPhi[:, :, 0] if self._section_mode else self._VPhi

    @property
    def components(self) -> np.ndarray:
        if self._section_mode:
            return np.stack(
                (self._VR[:, :, 0], self._VZ[:, :, 0], self._VPhi[:, :, 0]),
                axis=0,
            )
        return np.stack((self._VR, self._VZ, self._VPhi), axis=0)

    @property
    def components_3d(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._VR, self._VZ, self._VPhi

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(
            self._VR * self._VR + self._VZ * self._VZ + self._VPhi * self._VPhi
        )))

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def nR(self) -> int:
        return self._nR

    @property
    def nZ(self) -> int:
        return self._nZ

    @property
    def abs(self) -> np.ndarray:
        mag = np.sqrt(
            self._VR * self._VR + self._VZ * self._VZ + self._VPhi * self._VPhi
        )
        return mag[:, :, 0] if self._section_mode else mag

    @property
    def poloidal_abs(self) -> np.ndarray:
        mag = np.sqrt(self._VR * self._VR + self._VZ * self._VZ)
        return mag[:, :, 0] if self._section_mode else mag

    @property
    def is_axisymmetric(self) -> bool:
        return self._axisymmetric

    def _build_interps(self):
        if self._interp_VR is None:
            kw = dict(method='linear', bounds_error=False, fill_value=np.nan)
            phi = self._Phi
            vr = self._VR
            vz = self._VZ
            vphi = self._VPhi
            if self.field_periods > 1:
                period = 2.0 * np.pi / max(int(self.field_periods), 1)
                phi, vr = _extend_endpoint_false_periodic_phi(phi, vr, period)
                _, vz = _extend_endpoint_false_periodic_phi(self._Phi, vz, period)
                _, vphi = _extend_endpoint_false_periodic_phi(self._Phi, vphi, period)
            axes = (self._R, self._Z, phi)
            self._interp_VR   = RegularGridInterpolator(axes, vr,   **kw)
            self._interp_VZ   = RegularGridInterpolator(axes, vz,   **kw)
            self._interp_VPhi = RegularGridInterpolator(axes, vphi, **kw)

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
    def zero_like(cls, other: "VectorFieldCylind", label: str = "") -> "VectorFieldCylind":
        z = np.zeros_like(other.BR)
        return cls(
            R=other.R_arr,
            Z=other.Z_arr,
            Phi=None if other.is_section else other.Phi,
            BR=z,
            BZ=z,
            BPhi=z,
            phi=getattr(other, "phi", 0.0),
            field_periods=other.field_periods,
            label=label,
            section_mode=other.is_section,
        )

    @classmethod
    def from_components(
        cls,
        R: np.ndarray,
        Z: np.ndarray,
        Phi: np.ndarray,
        BR: np.ndarray,
        BZ: np.ndarray,
        BPhi: np.ndarray,
        *,
        label: str = "",
        field_periods: int = 1,
    ) -> "VectorFieldCylind":
        return cls(R=R, Z=Z, Phi=Phi, BR=BR, BZ=BZ, BPhi=BPhi,
                   label=label, field_periods=field_periods)

    @classmethod
    def from_field_cache(
        cls,
        cache: Mapping[str, Any],
        *,
        phi_idx: Optional[int] = None,
        label: str = "",
    ) -> "VectorFieldCylind":
        R = cache.get("R_grid", cache.get("R"))
        Z = cache.get("Z_grid", cache.get("Z"))
        Phi = cache.get("Phi_grid", cache.get("Phi"))
        if R is None or Z is None or Phi is None:
            raise KeyError("field cache must provide R_grid/Z_grid/Phi_grid or R/Z/Phi")
        field_periods = int(cache.get("field_periods", cache.get("nfp", 1)))
        if phi_idx is None:
            return cls(
                R=R,
                Z=Z,
                Phi=Phi,
                BR=cache["BR"],
                BZ=cache["BZ"],
                BPhi=cache["BPhi"],
                field_periods=field_periods,
                label=label or str(cache.get("label", "")),
            )
        phi_idx_i = int(phi_idx)
        return cls(
            R=R,
            Z=Z,
            BR=np.asarray(cache["BR"])[:, :, phi_idx_i],
            BZ=np.asarray(cache["BZ"])[:, :, phi_idx_i],
            BPhi=np.asarray(cache["BPhi"])[:, :, phi_idx_i],
            phi=float(np.asarray(Phi)[phi_idx_i]),
            field_periods=field_periods,
            label=label or f"cache_phi{phi_idx_i}",
            section_mode=True,
        )

    @classmethod
    def from_cache(cls, cache: Mapping[str, Any], phi_idx: int = 0, *, label: str = "") -> "VectorFieldCylind":
        """Backward-compatible fixed-section constructor."""

        return cls.from_field_cache(cache, phi_idx=phi_idx, label=label)

    def downsample(self, skip: int) -> "VectorFieldCylind":
        if self.is_section:
            return type(self)(
                R=self.R_arr[::skip],
                Z=self.Z_arr[::skip],
                BR=self.BR[::skip, ::skip],
                BZ=self.BZ[::skip, ::skip],
                BPhi=self.BPhi[::skip, ::skip],
                phi=self.phi,
                label=self.label,
                section_mode=True,
            )
        return type(self)(
            R=self.R_arr[::skip],
            Z=self.Z_arr[::skip],
            Phi=self.Phi,
            BR=self.BR[::skip, ::skip, :],
            BZ=self.BZ[::skip, ::skip, :],
            BPhi=self.BPhi[::skip, ::skip, :],
            phi=self.phi,
            field_periods=self.field_periods,
            label=self.label,
        )

    def div(self) -> "ScalarFieldCylind":
        from pyna.fields.diff_ops import divergence
        return divergence(self)

    divergence = div

    def curl(self) -> "VectorFieldCylind":
        from pyna.fields.diff_ops import curl
        return curl(self)

    def jacobian(self):
        from pyna.fields.diff_ops import jacobian_field
        return jacobian_field(self)

    def dot(self, other: Any) -> "ScalarFieldCylind":
        return _vector_dot(self, other)

    def cross(self, other: Any) -> "VectorFieldCylind":
        return _vector_cross(self, other)

    def magnitude(self) -> "ScalarFieldCylind":
        BR, BZ, BPhi = self.components_3d
        mag = np.sqrt(BR**2 + BZ**2 + BPhi**2)
        return _make_scalar_result(
            self.R_arr, self.Z_arr, self.Phi, mag,
            axisym=self.is_axisymmetric,
            section=self.is_section,
            name=f"|{self.name}|" if self.name else "",
            units=self.units,
        )

    norm = magnitude

    def __add__(self, other: Any) -> "VectorFieldCylind":
        return _vector_binary_op(self, other, np.add, "+")

    def __radd__(self, other: Any) -> "VectorFieldCylind":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "VectorFieldCylind":
        return _vector_binary_op(self, other, np.subtract, "-")

    def __rsub__(self, other: Any) -> "VectorFieldCylind":
        return _vector_binary_op(self, other, np.subtract, "-", reverse=True)

    def __neg__(self) -> "VectorFieldCylind":
        BR, BZ, BPhi = self.components_3d
        return _make_vector_result(
            self.R_arr, self.Z_arr, self.Phi, -BR, -BZ, -BPhi,
            axisym=self.is_axisymmetric,
            section=self.is_section,
            name=f"-({self.name})" if self.name else "",
            units=self.units,
            properties=self.properties,
        )

    def __mul__(self, other: Any) -> "VectorFieldCylind":
        return _vector_scalar_binary_op(self, other, np.multiply, "*")

    def __rmul__(self, other: Any) -> "VectorFieldCylind":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "VectorFieldCylind":
        return _vector_scalar_binary_op(self, other, np.divide, "/")

    def cyna_arrays(self, *, extend_phi: bool = False) -> CylindricalFieldArrays:
        """Return contiguous arrays for cyna without exposing tuple order."""

        BR, BZ, BPhi = self.components_3d
        Rg = np.ascontiguousarray(self.R_arr, dtype=np.float64)
        Zg = np.ascontiguousarray(self.Z_arr, dtype=np.float64)
        Pg = np.ascontiguousarray(self.Phi, dtype=np.float64)
        BR3 = np.asarray(BR, dtype=np.float64)
        BZ3 = np.asarray(BZ, dtype=np.float64)
        BP3 = np.asarray(BPhi, dtype=np.float64)

        if extend_phi and Pg.size > 0:
            period = 2.0 * np.pi / max(int(self.field_periods), 1)
            Pg, BR3, BZ3, BP3 = close_periodic_phi_grid(
                Pg, BR3, BZ3, BP3, period=period
            )

        return CylindricalFieldArrays(
            R_grid=Rg,
            Z_grid=Zg,
            Phi_grid=np.ascontiguousarray(Pg, dtype=np.float64),
            BR=np.ascontiguousarray(BR3, dtype=np.float64),
            BZ=np.ascontiguousarray(BZ3, dtype=np.float64),
            BPhi=np.ascontiguousarray(BP3, dtype=np.float64),
            field_periods=int(self.field_periods),
        )

    def to_field_cache(self, *, extend_phi: bool = False) -> dict[str, np.ndarray]:
        """Return a named field-cache dict in ``BR, BZ, BPhi`` order."""

        return self.cyna_arrays(extend_phi=extend_phi).as_field_cache()

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
    def from_npz(cls, path: str) -> "VectorFieldCylind":
        d = np.load(path, allow_pickle=True)
        # Support both VR/VZ/VPhi and BR/BZ/BPhi keys
        VR = d.get('VR', d.get('BR'))
        VZ = d.get('VZ', d.get('BZ'))
        VP = d.get('VPhi', d.get('BPhi'))
        return cls(R=d['R'], Z=d['Z'], Phi=d['Phi'], VR=VR, VZ=VZ, VPhi=VP,
                   field_periods=int(d.get('field_periods', d.get('nfp', 1))),
                   name=str(d.get('name', '')), units=str(d.get('units', '')))

    def __repr__(self) -> str:
        lbl = f", label={self.label!r}" if self.label else ""
        dims = f"{self.nR}x{self.nZ}" if self.is_section else f"{self.nR}x{self.nZ}x{self.nPhi}"
        return f"VectorFieldCylind({dims}, rms={self.rms:.3g}{lbl})"


class VectorFieldCylindAxisym(VectorFieldCylind):
    """Axisymmetric vector field: components depend only on (R, Z)."""

    __slots__ = ()

    def __init__(self, R, Z, VR_2d=None, VZ_2d=None, VPhi_2d=None,
                 *, BR=None, BZ=None, BPhi=None,
                 name="", units="", properties=FieldProperty.NONE):
        if BR is None:
            BR = VR_2d
        if BZ is None:
            BZ = VZ_2d
        if BPhi is None:
            BPhi = VPhi_2d
        if BR is None or BZ is None or BPhi is None:
            raise TypeError("VectorFieldCylindAxisym requires BR, BZ, BPhi components")
        Phi = np.array([0.0])
        def _e(a): return np.asarray(a, dtype=float)[:, :, np.newaxis]
        super().__init__(R, Z, Phi, _e(BR), _e(BZ), _e(BPhi),
                         field_periods=1, name=name, units=units,
                         properties=properties, section_mode=False,
                         axisymmetric=True)

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


class ScalarFieldCylindAxisym(ScalarFieldCylind):
    """Axisymmetric scalar field: value depends only on (R, Z)."""

    __slots__ = ()

    def __init__(self, R, Z, value_2d=None, *, value=None, B=None,
                 name="", units="", properties=FieldProperty.NONE):
        if value_2d is None:
            value_2d = value if value is not None else B
        if value_2d is None:
            raise TypeError("ScalarFieldCylindAxisym requires a value array")
        Phi = np.array([0.0])
        value_3d = np.asarray(value_2d, dtype=float)[:, :, np.newaxis]
        super().__init__(R, Z, Phi, value_3d, field_periods=1,
                         name=name, units=units, properties=properties,
                         axisymmetric=True)

    @property
    def is_axisymmetric(self) -> bool:
        return True

    @property
    def value_2d(self) -> np.ndarray:
        return (
            self._value_2d_view
            if self._value_2d_view is not None
            else self._value[:, :, 0]
        )
    @property
    def B(self) -> np.ndarray: return self.value_2d

    def __call__(self, coords: np.ndarray, **kwargs) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        coords_axi = coords.copy()
        coords_axi[..., 2] = 0.0
        return super().__call__(coords_axi, **kwargs)


def _same_axis(a: np.ndarray, b: np.ndarray, name: str) -> None:
    if len(a) != len(b) or not np.allclose(a, b, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} grids differ")


def _field_is_axisym(field: Any) -> bool:
    return isinstance(field, (ScalarFieldCylindAxisym, VectorFieldCylindAxisym))


def _binary_grid(a: Any, b: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    _same_axis(a.R, b.R, "R")
    _same_axis(a.Z, b.Z, "Z")
    axisym = _field_is_axisym(a) and _field_is_axisym(b)
    section = (
        not axisym
        and getattr(a, "is_section", False)
        and getattr(b, "is_section", False)
    )
    if axisym:
        return a.R, a.Z, np.array([0.0]), True, False

    a_phi = np.asarray(a.Phi, dtype=float)
    b_phi = np.asarray(b.Phi, dtype=float)
    if _field_is_axisym(a):
        return b.R, b.Z, b_phi, False, False
    if _field_is_axisym(b):
        return a.R, a.Z, a_phi, False, False
    _same_axis(a_phi, b_phi, "Phi")
    return a.R, a.Z, a_phi, False, section


def _broadcast_scalar_values(field: ScalarFieldCylind, Phi: np.ndarray) -> np.ndarray:
    values = np.asarray(field.value, dtype=float)
    if _field_is_axisym(field) and len(Phi) > 1:
        return np.repeat(values, len(Phi), axis=2)
    if values.shape[2] != len(Phi):
        raise ValueError(f"scalar field nPhi={values.shape[2]} cannot broadcast to {len(Phi)}")
    return values


def _broadcast_vector_components(field: VectorFieldCylind, Phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    BR, BZ, BPhi = (np.asarray(a, dtype=float) for a in field.components_3d)
    if _field_is_axisym(field) and len(Phi) > 1:
        return (
            np.repeat(BR, len(Phi), axis=2),
            np.repeat(BZ, len(Phi), axis=2),
            np.repeat(BPhi, len(Phi), axis=2),
        )
    if BR.shape[2] != len(Phi):
        raise ValueError(f"vector field nPhi={BR.shape[2]} cannot broadcast to {len(Phi)}")
    return BR, BZ, BPhi


def _make_scalar_result(
    R: np.ndarray,
    Z: np.ndarray,
    Phi: np.ndarray,
    value: np.ndarray,
    *,
    axisym: bool = False,
    section: bool = False,
    name: str = "",
    units: str = "",
    properties: FieldProperty = FieldProperty.NONE,
) -> ScalarFieldCylind:
    value = np.asarray(value, dtype=float)
    if axisym:
        return ScalarFieldCylindAxisym(R, Z, value[:, :, 0], name=name, units=units,
                                      properties=properties)
    return ScalarFieldCylind(R, Z, Phi, value, name=name, units=units,
                             properties=properties)


def _make_vector_result(
    R: np.ndarray,
    Z: np.ndarray,
    Phi: np.ndarray,
    BR: np.ndarray,
    BZ: np.ndarray,
    BPhi: np.ndarray,
    *,
    axisym: bool = False,
    section: bool = False,
    name: str = "",
    units: str = "",
    properties: FieldProperty = FieldProperty.NONE,
) -> VectorFieldCylind:
    BR = np.asarray(BR, dtype=float)
    BZ = np.asarray(BZ, dtype=float)
    BPhi = np.asarray(BPhi, dtype=float)
    if axisym:
        return VectorFieldCylindAxisym(R, Z, BR=BR[:, :, 0], BZ=BZ[:, :, 0],
                                      BPhi=BPhi[:, :, 0], name=name, units=units,
                                      properties=properties)
    if section:
        return VectorFieldCylind(R, Z, BR=BR[:, :, 0], BZ=BZ[:, :, 0],
                                 BPhi=BPhi[:, :, 0], phi=float(Phi[0]),
                                 section_mode=True, name=name, units=units,
                                 properties=properties)
    return VectorFieldCylind(R=R, Z=Z, Phi=Phi, BR=BR, BZ=BZ, BPhi=BPhi,
                             name=name, units=units, properties=properties)


def as_scalar_field_cylindrical(field_like: Any) -> ScalarFieldCylind:
    if isinstance(field_like, ScalarFieldCylind):
        return field_like
    if isinstance(field_like, Mapping):
        R = field_like.get("R_grid", field_like.get("R"))
        Z = field_like.get("Z_grid", field_like.get("Z"))
        Phi = field_like.get("Phi_grid", field_like.get("Phi"))
        value = field_like.get("value", field_like.get("B", None))
        if R is None or Z is None or value is None:
            raise KeyError("scalar field cache must provide R/Z and value")
        if Phi is None:
            return ScalarFieldCylindAxisym(R, Z, value)
        return ScalarFieldCylind(R, Z, Phi, value)
    raise TypeError("expected ScalarFieldCylind-compatible object")


def _scalar_binary_op(self: ScalarFieldCylind, other: Any, op, opname: str, *, reverse: bool = False) -> ScalarFieldCylind:
    if np.isscalar(other):
        value = op(other, self.value) if reverse else op(self.value, other)
        return _make_scalar_result(
            self.R, self.Z, self.Phi, value,
            axisym=_field_is_axisym(self),
            name=f"({self.name}{opname}{other})" if self.name else "",
            units=self.units,
            properties=self.properties,
        )
    other = as_scalar_field_cylindrical(other)
    R, Z, Phi, axisym, section = _binary_grid(self, other)
    a = _broadcast_scalar_values(self, Phi)
    b = _broadcast_scalar_values(other, Phi)
    value = op(b, a) if reverse else op(a, b)
    return _make_scalar_result(R, Z, Phi, value, axisym=axisym, section=section,
                               units=self.units)


def _vector_binary_op(self: VectorFieldCylind, other: Any, op, opname: str, *, reverse: bool = False) -> VectorFieldCylind:
    other = as_vector_field_cylindrical(other)
    R, Z, Phi, axisym, section = _binary_grid(self, other)
    aR, aZ, aP = _broadcast_vector_components(self, Phi)
    bR, bZ, bP = _broadcast_vector_components(other, Phi)
    if reverse:
        BR, BZ, BPhi = op(bR, aR), op(bZ, aZ), op(bP, aP)
    else:
        BR, BZ, BPhi = op(aR, bR), op(aZ, bZ), op(aP, bP)
    return _make_vector_result(R, Z, Phi, BR, BZ, BPhi, axisym=axisym,
                               section=section, units=self.units)


def _vector_scalar_binary_op(self: VectorFieldCylind, other: Any, op, opname: str) -> VectorFieldCylind:
    BR, BZ, BPhi = self.components_3d
    if np.isscalar(other):
        return _make_vector_result(
            self.R_arr, self.Z_arr, self.Phi,
            op(BR, other), op(BZ, other), op(BPhi, other),
            axisym=_field_is_axisym(self),
            section=self.is_section,
            units=self.units,
            properties=self.properties,
        )
    scalar = as_scalar_field_cylindrical(other)
    R, Z, Phi, axisym, section = _binary_grid(self, scalar)
    vR, vZ, vP = _broadcast_vector_components(self, Phi)
    s = _broadcast_scalar_values(scalar, Phi)
    return _make_vector_result(R, Z, Phi, op(vR, s), op(vZ, s), op(vP, s),
                               axisym=axisym, section=section, units=self.units,
                               properties=self.properties)


def _vector_dot(self: VectorFieldCylind, other: Any) -> ScalarFieldCylind:
    other = as_vector_field_cylindrical(other)
    R, Z, Phi, axisym, section = _binary_grid(self, other)
    aR, aZ, aP = _broadcast_vector_components(self, Phi)
    bR, bZ, bP = _broadcast_vector_components(other, Phi)
    value = aR * bR + aZ * bZ + aP * bP
    return _make_scalar_result(R, Z, Phi, value, axisym=axisym, section=section,
                               name=f"{self.name}·{other.name}" if self.name or other.name else "",
                               units="")


def _vector_cross(self: VectorFieldCylind, other: Any) -> VectorFieldCylind:
    other = as_vector_field_cylindrical(other)
    R, Z, Phi, axisym, section = _binary_grid(self, other)
    aR, aZ, aP = _broadcast_vector_components(self, Phi)
    bR, bZ, bP = _broadcast_vector_components(other, Phi)
    BR = aP * bZ - aZ * bP
    BZ = aR * bP - aP * bR
    BPhi = aZ * bR - aR * bZ
    return _make_vector_result(R, Z, Phi, BR, BZ, BPhi, axisym=axisym,
                               section=section, units=self.units)


def as_vector_field_cylindrical(field_like: Any, *, label: str = "") -> VectorFieldCylind:
    """Normalize common cylindrical field inputs to :class:`VectorFieldCylind`.

    New code should pass a ``VectorFieldCylind`` object.  Dict and tuple support
    exists to bridge legacy field-cache code while the rest of the repository
    migrates away from component tuples.
    """

    if isinstance(field_like, VectorFieldCylind):
        return field_like

    if isinstance(field_like, Mapping):
        return VectorFieldCylind.from_field_cache(field_like, phi_idx=None, label=label)

    required = ("R", "Z", "Phi", "BR", "BZ", "BPhi")
    if all(hasattr(field_like, name) for name in required):
        return VectorFieldCylind(
            R=getattr(field_like, "R"),
            Z=getattr(field_like, "Z"),
            Phi=getattr(field_like, "Phi"),
            BR=getattr(field_like, "BR"),
            BZ=getattr(field_like, "BZ"),
            BPhi=getattr(field_like, "BPhi"),
            field_periods=getattr(field_like, "field_periods", getattr(field_like, "nfp", 1)),
            label=label or getattr(field_like, "label", "") or getattr(field_like, "name", ""),
        )

    if isinstance(field_like, (tuple, list)):
        if len(field_like) == 6:
            R, Z, Phi, BR, BZ, BPhi = field_like
            return VectorFieldCylind(R=R, Z=Z, Phi=Phi, BR=BR, BZ=BZ, BPhi=BPhi, label=label)
        if len(field_like) == 5:
            R, Z, BR, BZ, BPhi = field_like
            return VectorFieldCylind(R=R, Z=Z, BR=BR, BZ=BZ, BPhi=BPhi, label=label)

    raise TypeError(
        "expected VectorFieldCylind, field-cache dict, object with "
        "R/Z/Phi/BR/BZ/BPhi, or legacy tuple"
    )


as_vector_field_cylind = as_vector_field_cylindrical
as_scalar_field_cylind = as_scalar_field_cylindrical
