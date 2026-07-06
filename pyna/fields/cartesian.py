"""Cartesian grid field containers."""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pyna.fields.base import VectorField3D
from pyna.fields.coords import CoordsCartesian
from pyna.fields.properties import FieldProperty


class VectorFieldCartesian(VectorField3D):
    """Vector field ``(Vx, Vy, Vz)(x, y, z)`` on a regular Cartesian grid."""

    __slots__ = (
        "_X",
        "_Y",
        "_Z",
        "_VX",
        "_VY",
        "_VZ",
        "_interp_VX",
        "_interp_VY",
        "_interp_VZ",
        "label",
    )

    component_order = ("x", "y", "z")

    def __init__(
        self,
        X: Sequence[float],
        Y: Sequence[float],
        Z: Sequence[float],
        VX: np.ndarray,
        VY: np.ndarray,
        VZ: np.ndarray,
        *,
        label: Optional[str] = None,
        name: str = "",
        units: str = "",
        properties: FieldProperty = FieldProperty.NONE,
    ) -> None:
        self._X = np.asarray(X, dtype=np.float64)
        self._Y = np.asarray(Y, dtype=np.float64)
        self._Z = np.asarray(Z, dtype=np.float64)
        self._VX = np.asarray(VX, dtype=np.float64)
        self._VY = np.asarray(VY, dtype=np.float64)
        self._VZ = np.asarray(VZ, dtype=np.float64)
        expected = (self._X.size, self._Y.size, self._Z.size)
        if self._VX.shape != expected or self._VY.shape != expected or self._VZ.shape != expected:
            raise ValueError(
                "VX, VY, VZ must all have shape "
                f"{expected}; got {self._VX.shape}, {self._VY.shape}, {self._VZ.shape}"
            )
        if min(expected) < 2:
            raise ValueError("VectorFieldCartesian requires at least two grid points on each axis.")
        if not (_is_strictly_increasing(self._X) and _is_strictly_increasing(self._Y) and _is_strictly_increasing(self._Z)):
            raise ValueError("X, Y, Z grids must be strictly increasing.")
        super().__init__(properties=properties, name=name, units=units, coords=CoordsCartesian(3))
        self.label = label
        self._interp_VX: RegularGridInterpolator | None = None
        self._interp_VY: RegularGridInterpolator | None = None
        self._interp_VZ: RegularGridInterpolator | None = None

    @property
    def X(self) -> np.ndarray:
        return self._X

    @property
    def Y(self) -> np.ndarray:
        return self._Y

    @property
    def Z(self) -> np.ndarray:
        return self._Z

    @property
    def VX(self) -> np.ndarray:
        return self._VX

    @property
    def VY(self) -> np.ndarray:
        return self._VY

    @property
    def VZ(self) -> np.ndarray:
        return self._VZ

    @property
    def coordinate_names(self) -> tuple[str, str, str]:
        return ("x", "y", "z")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._VX.shape

    @property
    def components(self) -> np.ndarray:
        return np.stack([self._VX, self._VY, self._VZ], axis=0)

    @property
    def abs(self) -> np.ndarray:
        return np.sqrt(self._VX * self._VX + self._VY * self._VY + self._VZ * self._VZ)

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(self._VX * self._VX + self._VY * self._VY + self._VZ * self._VZ)))

    def _build_interps(self) -> None:
        if self._interp_VX is not None:
            return
        kw = dict(method="linear", bounds_error=False, fill_value=np.nan)
        axes = (self._X, self._Y, self._Z)
        self._interp_VX = RegularGridInterpolator(axes, self._VX, **kw)
        self._interp_VY = RegularGridInterpolator(axes, self._VY, **kw)
        self._interp_VZ = RegularGridInterpolator(axes, self._VZ, **kw)

    def __call__(self, coords: np.ndarray, **kwargs: Any) -> np.ndarray:
        del kwargs
        self._build_interps()
        pts = np.asarray(coords, dtype=float)
        if pts.shape[-1:] != (3,):
            raise ValueError("VectorFieldCartesian coordinates must have final dimension 3.")
        shape = pts.shape[:-1]
        flat = pts.reshape(-1, 3)
        assert self._interp_VX is not None and self._interp_VY is not None and self._interp_VZ is not None
        vx = self._interp_VX(flat).reshape(shape)
        vy = self._interp_VY(flat).reshape(shape)
        vz = self._interp_VZ(flat).reshape(shape)
        return np.stack([vx, vy, vz], axis=-1)

    @classmethod
    def from_callable(
        cls,
        func,
        X: Sequence[float],
        Y: Sequence[float],
        Z: Sequence[float],
        *,
        name: str = "",
        units: str = "",
        label: str | None = None,
        **kwargs: Any,
    ) -> "VectorFieldCartesian":
        X_arr = np.asarray(X, dtype=np.float64)
        Y_arr = np.asarray(Y, dtype=np.float64)
        Z_arr = np.asarray(Z, dtype=np.float64)
        XX, YY, ZZ = np.meshgrid(X_arr, Y_arr, Z_arr, indexing="ij")
        try:
            result = func(XX.ravel(), YY.ravel(), ZZ.ravel())
            VX, VY, VZ = result
        except TypeError:
            pts = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
            values = np.asarray([func(pt) for pt in pts], dtype=np.float64)
            VX, VY, VZ = values[:, 0], values[:, 1], values[:, 2]
        shape = (X_arr.size, Y_arr.size, Z_arr.size)
        return cls(
            X_arr,
            Y_arr,
            Z_arr,
            np.asarray(VX, dtype=np.float64).reshape(shape),
            np.asarray(VY, dtype=np.float64).reshape(shape),
            np.asarray(VZ, dtype=np.float64).reshape(shape),
            name=name,
            units=units,
            label=label,
            **kwargs,
        )

    @classmethod
    def from_field_cache(cls, cache: Mapping[str, Any], *, label: str = "") -> "VectorFieldCartesian":
        X = cache.get("X_grid", cache.get("X", cache.get("x")))
        Y = cache.get("Y_grid", cache.get("Y", cache.get("y")))
        Z = cache.get("Z_grid", cache.get("Z", cache.get("z")))
        if X is None or Y is None or Z is None:
            raise KeyError("Cartesian field cache must provide X/Y/Z axes.")
        VX = cache.get("VX", cache.get("Vx", cache.get("Jx")))
        VY = cache.get("VY", cache.get("Vy", cache.get("Jy")))
        VZ = cache.get("VZ", cache.get("Vz", cache.get("Jz")))
        if VX is None or VY is None or VZ is None:
            raise KeyError("Cartesian field cache must provide VX/VY/VZ arrays.")
        return cls(
            X,
            Y,
            Z,
            VX,
            VY,
            VZ,
            label=label or str(cache.get("label", "")),
            name=str(cache.get("name", "")),
            units=str(cache.get("units", "")),
        )

    def to_field_cache(self) -> dict[str, np.ndarray]:
        return {
            "X": np.ascontiguousarray(self._X),
            "Y": np.ascontiguousarray(self._Y),
            "Z": np.ascontiguousarray(self._Z),
            "VX": np.ascontiguousarray(self._VX),
            "VY": np.ascontiguousarray(self._VY),
            "VZ": np.ascontiguousarray(self._VZ),
        }

    def to_npz(self, path: str) -> None:
        np.savez_compressed(
            path,
            X=self._X,
            Y=self._Y,
            Z=self._Z,
            VX=self._VX,
            VY=self._VY,
            VZ=self._VZ,
            name=self.name,
            units=self.units,
        )

    @classmethod
    def from_npz(cls, path: str) -> "VectorFieldCartesian":
        data = np.load(path, allow_pickle=False)
        return cls(
            data["X"],
            data["Y"],
            data["Z"],
            data["VX"],
            data["VY"],
            data["VZ"],
            name=str(data.get("name", "")),
            units=str(data.get("units", "")),
        )

    def __repr__(self) -> str:
        lbl = f", label={self.label!r}" if self.label else ""
        return f"VectorFieldCartesian({self.shape[0]}x{self.shape[1]}x{self.shape[2]}, rms={self.rms:.3g}{lbl})"


def as_vector_field_cartesian(field_like: Any, *, label: str = "") -> VectorFieldCartesian:
    """Normalize common Cartesian field inputs to :class:`VectorFieldCartesian`."""

    if isinstance(field_like, VectorFieldCartesian):
        return field_like
    if isinstance(field_like, Mapping):
        return VectorFieldCartesian.from_field_cache(field_like, label=label)
    raise TypeError("expected VectorFieldCartesian or a Cartesian field-cache mapping")


def _is_strictly_increasing(values: np.ndarray) -> bool:
    arr = np.asarray(values, dtype=np.float64)
    return bool(arr.ndim == 1 and arr.size >= 2 and np.all(np.diff(arr) > 0.0))


__all__ = ["VectorFieldCartesian", "as_vector_field_cartesian"]
