"""Concrete rank-2 tensor field on cylindrical grid."""
from __future__ import annotations
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pyna.fields.base import TensorField3D_rank2 as _TF3D_rank2_Base
from pyna.fields.properties import FieldProperty


class TensorField3D_rank2(_TF3D_rank2_Base):
    """Rank-2 tensor field T_ij(R, Z, φ) on a regular cylindrical grid.

    Data shape: (nR, nZ, nPhi, 3, 3) — spatial axes first, tensor indices last.

    Index convention (cylindrical):
      axis 0 = R component
      axis 1 = Z component
      axis 2 = φ component

    Parameters
    ----------
    R, Z, Phi : 1D ndarray
    data : ndarray, shape (nR, nZ, nPhi, 3, 3)
    name, units : str
    properties : FieldProperty
    """

    def __init__(self, R, Z, Phi, data, name="", units="",
                 properties=FieldProperty.NONE):
        super().__init__(properties=properties, name=name, units=units)
        self._R = np.asarray(R, dtype=float)
        self._Z = np.asarray(Z, dtype=float)
        self._Phi = np.asarray(Phi, dtype=float)
        self._data = np.asarray(data, dtype=float)
        expected = (len(self._R), len(self._Z), len(self._Phi), 3, 3)
        assert self._data.shape == expected, f"data shape {self._data.shape} != {expected}"
        self._interps = None

    @property
    def R(self): return self._R
    @property
    def Z(self): return self._Z
    @property
    def Phi(self): return self._Phi
    @property
    def data(self): return self._data

    def component(self, i: int, j: int) -> np.ndarray:
        """Return the (i,j) component grid, shape (nR, nZ, nPhi)."""
        return self._data[:, :, :, i, j]

    def trace(self) -> np.ndarray:
        """Return trace T_ii, shape (nR, nZ, nPhi)."""
        return sum(self._data[:,:,:,i,i] for i in range(3))

    def transpose(self) -> "TensorField3D_rank2":
        """Return T^T (swap last two axes)."""
        return TensorField3D_rank2(
            self._R, self._Z, self._Phi,
            np.transpose(self._data, (0,1,2,4,3)),
            name=f"({self.name})^T", units=self.units, properties=self._properties)

    def symmetrize(self) -> "TensorField3D_rank2":
        """Return (T + T^T)/2."""
        return TensorField3D_rank2(
            self._R, self._Z, self._Phi,
            0.5 * (self._data + np.transpose(self._data, (0,1,2,4,3))),
            name=f"sym({self.name})", units=self.units,
            properties=self._properties | FieldProperty.SYMMETRIC)

    def _build_interps(self):
        if self._interps is None:
            axes = (self._R, self._Z, self._Phi)
            kw = dict(method='linear', bounds_error=False, fill_value=np.nan)
            self._interps = [
                [RegularGridInterpolator(axes, self._data[:,:,:,i,j], **kw)
                 for j in range(3)]
                for i in range(3)
            ]

    def __call__(self, coords: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate at coords shape (..., 3), return (..., 3, 3)."""
        self._build_interps()
        coords = np.asarray(coords, dtype=float)
        shape = coords.shape[:-1]
        pts = coords.reshape(-1, 3)
        out = np.empty(pts.shape[0:1] + (3, 3), dtype=float)
        for i in range(3):
            for j in range(3):
                out[:,i,j] = self._interps[i][j](pts)
        return out.reshape(shape + (3, 3))
