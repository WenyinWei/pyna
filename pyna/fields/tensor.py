"""Concrete tensor fields on cylindrical grids."""
from __future__ import annotations
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pyna.fields.base import TensorField3DRank2 as _TF3D_rank2_Base
from pyna.fields.base import TensorField
from pyna.fields.properties import FieldProperty


class Tensor2FieldCylind(_TF3D_rank2_Base):
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
    @property
    def is_axisymmetric(self) -> bool: return False

    def component(self, i: int, j: int) -> np.ndarray:
        """Return the (i,j) component grid, shape (nR, nZ, nPhi)."""
        return self._data[:, :, :, i, j]

    def trace(self):
        """Return trace T_ii as a scalar field."""
        from pyna.fields.cylindrical import ScalarFieldCylind
        value = sum(self._data[:,:,:,i,i] for i in range(3))
        return ScalarFieldCylind(self._R, self._Z, self._Phi, value,
                                 name=f"tr({self.name})", units=self.units)

    def transpose(self) -> "Tensor2FieldCylind":
        """Return T^T (swap last two axes)."""
        return Tensor2FieldCylind(
            self._R, self._Z, self._Phi,
            np.transpose(self._data, (0,1,2,4,3)),
            name=f"({self.name})^T", units=self.units, properties=self._properties)

    def symmetrize(self) -> "Tensor2FieldCylind":
        """Return (T + T^T)/2."""
        return Tensor2FieldCylind(
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


class Tensor2FieldCylindAxisym(Tensor2FieldCylind):
    """Axisymmetric rank-2 tensor field T_ij(R, Z)."""

    def __init__(self, R, Z, data_2d, name="", units="",
                 properties=FieldProperty.NONE):
        data = np.asarray(data_2d, dtype=float)
        if data.ndim == 4:
            data = data[:, :, np.newaxis, :, :]
        elif data.ndim == 5 and data.shape[2] == 1:
            pass
        else:
            raise ValueError("axisymmetric rank-2 tensor data must have shape (nR,nZ,3,3)")
        super().__init__(R, Z, np.array([0.0]), data,
                         name=name, units=units, properties=properties)

    @property
    def is_axisymmetric(self) -> bool: return True

    @property
    def data_2d(self) -> np.ndarray:
        return self._data[:, :, 0, :, :]

    def __call__(self, coords: np.ndarray, **kwargs) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        coords_axi = coords.copy()
        coords_axi[..., 2] = 0.0
        return super().__call__(coords_axi, **kwargs)

    def trace(self):
        from pyna.fields.cylindrical import ScalarFieldCylindAxisym
        value = sum(self._data[:, :, 0, i, i] for i in range(3))
        return ScalarFieldCylindAxisym(self._R, self._Z, value,
                                      name=f"tr({self.name})", units=self.units)

    def transpose(self) -> "Tensor2FieldCylindAxisym":
        return Tensor2FieldCylindAxisym(
            self._R, self._Z, np.transpose(self.data_2d, (0,1,3,2)),
            name=f"({self.name})^T", units=self.units, properties=self._properties)

    def symmetrize(self) -> "Tensor2FieldCylindAxisym":
        data_t = np.transpose(self.data_2d, (0,1,3,2))
        return Tensor2FieldCylindAxisym(
            self._R, self._Z, 0.5 * (self.data_2d + data_t),
            name=f"sym({self.name})", units=self.units,
            properties=self._properties | FieldProperty.SYMMETRIC)


class TensorField4DRank2(TensorField):
    """Rank-2 tensor field T_ij(x) on a 4-D domain (e.g. spacetime).

    Data shape: (n0, n1, n2, n3, 4, 4) -- spatial axes first, tensor indices last.

    Primary use case: spacetime metric g_uv, stress-energy tensor T_uv,
    electromagnetic field tensor F_uv, Ricci tensor R_uv.

    Index convention follows the coordinate system's coord_names
    (e.g. for Schwarzschild: 0=t, 1=r, 2=theta, 3=phi).

    Note: Riemann tensor evaluations near or inside the Schwarzschild radius
    (r <= 2GM/c^2) will encounter numerical singularities.
    """
    @property
    def domain_dim(self) -> int: return 4
    @property
    def range_rank(self) -> int: return 2

    def __init__(self, axes, data, name="", units="", properties=FieldProperty.NONE):
        """
        Parameters
        ----------
        axes : tuple of 4 ndarrays (n0, n1, n2, n3)
            Grid axes for each coordinate.
        data : ndarray, shape (n0, n1, n2, n3, 4, 4)
        """
        super().__init__(properties=properties, name=name, units=units)
        self._axes = tuple(np.asarray(a, dtype=float) for a in axes)
        self._data = np.asarray(data, dtype=float)
        expected = tuple(len(a) for a in self._axes) + (4, 4)
        assert self._data.shape == expected, f"data shape {self._data.shape} != {expected}"
        self._interps = None

    @property
    def axes(self): return self._axes
    @property
    def data(self): return self._data

    def component(self, i: int, j: int) -> np.ndarray:
        return self._data[..., i, j]

    def trace(self, metric=None) -> np.ndarray:
        """Trace. If metric g^{ij} provided (shape n0,n1,n2,n3,4,4), uses g^{ij}T_{ij}."""
        if metric is None:
            return sum(self._data[..., i, i] for i in range(4))
        return np.einsum('...ij,...ij->...', metric, self._data)

    def transpose(self) -> "TensorField4DRank2":
        return TensorField4DRank2(
            self._axes, np.swapaxes(self._data, -2, -1),
            name=f"({self.name})^T", units=self.units, properties=self._properties)

    def symmetrize(self) -> "TensorField4DRank2":
        return TensorField4DRank2(
            self._axes, 0.5*(self._data + np.swapaxes(self._data, -2, -1)),
            name=f"sym({self.name})", units=self.units,
            properties=self._properties | FieldProperty.SYMMETRIC)

    def _build_interps(self):
        from scipy.interpolate import RegularGridInterpolator
        if self._interps is None:
            kw = dict(method='linear', bounds_error=False, fill_value=np.nan)
            self._interps = [
                [RegularGridInterpolator(self._axes, self._data[..., i, j], **kw)
                 for j in range(4)]
                for i in range(4)
            ]

    def __call__(self, coords: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate at coords shape (..., 4), return (..., 4, 4)."""
        self._build_interps()
        coords = np.asarray(coords, dtype=float)
        shape = coords.shape[:-1]
        pts = coords.reshape(-1, 4)
        out = np.empty((pts.shape[0], 4, 4), dtype=float)
        for i in range(4):
            for j in range(4):
                out[:, i, j] = self._interps[i][j](pts)
        return out.reshape(shape + (4, 4))

    @classmethod
    def from_metric(cls, coords, axes) -> "TensorField4DRank2":
        """Build metric tensor field g_ij from a CoordinateSystem on given grid axes."""
        grids = np.meshgrid(*axes, indexing='ij')
        pts = np.stack([g.ravel() for g in grids], axis=1)
        g = coords.metric_tensor(pts)  # shape (N, 4, 4)
        shape = tuple(len(a) for a in axes)
        data = g.reshape(shape + (4, 4))
        return cls(axes, data, name=f"g_{coords.__class__.__name__}",
                   units="", properties=FieldProperty.SYMMETRIC)


# Compatibility name retained for older tensor callers; canonical code should
# use Tensor2FieldCylind / Tensor2FieldCylindAxisym.
TensorField3DRank2 = Tensor2FieldCylind
