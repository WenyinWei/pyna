"""Cylindrical coordinate field data structures.

Ported from Jynamics.jl (Juna.jl) CylindricalVectorField struct.
These are the Python equivalents for storing and manipulating 3D grid data
of magnetic fields, velocities, and other vector/scalar fields.

Supports:
- Numerical grid fields on (R, Z, phi) meshes
- Fast trilinear interpolation via scipy RegularGridInterpolator
- Vector calculus operations in cylindrical coordinates
- I/O: numpy .npz, NetCDF (optional), HDF5 (optional)
- IMAS compatibility: from_imas_equilibrium(), to_npz(), from_npz()
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass, field
from typing import Optional, Tuple
import warnings


@dataclass
class CylindricalScalarField:
    """Scalar field f(R, Z, phi) on a regular cylindrical grid.

    Attributes
    ----------
    R : ndarray, shape (nR,)
        Major radius grid [m].
    Z : ndarray, shape (nZ,)
        Vertical coordinate grid [m].
    Phi : ndarray, shape (nPhi,)
        Toroidal angle grid [rad]. Can be full 2pi or a field period.
    value : ndarray, shape (nR, nZ, nPhi)
        Field values on grid.
    field_periods : int
        Number of field periods. 1 = no toroidal symmetry.
        If > 1, Phi covers [0, 2pi/field_periods] and is periodically extended.
    name : str
        Field name for labeling.
    units : str
        Physical units string.
    """
    R: np.ndarray
    Z: np.ndarray
    Phi: np.ndarray
    value: np.ndarray
    field_periods: int = 1
    name: str = ""
    units: str = ""
    _interp: Optional[object] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        assert self.value.shape == (len(self.R), len(self.Z), len(self.Phi)), \
            f"value shape {self.value.shape} != ({len(self.R)}, {len(self.Z)}, {len(self.Phi)})"

    def _build_interp(self):
        if self._interp is None:
            self._interp = RegularGridInterpolator(
                (self.R, self.Z, self.Phi), self.value,
                method='linear', bounds_error=False, fill_value=np.nan
            )

    def __call__(self, R, Z, phi):
        """Interpolate at (R, Z, phi). Handles toroidal periodicity."""
        self._build_interp()
        if self.field_periods > 1:
            phi = phi % (2 * np.pi / self.field_periods)
        pts = np.column_stack([np.asarray(R).ravel(), np.asarray(Z).ravel(), np.asarray(phi).ravel()])
        return self._interp(pts).reshape(np.asarray(R).shape)

    def to_npz(self, path):
        np.savez_compressed(path, R=self.R, Z=self.Z, Phi=self.Phi, value=self.value,
                            field_periods=self.field_periods, name=self.name, units=self.units)

    @classmethod
    def from_npz(cls, path):
        d = np.load(path, allow_pickle=True)
        return cls(R=d['R'], Z=d['Z'], Phi=d['Phi'], value=d['value'],
                   field_periods=int(d.get('field_periods', 1)),
                   name=str(d.get('name', '')), units=str(d.get('units', '')))


@dataclass
class CylindricalVectorField:
    """Vector field (V_R, V_Z, V_phi) on a regular cylindrical grid.

    Equivalent to Julia's CylindricalVectorField{T} struct in Jynamics.jl.

    Attributes
    ----------
    R, Z, Phi : ndarray
        Grid coordinates.
    VR, VZ, VPhi : ndarray, shape (nR, nZ, nPhi)
        Vector components on grid.
    field_periods : int
        Toroidal field periods (1 = full torus).
    name : str
        Field name.

    Examples
    --------
    >>> import numpy as np
    >>> R = np.linspace(1.0, 2.0, 10)
    >>> Z = np.linspace(-0.5, 0.5, 10)
    >>> Phi = np.linspace(0, 2*np.pi, 8, endpoint=False)
    >>> VR = np.zeros((10, 10, 8))
    >>> VZ = np.zeros((10, 10, 8))
    >>> VPhi = np.ones((10, 10, 8))
    >>> B = CylindricalVectorField(R=R, Z=Z, Phi=Phi, VR=VR, VZ=VZ, VPhi=VPhi)
    """
    R: np.ndarray
    Z: np.ndarray
    Phi: np.ndarray
    VR: np.ndarray
    VZ: np.ndarray
    VPhi: np.ndarray
    field_periods: int = 1
    name: str = ""
    _interp_R: Optional[object] = field(default=None, repr=False, compare=False)
    _interp_Z: Optional[object] = field(default=None, repr=False, compare=False)
    _interp_Phi_comp: Optional[object] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        shp = (len(self.R), len(self.Z), len(self.Phi))
        for arr, nm in [(self.VR, 'VR'), (self.VZ, 'VZ'), (self.VPhi, 'VPhi')]:
            assert arr.shape == shp, f"{nm} shape {arr.shape} != {shp}"

    def _build_interps(self):
        if self._interp_R is None:
            grid = (self.R, self.Z, self.Phi)
            self._interp_R = RegularGridInterpolator(grid, self.VR, method='linear',
                                                     bounds_error=False, fill_value=np.nan)
            self._interp_Z = RegularGridInterpolator(grid, self.VZ, method='linear',
                                                     bounds_error=False, fill_value=np.nan)
            self._interp_Phi_comp = RegularGridInterpolator(grid, self.VPhi, method='linear',
                                                            bounds_error=False, fill_value=np.nan)

    def __call__(self, R, Z, phi):
        """Interpolate vector components at (R, Z, phi). Returns (VR, VZ, VPhi)."""
        self._build_interps()
        if self.field_periods > 1:
            phi = np.asarray(phi) % (2 * np.pi / self.field_periods)
        pts = np.column_stack([np.asarray(R).ravel(), np.asarray(Z).ravel(),
                               np.asarray(phi).ravel()])
        vr = self._interp_R(pts).reshape(np.asarray(R).shape)
        vz = self._interp_Z(pts).reshape(np.asarray(R).shape)
        vph = self._interp_Phi_comp(pts).reshape(np.asarray(R).shape)
        return vr, vz, vph

    def as_field_func(self):
        """Return a field_func(rzphi) -> [dR, dZ, dphi]/dl callable for use with FieldLineTracer."""
        def field_func(rzphi):
            R, Z, phi = rzphi[0], rzphi[1], rzphi[2]
            vr, vz, vphi = self(R, Z, phi)
            Bmag = np.sqrt(vr**2 + vz**2 + vphi**2) + 1e-30
            return np.array([vr / Bmag, vz / Bmag, vphi / (R * Bmag)])
        return field_func

    @classmethod
    def from_callable(cls, field_func, R, Z, Phi, field_periods=1, name="", n_workers=8):
        """Build grid field from a callable.

        Parameters
        ----------
        field_func : callable
            Called as field_func(np.array([R, Z, phi])) -> (VR, VZ, VPhi).
        R, Z, Phi : array-like
            Grid coordinates.
        n_workers : int
            Number of threads for parallel evaluation.
        """
        R = np.asarray(R)
        Z = np.asarray(Z)
        Phi = np.asarray(Phi)
        nR, nZ, nPhi = len(R), len(Z), len(Phi)
        VR = np.zeros((nR, nZ, nPhi))
        VZ = np.zeros((nR, nZ, nPhi))
        VPhi = np.zeros((nR, nZ, nPhi))

        from concurrent.futures import ThreadPoolExecutor

        def fill_phi_slice(iphi):
            phi = Phi[iphi]
            for iR in range(nR):
                for iZ in range(nZ):
                    result = field_func(np.array([R[iR], Z[iZ], phi]))
                    VR[iR, iZ, iphi] = result[0]
                    VZ[iR, iZ, iphi] = result[1]
                    VPhi[iR, iZ, iphi] = result[2]

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(fill_phi_slice, range(nPhi)))

        return cls(R=R, Z=Z, Phi=Phi, VR=VR, VZ=VZ, VPhi=VPhi,
                   field_periods=field_periods, name=name)

    def to_npz(self, path):
        np.savez_compressed(path, R=self.R, Z=self.Z, Phi=self.Phi,
                            VR=self.VR, VZ=self.VZ, VPhi=self.VPhi,
                            field_periods=self.field_periods, name=self.name)

    @classmethod
    def from_npz(cls, path):
        d = np.load(path, allow_pickle=True)
        return cls(R=d['R'], Z=d['Z'], Phi=d['Phi'],
                   VR=d['VR'], VZ=d['VZ'], VPhi=d['VPhi'],
                   field_periods=int(d.get('field_periods', 1)),
                   name=str(d.get('name', '')))

    @classmethod
    def from_netcdf(cls, path, field_periods=1, R_var='B_R', Z_var='B_Z', Phi_var='B_phi',
                    R_dim='R', Z_dim='Z', Phi_dim='phi'):
        """Load from NetCDF file (e.g., VMEC, EFIT, EMC3 output).

        Requires netCDF4 package. Variable names default to typical B-field convention.
        The data must NOT be copyrighted numerical equilibrium data.
        Only use with your own simulation output or open data.
        """
        try:
            import netCDF4 as nc
        except ImportError:
            raise ImportError("netCDF4 required: pip install netCDF4")

        with nc.Dataset(path) as ds:
            R = ds.variables[R_dim][:]
            Z = ds.variables[Z_dim][:]
            Phi = ds.variables[Phi_dim][:]
            VR = ds.variables[R_var][:]
            VZ = ds.variables[Z_var][:]
            VPhi = ds.variables[Phi_var][:]

        if VR.ndim == 3:
            if VR.shape == (len(R), len(Z), len(Phi)):
                pass  # already correct
            elif VR.shape == (len(Phi), len(Z), len(R)):
                VR = np.transpose(VR, (2, 1, 0))
                VZ = np.transpose(VZ, (2, 1, 0))
                VPhi = np.transpose(VPhi, (2, 1, 0))

        return cls(R=np.asarray(R), Z=np.asarray(Z), Phi=np.asarray(Phi),
                   VR=np.asarray(VR), VZ=np.asarray(VZ), VPhi=np.asarray(VPhi),
                   field_periods=field_periods)


# ── Backward-compat aliases pointing to pyna.fields ──────────────────────────
# New code should import from pyna.fields directly.
from pyna.fields.cylindrical import (
    CylindricalScalarField3D as _NewScalar,
    CylindricalVectorField3D as _NewVector,
)

# Make old names equal to new classes (seamless upgrade)
CylindricalScalarField = _NewScalar
CylindricalVectorField = _NewVector
