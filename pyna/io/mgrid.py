"""VMEC mgrid readers and cylindrical current-density helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from pyna.fields.periodicity import ToroidalPeriodicity, normalize_nfp

MU0 = 4.0e-7 * np.pi


@dataclass(frozen=True)
class MGridField:
    """Magnetic field loaded from a VMEC ``mgrid`` file.

    Component arrays use the VMEC ordering ``(phi, Z, R)`` and store physical
    cylindrical components ``(B_R, B_phi, B_Z)``.
    """

    R: np.ndarray
    Z: np.ndarray
    phi: np.ndarray
    BR: np.ndarray
    BPhi: np.ndarray
    BZ: np.ndarray
    nfp: int
    period: float
    mode: str = ""
    source: Optional[str] = None
    coil_index: int = 1

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.BR.shape)

    @property
    def periodicity(self) -> ToroidalPeriodicity:
        return ToroidalPeriodicity(nfp=self.nfp, domain_period=self.period)

    @property
    def field_period(self) -> float:
        return self.periodicity.field_period


@dataclass(frozen=True)
class MGridCurrent:
    """Current density ``J = curl(B) / mu0`` on an mgrid lattice."""

    R: np.ndarray
    Z: np.ndarray
    phi: np.ndarray
    JR: np.ndarray
    JPhi: np.ndarray
    JZ: np.ndarray
    nfp: int
    period: float
    source: Optional[str] = None

    @property
    def Jabs(self) -> np.ndarray:
        return np.sqrt(self.JR * self.JR + self.JPhi * self.JPhi + self.JZ * self.JZ)


def _decode_chars(value: Any) -> str:
    arr = np.asarray(value)
    if arr.dtype.kind == "S":
        return b"".join(arr.ravel().tolist()).decode(errors="ignore").strip()
    if arr.dtype.kind == "U":
        return "".join(arr.ravel().tolist()).strip()
    return str(arr.item() if arr.shape == () else arr)


def _read_with_netcdf4(path: Path) -> dict[str, np.ndarray]:
    from netCDF4 import Dataset

    with Dataset(str(path), "r") as ds:
        return {name: np.asarray(var[:]).copy() if var.shape else np.asarray(var[()]).copy() for name, var in ds.variables.items()}


def _read_with_scipy(path: Path) -> dict[str, np.ndarray]:
    from scipy.io import netcdf_file

    with netcdf_file(str(path), "r", mmap=False) as ds:
        return {name: np.asarray(var.data).copy() for name, var in ds.variables.items()}


def read_mgrid_variables(path: Union[str, Path]) -> dict[str, np.ndarray]:
    """Read all variables from a VMEC mgrid NetCDF file.

    ``netCDF4`` is used when installed; SciPy's NetCDF3 reader is used as the
    no-extra-dependency fallback.
    """

    path = Path(path)
    try:
        return _read_with_netcdf4(path)
    except ImportError:
        return _read_with_scipy(path)
    except OSError:
        # Some VMEC mgrid files are NetCDF3 and cannot be opened by HDF5-only
        # stacks.  SciPy handles those files without adding a project dependency.
        return _read_with_scipy(path)


def load_vmec_mgrid(
    path: Union[str, Path],
    *,
    coil_index: int = 1,
    full_torus: bool = False,
) -> MGridField:
    """Load a VMEC-style mgrid magnetic field.

    Parameters
    ----------
    path:
        NetCDF mgrid file path.
    coil_index:
        One-based VMEC coil/current group index, mapping to variables such as
        ``br_001``, ``bp_001`` and ``bz_001``.
    full_torus:
        If true, repeat the stored field period ``nfp`` times and return a
        ``0..2*pi`` toroidal grid.  The default keeps the native field period.
    """

    path = Path(path)
    data = read_mgrid_variables(path)
    suffix = f"{int(coil_index):03d}"
    try:
        BR = np.asarray(data[f"br_{suffix}"], dtype=np.float64)
        BPhi = np.asarray(data[f"bp_{suffix}"], dtype=np.float64)
        BZ = np.asarray(data[f"bz_{suffix}"], dtype=np.float64)
    except KeyError as exc:
        raise KeyError(f"missing mgrid component for coil_index={coil_index}: {exc}") from exc

    nphi, nz, nr = BR.shape
    rmin = float(np.asarray(data["rmin"]).item())
    rmax = float(np.asarray(data["rmax"]).item())
    zmin = float(np.asarray(data["zmin"]).item())
    zmax = float(np.asarray(data["zmax"]).item())
    nfp = normalize_nfp(int(np.asarray(data.get("nfp", 1)).item()))
    native_period = ToroidalPeriodicity(nfp).field_period
    period = native_period

    if full_torus and nfp > 1:
        BR = np.concatenate([BR] * nfp, axis=0)
        BPhi = np.concatenate([BPhi] * nfp, axis=0)
        BZ = np.concatenate([BZ] * nfp, axis=0)
        nphi = BR.shape[0]
        period = 2.0 * np.pi

    R = np.linspace(rmin, rmax, nr)
    Z = np.linspace(zmin, zmax, nz)
    phi = np.arange(nphi, dtype=np.float64) * (period / nphi)
    mode = _decode_chars(data.get("mgrid_mode", ""))
    return MGridField(
        R=R,
        Z=Z,
        phi=phi,
        BR=BR,
        BPhi=BPhi,
        BZ=BZ,
        nfp=nfp,
        period=period,
        mode=mode,
        source=str(path),
        coil_index=int(coil_index),
    )


def mgrid_to_vector_field(field: MGridField, *, label: str | None = None):
    """Convert an ``MGridField`` into canonical ``VectorFieldCylind`` order.

    VMEC mgrid component arrays are stored as ``(phi, Z, R)``.  Pyna's regular
    cylindrical field object stores components as ``(R, Z, Phi)``.  Keeping this
    conversion explicit avoids silent dimension swaps in tracing and spectrum
    workflows.
    """

    from pyna.fields import VectorFieldCylind

    return VectorFieldCylind(
        R=np.asarray(field.R, dtype=np.float64),
        Z=np.asarray(field.Z, dtype=np.float64),
        Phi=np.asarray(field.phi, dtype=np.float64),
        BR=np.transpose(np.asarray(field.BR, dtype=np.float64), (2, 1, 0)),
        BZ=np.transpose(np.asarray(field.BZ, dtype=np.float64), (2, 1, 0)),
        BPhi=np.transpose(np.asarray(field.BPhi, dtype=np.float64), (2, 1, 0)),
        periodicity=field.periodicity,
        label=field.mode if label is None else label,
    )


def compute_current_density_cylindrical(field: MGridField) -> MGridCurrent:
    """Compute ``J = curl(B) / mu0`` on a cylindrical mgrid lattice."""

    R = np.asarray(field.R, dtype=np.float64)
    Z = np.asarray(field.Z, dtype=np.float64)
    BR = np.asarray(field.BR, dtype=np.float64)
    BPhi = np.asarray(field.BPhi, dtype=np.float64)
    BZ = np.asarray(field.BZ, dtype=np.float64)
    dphi = float(field.period) / BR.shape[0]
    dR = float(R[1] - R[0])
    dZ = float(Z[1] - Z[0])
    Rb = R[None, None, :]

    dBZ_dphi = (np.roll(BZ, -1, axis=0) - np.roll(BZ, 1, axis=0)) / (2.0 * dphi)
    dBR_dphi = (np.roll(BR, -1, axis=0) - np.roll(BR, 1, axis=0)) / (2.0 * dphi)
    dBPhi_dZ = np.gradient(BPhi, dZ, axis=1, edge_order=2)
    dBR_dZ = np.gradient(BR, dZ, axis=1, edge_order=2)
    dBZ_dR = np.gradient(BZ, dR, axis=2, edge_order=2)
    dRBPhi_dR = np.gradient(BPhi * Rb, dR, axis=2, edge_order=2)

    JR = (dBZ_dphi / Rb - dBPhi_dZ) / MU0
    JPhi = (dBR_dZ - dBZ_dR) / MU0
    JZ = (dRBPhi_dR - dBR_dphi) / (Rb * MU0)
    return MGridCurrent(
        R=R,
        Z=Z,
        phi=np.asarray(field.phi, dtype=np.float64),
        JR=JR,
        JPhi=JPhi,
        JZ=JZ,
        nfp=int(field.nfp),
        period=float(field.period),
        source=field.source,
    )


def toroidal_index(phi: float, period: float, nphi: int) -> int:
    """Return the nearest periodic toroidal grid index for ``phi``."""

    dphi = float(period) / int(nphi)
    return int(round(float(np.mod(phi, period)) / dphi)) % int(nphi)


def mgrid_toroidal_index(field: Union[MGridField, MGridCurrent], phi: float) -> int:
    """Return the nearest periodic toroidal index for an mgrid object."""

    nphi = field.BR.shape[0] if isinstance(field, MGridField) else field.JR.shape[0]
    return toroidal_index(phi, field.period, nphi)


def sample_plane_bilinear(
    values_zr: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    query_R: np.ndarray,
    query_Z: np.ndarray,
) -> np.ndarray:
    """Bilinearly sample a ``(Z, R)`` plane at arbitrary ``(R, Z)`` points."""

    values = np.asarray(values_zr, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    query_R = np.asarray(query_R, dtype=np.float64)
    query_Z = np.asarray(query_Z, dtype=np.float64)
    fr = (query_R - R[0]) / (R[1] - R[0])
    fz = (query_Z - Z[0]) / (Z[1] - Z[0])
    i0 = np.floor(fr).astype(np.int64)
    j0 = np.floor(fz).astype(np.int64)
    valid = (i0 >= 0) & (i0 < R.size - 1) & (j0 >= 0) & (j0 < Z.size - 1)
    i = np.clip(i0, 0, R.size - 2)
    j = np.clip(j0, 0, Z.size - 2)
    wr = np.clip(fr - i0, 0.0, 1.0)
    wz = np.clip(fz - j0, 0.0, 1.0)
    out = (
        (1.0 - wr) * (1.0 - wz) * values[j, i]
        + wr * (1.0 - wz) * values[j, i + 1]
        + (1.0 - wr) * wz * values[j + 1, i]
        + wr * wz * values[j + 1, i + 1]
    )
    return np.where(valid, out, np.nan)


__all__ = [
    "MU0",
    "MGridField",
    "MGridCurrent",
    "read_mgrid_variables",
    "load_vmec_mgrid",
    "mgrid_to_vector_field",
    "compute_current_density_cylindrical",
    "toroidal_index",
    "mgrid_toroidal_index",
    "sample_plane_bilinear",
]
