"""Magnetic-axis data reconstructed directly from a symmetric VMEC WOUT."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class VmecMagneticAxisBuild:
    """Endpoint-excluded native-period magnetic axis and on-axis iota."""

    phi_vals: NDArray[np.float64]
    R_axis_m: NDArray[np.float64]
    Z_axis_m: NDArray[np.float64]
    iota_axis: float
    nfp: int
    toroidal_period_rad: float
    source_sha256: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        phi = np.array(self.phi_vals, dtype=np.float64, copy=True).reshape(-1)
        radial = np.array(self.R_axis_m, dtype=np.float64, copy=True).reshape(-1)
        vertical = np.array(self.Z_axis_m, dtype=np.float64, copy=True).reshape(-1)
        if (
            phi.size < 2
            or radial.shape != phi.shape
            or vertical.shape != phi.shape
            or not np.all(np.isfinite(phi))
            or not np.all(np.isfinite(radial))
            or not np.all(np.isfinite(vertical))
        ):
            raise ValueError("VMEC magnetic-axis arrays must be finite and aligned")
        if int(self.nfp) < 1 or not np.isfinite(self.iota_axis):
            raise ValueError("VMEC magnetic-axis nfp and iota must be valid")
        for array in (phi, radial, vertical):
            array.setflags(write=False)
        object.__setattr__(self, "phi_vals", phi)
        object.__setattr__(self, "R_axis_m", radial)
        object.__setattr__(self, "Z_axis_m", vertical)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def build_vmec_magnetic_axis_from_wout(
    path: str | Path,
    *,
    n_phi_per_period: int = 64,
    include_source_path: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> VmecMagneticAxisBuild:
    """Evaluate the VMEC axis Fourier series on one native field period."""

    from netCDF4 import Dataset

    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    with Dataset(source) as dataset:
        nfp = int(np.asarray(dataset.variables["nfp"][...]).item())
        lasym = bool(
            np.asarray(dataset.variables["lasym__logical__"][...]).item()
        ) if "lasym__logical__" in dataset.variables else False
        if lasym:
            raise NotImplementedError(
                "asymmetric VMEC magnetic-axis reconstruction is not implemented"
            )
        radial_coefficients = np.asarray(
            dataset.variables["raxis_cc"][...], dtype=np.float64
        ).reshape(-1)
        vertical_coefficients = np.asarray(
            dataset.variables["zaxis_cs"][...], dtype=np.float64
        ).reshape(-1)
        iota_axis = float(np.asarray(dataset.variables["iotaf"][0]).item())
    count = int(n_phi_per_period)
    if nfp < 1 or count < 2 or radial_coefficients.shape != vertical_coefficients.shape:
        raise ValueError("VMEC magnetic-axis modes and sampling are invalid")
    period = 2.0 * np.pi / nfp
    phi = np.linspace(0.0, period, count, endpoint=False)
    harmonic = np.arange(radial_coefficients.size, dtype=np.float64)
    phase = -nfp * phi[:, None] * harmonic[None, :]
    radial = np.sum(radial_coefficients[None, :] * np.cos(phase), axis=1)
    vertical = np.sum(vertical_coefficients[None, :] * np.sin(phase), axis=1)
    audit: dict[str, Any] = {
        "source_kind": "vmec_wout",
        "coordinate_role": "magnetic_axis",
        "toroidal_domain": "native_field_period",
        "endpoint_excluded": True,
        "stellarator_symmetric": True,
        "healing_used": False,
    }
    if include_source_path:
        audit["source_path"] = str(source)
    if metadata:
        audit.update(dict(metadata))
    return VmecMagneticAxisBuild(
        phi_vals=phi,
        R_axis_m=radial,
        Z_axis_m=vertical,
        iota_axis=iota_axis,
        nfp=nfp,
        toroidal_period_rad=period,
        source_sha256=_file_sha256(source),
        metadata=audit,
    )


__all__ = ["VmecMagneticAxisBuild", "build_vmec_magnetic_axis_from_wout"]
