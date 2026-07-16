"""Three-dimensional PEST coordinates reconstructed from raw VMEC WOUT data.

The public builder in this module owns the generic VMEC Fourier evaluation and
VMEC-angle to PEST-angle inversion.  It returns one endpoint-excluded native
field period and never constructs a full-torus duplicate or modifies the
magnetic field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from pyna.toroidal.diagnostics.mgrid import SmoothPestCoordinates


_TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class VmecPestCoordinateBuild:
    """Raw-WOUT PEST coordinates plus VMEC surface quantities.

    ``coordinates.rho_vals`` is ``sqrt(s)`` while
    ``normalized_toroidal_flux`` retains VMEC's normalized toroidal flux
    coordinate ``s``.  ``contravariant_toroidal_field`` is the VMEC
    ``B^phi`` denominator sampled on the identical PEST grid.
    """

    coordinates: SmoothPestCoordinates
    normalized_toroidal_flux: NDArray[np.float64]
    iota_profile: NDArray[np.float64]
    contravariant_toroidal_field: NDArray[np.float64]
    source_sha256: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coordinates = self.coordinates
        if not isinstance(coordinates, SmoothPestCoordinates):
            raise TypeError("coordinates must be SmoothPestCoordinates")
        radial_s = _immutable_f64(
            self.normalized_toroidal_flux, "normalized_toroidal_flux"
        )
        iota = _immutable_f64(self.iota_profile, "iota_profile")
        bphi = np.array(
            self.contravariant_toroidal_field, dtype=np.float64, copy=True
        )
        expected_shape = np.asarray(coordinates.R_surf).shape
        if radial_s.shape != np.asarray(coordinates.rho_vals).shape:
            raise ValueError("normalized flux must contain one value per PEST surface")
        if iota.shape != radial_s.shape:
            raise ValueError("iota must contain one value per PEST surface")
        if bphi.shape != expected_shape or not np.all(np.isfinite(bphi)):
            raise ValueError("contravariant toroidal field must match the PEST mesh")
        if not np.allclose(
            np.asarray(coordinates.rho_vals) ** 2,
            radial_s,
            rtol=0.0,
            atol=8.0 * np.finfo(np.float64).eps,
        ):
            raise ValueError("PEST rho must equal sqrt(normalized toroidal flux)")
        source = _required_sha256(self.source_sha256)
        bphi.setflags(write=False)
        object.__setattr__(self, "normalized_toroidal_flux", radial_s)
        object.__setattr__(self, "iota_profile", iota)
        object.__setattr__(self, "contravariant_toroidal_field", bphi)
        object.__setattr__(self, "source_sha256", source)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def _immutable_f64(values: Any, label: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be non-empty and finite")
    array.setflags(write=False)
    return array


def _required_sha256(value: Any) -> str:
    text = str(value).strip()
    if not text.startswith("sha256:") or len(text) != 71:
        raise ValueError("source_sha256 must be a full sha256 identity")
    try:
        int(text[7:], 16)
    except ValueError as exc:
        raise ValueError("source_sha256 must be a full sha256 identity") from exc
    return text


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _interpolate_mode_rows(
    values: Any,
    source_s: Any,
    target_s: Any,
) -> NDArray[np.float64]:
    data = np.asarray(values, dtype=np.float64)
    source = np.asarray(source_s, dtype=np.float64).reshape(-1)
    target = np.asarray(target_s, dtype=np.float64).reshape(-1)
    if data.ndim != 2 or data.shape[0] != source.size or source.size < 2:
        raise ValueError("VMEC mode rows must have shape (len(source_s), n_modes)")
    order = np.argsort(source)
    source = source[order]
    data = data[order]
    clipped = np.clip(target, source[0], source[-1])
    upper = np.clip(
        np.searchsorted(source, clipped, side="right"), 1, source.size - 1
    )
    lower = upper - 1
    span = source[upper] - source[lower]
    alpha = np.divide(
        clipped - source[lower],
        span,
        out=np.zeros_like(clipped),
        where=span != 0.0,
    )
    return data[lower] * (1.0 - alpha[:, None]) + data[upper] * alpha[:, None]


def _evaluate_modes(
    coefficients: Any,
    m: Any,
    n: Any,
    theta_vmec: NDArray[np.float64],
    phi: NDArray[np.float64],
    *,
    sine: bool,
) -> NDArray[np.float64]:
    phase = (
        theta_vmec[:, :, None]
        * np.asarray(m, dtype=np.float64)[None, None, :]
        - phi[:, None, None]
        * np.asarray(n, dtype=np.float64)[None, None, :]
    )
    trig = np.sin(phase) if sine else np.cos(phase)
    return np.sum(
        np.asarray(coefficients, dtype=np.float64)[None, None, :] * trig,
        axis=-1,
    )


def _invert_pest_angle(
    theta_pest: NDArray[np.float64],
    phi: NDArray[np.float64],
    lmns: NDArray[np.float64],
    xm: NDArray[np.float64],
    xn: NDArray[np.float64],
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[NDArray[np.float64], float, int]:
    target = np.broadcast_to(theta_pest[None, :], (phi.size, theta_pest.size))
    theta_vmec = target.copy()
    residual_max = np.inf
    for iteration in range(max(1, int(max_iterations))):
        phase = (
            theta_vmec[:, :, None] * xm[None, None, :]
            - phi[:, None, None] * xn[None, None, :]
        )
        lam = np.sum(lmns[None, None, :] * np.sin(phase), axis=-1)
        derivative = 1.0 + np.sum(
            (lmns * xm)[None, None, :] * np.cos(phase), axis=-1
        )
        derivative = np.where(
            np.abs(derivative) < 1.0e-10,
            np.where(derivative < 0.0, -1.0e-10, 1.0e-10),
            derivative,
        )
        residual = theta_vmec + lam - target
        theta_vmec = theta_vmec - residual / derivative
        residual_max = float(np.max(np.abs(residual)))
        if residual_max <= float(tolerance):
            return np.mod(theta_vmec, _TWOPI), residual_max, iteration + 1
    raise RuntimeError(
        "VMEC-to-PEST angle inversion did not converge: "
        f"residual={residual_max:.3e}, iterations={int(max_iterations)}"
    )


def build_vmec_pest_coordinates_from_wout(
    path: str | Path,
    *,
    normalized_toroidal_flux: Sequence[float] = (0.81, 0.9025, 1.0),
    n_phi_per_period: int = 64,
    n_theta: int = 160,
    pest_tolerance: float = 1.0e-11,
    pest_max_iterations: int = 16,
    include_source_path: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> VmecPestCoordinateBuild:
    """Build a 3-D PEST mesh directly from raw VMEC Fourier coefficients.

    The toroidal mesh covers exactly one endpoint-excluded native field
    period.  VMEC's stored ``xn`` already includes ``nfp`` and is used without
    additional scaling.  Stellarator-asymmetric WOUT files currently fail
    closed because their sine/cosine partner arrays require a separate audit.
    """

    from netCDF4 import Dataset

    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    source_sha256 = _file_sha256(source)
    with Dataset(source) as dataset:
        periods = int(np.asarray(dataset.variables["nfp"][...]).item())
        ns = int(np.asarray(dataset.variables["ns"][...]).item())
        if "lasym__logical__" in dataset.variables:
            lasym = bool(
                np.asarray(dataset.variables["lasym__logical__"][...]).item()
            )
        else:
            lasym = False
        if lasym:
            raise NotImplementedError(
                "asymmetric VMEC surface reconstruction is not implemented"
            )
        xm = np.asarray(dataset.variables["xm"][...], dtype=np.float64)
        xn = np.asarray(dataset.variables["xn"][...], dtype=np.float64)
        xm_nyq = np.asarray(dataset.variables["xm_nyq"][...], dtype=np.float64)
        xn_nyq = np.asarray(dataset.variables["xn_nyq"][...], dtype=np.float64)
        rmnc = np.asarray(dataset.variables["rmnc"][...], dtype=np.float64)
        zmns = np.asarray(dataset.variables["zmns"][...], dtype=np.float64)
        lmns = np.asarray(dataset.variables["lmns"][...], dtype=np.float64)
        bsupvmnc = np.asarray(
            dataset.variables["bsupvmnc"][...], dtype=np.float64
        )
        iotaf = np.asarray(dataset.variables["iotaf"][...], dtype=np.float64)

    radial_s = np.asarray(
        tuple(normalized_toroidal_flux), dtype=np.float64
    ).reshape(-1)
    phi_count = int(n_phi_per_period)
    theta_count = int(n_theta)
    if (
        radial_s.size < 2
        or np.any(radial_s <= 0.0)
        or np.any(radial_s > 1.0)
        or np.any(np.diff(radial_s) <= 0.0)
    ):
        raise ValueError(
            "normalized_toroidal_flux must be strictly increasing in (0, 1]"
        )
    if periods <= 0 or phi_count < 2 or theta_count < 3 or ns < 2:
        raise ValueError("VMEC native-period mesh dimensions and nfp must be positive")
    if not np.isfinite(pest_tolerance) or float(pest_tolerance) <= 0.0:
        raise ValueError("pest_tolerance must be finite and positive")
    if int(pest_max_iterations) < 1:
        raise ValueError("pest_max_iterations must be positive")

    period = _TWOPI / float(periods)
    phi = np.linspace(0.0, period, phi_count, endpoint=False)
    theta = np.linspace(0.0, _TWOPI, theta_count, endpoint=False)
    full_s = np.linspace(0.0, 1.0, ns)
    half_s = (np.arange(1, ns, dtype=np.float64) - 0.5) / float(ns - 1)
    rm_rows = _interpolate_mode_rows(rmnc, full_s, radial_s)
    zm_rows = _interpolate_mode_rows(zmns, full_s, radial_s)
    lm_rows = _interpolate_mode_rows(lmns[1:], half_s, radial_s)
    bphi_rows = _interpolate_mode_rows(bsupvmnc[1:], half_s, radial_s)
    iota = _interpolate_mode_rows(iotaf[:, None], full_s, radial_s)[:, 0]

    R = np.empty((phi_count, radial_s.size, theta_count), dtype=np.float64)
    Z = np.empty_like(R)
    bphi = np.empty_like(R)
    inversion_residual = 0.0
    inversion_iterations = 0
    for radial_index in range(radial_s.size):
        theta_vmec, residual, iterations = _invert_pest_angle(
            theta,
            phi,
            lm_rows[radial_index],
            xm,
            xn,
            tolerance=float(pest_tolerance),
            max_iterations=int(pest_max_iterations),
        )
        inversion_residual = max(inversion_residual, residual)
        inversion_iterations = max(inversion_iterations, iterations)
        R[:, radial_index, :] = _evaluate_modes(
            rm_rows[radial_index], xm, xn, theta_vmec, phi, sine=False
        )
        Z[:, radial_index, :] = _evaluate_modes(
            zm_rows[radial_index], xm, xn, theta_vmec, phi, sine=True
        )
        bphi[:, radial_index, :] = _evaluate_modes(
            bphi_rows[radial_index],
            xm_nyq,
            xn_nyq,
            theta_vmec,
            phi,
            sine=False,
        )

    build_metadata: dict[str, Any] = {
        "source_kind": "vmec_wout",
        "source_id": source_sha256,
        "coordinate_system": "PEST",
        "vmec_radial_coordinate": "s_normalized_toroidal_flux",
        "pest_radial_coordinate": "rho=sqrt(s_normalized_toroidal_flux)",
        "theta_relation": "theta_PEST=theta_VMEC+lambda",
        "vmec_fourier_phase": "m*theta_VMEC-xn*phi",
        "xn_includes_field_periods": True,
        "toroidal_domain": "native_field_period",
        "toroidal_domain_period_rad": float(period),
        "native_field_period_sampling": True,
        "n_phi_domain": phi_count,
        "pest_inversion_max_residual": float(inversion_residual),
        "pest_inversion_max_iterations": int(inversion_iterations),
        "healing_used": False,
        "surface_displaced": False,
        "full_torus_intermediate_constructed": False,
    }
    if include_source_path:
        build_metadata["source_path"] = str(source)
    if metadata:
        build_metadata.update(dict(metadata))
    coordinates = SmoothPestCoordinates(
        R_surf=R,
        Z_surf=Z,
        rho_vals=np.sqrt(radial_s),
        theta_vals=theta,
        phi_vals=phi,
        source=source_sha256,
        nfp=periods,
        toroidal_period=period,
    )
    if not coordinates.stores_one_field_period:
        raise RuntimeError("VMEC PEST builder did not preserve one field period")
    return VmecPestCoordinateBuild(
        coordinates=coordinates,
        normalized_toroidal_flux=radial_s,
        iota_profile=iota,
        contravariant_toroidal_field=bphi,
        source_sha256=source_sha256,
        metadata=build_metadata,
    )


__all__ = [
    "VmecPestCoordinateBuild",
    "build_vmec_pest_coordinates_from_wout",
]
