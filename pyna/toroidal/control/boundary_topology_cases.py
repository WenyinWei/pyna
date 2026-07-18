"""Case assembly for active stellarator boundary-topology control.

The module connects healed/VMEC surface coordinates, finite-loop dipole
actuators, Nardon spectra, island/chaos diagnostics, heat forward models, and
the high-level bounded optimizer without embedding device-specific private
paths or names in the public package.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pyna.fields.periodicity import ToroidalPeriodicity, normalize_nfp
from pyna.toroidal.coils.boundary_local import BoundaryDipoleActuatorArray
from pyna.toroidal.control.boundary_field_basis import (
    BoundaryFieldActuatorArray,
    BoundaryFieldSuperposition,
    ScaledBoundaryFieldCandidate,
)
from pyna.toroidal.control.boundary_perturbation_candidates import (
    PerturbationCandidateSpectrumResponse,
    sample_perturbation_candidate_on_surfaces,
)
from pyna.toroidal.control.boundary_plasma_response import (
    BoundaryPlasmaResponseInput,
    BoundaryPlasmaResponseSnapshot,
    CorePreservationSnapshot,
)
from pyna.toroidal.control.boundary_topology_design import (
    BoundaryResponseObservables,
    boundary_response_observables,
    chaotic_layer_region_observables,
    resonant_chain_observables,
    stack_boundary_response_observables,
    wall_heat_region_observables,
)
from pyna.toroidal.control.heat_distribution import wall_heat_flux_observables
from pyna.toroidal.control.heat_contracts import (
    BoundaryTopologyHeatForwardModel,
    BoundaryTopologyHeatState,
    CallableBoundaryTopologyHeatForwardModel,
)
from pyna.toroidal.control.reduced_heat import ReducedSpectralHeatModel
from pyna.toroidal.perturbation_spectrum import (
    ChirikovOverlapBand,
    ChaoticLayerInterval,
    RadialPerturbationFourierSpectrum,
    ResonantIslandChain,
    analyze_resonant_island_chains_multi_n,
    chaotic_layer_intervals,
    chirikov_overlap_bands,
    cylindrical_field_grid_signature,
    nardon_radial_perturbation,
    radial_perturbation_Fourier_spectrum,
    require_matching_field_signature,
    surface_coordinate_signature,
)


TWOPI = 2.0 * np.pi


def _mapping_signature(metadata: Mapping[str, object] | None, *keys: str):
    values = dict(metadata or {})
    for key in keys:
        value = values.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return None


def _grid_field_signature(field):
    """Return declared or array-derived provenance for a cylindrical grid field."""

    signature = _mapping_signature(
        getattr(field, "metadata", None),
        "field_signature",
        "delta_field_signature",
        "background_field_signature",
    )
    if signature is not None:
        return signature
    factory = getattr(field, "field_signature", None)
    if callable(factory):
        value = factory()
        return dict(value) if isinstance(value, Mapping) else None
    names = ("R", "Z", "Phi", "BR", "BPhi", "BZ", "nfp")
    if not all(hasattr(field, name) for name in names):
        return None
    phi = getattr(field, "Phi")
    if phi is None:
        return None
    try:
        return cylindrical_field_grid_signature(
            getattr(field, "R"),
            getattr(field, "Z"),
            phi,
            getattr(field, "BR"),
            getattr(field, "BPhi"),
            getattr(field, "BZ"),
            nfp=int(getattr(field, "nfp")),
        )
    except (TypeError, ValueError):
        return None


def _case_spectrum_provenance(case: "BoundaryTopologyCase") -> dict[str, object]:
    metadata = dict(case.metadata or {})
    surface_signature = _mapping_signature(metadata, "surface_signature")
    background_signature = _mapping_signature(metadata, "background_field_signature")
    if background_signature is None and surface_signature is not None:
        bound_background = surface_signature.get(
            "background_field_signature",
            surface_signature.get("field_signature"),
        )
        if isinstance(bound_background, Mapping):
            background_signature = dict(bound_background)
    if background_signature is None and case.background_field is not None:
        background_signature = _grid_field_signature(case.background_field)
    if surface_signature is None and background_signature is not None:
        surface_signature = surface_coordinate_signature(
            case.R_surf,
            case.Z_surf,
            case.phi_vals,
            case.theta_vals,
            case.radial_labels,
            background_field_signature=background_signature,
            coordinate_system=case.coordinate_system,
            radial_coordinate=case.radial_coordinate,
        )
    elif surface_signature is not None and background_signature is not None:
        bound_background = surface_signature.get(
            "background_field_signature",
            surface_signature.get("field_signature"),
        )
        if bound_background is not None:
            require_matching_field_signature(
                bound_background,
                background_signature,
                context="boundary-topology case surface background field",
            )
    provenance: dict[str, object] = {}
    if surface_signature is not None:
        provenance["surface_signature"] = surface_signature
    if background_signature is not None:
        provenance["background_field_signature"] = background_signature
    return provenance


def _validate_grid_actuator_nfp(
    case: "BoundaryTopologyCase",
    actuators: BoundaryDipoleActuatorArray | BoundaryFieldActuatorArray,
) -> None:
    for actuator in actuators.actuators:
        grid_field = getattr(actuator, "grid_field", None)
        if grid_field is None:
            continue
        periods = getattr(grid_field, "nfp", None)
        if periods is None:
            continue
        if int(periods) != int(case.nfp):
            raise ValueError(
                f"grid-backed actuator {actuator.label!r} nfp={int(periods)} "
                f"does not match case.nfp={int(case.nfp)}"
            )


def _candidate_response_for_case(
    candidate,
    case: "BoundaryTopologyCase",
    *,
    m_max: int | None,
    n_max: int | None,
    metadata: Mapping[str, object],
) -> PerturbationCandidateSpectrumResponse:
    delta_BR, delta_BZ, delta_BPhi = sample_perturbation_candidate_on_surfaces(
        candidate,
        case.R_surf,
        case.Z_surf,
        case.phi_vals,
    )
    tilde = nardon_radial_perturbation(
        case.R_surf,
        case.Z_surf,
        case.phi_vals,
        case.theta_vals,
        delta_BR,
        delta_BZ,
        delta_BPhi,
        case.radial_labels,
        denominator_B3=case.denominator_B3,
    )
    response_metadata = {
        "candidate_kind": type(candidate).__name__,
        **dict(metadata),
    }
    spectrum = radial_perturbation_Fourier_spectrum(
        tilde,
        case.theta_vals,
        case.phi_vals,
        radial_labels=case.radial_labels,
        layout="phi-radial-theta",
        nfp=_surface_fourier_nfp(case),
        m_max=m_max,
        n_max=n_max,
        metadata=response_metadata,
    )
    return PerturbationCandidateSpectrumResponse(
        delta_BR=delta_BR,
        delta_BZ=delta_BZ,
        delta_BPhi=delta_BPhi,
        tilde_b1=tilde,
        spectrum=spectrum,
        metadata=response_metadata,
    )


def _surface_fourier_nfp(case: "BoundaryTopologyCase") -> int:
    """Resolve whether surfaces cover the full torus or one native field period."""

    phi = np.asarray(case.phi_vals, dtype=float).ravel()
    candidates = tuple(dict.fromkeys((int(case.nfp), 1)))
    for periods in candidates:
        period = ToroidalPeriodicity(periods).field_period
        scaled = np.unwrap(phi * (TWOPI / period)) * (period / TWOPI)
        span = float(scaled[-1] - scaled[0])
        has_endpoint = np.isclose(span, period, rtol=1.0e-10, atol=1.0e-12)
        count = phi.size - 1 if has_endpoint else phi.size
        if count < 1:
            continue
        axis = scaled[:-1] if has_endpoint else scaled
        if np.allclose(
            np.diff(axis),
            period / float(count),
            rtol=1.0e-9,
            atol=1.0e-12 * max(1.0, period),
        ):
            return periods
    raise ValueError(
        "case phi_vals must uniformly sample either the full torus or one native field period"
    )


@dataclass(frozen=True)
class BoundaryTopologyCase:
    """Background surfaces and profiles for one boundary-control case.

    Surface arrays use ``(phi, radial, theta)`` order.  ``theta_vals`` must be
    the angle named by ``coordinate_system``; production spectrum work should
    normally use a straight-field-line PEST-like angle.  ``denominator_B3`` is
    the background contravariant toroidal field ``B0 dot grad(phi)``.
    """

    name: str
    R_surf: np.ndarray
    Z_surf: np.ndarray
    phi_vals: np.ndarray
    theta_vals: np.ndarray
    radial_labels: np.ndarray
    iota_profile: np.ndarray
    denominator_B3: np.ndarray
    nfp: int = 1
    coordinate_system: str = "PEST"
    radial_coordinate: str = "s"
    q_profile: np.ndarray | None = None
    q_iota_sign: int = 1
    background_field: Any = None
    equilibrium: Any = None
    core_reference: CorePreservationSnapshot | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        R = np.asarray(self.R_surf, dtype=float)
        Z = np.asarray(self.Z_surf, dtype=float)
        phi = np.asarray(self.phi_vals, dtype=float).ravel()
        theta = np.asarray(self.theta_vals, dtype=float).ravel()
        radial = np.asarray(self.radial_labels, dtype=float).ravel()
        iota = np.asarray(self.iota_profile, dtype=float).ravel()
        B3 = np.asarray(self.denominator_B3, dtype=float)
        if R.shape != Z.shape or R.ndim != 3:
            raise ValueError("R_surf and Z_surf must have shape (n_phi, n_radial, n_theta)")
        if R.shape != (phi.size, radial.size, theta.size):
            raise ValueError("surface axes must match phi_vals, radial_labels, and theta_vals")
        if B3.shape != R.shape:
            raise ValueError("denominator_B3 must match the surface shape")
        if iota.size != radial.size:
            raise ValueError("iota_profile must match radial_labels")
        if not np.all(np.isfinite(R)) or not np.all(np.isfinite(Z)):
            raise ValueError("surface geometry must be finite")
        if not np.all(np.isfinite(B3)) or np.any(np.abs(B3) <= 1.0e-300):
            raise ValueError("denominator_B3 must be finite and nonzero")
        if not np.all(np.isfinite(radial)) or np.any(np.diff(radial) <= 0.0):
            raise ValueError("radial_labels must be finite and strictly increasing")
        if not np.all(np.isfinite(iota)) or np.any(np.abs(iota) <= 1.0e-300):
            raise ValueError("iota_profile must be finite and nonzero")
        relation_sign = int(self.q_iota_sign)
        if relation_sign not in (-1, 1):
            raise ValueError("q_iota_sign must be +1 or -1")
        q = relation_sign / iota if self.q_profile is None else np.asarray(self.q_profile, dtype=float).ravel()
        if q.size != radial.size or not np.all(np.isfinite(q)):
            raise ValueError("q_profile must be finite and match radial_labels")
        relation = q * iota
        if not np.allclose(relation, relation_sign, rtol=2.0e-6, atol=2.0e-8):
            raise ValueError("q_profile and iota_profile must satisfy q=q_iota_sign/iota")
        periods = normalize_nfp(self.nfp)
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "R_surf", R)
        object.__setattr__(self, "Z_surf", Z)
        object.__setattr__(self, "phi_vals", phi)
        object.__setattr__(self, "theta_vals", theta)
        object.__setattr__(self, "radial_labels", radial)
        object.__setattr__(self, "iota_profile", iota)
        object.__setattr__(self, "q_profile", q)
        object.__setattr__(self, "q_iota_sign", relation_sign)
        object.__setattr__(self, "denominator_B3", B3)
        object.__setattr__(self, "nfp", periods)
        object.__setattr__(self, "coordinate_system", str(self.coordinate_system))
        object.__setattr__(self, "radial_coordinate", str(self.radial_coordinate))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def boundary_R(self) -> np.ndarray:
        """Return the outermost control surface as ``(phi, theta)``."""

        return self.R_surf[:, -1, :]

    @property
    def periodicity(self) -> ToroidalPeriodicity:
        return ToroidalPeriodicity(nfp=self.nfp)

    @property
    def field_period(self) -> float:
        return self.periodicity.field_period

    @property
    def boundary_Z(self) -> np.ndarray:
        """Return the outermost control surface as ``(phi, theta)``."""

        return self.Z_surf[:, -1, :]

    def to_smooth_pest_coordinates(self):
        """Return the case geometry with explicit physical and mesh periodicity.

        ``nfp`` records the stellarator symmetry.  ``toroidal_period`` records
        whether this particular mesh stores one native field period or the
        full torus; keeping these separate prevents a full-torus display mesh
        from being mistaken for a native-period solver mesh.
        """

        from pyna.toroidal.diagnostics.mgrid import SmoothPestCoordinates

        domain_period = dict(self.metadata or {}).get("toroidal_domain_period_rad")
        if domain_period is None:
            sampled_periods = _surface_fourier_nfp(self)
            domain_period = ToroidalPeriodicity(sampled_periods).field_period
        return SmoothPestCoordinates(
            R_surf=self.R_surf,
            Z_surf=self.Z_surf,
            rho_vals=self.radial_labels,
            theta_vals=self.theta_vals,
            phi_vals=self.phi_vals,
            source=f"BoundaryTopologyCase:{self.name}",
            nfp=self.nfp,
            toroidal_period=float(domain_period),
        )


def boundary_topology_case_from_arrays(
    *,
    name: str,
    R_surf,
    Z_surf,
    phi_vals,
    theta_vals,
    radial_labels,
    iota_profile,
    denominator_B3=None,
    denominator_B_phi=None,
    nfp: int = 1,
    coordinate_system: str = "PEST",
    radial_coordinate: str = "s",
    q_profile=None,
    q_iota_sign: int = 1,
    background_field=None,
    equilibrium=None,
    core_reference=None,
    metadata: Mapping[str, object] | None = None,
) -> BoundaryTopologyCase:
    """Build a validated case from healed surfaces and background profiles."""

    R = np.asarray(R_surf, dtype=float)
    if denominator_B3 is None:
        if denominator_B_phi is None:
            raise ValueError("denominator_B3 or denominator_B_phi is required")
        denominator_B3 = np.asarray(denominator_B_phi, dtype=float) / np.maximum(R, 1.0e-300)
    core = None
    if core_reference is not None:
        core = core_reference if isinstance(core_reference, CorePreservationSnapshot) else CorePreservationSnapshot(**core_reference)
    return BoundaryTopologyCase(
        name=name,
        R_surf=R,
        Z_surf=Z_surf,
        phi_vals=phi_vals,
        theta_vals=theta_vals,
        radial_labels=radial_labels,
        iota_profile=iota_profile,
        q_profile=q_profile,
        q_iota_sign=q_iota_sign,
        denominator_B3=denominator_B3,
        nfp=nfp,
        coordinate_system=coordinate_system,
        radial_coordinate=radial_coordinate,
        background_field=background_field,
        equilibrium=equilibrium,
        core_reference=core,
        metadata={} if metadata is None else dict(metadata),
    )


def load_boundary_topology_case_npz(
    path: str | Path,
    *,
    name: str = "private stellarator",
    include_source_path: bool = False,
    metadata: Mapping[str, object] | None = None,
) -> BoundaryTopologyCase:
    """Load a private or public case bundle without recording its path by default."""

    source = Path(path).expanduser()
    with np.load(source, allow_pickle=False) as data:
        required = ("R_surf", "Z_surf", "phi_vals", "theta_vals", "radial_labels", "iota_profile")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"case bundle is missing required arrays: {missing}")
        if "denominator_B3" in data:
            B3 = np.asarray(data["denominator_B3"], dtype=float)
            Bphi = None
        elif "denominator_B_phi" in data:
            B3 = None
            Bphi = np.asarray(data["denominator_B_phi"], dtype=float)
        else:
            raise KeyError("case bundle requires denominator_B3 or denominator_B_phi")
        periods = int(np.asarray(data["nfp"]).ravel()[0]) if "nfp" in data else 1
        q = np.asarray(data["q_profile"], dtype=float) if "q_profile" in data else None
        q_iota_sign = int(np.asarray(data["q_iota_sign"]).ravel()[0]) if "q_iota_sign" in data else 1
        arrays = {key: np.asarray(data[key]) for key in required}
    md = {"source_kind": "npz_case_bundle", "source_id": _file_sha256_prefix(source)}
    if include_source_path:
        md["source_path"] = str(source)
    if metadata:
        md.update(dict(metadata))
    return boundary_topology_case_from_arrays(
        name=name,
        denominator_B3=B3,
        denominator_B_phi=Bphi,
        nfp=periods,
        q_profile=q,
        q_iota_sign=q_iota_sign,
        metadata=md,
        **arrays,
    )


def extend_boundary_topology_case_to_resonance(
    case: BoundaryTopologyCase,
    *,
    m: int,
    n: int,
    fit_points: int = 5,
    n_extra: int = 4,
    outer_margin: float = 0.02,
    max_extension: float = 0.15,
) -> BoundaryTopologyCase:
    """Linearly continue healed edge surfaces to one nearby q resonance.

    This is intended for boundary island chains whose rational surface lies
    just outside the last closed VMEC surface.  Geometry, ``B0 dot grad(phi)``,
    and q are continued from their edge slopes in the same radial coordinate.
    The operation is deliberately bounded and recorded in case metadata; it is
    not a replacement for rebuilding healed coordinates through the island
    region when that data is available.
    """

    m_int = int(m)
    n_int = int(n)
    if m_int <= 0 or n_int <= 0:
        raise ValueError("m and n must be positive")
    radial = np.asarray(case.radial_labels, dtype=float)
    q = np.asarray(case.q_profile, dtype=float)
    target_q = float(m_int) / float(n_int)
    diff = q - target_q
    if np.any(diff == 0.0) or np.any(diff[:-1] * diff[1:] < 0.0):
        return case
    count = min(max(2, int(fit_points)), radial.size)
    q_slope, q_intercept = np.polyfit(radial[-count:], q[-count:], 1)
    if not np.isfinite(q_slope) or abs(float(q_slope)) <= 1.0e-14:
        raise ValueError("edge q profile is too flat to continue to a resonance")
    root = float((target_q - q_intercept) / q_slope)
    edge = float(radial[-1])
    extension = root - edge
    if extension <= 0.0:
        raise ValueError("requested resonance is not an outward continuation of the case")
    if extension > float(max_extension):
        raise ValueError(
            f"resonance requires radial extension {extension:.6g}, exceeding max_extension"
        )
    stop = root + max(0.0, float(outer_margin))
    extra = np.linspace(edge, stop, int(n_extra) + 1, endpoint=True)[1:]
    radial_out = np.concatenate([radial, extra])

    def continue_edge(values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float)
        slope = (data[:, -1, :] - data[:, -2, :]) / (radial[-1] - radial[-2])
        appended = np.stack(
            [data[:, -1, :] + slope * (value - radial[-1]) for value in extra],
            axis=1,
        )
        return np.concatenate([data, appended], axis=1)

    q_extra = q_slope * extra + q_intercept
    q_out = np.concatenate([q, q_extra])
    metadata = dict(case.metadata)
    metadata["edge_resonance_extension"] = {
        "model": "linear_healed_edge_continuation",
        "m": m_int,
        "n": n_int,
        "target_q": target_q,
        "resonant_radial_label": root,
        "source_edge_radial_label": edge,
        "extension": extension,
        "quantitative_geometry": False,
    }
    return boundary_topology_case_from_arrays(
        name=case.name,
        R_surf=continue_edge(case.R_surf),
        Z_surf=continue_edge(case.Z_surf),
        phi_vals=case.phi_vals,
        theta_vals=case.theta_vals,
        radial_labels=radial_out,
        iota_profile=float(case.q_iota_sign) / q_out,
        q_profile=q_out,
        q_iota_sign=case.q_iota_sign,
        denominator_B3=continue_edge(case.denominator_B3),
        nfp=case.nfp,
        coordinate_system=case.coordinate_system,
        radial_coordinate=case.radial_coordinate,
        background_field=case.background_field,
        equilibrium=case.equilibrium,
        core_reference=case.core_reference,
        metadata=metadata,
    )


def _file_sha256_prefix(path: Path, length: int = 12) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[: int(length)]


def _interp_mode_rows(values: np.ndarray, source_s: np.ndarray, target_s: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    source = np.asarray(source_s, dtype=float).ravel()
    target = np.asarray(target_s, dtype=float).ravel()
    if data.ndim != 2 or data.shape[0] != source.size:
        raise ValueError("mode rows must have shape (len(source_s), n_modes)")
    order = np.argsort(source)
    source = source[order]
    data = data[order]
    target_clipped = np.clip(target, source[0], source[-1])
    upper = np.searchsorted(source, target_clipped, side="right")
    upper = np.clip(upper, 1, source.size - 1)
    lower = upper - 1
    span = source[upper] - source[lower]
    alpha = np.divide(
        target_clipped - source[lower],
        span,
        out=np.zeros_like(target_clipped),
        where=span != 0.0,
    )
    return data[lower] * (1.0 - alpha[:, None]) + data[upper] * alpha[:, None]


def _evaluate_cos_modes(coefficients, m, n, theta_grid, phi_vals) -> np.ndarray:
    phase = (
        np.asarray(theta_grid, dtype=float)[:, :, None] * np.asarray(m, dtype=float)[None, None, :]
        - np.asarray(phi_vals, dtype=float)[:, None, None] * np.asarray(n, dtype=float)[None, None, :]
    )
    return np.sum(np.asarray(coefficients, dtype=float)[None, None, :] * np.cos(phase), axis=-1)


def _evaluate_sin_modes(coefficients, m, n, theta_grid, phi_vals) -> np.ndarray:
    phase = (
        np.asarray(theta_grid, dtype=float)[:, :, None] * np.asarray(m, dtype=float)[None, None, :]
        - np.asarray(phi_vals, dtype=float)[:, None, None] * np.asarray(n, dtype=float)[None, None, :]
    )
    return np.sum(np.asarray(coefficients, dtype=float)[None, None, :] * np.sin(phase), axis=-1)


def _vmec_theta_from_pest(
    theta_pest: np.ndarray,
    phi_vals: np.ndarray,
    lmns: np.ndarray,
    xm: np.ndarray,
    xn: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, float, int]:
    target = np.broadcast_to(theta_pest[None, :], (phi_vals.size, theta_pest.size))
    theta_vmec = target.copy()
    residual_max = np.inf
    for iteration in range(max(1, int(max_iterations))):
        phase = theta_vmec[:, :, None] * xm[None, None, :] - phi_vals[:, None, None] * xn[None, None, :]
        lam = np.sum(lmns[None, None, :] * np.sin(phase), axis=-1)
        derivative = 1.0 + np.sum((lmns * xm)[None, None, :] * np.cos(phase), axis=-1)
        derivative = np.where(np.abs(derivative) < 1.0e-10, np.sign(derivative) * 1.0e-10 + (derivative == 0.0) * 1.0e-10, derivative)
        residual = theta_vmec + lam - target
        theta_vmec = theta_vmec - residual / derivative
        residual_max = float(np.max(np.abs(residual)))
        if residual_max <= float(tolerance):
            return np.mod(theta_vmec, TWOPI), residual_max, iteration + 1
    raise RuntimeError(
        f"VMEC-to-PEST angle inversion did not converge: residual={residual_max:.3e}, "
        f"iterations={int(max_iterations)}"
    )


def vmec_boundary_topology_case_from_wout(
    path: str | Path,
    *,
    name: str = "W7-X public",
    radial_labels: Sequence[float] | None = None,
    n_radial: int = 14,
    radial_min: float = 0.12,
    radial_max: float = 1.0,
    n_phi: int = 40,
    n_phi_per_period: int | None = None,
    n_theta: int = 96,
    pest_tolerance: float = 1.0e-11,
    pest_max_iterations: int = 16,
    include_source_path: bool = False,
    metadata: Mapping[str, object] | None = None,
) -> BoundaryTopologyCase:
    """Reconstruct PEST surfaces and ``B0^phi`` directly from a VMEC wout.

    The implementation currently supports stellarator-symmetric VMEC output.
    VMEC's stored ``xn`` values are used directly, so field-period factors are
    not applied a second time.  The radial label is normalized toroidal flux
    ``s`` and ``q = 1 / iota``.  By default, ``n_phi`` samples the full torus
    for backward-compatible visualization.  Supplying ``n_phi_per_period``
    instead constructs exactly that many endpoint-excluded samples on
    ``[0, 2*pi/nfp)``; it does not build or slice a full-torus mesh.
    """

    from netCDF4 import Dataset

    source = Path(path).expanduser()
    with Dataset(source) as dataset:
        periods = int(np.asarray(dataset.variables["nfp"][:]).item())
        ns = int(np.asarray(dataset.variables["ns"][:]).item())
        lasym = bool(np.asarray(dataset.variables.get("lasym__logical__", np.array(0))[:]).item()) if "lasym__logical__" in dataset.variables else False
        if lasym:
            raise NotImplementedError("asymmetric VMEC surface reconstruction is not implemented")
        xm = np.asarray(dataset.variables["xm"][:], dtype=float)
        xn = np.asarray(dataset.variables["xn"][:], dtype=float)
        xm_nyq = np.asarray(dataset.variables["xm_nyq"][:], dtype=float)
        xn_nyq = np.asarray(dataset.variables["xn_nyq"][:], dtype=float)
        rmnc = np.asarray(dataset.variables["rmnc"][:], dtype=float)
        zmns = np.asarray(dataset.variables["zmns"][:], dtype=float)
        lmns = np.asarray(dataset.variables["lmns"][:], dtype=float)
        bsupvmnc = np.asarray(dataset.variables["bsupvmnc"][:], dtype=float)
        iotaf = np.asarray(dataset.variables["iotaf"][:], dtype=float)
        raxis_cc = np.asarray(dataset.variables["raxis_cc"][:], dtype=float) if "raxis_cc" in dataset.variables else None
        zaxis_cs = np.asarray(dataset.variables["zaxis_cs"][:], dtype=float) if "zaxis_cs" in dataset.variables else None

    if radial_labels is None:
        radial = np.linspace(float(radial_min), float(radial_max), int(n_radial))
    else:
        radial = np.asarray(radial_labels, dtype=float).ravel()
    if radial.size < 3 or np.any(radial <= 0.0) or np.any(radial > 1.0) or np.any(np.diff(radial) <= 0.0):
        raise ValueError("VMEC radial_labels must be strictly increasing in (0, 1]")
    if n_phi_per_period is None:
        toroidal_domain = "full_torus"
        phi_count = int(n_phi)
        phi_period = TWOPI
    else:
        toroidal_domain = "native_field_period"
        phi_count = int(n_phi_per_period)
        phi_period = ToroidalPeriodicity(periods).field_period
    if phi_count < 1:
        raise ValueError("n_phi or n_phi_per_period must be positive")
    phi_vals = np.linspace(0.0, phi_period, phi_count, endpoint=False)
    theta_vals = np.linspace(0.0, TWOPI, int(n_theta), endpoint=False)
    full_s = np.linspace(0.0, 1.0, ns)
    half_s = (np.arange(1, ns, dtype=float) - 0.5) / float(ns - 1)
    rm_rows = _interp_mode_rows(rmnc, full_s, radial)
    zm_rows = _interp_mode_rows(zmns, full_s, radial)
    lm_rows = _interp_mode_rows(lmns[1:], half_s, radial)
    b3_rows = _interp_mode_rows(bsupvmnc[1:], half_s, radial)
    iota = _interp_mode_rows(iotaf[:, None], full_s, radial)[:, 0]

    R_surf = np.empty((phi_vals.size, radial.size, theta_vals.size), dtype=float)
    Z_surf = np.empty_like(R_surf)
    B3 = np.empty_like(R_surf)
    inversion_residual = 0.0
    inversion_iterations = 0
    for ir in range(radial.size):
        theta_vmec, residual, iterations = _vmec_theta_from_pest(
            theta_vals,
            phi_vals,
            lm_rows[ir],
            xm,
            xn,
            tolerance=pest_tolerance,
            max_iterations=pest_max_iterations,
        )
        inversion_residual = max(inversion_residual, residual)
        inversion_iterations = max(inversion_iterations, iterations)
        R_surf[:, ir, :] = _evaluate_cos_modes(rm_rows[ir], xm, xn, theta_vmec, phi_vals)
        Z_surf[:, ir, :] = _evaluate_sin_modes(zm_rows[ir], xm, xn, theta_vmec, phi_vals)
        B3[:, ir, :] = _evaluate_cos_modes(b3_rows[ir], xm_nyq, xn_nyq, theta_vmec, phi_vals)

    if raxis_cc is not None and zaxis_cs is not None:
        axis_modes = np.arange(raxis_cc.size, dtype=float) * periods
        axis_R = np.sum(raxis_cc[None, :] * np.cos(phi_vals[:, None] * axis_modes[None, :]), axis=1)
        axis_Z = np.sum(zaxis_cs[None, :] * np.sin(phi_vals[:, None] * axis_modes[None, :]), axis=1)
    else:
        axis_R = np.mean(R_surf[:, 0, :], axis=1)
        axis_Z = np.mean(Z_surf[:, 0, :], axis=1)
    core_count = max(1, min(radial.size, int(np.searchsorted(radial, 0.45, side="right"))))
    core_reference = CorePreservationSnapshot(
        axis=np.array([axis_R[0], axis_Z[0]], dtype=float),
        radial_labels=radial[:core_count],
        surface_R=R_surf[:, :core_count, :],
        surface_Z=Z_surf[:, :core_count, :],
        q_profile=1.0 / iota[:core_count],
        iota_profile=iota[:core_count],
        scalars={"nfp": float(periods)},
        metadata={"source_kind": "vmec_wout", "coordinate_system": "PEST"},
    )
    md: dict[str, object] = {
        "source_kind": "vmec_wout",
        "source_id": _file_sha256_prefix(source),
        "coordinate_system": "PEST",
        "radial_coordinate": "s_normalized_toroidal_flux",
        "theta_relation": "theta_PEST=theta_VMEC+lambda",
        "vmec_fourier_phase": "m*theta_VMEC-xn*phi",
        "nardon_fourier_basis": "exp(i*(m*theta_PEST+n_N*phi))",
        "nardon_index_mapping": "n_N=-xn",
        "xn_includes_nfp": True,
        "toroidal_domain": toroidal_domain,
        "toroidal_domain_period_rad": float(phi_period),
        "native_field_period_sampling": bool(n_phi_per_period is not None),
        "n_phi_domain": int(phi_count),
        "pest_inversion_max_residual": float(inversion_residual),
        "pest_inversion_max_iterations": int(inversion_iterations),
    }
    if include_source_path:
        md["source_path"] = str(source)
    if metadata:
        md.update(dict(metadata))
    return BoundaryTopologyCase(
        name=name,
        R_surf=R_surf,
        Z_surf=Z_surf,
        phi_vals=phi_vals,
        theta_vals=theta_vals,
        radial_labels=radial,
        iota_profile=iota,
        q_profile=1.0 / iota,
        denominator_B3=B3,
        nfp=periods,
        coordinate_system="PEST",
        radial_coordinate="s",
        core_reference=core_reference,
        metadata=md,
    )


@dataclass(frozen=True)
class BoundaryDipoleSpectrumLibrary:
    """Precomputed unit-command Nardon responses for a linear actuator array.

    The historical class name is retained for compatibility.  ``actuators``
    may be either an exact finite-loop dipole array or an arbitrary field basis
    with the same label-safe ``actuators/field/control_bounds`` interface.
    """

    case: BoundaryTopologyCase
    actuators: BoundaryDipoleActuatorArray | BoundaryFieldActuatorArray
    responses: tuple[PerturbationCandidateSpectrumResponse, ...]
    m_max: int | None = None
    n_max: int | None = None
    base_tilde_b1: np.ndarray | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        responses = tuple(self.responses)
        if len(responses) != len(self.actuators.actuators):
            raise ValueError("one unit spectrum response is required per actuator")
        expected = self.case.R_surf.shape
        for response in responses:
            if np.asarray(response.tilde_b1).shape != expected:
                raise ValueError("unit actuator tilde_b1 arrays must match case surfaces")
        base = np.zeros(expected, dtype=complex) if self.base_tilde_b1 is None else np.asarray(self.base_tilde_b1, dtype=complex)
        if base.shape != expected:
            raise ValueError("base_tilde_b1 must match case surfaces")
        object.__setattr__(self, "responses", responses)
        object.__setattr__(self, "base_tilde_b1", base)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def control_labels(self) -> tuple[str, ...]:
        """Return stable actuator response-column labels."""

        return self.actuators.control_labels

    def combined_tilde_b1(self, controls: Sequence[float]) -> np.ndarray:
        """Return the linear vacuum Nardon field for ``controls``."""

        commands = np.asarray(controls, dtype=float).ravel()
        if commands.size != len(self.responses):
            raise ValueError("controls length must match response-library columns")
        out = np.asarray(self.base_tilde_b1, dtype=complex).copy()
        for command, response in zip(commands, self.responses):
            if float(command) != 0.0:
                out += float(command) * np.asarray(response.tilde_b1, dtype=complex)
        return out

    def combined_spectrum(
        self,
        controls: Sequence[float],
        *,
        tilde_b1: np.ndarray | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> RadialPerturbationFourierSpectrum:
        """Return a packed Fourier spectrum for one command vector."""

        commands = np.asarray(controls, dtype=float).ravel()
        canonical_tilde = self.combined_tilde_b1(commands)
        tilde = canonical_tilde if tilde_b1 is None else np.asarray(tilde_b1, dtype=complex)
        if tilde.shape != self.case.R_surf.shape:
            raise ValueError("tilde_b1 must match case surfaces")
        md = {
            "case_name": self.case.name,
            "coordinate_system": self.case.coordinate_system,
            "radial_coordinate": self.case.radial_coordinate,
            "nfp": self.case.nfp,
            "response_kind": self.metadata.get(
                "response_kind",
                getattr(self.actuators, "metadata", {}).get("response_kind", "boundary_linear_field_basis"),
            ),
        }
        md.update(dict(self.metadata))
        md.update(_case_spectrum_provenance(self.case))
        can_sign_delta = tilde_b1 is None or np.array_equal(tilde, canonical_tilde)
        can_sign_delta = can_sign_delta and not np.any(np.asarray(self.base_tilde_b1) != 0.0)
        if can_sign_delta and isinstance(self.actuators, BoundaryFieldActuatorArray):
            try:
                combined_field = self.actuators.grid_field(commands)
            except TypeError:
                combined_field = None
            delta_signature = None if combined_field is None else _grid_field_signature(combined_field)
            if delta_signature is not None:
                md["delta_field_signature"] = delta_signature
        if metadata:
            md.update(dict(metadata))
        return radial_perturbation_Fourier_spectrum(
            tilde,
            self.case.theta_vals,
            self.case.phi_vals,
            radial_labels=self.case.radial_labels,
            layout="phi-radial-theta",
            nfp=_surface_fourier_nfp(self.case),
            m_max=self.m_max,
            n_max=self.n_max,
            metadata=md,
        )


def build_boundary_perturbation_spectrum_library(
    case: BoundaryTopologyCase,
    actuators: BoundaryDipoleActuatorArray | BoundaryFieldActuatorArray,
    *,
    m_max: int | None = None,
    n_max: int | None = None,
    base_tilde_b1: np.ndarray | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> BoundaryDipoleSpectrumLibrary:
    """Sample arbitrary unit perturbation fields into a reusable spectrum basis."""

    _validate_grid_actuator_nfp(case, actuators)
    case_provenance = _case_spectrum_provenance(case)
    responses = []
    total = len(actuators.actuators)
    for index, actuator in enumerate(actuators.actuators):
        if progress is not None:
            progress(index, total, actuator.label)
        response_metadata = {
            "case_name": case.name,
            "control_label": actuator.label,
            "coordinate_system": case.coordinate_system,
            "radial_coordinate": case.radial_coordinate,
            **case_provenance,
        }
        grid_field = getattr(actuator, "grid_field", None)
        if grid_field is not None:
            delta_signature = _grid_field_signature(grid_field)
            if delta_signature is not None:
                response_metadata["delta_field_signature"] = delta_signature
        responses.append(
            _candidate_response_for_case(
                actuator.field(1.0),
                case,
                m_max=m_max,
                n_max=n_max,
                metadata=response_metadata,
            )
        )
    if progress is not None:
        progress(total, total, "complete")
    return BoundaryDipoleSpectrumLibrary(
        case=case,
        actuators=actuators,
        responses=tuple(responses),
        m_max=m_max,
        n_max=n_max,
        base_tilde_b1=base_tilde_b1,
        metadata={} if metadata is None else dict(metadata),
    )


def build_boundary_dipole_spectrum_library(
    case: BoundaryTopologyCase,
    actuators: BoundaryDipoleActuatorArray,
    *,
    m_max: int | None = None,
    n_max: int | None = None,
    base_tilde_b1: np.ndarray | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> BoundaryDipoleSpectrumLibrary:
    """Build a spectrum basis for exact finite-loop dipole actuators."""

    return build_boundary_perturbation_spectrum_library(
        case,
        actuators,
        m_max=m_max,
        n_max=n_max,
        base_tilde_b1=base_tilde_b1,
        progress=progress,
        metadata=metadata,
    )


# Preferred general name; keep the original public class import working.
BoundaryPerturbationSpectrumLibrary = BoundaryDipoleSpectrumLibrary


@dataclass(frozen=True)
class BoundaryTopologyPlasmaFeedback:
    """Optional self-consistent plasma response applied to a vacuum spectrum.

    ``tilde_b1`` is the authoritative perturbation sampled for the spectrum.
    Field objects retain the explicit ``B0``/``delta B`` decomposition:
    ``vacuum_delta_field`` is ``delta B_vac``, ``plasma_delta_field`` is the
    plasma increment, and ``total_field`` is an authoritative ``B_total`` when
    supplied.  Aggregate fields are derived only when their components expose
    a supported cylindrical sampling interface.
    """

    tilde_b1: np.ndarray
    response_case: BoundaryTopologyCase | None = None
    core: CorePreservationSnapshot | None = None
    background_field: Any = None
    equilibrium: Any = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    vacuum_delta_field: Any = None
    plasma_delta_field: Any = None
    total_field: Any = None


@dataclass(frozen=True)
class BoundaryTopologyFieldContext(BoundaryPlasmaResponseInput):
    """Heat-callback request carrying the authoritative response field contract."""

    total_field: Any = None
    delta_field: Any = None
    plasma_delta_field: Any = None

    @property
    def background_field(self):
        """Alias ``baseline_field`` as the response ``B0`` field."""

        return self.baseline_field


@dataclass(frozen=True)
class BoundaryTopologyResponseSnapshot(BoundaryPlasmaResponseSnapshot):
    """Topology snapshot retaining the explicit vacuum and plasma increments."""

    vacuum_delta_field: Any = None
    plasma_delta_field: Any = None
    field_context: BoundaryTopologyFieldContext | None = None


BoundaryTopologyPlasmaFeedbackModel = Callable[
    [BoundaryTopologyCase, BoundaryPlasmaResponseInput, np.ndarray],
    BoundaryTopologyPlasmaFeedback | Mapping[str, object] | np.ndarray,
]


def _coerce_plasma_feedback(
    value,
    *,
    vacuum_tilde_b1: np.ndarray,
    case: BoundaryTopologyCase,
    request: BoundaryPlasmaResponseInput,
    default_vacuum_delta_field=None,
    default_vacuum_delta_field_source: str = "unavailable",
) -> BoundaryTopologyPlasmaFeedback:
    if value is None:
        background_field = request.baseline_field
        if background_field is None:
            background_field = case.background_field
        equilibrium = request.baseline_equilibrium
        if equilibrium is None:
            equilibrium = case.equilibrium
        return BoundaryTopologyPlasmaFeedback(
            tilde_b1=vacuum_tilde_b1,
            response_case=case,
            core=case.core_reference,
            background_field=background_field,
            equilibrium=equilibrium,
            metadata={
                "response_model": "vacuum",
                "spectrum_delta_components": ("vacuum_delta_field",),
                "field_component_sources": {
                    "background_field": (
                        "request.baseline_field"
                        if request.baseline_field is not None
                        else "case.background_field"
                    ),
                    "vacuum_delta_field": default_vacuum_delta_field_source,
                    "plasma_delta_field": "not_applicable_vacuum_response",
                    "total_field": (
                        "request.metadata.total_field"
                        if request.metadata.get("total_field") is not None
                        else "unavailable"
                    ),
                },
            },
            vacuum_delta_field=default_vacuum_delta_field,
            total_field=request.metadata.get("total_field"),
        )
    if isinstance(value, BoundaryTopologyPlasmaFeedback):
        feedback = value
    elif isinstance(value, Mapping):
        feedback = BoundaryTopologyPlasmaFeedback(**value)
    else:
        feedback = BoundaryTopologyPlasmaFeedback(tilde_b1=np.asarray(value, dtype=complex))
    response_case = case if feedback.response_case is None else feedback.response_case
    if not isinstance(response_case, BoundaryTopologyCase):
        raise TypeError("plasma-feedback response_case must be a BoundaryTopologyCase")
    tilde = np.asarray(feedback.tilde_b1, dtype=complex)
    if tilde.shape != response_case.R_surf.shape:
        raise ValueError("plasma-feedback tilde_b1 must match response_case surfaces")
    replacement_case = response_case is not case
    if feedback.core is not None:
        core = feedback.core
        core_source = "feedback.core"
    else:
        core = response_case.core_reference
        core_source = "response_case.core_reference"
    if feedback.background_field is not None:
        background_field = feedback.background_field
        background_source = "feedback.background_field"
    elif replacement_case:
        background_field = response_case.background_field
        background_source = "response_case.background_field"
    else:
        background_field = request.baseline_field
        background_source = "request.baseline_field"
        if background_field is None:
            background_field = response_case.background_field
            background_source = "response_case.background_field"
    if feedback.equilibrium is not None:
        equilibrium = feedback.equilibrium
        equilibrium_source = "feedback.equilibrium"
    elif replacement_case:
        equilibrium = response_case.equilibrium
        equilibrium_source = "response_case.equilibrium"
    else:
        equilibrium = request.baseline_equilibrium
        equilibrium_source = "request.baseline_equilibrium"
        if equilibrium is None:
            equilibrium = response_case.equilibrium
            equilibrium_source = "response_case.equilibrium"
    metadata = dict(feedback.metadata or {})
    metadata.setdefault(
        "field_component_sources",
        {
            "background_field": background_source,
            "vacuum_delta_field": (
                "feedback.vacuum_delta_field"
                if feedback.vacuum_delta_field is not None
                else default_vacuum_delta_field_source
            ),
            "plasma_delta_field": (
                "feedback.plasma_delta_field"
                if feedback.plasma_delta_field is not None
                else "unavailable"
            ),
            "total_field": (
                "feedback.total_field"
                if feedback.total_field is not None
                else "unavailable"
            ),
        },
    )
    metadata.setdefault(
        "response_context_sources",
        {
            "core": core_source,
            "equilibrium": equilibrium_source,
        },
    )
    return BoundaryTopologyPlasmaFeedback(
        tilde_b1=tilde,
        response_case=response_case,
        core=core,
        background_field=background_field,
        equilibrium=equilibrium,
        metadata=metadata,
        vacuum_delta_field=(
            default_vacuum_delta_field
            if feedback.vacuum_delta_field is None
            else feedback.vacuum_delta_field
        ),
        plasma_delta_field=feedback.plasma_delta_field,
        total_field=feedback.total_field,
    )


_FIELD_COMPONENT_NAMES = ("vacuum_delta_field", "plasma_delta_field")


def _spectrum_delta_components(
    feedback: BoundaryTopologyPlasmaFeedback,
    vacuum_tilde_b1: np.ndarray,
) -> tuple[str, ...]:
    configured = dict(feedback.metadata or {}).get("spectrum_delta_components")
    if configured is not None:
        if isinstance(configured, str):
            components = (configured,)
        else:
            components = tuple(str(value) for value in configured)
        unknown = [value for value in components if value not in _FIELD_COMPONENT_NAMES]
        if unknown or len(set(components)) != len(components):
            raise ValueError(
                "spectrum_delta_components must contain unique vacuum_delta_field and/or "
                "plasma_delta_field entries"
            )
        return components
    if feedback.plasma_delta_field is not None:
        if feedback.vacuum_delta_field is not None:
            return _FIELD_COMPONENT_NAMES
        return ("plasma_delta_field",)
    if (
        feedback.vacuum_delta_field is not None
        and feedback.tilde_b1.shape == vacuum_tilde_b1.shape
        and np.array_equal(feedback.tilde_b1, vacuum_tilde_b1)
    ):
        return ("vacuum_delta_field",)
    return ()


def _compose_field_objects(fields: Sequence[Any]):
    components = tuple(value for value in fields if value is not None)
    if not components:
        return None
    if len(components) == 1:
        return components[0]
    try:
        return BoundaryFieldSuperposition(
            tuple(ScaledBoundaryFieldCandidate(value, 1.0) for value in components)
        )
    except TypeError:
        return None


def _field_sampling_scope(field) -> str:
    if field is None:
        return "unavailable"
    return str(getattr(field, "sampling_scope", "declared_field_domain"))


def _authoritative_field_context(
    request: BoundaryPlasmaResponseInput,
    feedback: BoundaryTopologyPlasmaFeedback,
    vacuum_tilde_b1: np.ndarray,
) -> tuple[BoundaryTopologyFieldContext, dict[str, object]]:
    components = _spectrum_delta_components(feedback, vacuum_tilde_b1)
    component_fields = tuple(getattr(feedback, name) for name in components)
    missing_components = tuple(
        name for name, value in zip(components, component_fields) if value is None
    )
    if missing_components:
        delta_field = None
        delta_source = "unavailable_missing_declared_components"
    else:
        delta_field = _compose_field_objects(component_fields)
        if not components:
            delta_source = "unavailable_tilde_b1_only"
        elif delta_field is None:
            delta_source = "unavailable_unsupported_component_composition"
        elif len(components) == 1:
            delta_source = components[0]
        else:
            delta_source = "composed_vacuum_plus_plasma"
    component_scopes = tuple(_field_sampling_scope(value) for value in component_fields)
    delta_scope = (
        "unavailable"
        if delta_field is None
        else (
            "response_surface_stack"
            if "response_surface_stack" in component_scopes
            else "declared_field_domain"
        )
    )

    if feedback.total_field is not None:
        total_field = feedback.total_field
        total_source = "feedback.total_field"
        total_scope = _field_sampling_scope(total_field)
    elif (
        feedback.background_field is not None
        and delta_field is not None
        and delta_scope != "response_surface_stack"
    ):
        total_field = _compose_field_objects((feedback.background_field, delta_field))
        total_source = (
            "composed_background_plus_delta"
            if total_field is not None
            else "unavailable_unsupported_background_delta_composition"
        )
        total_scope = "unavailable" if total_field is None else "declared_field_domain"
    elif delta_scope == "response_surface_stack":
        total_field = None
        total_source = "unavailable_surface_samples_are_not_a_trace_field"
        total_scope = "unavailable"
    else:
        total_field = None
        total_source = "unavailable_missing_background_or_delta"
        total_scope = "unavailable"

    has_split = feedback.background_field is not None and delta_field is not None
    if total_field is not None:
        authoritative = "total_field"
    elif has_split:
        authoritative = "background_field_plus_delta_field"
    else:
        authoritative = "tilde_b1_only"
    contract: dict[str, object] = {
        "semantics": {
            "background_field": "B0",
            "vacuum_delta_field": "deltaB_vac",
            "plasma_delta_field": "deltaB_plasma",
            "delta_field": "deltaB represented by spectrum_delta_components",
            "total_field": "B_total",
        },
        "precedence": (
            "feedback.total_field",
            "composed background_field + delta_field",
            "B0/deltaB split",
            "tilde_b1_only",
        ),
        "component_sources": dict(feedback.metadata or {}).get(
            "field_component_sources",
            {},
        ),
        "spectrum_delta_components": components,
        "delta_field_source": delta_source,
        "delta_field_sampling_scope": delta_scope,
        "total_field_source": total_source,
        "total_field_sampling_scope": total_scope,
        "authoritative_field": authoritative,
        "has_b0_delta_split": has_split,
        "field_trace_consistent": (
            total_field is not None and total_scope != "response_surface_stack"
        )
        or (has_split and delta_scope != "response_surface_stack"),
    }
    if missing_components:
        contract["missing_spectrum_field_components"] = missing_components

    metadata = dict(request.metadata or {})
    metadata.update(
        {
            "field_contract": contract,
            "total_field": total_field,
            "background_field": feedback.background_field,
            "delta_field": delta_field,
            "vacuum_delta_field": feedback.vacuum_delta_field,
            "plasma_delta_field": feedback.plasma_delta_field,
        }
    )
    context = BoundaryTopologyFieldContext(
        controls=request.controls,
        control_labels=request.control_labels,
        baseline_equilibrium=feedback.equilibrium,
        baseline_field=feedback.background_field,
        vacuum_delta_field=feedback.vacuum_delta_field,
        metadata=metadata,
        total_field=total_field,
        delta_field=delta_field,
        plasma_delta_field=feedback.plasma_delta_field,
    )
    return context, contract


@dataclass(frozen=True)
class BoundaryTopologyForwardState:
    """Complete nonlinear diagnostic state for one actuator command vector."""

    controls: np.ndarray
    response_case: BoundaryTopologyCase
    vacuum_tilde_b1: np.ndarray
    plasma_tilde_b1: np.ndarray
    spectrum: RadialPerturbationFourierSpectrum
    chains: tuple[ResonantIslandChain, ...]
    overlap_bands: tuple[ChirikovOverlapBand, ...]
    chaotic_intervals: tuple[ChaoticLayerInterval, ...]
    heat: BoundaryTopologyHeatState | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    field_context: BoundaryTopologyFieldContext | None = None


@dataclass(frozen=True)
class BoundaryTopologyCaseBackend:
    """Plasma-response backend joining spectrum, topology, and heat diagnostics."""

    library: BoundaryDipoleSpectrumLibrary
    n_values: Sequence[int]
    m_values: Sequence[int] | Mapping[int, Sequence[int]] | None = None
    min_b_res: float = 0.0
    sigma_threshold: float = 1.0
    co_radial_tol: float = 0.0
    heat_model: BoundaryTopologyHeatForwardModel | None = None
    plasma_feedback: BoundaryTopologyPlasmaFeedbackModel | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_values = tuple(int(value) for value in self.n_values)
        if not n_values:
            raise ValueError("n_values must contain at least one physical toroidal mode")
        object.__setattr__(self, "n_values", n_values)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def forward_state(self, request: BoundaryPlasmaResponseInput) -> tuple[BoundaryTopologyForwardState, BoundaryTopologyPlasmaFeedback]:
        if tuple(request.control_labels) != self.library.control_labels:
            raise ValueError("plasma-response control labels do not match the dipole response library")
        controls = np.asarray(request.controls, dtype=float).ravel()
        vacuum_tilde = self.library.combined_tilde_b1(controls)
        default_vacuum_field = request.vacuum_delta_field
        default_vacuum_field_source = (
            "request.vacuum_delta_field"
            if default_vacuum_field is not None
            else "unavailable"
        )
        if default_vacuum_field is None and not np.any(
            np.asarray(self.library.base_tilde_b1, dtype=complex) != 0.0
        ):
            default_vacuum_field = self.library.actuators.field(controls)
            default_vacuum_field_source = "library.actuators.field"
        feedback_value = None
        if self.plasma_feedback is not None:
            feedback_value = self.plasma_feedback(self.library.case, request, vacuum_tilde.copy())
        feedback = _coerce_plasma_feedback(
            feedback_value,
            vacuum_tilde_b1=vacuum_tilde,
            case=self.library.case,
            request=request,
            default_vacuum_delta_field=default_vacuum_field,
            default_vacuum_delta_field_source=default_vacuum_field_source,
        )
        response_case = feedback.response_case
        field_context, field_contract = _authoritative_field_context(
            request,
            feedback,
            vacuum_tilde,
        )
        spectrum = radial_perturbation_Fourier_spectrum(
            feedback.tilde_b1,
            response_case.theta_vals,
            response_case.phi_vals,
            radial_labels=response_case.radial_labels,
            layout="phi-radial-theta",
            nfp=_surface_fourier_nfp(response_case),
            m_max=self.library.m_max,
            n_max=self.library.n_max,
            metadata={
                "case_name": response_case.name,
                "coordinate_system": response_case.coordinate_system,
                "radial_coordinate": response_case.radial_coordinate,
                "nfp": response_case.nfp,
                "response_kind": self.library.metadata.get(
                    "response_kind",
                    getattr(self.library.actuators, "metadata", {}).get(
                        "response_kind", "boundary_linear_field_basis"
                    ),
                ),
                "plasma_response_model": feedback.metadata.get("response_model", "custom"),
                "field_contract": field_contract,
            },
        )
        chains = tuple(
            analyze_resonant_island_chains_multi_n(
                spectrum,
                response_case.q_profile,
                n_values=self.n_values,
                radial_labels=response_case.radial_labels,
                m_values=self.m_values,
                min_b_res=float(self.min_b_res),
            )
        )
        bands = tuple(
            chirikov_overlap_bands(
                chains,
                include_cross_n=True,
                radial_min=float(response_case.radial_labels[0]),
                radial_max=float(response_case.radial_labels[-1]),
                co_radial_tol=float(self.co_radial_tol),
            )
        )
        intervals = tuple(chaotic_layer_intervals(bands, sigma_threshold=float(self.sigma_threshold)))
        heat = None
        if self.heat_model is not None:
            heat = self.heat_model.evaluate(
                response_case,
                field_context,
                spectrum,
                chains,
                intervals,
            )
            if not isinstance(heat, BoundaryTopologyHeatState):
                raise TypeError("heat_model.evaluate must return BoundaryTopologyHeatState")
        state = BoundaryTopologyForwardState(
            controls=controls.copy(),
            response_case=response_case,
            vacuum_tilde_b1=vacuum_tilde,
            plasma_tilde_b1=np.asarray(feedback.tilde_b1, dtype=complex),
            spectrum=spectrum,
            chains=chains,
            overlap_bands=bands,
            chaotic_intervals=intervals,
            heat=heat,
            metadata={
                "case_name": self.library.case.name,
                "response_model": feedback.metadata.get("response_model", "custom"),
                "sigma_threshold": float(self.sigma_threshold),
                "field_contract": field_contract,
            },
            field_context=field_context,
        )
        return state, feedback

    def evaluate(self, request: BoundaryPlasmaResponseInput) -> BoundaryPlasmaResponseSnapshot:
        """Evaluate a command vector and expose its full forward state."""

        state, feedback = self.forward_state(request)
        md = dict(self.metadata)
        md.update(dict(feedback.metadata or {}))
        md.update({
            "case_name": self.library.case.name,
            "boundary_topology_state": state,
            "boundary_topology_field_context": state.field_context,
            "field_contract": state.field_context.metadata["field_contract"],
            "vacuum_delta_field": feedback.vacuum_delta_field,
            "plasma_delta_field": feedback.plasma_delta_field,
            "has_b0_delta_split": bool(
                state.field_context.background_field is not None
                and state.field_context.delta_field is not None
            ),
        })
        return BoundaryTopologyResponseSnapshot(
            total_field=state.field_context.total_field,
            background_field=state.field_context.background_field,
            delta_field=state.field_context.delta_field,
            equilibrium=feedback.equilibrium,
            core=feedback.core,
            metadata=md,
            vacuum_delta_field=feedback.vacuum_delta_field,
            plasma_delta_field=feedback.plasma_delta_field,
            field_context=state.field_context,
        )


@dataclass(frozen=True)
class BoundaryHeatTargetRegion:
    """Named target region in wall ``(phi, s)`` heat-map coordinates."""

    label: str
    s_bounds: tuple[float, float]
    phi_bounds: tuple[float, float] | None = None
    weight: float = 1.0

    def mask(self, heat: BoundaryTopologyHeatState) -> np.ndarray:
        """Return this region's mask on a heat state."""

        s_lo, s_hi = (float(self.s_bounds[0]), float(self.s_bounds[1]))
        if s_hi < s_lo:
            raise ValueError("heat-region s_bounds must be ordered")
        s_mask = (heat.s_values >= s_lo) & (heat.s_values <= s_hi)
        if self.phi_bounds is None:
            phi_mask = np.ones(heat.phi_values.size, dtype=bool)
        else:
            raw_phi_lo = float(self.phi_bounds[0])
            raw_phi_hi = float(self.phi_bounds[1])
            if abs(raw_phi_hi - raw_phi_lo) >= TWOPI - 1.0e-12:
                return np.ones((heat.phi_values.size, 1), dtype=bool) & s_mask[None, :]
            phi_lo = raw_phi_lo % TWOPI
            phi_hi = raw_phi_hi % TWOPI
            wrapped = np.mod(heat.phi_values, TWOPI)
            if phi_lo <= phi_hi:
                phi_mask = (wrapped >= phi_lo) & (wrapped <= phi_hi)
            else:
                phi_mask = (wrapped >= phi_lo) | (wrapped <= phi_hi)
        return phi_mask[:, None] & s_mask[None, :]


@dataclass(frozen=True)
class BoundaryTopologyObservableSpec:
    """Standard joint island, chaos, heat, and core observable definition."""

    resonant_modes: Sequence[tuple[int, int]] = field(default_factory=tuple)
    resonant_quantities: Sequence[str] = (
        "half_width",
        "coefficient_real",
        "coefficient_imag",
    )
    resonant_weights: Any = None
    chaos_regions: Sequence[tuple[float, float]] = field(default_factory=tuple)
    chaos_labels: Sequence[str] = field(default_factory=tuple)
    chaos_weights: Any = None
    heat_quantities: Sequence[str] = (
        "total_power",
        "peak_flux",
        "centroid_s",
        "rms_width_s",
    )
    heat_weights: Any = None
    heat_regions: Sequence[BoundaryHeatTargetRegion] = field(default_factory=tuple)
    core_radial_max: float | None = 0.4
    core_field_weights: Sequence[float] = (20.0, 20.0)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        modes = tuple((int(m), int(n)) for m, n in self.resonant_modes)
        regions = tuple((float(low), float(high)) for low, high in self.chaos_regions)
        labels = tuple(str(label) for label in self.chaos_labels)
        heat_regions = tuple(self.heat_regions)
        if len(regions) != len(labels):
            raise ValueError("chaos_regions and chaos_labels must have the same length")
        if not all(isinstance(region, BoundaryHeatTargetRegion) for region in heat_regions):
            raise TypeError("heat_regions must contain BoundaryHeatTargetRegion objects")
        object.__setattr__(self, "resonant_modes", modes)
        object.__setattr__(self, "resonant_quantities", tuple(str(q) for q in self.resonant_quantities))
        object.__setattr__(self, "chaos_regions", regions)
        object.__setattr__(self, "chaos_labels", labels)
        object.__setattr__(self, "heat_quantities", tuple(str(q) for q in self.heat_quantities))
        object.__setattr__(self, "heat_regions", heat_regions)
        object.__setattr__(self, "core_field_weights", tuple(float(v) for v in self.core_field_weights))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


def _forward_state_from_snapshot(snapshot: BoundaryPlasmaResponseSnapshot) -> BoundaryTopologyForwardState:
    state = dict(snapshot.metadata or {}).get("boundary_topology_state")
    if not isinstance(state, BoundaryTopologyForwardState):
        raise ValueError("plasma response snapshot does not contain a BoundaryTopologyForwardState")
    return state


def boundary_topology_case_observable_builder(
    spec: BoundaryTopologyObservableSpec,
) -> Callable[[BoundaryPlasmaResponseSnapshot, BoundaryPlasmaResponseInput], BoundaryResponseObservables]:
    """Build the standard joint observable callback for a case backend."""

    if not isinstance(spec, BoundaryTopologyObservableSpec):
        raise TypeError("spec must be BoundaryTopologyObservableSpec")

    def build(snapshot: BoundaryPlasmaResponseSnapshot, request: BoundaryPlasmaResponseInput) -> BoundaryResponseObservables:
        del request
        state = _forward_state_from_snapshot(snapshot)
        groups: list[BoundaryResponseObservables] = []
        if spec.resonant_modes:
            groups.append(
                resonant_chain_observables(
                    state.chains,
                    spec.resonant_modes,
                    quantities=spec.resonant_quantities,
                    weights=spec.resonant_weights,
                    prefix="island",
                    metadata={"coordinate_system": state.spectrum.metadata.get("coordinate_system")},
                )
            )
        if spec.chaos_regions:
            groups.append(
                chaotic_layer_region_observables(
                    state.chaotic_intervals,
                    spec.chaos_regions,
                    spec.chaos_labels,
                    weights=spec.chaos_weights,
                    prefix="chaos",
                )
            )
        if state.heat is not None and spec.heat_quantities:
            phi_period = float(
                dict(state.heat.metadata or {}).get(
                    "field_period",
                    state.response_case.field_period,
                )
            )
            groups.append(
                wall_heat_flux_observables(
                    state.heat.heat,
                    phi_values=state.heat.phi_values,
                    s_values=state.heat.s_values,
                    cell_areas=state.heat.cell_areas,
                    phi_period=phi_period,
                    quantities=spec.heat_quantities,
                    weights=spec.heat_weights,
                    prefix="heat",
                    metadata=state.heat.metadata,
                )
            )
        if state.heat is not None and spec.heat_regions:
            groups.append(
                wall_heat_region_observables(
                    state.heat.heat if state.heat.cell_areas is None else state.heat.heat * state.heat.cell_areas,
                    [region.mask(state.heat) for region in spec.heat_regions],
                    [region.label for region in spec.heat_regions],
                    weights=[region.weight for region in spec.heat_regions],
                    normalize=True,
                    prefix="heat_region",
                )
            )
        if spec.core_radial_max is not None:
            radial = np.asarray(state.spectrum.radial_labels, dtype=float)
            selected = radial <= float(spec.core_radial_max)
            if not np.any(selected):
                selected[np.argmin(radial)] = True
            core_values = np.abs(np.asarray(state.plasma_tilde_b1)[:, selected, :])
            finite = core_values[np.isfinite(core_values)]
            if finite.size:
                rms = float(np.sqrt(np.mean(finite**2)))
                maximum = float(np.max(finite))
            else:
                rms = float("nan")
                maximum = float("nan")
            groups.append(
                boundary_response_observables(
                    ("radial_leakage_rms", "radial_leakage_max"),
                    (rms, maximum),
                    weights=spec.core_field_weights,
                    prefix="core.field",
                    metadata={"radial_max": float(spec.core_radial_max)},
                )
            )
        return stack_boundary_response_observables(groups, metadata=spec.metadata)

    return build


def make_boundary_topology_control_problem(
    library: BoundaryDipoleSpectrumLibrary,
    observable_spec: BoundaryTopologyObservableSpec,
    target: Mapping[str, float] | Sequence[float],
    *,
    extra_observable_builders: Sequence[Callable[..., BoundaryResponseObservables]] = (),
    initial_controls: Sequence[float] | None = None,
    n_values: Sequence[int] | None = None,
    m_values: Sequence[int] | Mapping[int, Sequence[int]] | None = None,
    heat_model: BoundaryTopologyHeatForwardModel | None = None,
    plasma_feedback: BoundaryTopologyPlasmaFeedbackModel | None = None,
    min_b_res: float = 0.0,
    sigma_threshold: float = 1.0,
    co_radial_tol: float = 0.0,
    steps: Any = 0.05,
    n_iterations: int = 5,
    bounds: Any = None,
    regularization: float = 1.0e-5,
    line_search: Sequence[float] = (1.0, 0.5, 0.25, 0.125),
    convergence_tolerance: float | None = None,
    core_weights: Mapping[str, float] | None = None,
    target_zero_prefixes: Sequence[str] = ("core.",),
    target_preserve_initial_prefixes: Sequence[str] = (),
    target_preserve_initial_labels: Sequence[str] = (),
    metadata: Mapping[str, object] | None = None,
):
    """Assemble a complete bounded control problem from a case response library.

    ``extra_observable_builders`` is the extension point for expensive or
    backend-specific DP^k, FTLE, Newton fixed-point, manifold, and traced wall
    diagnostics.  Each builder follows the standard plasma-response observable
    callback signature and is stacked with the spectrum/heat/core rows.
    """

    from pyna.toroidal.control.boundary_topology_control import BoundaryTopologyControlProblem

    labels = library.control_labels
    controls = np.zeros(len(labels), dtype=float) if initial_controls is None else np.asarray(initial_controls, dtype=float).ravel()
    if controls.size != len(labels):
        raise ValueError("initial_controls must match actuator columns")
    if n_values is None:
        n_values = tuple(dict.fromkeys(int(n) for _m, n in observable_spec.resonant_modes))
    backend = BoundaryTopologyCaseBackend(
        library=library,
        n_values=n_values,
        m_values=m_values,
        min_b_res=min_b_res,
        sigma_threshold=sigma_threshold,
        co_radial_tol=co_radial_tol,
        heat_model=heat_model,
        plasma_feedback=plasma_feedback,
        metadata={} if metadata is None else dict(metadata),
    )
    md = {
        "case_name": library.case.name,
        "coordinate_system": library.case.coordinate_system,
        "radial_coordinate": library.case.radial_coordinate,
        "nfp": library.case.nfp,
    }
    if metadata:
        md.update(dict(metadata))
    standard_builder = boundary_topology_case_observable_builder(observable_spec)
    return BoundaryTopologyControlProblem(
        backend=backend,
        initial_controls=controls,
        control_labels=labels,
        target=target,
        observable_builders=(standard_builder,) + tuple(extra_observable_builders),
        core_reference=library.case.core_reference,
        core_weights=core_weights,
        target_zero_prefixes=target_zero_prefixes,
        target_preserve_initial_prefixes=target_preserve_initial_prefixes,
        target_preserve_initial_labels=target_preserve_initial_labels,
        steps=steps,
        n_iterations=n_iterations,
        bounds=bounds,
        control_bounds=library.actuators.control_bounds,
        regularization=regularization,
        control_scale=library.actuators.control_scale,
        line_search=line_search,
        convergence_tolerance=convergence_tolerance,
        baseline_equilibrium=library.case.equilibrium,
        baseline_field=library.case.background_field,
        metadata=md,
    )


__all__ = [
    "BoundaryDipoleSpectrumLibrary",
    "BoundaryPerturbationSpectrumLibrary",
    "BoundaryHeatTargetRegion",
    "BoundaryTopologyCase",
    "BoundaryTopologyCaseBackend",
    "CallableBoundaryTopologyHeatForwardModel",
    "BoundaryTopologyFieldContext",
    "BoundaryTopologyForwardState",
    "BoundaryTopologyHeatForwardModel",
    "BoundaryTopologyHeatState",
    "BoundaryTopologyObservableSpec",
    "BoundaryTopologyPlasmaFeedback",
    "BoundaryTopologyPlasmaFeedbackModel",
    "BoundaryTopologyResponseSnapshot",
    "ReducedSpectralHeatModel",
    "boundary_topology_case_from_arrays",
    "boundary_topology_case_observable_builder",
    "build_boundary_dipole_spectrum_library",
    "build_boundary_perturbation_spectrum_library",
    "extend_boundary_topology_case_to_resonance",
    "load_boundary_topology_case_npz",
    "make_boundary_topology_control_problem",
    "vmec_boundary_topology_case_from_wout",
]
