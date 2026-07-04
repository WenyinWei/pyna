"""Generic beta-ramp state containers and spectrum diagnostics.

This module is intentionally data-source agnostic.  Private equilibrium or
continuation tools can adapt their outputs into :class:`BetaRampState`, while
pyna handles reusable field conversion, radial-spectrum analysis, small-divisor
diagnostics, and scan summaries.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field
from typing import Any

import numpy as np

from pyna.fields import VectorFieldCylind
from pyna.toroidal._periodic_grid import prepare_surface_arrays
from pyna.toroidal.perturbation_spectrum import (
    ChirikovOverlap,
    RadialPerturbationFourierSpectrum,
    ResonantIslandChain,
    analyze_resonant_island_chains_multi_n,
    chirikov_overlaps,
    nardon_radial_perturbation,
    radial_perturbation_Fourier_spectrum,
    sample_cylindrical_vector_grid_on_surfaces,
)


_PATH_KEY_PARTS = (
    "path",
    "file",
    "dir",
    "root",
    "screenshot",
    "image",
    "figure",
)


def _array_or_none(value: Any, *, name: str, dtype=float) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=dtype)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def _require_array(value: np.ndarray | None, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is required")
    return value


def _same_1d_grid(left: np.ndarray | None, right: np.ndarray | None, name: str) -> bool:
    if left is None or right is None:
        return False
    return left.shape == right.shape and np.allclose(left, right, rtol=0.0, atol=0.0)


def _finite_fraction(value: np.ndarray) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        return 1.0
    return float(np.count_nonzero(np.isfinite(arr)) / arr.size)


def _metadata_float(metadata: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key in metadata and metadata[key] is not None:
            try:
                return float(metadata[key])
            except (TypeError, ValueError):
                return None
    return None


def scrub_beta_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    allow_keys: Iterable[str] | None = None,
    redacted: str = "<redacted>",
) -> dict[str, Any]:
    """Return metadata safe for public summaries.

    When ``allow_keys`` is omitted, all keys are retained but path-like keys are
    redacted.  When ``allow_keys`` is provided, only those keys are retained.
    """

    if metadata is None:
        return {}
    allowed = None if allow_keys is None else {str(key) for key in allow_keys}
    public: dict[str, Any] = {}
    for key, value in dict(metadata).items():
        key_s = str(key)
        if allowed is not None and key_s not in allowed:
            continue
        lowered = key_s.lower()
        if any(part in lowered for part in _PATH_KEY_PARTS):
            public[key_s] = redacted
        else:
            public[key_s] = value
    return public


@dataclass(frozen=True)
class BetaRampState:
    """One equilibrium or field state in a beta-ramp scan.

    The field grid uses pyna's cylindrical component order:
    ``BR(R, Z, Phi)``, ``BZ(R, Z, Phi)``, and ``BPhi(R, Z, Phi)``.  Surface
    arrays use the usual ``(phi, radial, theta)`` layout consumed by
    :mod:`pyna.toroidal.perturbation_spectrum`.
    """

    beta: float | None = None
    label: str = ""
    R_grid: np.ndarray | None = None
    Z_grid: np.ndarray | None = None
    Phi_grid: np.ndarray | None = None
    BR: np.ndarray | None = None
    BZ: np.ndarray | None = None
    BPhi: np.ndarray | None = None
    field_periods: int = 1
    R_surf: np.ndarray | None = None
    Z_surf: np.ndarray | None = None
    phi_vals: np.ndarray | None = None
    theta_vals: np.ndarray | None = None
    radial_labels: np.ndarray | None = None
    q_profile: np.ndarray | None = None
    iota_profile: np.ndarray | None = None
    metadata: Mapping[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("R_grid", "Z_grid", "Phi_grid", "radial_labels", "q_profile", "iota_profile"):
            object.__setattr__(
                self,
                name,
                _array_or_none(getattr(self, name), name=name, dtype=float),
            )
        for name in ("BR", "BZ", "BPhi", "R_surf", "Z_surf", "phi_vals", "theta_vals"):
            object.__setattr__(
                self,
                name,
                _array_or_none(getattr(self, name), name=name, dtype=float),
            )
        field_periods = int(self.field_periods)
        if field_periods < 1:
            raise ValueError("field_periods must be positive")
        object.__setattr__(self, "field_periods", field_periods)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

        has_any_grid = any(
            getattr(self, name) is not None
            for name in ("R_grid", "Z_grid", "Phi_grid", "BR", "BZ", "BPhi")
        )
        if has_any_grid:
            R = _require_array(self.R_grid, "R_grid")
            Z = _require_array(self.Z_grid, "Z_grid")
            Phi = _require_array(self.Phi_grid, "Phi_grid")
            shape = (R.size, Z.size, Phi.size)
            for name in ("BR", "BZ", "BPhi"):
                arr = _require_array(getattr(self, name), name)
                if arr.shape != shape:
                    raise ValueError(f"{name} shape {arr.shape} does not match field grid {shape}")

        has_any_surface = any(
            getattr(self, name) is not None
            for name in ("R_surf", "Z_surf", "phi_vals", "theta_vals", "radial_labels")
        )
        if has_any_surface:
            R_surf = _require_array(self.R_surf, "R_surf")
            Z_surf = _require_array(self.Z_surf, "Z_surf")
            phi = _require_array(self.phi_vals, "phi_vals")
            theta = _require_array(self.theta_vals, "theta_vals")
            radial = _require_array(self.radial_labels, "radial_labels")
            R_prepared, _Z_prepared, _phi_prepared, _theta_prepared = prepare_surface_arrays(
                R_surf,
                Z_surf,
                phi,
                theta,
            )
            if radial.ndim != 1 or radial.size != R_prepared.shape[1]:
                raise ValueError("radial_labels must match the prepared surface radial count")
            if not np.all(np.isfinite(radial)) or np.any(np.diff(radial) <= 0.0):
                raise ValueError("radial_labels must be finite and strictly increasing")

        for name in ("q_profile", "iota_profile"):
            values = getattr(self, name)
            if values is None:
                continue
            radial = _require_array(self.radial_labels, "radial_labels")
            if values.shape != radial.shape:
                raise ValueError(f"{name} must match radial_labels")

    @classmethod
    def from_field(
        cls,
        field: VectorFieldCylind,
        *,
        beta: float | None = None,
        label: str = "",
        R_surf: np.ndarray | None = None,
        Z_surf: np.ndarray | None = None,
        phi_vals: np.ndarray | None = None,
        theta_vals: np.ndarray | None = None,
        radial_labels: np.ndarray | None = None,
        q_profile: np.ndarray | None = None,
        iota_profile: np.ndarray | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "BetaRampState":
        """Build a state from a :class:`VectorFieldCylind` instance."""

        BR, BZ, BPhi = field.components_3d
        return cls(
            beta=beta,
            label=label or field.label or field.name,
            R_grid=field.R_arr,
            Z_grid=field.Z_arr,
            Phi_grid=field.Phi,
            BR=BR,
            BZ=BZ,
            BPhi=BPhi,
            field_periods=field.field_periods,
            R_surf=R_surf,
            Z_surf=Z_surf,
            phi_vals=phi_vals,
            theta_vals=theta_vals,
            radial_labels=radial_labels,
            q_profile=q_profile,
            iota_profile=iota_profile,
            metadata={} if metadata is None else metadata,
        )

    @property
    def has_field_grid(self) -> bool:
        return all(getattr(self, name) is not None for name in ("R_grid", "Z_grid", "Phi_grid", "BR", "BZ", "BPhi"))

    @property
    def has_surfaces(self) -> bool:
        return all(
            getattr(self, name) is not None
            for name in ("R_surf", "Z_surf", "phi_vals", "theta_vals", "radial_labels")
        )

    def field_cache(self) -> dict[str, np.ndarray | int | str]:
        """Return a named field-cache dictionary for cyna/pyna bridge code."""

        if not self.has_field_grid:
            raise ValueError("state does not contain a field grid")
        return {
            "R_grid": _require_array(self.R_grid, "R_grid"),
            "Z_grid": _require_array(self.Z_grid, "Z_grid"),
            "Phi_grid": _require_array(self.Phi_grid, "Phi_grid"),
            "BR": _require_array(self.BR, "BR"),
            "BZ": _require_array(self.BZ, "BZ"),
            "BPhi": _require_array(self.BPhi, "BPhi"),
            "field_periods": int(self.field_periods),
            "label": self.label,
        }

    def as_vector_field(self, *, label: str | None = None) -> VectorFieldCylind:
        """Convert this state's field grid to :class:`VectorFieldCylind`."""

        return VectorFieldCylind.from_field_cache(self.field_cache(), label=self.label if label is None else label)

    def q_values(self) -> np.ndarray:
        """Return q on ``radial_labels``, deriving it from iota when needed."""

        if self.q_profile is not None:
            return np.asarray(self.q_profile, dtype=float)
        if self.iota_profile is None:
            raise ValueError("state must provide q_profile or iota_profile")
        iota = np.asarray(self.iota_profile, dtype=float)
        return 1.0 / iota

    def iota_values(self) -> np.ndarray:
        """Return iota on ``radial_labels``, deriving it from q when needed."""

        if self.iota_profile is not None:
            return np.asarray(self.iota_profile, dtype=float)
        if self.q_profile is None:
            raise ValueError("state must provide q_profile or iota_profile")
        q = np.asarray(self.q_profile, dtype=float)
        return 1.0 / q

    def public_metadata(self, *, allow_keys: Iterable[str] | None = None) -> dict[str, Any]:
        return scrub_beta_metadata(self.metadata, allow_keys=allow_keys)

    def delta_to(self, reference: "BetaRampState", **kwargs) -> "BetaRampState":
        """Return ``self - reference`` as a perturbation state."""

        return delta_beta_ramp_state(self, reference, **kwargs)


@dataclass(frozen=True)
class BetaRampSurfaceFieldSamples:
    """Delta-field samples on one state's magnetic-coordinate surfaces."""

    delta_BR: np.ndarray
    delta_BZ: np.ndarray
    delta_BPhi: np.ndarray
    denominator_BPhi: np.ndarray
    R_surf: np.ndarray
    Z_surf: np.ndarray
    phi_vals: np.ndarray
    theta_vals: np.ndarray
    radial_labels: np.ndarray


@dataclass(frozen=True)
class BetaRampRadialModeReport:
    """Small-divisor and RMP/nRMP split diagnostics for one radial surface."""

    radial_label: float
    iota: float
    min_abs_miota_plus_n: float
    mode_m: int | None
    mode_n: int | None
    resonant_mode_count: int
    near_resonant_mode_count: int
    nonresonant_mode_count: int
    resonant_norm: float
    near_resonant_norm: float
    nonresonant_norm: float
    filtered_modes: tuple[tuple[int, int, float], ...] = ()

    @property
    def has_small_divisor(self) -> bool:
        return bool(self.near_resonant_mode_count or self.resonant_mode_count)


@dataclass(frozen=True)
class BetaRampTrustReport:
    """Scan-level confidence label with scalar metrics and reasons."""

    status: str
    reasons: tuple[str, ...]
    metrics: Mapping[str, float] = dataclass_field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class BetaRampSpectrumDiagnostics:
    """Complete spectrum diagnosis for one beta-ramp state."""

    state: BetaRampState
    reference: BetaRampState | None
    perturbation: BetaRampState
    surface_samples: BetaRampSurfaceFieldSamples
    tilde_b1: np.ndarray
    spectrum: RadialPerturbationFourierSpectrum
    chains: tuple[ResonantIslandChain, ...]
    overlaps: tuple[ChirikovOverlap, ...]
    small_divisors: tuple[BetaRampRadialModeReport, ...]
    trust: BetaRampTrustReport

    def dominant_chains(self, max_count: int = 8) -> tuple[ResonantIslandChain, ...]:
        """Return strongest island-chain estimates by ``b_res``."""

        ordered = sorted(self.chains, key=lambda chain: chain.b_res, reverse=True)
        return tuple(ordered[: max(0, int(max_count))])

    def summary(self) -> dict[str, Any]:
        """Return a compact row suitable for JSON/CSV scan summaries."""

        return _summary_row(self)


def delta_beta_ramp_state(
    state: BetaRampState,
    reference: BetaRampState,
    *,
    label: str | None = None,
    surface_source: str = "state",
    metadata: Mapping[str, Any] | None = None,
) -> BetaRampState:
    """Return a state whose field is ``state - reference``.

    The returned object keeps q/iota profiles from ``state`` because those
    profiles classify the current beta point.  Surfaces are taken from
    ``state`` by default.
    """

    if not state.has_field_grid or not reference.has_field_grid:
        raise ValueError("state and reference must both contain field grids")
    for name in ("R_grid", "Z_grid", "Phi_grid"):
        if not _same_1d_grid(getattr(state, name), getattr(reference, name), name):
            raise ValueError(f"{name} differs between state and reference")

    if surface_source not in {"state", "reference"}:
        raise ValueError("surface_source must be 'state' or 'reference'")
    surface_state = state if surface_source == "state" else reference
    md = {
        "source_label": state.label,
        "reference_label": reference.label,
    }
    if state.beta is not None and reference.beta is not None:
        md["beta_delta"] = float(state.beta) - float(reference.beta)
    if metadata:
        md.update(dict(metadata))
    return BetaRampState(
        beta=state.beta,
        label=label or f"{state.label or 'state'} minus {reference.label or 'reference'}",
        R_grid=state.R_grid,
        Z_grid=state.Z_grid,
        Phi_grid=state.Phi_grid,
        BR=_require_array(state.BR, "BR") - _require_array(reference.BR, "reference.BR"),
        BZ=_require_array(state.BZ, "BZ") - _require_array(reference.BZ, "reference.BZ"),
        BPhi=_require_array(state.BPhi, "BPhi") - _require_array(reference.BPhi, "reference.BPhi"),
        field_periods=state.field_periods,
        R_surf=surface_state.R_surf,
        Z_surf=surface_state.Z_surf,
        phi_vals=surface_state.phi_vals,
        theta_vals=surface_state.theta_vals,
        radial_labels=surface_state.radial_labels,
        q_profile=state.q_profile,
        iota_profile=state.iota_profile,
        metadata=md,
    )


def sample_beta_ramp_delta_on_surfaces(
    state: BetaRampState,
    *,
    reference: BetaRampState | None = None,
    denominator_state: BetaRampState | None = None,
) -> BetaRampSurfaceFieldSamples:
    """Sample ``delta_B`` and denominator ``B_phi`` on magnetic surfaces."""

    if reference is None:
        perturbation = state
        denom_source = denominator_state or state
    else:
        perturbation = delta_beta_ramp_state(state, reference)
        denom_source = denominator_state or reference
    if not perturbation.has_surfaces:
        raise ValueError("state must contain surface coordinates")
    if not perturbation.has_field_grid:
        raise ValueError("state must contain a field grid or a precomputed perturbation field")
    if not denom_source.has_field_grid:
        raise ValueError("denominator_state/reference must contain a field grid")

    R_surf = _require_array(perturbation.R_surf, "R_surf")
    Z_surf = _require_array(perturbation.Z_surf, "Z_surf")
    phi_vals = _require_array(perturbation.phi_vals, "phi_vals")
    theta_vals = _require_array(perturbation.theta_vals, "theta_vals")
    radial_labels = _require_array(perturbation.radial_labels, "radial_labels")
    delta_BR, delta_BPhi, delta_BZ = sample_cylindrical_vector_grid_on_surfaces(
        _require_array(perturbation.R_grid, "R_grid"),
        _require_array(perturbation.Z_grid, "Z_grid"),
        _require_array(perturbation.Phi_grid, "Phi_grid"),
        _require_array(perturbation.BR, "BR"),
        _require_array(perturbation.BPhi, "BPhi"),
        _require_array(perturbation.BZ, "BZ"),
        R_surf,
        Z_surf,
        phi_vals,
        theta_vals,
    )
    _denom_BR, denom_BPhi, _denom_BZ = sample_cylindrical_vector_grid_on_surfaces(
        _require_array(denom_source.R_grid, "denominator.R_grid"),
        _require_array(denom_source.Z_grid, "denominator.Z_grid"),
        _require_array(denom_source.Phi_grid, "denominator.Phi_grid"),
        _require_array(denom_source.BR, "denominator.BR"),
        _require_array(denom_source.BPhi, "denominator.BPhi"),
        _require_array(denom_source.BZ, "denominator.BZ"),
        R_surf,
        Z_surf,
        phi_vals,
        theta_vals,
    )
    return BetaRampSurfaceFieldSamples(
        delta_BR=delta_BR,
        delta_BZ=delta_BZ,
        delta_BPhi=delta_BPhi,
        denominator_BPhi=denom_BPhi,
        R_surf=R_surf,
        Z_surf=Z_surf,
        phi_vals=phi_vals,
        theta_vals=theta_vals,
        radial_labels=radial_labels,
    )


def radial_small_divisor_reports(
    spectrum: RadialPerturbationFourierSpectrum,
    iota_profile: Sequence[float],
    *,
    radial_labels: Sequence[float] | None = None,
    resonance_tol: float = 1.0e-9,
    small_divisor_tol: float = 3.0e-2,
    min_mode_amplitude: float = 0.0,
    max_reported_modes: int = 8,
) -> tuple[BetaRampRadialModeReport, ...]:
    """Compute per-surface ``abs(m*iota+n)`` diagnostics for active modes."""

    iota = np.asarray(iota_profile, dtype=float)
    if spectrum.dBr.ndim == 1:
        dbr = np.asarray(spectrum.dBr, dtype=complex)[np.newaxis, :]
        radial = np.array([np.nan], dtype=float) if radial_labels is None else np.asarray(radial_labels, dtype=float)
        if radial.size != 1:
            raise ValueError("radial_labels must have length 1 for a single-surface spectrum")
        if iota.size != 1:
            raise ValueError("iota_profile must have length 1 for a single-surface spectrum")
    else:
        dbr = np.asarray(spectrum.dBr, dtype=complex)
        radial = spectrum.radial_labels if radial_labels is None else radial_labels
        if radial is None:
            raise ValueError("radial_labels are required for radial-stack diagnostics")
        radial = np.asarray(radial, dtype=float)
        if radial.shape != iota.shape or dbr.shape[0] != radial.size:
            raise ValueError("iota_profile and radial_labels must match the spectrum radial count")

    m = np.asarray(spectrum.m, dtype=int)
    n = np.asarray(spectrum.n, dtype=int)
    reports: list[BetaRampRadialModeReport] = []
    for ir, (s_val, iota_val) in enumerate(zip(radial, iota)):
        coeff = dbr[ir]
        amp = np.abs(coeff)
        active = np.isfinite(amp) & (amp > float(min_mode_amplitude))
        if not np.any(active):
            reports.append(
                BetaRampRadialModeReport(
                    radial_label=float(s_val),
                    iota=float(iota_val),
                    min_abs_miota_plus_n=float("nan"),
                    mode_m=None,
                    mode_n=None,
                    resonant_mode_count=0,
                    near_resonant_mode_count=0,
                    nonresonant_mode_count=0,
                    resonant_norm=0.0,
                    near_resonant_norm=0.0,
                    nonresonant_norm=0.0,
                )
            )
            continue
        denom = np.abs(m.astype(float) * float(iota_val) + n.astype(float))
        denom_active = np.where(active, denom, np.inf)
        idx_min = int(np.argmin(denom_active))
        resonant = active & (denom <= float(resonance_tol))
        near = active & ~resonant & (denom <= float(small_divisor_tol))
        nonres = active & ~resonant
        mode_order = np.argsort(np.where(active, denom, np.inf))
        filtered: list[tuple[int, int, float]] = []
        for idx in mode_order:
            if len(filtered) >= int(max_reported_modes):
                break
            if not active[idx] or not np.isfinite(denom[idx]) or denom[idx] > float(small_divisor_tol):
                continue
            filtered.append((int(m[idx]), int(n[idx]), float(denom[idx])))
        reports.append(
            BetaRampRadialModeReport(
                radial_label=float(s_val),
                iota=float(iota_val),
                min_abs_miota_plus_n=float(denom_active[idx_min]),
                mode_m=int(m[idx_min]),
                mode_n=int(n[idx_min]),
                resonant_mode_count=int(np.count_nonzero(resonant)),
                near_resonant_mode_count=int(np.count_nonzero(near)),
                nonresonant_mode_count=int(np.count_nonzero(nonres)),
                resonant_norm=float(np.linalg.norm(amp[resonant])),
                near_resonant_norm=float(np.linalg.norm(amp[near])),
                nonresonant_norm=float(np.linalg.norm(amp[nonres])),
                filtered_modes=tuple(filtered),
            )
        )
    return tuple(reports)


def classify_beta_ramp_trust(
    state: BetaRampState,
    spectrum: RadialPerturbationFourierSpectrum,
    chains: Sequence[ResonantIslandChain],
    overlaps: Sequence[ChirikovOverlap],
    small_divisors: Sequence[BetaRampRadialModeReport],
    *,
    small_divisor_watch: float = 3.0e-2,
    small_divisor_low: float = 1.0e-8,
    chirikov_watch: float = 0.7,
    chirikov_low: float = 1.0,
    nan_fraction_low: float = 1.0e-3,
    equilibrium_residual_low: float | None = None,
    condition_number_low: float = 1.0e12,
    fpt_residual_low: float | None = None,
    trace_exit_low: float = 1.0,
) -> BetaRampTrustReport:
    """Classify numerical confidence for one beta-ramp diagnostic result."""

    reasons: list[str] = []
    low = False
    watch = False

    max_chirikov = max((float(overlap.sigma) for overlap in overlaps), default=0.0)
    min_small = min(
        (
            report.min_abs_miota_plus_n
            for report in small_divisors
            if np.isfinite(report.min_abs_miota_plus_n)
        ),
        default=float("nan"),
    )
    n_resonant = int(sum(report.resonant_mode_count for report in small_divisors))
    n_near = int(sum(report.near_resonant_mode_count for report in small_divisors))
    max_half_width = max((float(chain.half_width) for chain in chains if np.isfinite(chain.half_width)), default=0.0)
    max_b_res = max((float(chain.b_res) for chain in chains if np.isfinite(chain.b_res)), default=0.0)
    finite_fraction = _finite_fraction(np.asarray(spectrum.dBr_grid))
    nan_fraction = 1.0 - finite_fraction
    q_shear_reversals = 0
    try:
        q = state.q_values()
        dq = np.diff(q)
        dq_nonzero = dq[dq != 0.0]
        if dq_nonzero.size > 1:
            signs = np.sign(dq_nonzero)
            q_shear_reversals = int(np.count_nonzero(signs[1:] != signs[:-1]))
    except ValueError:
        pass

    if max_chirikov >= float(chirikov_low):
        low = True
        reasons.append("chirikov_overlap_ge_1")
    elif max_chirikov >= float(chirikov_watch):
        watch = True
        reasons.append("chirikov_overlap_near_1")

    if np.isfinite(min_small) and min_small <= float(small_divisor_watch):
        watch = True
        reasons.append("small_divisor_near_resonance")
        if min_small <= float(small_divisor_low) and not chains:
            low = True
            reasons.append("small_divisor_without_resonant_chain")

    if nan_fraction > float(nan_fraction_low):
        low = True
        reasons.append("spectrum_contains_nan_or_out_of_grid_samples")

    if q_shear_reversals:
        watch = True
        reasons.append("q_profile_shear_reversal")

    md = state.metadata
    eq_residual = _metadata_float(md, ("equilibrium_residual", "solver_residual", "residual"))
    if equilibrium_residual_low is not None and eq_residual is not None and eq_residual > float(equilibrium_residual_low):
        low = True
        reasons.append("equilibrium_residual_above_threshold")

    condition = _metadata_float(md, ("condition_number", "linear_solve_condition", "closure_condition"))
    if condition is not None and condition > float(condition_number_low):
        low = True
        reasons.append("condition_number_above_threshold")

    fpt_residual = _metadata_float(md, ("fpt_residual", "closure_residual", "fixed_point_residual"))
    if fpt_residual_low is not None and fpt_residual is not None and fpt_residual > float(fpt_residual_low):
        low = True
        reasons.append("fpt_residual_above_threshold")

    trace_exits = _metadata_float(md, ("trace_exit_count", "grid_exit_count", "wall_hit_count"))
    if trace_exits is not None and trace_exits >= float(trace_exit_low):
        watch = True
        reasons.append("field_line_trace_exits_present")

    status = "low-confidence" if low else ("watch" if watch else "ok")
    metrics = {
        "max_chirikov": float(max_chirikov),
        "min_abs_miota_plus_n": float(min_small),
        "n_resonant_modes": float(n_resonant),
        "n_near_resonant_modes": float(n_near),
        "max_island_half_width": float(max_half_width),
        "max_b_res": float(max_b_res),
        "nan_fraction": float(nan_fraction),
        "q_shear_reversal_count": float(q_shear_reversals),
    }
    if eq_residual is not None:
        metrics["equilibrium_residual"] = float(eq_residual)
    if condition is not None:
        metrics["condition_number"] = float(condition)
    if fpt_residual is not None:
        metrics["fpt_residual"] = float(fpt_residual)
    if trace_exits is not None:
        metrics["trace_exit_count"] = float(trace_exits)
    return BetaRampTrustReport(status=status, reasons=tuple(dict.fromkeys(reasons)), metrics=metrics)


def diagnose_beta_ramp_state(
    state: BetaRampState,
    *,
    reference: BetaRampState | None = None,
    n_values: Iterable[int] | None = None,
    m_values: Iterable[int] | Mapping[int, Iterable[int]] | None = None,
    m_max: int | None = None,
    n_max: int | None = None,
    min_amplitude: float = 0.0,
    min_b_res: float = 0.0,
    resonance_tol: float = 1.0e-9,
    small_divisor_tol: float = 3.0e-2,
    min_mode_amplitude: float = 0.0,
    trust_kwargs: Mapping[str, Any] | None = None,
) -> BetaRampSpectrumDiagnostics:
    """Compute Nardon spectrum, island chains, Chirikov, and trust metrics."""

    if not state.has_surfaces:
        raise ValueError("state must contain surface coordinates")
    samples = sample_beta_ramp_delta_on_surfaces(state, reference=reference)
    tilde_b1 = nardon_radial_perturbation(
        samples.R_surf,
        samples.Z_surf,
        samples.phi_vals,
        samples.theta_vals,
        samples.delta_BR,
        samples.delta_BZ,
        samples.delta_BPhi,
        samples.radial_labels,
        denominator_B_phi=samples.denominator_BPhi,
    )
    spectrum = radial_perturbation_Fourier_spectrum(
        tilde_b1,
        samples.theta_vals,
        samples.phi_vals,
        radial_labels=samples.radial_labels,
        m_max=m_max,
        n_max=n_max,
        min_amplitude=min_amplitude,
    )
    q_profile = state.q_values()
    iota_profile = state.iota_values()
    chains = tuple(
        analyze_resonant_island_chains_multi_n(
            spectrum,
            q_profile,
            n_values=n_values,
            radial_labels=samples.radial_labels,
            m_values=m_values,
            min_b_res=min_b_res,
        )
    )
    overlaps = tuple(chirikov_overlaps(chains))
    small_divisors = radial_small_divisor_reports(
        spectrum,
        iota_profile,
        radial_labels=samples.radial_labels,
        resonance_tol=resonance_tol,
        small_divisor_tol=small_divisor_tol,
        min_mode_amplitude=min_mode_amplitude,
    )
    perturbation = state if reference is None else delta_beta_ramp_state(state, reference)
    trust = classify_beta_ramp_trust(
        state,
        spectrum,
        chains,
        overlaps,
        small_divisors,
        **dict(trust_kwargs or {}),
    )
    return BetaRampSpectrumDiagnostics(
        state=state,
        reference=reference,
        perturbation=perturbation,
        surface_samples=samples,
        tilde_b1=tilde_b1,
        spectrum=spectrum,
        chains=chains,
        overlaps=overlaps,
        small_divisors=small_divisors,
        trust=trust,
    )


def _summary_row(result: BetaRampSpectrumDiagnostics) -> dict[str, Any]:
    metrics = dict(result.trust.metrics)
    row = {
        "label": result.state.label,
        "beta": np.nan if result.state.beta is None else float(result.state.beta),
        "reference_label": None if result.reference is None else result.reference.label,
        "reference_beta": None if result.reference is None or result.reference.beta is None else float(result.reference.beta),
        "status": result.trust.status,
        "reasons": list(result.trust.reasons),
        "n_chains": int(len(result.chains)),
        "n_overlaps": int(len(result.overlaps)),
        "dominant_modes": [(chain.m, chain.n) for chain in result.dominant_chains(5)],
    }
    row.update(metrics)
    return row


def beta_scan_summary_rows(results: Sequence[BetaRampSpectrumDiagnostics]) -> list[dict[str, Any]]:
    """Return compact rows for plotting or serializing a beta scan."""

    return [_summary_row(result) for result in results]


__all__ = [
    "BetaRampRadialModeReport",
    "BetaRampSpectrumDiagnostics",
    "BetaRampState",
    "BetaRampSurfaceFieldSamples",
    "BetaRampTrustReport",
    "beta_scan_summary_rows",
    "classify_beta_ramp_trust",
    "delta_beta_ramp_state",
    "diagnose_beta_ramp_state",
    "radial_small_divisor_reports",
    "sample_beta_ramp_delta_on_surfaces",
    "scrub_beta_metadata",
]
