"""Production heat-load validation for one labelled manifold branch.

Cyna traces the global first-wall hits.  PyNA preserves the exact branch seed
orders, accounts for unresolved absolute power, bins resolved power on the full
toroidal wall, and returns the refreshed strike labels and local tangent
planes needed by a subsequent differentiable ESSOS inner solve.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

import numpy as np

from pyna.topo.manifold_correspondence import ManifoldBranchReference
from pyna.topo.manifold_strike import (
    local_wall_plane_from_strike,
    manifold_branch_strike_seed_bundle,
    refresh_manifold_strike_match,
)
from pyna.topo.manifold_strike_contracts import (
    LocalWallPlane,
    ManifoldStrikeLabel,
    ManifoldStrikeMatch,
)
from pyna.toroidal.control.heat_contracts import BoundaryTopologyHeatState
from pyna.toroidal.control.strike_heat import (
    StrikeSeedBundle,
    WallStrikeSamples,
    trace_wall_strikes_field,
    wall_heat_state_from_strikes,
)


def _nonnegative_finite(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


@dataclass(frozen=True)
class ManifoldHeatValidationReport:
    """Authoritative first hits, wall heat state, and acceptance decision."""

    branch_label: str
    seed_orders: np.ndarray
    strike_powers_W: np.ndarray
    power_provenance: str
    seed_bundle: StrikeSeedBundle
    wall_strikes: WallStrikeSamples
    heat_state: BoundaryTopologyHeatState
    strike_matches: tuple[ManifoldStrikeMatch, ...]
    wall_planes: tuple[LocalWallPlane, ...]
    peak_heat_flux_W_m2: float
    maximum_heat_flux_W_m2: float
    unresolved_power_W: float
    maximum_unresolved_power_W: float
    all_labels_resolved: bool
    accepted: bool
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        label = str(self.branch_label).strip()
        orders = np.asarray(self.seed_orders, dtype=int).ravel()
        powers = np.asarray(self.strike_powers_W, dtype=float).ravel()
        provenance = str(self.power_provenance).strip()
        matches = tuple(self.strike_matches)
        planes = tuple(self.wall_planes)
        if not label:
            raise ValueError("branch_label must not be empty")
        if orders.size == 0 or powers.shape != orders.shape:
            raise ValueError("seed orders and strike powers must be non-empty and aligned")
        if np.unique(orders).size != orders.size or np.any(orders < 0):
            raise ValueError("seed_orders must be unique non-negative integers")
        if not np.all(np.isfinite(powers)) or np.any(powers < 0.0):
            raise ValueError("strike_powers_W must be non-negative and finite")
        if not provenance:
            raise ValueError("power_provenance must not be empty")
        if not isinstance(self.seed_bundle, StrikeSeedBundle):
            raise TypeError("seed_bundle must be StrikeSeedBundle")
        if not isinstance(self.wall_strikes, WallStrikeSamples):
            raise TypeError("wall_strikes must be WallStrikeSamples")
        if not isinstance(self.heat_state, BoundaryTopologyHeatState):
            raise TypeError("heat_state must be BoundaryTopologyHeatState")
        if self.seed_bundle.label != label or self.wall_strikes.label != label:
            raise ValueError("production heat objects belong to a different branch")
        if not all(isinstance(value, ManifoldStrikeMatch) for value in matches):
            raise TypeError("strike_matches must contain ManifoldStrikeMatch values")
        if not all(isinstance(value, LocalWallPlane) for value in planes):
            raise TypeError("wall_planes must contain LocalWallPlane values")
        if len(planes) > len(matches):
            raise ValueError("wall planes cannot outnumber strike matches")
        for match, plane in zip(matches, planes):
            if match.label != plane.strike_label:
                raise ValueError("wall plane belongs to a different strike label")

        peak = _nonnegative_finite(
            self.peak_heat_flux_W_m2,
            "peak_heat_flux_W_m2",
        )
        maximum = _nonnegative_finite(
            self.maximum_heat_flux_W_m2,
            "maximum_heat_flux_W_m2",
        )
        unresolved = _nonnegative_finite(
            self.unresolved_power_W,
            "unresolved_power_W",
        )
        maximum_unresolved = _nonnegative_finite(
            self.maximum_unresolved_power_W,
            "maximum_unresolved_power_W",
        )
        metadata_unresolved = float(self.heat_state.metadata["unresolved_power"])
        if not np.isclose(unresolved, metadata_unresolved, rtol=0.0, atol=1.0e-14):
            raise ValueError("unresolved_power_W disagrees with heat_state")
        expected_peak = float(np.max(np.asarray(self.heat_state.heat, dtype=float)))
        if not np.isclose(peak, expected_peak, rtol=0.0, atol=1.0e-14):
            raise ValueError("peak_heat_flux_W_m2 disagrees with heat_state")

        all_resolved = bool(self.all_labels_resolved)
        accepted = bool(self.accepted)
        reason = None if self.rejection_reason is None else str(self.rejection_reason)
        if accepted:
            if reason is not None:
                raise ValueError("an accepted report cannot have a rejection reason")
            if (
                not all_resolved
                or len(matches) != orders.size
                or len(planes) != orders.size
            ):
                raise ValueError("an accepted report requires every strike label")
            if unresolved > maximum_unresolved or peak > maximum:
                raise ValueError("an accepted report violates a heat-load gate")
            if not bool(self.heat_state.metadata.get("quantitative", False)):
                raise ValueError("an accepted report requires quantitative heat state")
        elif reason is None:
            raise ValueError("a rejected report requires rejection_reason")

        object.__setattr__(self, "branch_label", label)
        object.__setattr__(self, "seed_orders", orders.copy())
        object.__setattr__(self, "strike_powers_W", powers.copy())
        object.__setattr__(self, "power_provenance", provenance)
        object.__setattr__(self, "strike_matches", matches)
        object.__setattr__(self, "wall_planes", planes)
        object.__setattr__(self, "peak_heat_flux_W_m2", peak)
        object.__setattr__(self, "maximum_heat_flux_W_m2", maximum)
        object.__setattr__(self, "unresolved_power_W", unresolved)
        object.__setattr__(
            self,
            "maximum_unresolved_power_W",
            maximum_unresolved,
        )
        object.__setattr__(self, "all_labels_resolved", all_resolved)
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "rejection_reason", reason)


def _initial_strike_matches(
    branch: ManifoldBranchReference,
    bundle: StrikeSeedBundle,
    strikes: WallStrikeSamples,
) -> tuple[ManifoldStrikeMatch, ...]:
    strike_by_seed = {
        int(seed_index): strike_index
        for strike_index, seed_index in enumerate(strikes.seed_index)
    }
    matches = []
    for bundle_seed_index, seed_order in enumerate(branch.seed_orders):
        strike_index = strike_by_seed[bundle_seed_index]
        point = np.asarray(
            [
                strikes.R[strike_index],
                strikes.Z[strike_index],
                strikes.phi[strike_index],
            ],
            dtype=float,
        )
        matches.append(
            ManifoldStrikeMatch(
                label=ManifoldStrikeLabel(
                    branch_label=bundle.label,
                    direction=bundle.direction,
                    seed_order=int(seed_order),
                ),
                point_RZPhi_m_rad=point,
                target_distance_m=0.0,
                connection_length_m=float(strikes.connection_length[strike_index]),
                bundle_seed_index=bundle_seed_index,
                distance_mode="xyz",
            )
        )
    return tuple(matches)


def _validate_previous_matches(
    previous_matches: Sequence[ManifoldStrikeMatch],
    branch: ManifoldBranchReference,
    direction: str,
) -> tuple[ManifoldStrikeMatch, ...]:
    matches = tuple(previous_matches)
    if len(matches) != branch.n_seeds:
        raise ValueError("previous_matches must contain one match per branch seed")
    for index, match in enumerate(matches):
        if not isinstance(match, ManifoldStrikeMatch):
            raise TypeError("previous_matches must contain ManifoldStrikeMatch values")
        expected = ManifoldStrikeLabel(
            branch_label=branch.label,
            direction=direction,
            seed_order=int(branch.seed_orders[index]),
        )
        if match.label != expected:
            raise ValueError("previous_matches must follow branch seed-order identity")
    return matches


def validate_manifold_heat_load(
    field: Any,
    branch_reference: ManifoldBranchReference,
    wall: Any,
    *,
    strike_powers_W: Any,
    power_provenance: str,
    phi_edges: Any,
    s_edges: Any,
    max_turns: int,
    DPhi: float,
    maximum_heat_flux_W_m2: float,
    maximum_unresolved_power_W: float = 0.0,
    maximum_projection_distance_m: float = 1.0e-3,
    previous_matches: Sequence[ManifoldStrikeMatch] | None = None,
    maximum_hit_displacement_m: float | None = None,
    field_period: float | None = None,
    extend_phi: bool = True,
    trace_function: Callable[..., Any] | None = None,
) -> ManifoldHeatValidationReport:
    """Retrace and gate a quantitative, exact-label production heat state.

    Absolute powers must be supplied in ``branch_reference.seed_orders`` order.
    If ``previous_matches`` is present, each exact label is refreshed under the
    supplied hit-displacement trust radius.  An accepted report requires every
    seed to have a first-wall hit; unresolved power is always reported even
    when it is below the configured numerical tolerance.
    """

    if not isinstance(branch_reference, ManifoldBranchReference):
        raise TypeError("branch_reference must be ManifoldBranchReference")
    powers = np.asarray(strike_powers_W, dtype=float).ravel()
    if powers.shape != (branch_reference.n_seeds,):
        raise ValueError("strike_powers_W must have one value per branch seed")
    if not np.all(np.isfinite(powers)) or np.any(powers < 0.0):
        raise ValueError("strike_powers_W must be non-negative and finite")
    if float(np.sum(powers)) <= 0.0:
        raise ValueError("strike_powers_W must contain positive total power")
    provenance = str(power_provenance).strip()
    if not provenance:
        raise ValueError("power_provenance must identify the quantitative source")
    maximum_heat = _nonnegative_finite(
        maximum_heat_flux_W_m2,
        "maximum_heat_flux_W_m2",
    )
    maximum_unresolved = _nonnegative_finite(
        maximum_unresolved_power_W,
        "maximum_unresolved_power_W",
    )
    maximum_projection = _nonnegative_finite(
        maximum_projection_distance_m,
        "maximum_projection_distance_m",
    )
    if previous_matches is None:
        previous = None
        if maximum_hit_displacement_m is not None:
            raise ValueError(
                "maximum_hit_displacement_m requires previous_matches"
            )
        maximum_displacement = None
    else:
        if maximum_hit_displacement_m is None:
            raise ValueError(
                "maximum_hit_displacement_m is required with previous_matches"
            )
        maximum_displacement = _nonnegative_finite(
            maximum_hit_displacement_m,
            "maximum_hit_displacement_m",
        )

    bundle = manifold_branch_strike_seed_bundle(
        branch_reference,
        weights=powers,
        weight_kind="power",
        quantitative=True,
    )
    bundle = replace(
        bundle,
        metadata={
            **bundle.metadata,
            "weight_provenance": provenance,
            "power_units": "W",
        },
    )
    if previous_matches is not None:
        previous = _validate_previous_matches(
            previous_matches,
            branch_reference,
            bundle.direction,
        )

    strikes = trace_wall_strikes_field(
        field,
        (bundle,),
        wall,
        max_turns=max_turns,
        DPhi=DPhi,
        extend_phi=extend_phi,
        trace_function=trace_function,
    )[0]
    heat_state = wall_heat_state_from_strikes(
        (strikes,),
        wall,
        phi_edges=phi_edges,
        s_edges=s_edges,
        field_period=field_period,
    )
    peak_heat = float(np.max(np.asarray(heat_state.heat, dtype=float)))
    unresolved_power = float(heat_state.metadata["unresolved_power"])
    all_labels_resolved = bool(
        strikes.seed_index.size == branch_reference.n_seeds
        and np.array_equal(
            strikes.seed_index,
            np.arange(branch_reference.n_seeds, dtype=int),
        )
    )

    matches: tuple[ManifoldStrikeMatch, ...] = ()
    planes: tuple[LocalWallPlane, ...] = ()
    rejection_reason = None
    if unresolved_power > maximum_unresolved:
        rejection_reason = "production_unresolved_power_exceeds_limit"
    elif not all_labels_resolved:
        rejection_reason = "production_strike_labels_unresolved"
    else:
        if previous is None:
            matches = _initial_strike_matches(
                branch_reference,
                bundle,
                strikes,
            )
        else:
            refreshed = []
            for previous_match in previous:
                refresh = refresh_manifold_strike_match(
                    previous_match,
                    bundle,
                    strikes,
                    previous_match.distance_coordinates_m,
                    maximum_hit_displacement_m=maximum_displacement,
                )
                if not refresh.accepted or refresh.candidate_match is None:
                    rejection_reason = (
                        "production_strike_refresh_rejected:"
                        f"{refresh.rejection_reason or 'unknown'}"
                    )
                    break
                refreshed.append(refresh.candidate_match)
            matches = tuple(refreshed)

    if rejection_reason is None:
        projected_planes = []
        for match in matches:
            try:
                plane = local_wall_plane_from_strike(
                    match,
                    wall,
                    maximum_projection_distance_m=maximum_projection,
                )
            except ValueError:
                rejection_reason = "production_strike_wall_projection_rejected"
                break
            projected_planes.append(plane)
        planes = tuple(projected_planes)
    if rejection_reason is None and peak_heat > maximum_heat:
        rejection_reason = "production_peak_heat_flux_exceeds_limit"

    return ManifoldHeatValidationReport(
        branch_label=branch_reference.label,
        seed_orders=branch_reference.seed_orders,
        strike_powers_W=powers,
        power_provenance=provenance,
        seed_bundle=bundle,
        wall_strikes=strikes,
        heat_state=heat_state,
        strike_matches=matches,
        wall_planes=planes,
        peak_heat_flux_W_m2=peak_heat,
        maximum_heat_flux_W_m2=maximum_heat,
        unresolved_power_W=unresolved_power,
        maximum_unresolved_power_W=maximum_unresolved,
        all_labels_resolved=all_labels_resolved,
        accepted=rejection_reason is None,
        rejection_reason=rejection_reason,
    )


__all__ = [
    "ManifoldHeatValidationReport",
    "validate_manifold_heat_load",
]
