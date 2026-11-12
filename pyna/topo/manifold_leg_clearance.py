"""Independent Cyna first-hit and pre-strike clearance validation.

Every exact seed is globally traced to its first wall hit, then its complete
trajectory is sampled by Cyna. Clearance is a per-leg hard requirement; no
power weighting can hide a failing leg. The excluded terminal fraction is
measured in signed, unwrapped toroidal-angle span, not arc length. This is a
sampled certificate whose integration and output spacing must be converged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from pyna.topo.manifold_correspondence import ManifoldBranchReference
from pyna.topo.manifold_strike import (
    local_wall_plane_from_strike,
    manifold_branch_strike_seed_bundle,
    refresh_manifold_strike_match,
)
from pyna.topo.manifold_strike_contracts import ManifoldStrikeLabel, ManifoldStrikeMatch
from pyna.topo.xline_clearance import _signed_clearance_from_projection
from pyna.toroidal.control.strike_heat import trace_wall_strikes_field
from pyna.toroidal.flt import trace_orbit_along_phi_field
from pyna.toroidal.geometry import project_points_to_toroidal_surface


def _nonnegative(value: object, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be non-negative and finite")
    return value


@dataclass(frozen=True)
class ManifoldLegClearanceSampleReport:
    """Retained samples and failure evidence for one exact strike label."""

    label: ManifoldStrikeLabel
    point_RZPhi_m_rad: np.ndarray
    signed_clearance_m: np.ndarray
    trace_complete: bool
    endpoint_error_m: float
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        points = np.asarray(self.point_RZPhi_m_rad, dtype=float).copy()
        clearance = np.asarray(self.signed_clearance_m, dtype=float).copy()
        if (
            points.ndim != 2 or points.shape[1:] != (3,)
            or clearance.shape != (len(points),)
        ):
            raise ValueError("leg points and clearances must be aligned")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(clearance)):
            raise ValueError("stored leg samples must be finite")
        if np.isnan(self.endpoint_error_m) or self.endpoint_error_m < 0:
            raise ValueError("endpoint_error_m must be non-negative or infinity")
        if self.rejection_reason is None and (
            not self.trace_complete or not len(points)
        ):
            raise ValueError("accepted leg requires a complete sampled trajectory")
        points.setflags(write=False)
        clearance.setflags(write=False)
        object.__setattr__(self, "point_RZPhi_m_rad", points)
        object.__setattr__(self, "signed_clearance_m", clearance)

    @property
    def minimum_clearance_m(self) -> float:
        if not self.signed_clearance_m.size:
            return -np.inf
        return float(np.min(self.signed_clearance_m))

    @property
    def accepted(self) -> bool:
        return self.rejection_reason is None


@dataclass(frozen=True)
class ManifoldLegClearanceReport:
    """All required labels and refreshed geometry, with one acceptance result."""

    branch_label: str
    legs: tuple[ManifoldLegClearanceSampleReport, ...]
    strike_matches: tuple[Any, ...]
    wall_planes: tuple[Any, ...]
    required_minimum_clearance_m: float
    terminal_exclusion_fraction: float
    accepted: bool
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        for name in ("legs", "strike_matches", "wall_planes"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        required = _nonnegative(
            self.required_minimum_clearance_m, "required_minimum_clearance_m",
        )
        fraction = float(self.terminal_exclusion_fraction)
        if not np.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError("terminal_exclusion_fraction must lie in (0, 1)")
        object.__setattr__(self, "required_minimum_clearance_m", required)
        object.__setattr__(self, "terminal_exclusion_fraction", fraction)
        if not self.legs or len({leg.label.key for leg in self.legs}) != len(self.legs):
            raise ValueError("leg reports must have non-empty unique labels")
        if any(leg.label.branch_label != self.branch_label for leg in self.legs):
            raise ValueError("leg report belongs to a different branch")
        if self.accepted:
            if self.rejection_reason is not None or not all(
                leg.accepted for leg in self.legs
            ):
                raise ValueError("accepted clearance requires every leg gate")
            if self.minimum_clearance_m < required:
                raise ValueError("accepted clearance violates required minimum")
            if (
                len(self.strike_matches) != len(self.legs)
                or len(self.wall_planes) != len(self.legs)
            ):
                raise ValueError("accepted clearance requires all matches and planes")
            for leg, match, plane in zip(
                self.legs, self.strike_matches, self.wall_planes, strict=True,
            ):
                if leg.label != match.label or leg.label != plane.strike_label:
                    raise ValueError("accepted clearance changed ordered labels")
        elif self.rejection_reason is None:
            raise ValueError("rejected clearance requires rejection_reason")

    @property
    def minimum_clearance_m(self) -> float:
        return min(leg.minimum_clearance_m for leg in self.legs)


def validate_manifold_leg_clearance(
    field: Any,
    branch_reference: ManifoldBranchReference,
    wall: Any,
    *,
    previous_matches: Sequence[ManifoldStrikeMatch],
    required_minimum_clearance_m: float,
    terminal_exclusion_fraction: float,
    maximum_hit_displacement_m: float,
    maximum_hit_phase_shift_rad: float,
    maximum_endpoint_error_m: float,
    maximum_projection_distance_m: float = 1.0e-3,
    max_turns: int = 100,
    DPhi: float = 0.01,
    extend_phi: bool = True,
    strike_trace_function: Callable[..., Any] | None = None,
    trajectory_trace_function: Callable[..., Any] | None = None,
) -> ManifoldLegClearanceReport:
    """Retrace every global first hit and gate its labelled interior leg.

    Output spacing equals DPhi. The full seed-to-hit trajectory must be alive,
    monotone, span-complete, and agree with the independently traced endpoint.
    Unwrapped phase trust prevents replacing a hit with the same Cartesian
    location on a different turn, including for axisymmetric strike metrics.
    """
    if not isinstance(branch_reference, ManifoldBranchReference):
        raise TypeError("branch_reference must be ManifoldBranchReference")
    required = _nonnegative(
        required_minimum_clearance_m, "required_minimum_clearance_m",
    )
    displacement = _nonnegative(
        maximum_hit_displacement_m, "maximum_hit_displacement_m",
    )
    phase_shift = _nonnegative(
        maximum_hit_phase_shift_rad, "maximum_hit_phase_shift_rad",
    )
    endpoint_limit = _nonnegative(
        maximum_endpoint_error_m, "maximum_endpoint_error_m",
    )
    projection_limit = _nonnegative(
        maximum_projection_distance_m, "maximum_projection_distance_m",
    )
    step = _nonnegative(DPhi, "DPhi")
    if step == 0:
        raise ValueError("DPhi must be positive")
    fraction = float(terminal_exclusion_fraction)
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("terminal_exclusion_fraction must lie in (0, 1)")
    if (
        isinstance(max_turns, (bool, np.bool_)) or not np.isfinite(max_turns)
        or int(max_turns) != max_turns or max_turns <= 0
    ):
        raise ValueError("max_turns must be a positive integer")
    tracer = (
        trace_orbit_along_phi_field if trajectory_trace_function is None
        else trajectory_trace_function
    )
    if not callable(tracer):
        raise TypeError("trajectory_trace_function must be callable")
    bundle = manifold_branch_strike_seed_bundle(branch_reference)
    previous = tuple(previous_matches)
    if len(previous) != branch_reference.n_seeds:
        raise ValueError("previous_matches must contain every branch seed")
    for index, match in enumerate(previous):
        expected = ManifoldStrikeLabel(
            branch_reference.label, bundle.direction,
            int(branch_reference.seed_orders[index]),
        )
        if not isinstance(match, ManifoldStrikeMatch) or match.label != expected:
            raise ValueError("previous_matches must preserve ordered branch identity")
    strikes = trace_wall_strikes_field(
        field, (bundle,), wall, max_turns=max_turns, DPhi=step,
        extend_phi=extend_phi, trace_function=strike_trace_function,
    )[0]
    legs, matches, planes = [], [], []
    for index, old in enumerate(previous):
        points, clearance = np.empty((0, 3)), np.empty(0)
        complete, endpoint_error, reason = False, np.inf, None
        refresh = refresh_manifold_strike_match(
            old, bundle, strikes, old.distance_coordinates_m,
            maximum_hit_displacement_m=displacement,
        )
        match = refresh.candidate_match
        if not refresh.accepted or match is None:
            reason = f"production_leg_strike_refresh:{refresh.rejection_reason}"
        elif abs(match.point_RZPhi_m_rad[2] - old.point_RZPhi_m_rad[2]) > phase_shift:
            reason = "production_leg_hit_phase_shift_exceeds_limit"
        else:
            start, end = float(bundle.phi[index]), float(match.point_RZPhi_m_rad[2])
            sign = 1 if bundle.direction == "+" else -1
            if sign * (end - start) <= 0:
                reason = "production_leg_hit_direction_invalid"
            else:
                R, Z, phi, _monodromy, alive = tracer(
                    field, float(bundle.R[index]), float(bundle.Z[index]), start,
                    end, step, extend_phi=extend_phi, dphi_out=step,
                )
                R, Z, phi = (np.asarray(a, dtype=float).ravel() for a in (R, Z, phi))
                alive = np.asarray(alive, dtype=bool).ravel()
                if not (R.size == Z.size == phi.size == alive.size):
                    raise ValueError("production leg trajectory arrays must align")
                full = np.column_stack((R, Z, phi))
                phase_tol = max(1e-12, 1e-8 * max(1, abs(end)))
                complete = bool(
                    len(phi) >= 2 and np.all(alive) and np.all(np.isfinite(full))
                    and np.all(R > 0) and np.all(sign * np.diff(phi) > 0)
                    and abs(phi[0] - start) <= phase_tol
                    and abs(phi[-1] - end) <= phase_tol
                )
                if not complete:
                    reason = "production_leg_trace_incomplete"
                else:
                    endpoint_error = float(np.linalg.norm(
                        full[-1, :2] - match.point_RZPhi_m_rad[:2]
                    ))
                    launch_error = float(np.linalg.norm(
                        full[0, :2] - [bundle.R[index], bundle.Z[index]]
                    ))
                    if launch_error > 1e-10:
                        reason = "production_leg_launch_mismatch"
                    elif endpoint_error > endpoint_limit:
                        reason = "production_leg_endpoint_mismatch"
                    else:
                        retained = (phi - start) / (end - start) < 1 - fraction
                        points = full[retained]
                        projection = project_points_to_toroidal_surface(
                            points[:, 0], points[:, 1], points[:, 2], wall,
                        )
                        clearance = _signed_clearance_from_projection(
                            points, projection,
                        )
                        if not np.all(np.isfinite(clearance)):
                            points, clearance = np.empty((0, 3)), np.empty(0)
                            reason = "production_leg_wall_projection_rejected"
                        elif float(np.min(clearance)) < required:
                            reason = "production_leg_clearance_below_limit"
            if reason is None:
                try:
                    plane = local_wall_plane_from_strike(
                        match, wall,
                        maximum_projection_distance_m=projection_limit,
                    )
                except ValueError:
                    reason = "production_leg_wall_projection_rejected"
                else:
                    matches.append(match)
                    planes.append(plane)
        legs.append(ManifoldLegClearanceSampleReport(
            old.label, points, clearance, complete, endpoint_error, reason,
        ))
    reason = next((leg.rejection_reason for leg in legs if not leg.accepted), None)
    return ManifoldLegClearanceReport(
        branch_reference.label, tuple(legs), tuple(matches), tuple(planes),
        required, fraction, reason is None, reason,
    )


__all__ = [
    "ManifoldLegClearanceSampleReport", "ManifoldLegClearanceReport",
    "validate_manifold_leg_clearance",
]
