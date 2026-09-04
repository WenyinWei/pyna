"""Production wall-clearance validation for a labelled periodic X-line.

The differentiable JAX trajectory is an inner optimization model.  This module
keeps the accepted-design decision on PyNA's production path: Cyna traces the
periodic orbit through a cylindrical grid field, and PyNA measures its signed
clearance from the continuously interpolated toroidal wall.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from pyna.topo.manifold_correspondence import ManifoldBranchReference
from pyna.toroidal.flt import trace_orbit_along_phi_field
from pyna.toroidal.geometry import project_points_to_toroidal_surface


@dataclass(frozen=True)
class PeriodicXLineClearanceReport:
    """Cyna trajectory, wall clearances, and one explicit acceptance result."""

    branch_label: str
    point_RZPhi_m_rad: np.ndarray
    signed_clearance_m: np.ndarray
    minimum_clearance_m: float
    required_minimum_clearance_m: float
    closure_error_m: float
    maximum_closure_error_m: float
    trace_complete: bool
    accepted: bool
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        label = str(self.branch_label).strip()
        points = np.asarray(self.point_RZPhi_m_rad, dtype=float)
        clearance = np.asarray(self.signed_clearance_m, dtype=float).ravel()
        if not label:
            raise ValueError("branch_label must not be empty")
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("point_RZPhi_m_rad must have shape (n_samples, 3)")
        if clearance.shape != (points.shape[0],):
            raise ValueError("signed_clearance_m must align with trajectory samples")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(clearance)):
            raise ValueError("stored trajectory samples and clearances must be finite")

        minimum = float(self.minimum_clearance_m)
        required = float(self.required_minimum_clearance_m)
        closure = float(self.closure_error_m)
        maximum_closure = float(self.maximum_closure_error_m)
        if np.isnan(minimum):
            raise ValueError("minimum_clearance_m must not be NaN")
        if not np.isfinite(required) or required < 0.0:
            raise ValueError(
                "required_minimum_clearance_m must be non-negative and finite"
            )
        if np.isnan(closure) or closure < 0.0:
            raise ValueError("closure_error_m must be non-negative or infinity")
        if not np.isfinite(maximum_closure) or maximum_closure < 0.0:
            raise ValueError(
                "maximum_closure_error_m must be non-negative and finite"
            )
        expected_minimum = float(np.min(clearance)) if clearance.size else -np.inf
        if minimum != expected_minimum:
            raise ValueError("minimum_clearance_m disagrees with stored samples")

        complete = bool(self.trace_complete)
        accepted = bool(self.accepted)
        reason = None if self.rejection_reason is None else str(
            self.rejection_reason
        )
        if accepted:
            if not complete or reason is not None:
                raise ValueError("an accepted report requires a complete trace")
            if minimum < required or closure > maximum_closure:
                raise ValueError("accepted report violates a clearance or closure gate")
        elif reason is None:
            raise ValueError("a rejected report requires rejection_reason")

        object.__setattr__(self, "branch_label", label)
        object.__setattr__(self, "point_RZPhi_m_rad", points.copy())
        object.__setattr__(self, "signed_clearance_m", clearance.copy())
        object.__setattr__(self, "minimum_clearance_m", minimum)
        object.__setattr__(self, "required_minimum_clearance_m", required)
        object.__setattr__(self, "closure_error_m", closure)
        object.__setattr__(self, "maximum_closure_error_m", maximum_closure)
        object.__setattr__(self, "trace_complete", complete)
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "rejection_reason", reason)

    @property
    def minimum_sample_index(self) -> int | None:
        """Index of the limiting stored sample, or ``None`` for an empty trace."""

        if self.signed_clearance_m.size == 0:
            return None
        return int(np.argmin(self.signed_clearance_m))


def _nonnegative_finite(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def _signed_clearance_from_projection(
    point_RZPhi_m_rad: np.ndarray,
    projection: Any,
) -> np.ndarray:
    radius = point_RZPhi_m_rad[:, 0]
    height = point_RZPhi_m_rad[:, 1]
    phi = point_RZPhi_m_rad[:, 2]
    point_xyz = np.column_stack(
        (radius * np.cos(phi), radius * np.sin(phi), height)
    )
    wrapped_xyz = np.asarray(projection.xyz, dtype=float)
    wrapped_normal = np.asarray(projection.normal, dtype=float)
    rotation_angle = phi - np.asarray(projection.phi, dtype=float)
    cos_rotation = np.cos(rotation_angle)
    sin_rotation = np.sin(rotation_angle)
    projected_xyz = wrapped_xyz.copy()
    projected_xyz[:, 0] = (
        cos_rotation * wrapped_xyz[:, 0]
        - sin_rotation * wrapped_xyz[:, 1]
    )
    projected_xyz[:, 1] = (
        sin_rotation * wrapped_xyz[:, 0]
        + cos_rotation * wrapped_xyz[:, 1]
    )
    normal = wrapped_normal.copy()
    normal[:, 0] = (
        cos_rotation * wrapped_normal[:, 0]
        - sin_rotation * wrapped_normal[:, 1]
    )
    normal[:, 1] = (
        sin_rotation * wrapped_normal[:, 0]
        + cos_rotation * wrapped_normal[:, 1]
    )
    outward_offset = np.einsum(
        "ij,ij->i",
        point_xyz - projected_xyz,
        normal,
    )
    distance = np.asarray(projection.distance, dtype=float)
    return np.where(outward_offset <= 0.0, distance, -distance)


def validate_periodic_xline_clearance(
    field: Any,
    branch_reference: ManifoldBranchReference,
    wall: Any,
    *,
    DPhi: float,
    required_minimum_clearance_m: float,
    maximum_closure_error_m: float,
    extend_phi: bool = True,
    trace_function: Callable[..., Any] | None = None,
) -> PeriodicXLineClearanceReport:
    """Validate one complete production X-line orbit against a toroidal wall.

    Clearance is positive on the inward side of the oriented wall and negative
    outside it.  Periodic wall projections are rotated back to each sample's
    signed, unwrapped toroidal phase before assigning that sign.  The trajectory
    output interval equals ``DPhi``; callers must converge this sampling just as
    they converge the production field grid.
    """

    if not isinstance(branch_reference, ManifoldBranchReference):
        raise TypeError("branch_reference must be a ManifoldBranchReference")
    step = float(DPhi)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("DPhi must be positive and finite")
    required = _nonnegative_finite(
        required_minimum_clearance_m,
        "required_minimum_clearance_m",
    )
    maximum_closure = _nonnegative_finite(
        maximum_closure_error_m,
        "maximum_closure_error_m",
    )
    tracer = trace_orbit_along_phi_field if trace_function is None else trace_function
    if not callable(tracer):
        raise TypeError("trace_function must be callable")

    phi_start = float(branch_reference.section_phi)
    phi_end = phi_start + float(branch_reference.map_span)
    R, Z, phi, _DPm, alive = tracer(
        field,
        float(branch_reference.anchor_RZ_m[0]),
        float(branch_reference.anchor_RZ_m[1]),
        phi_start,
        phi_end,
        step,
        extend_phi=bool(extend_phi),
        dphi_out=step,
    )
    R = np.asarray(R, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float).ravel()
    phi = np.asarray(phi, dtype=float).ravel()
    alive = np.asarray(alive, dtype=bool).ravel()
    if not (R.size == Z.size == phi.size == alive.size):
        raise ValueError("production trajectory arrays must have matching lengths")

    finite = np.isfinite(R) & np.isfinite(Z) & np.isfinite(phi)
    valid = alive & finite & (R > 0.0)
    phase_tolerance = max(1.0e-12, 1.0e-8 * max(1.0, abs(phi_end)))
    covers_span = bool(
        phi.size >= 2
        and abs(phi[0] - phi_start) <= phase_tolerance
        and abs(phi[-1] - phi_end) <= phase_tolerance
    )
    trace_complete = bool(valid.size >= 2 and np.all(valid) and covers_span)
    points = np.column_stack((R[valid], Z[valid], phi[valid]))

    if points.size:
        projection = project_points_to_toroidal_surface(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            wall,
        )
        signed_clearance = _signed_clearance_from_projection(points, projection)
        minimum_clearance = float(np.min(signed_clearance))
    else:
        signed_clearance = np.empty(0, dtype=float)
        minimum_clearance = -np.inf

    closure_error = (
        float(np.linalg.norm(points[-1, :2] - branch_reference.anchor_RZ_m))
        if trace_complete
        else np.inf
    )
    if not trace_complete:
        rejection_reason = "production_orbit_trace_incomplete"
    elif closure_error > maximum_closure:
        rejection_reason = "production_orbit_closure_exceeds_limit"
    elif minimum_clearance < required:
        rejection_reason = "production_xline_clearance_below_limit"
    else:
        rejection_reason = None

    return PeriodicXLineClearanceReport(
        branch_label=branch_reference.label,
        point_RZPhi_m_rad=points,
        signed_clearance_m=signed_clearance,
        minimum_clearance_m=minimum_clearance,
        required_minimum_clearance_m=required,
        closure_error_m=closure_error,
        maximum_closure_error_m=maximum_closure,
        trace_complete=trace_complete,
        accepted=rejection_reason is None,
        rejection_reason=rejection_reason,
    )


__all__ = [
    "PeriodicXLineClearanceReport",
    "validate_periodic_xline_clearance",
]
