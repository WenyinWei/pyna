"""Outer correspondence gate for differentiable and production wall strikes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pyna.topo.manifold_strike_contracts import (
    ManifoldStrikeLabel,
    ManifoldStrikeMatch,
    strike_distance_coordinates,
)


@dataclass(frozen=True)
class ManifoldStrikeCorrespondenceReport:
    """JAX/Cyna parity and event diagnostics for one exact strike label."""

    label: ManifoldStrikeLabel
    distance_mode: Literal["xyz", "rz"]
    production_coordinates_m: np.ndarray
    jax_coordinates_m: np.ndarray
    finite: bool
    periodic_point_converged: bool
    wall_intersection_converged: bool
    within_event_window: bool
    wall_residual: float
    transversality: float
    phi_shift_from_guess: float
    deviation_m: float
    absolute_tolerance_m: float
    accepted: bool
    rejection_reason: str | None = None


def _scalar(value: object, name: str) -> float:
    array = np.asarray(value)
    if array.shape != ():
        raise TypeError(f"{name} must be a scalar")
    return float(array)


def _boolean(value: object, name: str) -> bool:
    array = np.asarray(value)
    if array.shape != ():
        raise TypeError(f"{name} must be a scalar")
    return bool(array)


def compare_jax_manifold_strike(
    production_match: ManifoldStrikeMatch,
    jax_strike_state: Any,
    *,
    absolute_tolerance_m: float,
) -> ManifoldStrikeCorrespondenceReport:
    """Compare one differentiable local-plane hit with its exact Cyna hit.

    The comparison uses the metric selected when the production strike was
    labelled: symmetry-reduced ``(R,Z)`` for an axisymmetric tokamak or full
    Cartesian ``(X,Y,Z)`` for a non-axisymmetric configuration.  A close point
    is still rejected when either the periodic root or implicit wall event did
    not converge, or when Newton reached the edge of its frozen event window.
    """

    if not isinstance(production_match, ManifoldStrikeMatch):
        raise TypeError("production_match must be ManifoldStrikeMatch")
    tolerance = float(absolute_tolerance_m)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("absolute_tolerance_m must be non-negative and finite")
    try:
        periodic = jax_strike_state.periodic_point
        intersection = jax_strike_state.intersection
        point_RZPhi = np.asarray(intersection.point_RZPhi, dtype=float).ravel()
        periodic_converged = _boolean(
            periodic.converged,
            "periodic_point.converged",
        )
        intersection_converged = _boolean(
            intersection.converged,
            "intersection.converged",
        )
        within_window = _boolean(
            intersection.within_window,
            "intersection.within_window",
        )
        wall_residual = _scalar(
            intersection.wall_residual,
            "intersection.wall_residual",
        )
        transversality = _scalar(
            intersection.transversality,
            "intersection.transversality",
        )
        phi_shift = _scalar(
            intersection.phi_shift_from_guess,
            "intersection.phi_shift_from_guess",
        )
    except AttributeError as exc:
        raise TypeError(
            "jax_strike_state must provide periodic-point and wall-intersection "
            "diagnostics"
        ) from exc
    if point_RZPhi.shape != (3,):
        raise TypeError("intersection.point_RZPhi must have shape (3,)")

    diagnostics = np.asarray(
        [wall_residual, transversality, phi_shift],
        dtype=float,
    )
    finite = bool(np.all(np.isfinite(point_RZPhi)) and np.all(np.isfinite(diagnostics)))
    production_coordinates = production_match.distance_coordinates_m
    if finite:
        jax_coordinates = strike_distance_coordinates(
            point_RZPhi,
            production_match.distance_mode,
        )
        deviation = float(np.linalg.norm(jax_coordinates - production_coordinates))
    else:
        jax_coordinates = np.full(production_coordinates.shape, np.nan)
        deviation = np.inf

    if not periodic_converged:
        reason = "periodic_point_not_converged"
    elif not finite:
        reason = "jax_strike_nonfinite"
    elif not within_window:
        reason = "wall_intersection_left_event_window"
    elif not intersection_converged:
        reason = "wall_intersection_not_converged"
    elif deviation > tolerance:
        reason = "strike_deviation_exceeds_tolerance"
    else:
        reason = None

    return ManifoldStrikeCorrespondenceReport(
        label=production_match.label,
        distance_mode=production_match.distance_mode,
        production_coordinates_m=production_coordinates,
        jax_coordinates_m=jax_coordinates,
        finite=finite,
        periodic_point_converged=periodic_converged,
        wall_intersection_converged=intersection_converged,
        within_event_window=within_window,
        wall_residual=wall_residual,
        transversality=transversality,
        phi_shift_from_guess=phi_shift,
        deviation_m=deviation,
        absolute_tolerance_m=tolerance,
        accepted=reason is None,
        rejection_reason=reason,
    )


__all__ = [
    "ManifoldStrikeCorrespondenceReport",
    "compare_jax_manifold_strike",
]
