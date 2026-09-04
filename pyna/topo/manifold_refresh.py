"""Production refresh of a labelled invariant-manifold branch.

This module is deliberately NumPy/Cyna-facing.  It continues one accepted
section X-point on a candidate production field, retraces the same labelled
seed segment, and rebuilds the dense correspondence consumed by a parallel JAX
inner loop.  It has no dependency on ESSOS or JAX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from pyna.topo.manifold_correspondence import (
    ManifoldBranchReference,
    manifold_branch_reference_from_trace,
)


@dataclass(frozen=True)
class ManifoldProductionRefreshReport:
    """Outcome of continuing and retracing one production branch."""

    previous_branch_label: str
    candidate_fixed_point: Any | None
    production_payload: Mapping[str, object] | None
    candidate_branch: ManifoldBranchReference | None
    anchor_displacement_m: float
    maximum_anchor_displacement_m: float
    accepted: bool
    rejection_reason: str | None = None
    diagnostics: Mapping[str, object] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        label = str(self.previous_branch_label).strip()
        if not label:
            raise ValueError("previous_branch_label must not be empty")
        displacement = float(self.anchor_displacement_m)
        maximum = float(self.maximum_anchor_displacement_m)
        if np.isnan(displacement) or displacement < 0.0:
            raise ValueError("anchor_displacement_m must be non-negative or infinity")
        if not np.isfinite(maximum) or maximum < 0.0:
            raise ValueError(
                "maximum_anchor_displacement_m must be non-negative and finite"
            )
        reason = None if self.rejection_reason is None else str(self.rejection_reason)
        accepted = bool(self.accepted)
        if accepted:
            if (
                self.candidate_fixed_point is None
                or self.production_payload is None
                or self.candidate_branch is None
                or reason is not None
            ):
                raise ValueError(
                    "an accepted production refresh requires complete candidate data"
                )
            if displacement > maximum:
                raise ValueError("accepted production refresh exceeds its anchor limit")
            if self.candidate_branch.label != label:
                raise ValueError("accepted production refresh changed branch identity")
        elif reason is None:
            raise ValueError("a rejected production refresh requires rejection_reason")

        payload = (
            None
            if self.production_payload is None
            else dict(self.production_payload)
        )
        object.__setattr__(self, "previous_branch_label", label)
        object.__setattr__(self, "production_payload", payload)
        object.__setattr__(self, "anchor_displacement_m", displacement)
        object.__setattr__(self, "maximum_anchor_displacement_m", maximum)
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "rejection_reason", reason)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics or {}))


def _nonnegative_finite(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def _nonzero_finite(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or abs(result) <= 1.0e-14:
        raise ValueError(f"{name} must be finite and non-zero")
    return result


def _source_monodromy(branch: ManifoldBranchReference) -> np.ndarray:
    """Construct a hyperbolic surrogate used only as the Newton seed record."""

    direction = np.asarray(branch.direction_RZ, dtype=float)
    transverse = np.asarray([-direction[1], direction[0]], dtype=float)
    basis = np.column_stack((direction, transverse))
    sign = -1.0 if branch.orientation_reversing else 1.0
    if branch.stability == "unstable":
        eigenvalues = np.asarray(
            [sign * branch.expansion, sign / branch.expansion]
        )
    else:
        eigenvalues = np.asarray(
            [sign / branch.expansion, sign * branch.expansion]
        )
    return basis @ np.diag(eigenvalues) @ basis.T


def _source_fixed_point(branch: ManifoldBranchReference):
    from pyna.toroidal.flt import BoundaryIslandFixedPoint

    metadata = dict(branch.metadata)
    metadata.update(
        {
            "map_power": 1,
            "field_period": float(branch.map_span),
            "map_span": float(branch.map_span),
            "base_map_span": float(branch.map_span),
            "monodromy_field_period": float(branch.map_span),
            "monodromy_map_span": float(branch.map_span),
            "manifold_origin_label": branch.origin_label,
        }
    )
    monodromy = _source_monodromy(branch)
    anchor = np.asarray(branch.anchor_RZ_m, dtype=float)
    return BoundaryIslandFixedPoint(
        phi=float(branch.section_phi),
        R=float(anchor[0]),
        Z=float(anchor[1]),
        map_power=1,
        kind="X",
        DPm=monodromy,
        residual=0.0,
        eigenvalues=np.linalg.eigvals(monodromy),
        seed_R=float(anchor[0]),
        seed_Z=float(anchor[1]),
        map_span=float(branch.map_span),
        chain_id=metadata.get("chain_id"),
        orbit_id=metadata.get("orbit_id"),
        point_index=metadata.get("point_index"),
        metadata=metadata,
    )


def _rejected_report(
    previous_branch: ManifoldBranchReference,
    maximum_anchor_displacement_m: float,
    reason: str,
    *,
    candidate_fixed_point: Any | None = None,
    production_payload: Mapping[str, object] | None = None,
    anchor_displacement_m: float = np.inf,
    diagnostics: Mapping[str, object] | None = None,
) -> ManifoldProductionRefreshReport:
    return ManifoldProductionRefreshReport(
        previous_branch_label=previous_branch.label,
        candidate_fixed_point=candidate_fixed_point,
        production_payload=production_payload,
        candidate_branch=None,
        anchor_displacement_m=anchor_displacement_m,
        maximum_anchor_displacement_m=maximum_anchor_displacement_m,
        accepted=False,
        rejection_reason=reason,
        diagnostics={} if diagnostics is None else diagnostics,
    )


def refresh_manifold_branch_reference_field(
    field: Any,
    previous_branch: ManifoldBranchReference,
    *,
    maximum_anchor_displacement_m: float,
    minimum_direction_alignment: float,
    DPhi: float = 0.01,
    fd_eps: float = 1.0e-4,
    fixed_point_max_iter: int = 20,
    fixed_point_tolerance: float = 1.0e-10,
    fixed_point_residual_tolerance: float | None = None,
    seed_geometry_tolerance_m: float = 1.0e-10,
    wall_R: Sequence[float] | None = None,
    wall_Z: Sequence[float] | None = None,
    wall_phi: Sequence[float] | None = None,
    wall_R_all: np.ndarray | None = None,
    wall_Z_all: np.ndarray | None = None,
    RZlimit: tuple[float, float, float, float] | None = None,
    refine_stable_inverse_anchor: bool | None = None,
    extend_phi: bool = True,
    n_threads: int = -1,
) -> ManifoldProductionRefreshReport:
    """Continue and retrace one exact production branch on ``field``.

    Backend absence raises ``ImportError`` because it is an environment error.
    Loss of the fixed point, hyperbolicity, trust-region identity, labelled
    seed segment, or tangent correspondence returns a rejected report.
    """

    if not isinstance(previous_branch, ManifoldBranchReference):
        raise TypeError("previous_branch must be ManifoldBranchReference")
    maximum_displacement = _nonnegative_finite(
        maximum_anchor_displacement_m,
        "maximum_anchor_displacement_m",
    )
    alignment = float(minimum_direction_alignment)
    if not np.isfinite(alignment) or not 0.0 <= alignment <= 1.0:
        raise ValueError("minimum_direction_alignment must lie in [0, 1]")
    dphi = _nonzero_finite(DPhi, "DPhi")

    from pyna.toroidal.flt import (
        refine_fixed_points_monodromy_span_field,
        trace_fixed_point_manifolds_field,
    )

    source = _source_fixed_point(previous_branch)
    refined = refine_fixed_points_monodromy_span_field(
        field,
        [source],
        field_period=float(previous_branch.map_span),
        map_power=1,
        DPhi=dphi,
        fd_eps=float(fd_eps),
        max_iter=int(fixed_point_max_iter),
        tol=float(fixed_point_tolerance),
        residual_tol=fixed_point_residual_tolerance,
        keep_unconverged=False,
        extend_phi=bool(extend_phi),
        n_threads=int(n_threads),
    )
    if not refined:
        return _rejected_report(
            previous_branch,
            maximum_displacement,
            "fixed_point_continuation_failed",
        )

    fixed_point = refined[0]
    if str(getattr(fixed_point, "kind", "")).upper() != "X":
        return _rejected_report(
            previous_branch,
            maximum_displacement,
            "fixed_point_lost_hyperbolicity",
            candidate_fixed_point=fixed_point,
        )
    candidate_anchor = np.asarray(
        [getattr(fixed_point, "R"), getattr(fixed_point, "Z")],
        dtype=float,
    )
    displacement = float(
        np.linalg.norm(candidate_anchor - previous_branch.anchor_RZ_m)
    )
    if not np.isfinite(displacement):
        return _rejected_report(
            previous_branch,
            maximum_displacement,
            "fixed_point_nonfinite",
            candidate_fixed_point=fixed_point,
        )
    if displacement > maximum_displacement:
        return _rejected_report(
            previous_branch,
            maximum_displacement,
            "anchor_displacement_exceeds_limit",
            candidate_fixed_point=fixed_point,
            anchor_displacement_m=displacement,
        )

    refine_inverse = (
        previous_branch.stability == "stable"
        if refine_stable_inverse_anchor is None
        else bool(refine_stable_inverse_anchor)
    )
    try:
        payloads = trace_fixed_point_manifolds_field(
            field,
            [fixed_point],
            phi_section=float(previous_branch.section_phi),
            N_turns=previous_branch.n_generations,
            map_span=float(previous_branch.map_span),
            DPhi=dphi,
            fd_eps=float(fd_eps),
            seed_distances=previous_branch.seed_distances_m,
            seed_orders=previous_branch.seed_orders,
            wall_R=wall_R,
            wall_Z=wall_Z,
            wall_phi=wall_phi,
            wall_R_all=wall_R_all,
            wall_Z_all=wall_Z_all,
            RZlimit=RZlimit,
            extend_phi=bool(extend_phi),
            refine_stable_inverse_anchor=refine_inverse,
        )
    except (RuntimeError, ValueError) as exc:
        return _rejected_report(
            previous_branch,
            maximum_displacement,
            "production_trace_failed",
            candidate_fixed_point=fixed_point,
            anchor_displacement_m=displacement,
            diagnostics={"error": str(exc)},
        )
    if not payloads:
        return _rejected_report(
            previous_branch,
            maximum_displacement,
            "production_trace_failed",
            candidate_fixed_point=fixed_point,
            anchor_displacement_m=displacement,
            diagnostics={"error": "production tracer returned no X-point payload"},
        )

    payload = dict(payloads[0])
    payload["manifold_origin_label"] = previous_branch.origin_label
    try:
        candidate_branch = manifold_branch_reference_from_trace(
            payload,
            stability=previous_branch.stability,
            seed_side=previous_branch.seed_side,
            n_generations=previous_branch.n_generations,
            direction_reference=previous_branch.direction_RZ,
            minimum_direction_alignment=alignment,
            seed_geometry_tolerance_m=float(seed_geometry_tolerance_m),
        )
    except ValueError as exc:
        return _rejected_report(
            previous_branch,
            maximum_displacement,
            "branch_correspondence_failed",
            candidate_fixed_point=fixed_point,
            production_payload=payload,
            anchor_displacement_m=displacement,
            diagnostics={"error": str(exc)},
        )

    branch_displacement = float(
        np.linalg.norm(
            candidate_branch.anchor_RZ_m - previous_branch.anchor_RZ_m
        )
    )
    if not np.isfinite(branch_displacement):
        return _rejected_report(
            previous_branch,
            maximum_displacement,
            "branch_anchor_nonfinite",
            candidate_fixed_point=fixed_point,
            production_payload=payload,
            diagnostics={"fixed_point_anchor_displacement_m": displacement},
        )
    if branch_displacement > maximum_displacement:
        return _rejected_report(
            previous_branch,
            maximum_displacement,
            "anchor_displacement_exceeds_limit",
            candidate_fixed_point=fixed_point,
            production_payload=payload,
            anchor_displacement_m=branch_displacement,
            diagnostics={"fixed_point_anchor_displacement_m": displacement},
        )

    return ManifoldProductionRefreshReport(
        previous_branch_label=previous_branch.label,
        candidate_fixed_point=fixed_point,
        production_payload=payload,
        candidate_branch=candidate_branch,
        anchor_displacement_m=branch_displacement,
        maximum_anchor_displacement_m=maximum_displacement,
        accepted=True,
        diagnostics={
            "fixed_point_residual": float(getattr(fixed_point, "residual", np.nan)),
            "fixed_point_anchor_displacement_m": displacement,
            "direction_alignment": candidate_branch.metadata.get(
                "direction_alignment"
            ),
            "branch_complete": candidate_branch.complete,
        },
    )


__all__ = [
    "ManifoldProductionRefreshReport",
    "refresh_manifold_branch_reference_field",
]
