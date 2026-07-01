"""Adaptive Poincare seed densification utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from pyna.toroidal.flt.island_chain import (
    PoincareSectionTraces,
    _ray_polygon_radius,
    boundary_wall_fractions,
    trace_poincare_sections_from_same_orbits_field,
)


@dataclass(frozen=True)
class BoundarySeedDensificationResult:
    """Seed update proposed from a Poincare point-density diagnostic."""

    seed_R: np.ndarray
    seed_Z: np.ndarray
    added_R: np.ndarray
    added_Z: np.ndarray
    diagnostics: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        seed_R = np.asarray(self.seed_R, dtype=float).ravel()
        seed_Z = np.asarray(self.seed_Z, dtype=float).ravel()
        added_R = np.asarray(self.added_R, dtype=float).ravel()
        added_Z = np.asarray(self.added_Z, dtype=float).ravel()
        if seed_R.size != seed_Z.size:
            raise ValueError("seed_R and seed_Z must have the same length")
        if added_R.size != added_Z.size:
            raise ValueError("added_R and added_Z must have the same length")
        object.__setattr__(self, "seed_R", seed_R)
        object.__setattr__(self, "seed_Z", seed_Z)
        object.__setattr__(self, "added_R", added_R)
        object.__setattr__(self, "added_Z", added_Z)


@dataclass(frozen=True)
class AdaptivePoincareSectionTraces:
    """Aggregate Poincare traces from adaptive seed-refinement rounds."""

    traces: tuple[PoincareSectionTraces, ...]
    metadata: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        traces = tuple(self.traces)
        if not traces:
            raise ValueError("traces must not be empty")
        phi0 = np.asarray(traces[0].phi_sections, dtype=float)
        for trace in traces[1:]:
            if trace.n_section != traces[0].n_section:
                raise ValueError("all traces must use the same section count")
            if not np.allclose(trace.phi_sections, phi0):
                raise ValueError("all traces must use the same section phis")
        object.__setattr__(self, "traces", traces)

    @property
    def phi_sections(self) -> np.ndarray:
        return self.traces[0].phi_sections.copy()

    @property
    def seed_R(self) -> np.ndarray:
        return np.concatenate([trace.seed_R for trace in self.traces])

    @property
    def seed_Z(self) -> np.ndarray:
        return np.concatenate([trace.seed_Z for trace in self.traces])

    @property
    def n_seed(self) -> int:
        return int(sum(trace.n_seed for trace in self.traces))

    @property
    def n_section(self) -> int:
        return int(self.traces[0].n_section)

    @property
    def N_turns(self) -> int:
        return int(max(trace.N_turns for trace in self.traces))

    @property
    def direction(self) -> str:
        return str(self.traces[0].direction)

    def section_points(self, section: int | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        R_parts: list[np.ndarray] = []
        Z_parts: list[np.ndarray] = []
        seed_parts: list[np.ndarray] = []
        seed_offset = 0
        for trace in self.traces:
            R, Z, seed = trace.section_points(section)
            if R.size:
                R_parts.append(R)
                Z_parts.append(Z)
                seed_parts.append(seed + int(seed_offset))
            seed_offset += int(trace.n_seed)
        if not R_parts:
            return np.empty(0, dtype=float), np.empty(0, dtype=float), np.empty(0, dtype=int)
        return np.concatenate(R_parts), np.concatenate(Z_parts), np.concatenate(seed_parts)


def _seed_counts_for_trace(trace: PoincareSectionTraces) -> np.ndarray:
    return np.sum(np.asarray(trace.counts, dtype=int), axis=1)


def poincare_seed_hit_counts(traces) -> np.ndarray:
    """Return total recorded Poincare hits for every seed orbit."""

    if isinstance(traces, AdaptivePoincareSectionTraces):
        if not traces.traces:
            return np.empty(0, dtype=int)
        return np.concatenate([_seed_counts_for_trace(trace) for trace in traces.traces])
    return _seed_counts_for_trace(traces)


def _filter_single_trace_by_mask(trace: PoincareSectionTraces, mask: np.ndarray) -> PoincareSectionTraces:
    mask = np.asarray(mask, dtype=bool).ravel()
    if mask.size != trace.n_seed:
        raise ValueError("seed mask length must match trace.n_seed")
    R_cube = np.asarray(trace.R_flat, dtype=float).reshape(trace.n_seed, trace.n_section, trace.N_turns)
    Z_cube = np.asarray(trace.Z_flat, dtype=float).reshape(trace.n_seed, trace.n_section, trace.N_turns)
    return PoincareSectionTraces(
        phi_sections=trace.phi_sections,
        seed_R=trace.seed_R[mask],
        seed_Z=trace.seed_Z[mask],
        counts=trace.counts[mask],
        R_flat=R_cube[mask].ravel(),
        Z_flat=Z_cube[mask].ravel(),
        N_turns=trace.N_turns,
        direction=trace.direction,
        metadata={**dict(trace.metadata), "seed_filter_kept_count": int(np.count_nonzero(mask))},
    )


def _seed_filter_mask(
    trace: PoincareSectionTraces,
    *,
    min_total_count: int | None,
    min_count_per_section: int | None,
) -> np.ndarray:
    counts = np.asarray(trace.counts, dtype=int)
    mask = np.ones(trace.n_seed, dtype=bool)
    if min_total_count is not None:
        mask &= np.sum(counts, axis=1) >= int(min_total_count)
    if min_count_per_section is not None:
        mask &= np.min(counts, axis=1) >= int(min_count_per_section)
    return mask


def _survival_filter_metadata(original_counts: np.ndarray, kept_counts: np.ndarray, *, params: dict) -> dict:
    def _stats(values: np.ndarray) -> dict:
        arr = np.asarray(values, dtype=float).ravel()
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return {"count": 0, "min": None, "median": None, "p90": None, "max": None}
        return {
            "count": int(finite.size),
            "min": float(np.min(finite)),
            "median": float(np.median(finite)),
            "p90": float(np.percentile(finite, 90.0)),
            "max": float(np.max(finite)),
        }

    return {
        "filter": "seed_hit_count",
        "params": params,
        "original_seed_count": int(np.asarray(original_counts).size),
        "kept_seed_count": int(np.asarray(kept_counts).size),
        "dropped_seed_count": int(np.asarray(original_counts).size - np.asarray(kept_counts).size),
        "original_total_hit_stats": _stats(original_counts),
        "kept_total_hit_stats": _stats(kept_counts),
    }


def filter_poincare_traces_by_seed_count(
    traces,
    *,
    min_total_count: int | None = None,
    min_count_per_section: int | None = None,
):
    """Drop seed orbits that leave too few Poincare hits for clear plotting."""

    if min_total_count is None and min_count_per_section is None:
        return traces
    params = {
        "min_total_count": None if min_total_count is None else int(min_total_count),
        "min_count_per_section": None if min_count_per_section is None else int(min_count_per_section),
    }
    if isinstance(traces, AdaptivePoincareSectionTraces):
        filtered: list[PoincareSectionTraces] = []
        original_counts_parts: list[np.ndarray] = []
        kept_counts_parts: list[np.ndarray] = []
        for trace in traces.traces:
            counts = _seed_counts_for_trace(trace)
            mask = _seed_filter_mask(
                trace,
                min_total_count=min_total_count,
                min_count_per_section=min_count_per_section,
            )
            filtered.append(_filter_single_trace_by_mask(trace, mask))
            original_counts_parts.append(counts)
            kept_counts_parts.append(counts[mask])
        original_counts = np.concatenate(original_counts_parts) if original_counts_parts else np.empty(0, dtype=int)
        kept_counts = np.concatenate(kept_counts_parts) if kept_counts_parts else np.empty(0, dtype=int)
        metadata = {
            **dict(traces.metadata),
            "survival_filter": _survival_filter_metadata(original_counts, kept_counts, params=params),
        }
        return AdaptivePoincareSectionTraces(tuple(filtered), metadata=metadata)

    counts = _seed_counts_for_trace(traces)
    mask = _seed_filter_mask(
        traces,
        min_total_count=min_total_count,
        min_count_per_section=min_count_per_section,
    )
    filtered = _filter_single_trace_by_mask(traces, mask)
    filtered.metadata.update({
        "survival_filter": _survival_filter_metadata(counts, counts[mask], params=params),
    })
    return filtered


def _as_axis_by_section(axis_by_section, n_section: int) -> list[tuple[float, float]]:
    if len(axis_by_section) != int(n_section):
        raise ValueError("axis_by_section must have one (R, Z) pair per section")
    return [(float(axis[0]), float(axis[1])) for axis in axis_by_section]


def _as_wall_by_section(wall_by_section, n_section: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(wall_by_section) != int(n_section):
        raise ValueError("wall_by_section must have one (R, Z) pair per section")
    out = []
    for wall_R, wall_Z in wall_by_section:
        R = np.asarray(wall_R, dtype=float).ravel()
        Z = np.asarray(wall_Z, dtype=float).ravel()
        if R.size < 3 or Z.size != R.size:
            raise ValueError("each wall section must be a polygon with matching R/Z arrays")
        out.append((R, Z))
    return out


def wall_fraction_theta(
    axis_R: float,
    axis_Z: float,
    R: Sequence[float],
    Z: Sequence[float],
    wall_R: Sequence[float],
    wall_Z: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return wall fraction and poloidal angle for section points."""

    R_arr = np.asarray(R, dtype=float).ravel()
    Z_arr = np.asarray(Z, dtype=float).ravel()
    if R_arr.size != Z_arr.size:
        raise ValueError("R and Z must have the same length")
    wf = boundary_wall_fractions(float(axis_R), float(axis_Z), R_arr, Z_arr, wall_R, wall_Z)
    theta = np.mod(np.arctan2(Z_arr - float(axis_Z), R_arr - float(axis_R)), 2.0 * np.pi)
    theta[~np.isfinite(wf)] = np.nan
    return wf, theta


def poincare_wall_fraction_density(
    traces,
    *,
    axis_by_section,
    wall_by_section,
    wall_fraction_edges: Sequence[float],
    theta_edges: Sequence[float] | None = None,
) -> dict:
    """Histogram Poincare point density in wall-fraction and angle bins."""

    n_section = int(traces.n_section)
    axis = _as_axis_by_section(axis_by_section, n_section)
    walls = _as_wall_by_section(wall_by_section, n_section)
    wf_edges = np.asarray(wall_fraction_edges, dtype=float).ravel()
    if wf_edges.size < 2 or np.any(np.diff(wf_edges) <= 0.0):
        raise ValueError("wall_fraction_edges must be strictly increasing")
    if theta_edges is None:
        th_edges = np.linspace(0.0, 2.0 * np.pi, 33)
    else:
        th_edges = np.asarray(theta_edges, dtype=float).ravel()
    if th_edges.size < 2 or np.any(np.diff(th_edges) <= 0.0):
        raise ValueError("theta_edges must be strictly increasing")
    counts = np.zeros((n_section, wf_edges.size - 1, th_edges.size - 1), dtype=int)
    point_count_by_section: list[int] = []
    for section_index in range(n_section):
        R, Z, _seed = traces.section_points(section_index)
        point_count_by_section.append(int(R.size))
        if R.size == 0:
            continue
        wf, theta = wall_fraction_theta(
            axis[section_index][0],
            axis[section_index][1],
            R,
            Z,
            walls[section_index][0],
            walls[section_index][1],
        )
        finite = np.isfinite(wf) & np.isfinite(theta)
        if not np.any(finite):
            continue
        hist, _wf_edges, _th_edges = np.histogram2d(wf[finite], theta[finite], bins=(wf_edges, th_edges))
        counts[section_index] = hist.astype(int)
    return {
        "counts": counts,
        "wall_fraction_edges": wf_edges,
        "theta_edges": th_edges,
        "point_count_by_section": point_count_by_section,
    }


def _deduplicate_seed_points(seed_R: np.ndarray, seed_Z: np.ndarray, *, tol: float) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(seed_R) & np.isfinite(seed_Z)
    R = np.asarray(seed_R, dtype=float).ravel()[finite]
    Z = np.asarray(seed_Z, dtype=float).ravel()[finite]
    if R.size == 0 or float(tol) <= 0.0:
        return R.copy(), Z.copy()
    inv = 1.0 / float(tol)
    seen: set[tuple[int, int]] = set()
    keep_R: list[float] = []
    keep_Z: list[float] = []
    for r, z in zip(R, Z):
        key = (int(np.round(float(r) * inv)), int(np.round(float(z) * inv)))
        if key in seen:
            continue
        seen.add(key)
        keep_R.append(float(r))
        keep_Z.append(float(z))
    return np.asarray(keep_R, dtype=float), np.asarray(keep_Z, dtype=float)


def _seed_point_from_fraction_angle(
    axis_R: float,
    axis_Z: float,
    wall_R: np.ndarray,
    wall_Z: np.ndarray,
    wall_fraction: float,
    theta: float,
) -> tuple[float, float] | None:
    rho = _ray_polygon_radius(float(axis_R), float(axis_Z), float(theta), wall_R, wall_Z)
    if not np.isfinite(rho) or rho <= 0.0:
        return None
    return (
        float(axis_R + float(wall_fraction) * rho * np.cos(theta)),
        float(axis_Z + float(wall_fraction) * rho * np.sin(theta)),
    )


def adaptive_wall_fraction_seed_points(
    density: dict,
    *,
    seed_axis: tuple[float, float],
    seed_wall: tuple[Sequence[float], Sequence[float]],
    target_points_per_bin: int,
    reducer: str = "min",
    seeds_per_deficit_bin: int = 1,
    max_new_seeds: int | None = None,
    seed_dedup_tol: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate new seed points for under-populated density bins."""

    counts = np.asarray(density["counts"], dtype=int)
    if counts.ndim != 3:
        raise ValueError("density counts must have shape (section, wall_fraction_bin, theta_bin)")
    wf_edges = np.asarray(density["wall_fraction_edges"], dtype=float).ravel()
    theta_edges = np.asarray(density["theta_edges"], dtype=float).ravel()
    target = int(target_points_per_bin)
    if target <= 0:
        raise ValueError("target_points_per_bin must be positive")
    if seeds_per_deficit_bin <= 0:
        raise ValueError("seeds_per_deficit_bin must be positive")
    reducer_key = str(reducer).strip().lower()
    if reducer_key == "min":
        score = np.min(counts, axis=0)
    elif reducer_key == "median":
        score = np.median(counts, axis=0)
    else:
        raise ValueError("reducer must be 'min' or 'median'")
    deficit = np.maximum(0, target - score)
    deficient = np.argwhere(deficit > 0)
    if deficient.size == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float), {
            "target_points_per_bin": target,
            "deficient_bin_count": 0,
            "new_seed_count": 0,
        }

    wall_R = np.asarray(seed_wall[0], dtype=float).ravel()
    wall_Z = np.asarray(seed_wall[1], dtype=float).ravel()
    R_new: list[float] = []
    Z_new: list[float] = []
    for wf_bin, theta_bin in deficient:
        n_seed = int(seeds_per_deficit_bin)
        for k in range(n_seed):
            u = (float(k) + 0.5) / float(n_seed)
            v = np.mod((float(k) + 0.5) * 0.6180339887498949, 1.0)
            frac = float(wf_edges[wf_bin] + u * (wf_edges[wf_bin + 1] - wf_edges[wf_bin]))
            theta = float(theta_edges[theta_bin] + v * (theta_edges[theta_bin + 1] - theta_edges[theta_bin]))
            point = _seed_point_from_fraction_angle(
                float(seed_axis[0]),
                float(seed_axis[1]),
                wall_R,
                wall_Z,
                frac,
                theta,
            )
            if point is None:
                continue
            R_new.append(point[0])
            Z_new.append(point[1])
            if max_new_seeds is not None and len(R_new) >= int(max_new_seeds):
                break
        if max_new_seeds is not None and len(R_new) >= int(max_new_seeds):
            break

    R_arr, Z_arr = _deduplicate_seed_points(
        np.asarray(R_new, dtype=float),
        np.asarray(Z_new, dtype=float),
        tol=float(seed_dedup_tol),
    )
    return R_arr, Z_arr, {
        "target_points_per_bin": target,
        "reducer": reducer_key,
        "deficient_bin_count": int(deficient.shape[0]),
        "new_seed_count": int(R_arr.size),
        "score_min": float(np.min(score)) if score.size else None,
        "score_median": float(np.median(score)) if score.size else None,
        "score_max": float(np.max(score)) if score.size else None,
    }


def densify_boundary_poincare_seeds(
    traces,
    axis_R: float,
    axis_Z: float,
    wall_R: Sequence[float],
    wall_Z: Sequence[float],
    *,
    wall_fraction_bins: int | Sequence[float] = 8,
    theta_bins: int | Sequence[float] = 96,
    min_points_per_bin: int = 1,
    max_new_seeds: int | None = None,
    seeds_per_deficit_bin: int = 1,
    reducer: str = "min",
    dedup_tol: float = 1.0e-6,
) -> BoundarySeedDensificationResult:
    """Propose new seed points for sparse Poincare wall-fraction bins.

    This is the low-level post-compute primitive.  It does not retrace field
    lines; callers can cache the returned seed update and decide when to launch
    the next trace batch.
    """

    if isinstance(wall_fraction_bins, (int, np.integer)):
        wf_edges = np.linspace(0.0, 1.0, int(wall_fraction_bins) + 1)
    else:
        wf_edges = np.asarray(wall_fraction_bins, dtype=float).ravel()
    if isinstance(theta_bins, (int, np.integer)):
        theta_edges = np.linspace(0.0, 2.0 * np.pi, int(theta_bins) + 1)
    else:
        theta_edges = np.asarray(theta_bins, dtype=float).ravel()
    density = poincare_wall_fraction_density(
        traces,
        axis_by_section=[(float(axis_R), float(axis_Z))] * int(traces.n_section),
        wall_by_section=[(wall_R, wall_Z)] * int(traces.n_section),
        wall_fraction_edges=wf_edges,
        theta_edges=theta_edges,
    )
    added_R, added_Z, seed_diag = adaptive_wall_fraction_seed_points(
        density,
        seed_axis=(float(axis_R), float(axis_Z)),
        seed_wall=(wall_R, wall_Z),
        target_points_per_bin=int(min_points_per_bin),
        reducer=reducer,
        seeds_per_deficit_bin=int(seeds_per_deficit_bin),
        max_new_seeds=max_new_seeds,
        seed_dedup_tol=float(dedup_tol),
    )
    seed_R, seed_Z = _deduplicate_seed_points(
        np.concatenate([np.asarray(traces.seed_R, dtype=float).ravel(), added_R]),
        np.concatenate([np.asarray(traces.seed_Z, dtype=float).ravel(), added_Z]),
        tol=float(dedup_tol),
    )
    diagnostics = {
        "density": {
            "counts": density["counts"],
            "wall_fraction_edges": density["wall_fraction_edges"],
            "theta_edges": density["theta_edges"],
            "point_count_by_section": density["point_count_by_section"],
        },
        **seed_diag,
        "original_seed_count": int(traces.n_seed),
        "total_seed_count": int(seed_R.size),
    }
    return BoundarySeedDensificationResult(
        seed_R=seed_R,
        seed_Z=seed_Z,
        added_R=added_R,
        added_Z=added_Z,
        diagnostics=diagnostics,
    )


def _derive_target_points_per_bin(
    traces,
    *,
    axis_by_section,
    wall_by_section,
    reference_wall_fraction_range: tuple[float, float] | None,
    n_wall_fraction_bins: int,
    n_theta_bins: int,
    target_quantile: float,
    target_scale: float,
    min_target_points_per_bin: int,
) -> int:
    if reference_wall_fraction_range is None:
        return int(min_target_points_per_bin)
    ref_min, ref_max = map(float, reference_wall_fraction_range)
    ref_density = poincare_wall_fraction_density(
        traces,
        axis_by_section=axis_by_section,
        wall_by_section=wall_by_section,
        wall_fraction_edges=np.linspace(ref_min, ref_max, int(n_wall_fraction_bins) + 1),
        theta_edges=np.linspace(0.0, 2.0 * np.pi, int(n_theta_bins) + 1),
    )
    counts = np.asarray(ref_density["counts"], dtype=int)
    positive = counts[counts > 0]
    if positive.size == 0:
        return int(min_target_points_per_bin)
    target = int(np.ceil(float(target_scale) * float(np.quantile(positive, float(target_quantile)))))
    return max(int(min_target_points_per_bin), target)


def trace_adaptive_poincare_sections_from_same_orbits_field(
    field,
    seed_R: Sequence[float],
    seed_Z: Sequence[float],
    phi_sections: Sequence[float],
    *,
    axis_by_section,
    wall_by_section,
    N_turns: int,
    DPhi: float,
    density_wall_fraction_range: tuple[float, float] = (0.72, 1.0),
    reference_wall_fraction_range: tuple[float, float] | None = None,
    n_wall_fraction_bins: int = 4,
    n_theta_bins: int = 48,
    target_points_per_bin: int | None = None,
    target_quantile: float = 0.50,
    target_scale: float = 0.75,
    min_target_points_per_bin: int = 4,
    max_rounds: int = 2,
    seeds_per_deficit_bin: int = 1,
    max_new_seeds_per_round: int | None = 512,
    reducer: str = "min",
    seed_axis: tuple[float, float] | None = None,
    seed_wall: tuple[Sequence[float], Sequence[float]] | None = None,
    trace_wall_R: Sequence[float] | None = None,
    trace_wall_Z: Sequence[float] | None = None,
    seed_dedup_tol: float = 1.0e-8,
    extend_phi: bool = True,
    direction: str = "+",
    diagnostic_schema: str | None = None,
) -> AdaptivePoincareSectionTraces:
    """Trace Poincare sections and adaptively add seeds in sparse wall bins."""

    phi = np.asarray(phi_sections, dtype=float).ravel()
    if phi.size == 0:
        raise ValueError("phi_sections must not be empty")
    if int(max_rounds) < 0:
        raise ValueError("max_rounds must be non-negative")
    axis = _as_axis_by_section(axis_by_section, phi.size)
    walls = _as_wall_by_section(wall_by_section, phi.size)
    if seed_axis is None:
        seed_axis = axis[0]
    if seed_wall is None:
        seed_wall = walls[0]
    wf_min, wf_max = map(float, density_wall_fraction_range)
    if not (wf_min < wf_max):
        raise ValueError("density_wall_fraction_range must be increasing")

    traces: list[PoincareSectionTraces] = []
    rounds: list[dict] = []
    current_R = np.asarray(seed_R, dtype=float).ravel()
    current_Z = np.asarray(seed_Z, dtype=float).ravel()
    if current_R.size != current_Z.size or current_R.size == 0:
        raise ValueError("seed_R and seed_Z must be non-empty arrays with the same length")
    last_target = int(target_points_per_bin) if target_points_per_bin is not None else int(min_target_points_per_bin)

    for round_index in range(int(max_rounds) + 1):
        trace = trace_poincare_sections_from_same_orbits_field(
            field,
            current_R,
            current_Z,
            phi,
            N_turns=int(N_turns),
            DPhi=float(DPhi),
            wall_R=trace_wall_R,
            wall_Z=trace_wall_Z,
            extend_phi=extend_phi,
            direction=direction,
        )
        traces.append(trace)
        aggregate = AdaptivePoincareSectionTraces(tuple(traces))
        if round_index >= int(max_rounds):
            break
        if target_points_per_bin is None:
            target = _derive_target_points_per_bin(
                aggregate,
                axis_by_section=axis,
                wall_by_section=walls,
                reference_wall_fraction_range=reference_wall_fraction_range,
                n_wall_fraction_bins=int(n_wall_fraction_bins),
                n_theta_bins=int(n_theta_bins),
                target_quantile=float(target_quantile),
                target_scale=float(target_scale),
                min_target_points_per_bin=int(min_target_points_per_bin),
            )
        else:
            target = int(target_points_per_bin)
        last_target = int(target)
        density = poincare_wall_fraction_density(
            aggregate,
            axis_by_section=axis,
            wall_by_section=walls,
            wall_fraction_edges=np.linspace(wf_min, wf_max, int(n_wall_fraction_bins) + 1),
            theta_edges=np.linspace(0.0, 2.0 * np.pi, int(n_theta_bins) + 1),
        )
        next_R, next_Z, seed_diag = adaptive_wall_fraction_seed_points(
            density,
            seed_axis=seed_axis,
            seed_wall=seed_wall,
            target_points_per_bin=target,
            reducer=reducer,
            seeds_per_deficit_bin=int(seeds_per_deficit_bin),
            max_new_seeds=max_new_seeds_per_round,
            seed_dedup_tol=float(seed_dedup_tol),
        )
        rounds.append({
            "round_index": int(round_index),
            "input_seed_count": int(current_R.size),
            "target_points_per_bin": int(target),
            "density_point_count_by_section": list(density["point_count_by_section"]),
            **seed_diag,
        })
        if next_R.size == 0:
            break
        current_R = next_R
        current_Z = next_Z

    final_aggregate = AdaptivePoincareSectionTraces(tuple(traces))
    final_density = poincare_wall_fraction_density(
        final_aggregate,
        axis_by_section=axis,
        wall_by_section=walls,
        wall_fraction_edges=np.linspace(wf_min, wf_max, int(n_wall_fraction_bins) + 1),
        theta_edges=np.linspace(0.0, 2.0 * np.pi, int(n_theta_bins) + 1),
    )
    final_score = np.min(np.asarray(final_density["counts"], dtype=int), axis=0)
    metadata = {
        "trace_source": "adaptive_same_orbit_multi_section",
        "diagnostic_schema": None if diagnostic_schema is None else str(diagnostic_schema),
        "base_seed_count": int(np.asarray(seed_R).size),
        "total_seed_count": int(sum(trace.n_seed for trace in traces)),
        "trace_layer_count": int(len(traces)),
        "N_turns": int(N_turns),
        "DPhi": float(DPhi),
        "density_wall_fraction_range": [wf_min, wf_max],
        "reference_wall_fraction_range": (
            None if reference_wall_fraction_range is None else list(map(float, reference_wall_fraction_range))
        ),
        "n_wall_fraction_bins": int(n_wall_fraction_bins),
        "n_theta_bins": int(n_theta_bins),
        "adaptive_rounds": rounds,
        "final_density": {
            "point_count_by_section": list(final_density["point_count_by_section"]),
            "target_points_per_bin": int(last_target),
            "score_min": int(np.min(final_score)) if final_score.size else None,
            "score_median": float(np.median(final_score)) if final_score.size else None,
            "score_max": int(np.max(final_score)) if final_score.size else None,
            "below_target_bin_count": int(np.count_nonzero(final_score < int(last_target))),
        },
    }
    return AdaptivePoincareSectionTraces(tuple(traces), metadata=metadata)


__all__ = [
    "AdaptivePoincareSectionTraces",
    "BoundarySeedDensificationResult",
    "adaptive_wall_fraction_seed_points",
    "densify_boundary_poincare_seeds",
    "filter_poincare_traces_by_seed_count",
    "poincare_wall_fraction_density",
    "poincare_seed_hit_counts",
    "trace_adaptive_poincare_sections_from_same_orbits_field",
    "wall_fraction_theta",
]
