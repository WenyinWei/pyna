"""Poincare-first boundary-topology visualization helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from pyna.toroidal.visual.boundary_topology import boundary_dpk_recurrence_profile


@dataclass(frozen=True)
class PoincareDPKClassification:
    """DP^k-derived point classification for a Poincare section."""

    radial_labels: np.ndarray
    eigenvalue_growth: np.ndarray
    spectral_recurrence_min: np.ndarray
    recurrent_surface_indicator: np.ndarray
    chaotic_intervals: tuple[tuple[float, float], ...]
    point_radial_label: np.ndarray
    point_growth: np.ndarray
    point_spectral_recurrence_min: np.ndarray
    point_recurrent_surface_indicator: np.ndarray
    point_chaotic_mask: np.ndarray
    point_surface_mask: np.ndarray


@dataclass(frozen=True)
class PoincareTraceQuality:
    """Basic quality diagnostics for plotted Poincare traces."""

    n_traces: int
    n_points: int
    finite_fraction: float
    lost_trace_fraction: float
    median_points_per_trace: float
    median_step: float
    max_step: float
    suspicious_jump_fraction: float
    radial_drift_median: float
    radial_drift_p95: float


@dataclass(frozen=True)
class PoincareFixedPointValidation:
    """Summary of Newton residual metadata carried by plotted fixed points."""

    n_total: int
    n_with_residual: int
    n_converged_false: int
    n_residual_pass: int
    n_residual_fail: int
    max_residual: float


@dataclass(frozen=True)
class PoincareCurvedIslandBar:
    """Curved island-width bar geometry for a Poincare section.

    The intended use is to plot spectrum-predicted island widths after a
    ``B0 + delta B`` split.  ``R_path``/``Z_path`` should sample the curved
    radial segment in the section plane, and ``half_width`` should be the
    spectrum-predicted half width in the same radial coordinate used by the
    metadata.
    """

    R_path: np.ndarray
    Z_path: np.ndarray
    mode_m: int | None = None
    mode_n: int | None = None
    radial_label: float = np.nan
    half_width: float = np.nan
    amplitude: float = np.nan
    phase: float = np.nan
    kind: str | None = None
    branch: int | None = None
    color: str | tuple[float, ...] | None = None
    colormap: str | None = None
    label: str | None = None
    source: str = "B0 + delta B spectrum"


@dataclass(frozen=True)
class PoincareIslandSecondaryCoordinates:
    """Point labels in a local magnetic-island coordinate approximation.

    ``secondary_radius`` is normalized so zero is the predicted O-point helical
    axis and one is the pendulum-model separatrix.  Points outside all supplied
    islands have ``NaN`` radius and ``inside_island=False``.
    """

    secondary_radius: np.ndarray
    helical_phase: np.ndarray
    island_hamiltonian: np.ndarray
    inside_island: np.ndarray
    mode_m: np.ndarray
    mode_n: np.ndarray
    branch: np.ndarray


@dataclass(frozen=True)
class PoincareTopologyFigureStyle:
    """Reusable defaults for high-resolution Poincare topology figures."""

    figsize: tuple[float, float] = (7.2, 6.8)
    dpi: int = 360
    point_size: float = 1.15
    island_point_size: float = 1.05
    island_contour_levels: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
    island_contour_linewidth: float = 0.42
    island_contour_alpha: float = 0.78


@dataclass(frozen=True)
class PoincareTopologyReportPayload:
    """Reusable inputs for Poincare boundary-topology map/report figures.

    This container is deliberately data-only.  It carries already-computed
    Poincare points, DP^k diagnostics, Newton/cyna fixed points, spectrum bars,
    and island-chain context without locating fixed points or tracing field
    lines itself.
    """

    R: np.ndarray
    Z: np.ndarray
    radial_label: np.ndarray | None = None
    dpk_radial_labels: np.ndarray | None = None
    dpk_metrics: tuple[object, ...] = field(default_factory=tuple)
    point_growth: np.ndarray | None = None
    point_spectral_recurrence_min: np.ndarray | None = None
    point_recurrent_surface_indicator: np.ndarray | None = None
    fixed_points: tuple[object, ...] = field(default_factory=tuple)
    island_chains: tuple[object, ...] = field(default_factory=tuple)
    island_bars: tuple[object, ...] = field(default_factory=tuple)
    secondary_coordinates: PoincareIslandSecondaryCoordinates | None = None
    trace_index: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        R = np.asarray(self.R, dtype=float).ravel()
        Z = np.asarray(self.Z, dtype=float).ravel()
        if R.size != Z.size:
            raise ValueError("R and Z must have the same length")
        n_points = int(R.size)
        radial = _optional_point_array(self.radial_label, n_points, name="radial_label")
        growth = _optional_point_array(self.point_growth, n_points, name="point_growth")
        recurrence = _optional_point_array(
            self.point_spectral_recurrence_min,
            n_points,
            name="point_spectral_recurrence_min",
        )
        surface = _optional_point_array(
            self.point_recurrent_surface_indicator,
            n_points,
            name="point_recurrent_surface_indicator",
        )
        trace_index = None
        if self.trace_index is not None:
            trace_index = np.asarray(self.trace_index, dtype=int).ravel()
            if trace_index.size != n_points:
                raise ValueError("trace_index length must match R/Z")
        dpk_radial = None
        if self.dpk_radial_labels is not None:
            dpk_radial = np.asarray(self.dpk_radial_labels, dtype=float).ravel()
            if len(self.dpk_metrics) != dpk_radial.size:
                raise ValueError("dpk_metrics length must match dpk_radial_labels")
        elif self.dpk_metrics:
            raise ValueError("dpk_radial_labels are required when dpk_metrics are supplied")
        if self.secondary_coordinates is not None:
            secondary_size = int(np.asarray(self.secondary_coordinates.secondary_radius).size)
            if secondary_size != n_points:
                raise ValueError("secondary coordinate arrays must match R/Z point count")
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "Z", Z)
        object.__setattr__(self, "radial_label", radial)
        object.__setattr__(self, "dpk_radial_labels", dpk_radial)
        object.__setattr__(self, "dpk_metrics", tuple(self.dpk_metrics))
        object.__setattr__(self, "point_growth", growth)
        object.__setattr__(self, "point_spectral_recurrence_min", recurrence)
        object.__setattr__(self, "point_recurrent_surface_indicator", surface)
        object.__setattr__(self, "fixed_points", tuple(self.fixed_points))
        object.__setattr__(self, "island_chains", tuple(self.island_chains))
        object.__setattr__(self, "island_bars", tuple(self.island_bars))
        object.__setattr__(self, "trace_index", trace_index)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def n_points(self) -> int:
        """Number of Poincare section points carried by this payload."""

        return int(self.R.size)

    @property
    def has_dpk_profile(self) -> bool:
        """Whether the payload has a complete radial DP^k profile."""

        return self.dpk_radial_labels is not None and bool(self.dpk_metrics)

    def map_kwargs(self) -> dict[str, object]:
        """Return keyword arguments for :func:`plot_poincare_topology_map`."""

        kwargs: dict[str, object] = {
            "radial_label": self.radial_label,
            "point_growth": self.point_growth,
            "point_spectral_recurrence_min": self.point_spectral_recurrence_min,
            "point_recurrent_surface_indicator": self.point_recurrent_surface_indicator,
            "fixed_points": self.fixed_points,
            "island_bars": self.island_bars,
            "secondary_coordinates": self.secondary_coordinates,
            "secondary_trace_index": self.trace_index,
        }
        if self.has_dpk_profile:
            kwargs["dpk_radial_labels"] = self.dpk_radial_labels
            kwargs["dpk_metrics"] = self.dpk_metrics
        return kwargs

    def report_kwargs(self) -> dict[str, object]:
        """Return keyword arguments for :func:`plot_poincare_topology_report`."""

        if self.radial_label is None:
            raise ValueError("radial_label is required for a Poincare topology report")
        if not self.has_dpk_profile:
            raise ValueError("dpk_radial_labels and dpk_metrics are required for a report")
        kwargs = self.map_kwargs()
        kwargs["dpk_radial_labels"] = self.dpk_radial_labels
        kwargs["dpk_metrics"] = self.dpk_metrics
        kwargs["island_chains"] = self.island_chains
        return kwargs


def _optional_point_array(value, n_points: int, *, name: str) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size != int(n_points):
        raise ValueError(f"{name} length must match R/Z")
    return arr


def poincare_topology_report_payload(
    R,
    Z,
    *,
    radial_label=None,
    dpk_radial_labels=None,
    dpk_metrics: Sequence[object] = (),
    point_growth=None,
    point_spectral_recurrence_min=None,
    point_recurrent_surface_indicator=None,
    fixed_points: Sequence[object] = (),
    island_chains: Sequence[object] = (),
    island_bars: Sequence[object] = (),
    secondary_coordinates: PoincareIslandSecondaryCoordinates | None = None,
    trace_index=None,
    trace_counts=None,
    metadata: Mapping[str, object] | None = None,
) -> PoincareTopologyReportPayload:
    """Build a validated payload for Poincare topology map/report figures."""

    R_arr = np.asarray(R, dtype=float).ravel()
    if trace_index is None and trace_counts is not None:
        trace_index = poincare_trace_index_from_counts(trace_counts, n_points=R_arr.size)
    return PoincareTopologyReportPayload(
        R=R_arr,
        Z=Z,
        radial_label=radial_label,
        dpk_radial_labels=dpk_radial_labels,
        dpk_metrics=tuple(dpk_metrics),
        point_growth=point_growth,
        point_spectral_recurrence_min=point_spectral_recurrence_min,
        point_recurrent_surface_indicator=point_recurrent_surface_indicator,
        fixed_points=tuple(fixed_points),
        island_chains=tuple(island_chains),
        island_bars=tuple(island_bars),
        secondary_coordinates=secondary_coordinates,
        trace_index=trace_index,
        metadata={} if metadata is None else dict(metadata),
    )


def poincare_trace_index_from_counts(counts, *, n_points: int | None = None) -> np.ndarray:
    """Expand per-trace Poincare hit counts to one trace id per flat point."""

    counts_arr = np.asarray(counts, dtype=int).ravel()
    if counts_arr.ndim != 1:
        raise ValueError("counts must be one-dimensional after flattening")
    if np.any(counts_arr < 0):
        raise ValueError("counts must be non-negative")
    trace_index = np.repeat(np.arange(counts_arr.size, dtype=int), counts_arr)
    if n_points is not None and trace_index.size != int(n_points):
        raise ValueError("counts do not sum to the number of Poincare points")
    return trace_index


def _wrap_to_pi(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _magnetic_axis_from_object(eq) -> tuple[float, float]:
    axis = getattr(eq, "magnetic_axis", None)
    if axis is not None:
        return float(axis[0]), float(axis[1])
    return float(getattr(eq, "R0")), float(getattr(eq, "axis_Z", 0.0))


def circular_flux_coordinate_island_bars(
    components: Sequence[object],
    eq,
    *,
    phi: float = 0.0,
    kinds: Sequence[str] = ("O",),
    n_points: int = 33,
    label_prefix: str = "flux-coordinate bar",
    mode_colors: Mapping[tuple[int, int], str | tuple[float, ...]] | None = None,
    mode_colormaps: Mapping[tuple[int, int], str] | None = None,
) -> list[PoincareCurvedIslandBar]:
    """Return constant-theta island-width bars for circular flux coordinates.

    This is the lightweight tutorial/simple-stellarator bridge for drawing
    spectrum-predicted widths in magnetic-surface coordinates.  Each bar is a
    radial segment at one predicted X/O poloidal angle, sampled from
    ``r_res - half_width`` to ``r_res + half_width``.  For a circular tutorial
    equilibrium the path is visually straight; for plotting code it is still a
    flux-coordinate bar rather than a pair of annular circles.
    """

    axis_R, axis_Z = _magnetic_axis_from_object(eq)
    r0 = float(getattr(eq, "r0"))
    count = max(2, int(n_points))
    normalized_kinds = tuple(str(kind).upper() for kind in kinds)
    bars: list[PoincareCurvedIslandBar] = []
    for component in components:
        m = int(getattr(component, "m"))
        n = int(getattr(component, "n"))
        psi_res = float(getattr(component, "psi_res"))
        r_res = float(np.sqrt(max(psi_res, 0.0)) * r0)
        half_width = abs(float(getattr(component, "half_width_r")))
        amplitude = abs(complex(getattr(component, "b_mn", np.nan)))
        phase = float(np.angle(complex(getattr(component, "b_mn", np.nan))))
        fixed = component.fixed_points(float(phi))
        for kind in normalized_kinds:
            key = f"theta_{kind}"
            if key not in fixed:
                continue
            theta_values = np.asarray(fixed[key], dtype=float)
            if theta_values.ndim == 2:
                theta_values = theta_values[0]
            for branch, theta in enumerate(theta_values.ravel()):
                radii = r_res + np.linspace(-half_width, half_width, count)
                theta_f = float(theta)
                label = f"{label_prefix} {kind} ({m},{n})" if branch == 0 else None
                bars.append(PoincareCurvedIslandBar(
                    R_path=axis_R + radii * np.cos(theta_f),
                    Z_path=axis_Z + radii * np.sin(theta_f),
                    mode_m=m,
                    mode_n=n,
                    radial_label=r_res,
                    half_width=half_width,
                    amplitude=amplitude,
                    phase=phase,
                    kind=kind,
                    branch=int(branch),
                    color=None if mode_colors is None else mode_colors.get((m, n)),
                    colormap=None if mode_colormaps is None else mode_colormaps.get((m, n)),
                    label=label,
                    source=f"circular flux-coordinate {kind}-point width",
                ))
    return bars


def poincare_island_secondary_coordinates(
    R,
    Z,
    components: Sequence[object],
    eq,
    *,
    phi: float = 0.0,
    separatrix_tol: float = 0.03,
    branch_angle_gate: bool = True,
    branch_angle_tol: float = 0.08,
) -> PoincareIslandSecondaryCoordinates:
    """Assign Poincare points to approximate island secondary coordinates.

    The O-points predicted by each resonant component define local helical axes.
    The normalized secondary radius uses a pendulum island approximation:

    ``rho_i^2 = (2*((r-r_res)/w)^2 - cos(m*(theta-theta_O)) + 1) / 2``.

    This gives ``rho_i=0`` at the O axis and ``rho_i=1`` on the approximate
    separatrix, including both radial extrema at O phase and the X phase at the
    resonant radius.  If islands overlap, each point is assigned to the
    component/branch with smallest ``rho_i``.  The optional branch-angle gate
    breaks the ``cos(m theta)`` degeneracy between the ``m`` island lobes by
    requiring points to lie in the ordinary-poloidal-angle sector around the
    selected O point.
    """

    R_arr = np.asarray(R, dtype=float).ravel()
    Z_arr = np.asarray(Z, dtype=float).ravel()
    if R_arr.shape != Z_arr.shape:
        raise ValueError("R and Z must have matching shape")
    n_points = R_arr.size
    axis_R, axis_Z = _magnetic_axis_from_object(eq)
    r0 = float(getattr(eq, "r0"))
    r_minor = np.hypot(R_arr - axis_R, Z_arr - axis_Z)
    theta = np.arctan2(Z_arr - axis_Z, R_arr - axis_R)

    best_radius = np.full(n_points, np.inf, dtype=float)
    out_phase = np.full(n_points, np.nan, dtype=float)
    out_hamiltonian = np.full(n_points, np.nan, dtype=float)
    out_m = np.full(n_points, -1, dtype=int)
    out_n = np.full(n_points, -1, dtype=int)
    out_branch = np.full(n_points, -1, dtype=int)
    inside = np.zeros(n_points, dtype=bool)

    for component in components:
        m = int(getattr(component, "m"))
        n = int(getattr(component, "n"))
        psi_res = float(getattr(component, "psi_res"))
        r_res = float(np.sqrt(max(psi_res, 0.0)) * r0)
        half_width = abs(float(getattr(component, "half_width_r")))
        if not np.isfinite(half_width) or half_width <= 0.0 or m == 0:
            continue
        fixed = component.fixed_points(float(phi))
        theta_o = np.asarray(fixed.get("theta_O", []), dtype=float)
        if theta_o.ndim == 2:
            theta_o = theta_o[0]
        for branch, theta_axis in enumerate(theta_o.ravel()):
            ordinary_phase = _wrap_to_pi(theta - float(theta_axis))
            helical_phase = _wrap_to_pi(float(m) * ordinary_phase)
            radial_norm = (r_minor - r_res) / half_width
            hamiltonian = 2.0 * radial_norm * radial_norm - np.cos(helical_phase)
            rho_sq = 0.5 * (hamiltonian + 1.0)
            finite = np.isfinite(rho_sq)
            candidate_radius = np.sqrt(np.clip(rho_sq, 0.0, None))
            candidate_inside = finite & (rho_sq <= 1.0 + float(separatrix_tol))
            if branch_angle_gate:
                angular_limit = (np.pi / abs(float(m))) * (1.0 + float(branch_angle_tol))
                candidate_inside &= np.abs(ordinary_phase) <= angular_limit
            improve = candidate_inside & (candidate_radius < best_radius)
            if not np.any(improve):
                continue
            best_radius[improve] = candidate_radius[improve]
            out_phase[improve] = helical_phase[improve]
            out_hamiltonian[improve] = hamiltonian[improve]
            out_m[improve] = m
            out_n[improve] = n
            out_branch[improve] = int(branch)
            inside[improve] = True

    secondary_radius = best_radius.copy()
    secondary_radius[~inside] = np.nan
    return PoincareIslandSecondaryCoordinates(
        secondary_radius=secondary_radius,
        helical_phase=out_phase,
        island_hamiltonian=out_hamiltonian,
        inside_island=inside,
        mode_m=out_m,
        mode_n=out_n,
        branch=out_branch,
    )


def fixed_point_phase_comparison_markers(
    rows: Sequence[object],
    *,
    prefer_predicted_kind: bool = False,
) -> list[dict]:
    """Convert cyna/spectrum phase-comparison rows to X/O plot markers."""

    markers: list[dict] = []
    for row in rows:
        kind = getattr(row, "predicted_kind", "") if prefer_predicted_kind else getattr(row, "newton_kind", None)
        if kind is None or str(kind) == "":
            kind = getattr(row, "predicted_kind", "")
        m_value = getattr(row, "m", getattr(row, "mode_m", None))
        n_value = getattr(row, "n", getattr(row, "mode_n", None))
        mode_m = None if m_value is None else int(m_value)
        mode_n = None if n_value is None else int(n_value)
        markers.append({
            "R": float(getattr(row, "newton_R")),
            "Z": float(getattr(row, "newton_Z")),
            "kind": str(kind).upper(),
            "mode_m": mode_m,
            "mode_n": mode_n,
            "residual": float(getattr(row, "residual", np.nan)),
            "converged": bool(getattr(row, "converged", False)),
            "metadata": {
                "predicted_kind": str(getattr(row, "predicted_kind", "")),
                "mode_m": mode_m,
                "mode_n": mode_n,
                "branch": int(getattr(row, "branch", -1)),
                "theta_error": float(getattr(row, "theta_error", np.nan)),
                "helical_phase_error": float(getattr(row, "helical_phase_error", np.nan)),
                "radial_error": float(getattr(row, "radial_error", np.nan)),
                "map_span": float(getattr(row, "map_span", np.nan)),
                "phi": float(getattr(row, "phi", np.nan)),
            },
        })
    return markers


def _payload_attr(payload, *names, default=None):
    value = _bar_attr(payload, *names, default=None)
    if value is not None:
        return value
    metadata = _fixed_point_metadata(payload)
    for name in names:
        if name in metadata:
            return metadata[name]
    return default


def _payload_mode(payload) -> tuple[int, int] | None:
    mode = _payload_attr(payload, "mode", default=None)
    if mode is not None:
        arr = np.asarray(mode).ravel()
        if arr.size >= 2:
            return int(arr[0]), int(arr[1])
    m = _payload_attr(payload, "mode_m", "m", default=None)
    n = _payload_attr(payload, "mode_n", "n", default=None)
    if m is None or n is None:
        return None
    return int(m), int(n)


def _payload_branch(payload) -> int | None:
    branch = _payload_attr(payload, "branch", default=None)
    if branch is None:
        return None
    try:
        return int(branch)
    except Exception:
        return None


def _component_half_width(component) -> float:
    width = _payload_attr(
        component,
        "half_width_r",
        "half_width",
        "island_half_width",
        "width",
        default=None,
    )
    if width is None:
        return np.nan
    return abs(float(width))


def _component_radial_label(component, eq=None) -> float:
    radial = _payload_attr(component, "radial_label", "r_res", "minor_radius", default=None)
    if radial is not None:
        return float(radial)
    psi_res = _payload_attr(component, "psi_res", "s_res", default=None)
    if psi_res is not None and eq is not None and hasattr(eq, "r0"):
        return float(np.sqrt(max(float(psi_res), 0.0)) * float(getattr(eq, "r0")))
    return np.nan


def fixed_point_centered_island_bars(
    fixed_points: Sequence[object],
    components: Sequence[object] = (),
    *,
    eq=None,
    axis: tuple[float, float] | None = None,
    mode: tuple[int, int] | None = None,
    kinds: Sequence[str] = ("O",),
    n_points: int = 129,
    residual_tol: float | None = None,
    draw_unvalidated_fixed_points: bool = True,
    require_converged_fixed_points: bool = False,
    direction_angle: float | None = None,
    label_prefix: str = "fixed-point centered width",
    mode_colors: Mapping[tuple[int, int], str | tuple[float, ...]] | None = None,
    mode_colormaps: Mapping[tuple[int, int], str] | None = None,
) -> list[PoincareCurvedIslandBar]:
    """Anchor spectrum-predicted island-width bars at validated fixed points.

    This function does not locate X/O points.  It consumes fixed points from a
    Newton/cyna or equivalent solver, looks up the matching spectrum component
    width, and draws a local flux-radial bar centered on each selected point.
    """

    if axis is None:
        axis = _magnetic_axis_from_object(eq) if eq is not None else None
    normalized_kinds = tuple(str(kind).upper() for kind in kinds)
    count = max(2, int(n_points))

    components_by_mode: dict[tuple[int, int], object] = {}
    components_by_mode_branch: dict[tuple[int, int, int], object] = {}
    for component in components:
        component_mode = _payload_mode(component)
        if component_mode is None:
            continue
        components_by_mode.setdefault(component_mode, component)
        branch = _payload_branch(component)
        if branch is not None:
            components_by_mode_branch[(component_mode[0], component_mode[1], branch)] = component

    bars: list[PoincareCurvedIslandBar] = []
    for fixed_point in fixed_points:
        if not _fixed_point_passes_validation(
            fixed_point,
            residual_tol=residual_tol,
            draw_unvalidated=draw_unvalidated_fixed_points,
            require_converged=require_converged_fixed_points,
        ):
            continue
        item = _fixed_point_rz_kind(fixed_point)
        if item is None:
            continue
        R0, Z0, kind = item
        if kind not in normalized_kinds:
            continue
        fp_mode = mode if mode is not None else _payload_mode(fixed_point)
        if fp_mode is None and len(components_by_mode) == 1:
            fp_mode = next(iter(components_by_mode))
        branch = _payload_branch(fixed_point)
        component = None
        if fp_mode is not None and branch is not None:
            component = components_by_mode_branch.get((fp_mode[0], fp_mode[1], branch))
        if component is None and fp_mode is not None:
            component = components_by_mode.get(fp_mode)

        half_width = _payload_attr(fixed_point, "half_width_r", "half_width", default=None)
        half_width = abs(float(half_width)) if half_width is not None else _component_half_width(component)
        if not np.isfinite(half_width) or half_width <= 0.0:
            continue

        if direction_angle is not None:
            theta = float(direction_angle)
            unit_R = float(np.cos(theta))
            unit_Z = float(np.sin(theta))
        elif axis is not None:
            dR = float(R0) - float(axis[0])
            dZ = float(Z0) - float(axis[1])
            norm = float(np.hypot(dR, dZ))
            if norm <= 0.0 or not np.isfinite(norm):
                unit_R, unit_Z = 1.0, 0.0
            else:
                unit_R, unit_Z = dR / norm, dZ / norm
        else:
            unit_R, unit_Z = 1.0, 0.0

        t = np.linspace(-half_width, half_width, count)
        amplitude = _payload_attr(fixed_point, "amplitude", "b_mn_abs", default=None)
        phase = _payload_attr(fixed_point, "phase", "b_mn_phase", default=None)
        if component is not None:
            b_mn = _payload_attr(component, "b_mn", "b_res", default=None)
            if b_mn is not None:
                amplitude = abs(complex(b_mn))
                phase = float(np.angle(complex(b_mn)))
            else:
                amplitude = _payload_attr(component, "amplitude", "b_mn_abs", default=amplitude)
                phase = _payload_attr(component, "phase", "b_mn_phase", default=phase)
        amplitude = np.nan if amplitude is None else float(amplitude)
        phase = np.nan if phase is None else float(phase)
        label = f"{label_prefix} ({fp_mode[0]},{fp_mode[1]})" if fp_mode is not None and not bars else None
        bars.append(PoincareCurvedIslandBar(
            R_path=float(R0) + t * unit_R,
            Z_path=float(Z0) + t * unit_Z,
            mode_m=None if fp_mode is None else fp_mode[0],
            mode_n=None if fp_mode is None else fp_mode[1],
            radial_label=_component_radial_label(component, eq=eq) if component is not None else np.nan,
            half_width=half_width,
            amplitude=amplitude,
            phase=phase,
            kind=kind,
            branch=branch,
            color=None if mode_colors is None or fp_mode is None else mode_colors.get(fp_mode),
            colormap=None if mode_colormaps is None or fp_mode is None else mode_colormaps.get(fp_mode),
            label=label,
            source="B0 + deltaB spectrum width anchored to fixed point",
        ))
    return bars


def _as_point_array(value, n_points: int, *, default=np.nan, name: str) -> np.ndarray:
    if value is None:
        return np.full(n_points, default, dtype=float)
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size != n_points:
        raise ValueError(f"{name} length must match Poincare point count")
    return arr


def _flatten_trace_inputs(R, Z, trace_id=None):
    R_arr = np.asarray(R, dtype=float)
    Z_arr = np.asarray(Z, dtype=float)
    if R_arr.shape != Z_arr.shape:
        raise ValueError("R and Z must have matching shapes")
    if R_arr.ndim == 1:
        flat_R = R_arr.ravel()
        flat_Z = Z_arr.ravel()
        if trace_id is None:
            ids = np.zeros(flat_R.size, dtype=int)
        else:
            ids = np.asarray(trace_id, dtype=int).ravel()
            if ids.size != flat_R.size:
                raise ValueError("trace_id length must match R/Z")
    elif R_arr.ndim == 2:
        flat_R = R_arr.ravel()
        flat_Z = Z_arr.ravel()
        if trace_id is None:
            ids = np.repeat(np.arange(R_arr.shape[0], dtype=int), R_arr.shape[1])
        else:
            ids = np.asarray(trace_id, dtype=int)
            if ids.shape == R_arr.shape:
                ids = ids.ravel()
            elif ids.size == R_arr.shape[0]:
                ids = np.repeat(ids.ravel(), R_arr.shape[1])
            else:
                raise ValueError("trace_id must match R/Z shape or the number of traces")
    else:
        raise ValueError("R and Z must be 1-D or 2-D arrays")
    return flat_R, flat_Z, ids


def poincare_trace_quality(
    R,
    Z,
    *,
    trace_id=None,
    radial_label=None,
    center: tuple[float, float] = (0.0, 0.0),
    jump_threshold: float | None = None,
) -> PoincareTraceQuality:
    """Compute simple numerical sanity checks for Poincare point clouds.

    The diagnostics are intentionally generic: they detect common plotting and
    tracing failures, but they do not replace physics validation against the
    magnetic field, section convention, or ``B0 + delta B`` split.
    """

    flat_R, flat_Z, ids = _flatten_trace_inputs(R, Z, trace_id=trace_id)
    n_points = int(flat_R.size)
    finite = np.isfinite(flat_R) & np.isfinite(flat_Z)
    finite_fraction = 1.0 if n_points == 0 else float(np.count_nonzero(finite) / n_points)
    unique_ids = np.unique(ids) if ids.size else np.array([], dtype=int)
    n_traces = int(unique_ids.size)

    if radial_label is None:
        radial = np.hypot(flat_R - float(center[0]), flat_Z - float(center[1]))
    else:
        radial = np.asarray(radial_label, dtype=float)
        if radial.shape == np.asarray(R, dtype=float).shape:
            radial = radial.ravel()
        else:
            radial = radial.ravel()
        if radial.size != n_points:
            raise ValueError("radial_label length must match R/Z")

    finite_counts = []
    step_lengths = []
    radial_drifts = []
    lost = 0
    for trace in unique_ids:
        mask = ids == trace
        trace_finite = mask & finite
        count = int(np.count_nonzero(trace_finite))
        finite_counts.append(count)
        if count == 0:
            lost += 1
            continue
        if count < np.count_nonzero(mask):
            lost += 1
        R_i = flat_R[trace_finite]
        Z_i = flat_Z[trace_finite]
        radial_i = radial[trace_finite]
        if count > 1:
            step_lengths.extend(np.hypot(np.diff(R_i), np.diff(Z_i)).tolist())
        finite_radial = radial_i[np.isfinite(radial_i)]
        if finite_radial.size:
            radial_drifts.append(float(np.nanmax(finite_radial) - np.nanmin(finite_radial)))

    steps = np.asarray(step_lengths, dtype=float)
    if steps.size:
        median_step = float(np.nanmedian(steps))
        max_step = float(np.nanmax(steps))
        if jump_threshold is None:
            mad = float(np.nanmedian(np.abs(steps - median_step)))
            threshold = median_step + 8.0 * max(mad, 1.0e-12)
        else:
            threshold = float(jump_threshold)
        suspicious_jump_fraction = float(np.count_nonzero(steps > threshold) / steps.size)
    else:
        median_step = 0.0
        max_step = 0.0
        suspicious_jump_fraction = 0.0

    drifts = np.asarray(radial_drifts, dtype=float)
    return PoincareTraceQuality(
        n_traces=n_traces,
        n_points=n_points,
        finite_fraction=finite_fraction,
        lost_trace_fraction=0.0 if n_traces == 0 else float(lost / n_traces),
        median_points_per_trace=0.0 if not finite_counts else float(np.median(finite_counts)),
        median_step=median_step,
        max_step=max_step,
        suspicious_jump_fraction=suspicious_jump_fraction,
        radial_drift_median=0.0 if drifts.size == 0 else float(np.nanmedian(drifts)),
        radial_drift_p95=0.0 if drifts.size == 0 else float(np.nanpercentile(drifts, 95.0)),
    )


def _interp_profile(point_radial: np.ndarray, profile_radial: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.full(point_radial.shape, np.nan, dtype=float)
    finite_profile = np.isfinite(profile_radial) & np.isfinite(values)
    finite_points = np.isfinite(point_radial)
    if not np.any(finite_profile) or not np.any(finite_points):
        return out
    x = profile_radial[finite_profile]
    y = values[finite_profile]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if x.size == 1:
        out[finite_points] = float(y[0])
    else:
        out[finite_points] = np.interp(point_radial[finite_points], x, y, left=y[0], right=y[-1])
    return out


def poincare_dpk_classification(
    point_radial_label: Sequence[float],
    *,
    dpk_radial_labels: Sequence[float] | None = None,
    dpk_metrics: Sequence[object] | None = None,
    point_growth: Sequence[float] | None = None,
    point_spectral_recurrence_min: Sequence[float] | None = None,
    point_recurrent_surface_indicator: Sequence[float] | None = None,
    growth_threshold: float = 0.0,
    recurrence_threshold: float = 0.02,
    recurrent_surface_threshold: float = 0.5,
) -> PoincareDPKClassification:
    """Classify Poincare points using DP^k growth and recurrence diagnostics.

    Chaotic points require growth above threshold and no recurrent-surface
    signature.  This protects closed magnetic surfaces whose DP^k spectrum
    periodically returns close to one near the poloidal mode count.
    """

    point_radial = np.asarray(point_radial_label, dtype=float).ravel()
    n_points = point_radial.size
    if (dpk_radial_labels is None) != (dpk_metrics is None):
        raise ValueError("dpk_radial_labels and dpk_metrics must be supplied together")
    if dpk_radial_labels is None:
        profile_radial = np.array([], dtype=float)
        profile_growth = np.array([], dtype=float)
        profile_recurrence = np.array([], dtype=float)
        profile_surface = np.array([], dtype=float)
        intervals: tuple[tuple[float, float], ...] = ()
    else:
        profile = boundary_dpk_recurrence_profile(
            dpk_radial_labels,
            dpk_metrics or (),
            growth_threshold=growth_threshold,
            recurrence_threshold=recurrence_threshold,
            recurrent_surface_threshold=recurrent_surface_threshold,
        )
        profile_radial = profile.radial_labels
        profile_growth = profile.eigenvalue_growth
        profile_recurrence = profile.spectral_recurrence_min
        profile_surface = profile.recurrent_surface_indicator
        intervals = profile.chaotic_intervals

    growth = _as_point_array(point_growth, n_points, name="point_growth")
    recurrence = _as_point_array(point_spectral_recurrence_min, n_points, name="point_spectral_recurrence_min")
    surface = _as_point_array(
        point_recurrent_surface_indicator,
        n_points,
        default=np.nan,
        name="point_recurrent_surface_indicator",
    )
    if point_growth is None and profile_radial.size:
        growth = _interp_profile(point_radial, profile_radial, profile_growth)
    if point_spectral_recurrence_min is None and profile_radial.size:
        recurrence = _interp_profile(point_radial, profile_radial, profile_recurrence)
    if point_recurrent_surface_indicator is None and profile_radial.size:
        surface = _interp_profile(point_radial, profile_radial, profile_surface)
    if np.all(~np.isfinite(surface)):
        surface = np.asarray(np.isfinite(recurrence) & (recurrence <= float(recurrence_threshold)), dtype=float)

    surface_mask = np.isfinite(surface) & (surface >= float(recurrent_surface_threshold))
    chaotic_mask = (
        np.isfinite(growth)
        & (growth > float(growth_threshold))
        & ~surface_mask
    )
    return PoincareDPKClassification(
        radial_labels=profile_radial,
        eigenvalue_growth=profile_growth,
        spectral_recurrence_min=profile_recurrence,
        recurrent_surface_indicator=profile_surface,
        chaotic_intervals=intervals,
        point_radial_label=point_radial,
        point_growth=growth,
        point_spectral_recurrence_min=recurrence,
        point_recurrent_surface_indicator=surface,
        point_chaotic_mask=chaotic_mask,
        point_surface_mask=surface_mask & ~chaotic_mask,
    )


def _fixed_point_rz_kind(fp) -> tuple[float, float, str] | None:
    if isinstance(fp, Mapping):
        if "rzphi0" in fp:
            rzphi0 = np.asarray(fp["rzphi0"], dtype=float).ravel()
            R = rzphi0[0]
            Z = rzphi0[1]
        else:
            R = fp.get("R", fp.get("r"))
            Z = fp.get("Z", fp.get("z"))
        kind = fp.get("kind", fp.get("orbit_type", fp.get("type", "")))
    else:
        if hasattr(fp, "rzphi0"):
            rzphi0 = np.asarray(getattr(fp, "rzphi0"), dtype=float).ravel()
            R = rzphi0[0]
            Z = rzphi0[1]
        else:
            R = getattr(fp, "R", getattr(fp, "r", None))
            Z = getattr(fp, "Z", getattr(fp, "z", None))
        kind = getattr(fp, "kind", getattr(fp, "orbit_type", getattr(fp, "type", "")))
    if R is None or Z is None:
        return None
    return float(R), float(Z), str(kind).upper()


def _fixed_point_metadata(fp) -> Mapping:
    if isinstance(fp, Mapping):
        metadata = fp.get("metadata", {})
        return metadata if isinstance(metadata, Mapping) else {}
    metadata = getattr(fp, "metadata", {})
    return metadata if isinstance(metadata, Mapping) else {}


def _fixed_point_residual(fp) -> float:
    names = ("residual", "fpt_residual", "closure_residual", "fixed_point_residual")
    if isinstance(fp, Mapping):
        for name in names:
            if name in fp:
                return float(fp[name])
    else:
        for name in names:
            if hasattr(fp, name):
                return float(getattr(fp, name))
    metadata = _fixed_point_metadata(fp)
    for name in names:
        if name in metadata:
            return float(metadata[name])
    return np.nan


def _fixed_point_converged(fp) -> bool | None:
    names = ("converged", "newton_converged", "fixed_point_converged")
    if isinstance(fp, Mapping):
        for name in names:
            if name in fp:
                return bool(fp[name])
    else:
        for name in names:
            if hasattr(fp, name):
                return bool(getattr(fp, name))
    metadata = _fixed_point_metadata(fp)
    for name in names:
        if name in metadata:
            return bool(metadata[name])
    return None


def _fixed_point_passes_validation(
    fp,
    *,
    residual_tol: float | None,
    draw_unvalidated: bool,
    require_converged: bool,
) -> bool:
    converged = _fixed_point_converged(fp)
    if require_converged and converged is False:
        return False
    residual = _fixed_point_residual(fp)
    if residual_tol is None:
        return True if np.isfinite(residual) or draw_unvalidated else False
    if not np.isfinite(residual):
        return bool(draw_unvalidated)
    return float(residual) <= float(residual_tol)


def poincare_fixed_point_validation(
    fixed_points: Sequence[object],
    *,
    residual_tol: float = 1.0e-8,
) -> PoincareFixedPointValidation:
    """Summarize Newton residual metadata before drawing X/O markers."""

    residuals = np.asarray([_fixed_point_residual(fp) for fp in fixed_points], dtype=float)
    finite = np.isfinite(residuals)
    converged_values = [_fixed_point_converged(fp) for fp in fixed_points]
    converged_false = sum(value is False for value in converged_values)
    passes = finite & (residuals <= float(residual_tol))
    fails = finite & ~passes
    return PoincareFixedPointValidation(
        n_total=int(len(fixed_points)),
        n_with_residual=int(np.count_nonzero(finite)),
        n_converged_false=int(converged_false),
        n_residual_pass=int(np.count_nonzero(passes)),
        n_residual_fail=int(np.count_nonzero(fails)),
        max_residual=float(np.nanmax(residuals[finite])) if np.any(finite) else np.nan,
    )


def _chain_mode(chain) -> tuple[int, int] | None:
    try:
        if isinstance(chain, Mapping):
            return int(chain["m"]), int(chain["n"])
        return int(getattr(chain, "m")), int(getattr(chain, "n"))
    except Exception:
        return None


def _chain_radial_label(chain) -> float:
    for name in ("radial_label", "S_res", "s_res"):
        if hasattr(chain, name):
            return float(getattr(chain, name))
    if isinstance(chain, Mapping):
        for name in ("radial_label", "S_res", "s_res"):
            if name in chain:
                return float(chain[name])
    return np.nan


def _chain_half_width(chain) -> float:
    if isinstance(chain, Mapping):
        return float(chain.get("half_width", 0.0))
    return float(getattr(chain, "half_width", 0.0))


def _phase_response_attr(response, *names, default=None):
    if isinstance(response, Mapping):
        for name in names:
            if name in response:
                return response[name]
        return default
    for name in names:
        if hasattr(response, name):
            return getattr(response, name)
    return default


def _phase_response_kind(response) -> str:
    kind = _phase_response_attr(response, "kind", "point_kind", default=None)
    if kind is None:
        if bool(_phase_response_attr(response, "is_hyperbolic", default=False)):
            kind = "X"
        elif bool(_phase_response_attr(response, "is_elliptic", default=False)):
            kind = "O"
    normalized = "" if kind is None else str(kind).strip().upper()
    if normalized not in {"X", "O"}:
        raise ValueError("each phase response must identify kind='X' or kind='O'")
    return normalized


def _phase_response_coordinates(response) -> tuple[float, float, float, float]:
    s0 = _phase_response_attr(response, "s0", default=None)
    theta0 = _phase_response_attr(response, "theta_star0", default=None)
    if theta0 is None:
        z0 = _phase_response_attr(response, "z0", default=None)
        if z0 is not None:
            coordinates = np.asarray(z0, dtype=float)
            if coordinates.shape == (2,):
                theta0 = coordinates[1]
    dtheta = _phase_response_attr(
        response,
        "delta_theta_star_wrapped",
        "delta_theta_star",
        "delta_theta",
        default=None,
    )
    delta_s = _phase_response_attr(response, "delta_s", default=None)
    try:
        values = np.asarray([s0, theta0, dtheta, delta_s], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "phase response must provide finite s0, healed theta*0, delta theta*, and delta s"
        ) from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "phase response must provide finite s0, healed theta*0, delta theta*, and delta s"
        )
    return tuple(float(value) for value in values)


def sample_fixed_s_theta_curve(
    chart,
    *,
    s: float,
    theta0: float,
    delta_theta: float,
    n_points: int = 65,
) -> np.ndarray:
    """Sample a healed-chart curve from ``theta0`` to ``theta0+delta_theta``.

    The radial coordinate is held at the supplied ``s``.  This is also a
    constant-``psi`` curve because ``psi=s**2``.  The chart must expose an
    explicit ``s_theta_to_x`` method; no geometric polar angle is inferred.
    """

    mapper = getattr(chart, "s_theta_to_x", None)
    if not callable(mapper):
        raise TypeError("chart must expose callable s_theta_to_x")
    metadata = getattr(chart, "metadata", None)
    coordinate_choice = None if not isinstance(metadata, Mapping) else metadata.get("coordinate_choice")
    if coordinate_choice not in {"z=(s, theta*)", "z=(psi=s^2, theta*)"}:
        raise ValueError("chart metadata must state its healed phase coordinate choice")
    radial = float(s)
    start = float(theta0)
    angular_shift = float(delta_theta)
    if not np.all(np.isfinite([radial, start, angular_shift])) or radial < 0.0:
        raise ValueError("s, theta0, and delta_theta must be finite, with s non-negative")
    count = int(n_points)
    if count < 3:
        raise ValueError("n_points must be at least 3 for a curved arrow")
    theta = np.linspace(start, start + angular_shift, count)
    s_theta = np.column_stack([np.full(count, radial), theta])
    path = np.asarray(mapper(s_theta), dtype=float)
    if path.shape != (count, 2) or not np.all(np.isfinite(path)):
        raise ValueError("chart.s_theta_to_x must return finite shape (n_points, 2)")
    return path


def poincare_phase_response_radial_deltas(
    responses: Sequence[object],
    *,
    include_x: bool = True,
    include_o: bool = True,
    require_phase_space_valid: bool = True,
) -> dict[str, np.ndarray]:
    """Report radial ``delta s`` separately from fixed-``s`` angular arrows."""

    selected = ({"X"} if include_x else set()) | ({"O"} if include_o else set())
    grouped: dict[str, list[float]] = {kind: [] for kind in ("X", "O") if kind in selected}
    for response in responses:
        kind = _phase_response_kind(response)
        if kind not in selected:
            continue
        valid = bool(_phase_response_attr(response, "phase_space_valid", default=False))
        if require_phase_space_valid and not valid:
            continue
        _s0, _theta0, _dtheta, delta_s = _phase_response_coordinates(response)
        grouped[kind].append(delta_s)
    return {kind: np.asarray(values, dtype=float) for kind, values in grouped.items()}


def draw_poincare_phase_response_arrows(
    ax,
    responses: Sequence[object],
    chart,
    *,
    draw_x: bool = True,
    draw_o: bool = True,
    require_phase_space_valid: bool = True,
    n_points: int = 65,
    x_color: str = "#c43b4d",
    o_color: str = "#176b87",
    linewidth: float = 2.0,
    mutation_scale: float = 12.0,
    alpha: float = 0.95,
    label_x: str = "X phase response",
    label_o: str = "O phase response",
    zorder: float = 6.0,
):
    """Draw fixed-``s`` healed-theta response arrows for X and/or O points.

    The arrow path represents only ``delta theta*``.  Radial ``delta s`` is
    intentionally omitted from the geometry and is available through
    :func:`poincare_phase_response_radial_deltas` or the response object.
    """

    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path

    selected = ({"X"} if draw_x else set()) | ({"O"} if draw_o else set())
    colors = {"X": x_color, "O": o_color}
    labels = {"X": str(label_x), "O": str(label_o)}
    labelled: set[str] = set()
    artists = []
    for response in responses:
        kind = _phase_response_kind(response)
        if kind not in selected:
            continue
        valid = bool(_phase_response_attr(response, "phase_space_valid", default=False))
        if require_phase_space_valid and not valid:
            continue
        s0, theta0, delta_theta, delta_s = _phase_response_coordinates(response)
        if abs(delta_theta) <= np.finfo(float).eps:
            continue
        path_RZ = sample_fixed_s_theta_curve(
            chart,
            s=s0,
            theta0=theta0,
            delta_theta=delta_theta,
            n_points=n_points,
        )
        codes = np.full(path_RZ.shape[0], Path.LINETO, dtype=np.uint8)
        codes[0] = Path.MOVETO
        patch = FancyArrowPatch(
            path=Path(path_RZ, codes),
            arrowstyle="-|>",
            mutation_scale=float(mutation_scale),
            linewidth=float(linewidth),
            color=colors[kind],
            alpha=float(alpha),
            zorder=float(zorder),
            label=labels[kind] if kind not in labelled else None,
        )
        patch.phase_response_kind = kind
        patch.radial_delta_s = delta_s
        patch.sampled_path_RZ = path_RZ.copy()
        ax.add_patch(patch)
        artists.append(patch)
        labelled.add(kind)
    return artists


def _bar_attr(bar, *names, default=None):
    if isinstance(bar, Mapping):
        for name in names:
            if name in bar:
                return bar[name]
        return default
    for name in names:
        if hasattr(bar, name):
            return getattr(bar, name)
    return default


def _bar_path(bar) -> tuple[np.ndarray, np.ndarray]:
    R_path = _bar_attr(bar, "R_path", "R")
    Z_path = _bar_attr(bar, "Z_path", "Z")
    if R_path is not None and Z_path is not None:
        R_line = np.asarray(R_path, dtype=float).ravel()
        Z_line = np.asarray(Z_path, dtype=float).ravel()
    else:
        R_line = np.asarray([_bar_attr(bar, "R_inner"), _bar_attr(bar, "R_outer")], dtype=float)
        Z_line = np.asarray([_bar_attr(bar, "Z_inner"), _bar_attr(bar, "Z_outer")], dtype=float)
    if R_line.shape != Z_line.shape:
        raise ValueError("island bar R_path and Z_path must have matching shapes")
    if R_line.size < 2:
        raise ValueError("island bar path must contain at least two points")
    finite = np.isfinite(R_line) & np.isfinite(Z_line)
    return R_line[finite], Z_line[finite]


def _bar_mode(bar) -> tuple[int, int] | None:
    mode = _bar_attr(bar, "mode")
    if mode is not None and len(mode) >= 2:
        return int(mode[0]), int(mode[1])
    m = _bar_attr(bar, "mode_m", "m")
    n = _bar_attr(bar, "mode_n", "n")
    if m is not None and n is not None:
        return int(m), int(n)
    chain = _bar_attr(bar, "chain")
    if chain is not None:
        return _chain_mode(chain)
    return None


def _bar_half_width(bar) -> float:
    width = _bar_attr(bar, "half_width", default=None)
    if width is not None:
        return float(width)
    chain = _bar_attr(bar, "chain")
    if chain is not None:
        return _chain_half_width(chain)
    s_inner = _bar_attr(bar, "s_inner", default=None)
    s_outer = _bar_attr(bar, "s_outer", default=None)
    if s_inner is not None and s_outer is not None:
        return 0.5 * abs(float(s_outer) - float(s_inner))
    return np.nan


def _bar_phase(bar) -> float:
    phase = _bar_attr(bar, "phase", default=None)
    if phase is not None:
        return float(phase)
    chain = _bar_attr(bar, "chain")
    if chain is not None and hasattr(chain, "phase"):
        return float(getattr(chain, "phase"))
    return np.nan


def _bar_kind(bar) -> str | None:
    kind = _bar_attr(bar, "kind", default=None)
    return None if kind is None else str(kind).upper()


def _bar_branch(bar) -> int | None:
    branch = _bar_attr(bar, "branch", default=None)
    if branch is None:
        return None
    try:
        return int(branch)
    except Exception:
        return None


def _bar_amplitude(bar) -> float:
    amplitude = _bar_attr(bar, "amplitude", "b_res", "tilde_b", "tilde_b_mn", default=None)
    if amplitude is not None:
        return float(np.abs(amplitude))
    chain = _bar_attr(bar, "chain")
    if chain is not None and hasattr(chain, "b_res"):
        return float(np.abs(getattr(chain, "b_res")))
    return np.nan


def _bar_label(bar, label_prefix: str) -> str:
    explicit = _bar_attr(bar, "label", default=None)
    if explicit:
        return str(explicit)
    mode = _bar_mode(bar)
    width = _bar_half_width(bar)
    phase = _bar_phase(bar)
    parts = [str(label_prefix)]
    if mode is not None:
        parts.append(f"({mode[0]},{mode[1]})")
    if np.isfinite(width):
        parts.append(f"w={width:.2e}")
    if np.isfinite(phase):
        parts.append(f"phase={np.degrees(phase):.1f} deg")
    return " ".join(parts)


def _apply_line_halo(line, *, linewidth: float, color: str = "white", alpha: float = 0.84):
    import matplotlib.patheffects as pe

    line.set_path_effects(
        [
            pe.Stroke(linewidth=float(linewidth), foreground=color, alpha=float(alpha)),
            pe.Normal(),
        ]
    )


def draw_poincare_curved_island_bars(
    ax,
    island_bars: Sequence[object],
    *,
    colors: Sequence[str] | None = None,
    mode_colors: Mapping[tuple[int, int], str | tuple[float, ...]] | None = None,
    mode_colormaps: Mapping[tuple[int, int], str] | None = None,
    grayscale: bool = False,
    grayscale_color: str = "#9ca3af",
    linewidth: float = 3.0,
    alpha: float = 0.96,
    halo: bool = True,
    halo_linewidth: float | None = None,
    halo_color: str = "white",
    label_prefix: str = "spectrum bar",
    show_labels: bool = True,
    endpoint_markers: bool = True,
    marker_size: float = 16.0,
    zorder: float = 4.6,
):
    """Draw curved spectrum-predicted island-width bars on a Poincare axis."""

    import matplotlib.pyplot as plt

    palette = tuple(colors or ("#2ab7ca", "#f4a261", "#7d5fff", "#118ab2", "#ef476f"))
    artists = []
    labelled: set[tuple[int, int] | int] = set()
    color_order: dict[tuple[int, int] | int, int] = {}
    for idx, bar in enumerate(island_bars):
        R_line, Z_line = _bar_path(bar)
        if R_line.size < 2:
            continue
        mode = _bar_mode(bar)
        key: tuple[int, int] | int = mode if mode is not None else idx
        if key not in color_order:
            color_order[key] = len(color_order)
        explicit_color = _bar_attr(bar, "color", default=None)
        explicit_cmap = _bar_attr(bar, "colormap", default=None)
        if grayscale:
            color = grayscale_color
        elif explicit_color is not None:
            color = explicit_color
        elif mode is not None and mode_colors is not None and mode in mode_colors:
            color = mode_colors[mode]
        else:
            cmap_name = None
            if explicit_cmap is not None:
                cmap_name = str(explicit_cmap)
            elif mode is not None and mode_colormaps is not None and mode in mode_colormaps:
                cmap_name = str(mode_colormaps[mode])
            if cmap_name is not None:
                cmap = plt.get_cmap(cmap_name)
                branch = _bar_branch(bar)
                denom = max(abs(int(mode[0])) if mode is not None else len(island_bars), 1)
                frac = 0.5 if branch is None else (float(branch) + 0.5) / float(denom)
                color = cmap(0.22 + 0.68 * (frac % 1.0))
            else:
                color = palette[color_order[key] % len(palette)]
        label = None
        if show_labels and key not in labelled:
            label = _bar_label(bar, label_prefix)
            labelled.add(key)
        (line,) = ax.plot(
            R_line,
            Z_line,
            color=color,
            lw=float(linewidth),
            alpha=float(alpha),
            solid_capstyle="round",
            zorder=zorder,
            label=label,
        )
        if halo:
            _apply_line_halo(
                line,
                linewidth=float(halo_linewidth if halo_linewidth is not None else linewidth + 2.4),
                color=halo_color,
                alpha=0.78,
            )
        artists.append(line)
        if endpoint_markers:
            sc = ax.scatter(
                [R_line[0], R_line[-1]],
                [Z_line[0], Z_line[-1]],
                s=float(marker_size),
                color=color,
                edgecolors=halo_color,
                linewidths=0.55,
                alpha=float(alpha),
                zorder=zorder + 0.2,
            )
            artists.append(sc)
    return artists


def draw_poincare_island_secondary_points(
    ax,
    R,
    Z,
    secondary_coordinates: PoincareIslandSecondaryCoordinates,
    *,
    color_by: str = "rho",
    mode_colormaps: Mapping[tuple[int, int], str] | None = None,
    mode_colors: Mapping[tuple[int, int], str | tuple[float, ...]] | None = None,
    default_colormap: str = "viridis",
    grayscale: bool = False,
    grayscale_color: str = "#9ca3af",
    point_size: float = 1.4,
    alpha: float = 0.82,
    rho_limits: tuple[float, float] = (0.0, 1.0),
    zorder: float = 3.0,
    rasterized: bool = True,
):
    """Draw island-chain points colored by secondary island minor radius."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    R_arr = np.asarray(R, dtype=float).ravel()
    Z_arr = np.asarray(Z, dtype=float).ravel()
    if R_arr.shape != Z_arr.shape:
        raise ValueError("R and Z must have matching shape")
    if secondary_coordinates.secondary_radius.size != R_arr.size:
        raise ValueError("secondary coordinate arrays must match R/Z point count")

    inside = np.asarray(secondary_coordinates.inside_island, dtype=bool)
    artists = []
    modes = sorted({
        (int(m), int(n))
        for m, n, keep in zip(
            secondary_coordinates.mode_m,
            secondary_coordinates.mode_n,
            inside,
        )
        if keep and int(m) >= 0 and int(n) >= 0
    })
    color_key = str(color_by).lower()
    if color_key not in {"rho", "secondary_radius", "phase", "helical_phase", "hamiltonian", "mode", "branch"}:
        raise ValueError("color_by must be one of 'rho', 'phase', 'hamiltonian', 'mode', or 'branch'")
    norm = Normalize(vmin=float(rho_limits[0]), vmax=float(rho_limits[1]))
    for mode in modes:
        mask = (
            inside
            & (secondary_coordinates.mode_m == mode[0])
            & (secondary_coordinates.mode_n == mode[1])
            & np.isfinite(secondary_coordinates.secondary_radius)
        )
        if not np.any(mask):
            continue
        if grayscale:
            artist = ax.scatter(
                R_arr[mask],
                Z_arr[mask],
                s=float(point_size),
                c=grayscale_color,
                alpha=float(alpha),
                linewidths=0.0,
                rasterized=bool(rasterized),
                zorder=zorder,
            )
        elif color_key == "mode" and mode_colors is not None and mode in mode_colors:
            artist = ax.scatter(
                R_arr[mask],
                Z_arr[mask],
                s=float(point_size),
                c=mode_colors[mode],
                alpha=float(alpha),
                linewidths=0.0,
                rasterized=bool(rasterized),
                zorder=zorder,
            )
        else:
            cmap_name = default_colormap
            if mode_colormaps is not None and mode in mode_colormaps:
                cmap_name = mode_colormaps[mode]
            if color_key in {"rho", "secondary_radius"}:
                values = secondary_coordinates.secondary_radius[mask]
                active_norm = norm
            elif color_key in {"phase", "helical_phase"}:
                values = secondary_coordinates.helical_phase[mask]
                active_norm = Normalize(vmin=-np.pi, vmax=np.pi)
                if default_colormap == "viridis" and (mode_colormaps is None or mode not in mode_colormaps):
                    cmap_name = "twilight"
            elif color_key == "hamiltonian":
                values = secondary_coordinates.island_hamiltonian[mask]
                vmax = max(1.0, float(np.nanmax(np.abs(values))) if np.any(np.isfinite(values)) else 1.0)
                active_norm = Normalize(vmin=-vmax, vmax=vmax)
                if default_colormap == "viridis" and (mode_colormaps is None or mode not in mode_colormaps):
                    cmap_name = "coolwarm"
            else:
                values = secondary_coordinates.branch[mask]
                active_norm = None
            artist = ax.scatter(
                R_arr[mask],
                Z_arr[mask],
                s=float(point_size),
                c=values,
                cmap=plt.get_cmap(cmap_name),
                norm=active_norm,
                alpha=float(alpha),
                linewidths=0.0,
                rasterized=bool(rasterized),
                zorder=zorder,
            )
        artists.append(artist)
    return artists


def draw_poincare_island_secondary_contours(
    ax,
    R,
    Z,
    secondary_coordinates: PoincareIslandSecondaryCoordinates,
    *,
    levels: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
    mode_colormaps: Mapping[tuple[int, int], str] | None = None,
    mode_colors: Mapping[tuple[int, int], str | tuple[float, ...]] | None = None,
    default_colormap: str = "viridis",
    grayscale: bool = False,
    grayscale_color: str = "#6b7280",
    alpha: float = 0.78,
    linewidth: float = 0.42,
    min_points: int = 16,
    max_triangle_edge: float | None = None,
    auto_triangle_edge_factor: float | None = None,
    zorder: float = 3.45,
):
    """Draw approximate secondary flux-surface contours inside islands.

    The contours are grouped by ``(m, n, branch)`` so triangulation does not
    connect separate island lobes.  ``secondary_radius`` is used as the contour
    scalar, so the lines represent constant small radius in the local island
    coordinate approximation.
    """

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.tri as mtri

    R_arr = np.asarray(R, dtype=float).ravel()
    Z_arr = np.asarray(Z, dtype=float).ravel()
    if R_arr.shape != Z_arr.shape:
        raise ValueError("R and Z must have matching shape")
    rho = np.asarray(secondary_coordinates.secondary_radius, dtype=float).ravel()
    if rho.size != R_arr.size:
        raise ValueError("secondary coordinate arrays must match R/Z point count")

    inside = np.asarray(secondary_coordinates.inside_island, dtype=bool).ravel()
    m_arr = np.asarray(secondary_coordinates.mode_m, dtype=int).ravel()
    n_arr = np.asarray(secondary_coordinates.mode_n, dtype=int).ravel()
    branch_arr = np.asarray(secondary_coordinates.branch, dtype=int).ravel()
    finite = inside & np.isfinite(R_arr) & np.isfinite(Z_arr) & np.isfinite(rho)
    finite &= (m_arr >= 0) & (n_arr >= 0) & (branch_arr >= 0)
    if not np.any(finite):
        return []

    contour_levels = np.asarray(levels, dtype=float).ravel()
    contour_levels = contour_levels[np.isfinite(contour_levels)]
    if contour_levels.size == 0:
        return []

    artists = []
    groups = sorted({(int(m), int(n), int(b)) for m, n, b in zip(m_arr[finite], n_arr[finite], branch_arr[finite])})
    norm = Normalize(vmin=float(np.nanmin(contour_levels)), vmax=float(np.nanmax(contour_levels)))
    for m, n, branch in groups:
        mask = finite & (m_arr == m) & (n_arr == n) & (branch_arr == branch)
        if np.count_nonzero(mask) < int(min_points):
            continue
        values = rho[mask]
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
        active_levels = contour_levels[(contour_levels >= lo) & (contour_levels <= hi)]
        if active_levels.size == 0 or np.unique(values).size < 3:
            continue
        triangulation = None
        if max_triangle_edge is not None or auto_triangle_edge_factor is not None:
            triangulation = mtri.Triangulation(R_arr[mask], Z_arr[mask])
            triangles = triangulation.triangles
            if triangles.size:
                points = np.column_stack([R_arr[mask], Z_arr[mask]])
                tri_points = points[triangles]
                edge01 = np.linalg.norm(tri_points[:, 0, :] - tri_points[:, 1, :], axis=1)
                edge12 = np.linalg.norm(tri_points[:, 1, :] - tri_points[:, 2, :], axis=1)
                edge20 = np.linalg.norm(tri_points[:, 2, :] - tri_points[:, 0, :], axis=1)
                edge_max = np.maximum(edge01, np.maximum(edge12, edge20))
                edge_limit = None if max_triangle_edge is None else float(max_triangle_edge)
                if auto_triangle_edge_factor is not None:
                    finite_edges = edge_max[np.isfinite(edge_max) & (edge_max > 0.0)]
                    if finite_edges.size:
                        auto_limit = float(auto_triangle_edge_factor) * float(np.nanmedian(finite_edges))
                        edge_limit = auto_limit if edge_limit is None else min(edge_limit, auto_limit)
                if edge_limit is not None and np.isfinite(edge_limit) and edge_limit > 0.0:
                    triangulation.set_mask(edge_max > edge_limit)
        mode = (m, n)
        contour_args = (triangulation, values) if triangulation is not None else (R_arr[mask], Z_arr[mask], values)
        try:
            if grayscale:
                contour = ax.tricontour(
                    *contour_args,
                    levels=active_levels,
                    colors=grayscale_color,
                    linewidths=float(linewidth),
                    alpha=float(alpha),
                    zorder=zorder,
                )
            elif mode_colors is not None and mode in mode_colors:
                contour = ax.tricontour(
                    *contour_args,
                    levels=active_levels,
                    colors=mode_colors[mode],
                    linewidths=float(linewidth),
                    alpha=float(alpha),
                    zorder=zorder,
                )
            else:
                cmap_name = default_colormap
                if mode_colormaps is not None and mode in mode_colormaps:
                    cmap_name = mode_colormaps[mode]
                contour = ax.tricontour(
                    *contour_args,
                    levels=active_levels,
                    cmap=plt.get_cmap(cmap_name),
                    norm=norm,
                    linewidths=float(linewidth),
                    alpha=float(alpha),
                    zorder=zorder,
                )
        except Exception:
            continue
        artists.append(contour)
    return artists


def draw_poincare_island_secondary_trace_lines(
    ax,
    R,
    Z,
    secondary_coordinates: PoincareIslandSecondaryCoordinates,
    trace_index=None,
    *,
    counts=None,
    color_by: str = "rho",
    mode_colormaps: Mapping[tuple[int, int], str] | None = None,
    mode_colors: Mapping[tuple[int, int], str | tuple[float, ...]] | None = None,
    default_colormap: str = "viridis",
    grayscale: bool = False,
    grayscale_color: str = "#6b7280",
    rho_limits: tuple[float, float] = (0.0, 1.0),
    linewidth: float = 0.34,
    alpha: float = 0.55,
    connect_by: str = "helical_phase",
    max_segment_length: float | None = None,
    max_segment_length_factor: float | None = None,
    min_points_per_trace: int = 3,
    zorder: float = 3.35,
    rasterized: bool = True,
):
    """Draw island secondary surfaces by connecting points within each trace.

    Unlike scattered triangulation contours, this function never connects
    points from different seed traces.  By default points within a trace are
    ordered by local helical phase, not by Poincare return order, because
    successive map iterates can jump across the section.  Segments are split
    whenever mode, island branch, validity, or an optional maximum segment
    length changes.
    """

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    R_arr = np.asarray(R, dtype=float).ravel()
    Z_arr = np.asarray(Z, dtype=float).ravel()
    if R_arr.shape != Z_arr.shape:
        raise ValueError("R and Z must have matching shape")
    n_points = R_arr.size
    if trace_index is None:
        if counts is None:
            raise ValueError("trace_index or counts is required for trace-wise secondary lines")
        trace_arr = poincare_trace_index_from_counts(counts, n_points=n_points)
    else:
        trace_arr = np.asarray(trace_index, dtype=int).ravel()
        if trace_arr.size != n_points:
            raise ValueError("trace_index must match R/Z point count")

    rho = np.asarray(secondary_coordinates.secondary_radius, dtype=float).ravel()
    inside = np.asarray(secondary_coordinates.inside_island, dtype=bool).ravel()
    m_arr = np.asarray(secondary_coordinates.mode_m, dtype=int).ravel()
    n_arr = np.asarray(secondary_coordinates.mode_n, dtype=int).ravel()
    branch_arr = np.asarray(secondary_coordinates.branch, dtype=int).ravel()
    if rho.size != n_points:
        raise ValueError("secondary coordinate arrays must match R/Z point count")

    valid = inside & np.isfinite(R_arr) & np.isfinite(Z_arr) & np.isfinite(rho)
    valid &= (m_arr >= 0) & (n_arr >= 0) & (branch_arr >= 0) & (trace_arr >= 0)
    if not np.any(valid):
        return []

    color_key = str(color_by).lower()
    if color_key not in {"rho", "secondary_radius", "phase", "helical_phase", "hamiltonian", "mode", "branch"}:
        raise ValueError("color_by must be one of 'rho', 'phase', 'hamiltonian', 'mode', or 'branch'")
    connect_key = str(connect_by).lower()
    if connect_key not in {"helical_phase", "phase", "point_order", "trace_order"}:
        raise ValueError("connect_by must be 'helical_phase' or 'point_order'")
    phase_arr = np.asarray(secondary_coordinates.helical_phase, dtype=float).ravel()
    ham_arr = np.asarray(secondary_coordinates.island_hamiltonian, dtype=float).ravel()

    artists = []
    modes = sorted({(int(m), int(n)) for m, n in zip(m_arr[valid], n_arr[valid])})
    for mode in modes:
        mode_mask = valid & (m_arr == mode[0]) & (n_arr == mode[1])
        if not np.any(mode_mask):
            continue
        segments = []
        values = []
        for trace_id in np.unique(trace_arr[mode_mask]):
            idx = np.nonzero(mode_mask & (trace_arr == trace_id))[0]
            if idx.size < int(min_points_per_trace):
                continue
            if connect_key in {"helical_phase", "phase"}:
                good_phase = np.isfinite(phase_arr[idx])
                idx = idx[good_phase]
                if idx.size < int(min_points_per_trace):
                    continue
                idx = idx[np.argsort(phase_arr[idx])]
            else:
                idx = idx[np.argsort(idx)]
            points = np.column_stack([R_arr[idx], Z_arr[idx]])
            dr = np.linalg.norm(np.diff(points, axis=0), axis=1)
            edge_limit = None if max_segment_length is None else float(max_segment_length)
            if max_segment_length_factor is not None and dr.size:
                finite_dr = dr[np.isfinite(dr) & (dr > 0.0)]
                if finite_dr.size:
                    auto_limit = float(max_segment_length_factor) * float(np.nanmedian(finite_dr))
                    edge_limit = auto_limit if edge_limit is None else min(edge_limit, auto_limit)
            same_branch = branch_arr[idx[:-1]] == branch_arr[idx[1:]]
            keep = same_branch & np.isfinite(dr)
            if edge_limit is not None and np.isfinite(edge_limit) and edge_limit > 0.0:
                keep &= dr <= edge_limit
            if not np.any(keep):
                continue
            for local_i in np.nonzero(keep)[0]:
                segments.append(points[local_i:local_i + 2])
                if color_key in {"rho", "secondary_radius"}:
                    values.append(0.5 * (rho[idx[local_i]] + rho[idx[local_i + 1]]))
                elif color_key in {"phase", "helical_phase"}:
                    values.append(0.5 * (phase_arr[idx[local_i]] + phase_arr[idx[local_i + 1]]))
                elif color_key == "hamiltonian":
                    values.append(0.5 * (ham_arr[idx[local_i]] + ham_arr[idx[local_i + 1]]))
                elif color_key == "branch":
                    values.append(float(branch_arr[idx[local_i]]))
                else:
                    values.append(float(modes.index(mode)))
        if not segments:
            continue
        if grayscale:
            collection = LineCollection(
                segments,
                colors=grayscale_color,
                linewidths=float(linewidth),
                alpha=float(alpha),
                rasterized=bool(rasterized),
                zorder=zorder,
            )
        elif color_key == "mode" and mode_colors is not None and mode in mode_colors:
            collection = LineCollection(
                segments,
                colors=mode_colors[mode],
                linewidths=float(linewidth),
                alpha=float(alpha),
                rasterized=bool(rasterized),
                zorder=zorder,
            )
        else:
            cmap_name = default_colormap
            if mode_colormaps is not None and mode in mode_colormaps:
                cmap_name = mode_colormaps[mode]
            if color_key in {"rho", "secondary_radius"}:
                norm = Normalize(vmin=float(rho_limits[0]), vmax=float(rho_limits[1]))
            elif color_key in {"phase", "helical_phase"}:
                norm = Normalize(vmin=-np.pi, vmax=np.pi)
                if default_colormap == "viridis" and (mode_colormaps is None or mode not in mode_colormaps):
                    cmap_name = "twilight"
            elif color_key == "hamiltonian":
                value_arr = np.asarray(values, dtype=float)
                vmax = max(1.0, float(np.nanmax(np.abs(value_arr))) if np.any(np.isfinite(value_arr)) else 1.0)
                norm = Normalize(vmin=-vmax, vmax=vmax)
                if default_colormap == "viridis" and (mode_colormaps is None or mode not in mode_colormaps):
                    cmap_name = "coolwarm"
            else:
                norm = None
            collection = LineCollection(
                segments,
                cmap=plt.get_cmap(cmap_name),
                norm=norm,
                linewidths=float(linewidth),
                alpha=float(alpha),
                rasterized=bool(rasterized),
                zorder=zorder,
            )
            collection.set_array(np.asarray(values, dtype=float))
        ax.add_collection(collection)
        artists.append(collection)
    return artists


def _growth_limits(values: np.ndarray, threshold: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float(threshold), float(threshold) + 1.0
    vmin = min(float(np.nanmin(finite)), float(threshold))
    vmax = float(np.nanpercentile(finite, 98.0))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + max(1.0e-6, abs(vmin) * 0.1 + 1.0e-6)
    return vmin, vmax


def _draw_radial_contours(
    ax,
    R: np.ndarray,
    Z: np.ndarray,
    radial: np.ndarray,
    *,
    levels=None,
    color: str = "#e4c65d",
    alpha: float = 0.44,
    linewidth: float = 0.75,
):
    finite = np.isfinite(R) & np.isfinite(Z) & np.isfinite(radial)
    if np.count_nonzero(finite) < 8:
        return None
    values = radial[finite]
    if np.unique(values).size < 3:
        return None
    if levels is None:
        lo, hi = np.nanpercentile(values, [10.0, 96.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return None
        levels = np.linspace(float(lo), float(hi), 7)
    else:
        levels = np.asarray(levels, dtype=float)
    levels = np.asarray([level for level in levels if np.nanmin(values) <= level <= np.nanmax(values)], dtype=float)
    if levels.size == 0:
        return None
    try:
        return ax.tricontour(
            R[finite],
            Z[finite],
            values,
            levels=levels,
            colors=color,
            linewidths=float(linewidth),
            alpha=float(alpha),
            zorder=0.9,
        )
    except Exception:
        return None


def _interval_mask(values: np.ndarray, intervals: Sequence[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(values.shape, dtype=bool)
    finite = np.isfinite(values)
    for lo, hi in intervals:
        mask |= finite & (values >= float(lo)) & (values <= float(hi))
    return mask


def plot_poincare_topology_map(
    R: Sequence[float],
    Z: Sequence[float],
    *,
    style: PoincareTopologyFigureStyle | None = None,
    radial_label: Sequence[float] | None = None,
    dpk_radial_labels: Sequence[float] | None = None,
    dpk_metrics: Sequence[object] | None = None,
    point_growth: Sequence[float] | None = None,
    point_spectral_recurrence_min: Sequence[float] | None = None,
    point_recurrent_surface_indicator: Sequence[float] | None = None,
    fixed_points: Sequence[object] = (),
    fixed_point_residual_tol: float | None = None,
    draw_unvalidated_fixed_points: bool = True,
    require_converged_fixed_points: bool = False,
    island_bars: Sequence[object] = (),
    island_bar_kwargs: Mapping[str, object] | None = None,
    secondary_coordinates: PoincareIslandSecondaryCoordinates | None = None,
    secondary_color_by: str = "rho",
    secondary_point_kwargs: Mapping[str, object] | None = None,
    secondary_trace_index: Sequence[int] | None = None,
    secondary_trace_counts: Sequence[int] | None = None,
    secondary_trace_lines: bool = False,
    secondary_trace_line_kwargs: Mapping[str, object] | None = None,
    secondary_contours: bool = False,
    secondary_contour_kwargs: Mapping[str, object] | None = None,
    secondary_colorbar: bool = False,
    ax=None,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    cmap: str = "magma",
    background_color: str = "0.78",
    background_alpha: float = 0.35,
    surface_color: str = "#2f7d6d",
    surface_alpha: float = 0.58,
    chaotic_alpha: float = 0.88,
    xpoint_color: str = "#c0392b",
    opoint_color: str = "#178f4a",
    point_size: float = 2.2,
    growth_threshold: float = 0.0,
    recurrence_threshold: float = 0.02,
    recurrent_surface_threshold: float = 0.5,
    show_background_points: bool = True,
    show_surface_points: bool = True,
    show_chaotic_points: bool = True,
    show_island_bars: bool = True,
    show_fixed_points: bool = True,
    colorbar: bool = True,
    legend: bool = True,
    legend_loc: str = "lower left",
    legend_frame: bool = False,
    summary_box: bool = True,
    radial_contours: bool = False,
    radial_contour_levels=None,
    radial_contour_color: str = "#e4c65d",
    radial_contour_alpha: float = 0.44,
    radial_contour_linewidth: float = 0.75,
    chaotic_band_shading: bool = False,
    chaotic_band_color: str = "#f3c54a",
    chaotic_band_alpha: float = 0.10,
    grid: bool = True,
    xlabel: str = "R",
    ylabel: str = "Z",
    title: str | None = "Poincare topology map",
):
    """Plot a Poincare section with DP^k-colored chaotic-layer points."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    if style is not None:
        if figsize is None:
            figsize = style.figsize
        if dpi is None:
            dpi = style.dpi
        if point_size == 2.2:
            point_size = style.point_size
        if secondary_point_kwargs is None:
            secondary_point_kwargs = {"point_size": style.island_point_size}
        if secondary_contours and secondary_contour_kwargs is None:
            secondary_contour_kwargs = {
                "levels": style.island_contour_levels,
                "linewidth": style.island_contour_linewidth,
                "alpha": style.island_contour_alpha,
            }
        if secondary_trace_lines and secondary_trace_line_kwargs is None:
            secondary_trace_line_kwargs = {
                "linewidth": style.island_contour_linewidth,
                "alpha": style.island_contour_alpha,
            }
    R_arr = np.asarray(R, dtype=float).ravel()
    Z_arr = np.asarray(Z, dtype=float).ravel()
    if R_arr.size != Z_arr.size:
        raise ValueError("R and Z must have the same length")
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(6.4, 6.0) if figsize is None else figsize,
            dpi=None if dpi is None else int(dpi),
            constrained_layout=True,
        )
    else:
        fig = ax.figure
    radial = np.full(R_arr.size, np.nan, dtype=float) if radial_label is None else _as_point_array(radial_label, R_arr.size, name="radial_label")
    classification = poincare_dpk_classification(
        radial,
        dpk_radial_labels=dpk_radial_labels,
        dpk_metrics=dpk_metrics,
        point_growth=point_growth,
        point_spectral_recurrence_min=point_spectral_recurrence_min,
        point_recurrent_surface_indicator=point_recurrent_surface_indicator,
        growth_threshold=growth_threshold,
        recurrence_threshold=recurrence_threshold,
        recurrent_surface_threshold=recurrent_surface_threshold,
    )

    finite = np.isfinite(R_arr) & np.isfinite(Z_arr)
    chaotic = finite & classification.point_chaotic_mask
    surface = finite & classification.point_surface_mask & ~chaotic
    background = finite & ~(chaotic | surface)
    if chaotic_band_shading and classification.chaotic_intervals:
        band = finite & _interval_mask(radial, classification.chaotic_intervals)
        if np.any(band):
            ax.scatter(
                R_arr[band],
                Z_arr[band],
                s=float(point_size) * 4.8,
                c=chaotic_band_color,
                alpha=float(chaotic_band_alpha),
                linewidths=0.0,
                rasterized=True,
                zorder=0.5,
            )
    if radial_contours:
        _draw_radial_contours(
            ax,
            R_arr,
            Z_arr,
            radial,
            levels=radial_contour_levels,
            color=radial_contour_color,
            alpha=radial_contour_alpha,
            linewidth=radial_contour_linewidth,
        )
    if show_background_points and np.any(background):
        ax.scatter(
            R_arr[background],
            Z_arr[background],
            s=float(point_size),
            c=background_color,
            alpha=float(background_alpha),
            linewidths=0.0,
            rasterized=True,
            label="unclassified",
            zorder=1,
        )
    if show_surface_points and np.any(surface):
        ax.scatter(
            R_arr[surface],
            Z_arr[surface],
            s=float(point_size) * 1.15,
            c=surface_color,
            alpha=float(surface_alpha),
            linewidths=0.0,
            rasterized=True,
            label="recurrent closed surface",
            zorder=2,
        )

    scatter = None
    if show_chaotic_points and np.any(chaotic):
        vmin, vmax = _growth_limits(classification.point_growth[chaotic], growth_threshold)
        scatter = ax.scatter(
            R_arr[chaotic],
            Z_arr[chaotic],
            s=float(point_size) * 1.4,
            c=classification.point_growth[chaotic],
            cmap=cmap,
            norm=Normalize(vmin=vmin, vmax=vmax),
            alpha=float(chaotic_alpha),
            linewidths=0.0,
            rasterized=True,
            label="chaotic layer",
            zorder=3,
        )
        if colorbar:
            fig.colorbar(scatter, ax=ax, shrink=0.84, pad=0.02, label="DP^k eigenvalue growth")

    secondary_artists = []
    if secondary_coordinates is not None:
        secondary_kwargs = {} if secondary_point_kwargs is None else dict(secondary_point_kwargs)
        secondary_kwargs.setdefault("color_by", secondary_color_by)
        secondary_artists = draw_poincare_island_secondary_points(ax, R_arr, Z_arr, secondary_coordinates, **secondary_kwargs)
        if secondary_colorbar:
            for artist in secondary_artists:
                try:
                    if artist.get_array() is not None:
                        cbar = fig.colorbar(artist, ax=ax, shrink=0.70, pad=0.025)
                        label = "secondary island radius rho_i"
                        if str(secondary_color_by).lower() in {"phase", "helical_phase"}:
                            label = "secondary island helical phase"
                        elif str(secondary_color_by).lower() == "hamiltonian":
                            label = "secondary island Hamiltonian coordinate"
                        cbar.set_label(label)
                        break
                except Exception:
                    continue
        if secondary_contours:
            contour_kwargs = {} if secondary_contour_kwargs is None else dict(secondary_contour_kwargs)
            draw_poincare_island_secondary_contours(ax, R_arr, Z_arr, secondary_coordinates, **contour_kwargs)
        if secondary_trace_lines:
            trace_kwargs = {} if secondary_trace_line_kwargs is None else dict(secondary_trace_line_kwargs)
            trace_kwargs.setdefault("color_by", secondary_color_by)
            draw_poincare_island_secondary_trace_lines(
                ax,
                R_arr,
                Z_arr,
                secondary_coordinates,
                trace_index=secondary_trace_index,
                counts=secondary_trace_counts,
                **trace_kwargs,
            )

    if show_island_bars and island_bars:
        kwargs = {} if island_bar_kwargs is None else dict(island_bar_kwargs)
        draw_poincare_curved_island_bars(ax, island_bars, **kwargs)

    seen_o = False
    seen_x = False
    if show_fixed_points:
        for fp in fixed_points:
            if not _fixed_point_passes_validation(
                fp,
                residual_tol=fixed_point_residual_tol,
                draw_unvalidated=draw_unvalidated_fixed_points,
                require_converged=require_converged_fixed_points,
            ):
                continue
            item = _fixed_point_rz_kind(fp)
            if item is None:
                continue
            R0, Z0, kind = item
            if kind.startswith("O"):
                ax.scatter(
                    [R0],
                    [Z0],
                    s=50,
                    facecolors="none",
                    edgecolors=opoint_color,
                    linewidths=1.4,
                    marker="o",
                    label="O point" if not seen_o else None,
                    zorder=5,
                )
                seen_o = True
            elif kind.startswith("X"):
                ax.scatter(
                    [R0],
                    [Z0],
                    s=62,
                    c=xpoint_color,
                    linewidths=1.5,
                    marker="x",
                    label="X point" if not seen_x else None,
                    zorder=5,
                )
                seen_x = True

    if summary_box and finite.size:
        n_finite = max(1, int(np.count_nonzero(finite)))
        text = (
            f"chaotic points: {np.count_nonzero(chaotic) / n_finite:.1%}\n"
            f"closed-surface points: {np.count_nonzero(surface) / n_finite:.1%}"
        )
        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color="0.18",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.82", "alpha": 0.88},
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if grid:
        ax.grid(color="0.91", linewidth=0.55)
    if title:
        ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if legend and handles:
        unique = {}
        for handle, label in zip(handles, labels):
            if label and label not in unique:
                unique[label] = handle
        legend_obj = ax.legend(unique.values(), unique.keys(), loc=legend_loc, fontsize=8, frameon=legend_frame)
        if legend_frame:
            frame = legend_obj.get_frame()
            frame.set_facecolor("white")
            frame.set_edgecolor("0.80")
            frame.set_alpha(0.88)
    return fig, ax, classification, scatter


def plot_poincare_topology_report(
    R: Sequence[float],
    Z: Sequence[float],
    *,
    style: PoincareTopologyFigureStyle | None = None,
    radial_label: Sequence[float],
    dpk_radial_labels: Sequence[float],
    dpk_metrics: Sequence[object],
    island_chains: Sequence[object] = (),
    island_bars: Sequence[object] = (),
    secondary_coordinates: PoincareIslandSecondaryCoordinates | None = None,
    secondary_color_by: str = "rho",
    secondary_point_kwargs: Mapping[str, object] | None = None,
    secondary_trace_index: Sequence[int] | None = None,
    secondary_trace_counts: Sequence[int] | None = None,
    secondary_trace_lines: bool = False,
    secondary_trace_line_kwargs: Mapping[str, object] | None = None,
    secondary_contours: bool = False,
    secondary_contour_kwargs: Mapping[str, object] | None = None,
    secondary_colorbar: bool = False,
    fixed_points: Sequence[object] = (),
    fixed_point_residual_tol: float | None = None,
    draw_unvalidated_fixed_points: bool = True,
    require_converged_fixed_points: bool = False,
    point_growth: Sequence[float] | None = None,
    point_spectral_recurrence_min: Sequence[float] | None = None,
    point_recurrent_surface_indicator: Sequence[float] | None = None,
    cmap: str = "magma",
    growth_threshold: float = 0.0,
    recurrence_threshold: float = 0.02,
    recurrent_surface_threshold: float = 0.5,
    point_size: float = 1.7,
    radial_contours: bool = True,
    chaotic_band_shading: bool = True,
    summary_box: bool = False,
    legend: bool = True,
    legend_loc: str = "upper right",
    legend_frame: bool = True,
    horizontal_colorbar: bool = True,
    xlabel: str = "R",
    ylabel: str = "Z",
    title: str | None = "Poincare boundary-topology report",
    island_bar_kwargs: Mapping[str, object] | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
):
    """Plot a Poincare-centered report with DP^k and island-chain context."""

    import matplotlib.pyplot as plt

    if style is not None:
        if dpi is None:
            dpi = style.dpi
        if point_size == 1.7:
            point_size = style.point_size
        if secondary_point_kwargs is None:
            secondary_point_kwargs = {"point_size": style.island_point_size}
        if secondary_contours and secondary_contour_kwargs is None:
            secondary_contour_kwargs = {
                "levels": style.island_contour_levels,
                "linewidth": style.island_contour_linewidth,
                "alpha": style.island_contour_alpha,
            }
        if secondary_trace_lines and secondary_trace_line_kwargs is None:
            secondary_trace_line_kwargs = {
                "linewidth": style.island_contour_linewidth,
                "alpha": style.island_contour_alpha,
            }
    fig = plt.figure(
        figsize=(12.0, 7.2) if figsize is None else figsize,
        dpi=None if dpi is None else int(dpi),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(2, 3, width_ratios=(1.25, 1.25, 0.88))
    ax_map = fig.add_subplot(gs[:, :2])
    ax_growth = fig.add_subplot(gs[0, 2])
    ax_recur = fig.add_subplot(gs[1, 2])

    _fig, _ax, classification, _scatter = plot_poincare_topology_map(
        R,
        Z,
        radial_label=radial_label,
        dpk_radial_labels=dpk_radial_labels,
        dpk_metrics=dpk_metrics,
        point_growth=point_growth,
        point_spectral_recurrence_min=point_spectral_recurrence_min,
        point_recurrent_surface_indicator=point_recurrent_surface_indicator,
        fixed_points=fixed_points,
        fixed_point_residual_tol=fixed_point_residual_tol,
        draw_unvalidated_fixed_points=draw_unvalidated_fixed_points,
        require_converged_fixed_points=require_converged_fixed_points,
        island_bars=island_bars,
        island_bar_kwargs=island_bar_kwargs,
        secondary_coordinates=secondary_coordinates,
        secondary_color_by=secondary_color_by,
        secondary_point_kwargs=secondary_point_kwargs,
        secondary_trace_index=secondary_trace_index,
        secondary_trace_counts=secondary_trace_counts,
        secondary_trace_lines=secondary_trace_lines,
        secondary_trace_line_kwargs=secondary_trace_line_kwargs,
        secondary_contours=secondary_contours,
        secondary_contour_kwargs=secondary_contour_kwargs,
        secondary_colorbar=secondary_colorbar,
        ax=ax_map,
        style=None,
        cmap=cmap,
        point_size=point_size,
        growth_threshold=growth_threshold,
        recurrence_threshold=recurrence_threshold,
        recurrent_surface_threshold=recurrent_surface_threshold,
        colorbar=not horizontal_colorbar,
        legend=legend,
        legend_loc=legend_loc,
        legend_frame=legend_frame,
        summary_box=summary_box,
        radial_contours=radial_contours,
        chaotic_band_shading=chaotic_band_shading,
        xlabel=xlabel,
        ylabel=ylabel,
        title="Poincare section",
    )

    radial = classification.radial_labels
    for lo, hi in classification.chaotic_intervals:
        ax_growth.axhspan(lo, hi, color="#f3c54a", alpha=0.18, lw=0)
        ax_recur.axhspan(lo, hi, color="#f3c54a", alpha=0.18, lw=0)
    ax_growth.plot(classification.eigenvalue_growth, radial, color="#8f1d5b", lw=1.8, label="growth")
    ax_growth.axvline(float(growth_threshold), color="0.35", lw=0.9, ls="--")
    ax_growth.set_xlabel(r"$\lambda_{eig}$")
    ax_growth.set_ylabel("radial label")
    ax_growth.set_title("growth")
    ax_growth.grid(color="0.90", linewidth=0.45)

    max_growth = float(np.nanmax(classification.eigenvalue_growth)) if radial.size else 0.0
    label_x = max(max_growth, float(growth_threshold)) * 0.92
    for chain in island_chains:
        chain_radial = _chain_radial_label(chain)
        if not np.isfinite(chain_radial):
            continue
        mode = _chain_mode(chain)
        ax_growth.axhline(chain_radial, color="#3e5c76", alpha=0.35, lw=0.9)
        if mode is not None:
            ax_growth.annotate(
                f"({mode[0]},{mode[1]})",
                xy=(label_x, chain_radial),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                color="#334f68",
                va="bottom",
            )

    ax_recur.plot(
        classification.spectral_recurrence_min,
        radial,
        color="#2f7d6d",
        lw=1.6,
        label="recurrence min",
    )
    ax_recur.axvline(float(recurrence_threshold), color="0.35", lw=0.9, ls="--")
    ax_surface = ax_recur.twiny()
    ax_surface.plot(
        classification.recurrent_surface_indicator,
        radial,
        color="#415a77",
        lw=1.1,
        alpha=0.76,
        label="surface indicator",
    )
    ax_recur.set_xlabel(r"$\delta_{min}$")
    ax_recur.set_ylabel("radial label")
    ax_surface.set_xlabel("surface indicator")
    ax_recur.grid(color="0.90", linewidth=0.45)

    handles, labels = ax_recur.get_legend_handles_labels()
    handles2, labels2 = ax_surface.get_legend_handles_labels()
    ax_recur.legend(handles + handles2, labels + labels2, loc="best", fontsize=8, frameon=False)
    if horizontal_colorbar and _scatter is not None:
        cbar = fig.colorbar(
            _scatter,
            ax=[ax_map, ax_growth, ax_recur],
            orientation="horizontal",
            shrink=0.82,
            pad=0.065,
        )
        cbar.set_label(r"chaotic-layer color: DP$^k$ finite-return eigenvalue-growth exponent $\lambda_{eig}$")
    if title:
        fig.suptitle(title)
    axes = {"poincare": ax_map, "growth": ax_growth, "recurrence": ax_recur, "surface": ax_surface}
    return fig, axes, classification


def plot_poincare_topology_payload_map(
    payload: PoincareTopologyReportPayload,
    **kwargs,
):
    """Plot a Poincare topology map from a validated report payload."""

    plot_kwargs = payload.map_kwargs()
    plot_kwargs.update(kwargs)
    return plot_poincare_topology_map(payload.R, payload.Z, **plot_kwargs)


def plot_poincare_topology_payload_report(
    payload: PoincareTopologyReportPayload,
    **kwargs,
):
    """Plot a Poincare topology report from a validated report payload."""

    plot_kwargs = payload.report_kwargs()
    plot_kwargs.update(kwargs)
    return plot_poincare_topology_report(payload.R, payload.Z, **plot_kwargs)


__all__ = [
    "PoincareCurvedIslandBar",
    "PoincareDPKClassification",
    "PoincareFixedPointValidation",
    "PoincareIslandSecondaryCoordinates",
    "PoincareTopologyReportPayload",
    "PoincareTopologyFigureStyle",
    "PoincareTraceQuality",
    "circular_flux_coordinate_island_bars",
    "draw_poincare_curved_island_bars",
    "draw_poincare_phase_response_arrows",
    "draw_poincare_island_secondary_contours",
    "draw_poincare_island_secondary_points",
    "draw_poincare_island_secondary_trace_lines",
    "fixed_point_phase_comparison_markers",
    "fixed_point_centered_island_bars",
    "plot_poincare_topology_map",
    "plot_poincare_topology_payload_map",
    "plot_poincare_topology_payload_report",
    "plot_poincare_topology_report",
    "poincare_phase_response_radial_deltas",
    "poincare_island_secondary_coordinates",
    "poincare_topology_report_payload",
    "poincare_trace_index_from_counts",
    "poincare_fixed_point_validation",
    "poincare_dpk_classification",
    "poincare_trace_quality",
    "sample_fixed_s_theta_curve",
]
