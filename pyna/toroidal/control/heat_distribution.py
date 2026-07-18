"""Heat-distribution control helpers for boundary-divertor workflows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from pyna.toroidal.control.boundary_topology_design import (
    BoundaryLinearResponseSystem,
    BoundaryResponseObservables,
    BoundaryResponseSolveResult,
    boundary_response_observables,
    finite_difference_boundary_response_system,
)


@dataclass(frozen=True)
class FieldLineDiffusionSpec:
    """Simple wall-heat diffusion proxy for field-line diffusion workflows.

    ``sigma_phi_bins`` and ``sigma_s_bins`` are Gaussian diffusion widths in the
    wall heat-map bin coordinates.  This does not replace full diffusive field
    line tracing; it is a lightweight surrogate for optimization loops and for
    consuming heat maps produced by external tracers.
    """

    sigma_phi_bins: float = 0.0
    sigma_s_bins: float = 0.0
    periodic_phi: bool = True
    periodic_s: bool = False
    preserve_total: bool = True

    @classmethod
    def from_parallel_diffusion(
        cls,
        diffusion_coefficient: float,
        connection_length: float,
        *,
        phi_bin_width: float,
        s_bin_width: float,
        phi_scale: float = 1.0,
        s_scale: float = 1.0,
        periodic_phi: bool = True,
        periodic_s: bool = False,
        preserve_total: bool = True,
    ) -> "FieldLineDiffusionSpec":
        """Build bin-space widths from a diffusive tracing estimate.

        The RMS spread is estimated as ``sqrt(2 D L)`` and converted into bin
        units with separate scale factors for the toroidal and wall-arclength
        directions.
        """

        spread = float(np.sqrt(max(0.0, 2.0 * float(diffusion_coefficient) * float(connection_length))))
        phi_width = max(abs(float(phi_bin_width)), 1.0e-300)
        s_width = max(abs(float(s_bin_width)), 1.0e-300)
        return cls(
            sigma_phi_bins=float(spread * float(phi_scale) / phi_width),
            sigma_s_bins=float(spread * float(s_scale) / s_width),
            periodic_phi=bool(periodic_phi),
            periodic_s=bool(periodic_s),
            preserve_total=bool(preserve_total),
        )


@dataclass(frozen=True)
class WallHeatFluxMetrics:
    """Scalar strike-footprint metrics suitable for active heat control."""

    total_power: float
    peak_flux: float
    centroid_phi: float
    centroid_s: float
    rms_width_phi: float
    rms_width_s: float
    fwhm_s: float
    peak_to_mean: float
    effective_area: float


@dataclass(frozen=True)
class HeatDistributionControlResult:
    """One linearized heat-distribution control solve."""

    system: BoundaryLinearResponseSystem
    solve: BoundaryResponseSolveResult
    current_observables: BoundaryResponseObservables
    target_observables: BoundaryResponseObservables
    current_heat: np.ndarray
    target_heat: np.ndarray
    predicted_heat: np.ndarray
    diffusion: FieldLineDiffusionSpec | None = None


@dataclass(frozen=True)
class HeatDistributionControlStep:
    """One iteration of callback-driven heat-distribution control."""

    iteration: int
    controls_before: np.ndarray
    controls_after: np.ndarray
    result: HeatDistributionControlResult


def fusionsc_trace_endpoints_cylindrical(trace_result: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract final trace points from a ``fusionsc.flt.trace`` dict result.

    The preferred source is ``endPoints``.  If it is absent, the last finite
    sample in ``fieldLines`` is used.  Returned arrays are cylindrical
    ``(R, Z, phi)`` and are flattened over all traced start points.
    """

    if "endPoints" in trace_result:
        end = np.asarray(trace_result["endPoints"], dtype=float)
        if end.shape[0] < 3:
            raise ValueError("fusionsc endPoints must have first dimension at least 3")
        x = np.asarray(end[0], dtype=float).ravel()
        y = np.asarray(end[1], dtype=float).ravel()
        z = np.asarray(end[2], dtype=float).ravel()
    elif "fieldLines" in trace_result:
        lines = np.asarray(trace_result["fieldLines"], dtype=float)
        if lines.shape[0] != 3:
            raise ValueError("fusionsc fieldLines must have first dimension 3")
        flat = lines.reshape((3, -1, lines.shape[-1]))
        x = np.full(flat.shape[1], np.nan, dtype=float)
        y = np.full(flat.shape[1], np.nan, dtype=float)
        z = np.full(flat.shape[1], np.nan, dtype=float)
        finite = np.all(np.isfinite(flat), axis=0)
        for idx in range(flat.shape[1]):
            valid = np.nonzero(finite[idx])[0]
            if valid.size:
                last = int(valid[-1])
                x[idx], y[idx], z[idx] = flat[:, idx, last]
    else:
        raise ValueError("trace_result must contain endPoints or fieldLines")
    R = np.hypot(x, y)
    phi = np.arctan2(y, x)
    return R, z, phi


def wall_heat_footprint_from_fusionsc_trace(
    trace_result: Mapping[str, object],
    wall_phi: Sequence[float],
    wall_R: np.ndarray,
    wall_Z: np.ndarray,
    *,
    weights: Sequence[float] | None = None,
    n_phi_bins: int | None = None,
    n_s_bins: int = 160,
    field_period: float | None = None,
):
    """Convert a fusionsc diffusive trace result into a pyna wall heat footprint."""

    from pyna.plot.wall_heat import wall_heat_footprint_from_hits

    R, Z, phi = fusionsc_trace_endpoints_cylindrical(trace_result)
    return wall_heat_footprint_from_hits(
        R,
        Z,
        phi,
        wall_phi,
        wall_R,
        wall_Z,
        weights=weights,
        n_phi_bins=n_phi_bins,
        n_s_bins=n_s_bins,
        field_period=field_period,
    )


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    sigma_f = float(sigma)
    if sigma_f <= 0.0:
        return np.array([1.0], dtype=float)
    radius = max(1, int(np.ceil(4.0 * sigma_f)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_f) ** 2)
    return kernel / float(np.sum(kernel))


def _convolve_axis(values: np.ndarray, kernel: np.ndarray, *, axis: int, periodic: bool) -> np.ndarray:
    if kernel.size == 1:
        return values.copy()
    pad = kernel.size // 2
    mode = "wrap" if periodic else "edge"
    padded = np.pad(values, [(pad, pad) if idx == axis else (0, 0) for idx in range(values.ndim)], mode=mode)
    moved = np.moveaxis(padded, axis, 0)
    out = np.empty(np.moveaxis(values, axis, 0).shape, dtype=float)
    for idx in np.ndindex(moved.shape[1:]):
        out[(slice(None),) + idx] = np.convolve(moved[(slice(None),) + idx], kernel, mode="valid")
    return np.moveaxis(out, 0, axis)


def diffuse_wall_heat_distribution(
    heat: Sequence[Sequence[float]],
    diffusion: FieldLineDiffusionSpec | None = None,
) -> np.ndarray:
    """Apply a lightweight field-line diffusion proxy to wall heat bins."""

    heat_arr = np.asarray(heat, dtype=float)
    if heat_arr.ndim != 2:
        raise ValueError("heat must be a 2-D wall heat array")
    if diffusion is None:
        return heat_arr.copy()
    total_before = float(np.nansum(heat_arr))
    filled = np.nan_to_num(heat_arr, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        from scipy.ndimage import gaussian_filter

        mode = ("wrap" if diffusion.periodic_phi else "nearest", "wrap" if diffusion.periodic_s else "nearest")
        out = gaussian_filter(
            filled,
            sigma=(float(diffusion.sigma_phi_bins), float(diffusion.sigma_s_bins)),
            mode=mode,
        )
    except Exception:  # pragma: no cover - scipy is normally available here
        out = _convolve_axis(
            filled,
            _gaussian_kernel1d(diffusion.sigma_phi_bins),
            axis=0,
            periodic=diffusion.periodic_phi,
        )
        out = _convolve_axis(
            out,
            _gaussian_kernel1d(diffusion.sigma_s_bins),
            axis=1,
            periodic=diffusion.periodic_s,
        )
    out = np.maximum(np.asarray(out, dtype=float), 0.0)
    if diffusion.preserve_total:
        total_after = float(np.nansum(out))
        if total_after > 0.0 and np.isfinite(total_before):
            out = out * (total_before / total_after)
    return out


def wall_heat_flux_metrics(
    heat: Sequence[Sequence[float]],
    *,
    phi_values: Sequence[float] | None = None,
    s_values: Sequence[float] | None = None,
    cell_areas: Sequence[Sequence[float]] | None = None,
    periodic_phi: bool = True,
    phi_period: float = 2.0 * np.pi,
) -> WallHeatFluxMetrics:
    """Measure heat-load intensity, location, and footprint width.

    ``heat`` is interpreted as heat flux.  When ``cell_areas`` is supplied,
    integrated quantities and moments use ``heat * cell_areas`` while
    ``peak_flux`` remains the local heat-flux maximum.
    """

    flux = np.asarray(heat, dtype=float)
    if flux.ndim != 2:
        raise ValueError("heat must be a 2-D wall heat array")
    n_phi, n_s = flux.shape
    phi = (
        np.arange(n_phi, dtype=float)
        if phi_values is None
        else np.asarray(phi_values, dtype=float).ravel()
    )
    s = (
        np.arange(n_s, dtype=float)
        if s_values is None
        else np.asarray(s_values, dtype=float).ravel()
    )
    if phi.size != n_phi or s.size != n_s:
        raise ValueError("phi_values and s_values must match the heat-map axes")
    if cell_areas is None:
        area = np.ones_like(flux)
    else:
        area = np.asarray(cell_areas, dtype=float)
        if area.shape != flux.shape:
            raise ValueError("cell_areas must match heat shape")
        if np.any(area < 0.0) or not np.all(np.isfinite(area)):
            raise ValueError("cell_areas must be finite and non-negative")

    finite_flux = np.where(np.isfinite(flux), np.maximum(flux, 0.0), 0.0)
    power = finite_flux * area
    total = float(np.sum(power))
    peak = float(np.max(finite_flux)) if finite_flux.size else 0.0
    positive_area = float(np.sum(np.where(np.isfinite(flux), area, 0.0)))
    mean_flux = 0.0 if positive_area <= 0.0 else total / positive_area
    peak_to_mean = 0.0 if mean_flux <= 0.0 else peak / mean_flux
    effective_area = 0.0 if peak <= 0.0 else total / peak
    if total <= 0.0:
        return WallHeatFluxMetrics(
            total_power=0.0,
            peak_flux=0.0,
            centroid_phi=float("nan"),
            centroid_s=float("nan"),
            rms_width_phi=float("nan"),
            rms_width_s=float("nan"),
            fwhm_s=float("nan"),
            peak_to_mean=0.0,
            effective_area=0.0,
        )

    phi_weight = np.sum(power, axis=1)
    s_weight = np.sum(power, axis=0)
    centroid_s = float(np.sum(s_weight * s) / total)
    rms_s = float(np.sqrt(np.sum(s_weight * (s - centroid_s) ** 2) / total))

    if periodic_phi:
        period = float(phi_period)
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("phi_period must be positive and finite")
        phi0 = float(phi[0])
        angles = 2.0 * np.pi * (phi - phi0) / period
        resultant = np.sum(phi_weight * np.exp(1j * angles)) / total
        centroid_angle = float(np.angle(resultant)) % (2.0 * np.pi)
        centroid_phi = phi0 + period * centroid_angle / (2.0 * np.pi)
        wrapped_delta = np.angle(np.exp(1j * (angles - centroid_angle))) * period / (2.0 * np.pi)
        rms_phi = float(np.sqrt(np.sum(phi_weight * wrapped_delta**2) / total))
    else:
        centroid_phi = float(np.sum(phi_weight * phi) / total)
        rms_phi = float(np.sqrt(np.sum(phi_weight * (phi - centroid_phi) ** 2) / total))

    return WallHeatFluxMetrics(
        total_power=total,
        peak_flux=peak,
        centroid_phi=centroid_phi,
        centroid_s=centroid_s,
        rms_width_phi=rms_phi,
        rms_width_s=rms_s,
        fwhm_s=float(2.0 * np.sqrt(2.0 * np.log(2.0)) * rms_s),
        peak_to_mean=float(peak_to_mean),
        effective_area=float(effective_area),
    )


def wall_heat_flux_observables(
    heat: Sequence[Sequence[float]],
    *,
    phi_values: Sequence[float] | None = None,
    s_values: Sequence[float] | None = None,
    cell_areas: Sequence[Sequence[float]] | None = None,
    periodic_phi: bool = True,
    phi_period: float = 2.0 * np.pi,
    quantities: Sequence[str] = (
        "total_power",
        "peak_flux",
        "centroid_s",
        "rms_width_s",
        "peak_to_mean",
        "effective_area",
    ),
    weights=None,
    prefix: str = "heat",
    metadata: Mapping[str, object] | None = None,
) -> BoundaryResponseObservables:
    """Expose targetable heat intensity and strike-width scalar rows."""

    metrics = wall_heat_flux_metrics(
        heat,
        phi_values=phi_values,
        s_values=s_values,
        cell_areas=cell_areas,
        periodic_phi=periodic_phi,
        phi_period=phi_period,
    )
    allowed = tuple(WallHeatFluxMetrics.__dataclass_fields__)
    labels = tuple(str(quantity) for quantity in quantities)
    unknown = [label for label in labels if label not in allowed]
    if unknown:
        raise ValueError(f"unknown wall heat metric quantities: {unknown}")
    values = [float(getattr(metrics, label)) for label in labels]
    return boundary_response_observables(
        labels,
        values,
        weights=weights,
        prefix=prefix,
        metadata={} if metadata is None else dict(metadata),
    )


def _coarsen_heat(heat: np.ndarray, coarse_shape: tuple[int, int] | None) -> np.ndarray:
    arr = np.asarray(heat, dtype=float)
    if coarse_shape is None:
        return arr.copy()
    n_phi, n_s = int(coarse_shape[0]), int(coarse_shape[1])
    if n_phi <= 0 or n_s <= 0:
        raise ValueError("coarse_shape entries must be positive")
    phi_groups = np.array_split(np.arange(arr.shape[0]), n_phi)
    s_groups = np.array_split(np.arange(arr.shape[1]), n_s)
    out = np.empty((n_phi, n_s), dtype=float)
    for i, phi_idx in enumerate(phi_groups):
        for j, s_idx in enumerate(s_groups):
            out[i, j] = float(np.nansum(arr[np.ix_(phi_idx, s_idx)]))
    return out


def _prepare_heat_observable_array(
    heat: Sequence[Sequence[float]],
    *,
    diffusion: FieldLineDiffusionSpec | None = None,
    cell_areas: Sequence[Sequence[float]] | None = None,
    coarse_shape: tuple[int, int] | None = None,
    normalize: bool = True,
) -> np.ndarray:
    arr = diffuse_wall_heat_distribution(heat, diffusion)
    if cell_areas is not None:
        area = np.asarray(cell_areas, dtype=float)
        if area.shape != arr.shape:
            raise ValueError("cell_areas must match heat shape")
        if np.any(area < 0.0) or not np.all(np.isfinite(area)):
            raise ValueError("cell_areas must be finite and non-negative")
        arr = arr * area
    arr = _coarsen_heat(arr, coarse_shape)
    if normalize:
        total = float(np.nansum(arr))
        if total > 0.0:
            arr = arr / total
    return arr


def _coerce_bound_value(value) -> tuple[float, float]:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        magnitude = abs(float(arr))
        return -magnitude, magnitude
    flat = arr.ravel()
    if flat.size != 2:
        raise ValueError("mapping bounds values must be scalar magnitudes or (lower, upper) pairs")
    return float(flat[0]), float(flat[1])


def _bounds_pair_for_labels(bounds, labels: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    n_controls = len(labels)
    if bounds is None:
        return np.full(n_controls, -np.inf, dtype=float), np.full(n_controls, np.inf, dtype=float)
    if hasattr(bounds, "lb") and hasattr(bounds, "ub"):
        lower, upper = bounds.lb, bounds.ub
    elif isinstance(bounds, Mapping):
        control_index = {label: idx for idx, label in enumerate(labels)}
        unknown = [str(label) for label in bounds if str(label) not in control_index]
        if unknown:
            raise ValueError(f"bounds labels are not present in controls: {unknown}")
        lo = np.full(n_controls, -np.inf, dtype=float)
        hi = np.full(n_controls, np.inf, dtype=float)
        for label, value in bounds.items():
            lower_i, upper_i = _coerce_bound_value(value)
            idx = control_index[str(label)]
            lo[idx] = lower_i
            hi[idx] = upper_i
    else:
        lower, upper = bounds
        lo = np.broadcast_to(np.asarray(lower, dtype=float), (n_controls,)).copy()
        hi = np.broadcast_to(np.asarray(upper, dtype=float), (n_controls,)).copy()
    if hasattr(bounds, "lb") and hasattr(bounds, "ub"):
        lo = np.broadcast_to(np.asarray(lower, dtype=float), (n_controls,)).copy()
        hi = np.broadcast_to(np.asarray(upper, dtype=float), (n_controls,)).copy()
    if np.any(hi < lo):
        raise ValueError("upper bounds must be >= lower bounds")
    return lo, hi


def _increment_bounds_for_controls(
    increment_bounds,
    control_bounds,
    controls: np.ndarray,
    labels: tuple[str, ...],
):
    inc_lo, inc_hi = _bounds_pair_for_labels(increment_bounds, labels)
    if control_bounds is None:
        return inc_lo, inc_hi
    abs_lo, abs_hi = _bounds_pair_for_labels(control_bounds, labels)
    if np.any(controls < abs_lo - 1.0e-12) or np.any(controls > abs_hi + 1.0e-12):
        raise ValueError("initial/current controls are outside control_bounds")
    lo = np.maximum(inc_lo, abs_lo - controls)
    hi = np.minimum(inc_hi, abs_hi - controls)
    if np.any(hi < lo):
        raise ValueError("control_bounds and increment bounds leave no feasible control step")
    return lo, hi


def wall_heat_distribution_observables(
    heat: Sequence[Sequence[float]],
    *,
    diffusion: FieldLineDiffusionSpec | None = None,
    cell_areas: Sequence[Sequence[float]] | None = None,
    coarse_shape: tuple[int, int] | None = None,
    normalize: bool = True,
    weights=None,
    prefix: str = "heat",
    metadata: Mapping[str, object] | None = None,
) -> BoundaryResponseObservables:
    """Flatten heat flux into cell-power rows for distribution control.

    When ``cell_areas`` is supplied, each row represents ``heat * cell_areas``
    before coarsening and optional normalization.  Without areas, the legacy
    raw-heat behavior is retained.
    """

    arr = _prepare_heat_observable_array(
        heat,
        diffusion=diffusion,
        cell_areas=cell_areas,
        coarse_shape=coarse_shape,
        normalize=normalize,
    )
    labels = tuple(f"bin.p{i:02d}.s{j:02d}" for i in range(arr.shape[0]) for j in range(arr.shape[1]))
    return boundary_response_observables(
        labels,
        arr.ravel(),
        weights=weights,
        prefix=prefix,
        metadata={} if metadata is None else dict(metadata),
    )


def heat_distribution_response_system(
    current_heat: Sequence[Sequence[float]],
    plus_heats: Sequence[Sequence[Sequence[float]]],
    minus_heats: Sequence[Sequence[Sequence[float]]],
    *,
    steps=1.0,
    plus_steps=None,
    minus_steps=None,
    diffusion: FieldLineDiffusionSpec | None = None,
    cell_areas: Sequence[Sequence[float]] | None = None,
    coarse_shape: tuple[int, int] | None = None,
    normalize: bool = True,
    weights=None,
    control_labels: Sequence[str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> BoundaryLinearResponseSystem:
    """Assemble a finite-difference heat-distribution response system."""

    current = wall_heat_distribution_observables(
        current_heat,
        diffusion=diffusion,
        cell_areas=cell_areas,
        coarse_shape=coarse_shape,
        normalize=normalize,
        weights=weights,
        metadata={"kind": "current_heat"},
    )
    plus = [
        wall_heat_distribution_observables(
            heat,
            diffusion=diffusion,
            cell_areas=cell_areas,
            coarse_shape=coarse_shape,
            normalize=normalize,
            weights=current.weights,
        )
        for heat in plus_heats
    ]
    minus = [
        wall_heat_distribution_observables(
            heat,
            diffusion=diffusion,
            cell_areas=cell_areas,
            coarse_shape=coarse_shape,
            normalize=normalize,
            weights=current.weights,
        )
        for heat in minus_heats
    ]
    md = {
        "observable_shape": _prepare_heat_observable_array(
            current_heat,
            diffusion=diffusion,
            cell_areas=cell_areas,
            coarse_shape=coarse_shape,
            normalize=normalize,
        ).shape
    }
    if metadata:
        md.update(dict(metadata))
    return finite_difference_boundary_response_system(
        current,
        plus,
        minus,
        steps=steps,
        plus_steps=plus_steps,
        minus_steps=minus_steps,
        control_labels=control_labels,
        metadata=md,
    )


def solve_heat_distribution_control(
    current_heat: Sequence[Sequence[float]],
    target_heat: Sequence[Sequence[float]],
    plus_heats: Sequence[Sequence[Sequence[float]]],
    minus_heats: Sequence[Sequence[Sequence[float]]],
    *,
    steps=1.0,
    plus_steps=None,
    minus_steps=None,
    diffusion: FieldLineDiffusionSpec | None = None,
    cell_areas: Sequence[Sequence[float]] | None = None,
    coarse_shape: tuple[int, int] | None = None,
    normalize: bool = True,
    weights=None,
    control_labels: Sequence[str] | None = None,
    bounds=None,
    regularization: float = 0.0,
    control_scale=None,
    metadata: Mapping[str, object] | None = None,
) -> HeatDistributionControlResult:
    """Solve one bounded weighted least-squares heat-distribution control step."""

    system = heat_distribution_response_system(
        current_heat,
        plus_heats,
        minus_heats,
        steps=steps,
        plus_steps=plus_steps,
        minus_steps=minus_steps,
        diffusion=diffusion,
        cell_areas=cell_areas,
        coarse_shape=coarse_shape,
        normalize=normalize,
        weights=weights,
        control_labels=control_labels,
        metadata=metadata,
    )
    current_obs = wall_heat_distribution_observables(
        current_heat,
        diffusion=diffusion,
        cell_areas=cell_areas,
        coarse_shape=coarse_shape,
        normalize=normalize,
        weights=weights,
    )
    target_obs = wall_heat_distribution_observables(
        target_heat,
        diffusion=diffusion,
        cell_areas=cell_areas,
        coarse_shape=coarse_shape,
        normalize=normalize,
        weights=weights,
    )
    solve = system.solve(
        target_obs.values,
        bounds=bounds,
        regularization=regularization,
        control_scale=control_scale,
    )
    current_arr = current_obs.values.reshape(system.metadata["observable_shape"])
    target_arr = target_obs.values.reshape(system.metadata["observable_shape"])
    predicted = solve.predicted.reshape(system.metadata["observable_shape"])
    return HeatDistributionControlResult(
        system=system,
        solve=solve,
        current_observables=current_obs,
        target_observables=target_obs,
        current_heat=current_arr,
        target_heat=target_arr,
        predicted_heat=predicted,
        diffusion=diffusion,
    )


def iterative_heat_distribution_control(
    evaluate_heat: Callable[[np.ndarray], Sequence[Sequence[float]]],
    initial_controls: Sequence[float],
    target_heat: Sequence[Sequence[float]],
    *,
    control_labels: Sequence[str],
    steps=1.0,
    n_iterations: int = 3,
    diffusion: FieldLineDiffusionSpec | None = None,
    cell_areas: Sequence[Sequence[float]] | None = None,
    coarse_shape: tuple[int, int] | None = None,
    normalize: bool = True,
    weights=None,
    bounds=None,
    control_bounds=None,
    regularization: float = 0.0,
    control_scale=None,
) -> list[HeatDistributionControlStep]:
    """Iteratively update controls using callback-evaluated heat maps.

    ``evaluate_heat`` may wrap a full nonlinear tracer, including fusionsc
    diffusive tracing when available.  Each iteration builds centered
    finite-difference heat responses around the current controls, solves a
    bounded linear step, and updates the controls by the solved increment.
    ``bounds`` constrains each iteration's increment, while
    ``control_bounds`` constrains the absolute control values after the update.
    """

    controls = np.asarray(initial_controls, dtype=float).ravel().copy()
    labels = tuple(str(label) for label in control_labels)
    if controls.size != len(labels):
        raise ValueError("initial_controls length must match control_labels")
    step_arr = np.broadcast_to(np.asarray(steps, dtype=float), controls.shape).copy()
    if np.any(step_arr == 0.0) or not np.all(np.isfinite(step_arr)):
        raise ValueError("steps must be finite and nonzero")
    history: list[HeatDistributionControlStep] = []
    for iteration in range(int(n_iterations)):
        current_heat = evaluate_heat(controls)
        abs_lower, abs_upper = _bounds_pair_for_labels(control_bounds, labels)
        plus = []
        minus = []
        plus_step_arr = np.empty_like(step_arr, dtype=float)
        minus_step_arr = np.empty_like(step_arr, dtype=float)
        for idx, step in enumerate(step_arr):
            step_abs = abs(float(step))
            plus_step = step_abs
            minus_step = step_abs
            if control_bounds is not None:
                plus_step = min(plus_step, max(0.0, float(abs_upper[idx] - controls[idx])))
                minus_step = min(minus_step, max(0.0, float(controls[idx] - abs_lower[idx])))
            plus_step_arr[idx] = plus_step
            minus_step_arr[idx] = minus_step
            plus_delta = np.zeros_like(controls)
            plus_delta[idx] = plus_step
            minus_delta = np.zeros_like(controls)
            minus_delta[idx] = -minus_step
            plus.append(current_heat if plus_step == 0.0 else evaluate_heat(controls + plus_delta))
            minus.append(current_heat if minus_step == 0.0 else evaluate_heat(controls + minus_delta))
            if plus_step + minus_step == 0.0:
                plus_step_arr[idx] = 0.5
                minus_step_arr[idx] = 0.5
        iteration_bounds = _increment_bounds_for_controls(
            bounds,
            control_bounds,
            controls,
            labels,
        )
        result = solve_heat_distribution_control(
            current_heat,
            target_heat,
            plus,
            minus,
            steps=step_arr,
            plus_steps=plus_step_arr,
            minus_steps=minus_step_arr,
            diffusion=diffusion,
            cell_areas=cell_areas,
            coarse_shape=coarse_shape,
            normalize=normalize,
            weights=weights,
            control_labels=labels,
            bounds=iteration_bounds,
            regularization=regularization,
            control_scale=control_scale,
        )
        before = controls.copy()
        controls = controls + result.solve.controls
        history.append(
            HeatDistributionControlStep(
                iteration=iteration,
                controls_before=before,
                controls_after=controls.copy(),
                result=result,
            )
        )
    return history


__all__ = [
    "FieldLineDiffusionSpec",
    "HeatDistributionControlResult",
    "HeatDistributionControlStep",
    "WallHeatFluxMetrics",
    "diffuse_wall_heat_distribution",
    "fusionsc_trace_endpoints_cylindrical",
    "heat_distribution_response_system",
    "iterative_heat_distribution_control",
    "solve_heat_distribution_control",
    "wall_heat_footprint_from_fusionsc_trace",
    "wall_heat_distribution_observables",
    "wall_heat_flux_metrics",
    "wall_heat_flux_observables",
]
