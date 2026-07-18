"""Island O/X-point extraction from Poincar?? scatter data.

Provides
--------
* :class:`IslandChain` ???dataclass holding O/X points and widths.
* :func:`extract_island_width` ???infer island geometry from Poincar?? data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize

__all__ = [
    "IslandChain",
    "extract_island_width",
    "extract_island_width_newton",
    "detect_residual_islands",
]


@dataclass
class IslandChain:
    """Geometric description of an island chain.

    Attributes
    ----------
    O_points : ndarray of shape (n_islands, 2)
        O-point coordinates (R, Z) in metres.
    X_points : ndarray of shape (n_islands, 2)
        X-point coordinates (R, Z) in metres.
    half_width_r : float
        Average radial half-width in metres.
    half_width_psi : float
        Average half-width in normalised ?? coordinate.
    fixed_point_kinds : ndarray of str, optional
        Kinds of Newton-refined fixed points used for diagnostics.
    fixed_point_traces : ndarray, optional
        Traces of the Newton-refined monodromy matrices.
    fixed_point_residuals : ndarray, optional
        Final fixed-point residual norms.
    metadata : dict, optional
        Extra diagnostics from the extractor.
    """

    O_points: np.ndarray
    X_points: np.ndarray
    half_width_r: float
    half_width_psi: float
    fixed_point_kinds: np.ndarray = field(default_factory=lambda: np.empty(0, dtype="<U1"))
    fixed_point_traces: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    fixed_point_residuals: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    metadata: dict = field(default_factory=dict)


def extract_island_width(
    poincare_pts: np.ndarray,
    R_axis: float,
    Z_axis: float,
    mode_m: int,
    psi_func,
    max_newton_iter: int = 50,
    n_fallback_seeds: int = 8,
) -> IslandChain:
    """Extract O/X points and island half-widths from Poincar?? scatter data.

    ``max_newton_iter`` is retained for legacy API compatibility; this rough
    point-cloud extractor does not run Newton refinement.
    """
    _ = max_newton_iter
    return _rough_island_chain_from_points(
        poincare_pts,
        R_axis,
        Z_axis,
        mode_m,
        psi_func,
        n_fallback_seeds=n_fallback_seeds,
    )


def _rough_island_chain_from_points(
    poincare_pts: np.ndarray,
    R_axis: float,
    Z_axis: float,
    mode_m: int,
    psi_func,
    *,
    n_fallback_seeds: int = 8,
) -> IslandChain:
    """Extract O/X points and island half-widths from Poincar?? scatter data.

    Algorithm
    ---------
    1. Compute angles from magnetic axis for each Poincar?? point.
    2. Bin into *mode_m* groups by angle.
    3. Cluster centroid ???initial O-point candidate.
    4. Refine each O-point using Nelder-Mead to minimise the radial variance
       of the points in that cluster.  Fallback seeds if optimisation diverges.
    5. X-points: midpoints between O-points in angle (same radial distance).
    6. Half-widths computed from point-cloud spread around each O-point.

    Parameters
    ----------
    poincare_pts : ndarray of shape (N, 2) or (N, 3)
        (R, Z) columns (additional columns ignored).
    R_axis, Z_axis : float
        Magnetic axis coordinates.
    mode_m : int
        Number of islands in the chain.
    psi_func : callable
        ``psi_func(R, Z) ???psi_norm``.
    n_fallback_seeds : int
        Number of fallback seeds if Nelder-Mead diverges.

    Returns
    -------
    IslandChain
    """
    pts = np.asarray(poincare_pts, dtype=float)
    R_pts = pts[:, 0]
    Z_pts = pts[:, 1]

    # Angles from magnetic axis
    angles = np.arctan2(Z_pts - Z_axis, R_pts - R_axis)  # in (-??, ??]
    # Normalise to [0, 2??)
    angles = angles % (2 * np.pi)
    # Radial distances from axis
    r_pts = np.sqrt((R_pts - R_axis) ** 2 + (Z_pts - Z_axis) ** 2)

    # Bin into mode_m sectors by angle
    bin_edges = np.linspace(0, 2 * np.pi, mode_m + 1)
    labels = np.digitize(angles, bin_edges) - 1
    labels = np.clip(labels, 0, mode_m - 1)

    O_points = []
    half_widths_R = []
    half_widths_psi = []

    for k in range(mode_m):
        mask = labels == k
        if mask.sum() < 3:
            continue

        R_k = R_pts[mask]
        Z_k = Z_pts[mask]
        r_k = r_pts[mask]

        # Initial centroid
        R0_k = float(np.mean(R_k))
        Z0_k = float(np.mean(Z_k))

        # Minimise radial variance in cluster ???proxy for O-point location
        def objective(rz):
            R_c, Z_c = rz[0], rz[1]
            r_c = np.sqrt((R_k - R_c) ** 2 + (Z_k - Z_c) ** 2)
            return float(np.var(r_c))

        res = minimize(objective, [R0_k, Z0_k], method='Nelder-Mead',
                       options={'xatol': 1e-5, 'fatol': 1e-8, 'maxiter': 500})
        R_O, Z_O = res.x[0], res.x[1]

        # Check if optimum is reasonable (inside domain)
        r_O = np.sqrt((R_O - R_axis) ** 2 + (Z_O - Z_axis) ** 2)
        r_mean = float(np.mean(r_k))
        if r_O > 2.0 * r_mean or r_O < 0.01 * r_mean:
            # Fallback: try seeds along radial direction
            theta_k = float(np.mean(np.arctan2(Z_k - Z_axis, R_k - R_axis)))
            best_obj = np.inf
            R_O, Z_O = R0_k, Z0_k
            for j in range(n_fallback_seeds):
                r_seed = r_mean * (0.5 + j / n_fallback_seeds)
                R_s = R_axis + r_seed * np.cos(theta_k)
                Z_s = Z_axis + r_seed * np.sin(theta_k)
                res2 = minimize(objective, [R_s, Z_s], method='Nelder-Mead',
                                options={'xatol': 1e-5, 'fatol': 1e-8})
                if res2.fun < best_obj:
                    best_obj = res2.fun
                    R_O, Z_O = res2.x[0], res2.x[1]

        O_points.append([R_O, Z_O])

        # Half-width: from point scatter around O-point
        r_from_O = np.sqrt((R_k - R_O) ** 2 + (Z_k - Z_O) ** 2)
        r_min = float(np.min(r_from_O))
        r_max = float(np.max(r_from_O))
        half_widths_R.append((r_max - r_min) / 2.0)

        # ?? half-width
        try:
            psi_O = float(psi_func(R_O, Z_O))
            angle_O = float(np.arctan2(Z_O - Z_axis, R_O - R_axis))
            dr = (r_max - r_min) / 2.0
            R_plus = R_O + dr * np.cos(angle_O)
            Z_plus = Z_O + dr * np.sin(angle_O)
            psi_plus = float(psi_func(R_plus, Z_plus))
            half_widths_psi.append(abs(psi_plus - psi_O))
        except Exception:
            half_widths_psi.append(np.nan)

    if not O_points:
        return IslandChain(
            O_points=np.empty((0, 2)),
            X_points=np.empty((0, 2)),
            half_width_r=np.nan,
            half_width_psi=np.nan,
        )

    O_points_arr = np.array(O_points)

    # X-points: midpoints between consecutive O-points (in angle)
    angles_O = np.arctan2(
        O_points_arr[:, 1] - Z_axis,
        O_points_arr[:, 0] - R_axis,
    )
    order = np.argsort(angles_O)
    O_sorted = O_points_arr[order]

    X_points_list = []
    r_O_sorted = np.sqrt(
        (O_sorted[:, 0] - R_axis) ** 2 + (O_sorted[:, 1] - Z_axis) ** 2
    )
    angles_O_sorted = np.arctan2(O_sorted[:, 1] - Z_axis, O_sorted[:, 0] - R_axis)
    n_O = len(O_sorted)
    for k in range(n_O):
        angle1 = angles_O_sorted[k]
        angle2 = angles_O_sorted[(k + 1) % n_O]
        angle_mid = (angle1 + angle2) / 2.0
        r_mid = (r_O_sorted[k] + r_O_sorted[(k + 1) % n_O]) / 2.0
        X_points_list.append([
            R_axis + r_mid * np.cos(angle_mid),
            Z_axis + r_mid * np.sin(angle_mid),
        ])

    X_points_arr = np.array(X_points_list) if X_points_list else np.empty((0, 2))

    avg_hw_R = float(np.nanmean(half_widths_R)) if half_widths_R else np.nan
    avg_hw_psi = float(np.nanmean(half_widths_psi)) if half_widths_psi else np.nan

    return IslandChain(
        O_points=O_sorted,
        X_points=X_points_arr,
        half_width_r=avg_hw_R,
        half_width_psi=avg_hw_psi,
    )


def _sort_points_by_axis_angle(points: np.ndarray, R_axis: float, Z_axis: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.empty((0, 2), dtype=float)
    pts = pts.reshape((-1, 2))
    angles = np.arctan2(pts[:, 1] - float(Z_axis), pts[:, 0] - float(R_axis)) % (2.0 * np.pi)
    return pts[np.argsort(angles)]


def _x_seed_bases_from_opoints(O_points: np.ndarray, R_axis: float, Z_axis: float) -> np.ndarray:
    opts = _sort_points_by_axis_angle(O_points, R_axis, Z_axis)
    if len(opts) < 2:
        return np.empty((0, 2), dtype=float)
    rel = opts - np.asarray([float(R_axis), float(Z_axis)], dtype=float)
    angles = np.arctan2(rel[:, 1], rel[:, 0]) % (2.0 * np.pi)
    radii = np.linalg.norm(rel, axis=1)
    bases = []
    for i in range(len(opts)):
        a0 = float(angles[i])
        a1 = float(angles[(i + 1) % len(opts)])
        da = float(np.angle(np.exp(1j * (a1 - a0))))
        amid = (a0 + 0.5 * da) % (2.0 * np.pi)
        rmid = 0.5 * (float(radii[i]) + float(radii[(i + 1) % len(opts)]))
        bases.append([
            float(R_axis) + rmid * np.cos(amid),
            float(Z_axis) + rmid * np.sin(amid),
        ])
    return np.asarray(bases, dtype=float)


def _unique_phases(phases: list[float], period: float, tol: float = 1.0e-8) -> list[float]:
    unique: list[float] = []
    for phase in phases:
        value = float(phase) % float(period)
        if all(abs(((value - old + 0.5 * float(period)) % float(period)) - 0.5 * float(period)) > tol
               for old in unique):
            unique.append(value)
    return unique


def _cloud_sector_seed_sets(
    poincare_pts: np.ndarray,
    R_axis: float,
    Z_axis: float,
    mode_m: int,
) -> list[np.ndarray]:
    """Return centroid seed sets from m-fold angular sectors of the cloud."""
    pts = np.asarray(poincare_pts, dtype=float)
    if pts.size == 0 or int(mode_m) <= 0:
        return []
    pts = pts[:, :2]
    rel = pts - np.asarray([float(R_axis), float(Z_axis)], dtype=float)
    angles = np.arctan2(rel[:, 1], rel[:, 0]) % (2.0 * np.pi)
    bin_width = 2.0 * np.pi / int(mode_m)

    phases = [0.0, 0.5 * bin_width]
    harmonic = np.mean(np.exp(1j * int(mode_m) * angles))
    if np.isfinite(harmonic.real) and np.isfinite(harmonic.imag) and abs(harmonic) > 1.0e-12:
        phase = (float(np.angle(harmonic)) / int(mode_m)) % bin_width
        phases.extend([phase, phase + 0.5 * bin_width])
    phases = _unique_phases(phases, bin_width)

    seed_sets: list[np.ndarray] = []
    for phase in phases:
        centers = (phase + bin_width * np.arange(int(mode_m))) % (2.0 * np.pi)
        angular_dist = np.abs(np.angle(np.exp(1j * (angles[:, None] - centers[None, :]))))
        labels = np.argmin(angular_dist, axis=1)
        seeds = []
        for k in range(int(mode_m)):
            mask = labels == k
            if np.count_nonzero(mask) < 3:
                continue
            seeds.append(np.mean(pts[mask], axis=0))
        if seeds:
            seed_sets.append(_sort_points_by_axis_angle(np.asarray(seeds, dtype=float), R_axis, Z_axis))
    return seed_sets


def _candidate_newton_seeds(
    rough_chain: IslandChain,
    R_axis: float,
    Z_axis: float,
    dedup_tol: float,
    *,
    poincare_pts: np.ndarray | None = None,
    mode_m: int | None = None,
) -> np.ndarray:
    seeds = []
    rough_o = np.asarray(rough_chain.O_points, dtype=float).reshape((-1, 2))
    for point in rough_o:
        seeds.append(point)

    def add_x_base_seeds(x_bases: np.ndarray, reference_o: np.ndarray) -> None:
        if x_bases.size == 0:
            return
        rel_o = reference_o - np.asarray([float(R_axis), float(Z_axis)], dtype=float)
        radii = np.linalg.norm(rel_o, axis=1)
        scale = float(np.nanmedian(radii)) if radii.size else 0.0
        offset = max(10.0 * float(dedup_tol), 0.05 * scale)
        for base in x_bases:
            seeds.append(base)
            angle = float(np.arctan2(base[1] - float(Z_axis), base[0] - float(R_axis)))
            radial = np.asarray([np.cos(angle), np.sin(angle)], dtype=float)
            tangent = np.asarray([-np.sin(angle), np.cos(angle)], dtype=float)
            for direction in (radial, tangent):
                seeds.append(base + offset * direction)
                seeds.append(base - offset * direction)

    x_bases = _x_seed_bases_from_opoints(rough_o, R_axis, Z_axis)
    add_x_base_seeds(x_bases, rough_o)

    if poincare_pts is not None and mode_m is not None:
        for cloud_o in _cloud_sector_seed_sets(poincare_pts, R_axis, Z_axis, int(mode_m)):
            for point in cloud_o:
                seeds.append(point)
            add_x_base_seeds(_x_seed_bases_from_opoints(cloud_o, R_axis, Z_axis), cloud_o)

    rough_x = np.asarray(rough_chain.X_points, dtype=float).reshape((-1, 2))
    for point in rough_x:
        seeds.append(point)

    deduped = []
    for seed in seeds:
        seed = np.asarray(seed, dtype=float)
        if not np.all(np.isfinite(seed)):
            continue
        if all(np.linalg.norm(seed - prev) > 0.5 * float(dedup_tol) for prev in deduped):
            deduped.append(seed)
    return np.asarray(deduped, dtype=float).reshape((-1, 2)) if deduped else np.empty((0, 2), dtype=float)


def _endpoint_from_map_func(map_func, x: np.ndarray, period: int) -> np.ndarray | None:
    try:
        out = map_func(float(x[0]), float(x[1]), int(period))
    except Exception:
        return None
    arr = np.asarray(out, dtype=float).ravel()
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        return None
    return arr[:2].astype(float, copy=True)


def _endpoint_from_tracer(tracer, x: np.ndarray, period: int, phi_sec: float) -> np.ndarray | None:
    try:
        result = tracer.trace_poincare(
            np.asarray([float(x[0])], dtype=float),
            np.asarray([float(x[1])], dtype=float),
            float(phi_sec),
            N_turns=int(period),
            use_wall=False,
            verbose=False,
        )
    except Exception:
        return None
    if not result:
        return None
    try:
        R_arr, Z_arr = result[0]
    except Exception:
        return None
    R_arr = np.asarray(R_arr, dtype=float).ravel()
    Z_arr = np.asarray(Z_arr, dtype=float).ravel()
    if R_arr.size < int(period) or Z_arr.size < int(period):
        return None
    out = np.asarray([R_arr[-1], Z_arr[-1]], dtype=float)
    if not np.all(np.isfinite(out)):
        return None
    return out


def _finite_difference_DPm(endpoint, x: np.ndarray, period: int, fd_eps: float) -> np.ndarray | None:
    eps = float(fd_eps)
    if eps <= 0.0:
        raise ValueError("fd_eps must be positive")
    mat = np.empty((2, 2), dtype=float)
    for j in range(2):
        step = np.zeros(2, dtype=float)
        step[j] = eps
        plus = endpoint(x + step, period)
        minus = endpoint(x - step, period)
        if plus is None or minus is None:
            return None
        mat[:, j] = (plus - minus) / (2.0 * eps)
    if not np.all(np.isfinite(mat)):
        return None
    return mat


def _classify_DPm(DPm: np.ndarray) -> str:
    trace = float(np.trace(DPm))
    if not np.isfinite(trace):
        return "U"
    if abs(trace) < 2.0:
        return "O"
    if abs(trace) > 2.0:
        return "X"
    return "P"


def _newton_refine_seed(
    seed: np.ndarray,
    endpoint,
    period: int,
    *,
    max_iter: int,
    tol: float,
    fd_eps: float,
) -> dict | None:
    x = np.asarray(seed, dtype=float).reshape(2).copy()
    if not np.all(np.isfinite(x)):
        return None

    residual = np.full(2, np.inf, dtype=float)
    for _ in range(int(max_iter)):
        mapped = endpoint(x, period)
        if mapped is None:
            return None
        residual = mapped - x
        res_norm = float(np.linalg.norm(residual))
        if not np.isfinite(res_norm):
            return None
        if res_norm <= float(tol):
            break

        DPm = _finite_difference_DPm(endpoint, x, period, fd_eps)
        if DPm is None:
            return None
        J = DPm - np.eye(2)
        try:
            step = -np.linalg.solve(J, residual)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(step)):
            return None

        max_step = 0.25
        step_norm = float(np.linalg.norm(step))
        if step_norm > max_step:
            step *= max_step / step_norm

        accepted = False
        for damping in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial = x + damping * step
            trial_mapped = endpoint(trial, period)
            if trial_mapped is None:
                continue
            trial_residual = trial_mapped - trial
            trial_norm = float(np.linalg.norm(trial_residual))
            if np.isfinite(trial_norm) and trial_norm < res_norm:
                x = trial
                accepted = True
                break
        if not accepted:
            return None

    mapped = endpoint(x, period)
    if mapped is None:
        return None
    residual = mapped - x
    res_norm = float(np.linalg.norm(residual))
    if not np.isfinite(res_norm) or res_norm > float(tol):
        return None

    DPm = _finite_difference_DPm(endpoint, x, period, fd_eps)
    if DPm is None:
        return None
    trace = float(np.trace(DPm))
    return {
        "R": float(x[0]),
        "Z": float(x[1]),
        "DPm": DPm,
        "kind": _classify_DPm(DPm),
        "trace": trace,
        "residual": res_norm,
        "eigenvalues": np.linalg.eigvals(DPm),
        "seed_R": float(seed[0]),
        "seed_Z": float(seed[1]),
    }


def _deduplicate_fixed_points(fixed_points: list[dict], dedup_tol: float) -> list[dict]:
    out: list[dict] = []
    for fp in fixed_points:
        point = np.asarray([fp["R"], fp["Z"]], dtype=float)
        replace_index = None
        for i, old in enumerate(out):
            old_point = np.asarray([old["R"], old["Z"]], dtype=float)
            if np.linalg.norm(point - old_point) <= float(dedup_tol):
                replace_index = i
                break
        if replace_index is None:
            out.append(fp)
        elif float(fp["residual"]) < float(out[replace_index]["residual"]):
            out[replace_index] = fp
    return out


def _points_of_kind(fixed_points: list[dict], kind: str, R_axis: float, Z_axis: float) -> np.ndarray:
    pts = np.asarray(
        [[fp["R"], fp["Z"]] for fp in fixed_points if str(fp.get("kind", "")).upper() == kind],
        dtype=float,
    )
    return _sort_points_by_axis_angle(pts, R_axis, Z_axis)


def _point_cloud_width_from_opoints(
    poincare_pts: np.ndarray,
    O_points: np.ndarray,
    R_axis: float,
    Z_axis: float,
    psi_func,
) -> tuple[float, float]:
    pts = np.asarray(poincare_pts, dtype=float)
    if pts.size == 0:
        return float("nan"), float("nan")
    pts = pts[:, :2]
    opts = np.asarray(O_points, dtype=float).reshape((-1, 2))
    if opts.size == 0:
        return float("nan"), float("nan")

    distances = np.linalg.norm(pts[:, None, :] - opts[None, :, :], axis=2)
    labels = np.argmin(distances, axis=1)
    half_widths_R = []
    half_widths_psi = []
    for i, opt in enumerate(opts):
        mask = labels == i
        if not np.any(mask):
            continue
        r_from_O = np.linalg.norm(pts[mask] - opt[None, :], axis=1)
        r_min = float(np.min(r_from_O))
        r_max = float(np.max(r_from_O))
        dr = 0.5 * (r_max - r_min)
        half_widths_R.append(dr)

        try:
            psi_O = float(psi_func(float(opt[0]), float(opt[1])))
            direction = opt - np.asarray([float(R_axis), float(Z_axis)], dtype=float)
            norm = float(np.linalg.norm(direction))
            if norm <= 1.0e-30:
                direction = np.asarray([1.0, 0.0], dtype=float)
            else:
                direction = direction / norm
            plus = opt + dr * direction
            psi_plus = float(psi_func(float(plus[0]), float(plus[1])))
            half_widths_psi.append(abs(psi_plus - psi_O))
        except Exception:
            half_widths_psi.append(np.nan)

    avg_hw_R = float(np.nanmean(half_widths_R)) if half_widths_R else float("nan")
    avg_hw_psi = float(np.nanmean(half_widths_psi)) if half_widths_psi else float("nan")
    return avg_hw_R, avg_hw_psi


def _ox_distance_diagnostics(O_points: np.ndarray, X_points: np.ndarray) -> np.ndarray:
    opts = np.asarray(O_points, dtype=float).reshape((-1, 2))
    xpts = np.asarray(X_points, dtype=float).reshape((-1, 2))
    if opts.size == 0 or xpts.size == 0:
        return np.empty(0, dtype=float)
    distances = np.linalg.norm(opts[:, None, :] - xpts[None, :, :], axis=2)
    return np.min(distances, axis=1)


def extract_island_width_newton(
    poincare_pts,
    R_axis,
    Z_axis,
    mode_m,
    psi_func,
    *,
    map_func=None,
    tracer=None,
    phi_sec=0.0,
    period=None,
    max_iter=40,
    tol=1e-9,
    fd_eps=1e-4,
    dedup_tol=1e-3,
    fallback_to_point_cloud=True,
    verbose=False,
) -> IslandChain:
    """Extract island O/X points with Newton refinement of ``P^period(x) = x``.

    ``poincare_pts`` are used only to seed the Newton search and to measure the
    point-cloud half-width.  Provide exactly one map source: either
    ``map_func(R, Z, period) -> (R_end, Z_end)`` or a tracer with
    ``trace_poincare``.
    """
    if (map_func is None) == (tracer is None):
        raise ValueError("provide exactly one of map_func or tracer")
    period_value = int(mode_m if period is None else period)
    if period_value <= 0:
        raise ValueError("period must be positive")

    rough = _rough_island_chain_from_points(
        poincare_pts,
        float(R_axis),
        float(Z_axis),
        int(mode_m),
        psi_func,
    )
    seeds = _candidate_newton_seeds(
        rough,
        float(R_axis),
        float(Z_axis),
        float(dedup_tol),
        poincare_pts=poincare_pts,
        mode_m=int(mode_m),
    )

    if map_func is not None:
        endpoint = lambda x, p: _endpoint_from_map_func(map_func, x, p)
    else:
        endpoint = lambda x, p: _endpoint_from_tracer(tracer, x, p, float(phi_sec))

    refined = []
    for seed in seeds:
        fp = _newton_refine_seed(
            seed,
            endpoint,
            period_value,
            max_iter=int(max_iter),
            tol=float(tol),
            fd_eps=float(fd_eps),
        )
        if fp is not None:
            refined.append(fp)
            if verbose:
                print(
                    f"  [{fp['kind']}] R={fp['R']:.8f} Z={fp['Z']:.8f} "
                    f"res={fp['residual']:.3e} tr={fp['trace']:.6g}"
                )
    refined = _deduplicate_fixed_points(refined, float(dedup_tol))

    newton_O = _points_of_kind(refined, "O", float(R_axis), float(Z_axis))
    newton_X = _points_of_kind(refined, "X", float(R_axis), float(Z_axis))
    rough_O = _sort_points_by_axis_angle(rough.O_points, float(R_axis), float(Z_axis))
    rough_X = _x_seed_bases_from_opoints(rough_O, float(R_axis), float(Z_axis))
    if rough_X.size == 0:
        rough_X = _sort_points_by_axis_angle(rough.X_points, float(R_axis), float(Z_axis))

    used_fallback_O = bool(fallback_to_point_cloud and len(newton_O) < int(mode_m))
    used_fallback_X = bool(fallback_to_point_cloud and len(newton_X) < int(mode_m))
    O_points = rough_O if used_fallback_O else newton_O
    X_points = rough_X if used_fallback_X else newton_X

    half_width_r, half_width_psi = _point_cloud_width_from_opoints(
        poincare_pts,
        O_points,
        float(R_axis),
        float(Z_axis),
        psi_func,
    )
    fixed_point_kinds = np.asarray([str(fp["kind"]) for fp in refined], dtype="<U1")
    fixed_point_traces = np.asarray([float(fp["trace"]) for fp in refined], dtype=float)
    fixed_point_residuals = np.asarray([float(fp["residual"]) for fp in refined], dtype=float)
    metadata = {
        "method": "newton",
        "period": period_value,
        "phi_sec": float(phi_sec),
        "newton_converged_count": int(len(refined)),
        "newton_o_count": int(len(newton_O)),
        "newton_x_count": int(len(newton_X)),
        "used_point_cloud_fallback_O": used_fallback_O,
        "used_point_cloud_fallback_X": used_fallback_X,
        "rough_o_count": int(len(rough_O)),
        "rough_x_count": int(len(rough_X)),
        "ox_nearest_distances": _ox_distance_diagnostics(O_points, X_points),
    }
    return IslandChain(
        O_points=np.asarray(O_points, dtype=float).reshape((-1, 2)),
        X_points=np.asarray(X_points, dtype=float).reshape((-1, 2)),
        half_width_r=half_width_r,
        half_width_psi=half_width_psi,
        fixed_point_kinds=fixed_point_kinds,
        fixed_point_traces=fixed_point_traces,
        fixed_point_residuals=fixed_point_residuals,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Residual island detection in chaotic sea
# ---------------------------------------------------------------------------

def detect_residual_islands(
    poincare_pts: np.ndarray,
    R_axis: float,
    Z_axis: float,
    n_angle_bins: int = 72,
    min_cluster_fraction: float = 0.01,
    radial_bandwidth: float = 0.05,
    min_period: int = 1,
    max_period: int = 12,
) -> list:
    """Detect surviving (residual) magnetic islands embedded in a chaotic sea.

    In a chaotic Poincaré scatter plot, small regular islands appear as
    distinct, tightly-clustered arc-shaped structures amid the diffuse chaotic
    background.  This function identifies such clusters by:

    1. Partitioning the Poincaré points into annular radial shells.
    2. Within each shell, computing the angular density histogram.
    3. Looking for ``p``-fold periodic structure (for p in [min_period,
       max_period]) that exceeds the background by a statistically significant
       margin.
    4. Returning a list of candidate (period, radial_shell_index, angle_peaks)
       tuples from which callers can seed a fixed-point finder.

    Parameters
    ----------
    poincare_pts : ndarray, shape (N, 2) or (N, 3)
        Poincaré section points.  Only the first two columns (R, Z) are used.
    R_axis, Z_axis : float
        Magnetic axis coordinates.
    n_angle_bins : int
        Number of angular bins for the density histogram.  Higher values
        give finer angular resolution.  Default 72 (5° bins).
    min_cluster_fraction : float
        Minimum fraction of points in a radial shell that must fall in the
        candidate island peaks for it to be reported.  Default 0.01.
    radial_bandwidth : float
        Fractional width of radial shells (as a fraction of the max radius
        from axis).  Default 0.05 (5 % shells).
    min_period, max_period : int
        Range of periods (number of Poincaré-map iterations) to search.

    Returns
    -------
    candidates : list of dict
        Each dict contains:

        ``'period'`` : int
            Detected periodicity p (number of O-points expected in chain).
        ``'r_shell'`` : float
            Mean radial distance of the shell where the island was detected.
        ``'angle_peaks'`` : ndarray
            Approximate angular positions of the O-points.
        ``'seed_RZ'`` : ndarray, shape (p, 2)
            Rough seed points [R, Z] for fixed-point refinement.

    Notes
    -----
    This is a *rough* detection method intended to seed a Newton-refinement
    step (e.g. :func:`pyna.topo.fixed_points.find_periodic_orbit`).  The
    returned seed points are approximate and may not all converge to true
    fixed points.

    Examples
    --------
    >>> candidates = detect_residual_islands(poincare_pts, R_axis=1.0, Z_axis=0.0)
    >>> for c in candidates:
    ...     print(c['period'], c['r_shell'], c['seed_RZ'])
    """
    pts = np.asarray(poincare_pts, dtype=float)
    R_pts = pts[:, 0]
    Z_pts = pts[:, 1]

    r_pts = np.sqrt((R_pts - R_axis) ** 2 + (Z_pts - Z_axis) ** 2)
    angle_pts = np.arctan2(Z_pts - Z_axis, R_pts - R_axis) % (2 * np.pi)

    r_max = float(np.max(r_pts))
    if r_max < 1e-12:
        return []

    n_shells = max(1, int(1.0 / radial_bandwidth))
    shell_edges = np.linspace(0.0, r_max, n_shells + 1)

    candidates = []

    angle_bin_edges = np.linspace(0.0, 2.0 * np.pi, n_angle_bins + 1)

    for s in range(n_shells):
        r_lo, r_hi = shell_edges[s], shell_edges[s + 1]
        mask = (r_pts >= r_lo) & (r_pts < r_hi)
        if mask.sum() < max(10, int(len(pts) * min_cluster_fraction)):
            continue

        angles_in_shell = angle_pts[mask]
        hist, _ = np.histogram(angles_in_shell, bins=angle_bin_edges)
        hist = hist.astype(float)

        # Background level (robust median)
        bg = float(np.median(hist))
        hist_normalised = hist / (bg + 1e-10)

        for p in range(min_period, max_period + 1):
            # Look for p-fold peaks by folding the histogram
            n_per_segment = n_angle_bins // p
            if n_per_segment < 2:
                continue
            folded = np.zeros(n_per_segment)
            for k in range(p):
                segment = hist_normalised[k * n_per_segment: (k + 1) * n_per_segment]
                folded[:len(segment)] += segment
            # Detect peaks in the folded histogram
            folded_mean = float(np.mean(folded))
            folded_std = float(np.std(folded))
            threshold = folded_mean + 1.5 * folded_std
            peak_idx = np.where(folded > threshold)[0]
            if len(peak_idx) == 0:
                continue

            # The primary peak bin index in the folded array
            primary_bin = int(peak_idx[np.argmax(folded[peak_idx])])
            primary_angle = (primary_bin + 0.5) / n_per_segment * (2.0 * np.pi / p)

            # Replicate across p copies
            angle_peaks = np.array([
                (primary_angle + k * 2.0 * np.pi / p) % (2.0 * np.pi)
                for k in range(p)
            ])
            r_shell_mean = 0.5 * (r_lo + r_hi)

            # Seed points
            seed_RZ = np.column_stack([
                R_axis + r_shell_mean * np.cos(angle_peaks),
                Z_axis + r_shell_mean * np.sin(angle_peaks),
            ])

            candidates.append({
                'period': p,
                'r_shell': r_shell_mean,
                'angle_peaks': angle_peaks,
                'seed_RZ': seed_RZ,
            })

    return candidates
