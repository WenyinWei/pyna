"""Boundary island-chain search utilities for toroidal field-line maps."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from pyna.topo.toroidal import FixedPoint
from pyna.topo._monodromy_classification import classify_monodromy_2x2
from pyna.toroidal.flt.numba_poincare import (
    find_fixed_points_batch_field,
    find_fixed_points_batch_span_field,
    trace_map_batch_span_field,
    trace_poincare_batch_field,
    trace_poincare_batch_twall_field,
    trace_orbit_along_phi_field,
)


@dataclass(frozen=True)
class BoundaryIslandFixedPoint:
    """One X/O point found from a boundary-focused ``P^m`` Newton search."""

    phi: float
    R: float
    Z: float
    period: int
    kind: str
    DPm: np.ndarray
    residual: float
    eigenvalues: np.ndarray
    seed_R: float
    seed_Z: float
    lower_period_residual: float = np.inf

    def as_fixed_point(self) -> FixedPoint:
        kind = str(self.kind).upper()
        fp = FixedPoint(
            phi=float(self.phi),
            R=float(self.R),
            Z=float(self.Z),
            DPm=np.asarray(self.DPm, dtype=float).reshape(2, 2).copy(),
            kind=kind,
        )
        fp.period = int(self.period)
        fp.residual = float(self.residual)
        fp.eigenvalues = np.asarray(self.eigenvalues).copy()
        cls = classify_monodromy_2x2(self.DPm)
        fp.trace = cls.trace
        fp.determinant = cls.determinant
        fp.discriminant = cls.discriminant
        fp.monodromy_classification_reason = cls.reason
        if kind == "X":
            stable, unstable = _stable_unstable_eigenvectors(self.DPm)
            fp.stable_eigenvec = stable
            fp.unstable_eigenvec = unstable
        return fp


@dataclass(frozen=True)
class BoundaryIslandSeedCandidates:
    """Poincare-recurrence seeds for a boundary island-chain search."""

    seeds_by_period: dict[int, tuple[np.ndarray, np.ndarray]]
    diagnostics: dict = field(default_factory=dict)

    def seeds_for_period(self, period: int) -> tuple[np.ndarray, np.ndarray]:
        return self.seeds_by_period.get(
            int(period),
            (np.empty(0, dtype=float), np.empty(0, dtype=float)),
        )


@dataclass(frozen=True)
class BoundaryIslandSearchResult:
    """Result container for boundary island-chain fixed-point searches."""

    fixed_points: tuple[BoundaryIslandFixedPoint, ...]
    fp_by_sec: dict[float, dict[str, list[FixedPoint]]]
    seed_R: np.ndarray
    seed_Z: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    @property
    def x_points(self) -> list[BoundaryIslandFixedPoint]:
        return [fp for fp in self.fixed_points if fp.kind == "X"]

    @property
    def o_points(self) -> list[BoundaryIslandFixedPoint]:
        return [fp for fp in self.fixed_points if fp.kind == "O"]


def _as_periods(periods: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(periods, (int, np.integer)):
        periods_tuple = (int(periods),)
    else:
        periods_tuple = tuple(int(p) for p in periods)
    periods_tuple = tuple(p for p in periods_tuple if p > 0)
    if not periods_tuple:
        raise ValueError("periods must contain at least one positive integer")
    return tuple(sorted(set(periods_tuple)))


def _field_grid_wall(field, *, inset: float = 1.0e-10) -> tuple[np.ndarray, np.ndarray]:
    """Return a rectangular wall just inside the interpolation grid."""

    R_arr = np.asarray(getattr(field, "R_arr", getattr(field, "R", [])), dtype=float)
    Z_arr = np.asarray(getattr(field, "Z_arr", getattr(field, "Z", [])), dtype=float)
    if R_arr.size < 2 or Z_arr.size < 2:
        raise ValueError("field grid is required when no wall is supplied")
    R_min = float(np.nanmin(R_arr)) + float(inset)
    R_max = float(np.nanmax(R_arr)) - float(inset)
    Z_min = float(np.nanmin(Z_arr)) + float(inset)
    Z_max = float(np.nanmax(Z_arr)) - float(inset)
    if not (R_min < R_max and Z_min < Z_max):
        raise ValueError("field grid bounds are degenerate")
    return (
        np.asarray([R_min, R_max, R_max, R_min], dtype=float),
        np.asarray([Z_min, Z_min, Z_max, Z_max], dtype=float),
    )


def _deduplicate_seed_points(
    seed_R: Sequence[float],
    seed_Z: Sequence[float],
    *,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(seed_R, dtype=float).ravel()
    Z = np.asarray(seed_Z, dtype=float).ravel()
    finite = np.isfinite(R) & np.isfinite(Z)
    R = R[finite]
    Z = Z[finite]
    if R.size == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    if tol <= 0.0:
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


def _stable_unstable_eigenvectors(DPm: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return normalized stable/unstable real eigenvectors for a saddle map."""

    mat = np.asarray(DPm, dtype=float).reshape(2, 2)
    if not np.all(np.isfinite(mat)):
        return None, None
    try:
        eigvals, eigvecs = np.linalg.eig(mat)
    except np.linalg.LinAlgError:
        return None, None
    if eigvals.size != 2:
        return None, None
    mod = np.abs(eigvals)
    stable_idx = int(np.argmin(mod))
    unstable_idx = int(np.argmax(mod))
    if not (mod[stable_idx] < 1.0 < mod[unstable_idx]):
        return None, None

    def _real_unit(idx: int) -> np.ndarray | None:
        vec = np.asarray(eigvecs[:, idx], dtype=complex)
        if np.max(np.abs(vec.imag)) > 1.0e-7 * max(1.0, float(np.max(np.abs(vec.real)))):
            return None
        real_vec = np.asarray(vec.real, dtype=float)
        norm = float(np.linalg.norm(real_vec))
        if not np.isfinite(norm) or norm <= 0.0:
            return None
        return real_vec / norm

    return _real_unit(stable_idx), _real_unit(unstable_idx)


def _classify_DPm_kind(DPm: np.ndarray, point_type: int | None = None) -> str:
    """Classify a section fixed point from the monodromy eigen-structure."""

    cls = classify_monodromy_2x2(DPm)
    if cls.kind == "U" and int(point_type or 0) == 1 and cls.area_preserving:
        return "X"
    return cls.kind


def _trace_poincare_points_field(
    field,
    seed_R: Sequence[float],
    seed_Z: Sequence[float],
    *,
    phi_section: float,
    N_turns: int,
    DPhi: float,
    wall_R: Sequence[float] | None = None,
    wall_Z: Sequence[float] | None = None,
    wall_phi: Sequence[float] | None = None,
    wall_R_all: np.ndarray | None = None,
    wall_Z_all: np.ndarray | None = None,
    extend_phi: bool = True,
    direction: str = "+",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Trace seeds and return per-seed section point arrays."""

    R0 = np.asarray(seed_R, dtype=float).ravel()
    Z0 = np.asarray(seed_Z, dtype=float).ravel()
    if R0.size != Z0.size:
        raise ValueError("seed_R and seed_Z must have the same length")
    if R0.size == 0:
        return []

    if wall_phi is not None or wall_R_all is not None or wall_Z_all is not None:
        if wall_phi is None or wall_R_all is None or wall_Z_all is None:
            raise ValueError("wall_phi, wall_R_all, and wall_Z_all must be supplied together")
        counts, flat_R, flat_Z = trace_poincare_batch_twall_field(
            field,
            R0,
            Z0,
            float(phi_section),
            int(N_turns),
            float(DPhi),
            np.asarray(wall_phi, dtype=float),
            np.asarray(wall_R_all, dtype=float),
            np.asarray(wall_Z_all, dtype=float),
            extend_phi=extend_phi,
            direction=direction,
        )
    else:
        if wall_R is None or wall_Z is None:
            wall_R, wall_Z = _field_grid_wall(field)
        counts, flat_R, flat_Z = trace_poincare_batch_field(
            field,
            R0,
            Z0,
            float(phi_section),
            int(N_turns),
            float(DPhi),
            np.asarray(wall_R, dtype=float),
            np.asarray(wall_Z, dtype=float),
            extend_phi=extend_phi,
            direction=direction,
        )

    counts_arr = np.asarray(counts, dtype=int).ravel()
    R_flat = np.asarray(flat_R, dtype=float).ravel()
    Z_flat = np.asarray(flat_Z, dtype=float).ravel()
    traces: list[tuple[np.ndarray, np.ndarray]] = []
    stride = int(N_turns)
    for i in range(R0.size):
        n = max(0, min(int(counts_arr[i]), stride))
        start = i * stride
        stop = start + n
        traces.append((R_flat[start:stop].copy(), Z_flat[start:stop].copy()))
    return traces


def _trace_map_sequence_field(
    field,
    R0: float,
    Z0: float,
    phi_section: float,
    *,
    n_steps: int,
    map_period: float,
    DPhi: float,
    wall_R: Sequence[float] | None = None,
    wall_Z: Sequence[float] | None = None,
    extend_phi: bool = True,
    fd_eps: float = 1.0e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return section-equivalent map iterates for an arbitrary map period."""

    if n_steps <= 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    phi_end = float(phi_section) + float(n_steps) * float(map_period)
    try:
        R_t, Z_t, _phi_t, _DP_t, alive = trace_orbit_along_phi_field(
            field,
            float(R0),
            float(Z0),
            float(phi_section),
            phi_end,
            float(DPhi),
            extend_phi=extend_phi,
            dphi_out=abs(float(map_period)),
            fd_eps=fd_eps,
        )
    except Exception:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    R_arr = np.asarray(R_t, dtype=float)
    Z_arr = np.asarray(Z_t, dtype=float)
    alive_arr = np.asarray(alive, dtype=bool)
    if R_arr.size <= 1:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    keep = min(int(n_steps), R_arr.size - 1)
    R_map = R_arr[1:keep + 1].copy()
    Z_map = Z_arr[1:keep + 1].copy()
    ok = alive_arr[1:keep + 1] if alive_arr.size >= keep + 1 else np.ones(keep, dtype=bool)
    ok &= np.isfinite(R_map) & np.isfinite(Z_map)
    if wall_R is not None and wall_Z is not None and ok.size:
        ok &= _points_in_polygon(R_map, Z_map, np.asarray(wall_R), np.asarray(wall_Z))
    if not np.all(ok):
        first_bad = int(np.flatnonzero(~ok)[0])
        R_map = R_map[:first_bad]
        Z_map = Z_map[:first_bad]
    return R_map, Z_map


def _trace_map_points_field(
    field,
    seed_R: Sequence[float],
    seed_Z: Sequence[float],
    *,
    phi_section: float,
    N_turns: int,
    map_period: float,
    DPhi: float,
    wall_R: Sequence[float] | None = None,
    wall_Z: Sequence[float] | None = None,
    extend_phi: bool = True,
    fd_eps: float = 1.0e-4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Trace arbitrary-period map points for multiple seeds."""

    R0 = np.asarray(seed_R, dtype=float).ravel()
    Z0 = np.asarray(seed_Z, dtype=float).ravel()
    if R0.size != Z0.size:
        raise ValueError("seed_R and seed_Z must have the same length")
    try:
        counts, flat_R, flat_Z = trace_map_batch_span_field(
            field,
            R0,
            Z0,
            float(phi_section),
            float(map_period),
            int(N_turns),
            float(DPhi),
            wall_R=wall_R,
            wall_Z=wall_Z,
            extend_phi=extend_phi,
        )
        counts_arr = np.asarray(counts, dtype=int).ravel()
        R_flat = np.asarray(flat_R, dtype=float).ravel()
        Z_flat = np.asarray(flat_Z, dtype=float).ravel()
        traces: list[tuple[np.ndarray, np.ndarray]] = []
        stride = int(N_turns)
        for i in range(R0.size):
            n = max(0, min(int(counts_arr[i]), stride))
            start = i * stride
            stop = start + n
            traces.append((R_flat[start:stop].copy(), Z_flat[start:stop].copy()))
        return traces
    except ImportError:
        pass
    return [
        _trace_map_sequence_field(
            field,
            float(r),
            float(z),
            float(phi_section),
            n_steps=int(N_turns),
            map_period=float(map_period),
            DPhi=float(DPhi),
            wall_R=wall_R,
            wall_Z=wall_Z,
            extend_phi=extend_phi,
            fd_eps=fd_eps,
        )
        for r, z in zip(R0, Z0)
    ]


def _map_endpoint_field(
    field,
    R: float,
    Z: float,
    phi: float,
    period: int,
    map_period: float,
    DPhi: float,
    *,
    extend_phi: bool,
    fd_eps: float,
) -> tuple[float, float] | None:
    R_map, Z_map = _trace_map_sequence_field(
        field,
        float(R),
        float(Z),
        float(phi),
        n_steps=int(period),
        map_period=float(map_period),
        DPhi=float(DPhi),
        extend_phi=extend_phi,
        fd_eps=fd_eps,
    )
    if R_map.size < period:
        return None
    return float(R_map[-1]), float(Z_map[-1])


def _map_residual_field(
    field,
    x: np.ndarray,
    phi: float,
    period: int,
    map_period: float,
    DPhi: float,
    *,
    extend_phi: bool,
    fd_eps: float,
) -> np.ndarray | None:
    endpoint = _map_endpoint_field(
        field,
        float(x[0]),
        float(x[1]),
        float(phi),
        int(period),
        float(map_period),
        float(DPhi),
        extend_phi=extend_phi,
        fd_eps=fd_eps,
    )
    if endpoint is None:
        return None
    F = np.asarray([endpoint[0] - float(x[0]), endpoint[1] - float(x[1])], dtype=float)
    if not np.all(np.isfinite(F)):
        return None
    return F


def _finite_difference_DPm_field(
    field,
    x: np.ndarray,
    phi: float,
    period: int,
    map_period: float,
    DPhi: float,
    *,
    extend_phi: bool,
    fd_eps: float,
) -> np.ndarray | None:
    mat = np.empty((2, 2), dtype=float)
    for j in range(2):
        step = np.zeros(2, dtype=float)
        step[j] = float(fd_eps)
        plus = _map_endpoint_field(
            field,
            float(x[0] + step[0]),
            float(x[1] + step[1]),
            float(phi),
            int(period),
            float(map_period),
            float(DPhi),
            extend_phi=extend_phi,
            fd_eps=fd_eps,
        )
        minus = _map_endpoint_field(
            field,
            float(x[0] - step[0]),
            float(x[1] - step[1]),
            float(phi),
            int(period),
            float(map_period),
            float(DPhi),
            extend_phi=extend_phi,
            fd_eps=fd_eps,
        )
        if plus is None or minus is None:
            return None
        mat[:, j] = (
            np.asarray(plus, dtype=float) - np.asarray(minus, dtype=float)
        ) / (2.0 * float(fd_eps))
    return mat


def _newton_fixed_point_period_map_field(
    field,
    seed_R: float,
    seed_Z: float,
    phi: float,
    period: int,
    map_period: float,
    DPhi: float,
    *,
    extend_phi: bool,
    fd_eps: float,
    max_iter: int,
    tol: float,
    trust_radius: float,
) -> tuple[float, float, float, bool, np.ndarray, np.ndarray, int]:
    """Damped Newton solve for arbitrary-period maps."""

    x = np.asarray([float(seed_R), float(seed_Z)], dtype=float)
    best_res = np.inf
    for _it in range(int(max_iter)):
        F = _map_residual_field(
            field,
            x,
            float(phi),
            int(period),
            float(map_period),
            float(DPhi),
            extend_phi=extend_phi,
            fd_eps=fd_eps,
        )
        if F is None:
            break
        res = float(np.linalg.norm(F))
        best_res = min(best_res, res)
        if res <= float(tol):
            DPm = _finite_difference_DPm_field(
                field,
                x,
                float(phi),
                int(period),
                float(map_period),
                float(DPhi),
                extend_phi=extend_phi,
                fd_eps=fd_eps,
            )
            if DPm is None:
                DPm = np.eye(2)
            eig = np.linalg.eigvals(DPm)
            ptype = 1 if abs(float(np.trace(DPm))) > 2.0 else 0
            return float(x[0]), float(x[1]), res, True, DPm, eig, ptype
        JF = np.empty((2, 2), dtype=float)
        ok = True
        for j in range(2):
            step = np.zeros(2, dtype=float)
            step[j] = float(fd_eps)
            Fp = _map_residual_field(
                field, x + step, float(phi), int(period), float(map_period), float(DPhi),
                extend_phi=extend_phi, fd_eps=fd_eps,
            )
            Fm = _map_residual_field(
                field, x - step, float(phi), int(period), float(map_period), float(DPhi),
                extend_phi=extend_phi, fd_eps=fd_eps,
            )
            if Fp is None or Fm is None:
                ok = False
                break
            JF[:, j] = (Fp - Fm) / (2.0 * float(fd_eps))
        if not ok:
            break
        try:
            step = np.linalg.solve(JF, -F)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(JF, -F, rcond=None)[0]
        step_norm = float(np.linalg.norm(step))
        if not np.isfinite(step_norm) or step_norm <= 0.0:
            break
        if step_norm > trust_radius:
            step *= float(trust_radius) / step_norm
        accepted = False
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            trial = x + alpha * step
            Ft = _map_residual_field(
                field,
                trial,
                float(phi),
                int(period),
                float(map_period),
                float(DPhi),
                extend_phi=extend_phi,
                fd_eps=fd_eps,
            )
            if Ft is None:
                continue
            trial_res = float(np.linalg.norm(Ft))
            if np.isfinite(trial_res) and trial_res < res:
                x = trial
                accepted = True
                break
        if not accepted:
            break

    DPm_fail = np.eye(2)
    eig_fail = np.asarray([np.nan + 0j, np.nan + 0j])
    return np.nan, np.nan, float(best_res), False, DPm_fail, eig_fail, -1


def _ray_polygon_radius(
    axis_R: float,
    axis_Z: float,
    theta: float,
    wall_R: np.ndarray,
    wall_Z: np.ndarray,
) -> float:
    """Distance from axis to the first polygon hit along ``theta``."""

    dR = float(np.cos(theta))
    dZ = float(np.sin(theta))
    radii: list[float] = []
    Rv = np.asarray(wall_R, dtype=float).ravel()
    Zv = np.asarray(wall_Z, dtype=float).ravel()
    n = Rv.size
    if n < 3 or n != Zv.size:
        return np.nan
    for i in range(n):
        x1 = float(Rv[i] - axis_R)
        y1 = float(Zv[i] - axis_Z)
        x2 = float(Rv[(i + 1) % n] - axis_R)
        y2 = float(Zv[(i + 1) % n] - axis_Z)
        sR = x2 - x1
        sZ = y2 - y1
        denom = dR * sZ - dZ * sR
        if abs(denom) < 1.0e-14:
            continue
        t = (x1 * sZ - y1 * sR) / denom
        u = (x1 * dZ - y1 * dR) / denom
        if t > 0.0 and -1.0e-12 <= u <= 1.0 + 1.0e-12:
            radii.append(float(t))
    if not radii:
        return np.nan
    return min(radii)


def _points_in_polygon(
    R: np.ndarray,
    Z: np.ndarray,
    wall_R: np.ndarray,
    wall_Z: np.ndarray,
) -> np.ndarray:
    """Vectorized even-odd point-in-polygon test."""

    Rq = np.asarray(R, dtype=float)
    Zq = np.asarray(Z, dtype=float)
    x = np.asarray(wall_R, dtype=float).ravel()
    y = np.asarray(wall_Z, dtype=float).ravel()
    if x.size < 3 or x.size != y.size:
        return np.zeros(Rq.shape, dtype=bool)
    inside = np.zeros(Rq.shape, dtype=bool)
    j = x.size - 1
    for i in range(x.size):
        yi = y[i]
        yj = y[j]
        xi = x[i]
        xj = x[j]
        crosses = (yi > Zq) != (yj > Zq)
        x_at = (xj - xi) * (Zq - yi) / (yj - yi + 1.0e-300) + xi
        inside ^= crosses & (Rq < x_at)
        j = i
    return inside


def boundary_seed_grid(
    axis_R: float,
    axis_Z: float,
    *,
    field=None,
    wall_R: Sequence[float] | None = None,
    wall_Z: Sequence[float] | None = None,
    radii: Sequence[float] | None = None,
    r_min: float | None = None,
    r_max: float | None = None,
    wall_fraction_min: float = 0.62,
    wall_fraction_max: float = 0.96,
    n_r: int = 6,
    n_theta: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    """Build seed rings biased toward the wall-facing boundary region."""

    if n_r <= 0 or n_theta <= 0:
        raise ValueError("n_r and n_theta must be positive")

    theta = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False)
    if radii is not None:
        rr = np.asarray(radii, dtype=float).ravel()
        if rr.size == 0:
            raise ValueError("radii must not be empty")
        R = axis_R + rr[:, None] * np.cos(theta)[None, :]
        Z = axis_Z + rr[:, None] * np.sin(theta)[None, :]
        return R.ravel(), Z.ravel()

    if wall_R is not None and wall_Z is not None:
        wall_R_arr = np.asarray(wall_R, dtype=float)
        wall_Z_arr = np.asarray(wall_Z, dtype=float)
        fractions = np.linspace(float(wall_fraction_min), float(wall_fraction_max), int(n_r))
        R_list: list[float] = []
        Z_list: list[float] = []
        for th in theta:
            rho = _ray_polygon_radius(axis_R, axis_Z, float(th), wall_R_arr, wall_Z_arr)
            if not np.isfinite(rho) or rho <= 0.0:
                continue
            for frac in fractions:
                R_list.append(float(axis_R + frac * rho * np.cos(th)))
                Z_list.append(float(axis_Z + frac * rho * np.sin(th)))
        if R_list:
            return np.asarray(R_list, dtype=float), np.asarray(Z_list, dtype=float)

    if r_max is None:
        if field is not None:
            R_arr = np.asarray(getattr(field, "R_arr", getattr(field, "R", [])), dtype=float)
            Z_arr = np.asarray(getattr(field, "Z_arr", getattr(field, "Z", [])), dtype=float)
            margins = [
                axis_R - float(np.nanmin(R_arr)),
                float(np.nanmax(R_arr)) - axis_R,
                axis_Z - float(np.nanmin(Z_arr)),
                float(np.nanmax(Z_arr)) - axis_Z,
            ]
            finite_margins = [m for m in margins if np.isfinite(m) and m > 0.0]
            r_max = 0.82 * min(finite_margins) if finite_margins else 0.1
        else:
            r_max = 0.1
    if r_min is None:
        r_min = 0.45 * float(r_max)
    rr = np.linspace(float(r_min), float(r_max), int(n_r))
    R = float(axis_R) + rr[:, None] * np.cos(theta)[None, :]
    Z = float(axis_Z) + rr[:, None] * np.sin(theta)[None, :]
    return R.ravel(), Z.ravel()


def boundary_recurrence_seed_candidates_field(
    field,
    axis_R: float,
    axis_Z: float,
    *,
    phi_section: float = 0.0,
    periods: int | Iterable[int] = (2, 3, 4, 5, 6, 7, 8, 9, 10),
    seed_R: Sequence[float] | None = None,
    seed_Z: Sequence[float] | None = None,
    radii: Sequence[float] | None = None,
    r_min: float | None = None,
    r_max: float | None = None,
    wall_R: Sequence[float] | None = None,
    wall_Z: Sequence[float] | None = None,
    wall_phi: Sequence[float] | None = None,
    wall_R_all: np.ndarray | None = None,
    wall_Z_all: np.ndarray | None = None,
    wall_fraction_min: float = 0.62,
    wall_fraction_max: float = 0.96,
    n_r: int = 6,
    n_theta: int = 96,
    N_turns: int = 120,
    map_period: float = 2.0 * np.pi,
    DPhi: float = 0.01,
    fd_eps: float = 1.0e-4,
    recurrence_tol: float | None = None,
    candidates_per_period: int = 96,
    candidate_dedup_tol: float = 1.0e-3,
    extend_phi: bool = True,
    direction: str = "+",
) -> BoundaryIslandSeedCandidates:
    """Find Newton seeds from near-periodic returns of boundary Poincare traces.

    Candidate points are selected only from the discrete map residual
    ``|P^m(x)-x|`` along traced Poincare orbits.  No magnetic-field-strength
    extrema are used.
    """

    period_values = _as_periods(periods)
    if N_turns <= max(period_values):
        raise ValueError("N_turns must be larger than the largest requested period")
    if candidates_per_period <= 0:
        raise ValueError("candidates_per_period must be positive")

    if seed_R is None or seed_Z is None:
        base_R, base_Z = boundary_seed_grid(
            float(axis_R),
            float(axis_Z),
            field=field,
            wall_R=wall_R,
            wall_Z=wall_Z,
            radii=radii,
            r_min=r_min,
            r_max=r_max,
            wall_fraction_min=wall_fraction_min,
            wall_fraction_max=wall_fraction_max,
            n_r=n_r,
            n_theta=n_theta,
        )
    else:
        base_R = np.asarray(seed_R, dtype=float).ravel()
        base_Z = np.asarray(seed_Z, dtype=float).ravel()
        if base_R.size != base_Z.size:
            raise ValueError("seed_R and seed_Z must have the same length")

    if wall_R is not None and wall_Z is not None:
        inside = _points_in_polygon(base_R, base_Z, np.asarray(wall_R), np.asarray(wall_Z))
        base_R = base_R[inside]
        base_Z = base_Z[inside]
    if base_R.size == 0:
        raise ValueError("boundary seed grid is empty")

    if np.isclose(float(map_period), 2.0 * np.pi, rtol=0.0, atol=1.0e-12):
        traces = _trace_poincare_points_field(
            field,
            base_R,
            base_Z,
            phi_section=float(phi_section),
            N_turns=int(N_turns),
            DPhi=float(DPhi),
            wall_R=wall_R,
            wall_Z=wall_Z,
            wall_phi=wall_phi,
            wall_R_all=wall_R_all,
            wall_Z_all=wall_Z_all,
            extend_phi=extend_phi,
            direction=direction,
        )
    else:
        period_sign = 1.0 if direction != "-" else -1.0
        traces = _trace_map_points_field(
            field,
            base_R,
            base_Z,
            phi_section=float(phi_section),
            N_turns=int(N_turns),
            map_period=period_sign * abs(float(map_period)),
            DPhi=float(DPhi),
            wall_R=wall_R,
            wall_Z=wall_Z,
            extend_phi=extend_phi,
            fd_eps=fd_eps,
        )

    seeds_by_period: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    best_residual_by_period: dict[int, float] = {}
    raw_candidate_counts: dict[int, int] = {}
    accepted_counts: dict[int, int] = {}
    trace_lengths = [int(len(r)) for r, _ in traces]

    for period in period_values:
        scored: list[tuple[float, float, float, int, int]] = []
        for seed_idx, (R_trace, Z_trace) in enumerate(traces):
            n = int(len(R_trace))
            if n <= period:
                continue
            d = np.hypot(R_trace[period:] - R_trace[:-period], Z_trace[period:] - Z_trace[:-period])
            if d.size == 0:
                continue
            finite_idx = np.flatnonzero(np.isfinite(d))
            for k in finite_idx:
                residual = float(d[k])
                if recurrence_tol is not None and residual > recurrence_tol:
                    continue
                scored.append((residual, float(R_trace[k]), float(Z_trace[k]), seed_idx, int(k)))

        scored.sort(key=lambda item: item[0])
        raw_candidate_counts[int(period)] = int(len(scored))
        best_residual_by_period[int(period)] = float(scored[0][0]) if scored else np.inf

        keep_R: list[float] = []
        keep_Z: list[float] = []
        for _residual, r, z, _seed_idx, _k in scored:
            keep_R.append(r)
            keep_Z.append(z)
            if len(keep_R) >= int(candidates_per_period) * 4:
                break
        cand_R, cand_Z = _deduplicate_seed_points(
            keep_R,
            keep_Z,
            tol=float(candidate_dedup_tol),
        )
        if cand_R.size > candidates_per_period:
            cand_R = cand_R[:candidates_per_period]
            cand_Z = cand_Z[:candidates_per_period]
        seeds_by_period[int(period)] = (cand_R, cand_Z)
        accepted_counts[int(period)] = int(cand_R.size)

    return BoundaryIslandSeedCandidates(
        seeds_by_period=seeds_by_period,
        diagnostics={
            "periods": list(period_values),
            "n_base_seeds": int(base_R.size),
            "N_turns": int(N_turns),
            "map_period": float(map_period),
            "trace_length_min": int(min(trace_lengths)) if trace_lengths else 0,
            "trace_length_max": int(max(trace_lengths)) if trace_lengths else 0,
            "raw_candidate_counts": raw_candidate_counts,
            "accepted_counts": accepted_counts,
            "best_residual_by_period": best_residual_by_period,
        },
    )


def _lower_period_residual(
    field,
    R: float,
    Z: float,
    phi: float,
    period: int,
    map_period: float,
    DPhi: float,
    *,
    extend_phi: bool,
    fd_eps: float,
) -> float:
    if period <= 1:
        return np.inf
    divisors = [d for d in range(1, period) if period % d == 0]
    if not divisors:
        divisors = [1]
    best = np.inf
    for divisor in divisors:
        try:
            R_t, Z_t, _phi_t, _DP_t, alive = trace_orbit_along_phi_field(
                field,
                R,
                Z,
                phi,
                phi + float(map_period) * divisor,
                DPhi,
                extend_phi=extend_phi,
                dphi_out=abs(float(map_period) * divisor),
                fd_eps=fd_eps,
            )
        except Exception:
            continue
        if len(R_t) == 0 or not bool(np.asarray(alive)[-1]):
            continue
        res = float(np.hypot(float(R_t[-1]) - R, float(Z_t[-1]) - Z))
        if res < best:
            best = res
    return best


def _deduplicate_fixed_points(
    fixed_points: list[BoundaryIslandFixedPoint],
    *,
    tol: float,
) -> tuple[BoundaryIslandFixedPoint, ...]:
    kept: list[BoundaryIslandFixedPoint] = []
    for fp in sorted(fixed_points, key=lambda x: (x.period, x.kind, x.residual)):
        duplicate_index = -1
        for i, old in enumerate(kept):
            if fp.kind != old.kind:
                continue
            if np.hypot(fp.R - old.R, fp.Z - old.Z) <= tol:
                duplicate_index = i
                break
        if duplicate_index < 0:
            kept.append(fp)
            continue
        old = kept[duplicate_index]
        if (fp.period, fp.residual) < (old.period, old.residual):
            kept[duplicate_index] = fp
    return tuple(sorted(kept, key=lambda x: (x.period, x.kind, np.arctan2(x.Z, x.R))))


def fixed_points_by_section_payload(
    fixed_points: Sequence[BoundaryIslandFixedPoint],
    phi_sections: Sequence[float] | None = None,
) -> dict[float, dict[str, list[FixedPoint]]]:
    """Convert boundary island fixed points to the plotting ``fp_by_sec`` form."""

    if phi_sections is None:
        keys = sorted({float(fp.phi) for fp in fixed_points})
    else:
        keys = [float(p) for p in phi_sections]
    payload = {float(phi): {"xpts": [], "opts": []} for phi in keys}
    for fp in fixed_points:
        phi_key = min(payload, key=lambda p: abs(p - float(fp.phi))) if payload else float(fp.phi)
        if phi_key not in payload:
            payload[phi_key] = {"xpts": [], "opts": []}
        if fp.kind == "X":
            bucket = "xpts"
        elif fp.kind == "O":
            bucket = "opts"
        else:
            bucket = "unknown"
        payload[phi_key].setdefault(bucket, [])
        payload[phi_key][bucket].append(fp.as_fixed_point())
    return payload


def find_boundary_island_fixed_points_field(
    field,
    axis_R: float,
    axis_Z: float,
    *,
    phi_section: float = 0.0,
    periods: int | Iterable[int] = (2, 3, 4, 5, 6, 7, 8, 9, 10),
    radii: Sequence[float] | None = None,
    r_min: float | None = None,
    r_max: float | None = None,
    wall_R: Sequence[float] | None = None,
    wall_Z: Sequence[float] | None = None,
    wall_fraction_min: float = 0.62,
    wall_fraction_max: float = 0.96,
    n_r: int = 6,
    n_theta: int = 96,
    wall_phi: Sequence[float] | None = None,
    wall_R_all: np.ndarray | None = None,
    wall_Z_all: np.ndarray | None = None,
    candidate_strategy: str = "grid",
    recurrence_turns: int = 120,
    recurrence_tol: float | None = None,
    recurrence_candidates_per_period: int = 96,
    candidate_dedup_tol: float = 1.0e-3,
    map_period: float = 2.0 * np.pi,
    python_trust_radius: float = 0.08,
    DPhi: float = 0.01,
    fd_eps: float = 1.0e-4,
    max_iter: int = 60,
    tol: float = 1.0e-10,
    residual_tol: float | None = None,
    dedup_tol: float = 2.0e-3,
    exclude_lower_periods: bool = True,
    lower_period_tol: float = 2.0e-3,
    extend_phi: bool = True,
    n_threads: int = -1,
) -> BoundaryIslandSearchResult:
    """Find X/O points of boundary island chains with a damped Newton backend."""

    period_values = _as_periods(periods)
    strategy = str(candidate_strategy).strip().lower()
    if strategy not in {"grid", "recurrence", "both"}:
        raise ValueError("candidate_strategy must be 'grid', 'recurrence', or 'both'")
    seed_R, seed_Z = boundary_seed_grid(
        float(axis_R),
        float(axis_Z),
        field=field,
        wall_R=wall_R,
        wall_Z=wall_Z,
        radii=radii,
        r_min=r_min,
        r_max=r_max,
        wall_fraction_min=wall_fraction_min,
        wall_fraction_max=wall_fraction_max,
        n_r=n_r,
        n_theta=n_theta,
    )
    if wall_R is not None and wall_Z is not None:
        inside = _points_in_polygon(seed_R, seed_Z, np.asarray(wall_R), np.asarray(wall_Z))
        seed_R = seed_R[inside]
        seed_Z = seed_Z[inside]
    if seed_R.size == 0:
        raise ValueError("boundary seed grid is empty")

    recurrence_candidates: BoundaryIslandSeedCandidates | None = None
    if strategy in {"recurrence", "both"}:
        recurrence_candidates = boundary_recurrence_seed_candidates_field(
            field,
            float(axis_R),
            float(axis_Z),
            phi_section=float(phi_section),
            periods=period_values,
            seed_R=seed_R,
            seed_Z=seed_Z,
            wall_R=wall_R,
            wall_Z=wall_Z,
            wall_phi=wall_phi,
            wall_R_all=wall_R_all,
            wall_Z_all=wall_Z_all,
            N_turns=int(recurrence_turns),
            map_period=float(map_period),
            DPhi=float(DPhi),
            fd_eps=float(fd_eps),
            recurrence_tol=recurrence_tol,
            candidates_per_period=int(recurrence_candidates_per_period),
            candidate_dedup_tol=float(candidate_dedup_tol),
            extend_phi=extend_phi,
        )

    if residual_tol is None:
        residual_tol = max(20.0 * float(tol), 1.0e-8)

    candidates: list[BoundaryIslandFixedPoint] = []
    raw_counts: dict[int, int] = {}
    converged_counts: dict[int, int] = {}
    accepted_counts: dict[int, int] = {}
    period_seed_counts: dict[int, int] = {}
    all_used_seed_R: list[np.ndarray] = []
    all_used_seed_Z: list[np.ndarray] = []

    for period in period_values:
        if strategy == "grid":
            period_seed_R, period_seed_Z = seed_R, seed_Z
        else:
            assert recurrence_candidates is not None
            rec_R, rec_Z = recurrence_candidates.seeds_for_period(int(period))
            if strategy == "recurrence":
                period_seed_R, period_seed_Z = rec_R, rec_Z
            else:
                period_seed_R = np.concatenate([seed_R, rec_R])
                period_seed_Z = np.concatenate([seed_Z, rec_Z])
                period_seed_R, period_seed_Z = _deduplicate_seed_points(
                    period_seed_R,
                    period_seed_Z,
                    tol=float(candidate_dedup_tol),
                )
        period_seed_R = np.asarray(period_seed_R, dtype=float).ravel()
        period_seed_Z = np.asarray(period_seed_Z, dtype=float).ravel()
        period_seed_counts[int(period)] = int(period_seed_R.size)
        if period_seed_R.size == 0:
            raw_counts[int(period)] = 0
            converged_counts[int(period)] = 0
            accepted_counts[int(period)] = 0
            continue
        all_used_seed_R.append(period_seed_R.copy())
        all_used_seed_Z.append(period_seed_Z.copy())

        if np.isclose(float(map_period), 2.0 * np.pi, rtol=0.0, atol=1.0e-12):
            result = find_fixed_points_batch_field(
                field,
                period_seed_R,
                period_seed_Z,
                float(phi_section),
                int(period),
                1,
                float(DPhi),
                extend_phi=extend_phi,
                fd_eps=fd_eps,
                max_iter=max_iter,
                tol=tol,
                n_threads=n_threads,
            )
            R_out, Z_out, residual, converged, DPm_flat, eig_r, eig_i, point_type = result
            R_out = np.asarray(R_out, dtype=float)
            Z_out = np.asarray(Z_out, dtype=float)
            residual = np.asarray(residual, dtype=float)
            converged = np.asarray(converged).astype(bool)
            point_type = np.asarray(point_type)
            DPm_arr = np.asarray(DPm_flat, dtype=float).reshape(len(R_out), 2, 2)
            eig = np.asarray(eig_r, dtype=float) + 1j * np.asarray(eig_i, dtype=float)
            if eig.ndim == 1:
                eig = eig.reshape(len(R_out), -1)
        else:
            try:
                result = find_fixed_points_batch_span_field(
                    field,
                    period_seed_R,
                    period_seed_Z,
                    float(phi_section),
                    float(period) * float(map_period),
                    float(DPhi),
                    extend_phi=extend_phi,
                    fd_eps=float(fd_eps),
                    max_iter=int(max_iter),
                    tol=float(tol),
                    n_threads=n_threads,
                )
                R_out, Z_out, residual, converged, DPm_flat, eig_r, eig_i, point_type = result
                R_out = np.asarray(R_out, dtype=float)
                Z_out = np.asarray(Z_out, dtype=float)
                residual = np.asarray(residual, dtype=float)
                converged = np.asarray(converged).astype(bool)
                point_type = np.asarray(point_type)
                DPm_arr = np.asarray(DPm_flat, dtype=float).reshape(len(R_out), 2, 2)
                eig = np.asarray(eig_r, dtype=float) + 1j * np.asarray(eig_i, dtype=float)
                if eig.ndim == 1:
                    eig = eig.reshape(len(R_out), -1)
            except ImportError:
                rows = [
                    _newton_fixed_point_period_map_field(
                        field,
                        float(r),
                        float(z),
                        float(phi_section),
                        int(period),
                        float(map_period),
                        float(DPhi),
                        extend_phi=extend_phi,
                        fd_eps=float(fd_eps),
                        max_iter=int(max_iter),
                        tol=float(tol),
                        trust_radius=float(python_trust_radius),
                    )
                    for r, z in zip(period_seed_R, period_seed_Z)
                ]
                R_out = np.asarray([row[0] for row in rows], dtype=float)
                Z_out = np.asarray([row[1] for row in rows], dtype=float)
                residual = np.asarray([row[2] for row in rows], dtype=float)
                converged = np.asarray([row[3] for row in rows], dtype=bool)
                DPm_arr = np.asarray([row[4] for row in rows], dtype=float).reshape(len(rows), 2, 2)
                eig = np.asarray([row[5] for row in rows], dtype=complex)
                point_type = np.asarray([row[6] for row in rows], dtype=int)

        raw_counts[int(period)] = int(len(R_out))
        converged_counts[int(period)] = int(np.count_nonzero(converged))
        accepted_before = len(candidates)
        for i in range(len(R_out)):
            if not converged[i]:
                continue
            if not (
                np.isfinite(R_out[i])
                and np.isfinite(Z_out[i])
                and np.isfinite(residual[i])
                and residual[i] <= residual_tol
            ):
                continue
            if wall_R is not None and wall_Z is not None:
                if not bool(_points_in_polygon(
                    np.asarray([R_out[i]]),
                    np.asarray([Z_out[i]]),
                    np.asarray(wall_R),
                    np.asarray(wall_Z),
                )[0]):
                    continue
            lower_res = _lower_period_residual(
                field,
                float(R_out[i]),
                float(Z_out[i]),
                float(phi_section),
                int(period),
                float(map_period),
                float(DPhi),
                extend_phi=extend_phi,
                fd_eps=fd_eps,
            )
            if exclude_lower_periods and lower_res <= lower_period_tol:
                continue
            kind = _classify_DPm_kind(DPm_arr[i], int(point_type[i]))
            candidates.append(BoundaryIslandFixedPoint(
                phi=float(phi_section),
                R=float(R_out[i]),
                Z=float(Z_out[i]),
                period=int(period),
                kind=kind,
                DPm=DPm_arr[i].copy(),
                residual=float(residual[i]),
                eigenvalues=np.asarray(eig[i]).copy(),
                seed_R=float(period_seed_R[i]),
                seed_Z=float(period_seed_Z[i]),
                lower_period_residual=float(lower_res),
            ))
        accepted_counts[int(period)] = int(len(candidates) - accepted_before)

    fixed = _deduplicate_fixed_points(candidates, tol=float(dedup_tol))
    fp_by_sec = fixed_points_by_section_payload(fixed, [float(phi_section)])
    used_seed_R = (
        np.concatenate(all_used_seed_R) if all_used_seed_R else np.empty(0, dtype=float)
    )
    used_seed_Z = (
        np.concatenate(all_used_seed_Z) if all_used_seed_Z else np.empty(0, dtype=float)
    )
    used_seed_R, used_seed_Z = _deduplicate_seed_points(
        used_seed_R,
        used_seed_Z,
        tol=float(candidate_dedup_tol),
    )
    diagnostics = {
        "periods": list(period_values),
        "n_seeds": int(seed_R.size),
        "candidate_strategy": strategy,
        "map_period": float(map_period),
        "period_seed_counts": period_seed_counts,
        "raw_counts": raw_counts,
        "converged_counts": converged_counts,
        "accepted_counts": accepted_counts,
        "n_fixed_points": int(len(fixed)),
    }
    if recurrence_candidates is not None:
        diagnostics["recurrence"] = recurrence_candidates.diagnostics
    return BoundaryIslandSearchResult(
        fixed_points=fixed,
        fp_by_sec=fp_by_sec,
        seed_R=used_seed_R if used_seed_R.size else seed_R,
        seed_Z=used_seed_Z if used_seed_Z.size else seed_Z,
        diagnostics=diagnostics,
    )


def find_boundary_island_fixed_points_multi_section_field(
    field,
    axis_by_sec: Sequence[tuple[float, float]],
    phi_sections: Sequence[float],
    *,
    wall_by_sec: Sequence[tuple[Sequence[float], Sequence[float]]] | None = None,
    **kwargs,
) -> BoundaryIslandSearchResult:
    """Run boundary fixed-point searches on multiple toroidal sections."""

    if len(axis_by_sec) != len(phi_sections):
        raise ValueError("axis_by_sec and phi_sections must have the same length")
    if wall_by_sec is not None and len(wall_by_sec) != len(phi_sections):
        raise ValueError("wall_by_sec and phi_sections must have the same length")
    all_fixed: list[BoundaryIslandFixedPoint] = []
    all_seed_R: list[np.ndarray] = []
    all_seed_Z: list[np.ndarray] = []
    section_diag: dict[float, dict] = {}
    for idx, ((axis_R, axis_Z), phi) in enumerate(zip(axis_by_sec, phi_sections)):
        section_kwargs = dict(kwargs)
        if wall_by_sec is not None and "wall_R" not in section_kwargs and "wall_Z" not in section_kwargs:
            section_kwargs["wall_R"], section_kwargs["wall_Z"] = wall_by_sec[idx]
        result = find_boundary_island_fixed_points_field(
            field,
            float(axis_R),
            float(axis_Z),
            phi_section=float(phi),
            **section_kwargs,
        )
        all_fixed.extend(result.fixed_points)
        all_seed_R.append(result.seed_R)
        all_seed_Z.append(result.seed_Z)
        section_diag[float(phi)] = result.diagnostics
    fixed = tuple(all_fixed)
    return BoundaryIslandSearchResult(
        fixed_points=fixed,
        fp_by_sec=fixed_points_by_section_payload(fixed, phi_sections),
        seed_R=np.concatenate(all_seed_R) if all_seed_R else np.empty(0),
        seed_Z=np.concatenate(all_seed_Z) if all_seed_Z else np.empty(0),
        diagnostics={
            "sections": section_diag,
            "n_fixed_points": int(len(fixed)),
        },
    )


def _fixed_points_for_section(
    fp_by_sec: dict[float, dict[str, list[FixedPoint]]],
    phi: float,
) -> tuple[list[FixedPoint], list[FixedPoint]]:
    if not fp_by_sec:
        return [], []
    phi_key = min(fp_by_sec, key=lambda p: abs(float(p) - float(phi)))
    payload = fp_by_sec.get(phi_key, {})
    if isinstance(payload, dict):
        return list(payload.get("xpts", [])), list(payload.get("opts", []))
    xpts = [fp for fp in payload if str(getattr(fp, "kind", "")).upper() == "X"]
    opts = [fp for fp in payload if str(getattr(fp, "kind", "")).upper() == "O"]
    return xpts, opts


def _sorted_curve(R: Sequence[float], Z: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    R_arr = np.asarray(R, dtype=float).ravel()
    Z_arr = np.asarray(Z, dtype=float).ravel()
    finite = np.isfinite(R_arr) & np.isfinite(Z_arr)
    R_arr = R_arr[finite]
    Z_arr = Z_arr[finite]
    if R_arr.size < 3:
        return R_arr, Z_arr
    cR = float(np.mean(R_arr))
    cZ = float(np.mean(Z_arr))
    order = np.argsort(np.arctan2(Z_arr - cZ, R_arr - cR))
    return R_arr[order], Z_arr[order]


def trace_boundary_island_shapes_field(
    field,
    fixed_points: Sequence[BoundaryIslandFixedPoint | FixedPoint],
    axis_R: float,
    axis_Z: float,
    *,
    phi_section: float = 0.0,
    periods: int | Iterable[int] | None = None,
    shape_radius_fractions: Sequence[float] = (0.55, 0.78, 0.92),
    n_shape_angles: int = 6,
    N_turns: int = 80,
    map_period: float = 2.0 * np.pi,
    DPhi: float = 0.01,
    fd_eps: float = 1.0e-4,
    min_points: int = 12,
    wall_R: Sequence[float] | None = None,
    wall_Z: Sequence[float] | None = None,
    wall_phi: Sequence[float] | None = None,
    wall_R_all: np.ndarray | None = None,
    wall_Z_all: np.ndarray | None = None,
    extend_phi: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Trace readable island-shape curves around O-points.

    Seed scales are determined from X/O fixed-point geometry, then confirmed by
    actual Poincare tracing.  This is a plotting aid; X/O positions themselves
    still come from the fixed-point solver.
    """

    if n_shape_angles <= 0:
        raise ValueError("n_shape_angles must be positive")
    period_filter = None if periods is None else set(_as_periods(periods))
    fps = list(fixed_points)
    opts = [
        fp for fp in fps
        if str(getattr(fp, "kind", "")).upper() == "O"
        and (period_filter is None or int(getattr(fp, "period", 1)) in period_filter)
    ]
    xpts = [
        fp for fp in fps
        if str(getattr(fp, "kind", "")).upper() == "X"
        and (period_filter is None or int(getattr(fp, "period", 1)) in period_filter)
    ]
    if not opts:
        return []

    shapes: list[tuple[np.ndarray, np.ndarray]] = []
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_shape_angles), endpoint=False)
    frac_values = np.asarray(shape_radius_fractions, dtype=float).ravel()
    frac_values = frac_values[np.isfinite(frac_values) & (frac_values > 0.0)]
    if frac_values.size == 0:
        raise ValueError("shape_radius_fractions must contain positive values")

    for opt in opts:
        same_period_x = [
            x for x in xpts
            if int(getattr(x, "period", 1)) == int(getattr(opt, "period", 1))
        ]
        if same_period_x:
            distances = np.asarray([
                np.hypot(float(x.R) - float(opt.R), float(x.Z) - float(opt.Z))
                for x in same_period_x
            ], dtype=float)
            scale = float(np.nanmin(distances))
        else:
            scale = float(np.hypot(float(opt.R) - float(axis_R), float(opt.Z) - float(axis_Z)))
        if not np.isfinite(scale) or scale <= 0.0:
            continue

        for frac in frac_values:
            radius = float(frac) * scale
            seed_R = float(opt.R) + radius * np.cos(theta)
            seed_Z = float(opt.Z) + radius * np.sin(theta)
            if wall_R is not None and wall_Z is not None:
                inside = _points_in_polygon(seed_R, seed_Z, np.asarray(wall_R), np.asarray(wall_Z))
                seed_R = seed_R[inside]
                seed_Z = seed_Z[inside]
            if seed_R.size == 0:
                continue
            if np.isclose(float(map_period), 2.0 * np.pi, rtol=0.0, atol=1.0e-12):
                traces = _trace_poincare_points_field(
                    field,
                    seed_R,
                    seed_Z,
                    phi_section=float(phi_section),
                    N_turns=int(N_turns),
                    DPhi=float(DPhi),
                    wall_R=wall_R,
                    wall_Z=wall_Z,
                    wall_phi=wall_phi,
                    wall_R_all=wall_R_all,
                    wall_Z_all=wall_Z_all,
                    extend_phi=extend_phi,
                )
            else:
                traces = _trace_map_points_field(
                    field,
                    seed_R,
                    seed_Z,
                    phi_section=float(phi_section),
                    N_turns=int(N_turns),
                    map_period=float(map_period),
                    DPhi=float(DPhi),
                    wall_R=wall_R,
                    wall_Z=wall_Z,
                    extend_phi=extend_phi,
                    fd_eps=fd_eps,
                )
            for R_trace, Z_trace in traces:
                if len(R_trace) < min_points:
                    continue
                shapes.append(_sorted_curve(R_trace, Z_trace))
    return shapes


def boundary_island_edge_state_payload(
    shapes_by_sec: Sequence[Sequence[tuple[np.ndarray, np.ndarray]]],
) -> list[dict]:
    """Convert boundary island curves to topoquest ``edge_state_by_sec`` form."""

    payload: list[dict] = []
    for shapes in shapes_by_sec:
        curves = [(np.asarray(R, dtype=float), np.asarray(Z, dtype=float)) for R, Z in shapes]
        payload.append({
            "closed_core": [],
            "boundary_island": curves,
            "open_loss": [],
            "chaotic_edge": [],
            "counts": {
                "closed_core": 0,
                "boundary_island": int(len(curves)),
                "open_loss": 0,
                "chaotic_edge": 0,
            },
        })
    return payload


def trace_boundary_island_shapes_multi_section_field(
    field,
    fp_by_sec: dict[float, dict[str, list[FixedPoint]]],
    axis_by_sec: Sequence[tuple[float, float]],
    phi_sections: Sequence[float],
    *,
    wall_by_sec: Sequence[tuple[Sequence[float], Sequence[float]]] | None = None,
    **kwargs,
) -> list[dict]:
    """Trace island-shape curves on multiple sections as ``edge_state_by_sec``."""

    if len(axis_by_sec) != len(phi_sections):
        raise ValueError("axis_by_sec and phi_sections must have the same length")
    if wall_by_sec is not None and len(wall_by_sec) != len(phi_sections):
        raise ValueError("wall_by_sec and phi_sections must have the same length")
    shapes_by_sec: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for idx, ((axis_R, axis_Z), phi) in enumerate(zip(axis_by_sec, phi_sections)):
        xpts, opts = _fixed_points_for_section(fp_by_sec, float(phi))
        section_kwargs = dict(kwargs)
        if wall_by_sec is not None and "wall_R" not in section_kwargs and "wall_Z" not in section_kwargs:
            section_kwargs["wall_R"], section_kwargs["wall_Z"] = wall_by_sec[idx]
        shapes = trace_boundary_island_shapes_field(
            field,
            [*xpts, *opts],
            float(axis_R),
            float(axis_Z),
            phi_section=float(phi),
            **section_kwargs,
        )
        shapes_by_sec.append(shapes)
    return boundary_island_edge_state_payload(shapes_by_sec)


def trace_fixed_point_manifolds_field(
    field,
    fixed_points: Sequence[BoundaryIslandFixedPoint | FixedPoint],
    *,
    phi_section: float | None = None,
    N_turns: int = 24,
    map_period: float = 2.0 * np.pi,
    DPhi: float = 0.01,
    fd_eps: float = 1.0e-4,
    seed_distances: Sequence[float] | None = None,
    eps_min: float = 1.0e-6,
    eps_max: float = 1.0e-3,
    n_eps: int = 24,
    wall_R: Sequence[float] | None = None,
    wall_Z: Sequence[float] | None = None,
    wall_phi: Sequence[float] | None = None,
    wall_R_all: np.ndarray | None = None,
    wall_Z_all: np.ndarray | None = None,
    RZlimit: tuple[float, float, float, float] | None = None,
    extend_phi: bool = True,
) -> list[dict[str, np.ndarray]]:
    """Trace W^u/W^s point clouds from hyperbolic fixed points.

    ``W^u`` is traced forward from the unstable eigendirection; ``W^s`` is
    traced backward from the stable eigendirection.  The return format matches
    ``topoquest.plot.poincare``.
    """

    if N_turns <= 0:
        raise ValueError("N_turns must be positive")
    if seed_distances is None:
        if n_eps <= 0:
            raise ValueError("n_eps must be positive")
        distances = np.logspace(np.log10(float(eps_min)), np.log10(float(eps_max)), int(n_eps))
    else:
        distances = np.asarray(seed_distances, dtype=float).ravel()
    distances = distances[np.isfinite(distances) & (distances > 0.0)]
    if distances.size == 0:
        raise ValueError("seed_distances must contain positive finite values")

    def _flatten(traces: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
        if not traces:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        R_arr = np.concatenate([np.asarray(R, dtype=float).ravel() for R, _Z in traces])
        Z_arr = np.concatenate([np.asarray(Z, dtype=float).ravel() for _R, Z in traces])
        finite = np.isfinite(R_arr) & np.isfinite(Z_arr)
        if RZlimit is not None:
            r_min, r_max, z_min, z_max = map(float, RZlimit)
            finite &= (R_arr >= r_min) & (R_arr <= r_max) & (Z_arr >= z_min) & (Z_arr <= z_max)
        return R_arr[finite], Z_arr[finite]

    manifolds: list[dict[str, np.ndarray]] = []
    for fp in fixed_points:
        if str(getattr(fp, "kind", "")).upper() != "X":
            continue
        DPm = np.asarray(getattr(fp, "DPm", np.eye(2)), dtype=float).reshape(2, 2)
        stable, unstable = _stable_unstable_eigenvectors(DPm)
        if stable is None or unstable is None:
            continue

        phi = float(getattr(fp, "phi", 0.0) if phi_section is None else phi_section)
        R0 = float(getattr(fp, "R"))
        Z0 = float(getattr(fp, "Z"))
        sign = np.asarray([-1.0, 1.0], dtype=float)
        du = (sign[:, None] * distances[None, :]).ravel()
        ds = du.copy()
        seed_u_R = R0 + du * unstable[0]
        seed_u_Z = Z0 + du * unstable[1]
        seed_s_R = R0 + ds * stable[0]
        seed_s_Z = Z0 + ds * stable[1]

        if wall_R is not None and wall_Z is not None:
            inside_u = _points_in_polygon(seed_u_R, seed_u_Z, np.asarray(wall_R), np.asarray(wall_Z))
            seed_u_R = seed_u_R[inside_u]
            seed_u_Z = seed_u_Z[inside_u]
            inside_s = _points_in_polygon(seed_s_R, seed_s_Z, np.asarray(wall_R), np.asarray(wall_Z))
            seed_s_R = seed_s_R[inside_s]
            seed_s_Z = seed_s_Z[inside_s]

        if np.isclose(float(map_period), 2.0 * np.pi, rtol=0.0, atol=1.0e-12):
            u_traces = _trace_poincare_points_field(
                field,
                seed_u_R,
                seed_u_Z,
                phi_section=phi,
                N_turns=int(N_turns),
                DPhi=float(DPhi),
                wall_R=wall_R,
                wall_Z=wall_Z,
                wall_phi=wall_phi,
                wall_R_all=wall_R_all,
                wall_Z_all=wall_Z_all,
                extend_phi=extend_phi,
                direction="+",
            )
            s_traces = _trace_poincare_points_field(
                field,
                seed_s_R,
                seed_s_Z,
                phi_section=phi,
                N_turns=int(N_turns),
                DPhi=float(DPhi),
                wall_R=wall_R,
                wall_Z=wall_Z,
                wall_phi=wall_phi,
                wall_R_all=wall_R_all,
                wall_Z_all=wall_Z_all,
                extend_phi=extend_phi,
                direction="-",
            )
        else:
            u_traces = _trace_map_points_field(
                field,
                seed_u_R,
                seed_u_Z,
                phi_section=phi,
                N_turns=int(N_turns),
                map_period=abs(float(map_period)),
                DPhi=float(DPhi),
                wall_R=wall_R,
                wall_Z=wall_Z,
                extend_phi=extend_phi,
                fd_eps=fd_eps,
            )
            s_traces = _trace_map_points_field(
                field,
                seed_s_R,
                seed_s_Z,
                phi_section=phi,
                N_turns=int(N_turns),
                map_period=-abs(float(map_period)),
                DPhi=float(DPhi),
                wall_R=wall_R,
                wall_Z=wall_Z,
                extend_phi=extend_phi,
                fd_eps=fd_eps,
            )
        u_R, u_Z = _flatten(u_traces)
        s_R, s_Z = _flatten(s_traces)
        manifolds.append({
            "u_R": u_R,
            "u_Z": u_Z,
            "s_R": s_R,
            "s_Z": s_Z,
        })
    return manifolds


def trace_fixed_point_manifolds_multi_section_field(
    field,
    fp_by_sec: dict[float, dict[str, list[FixedPoint]]],
    phi_sections: Sequence[float],
    *,
    wall_by_sec: Sequence[tuple[Sequence[float], Sequence[float]]] | None = None,
    **kwargs,
) -> dict[float, list[dict[str, np.ndarray]]]:
    """Trace fixed-point manifolds for all supplied sections."""

    if wall_by_sec is not None and len(wall_by_sec) != len(phi_sections):
        raise ValueError("wall_by_sec and phi_sections must have the same length")
    out: dict[float, list[dict[str, np.ndarray]]] = {}
    for idx, phi in enumerate(phi_sections):
        xpts, _opts = _fixed_points_for_section(fp_by_sec, float(phi))
        section_kwargs = dict(kwargs)
        if wall_by_sec is not None and "wall_R" not in section_kwargs and "wall_Z" not in section_kwargs:
            section_kwargs["wall_R"], section_kwargs["wall_Z"] = wall_by_sec[idx]
        out[float(phi)] = trace_fixed_point_manifolds_field(
            field,
            xpts,
            phi_section=float(phi),
            **section_kwargs,
        )
    return out


def boundary_island_topology_payload_field(
    field,
    axis_by_sec: Sequence[tuple[float, float]],
    phi_sections: Sequence[float],
    *,
    periods: int | Iterable[int] = (2, 3, 4, 5, 6, 7, 8, 9, 10),
    wall_by_sec: Sequence[tuple[Sequence[float], Sequence[float]]] | None = None,
    search_kwargs: dict | None = None,
    shape_kwargs: dict | None = None,
    manifold_kwargs: dict | None = None,
) -> dict:
    """Build topoquest-ready boundary island-chain overlay payloads."""

    search = find_boundary_island_fixed_points_multi_section_field(
        field,
        axis_by_sec,
        phi_sections,
        wall_by_sec=wall_by_sec,
        periods=periods,
        **(search_kwargs or {}),
    )
    edge_state_by_sec = trace_boundary_island_shapes_multi_section_field(
        field,
        search.fp_by_sec,
        axis_by_sec,
        phi_sections,
        wall_by_sec=wall_by_sec,
        periods=periods,
        **(shape_kwargs or {}),
    )
    manifolds_by_sec = trace_fixed_point_manifolds_multi_section_field(
        field,
        search.fp_by_sec,
        phi_sections,
        wall_by_sec=wall_by_sec,
        **(manifold_kwargs or {}),
    )
    return {
        "fp_by_sec": search.fp_by_sec,
        "edge_state_by_sec": edge_state_by_sec,
        "manifolds_by_sec": manifolds_by_sec,
        "fixed_points": search.fixed_points,
        "diagnostics": search.diagnostics,
    }


__all__ = [
    "BoundaryIslandFixedPoint",
    "BoundaryIslandSeedCandidates",
    "BoundaryIslandSearchResult",
    "boundary_island_edge_state_payload",
    "boundary_island_topology_payload_field",
    "boundary_recurrence_seed_candidates_field",
    "boundary_seed_grid",
    "fixed_points_by_section_payload",
    "find_boundary_island_fixed_points_field",
    "find_boundary_island_fixed_points_multi_section_field",
    "trace_boundary_island_shapes_field",
    "trace_boundary_island_shapes_multi_section_field",
    "trace_fixed_point_manifolds_field",
    "trace_fixed_point_manifolds_multi_section_field",
]
