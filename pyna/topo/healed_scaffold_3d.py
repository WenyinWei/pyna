"""healed_scaffold_3d.py
=================================
Field-line-transported 3D scaffold for healed magnetic coordinates.

This module lifts the "reference section + field-line tracing" workflow from
project scripts (for example ``topoquest/scripts/w7x/w7x_healed_scaffold.py``)
into reusable ``pyna.topo`` infrastructure.

Besides discrete field-line transport, this module now also provides a more
robust section-local X/O ordering helper for building healed ``C_XO``
boundaries.  The key design choice is to avoid relying on a raw polar-angle
sort alone; instead we:

1. deduplicate points,
2. infer a smooth O-ring backbone,
3. assign each X-point to the most plausible O-slot using local chord geometry,
4. build an alternating O/X sequence with monotone arclength on that backbone,
5. validate the resulting curve against winding / slot / spacing heuristics.

This improves behaviour for edge island chains whose naive angular ordering can
break because of non-convexity, strong shaping, or boundary bending.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, List, Any, Dict

import numpy as np
from scipy.interpolate import CubicSpline


ArrayLike = np.ndarray
TraceFunction = Callable[[float, float, float, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]


def _wrap_angle(phi: float) -> float:
    """Wrap angle to [0, 2π)."""
    return float(phi % (2.0 * np.pi))


def _forward_span(phi_src: float, phi_tgt: float) -> float:
    """Forward toroidal span from ``phi_src`` to ``phi_tgt`` in [0, 2π)."""
    return float((phi_tgt - phi_src) % (2.0 * np.pi))


def _as_point_rows(points: Sequence[Any]) -> np.ndarray:
    if points is None:
        return np.empty((0, 2), dtype=float)
    try:
        arr0 = np.asarray(points, dtype=float)
    except Exception:
        arr0 = None
    if arr0 is not None and arr0.ndim == 2 and arr0.shape[1] >= 2:
        return np.asarray(arr0[:, :2], dtype=float)
    rows: List[List[float]] = []
    for pt in points:
        if hasattr(pt, "R") and hasattr(pt, "Z"):
            rows.append([float(pt.R), float(pt.Z)])
            continue
        if isinstance(pt, (tuple, list)) and len(pt) >= 2:
            try:
                rows.append([float(pt[0]), float(pt[1])])
                continue
            except Exception:
                pass
        arr = np.asarray(pt, dtype=float).ravel()
        if arr.size < 2:
            continue
        rows.append([float(arr[0]), float(arr[1])])
    return np.asarray(rows, dtype=float) if rows else np.empty((0, 2), dtype=float)


def _dedup_points(points: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    if len(points) <= 1:
        return points.copy()
    kept: List[np.ndarray] = []
    for p in points:
        if not kept:
            kept.append(p)
            continue
        if min(np.hypot(*(p - q)) for q in kept) > tol:
            kept.append(p)
    return np.asarray(kept, dtype=float)


def _pairwise_dist(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    if len(points_a) == 0 or len(points_b) == 0:
        return np.empty((len(points_a), len(points_b)), dtype=float)
    da = points_a[:, None, :] - points_b[None, :, :]
    return np.sqrt(np.sum(da * da, axis=2))


def _nearest_neighbor_cycle_order(points: np.ndarray) -> np.ndarray:
    n = len(points)
    if n <= 2:
        return np.arange(n, dtype=int)
    D = _pairwise_dist(points, points)
    np.fill_diagonal(D, np.inf)
    start = int(np.argmin(points[:, 0]))
    order = [start]
    used = {start}
    cur = start
    for _ in range(n - 1):
        cand = np.argsort(D[cur])
        nxt = next((int(j) for j in cand if int(j) not in used), None)
        if nxt is None:
            break
        order.append(nxt)
        used.add(nxt)
        cur = nxt
    if len(order) < n:
        order.extend([i for i in range(n) if i not in used])
    return np.asarray(order, dtype=int)


def _curve_cumulative_arclength(curve: np.ndarray) -> np.ndarray:
    if len(curve) == 0:
        return np.empty(0, dtype=float)
    ds = np.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1))
    return np.concatenate([[0.0], np.cumsum(ds)])


def _cyclic_dist(a: int, b: int, n: int) -> int:
    if n <= 0:
        return 0
    d = abs(int(a) - int(b)) % n
    return int(min(d, n - d))


def _polygon_signed_area(curve: np.ndarray) -> float:
    if len(curve) < 3:
        return 0.0
    x = np.asarray(curve[:, 0], dtype=float)
    y = np.asarray(curve[:, 1], dtype=float)
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _segments_intersect(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray, atol: float = 1e-9) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return (
            min(p[0], q[0]) - atol <= r[0] <= max(p[0], q[0]) + atol
            and min(p[1], q[1]) - atol <= r[1] <= max(p[1], q[1]) + atol
        )

    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)
    if ((o1 > atol and o2 < -atol) or (o1 < -atol and o2 > atol)) and ((o3 > atol and o4 < -atol) or (o3 < -atol and o4 > atol)):
        return True
    if abs(o1) <= atol and on_segment(a1, a2, b1):
        return True
    if abs(o2) <= atol and on_segment(a1, a2, b2):
        return True
    if abs(o3) <= atol and on_segment(b1, b2, a1):
        return True
    if abs(o4) <= atol and on_segment(b1, b2, a2):
        return True
    return False


def _segment_intersection_count(curve: np.ndarray, atol: float = 1e-9) -> int:
    if len(curve) < 4:
        return 0
    pts = np.asarray(curve, dtype=float)
    n = len(pts)
    count = 0
    for i in range(n):
        a1 = pts[i]
        a2 = pts[(i + 1) % n]
        for j in range(i + 1, n):
            if (i + 1) % n == j or (j + 1) % n == i:
                continue
            b1 = pts[j]
            b2 = pts[(j + 1) % n]
            if _segments_intersect(a1, a2, b1, b2, atol=atol):
                count += 1
    return count


def _cleanup_monotone_samples(
    s_vals: np.ndarray,
    R_vals: np.ndarray,
    Z_vals: np.ndarray,
    *,
    min_ds: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(s_vals) == 0:
        keep = np.empty(0, dtype=bool)
        return s_vals, R_vals, Z_vals, keep
    keep = np.ones(len(s_vals), dtype=bool)
    last = 0
    for i in range(1, len(s_vals)):
        ds = float(s_vals[i] - s_vals[last])
        dR = float(R_vals[i] - R_vals[last])
        dZ = float(Z_vals[i] - Z_vals[last])
        if ds <= min_ds or np.hypot(dR, dZ) <= min_ds:
            keep[i] = False
            continue
        last = i
    return s_vals[keep], R_vals[keep], Z_vals[keep], keep


def _enforce_forward_winding(sequence: List["XOArcPoint"]) -> List["XOArcPoint"]:
    if len(sequence) < 3:
        return list(sequence)
    curve = np.array([[p.R, p.Z] for p in sequence], dtype=float)
    if _polygon_signed_area(curve) >= 0.0:
        return list(sequence)
    seq_rev = list(reversed(sequence))
    total_s = float(sequence[-1].s)
    rebuilt: List[XOArcPoint] = []
    for idx, pt in enumerate(seq_rev):
        s_new = 0.0 if idx == 0 else total_s - float(pt.s)
        rebuilt.append(XOArcPoint(kind=pt.kind, index=pt.index, R=pt.R, Z=pt.Z, s=s_new, slot=pt.slot, local_t=pt.local_t))
    return rebuilt


def _project_to_segment_param(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[float, float, np.ndarray]:
    ab = b - a
    lab2 = float(np.dot(ab, ab))
    if lab2 <= 0.0:
        q = a.copy()
        return 0.0, float(np.hypot(*(p - q))), q
    t = float(np.dot(p - a, ab) / lab2)
    t_clip = min(1.0, max(0.0, t))
    q = a + t_clip * ab
    return t_clip, float(np.hypot(*(p - q))), q


@dataclass
class XOArcPoint:
    kind: str
    index: int
    R: float
    Z: float
    s: float
    slot: Optional[int] = None
    local_t: Optional[float] = None


@dataclass
class XOSequence:
    """Ordered / connected section-local X/O sequence for healed boundary work."""

    axis: Tuple[float, float]
    O_points: ArrayLike
    X_points: ArrayLike
    O_order: ArrayLike
    X_slot: ArrayLike
    sequence: List[XOArcPoint]
    s_closed: ArrayLike
    R_closed: ArrayLike
    Z_closed: ArrayLike
    diagnostics: Dict[str, Any]

    def periodic_splines(self) -> Tuple[Optional[CubicSpline], Optional[CubicSpline]]:
        if len(self.s_closed) < 4:
            return None, None
        if not np.all(np.diff(self.s_closed) > 0):
            return None, None
        try:
            return (
                CubicSpline(self.s_closed, self.R_closed, bc_type="periodic"),
                CubicSpline(self.s_closed, self.Z_closed, bc_type="periodic"),
            )
        except Exception:
            return None, None


def build_xo_sequence(
    O_points: Sequence[Any],
    X_points: Sequence[Any],
    *,
    axis: Optional[Tuple[float, float]] = None,
    rho_min: float = 0.0,
    dedup_tol: float = 1e-5,
    min_o_points: int = 3,
) -> Optional[XOSequence]:
    """Build a robust alternating X/O sequence on one section.

    The backbone is defined by a cyclic ordering of O-points, refined by local
    nearest-neighbour continuity rather than simple polar-angle sorting.  Each
    X-point is then assigned to the O-slot whose chord it best matches.

    Compared with the initial greedy implementation, this version keeps a full
    candidate table and then repairs slot gaps with lightweight global logic:

    * favour the best one-to-one X↔slot matching first,
    * allow missing-slot repair when an unassigned X fits an empty neighbour
      slot substantially better than the original winner,
    * softly prefer angular continuity between X and slot on the O-ring,
    * keep the final polygon rejection (self-intersection / winding) intact.
    """
    O = _dedup_points(_as_point_rows(O_points), tol=dedup_tol)
    X = _dedup_points(_as_point_rows(X_points), tol=dedup_tol)
    if axis is None:
        if len(O):
            axis = (float(np.mean(O[:, 0])), float(np.mean(O[:, 1])))
        elif len(X):
            axis = (float(np.mean(X[:, 0])), float(np.mean(X[:, 1])))
        else:
            return None
    R_ax, Z_ax = float(axis[0]), float(axis[1])
    axis_vec = np.array([R_ax, Z_ax], dtype=float)

    if rho_min > 0.0:
        if len(O):
            O = O[np.hypot(O[:, 0] - R_ax, O[:, 1] - Z_ax) > rho_min]
        if len(X):
            X = X[np.hypot(X[:, 0] - R_ax, X[:, 1] - Z_ax) > rho_min]
    if len(O) < min_o_points or len(X) < 1:
        return None

    order_seed = _nearest_neighbor_cycle_order(O)
    O_ord = O[order_seed]
    if _polygon_signed_area(O_ord) < 0.0:
        O_ord = O_ord[::-1].copy()
        order_seed = order_seed[::-1].copy()

    ang = np.unwrap(np.arctan2(O_ord[:, 1] - Z_ax, O_ord[:, 0] - R_ax))
    if np.ptp(ang) > 0:
        shift = int(np.argmin(ang))
        O_ord = np.roll(O_ord, -shift, axis=0)
        order_seed = np.roll(order_seed, -shift)
        ang = np.roll(ang, -shift)
    ang = np.unwrap(ang)

    O_closed = np.vstack([O_ord, O_ord[0]])
    sO = _curve_cumulative_arclength(O_closed)
    n_O = len(O_ord)

    x_slot = np.full(len(X), -1, dtype=int)
    x_angle = np.unwrap(np.arctan2(X[:, 1] - Z_ax, X[:, 0] - R_ax)) if len(X) else np.empty(0, dtype=float)
    if len(x_angle) and len(ang):
        x_angle = x_angle + 2.0 * np.pi * np.round((np.mean(ang) - np.mean(x_angle)) / (2.0 * np.pi))

    slot_candidates: Dict[int, List[Dict[str, float]]] = {k: [] for k in range(n_O)}
    x_candidates: Dict[int, List[Dict[str, float]]] = {ix: [] for ix in range(len(X))}

    for ix, xp in enumerate(X):
        xp = np.asarray(xp, dtype=float)
        for k in range(n_O):
            a = O_ord[k]
            b = O_ord[(k + 1) % n_O]
            tloc, dist, q = _project_to_segment_param(xp, a, b)
            chord = float(np.hypot(*(b - a))) + 1e-12
            mid = 0.5 * (a + b)
            radial_pen = abs(np.hypot(*(xp - axis_vec)) - np.hypot(*(mid - axis_vec)))
            endpoint_pen = min(tloc, 1.0 - tloc)
            theta_mid = 0.5 * (ang[k] + ang[(k + 1) % n_O])
            angle_pen = abs(float(x_angle[ix] - theta_mid)) / np.pi if len(x_angle) else 0.0
            base_score = dist / chord + 0.35 * radial_pen / chord - 0.15 * endpoint_pen
            score = base_score + 0.10 * angle_pen
            cand = {
                "slot": k,
                "score": float(score),
                "base_score": float(base_score),
                "angle_pen": float(angle_pen),
                "t": float(tloc),
                "s": float(sO[k] + tloc * chord),
                "R": float(xp[0]),
                "Z": float(xp[1]),
                "ix": ix,
                "dist": float(dist),
                "chord": float(chord),
            }
            slot_candidates[k].append(cand)
            x_candidates[ix].append(cand)

    for k in range(n_O):
        slot_candidates[k].sort(key=lambda rec: (rec["score"], rec["angle_pen"], rec["dist"]))
    for ix in range(len(X)):
        x_candidates[ix].sort(key=lambda rec: (rec["score"], rec["angle_pen"], rec["dist"]))

    slot_best: Dict[int, Dict[str, float]] = {}
    used_x: set[int] = set()

    def _assign(slot: int, rec: Dict[str, float]) -> None:
        slot_best[int(slot)] = rec
        x_slot[int(rec["ix"])] = int(slot)
        used_x.add(int(rec["ix"]))

    for k in range(n_O):
        for cand in slot_candidates[k]:
            ix = int(cand["ix"])
            if ix in used_x:
                continue
            _assign(k, cand)
            break

    empty_slots = [k for k in range(n_O) if k not in slot_best]
    if empty_slots:
        unassigned = [ix for ix in range(len(X)) if ix not in used_x]
        for k in empty_slots:
            cands = [rec for rec in slot_candidates[k] if int(rec["ix"]) in unassigned]
            if not cands:
                continue
            best = cands[0]
            if best["score"] <= 1.35:
                _assign(k, best)
                unassigned.remove(int(best["ix"]))

    empty_slots = [k for k in range(n_O) if k not in slot_best]
    if empty_slots:
        unassigned = [ix for ix in range(len(X)) if ix not in used_x]
        for k in empty_slots:
            for rec in slot_candidates[k]:
                ix = int(rec["ix"])
                if ix not in unassigned:
                    continue
                best_x = x_candidates[ix][0] if x_candidates[ix] else None
                if best_x is None:
                    continue
                best_slot = int(best_x["slot"])
                donor = slot_best.get(best_slot)
                score_gain = float(rec["score"] - best_x["score"])
                near_pref = _cyclic_dist(k, best_slot, n_O)
                if donor is None or score_gain <= 0.20 + 0.08 * near_pref:
                    _assign(k, rec)
                    unassigned.remove(ix)
                    break

    seq: List[XOArcPoint] = []
    for k in range(n_O):
        seq.append(XOArcPoint(kind="O", index=int(order_seed[k]), R=float(O_ord[k, 0]), Z=float(O_ord[k, 1]), s=float(sO[k]), slot=k, local_t=0.0))
        rec = slot_best.get(k)
        if rec is not None:
            seq.append(XOArcPoint(kind="X", index=int(rec["ix"]), R=float(rec["R"]), Z=float(rec["Z"]), s=float(rec["s"]), slot=k, local_t=float(rec["t"])))

    if len(seq) < 4:
        return None

    seq = sorted(seq, key=lambda p: p.s)
    seq = _enforce_forward_winding(seq)
    s_vals = np.array([p.s for p in seq], dtype=float)
    R_vals = np.array([p.R for p in seq], dtype=float)
    Z_vals = np.array([p.Z for p in seq], dtype=float)
    s_vals, R_vals, Z_vals, keep = _cleanup_monotone_samples(s_vals, R_vals, Z_vals, min_ds=1e-10)
    seq = [p for p, kk in zip(seq, keep) if kk]
    if len(seq) < 4:
        return None

    total_len = float(sO[-1])
    curve = np.column_stack([R_vals, Z_vals])
    turn = np.unwrap(np.arctan2(Z_vals - Z_ax, R_vals - R_ax))
    winding_monotone = bool(np.all(np.diff(turn) > -1e-8))
    self_intersections = int(_segment_intersection_count(curve))
    if self_intersections > 0:
        return None

    s_closed = np.append(s_vals, s_vals[0] + total_len)
    R_closed = np.append(R_vals, R_vals[0])
    Z_closed = np.append(Z_vals, Z_vals[0])

    diagnostics = {
        "n_O": int(len(O_ord)),
        "n_X": int(len(X)),
        "n_X_assigned": int(np.sum(x_slot >= 0)),
        "n_slots_filled": int(len(slot_best)),
        "coverage": float(len(slot_best) / max(len(O_ord), 1)),
        "total_length": total_len,
        "slot_fill_fraction": float(len(slot_best) / max(n_O, 1)),
        "sequence_cleanup_removed": int(np.sum(~keep)),
        "self_intersections": self_intersections,
        "winding_monotone": winding_monotone,
        "signed_area": float(_polygon_signed_area(curve)),
        "slot_gap_count": int(np.sum([k not in slot_best for k in range(n_O)])),
        "n_unassigned_X": int(np.sum(x_slot < 0)),
    }
    return XOSequence(
        axis=(R_ax, Z_ax),
        O_points=O,
        X_points=X,
        O_order=order_seed,
        X_slot=x_slot,
        sequence=seq,
        s_closed=s_closed,
        R_closed=R_closed,
        Z_closed=Z_closed,
        diagnostics=diagnostics,
    )


def xo_sequence_boundary_arcs(
    xo: Optional[XOSequence],
    *,
    include_o_segments: bool = True,
    include_x_segments: bool = True,
    include_cross_segments: bool = True,
    outward_quantile: float = 0.5,
    min_length_fraction: float = 0.0,
    min_points: int = 2,
) -> List[np.ndarray]:
    """Extract physically filtered slot/segment-aware local boundary arcs.

    The raw repaired X/O sequence contains many short local connections. For
    boundary correction we usually only want segments that are plausibly part of
    the outer healed envelope. This helper therefore supports lightweight
    physics-aware filtering by outwardness and relative segment length.
    """
    if xo is None or len(xo.sequence) < max(2, min_points):
        return []
    seq = list(xo.sequence)
    n = len(seq)
    axis = np.asarray(xo.axis, dtype=float)
    candidates: List[Dict[str, Any]] = []
    lengths = []
    outward_vals = []
    for i, p0 in enumerate(seq):
        p1 = seq[(i + 1) % n]
        pts = np.array([[p0.R, p0.Z], [p1.R, p1.Z]], dtype=float)
        if pts.shape[0] < min_points:
            continue
        kinds = {p0.kind, p1.kind}
        keep_kind = (
            (kinds == {"O"} and include_o_segments)
            or (kinds == {"X"} and include_x_segments)
            or (kinds == {"O", "X"} and include_cross_segments)
        )
        if not keep_kind:
            continue
        rho = np.hypot(pts[:, 0] - axis[0], pts[:, 1] - axis[1])
        outward = float(np.mean(rho))
        seg_len = float(np.hypot(*(pts[1] - pts[0])))
        candidates.append({
            "pts": pts,
            "outward": outward,
            "length": seg_len,
            "kinds": kinds,
        })
        lengths.append(seg_len)
        outward_vals.append(outward)
    if not candidates:
        return []
    outward_cut = float(np.quantile(outward_vals, np.clip(outward_quantile, 0.0, 1.0)))
    max_len = max(lengths) if lengths else 0.0
    arcs: List[np.ndarray] = []
    for rec in candidates:
        if rec["outward"] + 1e-12 < outward_cut:
            continue
        if max_len > 0.0 and rec["length"] + 1e-12 < float(min_length_fraction) * max_len:
            continue
        arcs.append(np.asarray(rec["pts"], dtype=float))
    return arcs


def build_cxo_spline(
    O_points: Sequence[Any],
    X_points: Sequence[Any],
    *,
    axis: Optional[Tuple[float, float]] = None,
    rho_min: float = 0.30,
    dedup_tol: float = 1e-5,
    min_o_points: int = 3,
    validate_winding: bool = True,
    sample_count: int = 360,
) -> Tuple[Optional[CubicSpline], Optional[CubicSpline], Optional[XOSequence]]:
    """Build a robust periodic ``C_XO`` spline from section-local O/X points."""
    xo = build_xo_sequence(
        O_points,
        X_points,
        axis=axis,
        rho_min=rho_min,
        dedup_tol=dedup_tol,
        min_o_points=min_o_points,
    )
    if xo is None:
        return None, None, None

    sR, sZ = xo.periodic_splines()
    if sR is None or sZ is None:
        return None, None, xo

    if validate_winding:
        ss = np.linspace(xo.s_closed[0], xo.s_closed[-1], sample_count, endpoint=False)
        ang = np.unwrap(np.arctan2(sZ(ss) - xo.axis[1], sR(ss) - xo.axis[0]))
        dang = np.diff(ang)
        if np.any(dang <= -1e-6):
            return None, None, xo
        rr = np.hypot(sR(ss) - xo.axis[0], sZ(ss) - xo.axis[1])
        if np.nanmin(rr) <= 0.0:
            return None, None, xo

    return sR, sZ, xo


@dataclass
class SectionFit:
    phi: float
    R_ax: float
    Z_ax: float
    spl_R_in: Optional[Sequence]
    spl_Z_in: Optional[Sequence]
    spl_R_ext: Optional[Sequence]
    spl_Z_ext: Optional[Sequence]
    r_max: float
    r_aug_ext: ArrayLike
    spl_R_CXO: Optional[Any]
    spl_Z_CXO: Optional[Any]
    cxo_param: Optional[ArrayLike] = None
    scaffold_valid: bool = False
    trace_valid_fraction: float = 0.0
    boundary_source: str = "none"
    boundary_valid_fraction: float = 0.0


@dataclass
class BoundarySection:
    """One toroidal section of a transported healed boundary family.

    Parameters
    ----------
    phi : float
        Toroidal angle [rad].
    R, Z : ndarray, shape (n_theta,)
        Boundary coordinates sampled on a shared boundary parameter grid.
    valid : ndarray, shape (n_theta,)
        Validity mask after field-line transport / repair.
    param : ndarray, shape (n_theta,)
        Shared boundary parameter in [0, 1).
    source : str
        Provenance label, e.g. ``local-cxo`` or ``transported-ref-cxo``.
    """
    phi: float
    R: ArrayLike
    Z: ArrayLike
    valid: ArrayLike
    param: ArrayLike
    source: str = "unknown"
    diagnostics: Optional[Dict[str, Any]] = None

    def valid_fraction(self) -> float:
        return float(np.mean(self.valid)) if np.size(self.valid) else 0.0


@dataclass
class BoundaryConstraintSet:
    """Section-local geometric constraints for boundary correction.

    This is intentionally generic: callers may provide invariant points from
    island chains, periodic orbits, manifolds, or any other section-local
    boundary markers. ``attract_points`` are sparse markers; ``attract_arcs``
    are ordered polyline-like samples that indicate boundary segments that
    should be followed locally.
    """

    attract_points: ArrayLike
    attract_arcs: Optional[Sequence[ArrayLike]] = None
    repel_points: Optional[ArrayLike] = None
    local_weight: float = 0.7
    snap_length_scale: Optional[float] = None
    arc_weight: Optional[float] = None


def _nearest_curve_samples(
    R_curve: Sequence[float],
    Z_curve: Sequence[float],
    points: Sequence[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Rc = np.asarray(R_curve, dtype=float)
    Zc = np.asarray(Z_curve, dtype=float)
    P = np.asarray(points, dtype=float)
    if P.size == 0 or Rc.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float), np.empty((0, 2), dtype=float)
    if P.ndim == 1:
        P = P[None, :]
    dR = Rc[None, :] - P[:, 0][:, None]
    dZ = Zc[None, :] - P[:, 1][:, None]
    dist = np.sqrt(dR * dR + dZ * dZ)
    idx = np.argmin(dist, axis=1)
    dmin = dist[np.arange(len(P)), idx]
    curve_pts = np.column_stack([Rc[idx], Zc[idx]])
    return idx.astype(int), dmin.astype(float), curve_pts


def correct_boundary_with_constraints(
    R_curve: Sequence[float],
    Z_curve: Sequence[float],
    *,
    constraints: Optional[BoundaryConstraintSet] = None,
    valid: Optional[Sequence[bool]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Apply a generic local correction from section-local boundary markers.

    Two complementary correction modes are supported:
    1. sparse point attraction for isolated invariant markers,
    2. arc-aware attraction for ordered local boundary segments.
    """
    R = np.asarray(R_curve, dtype=float).copy()
    Z = np.asarray(Z_curve, dtype=float).copy()
    ok = np.ones_like(R, dtype=bool) if valid is None else np.asarray(valid, dtype=bool).copy()
    diagnostics = {
        "n_attract": 0,
        "n_snap_used": 0,
        "mean_snap_distance": 0.0,
        "max_snap_distance": 0.0,
        "n_arcs": 0,
        "n_arc_samples_used": 0,
    }
    if constraints is None:
        return R, Z, ok, diagnostics

    moved = []
    length_scale = constraints.snap_length_scale
    weight = float(np.clip(constraints.local_weight, 0.0, 1.0))
    arc_weight = float(np.clip(constraints.arc_weight if constraints.arc_weight is not None else max(weight, 0.85), 0.0, 1.0))

    attract = np.asarray(constraints.attract_points, dtype=float)
    if attract.size != 0:
        if attract.ndim == 1:
            attract = attract[None, :]
        diagnostics["n_attract"] = int(len(attract))
        idx, dist, _ = _nearest_curve_samples(R, Z, attract)
        if length_scale is None:
            finite_dist = dist[np.isfinite(dist)]
            length_scale = float(np.median(finite_dist)) if finite_dist.size else 0.0
        for ip, j in enumerate(idx):
            if not np.isfinite(dist[ip]):
                continue
            if length_scale > 0.0 and dist[ip] > 3.0 * length_scale:
                continue
            R[j] = (1.0 - weight) * R[j] + weight * attract[ip, 0]
            Z[j] = (1.0 - weight) * Z[j] + weight * attract[ip, 1]
            ok[j] = True
            diagnostics["n_snap_used"] += 1
            moved.append(float(dist[ip]))

    arcs = constraints.attract_arcs or []
    diagnostics["n_arcs"] = int(len(arcs))
    for arc in arcs:
        arc_arr = np.asarray(arc, dtype=float)
        if arc_arr.ndim != 2 or arc_arr.shape[0] < 2 or arc_arr.shape[1] < 2:
            continue
        idx_arc, dist_arc, _ = _nearest_curve_samples(R, Z, arc_arr[:, :2])
        if length_scale is None:
            finite_dist = dist_arc[np.isfinite(dist_arc)]
            length_scale = float(np.median(finite_dist)) if finite_dist.size else 0.0
        used_idx = []
        used_pts = []
        for ip, j in enumerate(idx_arc):
            if not np.isfinite(dist_arc[ip]):
                continue
            if length_scale > 0.0 and dist_arc[ip] > 4.0 * length_scale:
                continue
            used_idx.append(int(j))
            used_pts.append(arc_arr[ip, :2])
        if len(used_idx) < 2:
            continue
        order = np.argsort(used_idx)
        used_idx = [used_idx[k] for k in order]
        used_pts = [used_pts[k] for k in order]
        for j, pt in zip(used_idx, used_pts):
            R[j] = (1.0 - arc_weight) * R[j] + arc_weight * float(pt[0])
            Z[j] = (1.0 - arc_weight) * Z[j] + arc_weight * float(pt[1])
            ok[j] = True
        diagnostics["n_arc_samples_used"] += int(len(used_idx))
        seg_move = np.sqrt((np.array([pt[0] for pt in used_pts]) - R[np.array(used_idx)])**2 + (np.array([pt[1] for pt in used_pts]) - Z[np.array(used_idx)])**2)
        moved.extend([float(v) for v in seg_move])

    if moved:
        diagnostics["mean_snap_distance"] = float(np.mean(moved))
        diagnostics["max_snap_distance"] = float(np.max(moved))
    return R, Z, ok, diagnostics


class BoundaryFamily3D:
    """Toroidally transported healed boundary family.

    This promotes the outer healed boundary from a section-local plotting helper
    to a first-class 3D object.  A boundary family is built from one reference
    closed curve plus optional local boundary observations at other toroidal
    sections.  Transport provides cross-section continuity; local observations
    refine the transported shape when available.
    """

    def __init__(
        self,
        *,
        phi_ref: float,
        phi_samples: Sequence[float],
        param_levels: Sequence[float],
        sections: Sequence[BoundarySection],
        ref_R: Sequence[float],
        ref_Z: Sequence[float],
        ref_source: str = "reference",
    ):
        self.phi_ref = _wrap_angle(phi_ref)
        self.phi_samples = np.asarray([_wrap_angle(phi) for phi in phi_samples], dtype=float)
        self.param_levels = np.asarray(param_levels, dtype=float)
        self.sections = list(sections)
        self.ref_R = np.asarray(ref_R, dtype=float)
        self.ref_Z = np.asarray(ref_Z, dtype=float)
        self.ref_source = str(ref_source)
        if len(self.sections) != len(self.phi_samples):
            raise ValueError("len(sections) must equal len(phi_samples)")
        n_theta = len(self.param_levels)
        if self.ref_R.shape != (n_theta,) or self.ref_Z.shape != (n_theta,):
            raise ValueError("reference boundary must have shape (n_theta,)")
        for sec in self.sections:
            if np.asarray(sec.R).shape != (n_theta,) or np.asarray(sec.Z).shape != (n_theta,):
                raise ValueError("all boundary sections must share the same parameter grid")

    @classmethod
    def from_reference_curve(
        cls,
        *,
        phi_ref: float,
        phi_samples: Sequence[float],
        ref_R: Sequence[float],
        ref_Z: Sequence[float],
        trace_func: TraceFunction,
        local_sections: Optional[Sequence[Optional[Tuple[Sequence[float], Sequence[float]]]]] = None,
        local_constraints: Optional[Sequence[Optional[BoundaryConstraintSet]]] = None,
        param_levels: Optional[Sequence[float]] = None,
        dphi_hint: float = 0.04,
        phi_hit_tol_factor: float = 5.0,
        blend_local_weight: float = 0.65,
        min_transport_fraction: float = 0.35,
    ) -> "BoundaryFamily3D":
        ref_R = np.asarray(ref_R, dtype=float)
        ref_Z = np.asarray(ref_Z, dtype=float)
        n_theta = len(ref_R)
        if n_theta < 8 or ref_Z.shape != ref_R.shape:
            raise ValueError("reference boundary must be a closed curve with >=8 samples")
        if param_levels is None:
            param_levels = np.linspace(0.0, 1.0, n_theta, endpoint=False)
        param_levels = np.asarray(param_levels, dtype=float)
        phi_samples = np.asarray([_wrap_angle(phi) for phi in phi_samples], dtype=float)
        ref_idx = int(np.argmin(np.abs(np.angle(np.exp(1j * (phi_samples - _wrap_angle(phi_ref)))))))
        sections: List[BoundarySection] = []
        for ip, phi in enumerate(phi_samples):
            if ip == ref_idx:
                valid = np.ones(n_theta, dtype=bool)
                source = "reference"
                R_use = ref_R.copy()
                Z_use = ref_Z.copy()
            else:
                R_t, Z_t, valid = trace_section_curve_to_phi(
                    ref_R, ref_Z,
                    phi_src=float(phi_ref), phi_tgt=float(phi), trace_func=trace_func,
                    dphi_hint=dphi_hint, phi_hit_tol_factor=phi_hit_tol_factor,
                )
                source = "transported-ref-cxo"
                R_use = np.asarray(R_t, dtype=float)
                Z_use = np.asarray(Z_t, dtype=float)
                valid = np.asarray(valid, dtype=bool)
            diagnostics = None
            if local_sections is not None and ip < len(local_sections) and local_sections[ip] is not None:
                R_loc = np.asarray(local_sections[ip][0], dtype=float)
                Z_loc = np.asarray(local_sections[ip][1], dtype=float)
                if R_loc.shape == (n_theta,) and Z_loc.shape == (n_theta,):
                    good_loc = np.isfinite(R_loc) & np.isfinite(Z_loc)
                    if np.mean(good_loc) >= min_transport_fraction:
                        if np.mean(valid) >= min_transport_fraction:
                            both = valid & good_loc
                            if np.any(both):
                                w = float(np.clip(blend_local_weight, 0.0, 1.0))
                                R_use[both] = (1.0 - w) * R_use[both] + w * R_loc[both]
                                Z_use[both] = (1.0 - w) * Z_use[both] + w * Z_loc[both]
                                valid = valid | good_loc
                                source = "blended-local+transport"
                        else:
                            R_use = R_loc.copy()
                            Z_use = Z_loc.copy()
                            valid = good_loc.copy()
                            source = "local-cxo"
            if local_constraints is not None and ip < len(local_constraints) and local_constraints[ip] is not None:
                R_use, Z_use, valid, diagnostics = correct_boundary_with_constraints(
                    R_use,
                    Z_use,
                    constraints=local_constraints[ip],
                    valid=valid,
                )
                if diagnostics.get("n_snap_used", 0) > 0:
                    source = f"{source}+constraint-correction"
            sections.append(BoundarySection(
                phi=float(phi),
                R=R_use,
                Z=Z_use,
                valid=valid,
                param=param_levels.copy(),
                source=source,
                diagnostics=diagnostics,
            ))
        return cls(
            phi_ref=float(phi_ref),
            phi_samples=phi_samples,
            param_levels=param_levels,
            sections=sections,
            ref_R=ref_R,
            ref_Z=ref_Z,
            ref_source="reference",
        )

    def nearest_section_index(self, phi: float) -> int:
        phi_w = _wrap_angle(phi)
        dphi = np.abs(np.angle(np.exp(1j * (self.phi_samples - phi_w))))
        return int(np.argmin(dphi))

    def section_at(self, phi: float) -> BoundarySection:
        return self.sections[self.nearest_section_index(phi)]

    def section_splines(self, phi: float) -> Tuple[Optional[CubicSpline], Optional[CubicSpline], Optional[np.ndarray], float, str]:
        sec = self.section_at(phi)
        ok = np.asarray(sec.valid, dtype=bool) & np.isfinite(sec.R) & np.isfinite(sec.Z)
        if np.sum(ok) < max(12, len(sec.param) // 3):
            return None, None, None, float(np.mean(ok)) if ok.size else 0.0, sec.source
        t = np.asarray(sec.param[ok], dtype=float)
        R = np.asarray(sec.R[ok], dtype=float)
        Z = np.asarray(sec.Z[ok], dtype=float)
        if len(t) < 4:
            return None, None, None, float(np.mean(ok)), sec.source
        order = np.argsort(t)
        t = t[order]
        R = R[order]
        Z = Z[order]
        axis = np.array([np.mean(R), np.mean(Z)], dtype=float)
        ang = np.unwrap(np.arctan2(Z - axis[1], R - axis[0]))
        winding = abs(float((ang[-1] - ang[0]) / (2.0 * np.pi))) if len(ang) > 1 else 0.0
        if winding > 1.5:
            return None, None, None, float(np.mean(ok)), sec.source + "+rejected-multiwrap"
        t_closed = np.append(t, 1.0)
        R_closed = np.append(R, R[0])
        Z_closed = np.append(Z, Z[0])
        try:
            return (
                CubicSpline(t_closed, R_closed, bc_type="periodic"),
                CubicSpline(t_closed, Z_closed, bc_type="periodic"),
                t_closed,
                float(np.mean(ok)),
                sec.source,
            )
        except Exception:
            return None, None, None, float(np.mean(ok)), sec.source


@dataclass
class SectionScaffoldBundle:
    scaffold_3d: FieldLineScaffold3D
    fits: Sequence[SectionFit]
    reference_index: int
    boundary_family: Optional[BoundaryFamily3D] = None


def trace_grid_to_phi(
    R_grid: ArrayLike,
    Z_grid: ArrayLike,
    *,
    phi_src: float,
    phi_tgt: float,
    trace_func: TraceFunction,
    dphi_hint: float = 0.04,
    phi_hit_tol_factor: float = 5.0,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Transport a 2D ``(r, θ)`` point grid from one toroidal section to another."""
    R_grid = np.asarray(R_grid, dtype=float)
    Z_grid = np.asarray(Z_grid, dtype=float)
    if R_grid.shape != Z_grid.shape:
        raise ValueError("R_grid and Z_grid must have the same shape")

    n_r, n_theta = R_grid.shape
    R_tgt = np.full((n_r, n_theta), np.nan, dtype=float)
    Z_tgt = np.full((n_r, n_theta), np.nan, dtype=float)
    valid = np.zeros((n_r, n_theta), dtype=bool)

    span = _forward_span(phi_src, phi_tgt)
    if span < 1e-12:
        return R_grid.copy(), Z_grid.copy(), np.ones_like(R_grid, dtype=bool)

    dphi_out = min(span / 30.0, dphi_hint) if span > 0 else dphi_hint
    phi_tgt_w = _wrap_angle(phi_tgt)
    phi_tol = phi_hit_tol_factor * dphi_out

    for i in range(n_r):
        for j in range(n_theta):
            R0 = float(R_grid[i, j])
            Z0 = float(Z_grid[i, j])
            if not (np.isfinite(R0) and np.isfinite(Z0)):
                continue
            R_arr, Z_arr, phi_arr = trace_func(R0, Z0, float(phi_src), float(span + 2.0 * dphi_out), float(dphi_out))
            if len(phi_arr) == 0:
                continue
            phi_mod = np.asarray(phi_arr, dtype=float) % (2.0 * np.pi)
            delta = np.abs(np.angle(np.exp(1j * (phi_mod - phi_tgt_w))))
            idx = int(np.argmin(delta))
            if float(delta[idx]) < phi_tol:
                R_tgt[i, j] = float(R_arr[idx])
                Z_tgt[i, j] = float(Z_arr[idx])
                valid[i, j] = True
    return R_tgt, Z_tgt, valid


@dataclass
class TransportedSection:
    phi: float
    R: ArrayLike
    Z: ArrayLike
    valid: ArrayLike


@dataclass
class SectionCorrespondence:
    phi_ref: float
    phi: float
    r_levels: ArrayLike
    theta_levels: ArrayLike
    R: ArrayLike
    Z: ArrayLike
    valid: ArrayLike

    def coverage_fraction(self) -> float:
        return float(np.mean(self.valid)) if self.valid.size else 0.0

    def valid_counts_by_r(self) -> ArrayLike:
        return np.sum(self.valid, axis=1)


class FieldLineScaffold3D:
    def __init__(
        self,
        phi_ref: float,
        r_levels: ArrayLike,
        theta_levels: ArrayLike,
        phi_samples: ArrayLike,
        sections: Sequence[TransportedSection],
        R_ref: ArrayLike,
        Z_ref: ArrayLike,
    ):
        self.phi_ref = _wrap_angle(phi_ref)
        self.r_levels = np.asarray(r_levels, dtype=float)
        self.theta_levels = np.asarray(theta_levels, dtype=float)
        self.phi_samples = np.asarray([_wrap_angle(phi) for phi in phi_samples], dtype=float)
        self.sections = list(sections)
        self.R_ref = np.asarray(R_ref, dtype=float)
        self.Z_ref = np.asarray(Z_ref, dtype=float)
        if len(self.sections) != len(self.phi_samples):
            raise ValueError("len(sections) must equal len(phi_samples)")
        if self.R_ref.shape != (len(self.r_levels), len(self.theta_levels)):
            raise ValueError("R_ref shape must be (n_r, n_theta)")
        if self.Z_ref.shape != self.R_ref.shape:
            raise ValueError("Z_ref shape mismatch")

    @classmethod
    def from_reference_map(
        cls,
        reference_map,
        phi_ref: float,
        r_levels: Sequence[float],
        theta_levels: Sequence[float],
        phi_samples: Sequence[float],
        trace_func: TraceFunction,
        *,
        dphi_hint: float = 0.04,
        phi_hit_tol_factor: float = 5.0,
    ) -> "FieldLineScaffold3D":
        phi_ref = _wrap_angle(phi_ref)
        r_levels = np.asarray(r_levels, dtype=float)
        theta_levels = np.asarray(theta_levels, dtype=float)
        phi_samples = np.asarray([_wrap_angle(phi) for phi in phi_samples], dtype=float)
        n_r = len(r_levels)
        n_theta = len(theta_levels)
        R_ref = np.empty((n_r, n_theta), dtype=float)
        Z_ref = np.empty((n_r, n_theta), dtype=float)
        for i, r in enumerate(r_levels):
            for j, theta in enumerate(theta_levels):
                R_ref[i, j], Z_ref[i, j] = reference_map.eval_RZ(float(r), float(theta))
        sections = []
        for phi_tgt in phi_samples:
            if abs(_forward_span(phi_ref, phi_tgt)) < 1e-12:
                valid = np.ones_like(R_ref, dtype=bool)
                sections.append(TransportedSection(phi=float(phi_tgt), R=R_ref.copy(), Z=Z_ref.copy(), valid=valid))
                continue
            R_tgt, Z_tgt, valid = trace_grid_to_phi(
                R_ref, Z_ref, phi_src=phi_ref, phi_tgt=float(phi_tgt), trace_func=trace_func,
                dphi_hint=dphi_hint, phi_hit_tol_factor=phi_hit_tol_factor,
            )
            sections.append(TransportedSection(phi=float(phi_tgt), R=R_tgt, Z=Z_tgt, valid=valid))
        return cls(phi_ref=phi_ref, r_levels=r_levels, theta_levels=theta_levels, phi_samples=phi_samples, sections=sections, R_ref=R_ref, Z_ref=Z_ref)

    def nearest_section_index(self, phi: float) -> int:
        phi_w = _wrap_angle(phi)
        dphi = np.abs(np.angle(np.exp(1j * (self.phi_samples - phi_w))))
        return int(np.argmin(dphi))

    def section_at(self, phi: float) -> TransportedSection:
        return self.sections[self.nearest_section_index(phi)]

    def correspondence_at(self, phi: float) -> SectionCorrespondence:
        sec = self.section_at(phi)
        return SectionCorrespondence(phi_ref=self.phi_ref, phi=sec.phi, r_levels=self.r_levels, theta_levels=self.theta_levels, R=sec.R, Z=sec.Z, valid=sec.valid)

    def sample_surface(self, r_index: int, phi: float) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        sec = self.section_at(phi)
        return sec.R[r_index], sec.Z[r_index], sec.valid[r_index]

    def sampled_arrays(self) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        R = np.stack([sec.R for sec in self.sections], axis=0)
        Z = np.stack([sec.Z for sec in self.sections], axis=0)
        valid = np.stack([sec.valid for sec in self.sections], axis=0)
        return R, Z, valid


def trace_section_curve_to_phi(
    R_curve: Sequence[float],
    Z_curve: Sequence[float],
    *,
    phi_src: float,
    phi_tgt: float,
    trace_func: TraceFunction,
    dphi_hint: float = 0.04,
    phi_hit_tol_factor: float = 5.0,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    R_curve = np.asarray(R_curve, dtype=float)[None, :]
    Z_curve = np.asarray(Z_curve, dtype=float)[None, :]
    R_t, Z_t, valid = trace_grid_to_phi(
        R_curve, Z_curve, phi_src=phi_src, phi_tgt=phi_tgt, trace_func=trace_func,
        dphi_hint=dphi_hint, phi_hit_tol_factor=phi_hit_tol_factor,
    )
    return R_t[0], Z_t[0], valid[0]


def trace_surface_family_to_sections(
    reference_map,
    phi_ref: float,
    r_levels: Sequence[float],
    theta_levels: Sequence[float],
    phi_samples: Sequence[float],
    trace_func: TraceFunction,
    *,
    dphi_hint: float = 0.04,
    phi_hit_tol_factor: float = 5.0,
) -> FieldLineScaffold3D:
    return FieldLineScaffold3D.from_reference_map(
        reference_map=reference_map,
        phi_ref=phi_ref,
        r_levels=r_levels,
        theta_levels=theta_levels,
        phi_samples=phi_samples,
        trace_func=trace_func,
        dphi_hint=dphi_hint,
        phi_hit_tol_factor=phi_hit_tol_factor,
    )


def fit_ring_fourier(theta: ArrayLike, R: ArrayLike, Z: ArrayLike, n_coeff: int) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    A = np.empty((len(theta), n_coeff), dtype=float)
    A[:, 0] = 1.0
    n_map = (n_coeff - 1) // 2
    for k in range(1, n_map + 1):
        A[:, 2 * k - 1] = np.cos(k * theta)
        A[:, 2 * k] = np.sin(k * theta)
    cR, *_ = np.linalg.lstsq(A, R, rcond=None)
    cZ, *_ = np.linalg.lstsq(A, Z, rcond=None)
    return cR, cZ


def build_pchip_family(r_pts: ArrayLike, cR_pts: ArrayLike, cZ_pts: ArrayLike):
    from scipy.interpolate import PchipInterpolator
    r_pts = np.asarray(r_pts, dtype=float)
    cR_pts = np.asarray(cR_pts, dtype=float)
    cZ_pts = np.asarray(cZ_pts, dtype=float)
    _, u = np.unique(np.round(r_pts, 8), return_index=True)
    r = r_pts[u]
    cR = cR_pts[u]
    cZ = cZ_pts[u]
    if len(r) < 2:
        return None, None, r
    return ([PchipInterpolator(r, cR[:, k]) for k in range(cR.shape[1])], [PchipInterpolator(r, cZ[:, k]) for k in range(cZ.shape[1])], r)


def eval_fourier_family(coeff_R: Sequence, coeff_Z: Sequence, r: float, theta: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    cR = np.array([spl(r) for spl in coeff_R], dtype=float)
    cZ = np.array([spl(r) for spl in coeff_Z], dtype=float)
    n_coeff = len(cR)
    A = np.empty((len(theta), n_coeff), dtype=float)
    A[:, 0] = 1.0
    n_map = (n_coeff - 1) // 2
    for k in range(1, n_map + 1):
        A[:, 2 * k - 1] = np.cos(k * theta)
        A[:, 2 * k] = np.sin(k * theta)
    return A @ cR, A @ cZ


class _ReferenceMapAdapter:
    def __init__(self, spl_R, spl_Z):
        self.spl_R = spl_R
        self.spl_Z = spl_Z
    def eval_RZ(self, r: float, theta: float) -> Tuple[float, float]:
        R, Z = eval_fourier_family(self.spl_R, self.spl_Z, r, np.array([theta], dtype=float))
        return float(R[0]), float(Z[0])


def build_section_scaffold_bundle(
    *,
    phi_samples: Sequence[float],
    phi_ref: float,
    reference_spl_R: Sequence,
    reference_spl_Z: Sequence,
    r_levels: Sequence[float],
    theta_levels: Sequence[float],
    trace_func: TraceFunction,
    section_axes: Sequence[Tuple[float, float]],
    cxo_local: Optional[Sequence[Optional[Tuple[Any, Any]]]] = None,
    local_boundary_constraints: Optional[Sequence[Optional[BoundaryConstraintSet]]] = None,
    fit_inner_limit: Optional[float] = None,
    n_coeff: int,
    dphi_hint: float = 0.04,
    phi_hit_tol_factor: float = 5.0,
    cxo_trace_theta: Optional[Sequence[float]] = None,
    boundary_family: Optional[BoundaryFamily3D] = None,
) -> SectionScaffoldBundle:
    phi_samples = np.asarray(phi_samples, dtype=float)
    ref_idx = int(np.argmin(np.abs(np.angle(np.exp(1j * (phi_samples - _wrap_angle(phi_ref)))))))
    scaffold = trace_surface_family_to_sections(
        reference_map=_ReferenceMapAdapter(reference_spl_R, reference_spl_Z),
        phi_ref=phi_ref,
        r_levels=r_levels,
        theta_levels=theta_levels,
        phi_samples=phi_samples,
        trace_func=trace_func,
        dphi_hint=dphi_hint,
        phi_hit_tol_factor=phi_hit_tol_factor,
    )
    fits: List[SectionFit] = []
    cxo_trace_theta = np.asarray(cxo_trace_theta if cxo_trace_theta is not None else theta_levels, dtype=float)

    if boundary_family is None and cxo_local is not None and ref_idx < len(cxo_local) and cxo_local[ref_idx] is not None:
        ref_cxo = cxo_local[ref_idx]
        ref_R = np.asarray(ref_cxo[0](cxo_trace_theta), dtype=float)
        ref_Z = np.asarray(ref_cxo[1](cxo_trace_theta), dtype=float)
        local_boundary_samples = []
        for item in cxo_local:
            if item is None:
                local_boundary_samples.append(None)
            else:
                local_boundary_samples.append((
                    np.asarray(item[0](cxo_trace_theta), dtype=float),
                    np.asarray(item[1](cxo_trace_theta), dtype=float),
                ))
        boundary_family = BoundaryFamily3D.from_reference_curve(
            phi_ref=float(phi_ref),
            phi_samples=phi_samples,
            ref_R=ref_R,
            ref_Z=ref_Z,
            trace_func=trace_func,
            local_sections=local_boundary_samples,
            local_constraints=local_boundary_constraints,
            param_levels=np.linspace(0.0, 1.0, len(cxo_trace_theta), endpoint=False),
            dphi_hint=dphi_hint,
            phi_hit_tol_factor=phi_hit_tol_factor,
        )

    for ip, phi in enumerate(phi_samples):
        sec = scaffold.sections[ip]
        R_ax, Z_ax = section_axes[ip]
        cR_ax = np.zeros(n_coeff, dtype=float); cR_ax[0] = R_ax
        cZ_ax = np.zeros(n_coeff, dtype=float); cZ_ax[0] = Z_ax
        r_ok = [0.0]
        cR_rows = [cR_ax]
        cZ_rows = [cZ_ax]
        for ir, r_val in enumerate(r_levels):
            ok = np.asarray(sec.valid[ir], dtype=bool)
            if int(np.sum(ok)) < max(8, len(theta_levels) // 3):
                continue
            cR, cZ = fit_ring_fourier(np.asarray(theta_levels)[ok], np.asarray(sec.R[ir])[ok], np.asarray(sec.Z[ir])[ok], n_coeff)
            r_ok.append(float(r_val))
            cR_rows.append(cR)
            cZ_rows.append(cZ)
        cR_arr = np.asarray(cR_rows, dtype=float)
        cZ_arr = np.asarray(cZ_rows, dtype=float)
        r_ok_arr = np.asarray(r_ok, dtype=float)
        if fit_inner_limit is None:
            inner_mask = np.ones_like(r_ok_arr, dtype=bool)
        else:
            inner_mask = r_ok_arr <= float(fit_inner_limit) + 1e-12
        spl_R_in, spl_Z_in, _ = build_pchip_family(r_ok_arr[inner_mask], cR_arr[inner_mask], cZ_arr[inner_mask])
        r_max = float(np.max(r_ok_arr[inner_mask])) if np.any(inner_mask) else 0.0

        spl_R_CXO = None
        spl_Z_CXO = None
        cxo_param = None
        boundary_source = "none"
        boundary_valid_fraction = 0.0
        if boundary_family is not None:
            spl_R_CXO, spl_Z_CXO, cxo_param, boundary_valid_fraction, boundary_source = boundary_family.section_splines(float(phi))
        elif cxo_local is not None and ip < len(cxo_local) and cxo_local[ip] is not None:
            spl_R_CXO, spl_Z_CXO = cxo_local[ip]
            boundary_source = "local-cxo"
            boundary_valid_fraction = 1.0

        cR_ext_arr = cR_arr.copy()
        cZ_ext_arr = cZ_arr.copy()
        r_ext_arr = r_ok_arr.copy()
        if spl_R_CXO is not None and spl_Z_CXO is not None:
            t_eval = np.asarray(cxo_trace_theta, dtype=float)
            if cxo_param is None:
                Rb = np.asarray(spl_R_CXO(t_eval), dtype=float)
                Zb = np.asarray(spl_Z_CXO(t_eval), dtype=float)
                theta_b = 2.0 * np.pi * t_eval
            else:
                tb = np.asarray(cxo_param[:-1], dtype=float) if len(cxo_param) > 1 else np.asarray(cxo_param, dtype=float)
                Rb = np.asarray(spl_R_CXO(tb), dtype=float)
                Zb = np.asarray(spl_Z_CXO(tb), dtype=float)
                theta_b = 2.0 * np.pi * tb
            if len(Rb) >= max(8, n_coeff):
                cR_bnd, cZ_bnd = fit_ring_fourier(theta_b[:len(Rb)], Rb, Zb, n_coeff)
                if np.any(np.isclose(r_ext_arr, 1.0, atol=1e-8)):
                    i1 = int(np.argmin(np.abs(r_ext_arr - 1.0)))
                    cR_ext_arr[i1] = cR_bnd
                    cZ_ext_arr[i1] = cZ_bnd
                else:
                    r_ext_arr = np.append(r_ext_arr, 1.0)
                    cR_ext_arr = np.vstack([cR_ext_arr, cR_bnd])
                    cZ_ext_arr = np.vstack([cZ_ext_arr, cZ_bnd])
                order = np.argsort(r_ext_arr)
                r_ext_arr = r_ext_arr[order]
                cR_ext_arr = cR_ext_arr[order]
                cZ_ext_arr = cZ_ext_arr[order]

        spl_R_ext, spl_Z_ext, r_aug_ext = build_pchip_family(r_ext_arr, cR_ext_arr, cZ_ext_arr)

        fits.append(SectionFit(
            phi=float(phi), R_ax=float(R_ax), Z_ax=float(Z_ax),
            spl_R_in=spl_R_in, spl_Z_in=spl_Z_in,
            spl_R_ext=spl_R_ext, spl_Z_ext=spl_Z_ext,
            r_max=r_max, r_aug_ext=r_aug_ext,
            spl_R_CXO=spl_R_CXO, spl_Z_CXO=spl_Z_CXO, cxo_param=cxo_param,
            scaffold_valid=(spl_R_in is not None),
            trace_valid_fraction=float(np.mean(sec.valid)),
            boundary_source=boundary_source,
            boundary_valid_fraction=boundary_valid_fraction,
        ))
    return SectionScaffoldBundle(scaffold_3d=scaffold, fits=fits, reference_index=ref_idx, boundary_family=boundary_family)

