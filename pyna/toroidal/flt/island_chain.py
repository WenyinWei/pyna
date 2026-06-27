"""Boundary island-chain search utilities for toroidal field-line maps."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import gcd
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
    chain_id: int | None = None
    cycle_id: int | None = None
    point_index: int | None = None
    map_span: float | None = None
    winding: tuple[int, int] | None = None
    reduced_winding: tuple[int, int] | None = None
    section_phi: float | None = None
    metadata: dict = field(default_factory=dict, compare=False, repr=False)

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
        fp.metadata.update({
            "period": int(self.period),
            "residual": float(self.residual),
            "lower_period_residual": float(self.lower_period_residual),
            **dict(self.metadata),
        })
        for key, value in (
            ("chain_id", self.chain_id),
            ("cycle_id", self.cycle_id),
            ("point_index", self.point_index),
            ("map_span", self.map_span),
            ("winding", self.winding),
            ("reduced_winding", self.reduced_winding),
            ("section_phi", self.section_phi),
        ):
            if value is not None:
                fp.metadata[key] = value
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


@dataclass(frozen=True)
class BoundaryIslandCycle:
    """One ordered fixed-point cycle under a toroidal-span map.

    ``points`` are ordered by repeated application of ``P_span``.  For a
    period-m point this is the m distinct points of the cycle; the closing
    endpoint ``P_span^m(x0)`` is summarized by ``closure_residual`` instead of
    being stored as a duplicate point.
    """

    points: tuple[FixedPoint, ...]
    period: int
    kind: str
    map_span: float
    source_index: int = -1
    cycle_id: int | None = None
    chain_id: int | None = None
    closure_residual: float = np.inf
    map_count: int = 0
    alive: bool = False
    winding: tuple[int, int] | None = None
    reduced_winding: tuple[int, int] | None = None
    metadata: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def R(self) -> np.ndarray:
        return np.asarray([float(fp.R) for fp in self.points], dtype=float)

    @property
    def Z(self) -> np.ndarray:
        return np.asarray([float(fp.Z) for fp in self.points], dtype=float)

    @property
    def phi(self) -> np.ndarray:
        return np.asarray([float(fp.phi) for fp in self.points], dtype=float)

    @property
    def seed_point(self) -> FixedPoint | None:
        return self.points[0] if self.points else None

    @property
    def is_closed(self) -> bool:
        return bool(self.alive and np.isfinite(self.closure_residual))


@dataclass(frozen=True)
class BoundaryIslandChain:
    """Connected boundary-island topology assembled from ordered cycles."""

    cycles: tuple[BoundaryIslandCycle, ...]
    period: int
    map_span: float
    chain_id: int
    winding: tuple[int, int]
    reduced_winding: tuple[int, int]
    metadata: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def x_cycles(self) -> list[BoundaryIslandCycle]:
        return [cyc for cyc in self.cycles if str(cyc.kind).upper() == "X"]

    @property
    def o_cycles(self) -> list[BoundaryIslandCycle]:
        return [cyc for cyc in self.cycles if str(cyc.kind).upper() == "O"]

    @property
    def fixed_points(self) -> tuple[FixedPoint, ...]:
        points: list[FixedPoint] = []
        for cycle in self.cycles:
            points.extend(cycle.points)
        return tuple(points)

    @property
    def connected_component_count(self) -> int:
        return max(1, gcd(abs(int(self.winding[0])), abs(int(self.winding[1]))))

    @property
    def points_per_connected_component(self) -> int:
        return max(1, abs(int(self.winding[0])) // self.connected_component_count)

    @property
    def map_advance_per_component(self) -> int:
        return int(self.reduced_winding[1])

    def as_fp_by_section(self, phi_sections: Sequence[float] | None = None) -> dict[float, dict[str, list[FixedPoint]]]:
        """Return a plotting payload with the same shape as ``fp_by_sec``."""

        if phi_sections is None:
            keys = sorted({float(fp.phi) for fp in self.fixed_points})
        else:
            keys = [float(p) for p in phi_sections]
        payload = {float(phi): {"xpts": [], "opts": []} for phi in keys}
        for fp in self.fixed_points:
            phi_key = min(payload, key=lambda p: abs(p - float(fp.phi))) if payload else float(fp.phi)
            payload.setdefault(phi_key, {"xpts": [], "opts": []})
            kind = str(getattr(fp, "kind", "")).upper()
            bucket = "xpts" if kind == "X" else "opts" if kind == "O" else "unknown"
            payload[phi_key].setdefault(bucket, [])
            payload[phi_key][bucket].append(fp)
        return payload

    def to_island_chain(self, *, proximity_tol: float = 1.0):
        """Convert this semantic chain to the existing toroidal ``IslandChain``."""

        from pyna.topo.toroidal import IslandChain

        return IslandChain.from_fixed_points(
            O_points=[fp for fp in self.fixed_points if str(fp.kind).upper() == "O"],
            X_points=[fp for fp in self.fixed_points if str(fp.kind).upper() == "X"],
            m=int(self.winding[0]),
            n=int(self.winding[1]),
            proximity_tol=float(proximity_tol),
            ambient_dim=2,
        )


@dataclass(frozen=True)
class BoundaryIslandDenseCycle:
    """Continuous sampled 3-D geometry for one periodic field-line cycle."""

    phi: np.ndarray
    R: np.ndarray
    Z: np.ndarray
    alive: np.ndarray
    period: int
    kind: str
    map_span: float
    source_phi: float
    cycle_id: int | None = None
    chain_id: int | None = None
    closure_residual: float = np.inf
    winding: tuple[int, int] | None = None
    reduced_winding: tuple[int, int] | None = None
    section_points: tuple[FixedPoint, ...] = field(default_factory=tuple, compare=False, repr=False)
    metadata: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "phi", np.asarray(self.phi, dtype=float))
        object.__setattr__(self, "R", np.asarray(self.R, dtype=float))
        object.__setattr__(self, "Z", np.asarray(self.Z, dtype=float))
        object.__setattr__(self, "alive", np.asarray(self.alive, dtype=bool))
        if not (self.phi.shape == self.R.shape == self.Z.shape == self.alive.shape):
            raise ValueError("phi, R, Z, and alive must have identical shapes")

    @property
    def n_samples(self) -> int:
        return int(self.phi.size)

    @property
    def xyz(self) -> np.ndarray:
        return np.column_stack([
            self.R * np.cos(self.phi),
            self.R * np.sin(self.phi),
            self.Z,
        ])

    @property
    def complete(self) -> bool:
        return bool(self.alive.size > 0 and bool(self.alive[-1]))

    def as_arrays(self, *, include_xyz: bool = False) -> dict[str, np.ndarray]:
        arrays = {
            "phi": self.phi.copy(),
            "R": self.R.copy(),
            "Z": self.Z.copy(),
            "alive": self.alive.copy(),
        }
        if include_xyz:
            xyz = self.xyz
            arrays.update({"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]})
        return arrays

    def save_npz(self, path, *, include_xyz: bool = True) -> None:
        arrays = self.as_arrays(include_xyz=include_xyz)
        arrays.update({
            "period": np.asarray(self.period, dtype=int),
            "kind": np.asarray(str(self.kind)),
            "map_span": np.asarray(float(self.map_span)),
            "source_phi": np.asarray(float(self.source_phi)),
            "cycle_id": np.asarray(-1 if self.cycle_id is None else int(self.cycle_id), dtype=int),
            "chain_id": np.asarray(-1 if self.chain_id is None else int(self.chain_id), dtype=int),
            "closure_residual": np.asarray(float(self.closure_residual)),
        })
        np.savez(str(path), **arrays)


@dataclass(frozen=True)
class BoundaryIslandDenseChain:
    """Continuous sampled 3-D geometry for all cycles in one island chain."""

    dense_cycles: tuple[BoundaryIslandDenseCycle, ...]
    period: int
    map_span: float
    chain_id: int
    winding: tuple[int, int]
    reduced_winding: tuple[int, int]
    metadata: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def x_cycles(self) -> list[BoundaryIslandDenseCycle]:
        return [cyc for cyc in self.dense_cycles if str(cyc.kind).upper() == "X"]

    @property
    def o_cycles(self) -> list[BoundaryIslandDenseCycle]:
        return [cyc for cyc in self.dense_cycles if str(cyc.kind).upper() == "O"]

    @property
    def connected_component_count(self) -> int:
        return max(1, gcd(abs(int(self.winding[0])), abs(int(self.winding[1]))))

    @property
    def points_per_connected_component(self) -> int:
        return max(1, abs(int(self.winding[0])) // self.connected_component_count)

    def as_arrays(self, *, include_xyz: bool = True) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for i, cycle in enumerate(self.dense_cycles):
            for key, value in cycle.as_arrays(include_xyz=include_xyz).items():
                arrays[f"cycle{i}_{key}"] = value
            arrays[f"cycle{i}_kind"] = np.asarray(str(cycle.kind))
            arrays[f"cycle{i}_cycle_id"] = np.asarray(
                -1 if cycle.cycle_id is None else int(cycle.cycle_id),
                dtype=int,
            )
        arrays.update({
            "period": np.asarray(self.period, dtype=int),
            "map_span": np.asarray(float(self.map_span)),
            "chain_id": np.asarray(int(self.chain_id), dtype=int),
            "winding": np.asarray(self.winding, dtype=int),
            "reduced_winding": np.asarray(self.reduced_winding, dtype=int),
        })
        return arrays

    def save_npz(self, path, *, include_xyz: bool = True) -> None:
        np.savez(str(path), **self.as_arrays(include_xyz=include_xyz))


def _as_periods(periods: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(periods, (int, np.integer)):
        periods_tuple = (int(periods),)
    else:
        periods_tuple = tuple(int(p) for p in periods)
    periods_tuple = tuple(p for p in periods_tuple if p > 0)
    if not periods_tuple:
        raise ValueError("periods must contain at least one positive integer")
    return tuple(sorted(set(periods_tuple)))


def _reduced_winding(m: int, n: int) -> tuple[int, int]:
    m_i = int(m)
    n_i = int(n)
    g = gcd(abs(m_i), abs(n_i))
    if g <= 0:
        return m_i, n_i
    return m_i // g, n_i // g


def _cycle_sort_angle(cycle: BoundaryIslandCycle) -> float:
    R = cycle.R
    Z = cycle.Z
    if R.size == 0:
        return 0.0
    return float(np.arctan2(np.nanmean(Z), np.nanmean(R)))


def _fixed_point_period(fp: BoundaryIslandFixedPoint | FixedPoint, default: int | None = None) -> int:
    value = getattr(fp, "period", None)
    if value is None:
        value = getattr(fp, "metadata", {}).get("period") if hasattr(fp, "metadata") else None
    if value is None:
        value = default
    if value is None:
        raise ValueError("fixed point period is missing; pass period=...")
    period = int(value)
    if period <= 0:
        raise ValueError("fixed point period must be positive")
    return period


def _fixed_point_kind(fp: BoundaryIslandFixedPoint | FixedPoint) -> str:
    return str(getattr(fp, "kind", "")).upper()


def _fixed_point_residual(fp: BoundaryIslandFixedPoint | FixedPoint) -> float:
    value = getattr(fp, "residual", None)
    if value is None:
        value = getattr(fp, "metadata", {}).get("residual") if hasattr(fp, "metadata") else None
    try:
        return float(value)
    except Exception:
        return np.inf


def _clone_cycle_fixed_point(
    source: BoundaryIslandFixedPoint | FixedPoint,
    *,
    R: float,
    Z: float,
    phi: float,
    point_index: int,
    period: int,
    map_span: float,
    closure_residual: float,
    cycle_id: int | None = None,
    chain_id: int | None = None,
    winding: tuple[int, int] | None = None,
    reduced_winding: tuple[int, int] | None = None,
    source_index: int | None = None,
) -> FixedPoint:
    kind = _fixed_point_kind(source)
    DPm = np.asarray(getattr(source, "DPm", np.eye(2)), dtype=float).reshape(2, 2).copy()
    fp = FixedPoint(phi=float(phi), R=float(R), Z=float(Z), DPm=DPm, kind=kind)
    fp.period = int(period)
    fp.residual = _fixed_point_residual(source)
    eig = getattr(source, "eigenvalues", None)
    if eig is not None:
        fp.eigenvalues = np.asarray(eig).copy()
    cls = classify_monodromy_2x2(DPm)
    fp.trace = cls.trace
    fp.determinant = cls.determinant
    fp.discriminant = cls.discriminant
    fp.monodromy_classification_reason = cls.reason
    metadata = dict(getattr(source, "metadata", {}) or {})
    metadata.update({
        "period": int(period),
        "map_span": float(map_span),
        "point_index": int(point_index),
        "closure_residual": float(closure_residual),
    })
    if source_index is not None:
        metadata["source_index"] = int(source_index)
    for key, value in (
        ("chain_id", chain_id),
        ("cycle_id", cycle_id),
        ("winding", winding),
        ("reduced_winding", reduced_winding),
        ("section_phi", float(getattr(source, "phi", phi))),
    ):
        if value is not None:
            metadata[key] = value
    fp.metadata.update(metadata)
    if kind == "X":
        stable, unstable = _stable_unstable_eigenvectors(DPm)
        fp.stable_eigenvec = stable
        fp.unstable_eigenvec = unstable
    return fp


def _annotate_cycle(
    cycle: BoundaryIslandCycle,
    *,
    cycle_id: int | None = None,
    chain_id: int | None = None,
    winding: tuple[int, int] | None = None,
    reduced_winding: tuple[int, int] | None = None,
) -> BoundaryIslandCycle:
    cycle_id = cycle.cycle_id if cycle_id is None else int(cycle_id)
    chain_id = cycle.chain_id if chain_id is None else int(chain_id)
    winding = cycle.winding if winding is None else tuple(map(int, winding))
    reduced_winding = (
        cycle.reduced_winding
        if reduced_winding is None
        else tuple(map(int, reduced_winding))
    )
    points = tuple(
        _clone_cycle_fixed_point(
            fp,
            R=float(fp.R),
            Z=float(fp.Z),
            phi=float(fp.phi),
            point_index=i,
            period=int(cycle.period),
            map_span=float(cycle.map_span),
            closure_residual=float(cycle.closure_residual),
            cycle_id=cycle_id,
            chain_id=chain_id,
            winding=winding,
            reduced_winding=reduced_winding,
            source_index=cycle.source_index,
        )
        for i, fp in enumerate(cycle.points)
    )
    metadata = dict(cycle.metadata)
    metadata.update({
        "cycle_id": cycle_id,
        "chain_id": chain_id,
    })
    if winding is not None:
        metadata["winding"] = winding
    if reduced_winding is not None:
        metadata["reduced_winding"] = reduced_winding
    return replace(
        cycle,
        points=points,
        cycle_id=cycle_id,
        chain_id=chain_id,
        winding=winding,
        reduced_winding=reduced_winding,
        metadata=metadata,
    )


def _cycle_distance(a: BoundaryIslandCycle, b: BoundaryIslandCycle) -> float:
    if len(a.points) != len(b.points) or len(a.points) == 0:
        return np.inf
    A = np.column_stack([a.R, a.Z])
    B = np.column_stack([b.R, b.Z])
    best = np.inf
    for shift in range(len(a.points)):
        rolled = np.roll(B, -shift, axis=0)
        dist = float(np.max(np.linalg.norm(A - rolled, axis=1)))
        if dist < best:
            best = dist
    return best


def _deduplicate_cycles(
    cycles: Sequence[BoundaryIslandCycle],
    *,
    tol: float,
) -> tuple[BoundaryIslandCycle, ...]:
    kept: list[BoundaryIslandCycle] = []
    ordered = sorted(
        cycles,
        key=lambda c: (
            int(c.period),
            str(c.kind).upper(),
            float(c.closure_residual),
            int(c.source_index),
        ),
    )
    for cycle in ordered:
        duplicate_index = -1
        for i, old in enumerate(kept):
            if int(cycle.period) != int(old.period):
                continue
            if str(cycle.kind).upper() != str(old.kind).upper():
                continue
            if _cycle_distance(cycle, old) <= float(tol):
                duplicate_index = i
                break
        if duplicate_index < 0:
            kept.append(cycle)
            continue
        old = kept[duplicate_index]
        if float(cycle.closure_residual) < float(old.closure_residual):
            kept[duplicate_index] = cycle
    return tuple(sorted(kept, key=lambda c: (int(c.period), str(c.kind).upper(), _cycle_sort_angle(c))))


def _cycle_source_phi(cycle: BoundaryIslandCycle) -> float:
    if "section_phi" in cycle.metadata:
        return float(cycle.metadata["section_phi"])
    if "source_phi" in cycle.metadata:
        return float(cycle.metadata["source_phi"])
    if cycle.points:
        fp = cycle.points[0]
        metadata = getattr(fp, "metadata", {}) or {}
        if "section_phi" in metadata:
            return float(metadata["section_phi"])
        return float(fp.phi)
    return 0.0


def _section_delta_phi(source_phi: float, section_phi: float, period: float) -> float:
    period = abs(float(period))
    if period <= 0.0 or not np.isfinite(period):
        raise ValueError("section period must be positive and finite")
    delta = float(np.mod(float(section_phi) - float(source_phi), period))
    if delta <= 1.0e-12 or abs(delta - period) <= 1.0e-12:
        return 0.0
    return delta


def _target_phis_for_section(
    *,
    source_phi: float,
    phi_end: float,
    section_phi: float,
    section_period: float,
) -> np.ndarray:
    period = abs(float(section_period))
    if period <= 0.0 or not np.isfinite(period):
        raise ValueError("section_period must be positive and finite")
    start = float(source_phi)
    end = float(phi_end)
    if end < start:
        start, end = end, start
    delta = _section_delta_phi(float(source_phi), float(section_phi), period)
    first = float(source_phi) + delta
    while first < start - 1.0e-10:
        first += period
    targets: list[float] = []
    val = first
    while val <= end + 1.0e-10:
        targets.append(float(val))
        val += period
    return np.asarray(targets, dtype=float)


def _interp_dense_cycle_at_phi(
    dense: BoundaryIslandDenseCycle,
    target_phi: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = np.asarray(target_phi, dtype=float).ravel()
    if targets.size == 0 or dense.n_samples < 2:
        return np.empty(0), np.empty(0), np.zeros(targets.size, dtype=bool)
    phi = np.asarray(dense.phi, dtype=float)
    R = np.asarray(dense.R, dtype=float)
    Z = np.asarray(dense.Z, dtype=float)
    alive = np.asarray(dense.alive, dtype=bool)
    if phi[0] > phi[-1]:
        phi = phi[::-1]
        R = R[::-1]
        Z = Z[::-1]
        alive = alive[::-1]
    order = np.argsort(phi)
    phi = phi[order]
    R = R[order]
    Z = Z[order]
    alive = alive[order]

    ok = (targets >= phi[0] - 1.0e-10) & (targets <= phi[-1] + 1.0e-10)
    R_out = np.full(targets.shape, np.nan, dtype=float)
    Z_out = np.full(targets.shape, np.nan, dtype=float)
    if np.any(ok):
        clipped = np.clip(targets[ok], phi[0], phi[-1])
        R_out[ok] = np.interp(clipped, phi, R)
        Z_out[ok] = np.interp(clipped, phi, Z)
        idx = np.searchsorted(phi, clipped, side="right") - 1
        idx = np.clip(idx, 0, len(phi) - 2)
        ok_alive = alive[idx] & alive[idx + 1]
        # Exact last-sample hits use the last alive flag.
        last = np.isclose(clipped, phi[-1], rtol=0.0, atol=1.0e-10)
        if np.any(last):
            ok_alive[last] = alive[-1]
        ok_indices = np.flatnonzero(ok)
        ok[ok_indices] &= ok_alive
    ok &= np.isfinite(R_out) & np.isfinite(Z_out)
    return R_out, Z_out, ok


def _deduplicate_section_crossings(
    R: Sequence[float],
    Z: Sequence[float],
    target_phi: Sequence[float],
    *,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    R_arr = np.asarray(R, dtype=float).ravel()
    Z_arr = np.asarray(Z, dtype=float).ravel()
    phi_arr = np.asarray(target_phi, dtype=float).ravel()
    finite = np.isfinite(R_arr) & np.isfinite(Z_arr) & np.isfinite(phi_arr)
    R_arr = R_arr[finite]
    Z_arr = Z_arr[finite]
    phi_arr = phi_arr[finite]
    source_index = np.flatnonzero(finite)
    if R_arr.size == 0:
        return R_arr, Z_arr, phi_arr, source_index.astype(int)
    keep: list[int] = []
    for i, (r, z) in enumerate(zip(R_arr, Z_arr)):
        if any(np.hypot(r - R_arr[j], z - Z_arr[j]) <= float(tol) for j in keep):
            continue
        keep.append(i)
    keep_arr = np.asarray(keep, dtype=int)
    return R_arr[keep_arr], Z_arr[keep_arr], phi_arr[keep_arr], source_index[keep_arr].astype(int)


def _section_cycle_from_dense_cycle(
    dense: BoundaryIslandDenseCycle,
    base_cycle: BoundaryIslandCycle,
    *,
    section_phi: float,
    section_index: int,
    wall_R: Sequence[float] | None = None,
    wall_Z: Sequence[float] | None = None,
    section_dedup_tol: float = 5.0e-4,
) -> BoundaryIslandCycle:
    section_period = abs(float(dense.map_span))
    target_phi = _target_phis_for_section(
        source_phi=float(dense.source_phi),
        phi_end=float(dense.phi[-1]) if dense.n_samples else float(dense.source_phi),
        section_phi=float(section_phi),
        section_period=section_period,
    )
    R_raw, Z_raw, ok = _interp_dense_cycle_at_phi(dense, target_phi)
    if wall_R is not None and wall_Z is not None and ok.size:
        ok &= _points_in_polygon(
            R_raw,
            Z_raw,
            np.asarray(wall_R, dtype=float),
            np.asarray(wall_Z, dtype=float),
        )
    R_keep, Z_keep, phi_keep, source_index = _deduplicate_section_crossings(
        R_raw[ok],
        Z_raw[ok],
        target_phi[ok],
        tol=float(section_dedup_tol),
    )
    if base_cycle.points:
        source_points = tuple(
            base_cycle.points[int(i) % len(base_cycle.points)]
            for i in source_index
        )
    else:
        source_points = ()
    point_indices = tuple(int(i) % max(1, int(base_cycle.period)) for i in source_index)
    section_cycle = _clone_section_cycle(
        base_cycle,
        section_phi=float(section_phi),
        section_index=int(section_index),
        delta_phi=_section_delta_phi(dense.source_phi, section_phi, section_period),
        R_values=R_keep,
        Z_values=Z_keep,
        source_points=source_points,
        point_indices=point_indices,
        alive=bool(R_keep.size >= min(int(base_cycle.period), max(1, int(base_cycle.period)))),
        map_count=int(R_keep.size),
    )
    metadata = dict(section_cycle.metadata)
    metadata.update({
        "section_cycle_source": "dense_orbit_crossings",
        "raw_crossing_count": int(np.count_nonzero(ok)),
        "dedup_crossing_count": int(R_keep.size),
        "section_dedup_tol": float(section_dedup_tol),
        "target_phi_min": float(np.min(phi_keep)) if phi_keep.size else None,
        "target_phi_max": float(np.max(phi_keep)) if phi_keep.size else None,
    })
    return replace(section_cycle, metadata=metadata)


def _clone_section_cycle(
    cycle: BoundaryIslandCycle,
    *,
    section_phi: float,
    section_index: int,
    delta_phi: float,
    R_values: Sequence[float],
    Z_values: Sequence[float],
    source_points: Sequence[FixedPoint] | None = None,
    point_indices: Sequence[int] | None = None,
    alive: bool,
    map_count: int,
) -> BoundaryIslandCycle:
    points = []
    if source_points is None:
        source_points = cycle.points
    if point_indices is None:
        point_indices = tuple(range(len(source_points)))
    for local_i, (source_fp, point_index, R, Z) in enumerate(
        zip(source_points, point_indices, R_values, Z_values)
    ):
        fp = _clone_cycle_fixed_point(
            source_fp,
            R=float(R),
            Z=float(Z),
            phi=float(section_phi),
            point_index=int(point_index),
            period=int(cycle.period),
            map_span=float(cycle.map_span),
            closure_residual=float(cycle.closure_residual),
            cycle_id=cycle.cycle_id,
            chain_id=cycle.chain_id,
            winding=cycle.winding,
            reduced_winding=cycle.reduced_winding,
            source_index=cycle.source_index,
        )
        fp.metadata.update({
            "section_phi": float(section_phi),
            "section_index": int(section_index),
            "section_delta_phi": float(delta_phi),
            "orbit_point_index": int(point_index),
            "section_local_index": int(local_i),
        })
        points.append(fp)
    metadata = dict(cycle.metadata)
    metadata.update({
        "section_phi": float(section_phi),
        "section_index": int(section_index),
        "section_delta_phi": float(delta_phi),
    })
    return replace(
        cycle,
        points=tuple(points),
        map_count=int(map_count),
        alive=bool(alive),
        metadata=metadata,
    )


def trace_fixed_point_cycles_span_field(
    field,
    fixed_points: Sequence[BoundaryIslandFixedPoint | FixedPoint],
    *,
    map_span: float = 2.0 * np.pi,
    period: int | None = None,
    DPhi: float = 0.01,
    wall_R: Sequence[float] | None = None,
    wall_Z: Sequence[float] | None = None,
    extend_phi: bool = True,
    n_threads: int = -1,
) -> tuple[BoundaryIslandCycle, ...]:
    """Trace ordered cycles from converged fixed points using ``P_span``.

    This is a batch wrapper around the existing cyna arbitrary-span map tracer.
    For each source point it stores the m distinct cycle points and summarizes
    the closing endpoint with ``closure_residual``.
    """

    if not np.isfinite(float(map_span)) or abs(float(map_span)) <= 1.0e-14:
        raise ValueError("map_span must be a nonzero finite toroidal angle")
    if not fixed_points:
        return ()

    by_period_phi: dict[tuple[int, float], list[tuple[int, BoundaryIslandFixedPoint | FixedPoint]]] = {}
    for source_index, fp in enumerate(fixed_points):
        p = _fixed_point_period(fp, default=period)
        phi = float(getattr(fp, "phi", 0.0))
        by_period_phi.setdefault((p, phi), []).append((source_index, fp))

    cycles: list[BoundaryIslandCycle] = []
    for p, phi_start in sorted(by_period_phi):
        items = by_period_phi[(p, phi_start)]
        R0 = np.asarray([float(getattr(fp, "R")) for _idx, fp in items], dtype=float)
        Z0 = np.asarray([float(getattr(fp, "Z")) for _idx, fp in items], dtype=float)
        phi0 = np.asarray([float(getattr(fp, "phi", 0.0)) for _idx, fp in items], dtype=float)
        counts, flat_R, flat_Z = trace_map_batch_span_field(
            field,
            R0,
            Z0,
            float(phi_start),
            float(map_span),
            int(p),
            float(DPhi),
            wall_R=wall_R,
            wall_Z=wall_Z,
            extend_phi=extend_phi,
            n_threads=n_threads,
        )
        counts_arr = np.asarray(counts, dtype=int).ravel()
        R_flat = np.asarray(flat_R, dtype=float).ravel()
        Z_flat = np.asarray(flat_Z, dtype=float).ravel()
        for local_index, (source_index, source_fp) in enumerate(items):
            base = local_index * int(p)
            count = int(counts_arr[local_index]) if local_index < counts_arr.size else 0
            if count >= int(p):
                endpoint_R = float(R_flat[base + int(p) - 1])
                endpoint_Z = float(Z_flat[base + int(p) - 1])
                closure = float(np.hypot(endpoint_R - R0[local_index], endpoint_Z - Z0[local_index]))
            else:
                closure = np.inf

            point_R = [float(R0[local_index])]
            point_Z = [float(Z0[local_index])]
            for step_index in range(1, int(p)):
                if count < step_index:
                    break
                point_R.append(float(R_flat[base + step_index - 1]))
                point_Z.append(float(Z_flat[base + step_index - 1]))

            points = tuple(
                _clone_cycle_fixed_point(
                    source_fp,
                    R=r,
                    Z=z,
                    phi=float(phi0[local_index]) + i * float(map_span),
                    point_index=i,
                    period=int(p),
                    map_span=float(map_span),
                    closure_residual=float(closure),
                    source_index=source_index,
                )
                for i, (r, z) in enumerate(zip(point_R, point_Z))
            )
            cycles.append(BoundaryIslandCycle(
                points=points,
                period=int(p),
                kind=_fixed_point_kind(source_fp),
                map_span=float(map_span),
                source_index=int(source_index),
                closure_residual=float(closure),
                map_count=int(count),
                alive=bool(count >= int(p)),
                metadata={
                    "source_index": int(source_index),
                    "source_phi": float(phi0[local_index]),
                },
            ))
    return tuple(cycles)


def trace_fixed_point_cycle_span_field(
    field,
    fixed_point: BoundaryIslandFixedPoint | FixedPoint,
    *,
    map_span: float = 2.0 * np.pi,
    period: int | None = None,
    DPhi: float = 0.01,
    wall_R: Sequence[float] | None = None,
    wall_Z: Sequence[float] | None = None,
    extend_phi: bool = True,
    n_threads: int = -1,
) -> BoundaryIslandCycle:
    """Trace one ordered fixed-point cycle under an arbitrary-span map."""

    cycles = trace_fixed_point_cycles_span_field(
        field,
        [fixed_point],
        map_span=map_span,
        period=period,
        DPhi=DPhi,
        wall_R=wall_R,
        wall_Z=wall_Z,
        extend_phi=extend_phi,
        n_threads=n_threads,
    )
    if not cycles:
        raise ValueError("no cycle was traced")
    return cycles[0]


def trace_fixed_point_cycle_sections_span_field(
    field,
    fixed_point_or_cycle: BoundaryIslandFixedPoint | FixedPoint | BoundaryIslandCycle,
    section_phis: Sequence[float],
    *,
    map_span: float = 2.0 * np.pi,
    period: int | None = None,
    DPhi: float = 0.01,
    dphi_out: float | None = None,
    fd_eps: float = 1.0e-4,
    section_dedup_tol: float = 5.0e-4,
    wall_by_section: Sequence[tuple[Sequence[float], Sequence[float]]] | None = None,
    extend_phi: bool = True,
    n_threads: int = -1,
) -> dict[float, BoundaryIslandCycle]:
    """Trace one periodic orbit to multiple toroidal sections.

    The fixed point is solved once on its reference section.  The returned
    cycles are extracted from one continuous dense field-line orbit, preserving
    Poincare-map ordering and removing near-duplicate closure hits.
    """

    if isinstance(fixed_point_or_cycle, BoundaryIslandCycle):
        base_cycle = fixed_point_or_cycle
    else:
        base_cycle = trace_fixed_point_cycle_span_field(
            field,
            fixed_point_or_cycle,
            map_span=map_span,
            period=period,
            DPhi=DPhi,
            extend_phi=extend_phi,
            n_threads=n_threads,
        )
    if not base_cycle.points:
        return {}
    if wall_by_section is not None and len(wall_by_section) != len(section_phis):
        raise ValueError("wall_by_section and section_phis must have the same length")

    dense = trace_fixed_point_cycle_dense_span_field(
        field,
        base_cycle,
        DPhi=DPhi,
        dphi_out=dphi_out,
        extend_phi=extend_phi,
        fd_eps=fd_eps,
        n_threads=n_threads,
    )
    out: dict[float, BoundaryIslandCycle] = {}
    for section_index, section_phi_raw in enumerate(section_phis):
        section_phi = float(section_phi_raw)
        wall_R = wall_Z = None
        if wall_by_section is not None:
            wall_R, wall_Z = wall_by_section[section_index]
        out[section_phi] = _section_cycle_from_dense_cycle(
            dense,
            base_cycle,
            section_phi=section_phi,
            section_index=section_index,
            wall_R=wall_R,
            wall_Z=wall_Z,
            section_dedup_tol=float(section_dedup_tol),
        )
    return out


def trace_boundary_island_chain_sections_span_field(
    field,
    chain: BoundaryIslandChain,
    section_phis: Sequence[float],
    *,
    DPhi: float = 0.01,
    dphi_out: float | None = None,
    fd_eps: float = 1.0e-4,
    section_dedup_tol: float = 5.0e-4,
    wall_by_section: Sequence[tuple[Sequence[float], Sequence[float]]] | None = None,
    extend_phi: bool = True,
    n_threads: int = -1,
) -> dict[float, BoundaryIslandChain]:
    """Trace every cycle in one boundary island chain to multiple sections."""

    section_cycles: dict[float, list[BoundaryIslandCycle]] = {
        float(phi): [] for phi in section_phis
    }
    for cycle in chain.cycles:
        traced = trace_fixed_point_cycle_sections_span_field(
            field,
            cycle,
            section_phis,
            DPhi=DPhi,
            dphi_out=dphi_out,
            fd_eps=fd_eps,
            section_dedup_tol=section_dedup_tol,
            wall_by_section=wall_by_section,
            extend_phi=extend_phi,
            n_threads=n_threads,
        )
        for phi, section_cycle in traced.items():
            section_cycles[float(phi)].append(section_cycle)

    out: dict[float, BoundaryIslandChain] = {}
    for phi in section_phis:
        phi_key = float(phi)
        metadata = dict(chain.metadata)
        metadata.update({
            "section_phi": phi_key,
            "section_cycle_source": "orbit_trace",
        })
        out[phi_key] = replace(
            chain,
            cycles=tuple(section_cycles.get(phi_key, [])),
            metadata=metadata,
        )
    return out


def trace_fixed_point_cycle_dense_span_field(
    field,
    fixed_point_or_cycle: BoundaryIslandFixedPoint | FixedPoint | BoundaryIslandCycle,
    *,
    map_span: float = 2.0 * np.pi,
    period: int | None = None,
    DPhi: float = 0.01,
    dphi_out: float | None = None,
    extend_phi: bool = True,
    fd_eps: float = 1.0e-4,
    n_threads: int = -1,
) -> BoundaryIslandDenseCycle:
    """Trace the continuous 3-D geometry of one periodic cycle.

    The dense output follows one representative field line through the full
    ``period * map_span`` orbit.  The section fixed points remain attached in
    ``section_points`` for plotting and indexing.
    """

    _ = n_threads  # kept for API symmetry; single-orbit cyna tracing is serial.
    if isinstance(fixed_point_or_cycle, BoundaryIslandCycle):
        base_cycle = fixed_point_or_cycle
    else:
        base_cycle = trace_fixed_point_cycle_span_field(
            field,
            fixed_point_or_cycle,
            map_span=map_span,
            period=period,
            DPhi=DPhi,
            extend_phi=extend_phi,
            n_threads=n_threads,
        )
    if not base_cycle.points:
        raise ValueError("cannot trace dense output for an empty cycle")
    if not np.isfinite(float(base_cycle.map_span)) or abs(float(base_cycle.map_span)) <= 1.0e-14:
        raise ValueError("cycle map_span must be a nonzero finite toroidal angle")

    seed = base_cycle.points[0]
    source_phi = float(seed.phi)
    phi_end = source_phi + int(base_cycle.period) * float(base_cycle.map_span)
    if dphi_out is None:
        dphi_out = abs(float(DPhi))
    if not np.isfinite(float(dphi_out)) or abs(float(dphi_out)) <= 0.0:
        raise ValueError("dphi_out must be positive and finite")
    R_t, Z_t, phi_t, _DP_t, alive_t = trace_orbit_along_phi_field(
        field,
        float(seed.R),
        float(seed.Z),
        source_phi,
        phi_end,
        float(DPhi),
        extend_phi=extend_phi,
        dphi_out=abs(float(dphi_out)),
        fd_eps=float(fd_eps),
    )
    R_arr = np.asarray(R_t, dtype=float)
    Z_arr = np.asarray(Z_t, dtype=float)
    phi_arr = np.asarray(phi_t, dtype=float)
    alive_arr = np.asarray(alive_t, dtype=bool)
    if R_arr.size and Z_arr.size and alive_arr.size and bool(alive_arr[-1]):
        closure = float(np.hypot(R_arr[-1] - float(seed.R), Z_arr[-1] - float(seed.Z)))
    else:
        closure = np.inf
    metadata = dict(base_cycle.metadata)
    metadata.update({
        "dense_output": True,
        "source_phi": float(source_phi),
        "phi_end": float(phi_end),
        "dphi_out": float(dphi_out),
        "n_samples": int(R_arr.size),
    })
    return BoundaryIslandDenseCycle(
        phi=phi_arr,
        R=R_arr,
        Z=Z_arr,
        alive=alive_arr,
        period=int(base_cycle.period),
        kind=str(base_cycle.kind).upper(),
        map_span=float(base_cycle.map_span),
        source_phi=float(source_phi),
        cycle_id=base_cycle.cycle_id,
        chain_id=base_cycle.chain_id,
        closure_residual=float(closure),
        winding=base_cycle.winding,
        reduced_winding=base_cycle.reduced_winding,
        section_points=tuple(base_cycle.points),
        metadata=metadata,
    )


def trace_boundary_island_chain_dense_span_field(
    field,
    chain: BoundaryIslandChain,
    *,
    DPhi: float = 0.01,
    dphi_out: float | None = None,
    extend_phi: bool = True,
    fd_eps: float = 1.0e-4,
    n_threads: int = -1,
) -> BoundaryIslandDenseChain:
    """Trace continuous 3-D geometry for every cycle in a boundary chain."""

    dense_cycles = tuple(
        trace_fixed_point_cycle_dense_span_field(
            field,
            cycle,
            DPhi=DPhi,
            dphi_out=dphi_out,
            extend_phi=extend_phi,
            fd_eps=fd_eps,
            n_threads=n_threads,
        )
        for cycle in chain.cycles
    )
    metadata = dict(chain.metadata)
    metadata.update({
        "dense_output": True,
        "n_dense_cycles": int(len(dense_cycles)),
    })
    return BoundaryIslandDenseChain(
        dense_cycles=dense_cycles,
        period=int(chain.period),
        map_span=float(chain.map_span),
        chain_id=int(chain.chain_id),
        winding=tuple(map(int, chain.winding)),
        reduced_winding=tuple(map(int, chain.reduced_winding)),
        metadata=metadata,
    )


def assemble_boundary_island_chains(
    cycles: Sequence[BoundaryIslandCycle],
    *,
    m: int | None = None,
    n: int = 1,
    cycle_dedup_tol: float = 5.0e-4,
    pairing_tol: float | None = None,
) -> tuple[BoundaryIslandChain, ...]:
    """Deduplicate ordered cycles and assemble O/X boundary island chains."""

    if not cycles:
        return ()
    unique_cycles = _deduplicate_cycles(cycles, tol=float(cycle_dedup_tol))
    by_period: dict[int, list[BoundaryIslandCycle]] = {}
    for cycle in unique_cycles:
        by_period.setdefault(int(cycle.period), []).append(cycle)

    chains: list[BoundaryIslandChain] = []
    next_chain_id = 0
    next_cycle_id = 0
    for period_value in sorted(by_period):
        period_cycles = sorted(
            by_period[period_value],
            key=lambda c: (0 if str(c.kind).upper() == "O" else 1, _cycle_sort_angle(c)),
        )
        winding = (int(period_value if m is None else m), int(n))
        reduced = _reduced_winding(*winding)
        o_cycles = [c for c in period_cycles if str(c.kind).upper() == "O"]
        x_cycles = [c for c in period_cycles if str(c.kind).upper() == "X"]
        other_cycles = [c for c in period_cycles if str(c.kind).upper() not in {"O", "X"}]
        used_x: set[int] = set()

        def _new_chain(raw_cycles: list[BoundaryIslandCycle]) -> None:
            nonlocal next_chain_id, next_cycle_id
            annotated: list[BoundaryIslandCycle] = []
            for raw in raw_cycles:
                annotated.append(_annotate_cycle(
                    raw,
                    cycle_id=next_cycle_id,
                    chain_id=next_chain_id,
                    winding=winding,
                    reduced_winding=reduced,
                ))
                next_cycle_id += 1
            chains.append(BoundaryIslandChain(
                cycles=tuple(annotated),
                period=int(period_value),
                map_span=float(annotated[0].map_span if annotated else period_cycles[0].map_span),
                chain_id=int(next_chain_id),
                winding=winding,
                reduced_winding=reduced,
                metadata={
                    "period": int(period_value),
                    "m": int(winding[0]),
                    "n": int(winding[1]),
                    "reduced_mn": reduced,
                    "connected_component_count": max(1, gcd(abs(int(winding[0])), abs(int(winding[1])))),
                    "points_per_connected_component": max(
                        1,
                        abs(int(winding[0])) // max(1, gcd(abs(int(winding[0])), abs(int(winding[1])))),
                    ),
                    "map_advance_per_component": int(reduced[1]),
                    "n_cycles": int(len(annotated)),
                },
            ))
            next_chain_id += 1

        for o_cycle in o_cycles:
            raw_group = [o_cycle]
            if x_cycles:
                distances = [
                    (_cycle_distance(o_cycle, x_cycle), i, x_cycle)
                    for i, x_cycle in enumerate(x_cycles)
                    if i not in used_x
                ]
                distances = [row for row in distances if np.isfinite(row[0])]
                if distances:
                    distance, idx, x_cycle = min(distances, key=lambda row: row[0])
                    if pairing_tol is None or distance <= float(pairing_tol):
                        raw_group.append(x_cycle)
                        used_x.add(idx)
            _new_chain(raw_group)

        for i, x_cycle in enumerate(x_cycles):
            if i not in used_x:
                _new_chain([x_cycle])
        for cycle in other_cycles:
            _new_chain([cycle])

    return tuple(chains)


def assemble_boundary_island_chains_field(
    field,
    fixed_points: Sequence[BoundaryIslandFixedPoint | FixedPoint],
    *,
    map_span: float = 2.0 * np.pi,
    period: int | None = None,
    m: int | None = None,
    n: int = 1,
    DPhi: float = 0.01,
    wall_R: Sequence[float] | None = None,
    wall_Z: Sequence[float] | None = None,
    extend_phi: bool = True,
    n_threads: int = -1,
    cycle_dedup_tol: float = 5.0e-4,
    pairing_tol: float | None = None,
) -> tuple[BoundaryIslandChain, ...]:
    """Trace fixed-point cycles and assemble semantic boundary island chains."""

    cycles = trace_fixed_point_cycles_span_field(
        field,
        fixed_points,
        map_span=map_span,
        period=period,
        DPhi=DPhi,
        wall_R=wall_R,
        wall_Z=wall_Z,
        extend_phi=extend_phi,
        n_threads=n_threads,
    )
    return assemble_boundary_island_chains(
        cycles,
        m=m,
        n=n,
        cycle_dedup_tol=cycle_dedup_tol,
        pairing_tol=pairing_tol,
    )


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
    "BoundaryIslandChain",
    "BoundaryIslandDenseChain",
    "BoundaryIslandDenseCycle",
    "BoundaryIslandCycle",
    "BoundaryIslandFixedPoint",
    "BoundaryIslandSeedCandidates",
    "BoundaryIslandSearchResult",
    "assemble_boundary_island_chains",
    "assemble_boundary_island_chains_field",
    "boundary_island_edge_state_payload",
    "boundary_island_topology_payload_field",
    "boundary_recurrence_seed_candidates_field",
    "boundary_seed_grid",
    "fixed_points_by_section_payload",
    "find_boundary_island_fixed_points_field",
    "find_boundary_island_fixed_points_multi_section_field",
    "trace_boundary_island_shapes_field",
    "trace_boundary_island_shapes_multi_section_field",
    "trace_boundary_island_chain_sections_span_field",
    "trace_boundary_island_chain_dense_span_field",
    "trace_fixed_point_cycle_span_field",
    "trace_fixed_point_cycle_sections_span_field",
    "trace_fixed_point_cycle_dense_span_field",
    "trace_fixed_point_cycles_span_field",
    "trace_fixed_point_manifolds_field",
    "trace_fixed_point_manifolds_multi_section_field",
]
