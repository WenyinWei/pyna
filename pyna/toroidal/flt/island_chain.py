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
    trace_poincare_multi_batch_field,
    trace_poincare_batch_twall_field,
    trace_orbit_along_phi_field,
)


@dataclass(frozen=True)
class BoundaryIslandFixedPoint:
    """One X/O point found from a boundary-focused ``P^m`` Newton search."""

    phi: float
    R: float
    Z: float
    map_power: int
    kind: str
    DPm: np.ndarray
    residual: float
    eigenvalues: np.ndarray
    seed_R: float
    seed_Z: float
    lower_map_power_residual: float = np.inf
    chain_id: int | None = None
    cycle_id: int | None = None
    point_index: int | None = None
    map_span: float | None = None
    winding: tuple[int, int] | None = None
    reduced_winding: tuple[int, int] | None = None
    section_phi: float | None = None
    metadata: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def field_period(self) -> float | None:
        if self.map_span is not None:
            return float(self.map_span)
        value = self.metadata.get("field_period")
        return None if value is None else float(value)

    def as_fixed_point(self) -> FixedPoint:
        kind = str(self.kind).upper()
        fp = FixedPoint(
            phi=float(self.phi),
            R=float(self.R),
            Z=float(self.Z),
            DPm=np.asarray(self.DPm, dtype=float).reshape(2, 2).copy(),
            kind=kind,
        )
        fp.map_power = int(self.map_power)
        if self.field_period is not None:
            fp.field_period = float(self.field_period)
        fp.residual = float(self.residual)
        fp.eigenvalues = np.asarray(self.eigenvalues).copy()
        cls = classify_monodromy_2x2(self.DPm)
        fp.trace = cls.trace
        fp.determinant = cls.determinant
        fp.discriminant = cls.discriminant
        fp.monodromy_classification_reason = cls.reason
        fp.metadata.update({
            "map_power": int(self.map_power),
            "residual": float(self.residual),
            "lower_map_power_residual": float(self.lower_map_power_residual),
            **dict(self.metadata),
        })
        for key, value in (
            ("chain_id", self.chain_id),
            ("cycle_id", self.cycle_id),
            ("point_index", self.point_index),
            ("map_span", self.map_span),
            ("field_period", self.field_period),
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

    seeds_by_map_power: dict[int, tuple[np.ndarray, np.ndarray]]
    diagnostics: dict = field(default_factory=dict)

    def seeds_for_map_power(self, map_power: int) -> tuple[np.ndarray, np.ndarray]:
        return self.seeds_by_map_power.get(
            int(map_power),
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
class PoincareSectionTraces:
    """Multi-section Poincare traces from one shared batch of seed orbits."""

    phi_sections: np.ndarray
    seed_R: np.ndarray
    seed_Z: np.ndarray
    counts: np.ndarray
    R_flat: np.ndarray
    Z_flat: np.ndarray
    N_turns: int
    direction: str = "+"
    metadata: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        phi = np.asarray(self.phi_sections, dtype=float).ravel()
        seed_R = np.asarray(self.seed_R, dtype=float).ravel()
        seed_Z = np.asarray(self.seed_Z, dtype=float).ravel()
        counts = np.asarray(self.counts, dtype=int)
        object.__setattr__(self, "phi_sections", phi)
        object.__setattr__(self, "seed_R", seed_R)
        object.__setattr__(self, "seed_Z", seed_Z)
        object.__setattr__(self, "counts", counts.reshape(seed_R.size, phi.size))
        object.__setattr__(self, "R_flat", np.asarray(self.R_flat, dtype=float).ravel())
        object.__setattr__(self, "Z_flat", np.asarray(self.Z_flat, dtype=float).ravel())
        object.__setattr__(self, "N_turns", int(self.N_turns))

    @property
    def n_seed(self) -> int:
        return int(self.seed_R.size)

    @property
    def n_section(self) -> int:
        return int(self.phi_sections.size)

    def seed_section_points(self, seed_index: int, section_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the points for one seed at one recorded section."""

        i = int(seed_index)
        j = int(section_index)
        if not (0 <= i < self.n_seed and 0 <= j < self.n_section):
            raise IndexError("seed_index or section_index out of range")
        n = max(0, min(int(self.counts[i, j]), int(self.N_turns)))
        base = (i * self.n_section + j) * int(self.N_turns)
        stop = base + n
        return self.R_flat[base:stop].copy(), self.Z_flat[base:stop].copy()

    def section_points(self, section: int | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(R, Z, seed_index)`` for one section."""

        if isinstance(section, (int, np.integer)):
            section_index = int(section)
        else:
            phi = float(section)
            section_index = int(np.argmin(np.abs(self.phi_sections - phi)))
        R_parts: list[np.ndarray] = []
        Z_parts: list[np.ndarray] = []
        seed_parts: list[np.ndarray] = []
        for i in range(self.n_seed):
            R_i, Z_i = self.seed_section_points(i, section_index)
            if R_i.size == 0:
                continue
            R_parts.append(R_i)
            Z_parts.append(Z_i)
            seed_parts.append(np.full(R_i.size, i, dtype=int))
        if not R_parts:
            empty = np.empty(0, dtype=float)
            return empty, empty, np.empty(0, dtype=int)
        return np.concatenate(R_parts), np.concatenate(Z_parts), np.concatenate(seed_parts)


@dataclass(frozen=True)
class BoundaryIslandCycle:
    """One ordered fixed-point cycle under a toroidal-span map.

    ``points`` are ordered by repeated application of ``P_span``.  For a
    map-power-m fixed point this stores the m section points in the discrete
    orbit; the closing endpoint ``P_span^m(x0)`` is summarized by
    ``closure_residual`` instead of being stored as a duplicate point.
    """

    points: tuple[FixedPoint, ...]
    cycle_orbit_size: int
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
    cycle_orbit_size: int
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
    cycle_orbit_size: int
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
            "cycle_orbit_size": np.asarray(self.cycle_orbit_size, dtype=int),
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
    cycle_orbit_size: int
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
            "cycle_orbit_size": np.asarray(self.cycle_orbit_size, dtype=int),
            "map_span": np.asarray(float(self.map_span)),
            "chain_id": np.asarray(int(self.chain_id), dtype=int),
            "winding": np.asarray(self.winding, dtype=int),
            "reduced_winding": np.asarray(self.reduced_winding, dtype=int),
        })
        return arrays

    def save_npz(self, path, *, include_xyz: bool = True) -> None:
        np.savez(str(path), **self.as_arrays(include_xyz=include_xyz))


def _as_map_powers(map_powers: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(map_powers, (int, np.integer)):
        map_powers_tuple = (int(map_powers),)
    else:
        map_powers_tuple = tuple(int(p) for p in map_powers)
    map_powers_tuple = tuple(p for p in map_powers_tuple if p > 0)
    if not map_powers_tuple:
        raise ValueError("map_powers must contain at least one positive integer")
    return tuple(sorted(set(map_powers_tuple)))


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


def _fixed_point_map_power(fp: BoundaryIslandFixedPoint | FixedPoint, default: int | None = None) -> int:
    value = getattr(fp, "map_power", None)
    if value is None:
        value = getattr(fp, "metadata", {}).get("map_power") if hasattr(fp, "metadata") else None
    if value is None:
        value = default
    if value is None:
        raise ValueError("fixed point map_power is missing; pass map_power=...")
    map_power = int(value)
    if map_power <= 0:
        raise ValueError("fixed point map_power must be positive")
    return map_power


def _fixed_point_base_map_span(
    fp: BoundaryIslandFixedPoint | FixedPoint,
    *,
    default: float = 2.0 * np.pi,
) -> float:
    metadata = dict(getattr(fp, "metadata", {}) or {})
    value = getattr(fp, "field_period", None)
    if value is None:
        value = getattr(fp, "map_span", None)
    if value is None:
        value = metadata.get("field_period")
    if value is None:
        value = metadata.get("map_span", metadata.get("base_map_span"))
    if value is None:
        value = default
    span = float(value)
    if not np.isfinite(span) or abs(span) <= 1.0e-14:
        raise ValueError("fixed point map_span must be a nonzero finite toroidal angle")
    return span


def _resolve_field_period(field, field_period: float | None) -> float:
    if field_period is not None:
        span = float(field_period)
    else:
        value = getattr(field, "field_period", None)
        if value is not None:
            span = float(value)
        else:
            nfp = int(getattr(field, "nfp", 1))
            span = 2.0 * np.pi / float(nfp)
    if not np.isfinite(span) or abs(span) <= 1.0e-14:
        raise ValueError("field_period must be a nonzero finite toroidal angle")
    return span


def _fixed_point_monodromy_map_span(
    fp: BoundaryIslandFixedPoint | FixedPoint,
    *,
    explicit_field_period: float | None,
) -> tuple[float, str]:
    if explicit_field_period is not None:
        base_span = float(explicit_field_period)
        if not np.isfinite(base_span) or abs(base_span) <= 1.0e-14:
            raise ValueError("field_period must be a nonzero finite toroidal angle")
        map_power = _fixed_point_map_power(fp, default=1)
        return float(map_power) * float(base_span), "map_power_times_field_period"

    metadata = dict(getattr(fp, "metadata", {}) or {})
    for key in ("monodromy_field_period", "monodromy_map_span", "monodromy_refine_total_span"):
        value = metadata.get(key)
        if value is not None:
            span = float(value)
            if np.isfinite(span) and abs(span) > 1.0e-14:
                return span, key

    map_power = _fixed_point_map_power(fp, default=1)
    base_span = _fixed_point_base_map_span(fp)
    return float(map_power) * float(base_span), "map_power_times_field_period"


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
    cycle_orbit_size: int,
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
    fp.cycle_orbit_size = int(cycle_orbit_size)
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
        "cycle_orbit_size": int(cycle_orbit_size),
        "map_span": float(map_span),
        "point_index": int(point_index),
        "map_order_index": int(point_index),
        "poincare_map_power": int(point_index),
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


def _clone_refined_fixed_point(
    source: BoundaryIslandFixedPoint | FixedPoint,
    *,
    R: float,
    Z: float,
    phi: float,
    map_power: int,
    map_span: float,
    DPm: np.ndarray,
    residual: float,
    point_type: int | None = None,
    converged: bool,
    metadata: dict | None = None,
) -> BoundaryIslandFixedPoint | FixedPoint:
    """Clone a fixed point with a freshly computed local monodromy matrix."""

    DPm_arr = np.asarray(DPm, dtype=float).reshape(2, 2).copy()
    eig = np.linalg.eigvals(DPm_arr)
    kind = _classify_DPm_kind(DPm_arr, point_type)
    cls = classify_monodromy_2x2(DPm_arr)
    meta = dict(getattr(source, "metadata", {}) or {})
    meta.update(metadata or {})
    meta.update({
        "map_power": int(map_power),
        "map_span": float(map_span),
        "monodromy_refined": bool(converged),
        "monodromy_residual": float(residual),
        "monodromy_trace": cls.trace,
        "monodromy_determinant": cls.determinant,
        "monodromy_discriminant": cls.discriminant,
        "monodromy_classification_reason": cls.reason,
    })

    if isinstance(source, BoundaryIslandFixedPoint):
        return replace(
            source,
            phi=float(phi),
            R=float(R),
            Z=float(Z),
            map_power=int(map_power),
            kind=kind,
            DPm=DPm_arr,
            residual=float(residual),
            eigenvalues=eig,
            map_span=float(map_span),
            metadata=meta,
        )

    fp = FixedPoint(phi=float(phi), R=float(R), Z=float(Z), DPm=DPm_arr, kind=kind)
    fp.map_power = int(map_power)
    fp.residual = float(residual)
    fp.eigenvalues = eig.copy()
    fp.trace = cls.trace
    fp.determinant = cls.determinant
    fp.discriminant = cls.discriminant
    fp.monodromy_classification_reason = cls.reason
    fp.metadata.update(meta)
    if kind == "X":
        stable, unstable = _stable_unstable_eigenvectors(DPm_arr)
        fp.stable_eigenvec = stable
        fp.unstable_eigenvec = unstable
    return fp


def refine_fixed_points_monodromy_span_field(
    field,
    fixed_points: Sequence[BoundaryIslandFixedPoint | FixedPoint],
    *,
    field_period: float | None = None,
    map_power: int | None = None,
    DPhi: float = 0.01,
    fd_eps: float = 1.0e-4,
    max_iter: int = 20,
    tol: float = 1.0e-10,
    residual_tol: float | None = None,
    keep_unconverged: bool = True,
    extend_phi: bool = True,
    n_threads: int = -1,
) -> tuple[BoundaryIslandFixedPoint | FixedPoint, ...]:
    """Recompute section-local ``DP^m`` for existing fixed points.

    Cycle tracing can clone one source ``DPm`` onto every map-ordered point.
    That preserves eigenvalues but not local eigenvectors.  This helper runs a
    short Newton refinement at each supplied point using the same field-period
    span, so manifold seeds can use the correct local stable/unstable
    directions at every ``P^k`` point.
    """

    if not fixed_points:
        return ()
    field_period_value = _resolve_field_period(field, field_period)
    if residual_tol is None:
        residual_tol = max(20.0 * float(tol), 1.0e-8)

    out: list[BoundaryIslandFixedPoint | FixedPoint | None] = [None] * len(fixed_points)
    by_map_power_phi: dict[tuple[int, float], list[tuple[int, BoundaryIslandFixedPoint | FixedPoint]]] = {}
    for idx, fp in enumerate(fixed_points):
        p = _fixed_point_map_power(fp, default=map_power)
        phi = float(getattr(fp, "phi", 0.0))
        by_map_power_phi.setdefault((p, phi), []).append((idx, fp))

    for (p, phi), items in sorted(by_map_power_phi.items()):
        R0 = np.asarray([float(getattr(fp, "R")) for _idx, fp in items], dtype=float)
        Z0 = np.asarray([float(getattr(fp, "Z")) for _idx, fp in items], dtype=float)
        try:
            result = find_fixed_points_batch_span_field(
                field,
                R0,
                Z0,
                float(phi),
                float(p) * float(field_period_value),
                float(DPhi),
                extend_phi=extend_phi,
                fd_eps=float(fd_eps),
                max_iter=int(max_iter),
                tol=float(tol),
                n_threads=n_threads,
            )
        except ImportError:
            result = None

        if result is None:
            for idx, fp in items:
                if keep_unconverged:
                    out[idx] = fp
            continue

        R_out, Z_out, residual, converged, DPm_flat, _eig_r, _eig_i, point_type = result
        R_arr = np.asarray(R_out, dtype=float).ravel()
        Z_arr = np.asarray(Z_out, dtype=float).ravel()
        residual_arr = np.asarray(residual, dtype=float).ravel()
        converged_arr = np.asarray(converged).astype(bool).ravel()
        point_type_arr = np.asarray(point_type).ravel()
        DPm_arr = np.asarray(DPm_flat, dtype=float).reshape(len(R_arr), 2, 2)
        for local_i, (idx, fp) in enumerate(items):
            ok = (
                local_i < len(R_arr)
                and local_i < len(residual_arr)
                and local_i < len(converged_arr)
                and bool(converged_arr[local_i])
                and np.isfinite(residual_arr[local_i])
                and float(residual_arr[local_i]) <= float(residual_tol)
                and np.all(np.isfinite(DPm_arr[local_i]))
            )
            if not ok:
                if keep_unconverged:
                    out[idx] = fp
                continue
            ptype = int(point_type_arr[local_i]) if local_i < point_type_arr.size else None
            out[idx] = _clone_refined_fixed_point(
                fp,
                R=float(R_arr[local_i]),
                Z=float(Z_arr[local_i]),
                phi=float(phi),
                map_power=int(p),
                map_span=float(field_period_value),
                DPm=DPm_arr[local_i],
                residual=float(residual_arr[local_i]),
                point_type=ptype,
                converged=True,
                metadata={
                    "monodromy_refine_source": "find_fixed_points_batch_span_field",
                    "monodromy_refine_phi": float(phi),
                    "monodromy_refine_total_span": float(p) * float(field_period_value),
                },
            )

    if keep_unconverged:
        return tuple(fp if fp is not None else fixed_points[i] for i, fp in enumerate(out))
    return tuple(fp for fp in out if fp is not None)


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
            cycle_orbit_size=int(cycle.cycle_orbit_size),
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
            int(c.cycle_orbit_size),
            str(c.kind).upper(),
            float(c.closure_residual),
            int(c.source_index),
        ),
    )
    for cycle in ordered:
        duplicate_index = -1
        for i, old in enumerate(kept):
            if int(cycle.cycle_orbit_size) != int(old.cycle_orbit_size):
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
    return tuple(sorted(kept, key=lambda c: (int(c.cycle_orbit_size), str(c.kind).upper(), _cycle_sort_angle(c))))


def deduplicate_boundary_island_cycles(
    cycles: Sequence[BoundaryIslandCycle],
    *,
    cycle_dedup_tol: float = 5.0e-4,
    start_cycle_id: int = 0,
    chain_id: int | None = None,
    winding: tuple[int, int] | None = None,
    reduced_winding: tuple[int, int] | None = None,
) -> tuple[BoundaryIslandCycle, ...]:
    """Deduplicate equivalent fixed-point cycles and assign stable IDs.

    The equivalence test is phase-invariant along the map orbit: cycles with
    the same ordered point set but different starting fixed points are treated
    as one physical cycle.  Returned cycles are annotated with ``cycle_id`` and
    their points carry matching ``cycle_id``, ``point_index`` and
    ``same_cycle_key`` metadata so section cuts can be identified across plots.
    """

    unique = _deduplicate_cycles(cycles, tol=float(cycle_dedup_tol))
    annotated: list[BoundaryIslandCycle] = []
    for offset, cycle in enumerate(unique):
        cycle_id = int(start_cycle_id) + offset
        annotated_cycle = _annotate_cycle(
            cycle,
            cycle_id=cycle_id,
            chain_id=chain_id,
            winding=winding,
            reduced_winding=reduced_winding,
        )
        same_cycle_key = f"chain={chain_id if chain_id is not None else 'none'}:cycle={cycle_id}:kind={annotated_cycle.kind}"
        points = []
        for fp in annotated_cycle.points:
            fp.metadata["same_cycle_key"] = same_cycle_key
            fp.metadata["cycle_point_key"] = f"{same_cycle_key}:point={fp.metadata.get('point_index')}"
            points.append(fp)
        metadata = dict(annotated_cycle.metadata)
        metadata.update({
            "same_cycle_key": same_cycle_key,
            "cycle_dedup_tol": float(cycle_dedup_tol),
        })
        annotated.append(replace(annotated_cycle, points=tuple(points), metadata=metadata))
    return tuple(annotated)


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


def _section_delta_phi(source_phi: float, section_phi: float, section_period: float) -> float:
    section_period = abs(float(section_period))
    if section_period <= 0.0 or not np.isfinite(section_period):
        raise ValueError("section_period must be positive and finite")
    delta = float(np.mod(float(section_phi) - float(source_phi), section_period))
    if delta <= 1.0e-12 or abs(delta - section_period) <= 1.0e-12:
        return 0.0
    return delta


def _target_phis_for_section(
    *,
    source_phi: float,
    phi_end: float,
    section_phi: float,
    section_period: float,
) -> np.ndarray:
    section_period_value = abs(float(section_period))
    if section_period_value <= 0.0 or not np.isfinite(section_period_value):
        raise ValueError("section_period must be positive and finite")
    start = float(source_phi)
    end = float(phi_end)
    if end < start:
        start, end = end, start
    delta = _section_delta_phi(float(source_phi), float(section_phi), section_period_value)
    first = float(source_phi) + delta
    while first < start - 1.0e-10:
        first += section_period_value
    targets: list[float] = []
    val = first
    while val <= end + 1.0e-10:
        targets.append(float(val))
        val += section_period_value
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


def _select_complete_section_crossings(
    R: Sequence[float],
    Z: Sequence[float],
    target_phi: Sequence[float],
    *,
    tol: float,
    expected_count: int,
    require_complete: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select one complete ordered tube crossing set for a section.

    Spatial de-duplication is intentionally limited to the closing endpoint of
    the same periodic orbit.  Distinct fixed points in a tube can be close on a
    section; dropping them by an all-pairs distance test changes the visible
    X/O count from section to section.
    """

    R_arr = np.asarray(R, dtype=float).ravel()
    Z_arr = np.asarray(Z, dtype=float).ravel()
    phi_arr = np.asarray(target_phi, dtype=float).ravel()
    expected = max(1, int(expected_count))
    finite = np.isfinite(R_arr) & np.isfinite(Z_arr) & np.isfinite(phi_arr)
    R_arr = R_arr[finite]
    Z_arr = Z_arr[finite]
    phi_arr = phi_arr[finite]
    source_index = np.flatnonzero(finite)
    if R_arr.size == 0:
        if require_complete:
            raise ValueError(
                f"section has no finite crossings; expected {expected} for a complete cycle"
            )
        return R_arr, Z_arr, phi_arr, source_index.astype(int)

    keep: list[int] = list(range(int(R_arr.size)))
    if len(keep) > expected:
        while len(keep) > expected:
            first = keep[0]
            last = keep[-1]
            if np.hypot(R_arr[first] - R_arr[last], Z_arr[first] - Z_arr[last]) <= float(tol):
                keep.pop()
                continue
            break

    if len(keep) < expected and require_complete:
        raise ValueError(
            f"section has {len(keep)} crossings after endpoint selection; expected {expected}"
        )
    if len(keep) > expected:
        if require_complete:
            raise ValueError(
                f"section has {len(keep)} crossings after endpoint selection; expected {expected}"
            )
        keep = keep[:expected]
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
    require_complete: bool = True,
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
    expected_count = int(base_cycle.cycle_orbit_size)
    effective_dedup_tol = float(section_dedup_tol)
    closure_residual = float(getattr(base_cycle, "closure_residual", np.inf))
    if np.isfinite(closure_residual) and closure_residual > 0.0:
        effective_dedup_tol = max(effective_dedup_tol, 1.05 * closure_residual)
    R_keep, Z_keep, phi_keep, source_index = _select_complete_section_crossings(
        R_raw[ok],
        Z_raw[ok],
        target_phi[ok],
        tol=effective_dedup_tol,
        expected_count=expected_count,
        require_complete=bool(require_complete),
    )
    if base_cycle.points:
        source_points = tuple(
            base_cycle.points[int(i) % len(base_cycle.points)]
            for i in source_index
        )
    else:
        source_points = ()
    point_indices = tuple(int(i) % max(1, int(base_cycle.cycle_orbit_size)) for i in source_index)
    section_cycle = _clone_section_cycle(
        base_cycle,
        section_phi=float(section_phi),
        section_index=int(section_index),
        delta_phi=_section_delta_phi(dense.source_phi, section_phi, section_period),
        R_values=R_keep,
        Z_values=Z_keep,
        source_points=source_points,
        point_indices=point_indices,
        alive=bool(R_keep.size == expected_count),
        map_count=int(R_keep.size),
    )
    metadata = dict(section_cycle.metadata)
    metadata.update({
        "section_cycle_source": "dense_orbit_crossings",
        "raw_crossing_count": int(np.count_nonzero(ok)),
        "dedup_crossing_count": int(R_keep.size),
        "expected_crossing_count": int(expected_count),
        "complete_crossing_count": bool(R_keep.size == expected_count),
        "section_dedup_tol": float(section_dedup_tol),
        "effective_section_dedup_tol": float(effective_dedup_tol),
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
            cycle_orbit_size=int(cycle.cycle_orbit_size),
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
            "map_order_index": int(point_index),
            "poincare_map_power": int(point_index),
            "section_local_index": int(local_i),
        })
        same_cycle_key = fp.metadata.get("same_cycle_key")
        if same_cycle_key is not None:
            fp.metadata["cycle_section_key"] = f"{same_cycle_key}:section={int(section_index)}"
            fp.metadata["cycle_section_point_key"] = (
                f"{same_cycle_key}:section={int(section_index)}:point={int(point_index)}"
            )
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
    map_power: int | None = None,
    DPhi: float = 0.01,
    wall_R: Sequence[float] | None = None,
    wall_Z: Sequence[float] | None = None,
    extend_phi: bool = True,
    n_threads: int = -1,
    deduplicate: bool = False,
    cycle_dedup_tol: float = 5.0e-4,
    start_cycle_id: int = 0,
    chain_id: int | None = None,
    winding: tuple[int, int] | None = None,
    reduced_winding: tuple[int, int] | None = None,
) -> tuple[BoundaryIslandCycle, ...]:
    """Trace ordered cycles from converged fixed points using ``P_span``.

    This is a batch wrapper around the existing cyna arbitrary-span map tracer.
    For each source point it stores the m distinct cycle points and summarizes
    the closing endpoint with ``closure_residual``.  Set ``deduplicate=True``
    to collapse equivalent cycles from different seeds and attach stable
    same-cycle metadata for cross-section plotting.
    """

    if not np.isfinite(float(map_span)) or abs(float(map_span)) <= 1.0e-14:
        raise ValueError("map_span must be a nonzero finite toroidal angle")
    if not fixed_points:
        return ()

    by_map_power_phi: dict[tuple[int, float], list[tuple[int, BoundaryIslandFixedPoint | FixedPoint]]] = {}
    for source_index, fp in enumerate(fixed_points):
        p = _fixed_point_map_power(fp, default=map_power)
        phi = float(getattr(fp, "phi", 0.0))
        by_map_power_phi.setdefault((p, phi), []).append((source_index, fp))

    cycles: list[BoundaryIslandCycle] = []
    for p, phi_start in sorted(by_map_power_phi):
        items = by_map_power_phi[(p, phi_start)]
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
                    cycle_orbit_size=int(p),
                    map_span=float(map_span),
                    closure_residual=float(closure),
                    source_index=source_index,
                )
                for i, (r, z) in enumerate(zip(point_R, point_Z))
            )
            cycles.append(BoundaryIslandCycle(
                points=points,
                cycle_orbit_size=int(p),
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
    raw_cycles = tuple(cycles)
    if not deduplicate:
        return raw_cycles
    return deduplicate_boundary_island_cycles(
        raw_cycles,
        cycle_dedup_tol=float(cycle_dedup_tol),
        start_cycle_id=int(start_cycle_id),
        chain_id=chain_id,
        winding=winding,
        reduced_winding=reduced_winding,
    )


def trace_fixed_point_cycle_span_field(
    field,
    fixed_point: BoundaryIslandFixedPoint | FixedPoint,
    *,
    map_span: float = 2.0 * np.pi,
    map_power: int | None = None,
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
        map_power=map_power,
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
    map_power: int | None = None,
    DPhi: float = 0.01,
    dphi_out: float | None = None,
    fd_eps: float = 1.0e-4,
    section_dedup_tol: float = 5.0e-4,
    require_complete_sections: bool = True,
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
            map_power=map_power,
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
            require_complete=bool(require_complete_sections),
        )
    return out


def _section_cycle_count_key(cycle: BoundaryIslandCycle) -> tuple[int | None, int | None, str, int]:
    return (
        cycle.chain_id,
        cycle.cycle_id,
        str(cycle.kind).upper(),
        int(cycle.source_index),
    )


def _validate_section_cycle_counts(
    section_cycles: dict[float, list[BoundaryIslandCycle]],
    section_phis: Sequence[float],
) -> dict[str, dict[str, int]]:
    counts: dict[tuple[int | None, int | None, str, int], dict[str, int]] = {}
    expected: dict[tuple[int | None, int | None, str, int], int] = {}
    for phi in section_phis:
        phi_key = float(phi)
        for cycle in section_cycles.get(phi_key, []):
            key = _section_cycle_count_key(cycle)
            count = int(len(cycle.points))
            expected_count = int(cycle.metadata.get("expected_crossing_count", cycle.cycle_orbit_size))
            if count != expected_count:
                raise ValueError(
                    "incomplete section cycle: "
                    f"phi={phi_key}, kind={cycle.kind}, cycle_id={cycle.cycle_id}, "
                    f"count={count}, expected={expected_count}"
                )
            if key in expected and expected[key] != count:
                raise ValueError(
                    "section cycle count changed across sections: "
                    f"kind={cycle.kind}, cycle_id={cycle.cycle_id}, "
                    f"previous={expected[key]}, current={count}, phi={phi_key}"
                )
            expected[key] = count
            counts.setdefault(key, {})[f"{phi_key:.17g}"] = count

    return {
        f"chain={key[0]} cycle={key[1]} kind={key[2]} source={key[3]}": value
        for key, value in counts.items()
    }


def trace_boundary_island_chain_sections_span_field(
    field,
    chain: BoundaryIslandChain,
    section_phis: Sequence[float],
    *,
    DPhi: float = 0.01,
    dphi_out: float | None = None,
    fd_eps: float = 1.0e-4,
    section_dedup_tol: float = 5.0e-4,
    require_complete_sections: bool = True,
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
            require_complete_sections=bool(require_complete_sections),
            wall_by_section=wall_by_section,
            extend_phi=extend_phi,
            n_threads=n_threads,
        )
        for phi, section_cycle in traced.items():
            section_cycles[float(phi)].append(section_cycle)

    count_by_cycle = (
        _validate_section_cycle_counts(section_cycles, section_phis)
        if require_complete_sections
        else {}
    )
    out: dict[float, BoundaryIslandChain] = {}
    for phi in section_phis:
        phi_key = float(phi)
        metadata = dict(chain.metadata)
        metadata.update({
            "section_phi": phi_key,
            "section_cycle_source": "orbit_trace",
            "section_cycle_count_by_cycle": count_by_cycle,
            "require_complete_sections": bool(require_complete_sections),
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
    map_power: int | None = None,
    DPhi: float = 0.01,
    dphi_out: float | None = None,
    extend_phi: bool = True,
    fd_eps: float = 1.0e-4,
    n_threads: int = -1,
) -> BoundaryIslandDenseCycle:
    """Trace the continuous 3-D geometry of one periodic cycle.

    The dense output follows one representative field line through the full
    ``cycle_orbit_size * map_span`` orbit.  The section fixed points remain attached in
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
            map_power=map_power,
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
    phi_end = source_phi + int(base_cycle.cycle_orbit_size) * float(base_cycle.map_span)
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
        cycle_orbit_size=int(base_cycle.cycle_orbit_size),
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
        cycle_orbit_size=int(chain.cycle_orbit_size),
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
    by_cycle_orbit_size: dict[int, list[BoundaryIslandCycle]] = {}
    for cycle in unique_cycles:
        by_cycle_orbit_size.setdefault(int(cycle.cycle_orbit_size), []).append(cycle)

    chains: list[BoundaryIslandChain] = []
    next_chain_id = 0
    next_cycle_id = 0
    for cycle_orbit_size_value in sorted(by_cycle_orbit_size):
        orbit_size_cycles = sorted(
            by_cycle_orbit_size[cycle_orbit_size_value],
            key=lambda c: (0 if str(c.kind).upper() == "O" else 1, _cycle_sort_angle(c)),
        )
        winding = (int(cycle_orbit_size_value if m is None else m), int(n))
        reduced = _reduced_winding(*winding)
        o_cycles = [c for c in orbit_size_cycles if str(c.kind).upper() == "O"]
        x_cycles = [c for c in orbit_size_cycles if str(c.kind).upper() == "X"]
        other_cycles = [c for c in orbit_size_cycles if str(c.kind).upper() not in {"O", "X"}]
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
                cycle_orbit_size=int(cycle_orbit_size_value),
                map_span=float(annotated[0].map_span if annotated else orbit_size_cycles[0].map_span),
                chain_id=int(next_chain_id),
                winding=winding,
                reduced_winding=reduced,
                metadata={
                    "cycle_orbit_size": int(cycle_orbit_size_value),
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
    map_power: int | None = None,
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
        map_power=map_power,
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


def _stable_unstable_eigenpairs(
    DPm: np.ndarray,
) -> tuple[float, np.ndarray, float, np.ndarray] | None:
    """Return real stable/unstable eigenvalues and unit eigenvectors."""

    mat = np.asarray(DPm, dtype=float).reshape(2, 2)
    if not np.all(np.isfinite(mat)):
        return None
    try:
        eigvals, eigvecs = np.linalg.eig(mat)
    except np.linalg.LinAlgError:
        return None
    if eigvals.size != 2:
        return None
    mod = np.abs(eigvals)
    order = np.argsort(mod)
    stable_idx = int(order[0])
    unstable_idx = int(order[-1])
    if not (mod[stable_idx] < 1.0 < mod[unstable_idx]):
        return None

    def _real_unit(idx: int) -> np.ndarray | None:
        vec = np.asarray(eigvecs[:, idx], dtype=complex)
        if np.max(np.abs(vec.imag)) > 1.0e-7 * max(1.0, float(np.max(np.abs(vec.real)))):
            return None
        real_vec = np.asarray(vec.real, dtype=float)
        norm = float(np.linalg.norm(real_vec))
        if not np.isfinite(norm) or norm <= 0.0:
            return None
        return real_vec / norm

    stable = _real_unit(stable_idx)
    unstable = _real_unit(unstable_idx)
    if stable is None or unstable is None:
        return None
    stable_eval = eigvals[stable_idx]
    unstable_eval = eigvals[unstable_idx]
    if abs(stable_eval.imag) > 1.0e-8 * max(1.0, abs(stable_eval.real)):
        return None
    if abs(unstable_eval.imag) > 1.0e-8 * max(1.0, abs(unstable_eval.real)):
        return None
    return float(stable_eval.real), stable, float(unstable_eval.real), unstable


def _manifold_seed_distances_from_expansion(
    expansion: float,
    *,
    eps_min: float,
    eps_max: float,
    n_eps: int,
) -> tuple[np.ndarray, float]:
    """Build one fundamental geometric seed segment for a manifold branch."""

    if n_eps <= 0:
        raise ValueError("n_eps must be positive")
    lam = float(abs(expansion))
    if not np.isfinite(lam) or lam <= 1.0:
        raise ValueError("manifold expansion must be finite and greater than 1")
    first = float(eps_min)
    if not np.isfinite(first) or first <= 0.0:
        raise ValueError("eps_min must be positive and finite")
    max_dist = float(eps_max)
    if not np.isfinite(max_dist) or max_dist <= 0.0:
        raise ValueError("eps_max must be positive and finite")

    ratio = lam if int(n_eps) == 1 else lam ** (1.0 / float(n_eps))
    last_factor = ratio ** max(0, int(n_eps) - 1)
    if first * last_factor > max_dist:
        first = max_dist / last_factor
    distances = first * ratio ** np.arange(int(n_eps), dtype=float)
    return distances, float(ratio)


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


def trace_poincare_sections_from_same_orbits_field(
    field,
    seed_R: Sequence[float],
    seed_Z: Sequence[float],
    phi_sections: Sequence[float],
    *,
    N_turns: int,
    DPhi: float,
    wall_R: Sequence[float] | None = None,
    wall_Z: Sequence[float] | None = None,
    extend_phi: bool = True,
    direction: str = "+",
) -> PoincareSectionTraces:
    """Trace one seed batch once and record all requested sections.

    This is the plotting-safe helper for multi-section Poincare backgrounds:
    every section is populated from the same physical seed orbits, not from
    independent per-section seed grids or separate map traces.
    """

    R0 = np.asarray(seed_R, dtype=float).ravel()
    Z0 = np.asarray(seed_Z, dtype=float).ravel()
    phi = np.asarray(phi_sections, dtype=float).ravel()
    if R0.size != Z0.size:
        raise ValueError("seed_R and seed_Z must have the same length")
    if R0.size == 0:
        raise ValueError("seed arrays must not be empty")
    if phi.size == 0:
        raise ValueError("phi_sections must not be empty")
    if int(N_turns) <= 0:
        raise ValueError("N_turns must be positive")
    if wall_R is None or wall_Z is None:
        wall_R, wall_Z = _field_grid_wall(field)
    counts, flat_R, flat_Z = trace_poincare_multi_batch_field(
        field,
        R0,
        Z0,
        phi,
        int(N_turns),
        float(DPhi),
        np.asarray(wall_R, dtype=float),
        np.asarray(wall_Z, dtype=float),
        extend_phi=extend_phi,
        direction=direction,
    )
    return PoincareSectionTraces(
        phi_sections=phi,
        seed_R=R0,
        seed_Z=Z0,
        counts=np.asarray(counts, dtype=int),
        R_flat=np.asarray(flat_R, dtype=float),
        Z_flat=np.asarray(flat_Z, dtype=float),
        N_turns=int(N_turns),
        direction=str(direction),
        metadata={
            "trace_source": "same_orbit_multi_section",
            "n_seed": int(R0.size),
            "n_section": int(phi.size),
            "N_turns": int(N_turns),
            "DPhi": float(DPhi),
        },
    )


def _trace_map_sequence_field(
    field,
    R0: float,
    Z0: float,
    phi_section: float,
    *,
    n_steps: int,
    map_span: float,
    DPhi: float,
    wall_R: Sequence[float] | None = None,
    wall_Z: Sequence[float] | None = None,
    extend_phi: bool = True,
    fd_eps: float = 1.0e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return section-equivalent map iterates for an arbitrary map span."""

    if n_steps <= 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    phi_end = float(phi_section) + float(n_steps) * float(map_span)
    try:
        R_t, Z_t, _phi_t, _DP_t, alive = trace_orbit_along_phi_field(
            field,
            float(R0),
            float(Z0),
            float(phi_section),
            phi_end,
            float(DPhi),
            extend_phi=extend_phi,
            dphi_out=abs(float(map_span)),
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
    map_span: float,
    DPhi: float,
    wall_R: Sequence[float] | None = None,
    wall_Z: Sequence[float] | None = None,
    extend_phi: bool = True,
    fd_eps: float = 1.0e-4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Trace arbitrary-span map points for multiple seeds."""

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
            float(map_span),
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
            map_span=float(map_span),
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
    map_power: int,
    map_span: float,
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
        n_steps=int(map_power),
        map_span=float(map_span),
        DPhi=float(DPhi),
        extend_phi=extend_phi,
        fd_eps=fd_eps,
    )
    if R_map.size < map_power:
        return None
    return float(R_map[-1]), float(Z_map[-1])


def _map_residual_field(
    field,
    x: np.ndarray,
    phi: float,
    map_power: int,
    map_span: float,
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
        int(map_power),
        float(map_span),
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
    map_power: int,
    map_span: float,
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
            int(map_power),
            float(map_span),
            float(DPhi),
            extend_phi=extend_phi,
            fd_eps=fd_eps,
        )
        minus = _map_endpoint_field(
            field,
            float(x[0] - step[0]),
            float(x[1] - step[1]),
            float(phi),
            int(map_power),
            float(map_span),
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


def _newton_fixed_point_map_power_field(
    field,
    seed_R: float,
    seed_Z: float,
    phi: float,
    map_power: int,
    map_span: float,
    DPhi: float,
    *,
    extend_phi: bool,
    fd_eps: float,
    max_iter: int,
    tol: float,
    trust_radius: float,
) -> tuple[float, float, float, bool, np.ndarray, np.ndarray, int]:
    """Damped Newton solve for arbitrary-span ``P^m`` maps."""

    x = np.asarray([float(seed_R), float(seed_Z)], dtype=float)
    best_res = np.inf
    for _it in range(int(max_iter)):
        F = _map_residual_field(
            field,
            x,
            float(phi),
            int(map_power),
            float(map_span),
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
                int(map_power),
                float(map_span),
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
                field, x + step, float(phi), int(map_power), float(map_span), float(DPhi),
                extend_phi=extend_phi, fd_eps=fd_eps,
            )
            Fm = _map_residual_field(
                field, x - step, float(phi), int(map_power), float(map_span), float(DPhi),
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
                int(map_power),
                float(map_span),
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


def boundary_wall_fractions(
    axis_R: float,
    axis_Z: float,
    R: Sequence[float],
    Z: Sequence[float],
    wall_R: Sequence[float],
    wall_Z: Sequence[float],
) -> np.ndarray:
    """Return each point's radial fraction of the wall radius from the axis.

    A value near 1 means the point lies close to the supplied wall along its
    ray from ``(axis_R, axis_Z)``.  ``nan`` is returned where the wall ray is
    degenerate or does not intersect the polygon.
    """

    R_arr = np.asarray(R, dtype=float).ravel()
    Z_arr = np.asarray(Z, dtype=float).ravel()
    if R_arr.size != Z_arr.size:
        raise ValueError("R and Z must have the same length")
    wall_R_arr = np.asarray(wall_R, dtype=float).ravel()
    wall_Z_arr = np.asarray(wall_Z, dtype=float).ravel()
    out = np.full(R_arr.shape, np.nan, dtype=float)
    for i, (r, z) in enumerate(zip(R_arr, Z_arr)):
        dR = float(r) - float(axis_R)
        dZ = float(z) - float(axis_Z)
        rho = float(np.hypot(dR, dZ))
        if not np.isfinite(rho):
            continue
        if rho <= 0.0:
            out[i] = 0.0
            continue
        wall_rho = _ray_polygon_radius(
            float(axis_R),
            float(axis_Z),
            float(np.arctan2(dZ, dR)),
            wall_R_arr,
            wall_Z_arr,
        )
        if np.isfinite(wall_rho) and wall_rho > 0.0:
            out[i] = rho / float(wall_rho)
    return out


def _fraction_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "min": None, "p05": None, "median": None, "p95": None, "max": None}
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


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
    map_powers: int | Iterable[int] = (2, 3, 4, 5, 6, 7, 8, 9, 10),
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
    field_period: float | None = None,
    DPhi: float = 0.01,
    fd_eps: float = 1.0e-4,
    recurrence_tol: float | None = None,
    candidate_wall_fraction_min: float | None = None,
    candidate_wall_fraction_max: float | None = None,
    candidate_order: str = "residual",
    candidates_per_map_power: int = 96,
    candidate_dedup_tol: float = 1.0e-3,
    extend_phi: bool = True,
    direction: str = "+",
) -> BoundaryIslandSeedCandidates:
    """Find Newton seeds from near-periodic returns of boundary Poincare traces.

    Candidate points are selected only from the discrete map residual
    ``|P^m(x)-x|`` along traced Poincare orbits.  No magnetic-field-strength
    extrema are used.
    """

    field_period_value = _resolve_field_period(field, field_period)
    map_power_values = _as_map_powers(map_powers)
    if N_turns <= max(map_power_values):
        raise ValueError("N_turns must be larger than the largest requested map_power")
    if candidates_per_map_power <= 0:
        raise ValueError("candidates_per_map_power must be positive")
    order = str(candidate_order).strip().lower().replace("-", "_")
    if order in {"edge", "outer_edge", "outer_residual", "wall_fraction"}:
        order = "outer"
    if order not in {"residual", "outer"}:
        raise ValueError("candidate_order must be 'residual' or 'outer'")
    if (
        candidate_wall_fraction_min is not None
        or candidate_wall_fraction_max is not None
        or order == "outer"
    ) and (wall_R is None or wall_Z is None):
        raise ValueError("wall_R and wall_Z are required for wall-fraction candidate selection")

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
    base_wall_fraction = None
    if wall_R is not None and wall_Z is not None:
        base_wall_fraction = boundary_wall_fractions(
            float(axis_R),
            float(axis_Z),
            base_R,
            base_Z,
            wall_R,
            wall_Z,
        )

    if np.isclose(float(field_period_value), 2.0 * np.pi, rtol=0.0, atol=1.0e-12):
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
            map_span=period_sign * abs(float(field_period_value)),
            DPhi=float(DPhi),
            wall_R=wall_R,
            wall_Z=wall_Z,
            extend_phi=extend_phi,
            fd_eps=fd_eps,
        )

    seeds_by_map_power: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    best_residual_by_map_power: dict[int, float] = {}
    raw_candidate_counts: dict[int, int] = {}
    accepted_counts: dict[int, int] = {}
    raw_wall_fraction_by_map_power: dict[int, dict[str, float | int | None]] = {}
    accepted_wall_fraction_by_map_power: dict[int, dict[str, float | int | None]] = {}
    trace_lengths = [int(len(r)) for r, _ in traces]

    for map_power in map_power_values:
        scored: list[tuple[float, float, float, int, int, float]] = []
        for seed_idx, (R_trace, Z_trace) in enumerate(traces):
            n = int(len(R_trace))
            if n <= map_power:
                continue
            d = np.hypot(
                R_trace[map_power:] - R_trace[:-map_power],
                Z_trace[map_power:] - Z_trace[:-map_power],
            )
            if d.size == 0:
                continue
            finite_idx = np.flatnonzero(np.isfinite(d))
            for k in finite_idx:
                residual = float(d[k])
                if recurrence_tol is not None and residual > recurrence_tol:
                    continue
                r = float(R_trace[k])
                z = float(Z_trace[k])
                wall_fraction = np.nan
                if wall_R is not None and wall_Z is not None:
                    wall_fraction = float(boundary_wall_fractions(
                        float(axis_R),
                        float(axis_Z),
                        [r],
                        [z],
                        wall_R,
                        wall_Z,
                    )[0])
                if candidate_wall_fraction_min is not None and (
                    not np.isfinite(wall_fraction)
                    or wall_fraction < float(candidate_wall_fraction_min)
                ):
                    continue
                if candidate_wall_fraction_max is not None and (
                    not np.isfinite(wall_fraction)
                    or wall_fraction > float(candidate_wall_fraction_max)
                ):
                    continue
                scored.append((residual, r, z, seed_idx, int(k), wall_fraction))

        raw_candidate_counts[int(map_power)] = int(len(scored))
        best_residual_by_map_power[int(map_power)] = (
            float(min(item[0] for item in scored)) if scored else np.inf
        )
        raw_wall_fraction_by_map_power[int(map_power)] = _fraction_stats([item[5] for item in scored])
        if order == "outer":
            scored.sort(key=lambda item: (
                not np.isfinite(item[5]),
                -item[5] if np.isfinite(item[5]) else 0.0,
                item[0],
            ))
        else:
            scored.sort(key=lambda item: (
                item[0],
                -item[5] if np.isfinite(item[5]) else 0.0,
            ))

        keep_R: list[float] = []
        keep_Z: list[float] = []
        for _residual, r, z, _seed_idx, _k, _wall_fraction in scored:
            keep_R.append(r)
            keep_Z.append(z)
            if len(keep_R) >= int(candidates_per_map_power) * 4:
                break
        cand_R, cand_Z = _deduplicate_seed_points(
            keep_R,
            keep_Z,
            tol=float(candidate_dedup_tol),
        )
        if cand_R.size > candidates_per_map_power:
            cand_R = cand_R[:candidates_per_map_power]
            cand_Z = cand_Z[:candidates_per_map_power]
        seeds_by_map_power[int(map_power)] = (cand_R, cand_Z)
        accepted_counts[int(map_power)] = int(cand_R.size)
        if wall_R is not None and wall_Z is not None:
            accepted_wall_fraction_by_map_power[int(map_power)] = _fraction_stats(
                boundary_wall_fractions(
                    float(axis_R),
                    float(axis_Z),
                    cand_R,
                    cand_Z,
                    wall_R,
                    wall_Z,
                )
            )
        else:
            accepted_wall_fraction_by_map_power[int(map_power)] = _fraction_stats([])

    return BoundaryIslandSeedCandidates(
        seeds_by_map_power=seeds_by_map_power,
        diagnostics={
            "map_powers": list(map_power_values),
            "n_base_seeds": int(base_R.size),
            "N_turns": int(N_turns),
            "field_period": float(field_period_value),
            "nfp": float(2.0 * np.pi / float(field_period_value)),
            "candidate_order": order,
            "candidate_wall_fraction_min": (
                None if candidate_wall_fraction_min is None else float(candidate_wall_fraction_min)
            ),
            "candidate_wall_fraction_max": (
                None if candidate_wall_fraction_max is None else float(candidate_wall_fraction_max)
            ),
            "base_seed_wall_fraction": (
                _fraction_stats(base_wall_fraction)
                if base_wall_fraction is not None
                else _fraction_stats([])
            ),
            "trace_length_min": int(min(trace_lengths)) if trace_lengths else 0,
            "trace_length_max": int(max(trace_lengths)) if trace_lengths else 0,
            "raw_candidate_counts": raw_candidate_counts,
            "accepted_counts": accepted_counts,
            "raw_candidate_wall_fraction_by_map_power": raw_wall_fraction_by_map_power,
            "accepted_candidate_wall_fraction_by_map_power": accepted_wall_fraction_by_map_power,
            "best_residual_by_map_power": best_residual_by_map_power,
        },
    )


def _lower_map_power_residual(
    field,
    R: float,
    Z: float,
    phi: float,
    map_power: int,
    field_period: float,
    DPhi: float,
    *,
    extend_phi: bool,
    fd_eps: float,
) -> float:
    if map_power <= 1:
        return np.inf
    divisors = [d for d in range(1, map_power) if map_power % d == 0]
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
                phi + float(field_period) * divisor,
                DPhi,
                extend_phi=extend_phi,
                dphi_out=abs(float(field_period) * divisor),
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
    for fp in sorted(fixed_points, key=lambda x: (x.map_power, x.kind, x.residual)):
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
        if (fp.map_power, fp.residual) < (old.map_power, old.residual):
            kept[duplicate_index] = fp
    return tuple(sorted(kept, key=lambda x: (x.map_power, x.kind, np.arctan2(x.Z, x.R))))


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
    map_powers: int | Iterable[int] = (2, 3, 4, 5, 6, 7, 8, 9, 10),
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
    recurrence_candidate_wall_fraction_min: float | None = None,
    recurrence_candidate_wall_fraction_max: float | None = None,
    recurrence_candidate_order: str = "residual",
    recurrence_candidates_per_map_power: int = 96,
    candidate_dedup_tol: float = 1.0e-3,
    field_period: float | None = None,
    python_trust_radius: float = 0.08,
    DPhi: float = 0.01,
    fd_eps: float = 1.0e-4,
    max_iter: int = 60,
    tol: float = 1.0e-10,
    residual_tol: float | None = None,
    dedup_tol: float = 2.0e-3,
    exclude_lower_map_powers: bool = True,
    lower_map_power_tol: float = 2.0e-3,
    extend_phi: bool = True,
    n_threads: int = -1,
) -> BoundaryIslandSearchResult:
    """Find X/O points of boundary island chains with a damped Newton backend."""

    field_period_value = _resolve_field_period(field, field_period)
    map_power_values = _as_map_powers(map_powers)
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
    seed_wall_fraction = None
    if wall_R is not None and wall_Z is not None:
        seed_wall_fraction = boundary_wall_fractions(
            float(axis_R),
            float(axis_Z),
            seed_R,
            seed_Z,
            wall_R,
            wall_Z,
        )

    recurrence_candidates: BoundaryIslandSeedCandidates | None = None
    if strategy in {"recurrence", "both"}:
        recurrence_candidates = boundary_recurrence_seed_candidates_field(
            field,
            float(axis_R),
            float(axis_Z),
            phi_section=float(phi_section),
            map_powers=map_power_values,
            seed_R=seed_R,
            seed_Z=seed_Z,
            wall_R=wall_R,
            wall_Z=wall_Z,
            wall_phi=wall_phi,
            wall_R_all=wall_R_all,
            wall_Z_all=wall_Z_all,
            N_turns=int(recurrence_turns),
            field_period=float(field_period_value),
            DPhi=float(DPhi),
            fd_eps=float(fd_eps),
            recurrence_tol=recurrence_tol,
            candidate_wall_fraction_min=recurrence_candidate_wall_fraction_min,
            candidate_wall_fraction_max=recurrence_candidate_wall_fraction_max,
            candidate_order=recurrence_candidate_order,
            candidates_per_map_power=int(recurrence_candidates_per_map_power),
            candidate_dedup_tol=float(candidate_dedup_tol),
            extend_phi=extend_phi,
        )

    if residual_tol is None:
        residual_tol = max(20.0 * float(tol), 1.0e-8)

    candidates: list[BoundaryIslandFixedPoint] = []
    raw_counts: dict[int, int] = {}
    converged_counts: dict[int, int] = {}
    accepted_counts: dict[int, int] = {}
    map_power_seed_counts: dict[int, int] = {}
    all_used_seed_R: list[np.ndarray] = []
    all_used_seed_Z: list[np.ndarray] = []

    for map_power in map_power_values:
        if strategy == "grid":
            map_power_seed_R, map_power_seed_Z = seed_R, seed_Z
        else:
            assert recurrence_candidates is not None
            rec_R, rec_Z = recurrence_candidates.seeds_for_map_power(int(map_power))
            if strategy == "recurrence":
                map_power_seed_R, map_power_seed_Z = rec_R, rec_Z
            else:
                map_power_seed_R = np.concatenate([seed_R, rec_R])
                map_power_seed_Z = np.concatenate([seed_Z, rec_Z])
                map_power_seed_R, map_power_seed_Z = _deduplicate_seed_points(
                    map_power_seed_R,
                    map_power_seed_Z,
                    tol=float(candidate_dedup_tol),
                )
        map_power_seed_R = np.asarray(map_power_seed_R, dtype=float).ravel()
        map_power_seed_Z = np.asarray(map_power_seed_Z, dtype=float).ravel()
        map_power_seed_counts[int(map_power)] = int(map_power_seed_R.size)
        if map_power_seed_R.size == 0:
            raw_counts[int(map_power)] = 0
            converged_counts[int(map_power)] = 0
            accepted_counts[int(map_power)] = 0
            continue
        all_used_seed_R.append(map_power_seed_R.copy())
        all_used_seed_Z.append(map_power_seed_Z.copy())
        map_power_seed_wall_fraction = None
        if wall_R is not None and wall_Z is not None:
            map_power_seed_wall_fraction = boundary_wall_fractions(
                float(axis_R),
                float(axis_Z),
                map_power_seed_R,
                map_power_seed_Z,
                wall_R,
                wall_Z,
            )

        if np.isclose(float(field_period_value), 2.0 * np.pi, rtol=0.0, atol=1.0e-12):
            result = find_fixed_points_batch_field(
                field,
                map_power_seed_R,
                map_power_seed_Z,
                float(phi_section),
                int(map_power),
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
                    map_power_seed_R,
                    map_power_seed_Z,
                    float(phi_section),
                    float(map_power) * float(field_period_value),
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
                    _newton_fixed_point_map_power_field(
                        field,
                        float(r),
                        float(z),
                        float(phi_section),
                        int(map_power),
                        float(field_period_value),
                        float(DPhi),
                        extend_phi=extend_phi,
                        fd_eps=float(fd_eps),
                        max_iter=int(max_iter),
                        tol=float(tol),
                        trust_radius=float(python_trust_radius),
                    )
                    for r, z in zip(map_power_seed_R, map_power_seed_Z)
                ]
                R_out = np.asarray([row[0] for row in rows], dtype=float)
                Z_out = np.asarray([row[1] for row in rows], dtype=float)
                residual = np.asarray([row[2] for row in rows], dtype=float)
                converged = np.asarray([row[3] for row in rows], dtype=bool)
                DPm_arr = np.asarray([row[4] for row in rows], dtype=float).reshape(len(rows), 2, 2)
                eig = np.asarray([row[5] for row in rows], dtype=complex)
                point_type = np.asarray([row[6] for row in rows], dtype=int)

        raw_counts[int(map_power)] = int(len(R_out))
        converged_counts[int(map_power)] = int(np.count_nonzero(converged))
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
            lower_res = _lower_map_power_residual(
                field,
                float(R_out[i]),
                float(Z_out[i]),
                float(phi_section),
                int(map_power),
                float(field_period_value),
                float(DPhi),
                extend_phi=extend_phi,
                fd_eps=fd_eps,
            )
            if exclude_lower_map_powers and lower_res <= lower_map_power_tol:
                continue
            kind = _classify_DPm_kind(DPm_arr[i], int(point_type[i]))
            fixed_wall_fraction = np.nan
            if wall_R is not None and wall_Z is not None:
                fixed_wall_fraction = float(boundary_wall_fractions(
                    float(axis_R),
                    float(axis_Z),
                    [float(R_out[i])],
                    [float(Z_out[i])],
                    wall_R,
                    wall_Z,
                )[0])
            seed_fraction = np.nan
            if map_power_seed_wall_fraction is not None and i < map_power_seed_wall_fraction.size:
                seed_fraction = float(map_power_seed_wall_fraction[i])
            candidates.append(BoundaryIslandFixedPoint(
                phi=float(phi_section),
                R=float(R_out[i]),
                Z=float(Z_out[i]),
                map_power=int(map_power),
                kind=kind,
                DPm=DPm_arr[i].copy(),
                residual=float(residual[i]),
                eigenvalues=np.asarray(eig[i]).copy(),
                seed_R=float(map_power_seed_R[i]),
                seed_Z=float(map_power_seed_Z[i]),
                lower_map_power_residual=float(lower_res),
                map_span=float(field_period_value),
                metadata={
                    "map_power": int(map_power),
                    "field_period": float(field_period_value),
                    "nfp": float(2.0 * np.pi / float(field_period_value)),
                    "map_span": float(field_period_value),
                    "base_map_span": float(field_period_value),
                    "monodromy_field_period": float(map_power) * float(field_period_value),
                    "monodromy_map_span": float(map_power) * float(field_period_value),
                    "wall_fraction": fixed_wall_fraction,
                    "seed_wall_fraction": seed_fraction,
                },
            ))
        accepted_counts[int(map_power)] = int(len(candidates) - accepted_before)

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
        "map_powers": list(map_power_values),
        "n_seeds": int(seed_R.size),
        "candidate_strategy": strategy,
        "field_period": float(field_period_value),
        "nfp": float(2.0 * np.pi / float(field_period_value)),
        "seed_wall_fraction": (
            _fraction_stats(seed_wall_fraction)
            if seed_wall_fraction is not None
            else _fraction_stats([])
        ),
        "fixed_point_wall_fraction": _fraction_stats([
            fp.metadata.get("wall_fraction", np.nan)
            for fp in fixed
        ]),
        "map_power_seed_counts": map_power_seed_counts,
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
    map_powers: int | Iterable[int] | None = None,
    shape_radius_fractions: Sequence[float] = (0.55, 0.78, 0.92),
    n_shape_angles: int = 6,
    N_turns: int = 80,
    field_period: float | None = None,
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
    field_period_value = _resolve_field_period(field, field_period)
    map_power_filter = None if map_powers is None else set(_as_map_powers(map_powers))
    fps = list(fixed_points)
    opts = [
        fp for fp in fps
        if str(getattr(fp, "kind", "")).upper() == "O"
        and (map_power_filter is None or _fixed_point_map_power(fp, default=1) in map_power_filter)
    ]
    xpts = [
        fp for fp in fps
        if str(getattr(fp, "kind", "")).upper() == "X"
        and (map_power_filter is None or _fixed_point_map_power(fp, default=1) in map_power_filter)
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
        same_map_power_x = [
            x for x in xpts
            if _fixed_point_map_power(x, default=1) == _fixed_point_map_power(opt, default=1)
        ]
        if same_map_power_x:
            distances = np.asarray([
                np.hypot(float(x.R) - float(opt.R), float(x.Z) - float(opt.Z))
                for x in same_map_power_x
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
            if np.isclose(float(field_period_value), 2.0 * np.pi, rtol=0.0, atol=1.0e-12):
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
                    map_span=float(field_period_value),
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
    field_period: float | None = None,
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
    include_arclength: bool = False,
    extend_phi: bool = True,
) -> list[dict[str, np.ndarray]]:
    """Trace W^u/W^s point clouds from hyperbolic fixed points.

    ``W^u`` is traced forward from the unstable eigendirection; ``W^s`` is
    traced backward from the stable eigendirection.  If ``field_period`` is not
    supplied, the traced span is inferred from the fixed point monodromy:
    ``map_power * field_period`` when that metadata is available.  Pass
    ``field_period`` explicitly to override this.
    """

    if N_turns <= 0:
        raise ValueError("N_turns must be positive")
    user_seed_distances = seed_distances is not None
    if not user_seed_distances:
        if n_eps <= 0:
            raise ValueError("n_eps must be positive")
        distances = None
    else:
        distances = np.asarray(seed_distances, dtype=float).ravel()
        distances = distances[np.isfinite(distances) & (distances > 0.0)]
        if distances.size == 0:
            raise ValueError("seed_distances must contain positive finite values")

    def _flatten_carousel(
        traces: list[tuple[np.ndarray, np.ndarray]],
        *,
        seed_R: np.ndarray,
        seed_Z: np.ndarray,
        seed_distance: np.ndarray,
        seed_side: np.ndarray,
        seed_order: np.ndarray,
        eigenvalue_sign: float,
        origin_R: float,
        origin_Z: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        seed_R = np.asarray(seed_R, dtype=float).ravel()
        seed_Z = np.asarray(seed_Z, dtype=float).ravel()
        seed_distance = np.asarray(seed_distance, dtype=float).ravel()
        seed_side = np.asarray(seed_side, dtype=float).ravel()
        seed_order = np.asarray(seed_order, dtype=int).ravel()
        if seed_R.size == 0:
            empty = np.empty(0, dtype=float)
            return empty, empty, empty, np.empty(0, dtype=int), np.empty(0, dtype=int), empty

        def _inside_limits(r: float, z: float) -> bool:
            if not (np.isfinite(r) and np.isfinite(z)):
                return False
            if RZlimit is None:
                return True
            r_min, r_max, z_min, z_max = map(float, RZlimit)
            return r_min <= r <= r_max and z_min <= z <= z_max

        R_out: list[float] = []
        Z_out: list[float] = []
        s_out: list[float] = []
        generation_out: list[int] = []
        seed_order_out: list[int] = []
        side_out: list[float] = []
        sign = -1.0 if float(eigenvalue_sign) < 0.0 else 1.0
        max_turns = 0
        for seed_i in range(seed_R.size):
            if int(seed_i) >= len(traces):
                continue
            R_trace = np.asarray(traces[int(seed_i)][0], dtype=float).ravel()
            Z_trace = np.asarray(traces[int(seed_i)][1], dtype=float).ravel()
            max_turns = max(max_turns, min(R_trace.size, Z_trace.size))

        for physical_side in (-1.0, 1.0):

            last_R = float(origin_R)
            last_Z = float(origin_Z)
            arc = 0.0

            def _append_point(seed_i: int, r: float, z: float, generation: int, side: float) -> bool:
                nonlocal arc, last_R, last_Z
                if not _inside_limits(r, z):
                    return False
                arc += float(np.hypot(r - last_R, z - last_Z))
                R_out.append(float(r))
                Z_out.append(float(z))
                s_out.append(arc)
                generation_out.append(int(generation))
                seed_order_out.append(int(seed_order[int(seed_i)]))
                side_out.append(float(side))
                last_R = float(r)
                last_Z = float(z)
                return True

            for generation in range(0, max_turns + 1):
                generation_sign = sign ** generation
                effective_side = seed_side * generation_sign
                side_idx = (
                    np.flatnonzero(effective_side < 0.0)
                    if physical_side < 0.0
                    else np.flatnonzero(effective_side > 0.0)
                )
                if side_idx.size == 0:
                    continue
                side_idx = side_idx[np.argsort(seed_distance[side_idx])]
                for seed_i_raw in side_idx:
                    seed_i = int(seed_i_raw)
                    if generation == 0:
                        _append_point(
                            seed_i,
                            float(seed_R[seed_i]),
                            float(seed_Z[seed_i]),
                            0,
                            physical_side,
                        )
                        continue
                    if seed_i >= len(traces):
                        continue
                    R_trace = np.asarray(traces[seed_i][0], dtype=float).ravel()
                    Z_trace = np.asarray(traces[seed_i][1], dtype=float).ravel()
                    trace_index = generation - 1
                    if trace_index >= min(R_trace.size, Z_trace.size):
                        continue
                    _append_point(
                        seed_i,
                        float(R_trace[trace_index]),
                        float(Z_trace[trace_index]),
                        generation,
                        physical_side,
                    )

        if not R_out:
            empty = np.empty(0, dtype=float)
            return empty, empty, empty, np.empty(0, dtype=int), np.empty(0, dtype=int), empty
        return (
            np.asarray(R_out, dtype=float),
            np.asarray(Z_out, dtype=float),
            np.asarray(s_out, dtype=float),
            np.asarray(generation_out, dtype=int),
            np.asarray(seed_order_out, dtype=int),
            np.asarray(side_out, dtype=float),
        )

    manifolds: list[dict[str, np.ndarray]] = []
    for fp in fixed_points:
        if str(getattr(fp, "kind", "")).upper() != "X":
            continue
        DPm = np.asarray(getattr(fp, "DPm", np.eye(2)), dtype=float).reshape(2, 2)
        eigpairs = _stable_unstable_eigenpairs(DPm)
        if eigpairs is None:
            continue
        stable_eigval, stable, unstable_eigval, unstable = eigpairs
        unstable_expansion = float(abs(unstable_eigval))
        stable_backward_expansion = np.inf if stable_eigval == 0.0 else float(1.0 / abs(stable_eigval))
        if not np.isfinite(stable_backward_expansion) or stable_backward_expansion <= 1.0:
            continue
        if user_seed_distances:
            u_distances = np.asarray(distances, dtype=float)
            s_distances = np.asarray(distances, dtype=float)
            u_seed_ratio = np.nan
            s_seed_ratio = np.nan
            seed_spacing = "user"
        else:
            u_distances, u_seed_ratio = _manifold_seed_distances_from_expansion(
                unstable_expansion,
                eps_min=float(eps_min),
                eps_max=float(eps_max),
                n_eps=int(n_eps),
            )
            s_distances, s_seed_ratio = _manifold_seed_distances_from_expansion(
                stable_backward_expansion,
                eps_min=float(eps_min),
                eps_max=float(eps_max),
                n_eps=int(n_eps),
            )
            seed_spacing = "eigenvalue_geometric"

        phi = float(getattr(fp, "phi", 0.0) if phi_section is None else phi_section)
        R0 = float(getattr(fp, "R"))
        Z0 = float(getattr(fp, "Z"))
        trace_map_span, trace_map_span_source = _fixed_point_monodromy_map_span(
            fp,
            explicit_field_period=field_period,
        )

        def _signed_seed_offsets(seed_distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            sign = np.asarray([-1.0, 1.0], dtype=float)
            offsets = (sign[:, None] * seed_distances[None, :]).ravel()
            order = np.broadcast_to(
                np.arange(seed_distances.size, dtype=int),
                (sign.size, seed_distances.size),
            ).ravel()
            return offsets, order

        du, u_seed_order = _signed_seed_offsets(u_distances)
        ds, s_seed_order = _signed_seed_offsets(s_distances)
        seed_u_R = R0 + du * unstable[0]
        seed_u_Z = Z0 + du * unstable[1]
        seed_s_R = R0 + ds * stable[0]
        seed_s_Z = Z0 + ds * stable[1]
        seed_u_distance = np.abs(du)
        seed_s_distance = np.abs(ds)
        seed_u_side = np.sign(du)
        seed_s_side = np.sign(ds)

        if wall_R is not None and wall_Z is not None:
            inside_u = _points_in_polygon(seed_u_R, seed_u_Z, np.asarray(wall_R), np.asarray(wall_Z))
            seed_u_R = seed_u_R[inside_u]
            seed_u_Z = seed_u_Z[inside_u]
            seed_u_distance = seed_u_distance[inside_u]
            seed_u_side = seed_u_side[inside_u]
            u_seed_order = u_seed_order[inside_u]
            inside_s = _points_in_polygon(seed_s_R, seed_s_Z, np.asarray(wall_R), np.asarray(wall_Z))
            seed_s_R = seed_s_R[inside_s]
            seed_s_Z = seed_s_Z[inside_s]
            seed_s_distance = seed_s_distance[inside_s]
            seed_s_side = seed_s_side[inside_s]
            s_seed_order = s_seed_order[inside_s]

        if np.isclose(abs(float(trace_map_span)), 2.0 * np.pi, rtol=0.0, atol=1.0e-12):
            forward_direction = "+" if trace_map_span > 0.0 else "-"
            backward_direction = "-" if trace_map_span > 0.0 else "+"
            u_traces = (
                _trace_poincare_points_field(
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
                    direction=forward_direction,
                )
                if seed_u_R.size
                else []
            )
            s_traces = (
                _trace_poincare_points_field(
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
                    direction=backward_direction,
                )
                if seed_s_R.size
                else []
            )
        else:
            u_traces = (
                _trace_map_points_field(
                    field,
                    seed_u_R,
                    seed_u_Z,
                    phi_section=phi,
                    N_turns=int(N_turns),
                    map_span=float(trace_map_span),
                    DPhi=float(DPhi),
                    wall_R=wall_R,
                    wall_Z=wall_Z,
                    extend_phi=extend_phi,
                    fd_eps=fd_eps,
                )
                if seed_u_R.size
                else []
            )
            s_traces = (
                _trace_map_points_field(
                    field,
                    seed_s_R,
                    seed_s_Z,
                    phi_section=phi,
                    N_turns=int(N_turns),
                    map_span=-float(trace_map_span),
                    DPhi=float(DPhi),
                    wall_R=wall_R,
                    wall_Z=wall_Z,
                    extend_phi=extend_phi,
                    fd_eps=fd_eps,
                )
                if seed_s_R.size
                else []
            )
        u_R, u_Z, u_lpol, u_generation, u_point_seed_order, u_point_side = _flatten_carousel(
            u_traces,
            seed_R=seed_u_R,
            seed_Z=seed_u_Z,
            seed_distance=seed_u_distance,
            seed_side=seed_u_side,
            seed_order=u_seed_order,
            eigenvalue_sign=np.sign(unstable_eigval),
            origin_R=R0,
            origin_Z=Z0,
        )
        s_R, s_Z, s_lpol, s_generation, s_point_seed_order, s_point_side = _flatten_carousel(
            s_traces,
            seed_R=seed_s_R,
            seed_Z=seed_s_Z,
            seed_distance=seed_s_distance,
            seed_side=seed_s_side,
            seed_order=s_seed_order,
            eigenvalue_sign=np.sign(stable_eigval),
            origin_R=R0,
            origin_Z=Z0,
        )
        metadata = dict(getattr(fp, "metadata", {}) or {})
        map_order_index = metadata.get(
            "map_order_index",
            metadata.get("orbit_point_index", metadata.get("point_index")),
        )
        cycle_id = metadata.get("cycle_id")
        if cycle_id is None:
            cycle_id = "?"
        payload = {
            "u_R": u_R,
            "u_Z": u_Z,
            "s_R": s_R,
            "s_Z": s_Z,
            "origin_R": R0,
            "origin_Z": Z0,
            "origin_phi": phi,
            "manifold_field_period": float(trace_map_span),
            "manifold_field_period_source": trace_map_span_source,
            "stable_eigenvector": stable.copy(),
            "unstable_eigenvector": unstable.copy(),
            "stable_eigenvalue": stable_eigval,
            "unstable_eigenvalue": unstable_eigval,
            "stable_orientation_reversing": bool(stable_eigval < 0.0),
            "unstable_orientation_reversing": bool(unstable_eigval < 0.0),
            "stable_backward_expansion": stable_backward_expansion,
            "unstable_expansion": unstable_expansion,
            "u_seed_R": seed_u_R.copy(),
            "u_seed_Z": seed_u_Z.copy(),
            "u_seed_distance": seed_u_distance.copy(),
            "u_seed_side": seed_u_side.copy(),
            "u_seed_order": u_seed_order.copy(),
            "u_seed_ratio": u_seed_ratio,
            "u_seed_next_distance": (
                np.nan if seed_u_distance.size == 0
                else unstable_expansion * float(np.min(seed_u_distance))
            ),
            "u_generation": u_generation,
            "u_point_seed_order": u_point_seed_order,
            "u_point_side": u_point_side,
            "s_seed_R": seed_s_R.copy(),
            "s_seed_Z": seed_s_Z.copy(),
            "s_seed_distance": seed_s_distance.copy(),
            "s_seed_side": seed_s_side.copy(),
            "s_seed_order": s_seed_order.copy(),
            "s_seed_ratio": s_seed_ratio,
            "s_seed_next_distance": (
                np.nan if seed_s_distance.size == 0
                else stable_backward_expansion * float(np.min(seed_s_distance))
            ),
            "s_generation": s_generation,
            "s_point_seed_order": s_point_seed_order,
            "s_point_side": s_point_side,
            "seed_spacing": seed_spacing,
            "seed_generation_order": "side_then_generation_then_distance",
            "seed_generation_semantics": (
                "each generation contains the full geometric seed segment; "
                "automatic spacing chooses q=lambda_expand^(1/n_eps) so "
                "lambda_expand*s1 is the next point after sN"
            ),
            "cycle_id": cycle_id,
            "chain_id": metadata.get("chain_id"),
            "point_index": metadata.get("point_index"),
            "orbit_point_index": metadata.get("orbit_point_index"),
            "map_order_index": map_order_index,
            "poincare_map_power": metadata.get("poincare_map_power", map_order_index),
            "same_cycle_key": metadata.get("same_cycle_key"),
            "cycle_point_key": metadata.get("cycle_point_key"),
            "cycle_section_key": metadata.get("cycle_section_key"),
            "cycle_section_point_key": metadata.get("cycle_section_point_key"),
            "manifold_origin_label": (
                None if map_order_index is None
                else f"{cycle_id}:P{int(map_order_index)}"
            ),
        }
        if include_arclength:
            payload.update({
                "u_lpol": u_lpol,
                "s_lpol": s_lpol,
                "arclength_coordinate": "poloidal_RZ_from_xpoint",
            })
        manifolds.append(payload)
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
    map_powers: int | Iterable[int] = (2, 3, 4, 5, 6, 7, 8, 9, 10),
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
        map_powers=map_powers,
        **(search_kwargs or {}),
    )
    edge_state_by_sec = trace_boundary_island_shapes_multi_section_field(
        field,
        search.fp_by_sec,
        axis_by_sec,
        phi_sections,
        wall_by_sec=wall_by_sec,
        map_powers=map_powers,
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
    "PoincareSectionTraces",
    "assemble_boundary_island_chains",
    "assemble_boundary_island_chains_field",
    "boundary_island_edge_state_payload",
    "boundary_island_topology_payload_field",
    "boundary_recurrence_seed_candidates_field",
    "boundary_seed_grid",
    "boundary_wall_fractions",
    "deduplicate_boundary_island_cycles",
    "fixed_points_by_section_payload",
    "find_boundary_island_fixed_points_field",
    "find_boundary_island_fixed_points_multi_section_field",
    "refine_fixed_points_monodromy_span_field",
    "trace_boundary_island_shapes_field",
    "trace_boundary_island_shapes_multi_section_field",
    "trace_boundary_island_chain_sections_span_field",
    "trace_boundary_island_chain_dense_span_field",
    "trace_poincare_sections_from_same_orbits_field",
    "trace_fixed_point_cycle_span_field",
    "trace_fixed_point_cycle_sections_span_field",
    "trace_fixed_point_cycle_dense_span_field",
    "trace_fixed_point_cycles_span_field",
    "trace_fixed_point_manifolds_field",
    "trace_fixed_point_manifolds_multi_section_field",
]
