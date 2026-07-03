"""Queryable and resumable trajectory datasets.

This module is intentionally independent from Prefect and from the memoize
cache.  It stores trajectory metadata in a small SQLite catalog and stores
large sampled arrays in append-only ``.npz`` segment files.  The format is
designed for selecting one trajectory, many trajectories, or trajectories
whose seeds satisfy a geometric predicate before extending them.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np


DATASET_SCHEMA_NAME = "pyna.trace_dataset"
DATASET_SCHEMA_VERSION = 1
DATASET_COMPATIBILITY = "v1 append-only; readers ignore unknown columns and fields"
SEGMENT_SCHEMA_NAME = "pyna.trace_dataset.segment"

_WALL_TERM_EVENTS = {
    1: "wall_hit",
    2: "field_grid_exit",
    3: "nonfinite_field",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_loads(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    return json.loads(str(value))


def _pyna_version() -> str | None:
    for package_name in ("pyna-chaos", "pyna"):
        try:
            return importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return None


def _canonical_hash(value: Any, *, prefix: str, length: int = 32) -> str:
    payload = _json_dumps(value).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:length]}"


def _array_summary(value: Any) -> dict[str, Any]:
    arr = np.asarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    if arr.dtype == object:
        h.update(repr(arr.tolist()).encode("utf-8"))
    else:
        h.update(np.ascontiguousarray(arr).tobytes())
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": h.hexdigest(),
    }


def _require_array_summary(
    row: sqlite3.Row,
    *,
    field: str,
    label: str,
    value: Any,
) -> None:
    expected = dict(_json_loads(row[field], {}) or {})
    actual = _array_summary(value)
    if expected != actual:
        raise ValueError(
            f"segment {row['segment_id']} {label} summary mismatch; "
            "the chunk may be stale or corrupted"
        )


def _require_segment_catalog_match(
    row: sqlite3.Row,
    *,
    independent: np.ndarray,
    states: np.ndarray,
) -> None:
    n_samples = int(row["n_samples"])
    ambient_dim = int(row["ambient_dim"])
    if independent.shape != (n_samples,):
        raise ValueError(f"segment {row['segment_id']} independent shape does not match catalog")
    if states.shape != (n_samples, ambient_dim):
        raise ValueError(f"segment {row['segment_id']} states shape does not match catalog")
    if not np.isclose(float(independent[0]), float(row["start_independent"]), rtol=0.0, atol=1.0e-14):
        raise ValueError(f"segment {row['segment_id']} start parameter does not match catalog")
    if not np.isclose(float(independent[-1]), float(row["end_independent"]), rtol=0.0, atol=1.0e-14):
        raise ValueError(f"segment {row['segment_id']} end parameter does not match catalog")


def _now() -> float:
    return float(time.time())


def _as_string_tuple(value: Sequence[str] | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(str(item) for item in value)


def _normalize_status(status: str | None) -> str:
    if status is None:
        return "extendable"
    status = str(status)
    if status == "running":
        return "incomplete"
    return status


def _point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    poly = np.asarray(polygon, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
        raise ValueError("seed_polygon must have shape (N, 2) with N >= 3")
    inside = False
    x0, y0 = poly[-1]
    for x1, y1 in poly:
        crosses = (y1 > y) != (y0 > y)
        if crosses:
            x_cross = (x0 - x1) * (y - y1) / ((y0 - y1) + 1.0e-300) + x1
            if x < x_cross:
                inside = not inside
        x0, y0 = x1, y1
    return bool(inside)


def _seed_value(seed: Any, names: Sequence[str] | None, coord: str) -> float | None:
    if isinstance(seed, dict):
        if coord not in seed:
            return None
        return float(seed[coord])
    values = list(seed)
    if names is None:
        if not str(coord).startswith("x"):
            return None
        try:
            idx = int(str(coord)[1:])
        except ValueError:
            return None
    else:
        try:
            idx = tuple(names).index(coord)
        except ValueError:
            return None
    if idx >= len(values):
        return None
    return float(values[idx])


def _seed_components(seed: Any, names: Sequence[str] | None) -> list[tuple[str, float | None]]:
    if seed is None:
        return []
    if isinstance(seed, dict):
        coord_names = [str(name) for name in names] if names is not None else sorted(str(name) for name in seed)
        out = []
        for name in coord_names[:3]:
            try:
                out.append((name, float(seed[name])))
            except (KeyError, TypeError, ValueError):
                out.append((name, None))
        return out
    try:
        values = list(seed)
    except TypeError:
        return []
    coord_names = [str(name) for name in names] if names is not None else [f"x{index}" for index in range(len(values))]
    out = []
    for name, value in list(zip(coord_names, values))[:3]:
        try:
            out.append((str(name), float(value)))
        except (TypeError, ValueError):
            out.append((str(name), None))
    return out


def _direction_suffix(direction: str) -> str:
    key = str(direction).strip().lower()
    if key in {"+", "plus", "forward", "fwd"}:
        return "plus"
    if key in {"-", "minus", "backward", "bwd"}:
        return "minus"
    raise ValueError(f"unknown direction {direction!r}")


def _direction_suffixes(directions: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(directions, str):
        if directions.strip().lower() == "both":
            return ("plus", "minus")
        return (_direction_suffix(directions),)
    seen = []
    for direction in directions:
        suffix = _direction_suffix(direction)
        if suffix not in seen:
            seen.append(suffix)
    return tuple(seen)


def _increment_counter(counter: dict[str, int], key: Any, amount: int = 1) -> None:
    label = "none" if key is None else str(key)
    counter[label] = int(counter.get(label, 0)) + int(amount)


def _signature_digest(value: Any) -> str:
    return hashlib.sha256(_json_dumps(value).encode("utf-8")).hexdigest()


def _short_signature(value: Any) -> dict[str, Any]:
    if value is None:
        return {"sha256": None}
    return {"sha256": _signature_digest(value), "summary": value}


def _compare_payload(expected: Any, actual: Any, *, field: str) -> dict[str, Any] | None:
    if expected == actual:
        return None
    return {
        "field": field,
        "expected": _short_signature(expected),
        "actual": _short_signature(actual),
    }


@dataclass(frozen=True)
class TraceTrajectoryRecord:
    """One trajectory row from a :class:`TraceDataset` catalog."""

    dataset: "TraceDataset"
    trajectory_id: str
    kind: str
    status: str
    seed: Any
    seed_coord_names: tuple[str, ...] | None
    direction: str | None
    independent_name: str
    coordinate_names: tuple[str, ...] | None
    settings: dict[str, Any]
    metadata: dict[str, Any]
    source_signature: Any
    n_segments: int
    n_samples: int
    ambient_dim: int | None
    start_independent: float | None
    end_independent: float | None
    last_state: np.ndarray | None

    @property
    def trace_id(self) -> str:
        """User-facing alias for ``trajectory_id``."""

        return self.trajectory_id

    @classmethod
    def from_row(cls, dataset: "TraceDataset", row: sqlite3.Row) -> "TraceTrajectoryRecord":
        last_state = _json_loads(row["last_state_json"], None)
        return cls(
            dataset=dataset,
            trajectory_id=str(row["trajectory_id"]),
            kind=str(row["kind"]),
            status=str(row["status"]),
            seed=_json_loads(row["seed_json"], None),
            seed_coord_names=(
                None
                if row["seed_coord_names_json"] is None
                else tuple(_json_loads(row["seed_coord_names_json"], []))
            ),
            direction=None if row["direction"] is None else str(row["direction"]),
            independent_name=str(row["independent_name"]),
            coordinate_names=(
                None
                if row["coordinate_names_json"] is None
                else tuple(_json_loads(row["coordinate_names_json"], []))
            ),
            settings=dict(_json_loads(row["settings_json"], {}) or {}),
            metadata=dict(_json_loads(row["metadata_json"], {}) or {}),
            source_signature=_json_loads(row["source_signature_json"], None),
            n_segments=int(row["n_segments"]),
            n_samples=int(row["n_samples"]),
            ambient_dim=None if row["ambient_dim"] is None else int(row["ambient_dim"]),
            start_independent=(
                None if row["start_independent"] is None else float(row["start_independent"])
            ),
            end_independent=(
                None if row["end_independent"] is None else float(row["end_independent"])
            ),
            last_state=None if last_state is None else np.asarray(last_state, dtype=float),
        )


TraceRecord = TraceTrajectoryRecord


class TraceSelection:
    """A concrete selection of trajectories, possibly across datasets."""

    def __init__(self, entries: Sequence[TraceTrajectoryRecord]):
        self._entries = list(entries)

    def __iter__(self):
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def ids(self) -> list[str]:
        return [entry.trajectory_id for entry in self._entries]

    def records(self) -> list[TraceTrajectoryRecord]:
        return list(self._entries)

    def summary(self, *, include_events: bool = False) -> dict[str, Any]:
        """Return a compact summary of the selected trajectories."""

        by_kind: dict[str, int] = {}
        by_status: dict[str, int] = {}
        by_direction: dict[str, int] = {}
        n_segments = 0
        n_samples = 0
        for entry in self._entries:
            _increment_counter(by_kind, entry.kind)
            _increment_counter(by_status, entry.status)
            _increment_counter(by_direction, entry.direction)
            n_segments += int(entry.n_segments)
            n_samples += int(entry.n_samples)
        out: dict[str, Any] = {
            "n_trajectories": len(self._entries),
            "by_kind": by_kind,
            "by_status": by_status,
            "by_direction": by_direction,
            "n_segments": int(n_segments),
            "n_samples": int(n_samples),
        }
        if include_events:
            event_counts: dict[str, int] = {}
            terminal_events = 0
            for entry in self._entries:
                for event in entry.dataset.events(entry.trajectory_id):
                    _increment_counter(event_counts, event["event_type"])
                    terminal_events += int(bool(event["terminal"]))
            out["events_by_type"] = event_counts
            out["terminal_events"] = int(terminal_events)
        return out

    def resume_fieldlines(self, field, *, target_phi: float, **kwargs: Any) -> list[TraceTrajectoryRecord]:
        """Extend selected field-line trajectories in their owning datasets."""

        out: list[TraceTrajectoryRecord] = []
        grouped: dict[TraceDataset, list[str]] = {}
        for entry in self._entries:
            grouped.setdefault(entry.dataset, []).append(entry.trajectory_id)
        for dataset, ids in grouped.items():
            out.extend(dataset.resume_fieldline_trajectories(field, target_phi=target_phi, trajectory_ids=ids, **kwargs))
        return out

    def trace_wall_hits(self, field, *, max_turns: int, **kwargs: Any) -> list[TraceTrajectoryRecord]:
        """Extend selected wall-hit trajectories in their owning datasets."""

        out: list[TraceTrajectoryRecord] = []
        grouped: dict[TraceDataset, list[str]] = {}
        for entry in self._entries:
            grouped.setdefault(entry.dataset, []).append(entry.trajectory_id)
        for dataset, ids in grouped.items():
            out.extend(dataset.trace_wall_hits_field(field, max_turns=max_turns, trajectory_ids=ids, **kwargs))
        return out

    def resume_trajectories(self, system, *, target_time: float, **kwargs: Any) -> list[TraceTrajectoryRecord]:
        """Extend selected finite-dimensional continuous trajectories."""

        out: list[TraceTrajectoryRecord] = []
        grouped: dict[TraceDataset, list[str]] = {}
        for entry in self._entries:
            grouped.setdefault(entry.dataset, []).append(entry.trajectory_id)
        for dataset, ids in grouped.items():
            out.extend(dataset.resume_trajectories(system, target_time=target_time, trajectory_ids=ids, **kwargs))
        return out

    def iterate_orbits(self, map_obj, *, target_step: int, **kwargs: Any) -> list[TraceTrajectoryRecord]:
        """Extend selected finite-dimensional discrete orbits."""

        out: list[TraceTrajectoryRecord] = []
        grouped: dict[TraceDataset, list[str]] = {}
        for entry in self._entries:
            grouped.setdefault(entry.dataset, []).append(entry.trajectory_id)
        for dataset, ids in grouped.items():
            out.extend(dataset.iterate_orbits(map_obj, target_step=target_step, trajectory_ids=ids, **kwargs))
        return out


class TraceDatasetCollection:
    """A selection facade over multiple :class:`TraceDataset` roots."""

    def __init__(self, datasets: Sequence["TraceDataset"]):
        self.datasets = list(datasets)

    @classmethod
    def open_many(cls, roots: Iterable[str | os.PathLike[str] | "TraceDataset"]) -> "TraceDatasetCollection":
        datasets = [root if isinstance(root, TraceDataset) else TraceDataset(root, create=False) for root in roots]
        return cls(datasets)

    def select(self, **kwargs: Any) -> TraceSelection:
        entries: list[TraceTrajectoryRecord] = []
        for dataset in self.datasets:
            entries.extend(dataset.select(**kwargs).records())
        return TraceSelection(entries)


class TraceDataset:
    """Append-only trajectory dataset with a queryable SQLite catalog."""

    catalog_name = "catalog.sqlite"

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        create: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root).expanduser()
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        elif not self.root.exists():
            raise FileNotFoundError(self.root)
        self.chunk_dir = self.root / "chunks"
        if create:
            self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.root / self.catalog_name
        self._conn = sqlite3.connect(str(self.catalog_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._init_schema(metadata=metadata or {})

    @classmethod
    def open_many(cls, roots: Iterable[str | os.PathLike[str] | "TraceDataset"]) -> TraceDatasetCollection:
        return TraceDatasetCollection.open_many(roots)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "TraceDataset":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def summary(
        self,
        *,
        kind: str | None = None,
        status: str | Sequence[str] | None = None,
        include_events: bool = True,
        include_storage: bool = True,
    ) -> dict[str, Any]:
        """Return user-facing counts for planning and progress checks."""

        selection = self.select(kind=kind, status=status)
        base = selection.summary(include_events=False)
        dataset_row = self._conn.execute("SELECT * FROM dataset WHERE id = 1").fetchone()
        out: dict[str, Any] = {
            "root": str(self.root),
            "schema_name": DATASET_SCHEMA_NAME if dataset_row is None else str(dataset_row["schema_name"]),
            "schema_version": DATASET_SCHEMA_VERSION if dataset_row is None else int(dataset_row["schema_version"]),
            "compatibility": DATASET_COMPATIBILITY if dataset_row is None else str(dataset_row["compatibility"]),
            **base,
        }
        if include_events:
            ids = selection.ids
            event_counts: dict[str, int] = {}
            terminal_events = 0
            if ids:
                sql = f"SELECT event_type, terminal FROM event WHERE trajectory_id IN ({','.join('?' for _ in ids)})"
                rows = self._conn.execute(sql, ids).fetchall()
            else:
                rows = []
            for row in rows:
                _increment_counter(event_counts, row["event_type"])
                terminal_events += int(bool(row["terminal"]))
            out["events_by_type"] = event_counts
            out["terminal_events"] = int(terminal_events)
        if include_storage:
            chunk_files = list(self.chunk_dir.rglob("*.npz")) if self.chunk_dir.exists() else []
            out["chunk_files"] = int(len(chunk_files))
            out["chunk_bytes"] = int(sum(path.stat().st_size for path in chunk_files if path.is_file()))
        return out

    def _init_schema(self, *, metadata: dict[str, Any]) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS dataset (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                schema_name TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                compatibility TEXT NOT NULL,
                pyna_version TEXT,
                metadata_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trajectory (
                trajectory_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                source_signature_json TEXT,
                identity_json TEXT NOT NULL,
                seed_json TEXT,
                seed_coord_names_json TEXT,
                seed0_name TEXT,
                seed0 REAL,
                seed1_name TEXT,
                seed1 REAL,
                seed2_name TEXT,
                seed2 REAL,
                direction TEXT,
                independent_name TEXT NOT NULL,
                coordinate_names_json TEXT,
                settings_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                status TEXT NOT NULL,
                n_segments INTEGER NOT NULL DEFAULT 0,
                n_samples INTEGER NOT NULL DEFAULT 0,
                ambient_dim INTEGER,
                start_independent REAL,
                end_independent REAL,
                last_state_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS segment (
                segment_id TEXT PRIMARY KEY,
                trajectory_id TEXT NOT NULL REFERENCES trajectory(trajectory_id) ON DELETE CASCADE,
                segment_index INTEGER NOT NULL,
                chunk_path TEXT NOT NULL,
                independent_name TEXT NOT NULL,
                start_independent REAL NOT NULL,
                end_independent REAL NOT NULL,
                n_samples INTEGER NOT NULL,
                ambient_dim INTEGER NOT NULL,
                states_summary_json TEXT NOT NULL,
                independent_summary_json TEXT NOT NULL,
                extra_summary_json TEXT NOT NULL,
                complete INTEGER NOT NULL,
                status TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                UNIQUE (trajectory_id, segment_index)
            );
            CREATE TABLE IF NOT EXISTS event (
                event_id TEXT PRIMARY KEY,
                trajectory_id TEXT NOT NULL REFERENCES trajectory(trajectory_id) ON DELETE CASCADE,
                event_index INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                independent_name TEXT NOT NULL,
                independent_value REAL,
                state_json TEXT,
                payload_json TEXT NOT NULL,
                terminal INTEGER NOT NULL,
                created_at REAL NOT NULL,
                UNIQUE (trajectory_id, event_index)
            );
            CREATE INDEX IF NOT EXISTS idx_trajectory_kind_status ON trajectory(kind, status);
            CREATE INDEX IF NOT EXISTS idx_trajectory_direction ON trajectory(direction);
            CREATE INDEX IF NOT EXISTS idx_segment_trajectory_index ON segment(trajectory_id, segment_index);
            CREATE INDEX IF NOT EXISTS idx_event_trajectory_index ON event(trajectory_id, event_index);
            CREATE INDEX IF NOT EXISTS idx_event_type ON event(event_type);
            """
        )
        self._ensure_trajectory_seed_columns()
        self._conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_trajectory_seed0 ON trajectory(seed0_name, seed0);
            CREATE INDEX IF NOT EXISTS idx_trajectory_seed1 ON trajectory(seed1_name, seed1);
            CREATE INDEX IF NOT EXISTS idx_trajectory_seed2 ON trajectory(seed2_name, seed2);
            """
        )
        row = self._conn.execute("SELECT schema_version FROM dataset WHERE id = 1").fetchone()
        now = _now()
        if row is None:
            self._conn.execute(
                """
                INSERT INTO dataset (
                    id, schema_name, schema_version, compatibility, pyna_version,
                    metadata_json, created_at, updated_at
                ) VALUES (1, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    DATASET_SCHEMA_NAME,
                    DATASET_SCHEMA_VERSION,
                    DATASET_COMPATIBILITY,
                    _pyna_version(),
                    _json_dumps(metadata),
                    now,
                    now,
                ),
            )
            self._conn.commit()
            return
        schema_version = int(row["schema_version"])
        if schema_version > DATASET_SCHEMA_VERSION:
            raise ValueError(
                "trace dataset was written by a newer schema "
                f"({schema_version}); this version supports {DATASET_SCHEMA_VERSION}"
            )

    def _ensure_trajectory_seed_columns(self) -> None:
        existing = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(trajectory)").fetchall()
        }
        specs = {
            "seed0_name": "TEXT",
            "seed0": "REAL",
            "seed1_name": "TEXT",
            "seed1": "REAL",
            "seed2_name": "TEXT",
            "seed2": "REAL",
        }
        for name, sql_type in specs.items():
            if name not in existing:
                self._conn.execute(f"ALTER TABLE trajectory ADD COLUMN {name} {sql_type}")
        self._conn.commit()

    def trajectory_identity(
        self,
        *,
        kind: str,
        source_signature: Any = None,
        seed: Any = None,
        seed_coord_names: Sequence[str] | None = None,
        direction: str | None = None,
        independent_name: str = "t",
        coordinate_names: Sequence[str] | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the stable identity payload for one trajectory.

        The requested end time/angle/iteration should not be included in
        ``settings``.  Identity is meant to describe the trajectory itself, not
        how far one particular run attempted to extend it.
        """

        return {
            "schema_name": f"{DATASET_SCHEMA_NAME}.trajectory_identity",
            "schema_version": 1,
            "kind": str(kind),
            "source_signature": source_signature,
            "seed": seed,
            "seed_coord_names": list(seed_coord_names) if seed_coord_names is not None else None,
            "direction": None if direction is None else str(direction),
            "independent_name": str(independent_name),
            "coordinate_names": list(coordinate_names) if coordinate_names is not None else None,
            "settings": dict(settings or {}),
        }

    def trajectory_id_for(self, **kwargs: Any) -> str:
        return _canonical_hash(self.trajectory_identity(**kwargs), prefix="traj")

    def ensure_trajectory(
        self,
        *,
        kind: str,
        source_signature: Any = None,
        seed: Any = None,
        seed_coord_names: Sequence[str] | None = None,
        direction: str | None = None,
        independent_name: str = "t",
        coordinate_names: Sequence[str] | None = None,
        settings: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        trajectory_id: str | None = None,
        status: str = "extendable",
    ) -> str:
        identity = self.trajectory_identity(
            kind=kind,
            source_signature=source_signature,
            seed=seed,
            seed_coord_names=seed_coord_names,
            direction=direction,
            independent_name=independent_name,
            coordinate_names=coordinate_names,
            settings=settings,
        )
        if trajectory_id is None:
            trajectory_id = _canonical_hash(identity, prefix="traj")
        components = _seed_components(seed, seed_coord_names)
        component_values: list[Any] = []
        for index in range(3):
            if index < len(components):
                name, value = components[index]
                component_values.extend([name, value])
            else:
                component_values.extend([None, None])
        now = _now()
        self._conn.execute(
            """
            INSERT INTO trajectory (
                trajectory_id, kind, source_signature_json, identity_json,
                seed_json, seed_coord_names_json,
                seed0_name, seed0, seed1_name, seed1, seed2_name, seed2,
                direction, independent_name,
                coordinate_names_json, settings_json, metadata_json, status,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trajectory_id) DO UPDATE SET
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                str(trajectory_id),
                str(kind),
                _json_dumps(source_signature),
                _json_dumps(identity),
                _json_dumps(seed),
                None if seed_coord_names is None else _json_dumps(list(seed_coord_names)),
                *component_values,
                None if direction is None else str(direction),
                str(independent_name),
                None if coordinate_names is None else _json_dumps(list(coordinate_names)),
                _json_dumps(settings or {}),
                _json_dumps(metadata or {}),
                _normalize_status(status),
                now,
                now,
            ),
        )
        self._conn.commit()
        return str(trajectory_id)

    def append_segment(
        self,
        trajectory_id: str,
        *,
        independent: Any,
        states: Any,
        independent_name: str | None = None,
        extra_arrays: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        status: str = "extendable",
        complete: bool = True,
    ) -> str:
        row = self._trajectory_row(trajectory_id)
        if row is None:
            raise KeyError(f"unknown trajectory_id {trajectory_id!r}")
        independent_arr = np.asarray(independent, dtype=float).reshape(-1)
        states_arr = np.asarray(states, dtype=float)
        if states_arr.ndim != 2:
            raise ValueError("states must have shape (N, d)")
        if states_arr.shape[0] != independent_arr.shape[0]:
            raise ValueError("independent and states lengths differ")
        if states_arr.shape[0] == 0:
            raise ValueError("cannot append an empty segment")
        independent_name = str(independent_name or row["independent_name"])
        segment_index = self._next_segment_index(trajectory_id)
        segment_id = _canonical_hash(
            {
                "trajectory_id": trajectory_id,
                "segment_index": segment_index,
                "start": float(independent_arr[0]),
                "end": float(independent_arr[-1]),
                "states": _array_summary(states_arr),
            },
            prefix="seg",
        )
        rel_chunk = Path("chunks") / str(trajectory_id) / f"{segment_index:08d}_{segment_id}.npz"
        abs_chunk = self.root / rel_chunk
        abs_chunk.parent.mkdir(parents=True, exist_ok=True)
        extra_arrays = {str(k): np.asarray(v) for k, v in dict(extra_arrays or {}).items()}
        for name, arr in extra_arrays.items():
            if arr.shape[0] != states_arr.shape[0]:
                raise ValueError(f"extra array {name!r} length does not match states")
        extra_summary = {name: _array_summary(arr) for name, arr in extra_arrays.items()}
        metadata = dict(metadata or {})
        arrays: dict[str, Any] = {
            "schema_name": np.asarray(SEGMENT_SCHEMA_NAME),
            "schema_version": np.asarray(DATASET_SCHEMA_VERSION, dtype=np.int64),
            "compatibility": np.asarray(DATASET_COMPATIBILITY),
            "complete": np.asarray(bool(complete)),
            "trajectory_id": np.asarray(str(trajectory_id)),
            "segment_id": np.asarray(segment_id),
            "segment_index": np.asarray(segment_index, dtype=np.int64),
            "independent_name": np.asarray(independent_name),
            "independent": independent_arr,
            "states": states_arr,
            "extra_keys_json": np.asarray(_json_dumps(sorted(extra_arrays))),
            "metadata_json": np.asarray(_json_dumps(metadata)),
        }
        for name, arr in extra_arrays.items():
            arrays[f"extra_{name}"] = arr

        tmp = abs_chunk.with_name(f".{abs_chunk.name}.tmp")
        try:
            np.savez_compressed(str(tmp), **arrays)
            actual_tmp = tmp if tmp.exists() else Path(str(tmp) + ".npz")
            actual_tmp.replace(abs_chunk)
        finally:
            if tmp.exists():
                tmp.unlink()
            tmp_npz = Path(str(tmp) + ".npz")
            if tmp_npz.exists():
                tmp_npz.unlink()

        now = _now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO segment (
                    segment_id, trajectory_id, segment_index, chunk_path,
                    independent_name, start_independent, end_independent,
                    n_samples, ambient_dim, states_summary_json,
                    independent_summary_json, extra_summary_json, complete,
                    status, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    segment_id,
                    str(trajectory_id),
                    int(segment_index),
                    rel_chunk.as_posix(),
                    independent_name,
                    float(independent_arr[0]),
                    float(independent_arr[-1]),
                    int(states_arr.shape[0]),
                    int(states_arr.shape[1]),
                    _json_dumps(_array_summary(states_arr)),
                    _json_dumps(_array_summary(independent_arr)),
                    _json_dumps(extra_summary),
                    1 if complete else 0,
                    _normalize_status(status),
                    _json_dumps(metadata),
                    now,
                ),
            )
            previous_samples = int(row["n_samples"])
            added_samples = int(states_arr.shape[0])
            if previous_samples > 0 and self._segment_continues_existing(row, independent_arr, states_arr):
                added_samples -= 1
            start_independent = row["start_independent"]
            if start_independent is None:
                start_independent = float(independent_arr[0])
            self._conn.execute(
                """
                UPDATE trajectory SET
                    status = ?,
                    n_segments = n_segments + 1,
                    n_samples = n_samples + ?,
                    ambient_dim = ?,
                    start_independent = ?,
                    end_independent = ?,
                    last_state_json = ?,
                    updated_at = ?
                WHERE trajectory_id = ?
                """,
                (
                    _normalize_status(status),
                    int(added_samples),
                    int(states_arr.shape[1]),
                    float(start_independent),
                    float(independent_arr[-1]),
                    _json_dumps(states_arr[-1].tolist()),
                    now,
                    str(trajectory_id),
                ),
            )
        return segment_id

    def append_event(
        self,
        trajectory_id: str,
        *,
        event_type: str,
        independent_name: str,
        independent_value: float | None = None,
        state: Any = None,
        payload: dict[str, Any] | None = None,
        terminal: bool = False,
    ) -> str:
        if self._trajectory_row(trajectory_id) is None:
            raise KeyError(f"unknown trajectory_id {trajectory_id!r}")
        event_index = self._next_event_index(trajectory_id)
        event_id = _canonical_hash(
            {
                "trajectory_id": trajectory_id,
                "event_index": event_index,
                "event_type": event_type,
                "independent_value": independent_value,
                "state": state,
            },
            prefix="event",
        )
        self._conn.execute(
            """
            INSERT INTO event (
                event_id, trajectory_id, event_index, event_type,
                independent_name, independent_value, state_json, payload_json,
                terminal, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                str(trajectory_id),
                int(event_index),
                str(event_type),
                str(independent_name),
                None if independent_value is None else float(independent_value),
                _json_dumps(state),
                _json_dumps(payload or {}),
                1 if terminal else 0,
                _now(),
            ),
        )
        self._conn.commit()
        return event_id

    def events(self, trajectory_id: str | None = None, *, event_type: str | None = None) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if trajectory_id is not None:
            clauses.append("trajectory_id = ?")
            params.append(str(trajectory_id))
        if event_type is not None:
            clauses.append("event_type = ?")
            params.append(str(event_type))
        sql = "SELECT * FROM event"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY trajectory_id, event_index"
        rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "event_id": str(row["event_id"]),
                "trajectory_id": str(row["trajectory_id"]),
                "event_index": int(row["event_index"]),
                "event_type": str(row["event_type"]),
                "independent_name": str(row["independent_name"]),
                "independent_value": (
                    None if row["independent_value"] is None else float(row["independent_value"])
                ),
                "state": _json_loads(row["state_json"], None),
                "payload": _json_loads(row["payload_json"], {}),
                "terminal": bool(row["terminal"]),
            }
            for row in rows
        ]

    def append_topo_trace(
        self,
        trace,
        *,
        kind: str | None = None,
        seed: Any = None,
        source_signature: Any = None,
        settings: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        status: str = "extendable",
        trajectory_id: str | None = None,
    ) -> str:
        from pyna.topo.core import Orbit, Trajectory

        if isinstance(trace, Trajectory):
            resolved_kind = "trajectory" if kind is None else str(kind)
            source_label = "pyna.topo.Trajectory"
        elif isinstance(trace, Orbit):
            resolved_kind = "orbit" if kind is None else str(kind)
            source_label = "pyna.topo.Orbit"
        else:
            raise TypeError("trace must be a pyna.topo Trajectory or Orbit")
        if resolved_kind not in {"trajectory", "orbit"}:
            raise ValueError("topo trace kind must be 'trajectory' or 'orbit'")

        trace_metadata = dict(getattr(trace, "metadata", {}) or {})
        if seed is None:
            seed = trace_metadata.get("seed", trace.initial.tolist())
        if source_signature is None:
            source_signature = trace_metadata.get("source_signature")
        if settings is None:
            settings = dict(trace_metadata.get("trace_settings", {}) or {})
        metadata = {**trace_metadata, **dict(metadata or {})}
        trajectory_id = self.ensure_trajectory(
            kind=resolved_kind,
            source_signature=source_signature,
            seed=seed,
            seed_coord_names=getattr(trace, "coordinate_names", None),
            independent_name=trace.independent_name,
            coordinate_names=getattr(trace, "coordinate_names", None),
            settings=settings or {},
            metadata=metadata,
            trajectory_id=trajectory_id,
            status=status,
        )
        self.append_segment(
            trajectory_id,
            independent=trace.independent,
            states=trace.states,
            metadata={"source": source_label},
            status=status,
            complete=True,
        )
        return trajectory_id

    def append_topo_trajectory(
        self,
        trajectory,
        *,
        seed: Any = None,
        source_signature: Any = None,
        settings: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        status: str = "extendable",
        trajectory_id: str | None = None,
    ) -> str:
        return self.append_topo_trace(
            trajectory,
            kind="trajectory",
            seed=seed,
            source_signature=source_signature,
            settings=settings,
            metadata=metadata,
            status=status,
            trajectory_id=trajectory_id,
        )

    def append_topo_orbit(
        self,
        orbit,
        *,
        seed: Any = None,
        source_signature: Any = None,
        settings: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        status: str = "extendable",
        trajectory_id: str | None = None,
    ) -> str:
        return self.append_topo_trace(
            orbit,
            kind="orbit",
            seed=seed,
            source_signature=source_signature,
            metadata=metadata,
            settings=settings,
            status=status,
            trajectory_id=trajectory_id,
        )

    def resume_trajectories(
        self,
        system,
        *,
        target_time: float,
        trajectory_ids: Sequence[str] | None = None,
        seed_polygon: Any = None,
        seed_coords: tuple[str, str] = ("x0", "x1"),
        dt: Any = None,
        t_eval: Any = None,
        segment_time_span: float | None = None,
        method_kwargs: dict[str, Any] | None = None,
        source_signature: Any | None = None,
        validate_source: bool = True,
    ) -> list[TraceTrajectoryRecord]:
        """Resume finite-dimensional continuous trajectories from their cursors.

        The system must expose ``trajectory(x0, t_span, **kwargs)`` or
        ``integrate(x0, t_span, **kwargs)`` and return a ``pyna.topo.Trajectory``.
        ``segment_time_span`` controls dataset-level dump cadence for long
        integrations.
        """

        if segment_time_span is not None:
            segment_time_span = abs(float(segment_time_span))
            if segment_time_span <= 0.0 or not np.isfinite(segment_time_span):
                raise ValueError("segment_time_span must be finite and positive")
            if t_eval is not None:
                raise ValueError("t_eval cannot be combined with segment_time_span")

        selection = self.select(
            trajectory_ids=trajectory_ids,
            kind="trajectory",
            seed_polygon=seed_polygon,
            seed_coords=seed_coords,
        )
        out: list[TraceTrajectoryRecord] = []
        for record in selection:
            if (
                validate_source
                and source_signature is not None
                and record.source_signature is not None
                and record.source_signature != source_signature
            ):
                raise ValueError(f"source signature does not match trajectory {record.trajectory_id}")
            if record.last_state is None or record.end_independent is None:
                raise ValueError(f"trajectory {record.trajectory_id} has no resumable cursor")
            current_time = float(record.end_independent)
            last_state = np.asarray(record.last_state, dtype=float)
            if np.isclose(current_time, float(target_time), rtol=0.0, atol=1.0e-14):
                out.append(record)
                continue
            while not np.isclose(current_time, float(target_time), rtol=0.0, atol=1.0e-14):
                segment_target = float(target_time)
                if segment_time_span is not None:
                    remaining = float(target_time) - current_time
                    step = math.copysign(min(abs(remaining), float(segment_time_span)), remaining)
                    segment_target = current_time + step
                trajectory = self._call_system_trajectory(
                    system,
                    last_state,
                    (current_time, segment_target),
                    dt=dt,
                    t_eval=t_eval,
                    method_kwargs=method_kwargs,
                )
                self.append_segment(
                    record.trajectory_id,
                    independent=np.asarray(trajectory.times, dtype=float),
                    states=np.asarray(trajectory.states, dtype=float),
                    metadata={
                        "source": "resume_trajectories",
                        "target_time": float(target_time),
                        "segment_target_time": float(segment_target),
                        "segment_time_span": None if segment_time_span is None else float(segment_time_span),
                    },
                    status="extendable",
                    complete=True,
                )
                next_time = float(np.asarray(trajectory.times, dtype=float)[-1])
                if np.isclose(next_time, current_time, rtol=0.0, atol=1.0e-14):
                    raise ValueError(f"trajectory {record.trajectory_id} did not advance from {current_time}")
                current_time = next_time
                last_state = np.asarray(trajectory.states[-1], dtype=float)
                del trajectory
                if segment_time_span is None:
                    break
            out.append(self.get(record.trajectory_id))
        return out

    def iterate_orbits(
        self,
        map_obj,
        *,
        target_step: int,
        trajectory_ids: Sequence[str] | None = None,
        seed_polygon: Any = None,
        seed_coords: tuple[str, str] = ("x0", "x1"),
        segment_steps: int | None = None,
        source_signature: Any | None = None,
        validate_source: bool = True,
    ) -> list[TraceTrajectoryRecord]:
        """Resume finite-dimensional map orbits from their last sampled state."""

        from pyna.topo.workflow import orbit_from_map

        target_step = int(target_step)
        if segment_steps is not None:
            segment_steps = int(segment_steps)
            if segment_steps <= 0:
                raise ValueError("segment_steps must be positive")
        selection = self.select(
            trajectory_ids=trajectory_ids,
            kind="orbit",
            seed_polygon=seed_polygon,
            seed_coords=seed_coords,
        )
        out: list[TraceTrajectoryRecord] = []
        for record in selection:
            if (
                validate_source
                and source_signature is not None
                and record.source_signature is not None
                and record.source_signature != source_signature
            ):
                raise ValueError(f"source signature does not match trajectory {record.trajectory_id}")
            if record.last_state is None or record.end_independent is None:
                raise ValueError(f"orbit {record.trajectory_id} has no resumable cursor")
            current_step = int(round(float(record.end_independent)))
            if target_step < current_step:
                raise ValueError(f"target_step is behind orbit {record.trajectory_id}")
            last_state = np.asarray(record.last_state, dtype=float)
            if target_step == current_step:
                out.append(record)
                continue
            while current_step < target_step:
                n_iter = target_step - current_step
                if segment_steps is not None:
                    n_iter = min(n_iter, int(segment_steps))
                orbit = orbit_from_map(map_obj, last_state, int(n_iter))
                local_steps = np.arange(orbit.n_samples, dtype=int) if orbit.steps is None else np.asarray(orbit.steps, dtype=int)
                independent = current_step + local_steps
                self.append_segment(
                    record.trajectory_id,
                    independent=independent,
                    states=np.asarray(orbit.states, dtype=float),
                    metadata={
                        "source": "iterate_orbits",
                        "target_step": int(target_step),
                        "segment_steps": None if segment_steps is None else int(segment_steps),
                    },
                    status="extendable",
                    complete=True,
                )
                next_step = int(independent[-1])
                if next_step <= current_step:
                    raise ValueError(f"orbit {record.trajectory_id} did not advance from step {current_step}")
                current_step = next_step
                last_state = np.asarray(orbit.states[-1], dtype=float)
                del orbit
                if segment_steps is None:
                    break
            out.append(self.get(record.trajectory_id))
        return out

    @staticmethod
    def _call_system_trajectory(
        system,
        x0,
        t_span: tuple[float, float],
        *,
        dt: Any = None,
        t_eval: Any = None,
        method_kwargs: dict[str, Any] | None = None,
    ):
        kwargs = dict(method_kwargs or {})
        if dt is not None:
            kwargs["dt"] = dt
        if t_eval is not None:
            kwargs["t_eval"] = t_eval
        for method_name in ("trajectory", "integrate"):
            method = getattr(system, method_name, None)
            if callable(method):
                return method(x0, t_span, **kwargs)
        raise TypeError(
            f"{type(system).__name__} does not expose callable trajectory(...) or integrate(...)"
        )

    def append_fieldline_trajectory(
        self,
        trajectory,
        *,
        field_signature: Any = None,
        settings: dict[str, Any] | None = None,
        direction: str | None = None,
        status: str | None = None,
        trajectory_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        meta = dict(getattr(trajectory, "metadata", {}) or {})
        if field_signature is None:
            field_signature = meta.get("field_signature")
        settings_payload = {
            "DPhi": meta.get("DPhi"),
            "dphi_out": meta.get("dphi_out"),
            "extend_phi": meta.get("extend_phi"),
            **dict(settings or {}),
        }
        direction = direction or ("backward" if float(meta.get("phi_end", 0.0)) < float(meta.get("phi_start", 0.0)) else "forward")
        seed = {
            "R": float(meta.get("R0", np.asarray(trajectory.R)[0])),
            "Z": float(meta.get("Z0", np.asarray(trajectory.Z)[0])),
            "phi": float(meta.get("phi_start", np.asarray(trajectory.phi)[0])),
        }
        status = _normalize_status(status or ("incomplete" if meta.get("status") == "incomplete" else "extendable"))
        trajectory_id = self.ensure_trajectory(
            kind="fieldline",
            source_signature=field_signature,
            seed=seed,
            seed_coord_names=("R", "Z", "phi"),
            direction=direction,
            independent_name="phi",
            coordinate_names=("R", "Z"),
            settings=settings_payload,
            metadata={**meta, **dict(metadata or {})},
            trajectory_id=trajectory_id,
            status=status,
        )
        alive = getattr(trajectory, "alive", None)
        self.append_segment(
            trajectory_id,
            independent=np.asarray(trajectory.phi, dtype=float),
            states=np.column_stack([np.asarray(trajectory.R, dtype=float), np.asarray(trajectory.Z, dtype=float)]),
            extra_arrays={} if alive is None else {"alive": np.asarray(alive, dtype=np.int8)},
            metadata={"source": "pyna.toroidal.flt.DenseFieldLineTrajectory"},
            status=status,
            complete=True,
        )
        return trajectory_id

    def resume_fieldline_trajectories(
        self,
        field,
        *,
        target_phi: float,
        trajectory_ids: Sequence[str] | None = None,
        seed_polygon: Any = None,
        seed_coords: tuple[str, str] = ("R", "Z"),
        DPhi: float | None = None,
        dphi_out: float | None = None,
        segment_phi_span: float | None = None,
        chunk_phi_span: float | None = None,
        storage: str = "memory",
        extend_phi: bool | None = None,
        stop_after_chunks: int | None = None,
        validate_field: bool = True,
    ) -> list[TraceTrajectoryRecord]:
        """Resume selected field-line trajectories up to ``target_phi``.

        ``segment_phi_span`` controls the outer dataset dump cadence.  When it
        is provided, a long resume is split into multiple trace calls and each
        completed segment is committed to disk before the next one starts.
        ``chunk_phi_span`` is passed through to the underlying tracer.
        """

        from pyna.toroidal.flt.trajectory import _field_signature, trace_fieldline_trajectory

        if segment_phi_span is not None:
            segment_phi_span = abs(float(segment_phi_span))
            if segment_phi_span <= 0.0 or not np.isfinite(segment_phi_span):
                raise ValueError("segment_phi_span must be finite and positive")

        field_signature = _field_signature(field)
        selection = self.select(
            trajectory_ids=trajectory_ids,
            kind="fieldline",
            seed_polygon=seed_polygon,
            seed_coords=seed_coords,
        )
        out: list[TraceTrajectoryRecord] = []
        for record in selection:
            if validate_field and record.source_signature is not None and record.source_signature != field_signature:
                raise ValueError(f"field signature does not match trajectory {record.trajectory_id}")
            if record.last_state is None or record.end_independent is None:
                raise ValueError(f"trajectory {record.trajectory_id} has no resumable cursor")
            current_phi = float(record.end_independent)
            if record.direction == "forward" and float(target_phi) < current_phi - 1.0e-14:
                raise ValueError(f"target_phi is behind forward trajectory {record.trajectory_id}")
            if record.direction == "backward" and float(target_phi) > current_phi + 1.0e-14:
                raise ValueError(f"target_phi is ahead of backward trajectory {record.trajectory_id}")
            if np.isclose(current_phi, float(target_phi), rtol=0.0, atol=1.0e-14):
                out.append(record)
                continue
            run_DPhi = DPhi if DPhi is not None else record.settings.get("DPhi")
            if run_DPhi is None:
                raise ValueError(f"trajectory {record.trajectory_id} lacks DPhi; pass DPhi explicitly")
            run_dphi_out = dphi_out if dphi_out is not None else record.settings.get("dphi_out")
            run_extend_phi = extend_phi if extend_phi is not None else record.settings.get("extend_phi", True)
            last_state = np.asarray(record.last_state, dtype=float)
            while not np.isclose(current_phi, float(target_phi), rtol=0.0, atol=1.0e-14):
                segment_target = float(target_phi)
                if segment_phi_span is not None:
                    remaining = float(target_phi) - current_phi
                    step = math.copysign(min(abs(remaining), float(segment_phi_span)), remaining)
                    segment_target = current_phi + step
                segment = trace_fieldline_trajectory(
                    field,
                    float(last_state[0]),
                    float(last_state[1]),
                    current_phi,
                    segment_target,
                    float(run_DPhi),
                    dphi_out=None if run_dphi_out is None else float(run_dphi_out),
                    chunk_phi_span=chunk_phi_span,
                    storage=storage,
                    extend_phi=bool(run_extend_phi),
                    stop_after_chunks=stop_after_chunks,
                )
                status = "extendable" if segment.status == "complete" else "incomplete"
                self.append_segment(
                    record.trajectory_id,
                    independent=np.asarray(segment.phi, dtype=float),
                    states=np.column_stack([np.asarray(segment.R, dtype=float), np.asarray(segment.Z, dtype=float)]),
                    extra_arrays={} if segment.alive is None else {"alive": np.asarray(segment.alive, dtype=np.int8)},
                    metadata={
                        "source": "resume_fieldline_trajectories",
                        "target_phi": float(target_phi),
                        "segment_target_phi": float(segment_target),
                        "segment_phi_span": None if segment_phi_span is None else float(segment_phi_span),
                        "field_signature": field_signature,
                    },
                    status=status,
                    complete=True,
                )
                current_phi = float(np.asarray(segment.phi, dtype=float)[-1])
                last_state = np.asarray([float(segment.R[-1]), float(segment.Z[-1])], dtype=float)
                del segment
                if status == "incomplete":
                    break
            out.append(self.get(record.trajectory_id))
        return out

    def trace_wall_hits_field(
        self,
        field,
        *,
        max_turns: int,
        wall_phi,
        wall_R_all=None,
        wall_Z_all=None,
        seed_R: Any = None,
        seed_Z: Any = None,
        phi_start: float | None = None,
        DPhi: float | None = None,
        directions: Sequence[str] | str = ("plus", "minus"),
        extend_phi: bool | None = None,
        batch_size: int | None = None,
        trajectory_ids: Sequence[str] | None = None,
        seed_polygon: Any = None,
        seed_coords: tuple[str, str] = ("R", "Z"),
        validate_field: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> list[TraceTrajectoryRecord]:
        """Trace or extend per-seed toroidal wall-hit attempts.

        Current cyna wall-hit tracing returns terminal summaries for a requested
        ``max_turns`` horizon.  It does not yet expose a cursor for true
        mid-orbit continuation, so extending an existing wall-hit trajectory
        selectively re-traces only the selected seeds/directions to the larger
        horizon.  Each attempt is recorded as an append-only segment and each
        nonzero termination code is recorded as an event.
        """

        from pyna.toroidal.flt.numba_poincare import trace_wall_hits_twall_field
        from pyna.toroidal.flt.postcompute import _field_cache_signature
        from pyna.toroidal.geometry import coerce_toroidal_surface_arrays

        max_turns = int(max_turns)
        if max_turns < 0:
            raise ValueError("max_turns must be non-negative")
        direction_suffixes = _direction_suffixes(directions)
        wall_phi, wall_R_all, wall_Z_all = coerce_toroidal_surface_arrays(wall_phi, wall_R_all, wall_Z_all)
        field_signature = _field_cache_signature(field)
        wall_signature = {
            "phi": _array_summary(wall_phi),
            "R": _array_summary(wall_R_all),
            "Z": _array_summary(wall_Z_all),
        }
        resolved_extend_phi = True if extend_phi is None else bool(extend_phi)

        ensured_ids: list[str] = []
        if seed_R is not None or seed_Z is not None:
            if seed_R is None or seed_Z is None:
                raise ValueError("seed_R and seed_Z must be supplied together")
            if phi_start is None or DPhi is None:
                raise ValueError("phi_start and DPhi are required when adding wall-hit seeds")
            R_seed = np.asarray(seed_R, dtype=float).reshape(-1)
            Z_seed = np.asarray(seed_Z, dtype=float).reshape(-1)
            if R_seed.shape != Z_seed.shape:
                raise ValueError("seed_R and seed_Z must have the same shape")
            for index, (R0, Z0) in enumerate(zip(R_seed, Z_seed)):
                for suffix in direction_suffixes:
                    ensured_ids.append(
                        self.ensure_trajectory(
                            kind="wall_hit",
                            source_signature=field_signature,
                            seed={"R": float(R0), "Z": float(Z0), "phi": float(phi_start)},
                            seed_coord_names=("R", "Z", "phi"),
                            direction=suffix,
                            independent_name="turn",
                            coordinate_names=("R", "Z", "phi"),
                            settings={
                                "DPhi": float(DPhi),
                                "extend_phi": bool(resolved_extend_phi),
                                "wall_signature": wall_signature,
                            },
                            metadata={
                                "seed_index": int(index),
                                "trace_source": "trace_wall_hits_twall_field",
                                **dict(metadata or {}),
                            },
                            status="extendable",
                        )
                    )
        if trajectory_ids is None and ensured_ids:
            trajectory_ids = ensured_ids

        selection = self.select(
            trajectory_ids=trajectory_ids,
            kind="wall_hit",
            seed_polygon=seed_polygon,
            seed_coords=seed_coords,
        )
        candidates = []
        for record in selection:
            if record.direction not in {"plus", "minus"}:
                continue
            if record.status == "terminal":
                continue
            if record.end_independent is not None and float(record.end_independent) >= float(max_turns):
                continue
            if validate_field and record.source_signature is not None and record.source_signature != field_signature:
                raise ValueError(f"field signature does not match trajectory {record.trajectory_id}")
            saved_wall_signature = record.settings.get("wall_signature")
            if saved_wall_signature is not None and saved_wall_signature != wall_signature:
                raise ValueError(f"wall signature does not match trajectory {record.trajectory_id}")
            record_DPhi = DPhi if DPhi is not None else record.settings.get("DPhi")
            if record_DPhi is None:
                raise ValueError(f"trajectory {record.trajectory_id} lacks DPhi; pass DPhi explicitly")
            if DPhi is not None and record.settings.get("DPhi") is not None:
                if not np.isclose(float(DPhi), float(record.settings["DPhi"]), rtol=0.0, atol=1.0e-14):
                    raise ValueError(f"DPhi does not match trajectory {record.trajectory_id}")
            record_phi_start = phi_start
            if record_phi_start is None:
                record_phi_start = _seed_value(record.seed, record.seed_coord_names, "phi")
            if record_phi_start is None:
                raise ValueError(f"trajectory {record.trajectory_id} lacks phi seed")
            R0 = _seed_value(record.seed, record.seed_coord_names, "R")
            Z0 = _seed_value(record.seed, record.seed_coord_names, "Z")
            if R0 is None or Z0 is None:
                raise ValueError(f"trajectory {record.trajectory_id} lacks R/Z seed")
            if extend_phi is not None and record.settings.get("extend_phi") is not None:
                if bool(extend_phi) != bool(record.settings["extend_phi"]):
                    raise ValueError(f"extend_phi does not match trajectory {record.trajectory_id}")
            candidates.append(
                {
                    "record": record,
                    "R": float(R0),
                    "Z": float(Z0),
                    "phi_start": float(record_phi_start),
                    "DPhi": float(record_DPhi),
                    "extend_phi": bool(record.settings.get("extend_phi", True) if extend_phi is None else extend_phi),
                }
            )

        if not candidates:
            return [self.get(record.trajectory_id) for record in selection]

        batch_size = len(candidates) if batch_size is None else max(1, int(batch_size))
        by_group: dict[tuple[float, float, bool], list[dict[str, Any]]] = {}
        for item in candidates:
            by_group.setdefault((item["phi_start"], item["DPhi"], item["extend_phi"]), []).append(item)

        for (group_phi_start, group_DPhi, group_extend_phi), items in by_group.items():
            unique: dict[tuple[float, float, float], list[dict[str, Any]]] = {}
            for item in items:
                unique.setdefault((item["R"], item["Z"], item["phi_start"]), []).append(item)
            unique_items = list(unique.items())
            for start in range(0, len(unique_items), batch_size):
                chunk = unique_items[start:start + batch_size]
                R_values = np.asarray([key[0] for key, _records in chunk], dtype=float)
                Z_values = np.asarray([key[1] for key, _records in chunk], dtype=float)
                hits = trace_wall_hits_twall_field(
                    field,
                    R_values,
                    Z_values,
                    float(group_phi_start),
                    int(max_turns),
                    float(group_DPhi),
                    wall_phi,
                    wall_R_all,
                    wall_Z_all,
                    extend_phi=bool(group_extend_phi),
                    direction="both",
                )
                for local_index, (_key, records) in enumerate(chunk):
                    for item in records:
                        record = item["record"]
                        suffix = str(record.direction)
                        term = int(np.asarray(hits[f"term_{suffix}"])[local_index])
                        hit = np.asarray(hits[f"hit_{suffix}"], dtype=float)[local_index]
                        Lc = float(np.asarray(hits[f"Lc_{suffix}"], dtype=float)[local_index])
                        event_type = _WALL_TERM_EVENTS.get(term)
                        status = "terminal" if event_type is not None else "extendable"
                        self.append_segment(
                            record.trajectory_id,
                            independent=np.asarray([record.end_independent or 0.0, float(max_turns)], dtype=float),
                            states=np.vstack(
                                [
                                    np.asarray([item["R"], item["Z"], item["phi_start"]], dtype=float),
                                    hit,
                                ]
                            ),
                            extra_arrays={
                                "term": np.asarray([term, term], dtype=np.int64),
                                "connection_length": np.asarray([np.nan, Lc], dtype=float),
                            },
                            metadata={
                                "source": "trace_wall_hits_field",
                                "max_turns": int(max_turns),
                                "term": int(term),
                                "connection_length": Lc,
                                "direction": suffix,
                                "wall_signature": wall_signature,
                            },
                            status=status,
                            complete=True,
                        )
                        if event_type is not None:
                            self.append_event(
                                record.trajectory_id,
                                event_type=event_type,
                                independent_name="turn",
                                independent_value=float(max_turns),
                                state=hit.tolist(),
                                payload={
                                    "term": int(term),
                                    "connection_length": Lc,
                                    "direction": suffix,
                                    "max_turns": int(max_turns),
                                },
                                terminal=True,
                            )
        return [self.get(record.trajectory_id) for record in selection]

    def explain_fieldline_request(
        self,
        trajectory_id: str,
        field,
        *,
        DPhi: float | None = None,
        dphi_out: float | None = None,
        extend_phi: bool | None = None,
        target_phi: float | None = None,
    ) -> dict[str, Any]:
        """Explain whether a field-line resume request matches a trajectory."""

        from pyna.toroidal.flt.trajectory import _field_signature

        record = self.get(trajectory_id)
        mismatches: list[dict[str, Any]] = []
        mismatch = _compare_payload(record.source_signature, _field_signature(field), field="field_signature")
        if mismatch is not None:
            mismatches.append(mismatch)
        if DPhi is not None and record.settings.get("DPhi") is not None:
            if not np.isclose(float(DPhi), float(record.settings["DPhi"]), rtol=0.0, atol=1.0e-14):
                mismatches.append({"field": "DPhi", "expected": record.settings["DPhi"], "actual": float(DPhi)})
        if dphi_out is not None and record.settings.get("dphi_out") is not None:
            if not np.isclose(float(dphi_out), float(record.settings["dphi_out"]), rtol=0.0, atol=1.0e-14):
                mismatches.append({"field": "dphi_out", "expected": record.settings["dphi_out"], "actual": float(dphi_out)})
        if extend_phi is not None and record.settings.get("extend_phi") is not None:
            if bool(extend_phi) != bool(record.settings["extend_phi"]):
                mismatches.append({"field": "extend_phi", "expected": bool(record.settings["extend_phi"]), "actual": bool(extend_phi)})
        if target_phi is not None and record.end_independent is not None:
            if record.direction == "forward" and float(target_phi) < float(record.end_independent) - 1.0e-14:
                mismatches.append(
                    {
                        "field": "target_phi",
                        "expected": f">= {record.end_independent} for forward trajectory",
                        "actual": float(target_phi),
                    }
                )
            if record.direction == "backward" and float(target_phi) > float(record.end_independent) + 1.0e-14:
                mismatches.append(
                    {
                        "field": "target_phi",
                        "expected": f"<= {record.end_independent} for backward trajectory",
                        "actual": float(target_phi),
                    }
                )
        return {
            "trajectory_id": record.trajectory_id,
            "kind": record.kind,
            "status": record.status,
            "matches": len(mismatches) == 0,
            "mismatches": mismatches,
            "current_end": record.end_independent,
            "settings": dict(record.settings),
        }

    def explain_wall_hit_request(
        self,
        trajectory_id: str,
        field,
        *,
        wall_phi,
        wall_R_all=None,
        wall_Z_all=None,
        DPhi: float | None = None,
        extend_phi: bool | None = None,
        max_turns: int | None = None,
    ) -> dict[str, Any]:
        """Explain whether a wall-hit extension request matches a trajectory."""

        from pyna.toroidal.flt.postcompute import _field_cache_signature
        from pyna.toroidal.geometry import coerce_toroidal_surface_arrays

        record = self.get(trajectory_id)
        wall_phi, wall_R_all, wall_Z_all = coerce_toroidal_surface_arrays(wall_phi, wall_R_all, wall_Z_all)
        wall_signature = {
            "phi": _array_summary(wall_phi),
            "R": _array_summary(wall_R_all),
            "Z": _array_summary(wall_Z_all),
        }
        mismatches: list[dict[str, Any]] = []
        mismatch = _compare_payload(record.source_signature, _field_cache_signature(field), field="field_signature")
        if mismatch is not None:
            mismatches.append(mismatch)
        saved_wall_signature = record.settings.get("wall_signature")
        if saved_wall_signature is not None:
            mismatch = _compare_payload(saved_wall_signature, wall_signature, field="wall_signature")
            if mismatch is not None:
                mismatches.append(mismatch)
        if DPhi is not None and record.settings.get("DPhi") is not None:
            if not np.isclose(float(DPhi), float(record.settings["DPhi"]), rtol=0.0, atol=1.0e-14):
                mismatches.append({"field": "DPhi", "expected": record.settings["DPhi"], "actual": float(DPhi)})
        if extend_phi is not None and record.settings.get("extend_phi") is not None:
            if bool(extend_phi) != bool(record.settings["extend_phi"]):
                mismatches.append({"field": "extend_phi", "expected": bool(record.settings["extend_phi"]), "actual": bool(extend_phi)})
        if max_turns is not None and record.end_independent is not None:
            if float(record.end_independent) >= float(max_turns):
                mismatches.append(
                    {
                        "field": "max_turns",
                        "expected": f"> {record.end_independent} to extend this trajectory",
                        "actual": int(max_turns),
                    }
                )
        return {
            "trajectory_id": record.trajectory_id,
            "kind": record.kind,
            "status": record.status,
            "matches": len(mismatches) == 0,
            "mismatches": mismatches,
            "current_end": record.end_independent,
            "settings": dict(record.settings),
        }

    def select(
        self,
        *,
        trajectory_ids: Sequence[str] | None = None,
        kind: str | None = None,
        status: str | Sequence[str] | None = None,
        direction: str | None = None,
        seed_polygon: Any = None,
        seed_coords: tuple[str, str] = ("R", "Z"),
        predicate: Callable[[TraceTrajectoryRecord], bool] | None = None,
    ) -> TraceSelection:
        clauses: list[str] = []
        params: list[Any] = []
        polygon = None
        if trajectory_ids is not None:
            ids = [str(item) for item in trajectory_ids]
            if not ids:
                return TraceSelection([])
            clauses.append(f"trajectory_id IN ({','.join('?' for _ in ids)})")
            params.extend(ids)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(str(kind))
        if status is not None:
            statuses = [str(status)] if isinstance(status, str) else [str(item) for item in status]
            if not statuses:
                return TraceSelection([])
            clauses.append(f"status IN ({','.join('?' for _ in statuses)})")
            params.extend(statuses)
        if direction is not None:
            clauses.append("direction = ?")
            params.append(str(direction))
        if seed_polygon is not None:
            polygon = np.asarray(seed_polygon, dtype=float)
            if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
                raise ValueError("seed_polygon must have shape (N, 2) with N >= 3")
            x_name, y_name = seed_coords
            x_min, y_min = np.min(polygon, axis=0)
            x_max, y_max = np.max(polygon, axis=0)

            def _bbox_clause(coord_name: str, low: float, high: float) -> str:
                params.extend([coord_name, float(low), float(high)] * 3)
                return (
                    "((seed0_name = ? AND seed0 BETWEEN ? AND ?) OR "
                    "(seed1_name = ? AND seed1 BETWEEN ? AND ?) OR "
                    "(seed2_name = ? AND seed2 BETWEEN ? AND ?))"
                )

            clauses.append(_bbox_clause(str(x_name), float(x_min), float(x_max)))
            clauses.append(_bbox_clause(str(y_name), float(y_min), float(y_max)))
        sql = "SELECT * FROM trajectory"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at, trajectory_id"
        rows = self._conn.execute(sql, params).fetchall()
        records = [TraceTrajectoryRecord.from_row(self, row) for row in rows]
        if polygon is not None:
            x_name, y_name = seed_coords
            filtered = []
            for record in records:
                x = _seed_value(record.seed, record.seed_coord_names, x_name)
                y = _seed_value(record.seed, record.seed_coord_names, y_name)
                if x is not None and y is not None and _point_in_polygon(x, y, polygon):
                    filtered.append(record)
            records = filtered
        if predicate is not None:
            records = [record for record in records if bool(predicate(record))]
        return TraceSelection(records)

    def get(self, trajectory_id: str) -> TraceTrajectoryRecord:
        row = self._trajectory_row(trajectory_id)
        if row is None:
            raise KeyError(trajectory_id)
        return TraceTrajectoryRecord.from_row(self, row)

    def _load_segment_row(self, row: sqlite3.Row) -> dict[str, Any]:
        path = self.root / str(row["chunk_path"])
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            schema_name = str(np.asarray(data["schema_name"]).item())
            if schema_name != SEGMENT_SCHEMA_NAME:
                raise ValueError(f"segment {row['segment_id']} has unknown schema {schema_name!r}")
            if "complete" in data.files and not bool(np.asarray(data["complete"]).item()):
                raise ValueError(f"segment {row['segment_id']} is not complete")
            chunk_trajectory_id = str(np.asarray(data["trajectory_id"]).item())
            if chunk_trajectory_id != str(row["trajectory_id"]):
                raise ValueError(f"segment {row['segment_id']} trajectory id does not match chunk")
            chunk_segment_id = str(np.asarray(data["segment_id"]).item())
            if chunk_segment_id != str(row["segment_id"]):
                raise ValueError(f"segment {row['segment_id']} id does not match chunk")
            chunk_segment_index = int(np.asarray(data["segment_index"]).item())
            if chunk_segment_index != int(row["segment_index"]):
                raise ValueError(f"segment {row['segment_id']} index does not match chunk")
            extra_keys = _json_loads(np.asarray(data["extra_keys_json"]).item(), [])
            independent_raw = np.asarray(data["independent"])
            states_raw = np.asarray(data["states"])
            independent = np.asarray(independent_raw, dtype=float)
            states = np.asarray(states_raw, dtype=float)
            _require_segment_catalog_match(row, independent=independent, states=states)
            _require_array_summary(
                row,
                field="independent_summary_json",
                label="independent",
                value=independent_raw,
            )
            _require_array_summary(row, field="states_summary_json", label="states", value=states_raw)
            expected_extra = dict(_json_loads(row["extra_summary_json"], {}) or {})
            if sorted(str(key) for key in extra_keys) != sorted(expected_extra):
                raise ValueError(f"segment {row['segment_id']} extra array keys do not match catalog")
            extra_arrays = {}
            for key in extra_keys:
                key = str(key)
                array_name = f"extra_{key}"
                if array_name not in data.files:
                    raise ValueError(f"segment {row['segment_id']} is missing extra array {key!r}")
                arr = np.asarray(data[array_name])
                expected_summary = expected_extra.get(key)
                if expected_summary != _array_summary(arr):
                    raise ValueError(
                        f"segment {row['segment_id']} extra array {key!r} summary mismatch; "
                        "the chunk may be stale or corrupted"
                    )
                extra_arrays[key] = arr
            return {
                "segment_id": str(row["segment_id"]),
                "segment_index": int(row["segment_index"]),
                "independent": independent,
                "states": states,
                "extra_arrays": extra_arrays,
                "metadata": _json_loads(np.asarray(data["metadata_json"]).item(), {}),
            }

    def load_segments(self, trajectory_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM segment WHERE trajectory_id = ? ORDER BY segment_index",
            (str(trajectory_id),),
        ).fetchall()
        return [self._load_segment_row(row) for row in rows]

    def validate_storage(self, *, check_orphans: bool = True) -> dict[str, Any]:
        """Validate catalog rows, chunk files, and trajectory aggregate cursors."""

        issues: list[dict[str, Any]] = []
        referenced_chunks: set[str] = set()
        failed_trajectories: set[str] = set()
        aggregates: dict[str, dict[str, Any]] = {}

        def add_issue(kind: str, message: str, **fields: Any) -> None:
            issues.append({"kind": kind, "message": message, **fields})

        rows = self._conn.execute(
            "SELECT * FROM segment ORDER BY trajectory_id, segment_index"
        ).fetchall()
        for row in rows:
            trajectory_id = str(row["trajectory_id"])
            referenced_chunks.add(str(row["chunk_path"]))
            try:
                segment = self._load_segment_row(row)
            except Exception as exc:
                failed_trajectories.add(trajectory_id)
                add_issue(
                    "segment_chunk_invalid",
                    str(exc),
                    trajectory_id=trajectory_id,
                    segment_id=str(row["segment_id"]),
                    segment_index=int(row["segment_index"]),
                    chunk_path=str(row["chunk_path"]),
                )
                continue

            independent = np.asarray(segment["independent"], dtype=float)
            states = np.asarray(segment["states"], dtype=float)
            aggregate = aggregates.setdefault(
                trajectory_id,
                {
                    "n_segments": 0,
                    "n_samples": 0,
                    "ambient_dim": int(states.shape[1]),
                    "start_independent": float(independent[0]),
                    "end_independent": None,
                    "last_state": None,
                },
            )
            added_samples = int(states.shape[0])
            previous_last = aggregate["last_state"]
            if (
                aggregate["n_segments"] > 0
                and aggregate["end_independent"] is not None
                and np.isclose(float(independent[0]), float(aggregate["end_independent"]), rtol=0.0, atol=1.0e-14)
                and previous_last is not None
                and np.asarray(previous_last).shape == states[0].shape
                and np.allclose(previous_last, states[0], rtol=0.0, atol=1.0e-12)
            ):
                added_samples -= 1
            aggregate["n_segments"] += 1
            aggregate["n_samples"] += int(added_samples)
            aggregate["ambient_dim"] = int(states.shape[1])
            aggregate["end_independent"] = float(independent[-1])
            aggregate["last_state"] = states[-1].tolist()

        trajectory_rows = self._conn.execute("SELECT * FROM trajectory ORDER BY trajectory_id").fetchall()
        for row in trajectory_rows:
            trajectory_id = str(row["trajectory_id"])
            if trajectory_id in failed_trajectories:
                continue
            aggregate = aggregates.get(
                trajectory_id,
                {
                    "n_segments": 0,
                    "n_samples": 0,
                    "ambient_dim": None,
                    "start_independent": None,
                    "end_independent": None,
                    "last_state": None,
                },
            )
            expected_pairs = (
                ("n_segments", int(row["n_segments"]), aggregate["n_segments"]),
                ("n_samples", int(row["n_samples"]), aggregate["n_samples"]),
            )
            for field, catalog_value, actual_value in expected_pairs:
                if catalog_value != actual_value:
                    add_issue(
                        "trajectory_aggregate_mismatch",
                        f"trajectory {trajectory_id} {field} does not match segments",
                        trajectory_id=trajectory_id,
                        field=field,
                        catalog=catalog_value,
                        actual=actual_value,
                    )
            if row["ambient_dim"] is not None and aggregate["ambient_dim"] != int(row["ambient_dim"]):
                add_issue(
                    "trajectory_aggregate_mismatch",
                    f"trajectory {trajectory_id} ambient_dim does not match segments",
                    trajectory_id=trajectory_id,
                    field="ambient_dim",
                    catalog=int(row["ambient_dim"]),
                    actual=aggregate["ambient_dim"],
                )
            for field in ("start_independent", "end_independent"):
                catalog_value = row[field]
                actual_value = aggregate[field]
                if catalog_value is None and actual_value is None:
                    continue
                if catalog_value is None or actual_value is None or not np.isclose(
                    float(catalog_value), float(actual_value), rtol=0.0, atol=1.0e-14
                ):
                    add_issue(
                        "trajectory_aggregate_mismatch",
                        f"trajectory {trajectory_id} {field} does not match segments",
                        trajectory_id=trajectory_id,
                        field=field,
                        catalog=None if catalog_value is None else float(catalog_value),
                        actual=actual_value,
                    )
            catalog_last = _json_loads(row["last_state_json"], None)
            actual_last = aggregate["last_state"]
            if catalog_last is not None or actual_last is not None:
                if catalog_last is None or actual_last is None or not np.allclose(
                    np.asarray(catalog_last, dtype=float),
                    np.asarray(actual_last, dtype=float),
                    rtol=0.0,
                    atol=1.0e-12,
                ):
                    add_issue(
                        "trajectory_aggregate_mismatch",
                        f"trajectory {trajectory_id} last_state does not match segments",
                        trajectory_id=trajectory_id,
                        field="last_state",
                        catalog=catalog_last,
                        actual=actual_last,
                    )

        chunk_files = []
        orphan_chunks = []
        if self.chunk_dir.exists():
            for path in self.chunk_dir.rglob("*.npz"):
                rel_path = path.relative_to(self.root).as_posix()
                chunk_files.append(rel_path)
                if check_orphans and rel_path not in referenced_chunks:
                    orphan_chunks.append(rel_path)
                    add_issue(
                        "orphan_chunk",
                        "chunk file is not referenced by the catalog",
                        chunk_path=rel_path,
                    )

        return {
            "ok": not issues,
            "issues": issues,
            "n_issues": int(len(issues)),
            "n_trajectories": int(len(trajectory_rows)),
            "n_segments": int(len(rows)),
            "n_chunks": int(len(chunk_files)),
            "n_orphan_chunks": int(len(orphan_chunks)),
        }

    def load_arrays(self, trajectory_id: str) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        segments = self.load_segments(trajectory_id)
        if not segments:
            raise ValueError(f"trajectory {trajectory_id!r} has no segments")
        independent_parts = []
        state_parts = []
        extra_parts: dict[str, list[np.ndarray]] = {}
        previous_end = None
        for segment in segments:
            independent = segment["independent"]
            states = segment["states"]
            start = 0
            if previous_end is not None and independent.size and np.isclose(
                float(independent[0]), float(previous_end), rtol=0.0, atol=1.0e-14
            ):
                start = 1
            independent_parts.append(independent[start:])
            state_parts.append(states[start:])
            for key, arr in segment["extra_arrays"].items():
                extra_parts.setdefault(key, []).append(np.asarray(arr)[start:])
            previous_end = float(independent[-1])
        independent_all = np.concatenate(independent_parts)
        states_all = np.vstack(state_parts)
        extras_all = {key: np.concatenate(parts) for key, parts in extra_parts.items()}
        return independent_all, states_all, extras_all

    def load_topo_trace(self, trajectory_id: str):
        from pyna.topo.core import Orbit, Trajectory

        record = self.get(trajectory_id)
        independent, states, _extras = self.load_arrays(trajectory_id)
        if record.kind == "orbit":
            return Orbit(
                states=states,
                steps=independent.astype(int),
                coordinate_names=record.coordinate_names,
                metadata={**record.metadata, "trace_dataset_id": trajectory_id},
            )
        return Trajectory(
            states=states,
            times=independent,
            time_name=record.independent_name,
            coordinate_names=record.coordinate_names,
            metadata={**record.metadata, "trace_dataset_id": trajectory_id},
        )

    def load_topo_trajectory(self, trajectory_id: str):
        return self.load_topo_trace(trajectory_id)

    def load_fieldline_trajectory(self, trajectory_id: str):
        from pyna.toroidal.flt.trajectory import DenseFieldLineTrajectory

        record = self.get(trajectory_id)
        if record.kind != "fieldline":
            raise ValueError(f"trajectory {trajectory_id!r} is not a fieldline")
        phi, states, extras = self.load_arrays(trajectory_id)
        alive = extras.get("alive")
        return DenseFieldLineTrajectory(
            phi=phi,
            R=states[:, 0],
            Z=states[:, 1],
            alive=alive,
            metadata={**record.metadata, "status": record.status, "trace_dataset_id": trajectory_id},
            storage_dir=str(self.root),
        )

    def _trajectory_row(self, trajectory_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM trajectory WHERE trajectory_id = ?",
            (str(trajectory_id),),
        ).fetchone()

    def _next_segment_index(self, trajectory_id: str) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(segment_index), -1) + 1 AS next_index FROM segment WHERE trajectory_id = ?",
            (str(trajectory_id),),
        ).fetchone()
        return int(row["next_index"])

    def _next_event_index(self, trajectory_id: str) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(event_index), -1) + 1 AS next_index FROM event WHERE trajectory_id = ?",
            (str(trajectory_id),),
        ).fetchone()
        return int(row["next_index"])

    @staticmethod
    def _segment_continues_existing(row: sqlite3.Row, independent: np.ndarray, states: np.ndarray) -> bool:
        if row["end_independent"] is None or row["last_state_json"] is None:
            return False
        previous_state = np.asarray(_json_loads(row["last_state_json"], []), dtype=float)
        return bool(
            np.isclose(float(independent[0]), float(row["end_independent"]), rtol=0.0, atol=1.0e-14)
            and previous_state.shape == states[0].shape
            and np.allclose(previous_state, states[0], rtol=0.0, atol=1.0e-12)
        )


__all__ = [
    "DATASET_COMPATIBILITY",
    "DATASET_SCHEMA_NAME",
    "DATASET_SCHEMA_VERSION",
    "SEGMENT_SCHEMA_NAME",
    "TraceDataset",
    "TraceDatasetCollection",
    "TraceRecord",
    "TraceSelection",
    "TraceTrajectoryRecord",
]
