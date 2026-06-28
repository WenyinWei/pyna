"""Post-compute views for toroidal field-line wall traces."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from prefect import flow, task
except ModuleNotFoundError:  # pragma: no cover - exercised only in slim envs
    def _identity_decorator(*args, **_kwargs):
        if args and callable(args[0]):
            return args[0]

        def _wrap(fn):
            return fn

        return _wrap

    flow = task = _identity_decorator

from pyna.toroidal.flt.numba_poincare import (
    strike_line_from_wall_hits,
    trace_wall_hits_twall_field,
)


_VIEW_ALIASES = {
    "+": "Lc_plus",
    "plus": "Lc_plus",
    "forward": "Lc_plus",
    "fwd": "Lc_plus",
    "Lc+": "Lc_plus",
    "Lc_plus": "Lc_plus",
    "-": "Lc_minus",
    "minus": "Lc_minus",
    "backward": "Lc_minus",
    "bwd": "Lc_minus",
    "Lc-": "Lc_minus",
    "Lc_minus": "Lc_minus",
    "sum": "Lc_sum",
    "total": "Lc_sum",
    "Lc_total": "Lc_sum",
    "Lc_sum": "Lc_sum",
    "max": "Lc_max",
    "maximum": "Lc_max",
    "Lc_max": "Lc_max",
    "min": "Lc_min",
    "minimum": "Lc_min",
    "Lc_min": "Lc_min",
}


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _array_digest(value) -> str:
    arr = np.asarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    if arr.dtype == object:
        h.update(repr(arr.tolist()).encode("utf-8"))
    else:
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _wall_trace_cache_signature(
    *,
    seed_R,
    seed_Z,
    phi_start: float,
    max_turns: int,
    DPhi: float,
    wall_phi,
    wall_R_all,
    wall_Z_all,
    extend_phi: bool,
) -> str:
    payload = {
        "seed_R": _array_digest(seed_R),
        "seed_Z": _array_digest(seed_Z),
        "phi_start": float(phi_start),
        "max_turns": int(max_turns),
        "DPhi": float(DPhi),
        "wall_phi": _array_digest(wall_phi),
        "wall_R_all": _array_digest(wall_R_all),
        "wall_Z_all": _array_digest(wall_Z_all),
        "extend_phi": bool(extend_phi),
    }
    encoded = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_connection_length_view(name: str) -> str:
    """Return the canonical connection-length view key."""

    key = str(name).strip()
    if key not in _VIEW_ALIASES:
        raise ValueError(f"unknown connection-length view: {name!r}")
    return _VIEW_ALIASES[key]


def connection_length_views_from_wall_hits(
    wall_hits: dict[str, np.ndarray],
    views: Sequence[str] = ("Lc_plus", "Lc_minus", "Lc_sum", "Lc_max"),
) -> dict[str, np.ndarray]:
    """Compute requested scalar Lc views from one bidirectional wall-hit trace."""

    plus = np.asarray(wall_hits["Lc_plus"], dtype=float)
    minus = np.asarray(wall_hits["Lc_minus"], dtype=float)
    base = {
        "Lc_plus": plus,
        "Lc_minus": minus,
        "Lc_sum": plus + minus,
        "Lc_max": np.maximum(plus, minus),
        "Lc_min": np.minimum(plus, minus),
    }
    return {normalize_connection_length_view(view): base[normalize_connection_length_view(view)].copy() for view in views}


@dataclass
class ToroidalWallTraceData:
    """Reusable raw toroidal-wall trace result plus post-compute helpers."""

    seed_R: np.ndarray
    seed_Z: np.ndarray
    phi_start: float
    max_turns: int
    DPhi: float
    wall_hits: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.seed_R = np.asarray(self.seed_R, dtype=float).ravel()
        self.seed_Z = np.asarray(self.seed_Z, dtype=float).ravel()
        if self.seed_R.size != self.seed_Z.size:
            raise ValueError("seed_R and seed_Z must have the same length")
        self.phi_start = float(self.phi_start)
        self.max_turns = int(self.max_turns)
        self.DPhi = float(self.DPhi)
        self.wall_hits = {key: np.asarray(value).copy() for key, value in self.wall_hits.items()}
        self.metadata = dict(self.metadata)

    @property
    def n_seed(self) -> int:
        return int(self.seed_R.size)

    def view(self, name: str) -> np.ndarray:
        """Return one scalar connection-length view without retracing."""

        return connection_length_views_from_wall_hits(self.wall_hits, (name,))[normalize_connection_length_view(name)]

    def views(self, names: Sequence[str]) -> dict[str, np.ndarray]:
        """Return multiple scalar connection-length views without retracing."""

        return connection_length_views_from_wall_hits(self.wall_hits, names)

    def strike(self, *, direction: str = "+", wall_term: int = 1) -> dict[str, np.ndarray]:
        """Return strike points derived from cached wall-hit data."""

        return strike_line_from_wall_hits(self.wall_hits, direction=direction, wall_term=int(wall_term))

    def save_npz(self, path: str | Path) -> None:
        """Persist this raw trace and metadata for later post-compute views."""

        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "seed_R": np.asarray(self.seed_R, dtype=float),
            "seed_Z": np.asarray(self.seed_Z, dtype=float),
            "phi_start": np.asarray(float(self.phi_start)),
            "max_turns": np.asarray(int(self.max_turns), dtype=int),
            "DPhi": np.asarray(float(self.DPhi)),
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True, default=_json_default)),
        }
        for key, value in self.wall_hits.items():
            arrays[f"wall_hits_{key}"] = np.asarray(value)
        np.savez(str(out), **arrays)

    def save_views_npz(
        self,
        path: str | Path,
        views: Sequence[str] = ("Lc_plus", "Lc_minus", "Lc_sum", "Lc_max"),
    ) -> dict[str, np.ndarray]:
        """Save derived connection-length views without retracing."""

        derived = self.views(tuple(views))
        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(out), **derived)
        return derived

    @classmethod
    def load_npz(cls, path: str | Path) -> "ToroidalWallTraceData":
        """Load a cached raw wall trace from :meth:`save_npz` output."""

        with np.load(str(Path(path).expanduser()), allow_pickle=False) as data:
            metadata_raw = str(np.asarray(data["metadata_json"]).item())
            wall_hits = {
                key[len("wall_hits_"):]: np.asarray(data[key])
                for key in data.files
                if key.startswith("wall_hits_")
            }
            seed_R = np.asarray(data["seed_R"], dtype=float)
            seed_Z = np.asarray(data["seed_Z"], dtype=float)
            phi_start = float(np.asarray(data["phi_start"]).item())
            max_turns = int(np.asarray(data["max_turns"]).item())
            DPhi = float(np.asarray(data["DPhi"]).item())
        return cls(
            seed_R=seed_R,
            seed_Z=seed_Z,
            phi_start=phi_start,
            max_turns=max_turns,
            DPhi=DPhi,
            wall_hits=wall_hits,
            metadata=json.loads(metadata_raw) if metadata_raw else {},
        )


def trace_toroidal_wall_data_field(
    field,
    seed_R,
    seed_Z,
    phi_start: float,
    max_turns: int,
    DPhi: float,
    wall_phi,
    wall_R_all,
    wall_Z_all,
    *,
    extend_phi: bool = True,
    cache_path: str | Path | None = None,
    overwrite: bool = False,
    validate_cache: bool = True,
    metadata: dict[str, Any] | None = None,
) -> ToroidalWallTraceData:
    """Trace bidirectional toroidal-wall hits once, optionally caching raw data."""

    signature = _wall_trace_cache_signature(
        seed_R=seed_R,
        seed_Z=seed_Z,
        phi_start=float(phi_start),
        max_turns=int(max_turns),
        DPhi=float(DPhi),
        wall_phi=wall_phi,
        wall_R_all=wall_R_all,
        wall_Z_all=wall_Z_all,
        extend_phi=bool(extend_phi),
    )
    if cache_path is not None:
        path = Path(cache_path).expanduser()
        if path.exists() and not overwrite:
            cached = ToroidalWallTraceData.load_npz(path)
            cached_signature = cached.metadata.get("cache_signature")
            if validate_cache and cached_signature is not None and cached_signature != signature:
                raise ValueError(
                    "cached toroidal wall trace does not match requested inputs; "
                    "use a different cache_path or pass overwrite=True"
                )
            return cached
    else:
        path = None

    hits = trace_wall_hits_twall_field(
        field,
        seed_R,
        seed_Z,
        float(phi_start),
        int(max_turns),
        float(DPhi),
        wall_phi,
        wall_R_all,
        wall_Z_all,
        extend_phi=extend_phi,
        direction="both",
    )
    payload = ToroidalWallTraceData(
        seed_R=np.asarray(seed_R, dtype=float),
        seed_Z=np.asarray(seed_Z, dtype=float),
        phi_start=float(phi_start),
        max_turns=int(max_turns),
        DPhi=float(DPhi),
        wall_hits=hits,
        metadata={
            **(metadata or {}),
            "trace_source": "trace_wall_hits_twall_field",
            "post_compute_reusable": True,
            "cache_signature": signature,
        },
    )
    if path is not None:
        payload.save_npz(path)
    return payload


@task
def wall_trace_post_compute_task(
    cache_path: str,
    views: Sequence[str],
    out_path: str | None = None,
) -> dict[str, np.ndarray]:
    """Prefect task: load one cached raw trace and derive scalar Lc views."""

    data = ToroidalWallTraceData.load_npz(cache_path)
    if out_path is not None:
        return data.save_views_npz(out_path, tuple(views))
    return data.views(tuple(views))


@flow(name="pyna-wall-trace-post-compute")
def wall_trace_post_compute_flow(
    cache_path: str,
    views: Sequence[str] = ("Lc_plus", "Lc_minus", "Lc_sum", "Lc_max"),
    out_path: str | None = None,
):
    """Prefect flow for post-compute views from cached wall-hit data."""

    return wall_trace_post_compute_task(cache_path, tuple(views), out_path)


def build_prefect_wall_post_compute_flow():
    """Return the canonical Prefect flow for wall-trace post-compute views."""

    return wall_trace_post_compute_flow


__all__ = [
    "ToroidalWallTraceData",
    "build_prefect_wall_post_compute_flow",
    "connection_length_views_from_wall_hits",
    "normalize_connection_length_view",
    "trace_toroidal_wall_data_field",
    "wall_trace_post_compute_flow",
    "wall_trace_post_compute_task",
]
