"""Post-compute views for toroidal field-line wall traces."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pyna.toroidal.flt.numba_poincare import (
    strike_line_from_wall_hits,
    trace_wall_hits_twall_field,
)
from pyna.toroidal.geometry import coerce_toroidal_surface_arrays


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

_WALL_TRACE_SCHEMA_NAME = "pyna.toroidal.flt.wall_trace_data"
_WALL_TRACE_SCHEMA_VERSION = 1
_WALL_TRACE_COMPATIBILITY = "v1 append-only; readers ignore unknown fields"


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


def _array_spec(value) -> dict[str, Any]:
    arr = np.asarray(value)
    return {"shape": list(arr.shape), "dtype": str(arr.dtype)}


def _npz_scalar(data, key: str, default=None):
    if key not in data.files:
        return default
    return np.asarray(data[key]).item()


def _json_from_npz_scalar(data, key: str, default):
    value = _npz_scalar(data, key, None)
    if value is None:
        return default
    return json.loads(str(value))


def _validate_npz_array_spec(data, key: str, spec: dict[str, Any]) -> None:
    if key not in data.files:
        raise ValueError(f"cached toroidal wall trace is missing array {key!r}")
    arr = np.asarray(data[key])
    expected_shape = spec.get("shape")
    if expected_shape is not None and list(arr.shape) != list(expected_shape):
        raise ValueError(
            f"cached toroidal wall trace array {key!r} has shape {arr.shape}; "
            f"expected {tuple(expected_shape)}"
        )
    expected_dtype = spec.get("dtype")
    if expected_dtype is not None and str(arr.dtype) != str(expected_dtype):
        raise ValueError(
            f"cached toroidal wall trace array {key!r} has dtype {arr.dtype}; "
            f"expected {expected_dtype}"
        )


def _field_array_signature(field, attr_names: tuple[str, ...]) -> dict[str, Any] | None:
    for name in attr_names:
        if not hasattr(field, name):
            continue
        value = getattr(field, name)
        if callable(value):
            continue
        arr = np.asarray(value)
        return {
            "attr": name,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "sha256": _array_digest(arr),
        }
    return None


def _field_cache_signature(field) -> dict[str, Any]:
    """Return a stable, compact cache signature for known field objects."""

    signature: dict[str, Any] = {
        "type": f"{field.__class__.__module__}.{field.__class__.__qualname__}",
    }
    for name in ("label", "name"):
        if not hasattr(field, name):
            continue
        value = getattr(field, name)
        if value is not None and not callable(value):
            signature[name] = str(value)
    for name in ("nfp", "field_periods", "nPhi"):
        if not hasattr(field, name):
            continue
        value = getattr(field, name)
        if value is not None and not callable(value):
            signature[name] = int(value)
    for name in ("field_period",):
        if not hasattr(field, name):
            continue
        value = getattr(field, name)
        if value is not None and not callable(value):
            signature[name] = float(value)
    for key, attr_names in {
        "R": ("R_arr", "R"),
        "Z": ("Z_arr", "Z"),
        "Phi": ("Phi",),
        "BR": ("BR", "VR"),
        "BZ": ("BZ", "VZ"),
        "BPhi": ("BPhi", "VPhi"),
    }.items():
        field_array = _field_array_signature(field, attr_names)
        if field_array is not None:
            signature[key] = field_array
    return signature


def _wall_trace_cache_signature_payload(
    *,
    field_signature,
    seed_R,
    seed_Z,
    phi_start: float,
    max_turns: int,
    DPhi: float,
    wall_phi,
    wall_R_all=None,
    wall_Z_all=None,
    extend_phi: bool,
) -> dict[str, Any]:
    return {
        "field": field_signature,
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


def _wall_trace_cache_signature_from_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _wall_trace_cache_signature(
    *,
    field_signature,
    seed_R,
    seed_Z,
    phi_start: float,
    max_turns: int,
    DPhi: float,
    wall_phi,
    wall_R_all=None,
    wall_Z_all=None,
    extend_phi: bool,
) -> str:
    payload = _wall_trace_cache_signature_payload(
        field_signature=field_signature,
        seed_R=seed_R,
        seed_Z=seed_Z,
        phi_start=phi_start,
        max_turns=max_turns,
        DPhi=DPhi,
        wall_phi=wall_phi,
        wall_R_all=wall_R_all,
        wall_Z_all=wall_Z_all,
        extend_phi=extend_phi,
    )
    return _wall_trace_cache_signature_from_payload(payload)


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
        wall_hit_items = sorted(((str(key), value) for key, value in self.wall_hits.items()), key=lambda item: item[0])
        wall_hit_keys = [key for key, _value in wall_hit_items]
        wall_hit_specs = {key: _array_spec(value) for key, value in wall_hit_items}
        array_specs = {
            "seed_R": _array_spec(self.seed_R),
            "seed_Z": _array_spec(self.seed_Z),
            **{f"wall_hits_{key}": spec for key, spec in wall_hit_specs.items()},
        }
        arrays: dict[str, np.ndarray] = {
            "schema_name": np.asarray(_WALL_TRACE_SCHEMA_NAME),
            "schema_version": np.asarray(_WALL_TRACE_SCHEMA_VERSION, dtype=np.int64),
            "compatibility": np.asarray(_WALL_TRACE_COMPATIBILITY),
            "complete": np.asarray(True),
            "n_seed": np.asarray(self.n_seed, dtype=np.int64),
            "seed_R": np.asarray(self.seed_R, dtype=float),
            "seed_Z": np.asarray(self.seed_Z, dtype=float),
            "phi_start": np.asarray(float(self.phi_start)),
            "max_turns": np.asarray(int(self.max_turns), dtype=int),
            "DPhi": np.asarray(float(self.DPhi)),
            "wall_hit_keys_json": np.asarray(json.dumps(wall_hit_keys, sort_keys=True)),
            "wall_hit_specs_json": np.asarray(json.dumps(wall_hit_specs, sort_keys=True)),
            "array_specs_json": np.asarray(json.dumps(array_specs, sort_keys=True)),
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True, default=_json_default)),
        }
        for key, value in wall_hit_items:
            arrays[f"wall_hits_{key}"] = np.asarray(value)
        tmp = out.with_name(f".{out.name}.tmp.npz")
        try:
            np.savez(str(tmp), **arrays)
            tmp.replace(out)
        finally:
            if tmp.exists():
                tmp.unlink()

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
            schema_name = str(_npz_scalar(data, "schema_name", _WALL_TRACE_SCHEMA_NAME))
            if schema_name != _WALL_TRACE_SCHEMA_NAME:
                raise ValueError(f"cached toroidal wall trace has unknown schema {schema_name!r}")
            schema_version = int(_npz_scalar(data, "schema_version", 1))
            if schema_version > _WALL_TRACE_SCHEMA_VERSION:
                raise ValueError(
                    "cached toroidal wall trace was written by a newer schema "
                    f"({schema_version}); this version supports {_WALL_TRACE_SCHEMA_VERSION}."
                )
            if schema_version < 1:
                raise ValueError("cached toroidal wall trace has an unsupported schema version")
            if "complete" in data.files and not bool(_npz_scalar(data, "complete")):
                raise ValueError("cached toroidal wall trace is not marked complete")

            metadata_raw = str(np.asarray(data["metadata_json"]).item())
            wall_hit_keys = _json_from_npz_scalar(data, "wall_hit_keys_json", None)
            if wall_hit_keys is None:
                wall_hit_keys = sorted(key[len("wall_hits_"):] for key in data.files if key.startswith("wall_hits_"))
            wall_hit_specs = _json_from_npz_scalar(data, "wall_hit_specs_json", {})
            for key in wall_hit_keys:
                array_key = f"wall_hits_{key}"
                if key in wall_hit_specs:
                    _validate_npz_array_spec(data, array_key, wall_hit_specs[key])
            array_specs = _json_from_npz_scalar(data, "array_specs_json", {})
            for key, spec in array_specs.items():
                _validate_npz_array_spec(data, key, spec)
            wall_hits = {key: np.asarray(data[f"wall_hits_{key}"]) for key in wall_hit_keys}
            seed_R = np.asarray(data["seed_R"], dtype=float)
            seed_Z = np.asarray(data["seed_Z"], dtype=float)
            n_seed = _npz_scalar(data, "n_seed", None)
            if n_seed is not None and int(n_seed) != int(seed_R.size):
                raise ValueError(
                    f"cached toroidal wall trace has n_seed={int(n_seed)} "
                    f"but seed_R contains {seed_R.size} entries"
                )
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
    wall_R_all=None,
    wall_Z_all=None,
    *,
    extend_phi: bool = True,
    cache_path: str | Path | None = None,
    overwrite: bool = False,
    validate_cache: bool = True,
    allow_legacy_unsigned_cache: bool = False,
    field_signature: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> ToroidalWallTraceData:
    """Trace bidirectional toroidal-wall hits once, optionally caching raw data."""

    wall_phi, wall_R_all, wall_Z_all = coerce_toroidal_surface_arrays(wall_phi, wall_R_all, wall_Z_all)
    resolved_field_signature = _field_cache_signature(field) if field_signature is None else field_signature
    signature_payload = _wall_trace_cache_signature_payload(
        field_signature=resolved_field_signature,
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
    signature = _wall_trace_cache_signature_from_payload(signature_payload)
    if cache_path is not None:
        path = Path(cache_path).expanduser()
        if path.exists() and not overwrite:
            cached = ToroidalWallTraceData.load_npz(path)
            cached_signature = cached.metadata.get("cache_signature")
            if validate_cache:
                if cached_signature is None:
                    if not bool(allow_legacy_unsigned_cache):
                        raise ValueError(
                            "cached toroidal wall trace lacks cache signature; "
                            "use a different cache_path, pass overwrite=True, or explicitly "
                            "pass allow_legacy_unsigned_cache=True"
                        )
                elif cached_signature != signature:
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
            "field_signature": resolved_field_signature,
            "cache_signature": signature,
            "cache_signature_inputs": signature_payload,
        },
    )
    if path is not None:
        payload.save_npz(path)
    return payload


def wall_trace_post_compute_task(
    cache_path: str,
    views: Sequence[str],
    out_path: str | None = None,
) -> dict[str, np.ndarray]:
    """Load one cached raw trace and derive scalar Lc views."""

    data = ToroidalWallTraceData.load_npz(cache_path)
    if out_path is not None:
        return data.save_views_npz(out_path, tuple(views))
    return data.views(tuple(views))


def _optional_prefect_flow_task():
    try:
        from pyna.workflow.prefect import optional_prefect
    except ModuleNotFoundError as exc:
        if exc.name and (
            exc.name == "pyna.workflow" or exc.name.startswith("pyna.workflow.")
        ):
            raise RuntimeError(
                "Prefect is required for workflow runtime support. "
                "Install it with `pyna-chaos[workflow]` or `pyna-chaos[prefect]`."
            ) from exc
        raise
    return optional_prefect()


def _make_prefect_wall_post_compute_flow(flow, task):
    @task(name="pyna-wall-trace-post-compute-task")
    def _wall_trace_post_compute_task(
        cache_path: str,
        views: Sequence[str],
        out_path: str | None = None,
    ) -> dict[str, np.ndarray]:
        return wall_trace_post_compute_task(cache_path, tuple(views), out_path)

    @flow(name="pyna-wall-trace-post-compute")
    def _wall_trace_post_compute_flow(
        cache_path: str,
        views: Sequence[str] = ("Lc_plus", "Lc_minus", "Lc_sum", "Lc_max"),
        out_path: str | None = None,
    ):
        return _wall_trace_post_compute_task(cache_path, tuple(views), out_path)

    return _wall_trace_post_compute_flow


def wall_trace_post_compute_flow(
    cache_path: str,
    views: Sequence[str] = ("Lc_plus", "Lc_minus", "Lc_sum", "Lc_max"),
    out_path: str | None = None,
):
    """Prefect flow for post-compute views from cached wall-hit data."""

    flow, task = _optional_prefect_flow_task()
    prefect_flow = _make_prefect_wall_post_compute_flow(flow, task)
    return prefect_flow(cache_path, tuple(views), out_path)


def build_prefect_wall_post_compute_flow():
    """Return the canonical Prefect flow for wall-trace post-compute views."""

    flow, task = _optional_prefect_flow_task()
    return _make_prefect_wall_post_compute_flow(flow, task)


__all__ = [
    "ToroidalWallTraceData",
    "build_prefect_wall_post_compute_flow",
    "connection_length_views_from_wall_hits",
    "normalize_connection_length_view",
    "trace_toroidal_wall_data_field",
    "wall_trace_post_compute_flow",
    "wall_trace_post_compute_task",
]
