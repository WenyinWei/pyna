"""pyna centralized cache infrastructure.

All expensive computations in pyna use this module for caching.
Provides two caching systems:

1. **pyna_cache (legacy)** — joblib.Memory-based, category-namespaced.
   Simple, robust, used throughout existing pyna code.

2. **memoize (new)** — source-aware, async I/O, multi-backend, LRU eviction.
   Feature-complete disk memoization with zero config.  Drop-in replacement
   for ``@pyna_cache`` with richer capabilities.

Quick start
-----------
    from pyna.cache import memoize, Session

    @memoize
    def expensive_simulation(params):
        # hours of number-crunching...
        return results

    # With session namespacing:
    with Session("tokamak_analysis"):
        @memoize(ignore=["verbose"])
        def fetch_equilibrium(shot, time, verbose=False):
            ...

Backward compatibility
----------------------
    from pyna.cache import pyna_cache, eq_hash, clear_cache  # all still work

Cache invalidation
------------------
    from pyna.cache import clear_cache
    clear_cache()           # clear all
    clear_cache('A_matrix') # clear specific category (prints instructions)
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
import pickle
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1. Legacy pyna_cache (joblib-based) — preserved for backward compat    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_CACHE_ROOT = Path.home() / '.pyna_cache'
_CACHE_ROOT.mkdir(exist_ok=True)

_memory = None  # lazy-init below


def _get_memory():
    """Lazy-init the joblib Memory object (avoids import cost when unused)."""
    global _memory
    if _memory is None:
        import joblib
        _memory = joblib.Memory(location=str(_CACHE_ROOT), verbose=0)
    return _memory


def eq_hash(**params) -> str:
    """Deterministic hash of equilibrium parameters.

    Example: eq_hash(R0=1.86, a=0.6, B0=5.3, kappa=1.7, delta=0.33, q0=3.0)
    Returns a short hex string.
    """
    s = json.dumps(params, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:12]


def array_hash(arr) -> str:
    """Hash a numpy array (full tobytes for exact matching)."""
    arr = np.asarray(arr)
    return hashlib.md5(arr.tobytes()).hexdigest()[:12]


def pyna_cache(category: str = 'general'):
    """Decorator: cache a function's result with joblib.Memory.

    Args to the function must be hashable (scalars, strings, tuples).
    numpy array arguments should be pre-converted to tuples or hashed.

    For new code, consider ``@memoize`` which handles arrays natively.
    """
    def decorator(fn):
        cached_fn = _get_memory().cache(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return cached_fn(*args, **kwargs)

        wrapper.clear_cache = cached_fn.clear
        wrapper.__wrapped__ = fn
        return wrapper
    return decorator


def clear_cache(category: Optional[str] = None):
    """Clear cached results.

    Parameters
    ----------
    category : str or None
        If None, clears ALL cached results.
        If specified (e.g., 'biot_savart'), prints instructions to delete
        the relevant subfolder.
    """
    if category is None:
        _get_memory().clear(warn=False)
        # Also clear the new-style cache if it exists
        global _global_cache
        if _global_cache is not None:
            _global_cache.clear()
        print(f"[pyna.cache] Cleared all cache at {_CACHE_ROOT}")
    else:
        print(f"[pyna.cache] To clear '{category}' cache, delete: {_CACHE_ROOT}")


def cache_info() -> dict:
    """Return info about cache usage (size, number of entries)."""
    total_size = sum(
        f.stat().st_size
        for f in _CACHE_ROOT.rglob('*')
        if f.is_file()
    )
    n_files = sum(1 for f in _CACHE_ROOT.rglob('*.pkl') if f.is_file())
    return {
        'cache_dir': str(_CACHE_ROOT),
        'total_size_mb': total_size / 1e6,
        'n_entries': n_files,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2. Serialization backends                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Backend type: (save_fn, load_fn, extension)
Backend = tuple[Callable[[Path, Any], None], Callable[[Path], Any], str]


# ── NumPy .npz backend (default) ──────────────────────────────────────────

def _npz_save(path: Path, data: Any) -> None:
    """Save to .npz. Dicts → named arrays; scalars → 'data' entry; objects → pickled."""
    if isinstance(data, dict):
        npz_data = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                npz_data[k] = v
            elif isinstance(v, (list, tuple)):
                arr = np.asarray(v)
                if arr.dtype == object:
                    npz_data[k] = np.void(pickle.dumps(v))
                else:
                    npz_data[k] = arr
            elif isinstance(v, dict):
                npz_data[k] = np.void(pickle.dumps(v))
            else:
                npz_data[k] = v
        np.savez_compressed(path, **npz_data)
    elif isinstance(data, np.ndarray):
        np.savez_compressed(path, data=data)
    else:
        np.savez_compressed(path, __pickle__=np.void(pickle.dumps(data)))


def _npz_load(path: Path) -> Any:
    """Load from .npz. Restores dict, array, or pickled object."""
    raw = np.load(path, allow_pickle=True)
    if "__pickle__" in raw:
        return pickle.loads(raw["__pickle__"].tobytes())
    if "data" in raw and len(raw) == 1:
        return raw["data"]
    result = {}
    for k in raw:
        val = raw[k]
        if isinstance(val, np.ndarray) and val.ndim == 0:
            result[k] = val.item()
        else:
            result[k] = val
    return result


# ── Pickle backend ────────────────────────────────────────────────────────

def _pickle_save(path: Path, data: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _pickle_load(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Backend registry ──────────────────────────────────────────────────────

BACKENDS: dict[str, Backend] = {
    "npz": (_npz_save, _npz_load, ".npz"),
    "pickle": (_pickle_save, _pickle_load, ".pkl"),
}

# ── HDF5 backend (optional) ───────────────────────────────────────────────

try:
    import h5py  # noqa: F401

    def _h5_save(path: Path, data: Any) -> None:
        import h5py
        with h5py.File(path, "w") as f:
            if isinstance(data, dict):
                for k, v in data.items():
                    f.create_dataset(k, data=np.asarray(v), compression="gzip")
            elif isinstance(data, np.ndarray):
                f.create_dataset("data", data=data, compression="gzip")
            else:
                f.attrs["pickle"] = np.void(pickle.dumps(data))

    def _h5_load(path: Path) -> Any:
        import h5py
        with h5py.File(path, "r") as f:
            if "pickle" in f.attrs:
                return pickle.loads(f.attrs["pickle"].tobytes())
            if "data" in f and len(f) == 1:
                return f["data"][:]
            return {k: f[k][:] for k in f}

    BACKENDS["hdf5"] = (_h5_save, _h5_load, ".h5")
except ImportError:
    pass


def get_backend(name: str = "npz") -> Backend:
    """Look up a serialization backend by name.

    Parameters
    ----------
    name : str
        Backend name: ``"npz"``, ``"pickle"``, or ``"hdf5"`` (if h5py installed).

    Returns
    -------
    tuple
        (save_fn, load_fn, extension)

    Raises
    ------
    ValueError
        If the backend is unknown.
    """
    if name not in BACKENDS:
        raise ValueError(
            f"Unknown backend '{name}'. Available: {list(BACKENDS)}"
        )
    return BACKENDS[name]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3. Content-addressable keys                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _canonicalize(obj: Any) -> Any:
    """Convert an arbitrary Python object to a JSON-serializable canonical form.

    Rules
    -----
    - NumPy arrays → {__ndarray__: sha256(shape + dtype + sample)}
    - Primitive types (int, float, str, bool, None) → themselves
    - list/tuple → JSON array
    - dict → JSON object (sorted keys)
    - Everything else → str(type) + repr (best-effort)
    """
    if isinstance(obj, np.ndarray):
        h = hashlib.sha256()
        h.update(str(obj.shape).encode())
        h.update(str(obj.dtype).encode())
        flat = obj.ravel()
        sample = flat[: min(256, flat.size)].tobytes()
        h.update(sample)
        return {"__ndarray__": h.hexdigest()}

    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj

    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]

    if isinstance(obj, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(obj.items())}

    if isinstance(obj, (set, frozenset)):
        return [_canonicalize(v) for v in sorted(obj, key=str)]

    # Best-effort for other types
    try:
        return repr(obj)
    except Exception:
        return str(type(obj).__name__)


def make_key(*args: Any, **kwargs: Any) -> str:
    """Generate a 32-char hex key from function arguments.

    Parameters
    ----------
    *args, **kwargs :
        The same arguments passed to the decorated function.

    Returns
    -------
    str
        A 32-character hexadecimal SHA-256 prefix, deterministic given
        identical arguments.  NumPy arrays are hashed by shape+dtype+sample.
    """
    payload = {
        "args": [_canonicalize(a) for a in args],
        "kwargs": _canonicalize(kwargs),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


def make_source_hash(fn: Callable) -> str:
    """Hash the source code of a function for cache invalidation.

    When the function body changes, previously cached results are
    automatically invalidated (stale keys won't match).
    """
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = fn.__name__
    return hashlib.sha256(src.encode()).hexdigest()[:16]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4. CacheStore — disk-backed cache with async I/O and LRU eviction      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass
class _Entry:
    """Internal cache entry metadata."""
    key: str
    path: Path
    created: float
    size_bytes: int
    ttl: float | None = None

    @property
    def expired(self) -> bool:
        if self.ttl is None:
            return False
        return (time.time() - self.created) > self.ttl


class CacheStore:
    """Filesystem cache with async writes and LRU eviction.

    Parameters
    ----------
    root : str or Path
        Cache directory.  Created if it doesn't exist.
    backend : str
        Serialization format: ``"npz"`` (default), ``"pickle"``, ``"hdf5"``.
    max_size_gb : float or None
        Maximum total cache size in GB.  Oldest entries evicted when
        exceeded.  ``None`` = unlimited.
    ttl_seconds : float or None
        Default time-to-live for new entries.  ``None`` = never expire.
    async_write : bool
        If True (default), writes happen in background daemon threads
        so I/O never blocks the main computation.
    """

    def __init__(
        self,
        root: str | Path,
        backend: str = "npz",
        max_size_gb: float | None = 20.0,
        ttl_seconds: float | None = None,
        async_write: bool = True,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_size_gb = max_size_gb
        self.ttl_seconds = ttl_seconds
        self.async_write = async_write

        self._save_fn, self._load_fn, self._ext = get_backend(backend)
        self._backend_name = backend

        self._index: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.Lock()

        # Scan existing entries on disk
        self._scan()

    # ── Public API ────────────────────────────────────────────────────

    def get(self, key: str) -> Any | None:
        """Read from cache.

        Returns
        -------
        Any or None
            The cached data, or ``None`` if the key is not found or the
            entry has expired (or the file is missing).
        """
        with self._lock:
            entry = self._index.get(key)
            if entry is None:
                return None
            if entry.expired:
                self._evict(key)
                return None
            if not entry.path.exists():
                self._evict(key)
                return None
            # Promote to most-recently-used (LRU)
            self._index.move_to_end(key)

        try:
            return self._load_fn(entry.path)
        except Exception:
            with self._lock:
                self._evict(key)
            return None

    def put(self, key: str, data: Any) -> None:
        """Store data in cache.

        When ``async_write=True`` (default), returns immediately and
        writes in a background thread.
        """
        if self.async_write:
            t = threading.Thread(
                target=self._put_sync, args=(key, data), daemon=True
            )
            t.start()
        else:
            self._put_sync(key, data)

    def _put_sync(self, key: str, data: Any) -> None:
        """Synchronous write (used by async thread or direct call).

        Writes to a temp file first, then atomically renames to the
        final path — crash-resilient on most filesystems.
        """
        final_path = self.root / f"{key}{self._ext}"
        tmp_base = self.root / f".{key}.tmp"
        self._save_fn(tmp_base, data)

        # numpy.savez_compressed appends .npz → find the actual output
        actual_tmp = tmp_base
        for candidate in [
            tmp_base,
            tmp_base.with_suffix(self._ext),
            Path(str(tmp_base) + self._ext),
        ]:
            if candidate.exists():
                actual_tmp = candidate
                break
        if not actual_tmp.exists():
            raise FileNotFoundError(f"Save did not produce output: {tmp_base}")
        os.replace(actual_tmp, final_path)

        with self._lock:
            self._index[key] = _Entry(
                key=key,
                path=final_path,
                created=time.time(),
                size_bytes=final_path.stat().st_size,
                ttl=self.ttl_seconds,
            )
            self._index.move_to_end(key)
            self._maybe_evict()

    def contains(self, key: str) -> bool:
        """Check if a key exists and is valid (not expired, file exists)."""
        with self._lock:
            entry = self._index.get(key)
            if entry is None or entry.expired:
                return False
            return entry.path.exists()

    def remove(self, key: str) -> None:
        """Remove a single entry (index + disk)."""
        with self._lock:
            self._evict(key)

    def clear(self) -> None:
        """Remove all entries from this store."""
        with self._lock:
            for key in list(self._index):
                self._evict(key)

    # ── Info ──────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Return cache statistics as a dict."""
        with self._lock:
            entries = len(self._index)
            total_bytes = sum(e.size_bytes for e in self._index.values())
        return {
            "entries": entries,
            "total_mb": round(total_bytes / (1024 * 1024), 2),
            "root": str(self.root),
            "backend": self._backend_name,
            "async_write": self.async_write,
        }

    def __repr__(self) -> str:
        s = self.stats
        return (
            f"CacheStore({s['entries']} entries, {s['total_mb']:.1f} MB, "
            f"root={s['root']})"
        )

    def __len__(self) -> int:
        return self.stats["entries"]

    # ── Internal ──────────────────────────────────────────────────────

    def _scan(self) -> None:
        """Scan cache directory for existing entries on startup."""
        for path in sorted(self.root.glob(f"*{self._ext}")):
            key = path.stem
            self._index[key] = _Entry(
                key=key,
                path=path,
                created=path.stat().st_mtime,
                size_bytes=path.stat().st_size,
                ttl=self.ttl_seconds,
            )

    def _evict(self, key: str) -> None:
        """Remove one entry from index and disk."""
        entry = self._index.pop(key, None)
        if entry and entry.path.exists():
            try:
                entry.path.unlink()
            except OSError:
                pass

    def _maybe_evict(self) -> None:
        """Evict oldest entries if total size exceeds max_size_gb."""
        if self.max_size_gb is None:
            return
        max_bytes = self.max_size_gb * 1024**3
        total = sum(e.size_bytes for e in self._index.values())
        while total > max_bytes and self._index:
            _, oldest = self._index.popitem(last=False)
            total -= oldest.size_bytes
            if oldest.path.exists():
                try:
                    oldest.path.unlink()
                except OSError:
                    pass


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5. Memoize decorator + Session context manager                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Global default cache ──────────────────────────────────────────────────

_DEFAULT_MEMOIZE_ROOT = Path.home() / ".cache" / "pyna"

_global_cache: CacheStore | None = None
_cache_lock = threading.Lock()

# ── Session stack ─────────────────────────────────────────────────────────

_session_stack: list[str] = []
_session_lock = threading.Lock()


def get_cache() -> CacheStore:
    """Return the global (thread-safe, lazy-init) CacheStore instance.

    Default: ``~/.cache/pyna``, npz backend, 20 GB max, async writes.
    """
    global _global_cache
    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                _global_cache = CacheStore(
                    _DEFAULT_MEMOIZE_ROOT,
                    backend="npz",
                    max_size_gb=20,
                    async_write=True,
                )
    return _global_cache


def set_cache_root(path: str | Path) -> None:
    """Change the default cache directory for ``@memoize``.

    Must be called before the first ``@memoize`` usage, or call
    ``clear_global_cache()`` first to reset.

    Parameters
    ----------
    path : str or Path
        New cache root directory.
    """
    global _global_cache, _DEFAULT_MEMOIZE_ROOT
    _DEFAULT_MEMOIZE_ROOT = Path(path)
    _global_cache = None


def clear_global_cache() -> None:
    """Reset the global CacheStore instance.  Does NOT delete files."""
    global _global_cache
    _global_cache = None


# ── Session context manager ──────────────────────────────────────────────

@contextmanager
def Session(namespace: str):
    """Context manager that prefixes all ``@memoize`` keys within its scope.

    Parameters
    ----------
    namespace : str
        A short identifier for the project or analysis phase, e.g.
        ``"east_efit"``, ``"parameter_scan_v2"``.

    Example
    -------
    >>> with Session("tokamak_response"):
    ...     @memoize
    ...     def compute_plasma_response(shot, time):
    ...         ...

    Session is a reentrant context manager — nested sessions compose cleanly
    (keys are joined with ``:``).
    """
    _session_lock.acquire()
    _session_stack.append(namespace)
    _session_lock.release()
    try:
        yield
    finally:
        _session_lock.acquire()
        _session_stack.pop()
        _session_lock.release()


def get_session_namespace() -> str:
    """Return the current namespace (composed from the session stack)."""
    with _session_lock:
        return ":".join(_session_stack) if _session_stack else ""


# ── Memoize decorator ─────────────────────────────────────────────────────

def memoize(
    fn: Callable | None = None,
    *,
    namespace: str | None = None,
    ignore: list[str] | None = None,
    cache: CacheStore | None = None,
    source_aware: bool = True,
    backend: str | None = None,
) -> Callable:
    """Disk-backed memoization decorator.  Zero configuration needed.

    Usage (decorator form)::

        @memoize
        def my_func(a, b, c=3):
            return expensive_work(a, b, c)

    Usage (explicit form)::

        @memoize(namespace="project_x", ignore=["verbose"], backend="hdf5")
        def my_func(data, verbose=False):
            return expensive_work(data)

    Parameters
    ----------
    fn : callable or None
        The function to decorate.  When called as ``@memoize`` (no
        parentheses), ``fn`` is the function.  When called with keyword
        arguments like ``@memoize(namespace=...)``, ``fn`` is None and
        we return the actual decorator.
    namespace : str or None
        Explicit namespace prefix.  Overrides the session stack.
        If None, uses the active session (via ``with Session(...)``).
    ignore : list of str or None
        Keyword argument names to exclude from cache key generation.
        Useful for verbosity flags, loggers, etc. that don't affect output.
    cache : CacheStore or None
        Use a specific cache instance instead of the global default.
    source_aware : bool
        If True (default), the function's source code is included in the
        cache key, so editing the function automatically invalidates
        stale cache entries.
    backend : str or None
        Override the serialization backend for this function only.
        One of ``"npz"``, ``"pickle"``, ``"hdf5"``.

    Returns
    -------
    callable
        The wrapped function with identical signature and return type.
        The wrapper exposes ``wrapper._clear_cache()`` to clear all
        entries for this specific function.
    """
    # Support both @memoize and @memoize(...)
    if fn is not None:
        return _memoize_impl(fn, namespace, ignore, cache, source_aware, backend)

    def decorator(f):
        return _memoize_impl(f, namespace, ignore, cache, source_aware, backend)

    return decorator


def _memoize_impl(
    fn: Callable,
    namespace: str | None,
    ignore: list[str] | None,
    cache: CacheStore | None,
    source_aware: bool,
    backend: str | None,
) -> Callable:
    """Internal: build the memoized wrapper."""
    # Resolve cache: per-function > global
    if cache is not None:
        store = cache
    elif backend is not None:
        # Create a dedicated store with the requested backend
        store = CacheStore(
            _DEFAULT_MEMOIZE_ROOT / f"_backend_{backend}",
            backend=backend,
            max_size_gb=20,
            async_write=True,
        )
    else:
        store = get_cache()

    ignore_set = set(ignore or [])
    src_hash = make_source_hash(fn) if source_aware else ""

    # In-memory stash: when async_write is on, recently written results
    # live here so the next get() within the same process finds them
    # instantly even before the background thread finishes the disk write.
    _stash: dict[str, Any] = {}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Filter out ignored kwargs
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ignore_set}

        # Build cache key
        ns = namespace or get_session_namespace()
        arg_key = make_key(*args, **clean_kwargs)
        if ns:
            full_key = f"{ns}:{fn.__qualname__}:{src_hash}:{arg_key}"
        else:
            full_key = f"{fn.__qualname__}:{src_hash}:{arg_key}"

        # Check in-memory stash first (for async writes not yet on disk)
        if full_key in _stash:
            return _stash[full_key]

        # Try disk cache
        cached = store.get(full_key)
        if cached is not None:
            _stash[full_key] = cached
            return cached

        # Compute and cache
        result = fn(*args, **kwargs)
        _stash[full_key] = result
        store.put(full_key, result)
        return result

    # Expose cache control on the wrapper
    wrapper._pyna_key_fn = fn.__qualname__  # type: ignore[attr-defined]

    def _clear_fn_entries():
        """Clear all cache entries for this function."""
        prefix = f"{fn.__qualname__}:{src_hash}:"
        count = 0
        with store._lock:
            for key in list(store._index):
                if prefix in key:
                    store._evict(key)
                    count += 1
        return count

    wrapper._clear_cache = _clear_fn_entries  # type: ignore[attr-defined]

    return wrapper


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  6. HDF5 large-data backend (optional, from anabranch.backends_hdf5)    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# The basic HDF5 backend is registered above in §2.  When h5py is available,
# users can also access the advanced HDF5 backend with chunking, compression
# tuning, and selective reads.


class HDF5Backend:
    """HDF5 serialization backend for large scientific arrays.

    Parameters
    ----------
    compression : str
        HDF5 compression filter (``"gzip"``, ``"lzf"``).
    compression_opts : int
        Compression level (1–9 for gzip).
    chunk_size : int or bool or None
        Chunk size in elements, ``True`` for auto, ``None`` for no chunking.
    shuffle : bool
        Enable HDF5 shuffle filter (improves compression).
    fletcher32 : bool
        Enable checksum filter for data integrity.
    """

    def __init__(
        self,
        compression: str = "gzip",
        compression_opts: int = 4,
        chunk_size: int | bool | None = 64 * 1024,
        shuffle: bool = True,
        fletcher32: bool = False,
    ) -> None:
        try:
            import h5py  # noqa: F401
        except ImportError:
            raise ImportError(
                "h5py is required for HDF5Backend. Install with: pip install h5py"
            )
        self.compression = compression
        self.compression_opts = compression_opts
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.fletcher32 = fletcher32

    def save(self, path: Path, data: Any) -> None:
        """Save data to an HDF5 file with compression and chunking."""
        import h5py as _h5py

        with _h5py.File(path, "w") as f:
            self._write_object(f, data)

    def _write_object(self, group, data: Any) -> None:
        """Recursively write an object into an HDF5 group."""
        import h5py as _h5py
        import pickle as _pickle

        if isinstance(data, dict):
            for key, value in data.items():
                safe_key = str(key).replace("/", "_")
                if isinstance(value, dict):
                    sub = group.create_group(safe_key)
                    self._write_object(sub, value)
                elif isinstance(value, (np.ndarray, list, tuple)):
                    arr = np.asarray(value)
                    group.create_dataset(
                        safe_key,
                        data=arr,
                        compression=self.compression,
                        compression_opts=self.compression_opts,
                        shuffle=self.shuffle,
                        fletcher32=self.fletcher32,
                        chunks=self._chunks_for(arr),
                    )
                elif isinstance(value, (int, float, str, bool, type(None))):
                    group.attrs[safe_key] = value
                else:
                    group.attrs[safe_key] = np.void(_pickle.dumps(value))
        elif isinstance(data, np.ndarray):
            group.create_dataset(
                "data",
                data=data,
                compression=self.compression,
                compression_opts=self.compression_opts,
                shuffle=self.shuffle,
                fletcher32=self.fletcher32,
                chunks=self._chunks_for(data),
            )
        elif isinstance(data, (list, tuple)):
            arr = np.asarray(data)
            group.create_dataset(
                "data",
                data=arr,
                compression=self.compression,
                compression_opts=self.compression_opts,
                shuffle=self.shuffle,
                fletcher32=self.fletcher32,
                chunks=self._chunks_for(arr),
            )
        else:
            group.attrs["__pickle__"] = np.void(_pickle.dumps(data))

    def _chunks_for(self, arr: np.ndarray):
        """Determine chunk shape for an array."""
        if self.chunk_size is None or self.chunk_size is False:
            return None
        if self.chunk_size is True:
            return True
        total = arr.size
        if total == 0:
            return True
        chunk_elems = int(self.chunk_size)
        chunks = [1] * (arr.ndim - 1) + [min(chunk_elems, arr.shape[-1])]
        return tuple(chunks)

    def load(self, path: Path) -> Any:
        """Load data from an HDF5 file."""
        import h5py as _h5py
        import pickle as _pickle

        with _h5py.File(path, "r") as f:
            return self._read_object(f, _pickle)

    def _read_object(self, group, _pickle) -> Any:
        """Recursively read an object from an HDF5 group."""
        import h5py as _h5py

        if "__pickle__" in group.attrs:
            return _pickle.loads(group.attrs["__pickle__"].tobytes())

        if "data" in group and len(group) == 1:
            return group["data"][:]

        result: dict[str, Any] = {}
        for key in group:
            val = group[key]
            if isinstance(val, _h5py.Dataset):
                result[key] = val[:]
            elif isinstance(val, _h5py.Group):
                result[key] = self._read_object(val, _pickle)

        for key in group.attrs:
            if key.startswith("__"):
                continue
            val = group.attrs[key]
            if isinstance(val, np.ndarray) and val.dtype == np.dtype("V"):
                result[key] = _pickle.loads(val.tobytes())
            elif isinstance(val, np.ndarray) and val.ndim == 0:
                result[key] = val.item()
            else:
                result[key] = val

        return result if result else None

    def load_keys(self, path: Path, keys: list[str]) -> dict[str, Any]:
        """Load only specified top-level datasets/groups (fast partial read)."""
        import h5py as _h5py
        import pickle as _pickle

        result: dict[str, Any] = {}
        with _h5py.File(path, "r") as f:
            for key in keys:
                if key in f:
                    val = f[key]
                    if isinstance(val, _h5py.Dataset):
                        result[key] = val[:]
                    elif isinstance(val, _h5py.Group):
                        result[key] = self._read_object(val, _pickle)
            for key in keys:
                if key in f.attrs:
                    result[key] = f.attrs[key]
        return result

    def list_keys(self, path: Path) -> list[str]:
        """List top-level dataset and group names in an HDF5 file."""
        import h5py as _h5py

        with _h5py.File(path, "r") as f:
            return list(f.keys())

    def as_backend(self) -> tuple:
        """Return a (save_fn, load_fn, extension) tuple for the backend registry."""
        return (self.save, self.load, ".h5")


def register_hdf5_backend(
    name: str = "hdf5_large",
    compression: str = "gzip",
    compression_opts: int = 4,
    chunk_size: int | bool | None = 64 * 1024,
) -> None:
    """Register the large-data HDF5 backend in the global registry.

    Parameters
    ----------
    name : str
        Backend name for the registry (default ``"hdf5_large"``).
    compression : str
        Compression filter (``"gzip"``, ``"lzf"``).
    compression_opts : int
        Compression level (1–9 for gzip).
    chunk_size : int or bool or None
        Chunk size in elements.
    """
    backend = HDF5Backend(
        compression=compression,
        compression_opts=compression_opts,
        chunk_size=chunk_size,
    )
    BACKENDS[name] = backend.as_backend()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  7. Public API                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

__all__ = [
    # Legacy (backward compatible)
    "pyna_cache",
    "eq_hash",
    "array_hash",
    "clear_cache",
    "cache_info",
    # New memoize system
    "memoize",
    "Session",
    "CacheStore",
    "set_cache_root",
    "get_cache",
    "clear_global_cache",
    # Key utilities
    "make_key",
    "make_source_hash",
    # Backend utilities
    "get_backend",
    "BACKENDS",
    "HDF5Backend",
    "register_hdf5_backend",
]
