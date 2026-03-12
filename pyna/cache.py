"""pyna centralized cache infrastructure.

All expensive computations in pyna use this module for caching.
Cache is disk-persistent (joblib.Memory), stored in ~/.pyna_cache/.

Cache categories and their keys:
  'biot_savart'   : (coil_R, coil_Z, coil_I, R_pts, Z_pts) → (BR, BZ, Bphi)
  'A_matrix'      : (eq_hash, R, Z, phi, eps) → 2×2 array
  'DPm'           : (eq_hash, R, Z) → 2×2 array
  'manifold'      : (eq_hash, R_xpt, Z_xpt, s_max) → (N,2) array
  'flux_surface'  : (eq_hash, psi_norm) → (R_arr, Z_arr)
  'xpoint'        : (eq_hash, R0, Z0) → (R_xpt, Z_xpt)

Usage
-----
    from pyna.cache import pyna_cache, eq_hash

    # Decorator style
    @pyna_cache('A_matrix')
    def compute_A(eq_hash_str, R, Z, phi, eps): ...

Cache invalidation
------------------
    from pyna.cache import clear_cache
    clear_cache()           # clear all
    clear_cache('A_matrix') # clear specific category (prints instructions)
"""

import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Optional
import joblib
from functools import wraps

_CACHE_ROOT = Path.home() / '.pyna_cache'
_CACHE_ROOT.mkdir(exist_ok=True)

_memory = joblib.Memory(location=str(_CACHE_ROOT), verbose=0)


def eq_hash(**params) -> str:
    """Deterministic hash of equilibrium parameters.

    Example: eq_hash(R0=1.86, a=0.6, B0=5.3, kappa=1.7, delta=0.33, q0=3.0)
    Returns a short hex string.
    """
    s = json.dumps(params, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:12]


def array_hash(arr) -> str:
    """Hash a numpy array (shape + values summary)."""
    arr = np.asarray(arr)
    return hashlib.md5(arr.tobytes()).hexdigest()[:12]


def pyna_cache(category: str = 'general'):
    """Decorator: cache a function's result with joblib.Memory.

    Args to the function must be hashable (scalars, strings, tuples).
    numpy array arguments should be pre-converted to tuples or hashed.
    """
    def decorator(fn):
        cached_fn = _memory.cache(fn)

        @wraps(fn)
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
        _memory.clear(warn=False)
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
