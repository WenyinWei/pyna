"""
pyna._cyna — thin Python wrapper around the _cyna_ext C++ extension.
"""
import importlib.util
import pathlib

_cyna_ext = None
_available = False

# Try to load _cyna_ext.pyd from this package directory
_p = pathlib.Path(__file__).parent
for _f in list(_p.glob("_cyna_ext*.pyd")) + list(_p.glob("_cyna_ext*.so")):
    try:
        _spec = importlib.util.spec_from_file_location("_cyna_ext", _f)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _cyna_ext = _mod
        _available = True
        break
    except Exception:
        continue

if not _available:
    # Last resort: try normal import (e.g., installed as package)
    try:
        import importlib
        _cyna_ext = importlib.import_module("pyna._cyna._cyna_ext")
        _available = True
    except Exception:
        pass


def is_available() -> bool:
    """Return True if the C++ extension was successfully loaded."""
    return _available


def get_version() -> str:
    """Return the extension version string."""
    return _cyna_ext.__version__ if _available else "not available"


if _available:
    trace_poincare_batch = _cyna_ext.trace_poincare_batch
    trace_poincare_multi = _cyna_ext.trace_poincare_multi
    trace_poincare_batch_twall = getattr(_cyna_ext, "trace_poincare_batch_twall", None)
    trace_connection_length_twall = getattr(_cyna_ext, "trace_connection_length_twall", None)
    trace_wall_hits_twall = getattr(_cyna_ext, "trace_wall_hits_twall", None)

__all__ = [
    "is_available",
    "get_version",
    "trace_poincare_batch",
    "trace_poincare_multi",
    "trace_poincare_batch_twall",
    "trace_connection_length_twall",
    "trace_wall_hits_twall",
]
