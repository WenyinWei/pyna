"""pyna._cyna — optional C++ acceleration layer.

When cyna is compiled and installed, this module provides fast C++
implementations of performance-critical operations.

If cyna is not compiled, all operations fall back to pure Python
implementations in pyna.flt, pyna.topo, etc.

Usage
-----
    from pyna._cyna import is_available, get_version

    if is_available():
        # Use C++ accelerated version
        from pyna._cyna import trace_fieldlines_batch
    else:
        # Automatic fallback — you don't need to do anything
        pass  # pyna.flt handles this transparently

Building cyna
-------------
    cd cyna/
    xmake build cyna_python
    xmake install -o ../pyna/_cyna/
"""

_cyna_ext = None
_available = False

try:
    from pyna._cyna import _cyna_ext  # type: ignore[attr-defined]
    _available = True
except ImportError:
    pass


def is_available() -> bool:
    """Return True if the C++ cyna extension is compiled and available."""
    return _available


def get_version() -> str:
    """Return cyna version string, or 'not available'."""
    if _available and _cyna_ext is not None:
        return _cyna_ext.__version__
    return "not available (pure Python fallback active)"


__all__ = ["is_available", "get_version"]
