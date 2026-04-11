"""pyna.MCF — legacy compatibility facade for :mod:`pyna.toroidal`.

New code should import from :mod:`pyna.toroidal`. This package remains only so
historical notebooks and scripts continue to resolve a coherent namespace.
"""

from importlib import import_module

from pyna.toroidal import __all__ as _TOROIDAL_ALL

__all__ = list(_TOROIDAL_ALL)


def __getattr__(name):
    return getattr(import_module("pyna.toroidal"), name)
