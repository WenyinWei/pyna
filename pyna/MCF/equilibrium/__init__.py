"""pyna.MCF.equilibrium — thin package-level facade over :mod:`pyna.toroidal.equilibrium`.

The old per-module wrappers in ``pyna.MCF.equilibrium`` were removed. Import
from ``pyna.toroidal.equilibrium`` directly for new code, or from this package
root when maintaining older notebooks.
"""

from importlib import import_module

from pyna.toroidal.equilibrium import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(import_module("pyna.toroidal.equilibrium"), name)
