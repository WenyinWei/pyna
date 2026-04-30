"""pyna.toroidal.perturbation — canonical landing zone for toroidal perturbative theory.

This package is intentionally lightweight.  It provides an honest namespace for
future toroidal functional-perturbation / plasma-response work without forcing a
large refactor today.

Ownership boundary
------------------
- :mod:`pyna.control` holds geometry/topology FPT that is generic across
  dynamical systems.
- :mod:`pyna.toroidal.perturbation` is the toroidal / MHD umbrella for
  perturbative equilibrium and plasma-response theory.
- Heavy implementations continue to live in their mature homes under
  :mod:`pyna.toroidal.equilibrium` and :mod:`pyna.toroidal.plasma_response`
  until a future consolidation is justified.

Near-term buckets
-----------------
- ``equilibrium``: finite-β continuation, perturbed Grad–Shafranov, force-
  balance correction workflows, equilibrium linearisation helpers.
- ``response``: plasma current closure models, coupled response solvers,
  vacuum→plasma transfer operators, and diagnostics for response validity.

This module should stay architectural until real shared perturbation APIs exist.
"""

from importlib import import_module

_SUBMODULES = {
    "equilibrium",
    "response",
}

__all__ = sorted(_SUBMODULES)


def __getattr__(name):
    if name in _SUBMODULES:
        return import_module(f"pyna.toroidal.perturbation.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
