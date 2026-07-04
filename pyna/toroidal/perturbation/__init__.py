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
    "beta_ramp",
    "equilibrium",
    "response",
}

_SYMBOL_MODULES = {
    "BetaRampRadialModeReport": "pyna.toroidal.perturbation.beta_ramp",
    "BetaRampSpectrumDiagnostics": "pyna.toroidal.perturbation.beta_ramp",
    "BetaRampState": "pyna.toroidal.perturbation.beta_ramp",
    "BetaRampSurfaceFieldSamples": "pyna.toroidal.perturbation.beta_ramp",
    "BetaRampTrustReport": "pyna.toroidal.perturbation.beta_ramp",
    "beta_scan_summary_rows": "pyna.toroidal.perturbation.beta_ramp",
    "classify_beta_ramp_trust": "pyna.toroidal.perturbation.beta_ramp",
    "delta_beta_ramp_state": "pyna.toroidal.perturbation.beta_ramp",
    "diagnose_beta_ramp_state": "pyna.toroidal.perturbation.beta_ramp",
    "radial_small_divisor_reports": "pyna.toroidal.perturbation.beta_ramp",
    "sample_beta_ramp_delta_on_surfaces": "pyna.toroidal.perturbation.beta_ramp",
    "scrub_beta_metadata": "pyna.toroidal.perturbation.beta_ramp",
}

__all__ = sorted(_SUBMODULES) + sorted(_SYMBOL_MODULES)


def __getattr__(name):
    if name in _SUBMODULES:
        return import_module(f"pyna.toroidal.perturbation.{name}")
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is not None:
        module = import_module(module_name)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
