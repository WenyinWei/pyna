"""Toroidal plasma-response namespace.

This shim collects toroidal plasma-response theory under
:mod:`pyna.toroidal.perturbation` without moving battle-tested modules yet.

Canonical implementation owners today
-------------------------------------
- :mod:`pyna.toroidal.plasma_response.PerturbGS`
- :mod:`pyna.toroidal.plasma_response.coupled_gs`
- selected response-oriented helpers in
  :mod:`pyna.toroidal.equilibrium.feedback_cylindrical` and
  :mod:`pyna.toroidal.equilibrium.feedback_boozer`
"""

from importlib import import_module

_SYMBOL_MODULES = {
    "BoozerPerturbation": "pyna.toroidal.equilibrium.feedback_boozer",
    "BoozerSurface": "pyna.toroidal.equilibrium.feedback_boozer",
    "compute_boozer_response": "pyna.toroidal.equilibrium.feedback_boozer",
    "CylindricalGrid": "pyna.toroidal.equilibrium.feedback_cylindrical",
    "PerturbationField": "pyna.toroidal.equilibrium.feedback_cylindrical",
    "PlasmaResponse": "pyna.toroidal.equilibrium.feedback_cylindrical",
    "compute_cylindrical_response": "pyna.toroidal.equilibrium.feedback_cylindrical",
    "feedback_correction_field": "pyna.toroidal.equilibrium.feedback_cylindrical",
    "iterative_equilibrium_correction": "pyna.toroidal.equilibrium.feedback_cylindrical",
    "solve_perturbed_gs": "pyna.toroidal.plasma_response",
    "solve_perturbed_gs_coupled": "pyna.toroidal.plasma_response",
    "compute_plasma_response": "pyna.toroidal.plasma_response",
    "compute_equilibrium_currents": "pyna.toroidal.plasma_response",
    "compute_diamagnetic_current": "pyna.toroidal.plasma_response",
    "compute_pfirsch_schlueter_current": "pyna.toroidal.plasma_response",
    "solve_coupled_mhd": "pyna.toroidal.plasma_response",
}

_ALIASES = {
    "compute_cylindrical_response": "compute_plasma_response",
}

__all__ = list(_SYMBOL_MODULES)


def __getattr__(name):
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    target_name = _ALIASES.get(name, name)
    return getattr(module, target_name)
