"""Toroidal plasma-response namespace.

Delegates to :mod:`topoquest.analysis.fem` for PerturbGS + coupled MHD solvers.
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
    # PerturbGS + coupled_gs → topoquest
    "solve_perturbed_gs": "topoquest.analysis.fem.perturb_gs",
    "solve_perturbed_gs_coupled": "topoquest.analysis.fem.perturb_gs",
    "compute_plasma_response": "topoquest.analysis.fem.perturb_gs",
    "compute_equilibrium_currents": "topoquest.analysis.fem.perturb_gs",
    "compute_diamagnetic_current": "topoquest.analysis.fem.perturb_gs",
    "compute_pfirsch_schlueter_current": "topoquest.analysis.fem.perturb_gs",
    "solve_coupled_mhd": "topoquest.analysis.fem.coupled_gs",
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
