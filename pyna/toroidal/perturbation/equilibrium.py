"""Toroidal perturbative-equilibrium namespace.

Delegates to :mod:`topoquest.analysis.fem` for FEniCSx + finite-beta solvers.
"""
from importlib import import_module

_SYMBOL_MODULES = {
    "recover_pressure_simplest": "pyna.toroidal.equilibrium.GradShafranov",
    "solve_GS_perturbed": "pyna.toroidal.equilibrium.GradShafranov",
    # fenicsx_corrector → topoquest
    "MU0_DEFAULT": "topoquest.analysis.fem.fenicsx_corrector",
    "AndersonMixer": "topoquest.analysis.fem.fenicsx_corrector",
    "array_to_dolfinx_function": "topoquest.analysis.fem.fenicsx_corrector",
    "build_rz_mesh": "topoquest.analysis.fem.fenicsx_corrector",
    "compute_curl_cylindrical": "topoquest.analysis.fem.fenicsx_corrector",
    "compute_force_residual": "topoquest.analysis.fem.fenicsx_corrector",
    "extract_to_grid": "topoquest.analysis.fem.fenicsx_corrector",
    "fpt_fenicsx_beta_step": "topoquest.analysis.fem.fenicsx_corrector",
    "interpolate_vector_field": "topoquest.analysis.fem.fenicsx_corrector",
    "solve_force_balance_correction": "topoquest.analysis.fem.fenicsx_corrector",
    "solve_linearised_fb": "topoquest.analysis.fem.fenicsx_corrector",
    # finite_beta_perturbation → topoquest
    "CoilVacuumField": "topoquest.analysis.fem.finite_beta_perturbation",
    "CurrentComponents": "topoquest.analysis.fem.finite_beta_perturbation",
    "FiniteBetaPerturbation": "topoquest.analysis.fem.finite_beta_perturbation",
    "PerturbationState": "topoquest.analysis.fem.finite_beta_perturbation",
}

__all__ = list(_SYMBOL_MODULES)


def __getattr__(name):
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
