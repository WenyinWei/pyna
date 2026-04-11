"""Toroidal perturbative-equilibrium namespace.

This shim groups perturbative equilibrium theory under the canonical
:mod:`pyna.toroidal.perturbation` umbrella while keeping the actual solver
implementations in the equilibrium package for now.

Canonical implementation owners today
-------------------------------------
- :mod:`pyna.toroidal.equilibrium.finite_beta_perturbation`
- :mod:`pyna.toroidal.equilibrium.GradShafranov`
- :mod:`pyna.toroidal.equilibrium.fenicsx_corrector`
"""

from importlib import import_module

_SYMBOL_MODULES = {
    "recover_pressure_simplest": "pyna.toroidal.equilibrium.GradShafranov",
    "solve_GS_perturbed": "pyna.toroidal.equilibrium.GradShafranov",
    "MU0_DEFAULT": "pyna.toroidal.equilibrium.fenicsx_corrector",
    "AndersonMixer": "pyna.toroidal.equilibrium.fenicsx_corrector",
    "array_to_dolfinx_function": "pyna.toroidal.equilibrium.fenicsx_corrector",
    "build_rz_mesh": "pyna.toroidal.equilibrium.fenicsx_corrector",
    "compute_curl_cylindrical": "pyna.toroidal.equilibrium.fenicsx_corrector",
    "compute_force_residual": "pyna.toroidal.equilibrium.fenicsx_corrector",
    "extract_to_grid": "pyna.toroidal.equilibrium.fenicsx_corrector",
    "fpt_fenicsx_beta_step": "pyna.toroidal.equilibrium.fenicsx_corrector",
    "interpolate_vector_field": "pyna.toroidal.equilibrium.fenicsx_corrector",
    "solve_force_balance_correction": "pyna.toroidal.equilibrium.fenicsx_corrector",
    "solve_linearised_fb": "pyna.toroidal.equilibrium.fenicsx_corrector",
    "CoilVacuumField": "pyna.toroidal.equilibrium.finite_beta_perturbation",
    "CurrentComponents": "pyna.toroidal.equilibrium.finite_beta_perturbation",
    "FiniteBetaPerturbation": "pyna.toroidal.equilibrium.finite_beta_perturbation",
    "PerturbationState": "pyna.toroidal.equilibrium.finite_beta_perturbation",
}

__all__ = list(_SYMBOL_MODULES)


def __getattr__(name):
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
