"""pyna.toroidal.equilibrium — toroidal MHD equilibrium models.

FEniCSx solvers → :mod:`topoquest.analysis.fem`
Plasma response   → :mod:`topoquest.analysis.response`
Equilibrium ABCs  → stays here in pyna (General-Dynamical-Systems layer)
Analytic models   → stays here (Solov'ev, stellarator)
"""
from importlib import import_module

from pyna.toroidal.equilibrium.axisymmetric import EquilibriumAxisym, EquilibriumTokamakCircularSynthetic
from pyna.toroidal.equilibrium.Solovev import EquilibriumSolovev
from pyna.toroidal.equilibrium.GradShafranov import recover_pressure_simplest, solve_GS_perturbed
from pyna.toroidal.equilibrium.stellarator import StellaratorSimple, simple_stellarator

_OPTIONAL_TOPOQUEST_EXPORTS = {
    "BoozerSurface": "topoquest.analysis.response.feedback_boozer",
    "BoozerPerturbation": "topoquest.analysis.response.feedback_boozer",
    "compute_boozer_response": "topoquest.analysis.response.feedback_boozer",
    "CylindricalGrid": "topoquest.analysis.response.feedback_cylindrical",
    "PerturbationField": "topoquest.analysis.response.feedback_cylindrical",
    "PlasmaResponse": "topoquest.analysis.response.feedback_cylindrical",
    "compute_plasma_response": "topoquest.analysis.response.feedback_cylindrical",
    "feedback_correction_field": "topoquest.analysis.response.feedback_cylindrical",
    "iterative_equilibrium_correction": "topoquest.analysis.response.feedback_cylindrical",
    "greens_function_cylinder": "topoquest.analysis.response.feedback_cylindrical_utils",
    "lundquist_number": "topoquest.analysis.response.feedback_cylindrical_utils",
    "toroidal_fft": "topoquest.analysis.response.feedback_cylindrical_utils",
    "convergence_monitor": "topoquest.analysis.response.feedback_cylindrical_utils",
}

__all__ = [
    "EquilibriumAxisym",
    "EquilibriumTokamakCircularSynthetic",
    "EquilibriumSolovev",
    "recover_pressure_simplest",
    "solve_GS_perturbed",
    "StellaratorSimple",
    "simple_stellarator",
    *_OPTIONAL_TOPOQUEST_EXPORTS,
]


def __getattr__(name):
    module_name = _OPTIONAL_TOPOQUEST_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == "topoquest":
            raise ImportError(
                f"{name} is provided by optional dependency 'topoquest'. "
                "Install topoquest to use plasma-response equilibrium exports."
            ) from exc
        raise
    value = getattr(module, name)
    globals()[name] = value
    return value
