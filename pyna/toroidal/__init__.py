"""pyna.toroidal — toroidal magnetic-geometry utilities.

This package groups toroidal coordinate, coil, control, diagnostic, and
visualisation helpers used by pyna.
"""

from importlib import import_module

_SUBMODULES = {
    "coords",
    "equilibrium",
    "coils",
    "control",
    "diagnostics",
    "flt",
    "optimize",
    "perturbation",
    "torus_deformation",
    "visual",
}

_SYMBOL_MODULES = {
    "EquilibriumSolovev": "pyna.toroidal.equilibrium.Solovev",
    "EquilibriumAxisym": "pyna.toroidal.equilibrium.axisymmetric",
    "TorusDeformationSpectrum": "pyna.toroidal.torus_deformation",
    "non_resonant_deformation_spectrum": "pyna.toroidal.torus_deformation",
    "poincare_section_deformation": "pyna.toroidal.torus_deformation",
    "iota_variation_pf": "pyna.toroidal.torus_deformation",
    "mean_radial_displacement": "pyna.toroidal.torus_deformation",
    "mean_radial_displacement_pf": "pyna.toroidal.torus_deformation",
    "mean_radial_displacement_dc": "pyna.toroidal.torus_deformation",
    "mean_radial_displacement_second_order": "pyna.toroidal.torus_deformation",
    "deformation_peak_valley": "pyna.toroidal.torus_deformation",
    "green_function_spectrum": "pyna.toroidal.torus_deformation",
    "iota_to_q": "pyna.toroidal.torus_deformation",
    "q_to_iota": "pyna.toroidal.torus_deformation",
    "iota_prime_from_q_prime": "pyna.toroidal.torus_deformation",
}


__all__ = sorted(_SUBMODULES) + list(_SYMBOL_MODULES)


def __getattr__(name):
    if name in _SUBMODULES:
        return import_module(f"pyna.toroidal.{name}")
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is not None:
        module = import_module(module_name)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
