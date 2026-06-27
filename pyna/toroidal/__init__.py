"""pyna.toroidal — toroidal magnetic-geometry utilities.

This package groups toroidal coordinate, coil, control, diagnostic, and
visualisation helpers used by pyna.
"""

from importlib import import_module

_SUBMODULES = {
    "boozer_coords",
    "coords",
    "equilibrium",
    "coils",
    "control",
    "diagnostics",
    "flt",
    "minor_radius",
    "optimize",
    "pest_coords",
    "perturbation",
    "perturbation_spectrum",
    "surface_coordinates",
    "torus_deformation",
    "visual",
}

_SYMBOL_MODULES = {
    "EquilibriumSolovev": "pyna.toroidal.equilibrium.Solovev",
    "EquilibriumAxisym": "pyna.toroidal.equilibrium.axisymmetric",
    "GeometricMinorRadiusProvider": "pyna.toroidal.minor_radius",
    "TorusDeformationSpectrum": "pyna.toroidal.torus_deformation",
    "RadialPerturbationSplit": "pyna.toroidal.torus_deformation",
    "BoozerCoordinateMesh": "pyna.toroidal.boozer_coords",
    "RadialPerturbationFourierSpectrum": "pyna.toroidal.perturbation_spectrum",
    "build_Boozer_coordinates": "pyna.toroidal.boozer_coords",
    "minor_radius_label": "pyna.toroidal.minor_radius",
    "radial_perturbation_Fourier_spectrum": "pyna.toroidal.perturbation_spectrum",
    "radial_perturbation_component": "pyna.toroidal.perturbation_spectrum",
    "surface_unit_normal_cylindrical": "pyna.toroidal.perturbation_spectrum",
    "split_radial_perturbation_spectrum": "pyna.toroidal.torus_deformation",
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
