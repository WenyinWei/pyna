"""pyna.MCF — Magnetic Confinement Fusion extensions for pyna.

Subpackages
-----------
equilibrium
    Axisymmetric and stellarator MHD equilibria: Solov'ev analytic solutions,
    numerical Grad-Shafranov solver, stellarator configurations.
coords
    Magnetic coordinate systems: PEST, Boozer, Hamada, Equal-arc
    straight-field-line transformations.
coils
    Coil geometry, Biot-Savart field computation, and RMP coil-set models.
control
    MCF topology control: gap response functions and q-profile response.
plasma_response
    Perturbed GS solver for linear plasma equilibrium response.
diagnostics
    Synthetic plasma diagnostics and observable extraction.
visual
    Publication-quality tokamak figures.  The :mod:`~pyna.MCF.visual.tokamak_manifold`
    module provides composable axes-based functions for EAST-tokamak research plots
    (equilibrium cross-sections, Poincaré orbits, stable/unstable manifold bundles).
torus_deformation
    Non-resonant torus (flux-surface) deformation under external magnetic
    perturbations.  Implements the analytic BNF-derived spectral theory of
    Wei (2025): full (δr, δθ, δφ) Fourier spectra, mean radial displacement,
    Poincaré-section projection, and Green's function coefficients.

Top-level exports
-----------------
The following names are importable directly from ``pyna.MCF``:

From :mod:`pyna.MCF.equilibrium`:
    SolovevEquilibrium, AxisymEquilibrium

From :mod:`pyna.MCF.torus_deformation`:
    TorusDeformationSpectrum,
    non_resonant_deformation_spectrum,
    poincare_section_deformation,
    iota_variation_pf,
    mean_radial_displacement,
    mean_radial_displacement_pf,
    mean_radial_displacement_dc,
    mean_radial_displacement_second_order,
    deformation_peak_valley,
    green_function_spectrum,
    iota_to_q, q_to_iota, iota_prime_from_q_prime
"""
from pyna.MCF.equilibrium.Solovev import SolovevEquilibrium
from pyna.MCF.equilibrium.axisymmetric import AxisymEquilibrium
from pyna.MCF.torus_deformation import (
    TorusDeformationSpectrum,
    non_resonant_deformation_spectrum,
    poincare_section_deformation,
    iota_variation_pf,
    mean_radial_displacement,
    mean_radial_displacement_pf,
    mean_radial_displacement_dc,
    mean_radial_displacement_second_order,
    deformation_peak_valley,
    green_function_spectrum,
    iota_to_q,
    q_to_iota,
    iota_prime_from_q_prime,
)
