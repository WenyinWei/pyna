"""pyna.MCF — Magnetic Confinement Fusion specific extensions to pyna.

Submodules
----------
equilibrium     : Axisymmetric/stellarator MHD equilibria (Solov'ev, GS, etc.)
coords          : Magnetic coordinate systems (PEST, Boozer, Hamada, Equal-arc)
coils           : Coil geometry, Biot-Savart, RMP coil sets
control         : MCF-specific topology control (gap response, q-profile)
plasma_response : Perturbed GS solver for plasma equilibrium response
diagnostics     : Plasma diagnostic observables
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
