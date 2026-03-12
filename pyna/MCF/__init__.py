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
