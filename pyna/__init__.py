"""pyna — Python library for dynamical systems and MHD plasma physics.

Submodules
----------
system   : Abstract dynamical system hierarchy
flt      : Field line tracer
mag      : Magnetic equilibrium, RMP analysis (tokamak/stellarator)
topo     : Topological analysis (Poincaré maps, island chains)
coord    : Magnetic coordinate systems (PEST, Boozer)
"""
__version__ = "0.1.0"

from pyna.system import (
    DynamicalSystem,
    NonAutonomousDynamicalSystem,
    AutonomousDynamicalSystem,
    VectorField,
    VectorField1D,
    VectorField2D,
    VectorField3D,
    VectorField4D,
    AxiSymmetricVectorField3D,
)
