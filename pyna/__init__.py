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

from pyna import field_data
from pyna import vector_calc
from pyna.field_data import CylindricalVectorField, CylindricalScalarField
from pyna.topo import classical_maps
from pyna.io import poincare_io

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
