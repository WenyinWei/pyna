"""pyna — Python library for dynamical systems and toroidal / MHD plasma physics.

Submodules
----------
system    : Abstract dynamical system hierarchy
flt       : Field line tracer
topo      : Topological analysis (generic + toroidal specializations)
coord     : Magnetic coordinate systems (legacy re-export)
mag       : Magnetic equilibrium and analysis (legacy re-export)
toroidal  : Toroidal geometry and field-line utilities
"""
__version__ = "0.5.0"

from importlib import import_module

from pyna import field_data
from pyna import vector_calc
from pyna.fields import (
    ScalarField3DCylindrical,
    VectorField3DCylindrical,
    VectorField3DAxiSymmetric,
    ScalarField3DAxiSymmetric,
)
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
    VectorField3DAxiSymmetric,
)

def __getattr__(name):
    if name == "toroidal":
        return import_module("pyna.toroidal")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "field_data",
    "vector_calc",
    "toroidal",
    "ScalarField3DCylindrical",
    "VectorField3DCylindrical",
    "VectorField3DAxiSymmetric",
    "ScalarField3DAxiSymmetric",
    "classical_maps",
    "poincare_io",
    "DynamicalSystem",
    "NonAutonomousDynamicalSystem",
    "AutonomousDynamicalSystem",
    "VectorField",
    "VectorField1D",
    "VectorField2D",
    "VectorField3D",
    "VectorField4D",
    "VectorField3DAxiSymmetric",
]
