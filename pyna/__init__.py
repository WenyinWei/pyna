"""pyna — Python library for dynamical systems and toroidal / MHD plasma physics.

Submodules
----------
system    : Abstract dynamical system hierarchy
flt       : Field line tracer
topo      : Topological analysis (generic + toroidal specializations)
toroidal  : Toroidal geometry and field-line utilities

.. deprecated:: 0.8.6
    ``pyna.coord`` has been removed. ``pyna.mag`` remains as a
    compatibility facade; prefer ``pyna.toroidal.control`` in new code.
"""
__version__ = "0.9.10"

from importlib import import_module

from pyna import field_data
from pyna import vector_calc
from pyna.fields import (
    CylindricalFieldArrays,
    ScalarFieldCylind,
    ScalarFieldCylindAxisym,
    VectorFieldCartesian,
    VectorFieldCylind,
    VectorFieldCylindAxisym,
    as_scalar_field_cylindrical,
    as_scalar_field_cylind,
    as_vector_field_cartesian,
    as_vector_field_cylindrical,
    as_vector_field_cylind,
    validate_phi_grid,
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
)

def __getattr__(name):
    if name == "toroidal":
        return import_module("pyna.toroidal")
    if name == "dynamics":
        return import_module("pyna.dynamics")
    if name == "trace_dataset":
        return import_module("pyna.trace_dataset")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "field_data",
    "vector_calc",
    "toroidal",
    "dynamics",
    "trace_dataset",
    "CylindricalFieldArrays",
    "VectorFieldCartesian",
    "VectorFieldCylind",
    "VectorFieldCylindAxisym",
    "ScalarFieldCylind",
    "ScalarFieldCylindAxisym",
    "as_scalar_field_cylindrical",
    "as_scalar_field_cylind",
    "as_vector_field_cartesian",
    "as_vector_field_cylindrical",
    "as_vector_field_cylind",
    "validate_phi_grid",
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
    "VectorFieldCylindAxisym",
]
