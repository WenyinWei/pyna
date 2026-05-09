"""Compatibility re-export for fixed-section toroidal field helpers.

The canonical field hierarchy lives in :mod:`pyna.fields`.  This module remains
only so historical ``pyna.topo.field`` imports continue to resolve.
"""

from pyna.fields import ScalarField, TensorField
from pyna.fields.toroidal import (
    MU0,
    AxisymmetricField,
    Equilibrium,
    EquilibriumLike,
    ToroidalField,
    VectorFieldCylind,
    VectorFieldCylindAxisym,
    compute_J_by_curl,
)
from pyna.fields.base import VectorField

__all__ = [
    "ScalarField",
    "VectorField",
    "TensorField",
    "VectorFieldCylind",
    "VectorFieldCylindAxisym",
    "ToroidalField",
    "AxisymmetricField",
    "Equilibrium",
    "EquilibriumLike",
    "compute_J_by_curl",
    "MU0",
]
