"""Cylindrical-grid field classes for MCF coil computations.

This module re-exports the canonical field classes from
:mod:`pyna.fields.cylindrical`.  The redundant
``CylindricalGridVectorField3D`` / ``CylindricalGridAxiVectorField3D``
classes (and the misspelled ``RegualrCylindricalGridField`` alias) have
been eliminated — all code should use the canonical names below.

Canonical class map
-------------------

+----------------------------------------------+----------------------------+
| Old name (eliminated)                        | Canonical class            |
+==============================================+============================+
| ``CylindricalGridVectorField3D``             | ``VectorFieldCylind``       |
+----------------------------------------------+----------------------------+
| ``CylindricalGridAxiVectorField3D``          | ``VectorFieldCylindAxisym`` |
+----------------------------------------------+----------------------------+
| ``RegualrCylindricalGridField`` (misspelled) | ``VectorFieldCylind``       |
+----------------------------------------------+----------------------------+
"""
from pyna.fields.cylindrical import (
    ScalarFieldCylind,
    VectorFieldCylind,
    VectorFieldCylindAxisym,
    ScalarFieldCylindAxisym,
)

__all__ = [
    "ScalarFieldCylind",
    "VectorFieldCylind",
    "VectorFieldCylindAxisym",
    "ScalarFieldCylindAxisym",
]
