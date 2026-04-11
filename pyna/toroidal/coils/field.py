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
| ``CylindricalGridVectorField3D``             | ``VectorField3DCylindrical``|
+----------------------------------------------+----------------------------+
| ``CylindricalGridAxiVectorField3D``          | ``VectorField3DAxiSymmetric``|
+----------------------------------------------+----------------------------+
| ``RegualrCylindricalGridField`` (misspelled) | ``VectorField3DCylindrical``|
+----------------------------------------------+----------------------------+
"""
from pyna.fields.cylindrical import (
    ScalarField3DCylindrical,
    VectorField3DCylindrical,
    ScalarField3DAxiSymmetric,
    VectorField3DAxiSymmetric,
)

__all__ = [
    "ScalarField3DCylindrical",
    "VectorField3DCylindrical",
    "ScalarField3DAxiSymmetric",
    "VectorField3DAxiSymmetric",
]
