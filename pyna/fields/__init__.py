"""pyna.fields — unified field hierarchy for scalar, vector, and tensor fields.

New hierarchy replacing:
  - pyna.field_data.CylindricalScalarField / CylindricalVectorField
  - pyna.system.VectorField3D / AxiSymmetricVectorField3D
  - pyna.field.RegualrCylindricalGridField / CylindricalGridVectorField3D

All old names remain importable via backward-compat aliases in their
original modules. New code should import from pyna.fields directly.
"""
from pyna.fields.properties import FieldProperty
from pyna.fields.base import (
    Field,
    ScalarField,
    VectorField,
    TensorField,
    ScalarField1D, ScalarField2D, ScalarField3D, ScalarField4D,
    VectorField1D, VectorField2D, VectorField3D, VectorField4D,
)
from pyna.fields.cylindrical import (
    CylindricalScalarField3D,
    CylindricalVectorField3D,
    AxiSymmetricScalarField3D,
    AxiSymmetricVectorField3D,
)
from pyna.fields.diff_ops import (
    gradient,
    divergence,
    curl,
    laplacian,
    hessian,
    jacobian_field,
    field_line_curvature,
)
from pyna.fields.tensor import TensorField3D_rank2
from pyna.fields.coords import (
    CartesianCoords,
    CylindricalCoords3D,
    SphericalCoords3D,
    TorCoords3D,
    MinkowskiCoords4D,
    SchwarzschildCoords4D,
    KerrCoords4D,
)

__all__ = [
    "Field", "ScalarField", "VectorField", "TensorField",
    "ScalarField1D", "ScalarField2D", "ScalarField3D", "ScalarField4D",
    "VectorField1D", "VectorField2D", "VectorField3D", "VectorField4D",
    "TensorField3D_rank2",
    "FieldProperty",
    "CylindricalScalarField3D", "CylindricalVectorField3D",
    "AxiSymmetricScalarField3D", "AxiSymmetricVectorField3D",
    "gradient", "divergence", "curl", "laplacian",
    "hessian", "jacobian_field", "field_line_curvature",
    # coordinate systems
    "CartesianCoords",
    "CylindricalCoords3D",
    "SphericalCoords3D",
    "TorCoords3D",
    "MinkowskiCoords4D",
    "SchwarzschildCoords4D",
    "KerrCoords4D",
]
