"""pyna.fields — unified field hierarchy for scalar, vector, and tensor fields.

New hierarchy replacing:
  - pyna.field_data.CylindricalScalarField / CylindricalVectorField
  - pyna.system.VectorField3D / VectorField3DAxiSymmetric

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
    ScalarField3DCylindrical,
    VectorField3DCylindrical,
    ScalarField3DAxiSymmetric,
    VectorField3DAxiSymmetric,
)
from pyna.fields.diff_ops import (
    gradient,
    divergence,
    curl,
    laplacian,
    hessian,
    jacobian_field,
    field_line_curvature,
    covariant_derivative_of_vector,
    riemann_tensor,
    ricci_tensor,
    ricci_scalar,
    strain_rate_tensor,
    helmholtz_decomposition,
)
from pyna.fields.tensor import TensorField3DRank2, TensorField4DRank2
from pyna.fields.coords import (
    CoordsCartesian,
    Coords3DCylindrical,
    Coords3DSpherical,
    Coords3DToroidal,
    Coords4DMinkowski,
    Coords4DSchwarzschild,
    Coords4DKerr,
)

__all__ = [
    "Field", "ScalarField", "VectorField", "TensorField",
    "ScalarField1D", "ScalarField2D", "ScalarField3D", "ScalarField4D",
    "VectorField1D", "VectorField2D", "VectorField3D", "VectorField4D",
    "TensorField3DRank2", "TensorField4DRank2",
    "FieldProperty",
    "ScalarField3DCylindrical", "VectorField3DCylindrical",
    "ScalarField3DAxiSymmetric", "VectorField3DAxiSymmetric",
    "gradient", "divergence", "curl", "laplacian",
    "hessian", "jacobian_field", "field_line_curvature",
    "covariant_derivative_of_vector",
    "riemann_tensor", "ricci_tensor", "ricci_scalar",
    "strain_rate_tensor", "helmholtz_decomposition",
    # coordinate systems
    "CoordsCartesian",
    "Coords3DCylindrical",
    "Coords3DSpherical",
    "Coords3DToroidal",
    "Coords4DMinkowski",
    "Coords4DSchwarzschild",
    "Coords4DKerr",
]
