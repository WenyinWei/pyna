"""pyna.fields — unified field hierarchy for scalar, vector, and tensor fields."""
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
    CylindricalFieldArrays,
    ScalarFieldCylind,
    ScalarFieldCylindAxisym,
    VectorFieldCylind,
    VectorFieldCylindAxisym,
    as_scalar_field_cylindrical,
    as_scalar_field_cylind,
    as_vector_field_cylindrical,
    as_vector_field_cylind,
    validate_phi_grid,
)
from pyna.fields.cartesian import (
    VectorFieldCartesian,
    as_vector_field_cartesian,
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
from pyna.fields.tensor import (
    Tensor2FieldCylind,
    Tensor2FieldCylindAxisym,
    TensorField4DRank2,
)
from pyna.fields.toroidal import (
    ToroidalField,
    AxisymmetricField,
    Equilibrium,
    EquilibriumLike,
    compute_J_by_curl,
    MU0,
)
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
    "Tensor2FieldCylind", "Tensor2FieldCylindAxisym",
    "TensorField4DRank2",
    "FieldProperty",
    "VectorFieldCartesian", "as_vector_field_cartesian",
    "CylindricalFieldArrays",
    "ScalarFieldCylind", "ScalarFieldCylindAxisym",
    "VectorFieldCylind", "VectorFieldCylindAxisym",
    "ToroidalField", "AxisymmetricField",
    "Equilibrium", "EquilibriumLike",
    "as_scalar_field_cylindrical", "as_scalar_field_cylind",
    "as_vector_field_cylindrical", "as_vector_field_cylind",
    "validate_phi_grid",
    "compute_J_by_curl", "MU0",
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
