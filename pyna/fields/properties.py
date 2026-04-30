"""Field property tags for communicating mathematical structure."""
from enum import auto, Flag


class FieldProperty(Flag):
    """Bit-flags describing mathematical properties of a field.

    Multiple properties can be combined: e.g.
    ``FieldProperty.DIVERGENCE_FREE | FieldProperty.IRROTATIONAL``
    """
    NONE              = 0
    DIVERGENCE_FREE   = auto()  # ∇·F = 0  (magnetic fields, incompressible velocity)
    CURL_FREE         = auto()  # ∇×F = 0  (electrostatic field, gradient fields)
    IRROTATIONAL      = CURL_FREE          # alias
    SOLENOIDAL        = DIVERGENCE_FREE    # alias
    CONSERVATIVE      = auto()  # F = -∇φ  (implies CURL_FREE)
    HARMONIC          = auto()  # ∇²φ = 0  (both div-free and curl-free potential)
    SYMMETRIC         = auto()  # tensor: T_ij = T_ji
    ANTISYMMETRIC     = auto()  # tensor: T_ij = -T_ji
    POSITIVE_DEFINITE = auto()  # tensor: x^T T x > 0 for all x ≠ 0
