"""Cylindrical coordinate vector calculus operations.

Ported from Jynamics.jl (Juna.jl) notebook cells:

  divergence(), magnitude(), cross(),

  directional_derivative_along_v_of_s(),

  directional_derivative_along_v1_of_v2()

All operations use second-order finite differences on the structured grid,

with periodic boundary conditions in phi (and optional periodicity in R, Z).

Reference:

  ∇·V = ∂_R V_R + ∂_Z V_Z + (V_R + ∂_phi V_phi) / R

"""

import numpy as np

from pyna.fields.cylindrical import (
    VectorFieldCylind,
    ScalarFieldCylind,
    _binary_grid,
    _broadcast_vector_components,
    _make_vector_result,
)

def magnitude(v: VectorFieldCylind) -> ScalarFieldCylind:

    """Pointwise magnitude |V| as a scalar field."""

    return v.magnitude()

def cross(v1: VectorFieldCylind, v2: VectorFieldCylind) -> VectorFieldCylind:

    """Cross product v1 × v2 in cylindrical coordinates.

    (v1 × v2)_R   = v1_Phi * v2_Z - v1_Z * v2_Phi

    (v1 × v2)_Z   = v1_R * v2_Phi  - v1_Phi * v2_R

    (v1 × v2)_Phi = v1_Z * v2_R    - v1_R * v2_Z

    """

    return v1.cross(v2)

def _grad_R(arr, R):

    """Second-order central differences along R (axis 0), one-sided at boundaries."""

    out = np.empty_like(arr)

    # interior: central diff

    dR = R[2:] - R[:-2]  # shape (nR-2,)

    out[1:-1] = (arr[2:] - arr[:-2]) / dR[:, None, None]

    # boundaries: one-sided

    out[0] = (arr[1] - arr[0]) / (R[1] - R[0])

    out[-1] = (arr[-1] - arr[-2]) / (R[-1] - R[-2])

    return out

def _grad_Z(arr, Z):

    """Second-order central differences along Z (axis 1), one-sided at boundaries."""

    out = np.empty_like(arr)

    dZ = Z[2:] - Z[:-2]  # shape (nZ-2,)

    out[:, 1:-1, :] = (arr[:, 2:, :] - arr[:, :-2, :]) / dZ[None, :, None]

    out[:, 0, :] = (arr[:, 1, :] - arr[:, 0, :]) / (Z[1] - Z[0])

    out[:, -1, :] = (arr[:, -1, :] - arr[:, -2, :]) / (Z[-1] - Z[-2])

    return out

def _grad_phi(arr, Phi, periodic=True):

    """Second-order central differences along phi (axis 2), periodic BCs."""

    if len(Phi) < 2:
        return np.zeros_like(arr)

    out = np.empty_like(arr)

    if periodic:

        # uniform spacing assumed; use roll

        dphi = Phi[1] - Phi[0]  # assumes uniform grid

        out = (np.roll(arr, -1, axis=2) - np.roll(arr, 1, axis=2)) / (2 * dphi)

    else:

        dPhi = Phi[2:] - Phi[:-2]

        out[:, :, 1:-1] = (arr[:, :, 2:] - arr[:, :, :-2]) / dPhi[None, None, :]

        out[:, :, 0] = (arr[:, :, 1] - arr[:, :, 0]) / (Phi[1] - Phi[0])

        out[:, :, -1] = (arr[:, :, -1] - arr[:, :, -2]) / (Phi[-1] - Phi[-2])

    return out

def divergence(v: VectorFieldCylind) -> ScalarFieldCylind:

    """Divergence ∇·V in cylindrical coords using 2nd-order finite differences.

    ∇·V = ∂_R V_R + ∂_Z V_Z + (V_R + ∂_phi V_phi) / R

    Boundary cells use one-sided differences.

    Phi direction uses periodic BCs.

    """

    return v.div()

def directional_derivative_of_scalar(

    v: VectorFieldCylind, s: ScalarFieldCylind

) -> ScalarFieldCylind:

    """v·∇s in cylindrical coordinates.

    v·∇s = v_R ∂_R s + v_Z ∂_Z s + (v_phi/R) ∂_phi s

    """

    return v.dot(s.grad())

def directional_derivative_of_vector(

    v1: VectorFieldCylind, v2: VectorFieldCylind

) -> VectorFieldCylind:

    """v1·∇v2 in cylindrical coordinates (includes Christoffel terms).

    (v1·∇v2)_R   = v1_R ∂_R v2_R + v1_Z ∂_Z v2_R + (v1_phi/R) ∂_phi v2_R - v1_phi * v2_phi / R

    (v1·∇v2)_Z   = v1_R ∂_R v2_Z + v1_Z ∂_Z v2_Z + (v1_phi/R) ∂_phi v2_Z

    (v1·∇v2)_phi = v1_R ∂_R v2_phi + v1_Z ∂_Z v2_phi + (v1_phi/R) ∂_phi v2_phi + v1_phi * v2_R / R

    The last terms in R and phi are Christoffel correction terms arising because

    cylindrical basis vectors ê_R, ê_phi depend on phi.

    """

    R, Z, Phi, axisym, section = _binary_grid(v1, v2)
    v1R, v1Z, v1Phi = _broadcast_vector_components(v1, Phi)
    v2R, v2Z, v2Phi = _broadcast_vector_components(v2, Phi)
    R3d = R[:, None, None]

    v1phi_over_R = v1Phi / R3d

    dv2R_dR = _grad_R(v2R, R)
    dv2R_dZ = _grad_Z(v2R, Z)
    dv2R_dphi = _grad_phi(v2R, Phi)

    dv2Z_dR = _grad_R(v2Z, R)
    dv2Z_dZ = _grad_Z(v2Z, Z)
    dv2Z_dphi = _grad_phi(v2Z, Phi)

    dv2Phi_dR = _grad_R(v2Phi, R)
    dv2Phi_dZ = _grad_Z(v2Phi, Z)
    dv2Phi_dphi = _grad_phi(v2Phi, Phi)

    res_R = (v1R * dv2R_dR + v1Z * dv2R_dZ + v1phi_over_R * dv2R_dphi
             - v1Phi * v2Phi / R3d)
    res_Z = v1R * dv2Z_dR + v1Z * dv2Z_dZ + v1phi_over_R * dv2Z_dphi
    res_Phi = (v1R * dv2Phi_dR + v1Z * dv2Phi_dZ + v1phi_over_R * dv2Phi_dphi
               + v1Phi * v2R / R3d)

    return _make_vector_result(
        R, Z, Phi, res_R, res_Z, res_Phi,
        axisym=axisym,
        section=section,
        name=f"({v1.name})·∇({v2.name})",
    )

