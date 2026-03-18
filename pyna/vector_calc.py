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

from pyna.fields.cylindrical import VectorField3DCylindrical, ScalarField3DCylindrical





def magnitude(v: VectorField3DCylindrical) -> np.ndarray:

    """Pointwise magnitude |V| = sqrt(VR² + VZ² + VPhi²)."""

    return np.sqrt(v.VR**2 + v.VZ**2 + v.VPhi**2)





def cross(v1: VectorField3DCylindrical, v2: VectorField3DCylindrical) -> VectorField3DCylindrical:

    """Cross product v1 × v2 in cylindrical coordinates.



    (v1 × v2)_R   = v1_Z * v2_Phi - v1_Phi * v2_Z

    (v1 × v2)_Z   = v1_Phi * v2_R  - v1_R  * v2_Phi

    (v1 × v2)_Phi = v1_R  * v2_Z   - v1_Z  * v2_R

    """

    return VectorField3DCylindrical(

        R=v1.R, Z=v1.Z, Phi=v1.Phi, field_periods=v1.field_periods,

        VR=v1.VZ * v2.VPhi - v1.VPhi * v2.VZ,

        VZ=v1.VPhi * v2.VR - v1.VR * v2.VPhi,

        VPhi=v1.VR * v2.VZ - v1.VZ * v2.VR,

    )





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





def divergence(v: VectorField3DCylindrical) -> ScalarField3DCylindrical:

    """Divergence ∇·V in cylindrical coords using 2nd-order finite differences.



    ∇·V = ∂_R V_R + ∂_Z V_Z + (V_R + ∂_phi V_phi) / R



    Boundary cells use one-sided differences.

    Phi direction uses periodic BCs.

    """

    R3d = v.R[:, None, None]  # broadcast shape (nR, 1, 1)



    dVR_dR = _grad_R(v.VR, v.R)

    dVZ_dZ = _grad_Z(v.VZ, v.Z)

    dVPhi_dphi = _grad_phi(v.VPhi, v.Phi, periodic=True)



    div = dVR_dR + dVZ_dZ + (v.VR + dVPhi_dphi) / R3d



    return ScalarField3DCylindrical(

        R=v.R, Z=v.Z, Phi=v.Phi,

        value=div,

        field_periods=v.field_periods,

        name=f"div({v.name})",

        units=""

    )





def directional_derivative_of_scalar(

    v: VectorField3DCylindrical, s: ScalarField3DCylindrical

) -> ScalarField3DCylindrical:

    """v·∇s in cylindrical coordinates.



    v·∇s = v_R ∂_R s + v_Z ∂_Z s + (v_phi/R) ∂_phi s

    """

    R3d = v.R[:, None, None]



    ds_dR = _grad_R(s.value, s.R)

    ds_dZ = _grad_Z(s.value, s.Z)

    ds_dphi = _grad_phi(s.value, s.Phi, periodic=True)



    result = v.VR * ds_dR + v.VZ * ds_dZ + (v.VPhi / R3d) * ds_dphi



    return ScalarField3DCylindrical(

        R=s.R, Z=s.Z, Phi=s.Phi,

        value=result,

        field_periods=s.field_periods,

        name=f"({v.name})·∇({s.name})",

        units=s.units

    )





def directional_derivative_of_vector(

    v1: VectorField3DCylindrical, v2: VectorField3DCylindrical

) -> VectorField3DCylindrical:

    """v1·∇v2 in cylindrical coordinates (includes Christoffel terms).



    (v1·∇v2)_R   = v1_R ∂_R v2_R + v1_Z ∂_Z v2_R + (v1_phi/R) ∂_phi v2_R - v1_phi * v2_phi / R

    (v1·∇v2)_Z   = v1_R ∂_R v2_Z + v1_Z ∂_Z v2_Z + (v1_phi/R) ∂_phi v2_Z

    (v1·∇v2)_phi = v1_R ∂_R v2_phi + v1_Z ∂_Z v2_phi + (v1_phi/R) ∂_phi v2_phi + v1_phi * v2_R / R



    The last terms in R and phi are Christoffel correction terms arising because

    cylindrical basis vectors ê_R, ê_phi depend on phi.

    """

    R3d = v1.R[:, None, None]

    v1phi_over_R = v1.VPhi / R3d



    dv2R_dR = _grad_R(v2.VR, v2.R)

    dv2R_dZ = _grad_Z(v2.VR, v2.Z)

    dv2R_dphi = _grad_phi(v2.VR, v2.Phi)



    dv2Z_dR = _grad_R(v2.VZ, v2.R)

    dv2Z_dZ = _grad_Z(v2.VZ, v2.Z)

    dv2Z_dphi = _grad_phi(v2.VZ, v2.Phi)



    dv2Phi_dR = _grad_R(v2.VPhi, v2.R)

    dv2Phi_dZ = _grad_Z(v2.VPhi, v2.Z)

    dv2Phi_dphi = _grad_phi(v2.VPhi, v2.Phi)



    res_R = (v1.VR * dv2R_dR + v1.VZ * dv2R_dZ + v1phi_over_R * dv2R_dphi

             - v1.VPhi * v2.VPhi / R3d)



    res_Z = v1.VR * dv2Z_dR + v1.VZ * dv2Z_dZ + v1phi_over_R * dv2Z_dphi



    res_Phi = (v1.VR * dv2Phi_dR + v1.VZ * dv2Phi_dZ + v1phi_over_R * dv2Phi_dphi

               + v1.VPhi * v2.VR / R3d)



    return VectorField3DCylindrical(

        R=v1.R, Z=v1.Z, Phi=v1.Phi,

        VR=res_R, VZ=res_Z, VPhi=res_Phi,

        field_periods=v1.field_periods,

        name=f"({v1.name})·∇({v2.name})"

    )

