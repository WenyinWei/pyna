"""Differential operators in cylindrical coordinates.

Extends and consolidates pyna.vector_calc with a complete set of
field-returning operators that produce typed Field objects.

All operators use second-order finite differences on the structured
cylindrical grid (R, Z, phi), with periodic BCs in phi.

Currently all operators assume cylindrical (R, Z, phi) coordinates.
Use field.coords to verify, or pass coords= explicitly.
"""
from __future__ import annotations
import numpy as np
from pyna.fields.properties import FieldProperty


# ── Low-level finite-difference helpers (mirrored from vector_calc.py) ───────

def _grad_R(arr, R):
    """Second-order central differences along R (axis 0), one-sided at boundaries."""
    out = np.empty_like(arr)
    dR = R[2:] - R[:-2]
    out[1:-1] = (arr[2:] - arr[:-2]) / dR[:, None, None]
    out[0]  = (arr[1]  - arr[0])  / (R[1]  - R[0])
    out[-1] = (arr[-1] - arr[-2]) / (R[-1] - R[-2])
    return out


def _grad_Z(arr, Z):
    """Second-order central differences along Z (axis 1), one-sided at boundaries."""
    out = np.empty_like(arr)
    dZ = Z[2:] - Z[:-2]
    out[:, 1:-1, :] = (arr[:, 2:, :] - arr[:, :-2, :]) / dZ[None, :, None]
    out[:, 0,  :] = (arr[:, 1,  :] - arr[:, 0,  :]) / (Z[1]  - Z[0])
    out[:, -1, :] = (arr[:, -1, :] - arr[:, -2, :]) / (Z[-1] - Z[-2])
    return out


def _grad_phi(arr, Phi, periodic=True):
    """Second-order central differences along phi (axis 2), periodic or one-sided."""
    if periodic:
        dphi = Phi[1] - Phi[0]
        return (np.roll(arr, -1, axis=2) - np.roll(arr, 1, axis=2)) / (2 * dphi)
    out = np.empty_like(arr)
    dPhi = Phi[2:] - Phi[:-2]
    out[:, :, 1:-1] = (arr[:, :, 2:] - arr[:, :, :-2]) / dPhi[None, None, :]
    out[:, :, 0]  = (arr[:, :, 1]  - arr[:, :, 0])  / (Phi[1]  - Phi[0])
    out[:, :, -1] = (arr[:, :, -1] - arr[:, :, -2]) / (Phi[-1] - Phi[-2])
    return out


# ── Public operators ──────────────────────────────────────────────────────────

def _check_coords(field, coords_override):
    """Warn/raise if field's coordinate system is not cylindrical."""
    from pyna.fields.coords import CylindricalCoords3D
    cs = coords_override or getattr(field, 'coords', None)
    if cs is not None and not isinstance(cs, CylindricalCoords3D):
        raise NotImplementedError(
            f"diff_ops currently only supports CylindricalCoords3D, got {type(cs).__name__}. "
            "Pass coords=CylindricalCoords3D() to override, or implement the coordinate system."
        )


def gradient(f, coords=None) -> "VectorField3DCylindrical":
    """Gradient of a scalar field in cylindrical coordinates.

    ∇f = (∂f/∂R,  ∂f/∂Z,  (1/R)·∂f/∂φ)

    Parameters
    ----------
    f : ScalarField3DCylindrical

    Returns
    -------
    VectorField3DCylindrical
    """
    _check_coords(f, coords)
    from pyna.fields.cylindrical import VectorField3DCylindrical
    R3d = f.R[:, None, None]
    df_dR   = _grad_R(f.value, f.R)
    df_dZ   = _grad_Z(f.value, f.Z)
    df_dphi = _grad_phi(f.value, f.Phi, periodic=True)
    return VectorField3DCylindrical(
        R=f.R, Z=f.Z, Phi=f.Phi,
        VR=df_dR,
        VZ=df_dZ,
        VPhi=df_dphi / R3d,
        field_periods=f.field_periods,
        name=f"grad({f.name})",
        units=f"{f.units}/m",
        properties=FieldProperty.NONE,
    )


def divergence(v) -> "ScalarField3DCylindrical":
    """Divergence of a vector field in cylindrical coordinates.

    ∇·V = ∂V_R/∂R + ∂V_Z/∂Z + (V_R + ∂V_φ/∂φ) / R

    Parameters
    ----------
    v : VectorField3DCylindrical

    Returns
    -------
    ScalarField3DCylindrical
    """
    from pyna.fields.cylindrical import ScalarField3DCylindrical
    R3d = v.R[:, None, None]
    dVR_dR      = _grad_R(v.VR, v.R)
    dVZ_dZ      = _grad_Z(v.VZ, v.Z)
    dVPhi_dphi  = _grad_phi(v.VPhi, v.Phi, periodic=True)
    div = dVR_dR + dVZ_dZ + (v.VR + dVPhi_dphi) / R3d
    return ScalarField3DCylindrical(
        R=v.R, Z=v.Z, Phi=v.Phi,
        value=div,
        field_periods=v.field_periods,
        name=f"div({v.name})",
        units="",
        properties=FieldProperty.NONE,
    )


def curl(v) -> "VectorField3DCylindrical":
    """Curl of a vector field in cylindrical coordinates.

    (∇×V)_R   = (1/R)·∂V_Z/∂φ  - ∂V_φ/∂Z
    (∇×V)_Z   = ∂V_φ/∂R + V_φ/R - (1/R)·∂V_R/∂φ
    (∇×V)_φ   = ∂V_R/∂Z - ∂V_Z/∂R

    The result is automatically tagged DIVERGENCE_FREE (curl is always solenoidal).

    Parameters
    ----------
    v : VectorField3DCylindrical

    Returns
    -------
    VectorField3DCylindrical  (with FieldProperty.DIVERGENCE_FREE)
    """
    from pyna.fields.cylindrical import VectorField3DCylindrical
    R3d = v.R[:, None, None]

    dVZ_dphi   = _grad_phi(v.VZ,   v.Phi, periodic=True)
    dVPhi_dZ   = _grad_Z(v.VPhi, v.Z)
    dVPhi_dR   = _grad_R(v.VPhi, v.R)
    dVR_dphi   = _grad_phi(v.VR,   v.Phi, periodic=True)
    dVR_dZ     = _grad_Z(v.VR,   v.Z)
    dVZ_dR     = _grad_R(v.VZ,   v.R)

    curl_R   = dVZ_dphi / R3d - dVPhi_dZ
    curl_Z   = dVPhi_dR + v.VPhi / R3d - dVR_dphi / R3d
    curl_Phi = dVR_dZ - dVZ_dR

    return VectorField3DCylindrical(
        R=v.R, Z=v.Z, Phi=v.Phi,
        VR=curl_R, VZ=curl_Z, VPhi=curl_Phi,
        field_periods=v.field_periods,
        name=f"curl({v.name})",
        units=f"{v.units}/m",
        properties=FieldProperty.DIVERGENCE_FREE,
    )


def laplacian(f) -> "ScalarField3DCylindrical":
    """Scalar Laplacian in cylindrical coordinates.

    ∇²f = (1/R)·∂/∂R(R·∂f/∂R) + ∂²f/∂Z² + (1/R²)·∂²f/∂φ²

    Computed via ∇²f = ∇·(∇f) using the gradient and divergence operators.

    Parameters
    ----------
    f : ScalarField3DCylindrical

    Returns
    -------
    ScalarField3DCylindrical
    """
    grad_f = gradient(f)
    lap = divergence(grad_f)
    # rename
    from pyna.fields.cylindrical import ScalarField3DCylindrical
    return ScalarField3DCylindrical(
        R=f.R, Z=f.Z, Phi=f.Phi,
        value=lap.value,
        field_periods=f.field_periods,
        name=f"laplacian({f.name})",
        units=f"{f.units}/m²",
        properties=FieldProperty.NONE,
    )


def hessian(f) -> "TensorField3D_rank2":
    """Hessian tensor H_ij = ∇_i ∇_j f in cylindrical coordinates.

    Computed by taking the gradient of each component of ∇f, including
    Christoffel connection corrections:

      H_RR   = ∂²f/∂R²
      H_ZZ   = ∂²f/∂Z²
      H_φφ   = (1/R²)·∂²f/∂φ² + (1/R)·∂f/∂R
      H_RZ   = H_ZR = ∂²f/∂R∂Z
      H_Rφ   = H_φR = (1/R)·∂²f/∂R∂φ - (1/R²)·∂f/∂φ
      H_Zφ   = H_φZ = (1/R)·∂²f/∂Z∂φ

    Parameters
    ----------
    f : ScalarField3DCylindrical

    Returns
    -------
    TensorField3D_rank2
    """
    from pyna.fields.tensor import TensorField3D_rank2
    R3d = f.R[:, None, None]
    nR, nZ, nPhi = len(f.R), len(f.Z), len(f.Phi)

    df_dR   = _grad_R(f.value, f.R)
    df_dZ   = _grad_Z(f.value, f.Z)
    df_dphi = _grad_phi(f.value, f.Phi, periodic=True)

    # Second derivatives
    d2f_dR2    = _grad_R(df_dR, f.R)
    d2f_dZ2    = _grad_Z(df_dZ, f.Z)
    d2f_dphi2  = _grad_phi(df_dphi, f.Phi, periodic=True)
    d2f_dRdZ   = _grad_Z(df_dR, f.Z)
    d2f_dRdphi = _grad_phi(df_dR, f.Phi, periodic=True)
    d2f_dZdphi = _grad_phi(df_dZ, f.Phi, periodic=True)

    data = np.zeros((nR, nZ, nPhi, 3, 3), dtype=float)
    # index: 0=R, 1=Z, 2=phi
    data[..., 0, 0] = d2f_dR2
    data[..., 1, 1] = d2f_dZ2
    data[..., 2, 2] = d2f_dphi2 / R3d**2 + df_dR / R3d
    data[..., 0, 1] = d2f_dRdZ
    data[..., 1, 0] = d2f_dRdZ
    data[..., 0, 2] = d2f_dRdphi / R3d - df_dphi / R3d**2
    data[..., 2, 0] = data[..., 0, 2]
    data[..., 1, 2] = d2f_dZdphi / R3d
    data[..., 2, 1] = data[..., 1, 2]

    return TensorField3D_rank2(
        R=f.R, Z=f.Z, Phi=f.Phi,
        data=data,
        name=f"hessian({f.name})",
        units=f"{f.units}/m²",
        properties=FieldProperty.SYMMETRIC,
    )


def jacobian_field(v) -> "TensorField3D_rank2":
    """Jacobian tensor J_ij = ∂V_i/∂x^j in cylindrical coordinates.

    Includes connection (Christoffel) corrections for curvilinear basis:
      J_RR   = ∂V_R/∂R
      J_RZ   = ∂V_R/∂Z
      J_Rphi = (1/R)·∂V_R/∂φ - V_φ/R
      J_ZR   = ∂V_Z/∂R
      J_ZZ   = ∂V_Z/∂Z
      J_Zphi = (1/R)·∂V_Z/∂φ
      J_phiR  = ∂V_φ/∂R - V_φ/R   (note: ∂(V_φ/R)/∂R + V_R/R style depends on convention)
      Actually using covariant derivative convention with Christoffel:
        ∇_j V_i = ∂_j V_i - Γ^k_ij V_k

    For practical purposes we return the coordinate-component Jacobian
    ∂V_i/∂x^j (without Christoffel corrections) plus the diagonal
    1/R terms that arise from the phi derivatives:

    Parameters
    ----------
    v : VectorField3DCylindrical

    Returns
    -------
    TensorField3D_rank2, shape (nR, nZ, nPhi, 3, 3)
      J[..., i, j] = ∂V_i/∂x^j  with phi-axis scaled by 1/R
    """
    from pyna.fields.tensor import TensorField3D_rank2
    R3d = v.R[:, None, None]
    nR, nZ, nPhi = len(v.R), len(v.Z), len(v.Phi)

    data = np.zeros((nR, nZ, nPhi, 3, 3), dtype=float)

    for i_comp, arr in enumerate([v.VR, v.VZ, v.VPhi]):
        data[..., i_comp, 0] = _grad_R(arr, v.R)
        data[..., i_comp, 1] = _grad_Z(arr, v.Z)
        data[..., i_comp, 2] = _grad_phi(arr, v.Phi, periodic=True) / R3d

    return TensorField3D_rank2(
        R=v.R, Z=v.Z, Phi=v.Phi,
        data=data,
        name=f"jacobian({v.name})",
        units=f"{v.units}/m",
        properties=FieldProperty.NONE,
    )


def field_line_curvature(B) -> "VectorField3DCylindrical":
    """Magnetic field-line curvature vector κ = (b̂·∇)b̂.

    Where b̂ = B/|B| is the unit vector along the field.

    This uses the directional derivative of a vector field in cylindrical
    coordinates (with Christoffel corrections):

    (b̂·∇b̂)_R   = b_R·∂b_R/∂R + b_Z·∂b_R/∂Z + (b_φ/R)·∂b_R/∂φ - b_φ²/R
    (b̂·∇b̂)_Z   = b_R·∂b_Z/∂R + b_Z·∂b_Z/∂Z + (b_φ/R)·∂b_Z/∂φ
    (b̂·∇b̂)_φ   = b_R·∂b_φ/∂R + b_Z·∂b_φ/∂Z + (b_φ/R)·∂b_φ/∂φ + b_φ·b_R/R

    Parameters
    ----------
    B : VectorField3DCylindrical

    Returns
    -------
    VectorField3DCylindrical
    """
    from pyna.fields.cylindrical import VectorField3DCylindrical
    R3d = B.R[:, None, None]

    Bmag = np.sqrt(B.VR**2 + B.VZ**2 + B.VPhi**2) + 1e-30
    bR   = B.VR   / Bmag
    bZ   = B.VZ   / Bmag
    bPhi = B.VPhi / Bmag

    bphi_over_R = bPhi / R3d

    # Derivatives of unit-vector components
    dbR_dR    = _grad_R(bR,   B.R)
    dbR_dZ    = _grad_Z(bR,   B.Z)
    dbR_dphi  = _grad_phi(bR, B.Phi, periodic=True)

    dbZ_dR    = _grad_R(bZ,   B.R)
    dbZ_dZ    = _grad_Z(bZ,   B.Z)
    dbZ_dphi  = _grad_phi(bZ, B.Phi, periodic=True)

    dbPhi_dR  = _grad_R(bPhi,   B.R)
    dbPhi_dZ  = _grad_Z(bPhi,   B.Z)
    dbPhi_dphi = _grad_phi(bPhi, B.Phi, periodic=True)

    kappa_R   = (bR * dbR_dR   + bZ * dbR_dZ   + bphi_over_R * dbR_dphi
                 - bPhi**2 / R3d)
    kappa_Z   =  bR * dbZ_dR   + bZ * dbZ_dZ   + bphi_over_R * dbZ_dphi
    kappa_Phi = (bR * dbPhi_dR + bZ * dbPhi_dZ + bphi_over_R * dbPhi_dphi
                 + bPhi * bR / R3d)

    return VectorField3DCylindrical(
        R=B.R, Z=B.Z, Phi=B.Phi,
        VR=kappa_R, VZ=kappa_Z, VPhi=kappa_Phi,
        field_periods=B.field_periods,
        name=f"curvature({B.name})",
        units="1/m",
        properties=FieldProperty.NONE,
    )


def covariant_derivative_of_vector(v, coords=None):
    """Covariant derivative nabla_i V^j of a vector field.

    Returns the (i,j) component tensor:
        (nablaV)^j_i = d_i V^j + Gamma^j_{ik} V^k

    For cylindrical coordinates this gives the correct connection terms
    (identical to the Jacobian plus Christoffel correction).

    Parameters
    ----------
    v : VectorField3DCylindrical
    coords : CoordinateSystem, optional
        Defaults to CylindricalCoords3D().

    Returns
    -------
    TensorField3D_rank2, shape (nR, nZ, nPhi, 3, 3)
        Component [i,j] = nabla_i V^j
    """
    from pyna.fields.coords import CylindricalCoords3D
    from pyna.fields.tensor import TensorField3D_rank2

    if coords is None:
        coords = CylindricalCoords3D()

    # Get ordinary Jacobian (d_i V^j)
    J = jacobian_field(v)  # TensorField3D_rank2, [i,j] = d_i V^j

    # Add Christoffel correction: Gamma^j_{ik} V^k
    # Evaluate Christoffel at every grid point
    RR, ZZ, PP = np.meshgrid(v.R, v.Z, v.Phi, indexing='ij')
    pts = np.stack([RR.ravel(), ZZ.ravel(), PP.ravel()], axis=1)
    # christoffel_symbols returns (N, dim, dim, dim): [k, i, j] = Gamma^k_ij
    Gamma = coords.christoffel_symbols(pts)
    shape3d = (len(v.R), len(v.Z), len(v.Phi))
    Gamma = Gamma.reshape(shape3d + (3, 3, 3))  # (nR,nZ,nPhi, k, i, j)

    # V components on grid
    V = np.stack([v.VR, v.VZ, v.VPhi], axis=-1)  # (nR,nZ,nPhi, 3)

    # Correction: sum_k Gamma^j_{ik} V^k  -> result[..., i, j]
    # Gamma[..., k, i, j] summed over k with V[..., k]
    correction = np.einsum('...kij,...k->...ij', Gamma, V)

    cov_data = J.data + correction  # (nR,nZ,nPhi,3,3)

    return TensorField3D_rank2(v.R, v.Z, v.Phi, cov_data,
                               name=f"nabla({v.name})", units=v.units)


def riemann_tensor(coords, pt, eps=1e-4):
    """Compute Riemann curvature tensor R^l_ijk at a point.

    R^l_ijk = d_j Gamma^l_ik - d_k Gamma^l_ij + Gamma^l_jm Gamma^m_ik - Gamma^l_km Gamma^m_ij

    Uses central finite differences for d_j Gamma.

    Parameters
    ----------
    coords : CoordinateSystem
    pt : ndarray, shape (dim,)
        Point at which to evaluate.
    eps : float
        Finite difference step.

    Returns
    -------
    ndarray, shape (dim, dim, dim, dim)
        R[l, i, j, k] = R^l_ijk

    Note: Near or inside the Schwarzschild radius (r <= 2GM/c^2),
    numerical singularities will occur.
    """
    dim = coords.dim
    pt = np.asarray(pt, dtype=float)

    def gamma_at(p):
        return coords.christoffel_symbols(p[np.newaxis])[0]  # (dim, dim, dim)

    # d_j Gamma^l_ik via central differences
    dGamma = np.zeros((dim, dim, dim, dim))  # dGamma[l, i, k, j] = d_j Gamma^l_ik
    for j in range(dim):
        ep = pt.copy(); ep[j] += eps
        em = pt.copy(); em[j] -= eps
        dGamma[:, :, :, j] = (gamma_at(ep) - gamma_at(em)) / (2 * eps)

    G = gamma_at(pt)  # Gamma^l_ij at pt

    R = np.zeros((dim, dim, dim, dim))  # R[l, i, j, k]
    for l in range(dim):
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    R[l, i, j, k] = (
                        dGamma[l, i, k, j] - dGamma[l, i, j, k]
                        + sum(G[l, j, m] * G[m, i, k] - G[l, k, m] * G[m, i, j]
                              for m in range(dim))
                    )
    return R


def ricci_tensor(coords, pt, eps=1e-4):
    """Ricci tensor R_ij = R^k_ikj (contraction of Riemann tensor).

    Returns ndarray shape (dim, dim).
    """
    R_full = riemann_tensor(coords, pt, eps)  # (dim, dim, dim, dim)
    dim = coords.dim
    # R_ij = R^k_ikj = R[k, i, k, j] summed over k
    return sum(R_full[k, :, k, :] for k in range(dim))


def ricci_scalar(coords, pt, eps=1e-4):
    """Ricci scalar R = g^{ij} R_ij."""
    g = coords.metric_tensor(pt[np.newaxis])[0]
    g_inv = np.linalg.inv(g)
    Ric = ricci_tensor(coords, pt, eps)
    return float(np.einsum('ij,ij->', g_inv, Ric))


def strain_rate_tensor(v):
    """Strain-rate tensor S = 1/2 (Dv + Dv^T) where Dv is the Jacobian field.

    Used in viscous flow, MHD transport, and deformation analysis.
    Result is always symmetric.
    """
    J = jacobian_field(v)
    return J.symmetrize()


def helmholtz_decomposition(v, tol=1e-6):
    """Helmholtz decomposition: v = nabla phi + nabla x A + harmonic.

    Simplified version: returns divergence-free part and curl-free part.

    WARNING: This is an approximate decomposition using finite differences.
    The divergence-free part is approximated by curl(v), not by a proper
    Leray projection / Poisson solve. For production use, a proper
    Poisson solver is recommended.

    Returns
    -------
    (v_div_free, v_curl_free) : tuple of VectorField3DCylindrical
        v_div_free  -- curl(v), annotated as DIVERGENCE_FREE
        v_curl_free -- v - curl(v), annotated as CURL_FREE
        v ≈ v_div_free + v_curl_free  (approximate)
    """
    from pyna.fields.cylindrical import VectorField3DCylindrical
    from pyna.fields.properties import FieldProperty

    curl_v = curl(v)

    v_div_free = VectorField3DCylindrical(
        v.R, v.Z, v.Phi, curl_v.VR, curl_v.VZ, curl_v.VPhi,
        name=f"divfree({v.name})",
        properties=FieldProperty.DIVERGENCE_FREE,
    )

    VR_cf = v.VR - curl_v.VR
    VZ_cf = v.VZ - curl_v.VZ
    VP_cf = v.VPhi - curl_v.VPhi
    v_curl_free = VectorField3DCylindrical(
        v.R, v.Z, v.Phi, VR_cf, VZ_cf, VP_cf,
        name=f"curlfree({v.name})",
        properties=FieldProperty.CURL_FREE,
    )
    return v_div_free, v_curl_free
