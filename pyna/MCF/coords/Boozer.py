"""Boozer magnetic coordinates.

Boozer coordinates (ψ, θ_B, φ) satisfy the covariant form:
    B = I(ψ) ∇φ + G(ψ) ∇θ_B    [approximate covariant form for tokamak]

Key properties:
    - Field lines are straight in (θ_B, φ) space (like PEST).
    - B_θ/B_φ = q(ψ) is constant on flux surfaces.
    - The toroidal current density satisfies J·∇φ = const(ψ).

Construction from PEST mesh
---------------------------
For axisymmetric equilibria the Boozer angle differs from the PEST angle by
a periodic function:

    θ_B = θ* + λ(ψ, θ*)

where λ satisfies (on each flux surface):

    ∂λ/∂θ* = <sqrt(g)> / sqrt(g) - 1

with sqrt(g) the Jacobian of PEST coordinates and <·> its flux-surface
average.  This ensures that the Jacobian of Boozer coordinates is a flux-
surface quantity (the Boozer condition).

References
----------
Boozer, Phys. Fluids 23, 904 (1980).
D'haeseleer et al., "Flux Coordinates and Magnetic Field Structure",
    Springer (1991), Chapter 6.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d


def _compute_PEST_jacobian(S, TET, R_mesh, Z_mesh):
    """Compute the PEST Jacobian sqrt(g) on the (S, TET) mesh.

    In PEST coordinates, the Jacobian is
        sqrt(g) = -(g_1 × g_2) · ê_φ  · R
    where g_1 = ∂r/∂S, g_2 = ∂r/∂θ* are the covariant basis vectors.

    Returns
    -------
    jac : ndarray, shape (ns, ntheta)
        Jacobian sqrt(g).  Boundary rows are NaN.
    """
    ns, ntheta = R_mesh.shape

    # ∂R/∂S, ∂Z/∂S (central differences, NaN at boundaries)
    dRdS = np.full((ns, ntheta), np.nan)
    dZdS = np.full((ns, ntheta), np.nan)
    dRdS[1:-1, :] = (R_mesh[2:, :] - R_mesh[:-2, :]) / (S[2:] - S[:-2])[:, None]
    dZdS[1:-1, :] = (Z_mesh[2:, :] - Z_mesh[:-2, :]) / (S[2:] - S[:-2])[:, None]

    # ∂R/∂θ*, ∂Z/∂θ* (central differences, periodic in θ*)
    dRdTET = np.empty((ns, ntheta))
    dZdTET = np.empty((ns, ntheta))
    dRdTET[:, 1:-1] = (R_mesh[:, 2:] - R_mesh[:, :-2]) / (TET[2:] - TET[:-2])[None, :]
    dZdTET[:, 1:-1] = (Z_mesh[:, 2:] - Z_mesh[:, :-2]) / (TET[2:] - TET[:-2])[None, :]
    dTET_wrap = TET[-1] - TET[-2] + TET[1] - TET[0]
    dRdTET[:, 0] = dRdTET[:, -1] = (R_mesh[:, 1] - R_mesh[:, -2]) / dTET_wrap
    dZdTET[:, 0] = dZdTET[:, -1] = (Z_mesh[:, 1] - Z_mesh[:, -2]) / dTET_wrap

    # sqrt(g) = R * (∂R/∂S * ∂Z/∂θ* - ∂R/∂θ* * ∂Z/∂S)
    jac = R_mesh * (dRdS * dZdTET - dRdTET * dZdS)
    return jac


def build_Boozer_mesh(S, TET, R_mesh, Z_mesh, q_iS,
                      B_R=None, B_Z=None, B_Phi=None,
                      equilibrium=None,
                      n_theta: int = 181):
    """Build Boozer coordinate mesh from PEST mesh.

    Parameters
    ----------
    S : ndarray, shape (ns,)
        PEST flux labels (sqrt of normalised flux).
    TET : ndarray, shape (ntheta,)
        PEST poloidal angles (0 to 2π inclusive).
    R_mesh, Z_mesh : ndarray, shape (ns, ntheta)
        PEST mesh in cylindrical (R, Z).
    q_iS : ndarray, shape (ns,)
        Safety factor profile.
    B_R, B_Z, B_Phi : ndarray or None
        Magnetic field components on the PEST mesh.
        At least one of these or ``equilibrium`` must be provided.
        If None, ``equilibrium.BR_BZ`` and ``equilibrium.Bphi`` are used.
    equilibrium : EquilibriumSolovev or compatible, optional
        Used to compute B fields when B_R / B_Z / B_Phi are None.
    n_theta : int
        Number of poloidal points in the output.

    Returns
    -------
    S : ndarray, shape (ns,)
    TET_B : ndarray, shape (n_theta,)
        Boozer poloidal angles (0 to 2π inclusive).
    R_mesh_B, Z_mesh_B : ndarray, shape (ns, n_theta)
        Boozer mesh.
    lambda_correction : ndarray, shape (ns, ntheta)
        Angle correction λ(ψ, θ*) = θ_B − θ*.
    """
    ns, ntheta = R_mesh.shape

    # ------------------------------------------------------------------
    # 1. Compute B field on PEST mesh if not provided
    # ------------------------------------------------------------------
    if B_R is None or B_Z is None or B_Phi is None:
        if equilibrium is None:
            raise ValueError("Either B_R/B_Z/B_Phi or equilibrium must be provided.")
        BR, BZ = equilibrium.BR_BZ(R_mesh, Z_mesh)
        BPhi = equilibrium.Bphi(R_mesh)
    else:
        BR = np.asarray(B_R)
        BZ = np.asarray(B_Z)
        BPhi = np.asarray(B_Phi)

    # ------------------------------------------------------------------
    # 2. Compute PEST Jacobian
    # ------------------------------------------------------------------
    jac = _compute_PEST_jacobian(S, TET, R_mesh, Z_mesh)

    # ------------------------------------------------------------------
    # 3. Compute angle correction λ per flux surface
    #    ∂λ/∂θ* = <jac> / jac - 1
    #    where <jac> is the flux-surface average (1/2π) ∫₀²π jac dθ*
    # ------------------------------------------------------------------
    lambda_correction = np.zeros((ns, ntheta))

    for i in range(1, ns - 1):  # skip axis and LCFS boundary (NaN Jacobian)
        jac_s = jac[i, :]  # shape (ntheta,)
        if np.any(~np.isfinite(jac_s)):
            continue

        # Flux-surface average of Jacobian
        jac_avg = np.trapezoid(jac_s, TET) / (2 * np.pi)
        if abs(jac_avg) < 1e-30:
            continue

        # ∂λ/∂θ* integrand
        integrand = jac_avg / jac_s - 1.0

        # Integrate with periodic boundary condition
        # λ(θ*=0) = 0  (choice of origin)
        lam = np.zeros(ntheta)
        lam[1:] = cumulative_trapezoid(integrand, TET)
        # Enforce exact periodicity: λ(2π) should equal λ(0) = 0
        # (The integral of integrand over [0,2π] should be 0 by construction)
        # Correct any numerical drift linearly
        drift = lam[-1]  # should be 0
        lam -= np.linspace(0, drift, ntheta)

        lambda_correction[i, :] = lam

    # ------------------------------------------------------------------
    # 4. Build new Boozer angle and remap mesh points
    # ------------------------------------------------------------------
    TET_B_out = np.linspace(0, 2 * np.pi, n_theta, endpoint=True)
    R_mesh_B = np.empty((ns, n_theta))
    Z_mesh_B = np.empty((ns, n_theta))

    for i in range(ns):
        R_s = R_mesh[i, :]
        Z_s = Z_mesh[i, :]
        lam = lambda_correction[i, :]

        # Axis or LCFS: copy as-is
        if np.allclose(R_s, R_s[0]) and np.allclose(Z_s, Z_s[0]):
            R_mesh_B[i, :] = R_s[0]
            Z_mesh_B[i, :] = Z_s[0]
            continue

        # Boozer angle at each PEST grid point
        TET_B_pest = TET + lam  # θ_B on the PEST grid

        # Sort by Boozer angle (should be monotone, but enforce it)
        sort_idx = np.argsort(TET_B_pest)
        TET_B_sorted = TET_B_pest[sort_idx]
        R_sorted = R_s[sort_idx]
        Z_sorted = Z_s[sort_idx]

        # Interpolate R, Z to the uniform Boozer grid
        R_mesh_B[i] = interp1d(TET_B_sorted, R_sorted,
                                kind='linear', fill_value='extrapolate')(TET_B_out)
        Z_mesh_B[i] = interp1d(TET_B_sorted, Z_sorted,
                                kind='linear', fill_value='extrapolate')(TET_B_out)

    return S, TET_B_out, R_mesh_B, Z_mesh_B, lambda_correction
