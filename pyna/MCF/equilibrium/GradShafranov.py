"""
Perturbed Grad-Shafranov solver for axisymmetric plasma equilibria.

This module provides tools for computing first-order plasma responses to
external magnetic perturbations, based on the linearised ideal-MHD force
balance:

    δJ × B₀ + J₀ × δB_plasma − ∇δp = −J₀ × δB_ext

supplemented by Ampère's law:

    ∇ × δB_plasma = μ₀ δJ

and the divergence-free constraint:

    ∇ · δB_plasma = 0

The problem is discretised on a regular (R, Z) grid and solved as a sparse
least-squares system.

Functions
---------
recover_pressure_simplest
    Recover the pressure field from its gradient by 2-D integration.
solve_GS_perturbed
    Full perturbed Grad-Shafranov solver (returns δB, δJ, δp).
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr, bicgstab

from pyna.fields.cylindrical import VectorField3DAxiSymmetric, ScalarField3DAxiSymmetric


# ---------------------------------------------------------------------------
# Pressure recovery
# ---------------------------------------------------------------------------

def recover_pressure_simplest(grad_p):
    """Recover the pressure from its gradient by simple path integration.

    The pressure is estimated by integrating the gradient along four paths
    (R-only forward/backward, Z-only forward/backward) and averaging the
    results.  This is the simplest possible approach and is only accurate
    when the gradient field is curl-free.

    Parameters
    ----------
    grad_p : VectorField3DAxiSymmetric or compatible
        Gradient of the pressure.  Must expose ``.R``, ``.Z``, ``.BR``
        (radial component ∂p/∂R) and ``.BZ`` (axial component ∂p/∂Z),
        each of shape (nR, nZ).

    Returns
    -------
    delta_p : ScalarField3DAxiSymmetric
        Recovered pressure perturbation on the same (R, Z) grid.
    """
    R, Z = grad_p.R, grad_p.Z
    nR, nZ = len(R), len(Z)
    dR, dZ = R[1] - R[0], Z[1] - Z[0]

    # Path 1: integrate ∂p/∂R from Rmin to Rmax (forward in R)
    p_R_fwd = np.zeros((nR, nZ))
    for j in range(nZ):
        for i in range(1, nR):
            p_R_fwd[i, j] = (p_R_fwd[i - 1, j]
                             + 0.5 * (grad_p.BR[i, j] + grad_p.BR[i - 1, j]) * dR)

    # Path 2: integrate ∂p/∂Z from Zmin to Zmax (forward in Z)
    p_Z_fwd = np.zeros((nR, nZ))
    for i in range(nR):
        for j in range(1, nZ):
            p_Z_fwd[i, j] = (p_Z_fwd[i, j - 1]
                             + 0.5 * (grad_p.BZ[i, j] + grad_p.BZ[i, j - 1]) * dZ)

    # Path 3: integrate ∂p/∂R from Rmax to Rmin (backward in R)
    p_R_bwd = np.zeros((nR, nZ))
    for j in range(nZ):
        for i in range(nR - 2, -1, -1):
            p_R_bwd[i, j] = (p_R_bwd[i + 1, j]
                             - 0.5 * (grad_p.BR[i + 1, j] + grad_p.BR[i, j]) * dR)

    # Path 4: integrate ∂p/∂Z from Zmax to Zmin (backward in Z)
    p_Z_bwd = np.zeros((nR, nZ))
    for i in range(nR):
        for j in range(nZ - 2, -1, -1):
            p_Z_bwd[i, j] = (p_Z_bwd[i, j + 1]
                             - 0.5 * (grad_p.BZ[i, j + 1] + grad_p.BZ[i, j]) * dZ)

    delta_p = 0.25 * (p_R_fwd + p_Z_fwd + p_R_bwd + p_Z_bwd)
    return ScalarField3DAxiSymmetric(R, Z, delta_p)


# ---------------------------------------------------------------------------
# Perturbed Grad-Shafranov solver
# ---------------------------------------------------------------------------

def solve_GS_perturbed(
        B0: VectorField3DAxiSymmetric,
        J0: VectorField3DAxiSymmetric,
        p0: ScalarField3DAxiSymmetric,
        delta_B_ext: VectorField3DAxiSymmetric,
        x0):
    """Solve the linearised ideal-MHD equations for a perturbed equilibrium.

    Given an axisymmetric equilibrium (B₀, J₀, p₀) and an external magnetic
    perturbation δB_ext, solve the system:

        ∇ × δB_plasma = μ₀ δJ                     (Ampère)
        δJ × B₀ + J₀ × δB_plasma − ∇δp = −J₀ × δB_ext   (force balance)
        ∇ · δB_plasma = 0                          (divergence-free)

    with boundary conditions δB = 0, δJ = 0, δp = 0 at grid boundaries
    and in vacuum regions (where p₀ ≈ 0).

    The system is assembled as a sparse linear system and solved using
    BiCGSTAB (normal-equation form A^T A x = A^T b).

    Parameters
    ----------
    B0 : VectorField3DAxiSymmetric
        Background equilibrium magnetic field.
    J0 : VectorField3DAxiSymmetric
        Background equilibrium current density.
    p0 : ScalarField3DAxiSymmetric
        Background equilibrium pressure.  Used to identify vacuum regions.
    delta_B_ext : VectorField3DAxiSymmetric
        External magnetic field perturbation.
    x0 : ndarray
        Initial guess for the linear solver, shape (7 * nR * nZ,).
        The first 3*n entries are δB (R, φ, Z), next 3*n are δJ, last n are δp.

    Returns
    -------
    delta_B_plasma : VectorField3DAxiSymmetric
        Plasma response magnetic field perturbation.
    delta_J : VectorField3DAxiSymmetric
        Plasma response current density perturbation.
    delta_p : ScalarField3DAxiSymmetric
        Plasma response pressure perturbation.
    """
    mu0 = 4e-7 * np.pi  # vacuum permeability [T·m/A]

    R, Z = B0.R, B0.Z
    nR, nZ = len(R), len(Z)
    dR, dZ = R[1] - R[0], Z[1] - Z[0]
    n = nR * nZ

    # Number of equations per grid point
    nEq = 14

    # Physical component indices: R=0, φ=1, Z=2
    iR, iPHI, iZ = 0, 1, 2

    # --- Pre-compute force-balance RHS: −J₀ × δB_ext ---
    RHS_fb = -J0.cross(delta_B_ext)

    # --- Sparse matrix assembly ---
    A = lil_matrix((nEq * n, 7 * n))
    b = np.zeros(nEq * n)

    def idx(i, j):
        """Flatten (i, j) grid index."""
        return i * nZ + j

    boundary_i = {0, 1, 2, 3, 4, nR - 1, nR - 2, nR - 3, nR - 4, nR - 5}
    boundary_j = {0, 1, 2, nZ - 1, nZ - 2, nZ - 3}

    for i in range(nR):
        for j in range(nZ):
            k = idx(i, j)

            # Neighbour indices
            k_pR  = idx(i + 1, j); k_mR  = idx(i - 1, j)
            k_pZ  = idx(i, j + 1); k_mZ  = idx(i, j - 1)
            k_p2R = idx(i + 2, j); k_m2R = idx(i - 2, j)
            k_p2Z = idx(i, j + 2); k_m2Z = idx(i, j - 2)
            k_p3R = idx(i + 3, j); k_m3R = idx(i - 3, j)
            k_p3Z = idx(i, j + 3); k_m3Z = idx(i, j - 3)

            is_interior = (i not in boundary_i) and (j not in boundary_j)

            if is_interior:
                # -------------------------------------------------------
                # 1.  Ampère's law  ∇ × δB_plasma = μ₀ δJ
                #     (sixth-order central differences)
                # -------------------------------------------------------
                def _d6z(col_offset):
                    """Sixth-order ∂/∂Z stencil weights for column col_offset."""
                    return [
                        (k_p3Z * 3 + col_offset, -1 / (60 * dZ)),
                        (k_p2Z * 3 + col_offset, +9 / (60 * dZ)),
                        (k_pZ  * 3 + col_offset, -45 / (60 * dZ)),
                        (k_mZ  * 3 + col_offset, +45 / (60 * dZ)),
                        (k_m2Z * 3 + col_offset, -9 / (60 * dZ)),
                        (k_m3Z * 3 + col_offset, +1 / (60 * dZ)),
                    ]

                def _d6r(col_offset):
                    """Sixth-order ∂/∂R stencil weights."""
                    return [
                        (k_p3R * 3 + col_offset, -1 / (60 * dR)),
                        (k_p2R * 3 + col_offset, +9 / (60 * dR)),
                        (k_pR  * 3 + col_offset, -45 / (60 * dR)),
                        (k_mR  * 3 + col_offset, +45 / (60 * dR)),
                        (k_m2R * 3 + col_offset, -9 / (60 * dR)),
                        (k_m3R * 3 + col_offset, +1 / (60 * dR)),
                    ]

                # R component:  −∂Bφ/∂Z = μ₀ J_R
                for col, w in _d6z(iPHI):
                    A[nEq * k + iR, col] = w
                A[nEq * k + iR, 3 * k + 3 * n + iR] = -mu0

                # φ component:  ∂B_R/∂Z − ∂B_Z/∂R = μ₀ J_φ
                for col, w in _d6z(iR):
                    A[nEq * k + iPHI, col] = w
                for col, w in _d6r(iZ):
                    A[nEq * k + iPHI, col] -= w          # note: sign flip
                    A[nEq * k + iPHI, col] = -w
                A[nEq * k + iPHI, 3 * k + 3 * n + iPHI] = -mu0

                # Z component:  Bφ/R + ∂Bφ/∂R = μ₀ J_Z
                A[nEq * k + iZ, 3 * k + iPHI] = 1.0 / R[i]
                for col, w in _d6r(iPHI):
                    A[nEq * k + iZ, col] = w
                A[nEq * k + iZ, 3 * k + 3 * n + iZ] = -mu0

                # -------------------------------------------------------
                # 2.  Force balance
                #     δJ × B₀ + J₀ × δB − ∇δp = RHS
                # -------------------------------------------------------
                def cross_matrix(v):
                    """Matrix representation of  v × ·."""
                    return np.array([
                        [0,       -v[iZ], v[iPHI]],
                        [v[iZ],   0,     -v[iR]],
                        [-v[iPHI], v[iR], 0],
                    ])

                # δJ × B₀
                A[nEq * k + 3:nEq * k + 6, 3 * k + 3 * n:3 * k + 3 * n + 3] = \
                    -cross_matrix(np.array([B0.BR[i, j], B0.BPhi[i, j], B0.BZ[i, j]]))

                # J₀ × δB
                A[nEq * k + 3:nEq * k + 6, 3 * k:3 * k + 3] = \
                    cross_matrix(np.array([J0.BR[i, j], J0.BPhi[i, j], J0.BZ[i, j]]))

                # −∇δp  (second-order central differences)
                A[nEq * k + 3 + iR, k_pR + 6 * n] = -1.0 / (2 * dR)
                A[nEq * k + 3 + iR, k_mR + 6 * n] =  1.0 / (2 * dR)
                A[nEq * k + 3 + iZ, k_pZ + 6 * n] = -1.0 / (2 * dZ)
                A[nEq * k + 3 + iZ, k_mZ + 6 * n] =  1.0 / (2 * dZ)

                # RHS
                b[nEq * k + 3 + iR]   = RHS_fb.BR[i, j]
                b[nEq * k + 3 + iPHI] = RHS_fb.BPhi[i, j]
                b[nEq * k + 3 + iZ]   = RHS_fb.BZ[i, j]

                # Scale force-balance rows to match Ampère
                scale_fb = 1e-2
                A[nEq * k + 3:nEq * k + 6, :] *= scale_fb
                b[nEq * k + 3:nEq * k + 6]    *= scale_fb

                # -------------------------------------------------------
                # 3.  Divergence-free  ∇ · δB = 0
                # -------------------------------------------------------
                A[nEq * k + 6, 3 * k + iR]      =  1.0 / R[i]
                A[nEq * k + 6, 3 * k_pR + iR]   =  1.0 / (2 * dR)
                A[nEq * k + 6, 3 * k_mR + iR]   = -1.0 / (2 * dR)
                A[nEq * k + 6, 3 * k_pZ + iZ]   =  1.0 / (2 * dZ)
                A[nEq * k + 6, 3 * k_mZ + iZ]   = -1.0 / (2 * dZ)

            # -----------------------------------------------------------
            # 4.  Boundary / vacuum conditions  δB=0, δJ=0, δp=0
            # -----------------------------------------------------------
            large = 1e10
            is_vacuum = (
                (i in boundary_i) or (j in boundary_j)
                or (i > 0 and j > 0 and i < nR - 1 and j < nZ - 1
                    and p0.B[i, j] < 1e-1)
            )

            if is_vacuum:
                A[nEq * k + 11, 3 * k + iR]              = large
                A[nEq * k + 12, 3 * k + iPHI]            = large
                A[nEq * k + 13, 3 * k + iZ]              = large
                A[nEq * k + 7,  3 * k + 3 * n + iR]     = large
                A[nEq * k + 8,  3 * k + 3 * n + iPHI]   = large
                A[nEq * k + 9,  3 * k + 3 * n + iZ]     = large
                A[nEq * k + 10, k + 6 * n]              = large

    # --- Solve the normal equations ---
    Ac = A.tocsc()
    AtA = Ac.T @ Ac
    Atb = Ac.T @ b

    x, info = bicgstab(AtA, Atb, x0=x0, maxiter=2000, rtol=1e-16)

    # --- Extract solution ---
    delta_B_arr = np.zeros((nR, nZ, 3))
    delta_J_arr = np.zeros((nR, nZ, 3))
    delta_p_arr = np.zeros((nR, nZ))

    for i in range(nR):
        for j in range(nZ):
            k = idx(i, j)
            delta_B_arr[i, j, :] = x[3 * k:3 * k + 3]
            delta_J_arr[i, j, :] = x[3 * n + 3 * k:3 * n + 3 * k + 3]
            delta_p_arr[i, j]    = x[6 * n + k]

    delta_B_plasma = VectorField3DAxiSymmetric(
        R, Z,
        BR=delta_B_arr[:, :, iR],
        BPhi=delta_B_arr[:, :, iPHI],
        BZ=delta_B_arr[:, :, iZ],
    )
    delta_J = VectorField3DAxiSymmetric(
        R, Z,
        BR=delta_J_arr[:, :, iR],
        BPhi=delta_J_arr[:, :, iPHI],
        BZ=delta_J_arr[:, :, iZ],
    )
    delta_p = ScalarField3DAxiSymmetric(R, Z, B=delta_p_arr)

    return delta_B_plasma, delta_J, delta_p
