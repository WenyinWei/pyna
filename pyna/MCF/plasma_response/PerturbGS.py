"""Perturbed Grad-Shafranov solver for axisymmetric plasma equilibrium response.

Ported from MHDpy/mhdpy/perturb_GradShafranov.py (solve_GS_perturbed_inone).

Scientific background
---------------------
For response matrix computation, the vacuum field δB_ext alone is insufficient.
The plasma response δB_plasma can be comparable in magnitude, especially for
core q-profile.  The correct total perturbation is::

    δB_total = δB_ext + δB_plasma

where δB_plasma solves the linearised MHD equilibrium::

    ∇ × δB_plasma = μ₀ δJ          (Ampère's law)
    δJ × B₀ + J₀ × δB_total = ∇δp  (force balance)
    ∇ · δB_plasma = 0               (div-free)

This is formulated as a sparse least-squares system on the R-Z grid.

Notes
-----
* Uses scipy sparse backend (lsqr / lgmres / gmres / bicgstab).
* Joblib caching for expensive solves.
* Field arrays are axisymmetric (2D R-Z), stored as 3D with singleton Phi dim.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr, lgmres, gmres, bicgstab
import os
from typing import Optional

from pyna.fields.cylindrical import VectorField3DCylindrical, ScalarField3DCylindrical
CylindricalVectorField = VectorField3DCylindrical
CylindricalScalarField = ScalarField3DCylindrical

# ---------------------------------------------------------------------------
# Joblib cache
# ---------------------------------------------------------------------------
from joblib import Memory as _Memory

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), ".cache", "plasma_response")
memory = _Memory(_CACHE_DIR, verbose=0)


# ---------------------------------------------------------------------------
# Helper: make axisymmetric 2D grids from CylindricalVectorField
# ---------------------------------------------------------------------------

def _to_2d_arrays(field: CylindricalVectorField):
    """Extract 2D (R, Z) arrays from an axisymmetric CylindricalVectorField.

    Takes the phi=0 slice (index 0 along the Phi axis).
    Returns (R, Z, VR_2d, VZ_2d, VPhi_2d).
    """
    R = field.R
    Z = field.Z
    VR = field.VR[:, :, 0]      # shape (nR, nZ)
    VZ = field.VZ[:, :, 0]
    VPhi = field.VPhi[:, :, 0]
    return R, Z, VR, VZ, VPhi


def _scalar_to_2d(field: CylindricalScalarField):
    """Extract 2D (R, Z) value array from an axisymmetric CylindricalScalarField."""
    return field.R, field.Z, field.value[:, :, 0]


def _make_axi_vector_field(R, Z, VR_2d, VZ_2d, VPhi_2d, name="") -> CylindricalVectorField:
    """Wrap 2D arrays into axisymmetric CylindricalVectorField (single Phi slice)."""
    Phi = np.array([0.0])
    VR = VR_2d[:, :, np.newaxis]
    VZ = VZ_2d[:, :, np.newaxis]
    VPhi = VPhi_2d[:, :, np.newaxis]
    return CylindricalVectorField(R=R, Z=Z, Phi=Phi, VR=VR, VZ=VZ, VPhi=VPhi, name=name)


def _make_axi_scalar_field(R, Z, val_2d, name="", units="") -> CylindricalScalarField:
    """Wrap 2D array into axisymmetric CylindricalScalarField."""
    Phi = np.array([0.0])
    value = val_2d[:, :, np.newaxis]
    return CylindricalScalarField(R=R, Z=Z, Phi=Phi, value=value, name=name, units=units)


# ---------------------------------------------------------------------------
# Cross product helper for 2D arrays
# ---------------------------------------------------------------------------

def _cross_2d(vR, vZ, vPhi, wR, wZ, wPhi):
    """Cylindrical cross product v × w → (R, Z, Phi) components."""
    cR   =  vPhi * wZ   - vZ   * wPhi
    cPhi = -vR   * wZ   + vZ   * wR
    cZ   =  vR   * wPhi - vPhi * wR
    return cR, cZ, cPhi


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

@memory.cache
def solve_perturbed_gs(
    B0: CylindricalVectorField,
    J0: CylindricalVectorField,
    p0: CylindricalScalarField,
    delta_B_ext: CylindricalVectorField,
    x0=None,
    solver: str = 'lsqr',
    max_iter: int = 1000,
    tol: float = 1e-6,
    R_axis=None,
    Z_axis=None,
    a_eff=None,
    beta_val: float = 0.0,
    alpha_pressure: float = 2.0,
    div_weight: float = 1e4,
    bc_J_weight: float = 1e5,
    bc_p_weight: float = 1e6,
) -> tuple:
    """Solve linearised MHD equilibrium for plasma response.

    Solves the coupled system::

        δJ × B₀ + J₀ × δB_plasma − ∇δp = −J₀ × δB_ext   (force balance)
        ∇ · δB_plasma = 0                                   (div-free)
        δJ = (1/μ₀) ∇ × δB_plasma                         (Ampère, implicit)

    on the R-Z grid using a sparse least-squares / iterative solver.

    Parameters
    ----------
    B0 : CylindricalVectorField
        Background magnetic field on R-Z grid (axisymmetric, phi=0 slice used).
    J0 : CylindricalVectorField
        Background current density J = ∇×B/μ₀.
    p0 : CylindricalScalarField
        Background pressure.
    delta_B_ext : CylindricalVectorField
        External (vacuum) field perturbation from coil currents.
    x0 : ndarray or None
        Initial guess vector (warm start). Shape (4*nR*nZ,).
    solver : str
        Sparse solver to use: 'lsqr', 'lgmres', 'gmres', 'bicgstab'.
    max_iter : int
        Maximum iterations.
    tol : float
        Solver tolerance.

    Returns
    -------
    delta_B_plasma : CylindricalVectorField
        Plasma response field δB_plasma.
    delta_J : CylindricalVectorField
        Induced current density δJ = (1/μ₀) ∇×δB_plasma.
    delta_p : CylindricalScalarField
        Pressure perturbation δp.
    """
    mu0 = 4e-7 * np.pi

    # Auto-compute J0 and p0 from equilibrium currents when beta_val > 0
    if beta_val > 0.0 and R_axis is not None and Z_axis is not None and a_eff is not None:
        R_arr = B0.R
        Z_arr = B0.Z
        Phi_arr = B0.Phi

        # Volume-average |B0| for pressure normalisation
        B_mag_all = np.sqrt(B0.VR**2 + B0.VZ**2 + B0.VPhi**2)
        B_vol_avg = float(np.mean(B_mag_all))
        p0_pa     = beta_val * B_vol_avg ** 2 / (2.0 * mu0)

        def _pressure_profile(psi_norm):
            return p0_pa * max(0.0, 1.0 - float(psi_norm)) ** alpha_pressure

        J0, p0 = compute_equilibrium_currents(
            B0, _pressure_profile, R_arr, Z_arr, Phi_arr,
            R_axis, Z_axis, a_eff,
        )

    R, Z, B0R, B0Z, B0Phi = _to_2d_arrays(B0)
    _,  _, J0R, J0Z, J0Phi = _to_2d_arrays(J0)
    _,  _, p0_val = _scalar_to_2d(p0)
    _,  _, dBextR, dBextZ, dBextPhi = _to_2d_arrays(delta_B_ext)

    nR, nZ = len(R), len(Z)
    dR = R[1] - R[0]
    dZ = Z[1] - Z[0]
    n = nR * nZ
    nEq = 8  # 3 force-balance + 1 div-free + 3 J=0 BC + 1 p=0 BC

    # RHS: -J0 × δB_ext  (force balance right-hand side)
    RHS_R, RHS_Z, RHS_Phi = _cross_2d(J0R, J0Z, J0Phi, dBextR, dBextZ, dBextPhi)
    RHS_R   = -RHS_R
    RHS_Z   = -RHS_Z
    RHS_Phi = -RHS_Phi

    # Variable layout: x = [δB_R(0..n-1), δB_Z(n..2n-1), δB_Phi(2n..3n-1), δp(3n..4n-1)]
    # We use R_=0, Z_=1, PHI_=2 but mapped to [δBR, δBZ, δBPhi]
    # Component offsets in x vector:
    OFF_BR   = 0
    OFF_BZ   = n
    OFF_BPHI = 2 * n
    OFF_P    = 3 * n

    def k(i, j):
        return i * nZ + j

    A = lil_matrix((nEq * n, 4 * n))
    b = np.zeros(nEq * n)

    # Cross matrix: v × (.) as 3×3 matrix in (R, Z, Phi) order
    # (v × w)_R   = v_Z * w_Phi - v_Phi * w_Z
    # (v × w)_Z   = v_Phi * w_R  - v_R  * w_Phi
    # (v × w)_Phi = v_R  * w_Z   - v_Z  * w_R

    def cross_matrix(vR, vZ, vPhi):
        """Matrix M such that (v×w) = M @ [w_R, w_Z, w_Phi]."""
        return np.array([
            [0,      vPhi,  -vZ  ],   # (v×w)_R
            [-vPhi,  0,      vR  ],   # (v×w)_Z
            [vZ,    -vR,     0   ],   # (v×w)_Phi
        ])

    # Equation rows per grid point k(i,j):
    # row 0: force balance R component
    # row 1: force balance Z component
    # row 2: force balance Phi component
    # row 3: div-free
    # row 4: J=0 BC (R)
    # row 5: J=0 BC (Z)
    # row 6: J=0 BC (Phi)
    # row 7: p=0 BC

    EqBdiv_weight   = div_weight    # default 1e4 (was 1e8 -- caused kappa > 1e20)
    BC_no_J_weight  = bc_J_weight  # default 1e5 (was 1e9)
    BC_no_p_weight  = bc_p_weight  # default 1e6 (was 1e12)

    for i in range(nR):
        for j in range(nZ):
            kij = k(i, j)

            # ---- Neighbour indices ----
            kpR = k(i + 1, j) if i < nR - 1 else k(i, j)
            kmR = k(i - 1, j) if i > 0      else k(i, j)
            kpZ = k(i, j + 1) if j < nZ - 1 else k(i, j)
            kmZ = k(i, j - 1) if j > 0      else k(i, j)

            interior = (0 < i < nR - 1) and (0 < j < nZ - 1)
            near_boundary = (i in [1, 2, nR - 2, nR - 3]) or (j in [1, 2, nZ - 2, nZ - 3])
            eq_base = nEq * kij

            if interior:
                # --------------------------------------------------------
                # 1. Force balance:
                #    (1/μ₀)(∇×δB_plasma)×B₀ + J₀×δB_plasma − ∇δp = RHS
                #
                # (∇×δB_plasma) components in cylindrical coords:
                #   curl_R   = ∂δBPhi/∂Z ... wait, actually:
                #   (∇×B)_R   = (1/R)∂Bphi/∂phi − ∂Bphi/∂Z  ... axi → (∇×B)_R = −∂Bphi/∂Z  → -dBPhi/dZ
                #   (∇×B)_Z   = (1/R)∂(R Bphi)/∂R              → Bphi/R + dBPhi/dR
                #   (∇×B)_Phi = ∂BR/∂Z − ∂BZ/∂R
                #
                # We substitute δJ = (1/μ₀) ∇×δB_plasma directly:
                # --------------------------------------------------------

                # 2nd-order central differences
                def d_dR(off_col):
                    """(∂f/∂R) using 2nd-order central diff → coefficients on A."""
                    return [(k(i + 1, j) + off_col, 1 / (2 * dR)),
                            (k(i - 1, j) + off_col, -1 / (2 * dR))]

                def d_dZ(off_col):
                    return [(k(i, j + 1) + off_col, 1 / (2 * dZ)),
                            (k(i, j - 1) + off_col, -1 / (2 * dZ))]

                def add_curl_x_B0_row(eq_row):
                    """
                    Set A rows for (1/μ₀)(∇×δB)×B₀ in R, Z, Phi components.

                    curl_R   = −∂δBPhi/∂Z
                    curl_Z   = δBPhi/R + ∂δBPhi/∂R
                    curl_Phi =  ∂δBR/∂Z − ∂δBZ/∂R

                    (curl × B0)_R   = curl_Z * B0Phi - curl_Phi * B0Z
                    (curl × B0)_Z   = curl_Phi * B0R - curl_R  * B0Phi
                    (curl × B0)_Phi = curl_R   * B0Z - curl_Z  * B0R
                    """
                    # curl components as functions of δB fields
                    curlR_coeffs   = [(k(i, j + 1) + OFF_BPHI, -1 / (2 * dZ)),
                                      (k(i, j - 1) + OFF_BPHI,  1 / (2 * dZ))]
                    curlZ_coeffs   = [(kij + OFF_BPHI, 1 / R[i])] + \
                                     [(k(i + 1, j) + OFF_BPHI,  1 / (2 * dR)),
                                      (k(i - 1, j) + OFF_BPHI, -1 / (2 * dR))]
                    curlPhi_coeffs = [(k(i, j + 1) + OFF_BR,   1 / (2 * dZ)),
                                      (k(i, j - 1) + OFF_BR,  -1 / (2 * dZ)),
                                      (k(i + 1, j) + OFF_BZ,  -1 / (2 * dR)),
                                      (k(i - 1, j) + OFF_BZ,   1 / (2 * dR))]

                    b0R, b0Z, b0Ph = B0R[i, j], B0Z[i, j], B0Phi[i, j]
                    factor = 1.0 / mu0

                    # Row R: (curlZ * b0Ph - curlPhi * b0Z) / mu0
                    for col, coef in curlZ_coeffs:
                        A[eq_base + 0, col] += factor * b0Ph * coef
                    for col, coef in curlPhi_coeffs:
                        A[eq_base + 0, col] -= factor * b0Z * coef

                    # Row Z: (curlPhi * b0R - curlR * b0Ph) / mu0
                    for col, coef in curlPhi_coeffs:
                        A[eq_base + 1, col] += factor * b0R * coef
                    for col, coef in curlR_coeffs:
                        A[eq_base + 1, col] -= factor * b0Ph * coef

                    # Row Phi: (curlR * b0Z - curlZ * b0R) / mu0
                    for col, coef in curlR_coeffs:
                        A[eq_base + 2, col] += factor * b0Z * coef
                    for col, coef in curlZ_coeffs:
                        A[eq_base + 2, col] -= factor * b0R * coef

                add_curl_x_B0_row(eq_base)

                # J0 × δB_plasma
                CM = cross_matrix(J0R[i, j], J0Z[i, j], J0Phi[i, j])
                # [R, Z, Phi] order in x: OFF_BR, OFF_BZ, OFF_BPHI
                cols_B = [kij + OFF_BR, kij + OFF_BZ, kij + OFF_BPHI]
                for row_off in range(3):
                    for col_off, col in enumerate(cols_B):
                        A[eq_base + row_off, col] += CM[row_off, col_off]

                # −∇δp
                A[eq_base + 0, k(i + 1, j) + OFF_P] -= 1 / (2 * dR)
                A[eq_base + 0, k(i - 1, j) + OFF_P] += 1 / (2 * dR)
                A[eq_base + 1, k(i, j + 1) + OFF_P] -= 1 / (2 * dZ)
                A[eq_base + 1, k(i, j - 1) + OFF_P] += 1 / (2 * dZ)
                # (no Z force balance for Phi direction — Phi component has no ∇δp)

                # RHS
                b[eq_base + 0] = RHS_R[i, j]
                b[eq_base + 1] = RHS_Z[i, j]
                b[eq_base + 2] = RHS_Phi[i, j]

            # --------------------------------------------------------
            # 2. Div-free: ∇·δB = δBR/R + ∂δBR/∂R + ∂δBZ/∂Z = 0
            # --------------------------------------------------------
            w = EqBdiv_weight
            if interior:
                A[eq_base + 3, kij + OFF_BR]  += (1 / R[i]) * w
                i_fwd = min(i + 1, nR - 1); i_bwd = max(i - 1, 0)
                j_fwd = min(j + 1, nZ - 1); j_bwd = max(j - 1, 0)
                dr_eff = (R[i_fwd] - R[i_bwd])
                dz_eff = (Z[j_fwd] - Z[j_bwd])
                A[eq_base + 3, k(i_fwd, j) + OFF_BR] +=  (1 / dr_eff) * w
                A[eq_base + 3, k(i_bwd, j) + OFF_BR] -= (1 / dr_eff) * w
                A[eq_base + 3, k(i, j_fwd) + OFF_BZ] +=  (1 / dz_eff) * w
                A[eq_base + 3, k(i, j_bwd) + OFF_BZ] -= (1 / dz_eff) * w
            else:
                # One-sided at boundary
                if i == 0:
                    A[eq_base + 3, kij + OFF_BR] += (1 / R[i] - 1 / dR) * w
                    A[eq_base + 3, k(i + 1, j) + OFF_BR] += (1 / dR) * w
                elif i == nR - 1:
                    A[eq_base + 3, kij + OFF_BR] += (1 / R[i] + 1 / dR) * w
                    A[eq_base + 3, k(i - 1, j) + OFF_BR] -= (1 / dR) * w
                else:
                    A[eq_base + 3, kij + OFF_BR] += (1 / R[i]) * w
                    A[eq_base + 3, k(i + 1, j) + OFF_BR] +=  (1 / (2 * dR)) * w
                    A[eq_base + 3, k(i - 1, j) + OFF_BR] -= (1 / (2 * dR)) * w

                if j == 0:
                    A[eq_base + 3, k(i, j + 1) + OFF_BZ] += (1 / dZ) * w
                    A[eq_base + 3, kij + OFF_BZ]          -= (1 / dZ) * w
                elif j == nZ - 1:
                    A[eq_base + 3, kij + OFF_BZ]          += (1 / dZ) * w
                    A[eq_base + 3, k(i, j - 1) + OFF_BZ] -= (1 / dZ) * w
                else:
                    A[eq_base + 3, k(i, j + 1) + OFF_BZ] +=  (1 / (2 * dZ)) * w
                    A[eq_base + 3, k(i, j - 1) + OFF_BZ] -= (1 / (2 * dZ)) * w

            # --------------------------------------------------------
            # 3. Boundary conditions: J=0, p=0 in vacuum region
            # --------------------------------------------------------
            is_boundary = (i in [0, 1, 2, nR - 3, nR - 2, nR - 1] or
                           j in [0, 1, 2, nZ - 3, nZ - 2, nZ - 1])

            # Pressure BC
            in_vacuum = True
            if interior and not is_boundary:
                # Check if in plasma region (p0 > threshold in neighbourhood)
                nbrs = [(i, j), (i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1),
                        (i + 1, j + 1), (i + 1, j - 1), (i - 1, j + 1), (i - 1, j - 1)]
                for ni, nj in nbrs:
                    if 0 <= ni < nR and 0 <= nj < nZ and p0_val[ni, nj] > 1e-1:
                        in_vacuum = False
                        break

            if is_boundary or in_vacuum:
                # δJ = 0 (via ∇×δB = 0)
                # We enforce curl components = 0
                wJ = BC_no_J_weight

                # (∇×δB)_R = −∂δBPhi/∂Z = 0
                jpZ = min(j + 1, nZ - 1); jmZ = max(j - 1, 0)
                coef_dZ = 1 / (Z[jpZ] - Z[jmZ]) if jpZ != jmZ else 1 / dZ
                A[eq_base + 4, k(i, jpZ) + OFF_BPHI] -= coef_dZ * wJ
                A[eq_base + 4, k(i, jmZ) + OFF_BPHI] += coef_dZ * wJ

                # (∇×δB)_Phi = ∂δBR/∂Z − ∂δBZ/∂R = 0
                A[eq_base + 5, k(i, jpZ) + OFF_BR] +=  coef_dZ * wJ
                A[eq_base + 5, k(i, jmZ) + OFF_BR] -= coef_dZ * wJ
                ipR = min(i + 1, nR - 1); imR = max(i - 1, 0)
                coef_dR = 1 / (R[ipR] - R[imR]) if ipR != imR else 1 / dR
                A[eq_base + 5, k(ipR, j) + OFF_BZ] -= coef_dR * wJ
                A[eq_base + 5, k(imR, j) + OFF_BZ] += coef_dR * wJ

                # (∇×δB)_Z = δBPhi/R + ∂δBPhi/∂R = 0
                A[eq_base + 6, kij + OFF_BPHI]       +=  (1 / R[i]) * wJ
                A[eq_base + 6, k(ipR, j) + OFF_BPHI] +=  coef_dR * wJ
                A[eq_base + 6, k(imR, j) + OFF_BPHI] -= coef_dR * wJ

                # δp = 0
                A[eq_base + 7, kij + OFF_P] = BC_no_p_weight

    # ----------------------------------------------------------------
    # Solve
    # ----------------------------------------------------------------
    A_csc = A.tocsc()

    if solver == 'lsqr':
        result = lsqr(A_csc, b, damp=1e-4, iter_lim=max_iter,
                      atol=tol, btol=tol, x0=x0)
        x = result[0]
    elif solver == 'lgmres':
        ATA = A_csc.T @ A_csc
        ATb = A_csc.T @ b
        x, _ = lgmres(ATA, ATb, x0=x0, maxiter=max_iter, rtol=tol)
    elif solver == 'gmres':
        ATA = A_csc.T @ A_csc
        ATb = A_csc.T @ b
        x, _ = gmres(ATA, ATb, x0=x0, maxiter=max_iter, rtol=tol)
    elif solver == 'bicgstab':
        ATA = A_csc.T @ A_csc
        ATb = A_csc.T @ b
        x, _ = bicgstab(ATA, ATb, x0=x0, maxiter=max_iter, rtol=tol)
    else:
        raise ValueError(f"Unknown solver '{solver}'. Choose from: lsqr, lgmres, gmres, bicgstab")

    # ----------------------------------------------------------------
    # Extract solution
    # ----------------------------------------------------------------
    dBR_2d   = np.zeros((nR, nZ))
    dBZ_2d   = np.zeros((nR, nZ))
    dBPhi_2d = np.zeros((nR, nZ))
    dp_2d    = np.zeros((nR, nZ))

    for i in range(nR):
        for j in range(nZ):
            kij = k(i, j)
            dBR_2d[i, j]   = x[kij + OFF_BR]
            dBZ_2d[i, j]   = x[kij + OFF_BZ]
            dBPhi_2d[i, j] = x[kij + OFF_BPHI]
            dp_2d[i, j]    = x[kij + OFF_P]

    # Compute δJ = (1/μ₀) ∇×δB_plasma numerically
    dJR_2d   = np.zeros((nR, nZ))
    dJZ_2d   = np.zeros((nR, nZ))
    dJPhi_2d = np.zeros((nR, nZ))

    # Interior points: second-order central differences
    dBPhi_dZ = np.gradient(dBPhi_2d, dZ, axis=1)
    d_RBPhi_dR = np.gradient(
        np.einsum('i,ij->ij', R, dBPhi_2d),
        dR, axis=0
    )
    dBR_dZ = np.gradient(dBR_2d, dZ, axis=1)
    dBZ_dR = np.gradient(dBZ_2d, dR, axis=0)

    RR, _ = np.meshgrid(R, np.ones(nZ), indexing='ij')
    dJR_2d   = -dBPhi_dZ / mu0
    dJZ_2d   =  d_RBPhi_dR / (RR * mu0)
    dJPhi_2d = (dBR_dZ - dBZ_dR) / mu0

    delta_B_plasma = _make_axi_vector_field(R, Z, dBR_2d, dBZ_2d, dBPhi_2d,
                                             name="delta_B_plasma")
    delta_J        = _make_axi_vector_field(R, Z, dJR_2d, dJZ_2d, dJPhi_2d,
                                             name="delta_J")
    delta_p        = _make_axi_scalar_field(R, Z, dp_2d,
                                             name="delta_p", units="Pa")

    return delta_B_plasma, delta_J, delta_p


# ---------------------------------------------------------------------------
# Preconditioning helper for coupled system
# ---------------------------------------------------------------------------

def _scale_coupled_system(A_lil, b, n, nR, nZ, mu0):
    """Row- and column-scale the 8n×7n coupled system for better conditioning.

    The raw system has wildly different coefficient magnitudes:
    - Ampère rows: ~1/μ₀ ~10⁸  A/m²/T
    - Force rows:  ~J×B  ~10⁶  N/m³
    - Div-free/DivJ rows: ~1/m  ~1–100

    Strategy
    --------
    Row scaling:
        - Ampère rows (eq 0,1,2): multiply by μ₀  → O(1)
        - Force rows  (eq 3,4,5): no extra scaling needed (already O(1) after
          weight_force=1.0; if weights differ, divide by weight)
        - Div-free row (eq 6):   multiply by L_ref (typical grid spacing)
        - DivJ row (eq 7):       multiply by L_ref

    Column scaling:
        - δJ columns (0–2n): multiply by μ₀  → brings δJ~O(J/μ₀) to O(J·μ₀/μ₀)=O(J)
        - δB columns (3n–5n): unchanged (B ~O(1 T))
        - δp column  (6n–7n): multiply by 1 (p in Pa, gradients ~B²/μ₀L ~10⁶)
          → scale by μ₀ as well so δp effectively in units of B²

    Parameters
    ----------
    A_lil : scipy.sparse.lil_matrix  — assembled system matrix (modified in-place copy)
    b     : ndarray — RHS vector (modified in-place copy)
    n     : int — nR*nZ
    nR, nZ : int
    mu0   : float

    Returns
    -------
    A_csc      : scaled CSC matrix
    b_scaled   : scaled RHS
    row_scale  : ndarray (n_rows,) — applied row multipliers
    col_scale  : ndarray (n_cols,) — applied column multipliers
                 To recover original solution: x_orig = x_scaled / col_scale
    """
    from scipy.sparse import diags as sp_diags

    N_EQ_PP = 8
    n_rows  = N_EQ_PP * n
    n_cols  = 7 * n

    # Estimate typical grid spacing from n
    import math
    L_ref = 1.0 / math.sqrt(max(n, 1))  # rough O(grid spacing)

    row_scale = np.ones(n_rows, dtype=np.float64)
    for kij in range(n):
        eq_base = N_EQ_PP * kij
        # Ampère rows (0,1,2): scale by μ₀
        row_scale[eq_base + 0] = mu0
        row_scale[eq_base + 1] = mu0
        row_scale[eq_base + 2] = mu0
        # Force rows (3,4,5): keep as-is (O(1) with weight_force=1)
        # Div-free row (6): scale by L_ref
        row_scale[eq_base + 6] = L_ref
        # DivJ row (7): scale by L_ref
        row_scale[eq_base + 7] = L_ref

    # Column scaling: δJ cols get *μ₀, rest unchanged
    col_scale = np.ones(n_cols, dtype=np.float64)
    # δJ_R  : cols [0*n, 1*n)
    col_scale[0 * n : 1 * n] = mu0
    # δJ_Z  : cols [1*n, 2*n)
    col_scale[1 * n : 2 * n] = mu0
    # δJ_Phi: cols [2*n, 3*n)
    col_scale[2 * n : 3 * n] = mu0
    # δB cols: unchanged
    # δp cols: scale by μ₀ so pressure is in "magnetic pressure" units
    col_scale[6 * n : 7 * n] = mu0

    # Apply row scaling: A_scaled = diag(row_scale) @ A
    # Apply col scaling: A_scaled = A_scaled @ diag(col_scale)
    # Combined: A_scaled[i, j] = row_scale[i] * A[i, j] * col_scale[j]
    A_csc = A_lil.tocsc()
    R_diag = sp_diags(row_scale, format='csr')
    C_diag = sp_diags(col_scale, format='csr')
    A_scaled_csc = (R_diag @ A_csc @ C_diag).tocsc()
    b_scaled = row_scale * b

    return A_scaled_csc, b_scaled, row_scale, col_scale


# ---------------------------------------------------------------------------
# Coupled (δJ, δB, δp) solver
# ---------------------------------------------------------------------------

def solve_perturbed_gs_coupled(
    B0: CylindricalVectorField,
    J0: CylindricalVectorField,
    p0: CylindricalScalarField,
    delta_B_ext: CylindricalVectorField,
    solver: str = 'lsqr',
    max_iter: int = 2000,
    tol: float = 1e-6,
    weight_ampere: float = 1e4,
    weight_force: float = 1.0,
    weight_div: float = 1e4,
    weight_divJ: float = 1e2,
    weight_BC_J: float = 1e2,
    weight_BC_p: float = 1e6,
    delta_p_2d: Optional[np.ndarray] = None,
) -> tuple:
    """Solve coupled (δJ, δB, δp) linearised MHD system simultaneously.

    Unlike :func:`solve_perturbed_gs` which only solves for (δB, δp) and
    computes δJ post-hoc from curl(δB)/μ₀, this function solves for δJ, δB,
    δp simultaneously, enforcing the Ampère constraint μ₀δJ = ∇×δB as an
    explicit equation in the sparse system.

    Variables (7n unknowns, n = nR×nZ)::

        x[0*n : 1*n]  = δJ_R   (flattened row-major i*nZ+j)
        x[1*n : 2*n]  = δJ_Z
        x[2*n : 3*n]  = δJ_Phi
        x[3*n : 4*n]  = δB_R
        x[4*n : 5*n]  = δB_Z
        x[5*n : 6*n]  = δB_Phi
        x[6*n : 7*n]  = δp

    8 equations per interior grid point (overdetermined → LSQR):

    * Ampère R/Z/Phi (weight_ampere)
    * Force-balance R/Z/Phi (weight_force)
    * Div-free (weight_div)
    * Div-J (weight_divJ)

    Boundary / vacuum regions: δJ = 0 (weight_BC_J), δp = 0 (weight_BC_p).

    Parameters
    ----------
    B0, J0, p0, delta_B_ext : CylindricalVectorField / CylindricalScalarField
        Background and external perturbation fields on the same R-Z grid.
    solver : str
        Sparse solver ('lsqr', 'lgmres', 'gmres', 'bicgstab').
    max_iter : int
        Maximum solver iterations.
    tol : float
        Solver tolerance.
    weight_ampere : float
        Weight on Ampère equations (enforce μ₀δJ = ∇×δB tightly).
    weight_force : float
        Weight on force-balance equations.
    weight_div : float
        Weight on div-free constraint.
    weight_divJ : float
        Weight on current conservation ∇·δJ = 0.
    weight_BC_J : float
        Penalty weight for δJ = 0 on boundary / vacuum.
    weight_BC_p : float
        Penalty weight for δp = 0 on boundary / vacuum.

    Returns
    -------
    delta_B_plasma : CylindricalVectorField
        Plasma response field δB_plasma.
    delta_J : CylindricalVectorField
        Induced current density δJ (directly solved, not computed post-hoc).
    delta_p : CylindricalScalarField
        Pressure perturbation δp.
    """
    mu0 = 4e-7 * np.pi

    R, Z, B0R, B0Z, B0Phi = _to_2d_arrays(B0)
    _,  _, J0R, J0Z, J0Phi = _to_2d_arrays(J0)
    _,  _, p0_val = _scalar_to_2d(p0)
    _,  _, dBextR, dBextZ, dBextPhi = _to_2d_arrays(delta_B_ext)

    nR, nZ = len(R), len(Z)
    dR = R[1] - R[0]
    dZ = Z[1] - Z[0]
    n = nR * nZ

    # Variable offsets in x-vector (7n unknowns)
    OFF_JR   = 0
    OFF_JZ   = n
    OFF_JPhi = 2 * n
    OFF_BR   = 3 * n
    OFF_BZ   = 4 * n
    OFF_BPhi = 5 * n
    OFF_P    = 6 * n

    # 8 equations per grid point
    N_EQ_PP = 8
    n_rows   = N_EQ_PP * n
    n_cols   = 7 * n

    def k(i, j):
        return i * nZ + j

    # RHS for force balance equations.
    # For a currentless stellarator (J0=0): force balance is simply
    #   δJ × B0 = ∇δp
    # If delta_p_2d is provided, use its gradient as the driving RHS.
    # Otherwise fall back to -J0 × δB_ext (tokamak-style perturbation response).
    if delta_p_2d is not None:
        dp = np.asarray(delta_p_2d, dtype=float)
        # Central differences for ∇δp
        RHS_R   = np.gradient(dp, R, axis=0)
        RHS_Z   = np.gradient(dp, Z, axis=1)
        RHS_Phi = np.zeros_like(RHS_R)   # axisymmetric p: no toroidal gradient
    else:
        RHS_R, RHS_Z, RHS_Phi = _cross_2d(J0R, J0Z, J0Phi, dBextR, dBextZ, dBextPhi)
        RHS_R   = -RHS_R
        RHS_Z   = -RHS_Z
        RHS_Phi = -RHS_Phi

    A = lil_matrix((n_rows, n_cols))
    b = np.zeros(n_rows)

    for i in range(nR):
        for j in range(nZ):
            kij      = k(i, j)
            eq_base  = N_EQ_PP * kij

            # Stencil neighbour indices (clamped for one-sided diffs at boundary)
            ipR = min(i + 1, nR - 1);  imR = max(i - 1, 0)
            jpZ = min(j + 1, nZ - 1);  jmZ = max(j - 1, 0)

            interior = (0 < i < nR - 1) and (0 < j < nZ - 1)

            # Effective step for FD (halved when at grid edge)
            dr_eff = R[ipR] - R[imR]
            dz_eff = Z[jpZ] - Z[jmZ]
            inv_dR = 1.0 / dr_eff
            inv_dZ = 1.0 / dz_eff

            # --------------------------------------------------------
            # Classify point
            # --------------------------------------------------------
            is_boundary = (i in (0, 1, 2, nR - 3, nR - 2, nR - 1) or
                           j in (0, 1, 2, nZ - 3, nZ - 2, nZ - 1))

            # Plasma region: use delta_p_2d if provided (vacuum stellarator
            # has p0=0, so we must use the target pressure perturbation to
            # identify the plasma interior). Fallback to p0_val for tokamak.
            _p_ref = delta_p_2d if delta_p_2d is not None else p0_val
            in_vacuum = True
            if not is_boundary:
                nbrs = [(i, j), (i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1),
                        (i + 1, j + 1), (i + 1, j - 1), (i - 1, j + 1), (i - 1, j - 1)]
                for ni, nj in nbrs:
                    if 0 <= ni < nR and 0 <= nj < nZ and _p_ref[ni, nj] > 1e-1:
                        in_vacuum = False
                        break

            plasma_interior = interior and not is_boundary and not in_vacuum

            # --------------------------------------------------------
            # Rows 0-2: Ampère equations  (interior: physics, else: δJ = 0 BC)
            # --------------------------------------------------------
            if plasma_interior:
                wA = weight_ampere
                # [AR]: μ₀·δJ_R + ∂δB_Phi/∂Z = 0
                A[eq_base + 0, OFF_JR   + kij]          = mu0 * wA
                A[eq_base + 0, OFF_BPhi + k(i, jpZ)]   += +wA * inv_dZ
                A[eq_base + 0, OFF_BPhi + k(i, jmZ)]   += -wA * inv_dZ

                # [AZ]: μ₀·δJ_Z - δB_Phi/R - ∂δB_Phi/∂R = 0
                A[eq_base + 1, OFF_JZ   + kij]          = mu0 * wA
                A[eq_base + 1, OFF_BPhi + kij]         += -wA / R[i]
                A[eq_base + 1, OFF_BPhi + k(ipR, j)]   += -wA * inv_dR
                A[eq_base + 1, OFF_BPhi + k(imR, j)]   +=  wA * inv_dR

                # [APhi]: μ₀·δJ_Phi - ∂δB_R/∂Z + ∂δB_Z/∂R = 0
                A[eq_base + 2, OFF_JPhi + kij]          = mu0 * wA
                A[eq_base + 2, OFF_BR   + k(i, jpZ)]   += -wA * inv_dZ
                A[eq_base + 2, OFF_BR   + k(i, jmZ)]   +=  wA * inv_dZ
                A[eq_base + 2, OFF_BZ   + k(ipR, j)]   +=  wA * inv_dR
                A[eq_base + 2, OFF_BZ   + k(imR, j)]   += -wA * inv_dR
            else:
                # BC: δJ = 0 (penalty)
                A[eq_base + 0, OFF_JR   + kij] = weight_BC_J
                A[eq_base + 1, OFF_JZ   + kij] = weight_BC_J
                A[eq_base + 2, OFF_JPhi + kij] = weight_BC_J

            # --------------------------------------------------------
            # Rows 3-5: Force balance  (interior: physics, else: δp = 0 + δB_Phi = 0)
            # --------------------------------------------------------
            if plasma_interior:
                wF = weight_force
                b0R, b0Z, b0Ph = B0R[i, j], B0Z[i, j], B0Phi[i, j]
                j0R, j0Z, j0Ph = J0R[i, j], J0Z[i, j], J0Phi[i, j]

                # [FR]: (δJ×B₀)_R + (J₀×δB)_R - ∂δp/∂R = RHS_R
                # (δJ×B₀)_R = δJ_Z·B₀_Phi - δJ_Phi·B₀_Z
                A[eq_base + 3, OFF_JZ   + kij] +=  wF * b0Ph
                A[eq_base + 3, OFF_JPhi + kij] += -wF * b0Z
                # (J₀×δB)_R = J₀_Z·δB_Phi - J₀_Phi·δB_Z
                A[eq_base + 3, OFF_BPhi + kij] +=  wF * j0Z
                A[eq_base + 3, OFF_BZ   + kij] += -wF * j0Ph
                # -∂δp/∂R
                A[eq_base + 3, OFF_P + k(ipR, j)] += -wF * inv_dR
                A[eq_base + 3, OFF_P + k(imR, j)] +=  wF * inv_dR
                b[eq_base + 3] = wF * RHS_R[i, j]

                # [FZ]: (δJ×B₀)_Z + (J₀×δB)_Z - ∂δp/∂Z = RHS_Z
                # (δJ×B₀)_Z = δJ_Phi·B₀_R - δJ_R·B₀_Phi
                A[eq_base + 4, OFF_JPhi + kij] +=  wF * b0R
                A[eq_base + 4, OFF_JR   + kij] += -wF * b0Ph
                # (J₀×δB)_Z = J₀_Phi·δB_R - J₀_R·δB_Phi
                A[eq_base + 4, OFF_BR   + kij] +=  wF * j0Ph
                A[eq_base + 4, OFF_BPhi + kij] += -wF * j0R
                # -∂δp/∂Z
                A[eq_base + 4, OFF_P + k(i, jpZ)] += -wF * inv_dZ
                A[eq_base + 4, OFF_P + k(i, jmZ)] +=  wF * inv_dZ
                b[eq_base + 4] = wF * RHS_Z[i, j]

                # [FPhi]: (δJ×B₀)_Phi + (J₀×δB)_Phi = RHS_Phi
                # (δJ×B₀)_Phi = δJ_R·B₀_Z - δJ_Z·B₀_R
                A[eq_base + 5, OFF_JR + kij] +=  wF * b0Z
                A[eq_base + 5, OFF_JZ + kij] += -wF * b0R
                # (J₀×δB)_Phi = J₀_R·δB_Z - J₀_Z·δB_R
                A[eq_base + 5, OFF_BZ  + kij] +=  wF * j0R
                A[eq_base + 5, OFF_BR  + kij] += -wF * j0Z
                b[eq_base + 5] = wF * RHS_Phi[i, j]
            else:
                # BC: δp = 0, δB_Phi = 0 on boundary
                A[eq_base + 3, OFF_P    + kij] = weight_BC_p
                A[eq_base + 4, OFF_BPhi + kij] = weight_BC_J   # suppress toroidal δB
                # row 5 left empty (no penalty needed for δB_R/Z at boundary from here)

            # --------------------------------------------------------
            # Row 6: Div-free  ∇·δB = δB_R/R + ∂δB_R/∂R + ∂δB_Z/∂Z = 0
            # (always applied, weight_div)
            # --------------------------------------------------------
            wD = weight_div
            A[eq_base + 6, OFF_BR   + kij]          += wD / R[i]
            A[eq_base + 6, OFF_BR   + k(ipR, j)]    +=  wD * inv_dR
            A[eq_base + 6, OFF_BR   + k(imR, j)]    += -wD * inv_dR
            A[eq_base + 6, OFF_BZ   + k(i, jpZ)]    +=  wD * inv_dZ
            A[eq_base + 6, OFF_BZ   + k(i, jmZ)]    += -wD * inv_dZ

            # --------------------------------------------------------
            # Row 7: Div-J  ∇·δJ = δJ_R/R + ∂δJ_R/∂R + ∂δJ_Z/∂Z = 0
            # (always applied, weight_divJ)
            # --------------------------------------------------------
            wJ = weight_divJ
            A[eq_base + 7, OFF_JR   + kij]          += wJ / R[i]
            A[eq_base + 7, OFF_JR   + k(ipR, j)]    +=  wJ * inv_dR
            A[eq_base + 7, OFF_JR   + k(imR, j)]    += -wJ * inv_dR
            A[eq_base + 7, OFF_JZ   + k(i, jpZ)]    +=  wJ * inv_dZ
            A[eq_base + 7, OFF_JZ   + k(i, jmZ)]    += -wJ * inv_dZ

    # ----------------------------------------------------------------
    # Scale the system for better conditioning, then solve
    # ----------------------------------------------------------------
    A_csc, b, row_scale, col_scale = _scale_coupled_system(
        A, b, n, nR, nZ, mu0
    )

    if solver == 'lsqr':
        result = lsqr(A_csc, b, damp=0.0, iter_lim=max_iter,
                      atol=tol, btol=tol)
        x_scaled = result[0]
    elif solver == 'lgmres':
        ATA = A_csc.T @ A_csc
        ATb = A_csc.T @ b
        x_scaled, _ = lgmres(ATA, ATb, maxiter=max_iter, rtol=tol)
    elif solver == 'gmres':
        ATA = A_csc.T @ A_csc
        ATb = A_csc.T @ b
        x_scaled, _ = gmres(ATA, ATb, maxiter=max_iter, rtol=tol)
    elif solver == 'bicgstab':
        ATA = A_csc.T @ A_csc
        ATb = A_csc.T @ b
        x_scaled, _ = bicgstab(ATA, ATb, maxiter=max_iter, rtol=tol)
    else:
        raise ValueError(f"Unknown solver '{solver}'.")

    # Recover original-scale solution.
    # The scaled system was A_scaled = diag(row_scale) @ A @ diag(col_scale),
    # so A_scaled @ x_scaled = b_scaled  implies  A @ (col_scale * x_scaled) = b.
    # Therefore x_true = col_scale * x_scaled  (NOT x_scaled / col_scale).
    x = x_scaled * col_scale

    # ----------------------------------------------------------------
    # Extract solution components
    # ----------------------------------------------------------------
    dJR_2d   = x[OFF_JR   : OFF_JR   + n].reshape(nR, nZ)
    dJZ_2d   = x[OFF_JZ   : OFF_JZ   + n].reshape(nR, nZ)
    dJPhi_2d = x[OFF_JPhi : OFF_JPhi + n].reshape(nR, nZ)
    dBR_2d   = x[OFF_BR   : OFF_BR   + n].reshape(nR, nZ)
    dBZ_2d   = x[OFF_BZ   : OFF_BZ   + n].reshape(nR, nZ)
    dBPhi_2d = x[OFF_BPhi : OFF_BPhi + n].reshape(nR, nZ)
    dp_2d    = x[OFF_P    : OFF_P    + n].reshape(nR, nZ)

    delta_B_plasma = _make_axi_vector_field(R, Z, dBR_2d, dBZ_2d, dBPhi_2d,
                                             name="delta_B_plasma")
    delta_J        = _make_axi_vector_field(R, Z, dJR_2d, dJZ_2d, dJPhi_2d,
                                             name="delta_J")
    delta_p        = _make_axi_scalar_field(R, Z, dp_2d,
                                             name="delta_p", units="Pa")

    return delta_B_plasma, delta_J, delta_p


# ---------------------------------------------------------------------------
# Stellarator equilibrium current computations
# ---------------------------------------------------------------------------

def compute_pfirsch_schlueter_current(
    B0_field: CylindricalVectorField,
    p_profile_func,
    R_arr,
    Z_arr,
    Phi_arr,
    R_axis: float,
    Z_axis: float,
    a_eff: float,
    n_phi: int = 32,
) -> CylindricalVectorField:
    """Compute the Pfirsch-Schlüter current in the cylindrical approximation.

    In a stellarator without net toroidal current, the Pfirsch-Schlüter (PS)
    current arises from ``∇·J = 0`` with toroidal variation of the magnetic
    field curvature.  The 2-D cylindrical estimate of the PS current density
    parallel to B is::

        J_PS = -(2/B²) * (dp/dr) * (B_pol/B) * (r/R) * B̂

    where r is the local minor radius from the magnetic axis and
    ``B_pol = sqrt(BR² + BZ²)`` is the poloidal field magnitude.

    Parameters
    ----------
    B0_field : CylindricalVectorField
        Background vacuum magnetic field.
    p_profile_func : callable
        ``p_profile_func(psi_norm) -> pressure [Pa]``.
        The pressure as a function of normalised flux (0 at axis, 1 at edge).
    R_arr, Z_arr, Phi_arr : array-like
        1-D coordinate arrays defining the output grid.
    R_axis, Z_axis : float
        Magnetic axis position [m].
    a_eff : float
        Effective minor radius [m] used to normalise psi.
    n_phi : int
        Not used (kept for API compatibility; the PS current is averaged
        over phi in the axisymmetric approximation).

    Returns
    -------
    J_PS : CylindricalVectorField
        Pfirsch-Schlüter current density [A m⁻²].
    """
    R_arr = np.asarray(R_arr)
    Z_arr = np.asarray(Z_arr)
    Phi_arr = np.asarray(Phi_arr)

    nR, nZ, nPhi = len(R_arr), len(Z_arr), len(Phi_arr)
    RR, ZZ = np.meshgrid(R_arr, Z_arr, indexing='ij')

    # Local minor radius from magnetic axis
    r_loc = np.sqrt((RR - R_axis) ** 2 + (ZZ - Z_axis) ** 2)
    psi_norm = np.clip(r_loc / a_eff, 0.0, 1.0)

    # dp/dr via finite difference of pressure profile
    delta_psi = 1e-4
    p_plus  = np.vectorize(p_profile_func)(np.clip(psi_norm + delta_psi, 0.0, 1.0))
    p_minus = np.vectorize(p_profile_func)(np.clip(psi_norm - delta_psi, 0.0, 1.0))
    dp_dpsi = (p_plus - p_minus) / (2.0 * delta_psi)   # Pa per unit psi_norm
    dp_dr   = dp_dpsi / (a_eff + 1e-30)                # Pa m⁻¹

    # Background field components (phi=0 slice, then broadcast)
    B0R_2d   = B0_field.VR[:, :, 0]
    B0Z_2d   = B0_field.VZ[:, :, 0]
    B0Phi_2d = B0_field.VPhi[:, :, 0]

    B2      = B0R_2d**2 + B0Z_2d**2 + B0Phi_2d**2 + 1e-30
    B_mag   = np.sqrt(B2)
    B_pol   = np.sqrt(B0R_2d**2 + B0Z_2d**2)

    # PS scalar amplitude (parallel to B)
    # J_PS_parallel = -(2/B²) * (dp/dr) * (B_pol/B) * (r/R)
    safe_R = np.where(RR > 0, RR, 1e-30)
    j_ps_parallel = -(2.0 / B2) * dp_dr * (B_pol / B_mag) * (r_loc / safe_R)

    # Project parallel component onto Cartesian-like B components
    JPS_R_2d   = j_ps_parallel * B0R_2d   / B_mag
    JPS_Z_2d   = j_ps_parallel * B0Z_2d   / B_mag
    JPS_Phi_2d = j_ps_parallel * B0Phi_2d / B_mag

    # Broadcast to 3-D (axisymmetric: replicate over Phi)
    JPS_R   = np.repeat(JPS_R_2d  [:, :, np.newaxis], nPhi, axis=2)
    JPS_Z   = np.repeat(JPS_Z_2d  [:, :, np.newaxis], nPhi, axis=2)
    JPS_Phi = np.repeat(JPS_Phi_2d[:, :, np.newaxis], nPhi, axis=2)

    return CylindricalVectorField(
        R=R_arr, Z=Z_arr, Phi=Phi_arr,
        VR=JPS_R, VZ=JPS_Z, VPhi=JPS_Phi,
        name="J_PS",
    )


def compute_diamagnetic_current(
    B0_field: CylindricalVectorField,
    p_profile_func,
    R_arr,
    Z_arr,
    Phi_arr,
    R_axis: float,
    Z_axis: float,
    a_eff: float,
) -> CylindricalVectorField:
    """Compute the diamagnetic current ``J_dia = (B × ∇p) / B²``.

    Parameters
    ----------
    B0_field : CylindricalVectorField
        Background vacuum magnetic field.
    p_profile_func : callable
        ``p_profile_func(psi_norm) -> pressure [Pa]``.
    R_arr, Z_arr, Phi_arr : array-like
        1-D coordinate arrays defining the output grid.
    R_axis, Z_axis : float
        Magnetic axis position [m].
    a_eff : float
        Effective minor radius [m].

    Returns
    -------
    J_dia : CylindricalVectorField
        Diamagnetic current density [A m⁻²].
    """
    R_arr = np.asarray(R_arr)
    Z_arr = np.asarray(Z_arr)
    Phi_arr = np.asarray(Phi_arr)

    nR, nZ, nPhi = len(R_arr), len(Z_arr), len(Phi_arr)
    RR, ZZ = np.meshgrid(R_arr, Z_arr, indexing='ij')

    # Pressure on grid
    r_loc    = np.sqrt((RR - R_axis) ** 2 + (ZZ - Z_axis) ** 2)
    psi_norm = np.clip(r_loc / a_eff, 0.0, 1.0)
    p_2d     = np.vectorize(p_profile_func)(psi_norm)

    # ∇p in cylindrical coords (axisymmetric → phi component is zero)
    grad_p_R   = np.gradient(p_2d, R_arr, axis=0)
    grad_p_Z   = np.gradient(p_2d, Z_arr, axis=1)
    grad_p_Phi = np.zeros_like(p_2d)

    # Background field (phi=0 slice)
    B0R_2d   = B0_field.VR[:, :, 0]
    B0Z_2d   = B0_field.VZ[:, :, 0]
    B0Phi_2d = B0_field.VPhi[:, :, 0]
    B2       = B0R_2d**2 + B0Z_2d**2 + B0Phi_2d**2 + 1e-30

    # J_dia = (B × ∇p) / B²
    # (B × ∇p)_R   = B_Z  * grad_p_Phi - B_Phi * grad_p_Z
    # (B × ∇p)_Z   = B_Phi * grad_p_R  - B_R  * grad_p_Phi
    # (B × ∇p)_Phi = B_R  * grad_p_Z  - B_Z  * grad_p_R
    Jdia_R_2d   = (B0Z_2d   * grad_p_Phi - B0Phi_2d * grad_p_Z  ) / B2
    Jdia_Z_2d   = (B0Phi_2d * grad_p_R   - B0R_2d   * grad_p_Phi) / B2
    Jdia_Phi_2d = (B0R_2d   * grad_p_Z   - B0Z_2d   * grad_p_R  ) / B2

    # Broadcast to 3-D
    Jdia_R   = np.repeat(Jdia_R_2d  [:, :, np.newaxis], nPhi, axis=2)
    Jdia_Z   = np.repeat(Jdia_Z_2d  [:, :, np.newaxis], nPhi, axis=2)
    Jdia_Phi = np.repeat(Jdia_Phi_2d[:, :, np.newaxis], nPhi, axis=2)

    return CylindricalVectorField(
        R=R_arr, Z=Z_arr, Phi=Phi_arr,
        VR=Jdia_R, VZ=Jdia_Z, VPhi=Jdia_Phi,
        name="J_dia",
    )


def compute_equilibrium_currents(
    B0_field: CylindricalVectorField,
    p_profile_func,
    R_arr,
    Z_arr,
    Phi_arr,
    R_axis: float,
    Z_axis: float,
    a_eff: float,
) -> tuple:
    """Compute total equilibrium currents (diamagnetic + Pfirsch-Schlüter).

    This gives the background ``J0`` and ``p0`` for the linearised GS solver
    without requiring a tokamak-style Ohmic current.

    Parameters
    ----------
    B0_field : CylindricalVectorField
        Background vacuum magnetic field.
    p_profile_func : callable
        ``p_profile_func(psi_norm) -> pressure [Pa]``.
    R_arr, Z_arr, Phi_arr : array-like
        1-D coordinate arrays defining the output grid.
    R_axis, Z_axis : float
        Magnetic axis position [m].
    a_eff : float
        Effective minor radius [m].

    Returns
    -------
    J_total : CylindricalVectorField
        Total equilibrium current density ``J_dia + J_PS`` [A m⁻²].
    p_grid : CylindricalScalarField
        Pressure on the grid [Pa].
    """
    R_arr   = np.asarray(R_arr)
    Z_arr   = np.asarray(Z_arr)
    Phi_arr = np.asarray(Phi_arr)
    nPhi    = len(Phi_arr)

    J_dia = compute_diamagnetic_current(
        B0_field, p_profile_func, R_arr, Z_arr, Phi_arr, R_axis, Z_axis, a_eff
    )
    J_PS = compute_pfirsch_schlueter_current(
        B0_field, p_profile_func, R_arr, Z_arr, Phi_arr, R_axis, Z_axis, a_eff
    )

    JR_tot   = J_dia.VR   + J_PS.VR
    JZ_tot   = J_dia.VZ   + J_PS.VZ
    JPhi_tot = J_dia.VPhi + J_PS.VPhi

    J_total = CylindricalVectorField(
        R=R_arr, Z=Z_arr, Phi=Phi_arr,
        VR=JR_tot, VZ=JZ_tot, VPhi=JPhi_tot,
        name="J_total",
    )

    # Build pressure scalar field
    RR, ZZ = np.meshgrid(R_arr, Z_arr, indexing='ij')
    r_loc    = np.sqrt((RR - R_axis) ** 2 + (ZZ - Z_axis) ** 2)
    psi_norm = np.clip(r_loc / a_eff, 0.0, 1.0)
    p_2d     = np.vectorize(p_profile_func)(psi_norm)
    p_3d     = np.repeat(p_2d[:, :, np.newaxis], nPhi, axis=2)

    p_grid = CylindricalScalarField(
        R=R_arr, Z=Z_arr, Phi=Phi_arr,
        value=p_3d, name="p0", units="Pa",
    )

    return J_total, p_grid


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def compute_plasma_response(
    eq,
    delta_B_ext: CylindricalVectorField,
    R_grid=None,
    Z_grid=None,
    **kwargs,
) -> CylindricalVectorField:
    """Compute total δB = δB_ext + δB_plasma using perturbed GS solver.

    Parameters
    ----------
    eq : object
        Equilibrium object.  Must expose:
        - ``eq.B_field_grid(R, Z)`` → ``CylindricalVectorField``, OR
        - ``eq.B0``, ``eq.J0``, ``eq.p0`` as ``CylindricalVectorField``/
          ``CylindricalScalarField`` already on the same grid.
        Also accepts a :class:`~pyna.mag.Solovev.EquilibriumSolovev` directly.
    delta_B_ext : CylindricalVectorField
        External (vacuum) perturbation field on R-Z grid.
    R_grid, Z_grid : array-like or None
        Grid to evaluate equilibrium fields on.  If None, the grid from
        ``delta_B_ext`` is used.
    **kwargs
        Forwarded to :func:`solve_perturbed_gs`.

    Returns
    -------
    delta_B_total : CylindricalVectorField
        Total perturbation δB_ext + δB_plasma on the same R-Z grid.
    """
    from pyna.MCF.equilibrium.Solovev import EquilibriumSolovev

    if R_grid is None:
        R_grid = delta_B_ext.R
    if Z_grid is None:
        Z_grid = delta_B_ext.Z

    R_arr = np.asarray(R_grid)
    Z_arr = np.asarray(Z_grid)

    # Build B0, J0, p0 from equilibrium
    if isinstance(eq, EquilibriumSolovev):
        B0, J0, p0 = _solovev_grid_fields(eq, R_arr, Z_arr)
    elif hasattr(eq, 'B0') and isinstance(eq.B0, CylindricalVectorField):
        B0, J0, p0 = eq.B0, eq.J0, eq.p0
    else:
        raise TypeError(
            f"eq must be a EquilibriumSolovev or have .B0/.J0/.p0 CylindricalVectorField attributes. "
            f"Got {type(eq)}"
        )

    delta_B_plasma, _, _ = solve_perturbed_gs(B0, J0, p0, delta_B_ext, **kwargs)

    # δB_total = δB_ext + δB_plasma  (component-wise)
    dBtot_R   = delta_B_ext.VR[:, :, 0]   + delta_B_plasma.VR[:, :, 0]
    dBtot_Z   = delta_B_ext.VZ[:, :, 0]   + delta_B_plasma.VZ[:, :, 0]
    dBtot_Phi = delta_B_ext.VPhi[:, :, 0] + delta_B_plasma.VPhi[:, :, 0]

    return _make_axi_vector_field(R_arr, Z_arr, dBtot_R, dBtot_Z, dBtot_Phi,
                                  name="delta_B_total")


def _solovev_grid_fields(eq, R_arr, Z_arr):
    """Build B0, J0, p0 CylindricalVectorField / CylindricalScalarField from EquilibriumSolovev."""
    from pyna.MCF.equilibrium.Solovev import EquilibriumSolovev

    nR, nZ = len(R_arr), len(Z_arr)
    RR, ZZ = np.meshgrid(R_arr, Z_arr, indexing='ij')

    BR_2d, BZ_2d = eq.BR_BZ(RR, ZZ)
    BPhi_2d = eq.Bphi(RR)

    # J = ∇×B / μ₀ using eq.J_grid if available
    if hasattr(eq, 'J_grid'):
        JR_2d, JZ_2d, JPhi_2d = eq.J_grid(R_arr, Z_arr)
    else:
        mu0 = 4e-7 * np.pi
        dR = R_arr[1] - R_arr[0]
        dZ = Z_arr[1] - Z_arr[0]
        dBPhi_dZ = np.gradient(BPhi_2d, dZ, axis=1)
        d_RBPhi_dR = np.gradient(RR * BPhi_2d, dR, axis=0)
        dBR_dZ = np.gradient(BR_2d, dZ, axis=1)
        dBZ_dR = np.gradient(BZ_2d, dR, axis=0)
        JR_2d   = -dBPhi_dZ / mu0
        JZ_2d   =  d_RBPhi_dR / (RR * mu0)
        JPhi_2d = (dBR_dZ - dBZ_dR) / mu0

    # p from J_grid if available, else zeros
    if hasattr(eq, 'p_grid'):
        p_2d = eq.p_grid(R_arr, Z_arr)
    else:
        p_2d = np.zeros((nR, nZ))

    B0 = _make_axi_vector_field(R_arr, Z_arr, BR_2d, BZ_2d, BPhi_2d, name="B0")
    J0 = _make_axi_vector_field(R_arr, Z_arr, JR_2d, JZ_2d, JPhi_2d, name="J0")
    p0 = _make_axi_scalar_field(R_arr, Z_arr, p_2d, name="p0", units="Pa")
    return B0, J0, p0
