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

from pyna.field_data import CylindricalVectorField, CylindricalScalarField

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

    EqBdiv_weight   = 1e8
    BC_no_J_weight  = 1e9
    BC_no_p_weight  = 1e12

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
        Also accepts a :class:`~pyna.mag.Solovev.SolovevEquilibrium` directly.
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
    from pyna.mag.Solovev import SolovevEquilibrium

    if R_grid is None:
        R_grid = delta_B_ext.R
    if Z_grid is None:
        Z_grid = delta_B_ext.Z

    R_arr = np.asarray(R_grid)
    Z_arr = np.asarray(Z_grid)

    # Build B0, J0, p0 from equilibrium
    if isinstance(eq, SolovevEquilibrium):
        B0, J0, p0 = _solovev_grid_fields(eq, R_arr, Z_arr)
    elif hasattr(eq, 'B0') and isinstance(eq.B0, CylindricalVectorField):
        B0, J0, p0 = eq.B0, eq.J0, eq.p0
    else:
        raise TypeError(
            f"eq must be a SolovevEquilibrium or have .B0/.J0/.p0 CylindricalVectorField attributes. "
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
    """Build B0, J0, p0 CylindricalVectorField / CylindricalScalarField from SolovevEquilibrium."""
    from pyna.mag.Solovev import SolovevEquilibrium

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
