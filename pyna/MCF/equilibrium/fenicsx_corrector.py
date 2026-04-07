"""
pyna/pyna/MCF/equilibrium/fenicsx_corrector.py
==============================================
FEniCSx-based force-balance corrector for MHD equilibria.

Given a predicted field B_pred (from FPT linearisation), computes
the correction δB such that (∇×(B_pred+δB)/μ₀)×(B_pred+δB) ≈ ∇p.

Uses dolfinx 0.10.x API on a 2D (R,Z) rectangular mesh.

Numerical improvements
----------------------
- Anderson acceleration (DIIS) for the outer Newton loop
- Backtracking line search (Armijo condition) to prevent overshooting
- Convergence monitoring: prints residual ratio each iteration
- Early exit if residual grows > 2× (diverging)
"""
from __future__ import annotations

import copy
import numpy as np
from typing import Optional, Tuple
from math import pi

MU0_DEFAULT = 4e-7 * pi


# ---------------------------------------------------------------------------
# Anderson acceleration
# ---------------------------------------------------------------------------

class AndersonMixer:
    """Anderson acceleration for fixed-point iteration x_{k+1} = G(x_k).

    Stores the last m iterates and residuals, computes optimal linear
    combination to minimise the residual norm.

    Reference: Walker & Ni (2011), SIAM J. Numer. Anal.
    """

    def __init__(self, m: int = 5, beta: float = 1.0):
        """
        Parameters
        ----------
        m    : history depth (5–10 is typical)
        beta : mixing parameter (1.0 = full step, <1 = damping)
        """
        self.m = m
        self.beta = beta
        self._F_hist: list = []   # residual history
        self._x_hist: list = []   # iterate history

    def update(self, x_k: np.ndarray, F_k: np.ndarray) -> np.ndarray:
        """Return the Anderson-accelerated next iterate.

        Parameters
        ----------
        x_k : ndarray  — flattened current field state
        F_k : ndarray  — flattened residual = G(x_k) - x_k  (= δB correction)

        Returns
        -------
        x_next : ndarray — accelerated next iterate
        """
        self._x_hist.append(x_k.copy())
        self._F_hist.append(F_k.copy())

        if len(self._F_hist) > self.m + 1:
            self._F_hist.pop(0)
            self._x_hist.pop(0)

        m_k = len(self._F_hist) - 1
        if m_k == 0:
            return x_k + self.beta * F_k

        # Build ΔF matrix: columns are F_{i+1} - F_i
        F_mat = np.column_stack([
            self._F_hist[i + 1] - self._F_hist[i]
            for i in range(m_k)
        ])  # shape (N, m_k)

        # Solve min_θ ‖F_k - ΔF @ θ‖²
        theta, _, _, _ = np.linalg.lstsq(F_mat, self._F_hist[-1], rcond=None)

        # Anderson update
        x_next = x_k + self.beta * self._F_hist[-1]
        for i in range(m_k):
            x_next -= theta[i] * (
                (self._x_hist[i + 1] + self.beta * self._F_hist[i + 1]) -
                (self._x_hist[i]     + self.beta * self._F_hist[i])
            )
        return x_next

    def reset(self):
        self._F_hist.clear()
        self._x_hist.clear()


# ---------------------------------------------------------------------------
# Mesh builder
# ---------------------------------------------------------------------------

def build_rz_mesh(R_arr, Z_arr):
    """Build a dolfinx rectangular mesh from 1D R and Z coordinate arrays."""
    from dolfinx import mesh as dmesh
    from mpi4py import MPI

    R_arr = np.asarray(R_arr, dtype=np.float64)
    Z_arr = np.asarray(Z_arr, dtype=np.float64)
    nR, nZ = len(R_arr), len(Z_arr)

    msh = dmesh.create_rectangle(
        MPI.COMM_WORLD,
        [[float(R_arr[0]), float(Z_arr[0])],
         [float(R_arr[-1]), float(Z_arr[-1])]],
        [nR - 1, nZ - 1],
        cell_type=dmesh.CellType.triangle,
    )
    return msh


# ---------------------------------------------------------------------------
# Array → dolfinx function
# ---------------------------------------------------------------------------

def array_to_dolfinx_function(mesh, values_2d, R_arr, Z_arr, degree: int = 1):
    """Interpolate a 2D numpy array (shape nR×nZ) onto a dolfinx scalar Function."""
    from dolfinx import fem
    V = fem.functionspace(mesh, ("CG", degree))
    f = fem.Function(V)

    R_arr = np.asarray(R_arr, dtype=np.float64)
    Z_arr = np.asarray(Z_arr, dtype=np.float64)
    vals  = np.asarray(values_2d, dtype=np.float64)

    def _interp(x):
        R_q = np.clip(x[0], R_arr[0], R_arr[-1])
        Z_q = np.clip(x[1], Z_arr[0], Z_arr[-1])
        i_R = np.clip(np.searchsorted(R_arr, R_q, 'right') - 1, 0, len(R_arr) - 2)
        i_Z = np.clip(np.searchsorted(Z_arr, Z_q, 'right') - 1, 0, len(Z_arr) - 2)
        dR = np.where((R_arr[i_R + 1] - R_arr[i_R]) == 0, 1.0, R_arr[i_R + 1] - R_arr[i_R])
        dZ = np.where((Z_arr[i_Z + 1] - Z_arr[i_Z]) == 0, 1.0, Z_arr[i_Z + 1] - Z_arr[i_Z])
        t_R = (R_q - R_arr[i_R]) / dR
        t_Z = (Z_q - Z_arr[i_Z]) / dZ
        v00 = vals[i_R,     i_Z    ]
        v10 = vals[i_R + 1, i_Z    ]
        v01 = vals[i_R,     i_Z + 1]
        v11 = vals[i_R + 1, i_Z + 1]
        return (1 - t_R) * (1 - t_Z) * v00 + t_R * (1 - t_Z) * v10 \
             + (1 - t_R) * t_Z * v01        + t_R * t_Z * v11

    f.interpolate(_interp)
    return f


# ---------------------------------------------------------------------------
# Interpolate 3-component vector field onto dolfinx
# ---------------------------------------------------------------------------

def interpolate_vector_field(mesh, V_2d, R_arr, Z_arr):
    """Interpolate a (3, nR, nZ) numpy array onto a dolfinx vector Function."""
    from dolfinx import fem
    V = fem.functionspace(mesh, ("CG", 1, (3,)))
    f = fem.Function(V)

    R_arr = np.asarray(R_arr, dtype=np.float64)
    Z_arr = np.asarray(Z_arr, dtype=np.float64)
    V_2d  = np.asarray(V_2d,  dtype=np.float64)

    def _interp(x):
        R_q = np.clip(x[0], R_arr[0], R_arr[-1])
        Z_q = np.clip(x[1], Z_arr[0], Z_arr[-1])
        nR  = len(R_arr) - 1
        nZ  = len(Z_arr) - 1
        i_R = np.clip(np.searchsorted(R_arr, R_q, 'right') - 1, 0, nR - 1)
        i_Z = np.clip(np.searchsorted(Z_arr, Z_q, 'right') - 1, 0, nZ - 1)
        dR  = np.where((R_arr[i_R + 1] - R_arr[i_R]) == 0, 1.0, R_arr[i_R + 1] - R_arr[i_R])
        dZ  = np.where((Z_arr[i_Z + 1] - Z_arr[i_Z]) == 0, 1.0, Z_arr[i_Z + 1] - Z_arr[i_Z])
        t_R = (R_q - R_arr[i_R]) / dR
        t_Z = (Z_q - Z_arr[i_Z]) / dZ
        out = np.zeros((3, x.shape[1]))
        for c in range(3):
            v00 = V_2d[c, i_R,     i_Z    ]
            v10 = V_2d[c, i_R + 1, i_Z    ]
            v01 = V_2d[c, i_R,     i_Z + 1]
            v11 = V_2d[c, i_R + 1, i_Z + 1]
            out[c] = ((1 - t_R) * (1 - t_Z) * v00
                    + t_R       * (1 - t_Z) * v10
                    + (1 - t_R) * t_Z       * v01
                    + t_R       * t_Z       * v11)
        return out

    f.interpolate(_interp)
    return f


# ---------------------------------------------------------------------------
# Extract dolfinx Function back to numpy grid
# ---------------------------------------------------------------------------

def extract_to_grid(sol, R_arr, Z_arr):
    """Evaluate a dolfinx vector Function (3 components) on the (R, Z) grid."""
    import dolfinx
    R_arr = np.asarray(R_arr, dtype=np.float64)
    Z_arr = np.asarray(Z_arr, dtype=np.float64)
    nR, nZ = len(R_arr), len(Z_arr)

    RR, ZZ = np.meshgrid(R_arr, Z_arr, indexing='ij')
    pts = np.column_stack([RR.ravel(), ZZ.ravel(), np.zeros(nR * nZ)])

    msh = sol.function_space.mesh
    bb_tree = dolfinx.geometry.bb_tree(msh, msh.topology.dim)
    cells   = []
    points_on_proc = []
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(msh, cell_candidates, pts)
    for i, pt in enumerate(pts):
        cs = colliding_cells.links(i)
        if len(cs) > 0:
            cells.append(cs[0])
            points_on_proc.append(pt)
        else:
            cells.append(0)
            points_on_proc.append(pt)

    pts_arr = np.array(points_on_proc, dtype=np.float64)
    vals = sol.eval(pts_arr, cells)

    if vals.ndim == 1:
        vals = vals[:, np.newaxis]

    ncomp = vals.shape[1] if vals.ndim > 1 else 1
    out = np.zeros((3, nR, nZ))
    for c in range(min(ncomp, 3)):
        out[c] = vals[:, c].reshape(nR, nZ)
    return out


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def compute_curl_cylindrical(B_2d, R_arr, Z_arr, mu0=MU0_DEFAULT):
    """Compute J = ∇×B/μ₀ on the (R,Z) grid using numpy.gradient.

    Parameters
    ----------
    B_2d : ndarray, shape (3, nR, nZ)  — [BR, BPhi, BZ]
    R_arr, Z_arr : 1D arrays
    mu0 : float

    Returns
    -------
    J_2d : ndarray, shape (3, nR, nZ)  — [JR, JZ, JPhi]
    """
    BR, BPhi, BZ = B_2d[0], B_2d[1], B_2d[2]
    RR, _ = np.meshgrid(R_arr, np.ones(len(Z_arr)), indexing='ij')

    dBPhi_dZ  = np.gradient(BPhi, Z_arr, axis=1)
    dBPhi_dR  = np.gradient(BPhi, R_arr, axis=0)
    dBR_dZ    = np.gradient(BR,   Z_arr, axis=1)
    dBZ_dR    = np.gradient(BZ,   R_arr, axis=0)

    J_R   = -dBPhi_dZ / mu0
    J_Z   = (BPhi / (RR + 1e-30) + dBPhi_dR) / mu0
    J_Phi = (dBR_dZ - dBZ_dR) / mu0

    return np.stack([J_R, J_Z, J_Phi], axis=0)


def compute_force_residual(J_2d, B_2d, p_2d, R_arr, Z_arr):
    """Compute r = (J×B) - ∇p in (R, Z) components.

    Parameters
    ----------
    J_2d : (3, nR, nZ) — [JR, JZ, JPhi]
    B_2d : (3, nR, nZ) — [BR, BPhi, BZ]
    p_2d : (nR, nZ)
    R_arr, Z_arr : 1D arrays

    Returns
    -------
    r_R, r_Z : each (nR, nZ)
    """
    JR, JZ, JPhi = J_2d[0], J_2d[1], J_2d[2]
    BR, BPhi, BZ = B_2d[0], B_2d[1], B_2d[2]

    fR = JZ * BPhi - JPhi * BZ
    fZ = JPhi * BR - JR * BPhi

    grad_p_R = np.gradient(p_2d, R_arr, axis=0)
    grad_p_Z = np.gradient(p_2d, Z_arr, axis=1)

    r_R = fR - grad_p_R
    r_Z = fZ - grad_p_Z
    return r_R, r_Z


def _residual_norm(B_2d, p_2d, R_arr, Z_arr, mu0=MU0_DEFAULT):
    """Compute scalar force-balance residual ‖J×B - ∇p‖ (RMS)."""
    J_2d = compute_curl_cylindrical(B_2d, R_arr, Z_arr, mu0)
    r_R, r_Z = compute_force_residual(J_2d, B_2d, p_2d, R_arr, Z_arr)
    return float(np.sqrt(np.mean(r_R ** 2 + r_Z ** 2)))


# ---------------------------------------------------------------------------
# Line search
# ---------------------------------------------------------------------------

def _line_search_damp(B_curr, delta_B, p_2d, R_arr, Z_arr, mu0=MU0_DEFAULT,
                      max_damp_iter: int = 8, alpha_init: float = 1.0,
                      armijo_c: float = 0.5):
    """Backtracking line search: find α ∈ (0,1] such that Armijo condition holds.

    res(B + α·δB) < (1 - armijo_c·α) * res(B)

    Parameters
    ----------
    B_curr, delta_B : (3, nR, nZ) arrays
    p_2d : (nR, nZ)
    R_arr, Z_arr : 1D grids
    mu0, max_damp_iter, alpha_init, armijo_c : scalars

    Returns
    -------
    alpha : float — accepted step size
    """
    res_curr = _residual_norm(B_curr, p_2d, R_arr, Z_arr, mu0)
    alpha = alpha_init
    for _ in range(max_damp_iter):
        B_trial = B_curr + alpha * delta_B
        res_trial = _residual_norm(B_trial, p_2d, R_arr, Z_arr, mu0)
        if res_trial < (1.0 - armijo_c * alpha) * res_curr:
            return alpha
        alpha *= 0.5
    return alpha  # accept best found even if Armijo not satisfied


# ---------------------------------------------------------------------------
# FEniCSx linearised force-balance solver
# ---------------------------------------------------------------------------

def solve_linearised_fb(B_curr, J_curr, r_R, r_Z, R_arr, Z_arr, eps_reg,
                        lambda_div=1.0, mu0=MU0_DEFAULT,
                        petsc_solver: str = "gamg"):
    """Solve the linearised force-balance system for δB using FEniCSx.

    Parameters
    ----------
    B_curr : (3, nR, nZ)  — [BR, BPhi, BZ]
    J_curr : (3, nR, nZ)  — [JR, JZ, JPhi]
    r_R, r_Z : (nR, nZ)  — force residual components
    R_arr, Z_arr : 1D grids
    eps_reg : float  — regularisation weight
    mu0 : float

    Returns
    -------
    delta : ndarray shape (3, nR, nZ)  — [δBR, δBPhi, δBZ]
    """
    import dolfinx
    from dolfinx import fem
    from dolfinx.fem.petsc import LinearProblem
    import ufl

    msh = build_rz_mesh(R_arr, Z_arr)

    V = fem.functionspace(msh, ("CG", 1, (3,)))
    delta = ufl.TrialFunction(V)
    v     = ufl.TestFunction(V)

    B_f = interpolate_vector_field(msh, B_curr, R_arr, Z_arr)
    J_f = interpolate_vector_field(msh, J_curr, R_arr, Z_arr)

    x       = ufl.SpatialCoordinate(msh)
    R_coord = x[0]

    dBR, dBPhi, dBZ = delta[0], delta[1], delta[2]

    curlR   = -dBPhi.dx(1)
    curlZ   =  dBPhi / R_coord + dBPhi.dx(0)
    curlPhi =  dBR.dx(1) - dBZ.dx(0)

    dJ_R   = curlR   / mu0
    dJ_Z   = curlZ   / mu0
    dJ_Phi = curlPhi / mu0

    BR_f, BPhi_f, BZ_f = B_f[0], B_f[1], B_f[2]
    JR_f, JPhi_f, JZ_f = J_f[0], J_f[1], J_f[2]

    df_R   = (dJ_Z * BPhi_f - dJ_Phi * BZ_f) + (JZ_f * delta[1] - JPhi_f * delta[2])
    df_Phi = (dJ_R * BZ_f   - dJ_Z   * BR_f) + (JR_f * delta[2] - JZ_f   * delta[0])
    df_Z   = (dJ_Phi * BR_f - dJ_R   * BPhi_f) + (JPhi_f * delta[0] - JR_f * delta[1])

    div_delta = dBR / R_coord + dBR.dx(0) + dBZ.dx(1)
    div_v     = v[0] / R_coord + v[0].dx(0) + v[2].dx(1)

    a_form  = (df_R   * v[0] + df_Phi * v[1] + df_Z * v[2]) * R_coord * ufl.dx
    a_form += lambda_div * div_delta * div_v * R_coord * ufl.dx
    a_form += eps_reg * ufl.inner(delta, v) * R_coord * ufl.dx

    JR_v, JPhi_v, JZ_v = J_curr[0], J_curr[1], J_curr[2]
    BR_v, BPhi_v, BZ_v = B_curr[0], B_curr[1], B_curr[2]
    r_Phi_arr = JPhi_v * BR_v - JR_v * BZ_v
    r_3d = np.stack([r_R, r_Phi_arr, r_Z], axis=0)
    r_f  = interpolate_vector_field(msh, r_3d, R_arr, Z_arr)
    L_form = -(r_f[0] * v[0] + r_f[1] * v[1] + r_f[2] * v[2]) * R_coord * ufl.dx

    # Build PETSc solver options (gamg=algebraic multigrid, ilu=fallback)
    if petsc_solver == "gamg":
        _petsc_opts = {
            "ksp_type": "gmres",
            "pc_type": "gamg",
            "pc_gamg_type": "agg",
            "pc_gamg_agg_nsmooths": 1,
            "ksp_rtol": 1e-8,
            "ksp_max_it": 500,
            "ksp_gmres_restart": 50,
        }
    elif petsc_solver == "hypre":
        _petsc_opts = {
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "ksp_rtol": 1e-8,
            "ksp_max_it": 500,
            "ksp_gmres_restart": 50,
        }
    else:
        # "ilu" or unknown -- fallback
        if petsc_solver not in ("ilu", "gamg", "hypre"):
            import warnings
            warnings.warn(
                f"Unknown petsc_solver={petsc_solver!r}; falling back to ILU.",
                stacklevel=3,
            )
        _petsc_opts = {
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "ksp_rtol": 1e-8,
            "ksp_max_it": 500,
        }

    problem = LinearProblem(
        a_form, L_form,
        bcs=[],
        petsc_options_prefix="fenicsx_mhd_",
        petsc_options=_petsc_opts,
    )
    sol = problem.solve()
    return extract_to_grid(sol, R_arr, Z_arr)


# ---------------------------------------------------------------------------
# Main corrector  (with Anderson + line search)
# ---------------------------------------------------------------------------

def solve_force_balance_correction(
    B_pred_RZPhi_2d,
    p_2d,
    R_arr,
    Z_arr,
    mu0: float = MU0_DEFAULT,
    eps_reg: float = 1e-6,
    lambda_div: float = 1.0,
    max_iter: int = 3,
    anderson_depth: int = 5,
    anderson_beta: float = 0.8,
    use_line_search: bool = True,
    diverge_tol: float = 2.0,
    petsc_solver: str = "gamg",
):
    """Newton-iteration FEniCSx corrector for force-balance.

    Includes Anderson acceleration (DIIS) and backtracking line search.

    Parameters
    ----------
    B_pred_RZPhi_2d : ndarray, shape (3, nR, nZ)  — [BR, BPhi, BZ]
    p_2d : ndarray, shape (nR, nZ)
    R_arr, Z_arr : 1D coordinate arrays
    mu0 : float
    eps_reg : float — Tikhonov regularisation weight
    lambda_div : float — div-free penalty weight
    max_iter : int  — maximum Newton iterations
    anderson_depth : int — Anderson mixing history depth (0 = plain Newton)
    anderson_beta : float — Anderson mixing parameter
    use_line_search : bool — enable backtracking line search
    diverge_tol : float — if residual_ratio > diverge_tol, exit early (diverging)

    Returns
    -------
    delta_B_2d : ndarray, shape (3, nR, nZ) — total correction [δBR, δBPhi, δBZ]
    residual_history : list[float] — residual at each iteration
    """
    B_curr = np.array(B_pred_RZPhi_2d, dtype=np.float64)
    p_2d   = np.asarray(p_2d, dtype=np.float64)
    R_arr  = np.asarray(R_arr, dtype=np.float64)
    Z_arr  = np.asarray(Z_arr, dtype=np.float64)

    mixer = AndersonMixer(m=anderson_depth, beta=anderson_beta)
    B_flat = B_curr.ravel()
    residual_history = []

    resid_0 = _residual_norm(B_curr, p_2d, R_arr, Z_arr, mu0)
    residual_history.append(resid_0)
    print(f"  [corrector] iter 0 (initial): residual = {resid_0:.4e}")

    if resid_0 < 1e-8:
        return B_curr - B_pred_RZPhi_2d, residual_history

    for iteration in range(max_iter):
        J_curr = compute_curl_cylindrical(B_curr, R_arr, Z_arr, mu0)
        r_R, r_Z = compute_force_residual(J_curr, B_curr, p_2d, R_arr, Z_arr)

        delta = solve_linearised_fb(B_curr, J_curr, r_R, r_Z, R_arr, Z_arr,
                                    eps_reg, lambda_div=lambda_div, mu0=mu0,
                                    petsc_solver=petsc_solver)

        # Line search
        if use_line_search:
            alpha = _line_search_damp(B_curr, delta, p_2d, R_arr, Z_arr, mu0)
            if alpha < 1.0:
                print(f"  [corrector]   line search: α={alpha:.3f}")
            delta_damped = alpha * delta
        else:
            # Fallback: hard clamp if step is enormous
            B_rms = float(np.sqrt(np.mean(B_curr ** 2))) + 1e-30
            d_rms = float(np.sqrt(np.mean(delta ** 2))) + 1e-30
            if d_rms > 0.01 * B_rms:
                delta *= 0.01 * B_rms / d_rms
            delta_damped = delta

        # Anderson-accelerated update
        delta_flat = delta_damped.ravel()
        B_flat_new = mixer.update(B_flat, delta_flat)
        B_curr = B_flat_new.reshape(B_curr.shape)
        B_flat = B_flat_new

        resid_new = _residual_norm(B_curr, p_2d, R_arr, Z_arr, mu0)
        residual_history.append(resid_new)
        ratio = resid_new / (resid_0 + 1e-30)
        print(f"  [corrector] iter {iteration + 1}: "
              f"residual = {resid_new:.4e}  (ratio={ratio:.3f})")

        if resid_new < 1e-8:
            print("  [corrector] converged.")
            break

        if resid_new > diverge_tol * resid_0 and iteration > 0:
            print(f"  [corrector] WARNING: residual growing (ratio={ratio:.2f}), "
                  "stopping early.")
            break

    return B_curr - B_pred_RZPhi_2d, residual_history


# ---------------------------------------------------------------------------
# Main interface: one FPT-predictor + FEniCSx-corrector beta step
# ---------------------------------------------------------------------------

def fpt_fenicsx_beta_step(
    field_cache: dict,
    beta_old: float,
    beta_new: float,
    R_axis: float,
    Z_axis: float,
    a_eff: float,
    alpha_pressure: float = 2.0,
    phi_slice: int = 0,
    eps_reg: float = 1e-6,
    max_newton_iter: int = 3,
    verbose: bool = True,
    mu0: float = MU0_DEFAULT,
    anderson_depth: int = 5,
    anderson_beta: float = 0.8,
    use_line_search: bool = True,
    iota_profile: Optional[np.ndarray] = None,
    r_eff_surfaces: Optional[Tuple] = None,
) -> dict:
    """One step of FPT predictor + FEniCSx corrector.

    Parameters
    ----------
    field_cache : dict
        Standard pyna field cache with keys:
        'BR', 'BPhi', 'BZ' : shape (nR, nZ, nPhi_ext)
        'R_grid', 'Z_grid', 'Phi_grid' : 1D arrays
    beta_old, beta_new : float
        Previous and target β values.
    R_axis, Z_axis : float
        Magnetic axis position [m].
    a_eff : float
        Effective minor radius [m].
    alpha_pressure : float
        Pressure profile exponent.
    phi_slice : int
        Which φ index to use for the 2D solve.
    eps_reg : float
        FEniCSx regularisation weight.
    max_newton_iter : int
        Newton iterations inside the corrector.
    verbose : bool
    anderson_depth : int
        Anderson mixing history depth (0 = plain Newton).
    anderson_beta : float
        Anderson mixing parameter (damping, 0 < β ≤ 1).
    use_line_search : bool
        Enable backtracking line search inside the corrector.
    iota_profile : ndarray or None
        Optional shape (n_surfaces, 2) array of (r_eff, iota) values.
        Currently reserved for future use; not consumed in this function.
    r_eff_surfaces : tuple or None
        Optional ``(r_eff_arr, R_seeds, Z_seeds)`` triple describing
        magnetic surface seed points on the phi=0 plane (from pyna tracing).
        ``r_eff_arr`` : 1D array of effective minor radii [m].
        ``R_seeds``, ``Z_seeds`` : 1D arrays of seed positions [m].
        When provided, the pressure profile is computed by nearest-neighbour
        mapping from grid points to magnetic surfaces, giving a better
        approximation for non-circular cross-sections (e.g. HAO stellarators).
        When None (default), the original circular-cross-section approximation
        is used (fully backwards-compatible).

    Returns
    -------
    dict with keys:
        'field_cache'        : updated field cache
        'residual_before'    : force-balance residual norm before correction
        'residual_after'     : force-balance residual norm after correction
        'delta_beta'         : beta_new - beta_old
        'B_correction_norm'  : L2 norm of δB correction applied
        'newton_iters'       : number of Newton iterations taken
        'residual_history'   : list of residuals per Newton iteration
    """
    R_arr   = np.asarray(field_cache['R_grid'], dtype=np.float64)
    Z_arr   = np.asarray(field_cache['Z_grid'], dtype=np.float64)
    nR, nZ  = len(R_arr), len(Z_arr)
    nPhi_ext = field_cache['BR'].shape[2]
    n_phi    = nPhi_ext - 1

    BR_2d   = field_cache['BR']  [:, :, phi_slice]
    BPhi_2d = field_cache['BPhi'][:, :, phi_slice]
    BZ_2d   = field_cache['BZ']  [:, :, phi_slice]
    B_pred_2d = np.stack([BR_2d, BPhi_2d, BZ_2d], axis=0)

    # Build pressure profile for beta_new
    RR, ZZ = np.meshgrid(R_arr, Z_arr, indexing='ij')

    if r_eff_surfaces is not None:
        # ------------------------------------------------------------------
        # r_eff-based pressure profile (better for non-circular stellarators)
        # ------------------------------------------------------------------
        r_eff_arr = np.asarray(r_eff_surfaces[0], dtype=np.float64)
        R_seeds   = np.asarray(r_eff_surfaces[1], dtype=np.float64)
        Z_seeds   = np.asarray(r_eff_surfaces[2], dtype=np.float64)

        # For each grid point find the nearest seed (in R-Z plane)
        # Shapes: RR/ZZ (nR, nZ), seeds (n_surf,)
        dR = RR[:, :, np.newaxis] - R_seeds[np.newaxis, np.newaxis, :]  # (nR,nZ,n_surf)
        dZ = ZZ[:, :, np.newaxis] - Z_seeds[np.newaxis, np.newaxis, :]
        dist2 = dR ** 2 + dZ ** 2
        nearest_idx = np.argmin(dist2, axis=2)  # (nR, nZ)
        r_eff_grid  = r_eff_arr[nearest_idx]    # (nR, nZ)

        p_shape = np.maximum(0.0, 1.0 - (r_eff_grid / max(a_eff, 1e-15)) ** 2) ** alpha_pressure
        if verbose:
            print("  [pressure] using r_eff-based profile "
                  f"(n_surfaces={len(r_eff_arr)})")
    else:
        # ------------------------------------------------------------------
        # Original circular cross-section approximation (backwards-compatible)
        # ------------------------------------------------------------------
        psi_norm = ((RR - R_axis) ** 2 + (ZZ - Z_axis) ** 2) / max(a_eff ** 2, 1e-30)
        p_shape  = np.maximum(0.0, 1.0 - psi_norm) ** alpha_pressure

    B2_mean = float(np.mean(BR_2d ** 2 + BPhi_2d ** 2 + BZ_2d ** 2))
    p0_new  = beta_new * B2_mean / (2.0 * mu0) * (alpha_pressure + 1.0)
    p_new_2d = p0_new * p_shape

    # FPT predictor
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _pyna_root = str(_Path(__file__).resolve().parents[3])
        if _pyna_root not in _sys.path:
            _sys.path.insert(0, _pyna_root)
        from pyna.MCF.plasma_response.PerturbGS import solve_perturbed_gs_coupled
        from pyna.fields.cylindrical import VectorField3DCylindrical as _CVF
        from pyna.fields.cylindrical import ScalarField3DCylindrical as _CSF

        Phi_1d = np.array([0.0])

        B0_field = _CVF(
            R=R_arr, Z=Z_arr, Phi=Phi_1d,
            VR=B_pred_2d[0, :, :, np.newaxis],
            VZ=B_pred_2d[2, :, :, np.newaxis],
            VPhi=B_pred_2d[1, :, :, np.newaxis],
        )

        J_pred_3d = compute_curl_cylindrical(B_pred_2d, R_arr, Z_arr, mu0)
        J0_field = _CVF(
            R=R_arr, Z=Z_arr, Phi=Phi_1d,
            VR=J_pred_3d[0, :, :, np.newaxis],
            VZ=J_pred_3d[2, :, :, np.newaxis],
            VPhi=J_pred_3d[1, :, :, np.newaxis],
        )

        p0_field = _CSF(
            R=R_arr, Z=Z_arr, Phi=Phi_1d,
            value=p_new_2d[:, :, np.newaxis],
            name="p0", units="Pa",
        )

        zeros = np.zeros_like(B_pred_2d[0, :, :, np.newaxis])
        dB_ext_field = _CVF(
            R=R_arr, Z=Z_arr, Phi=Phi_1d,
            VR=zeros, VZ=zeros, VPhi=zeros,
        )

        delta_B_field, delta_J_field, delta_p_field = solve_perturbed_gs_coupled(
            B0_field, J0_field, p0_field, dB_ext_field,
            solver='lsqr', max_iter=3000, tol=1e-6,
            weight_ampere=1e6, weight_force=1.0,
            weight_div=1e4,   # reduced from 1e8 to improve conditioning
            weight_divJ=1e5,  # reduced from 1e6 to improve conditioning
        )

        dBR_pred   = delta_B_field.VR[:, :, 0]
        dBPhi_pred = delta_B_field.VPhi[:, :, 0]
        dBZ_pred   = delta_B_field.VZ[:, :, 0]
        B_pred_2d  = B_pred_2d + np.stack([dBR_pred, dBPhi_pred, dBZ_pred], axis=0)

        if verbose:
            print("  [FPT predictor] solve_perturbed_gs_coupled succeeded.")

    except Exception as _e:
        if verbose:
            print(f"  [FPT predictor] solve_perturbed_gs_coupled failed ({_e}); "
                  "falling back to diamagnetic estimate.")
        if beta_new != 0.0 and BPhi_2d is not None:
            safe_BPhi = np.where(np.abs(BPhi_2d) > 1e-10, BPhi_2d, 1e-10)
            dBPhi_pred = mu0 * p_new_2d / safe_BPhi
            B_tot2  = BR_2d ** 2 + BPhi_2d ** 2 + BZ_2d ** 2
            dBR_pred = -mu0 * p_new_2d * BR_2d / (B_tot2 + 1e-30)
            dBZ_pred = -mu0 * p_new_2d * BZ_2d / (B_tot2 + 1e-30)
            B_pred_2d = np.stack([
                BR_2d + dBR_pred,
                BPhi_2d + dBPhi_pred,
                BZ_2d + dBZ_pred,
            ], axis=0)

    # Force residual before correction
    J_pred = compute_curl_cylindrical(B_pred_2d, R_arr, Z_arr, mu0)
    r_R_before, r_Z_before = compute_force_residual(J_pred, B_pred_2d, p_new_2d, R_arr, Z_arr)
    residual_before = float(np.sqrt(np.mean(r_R_before ** 2 + r_Z_before ** 2)))
    if verbose:
        print(f"β step {beta_old*100:.1f}% → {beta_new*100:.1f}%: "
              f"residual before = {residual_before:.4e}")

    # FEniCSx corrector with Anderson + line search
    delta_B, residual_history = solve_force_balance_correction(
        B_pred_2d, p_new_2d, R_arr, Z_arr,
        mu0=mu0, eps_reg=eps_reg, lambda_div=1.0, max_iter=max_newton_iter,
        anderson_depth=anderson_depth, anderson_beta=anderson_beta,
        use_line_search=use_line_search,
    )
    B_corrected_2d = B_pred_2d + delta_B

    # Force residual after correction
    J_corr = compute_curl_cylindrical(B_corrected_2d, R_arr, Z_arr, mu0)
    r_R_after, r_Z_after = compute_force_residual(J_corr, B_corrected_2d, p_new_2d, R_arr, Z_arr)
    residual_after = float(np.sqrt(np.mean(r_R_after ** 2 + r_Z_after ** 2)))
    if verbose:
        print(f"β step {beta_new*100:.1f}%: "
              f"residual after  = {residual_after:.4e}")

    B_correction_norm = float(np.sqrt(np.mean(delta_B ** 2)))
    newton_iters = len(residual_history) - 1  # excluding initial

    # Apply 2D correction to all φ slices
    new_cache = {
        'R_grid':   R_arr,
        'Z_grid':   Z_arr,
        'Phi_grid': field_cache['Phi_grid'].copy(),
        'BR':       field_cache['BR'].copy(),
        'BPhi':     field_cache['BPhi'].copy(),
        'BZ':       field_cache['BZ'].copy(),
    }
    delta_BR_2d   = B_corrected_2d[0] - field_cache['BR']  [:, :, phi_slice]
    delta_BPhi_2d = B_corrected_2d[1] - field_cache['BPhi'][:, :, phi_slice]
    delta_BZ_2d   = B_corrected_2d[2] - field_cache['BZ']  [:, :, phi_slice]

    for k in range(n_phi):
        new_cache['BR']  [:, :, k] += delta_BR_2d
        new_cache['BPhi'][:, :, k] += delta_BPhi_2d
        new_cache['BZ']  [:, :, k] += delta_BZ_2d
    new_cache['BR']  [:, :, -1] = new_cache['BR']  [:, :, 0]
    new_cache['BPhi'][:, :, -1] = new_cache['BPhi'][:, :, 0]
    new_cache['BZ']  [:, :, -1] = new_cache['BZ']  [:, :, 0]

    return {
        'field_cache':       new_cache,
        'residual_before':   residual_before,
        'residual_after':    residual_after,
        'delta_beta':        beta_new - beta_old,
        'B_correction_norm': B_correction_norm,
        'newton_iters':      newton_iters,
        'residual_history':  residual_history,
    }
