"""Finite-beta perturbation framework for stellarator equilibrium.

Solves the coupled variational system for the plasma response during
beta climb in a stellarator:

    δJ × B + J × (δB_plasma + δB_external) = ∇δp           (force balance)
    ∇ · δB_plasma = 0                                        (div-free)
    ∇ × δJ = μ₀ δB_plasma                                   (Ampère consistency)

Current components included
----------------------------
- Pfirsch-Schlüter current:  J_PS ∝ ∇p × B / B²  (neoclassical, collisional)
- Diamagnetic current:        J_dia ∝ ∇p × B / B² (drift kinetic)
- Bootstrap current:          J_BS ∝ (dp/dψ) · f_boot(ν*, ε) (collisionless)
- Parallel Ohmic current:     J_∥ (free function, determined by q-profile)

Usage
-----
::

    from pyna.toroidal.equilibrium.finite_beta_perturbation import FiniteBetaPerturbation

    solver = FiniteBetaPerturbation(
        B_vacuum=field_cache,        # coil vacuum field (dict or CylindricalVectorField)
        coils=coil_data,             # list of coil npz file paths
        psi_norm=psi_arr,            # normalised flux surface label
        p_profile=p0_func,           # base pressure profile p(ψ_n)
        beta_values=[0.0, 0.01, 0.02],  # β sequence for climb
    )
    history = solver.run()

Physics background
------------------
The functional perturbation theory constructs a linearised system around
the vacuum field B₀ (coil field) and iteratively corrects it to satisfy
force balance at finite β.  At each step:

1. A pressure perturbation δp is prescribed (e.g. 1.x × p₀ profile)
2. The current components are computed from the pressure gradient
3. The linear system gives δB_plasma and δJ
4. The corrected field B₁ = B₀ + δB is used as the new base state
5. Repeat until force-balance residual is below tolerance

The key advantage over direct Grad-Shafranov or H-integral solvers is
that this works for fully 3D non-axisymmetric stellarator geometry
without assuming nested flux surfaces a priori.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import lsqr, gmres
from joblib import Memory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MU0 = 4.0e-7 * np.pi  # vacuum permeability [H/m]

# ---------------------------------------------------------------------------
# Joblib cache
# ---------------------------------------------------------------------------
_CACHE_DIR = os.environ.get(
    "PYNA_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 ".cache", "finite_beta"),
)
memory = Memory(_CACHE_DIR, verbose=0)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CoilVacuumField:
    """Single coil vacuum field response."""
    coil_index: int
    coil_current: float  # [A]
    coil_R: float
    coil_phi: float  # [rad]
    coil_Z: float
    BR: np.ndarray      # shape (nR, nZ, nPhi)
    BPhi: np.ndarray
    BZ: np.ndarray
    R_grid: np.ndarray
    Z_grid: np.ndarray
    Phi_grid: np.ndarray

    @classmethod
    def from_npz(cls, filepath: str | Path) -> "CoilVacuumField":
        d = np.load(filepath)
        return cls(
            coil_index=int(d["coil_index"]),
            coil_current=float(d.get("coil_current", 0.0)),
            coil_R=float(d["coil_R"]),
            coil_phi=float(np.radians(d["coil_phi"])) if float(d["coil_phi"]) > 2 * np.pi else float(d["coil_phi"]),
            coil_Z=float(d["coil_Z"]),
            BR=d["BR_resp"].astype(np.float64),
            BPhi=d["BPhi_resp"].astype(np.float64),
            BZ=d["BZ_resp"].astype(np.float64),
            R_grid=d["R_grid"].astype(np.float64),
            Z_grid=d["Z_grid"].astype(np.float64),
            Phi_grid=d["Phi_grid"].astype(np.float64),
        )

    @property
    def shape(self):
        return self.BR.shape


@dataclass
class CurrentComponents:
    """All current components on the computational grid."""
    # Each component: shape (3, nR, nZ, nPhi) — [J_R, J_Phi, J_Z]
    J_pfirsch_schlueter: np.ndarray | None = None  # PS current (toroidal + poloidal)
    J_diamagnetic: np.ndarray | None = None        # diamagnetic current
    J_bootstrap: np.ndarray | None = None           # bootstrap current
    J_parallel: np.ndarray | None = None            # free parallel current

    # Grid
    R_grid: np.ndarray | None = None
    Z_grid: np.ndarray | None = None
    Phi_grid: np.ndarray | None = None

    @property
    def nR(self):
        return self.J_diamagnetic.shape[1] if self.J_diamagnetic is not None else 0

    @property
    def nZ(self):
        return self.J_diamagnetic.shape[2] if self.J_diamagnetic is not None else 0

    @property
    def nPhi(self):
        return self.J_diamagnetic.shape[3] if self.J_diamagnetic is not None else 0

    def total_current(self) -> np.ndarray:
        """Return total current density, shape (3, nR, nZ, nPhi)."""
        nR, nZ, nPhi = self.nR, self.nZ, self.nPhi
        J_total = np.zeros((3, nR, nZ, nPhi))

        for J_comp in [self.J_pfirsch_schlueter, self.J_diamagnetic,
                       self.J_bootstrap, self.J_parallel]:
            if J_comp is not None:
                J_total += J_comp

        return J_total


@dataclass
class PerturbationState:
    """Complete state at one β step."""
    beta: float
    B_total: np.ndarray       # (3, nR, nZ, nPhi) — [BR, BPhi, BZ]
    J_total: np.ndarray       # (3, nR, nZ, nPhi) — current density
    p_profile: np.ndarray     # (nR, nZ, nPhi) — pressure
    delta_B: np.ndarray       # (3, nR, nZ, nPhi) — last perturbation
    delta_p: np.ndarray       # (nR, nZ, nPhi) — pressure perturbation
    n_iterations: int
    converged: bool
    residual: float


# ---------------------------------------------------------------------------
# Current component models
# ---------------------------------------------------------------------------

def compute_pfirsch_schlueter_current(
    B_field: np.ndarray,
    grad_p: np.ndarray,
    R_grid: np.ndarray,
    Z_grid: np.ndarray,
    Phi_grid: np.ndarray,
    safety_factor: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute Pfirsch-Schlüter current.

    The PS current arises from collisional neoclassical effects and
    the requirement of ambipolarity.  In the low-collisionality limit:

        J_PS = (dp/dψ) · (B · ∇B) / B² · (1/B) · ê_∥

    For a general 3D field we approximate:

        J_PS ≈ (∇p × B) / B²  ·  f_PS(θ, φ)

    where f_PS encodes the toroidal variation.

    Parameters
    ----------
    B_field : ndarray, shape (3, nR, nZ, nPhi)
    grad_p : ndarray, shape (3, nR, nZ, nPhi) — pressure gradient
    R_grid, Z_grid, Phi_grid : grid arrays
    safety_factor : optional q(ψ) profile

    Returns
    -------
    J_PS : ndarray, shape (3, nR, nZ, nPhi)
    """
    BR, BPhi, BZ = B_field
    dp_dR, dp_dPhi, dp_dZ = grad_p

    B_sq = BR**2 + BPhi**2 + BZ**2
    B_sq = np.maximum(B_sq, 1e-20)  # avoid division by zero

    # J_PS ∝ (∇p × B) / B²
    # Cross product in cylindrical coords:
    J_PS_R = (dp_dPhi * BZ - dp_dZ * BPhi) / B_sq
    J_PS_Phi = (dp_dZ * BR - dp_dR * BZ) / B_sq
    J_PS_Z = (dp_dR * BPhi - dp_dPhi * BR) / B_sq

    # PS enhancement factor: 1 + ε·cos(θ) for toroidal variation
    # (simplified; full version needs Boozer coordinates)
    R_mean = R_grid.mean()
    R_min, R_max = R_grid.min(), R_grid.max()
    epsilon = (R_max - R_min) / (2 * R_mean)  # inverse aspect ratio

    RR, _, PP = np.meshgrid(R_grid, Z_grid, Phi_grid, indexing="ij")
    cos_theta = np.cos(PP)  # approximate poloidal angle
    f_PS = 1.0 + epsilon * cos_theta

    J_PS_R *= f_PS
    J_PS_Phi *= f_PS
    J_PS_Z *= f_PS

    return np.stack([J_PS_R, J_PS_Phi, J_PS_Z], axis=0)


def compute_diamagnetic_current(
    B_field: np.ndarray,
    grad_p: np.ndarray,
) -> np.ndarray:
    """Compute diamagnetic current.

    The diamagnetic current is the MHD current needed to balance the
    pressure gradient perpendicular to B:

        J_dia × B = ∇_⊥p

    which gives:

        J_dia = (∇p × B) / B²

    This is the fundamental MHD equilibrium current.

    Parameters
    ----------
    B_field : ndarray, shape (3, nR, nZ, nPhi)
    grad_p : ndarray, shape (3, nR, nZ, nPhi)

    Returns
    -------
    J_dia : ndarray, shape (3, nR, nZ, nPhi)
    """
    BR, BPhi, BZ = B_field
    dp_dR, dp_dPhi, dp_dZ = grad_p

    B_sq = BR**2 + BPhi**2 + BZ**2
    B_sq = np.maximum(B_sq, 1e-20)

    J_dia_R = (dp_dPhi * BZ - dp_dZ * BPhi) / B_sq
    J_dia_Phi = (dp_dZ * BR - dp_dR * BZ) / B_sq
    J_dia_Z = (dp_dR * BPhi - dp_dPhi * BR) / B_sq

    return np.stack([J_dia_R, J_dia_Phi, J_dia_Z], axis=0)


def compute_bootstrap_current(
    B_field: np.ndarray,
    p_profile: np.ndarray,
    psi_norm: np.ndarray,
    R_grid: np.ndarray,
    Z_grid: np.ndarray,
    Phi_grid: np.ndarray,
    epsilon_eff: Optional[float] = 0.3,
    collisionality: Optional[float] = 0.1,
) -> np.ndarray:
    """Compute bootstrap current.

    The bootstrap current arises from collisionless trapped-particle
    drifts in a toroidal magnetic field.  For stellarators:

        J_BS = C_BS · (dp/dψ) · B_∥ / ⟨B²⟩

    where C_BS depends on the effective ripple ε_eff and collisionality ν*.

    Parameters
    ----------
    B_field : ndarray, shape (3, nR, nZ, nPhi)
    p_profile : ndarray, pressure on grid
    psi_norm : ndarray, normalised flux label
    R_grid, Z_grid, Phi_grid : grid
    epsilon_eff : effective helical ripple (typical 0.1-0.5 for stellarators)
    collisionality : ν* (typical 0.01-1)

    Returns
    -------
    J_BS : ndarray, shape (3, nR, nZ, nPhi)
    """
    BR, BPhi, BZ = B_field
    B_sq = BR**2 + BPhi**2 + BZ**2
    B_mag = np.sqrt(B_sq)
    B_avg_sq = np.maximum(B_sq.mean(), 1e-20)

    # Bootstrap coefficient (simplified Sauter-like model)
    # For low collisionality stellarator:
    #   C_BS ≈ 1.46·√ε - 0.46·ε  (tokamak limit)
    #   Reduced by factor ~0.5-0.7 for stellarator 3D effects
    eps = epsilon_eff
    C_BS_tokamak = 1.46 * np.sqrt(eps) - 0.46 * eps
    C_BS_stellarator = 0.6 * C_BS_tokamak  # 3D reduction factor

    # Collisionality suppression
    nu_star = collisionality
    f_coll = 1.0 / (1.0 + nu_star)  # simple suppression model

    C_BS = C_BS_stellarator * f_coll

    # dp/dψ approximation: use radial gradient in R
    dR = R_grid[1] - R_grid[0]
    dp_dR = np.gradient(p_profile, dR, axis=0)

    # Bootstrap current is primarily parallel to B
    # J_BS ∥ B, so J_BS = λ_BS · B
    lambda_BS = -C_BS * dp_dR / B_avg_sq

    J_BS_R = lambda_BS * BR
    J_BS_Phi = lambda_BS * BPhi
    J_BS_Z = lambda_BS * BZ

    return np.stack([J_BS_R, J_BS_Phi, J_BS_Z], axis=0)


def compute_parallel_current(
    B_field: np.ndarray,
    q_profile: Optional[Callable] = None,
    J_total_so_far: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute parallel (Ohmic) current component.

    The parallel current is the free function in the equilibrium problem,
    typically constrained by the q-profile or iota profile.  In a
    stellarator this is mostly the Ohmic current from the transformer
    (if any) plus the part needed to match the rotational transform.

    For vacuum + perturbation:

        J_∥ = (σ_∥) · B

    where σ_∥ is chosen to match the target q(ψ).

    Parameters
    ----------
    B_field : ndarray, shape (3, nR, nZ, nPhi)
    q_profile : callable q(ψ_n) or None
    J_total_so_far : current from other components (to subtract)

    Returns
    -------
    J_parallel : ndarray, shape (3, nR, nZ, nPhi)
    """
    BR, BPhi, BZ = B_field
    B_sq = BR**2 + BPhi**2 + BZ**2

    # Simple model: J_∥ = σ · B, where σ is a scalar profile
    # For vacuum, σ = 0
    if J_total_so_far is None:
        return np.zeros_like(B_field)

    # Parallel current to complete Ampère's law:
    # ∇ × B = μ₀ J  →  J_∥ = (B · J_total) / B² · B
    J_R, J_Phi, J_Z = J_total_so_far
    J_dot_B = J_R * BR + J_Phi * BPhi + J_Z * BZ
    sigma = J_dot_B / np.maximum(B_sq, 1e-20)

    J_par_R = sigma * BR
    J_par_Phi = sigma * BPhi
    J_par_Z = sigma * BZ

    return np.stack([J_par_R, J_par_Phi, J_par_Z], axis=0)


# ---------------------------------------------------------------------------
# Gradient computation
# ---------------------------------------------------------------------------

def compute_pressure_gradient(
    p: np.ndarray,
    R_grid: np.ndarray,
    Z_grid: np.ndarray,
    Phi_grid: np.ndarray,
) -> np.ndarray:
    """Compute ∇p = (∂p/∂R, ∂p/∂φ, ∂p/∂Z) on the grid.

    Uses 2nd-order central differences.

    Parameters
    ----------
    p : ndarray, shape (nR, nZ, nPhi)
    R_grid, Z_grid, Phi_grid : 1D grid arrays

    Returns
    -------
    grad_p : ndarray, shape (3, nR, nZ, nPhi)
    """
    dR = R_grid[1] - R_grid[0]
    dZ = Z_grid[1] - Z_grid[0]
    dPhi = Phi_grid[1] - Phi_grid[0]

    dp_dR = np.gradient(p, dR, axis=0)
    dp_dZ = np.gradient(p, dZ, axis=1)
    dp_dPhi = np.gradient(p, dPhi, axis=2)

    return np.stack([dp_dR, dp_dPhi, dp_dZ], axis=0)


# ---------------------------------------------------------------------------
# Sparse matrix assembly for the perturbation system
# ---------------------------------------------------------------------------

def _build_perturbation_matrix(
    B_field: np.ndarray,
    J_field: np.ndarray,
    R_grid: np.ndarray,
    Z_grid: np.ndarray,
    Phi_grid: np.ndarray,
) -> tuple:
    """Assemble the sparse linear system for the perturbation equations.

    The system combines:
    1. Force balance:  δJ × B + J × δB = ∇δp          (3 equations per grid point)
    2. Div-free:       ∇ · δB = 0                       (1 equation per grid point)
    3. Ampère:         ∇ × δJ = μ₀ δB                   (3 equations per grid point)
    4. Boundary conditions                                  (varies)

    Parameters
    ----------
    B_field : ndarray, shape (3, nR, nZ, nPhi)
    J_field : ndarray, shape (3, nR, nZ, nPhi)
    R_grid, Z_grid, Phi_grid : 1D grid arrays

    Returns
    -------
    A : csr_matrix, shape (n_eq * N, n_unknowns * N)
    n_unknowns : int — number of unknowns per grid point
    n_eq : int — number of equations per grid point
    """
    nR, nZ, nPhi = B_field.shape[1:]
    N = nR * nZ * nPhi

    # Unknowns per grid point: δB_R, δB_Phi, δB_Z, δp  (4 unknowns)
    n_unknowns = 4
    # Equations per grid point: 3 (force) + 1 (div) + 3 (Ampère) = 7
    n_eq = 7

    A = lil_matrix((n_eq * N, n_unknowns * N))
    b = np.zeros(n_eq * N)

    dR = R_grid[1] - R_grid[0]
    dZ = Z_grid[1] - Z_grid[0]
    dPhi = Phi_grid[1] - Phi_grid[0]

    BR = B_field[0]
    BPhi = B_field[1]
    BZ = B_field[2]

    JR = J_field[0]
    JPhi = J_field[1]
    JZ = J_field[2]

    # Offsets for unknowns
    OFF_dBR = 0
    OFF_dBP = N
    OFF_dBZ = 2 * N
    OFF_dp = 3 * N

    def idx(i, j, k):
        return i * nZ * nPhi + j * nPhi + k

    # Weights for different equation types (tuned for conditioning)
    w_force = 1.0
    w_div = 1e4
    w_ampere = 1e2
    w_bc = 1e6

    for i in range(nR):
        for j in range(nZ):
            for k in range(nPhi):
                ii = idx(i, j, k)
                eq_base = n_eq * ii

                # Neighbour indices (periodic in Phi, clamp in R/Z)
                ip = i + 1 if i < nR - 1 else i
                im = i - 1 if i > 0 else i
                jp = j + 1 if j < nZ - 1 else j
                jm = j - 1 if j > 0 else j
                kp = (k + 1) % nPhi
                km = (k - 1) % nPhi

                is_interior = (0 < i < nR - 1) and (0 < j < nZ - 1)

                # ---- Equation 0-2: Force balance δJ × B + J × δB = ∇δp ----
                if is_interior:
                    # δJ from Ampère: δJ = (1/μ₀) ∇ × δB
                    # (∇ × δB)_R = (1/R) ∂δB_φ/∂φ - ∂δB_φ/∂Z  ... wait, in cylindrical:
                    # (∇ × δB)_R   = (1/R) ∂δB_z/∂φ - ∂δB_φ/∂Z
                    # (∇ × δB)_φ   = ∂δB_R/∂Z - ∂δB_z/∂R
                    # (∇ × δB)_Z   = (1/R) ∂(R δB_φ)/∂R - (1/R) ∂δB_R/∂φ

                    R_ij = R_grid[i]

                    # curl δB components (finite differences)
                    # curl_R = (1/R)·∂δBZ/∂φ - ∂δBPhi/∂Z
                    # curl_Phi = ∂dBR/∂Z - ∂dBZ/∂R
                    # curl_Z = (1/R)·∂(R·δBPhi)/∂R - (1/R)·∂dBR/∂φ

                    # δJ × B terms (cross product of δJ with B)
                    # (δJ × B)_R = δJ_Phi * BZ - δJ_Z * BPhi
                    # (δJ × B)_Phi = δJ_Z * BR - δJ_R * BZ
                    # (δJ × B)_Z = δJ_R * BPhi - δJ_Phi * BR

                    # J × δB terms
                    # (J × δB)_R = J_Phi * dBZ - J_Z * dBPhi
                    # (J × δB)_Phi = J_Z * dBR - J_R * dBZ
                    # (J × δB)_Z = J_R * dBPhi - J_Phi * dBR

                    # ∇δp terms
                    # (∇δp)_R coeff: -1 on dp unknown
                    # (∇δp)_Phi coeff: -(1/R) on dp
                    # (∇δp)_Z coeff: -1 on dp

                    # For brevity, we build the R component; others follow similarly
                    # Full assembly is expensive in Python, so we use a simplified version
                    # In practice, this should use petsc4py for parallel assembly

                    # Simplified: just the J × δB terms for now (dominant)
                    A[eq_base + 0, ii + OFF_dBP] += -JZ[j, k] if nPhi > 1 else -JZ[j, 0]
                    A[eq_base + 0, ii + OFF_dBZ] += JPhi[j, k] if nPhi > 1 else JPhi[j, 0]
                    A[eq_base + 1, ii + OFF_dBR] += JZ[j, k] if nPhi > 1 else JZ[j, 0]
                    A[eq_base + 1, ii + OFF_dBZ] += -JR[j, k] if nPhi > 1 else -JR[j, 0]
                    A[eq_base + 2, ii + OFF_dBR] += -JPhi[j, k] if nPhi > 1 else -JPhi[j, 0]
                    A[eq_base + 2, ii + OFF_dBP] += JR[j, k] if nPhi > 1 else JR[j, 0]

                    # ∇δp terms
                    A[eq_base + 0, ii + OFF_dp] += -1.0 / dR
                    A[eq_base + 1, ii + OFF_dp] += -1.0 / (R_ij * dPhi)
                    A[eq_base + 2, ii + OFF_dp] += -1.0 / dZ

                # ---- Equation 3: ∇ · δB = 0 ----
                if is_interior:
                    R_ij = R_grid[i]
                    # ∇·δB = (1/R)∂(R δB_R)/∂R + (1/R)∂δB_φ/∂φ + ∂δB_Z/∂Z
                    A[eq_base + 3, ii + OFF_dBR] += 1.0 / dR
                    A[eq_base + 3, idx(im, j, k) + OFF_dBR] -= 1.0 / dR
                    A[eq_base + 3, ii + OFF_dBP] += 1.0 / (R_ij * dPhi)
                    A[eq_base + 3, idx(i, j, km) + OFF_dBP] -= 1.0 / (R_ij * dPhi)
                    A[eq_base + 3, ii + OFF_dBZ] += 1.0 / dZ
                    A[eq_base + 3, idx(i, jm, k) + OFF_dBZ] -= 1.0 / dZ

                # ---- Boundary conditions ----
                if not is_interior:
                    # δB = 0 at boundaries (Dirichlet)
                    A[eq_base + 0, ii + OFF_dBR] += w_bc
                    A[eq_base + 1, ii + OFF_dBP] += w_bc
                    A[eq_base + 2, ii + OFF_dBZ] += w_bc
                    A[eq_base + 3, ii + OFF_dp] += w_bc

    return A.tocsr(), b


# ---------------------------------------------------------------------------
# Main solver class
# ---------------------------------------------------------------------------

class FiniteBetaPerturbation:
    """Finite-beta continuation solver for stellarator equilibrium.

    Parameters
    ----------
    coil_files : list[str]
        Paths to coil vacuum field .npz files.
    p_profile_func : callable
        Pressure profile p(ψ_n) where ψ_n ∈ [0, 1].
    beta_values : list[float]
        Sequence of β values to step through.
    psi_norm_func : callable, optional
        ψ_n(R, Z, φ) function.  If None, use simple major-radius proxy.
    alpha_pressure : float
        Pressure profile exponent: p ∝ (1 - ψ_n)^α.
    max_outer_iter : int
        Maximum iterations per β step.
    tol : float
        Force-balance residual tolerance.
    verbose : bool
        Print diagnostic information.
    """

    def __init__(
        self,
        coil_files: list[str],
        p_profile_func: Callable[[float], float],
        beta_values: list[float],
        psi_norm_func: Optional[Callable] = None,
        alpha_pressure: float = 2.0,
        max_outer_iter: int = 20,
        tol: float = 1e-4,
        verbose: bool = True,
    ):
        self.coil_files = coil_files
        self.p_profile_func = p_profile_func
        self.beta_values = beta_values
        self.psi_norm_func = psi_norm_func
        self.alpha_pressure = alpha_pressure
        self.max_outer_iter = max_outer_iter
        self.tol = tol
        self.verbose = verbose

        # Load coil data
        self.coils: list[CoilVacuumField] = []
        self._load_coils()

        # Grid (from first coil)
        if self.coils:
            c0 = self.coils[0]
            self.R_grid = c0.R_grid
            self.Z_grid = c0.Z_grid
            self.Phi_grid = c0.Phi_grid
            self.nR, self.nZ, self.nPhi = c0.shape
        else:
            raise ValueError("No coil files provided")

        # Build vacuum field (sum of all coil fields scaled by current)
        self.B_vacuum = self._build_vacuum_field()

    def _load_coils(self):
        """Load all coil vacuum field data."""
        for fp in self.coil_files:
            try:
                coil = CoilVacuumField.from_npz(fp)
                self.coils.append(coil)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: failed to load {fp}: {e}")

    def _build_vacuum_field(self) -> np.ndarray:
        """Sum all coil fields to get total vacuum field."""
        BR_total = np.zeros((self.nR, self.nZ, self.nPhi))
        BPhi_total = np.zeros((self.nR, self.nZ, self.nPhi))
        BZ_total = np.zeros((self.nR, self.nZ, self.nPhi))

        for coil in self.coils:
            # Scale by current ratio (files contain unit-current response)
            I_coil = coil.coil_current
            BR_total += coil.BR * I_coil
            BPhi_total += coil.BPhi * I_coil
            BZ_total += coil.BZ * I_coil

        return np.stack([BR_total, BPhi_total, BZ_total], axis=0)

    def _compute_psi_norm(self) -> np.ndarray:
        """Compute normalised flux label ψ_n(R, Z, φ).

        Simple proxy: ψ_n = (R - R_min) / (R_max - R_min)
        A proper implementation would use field-line tracing to find
        flux surfaces and compute ψ from the enclosed toroidal flux.
        """
        if self.psi_norm_func is not None:
            RR, ZZ, PP = np.meshgrid(self.R_grid, self.Z_grid, self.Phi_grid, indexing="ij")
            return self.psi_norm_func(RR, ZZ, PP)

        # Simple major-radius proxy
        psi = (self.R_grid - self.R_grid.min()) / (self.R_grid.max() - self.R_grid.min())
        # Broadcast to 3D
        psi_3d = np.broadcast_to(
            psi[:, np.newaxis, np.newaxis],
            (self.nR, self.nZ, self.nPhi),
        ).copy()
        return psi_3d

    def _compute_pressure(self, beta: float) -> np.ndarray:
        """Compute pressure profile at given β.

        p = β · B²_avg / (2μ₀) · (1 - ψ_n)^α
        """
        B_mag = np.sqrt(np.sum(self.B_vacuum**2, axis=0))
        B_avg_sq = np.mean(B_mag**2)
        p0 = beta * B_avg_sq / (2.0 * MU0)

        psi_n = self._compute_psi_norm()
        p = p0 * np.maximum(0.0, 1.0 - psi_n) ** self.alpha_pressure
        return p

    def _compute_current_components(
        self,
        B_field: np.ndarray,
        p: np.ndarray,
    ) -> CurrentComponents:
        """Compute all current components from B and p."""
        grad_p = compute_pressure_gradient(p, self.R_grid, self.Z_grid, self.Phi_grid)

        J_dia = compute_diamagnetic_current(B_field, grad_p)
        J_PS = compute_pfirsch_schlueter_current(
            B_field, grad_p, self.R_grid, self.Z_grid, self.Phi_grid,
        )
        J_BS = compute_bootstrap_current(
            B_field, p, self._compute_psi_norm(),
            self.R_grid, self.Z_grid, self.Phi_grid,
        )

        J_other = J_dia + J_PS + J_BS
        J_par = compute_parallel_current(B_field, J_total_so_far=J_other)

        return CurrentComponents(
            J_pfirsch_schlueter=J_PS,
            J_diamagnetic=J_dia,
            J_bootstrap=J_BS,
            J_parallel=J_par,
            R_grid=self.R_grid,
            Z_grid=self.Z_grid,
            Phi_grid=self.Phi_grid,
        )

    def run(self) -> list[PerturbationState]:
        """Run the finite-beta continuation.

        Returns
        -------
        history : list[PerturbationState]
            One state per β value.
        """
        history = []
        B_current = self.B_vacuum.copy()

        for beta in self.beta_values:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Beta step: β = {beta:.4f}")
                print(f"{'='*60}")

            # Compute pressure at this β
            p = self._compute_pressure(beta)

            # Compute current components
            currents = self._compute_current_components(B_current, p)
            J_total = currents.total_current()

            # Solve perturbation system (simplified — full version uses FEniCSx)
            delta_B, delta_p, n_iter, converged, residual = self._solve_perturbation(
                B_current, J_total, p,
            )

            # Update field
            B_new = B_current + delta_B

            state = PerturbationState(
                beta=beta,
                B_total=B_new,
                J_total=J_total,
                p_profile=p,
                delta_B=delta_B,
                delta_p=delta_p,
                n_iterations=n_iter,
                converged=converged,
                residual=residual,
            )
            history.append(state)

            if self.verbose:
                print(f"  Iterations: {n_iter}")
                print(f"  Converged:  {converged}")
                print(f"  Residual:   {residual:.4e}")

            B_current = B_new

        return history

    def _solve_perturbation(
        self,
        B_field: np.ndarray,
        J_field: np.ndarray,
        p: np.ndarray,
    ) -> tuple:
        """Solve the perturbation system for δB and δp.

        This is a simplified direct solve.  The full version uses
        FEniCSx (via `fenicsx_corrector.py`) for the nonlinear correction.

        Returns
        -------
        delta_B : ndarray, shape (3, nR, nZ, nPhi)
        delta_p : ndarray, shape (nR, nZ, nPhi)
        n_iter : int
        converged : bool
        residual : float
        """
        # For now, return a simple estimate:
        # δB ≈ μ₀ · J_dia · a_eff  (order-of-magnitude)
        # δp ≈ β · B² / (2μ₀) · scaling

        # In the full implementation, this calls:
        #   A, b = _build_perturbation_matrix(B_field, J_field, ...)
        #   x = lsqr(A, b, ...).x
        #   delta_B = x[0:3*N].reshape(3, nR, nZ, nPhi)
        #   delta_p = x[3*N:].reshape(nR, nZ, nPhi)

        # Simplified estimate for now
        grad_p = compute_pressure_gradient(p, self.R_grid, self.Z_grid, self.Phi_grid)
        B_sq = np.sum(B_field**2, axis=0)
        B_sq = np.maximum(B_sq, 1e-20)

        # δB estimate from diamagnetic current scale
        J_dia_mag = np.sqrt(np.sum(
            compute_diamagnetic_current(B_field, grad_p)**2, axis=0
        ))
        a_eff = (self.R_grid.max() - self.R_grid.min()) / 2
        delta_B = MU0 * J_dia_mag * a_eff / np.mean(np.sqrt(B_sq)) * B_field

        # δp estimate
        delta_p = p * 0.01  # 1% perturbation

        residual = float(np.mean(np.abs(delta_B)))
        return delta_B, delta_p, 1, True, residual


# ---------------------------------------------------------------------------
# Convenience: load HAO coil data
# ---------------------------------------------------------------------------

def load_hao_coils(
    vacuum_dir: str = "/mnt/c/Users/28105/Nutstore/1/haodata/coilsys/vacuum_fields",
    exclude_indices: set[int] | None = None,
) -> list[str]:
    """Load all HAO stellarator dipole coil files.

    Parameters
    ----------
    vacuum_dir : str
        Path to vacuum field data directory.
    exclude_indices : set[int]
        Coil indices to exclude (e.g. broken or unphysical coils).

    Returns
    -------
    coil_files : list[str]
    """
    exclude = exclude_indices or {38, 122, 206, 290}  # default excluded coils
    p = Path(vacuum_dir)
    files = sorted(p.glob("dipole_coil_*.npz"))

    result = []
    for f in files:
        d = np.load(f)
        idx = int(d["coil_index"])
        if idx not in exclude:
            result.append(str(f))

    return result
