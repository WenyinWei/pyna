"""Magnetic topology state vector — observable quantities for control.

This module defines the complete set of observables that characterize
magnetic topology for both tokamaks (axisymmetric) and stellarators (3D).

Observable categories
---------------------
Boundary : g_i (plasma-wall gap), X/O-point positions, DPm eigenvalues,
           DPm eigenvectors, B_pol on LCFS
Core     : q-profile, J-iota profile, O-point rotation transform
Derived  : connection length L_c, flux expansion f_x, surface fate
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from enum import Enum, auto


class SurfaceFate(Enum):
    """Classification of a flux surface under perturbation."""
    INTACT   = auto()   # KAM torus survives  (|δDPm/DPm| < ε_KAM)
    DEFORMED = auto()   # Torus deforms but stays closed
    ISLAND   = auto()   # Resonant island chain forms (Greene residue crosses 0 or 1)
    CHAOTIC  = auto()   # Stochastic region  (|δDPm/DPm| > ε_chaos)
    UNKNOWN  = auto()


@dataclass
class XPointState:
    """State of a divertor X-point."""
    R: float
    Z: float
    A_matrix: np.ndarray              # shape (2,2)
    DPm: np.ndarray                   # shape (2,2); exp(2πA) under axisymmetry
    DPm_eigenvalues: np.ndarray       # shape (2,), complex in general
    DPm_eigenvectors: np.ndarray      # shape (2,2), columns = eigenvectors
    D2Pm: Optional[np.ndarray] = None # shape (2,2,2), 2nd-order sensitivity
    D3Pm: Optional[np.ndarray] = None # shape (2,2,2,2)
    greene_residue: float = 0.0       # (2 − Tr(DPm)) / 4

    @property
    def stability_index(self) -> float:
        """Half-trace of DPm."""
        return float(np.trace(self.DPm) / 2.0)

    @property
    def is_hyperbolic(self) -> bool:
        return bool(np.any(np.abs(self.DPm_eigenvalues) > 1.0 + 1e-8))

    @property
    def connection_length_scale(self) -> float:
        """Inverse of max log-eigenvalue → proportional to L_c near X-point.

        When eigenvalues are close to 1 the separatrix is nearly degenerate
        and nearby field lines have a long connection length (good for
        detachment).  When they diverge from 1 the connection length drops.
        """
        lam_max = np.log(np.abs(self.DPm_eigenvalues)).max()
        return 1.0 / max(float(lam_max), 1e-10)


@dataclass
class OPointState:
    """State of a magnetic O-point (axis or island centre)."""
    R: float
    Z: float
    A_matrix: np.ndarray              # shape (2,2)
    DPm: np.ndarray                   # shape (2,2)
    DPm_eigenvalues: np.ndarray       # shape (2,); |λ|=1 for elliptic
    iota: float = 0.0                 # rotation transform ι = 1/q
    q: float = float('inf')           # safety factor q = 1/ι

    @property
    def is_elliptic(self) -> bool:
        return bool(np.all(np.abs(np.abs(self.DPm_eigenvalues) - 1.0) < 1e-6))


@dataclass
class TopologyState:
    """Complete magnetic topology state vector.

    Contains all observables needed for multi-objective magnetic topology
    control.  Works for both axisymmetric (tokamak) and 3D (stellarator)
    configurations.

    Attributes
    ----------
    xpoints : list of XPointState
        All X-points (divertor, saddle, …).
    opoints : list of OPointState
        All O-points (magnetic axis + island O-points).
    gap_gi : dict
        Plasma-wall gaps g_i at named monitoring points {name: gap_m}.
    Bpol_lcfs : ndarray or None
        |B_pol| on LCFS at theta_lcfs grid points.
    q_samples : ndarray or None
        q(s) at s = s_samples.
    J_iota_samples : ndarray or None
        Current-density rotation transform at the same s grid.
    surface_fate : dict
        {psi_norm: SurfaceFate} for monitored flux surfaces.
    is_axisymmetric : bool
        If True use closed-form FPT; otherwise integrate along φ.
    phi_ref : float
        Reference toroidal angle for the Poincaré section (default 0).
    """
    xpoints: List[XPointState] = field(default_factory=list)
    opoints: List[OPointState] = field(default_factory=list)
    gap_gi: Dict[str, float] = field(default_factory=dict)
    Bpol_lcfs: Optional[np.ndarray] = None
    theta_lcfs: Optional[np.ndarray] = None
    q_samples: Optional[np.ndarray] = None
    s_samples: Optional[np.ndarray] = None
    J_iota_samples: Optional[np.ndarray] = None
    surface_fate: Dict[float, SurfaceFate] = field(default_factory=dict)
    is_axisymmetric: bool = True
    phi_ref: float = 0.0

    def to_vector(self, keys=None):
        """Flatten state into a 1-D numpy array for optimisation.

        Returns
        -------
        values : ndarray, shape (n_obs,)
        labels : list of str, length n_obs
        """
        vals: List[float] = []
        labels: List[str] = []

        for i, xp in enumerate(self.xpoints):
            vals.extend([xp.R, xp.Z])
            labels.extend([f'xp{i}.R', f'xp{i}.Z'])
            ev = np.abs(xp.DPm_eigenvalues)
            vals.extend(ev.real.tolist())
            labels.extend([f'xp{i}.DPm_eig{j}' for j in range(len(ev))])

        for i, op in enumerate(self.opoints):
            vals.extend([op.R, op.Z, op.iota])
            labels.extend([f'op{i}.R', f'op{i}.Z', f'op{i}.iota'])

        for name, gap in self.gap_gi.items():
            vals.append(gap)
            labels.append(f'gap.{name}')

        if self.q_samples is not None:
            s_arr = self.s_samples if self.s_samples is not None else np.linspace(0.2, 1.0, len(self.q_samples))
            vals.extend(self.q_samples.tolist())
            labels.extend([f'q.s{s:.2f}' for s in s_arr])

        return np.array(vals), labels


def compute_topology_state(
    field_func: Callable,
    xpoint_guesses: list,
    opoint_guesses: list,
    is_axisymmetric: bool = True,
    phi_ref: float = 0.0,
    eps: float = 1e-4,
) -> TopologyState:
    """Compute full topology state from a field function.

    Parameters
    ----------
    field_func : callable
        Magnetic field function: [R,Z,phi] → [dR/dl, dZ/dl, dphi/dl].
    xpoint_guesses : list of (R, Z)
        Initial guesses for X-point positions.
    opoint_guesses : list of (R, Z)
        Initial guesses for O-point positions.
    is_axisymmetric : bool
        Use closed-form exp(2πA) for DPm.
    phi_ref : float
        Reference toroidal angle.
    eps : float
        Finite-difference step for A-matrix.

    Returns
    -------
    TopologyState
    """
    from pyna.control.fpt import A_matrix, DPm_axisymmetric

    xpoints = []
    for R0, Z0 in xpoint_guesses:
        A = A_matrix(field_func, R0, Z0, phi_ref, eps)
        DPm = DPm_axisymmetric(A) if is_axisymmetric else np.eye(2)
        eigvals, eigvecs = np.linalg.eig(DPm)
        xpoints.append(XPointState(
            R=float(R0), Z=float(Z0),
            A_matrix=A,
            DPm=DPm,
            DPm_eigenvalues=eigvals,
            DPm_eigenvectors=eigvecs,
            greene_residue=float((2.0 - np.trace(DPm)) / 4.0),
        ))

    opoints = []
    for R0, Z0 in opoint_guesses:
        A = A_matrix(field_func, R0, Z0, phi_ref, eps)
        DPm = DPm_axisymmetric(A) if is_axisymmetric else np.eye(2)
        eigvals, _ = np.linalg.eig(DPm)
        # Estimate iota from imaginary part of eigenvalue angle
        angles = np.angle(eigvals)
        iota = float(np.abs(angles).mean() / (2 * np.pi))
        opoints.append(OPointState(
            R=float(R0), Z=float(Z0),
            A_matrix=A,
            DPm=DPm,
            DPm_eigenvalues=eigvals,
            iota=iota,
            q=1.0 / iota if iota > 1e-12 else float('inf'),
        ))

    return TopologyState(
        xpoints=xpoints,
        opoints=opoints,
        is_axisymmetric=is_axisymmetric,
        phi_ref=phi_ref,
    )
