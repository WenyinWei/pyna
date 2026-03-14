"""
pyna.MCF.equilibrium.feedback_boozer
======================================
Linear plasma response (β-feedback) to magnetic perturbations
in Boozer coordinates (ψ, θ_B, φ_B).

Valid only where flux surfaces exist (non-chaotic regions).
For chaotic/divertor regions use feedback_cylindrical.py instead.

Companion module: pyna/MCF/equilibrium/feedback_cylindrical.py
(cylindrical formulation, valid also in chaotic/divertor regions)

Advantages over cylindrical formulation:
- Response operator is diagonal in Fourier (m,n) space
- Resonant surfaces identified analytically via iota(ψ)·n = m
- Faster computation when flux surfaces are intact
- Direct connection to neoclassical transport (ε_eff)

Key references:
  - Boozer (1981): coordinates and MHD equilibria
  - Mynick & Boozer (2004): resonant field amplification in stellarators
  - Lazerson et al. (2016): VMEC+PIES response computation
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Optional, Tuple, Dict, List
from dataclasses import dataclass, field


@dataclass
class BoozerSurface:
    """A single flux surface in Boozer coordinates.
    
    Attributes
    ----------
    psi_norm : float
        Normalized poloidal flux ψ/ψ_LCFS ∈ [0, 1].
    iota : float
        Rotational transform ι = 1/q on this surface.
    theta_B : 1D array of shape (Ntheta,)
        Boozer poloidal angles.
    phi_B : 1D array of shape (Nphi,)
        Boozer toroidal angles.
    B_mn : dict {(m, n): complex}
        Fourier decomposition of |B| on this surface.
    R_mn, Z_mn : dict {(m, n): complex}
        Geometry (R, Z) Fourier components.
    """
    psi_norm: float
    iota: float
    theta_B: np.ndarray
    phi_B: np.ndarray
    B_mn: Dict[Tuple[int, int], complex] = field(default_factory=dict)
    R_mn: Dict[Tuple[int, int], complex] = field(default_factory=dict)
    Z_mn: Dict[Tuple[int, int], complex] = field(default_factory=dict)

    @property
    def resonant_modes(self) -> List[Tuple[int, int]]:
        """Return (m,n) pairs where m - iota*n ≈ 0 (resonant modes).
        
        Scans n from 1..10 and finds m = round(iota*n).
        """
        # TODO: make n_max configurable; currently hard-coded to 10
        n_max = 10
        modes = []
        for n in range(1, n_max + 1):
            m = round(self.iota * n)
            if m > 0:
                modes.append((m, n))
        return modes


@dataclass
class BoozerPerturbation:
    """Perturbation field decomposed in Boozer Fourier modes.
    
    Attributes
    ----------
    surfaces : list of BoozerSurface
    modes : dict {psi_norm: {(m,n): complex}}
        For each surface, the Fourier amplitude of δB_mn = δB·∇ψ / |∇ψ|².
    """
    surfaces: List[BoozerSurface]
    modes: Dict[float, Dict[Tuple[int, int], complex]]

    @classmethod
    def from_cylindrical_perturbation(cls, cylindrical_pert, equilibrium,
                                      psi_grid: np.ndarray):
        """Convert a PerturbationField (R,Z,phi) to Boozer decomposition.
        
        This is a key bridge between the two formulations.
        Requires valid flux surfaces (non-chaotic region).

        Parameters
        ----------
        cylindrical_pert : PerturbationField
            Perturbation defined in cylindrical (R, Z, φ) coordinates.
            See feedback_cylindrical.py for the PerturbationField class.
        equilibrium : equilibrium object
            Must provide flux-surface geometry for coordinate mapping.
        psi_grid : 1D array
            Normalized ψ values at which to evaluate the decomposition.

        Returns
        -------
        BoozerPerturbation
        """
        # TODO: for each flux surface, integrate δB·∇ψ in Boozer coords
        raise NotImplementedError


def mhd_response_operator(
    surface: BoozerSurface,
    mode: Tuple[int, int],
    beta_local: float,
    model: str = 'ideal_mhd',
    lundquist: Optional[float] = None,
) -> complex:
    """Linear MHD response factor C_{mn}(ψ) for a single mode on a surface.
    
    Parameters
    ----------
    surface : BoozerSurface
    mode : (m, n)
    beta_local : float
        Local plasma beta on this surface.
    model : {'ideal_mhd', 'resistive', 'kinetic_screening'}
    lundquist : float, optional
        Lundquist number S for resistive model. Required if model='resistive'.
    
    Returns
    -------
    C_mn : complex
        Response amplification factor. 
        - |C_mn| > 1 → resonant amplification (ideal MHD near resonance)
        - Im(C_mn) ≠ 0 → phase shift (resistive / kinetic)
    
    Notes
    -----
    Ideal MHD near rational surface q = m/n:
        C_mn ≈ 1 / (1 - beta * W_mn)
    where W_mn is the MHD potential well depth.
    
    Resistive (Fitzpatrick):
        C_mn = C_ideal * (1 + i/S^{1/3})^{-1}
    
    Kinetic screening (Nave & Wesson):
        C_mn = C_ideal * K(omega_star, omega_A)
    """
    if model not in ('ideal_mhd', 'resistive', 'kinetic_screening'):
        raise ValueError(f"Unknown model '{model}'. Choose from: "
                         "'ideal_mhd', 'resistive', 'kinetic_screening'.")
    if model == 'resistive' and lundquist is None:
        raise ValueError("lundquist (Lundquist number S) must be provided for model='resistive'.")
    # TODO: implement each model
    raise NotImplementedError


def compute_boozer_response(
    equilibrium,
    perturbation: BoozerPerturbation,
    beta_profile: Optional[Callable] = None,
    iota_profile: Optional[Callable] = None,
    model: str = 'ideal_mhd',
    m_max: int = 10,
    n_max: int = 5,
) -> BoozerPerturbation:
    """Compute total resonant field including plasma response in Boozer frame.
    
    For each flux surface and each (m,n) mode:
        δB_total_mn = δB_ext_mn * C_mn(psi)
    
    Parameters
    ----------
    equilibrium : equilibrium object
        Must have iota_profile (or q_of_psi) and optional beta_profile.
    perturbation : BoozerPerturbation
        External perturbation in Boozer Fourier decomposition.
    beta_profile : callable, optional
        beta(psi_norm) -> float
    iota_profile : callable, optional
        iota(psi_norm) -> float. If None, uses 1/equilibrium.q_of_psi.
    model : str
        Response model (see mhd_response_operator).
    m_max, n_max : int
        Truncation of Fourier series.
    
    Returns
    -------
    total_perturbation : BoozerPerturbation
        The total field (external + plasma response).
    
    Notes
    -----
    Comparison with cylindrical formulation (feedback_cylindrical.py):
    - Faster when flux surfaces exist (diagonal in (m,n) space)
    - Cannot handle chaotic regions
    - Results should agree in non-chaotic interior
    - Disagreement at edge → signature of chaos onset
    """
    # TODO: implement
    raise NotImplementedError


def island_width_with_response(
    equilibrium,
    mode: Tuple[int, int],
    perturbation: BoozerPerturbation,
    response: BoozerPerturbation,
) -> Dict[str, float]:
    """Compute island width before and after plasma response.
    
    Uses the Chirikov / Rutherford island half-width formula.
    See also: pyna.topo.island_halfwidth for the underlying calculation.
    
    Parameters
    ----------
    equilibrium : equilibrium object
    mode : (m, n) — the resonant Fourier mode
    perturbation : BoozerPerturbation
        External perturbation (before plasma response).
    response : BoozerPerturbation
        Total perturbation including plasma response (output of
        compute_boozer_response).
    
    Returns
    -------
    dict with keys:
        'w_external': island half-width from external field alone
        'w_total': island half-width including plasma response  
        'amplification': w_total / w_external
    """
    # TODO: use island_halfwidth from pyna.topo
    raise NotImplementedError
