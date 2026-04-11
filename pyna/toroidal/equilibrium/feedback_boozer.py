"""
pyna.toroidal.equilibrium.feedback_boozer
=========================================
Linear plasma response (β-feedback) to magnetic perturbations
in Boozer coordinates (ψ, θ_B, φ_B).

Valid only where flux surfaces exist (non-chaotic regions).
For chaotic/divertor regions use feedback_cylindrical.py instead.

Companion module: pyna/toroidal/equilibrium/feedback_cylindrical.py
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
        Only includes modes within 15% of exact resonance.
        """
        # TODO: make n_max configurable; currently hard-coded to 10
        n_max = 10
        modes = []
        for n in range(1, n_max + 1):
            m = round(self.iota * n)
            if m > 0 and abs(m - self.iota * n) < 0.15:
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
        surfaces_list = []
        modes_dict: Dict[float, Dict[Tuple[int, int], complex]] = {}

        for psi_n in psi_grid:
            # Approximate flux-surface geometry from equilibrium
            iota = 1.0 / float(equilibrium.q_of_psi(psi_n)) if hasattr(equilibrium, 'q_of_psi') else 1.0
            theta_B = np.linspace(0, 2 * np.pi, 64, endpoint=False)
            phi_B = np.linspace(0, 2 * np.pi, 32, endpoint=False)

            surface = BoozerSurface(
                psi_norm=float(psi_n),
                iota=iota,
                theta_B=theta_B,
                phi_B=phi_B,
            )
            surfaces_list.append(surface)

            # Sample the cylindrical perturbation along this flux surface.
            # We build a simple toroidal-angle-based loop (tokamak approximation):
            # R(θ) ≈ R0 + r*cos(θ), Z(θ) ≈ Z0 + r*sin(θ), with r ~ sqrt(psi_n)*a.
            R0 = getattr(equilibrium, 'R0', 1.65)
            a = getattr(equilibrium, 'r0', getattr(equilibrium, 'a', 0.5))
            Z0 = getattr(equilibrium, 'Z0', 0.0)
            r = np.sqrt(psi_n) * a

            grid = cylindrical_pert.grid
            surface_modes: Dict[Tuple[int, int], complex] = {}

            # Sampling resolution for the flux-surface Fourier decomposition.
            # n_phi_s and n_theta are the number of sample points; the Nyquist
            # limits for independent Fourier modes are n_phi_s//2 and n_theta//2.
            n_theta = 64
            n_phi_s = 32
            n_max_tor = n_phi_s // 2   # max resolvable toroidal mode number
            n_max_pol = n_theta // 2   # max resolvable poloidal mode number

            # Sample δBR along the flux surface at each toroidal angle
            from scipy.interpolate import RegularGridInterpolator
            interp_BR = RegularGridInterpolator(
                (grid.R, grid.Z, grid.phi), cylindrical_pert.dBR,
                bounds_error=False, fill_value=0.0,
            )
            interp_BZ = RegularGridInterpolator(
                (grid.R, grid.Z, grid.phi), cylindrical_pert.dBZ,
                bounds_error=False, fill_value=0.0,
            )

            theta_arr = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
            phi_arr = np.linspace(0, 2 * np.pi, n_phi_s, endpoint=False)

            R_surf = R0 + r * np.cos(theta_arr)
            Z_surf = Z0 + r * np.sin(theta_arr)

            # Build δB_normal = δB · ê_r on this surface
            # ê_r ~ (cos θ, sin θ, 0) in (R, Z, phi) frame
            B_normal = np.zeros((n_theta, n_phi_s))
            for ip, phi_val in enumerate(phi_arr):
                pts = np.column_stack([R_surf, Z_surf,
                                       np.full(n_theta, phi_val % (2 * np.pi))])
                dBR_s = interp_BR(pts)
                dBZ_s = interp_BZ(pts)
                B_normal[:, ip] = dBR_s * np.cos(theta_arr) + dBZ_s * np.sin(theta_arr)

            # 2D FFT: axis 0 → m (poloidal), axis 1 → n (toroidal)
            fft2 = np.fft.fft2(B_normal) / (n_theta * n_phi_s)

            for n_tor in range(n_max_tor):
                for m_pol in range(n_max_pol):
                    amp = fft2[m_pol, n_tor]
                    if abs(amp) > 1e-20:
                        surface_modes[(m_pol, n_tor)] = complex(amp)

            modes_dict[float(psi_n)] = surface_modes

        return cls(surfaces=surfaces_list, modes=modes_dict)


def MHD_response_operator(
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
    m, n = mode
    delta_iota = surface.iota - m / n

    if model == 'ideal_mhd':
        epsilon = 1e-3
        C_mn = 1.0 / (delta_iota**2 + epsilon**2)**0.5 * beta_local * 0.1
        return complex(1.0 + C_mn, 0.0)

    elif model == 'resistive':
        if lundquist is None:
            raise ValueError("lundquist number required for resistive model")
        S = lundquist
        C_ideal = MHD_response_operator(surface, mode, beta_local, 'ideal_mhd')
        shield = 1.0 / (1.0 + 1j * S**(-1/3) / (abs(delta_iota) + 1e-6))
        return C_ideal * shield

    elif model == 'kinetic_screening':
        C_ideal = MHD_response_operator(surface, mode, beta_local, 'ideal_mhd')
        screening = 1.0 / (1.0 + 1j * 0.5 / (abs(delta_iota) + 1e-4))
        return C_ideal * screening

    else:
        raise ValueError(f"Unknown model: {model}. Choose ideal_mhd, resistive, or kinetic_screening")


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
        Response model (see MHD_response_operator).
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
    if beta_profile is None:
        beta_profile = lambda psi_n: 0.02 * (1 - psi_n)
    if iota_profile is None:
        if hasattr(equilibrium, 'q_of_psi'):
            iota_profile = lambda psi_n: 1.0 / float(equilibrium.q_of_psi(psi_n))
        else:
            raise ValueError("equilibrium must have q_of_psi or provide iota_profile")

    new_modes = {}
    for psi_n, surface_modes in perturbation.modes.items():
        iota = iota_profile(psi_n)
        beta = beta_profile(psi_n)
        surface = BoozerSurface(
            psi_norm=psi_n, iota=iota,
            theta_B=np.linspace(0, 2*np.pi, 64),
            phi_B=np.linspace(0, 2*np.pi, 32),
        )
        new_surface_modes = {}
        for (m, n), amplitude in surface_modes.items():
            C_mn = MHD_response_operator(surface, (m, n), beta, model)
            new_surface_modes[(m, n)] = amplitude * C_mn
        new_modes[psi_n] = new_surface_modes

    return BoozerPerturbation(surfaces=perturbation.surfaces, modes=new_modes)


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
    m, n = mode

    def get_amplitude(pert, psi_n, m, n):
        if psi_n in pert.modes and (m, n) in pert.modes[psi_n]:
            return abs(pert.modes[psi_n][(m, n)])
        return 0.0

    if hasattr(equilibrium, 'resonant_psi'):
        psi_list = equilibrium.resonant_psi(m, n)
    else:
        psi_list = []

    if not psi_list:
        return {'w_external': 0.0, 'w_total': 0.0, 'amplification': 1.0}

    psi_res = psi_list[0]
    b_ext = get_amplitude(perturbation, psi_res, m, n)
    b_tot = get_amplitude(response, psi_res, m, n)

    shear = abs(float(equilibrium.q_of_psi(min(psi_res + 0.05, 0.95))) -
                float(equilibrium.q_of_psi(max(psi_res - 0.05, 0.05)))) / 0.1

    factor = 4.0 / (m * shear + 1e-6)
    w_ext = factor * np.sqrt(max(b_ext, 0))
    w_tot = factor * np.sqrt(max(b_tot, 0))
    amp = w_tot / (w_ext + 1e-30)

    return {'w_external': w_ext, 'w_total': w_tot, 'amplification': amp}
