"""Utilities: Green's function kernels, Fourier helpers, convergence monitors."""

import numpy as np
from typing import Optional


def greens_function_cylinder(
    R_src: float, Z_src: float, phi_src: float,
    R_obs: np.ndarray, Z_obs: np.ndarray, phi_obs: np.ndarray,
) -> np.ndarray:
    """Magnetic Green's function G(x_obs | x_src) for a unit current element.
    
    Returns the (3,) vector field (BR, BZ, Bphi) at observation points
    due to a unit current element at source point.
    Uses complete elliptic integrals for exact cylindrical form.
    
    Parameters
    ----------
    R_src, Z_src, phi_src : float
        Source point in cylindrical coordinates.
    R_obs, Z_obs, phi_obs : np.ndarray
        Observation point arrays (must be broadcastable).
    
    Returns
    -------
    G : np.ndarray, shape (..., 3)
        Green's function vector (BR, BZ, Bphi) at observation points.
    
    Notes
    -----
    The exact form uses Neumann's formula with complete elliptic integrals
    K(k) and E(k) from scipy.special. The modulus k is:
        k^2 = 4 R_src R_obs / ((R_src + R_obs)^2 + (Z_src - Z_obs)^2)
    """
    from scipy.special import ellipk, ellipe
    mu0 = 4*np.pi*1e-7
    a = R_src
    z = Z_obs - Z_src
    R = R_obs
    denom = (R+a)**2 + z**2
    k2 = 4*a*R / (denom + 1e-30)
    k2 = np.clip(k2, 0, 0.9999)
    K = ellipk(k2)
    E = ellipe(k2)
    common = mu0/(2*np.pi) / np.sqrt(denom + 1e-30)
    B_Z = common * ((a**2 + R**2 + z**2) / ((R-a)**2 + z**2 + 1e-30) * E - K)
    B_R = common * (z/R) * ((a**2 + R**2 + z**2) / ((R-a)**2 + z**2 + 1e-30) * E - K)
    return np.array([B_R, B_Z, np.zeros_like(B_R)])


def lundquist_number(
    equilibrium,
    R: float, Z: float,
    T_eV: float = 1000.0,
) -> float:
    """Estimate local Lundquist number S = tau_R / tau_A.
    
    S ~ 10^6-10^8 for fusion plasmas.
    Resistive shielding factor ~ S^{-1/3}.
    
    Parameters
    ----------
    equilibrium : equilibrium object
        Must provide local B and density.
    R, Z : float
        Cylindrical position.
    T_eV : float
        Electron temperature in eV (default 1 keV).
    
    Returns
    -------
    S : float
        Lundquist number (dimensionless).
    
    Notes
    -----
    Spitzer resistivity: eta ~ T_eV^{-3/2} * ln_Lambda / 1.65e-9  [Ohm·m]
    Alfven time: tau_A = L / v_A,  v_A = B / sqrt(mu0 * rho)
    Resistive time: tau_R = mu0 * L^2 / eta
    S = tau_R / tau_A
    """
    Z_eff = 1.0; ln_Lambda = 15.0
    eta = 5.2e-5 * Z_eff * ln_Lambda / (T_eV**1.5)
    mu0 = 4*np.pi*1e-7
    n_e = 1e20; m_p = 1.67e-27
    B = equilibrium.B0 * equilibrium.R0 / R
    v_A = B / np.sqrt(mu0 * n_e * m_p)
    a = getattr(equilibrium, 'r0', 0.3)
    tau_R = mu0 * a**2 / eta
    tau_A = a / v_A
    return tau_R / tau_A


def toroidal_fft(arr: np.ndarray, n_max: int) -> dict:
    """Compute toroidal Fourier decomposition of a 3D array.
    
    Parameters
    ----------
    arr : np.ndarray, shape (NR, NZ, Nphi)
        Input array on a uniform toroidal grid.
    n_max : int
        Maximum toroidal mode number to return.
    
    Returns
    -------
    modes : dict {n: complex array of shape (NR, NZ)}
        Fourier amplitudes for n = 0, 1, ..., n_max.
    """
    # FFT along the phi axis (axis=2)
    fft_arr = np.fft.rfft(arr, axis=2)
    Nphi = arr.shape[2]
    modes = {}
    for n in range(min(n_max + 1, fft_arr.shape[2])):
        modes[n] = fft_arr[:, :, n] / Nphi
    return modes


def convergence_monitor(residuals: list, tol: float) -> bool:
    """Check if iteration has converged.
    
    Parameters
    ----------
    residuals : list of float
        L2 norm of field change at each iteration.
    tol : float
        Convergence tolerance.
    
    Returns
    -------
    converged : bool
    """
    if len(residuals) < 2:
        return False
    return residuals[-1] < tol
