"""RMP Fourier spectrum analysis and island width visualization.

For a StellaratorSimple, compute resonant components analytically:
  - Resonant surface location from q(ψ) = n/m
  - Island half-width via Rutherford formula
  - O-point phase from the RMP field structure

O-point Phase Convention
------------------------
For resonant component b_{m,-n} = |b|·exp(iφ) in the Fourier expansion
  δBψ = Σ b_{mn} exp(i(mθ* + nφ))

The fixed points of the Poincaré map (one toroidal turn) satisfy:
  δBψ = 0  →  mθ* − nφ + φ_mn = ±π/2

Stability analysis (q' > 0):
  O-point: mθ_O + φ_mn = −π/2  →  θ_O = (−π/2 − φ_mn)/m
  X-point: mθ_X + φ_mn = +π/2  →  θ_X = (+π/2 − φ_mn)/m

Reference: Rutherford (1973); Nardon (2007) thesis App. A;

General φ-section O/X-point formula
-------------------------------------
At an arbitrary toroidal angle φ, the m O-points lie at poloidal angles:

    θ_O^(k)(φ) = [nφ − π/2 − arg(b_{m,−n})] / m  +  2πk/m,   k = 0…m−1

and the m X-points at:

    θ_X^(k)(φ) = [nφ + π/2 − arg(b_{m,−n})] / m  +  2πk/m,   k = 0…m−1

All angles in radians; results should be taken mod 2π.
For reversed shear (q' < 0) swap O ↔ X.

References
----------
Boozer, Phys. Fluids B 3 (1991) — resonance condition
Rutherford, Phys. Fluids 16 (1973) — island width formula
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Union
import matplotlib.pyplot as plt

from .equilibrium import ISLAND_CMAPS


def island_fixed_points(
    m: int,
    n: int,
    b_mn: complex,
    phi: Union[float, np.ndarray],
    q_prime_sign: int = 1,
) -> dict:
    """Return poloidal angles of all O-points and X-points at toroidal angle φ.

    For the resonant component b_{m,−n} of the RMP field, the island fixed
    points at an arbitrary toroidal cross-section φ satisfy:

        mθ* − nφ + arg(b_{m,−n}) = ±π/2   (mod 2π)

    For q' > 0 (normal shear):
        O-points: mθ_O^(k) = nφ − π/2 − arg(b)   →  k = 0 … m−1
        X-points: mθ_X^(k) = nφ + π/2 − arg(b)

    Parameters
    ----------
    m, n : int
        Poloidal and toroidal mode numbers.
    b_mn : complex
        Fourier coefficient b_{m,−n} from the RMP spectrum.
    phi : float or array_like
        Toroidal angle(s) φ in radians at which to evaluate (can be scalar or
        1-D array for a sweep over multiple sections).
    q_prime_sign : int
        +1 for normal shear (q' > 0, default), −1 for reversed shear.
        Reversed shear swaps O and X.

    Returns
    -------
    dict with keys:
        'phi'      : input toroidal angles, shape (N,)
        'theta_O'  : O-point poloidal angles, shape (N, m)  — each row has m O-points
        'theta_X'  : X-point poloidal angles, shape (N, m)
        'theta_O_deg', 'theta_X_deg' : same in degrees

    Examples
    --------
    >>> pts = island_fixed_points(m=2, n=1, b_mn=0.002+0j, phi=0.0)
    >>> print(np.degrees(pts['theta_O']))   # O-point angles at φ=0

    >>> phis = np.linspace(0, 2*np.pi, 100)
    >>> pts = island_fixed_points(m=2, n=1, b_mn=0.002+0j, phi=phis)
    >>> # pts['theta_O'] shape: (100, 2)  — 2 O-points per section
    """
    arg_b = np.angle(b_mn)
    phi = np.atleast_1d(np.asarray(phi, dtype=float))  # shape (N,)
    N = len(phi)

    # Base angle before distributing k branches
    # q' > 0: O-point at mθ = nφ − π/2 − arg(b)
    #         X-point at mθ = nφ + π/2 − arg(b)
    if q_prime_sign >= 0:
        base_O = n * phi - np.pi / 2 - arg_b   # shape (N,)
        base_X = n * phi + np.pi / 2 - arg_b
    else:
        base_O = n * phi + np.pi / 2 - arg_b   # swap for reversed shear
        base_X = n * phi - np.pi / 2 - arg_b

    # k branches: k = 0, 1, …, m−1  → add 2πk
    k = np.arange(m)  # shape (m,)
    theta_O = (base_O[:, None] + 2 * np.pi * k[None, :]) / m  # (N, m)
    theta_X = (base_X[:, None] + 2 * np.pi * k[None, :]) / m  # (N, m)

    # Normalize to [0, 2π)
    theta_O = theta_O % (2 * np.pi)
    theta_X = theta_X % (2 * np.pi)

    return {
        'phi':         phi,
        'theta_O':     theta_O,
        'theta_X':     theta_X,
        'theta_O_deg': np.degrees(theta_O),
        'theta_X_deg': np.degrees(theta_X),
    }


@dataclass
class ResonantComponent:
    """One resonant (m, n) Fourier component of the RMP field."""
    m: int
    n: int
    harmonic_order: int
    b_mn: complex
    psi_res: float
    q_res: float
    half_width_psi: float
    half_width_r: float
    opoint_theta: float    # first O-point at φ=0, in [0, 2π/m)
    xpoint_theta: float    # first X-point at φ=0, in [0, 2π/m)
    q_prime_sign: int = 1  # +1 normal shear, −1 reversed shear

    def fixed_points(self, phi: Union[float, np.ndarray]) -> dict:
        """O-points and X-points at arbitrary toroidal section(s) φ.

        Parameters
        ----------
        phi : float or array_like
            Toroidal angle(s) in radians.

        Returns
        -------
        dict with 'theta_O', 'theta_X' (shape (N, m)) and degree variants.

        Example
        -------
        >>> comp.fixed_points(0.0)['theta_O_deg']   # at φ=0
        >>> comp.fixed_points(np.linspace(0, 2*np.pi, 36))['theta_O']
        """
        return island_fixed_points(
            self.m, self.n, self.b_mn, phi, self.q_prime_sign
        )


def find_resonant_components_analytic(
    eq,
    delta_B_func,
    base_m: int,
    base_n: int,
    max_harmonic: int = 3,
    n_theta: int = 128,
    n_phi: int = 64,
    min_amplitude: float = 1e-8,
) -> List[ResonantComponent]:
    """Find resonant RMP components using analytic surface sampling.

    For each harmonic k, finds the resonant surface ψ_res where
    q(ψ_res) = k*base_n / (k*base_m) = base_n/base_m, then computes
    the Fourier coefficient b_{km, kn} by sampling the RMP on that surface.

    Works with StellaratorSimple's psi_ax / q_of_psi / resonant_psi API.
    """
    components = []

    for k in range(1, max_harmonic + 1):
        m_k = k * base_m
        n_k = k * base_n

        # Find resonant ψ: resonance condition q = m/n (mode numbers)
        # eq.resonant_psi(m, n) gives q = m/n
        # We want q = m_k/n_k, so call resonant_psi(m_k, n_k)
        psi_list = eq.resonant_psi(m_k, n_k)
        if not psi_list:
            print(f"  k={k}: ({m_k},{n_k}) — no resonant surface in [0,1], skipping")
            continue

        psi_res = float(psi_list[0])
        r_res = np.sqrt(psi_res) * eq.r0
        q_res = float(eq.q_of_psi(psi_res))

        # Sample RMP on flux surface (circular approximation)
        theta_arr = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
        phi_arr   = np.linspace(0, 2*np.pi, n_phi,   endpoint=False)

        R_surf = eq.R0 + r_res * np.cos(theta_arr)
        Z_surf =         r_res * np.sin(theta_arr)

        # Compute δB^ψ on (theta, phi) grid
        # δB^ψ ≈ δBR * cos(θ) + δBZ * sin(θ)  (radial projection)
        dBpsi = np.zeros((n_theta, n_phi), dtype=complex)
        for j, phi in enumerate(phi_arr):
            for i in range(n_theta):
                db = delta_B_func(R_surf[i], Z_surf[i], phi)
                # Project onto outward radial direction
                dBpsi[i, j] = db[0] * np.cos(theta_arr[i]) + db[1] * np.sin(theta_arr[i])

        # 2D FFT → b_{m_k, n_k}
        b_fft = np.fft.fft2(dBpsi) / (n_theta * n_phi)
        m_freq = np.fft.fftfreq(n_theta, 1/n_theta).astype(int)
        n_freq = np.fft.fftfreq(n_phi,   1/n_phi).astype(int)

        # DFT convention: exp(-2πi*k*j/N), so cos(m*θ - n*φ) gives components
        # at (m_freq=+m, n_freq=-n) and (m_freq=-m, n_freq=+n).
        # We want the (+m, -n) component (forward-running wave).
        m_idx_arr = np.where(m_freq == m_k)[0]
        n_idx_arr = np.where(n_freq == -n_k)[0]  # note: -n_k (conjugate convention)

        if len(m_idx_arr) == 0 or len(n_idx_arr) == 0:
            print(f"  k={k}: ({m_k},{n_k}) — mode not in FFT grid, skipping")
            continue

        b_mn = b_fft[m_idx_arr[0], n_idx_arr[0]]

        if abs(b_mn) < min_amplitude:
            print(f"  k={k}: ({m_k},{n_k}) — |b_mn|={abs(b_mn):.2e} below threshold")
            continue

        # dq/dψ at resonant surface (from linear profile: dq/dψ = q1 - q0)
        dq_dpsi = eq.q1 - eq.q0   # constant for linear profile

        if abs(dq_dpsi) < 1e-12:
            continue

        # Rutherford formula: w_ψ = 4 * sqrt(|b_mn| / (m * |dq/dψ|))
        half_width_psi = 4.0 * np.sqrt(abs(b_mn) / (m_k * abs(dq_dpsi) + 1e-30))

        # Convert to meters: r ≈ sqrt(ψ) * r0
        half_width_r = half_width_psi * eq.r0 / (2.0 * np.sqrt(max(psi_res, 0.01)))

        # O-point phase (derived from stability analysis of island Hamiltonian)
        # Fixed points where δBψ = 2|b|cos(mθ + arg(b)) = 0, i.e., mθ + arg(b) = ±π/2
        # For q' > 0 (normal shear):
        #   O-point (stable):   mθ_O + arg(b) = -π/2  →  θ_O = (-π/2 - arg(b)) / m
        #   X-point (unstable): mθ_X + arg(b) = +π/2  →  θ_X = (+π/2 - arg(b)) / m
        # Reference: Nardon (2007) thesis, App. A; Rutherford (1973)
        phi_mn = np.angle(b_mn)
        # Determine sign of q' to handle reversed-shear case
        q_prime_sign = 1 if (getattr(eq, 'q1', 1.0) - getattr(eq, 'q0', 0.5)) >= 0 else -1
        opoint_theta = ((-np.pi/2) * q_prime_sign - phi_mn) / m_k % (2 * np.pi / m_k)
        xpoint_theta = ((+np.pi/2) * q_prime_sign - phi_mn) / m_k % (2 * np.pi / m_k)

        print(f"  k={k}: ({m_k},{n_k}) ψ_res={psi_res:.3f} q_res={q_res:.3f} "
              f"|b_mn|={abs(b_mn):.3e} phase_arg={np.degrees(phi_mn):.1f}° "
              f"w_ψ={half_width_psi:.4f} ({half_width_r*100:.2f} cm) "
              f"θ_O={np.degrees(opoint_theta):.1f}° θ_X={np.degrees(xpoint_theta):.1f}°")

        components.append(ResonantComponent(
            m=m_k, n=n_k,
            harmonic_order=k,
            b_mn=b_mn,
            psi_res=psi_res,
            q_res=q_res,
            half_width_psi=half_width_psi,
            half_width_r=half_width_r,
            opoint_theta=opoint_theta,
            xpoint_theta=xpoint_theta,
            q_prime_sign=q_prime_sign,
        ))

    return components


def plot_island_width_bars(
    ax,
    components: List[ResonantComponent],
    eq,
    phi_section: float = 0.0,
    colors: list = None,
    label_harmonics: bool = True,
) -> None:
    """Draw island width bars at O-point positions on R-Z cross-section.

    Parameters
    ----------
    phi_section : float
        Toroidal angle φ (radians) of this Poincaré cross-section.
        O/X-point angles are computed via the general formula at this φ.
    """
    if colors is None:
        colors = ISLAND_CMAPS

    R0 = eq.R0
    r0 = eq.r0

    for comp in components:
        color = colors[(comp.harmonic_order - 1) % len(colors)]
        r_res = np.sqrt(comp.psi_res) * r0

        # Use the general φ-aware formula
        pts = comp.fixed_points(phi_section)
        theta_O_all = pts['theta_O'][0]   # shape (m,)  — all m O-points
        theta_X_all = pts['theta_X'][0]   # shape (m,)

        for i_op in range(comp.m):
            theta_op = theta_O_all[i_op]
            theta_xp = theta_X_all[i_op]

            R_O = R0 + r_res * np.cos(theta_op)
            Z_O =      r_res * np.sin(theta_op)
            R_X = R0 + r_res * np.cos(theta_xp)
            Z_X =      r_res * np.sin(theta_xp)

            r_inner = max(0.01, r_res - comp.half_width_r)
            r_outer = r_res + comp.half_width_r

            R_in  = R0 + r_inner * np.cos(theta_op)
            Z_in  =      r_inner * np.sin(theta_op)
            R_out = R0 + r_outer * np.cos(theta_op)
            Z_out =      r_outer * np.sin(theta_op)

            # Island width bar at O-point
            ax.plot([R_in, R_out], [Z_in, Z_out],
                    color=color, linewidth=3.5, alpha=0.85,
                    solid_capstyle='round', zorder=5)
            ax.plot(R_O, Z_O, 'o', color=color, markersize=6, zorder=6)
            # X-point marker
            ax.plot(R_X, Z_X, 'x', color=color, markersize=7, markeredgewidth=1.5,
                    zorder=6, alpha=0.7)

        if label_harmonics:
            theta_op0 = theta_O_all[0]
            r_label = r_res + comp.half_width_r + 0.015
            R_label = R0 + r_label * np.cos(theta_op0)
            Z_label =      r_label * np.sin(theta_op0)
            ax.annotate(
                f'$({comp.m},{comp.n})$',
                xy=(R_label, Z_label),
                fontsize=7, color=color,
                ha='center', va='center', zorder=7,
                fontweight='bold',
            )


# ---------------------------------------------------------------------------
# 2-D (m, n) Fourier spectrum heatmap utilities
# ---------------------------------------------------------------------------

def compute_mn_spectrum(
    delta_B_func,
    S: float,
    equilibrium,
    m_max: int = 6,
    n_max: int = 4,
    n_theta: int = 64,
    n_phi: int = 64,
    phi0: float = 0.0,
) -> np.ndarray:
    """Compute the 2-D (m, n) Fourier spectrum of delta_B on a flux surface.

    Samples the radial perturbation field delta_B^psi = delta_BR * cos(theta)
    + delta_BZ * sin(theta) on a flux surface at normalised label S, then
    returns a (2*m_max+1) x (2*n_max+1) array of complex Fourier amplitudes
    b_{m,n} for m in [-m_max, m_max] and n in [-n_max, n_max].

    Parameters
    ----------
    delta_B_func : callable
        ``(R, Z, phi) -> [dBR, dBZ, dBphi]``
    S : float
        Normalised flux label (r_minor / r0)^2.
    equilibrium :
        Provides ``R0``, ``r0``.
    m_max, n_max : int
        Maximum poloidal / toroidal mode numbers.
    n_theta, n_phi : int
        Sampling resolution in theta and phi.
    phi0 : float
        Starting toroidal angle (unused; sampling covers full [0, 2pi)).

    Returns
    -------
    b_mn : ndarray, shape (2*m_max+1, 2*n_max+1), complex
        b_mn[i, j] = amplitude for m = i - m_max, n = j - n_max.
    """
    R0, r0 = equilibrium.R0, equilibrium.r0
    r = np.sqrt(S) * r0

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi   = np.linspace(0, 2 * np.pi, n_phi,   endpoint=False)

    R_surf = R0 + r * np.cos(theta)
    Z_surf =      r * np.sin(theta)

    dBpsi = np.zeros((n_theta, n_phi), dtype=complex)
    for j_phi in range(n_phi):
        for i_th in range(n_theta):
            try:
                db = delta_B_func(R_surf[i_th], Z_surf[i_th], phi[j_phi])
                dBpsi[i_th, j_phi] = (
                    db[0] * np.cos(theta[i_th]) + db[1] * np.sin(theta[i_th])
                )
            except Exception:
                pass

    # Full 2-D DFT: b_{m,n} at fftfreq indices
    B_fft = np.fft.fft2(dBpsi) / (n_theta * n_phi)
    m_freq = np.fft.fftfreq(n_theta, 1 / n_theta).astype(int)
    n_freq = np.fft.fftfreq(n_phi,   1 / n_phi).astype(int)

    b_mn = np.zeros((2 * m_max + 1, 2 * n_max + 1), dtype=complex)
    for i, m in enumerate(range(-m_max, m_max + 1)):
        im = np.where(m_freq == m)[0]
        if not len(im):
            continue
        for j, n in enumerate(range(-n_max, n_max + 1)):
            jn = np.where(n_freq == n)[0]
            if not len(jn):
                continue
            b_mn[i, j] = B_fft[im[0], jn[0]]

    return b_mn


def plot_mn_heatmap(
    b_mn: np.ndarray,
    m_max: int = 6,
    n_max: int = 4,
    ax=None,
    log_scale: bool = True,
    title: str = r'$|\tilde{b}_{mn}|$ spectrum',
    cmap: str = 'hot_r',
    vmin: float = None,
    annotate: bool = True,
    highlight_modes: list = None,
) -> "tuple[plt.Figure, plt.Axes]":
    """Plot a (m, n) Fourier amplitude heatmap.

    Parameters
    ----------
    b_mn : ndarray, shape (2*m_max+1, 2*n_max+1)
        Complex Fourier amplitudes from ``compute_mn_spectrum``.
    m_max, n_max : int
        Must match the shape of b_mn.
    ax : matplotlib Axes or None
    log_scale : bool
        Use log10 colour scale (recommended for large dynamic range).
    title : str
    cmap : str
        Matplotlib colourmap name.
    vmin : float or None
        Minimum value for colour scale (log10 units if log_scale=True).
    annotate : bool
        Annotate each cell with its numeric value.
    highlight_modes : list of (m, n) tuples
        Draw a red box around these specific modes.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    amps = np.abs(b_mn)
    m_range = np.arange(-m_max, m_max + 1)
    n_range = np.arange(-n_max, n_max + 1)

    if log_scale:
        plot_data = np.log10(amps + 1e-30)
        cbar_label = r'$\log_{10}|\tilde{b}_{mn}|$'
        if vmin is None:
            vmin = plot_data.max() - 6  # show 6 decades
    else:
        plot_data = amps
        cbar_label = r'$|\tilde{b}_{mn}|$'
        if vmin is None:
            vmin = 0.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(n_range) + 1.5),
                                        max(4, 0.5 * len(m_range) + 1.5)))
    else:
        fig = ax.figure

    im = ax.imshow(
        plot_data,
        origin='lower',
        aspect='auto',
        cmap=cmap,
        vmin=vmin,
        vmax=plot_data.max(),
        extent=[-n_max - 0.5, n_max + 0.5, -m_max - 0.5, m_max + 0.5],
        interpolation='nearest',
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label(cbar_label, fontsize=9)

    ax.set_xlabel('n  (toroidal mode)', fontsize=10)
    ax.set_ylabel('m  (poloidal mode)',  fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(n_range)
    ax.set_yticks(m_range)
    ax.axvline(0, color='white', lw=0.5, alpha=0.4)
    ax.axhline(0, color='white', lw=0.5, alpha=0.4)

    if annotate:
        for i, m in enumerate(m_range):
            for j, n in enumerate(n_range):
                val = amps[i, j]
                if log_scale:
                    txt = f'{np.log10(val+1e-30):.1f}'
                else:
                    txt = f'{val:.1e}'
                ax.text(n, m, txt, ha='center', va='center',
                        fontsize=5.5, color='white' if plot_data[i, j] > (plot_data.max() + vmin) / 2 else 'black')

    if highlight_modes:
        for (hm, hn) in highlight_modes:
            if abs(hm) <= m_max and abs(hn) <= n_max:
                ax.add_patch(plt.Rectangle(
                    (hn - 0.5, hm - 0.5), 1, 1,
                    linewidth=2, edgecolor='red', facecolor='none', zorder=5,
                ))

    return fig, ax


def plot_mn_heatmap_radial(
    delta_B_func,
    equilibrium,
    S_values: np.ndarray,
    m_max: int = 4,
    n_max: int = 3,
    n_theta: int = 32,
    n_phi: int = 32,
    target_modes: list = None,
    fig_title: str = 'Fourier spectrum vs flux surface',
    cmap: str = 'hot_r',
) -> "tuple[plt.Figure, list]":
    """Plot one (m,n)-heatmap per flux surface, arranged in a row.

    For each S in S_values, compute the full (m,n) spectrum and plot
    a heatmap.  Useful for showing how the resonant structure varies
    radially across the plasma.

    Parameters
    ----------
    S_values : array_like
        Normalised flux labels at which to evaluate the spectrum.
    target_modes : list of (m,n) or None
        Highlight these modes with a red box in every panel.

    Returns
    -------
    fig, axes
    """
    S_values = np.atleast_1d(S_values)
    nS = len(S_values)

    fig, axes = plt.subplots(1, nS, figsize=(3.5 * nS, 3.5))
    if nS == 1:
        axes = [axes]

    for ax, S in zip(axes, S_values):
        b_mn = compute_mn_spectrum(
            delta_B_func, S, equilibrium,
            m_max=m_max, n_max=n_max,
            n_theta=n_theta, n_phi=n_phi,
        )
        plot_mn_heatmap(
            b_mn, m_max=m_max, n_max=n_max,
            ax=ax, log_scale=True,
            title=f'S={S:.2f}  (q={equilibrium.q_of_psi(S):.2f})',
            cmap=cmap,
            annotate=(nS <= 4),
            highlight_modes=target_modes,
        )

    fig.suptitle(fig_title, fontsize=12)
    plt.tight_layout()
    return fig, axes
