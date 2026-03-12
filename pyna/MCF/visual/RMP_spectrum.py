"""RMP Fourier spectrum analysis and island width visualization.

For a SimpleStellarartor, compute resonant components analytically:
  - Resonant surface location from q(ψ) = n/m
  - Island half-width via Rutherford formula
  - O-point phase from the RMP field structure

References
----------
Boozer, Phys. Fluids B 3 (1991) — resonance condition
Rutherford, Phys. Fluids 16 (1973) — island width formula
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import matplotlib.pyplot as plt

from .equilibrium import ISLAND_CMAPS


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
    opoint_theta: float
    xpoint_theta: float


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

    Works with SimpleStellarartor's psi_ax / q_of_psi / resonant_psi API.
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

        # O-point phase
        phi_mn = np.angle(b_mn)
        opoint_theta = (phi_mn / m_k) % (2*np.pi / m_k)
        xpoint_theta = opoint_theta + np.pi / m_k

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
        ))

        print(f"  k={k}: ({m_k},{n_k}) ψ_res={psi_res:.3f} q_res={q_res:.3f} "
              f"|b_mn|={abs(b_mn):.3e} "
              f"w_ψ={half_width_psi:.4f} ({half_width_r*100:.2f} cm) "
              f"θ_O={np.degrees(opoint_theta):.1f}°")

    return components


def plot_island_width_bars(
    ax,
    components: List[ResonantComponent],
    eq,
    colors: list = None,
    label_harmonics: bool = True,
) -> None:
    """Draw island width bars at O-point positions on R-Z cross-section."""
    if colors is None:
        colors = ISLAND_CMAPS

    R0   = eq.R0
    r0   = eq.r0

    for comp in components:
        color = colors[(comp.harmonic_order - 1) % len(colors)]
        r_res = np.sqrt(comp.psi_res) * r0

        for i_op in range(comp.m):
            theta_op = comp.opoint_theta + i_op * 2*np.pi / comp.m

            R_O = R0 + r_res * np.cos(theta_op)
            Z_O =      r_res * np.sin(theta_op)

            r_inner = max(0.01, r_res - comp.half_width_r)
            r_outer = r_res + comp.half_width_r

            R_in  = R0 + r_inner * np.cos(theta_op)
            Z_in  =      r_inner * np.sin(theta_op)
            R_out = R0 + r_outer * np.cos(theta_op)
            Z_out =      r_outer * np.sin(theta_op)

            ax.plot([R_in, R_out], [Z_in, Z_out],
                    color=color, linewidth=3.5, alpha=0.85,
                    solid_capstyle='round', zorder=5)
            ax.plot(R_O, Z_O, 'o', color=color, markersize=5, zorder=6)

        if label_harmonics:
            theta_op0 = comp.opoint_theta
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
