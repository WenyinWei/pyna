"""RMP (Resonant Magnetic Perturbation) spectrum analysis.

Functions for computing the normalised RMP amplitude :math:`\\tilde{b}`,
its Fourier decomposition into (m, n) modes, and estimating the
resulting magnetic island widths at rational surfaces.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

from pyna.topo.island import (
    locate_all_rational_surfaces,
    island_halfwidth,
)


def normalize_b(
    B_perturb: np.ndarray,
    B0_magnitude: np.ndarray,
    psi_coord: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Compute the normalised RMP field :math:`\tilde{b} = \delta B / B_0`.

    Parameters
    ----------
    B_perturb:
        Array of perturbed field component values.  Shape may be
        ``(nR, nZ, nPhi)`` or any broadcast-compatible shape.
    B0_magnitude:
        Equilibrium field magnitude at the same locations.  Must be
        broadcast-compatible with ``B_perturb``.
    psi_coord:
        Optional 1-D array of ψ_norm values used as a flux-surface
        coordinate.  Currently not used for the normalisation itself
        but passed through for downstream consumers.

    Returns
    -------
    ndarray
        :math:`\\tilde{b}` with the same shape as ``B_perturb``.
    """
    return np.asarray(B_perturb, dtype=float) / np.asarray(B0_magnitude, dtype=float)


def RMP_spectrum_2d(
    tilde_b: np.ndarray,
    section_coord: int = 2,
) -> np.ndarray:
    r"""Decompose :math:`\tilde{b}` into poloidal/toroidal Fourier modes.

    Performs a 2-D real FFT over the toroidal (φ) and poloidal (θ)
    dimensions of ``tilde_b``.

    Parameters
    ----------
    tilde_b:
        Normalised perturbation field with shape ``(nS, nTheta, nPhi)``
        where *nS* is the number of flux surfaces, *nTheta* is the
        number of poloidal samples, and *nPhi* is the number of
        toroidal samples.
    section_coord:
        Not currently used; reserved for future extension.

    Returns
    -------
    ndarray
        Complex array of shape ``(nS, nTheta, nPhi//2 + 1)`` — the
        one-sided FFT spectrum over (m, n) modes.  The *m* index runs
        from ``-nTheta//2`` to ``nTheta//2`` (after fftshift on the
        poloidal axis); the *n* index is the non-negative toroidal
        mode number (0 … nPhi//2).

    Notes
    -----
    The returned spectrum is **not** yet normalised by the grid size;
    use ``tilde_b_mn / (nTheta * nPhi)`` to get physical amplitudes.
    """
    tilde_b = np.asarray(tilde_b, dtype=float)
    # FFT over theta (axis 1) and phi (axis 2)
    spectrum = np.fft.rfft2(tilde_b, axes=(1, 2))
    return spectrum


# Backward-compatibility alias removed — use RMP_spectrum_2d directly.

def island_width_at_rational_surfaces(
    tilde_b_mn: np.ndarray,
    equilibrium,
    m_max: int = 10,
    n_max: int = 3,
) -> Dict[int, Dict[int, List[float]]]:
    r"""Estimate island widths at all q = m/n rational surfaces.

    Combines :func:`RMP_spectrum_2d` output with the equilibrium's q
    profile to compute island half-widths using
    :func:`~pyna.topo.island.island_halfwidth`.

    Parameters
    ----------
    tilde_b_mn:
        Normalised RMP spectrum, shape ``(nS, nTheta, nPhi_rfft)``,
        as returned by :func:`RMP_spectrum_2d`.
    equilibrium:
        An :class:`~pyna.mag.equilibrium.EquilibriumAxisym` instance
        providing ``.S`` and ``.q(S)``.
    m_max:
        Maximum poloidal mode number.
    n_max:
        Maximum toroidal mode number.

    Returns
    -------
    dict[int, dict[int, list[float]]]
        ``result[m][n]`` is a list of island half-widths (in S
        coordinate) at each rational surface with q = m/n.  Empty
        list if no surface exists in the domain.
    """
    S = equilibrium.S
    q_prof = equilibrium.q(S)

    # Locate all rational surfaces
    surfaces = locate_all_rational_surfaces(S, q_prof, m_max=m_max, n_max=n_max)

    nS, nTheta, nPhi_rfft = tilde_b_mn.shape
    result: Dict[int, Dict[int, List[float]]] = {}

    for m in range(1, m_max + 1):
        result[m] = {}
        for n in range(1, n_max + 1):
            s_list = surfaces.get(m, {}).get(n, [])
            widths = []
            for S_res in s_list:
                # rfft2 on axes (theta, phi):
                #   axis 1 (theta, nTheta points): full FFT, positive m at index m
                #   axis 2 (phi, nPhi_rfft points): rfft, n at index n
                nPhi_full = 2 * (nPhi_rfft - 1)
                if m >= nTheta or n >= nPhi_rfft:
                    widths.append(float("nan"))
                    continue
                # Average positive and negative m contributions
                b_pos = np.abs(tilde_b_mn[:, m, n]) / (nTheta * nPhi_full)
                if nTheta - m != m and (nTheta - m) < nTheta:
                    b_neg = np.abs(tilde_b_mn[:, nTheta - m, n]) / (nTheta * nPhi_full)
                else:
                    b_neg = b_pos
                b_profile = b_pos + b_neg  # physical amplitude = sum of conjugate pair
                w = island_halfwidth(m, n, S_res, S, q_prof, b_profile)
                widths.append(w)
            result[m][n] = widths

    return result
