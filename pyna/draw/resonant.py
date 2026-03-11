"""Visualisation of resonant magnetic perturbation spectra.

Provides functions to plot the :math:`\\tilde{b}_{mn}` spectrum as
poloidal/toroidal (m, n) 2-D colour maps and 3-D bar charts.

These functions require matplotlib.  Import errors are deferred so
that the rest of the package can be used without a display.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def plot_tilde_b_mn_spectrum(
    tilde_b_mn_S: np.ndarray,
    S: np.ndarray,
    m_values: Optional[Sequence[int]] = None,
    n_values: Optional[Sequence[int]] = None,
    ax=None,
    cmap: str = "viridis",
    log_scale: bool = True,
    title: str = r"$|\tilde{b}_{mn}|$ spectrum",
):
    """Plot the :math:`|\\tilde{b}_{mn}|` amplitude as a 2-D colour map.

    Parameters
    ----------
    tilde_b_mn_S:
        2-D array of shape ``(n_modes, nS)`` containing the amplitude
        of each (m, n) mode as a function of flux surface label S.
        Row *i* corresponds to the mode ``(m_values[i], n_values[i])``.
    S:
        1-D array of flux-surface labels.
    m_values, n_values:
        Poloidal and toroidal mode numbers for each row.  If ``None``
        the row index is used.
    ax:
        Matplotlib ``Axes`` object.  A new figure is created if
        ``None``.
    cmap:
        Colour map name.
    log_scale:
        If ``True``, plot :math:`\\log_{10}|\\tilde{b}_{mn}|`.
    title:
        Figure title.

    Returns
    -------
    fig, ax
        The matplotlib Figure and Axes.
    """
    import matplotlib.pyplot as plt

    tilde_b_mn_S = np.asarray(tilde_b_mn_S, dtype=float)
    n_modes, nS = tilde_b_mn_S.shape
    mode_labels = (
        [f"({m},{n})" for m, n in zip(m_values, n_values)]
        if (m_values is not None and n_values is not None)
        else [str(i) for i in range(n_modes)]
    )

    data = np.abs(tilde_b_mn_S)
    if log_scale:
        with np.errstate(divide="ignore"):
            data = np.log10(np.where(data > 0, data, np.nan))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(3, n_modes * 0.4)))
    else:
        fig = ax.figure

    pcm = ax.pcolormesh(S, np.arange(n_modes), data, cmap=cmap, shading="auto")
    fig.colorbar(pcm, ax=ax, label=r"$\log_{10}|\tilde{b}_{mn}|$" if log_scale else r"$|\tilde{b}_{mn}|$")
    ax.set_yticks(np.arange(n_modes) + 0.5)
    ax.set_yticklabels(mode_labels, fontsize=7)
    ax.set_xlabel("S")
    ax.set_title(title)
    return fig, ax


def bar3d_tilde_b_mn_on_surface(
    tilde_b_mn: np.ndarray,
    m_max: int,
    n_max: int,
    S_value: Optional[float] = None,
    ax=None,
    title: str = r"$|\tilde{b}_{mn}|$ at surface",
):
    """3-D bar chart of :math:`|\\tilde{b}_{mn}|` for all (m, n) modes.

    Parameters
    ----------
    tilde_b_mn:
        Array of shape ``(m_max+1, n_max+1)`` (or ``(nS, m_max+1, n_max+1)``
        if a flux surface is specified via ``S_value``).
    m_max, n_max:
        Maximum poloidal and toroidal mode numbers.
    S_value:
        If ``tilde_b_mn`` has shape ``(nS, m_max+1, n_max+1)``, the
        S index (integer) to extract.
    ax:
        Matplotlib 3-D ``Axes``.  Created if ``None``.
    title:
        Plot title.

    Returns
    -------
    fig, ax
        The matplotlib Figure and Axes.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    tilde_b_mn = np.asarray(tilde_b_mn, dtype=float)
    if tilde_b_mn.ndim == 3:
        idx = int(S_value) if S_value is not None else tilde_b_mn.shape[0] // 2
        data_2d = np.abs(tilde_b_mn[idx])
    else:
        data_2d = np.abs(tilde_b_mn)

    m_arr = np.arange(1, m_max + 1)
    n_arr = np.arange(1, n_max + 1)
    MM, NN = np.meshgrid(m_arr, n_arr, indexing="ij")
    heights = data_2d[:m_max, :n_max].ravel()

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    xpos = MM.ravel() - 0.5
    ypos = NN.ravel() - 0.5
    zpos = np.zeros_like(heights)
    dx = dy = 0.8

    ax.bar3d(xpos, ypos, zpos, dx, dy, heights, zsort="average")
    ax.set_xlabel("m")
    ax.set_ylabel("n")
    ax.set_zlabel(r"$|\tilde{b}_{mn}|$")
    ax.set_title(title)
    return fig, ax
