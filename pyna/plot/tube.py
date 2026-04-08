"""pyna.plot.tube -- First-class plotting for Tube and TubeChain objects.

These functions treat Tube and TubeChain as the primary objects:
  - A Tube is an invariant-torus structure; its section cut yields Islands.
  - Islands from different Tubes within the same TubeChain are visually
    distinguished by per-Tube colour/marker (resonance_index-based).

Main entry points
-----------------
plot_tube_section(tube, section, ax, **style)
    Plot the section cut of a single Tube (its O-point Island(s)).

plot_tube_chain_section(tube_chain, section, ax, **style)
    Plot the section cut of an entire TubeChain.  Islands from different
    Tubes get different colours/markers automatically.

plot_tube_chain_poincare(tube_chain, sections, msp, n_turns, figsize, ...)
    Full 2×2 (or n-panel) Poincaré figure: each panel is one section,
    showing the TubeChain's Islands with per-Tube visual identity.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# ── Default per-tube visual styles ───────────────────────────────────────────

TUBE_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
    '#ff7f00', '#a65628', '#f781bf', '#999999',
    '#66c2a5', '#fc8d62',
]
TUBE_MARKERS_O = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '+']
TUBE_MARKERS_X = ['x', '+', '*', 'd', 'v', 'P', 'o', 'X', 'h', 's']


def _tube_style(tube_idx: int, kind: str = 'O') -> dict:
    """Return marker/color style for Tube index tube_idx."""
    color = TUBE_COLORS[tube_idx % len(TUBE_COLORS)]
    marker = (TUBE_MARKERS_O if kind == 'O' else TUBE_MARKERS_X)[tube_idx % 10]
    return {'color': color, 'marker': marker, 'ms': 7, 'mew': 1.8, 'ls': 'None'}


# ── Single-Tube section plot ──────────────────────────────────────────────────

def plot_tube_section(
    tube,
    section,
    ax=None,
    *,
    tube_idx: int = 0,
    show_o: bool = True,
    show_x: bool = True,
    show_arrows: bool = True,
    style: Optional[Dict[str, Any]] = None,
):
    """Plot the section cut of a single Tube.

    The Tube is cut at ``section`` (ToroidalSection or float phi).
    O-point Islands are drawn with filled markers; X-point regions
    (from tube.x_cycles) are drawn with crosses.

    Parameters
    ----------
    tube : Tube
    section : Section | float
    ax : matplotlib.axes.Axes, optional
    tube_idx : int
        Index used to pick colour/marker (0 = first tube colour).
    show_o, show_x : bool
        Whether to draw O-point / X-point markers.
    show_arrows : bool
        Draw eigenvector arrows at X-points.
    style : dict, optional
        Override default style.
    """
    import matplotlib.pyplot as plt
    from pyna.topo.section import ToroidalSection

    if isinstance(section, (int, float)):
        section = ToroidalSection(float(section))
    phi = section.phi if hasattr(section, 'phi') else None

    if ax is None:
        _, ax = plt.subplots()

    # O-point Islands from section cut
    if show_o:
        islands = tube.section_cut(section)
        s = {**_tube_style(tube_idx, 'O'), **(style or {})}
        for island in islands:
            R, Z = float(island.O_point[0]), float(island.O_point[1])
            ax.plot(R, Z, marker=s['marker'], color=s['color'],
                    ms=s['ms'], mew=0, ls='None', zorder=20)

    # X-point markers from x_cycles
    if show_x and phi is not None:
        for x_tube in tube.x_cycles:
            fps = x_tube.at_section(phi) if hasattr(x_tube, 'at_section') else []
            s_x = {**_tube_style(tube_idx, 'X'), **(style or {})}
            for fp in fps:
                ax.plot(float(fp.R), float(fp.Z),
                        marker=s_x['marker'], color=s_x['color'],
                        ms=s_x['ms'], mew=s_x['mew'], ls='None', zorder=20)
                # Eigenvector arrows (unstable manifold direction)
                if show_arrows and hasattr(fp, 'unstable_eigenvec'):
                    evec = fp.unstable_eigenvec
                    if evec is not None:
                        arrow_len = 0.015
                        ax.annotate(
                            '', xy=(fp.R + arrow_len * evec[0], fp.Z + arrow_len * evec[1]),
                            xytext=(fp.R, fp.Z),
                            arrowprops=dict(arrowstyle='->', color=s_x['color'], lw=1.2),
                            zorder=21,
                        )

    return ax


# ── TubeChain section plot ────────────────────────────────────────────────────

def plot_tube_chain_section(
    tube_chain,
    section,
    ax=None,
    *,
    show_o: bool = True,
    show_x: bool = True,
    show_arrows: bool = True,
    per_tube_style: bool = True,
    style: Optional[Dict[str, Any]] = None,
):
    """Plot the section cut of an entire TubeChain.

    Each Tube gets a distinct colour/marker (based on resonance_index).
    Islands from the same Tube have the same visual identity across all
    panels of a multi-section plot.

    Parameters
    ----------
    tube_chain : TubeChain
    section : Section | float
    ax : matplotlib.axes.Axes, optional
    per_tube_style : bool
        If True, colour/marker varies per Tube.  If False, uniform style.
    """
    import matplotlib.pyplot as plt
    from pyna.topo.section import ToroidalSection

    if isinstance(section, (int, float)):
        section = ToroidalSection(float(section))

    if ax is None:
        _, ax = plt.subplots()

    for idx, tube in enumerate(tube_chain.tubes):
        tidx = idx if per_tube_style else 0
        plot_tube_section(
            tube, section, ax=ax,
            tube_idx=tidx,
            show_o=show_o,
            show_x=show_x,
            show_arrows=show_arrows,
            style=style,
        )
    return ax


# ── Multi-section Poincaré figure ─────────────────────────────────────────────

def plot_tube_chain_poincare(
    tube_chain,
    sections: Sequence,
    *,
    ax_labels: Optional[Sequence[str]] = None,
    figsize: tuple = (12, 10),
    dpi: int = 150,
    show_o: bool = True,
    show_x: bool = True,
    show_arrows: bool = True,
    per_tube_style: bool = True,
    background_fn=None,
    title: str = '',
    out_path=None,
):
    """Draw a multi-panel Poincaré figure for a TubeChain.

    Each panel corresponds to one section in ``sections``.  The same
    per-Tube colour/marker scheme is used across all panels, so Islands
    from the same Tube are immediately recognisable across sections.

    Parameters
    ----------
    tube_chain : TubeChain
    sections : list of Section or float
        One panel per section.
    ax_labels : list of str, optional
        Panel titles.  Defaults to section.label or 'φ=...' strings.
    background_fn : callable(ax, section), optional
        Called for each panel to draw background (Lc heatmap, LCFS, etc.)
        before the TubeChain Islands are plotted.
    title : str
        Figure suptitle.
    out_path : str | Path, optional
        Save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pyna.topo.section import ToroidalSection

    n_panels = len(sections)
    n_cols = min(n_panels, 2)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                              constrained_layout=True)
    axes_flat = np.asarray(axes).flat if n_panels > 1 else [axes]

    secs = [ToroidalSection(float(s)) if isinstance(s, (int, float)) else s
            for s in sections]
    labels = ax_labels or [str(s) for s in secs]

    for i, (sec, ax, lbl) in enumerate(zip(secs, axes_flat, labels)):
        if background_fn is not None:
            background_fn(ax, sec)
        plot_tube_chain_section(
            tube_chain, sec, ax=ax,
            show_o=show_o, show_x=show_x,
            show_arrows=show_arrows,
            per_tube_style=per_tube_style,
        )
        ax.set_title(lbl, fontsize=9)
        ax.set_xlabel('R [m]', fontsize=8)
        ax.set_ylabel('Z [m]', fontsize=8)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)

    # Legend: one entry per Tube
    handles = []
    for idx, tube in enumerate(tube_chain.tubes):
        import matplotlib.lines as ml
        s = _tube_style(idx, 'O')
        lbl = tube.label or f'Tube {idx}'
        handles.append(ml.Line2D([], [], marker=s['marker'], color=s['color'],
                                  ls='None', ms=6, label=lbl))

    if handles:
        fig.legend(handles=handles, loc='lower center',
                   ncol=min(len(handles), 5), fontsize=8,
                   framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    if title:
        fig.suptitle(title, fontsize=11, y=1.01)

    if out_path is not None:
        fig.savefig(str(out_path), dpi=dpi, bbox_inches='tight')

    return fig


# ── Legend helpers ────────────────────────────────────────────────────────────

def tube_legend_handles(tube_chain, kind: str = 'O'):
    """Return matplotlib legend handles for each Tube in a TubeChain."""
    import matplotlib.lines as ml
    handles = []
    for idx, tube in enumerate(tube_chain.tubes):
        s = _tube_style(idx, kind)
        lbl = tube.label or f'Tube {idx}  (resonance_index={idx})'
        handles.append(ml.Line2D([], [], marker=s['marker'], color=s['color'],
                                  ls='None', ms=6, label=lbl))
    return handles
