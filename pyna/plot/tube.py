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
    Full (2 by 2) or n-panel Poincare figure: each panel is one section,
    showing the TubeChain's Islands with per-Tube visual identity.

plot_island_chain_by_tube(chain, ax, **style)
    Plot a pyna.topo.island.IslandChain with per-Tube colour coding based
    on island.resonance_index.

plot_resonance_structure_section(resonance_structure, section, ax, **kw)
    Plot a ResonanceStructure (O-chain + X-chain) at a given section.

tube_chain_legend(tube_chain, ax, kind, loc)
    Add a per-Tube legend to an existing axes.
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

_FALLBACK_COLOR = '#aaaaaa'   # used when resonance_index is None


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
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting.") from e

    from pyna.topo.section import ToroidalSection

    if isinstance(section, (int, float)):
        section = ToroidalSection(float(section))
    phi = section.phi if hasattr(section, 'phi') else None

    if ax is None:
        _, ax = plt.subplots()

    # O-point Islands from section cut
    if show_o:
        islands = tube.section_islands(section)
        s = {**_tube_style(tube_idx, 'O'), **(style or {})}
        for island in islands:
            R, Z = float(island.O_point[0]), float(island.O_point[1])
            ax.plot(R, Z, marker=s['marker'], color=s['color'],
                    ms=s['ms'], mew=0, ls='None', zorder=20)

    # X-point markers from x_cycles
    if show_x and phi is not None:
        for xc in tube.x_cycles:
            # x_cycles are Cycle objects; use section_points(phi)
            if hasattr(xc, 'section_points'):
                fps = xc.section_points(phi)
            elif hasattr(xc, 'at_section'):
                fps = xc.at_section(phi)
            else:
                fps = []
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
    show_connectivity: bool = False,
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
    show_connectivity : bool
        If True and connectivity is wired (Island._next is set), draw thin
        lines connecting island → island.next() to visualise orbit order.
        Line colour matches the tube colour, alpha=0.3.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting.") from e

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

    # Draw connectivity lines between sequentially-connected Islands
    if show_connectivity:
        chain = tube_chain.to_island_chain_connected(section.phi)
        islands = chain.O_islands if hasattr(chain, 'O_islands') else []
        if not islands and hasattr(chain, 'islands'):
            islands = chain.islands
        for isl in islands:
            if isl is None:
                continue
            nxt = isl.next()
            if nxt is isl:
                continue  # disconnected, skip
            tidx = isl.resonance_index if isl.resonance_index is not None else 0
            color = TUBE_COLORS[tidx % len(TUBE_COLORS)] if per_tube_style else _FALLBACK_COLOR
            R0, Z0 = float(isl.O_point[0]), float(isl.O_point[1])
            R1, Z1 = float(nxt.O_point[0]), float(nxt.O_point[1])
            ax.plot([R0, R1], [Z0, Z1], '-', color=color, alpha=0.3, lw=0.8, zorder=5)

    return ax


# ── IslandChain plot with per-Tube colour coding ──────────────────────────────

def plot_island_chain_by_tube(
    chain,
    ax=None,
    *,
    show_o: bool = True,
    show_x: bool = True,
    show_arrows: bool = False,
    per_tube_style: bool = True,
    style: Optional[Dict[str, Any]] = None,
):
    """Plot a pyna.topo.island.IslandChain with per-Tube colour coding.

    Each Island is coloured by its ``resonance_index`` attribute (set by
    :meth:`TubeChain._attach_chain_refs`).  Islands without a resonance_index
    are drawn in a neutral grey.

    Parameters
    ----------
    chain : IslandChain
        The island chain to plot (from pyna.topo.island).
    ax : matplotlib.axes.Axes, optional
    show_o : bool
        Plot O-point markers.
    show_x : bool
        Plot X-point markers (from island.X_points).
    show_arrows : bool
        Draw eigenvector arrows (requires island.x_orbit data).
    per_tube_style : bool
        Use resonance_index-based colours.  False → uniform grey.
    style : dict, optional
        Override per-Tube style.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting.") from e

    if ax is None:
        _, ax = plt.subplots()

    # Gather islands from chain
    islands = []
    if hasattr(chain, 'O_islands') and chain.O_islands:
        islands = chain.O_islands
    elif hasattr(chain, 'islands') and chain.islands:
        islands = chain.islands
    elif hasattr(chain, 'fixed_points'):
        # Fallback: iterable of Island-like objects
        islands = list(chain.fixed_points)

    for isl in islands:
        if isl is None:
            continue
        ridx = getattr(isl, 'resonance_index', None)
        if per_tube_style and ridx is not None:
            color = TUBE_COLORS[int(ridx) % len(TUBE_COLORS)]
            marker_o = TUBE_MARKERS_O[int(ridx) % 10]
            marker_x = TUBE_MARKERS_X[int(ridx) % 10]
        else:
            color = _FALLBACK_COLOR
            marker_o = 'o'
            marker_x = 'x'

        s = style or {}

        if show_o:
            R, Z = float(isl.O_point[0]), float(isl.O_point[1])
            ax.plot(R, Z, marker=marker_o, color=color,
                    ms=s.get('ms', 7), mew=0, ls='None', zorder=20)

        if show_x:
            for xpt in isl.X_points:
                xpt = np.asarray(xpt, dtype=float)
                ax.plot(float(xpt[0]), float(xpt[1]),
                        marker=marker_x, color=color,
                        ms=s.get('ms', 7), mew=s.get('mew', 1.8), ls='None', zorder=20)

    return ax


# ── TubeChain/resonance section plot ─────────────────────────────────────────

def plot_resonance_section(
    tubechain,
    section,
    ax=None,
    *,
    show_o: bool = True,
    show_x: bool = True,
    show_arrows: bool = True,
    per_tube_style: bool = True,
    style: Optional[Dict[str, Any]] = None,
):
    """Plot O-points and X-points of a TubeChain at a given Poincare section.

    Parameters
    ----------
    tubechain : TubeChain
        Contains both O-cycles (in each Tube.o_cycle) and X-cycles
        (in each Tube.x_cycles).
    section : Section | float
    ax : matplotlib.axes.Axes, optional
    show_o, show_x : bool
    show_arrows : bool
        Draw eigenvector arrows at X-points.
    per_tube_style : bool
    style : dict, optional
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting.") from e

    from pyna.topo.section import ToroidalSection

    if isinstance(section, (int, float)):
        section = ToroidalSection(float(section))
    phi = section.phi if hasattr(section, 'phi') else 0.0

    if ax is None:
        _, ax = plt.subplots()

    if show_o:
        plot_tube_chain_section(
            tubechain, section, ax=ax,
            show_o=True, show_x=False, show_arrows=False,
            per_tube_style=per_tube_style, style=style,
        )

    if show_x:
        for idx, tube in enumerate(tubechain.tubes):
            tidx = idx if per_tube_style else 0
            color = f'C{tidx % 10}'
            if style and 'x' in style:
                s = style['x']
                color = s.get('color', color)
            for xc in tube.x_cycles:
                x_fps = xc.section_points(phi)
                for fp in x_fps:
                    ax.plot(fp.R, fp.Z, marker='x', color=color,
                            ms=7, mew=1.8, ls='None', zorder=20)
                    if show_arrows and fp.monodromy.stability.name == 'HYPERBOLIC':
                        evals, evecs = np.linalg.eig(fp.DPm)
                        for evec in evecs.T:
                            evec = evec.real
                            norm = np.linalg.norm(evec)
                            if norm > 1e-10:
                                evec /= norm
                                ax.annotate('', xy=(fp.R + 0.02*evec[0], fp.Z + 0.02*evec[1]),
                                            xytext=(fp.R, fp.Z),
                                            arrowprops=dict(arrowstyle='->', color=color))
    return ax


def plot_resonance_structure_section(
    resonance_structure,
    section,
    ax=None,
    **kwargs,
):
    """Deprecated: use plot_resonance_section(tubechain, section) instead.

    This function used to accept a ResonanceStructure object.  Since
    ResonanceStructure has been removed, pass a TubeChain directly.
    """
    import warnings
    warnings.warn(
        "plot_resonance_structure_section is deprecated.  "
        "Pass a TubeChain to plot_resonance_section() instead.",
        DeprecationWarning, stacklevel=2,
    )
    return plot_resonance_section(resonance_structure, section, ax=ax, **kwargs)


# ── Multi-section Poincare figure ─────────────────────────────────────────────

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
    """Draw a multi-panel Poincare figure for a TubeChain.

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
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.lines as ml
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting.") from e

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
    handles = tube_legend_handles(tube_chain)
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
    """Return matplotlib legend handles for each Tube in a TubeChain.

    Parameters
    ----------
    tube_chain : TubeChain
    kind : 'O' or 'X'
        Which marker style to use in the legend.
    """
    try:
        import matplotlib.lines as ml
    except ImportError as e:
        raise ImportError("matplotlib is required for tube_legend_handles.") from e

    handles = []
    for idx, tube in enumerate(tube_chain.tubes):
        s = _tube_style(idx, kind)
        lbl = tube.label or f'Tube {idx}  (resonance_index={idx})'
        handles.append(ml.Line2D([], [], marker=s['marker'], color=s['color'],
                                  ls='None', ms=6, label=lbl))
    return handles


def tube_chain_legend(
    tube_chain,
    ax,
    kind: str = 'O',
    loc: str = 'upper right',
    **legend_kw,
):
    """Add a per-Tube legend to an existing axes.

    Parameters
    ----------
    tube_chain : TubeChain
    ax : matplotlib.axes.Axes
    kind : 'O' or 'X'
        Marker style shown in the legend.
    loc : str
        Legend location string (passed to ax.legend).
    **legend_kw
        Additional keyword arguments forwarded to ax.legend.
    """
    handles = tube_legend_handles(tube_chain, kind=kind)
    if handles:
        ax.legend(handles=handles, loc=loc, fontsize=8, **legend_kw)
    return ax

