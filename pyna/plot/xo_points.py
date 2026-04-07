# -*- coding: utf-8 -*-
"""
pyna.plot.xo_points
===================
Generic helpers for drawing X-point (saddle) and O-point (elliptic)
fixed-point markers on matplotlib Axes.

These utilities are deliberately **style-dict agnostic**: the caller
passes a plain ``dict`` of style keys, or uses the exported
:data:`XO_STYLE` defaults.  This makes them reusable by any downstream
package (e.g. topoquest) without coupling to that package's own STYLE
dict.

Typical usage::

    from pyna.plot.xo_points import draw_xo_points, XO_STYLE

    style = {**XO_STYLE, 'xpt_color': 'orange'}
    draw_xo_points(ax, xpts, opts, style, show_arrows=True)

Style keys
----------
``xpt_color``, ``xpt_ms``, ``xpt_mew``
    Appearance of X-point markers (plotted as ``'x'``).
``opt_color``, ``opt_ms``, ``opt_mew``
    Appearance of O-point markers (plotted as ``'o'``).
``arrow_len``
    Half-length of eigenvector arrows in data units.
``arrow_lw``
    Line width of eigenvector arrows.
``arrow_stable``, ``arrow_unstable``
    Colours for stable / unstable eigenvector arrows.
"""
from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Default style values (kept in sync with topoquest.plot.style.STYLE)
# ---------------------------------------------------------------------------

XO_STYLE: dict[str, Any] = {
    # X-point (saddle)
    'xpt_color': 'crimson',
    'xpt_ms':    9,
    'xpt_mew':   2.2,
    # O-point (elliptic)
    'opt_color': 'limegreen',
    'opt_ms':    7,
    'opt_mew':   1.5,
    # Eigenvector arrows
    'arrow_len':      0.022,
    'arrow_lw':       1.3,
    'arrow_stable':   '#00cfff',
    'arrow_unstable': '#ff8800',
}


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def draw_xo_points(
    ax: plt.Axes,
    xpts: Sequence,
    opts: Sequence,
    style: dict[str, Any] | None = None,
    *,
    show_arrows: bool = True,
) -> None:
    """Draw X/O fixed-point markers (and optional eigenvector arrows).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    xpts : sequence
        Iterable of X-point objects.  Each must expose ``.R`` and ``.Z``
        float attributes.  If *show_arrows* is *True*, the objects may
        additionally expose ``.stable_eigenvec`` and
        ``.unstable_eigenvec`` (each a length-2 sequence or *None*).
    opts : sequence
        Iterable of O-point objects (same protocol as *xpts*).
    style : dict, optional
        Style overrides merged on top of :data:`XO_STYLE`.
    show_arrows : bool
        Whether to draw eigenvector arrows for each X-point.
    """
    s: dict[str, Any] = {**XO_STYLE, **(style or {})}
    L: float = s['arrow_len']

    # O-points
    for fp in opts:
        ax.plot(
            fp.R, fp.Z, 'o',
            ms=s['opt_ms'],
            mec='k',
            mfc=s['opt_color'],
            mew=s['opt_mew'],
            zorder=12,
        )

    # X-points
    for fp in xpts:
        ax.plot(
            fp.R, fp.Z, 'x',
            ms=s['xpt_ms'],
            color=s['xpt_color'],
            mew=s['xpt_mew'],
            zorder=12,
        )
        if show_arrows:
            stable_ev   = getattr(fp, 'stable_eigenvec',   None)
            unstable_ev = getattr(fp, 'unstable_eigenvec', None)
            for evec, clr in [
                (stable_ev,   s['arrow_stable']),
                (unstable_ev, s['arrow_unstable']),
            ]:
                if evec is None:
                    continue
                ax.annotate(
                    '',
                    xy=(fp.R + L * evec[0], fp.Z + L * evec[1]),
                    xytext=(fp.R - L * evec[0], fp.Z - L * evec[1]),
                    arrowprops=dict(
                        arrowstyle='->',
                        color=clr,
                        lw=s['arrow_lw'],
                    ),
                    zorder=11,
                )
