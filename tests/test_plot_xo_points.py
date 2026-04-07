# -*- coding: utf-8 -*-
"""Tests for pyna.plot.xo_points (generic X/O point drawing helpers)."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import pytest

from pyna.plot.xo_points import draw_xo_points, XO_STYLE


# ---------------------------------------------------------------------------
# Minimal stub for a fixed-point object
# ---------------------------------------------------------------------------

class _FP:
    """Minimal fixed-point stub: just R, Z and optional eigenvectors."""
    def __init__(self, R, Z, stable_ev=None, unstable_ev=None):
        self.R = R
        self.Z = Z
        self.stable_eigenvec   = stable_ev
        self.unstable_eigenvec = unstable_ev


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_xo_style_has_required_keys():
    required = {
        'xpt_color', 'xpt_ms', 'xpt_mew',
        'opt_color', 'opt_ms', 'opt_mew',
        'arrow_len', 'arrow_lw', 'arrow_stable', 'arrow_unstable',
    }
    assert required.issubset(XO_STYLE.keys())


def test_draw_xo_points_smoke():
    """draw_xo_points should not raise for typical inputs."""
    fig, ax = plt.subplots()
    xpts = [_FP(1.5, 0.2), _FP(1.6, -0.1)]
    opts = [_FP(1.55, 0.0)]
    draw_xo_points(ax, xpts, opts, show_arrows=False)
    plt.close(fig)


def test_draw_xo_points_with_arrows():
    """Eigenvector arrows are drawn when stable_eigenvec is set."""
    fig, ax = plt.subplots()
    xpts = [_FP(1.5, 0.0, stable_ev=[1.0, 0.0], unstable_ev=[0.0, 1.0])]
    opts: list = []
    draw_xo_points(ax, xpts, opts, show_arrows=True)
    # ax.annotate(...) with arrowprops creates Annotation artists
    import matplotlib.text as mtext
    annotations = [c for c in ax.get_children()
                   if isinstance(c, mtext.Annotation)]
    assert len(annotations) >= 2
    plt.close(fig)


def test_draw_xo_points_style_override():
    """Custom style dict is respected."""
    fig, ax = plt.subplots()
    opts = [_FP(1.5, 0.0)]
    draw_xo_points(ax, [], opts, style={'opt_color': 'blue', 'opt_ms': 12})
    lines = ax.get_lines()
    assert len(lines) == 1
    assert lines[0].get_markersize() == 12
    plt.close(fig)


def test_draw_xo_points_empty():
    """Empty lists should produce no artists and not raise."""
    fig, ax = plt.subplots()
    draw_xo_points(ax, [], [])
    assert len(ax.get_lines()) == 0
    plt.close(fig)


def test_import_from_pyna_plot():
    """draw_xo_points and XO_STYLE are importable from pyna.plot."""
    from pyna.plot import draw_xo_points as dxo, XO_STYLE as xs  # noqa: F401
    assert callable(dxo)
    assert isinstance(xs, dict)
