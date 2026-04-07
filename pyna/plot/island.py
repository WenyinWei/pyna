from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def _as_point_xy(pt) -> tuple[float, float]:
    if hasattr(pt, "R") and hasattr(pt, "Z"):
        return float(pt.R), float(pt.Z)
    arr = np.asarray(pt, dtype=float)
    return float(arr[0]), float(arr[1])


def island_section_points(island, phi: float | None = None) -> dict:
    """Return section-level geometry for one Island.

    This is intentionally generic:
    - In 2D map contexts, ``phi`` is ignored.
    - In 3D toroidal contexts, if the island stores section-aware geometry,
      this accessor may use it later; for now it falls back to O/X points.
    """
    O = _as_point_xy(island.O_point) if getattr(island, "O_point", None) is not None else None
    X = [_as_point_xy(x) for x in getattr(island, "X_points", [])]
    return {
        "phi": phi,
        "O_points": [O] if O is not None else [],
        "X_points": X,
        "boundary_curves": [],
        "theta_lines": [],
    }


def plot_island(
    island,
    ax=None,
    phi: float | None = None,
    *,
    show_O: bool = True,
    show_X: bool = True,
    show_label: bool = True,
    show_boundary: bool = True,
):
    """Plot a single Island object.

    Design intent:
    - Plot only the target Island, not the whole chain.
    - Work for both 2D map islands and 3D toroidal-section footprints.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    sec = island_section_points(island, phi=phi)

    if show_boundary:
        for curve in sec["boundary_curves"]:
            curve = np.asarray(curve, dtype=float)
            if curve.ndim == 2 and curve.shape[1] >= 2:
                ax.plot(curve[:, 0], curve[:, 1], color="#ff8800", lw=1.8, zorder=3)

    if show_O:
        for R, Z in sec["O_points"]:
            ax.plot(R, Z, "o", color="limegreen", ms=7, mec="k", mew=0.6, zorder=10)
            if show_label and getattr(island, "label", None):
                ax.text(R, Z, f" {island.label}", color="limegreen", fontsize=9, zorder=11)

    if show_X:
        for R, Z in sec["X_points"]:
            ax.plot(R, Z, "x", color="crimson", ms=8, mew=2.0, zorder=10)

    ax.set_aspect("equal")
    return ax
