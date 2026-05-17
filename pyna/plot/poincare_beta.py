"""Poincare-section comparison plots for beta scans.

The helpers here deliberately consume plain dictionaries/lists of traced
Poincare points.  Domain-specific projects such as topoquest can use their
preferred cyna/pyna tracers to build the payload, while the plotting layer
stays reusable for the wider pyna community.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def plot_poincare_beta_grid(
    rows: Sequence[Mapping],
    phi_sections: Sequence[float],
    *,
    out_path: str | Path | None = None,
    vacuum_axis_R: float | Sequence[float] | None = None,
    vacuum_axis_Z: float | Sequence[float] | None = None,
    title: str = "",
    figsize: tuple[float, float] | None = None,
    dpi: int = 170,
    point_size: float = 1.8,
    point_alpha: float = 0.62,
    axis_limits: tuple[float, float, float, float] | None = None,
):
    """Plot an ``n_beta x n_phi`` grid of Poincare sections.

    Parameters
    ----------
    rows:
        Sequence of dictionaries.  Each row should provide ``beta`` and
        ``core_by_sec``.  ``core_by_sec[s][i]`` is a ``(R_points, Z_points)``
        tuple for seed ``i`` at section ``s``.  Optional keys are
        ``r_norm_core`` and ``axis_by_sec``.
    phi_sections:
        Toroidal section angles in radians.  Four W7-X sections are typical,
        but the routine accepts any positive number of columns.
    vacuum_axis_R, vacuum_axis_Z:
        Scalar or per-section axis coordinates used as reference guide lines.
    out_path:
        If given, save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    n_rows = len(rows)
    n_cols = len(phi_sections)
    if n_rows == 0 or n_cols == 0:
        raise ValueError("rows and phi_sections must be non-empty")

    if figsize is None:
        figsize = (3.4 * n_cols, 2.75 * n_rows)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    cmap = plt.get_cmap("viridis")

    def _per_section(value, idx):
        if value is None:
            return None
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return float(arr)
        return float(arr[min(idx, len(arr) - 1)])

    for ridx, row in enumerate(rows):
        beta = float(row.get("beta", np.nan))
        label = row.get("label", f"<beta>={beta:.4g}")
        core_by_sec = row.get("core_by_sec", [])
        r_norm = np.asarray(row.get("r_norm_core", []), dtype=float)
        axis_by_sec = row.get("axis_by_sec", [None] * n_cols)
        for cidx, phi in enumerate(phi_sections):
            ax = axes[ridx, cidx]
            traces = core_by_sec[cidx] if cidx < len(core_by_sec) else []
            for tidx, trace in enumerate(traces):
                if trace is None:
                    continue
                Rp, Zp = trace
                Rp = np.asarray(Rp, dtype=float)
                Zp = np.asarray(Zp, dtype=float)
                if Rp.size == 0:
                    continue
                rn = float(r_norm[tidx]) if tidx < len(r_norm) else 0.5
                ax.scatter(
                    Rp,
                    Zp,
                    s=point_size,
                    alpha=point_alpha,
                    color=cmap(np.clip(rn, 0.0, 1.0)),
                    linewidths=0,
                    rasterized=True,
                )

            ref_R = _per_section(vacuum_axis_R, cidx)
            if ref_R is not None:
                ax.axvline(ref_R, color="0.15", lw=0.8, ls="--", alpha=0.72)
            ref_Z = _per_section(vacuum_axis_Z, cidx)
            if ref_Z is not None:
                ax.axhline(ref_Z, color="0.35", lw=0.6, ls=":", alpha=0.45)

            axis = axis_by_sec[cidx] if cidx < len(axis_by_sec) else None
            if axis is not None:
                ax.plot(float(axis[0]), float(axis[1]), "o", ms=4.2, mfc="white", mec="black", mew=0.9)

            if ridx == 0:
                ax.set_title(f"phi={float(phi):.3f} rad", fontsize=10)
            if cidx == 0:
                ax.set_ylabel(f"{label}\nZ [m]", fontsize=9)
            if ridx == n_rows - 1:
                ax.set_xlabel("R [m]", fontsize=9)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, lw=0.25, color="0.85", alpha=0.55)
            if axis_limits is not None:
                ax.set_xlim(axis_limits[0], axis_limits[1])
                ax.set_ylim(axis_limits[2], axis_limits[3])

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    if out_path is not None:
        out = Path(out_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return fig


__all__ = ["plot_poincare_beta_grid"]
