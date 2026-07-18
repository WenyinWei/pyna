"""Convenience plotting helpers for Poincare section data."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from pyna.plot.section_geometry import draw_poincare_points, save_figure


def _as_2d_points(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("R and Z arrays must be one- or two-dimensional")
    return arr


def _split_flat_traces(R_flat, Z_flat, counts, turns: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    R_arr = np.asarray(R_flat, dtype=float).ravel()
    Z_arr = np.asarray(Z_flat, dtype=float).ravel()
    counts_arr = np.asarray(counts, dtype=int).ravel()
    if R_arr.shape != Z_arr.shape:
        raise ValueError("R_flat and Z_flat must have the same shape")
    if counts_arr.size == 0:
        return []

    n_seed = int(counts_arr.size)
    if turns is not None and int(turns) > 0 and R_arr.size == n_seed * int(turns):
        n_turns = int(turns)
        R_grid = R_arr.reshape(n_seed, n_turns)
        Z_grid = Z_arr.reshape(n_seed, n_turns)
        return [
            (R_grid[i, :max(0, min(int(count), n_turns))].copy(),
             Z_grid[i, :max(0, min(int(count), n_turns))].copy())
            for i, count in enumerate(counts_arr)
        ]

    if R_arr.size == int(np.sum(counts_arr)):
        traces = []
        start = 0
        for count in counts_arr:
            stop = start + max(0, int(count))
            traces.append((R_arr[start:stop].copy(), Z_arr[start:stop].copy()))
            start = stop
        return traces

    if R_arr.size % n_seed == 0:
        n_turns = R_arr.size // n_seed
        R_grid = R_arr.reshape(n_seed, n_turns)
        Z_grid = Z_arr.reshape(n_seed, n_turns)
        return [
            (R_grid[i, :max(0, min(int(count), n_turns))].copy(),
             Z_grid[i, :max(0, min(int(count), n_turns))].copy())
            for i, count in enumerate(counts_arr)
        ]

    raise ValueError("could not infer flat Poincare trace layout from counts")


def _traces_from_arrays(R, Z, counts=None, turns: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    R_arr = np.asarray(R, dtype=float)
    Z_arr = np.asarray(Z, dtype=float)
    if R_arr.shape != Z_arr.shape:
        raise ValueError("R and Z must have the same shape")
    if counts is not None and R_arr.ndim == 1:
        return _split_flat_traces(R_arr, Z_arr, counts, turns=turns)

    R_2d = _as_2d_points(R_arr)
    Z_2d = _as_2d_points(Z_arr)
    if R_2d.shape != Z_2d.shape:
        raise ValueError("R and Z must have the same shape")

    counts_arr = None if counts is None else np.asarray(counts, dtype=int).ravel()
    traces = []
    for i in range(R_2d.shape[0]):
        count = R_2d.shape[1] if counts_arr is None or i >= counts_arr.size else int(counts_arr[i])
        count = max(0, min(count, R_2d.shape[1]))
        traces.append((R_2d[i, :count].copy(), Z_2d[i, :count].copy()))
    return traces


def _traces_from_payload(poincare_data: Mapping | None, R=None, Z=None, counts=None, turns=None):
    data = {} if poincare_data is None else dict(poincare_data)
    if R is None:
        R = data.get("R_flat", data.get("R"))
    if Z is None:
        Z = data.get("Z_flat", data.get("Z"))
    if counts is None:
        counts = data.get("counts")
    if turns is None:
        turns = data.get("turns", data.get("N_turns"))
    if R is None or Z is None:
        raise ValueError("provide poincare_data with R/Z data, or pass R and Z directly")
    return _traces_from_arrays(R, Z, counts=counts, turns=turns)


def _load_boundary_curve(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        data = np.loadtxt(path)
    except ValueError:
        data = np.loadtxt(path, skiprows=1)
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("boundary file must contain at least two columns: R Z")
    return arr[:, 0], arr[:, 1]


def _append_totals(traces: Sequence[tuple[np.ndarray, np.ndarray]]):
    if not traces:
        empty = np.empty(0, dtype=float)
        return empty, empty, empty.astype(int)
    R_parts = []
    Z_parts = []
    seed_parts = []
    for i, (R, Z) in enumerate(traces):
        R_arr = np.asarray(R, dtype=float).ravel()
        Z_arr = np.asarray(Z, dtype=float).ravel()
        finite = np.isfinite(R_arr) & np.isfinite(Z_arr)
        R_parts.append(R_arr[finite])
        Z_parts.append(Z_arr[finite])
        seed_parts.append(np.full(np.count_nonzero(finite), i, dtype=int))
    return np.concatenate(R_parts), np.concatenate(Z_parts), np.concatenate(seed_parts)


def plot_poincare(
    poincare_data: Mapping | None = None,
    *,
    R=None,
    Z=None,
    counts=None,
    turns: int | None = None,
    phi0: float | None = None,
    ax=None,
    figsize: tuple[float, float] = (7.0, 7.0),
    point_size: float = 0.5,
    s: float | None = None,
    alpha: float = 0.6,
    cmap: str = "tab20",
    color=None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    boundary: tuple[Sequence[float], Sequence[float]] | None = None,
    plot_d1b: bool = False,
    d1b_file: str | Path | None = None,
    boundary_color: str = "k",
    boundary_lw: float = 1.0,
    title: str | None = None,
    grid: bool = True,
    save_png: bool = False,
    png_file: str | Path = "poincare.png",
    save_path: str | Path | None = None,
    dpi: int = 180,
    rasterized: bool = True,
    return_data: bool = False,
    **scatter_kwargs,
):
    """Plot Poincare section points in the R-Z plane.

    The input may be a physics-tools-style dictionary containing
    ``counts``, ``R_flat`` and ``Z_flat``, or pyna-style ``R`` and ``Z`` arrays
    with shape ``(n_seed, n_crossings)``.  Direct ``R=`` and ``Z=`` arguments
    are also accepted.
    """

    import matplotlib.pyplot as plt

    data = {} if poincare_data is None else dict(poincare_data)
    if turns is None:
        turns = data.get("turns", data.get("N_turns"))
    if phi0 is None:
        phi0 = data.get("phi0", data.get("phi"))
        if np.asarray(phi0).ndim > 0:
            phi0 = np.asarray(phi0, dtype=float).ravel()[0]

    traces = _traces_from_payload(data, R=R, Z=Z, counts=counts, turns=turns)
    R_total, Z_total, seed_index = _append_totals(traces)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    if s is not None:
        point_size = float(s)

    draw_poincare_points(
        ax,
        R_total,
        Z_total,
        seed_index,
        point_size=point_size,
        alpha=alpha,
        cmap=cmap,
        color=color,
        rasterized=rasterized,
        **scatter_kwargs,
    )

    if boundary is not None:
        wall_R, wall_Z = boundary
        ax.plot(wall_R, wall_Z, color=boundary_color, lw=boundary_lw)
    if plot_d1b:
        if d1b_file is None:
            raise ValueError("d1b_file is required when plot_d1b=True")
        wall_R, wall_Z = _load_boundary_curve(d1b_file)
        ax.plot(wall_R, wall_Z, color=boundary_color, lw=boundary_lw)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    if title is None:
        title_parts = ["Poincare"]
        if phi0 is not None and np.isfinite(float(phi0)):
            title_parts.append(f"phi={float(phi0):.3f}")
        if turns is not None:
            title_parts.append(f"turns={int(turns)}")
        title = ", ".join(title_parts)
    if title:
        ax.set_title(title)
    if grid:
        ax.grid(True, color="0.85", lw=0.35, alpha=0.6)
    fig.tight_layout()

    if save_path is None and save_png:
        save_path = png_file
    if save_path is not None:
        save_figure(fig, save_path, dpi=dpi)

    plotted_data = dict(data)
    plotted_data.update({
        "R_total": R_total,
        "Z_total": Z_total,
        "seed_index": seed_index,
        "traces": traces,
    })
    if return_data:
        return fig, ax, plotted_data
    return fig, ax


def plot_poincare_data(
    poincare_data: Mapping,
    save_png: bool = False,
    png_file: str | Path = "poincare.png",
    ax=None,
    plot_d1b: bool = False,
    d1b_file: str | Path | None = None,
    xlim: tuple[float, float] | None = (0.75, 2.0),
    ylim: tuple[float, float] | None = (-0.45, 0.45),
    s: float = 0.5,
    save_html: bool = False,
    html_file: str | Path | None = None,
    **kwargs,
):
    """Compatibility wrapper for physics-tools-style Poincare dictionaries."""

    if save_html:
        raise NotImplementedError("HTML export is not supported by pyna.plot.poincare")
    _ = html_file
    save_path = png_file if save_png else None
    return plot_poincare(
        poincare_data,
        ax=ax,
        plot_d1b=plot_d1b,
        d1b_file=d1b_file,
        xlim=xlim,
        ylim=ylim,
        point_size=s,
        save_path=save_path,
        return_data=True,
        **kwargs,
    )


__all__ = ["plot_poincare", "plot_poincare_data"]
