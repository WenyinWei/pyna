"""Poloidal-section field plotting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class StreamField:
    """Arrays prepared for ``matplotlib.axes.Axes.streamplot``."""

    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    magnitude: np.ndarray
    mask: np.ndarray | None = None


def _as_1d_axis(values: np.ndarray, axis: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim != 2:
        raise ValueError("R and Z must be 1D axes or 2D mesh grids")
    return arr[0, :] if axis == 0 else arr[:, 0]


def stream_field_from_components(
    R,
    Z,
    BR,
    BZ,
    *,
    mask=None,
) -> StreamField:
    """Normalize poloidal field arrays for Matplotlib streamplot.

    ``R`` and ``Z`` may be 1D axes or 2D mesh grids. ``BR``/``BZ`` may be
    stored either as ``[R, Z]`` or as streamplot-native ``[Z, R]``; the shape is
    inferred from the axis lengths.
    """

    x = _as_1d_axis(np.asarray(R, dtype=float), axis=0)
    y = _as_1d_axis(np.asarray(Z, dtype=float), axis=1)
    br = np.asarray(BR, dtype=float)
    bz = np.asarray(BZ, dtype=float)
    if br.shape != bz.shape:
        raise ValueError("BR and BZ must have the same shape")

    native_shape = (y.size, x.size)
    rz_shape = (x.size, y.size)
    if br.shape == native_shape:
        u, v = br, bz
        stream_mask = None if mask is None else np.asarray(mask, dtype=bool)
    elif br.shape == rz_shape:
        u, v = br.T, bz.T
        stream_mask = None if mask is None else np.asarray(mask, dtype=bool).T
    else:
        raise ValueError(
            "BR/BZ shape must be either (len(Z), len(R)) or (len(R), len(Z)); "
            f"got {br.shape}, expected {native_shape} or {rz_shape}"
        )
    mag = np.hypot(u, v)
    return StreamField(x=x, y=y, u=u, v=v, magnitude=mag, mask=stream_mask)


def robust_field_max(magnitude, *, percentile: float = 99.5) -> float:
    """Return a robust positive scale for field-strength styling."""

    values = np.asarray(magnitude, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    return max(float(np.nanpercentile(values, percentile)), 1e-30)


def poloidal_field_section(
    R=None,
    Z=None,
    BR=None,
    BZ=None,
    *,
    fig=None,
    ax=None,
    mask=None,
    cmap: str = "turbo",
    norm=None,
    field_max: float | None = None,
    percentile: float = 99.5,
    linewidth_max: float = 1.55,
    linewidth_min: float = 0.16,
    density: float | tuple[float, float] = 1.35,
    strength_density: bool = True,
    density_levels: Iterable[float] = (8.0, 35.0, 68.0),
    density_values: Iterable[float] = (0.55, 0.85, 1.2),
    arrowsize: float = 0.62,
    colorbar: bool = True,
    cax=None,
    cbar_label: str = r"$|B_{\mathrm{pol}}|$",
    stream_kw: dict | None = None,
    contour=None,
    contour_kw: dict | None = None,
):
    """Draw a poloidal field using streamlines only.

    The local field strength is encoded by streamline color and linewidth. If
    ``strength_density`` is true, progressively stronger bands are overlaid so
    high-field regions also receive denser streamlines.
    """

    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5.0, 6.4))
    elif fig is None:
        fig = ax.figure
    elif ax is None:
        ax = fig.add_subplot(111)

    field = stream_field_from_components(R, Z, BR, BZ, mask=mask)
    valid = np.isfinite(field.u) & np.isfinite(field.v) & np.isfinite(field.magnitude)
    if field.mask is not None:
        valid &= field.mask
    sample = field.magnitude[valid]
    if sample.size == 0:
        ax.set_aspect("equal")
        return fig, ax, None

    scale = robust_field_max(sample, percentile=percentile) if field_max is None else max(float(field_max), 1e-30)
    norm = Normalize(vmin=0.0, vmax=scale) if norm is None else norm
    line_width = linewidth_min + (linewidth_max - linewidth_min) * np.clip(field.magnitude / scale, 0.0, 1.0)
    line_color = np.clip(field.magnitude, 0.0, scale)
    base_mask = ~valid
    base_kw = {
        "cmap": cmap,
        "norm": norm,
        "arrowsize": arrowsize,
    }
    if stream_kw:
        base_kw.update(stream_kw)

    streams = []
    if strength_density:
        for level, layer_density in zip(density_levels, density_values):
            cutoff = float(np.nanpercentile(sample, level))
            layer_mask = base_mask | (field.magnitude < cutoff)
            streams.append(
                ax.streamplot(
                    field.x,
                    field.y,
                    np.ma.array(field.u, mask=layer_mask),
                    np.ma.array(field.v, mask=layer_mask),
                    color=np.ma.array(line_color, mask=layer_mask),
                    linewidth=np.ma.array(line_width, mask=layer_mask),
                    density=layer_density,
                    **base_kw,
                )
            )
    else:
        streams.append(
            ax.streamplot(
                field.x,
                field.y,
                np.ma.array(field.u, mask=base_mask),
                np.ma.array(field.v, mask=base_mask),
                color=np.ma.array(line_color, mask=base_mask),
                linewidth=np.ma.array(line_width, mask=base_mask),
                density=density,
                **base_kw,
            )
        )

    if contour is not None:
        contour_data = np.asarray(contour)
        contour_kw = {"levels": [0.5], "colors": "black", "linewidths": 0.8, **(contour_kw or {})}
        if contour_data.shape == (field.x.size, field.y.size):
            contour_data = contour_data.T
        ax.contour(field.x, field.y, contour_data, **contour_kw)

    cbar = None
    if colorbar:
        if cax is None:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes

            cax = inset_axes(ax, width="5%", height="28%", loc="lower right", borderpad=0.038)
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, cax=cax, extend="max")
        cbar.set_label(cbar_label)

    ax.set_aspect("equal")
    return fig, ax, cbar


def B_pol_section(*args, nocbar: bool = False, strm_kwarg: dict | None = None, **kwargs):
    """Backward-compatible wrapper for the legacy MHDpy helper name."""

    if strm_kwarg:
        kwargs.setdefault("stream_kw", {}).update(strm_kwarg)
    kwargs["colorbar"] = not nocbar
    fig, ax, cbar = poloidal_field_section(*args, **kwargs)
    return fig, ax, cbar.ax if cbar is not None else None


def s_isolines(R, Z, psi_norm, *, fig=None, ax=None, levels=None, **contour_kw):
    """Draw normalized-flux isolines on a poloidal section."""

    import matplotlib.pyplot as plt

    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5.0, 6.4))
    elif fig is None:
        fig = ax.figure
    elif ax is None:
        ax = fig.add_subplot(111)
    x = _as_1d_axis(np.asarray(R, dtype=float), axis=0)
    y = _as_1d_axis(np.asarray(Z, dtype=float), axis=1)
    data = np.asarray(psi_norm, dtype=float)
    if data.shape == (x.size, y.size):
        data = data.T
    if levels is None:
        levels = np.arange(1, 8) / 5.0
    contour = ax.contour(x, y, data, levels=levels, **contour_kw)
    ax.clabel(contour, inline=True, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$R$ [m]")
    ax.set_ylabel(r"$Z$ [m]")
    return fig, ax, contour


__all__ = [
    "StreamField",
    "stream_field_from_components",
    "robust_field_max",
    "poloidal_field_section",
    "B_pol_section",
    "s_isolines",
]
