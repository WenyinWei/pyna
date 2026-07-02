"""Visual helpers for classical magnetic-spectrum island-chain analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from pyna.toroidal._periodic_grid import TWOPI, prepare_surface_arrays
from pyna.toroidal.perturbation_spectrum import (
    ChirikovOverlap,
    RadialPerturbationFourierSpectrum,
    ResonantIslandChain,
)


@dataclass(frozen=True)
class SectionIslandBar:
    """Geometry of one plotted island-width bar on a poloidal section."""

    chain: ResonantIslandChain
    branch: int
    theta_O: float
    theta_X: float
    R_O: float
    Z_O: float
    R_X: float
    Z_X: float
    R_inner: float
    Z_inner: float
    R_outer: float
    Z_outer: float


@dataclass(frozen=True)
class SpectrumSurfaceMatrix:
    """Packed ``(m, n)`` spectrum expanded to a rectangular plotting matrix."""

    m_values: np.ndarray
    n_values: np.ndarray
    coefficient: np.ndarray
    radial_index: int
    radial_label: float | None = None

    @property
    def amplitude(self) -> np.ndarray:
        return np.abs(self.coefficient)

    @property
    def phase(self) -> np.ndarray:
        return np.angle(self.coefficient)


@dataclass(frozen=True)
class RadialModeSpectrum:
    """Radial stack for a fixed physical ``n`` or fixed physical ``m`` family."""

    fixed_axis: str
    fixed_value: int
    mode_axis: str
    mode_values: np.ndarray
    radial_labels: np.ndarray
    coefficient: np.ndarray
    fourier_m: np.ndarray
    fourier_n: np.ndarray

    @property
    def amplitude(self) -> np.ndarray:
        return np.abs(self.coefficient)

    @property
    def phase(self) -> np.ndarray:
        return np.angle(self.coefficient)


@dataclass(frozen=True)
class RationalSurfaceMarker:
    """One low-order rational surface ``q(s)=m/n`` in a radial profile."""

    m: int
    n: int
    radial_label: float
    q: float

    @property
    def label(self) -> str:
        return f"{self.m}/{self.n}"


@dataclass(frozen=True)
class PoincareRationalTrace:
    """Poincare crossings projected to the ``m/n``-or-``q`` versus radius plane."""

    ratio: np.ndarray
    radial_label: np.ndarray
    value: np.ndarray | None = None
    label: str = "Poincare trace"


def _surface_radial_index(
    spectrum: RadialPerturbationFourierSpectrum,
    radial_index: int | None,
) -> tuple[int, float | None]:
    if spectrum.dBr.ndim == 2:
        idx = spectrum.dBr.shape[0] // 2 if radial_index is None else int(radial_index)
        if idx < 0 or idx >= spectrum.dBr.shape[0]:
            raise IndexError("radial_index is out of range for the spectrum radial stack")
        label = None if spectrum.radial_labels is None else float(spectrum.radial_labels[idx])
        return idx, label
    return 0, None


def _default_m_values(
    spectrum: RadialPerturbationFourierSpectrum,
    *,
    m_max: int | None,
    positive_only: bool = True,
) -> np.ndarray:
    m_lim = int(np.max(np.abs(spectrum.m)) if m_max is None else m_max)
    if positive_only:
        return np.arange(1, m_lim + 1, dtype=int)
    return np.arange(-m_lim, m_lim + 1, dtype=int)


def _default_n_values(
    spectrum: RadialPerturbationFourierSpectrum,
    *,
    n_max: int | None,
) -> np.ndarray:
    n_lim = int(np.max(np.abs(spectrum.n)) if n_max is None else n_max)
    return np.arange(-n_lim, n_lim + 1, dtype=int)


def _mode_coeff_matrix(
    spectrum: RadialPerturbationFourierSpectrum,
    *,
    radial_index: int,
    m_values: Sequence[int],
    n_values: Sequence[int],
) -> np.ndarray:
    data = np.zeros((len(m_values), len(n_values)), dtype=complex)
    for i, m_val in enumerate(m_values):
        for j, n_val in enumerate(n_values):
            idx = spectrum.mode_index(int(m_val), int(n_val))
            if idx is None:
                continue
            coeff = spectrum.dBr[idx] if spectrum.dBr.ndim == 1 else spectrum.dBr[int(radial_index), idx]
            data[i, j] = coeff
    return data


def _mode_matrix(
    spectrum: RadialPerturbationFourierSpectrum,
    *,
    radial_index: int,
    m_values: Sequence[int],
    n_values: Sequence[int],
) -> np.ndarray:
    return np.abs(
        _mode_coeff_matrix(
            spectrum,
            radial_index=radial_index,
            m_values=m_values,
            n_values=n_values,
        )
    )


def _scale_name(log_scale: bool, amplitude_scale: str | None) -> str:
    if amplitude_scale is None:
        return "log10" if log_scale else "linear"
    key = str(amplitude_scale).lower()
    aliases = {
        "log": "log10",
        "log10": "log10",
        "linear": "linear",
        "sqrt": "sqrt",
        "asinh": "asinh",
    }
    if key not in aliases:
        raise ValueError("amplitude_scale must be 'linear', 'sqrt', 'asinh', or 'log10'")
    return aliases[key]


def _plot_values(
    amplitude: np.ndarray,
    *,
    log_scale: bool,
    amplitude_scale: str | None = None,
    floor: float = 1.0e-300,
    mask_zeros: bool = False,
    asinh_linear_width: float | None = None,
) -> tuple[np.ndarray, str]:
    amps = np.asarray(amplitude, dtype=float)
    scale = _scale_name(log_scale, amplitude_scale)
    if scale == "log10":
        values = np.log10(amps + float(floor))
        label = r"$\log_{10}|\tilde{b}^{1}_{mn}|$"
    elif scale == "linear":
        values = amps
        label = r"$|\tilde{b}^{1}_{mn}|$"
    elif scale == "sqrt":
        values = np.sqrt(np.maximum(amps, 0.0))
        label = r"$\sqrt{|\tilde{b}^{1}_{mn}|}$"
    else:
        positive = amps[np.isfinite(amps) & (amps > float(floor))]
        if asinh_linear_width is None:
            if positive.size:
                width = max(float(np.nanpercentile(positive, 25.0)), float(np.nanmax(positive)) / 100.0)
            else:
                width = 1.0
        else:
            width = float(asinh_linear_width)
        width = max(width, float(floor))
        values = np.arcsinh(amps / width)
        label = rf"$\operatorname{{asinh}}(|\tilde{{b}}^{{1}}_{{mn}}|/{width:.2e})$"
    if mask_zeros:
        values = np.where(amps > float(floor), values, np.nan)
    return values, label


def _plain_plot_label(log_scale: bool, amplitude_scale: str | None) -> str:
    scale = _scale_name(log_scale, amplitude_scale)
    if scale == "log10":
        return "log10 |b~1_mn|"
    if scale == "sqrt":
        return "sqrt |b~1_mn|"
    if scale == "asinh":
        return "asinh |b~1_mn|"
    return "|b~1_mn|"


def _plot_limits(values: np.ndarray, *, log_scale: bool, amplitude_scale: str | None) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    vmax = float(np.max(finite)) if finite.size else 0.0
    scale = _scale_name(log_scale, amplitude_scale)
    if scale == "log10":
        return vmax - 8.0, vmax
    return 0.0, vmax


def _cmap_with_bad(
    plt,
    cmap: str,
    bad_color: str | None,
    *,
    low_color: str | None = None,
    low_fraction: float = 0.18,
):
    base = plt.get_cmap(cmap)
    if low_color is None:
        cmap_obj = base.copy()
    else:
        from matplotlib.colors import ListedColormap, to_rgba

        colors = base(np.linspace(0.0, 1.0, 256))
        anchor = int(np.clip(round(float(low_fraction) * 255), 1, 255))
        start = np.asarray(to_rgba(low_color), dtype=float)
        stop = np.asarray(colors[anchor], dtype=float)
        for idx in range(anchor + 1):
            weight = idx / anchor
            colors[idx] = (1.0 - weight) * start + weight * stop
        cmap_obj = ListedColormap(colors, name=f"{base.name}_low_{low_color.lstrip('#')}")
    if bad_color is not None:
        cmap_obj.set_bad(bad_color)
    return cmap_obj


def _low_color_for_scale(log_scale: bool, amplitude_scale: str | None, zero_color: str | None) -> str | None:
    if zero_color is None:
        return None
    return None if _scale_name(log_scale, amplitude_scale) == "log10" else zero_color


def _plotly_colorscale_from_matplotlib(cmap: str, *, low_color: str | None = None) -> list[list[float | str]]:
    if cmap.lower() in {"magnetic", "magnetic_bar", "spectrum_bar"}:
        return [
            [0.00, "#ffffff"],
            [0.10, "#eef2ff"],
            [0.24, "#a5b4fc"],
            [0.40, "#38bdf8"],
            [0.58, "#2dd4bf"],
            [0.74, "#a3e635"],
            [0.90, "#facc15"],
            [1.00, "#f97316"],
        ]

    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    cmap_obj = _cmap_with_bad(plt, cmap, None, low_color=low_color)
    stops = np.linspace(0.0, 1.0, 16)
    return [[float(stop), to_hex(cmap_obj(float(stop)))] for stop in stops]


def _edges_from_centers(values: Sequence[float]) -> np.ndarray:
    centers = np.asarray(values, dtype=float)
    if centers.ndim != 1 or centers.size == 0:
        raise ValueError("centers must be a non-empty one-dimensional array")
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5], dtype=float)
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def _validated_profile(radial_labels: Sequence[float], q_profile: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    radial = np.asarray(radial_labels, dtype=float)
    q_arr = np.asarray(q_profile, dtype=float)
    if radial.ndim != 1 or q_arr.ndim != 1 or radial.shape != q_arr.shape:
        raise ValueError("radial_labels and q_profile must be one-dimensional arrays with the same shape")
    if radial.size < 2:
        raise ValueError("at least two radial samples are required")
    if not np.all(np.isfinite(radial)) or not np.all(np.isfinite(q_arr)):
        raise ValueError("radial_labels and q_profile must be finite")
    if np.any(np.diff(radial) <= 0.0):
        raise ValueError("radial_labels must be strictly increasing")
    return radial, q_arr


def _profile_crossings(radial: np.ndarray, values: np.ndarray, target: float) -> list[float]:
    roots: list[float] = []
    diff = np.asarray(values, dtype=float) - float(target)
    for i in range(radial.size - 1):
        f0 = diff[i]
        f1 = diff[i + 1]
        if not np.isfinite(f0) or not np.isfinite(f1):
            continue
        if f0 == 0.0:
            roots.append(float(radial[i]))
        if f0 * f1 < 0.0:
            t = -f0 / (f1 - f0)
            roots.append(float(radial[i] + t * (radial[i + 1] - radial[i])))
    if diff[-1] == 0.0:
        roots.append(float(radial[-1]))

    deduped: list[float] = []
    for root in sorted(roots):
        if not deduped or abs(root - deduped[-1]) > 1.0e-12:
            deduped.append(root)
    return deduped


def _m_values_for_rationals(
    m_values: Sequence[int] | dict[int, Sequence[int]] | None,
    *,
    n: int,
    q_profile: np.ndarray,
) -> list[int]:
    if isinstance(m_values, dict):
        raw = m_values.get(int(n), ())
        return sorted({int(m) for m in raw if int(m) > 0})
    if m_values is not None:
        return sorted({int(m) for m in m_values if int(m) > 0})
    q_min = float(np.nanmin(q_profile))
    q_max = float(np.nanmax(q_profile))
    lo = int(np.floor(min(q_min, q_max) * int(n))) - 1
    hi = int(np.ceil(max(q_min, q_max) * int(n))) + 1
    return list(range(max(1, lo), max(1, hi) + 1))


def _apply_line_halo(artist, *, color: str = "white", linewidth: float = 2.8, alpha: float = 0.9):
    try:
        import matplotlib.patheffects as pe
    except Exception:  # pragma: no cover - path effects are normally present with matplotlib
        return artist
    artist.set_path_effects([pe.Stroke(linewidth=float(linewidth), foreground=color, alpha=float(alpha)), pe.Normal()])
    return artist


def rational_surface_markers(
    radial_labels: Sequence[float],
    q_profile: Sequence[float],
    *,
    n_values: Sequence[int] = (1, 2, 3),
    m_values: Sequence[int] | dict[int, Sequence[int]] | None = None,
) -> list[RationalSurfaceMarker]:
    """Find rational surfaces ``q(s)=m/n`` for plotting overlays.

    ``n_values`` are positive physical toroidal mode numbers.  If ``m_values``
    is omitted, the function scans the positive ``m`` range covered by the
    supplied q-profile for each ``n``.
    """

    radial, q_arr = _validated_profile(radial_labels, q_profile)
    markers: list[RationalSurfaceMarker] = []
    for n_int in sorted({int(n) for n in n_values if int(n) > 0}):
        for m_int in _m_values_for_rationals(m_values, n=n_int, q_profile=q_arr):
            q_target = float(m_int) / float(n_int)
            for s_res in _profile_crossings(radial, q_arr, q_target):
                markers.append(
                    RationalSurfaceMarker(
                        m=m_int,
                        n=n_int,
                        radial_label=float(s_res),
                        q=float(q_target),
                    )
                )
    markers.sort(key=lambda marker: (marker.radial_label, marker.n, marker.m))
    return markers


def overlay_surface_resonance_line(
    ax,
    q_value: float,
    *,
    resonant_sign: int = -1,
    color: str = "#111827",
    linewidth: float = 1.45,
    linestyle: str = "--",
    alpha: float = 0.96,
    label: str | None = None,
    zorder: float = 7.0,
    halo: bool = True,
):
    """Overlay the resonant branch on a surface ``(Fourier n, m)`` spectrum.

    The spectrum convention is ``exp(i(m theta + n_F phi))``.  For the usual
    RMP coefficient ``b_{m,-n}`` and physical ``q=m/n``, the default
    ``resonant_sign=-1`` draws ``m=-q n_F``.
    """

    q = float(q_value)
    if not np.isfinite(q):
        raise ValueError("q_value must be finite")
    sign = -1 if int(np.sign(resonant_sign)) < 0 else 1
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    n_line = np.linspace(min(x0, x1), max(x0, x1), 512)
    m_line = sign * q * n_line
    lo, hi = min(y0, y1), max(y0, y1)
    mask = np.isfinite(m_line) & (m_line >= lo) & (m_line <= hi)
    if not np.any(mask):
        return None
    line_label = label if label is not None else rf"resonant branch $q={q:.3g}$"
    (line,) = ax.plot(
        n_line[mask],
        m_line[mask],
        color=color,
        lw=float(linewidth),
        ls=linestyle,
        alpha=float(alpha),
        label=line_label,
        zorder=zorder,
    )
    if halo:
        _apply_line_halo(line, linewidth=float(linewidth) + 2.2, alpha=0.82)
    return line


def spectrum_surface_matrix(
    spectrum: RadialPerturbationFourierSpectrum,
    *,
    radial_index: int | None = None,
    m_values: Sequence[int] | None = None,
    n_values: Sequence[int] | None = None,
    m_max: int | None = None,
    n_max: int | None = None,
    positive_m: bool = True,
) -> SpectrumSurfaceMatrix:
    """Return a rectangular ``(m, n)`` matrix for one radial surface.

    Missing packed modes are filled with zero coefficients.  By default the
    matrix follows the magnetic-confinement convention ``m > 0`` and includes
    both signs of Fourier ``n``.
    """

    ridx, label = _surface_radial_index(spectrum, radial_index)
    m_arr = np.asarray(m_values if m_values is not None else _default_m_values(spectrum, m_max=m_max, positive_only=positive_m), dtype=int)
    n_arr = np.asarray(n_values if n_values is not None else _default_n_values(spectrum, n_max=n_max), dtype=int)
    coeff = _mode_coeff_matrix(spectrum, radial_index=ridx, m_values=m_arr, n_values=n_arr)
    return SpectrumSurfaceMatrix(
        m_values=m_arr,
        n_values=n_arr,
        coefficient=coeff,
        radial_index=ridx,
        radial_label=label,
    )


def plot_spectrum_heatmap(
    spectrum: RadialPerturbationFourierSpectrum,
    *,
    radial_index: int | None = None,
    m_max: int | None = None,
    n_max: int | None = None,
    m_values: Sequence[int] | None = None,
    n_values: Sequence[int] | None = None,
    chains: Iterable[ResonantIslandChain] = (),
    q_value: float | None = None,
    resonant_sign: int = -1,
    show_resonance_line: bool = True,
    resonance_line_kwargs: dict | None = None,
    show_island_boxes: bool = True,
    annotate_islands: bool = True,
    log_scale: bool = True,
    amplitude_scale: str | None = None,
    mask_zeros: bool = False,
    zero_color: str = "#ffffff",
    renderer: str = "pcolormesh",
    ax=None,
    cmap: str = "magma",
    title: str | None = None,
):
    """Plot ``|tilde_b^1_{mn}|`` for one radial surface.

    ``renderer="pcolormesh"`` is the default because it also handles nonuniform
    axes and matches radial profile maps.  ``renderer="imshow"`` remains
    available for crisp integer-grid heatmaps.  ``amplitude_scale="sqrt"`` or
    ``"asinh"`` is often more readable than ``"log10"`` for sparse spectra.
    """

    import matplotlib.pyplot as plt

    matrix = spectrum_surface_matrix(
        spectrum,
        radial_index=radial_index,
        m_values=m_values,
        n_values=n_values,
        m_max=m_max,
        n_max=n_max,
    )
    plot_data, label = _plot_values(
        matrix.amplitude,
        log_scale=log_scale,
        amplitude_scale=amplitude_scale,
        mask_zeros=mask_zeros,
    )
    plot_cmap = _cmap_with_bad(
        plt,
        cmap,
        zero_color if mask_zeros else None,
        low_color=_low_color_for_scale(log_scale, amplitude_scale, zero_color),
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6.0, 0.42 * matrix.n_values.size), max(4.0, 0.26 * matrix.m_values.size)))
    else:
        fig = ax.figure

    vmin, vmax = _plot_limits(plot_data, log_scale=log_scale, amplitude_scale=amplitude_scale)
    renderer_key = renderer.lower()
    if renderer_key in {"imshow", "heatmap", "image"}:
        im = ax.imshow(
            plot_data,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[
                matrix.n_values[0] - 0.5,
                matrix.n_values[-1] + 0.5,
                matrix.m_values[0] - 0.5,
                matrix.m_values[-1] + 0.5,
            ],
            cmap=plot_cmap,
            vmin=vmin,
            vmax=vmax,
        )
    elif renderer_key in {"pcolormesh", "mesh"}:
        im = ax.pcolormesh(
            _edges_from_centers(matrix.n_values),
            _edges_from_centers(matrix.m_values),
            plot_data,
            shading="auto",
            cmap=plot_cmap,
            vmin=vmin,
            vmax=vmax,
        )
    else:
        raise ValueError("renderer must be 'imshow' or 'pcolormesh'")
    fig.colorbar(im, ax=ax, pad=0.02, label=label)
    ax.axvline(0.0, color="white", lw=0.7, alpha=0.65)
    ax.set_xlabel("n")
    ax.set_ylabel("m")
    if title is None:
        title = f"Magnetic perturbation spectrum at radial index {matrix.radial_index}"
    if matrix.radial_label is not None:
        title += f"  s={matrix.radial_label:.4g}"
    ax.set_title(title)

    resonance_line = None
    if q_value is not None and show_resonance_line:
        kwargs = {} if resonance_line_kwargs is None else dict(resonance_line_kwargs)
        resonance_line = overlay_surface_resonance_line(ax, q_value, resonant_sign=resonant_sign, **kwargs)

    if show_island_boxes:
        m_set = set(matrix.m_values.tolist())
        n_set = set(matrix.n_values.tolist())
        for chain in chains:
            n_plot = resonant_sign * chain.n
            if chain.m not in m_set or n_plot not in n_set:
                continue
            rect = plt.Rectangle(
                (n_plot - 0.5, chain.m - 0.5),
                1.0,
                1.0,
                edgecolor="#2dd4bf",
                facecolor="none",
                linewidth=1.8,
                zorder=5,
            )
            ax.add_patch(rect)
            if annotate_islands:
                text = ax.text(
                    n_plot,
                    chain.m,
                    f"{chain.m}/{chain.n}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#111827",
                    zorder=6,
                )
                _apply_line_halo(text, linewidth=2.4, alpha=0.88)
    if resonance_line is not None:
        ax.legend(loc="best", fontsize=8, frameon=True, framealpha=0.86)
    return fig, ax


def plot_spectrum_pcolormesh(*args, **kwargs):
    """Convenience wrapper around :func:`plot_spectrum_heatmap` using pcolormesh."""

    kwargs["renderer"] = "pcolormesh"
    return plot_spectrum_heatmap(*args, **kwargs)


def plot_spectrum_bar3d(
    spectrum: RadialPerturbationFourierSpectrum,
    *,
    radial_index: int | None = None,
    m_max: int | None = None,
    n_max: int | None = None,
    m_values: Sequence[int] | None = None,
    n_values: Sequence[int] | None = None,
    log_scale: bool = True,
    amplitude_scale: str | None = None,
    mask_zeros: bool = True,
    zero_color: str = "#ffffff",
    bar_width: float = 0.82,
    range_mode: str = "requested",
    range_padding: float = 0.75,
    z_aspect: float = 0.58,
    show_edges: bool = True,
    edge_color: str = "rgba(15, 23, 42, 0.42)",
    edge_width: float = 2.2,
    ax=None,
    cmap: str = "magnetic",
    title: str | None = None,
):
    """Plot one radial surface of ``|tilde_b^1_{mn}|`` as an interactive Plotly 3-D bar chart."""

    if ax is not None:
        raise ValueError("plot_spectrum_bar3d now returns a Plotly figure and does not accept a Matplotlib ax")
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - exercised only without optional plotting deps
        raise ImportError("plot_spectrum_bar3d requires plotly; install pyna-chaos with plotly support") from exc

    matrix = spectrum_surface_matrix(
        spectrum,
        radial_index=radial_index,
        m_values=m_values,
        n_values=n_values,
        m_max=m_max,
        n_max=n_max,
    )
    plot_data, z_label = _plot_values(
        matrix.amplitude,
        log_scale=log_scale,
        amplitude_scale=amplitude_scale,
        mask_zeros=mask_zeros,
    )
    z_label_text = _plain_plot_label(log_scale, amplitude_scale)
    vmin, vmax = _plot_limits(plot_data, log_scale=log_scale, amplitude_scale=amplitude_scale)
    values = np.clip(plot_data, vmin, vmax)
    heights = np.maximum(values - vmin, 0.0) if _scale_name(log_scale, amplitude_scale) == "log10" else np.maximum(values, 0.0)

    nn, mm = np.meshgrid(matrix.n_values, matrix.m_values)
    centers_n = nn.ravel()
    centers_m = mm.ravel()
    raw_amplitude = matrix.amplitude.ravel()
    flat_values = values.ravel()
    flat_heights = heights.ravel()
    valid = np.isfinite(flat_values) & np.isfinite(flat_heights)
    if mask_zeros:
        valid &= raw_amplitude > 0.0
    valid &= flat_heights > 0.0

    requested_x_range = [float(matrix.n_values[0]) - 0.5, float(matrix.n_values[-1]) + 0.5]
    requested_y_range = [float(matrix.m_values[0]) - 0.5, float(matrix.m_values[-1]) + 0.5]
    range_key = str(range_mode).lower()
    if range_key in {"requested", "full"} or not np.any(valid):
        x_range = requested_x_range
        y_range = requested_y_range
    elif range_key in {"nonzero", "data", "occupied"}:
        pad = max(float(range_padding), 0.5 * float(bar_width))
        x_range = [float(np.nanmin(centers_n[valid])) - pad, float(np.nanmax(centers_n[valid])) + pad]
        y_range = [float(np.nanmin(centers_m[valid])) - pad, float(np.nanmax(centers_m[valid])) + pad]
    else:
        raise ValueError("range_mode must be 'requested' or 'nonzero'")
    x_span = max(x_range[1] - x_range[0], 1.0)
    y_span = max(y_range[1] - y_range[0], 1.0)
    xy_span = max(x_span, y_span)
    aspectratio = {
        "x": x_span / xy_span,
        "y": y_span / xy_span,
        "z": float(z_aspect),
    }

    dx = dy = float(bar_width)
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    ii: list[int] = []
    jj: list[int] = []
    kk: list[int] = []
    intensity: list[float] = []
    hover_text: list[str] = []
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    faces = (
        (0, 1, 2),
        (0, 2, 3),
        (4, 6, 5),
        (4, 7, 6),
        (0, 4, 5),
        (0, 5, 1),
        (1, 5, 6),
        (1, 6, 2),
        (2, 6, 7),
        (2, 7, 3),
        (3, 7, 4),
        (3, 4, 0),
    )
    edge_pairs = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    for n0, m0, amp, value, height in zip(centers_n[valid], centers_m[valid], raw_amplitude[valid], flat_values[valid], flat_heights[valid]):
        x0 = float(n0) - 0.5 * dx
        x1 = float(n0) + 0.5 * dx
        y0 = float(m0) - 0.5 * dy
        y1 = float(m0) + 0.5 * dy
        z1 = float(height)
        base = len(xs)
        vertices = (
            (x0, y0, 0.0),
            (x1, y0, 0.0),
            (x1, y1, 0.0),
            (x0, y1, 0.0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        )
        text = f"m={int(m0)}<br>n={int(n0)}<br>|b|={float(amp):.4e}<br>{z_label_text}={float(value):.4e}"
        for vertex_i, (x, y, z) in enumerate(vertices):
            xs.append(x)
            ys.append(y)
            zs.append(z)
            intensity.append(float(vmin if vertex_i < 4 else value))
            hover_text.append(text)
        for a, b, c in faces:
            ii.append(base + a)
            jj.append(base + b)
            kk.append(base + c)
        if show_edges:
            for a, b in edge_pairs:
                edge_x.extend([vertices[a][0], vertices[b][0], None])
                edge_y.extend([vertices[a][1], vertices[b][1], None])
                edge_z.extend([vertices[a][2], vertices[b][2], None])

    colorscale = _plotly_colorscale_from_matplotlib(
        cmap,
        low_color=_low_color_for_scale(log_scale, amplitude_scale, zero_color),
    )
    if title is None:
        title = f"3-D magnetic spectrum at radial index {matrix.radial_index}"
    if matrix.radial_label is not None:
        title += f"  s={matrix.radial_label:.4g}"

    fig = go.Figure()
    if xs:
        fig.add_trace(
            go.Mesh3d(
                x=xs,
                y=ys,
                z=zs,
                i=ii,
                j=jj,
                k=kk,
                intensity=intensity,
                colorscale=colorscale,
                cmin=vmin,
                cmax=vmax,
                colorbar={"title": {"text": z_label_text}, "len": 0.68, "thickness": 18, "x": 0.86},
                flatshading=False,
                lighting={"ambient": 0.7, "diffuse": 0.64, "specular": 0.14, "roughness": 0.82},
                text=hover_text,
                hoverinfo="text",
                name="spectrum bars",
            )
        )
    if show_edges and edge_x:
        fig.add_trace(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line={"color": edge_color, "width": float(edge_width)},
                hoverinfo="skip",
                showlegend=False,
                name="bar edges",
            )
        )
    fig.update_layout(
        title=title,
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        paper_bgcolor="white",
        plot_bgcolor="white",
        scene={
            "xaxis": {
                "title": "n",
                "range": x_range,
                "backgroundcolor": "white",
                "gridcolor": "#e5e7eb",
                "zerolinecolor": "#9ca3af",
            },
            "yaxis": {
                "title": "m",
                "range": y_range,
                "backgroundcolor": "white",
                "gridcolor": "#e5e7eb",
                "zerolinecolor": "#9ca3af",
            },
            "zaxis": {
                "title": z_label_text,
                "range": [0.0, float(np.nanmax(flat_heights[valid]) * 1.12 if np.any(valid) else 1.0)],
                "backgroundcolor": "white",
                "gridcolor": "#e5e7eb",
                "zerolinecolor": "#9ca3af",
            },
            "camera": {
                "eye": {"x": 1.45, "y": -1.55, "z": 0.9},
                "projection": {"type": "orthographic"},
            },
            "aspectmode": "manual",
            "aspectratio": aspectratio,
        },
    )
    return fig


def _positive_mode_values(values: np.ndarray) -> np.ndarray:
    return np.asarray(sorted({abs(int(value)) for value in values if int(value) != 0}), dtype=int)


def radial_mode_spectrum(
    spectrum: RadialPerturbationFourierSpectrum,
    *,
    fixed_n: int | None = None,
    fixed_m: int | None = None,
    mode_values: Sequence[int] | None = None,
    resonant_sign: int = -1,
) -> RadialModeSpectrum:
    """Extract a radial ``mode x surface`` matrix for one physical mode family.

    Pass ``fixed_n`` to inspect signed poloidal rows at the same Fourier
    toroidal index ``resonant_sign*fixed_n``.  Pass ``fixed_m`` to inspect
    signed toroidal rows at the same positive poloidal index.  The two sides of
    a signed axis are actual Fourier rows; they are not forced to be conjugate.
    """

    if (fixed_n is None) == (fixed_m is None):
        raise ValueError("pass exactly one of fixed_n or fixed_m")
    if spectrum.dBr.ndim != 2 or spectrum.radial_labels is None:
        raise ValueError("radial_mode_spectrum requires a radial stack spectrum with radial_labels")
    radial = np.asarray(spectrum.radial_labels, dtype=float)
    sign = -1 if int(np.sign(resonant_sign)) < 0 else 1

    if fixed_n is not None:
        fixed = int(fixed_n)
        if fixed <= 0:
            raise ValueError("fixed_n must be a positive physical toroidal mode number")
        modes = np.asarray(mode_values if mode_values is not None else _positive_mode_values(spectrum.m), dtype=int)
        fourier_m = modes.copy()
        fourier_n = np.full(modes.shape, sign * fixed, dtype=int)
        fixed_axis = "n"
        mode_axis = "m"
    else:
        fixed = int(fixed_m)
        if fixed <= 0:
            raise ValueError("fixed_m must be a positive poloidal mode number")
        modes = np.asarray(mode_values if mode_values is not None else _positive_mode_values(spectrum.n), dtype=int)
        fourier_m = np.full(modes.shape, fixed, dtype=int)
        fourier_n = sign * modes
        fixed_axis = "m"
        mode_axis = "n"

    coeff = np.zeros((radial.size, modes.size), dtype=complex)
    for j, (m_val, n_val) in enumerate(zip(fourier_m, fourier_n)):
        idx = spectrum.mode_index(int(m_val), int(n_val))
        if idx is not None:
            coeff[:, j] = spectrum.dBr[:, idx]

    return RadialModeSpectrum(
        fixed_axis=fixed_axis,
        fixed_value=fixed,
        mode_axis=mode_axis,
        mode_values=modes,
        radial_labels=radial,
        coefficient=coeff,
        fourier_m=fourier_m,
        fourier_n=fourier_n,
    )


def _chains_for_radial_map(
    chains: Iterable[ResonantIslandChain],
    radial_map: RadialModeSpectrum,
) -> list[ResonantIslandChain]:
    if radial_map.fixed_axis == "n":
        return [chain for chain in chains if int(chain.n) == radial_map.fixed_value]
    return [chain for chain in chains if int(chain.m) == radial_map.fixed_value]


def _resonance_curve(radial_map: RadialModeSpectrum, q_profile: np.ndarray | None) -> np.ndarray | None:
    return _radial_resonance_curve(radial_map, q_profile, axis_convention="physical")


def _radial_axis_convention(axis_convention: str) -> str:
    key = str(axis_convention).lower().replace("-", "_")
    aliases = {
        "physical": "physical",
        "physical_n": "physical",
        "n_phys": "physical",
        "fourier": "fourier",
        "fourier_n": "fourier",
        "n_fourier": "fourier",
        "nf": "fourier",
    }
    if key not in aliases:
        raise ValueError("axis_convention must be 'physical' or 'fourier'")
    return aliases[key]


def _radial_axis_values(radial_map: RadialModeSpectrum, axis_convention: str) -> np.ndarray:
    if radial_map.fixed_axis == "m" and _radial_axis_convention(axis_convention) == "fourier":
        return np.asarray(radial_map.fourier_n, dtype=float)
    if radial_map.fixed_axis == "n" and _radial_axis_convention(axis_convention) == "fourier":
        return np.asarray(radial_map.fourier_m, dtype=float)
    return np.asarray(radial_map.mode_values, dtype=float)


def _fixed_fourier_n(radial_map: RadialModeSpectrum) -> int:
    if radial_map.fixed_axis != "n" or radial_map.fourier_n.size == 0:
        raise ValueError("fixed Fourier n is only defined for fixed-n radial maps")
    return int(radial_map.fourier_n[0])


def _radial_axis_label(radial_map: RadialModeSpectrum, axis_convention: str) -> str:
    if radial_map.fixed_axis == "n":
        return "m"
    if _radial_axis_convention(axis_convention) == "fourier":
        return "n"
    return r"physical $n=-n_F$"


def _rational_surface_label(m: int, n: int, *, prefix: str = "q=") -> str:
    m_int = abs(int(m))
    n_int = abs(int(n))
    div = int(np.gcd(m_int, n_int)) if n_int else 1
    m_red = m_int // max(div, 1)
    n_red = n_int // max(div, 1)
    return f"{prefix}{m_red}/{n_red}"


def _signed_q_profile_label(axis: str, coefficient: float) -> str:
    coeff = int(round(float(coefficient)))
    if np.isclose(coefficient, coeff):
        if axis == "m" and coeff == 1:
            factor = ""
        elif axis == "m" and coeff == -1:
            factor = "-"
        else:
            factor = f"{coeff}"
    else:
        factor = f"{float(coefficient):.3g}"
    if axis == "m":
        return rf"$m={factor}q(s)$"
    return rf"$n={factor}/q(s)$"


def _radial_resonance_curve(
    radial_map: RadialModeSpectrum,
    q_profile: np.ndarray | None,
    *,
    axis_convention: str,
) -> np.ndarray | None:
    if q_profile is None:
        return None
    q_arr = np.asarray(q_profile, dtype=float)
    if q_arr.shape != radial_map.radial_labels.shape:
        raise ValueError("q_profile must have the same shape as spectrum.radial_labels")
    if radial_map.fixed_axis == "n":
        return -float(_fixed_fourier_n(radial_map)) * q_arr
    with np.errstate(divide="ignore", invalid="ignore"):
        curve = float(radial_map.fixed_value) / q_arr
    if _radial_axis_convention(axis_convention) == "fourier":
        return -curve
    return curve


def _radial_chain_x(chain: ResonantIslandChain, radial_map: RadialModeSpectrum, axis_convention: str) -> float:
    if radial_map.fixed_axis == "n":
        n_fourier = _fixed_fourier_n(radial_map)
        return -float(np.sign(n_fourier) or 1.0) * float(chain.m)
    if _radial_axis_convention(axis_convention) == "fourier":
        hits = np.where(np.asarray(radial_map.mode_values, dtype=int) == int(chain.n))[0]
        if hits.size:
            return float(radial_map.fourier_n[int(hits[0])])
        return -float(chain.n)
    return float(chain.n)


def overlay_radial_resonance_curve(
    ax,
    radial_map: RadialModeSpectrum,
    q_profile: Sequence[float],
    *,
    axis_convention: str = "physical",
    color: str = "#111827",
    linewidth: float = 1.65,
    linestyle: str = "-",
    alpha: float = 0.96,
    label: str | None = None,
    halo: bool = True,
    zorder: float = 6.0,
):
    """Overlay the q-profile resonance curve on a fixed-``n`` or fixed-``m`` map."""

    curve = _radial_resonance_curve(radial_map, np.asarray(q_profile, dtype=float), axis_convention=axis_convention)
    if curve is None:
        return None
    line_label = label
    if line_label is None:
        if radial_map.fixed_axis == "n":
            n_fourier = _fixed_fourier_n(radial_map)
            line_label = _signed_q_profile_label("m", -float(n_fourier))
        elif _radial_axis_convention(axis_convention) == "fourier":
            line_label = _signed_q_profile_label("n", -float(radial_map.fixed_value))
        else:
            line_label = _signed_q_profile_label("n", float(radial_map.fixed_value))
    (line,) = ax.plot(
        curve,
        radial_map.radial_labels,
        color=color,
        lw=float(linewidth),
        ls=linestyle,
        alpha=float(alpha),
        label=line_label,
        zorder=zorder,
    )
    if halo:
        _apply_line_halo(line, linewidth=float(linewidth) + 2.2, alpha=0.86)
    return line


def overlay_radial_mode_island_bars(
    ax,
    chains: Iterable[ResonantIslandChain],
    radial_map: RadialModeSpectrum,
    *,
    axis_convention: str = "physical",
    max_island_bars: int | None = None,
    annotate: bool = True,
    color: str = "#f59e0b",
    outline_color: str = "#111827",
    zorder: float = 8.0,
):
    """Overlay Nardon island-width bars on a fixed-``n`` or fixed-``m`` radial map."""

    island_chains = sorted(_chains_for_radial_map(chains, radial_map), key=lambda chain: chain.b_res, reverse=True)
    if max_island_bars is not None:
        island_chains = island_chains[: int(max_island_bars)]
    max_b = max((chain.b_res for chain in island_chains), default=0.0)
    artists = []
    for chain in island_chains:
        y0 = float(chain.radial_label)
        dy = float(chain.half_width)
        weight = 0.0 if max_b <= 0.0 else float(np.sqrt(max(chain.b_res, 0.0) / max_b))
        lw = 1.45 + 2.5 * weight
        x = _radial_chain_x(chain, radial_map, axis_convention)
        (bar,) = ax.plot([x, x], [y0 - dy, y0 + dy], color=color, lw=lw, solid_capstyle="round", zorder=zorder)
        _apply_line_halo(bar, color=outline_color, linewidth=lw + 1.05, alpha=0.42)
        (point,) = ax.plot(x, y0, "o", ms=4.2 + 3.2 * weight, color=color, mec=outline_color, mew=0.55, zorder=zorder + 1)
        artists.extend([bar, point])
        if annotate:
            text = ax.annotate(
                _rational_surface_label(chain.m, chain.n),
                xy=(x, y0),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
                color="#111827",
                zorder=zorder + 2,
            )
            _apply_line_halo(text, linewidth=2.4, alpha=0.9)
            artists.append(text)
    return artists


def overlay_rational_surface_markers(
    ax,
    markers: Sequence[RationalSurfaceMarker],
    *,
    show_verticals: bool = True,
    annotate: bool = True,
    max_labels: int = 24,
    color: str = "#2563eb",
    vertical_color: str = "#cbd5e1",
    zorder: float = 4.0,
):
    """Overlay low-order rational-surface markers in the ``q``-radius plane."""

    if not markers:
        return []
    artists = []
    if show_verticals:
        for q_val in sorted({round(float(marker.q), 12) for marker in markers}):
            artists.append(
                ax.axvline(
                    q_val,
                    color=vertical_color,
                    lw=0.75,
                    ls=":",
                    alpha=0.72,
                    zorder=zorder - 2,
                )
            )
    q_vals = np.array([marker.q for marker in markers], dtype=float)
    s_vals = np.array([marker.radial_label for marker in markers], dtype=float)
    n_vals = np.array([marker.n for marker in markers], dtype=float)
    sizes = 24.0 + 26.0 / np.maximum(n_vals, 1.0)
    scatter = ax.scatter(
        q_vals,
        s_vals,
        s=sizes,
        c=color,
        edgecolors="white",
        linewidths=0.65,
        alpha=0.95,
        label=r"rational $q=m/n$",
        zorder=zorder,
    )
    artists.append(scatter)
    if annotate and len(markers) <= int(max_labels):
        for marker in markers:
            text = ax.annotate(
                marker.label,
                xy=(marker.q, marker.radial_label),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
                color="#111827",
                zorder=zorder + 1,
            )
            _apply_line_halo(text, linewidth=2.2, alpha=0.9)
            artists.append(text)
    return artists


def overlay_rational_island_bars(
    ax,
    chains: Iterable[ResonantIslandChain],
    *,
    x_axis: str = "ratio",
    max_island_bars: int | None = None,
    annotate: bool = True,
    color: str = "#f59e0b",
    outline_color: str = "#111827",
    zorder: float = 7.0,
):
    """Overlay island-width bars on a ``q``/``m/n`` versus radius plot."""

    ordered = sorted(chains, key=lambda chain: chain.b_res, reverse=True)
    if max_island_bars is not None:
        ordered = ordered[: int(max_island_bars)]
    max_b = max((chain.b_res for chain in ordered), default=0.0)
    artists = []
    use_profile_q = str(x_axis).lower() in {"q", "profile_q", "q_profile"}
    for chain in ordered:
        x = float(chain.q if use_profile_q else chain.m / chain.n)
        y0 = float(chain.radial_label)
        dy = float(chain.half_width)
        weight = 0.0 if max_b <= 0.0 else float(np.sqrt(max(chain.b_res, 0.0) / max_b))
        lw = 1.45 + 2.7 * weight
        (bar,) = ax.plot([x, x], [y0 - dy, y0 + dy], color=color, lw=lw, solid_capstyle="round", zorder=zorder)
        _apply_line_halo(bar, color=outline_color, linewidth=lw + 1.15, alpha=0.42)
        (point,) = ax.plot(x, y0, "o", ms=4.2 + 3.3 * weight, color=color, mec=outline_color, mew=0.55, zorder=zorder + 1)
        artists.extend([bar, point])
        if annotate:
            text = ax.annotate(
                _rational_surface_label(chain.m, chain.n),
                xy=(x, y0),
                xytext=(5, 3),
                textcoords="offset points",
                fontsize=7,
                color="#111827",
                zorder=zorder + 2,
            )
            _apply_line_halo(text, linewidth=2.4, alpha=0.9)
            artists.append(text)
    return artists


def overlay_poincare_rational_trace(
    ax,
    trace: PoincareRationalTrace,
    *,
    color: str = "#7c3aed",
    cmap: str = "viridis",
    size: float = 10.0,
    alpha: float = 0.48,
    zorder: float = 5.6,
):
    """Overlay Poincare crossings already projected to ``(m/n, radial)``."""

    ratio = np.asarray(trace.ratio, dtype=float)
    radial = np.asarray(trace.radial_label, dtype=float)
    if ratio.shape != radial.shape:
        raise ValueError("trace.ratio and trace.radial_label must have the same shape")
    if trace.value is None:
        return ax.scatter(
            ratio,
            radial,
            s=float(size),
            c=color,
            alpha=float(alpha),
            edgecolors="white",
            linewidths=0.2,
            label=trace.label,
            zorder=zorder,
        )
    value = np.asarray(trace.value, dtype=float)
    if value.shape != ratio.shape:
        raise ValueError("trace.value must have the same shape as trace.ratio")
    return ax.scatter(
        ratio,
        radial,
        s=float(size),
        c=value,
        cmap=cmap,
        alpha=float(alpha),
        edgecolors="white",
        linewidths=0.2,
        label=trace.label,
        zorder=zorder,
    )


def _coerce_poincare_traces(
    poincare: PoincareRationalTrace | Sequence[PoincareRationalTrace] | None,
    poincare_ratio: Sequence[float] | None,
    poincare_radial: Sequence[float] | None,
) -> list[PoincareRationalTrace]:
    traces: list[PoincareRationalTrace] = []
    if poincare is not None:
        if isinstance(poincare, PoincareRationalTrace):
            traces.append(poincare)
        else:
            traces.extend(poincare)
    if poincare_ratio is not None or poincare_radial is not None:
        if poincare_ratio is None or poincare_radial is None:
            raise ValueError("poincare_ratio and poincare_radial must be supplied together")
        traces.append(
            PoincareRationalTrace(
                ratio=np.asarray(poincare_ratio, dtype=float),
                radial_label=np.asarray(poincare_radial, dtype=float),
            )
        )
    return traces


def plot_rational_surface_map(
    radial_labels: Sequence[float],
    q_profile: Sequence[float],
    *,
    n_values: Sequence[int] = (1, 2, 3),
    m_values: Sequence[int] | dict[int, Sequence[int]] | None = None,
    markers: Sequence[RationalSurfaceMarker] | None = None,
    chains: Iterable[ResonantIslandChain] = (),
    poincare: PoincareRationalTrace | Sequence[PoincareRationalTrace] | None = None,
    poincare_ratio: Sequence[float] | None = None,
    poincare_radial: Sequence[float] | None = None,
    show_q_profile: bool = True,
    show_rational_surfaces: bool = True,
    show_poincare: bool = True,
    show_island_bars: bool = True,
    annotate_rationals: bool = True,
    annotate_islands: bool = True,
    max_island_bars: int | None = None,
    ax=None,
    title: str | None = None,
):
    """Plot a modular ``q``/``m/n`` versus radius resonance atlas.

    This is the light-weight companion to the full Fourier heatmaps: users can
    independently combine the q-profile, low-order rational intersections,
    Poincare trace projections, and island-width bars.
    """

    import matplotlib.pyplot as plt

    radial, q_arr = _validated_profile(radial_labels, q_profile)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
    else:
        fig = ax.figure

    if show_q_profile:
        (line,) = ax.plot(q_arr, radial, color="#111827", lw=1.85, label="q-profile", zorder=5)
        _apply_line_halo(line, linewidth=4.0, alpha=0.88)

    marker_list = list(markers) if markers is not None else rational_surface_markers(
        radial,
        q_arr,
        n_values=n_values,
        m_values=m_values,
    )
    if show_rational_surfaces:
        overlay_rational_surface_markers(ax, marker_list, annotate=annotate_rationals)

    traces = _coerce_poincare_traces(poincare, poincare_ratio, poincare_radial)
    if show_poincare:
        for trace in traces:
            overlay_poincare_rational_trace(ax, trace)

    chain_list = list(chains)
    if show_island_bars:
        overlay_rational_island_bars(
            ax,
            chain_list,
            max_island_bars=max_island_bars,
            annotate=annotate_islands,
        )

    ax.set_xlabel(r"$q$ or $m/n$")
    ax.set_ylabel("radial label")
    ax.set_ylim(float(radial[0]), float(radial[-1]))
    ax.grid(True, color="#e5e7eb", lw=0.75, alpha=0.86)
    if title is None:
        title = "q-profile, rational surfaces, and island-width overlays"
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="best", fontsize=8, frameon=True, framealpha=0.9)
    return fig, ax, marker_list


def plot_radial_mode_heatmap(
    spectrum: RadialPerturbationFourierSpectrum,
    *,
    fixed_n: int | None = None,
    fixed_m: int | None = None,
    mode_values: Sequence[int] | None = None,
    axis_convention: str = "physical",
    q_profile: np.ndarray | None = None,
    chains: Iterable[ResonantIslandChain] = (),
    resonant_sign: int = -1,
    log_scale: bool = True,
    amplitude_scale: str | None = None,
    mask_zeros: bool = False,
    zero_color: str = "#ffffff",
    renderer: str = "pcolormesh",
    ax=None,
    cmap: str = "magma",
    title: str | None = None,
    show_resonance_curve: bool = True,
    resonance_curve_kwargs: dict | None = None,
    show_island_bars: bool = True,
    island_bar_kwargs: dict | None = None,
    annotate_islands: bool = True,
    max_island_bars: int | None = None,
):
    """Plot fixed-``n`` or fixed-``m`` radial magnetic-spectrum maps.

    With ``fixed_n``, the selected Fourier row is ``n_F=resonant_sign*n`` and
    the horizontal axis is Fourier ``m``; the positive-q resonant branch is
    ``m=-n_F*q(s)``.  With ``fixed_m``, ``axis_convention="physical"`` plots
    the physical toroidal number ``n_phys=-n_F`` while
    ``axis_convention="fourier"`` plots the actual Fourier index ``n_F`` and
    draws the branch ``n_F=-m/q(s)``.  Island bars span
    ``s_res +/- half_width`` at the corresponding low-order rational surface.
    """

    import matplotlib.pyplot as plt

    radial_map = radial_mode_spectrum(
        spectrum,
        fixed_n=fixed_n,
        fixed_m=fixed_m,
        mode_values=mode_values,
        resonant_sign=resonant_sign,
    )
    axis_key = _radial_axis_convention(axis_convention)
    plot_data, label = _plot_values(
        radial_map.amplitude,
        log_scale=log_scale,
        amplitude_scale=amplitude_scale,
        mask_zeros=mask_zeros,
    )
    vmin, vmax = _plot_limits(plot_data, log_scale=log_scale, amplitude_scale=amplitude_scale)
    plot_cmap = _cmap_with_bad(
        plt,
        cmap,
        zero_color if mask_zeros else None,
        low_color=_low_color_for_scale(log_scale, amplitude_scale, zero_color),
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(7.0, 0.38 * radial_map.mode_values.size), 4.8))
    else:
        fig = ax.figure

    x_values = _radial_axis_values(radial_map, axis_key)
    x_order = np.argsort(x_values)
    x_plot = x_values[x_order]
    plot_data_ordered = plot_data[:, x_order]
    renderer_key = renderer.lower()
    if renderer_key in {"pcolormesh", "mesh"}:
        im = ax.pcolormesh(
            _edges_from_centers(x_plot),
            _edges_from_centers(radial_map.radial_labels),
            plot_data_ordered,
            shading="auto",
            cmap=plot_cmap,
            vmin=vmin,
            vmax=vmax,
        )
    elif renderer_key in {"imshow", "heatmap", "image"}:
        x_edges = _edges_from_centers(x_plot)
        y_edges = _edges_from_centers(radial_map.radial_labels)
        im = ax.imshow(
            plot_data_ordered,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            cmap=plot_cmap,
            vmin=vmin,
            vmax=vmax,
        )
    else:
        raise ValueError("renderer must be 'imshow' or 'pcolormesh'")
    fig.colorbar(im, ax=ax, pad=0.02, label=label)

    curve = None
    if q_profile is not None and show_resonance_curve:
        kwargs = {} if resonance_curve_kwargs is None else dict(resonance_curve_kwargs)
        kwargs.setdefault("axis_convention", axis_key)
        curve = overlay_radial_resonance_curve(ax, radial_map, q_profile, **kwargs)

    if show_island_bars:
        kwargs = {} if island_bar_kwargs is None else dict(island_bar_kwargs)
        kwargs.setdefault("axis_convention", axis_key)
        overlay_radial_mode_island_bars(
            ax,
            chains,
            radial_map,
            max_island_bars=max_island_bars,
            annotate=annotate_islands,
            **kwargs,
        )

    ax.set_xlabel(_radial_axis_label(radial_map, axis_key))
    ax.set_ylabel("radial label")
    if title is None:
        if radial_map.fixed_axis == "n":
            title = f"Radial spectrum at fixed physical n={radial_map.fixed_value} (Fourier n_F={_fixed_fourier_n(radial_map)})"
        elif axis_key == "fourier":
            title = f"Radial spectrum at fixed Fourier m={radial_map.fixed_value}"
        else:
            title = f"Radial spectrum at fixed physical m={radial_map.fixed_value}"
    ax.set_title(title)
    ax.grid(True, alpha=0.18)
    if curve is not None:
        ax.legend(loc="best", fontsize=8)
    return fig, ax, radial_map


def plot_radial_mode_pcolormesh(*args, **kwargs):
    """Convenience wrapper around :func:`plot_radial_mode_heatmap` using pcolormesh."""

    kwargs["renderer"] = "pcolormesh"
    return plot_radial_mode_heatmap(*args, **kwargs)


def plot_resonant_radial_profiles(
    spectrum: RadialPerturbationFourierSpectrum,
    chains: Sequence[ResonantIslandChain],
    *,
    ax=None,
    max_modes: int = 10,
    title: str = r"Resonant spectrum and island-width estimates",
):
    """Plot radial profiles of ``2|tilde_b^1_{m,-n}|`` for resonant chains."""

    import matplotlib.pyplot as plt

    if spectrum.dBr.ndim != 2 or spectrum.radial_labels is None:
        raise ValueError("radial profiles require a radial stack spectrum with radial_labels")
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 4.6))
    else:
        fig = ax.figure

    radial = spectrum.radial_labels
    ordered = sorted(chains, key=lambda c: c.b_res, reverse=True)[: int(max_modes)]
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    for i, chain in enumerate(ordered):
        idx = spectrum.mode_index(chain.m, -chain.n)
        if idx is None:
            continue
        color = colors[i % len(colors)]
        profile = 2.0 * np.abs(spectrum.dBr[:, idx])
        ax.plot(radial, profile, marker="o", ms=3.5, color=color, label=f"({chain.m},{chain.n})")
        ax.axvline(chain.radial_label, color=color, lw=0.9, alpha=0.55)
        ax.annotate(
            f"w={chain.half_width:.2e}\nphase={np.degrees(chain.phase):.1f} deg",
            xy=(chain.radial_label, chain.b_res),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=7,
            color=color,
        )
    ax.set_yscale("log")
    ax.set_xlabel("s")
    ax.set_ylabel(r"$\tilde{b}^{1}_{res}=2|\tilde{b}^{1}_{m,-n}|$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    if ordered:
        ax.legend(loc="best", fontsize=8, ncol=min(3, len(ordered)))
    return fig, ax


def _interp_periodic(theta: np.ndarray, values: np.ndarray, theta0: float) -> float:
    src = np.asarray(theta, dtype=float)
    vals = np.asarray(values, dtype=float)
    tgt = float(np.mod(theta0, TWOPI))
    src_ext = np.concatenate([src[-1:] - TWOPI, src, src[:1] + TWOPI])
    vals_ext = np.concatenate([vals[-1:], vals, vals[:1]])
    return float(np.interp(tgt, src_ext, vals_ext))


def _interp_extrap(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    target = float(x0)
    if target < x_arr[0] and x_arr.size >= 2:
        slope = (y_arr[1] - y_arr[0]) / (x_arr[1] - x_arr[0])
        return float(y_arr[0] + slope * (target - x_arr[0]))
    if target > x_arr[-1] and x_arr.size >= 2:
        slope = (y_arr[-1] - y_arr[-2]) / (x_arr[-1] - x_arr[-2])
        return float(y_arr[-1] + slope * (target - x_arr[-1]))
    return float(np.interp(target, x_arr, y_arr))


def _surface_point(
    R_section: np.ndarray,
    Z_section: np.ndarray,
    theta: np.ndarray,
    radial_labels: np.ndarray,
    *,
    s: float,
    theta0: float,
) -> tuple[float, float]:
    R_theta = np.array([_interp_periodic(theta, R_section[ir], theta0) for ir in range(radial_labels.size)])
    Z_theta = np.array([_interp_periodic(theta, Z_section[ir], theta0) for ir in range(radial_labels.size)])
    return _interp_extrap(radial_labels, R_theta, s), _interp_extrap(radial_labels, Z_theta, s)


def island_bars_on_section(
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    radial_labels: np.ndarray,
    chains: Sequence[ResonantIslandChain],
    *,
    phi_section: float = 0.0,
    width_scale: float = 1.0,
) -> list[SectionIslandBar]:
    """Return island-width bar geometry on the nearest available phi section."""

    R, Z, phi, theta = prepare_surface_arrays(R_surf, Z_surf, phi_vals, theta_vals)
    radial = np.asarray(radial_labels, dtype=float)
    if radial.shape != (R.shape[1],):
        raise ValueError("radial_labels must match the radial surface count")
    iphi = int(np.argmin(np.abs(np.angle(np.exp(1j * (phi - float(phi_section)))))))
    bars: list[SectionIslandBar] = []
    for chain in chains:
        pts = chain.fixed_points(phi[iphi])
        theta_O = pts["theta_O"][0]
        theta_X = pts["theta_X"][0]
        for branch in range(chain.m):
            th_o = float(theta_O[branch])
            th_x = float(theta_X[branch])
            R_O, Z_O = _surface_point(R[iphi], Z[iphi], theta, radial, s=chain.radial_label, theta0=th_o)
            R_X, Z_X = _surface_point(R[iphi], Z[iphi], theta, radial, s=chain.radial_label, theta0=th_x)
            R_inner, Z_inner = _surface_point(
                R[iphi],
                Z[iphi],
                theta,
                radial,
                s=chain.radial_label - width_scale * chain.half_width,
                theta0=th_o,
            )
            R_outer, Z_outer = _surface_point(
                R[iphi],
                Z[iphi],
                theta,
                radial,
                s=chain.radial_label + width_scale * chain.half_width,
                theta0=th_o,
            )
            bars.append(
                SectionIslandBar(
                    chain=chain,
                    branch=branch,
                    theta_O=th_o,
                    theta_X=th_x,
                    R_O=R_O,
                    Z_O=Z_O,
                    R_X=R_X,
                    Z_X=Z_X,
                    R_inner=R_inner,
                    Z_inner=Z_inner,
                    R_outer=R_outer,
                    Z_outer=Z_outer,
                )
            )
    return bars


def plot_island_chains_on_section(
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    radial_labels: np.ndarray,
    chains: Sequence[ResonantIslandChain],
    *,
    phi_section: float = 0.0,
    max_chains: int = 4,
    width_scale: float = 1.0,
    show_legend: bool = True,
    ax=None,
    title: str | None = None,
):
    """Plot flux surfaces and Nardon island-width bars at O-points."""

    import matplotlib.pyplot as plt

    R, Z, phi, _ = prepare_surface_arrays(R_surf, Z_surf, phi_vals, theta_vals)
    iphi = int(np.argmin(np.abs(np.angle(np.exp(1j * (phi - float(phi_section)))))))
    ordered = sorted(chains, key=lambda c: c.b_res, reverse=True)[: int(max_chains)]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.6, 5.8))
    else:
        fig = ax.figure

    for ir in range(R.shape[1]):
        ax.plot(
            np.r_[R[iphi, ir], R[iphi, ir, 0]],
            np.r_[Z[iphi, ir], Z[iphi, ir, 0]],
            color="0.78",
            lw=0.8,
            zorder=1,
        )

    bars = island_bars_on_section(
        R_surf,
        Z_surf,
        phi_vals,
        theta_vals,
        radial_labels,
        ordered,
        phi_section=phi[iphi],
        width_scale=width_scale,
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    chain_order = {id(chain): i for i, chain in enumerate(ordered)}
    labelled: set[int] = set()
    for bar in bars:
        ci = chain_order[id(bar.chain)]
        color = colors[ci % len(colors)]
        label = None
        if id(bar.chain) not in labelled:
            label = (
                f"({bar.chain.m},{bar.chain.n}) "
                f"w={bar.chain.half_width:.2e}, phase={np.degrees(bar.chain.phase):.1f} deg"
            )
            labelled.add(id(bar.chain))
        ax.plot(
            [bar.R_inner, bar.R_outer],
            [bar.Z_inner, bar.Z_outer],
            color=color,
            lw=2.6,
            solid_capstyle="round",
            zorder=4,
            label=label,
        )
        ax.plot(bar.R_O, bar.Z_O, "o", ms=4.5, color=color, zorder=5)
        ax.plot(bar.R_X, bar.Z_X, "x", ms=5.5, mew=1.2, color=color, zorder=5)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    if title is None:
        title = f"Island-width bars at O-points, phi={np.degrees(phi[iphi]):.1f} deg"
    ax.set_title(title)
    if ordered and show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, borderaxespad=0.0)
    return fig, ax, bars


def plot_island_phase_scan(
    chain: ResonantIslandChain,
    *,
    phase_shifts: np.ndarray,
    phi_section: float = 0.0,
    ax=None,
):
    """Plot O-point angular rotation as the resonant Fourier phase is changed."""

    import matplotlib.pyplot as plt

    shifts = np.asarray(phase_shifts, dtype=float)
    base = chain.fixed_points(phi_section)["theta_O"][0, 0]
    tracked = []
    for shift in shifts:
        theta_o = chain.with_phase_shift(shift).fixed_points(phi_section)["theta_O"][0]
        expected = base - float(shift) / float(chain.m)
        idx = int(np.argmin(np.abs(np.angle(np.exp(1j * (theta_o - expected))))))
        tracked.append(float(theta_o[idx]))
    theta = np.asarray(tracked, dtype=float)
    dtheta = np.angle(np.exp(1j * (theta - base)))
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
    else:
        fig = ax.figure
    ax.plot(np.degrees(shifts), np.degrees(dtheta), marker="o", label="computed O-point shift")
    ax.plot(np.degrees(shifts), -np.degrees(shifts) / chain.m, "--", label=r"$-\Delta\arg(b)/m$")
    ax.set_xlabel("Fourier phase shift [deg]")
    ax.set_ylabel("O-point poloidal shift [deg]")
    ax.set_title(f"Island phase control for ({chain.m},{chain.n}) at phi={np.degrees(phi_section):.1f} deg")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return fig, ax


def plot_chirikov_overlaps(
    overlaps: Sequence[ChirikovOverlap],
    *,
    ax=None,
    title: str = "Chirikov island-overlap parameters",
):
    """Plot Chirikov overlap parameters for adjacent island chains."""

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(5.5, 1.1 * max(1, len(overlaps))), 4.0))
    else:
        fig = ax.figure
    labels = [f"({o.left.m},{o.left.n})-({o.right.m},{o.right.n})" for o in overlaps]
    sigma = np.array([o.sigma for o in overlaps], dtype=float)
    x = np.arange(len(overlaps))
    colors = ["#d62728" if val >= 1.0 else "#1f77b4" for val in sigma]
    ax.bar(x, sigma, color=colors, alpha=0.85)
    ax.axhline(1.0, color="0.2", lw=1.0, ls="--", label=r"$\sigma=1$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(r"$\sigma_{Chir}$")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    if len(overlaps):
        ax.legend(loc="best")
    return fig, ax


__all__ = [
    "PoincareRationalTrace",
    "RadialModeSpectrum",
    "RationalSurfaceMarker",
    "SectionIslandBar",
    "SpectrumSurfaceMatrix",
    "island_bars_on_section",
    "overlay_poincare_rational_trace",
    "overlay_radial_mode_island_bars",
    "overlay_radial_resonance_curve",
    "overlay_rational_island_bars",
    "overlay_rational_surface_markers",
    "overlay_surface_resonance_line",
    "plot_chirikov_overlaps",
    "plot_island_chains_on_section",
    "plot_island_phase_scan",
    "plot_radial_mode_heatmap",
    "plot_radial_mode_pcolormesh",
    "plot_rational_surface_map",
    "plot_resonant_radial_profiles",
    "plot_spectrum_bar3d",
    "plot_spectrum_heatmap",
    "plot_spectrum_pcolormesh",
    "radial_mode_spectrum",
    "rational_surface_markers",
    "spectrum_surface_matrix",
]
