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


def _mode_matrix(
    spectrum: RadialPerturbationFourierSpectrum,
    *,
    radial_index: int,
    m_values: Sequence[int],
    n_values: Sequence[int],
) -> np.ndarray:
    data = np.zeros((len(m_values), len(n_values)), dtype=float)
    for i, m_val in enumerate(m_values):
        for j, n_val in enumerate(n_values):
            idx = spectrum.mode_index(int(m_val), int(n_val))
            if idx is None:
                continue
            coeff = spectrum.dBr[idx] if spectrum.dBr.ndim == 1 else spectrum.dBr[int(radial_index), idx]
            data[i, j] = abs(coeff)
    return data


def plot_spectrum_heatmap(
    spectrum: RadialPerturbationFourierSpectrum,
    *,
    radial_index: int | None = None,
    m_max: int | None = None,
    n_max: int | None = None,
    chains: Iterable[ResonantIslandChain] = (),
    resonant_sign: int = -1,
    log_scale: bool = True,
    ax=None,
    cmap: str = "magma",
    title: str | None = None,
):
    """Plot ``|tilde_b^1_{mn}|`` for one radial surface."""

    import matplotlib.pyplot as plt

    if spectrum.dBr.ndim == 2:
        if radial_index is None:
            radial_index = spectrum.dBr.shape[0] // 2
    else:
        radial_index = 0
    m_lim = int(np.max(np.abs(spectrum.m)) if m_max is None else m_max)
    n_lim = int(np.max(np.abs(spectrum.n)) if n_max is None else n_max)
    m_values = np.arange(1, m_lim + 1, dtype=int)
    n_values = np.arange(-n_lim, n_lim + 1, dtype=int)
    amps = _mode_matrix(spectrum, radial_index=int(radial_index), m_values=m_values, n_values=n_values)
    plot_data = np.log10(amps + 1.0e-300) if log_scale else amps

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6.0, 0.42 * n_values.size), max(4.0, 0.26 * m_values.size)))
    else:
        fig = ax.figure

    finite = plot_data[np.isfinite(plot_data)]
    vmax = float(np.max(finite)) if finite.size else 0.0
    vmin = vmax - 8.0 if log_scale else 0.0
    im = ax.imshow(
        plot_data,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[n_values[0] - 0.5, n_values[-1] + 0.5, 0.5, m_values[-1] + 0.5],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    label = r"$\log_{10}|\tilde{b}^{1}_{mn}|$" if log_scale else r"$|\tilde{b}^{1}_{mn}|$"
    fig.colorbar(im, ax=ax, pad=0.02, label=label)
    ax.axvline(0.0, color="white", lw=0.7, alpha=0.65)
    ax.set_xlabel("n")
    ax.set_ylabel("m")
    if title is None:
        title = f"Magnetic perturbation spectrum at radial index {radial_index}"
    if spectrum.radial_labels is not None and spectrum.dBr.ndim == 2:
        title += f"  s={spectrum.radial_labels[int(radial_index)]:.4g}"
    ax.set_title(title)

    for chain in chains:
        n_plot = resonant_sign * chain.n
        if 1 <= chain.m <= m_lim and -n_lim <= n_plot <= n_lim:
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
            ax.text(
                n_plot,
                chain.m,
                f"{chain.m}/{chain.n}",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                zorder=6,
            )
    return fig, ax


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
    theta = np.array([chain.with_phase_shift(shift).fixed_points(phi_section)["theta_O"][0, 0] for shift in shifts])
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
    "SectionIslandBar",
    "island_bars_on_section",
    "plot_chirikov_overlaps",
    "plot_island_chains_on_section",
    "plot_island_phase_scan",
    "plot_resonant_radial_profiles",
    "plot_spectrum_heatmap",
]
