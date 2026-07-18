#!/usr/bin/env python3
"""Generate a public synthetic audit figure for Nardon's spectrum convention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from pyna.toroidal.perturbation_spectrum import (
    analyze_resonant_island_chains,
    radial_perturbation_Fourier_spectrum,
)


TWOPI = 2.0 * np.pi


def _build_case():
    theta = 0.17 + np.arange(160) * (TWOPI / 160)
    phi = -0.23 + np.arange(120) * (TWOPI / 120)
    radial = np.linspace(0.36, 0.84, 25)
    m, n0 = 3, 2
    resonant = 2.5e-4 * np.exp(0.63j)
    opposite = 0.8e-4 * np.exp(-0.27j)
    resonant_phase = m * theta[None, :] - n0 * phi[:, None]
    opposite_phase = m * theta[None, :] + n0 * phi[:, None]
    surface_field = 2.0 * np.real(
        resonant * np.exp(1j * resonant_phase)
        + opposite * np.exp(1j * opposite_phase)
    )
    radial_field = (
        (1.0 + 0.2 * (radial - 0.6))[:, None, None]
        * surface_field[None, :, :]
    )
    spectrum = radial_perturbation_Fourier_spectrum(
        radial_field,
        theta,
        phi,
        radial_labels=radial,
        layout="radial-phi-theta",
        m_max=6,
        n_max=5,
        min_amplitude=1.0e-14,
    )
    q_profile = 1.2 + 0.5 * radial
    chains = analyze_resonant_island_chains(
        spectrum,
        q_profile,
        n=n0,
        m_values=[m],
    )
    if len(chains) != 1:
        raise RuntimeError("synthetic convention case did not produce one chain")
    return theta, phi, radial, surface_field, spectrum, q_profile, chains[0], resonant, opposite


def _plot_report(output: Path, *, dpi: int) -> dict[str, object]:
    (
        theta,
        phi,
        radial,
        surface_field,
        spectrum,
        q_profile,
        chain,
        resonant,
        opposite,
    ) = _build_case()

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.2), constrained_layout=True)
    ax_field, ax_spectrum, ax_radial, ax_complex = axes.ravel()

    mesh = ax_field.pcolormesh(
        phi / np.pi,
        theta / np.pi,
        surface_field.T,
        shading="auto",
        cmap="RdBu_r",
    )
    fig.colorbar(mesh, ax=ax_field, label=r"synthetic $\widetilde b^1$")
    ax_field.set(
        xlabel=r"$\varphi/\pi$",
        ylabel=r"$\theta^*/\pi$",
        title=r"Real field: unequal $(m,-n_0)$ and $(m,+n_0)$ branches",
    )

    radial_index = int(np.argmin(np.abs(radial - chain.radial_label)))
    amplitude = np.abs(spectrum.dBr[radial_index])
    positive = amplitude[amplitude > 0.0]
    norm = LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(positive)))
    scatter = ax_spectrum.scatter(
        spectrum.nardon_n,
        spectrum.m,
        c=amplitude,
        s=45,
        cmap="viridis",
        norm=norm,
        edgecolors="none",
    )
    fig.colorbar(scatter, ax=ax_spectrum, label=r"$|\widetilde b^1_{m n_N}|$")
    ax_spectrum.scatter([chain.coefficient_n], [chain.m], marker="s", s=130,
                        facecolors="none", edgecolors="#d62728", linewidths=2.0,
                        label=r"selected $(m,-n_0)$")
    ax_spectrum.scatter([+chain.n], [chain.m], marker="D", s=85,
                        facecolors="none", edgecolors="#1f77b4", linewidths=1.8,
                        label=r"distinct $(m,+n_0)$")
    ax_spectrum.set(
        xlabel=r"signed Nardon index $n_N$",
        ylabel=r"poloidal index $m$",
        title=r"Fourier basis $e^{i(m\theta^*+n_N\varphi)}$",
        xlim=(-5.5, 5.5),
        ylim=(-6.5, 6.5),
    )
    ax_spectrum.axvline(0.0, color="0.72", lw=0.8)
    ax_spectrum.axhline(0.0, color="0.72", lw=0.8)
    ax_spectrum.legend(loc="upper left", fontsize=8)

    ax_radial.plot(radial, q_profile, color="#25364a", lw=2.1, label=r"$q(s)$ from $B_0$")
    ax_radial.axhline(chain.m / chain.n, color="#d62728", ls="--", lw=1.2,
                      label=rf"$q={chain.m}/{chain.n}$")
    ax_radial.axvspan(
        chain.radial_label - chain.half_width,
        chain.radial_label + chain.half_width,
        color="#e69f00",
        alpha=0.28,
        label="Nardon full island-width interval",
    )
    ax_radial.scatter([chain.radial_label], [chain.q], color="#d62728", s=48, zorder=4)
    ax_radial.set(
        xlabel=r"$s=\sqrt{\psi}$",
        ylabel=r"$q$",
        title=rf"Half-width $\delta={chain.half_width:.4f}$ in the same $s$ coordinate",
    )
    ax_radial.legend(fontsize=8)

    for coefficient, color, label in (
        (resonant, "#d62728", r"$\widetilde b^1_{m,-n_0}$"),
        (opposite, "#1f77b4", r"$\widetilde b^1_{m,+n_0}$"),
        (resonant.conjugate(), "#009e73", r"$\widetilde b^1_{-m,+n_0}$"),
    ):
        ax_complex.arrow(
            0.0,
            0.0,
            coefficient.real,
            coefficient.imag,
            color=color,
            width=2.0e-6,
            head_width=2.0e-5,
            length_includes_head=True,
            label=label,
        )
    scale = 1.2 * max(abs(resonant), abs(opposite))
    ax_complex.set(
        xlim=(-scale, scale),
        ylim=(-scale, scale),
        xlabel="real coefficient",
        ylabel="imaginary coefficient",
        title="Conjugacy flips both mode indices",
        aspect="equal",
    )
    ax_complex.axhline(0.0, color="0.75", lw=0.8)
    ax_complex.axvline(0.0, color="0.75", lw=0.8)
    ax_complex.legend(fontsize=8, loc="lower right")

    fig.suptitle("Synthetic Nardon magnetic-spectrum convention audit", fontsize=15)
    figure_path = output / "nardon_convention_audit.png"
    fig.savefig(figure_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)

    summary = {
        "schema": "pyna_synthetic_nardon_convention_v1",
        "fourier_basis": "exp(i*(m*theta_star+n_N*phi))",
        "reality_conjugacy": "(m,n_N)<->(-m,-n_N)",
        "positive_q_resonant_branch": "(m,-n0)",
        "mode": [int(chain.m), int(chain.n)],
        "coefficient_nardon_n": int(chain.coefficient_n),
        "radial_coordinate": "s=sqrt(psi)",
        "resonant_surface": float(chain.radial_label),
        "b_res": float(chain.b_res),
        "half_width": float(chain.half_width),
        "figure": figure_path.name,
    }
    (output / "nardon_convention_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "MCFdata" / "pyna" / "nardon_convention",
    )
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = _plot_report(args.output, dpi=args.dpi)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
