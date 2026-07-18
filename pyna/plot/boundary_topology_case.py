"""Scientific audit plots for boundary-topology control cases."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def _save(fig, out_path, dpi: int):
    if out_path is None:
        return None
    path = Path(out_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    return path


def _chain_by_mode(chains, mode):
    selected = [chain for chain in chains if (int(chain.m), int(chain.n)) == tuple(mode)]
    return max(selected, key=lambda chain: float(chain.half_width), default=None)


def _target_complex(result, mode):
    if result is None:
        return None
    m, n = (int(mode[0]), int(mode[1]))
    labels = tuple(result.validation.target_observables.labels)
    values = np.asarray(result.validation.target_observables.values, dtype=float)
    index = {label: idx for idx, label in enumerate(labels)}
    real_label = f"island.m{m}.n{n}.coefficient_real"
    imag_label = f"island.m{m}.n{n}.coefficient_imag"
    if real_label not in index or imag_label not in index:
        return None
    return complex(values[index[real_label]], values[index[imag_label]])


def _effective_dipole_sites(actuators, controls):
    sites = {}
    actuator_specs = getattr(actuators, "actuators", ())
    for command, actuator in zip(np.asarray(controls, dtype=float).ravel(), actuator_specs):
        for dipole in getattr(actuator, "dipoles", ()):
            if dipole.anchor_phi is None or dipole.anchor_R is None or dipole.anchor_Z is None:
                continue
            key = (
                round(float(dipole.anchor_phi), 12),
                round(float(dipole.anchor_theta or 0.0), 12),
                round(float(dipole.anchor_R), 12),
                round(float(dipole.anchor_Z), 12),
            )
            sites[key] = sites.get(key, 0.0) + float(command) * float(dipole.magnetic_moment)
    return sites


def plot_boundary_topology_control_audit(
    case,
    actuators,
    initial_state,
    final_state,
    *,
    result=None,
    modes: Sequence[tuple[int, int]] | None = None,
    section_phi: float = 0.0,
    out_path: str | Path | None = None,
    save_dpi: int = 220,
    title: str | None = None,
):
    """Plot geometry, spectrum, islands, heat, and residuals for one control run."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    if modes is None:
        modes = tuple(
            dict.fromkeys(
                (int(chain.m), int(chain.n))
                for chain in tuple(initial_state.chains) + tuple(final_state.chains)
            )
        )
    modes = tuple((int(m), int(n)) for m, n in modes)
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.2), constrained_layout=True)

    phi_delta = np.angle(np.exp(1j * (case.phi_vals - float(section_phi))))
    iphi = int(np.argmin(np.abs(phi_delta)))
    ax = axes[0, 0]
    for ir in range(case.radial_labels.size):
        edge = ir == case.radial_labels.size - 1
        ax.plot(
            case.R_surf[iphi, ir],
            case.Z_surf[iphi, ir],
            color="#2f3e46" if edge else "#9aa3a8",
            lw=1.6 if edge else 0.55,
            alpha=0.95 if edge else 0.6,
        )
    response_case = getattr(final_state, "response_case", case)
    if response_case is not case:
        response_delta = np.angle(np.exp(1j * (response_case.phi_vals - float(section_phi))))
        response_iphi = int(np.argmin(np.abs(response_delta)))
        ax.plot(
            response_case.R_surf[response_iphi, -1],
            response_case.Z_surf[response_iphi, -1],
            color="#2f7d6d",
            lw=1.4,
            ls="--",
            label="response LCFS",
        )
    controls = final_state.controls
    sites = _effective_dipole_sites(actuators, controls)
    selected_sites = [
        (key, value)
        for key, value in sites.items()
        if abs(float(np.angle(np.exp(1j * (key[0] - case.phi_vals[iphi])))))
        <= max(1.0e-10, np.pi / max(2, case.phi_vals.size))
    ]
    if selected_sites:
        values = np.asarray([value for _key, value in selected_sites], dtype=float)
        scale = max(float(np.max(np.abs(values))), 1.0e-300)
        scatter = ax.scatter(
            [key[2] for key, _value in selected_sites],
            [key[3] for key, _value in selected_sites],
            c=values,
            cmap="coolwarm",
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-scale, vmax=scale),
            s=30,
            edgecolors="black",
            linewidths=0.35,
            zorder=5,
        )
        fig.colorbar(scatter, ax=ax, label="effective dipole moment [A m$^2$]")
    site_suffix = " and dipoles" if sites else ""
    ax.set_title(f"PEST surfaces{site_suffix}, phi={case.phi_vals[iphi]:.3f}")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal", adjustable="box")
    if response_case is not case:
        ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    colors = plt.get_cmap("tab10")
    coefficients = []
    for idx, mode in enumerate(modes):
        before = _chain_by_mode(initial_state.chains, mode)
        after = _chain_by_mode(final_state.chains, mode)
        c0 = 0.0j if before is None else complex(before.coefficient)
        c1 = 0.0j if after is None else complex(after.coefficient)
        target = _target_complex(result, mode)
        color = colors(idx % 10)
        ax.plot([c0.real, c1.real], [c0.imag, c1.imag], color=color, lw=1.3)
        ax.scatter(c0.real, c0.imag, facecolors="none", edgecolors=[color], s=45, marker="o")
        ax.scatter(c1.real, c1.imag, color=[color], s=38, marker="o", label=f"({mode[0]}, {mode[1]})")
        if target is not None:
            ax.scatter(target.real, target.imag, color=[color], s=55, marker="x", linewidths=1.8)
        coefficients.extend((c0, c1))
        if target is not None:
            coefficients.append(target)
    extent = max([abs(value) for value in coefficients] + [1.0e-12]) * 1.18
    ax.axhline(0.0, color="#9aa3a8", lw=0.6)
    ax.axvline(0.0, color="#9aa3a8", lw=0.6)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("resonant coefficient phase plane")
    ax.set_xlabel("Re(tilde b_mn)")
    ax.set_ylabel("Im(tilde b_mn)")
    if modes:
        ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 2]
    for idx, mode in enumerate(modes):
        color = colors(idx % 10)
        before = _chain_by_mode(initial_state.chains, mode)
        after = _chain_by_mode(final_state.chains, mode)
        if before is not None:
            ax.scatter(before.radial_label, 2.0 * before.half_width, facecolors="none", edgecolors=[color], s=52)
        if after is not None:
            ax.scatter(after.radial_label, 2.0 * after.half_width, color=[color], s=44, label=f"({mode[0]}, {mode[1]})")
    for interval in final_state.chaotic_intervals:
        ax.axvspan(interval.inner, interval.outer, color="#d95f02", alpha=0.16)
    sigma_threshold = dict(final_state.metadata or {}).get("sigma_threshold")
    band_title = "island full width and overlap bands"
    if sigma_threshold is not None:
        band_title += f" (sigma >= {float(sigma_threshold):g})"
    ax.set_title(band_title)
    ax.set_xlabel(case.radial_coordinate)
    ax.set_ylabel("predicted full island width")
    ax.grid(True, alpha=0.22)
    ax.text(0.02, 0.98, "open: initial; filled: controlled", transform=ax.transAxes, va="top", fontsize=8)
    if modes:
        ax.legend(frameon=False, fontsize=8)

    heat_states = (initial_state.heat, final_state.heat)
    positive = np.concatenate(
        [state.heat[state.heat > 0.0] for state in heat_states if state is not None and np.any(state.heat > 0.0)]
    ) if any(state is not None and np.any(state.heat > 0.0) for state in heat_states) else np.array([])
    norm = None
    if positive.size:
        vmax = float(np.max(positive))
        vmin = max(float(np.percentile(positive, 2.0)), vmax * 1.0e-5)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    heat_mesh = None
    for column, (label, state) in enumerate((("initial", initial_state.heat), ("controlled", final_state.heat))):
        ax = axes[1, column]
        if state is None:
            ax.text(0.5, 0.5, "heat backend not supplied", ha="center", va="center", transform=ax.transAxes)
        else:
            heat_mesh = ax.pcolormesh(
                state.phi_values,
                state.s_values,
                state.heat.T,
                shading="auto",
                cmap="inferno",
                norm=norm,
            )
        ax.set_title(f"{label} wall heat flux")
        ax.set_xlabel("phi [rad]")
        ax.set_ylabel("wall arclength fraction")
    if heat_mesh is not None:
        fig.colorbar(heat_mesh, ax=[axes[1, 0], axes[1, 1]], label="heat flux [model units]")

    ax = axes[1, 2]
    if result is not None:
        rows = tuple(result.validation.residual_rows[:10])
        labels = [row.label for row in rows][::-1]
        values = [row.weighted_abs_residual for row in rows][::-1]
        ax.barh(np.arange(len(rows)), values, color="#2f7d6d", alpha=0.88)
        ax.set_yticks(np.arange(len(rows)), labels=labels, fontsize=7)
        ax.set_xlabel("weighted absolute residual")
        ax.set_title("largest final target residuals")
        ax.grid(True, axis="x", alpha=0.22)
    else:
        ax.axis("off")
    fig.suptitle(title or f"{case.name} boundary topology control audit", fontsize=14)
    _save(fig, out_path, save_dpi)
    return fig, axes


def plot_boundary_response_matrix_audit(
    system,
    *,
    out_path: str | Path | None = None,
    save_dpi: int = 220,
    title: str | None = "Boundary response matrix audit",
):
    """Plot weighted response rows, actuator correlations, and singular values."""

    import matplotlib.pyplot as plt

    matrix = np.asarray(system.response_matrix, dtype=float)
    weighted = np.sqrt(np.asarray(system.weights, dtype=float))[:, None] * matrix
    row_scale_all = np.max(np.abs(weighted), axis=1)
    active_rows = row_scale_all > max(1.0e-15, float(np.max(row_scale_all, initial=0.0)) * 1.0e-14)
    if not np.any(active_rows):
        active_rows = np.ones(row_scale_all.shape, dtype=bool)
    display_weighted = weighted[active_rows]
    display_labels = tuple(label for label, active in zip(system.labels, active_rows) if active)
    row_scale = np.max(np.abs(display_weighted), axis=1, keepdims=True)
    normalized = np.divide(
        display_weighted,
        row_scale,
        out=np.zeros_like(display_weighted),
        where=row_scale > 0.0,
    )
    norms = np.linalg.norm(weighted, axis=0)
    correlation = np.divide(
        weighted.T @ weighted,
        norms[:, None] * norms[None, :],
        out=np.zeros((weighted.shape[1], weighted.shape[1]), dtype=float),
        where=(norms[:, None] * norms[None, :]) > 0.0,
    )
    singular = np.linalg.svd(weighted, compute_uv=False)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), constrained_layout=True)

    image = axes[0].imshow(normalized, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axes[0].set_title("weighted response, row normalized")
    axes[0].set_xlabel("actuator column")
    axes[0].set_ylabel("observable row")
    if len(system.control_labels) <= 14:
        axes[0].set_xticks(np.arange(len(system.control_labels)), labels=system.control_labels, rotation=75, ha="right", fontsize=7)
    if len(display_labels) <= 24:
        axes[0].set_yticks(np.arange(len(display_labels)), labels=display_labels, fontsize=7)
    fig.colorbar(image, ax=axes[0], label="normalized sensitivity")

    corr_image = axes[1].imshow(correlation, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axes[1].set_title("actuator column correlation")
    axes[1].set_xlabel("actuator column")
    axes[1].set_ylabel("actuator column")
    if len(system.control_labels) <= 14:
        ticks = np.arange(len(system.control_labels))
        axes[1].set_xticks(ticks, labels=system.control_labels, rotation=75, ha="right", fontsize=7)
        axes[1].set_yticks(ticks, labels=system.control_labels, fontsize=7)
    fig.colorbar(corr_image, ax=axes[1], label="correlation")

    axes[2].semilogy(np.arange(1, singular.size + 1), singular, marker="o", color="#2f7d6d")
    axes[2].set_title(f"singular values, rank={system.diagnostics.rank}")
    axes[2].set_xlabel("singular index")
    axes[2].set_ylabel("weighted singular value")
    axes[2].grid(True, which="both", alpha=0.24)
    if title:
        fig.suptitle(title)
    _save(fig, out_path, save_dpi)
    return fig, axes


__all__ = [
    "plot_boundary_response_matrix_audit",
    "plot_boundary_topology_control_audit",
]
