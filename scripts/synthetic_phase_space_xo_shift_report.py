#!/usr/bin/env python3
"""Generate a public synthetic audit of phase-space X/O point responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from pyna.plot import (
    draw_poincare_phase_response_arrows,
    poincare_phase_response_radial_deltas,
)
from pyna.toroidal.control import (
    HealedSurfaceSectionChart,
    solve_periodic_point_phase_response,
)


TWOPI = 2.0 * np.pi


def _healed_chart() -> HealedSurfaceSectionChart:
    axis_R = 5.55
    axis_Z = 0.02

    def s_theta_to_x(s_theta):
        values = np.asarray(s_theta, dtype=float)
        s = values[..., 0]
        theta = values[..., 1]
        triangularity = 0.10 * s * np.cos(2.0 * theta)
        R = axis_R + 0.62 * s * np.cos(theta) + triangularity
        Z = axis_Z + 0.39 * s * np.sin(theta) - 0.045 * s * np.sin(2.0 * theta)
        return np.stack([R, Z], axis=-1)

    def x_to_s_theta(x_RZ):
        points = np.asarray(x_RZ, dtype=float)
        flat = points.reshape(-1, 2)
        output = np.empty_like(flat)
        for index, point in enumerate(flat):
            s = np.hypot((point[0] - axis_R) / 0.62, (point[1] - axis_Z) / 0.39)
            theta = np.arctan2((point[1] - axis_Z) / 0.39, (point[0] - axis_R) / 0.62)
            for _ in range(12):
                mapped = s_theta_to_x(np.asarray([s, theta]))
                residual = mapped - point
                J = jacobian_s_theta(np.asarray([s, theta]))
                correction = np.linalg.solve(J, residual)
                s -= correction[0]
                theta -= correction[1]
            output[index] = [max(s, 0.0), theta % TWOPI]
        return output.reshape(points.shape)

    def jacobian_s_theta(s_theta):
        s, theta = np.asarray(s_theta, dtype=float)
        return np.asarray(
            [
                [
                    0.62 * np.cos(theta) + 0.10 * np.cos(2.0 * theta),
                    -0.62 * s * np.sin(theta) - 0.20 * s * np.sin(2.0 * theta),
                ],
                [
                    0.39 * np.sin(theta) - 0.045 * np.sin(2.0 * theta),
                    0.39 * s * np.cos(theta) - 0.09 * s * np.cos(2.0 * theta),
                ],
            ]
        )

    return HealedSurfaceSectionChart(
        s_theta_to_x,
        x_to_s_theta,
        jacobian_s_theta,
        radial_phase="psi",
        canonical=True,
        name="synthetic_healed_boundary_chart",
        metadata={"case": "synthetic", "theta_origin": "outboard midplane"},
    )


def _response(
    chart: HealedSurfaceSectionChart,
    *,
    s0: float,
    theta0: float,
    kind: str,
    delta_s: float,
    delta_theta: float,
    map_power: int,
):
    z0 = np.asarray([s0**2, theta0])
    J = chart.jacobian(z0)
    if kind == "O":
        angle = 0.56
        DP_z = np.asarray(
            [
                [np.cos(angle), -0.15 * np.sin(angle)],
                [np.sin(angle) / 0.15, np.cos(angle)],
            ]
        )
    else:
        growth = np.exp(0.42)
        rotation = np.asarray(
            [[np.cos(0.28), -np.sin(0.28)], [np.sin(0.28), np.cos(0.28)]]
        )
        DP_z = rotation @ np.diag([growth, 1.0 / growth]) @ rotation.T
    desired_delta_z = np.asarray([2.0 * s0 * delta_s, delta_theta])
    forcing_z = (np.eye(2) - DP_z) @ desired_delta_z
    DP_x = J @ DP_z @ np.linalg.inv(J)
    forcing_x = J @ forcing_z
    return solve_periodic_point_phase_response(
        DP_x,
        forcing_x,
        chart,
        z0=z0,
        kind=kind,
        map_power=map_power,
    )


def _build_responses(chart: HealedSurfaceSectionChart, *, m: int = 5):
    s0 = 0.84
    responses = []
    for branch in range(m):
        theta_O = 0.18 + TWOPI * branch / m
        theta_X = theta_O + np.pi / m
        responses.append(
            _response(
                chart,
                s0=s0,
                theta0=theta_O,
                kind="O",
                delta_s=0.0045 * np.cos(theta_O),
                delta_theta=0.075 + 0.012 * np.sin(theta_O),
                map_power=m,
            )
        )
        responses.append(
            _response(
                chart,
                s0=s0,
                theta0=theta_X,
                kind="X",
                delta_s=-0.0035 * np.sin(theta_X),
                delta_theta=-0.052 + 0.010 * np.cos(theta_X),
                map_power=m,
            )
        )
    return responses


def _plot(output: Path, *, dpi: int) -> dict[str, object]:
    chart = _healed_chart()
    responses = _build_responses(chart)
    fig = plt.figure(figsize=(12.4, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.55, 1.0))
    ax_map = fig.add_subplot(grid[:, 0])
    ax_phase = fig.add_subplot(grid[0, 1])
    ax_radial = fig.add_subplot(grid[1, 1])

    theta = np.linspace(0.0, TWOPI, 500)
    for s in np.linspace(0.18, 1.0, 10):
        path = chart.s_theta_to_x(np.column_stack([np.full(theta.size, s), theta]))
        ax_map.plot(path[:, 0], path[:, 1], color="#aeb8c4", lw=0.65, alpha=0.68)
    boundary = chart.s_theta_to_x(np.column_stack([np.ones(theta.size), theta]))
    ax_map.plot(boundary[:, 0], boundary[:, 1], color="#35495e", lw=1.6)

    for kind, marker, color, label in (
        ("O", "o", "#176b87", "O point (Newton/fixed point)"),
        ("X", "x", "#c43b4d", "X point (Newton/fixed point)"),
    ):
        selected = [response for response in responses if response.kind == kind]
        points = np.asarray([response.x0_RZ_m for response in selected])
        ax_map.scatter(
            points[:, 0],
            points[:, 1],
            marker=marker,
            s=58 if kind == "O" else 62,
            facecolors="none" if kind == "O" else color,
            edgecolors=color,
            linewidths=1.7,
            label=label,
            zorder=7,
        )
    draw_poincare_phase_response_arrows(
        ax_map,
        responses,
        chart,
        n_points=81,
        linewidth=2.2,
        mutation_scale=13.0,
    )
    ax_map.set(
        xlabel="R [m]",
        ylabel="Z [m]",
        title=r"Phase response: curved arrows follow healed constant-$s$ surfaces",
        aspect="equal",
    )
    ax_map.legend(loc="lower left", fontsize=8, frameon=False)

    kinds = np.asarray([response.kind for response in responses])
    branch = np.arange(len(responses)) // 2
    for kind, color, marker in (("O", "#176b87", "o"), ("X", "#c43b4d", "x")):
        mask = kinds == kind
        theta_shift = np.asarray(
            [response.delta_theta_star_wrapped for response in responses]
        )[mask]
        ax_phase.scatter(branch[mask], theta_shift, color=color, marker=marker, s=48, label=kind)
    ax_phase.axhline(0.0, color="0.65", lw=0.8)
    ax_phase.set(
        xlabel="island branch",
        ylabel=r"$\delta\theta^*$ [rad]",
        title=r"Authoritative healed phase response from $(I-DP_z^K)\delta z=\delta P_z^K$",
    )
    ax_phase.legend(frameon=False)

    radial = poincare_phase_response_radial_deltas(responses)
    width = 0.34
    indices = np.arange(5)
    ax_radial.bar(indices - width / 2, radial["O"], width, color="#176b87", alpha=0.82, label="O")
    ax_radial.bar(indices + width / 2, radial["X"], width, color="#c43b4d", alpha=0.82, label="X")
    ax_radial.axhline(0.0, color="0.55", lw=0.8)
    ax_radial.set(
        xlabel="island branch",
        ylabel=r"separate radial response $\delta s$",
        title="Radial response is reported separately, not drawn as a straight arrow",
    )
    ax_radial.legend(frameon=False)

    fig.suptitle("Synthetic phase-space X/O shift audit (canonical z=(psi, theta*))", fontsize=14)
    figure_path = output / "phase_space_xo_shift_audit.png"
    fig.savefig(figure_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)

    summary = {
        "schema": "pyna_synthetic_phase_space_xo_shift_v1",
        "case": "synthetic",
        "equation": "(I-DPk_z) delta_z = deltaPk_z",
        "coordinate_choice": chart.coordinate_choice,
        "canonical": chart.canonical,
        "map_power": 5,
        "n_responses": len(responses),
        "all_phase_space_valid": all(response.phase_space_valid for response in responses),
        "max_solve_condition_number": float(
            max(response.solve_condition_number for response in responses)
        ),
        "max_relative_residual": float(max(response.relative_residual for response in responses)),
        "figure": figure_path.name,
    }
    (output / "phase_space_xo_shift_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "MCFdata" / "pyna" / "phase_space_xo_shift",
    )
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(json.dumps(_plot(args.output, dpi=args.dpi), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
