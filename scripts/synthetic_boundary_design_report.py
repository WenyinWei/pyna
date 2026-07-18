#!/usr/bin/env python3
"""Generate a synthetic boundary-design report with topology, optimization, and heat plots.

The data are analytic toy arrays intended for public plotting smoke checks.  No
real device geometry, field files, or private case paths are used.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib
import numpy as np

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyna.plot import (  # noqa: E402
    camera_xyz_from_cylindrical,
    plot_boundary_response_optimization_history,
    plot_poincare_topology_payload_report,
    plot_wall_heat_camera_surface,
    plot_wall_heat_footprint,
    wall_heat_footprint_from_hits,
    wall_surface_heat_from_footprint,
)
from pyna.toroidal.control import (  # noqa: E402
    BoundaryTopologyControlProblem,
    CallableBoundaryPlasmaResponseBackend,
    boundary_response_observables,
    core_preservation_snapshot,
    format_boundary_topology_control_summary,
    solve_boundary_topology_control,
)
from synthetic_poincare_topology_report import synthetic_poincare_topology_payload  # noqa: E402


def _close(fig):
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass


def synthetic_boundary_control_result():
    """Return a toy nonlinear control result for topology/heat observables."""

    matrix = np.array(
        [
            [0.80, 0.18],
            [0.30, 0.70],
            [0.10, 0.92],
        ],
        dtype=float,
    )
    base = np.array([0.22, 0.36, 0.18], dtype=float)
    radial = np.asarray([0.25, 0.55, 0.75], dtype=float)
    theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    surface_R = 3.0 + radial[:, None] * np.cos(theta)[None, :]
    surface_Z = 0.72 * radial[:, None] * np.sin(theta)[None, :]
    core_reference = core_preservation_snapshot(
        axis=[3.0, 0.0],
        radial_labels=radial,
        surface_R=surface_R,
        surface_Z=surface_Z,
        q_profile=[1.10, 1.28, 1.55],
        iota_profile=[0.91, 0.78, 0.65],
        scalars={"volume": 12.0, "minor_radius": 0.75},
    )

    def plasma_response(request):
        controls = request.controls
        core = core_preservation_snapshot(
            axis=[3.0 + 0.012 * controls[0] - 0.006 * controls[1], -0.008 * controls[0]],
            radial_labels=radial,
            surface_R=surface_R + (0.010 * controls[0] - 0.004 * controls[1]),
            surface_Z=surface_Z + (0.006 * controls[1]),
            q_profile=np.asarray(core_reference.q_profile) + np.array([0.006, -0.010, 0.014]) * controls[0],
            iota_profile=np.asarray(core_reference.iota_profile) + np.array([-0.004, 0.006, -0.008]) * controls[1],
            scalars={
                "volume": 12.0 + 0.08 * controls[0] - 0.03 * controls[1],
                "minor_radius": 0.75 + 0.010 * controls[0],
            },
        )
        return {"B0": "synthetic_healed_B0", "delta_B": "synthetic_response_deltaB", "core": core}

    backend = CallableBoundaryPlasmaResponseBackend(plasma_response)

    def boundary_observable_builder(_snapshot, request):
        controls = request.controls
        linear = base + matrix @ controls
        nonlinear = np.array(
            [
                linear[0] + 0.035 * controls[0] * controls[1],
                np.tanh(linear[1]),
                linear[2] + 0.06 * controls[1] ** 2,
            ],
            dtype=float,
        )
        return boundary_response_observables(
            ["island.width.5_2", "chaos.layer.mid", "heat.outer.target"],
            nonlinear,
            weights=[3.0, 2.0, 1.5],
        )

    problem = BoundaryTopologyControlProblem(
        backend,
        initial_controls=[0.0, 0.0],
        control_labels=["spectral_mode_5_2", "strike_trim"],
        target={
            "island.width.5_2": 0.56,
            "chaos.layer.mid": 0.62,
            "heat.outer.target": 0.74,
        },
        observable_builders=[boundary_observable_builder],
        core_reference=core_reference,
        core_weights={"axis": 25.0, "surface": 10.0, "q_profile": 8.0, "iota_profile": 8.0, "scalar": 2.0},
        steps=[0.08, 0.08],
        n_iterations=4,
        bounds=(-0.8, 0.8),
        control_bounds={"spectral_mode_5_2": (-0.6, 0.6), "strike_trim": (-0.2, 0.7)},
        line_search=(1.0, 0.5, 0.25),
        metadata={"case": "synthetic_boundary_design"},
    )
    return solve_boundary_topology_control(problem)


def synthetic_wall_geometry(n_phi: int = 24, n_theta: int = 96):
    """Return a smooth toy wall grid in cylindrical coordinates."""

    wall_phi = np.linspace(0.0, 2.0 * np.pi, int(n_phi), endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False)
    wall_R = 3.05 + 0.68 * np.cos(theta)[None, :] + 0.05 * np.cos(2.0 * wall_phi[:, None])
    wall_Z = 0.78 * np.sin(theta)[None, :] + 0.035 * np.sin(3.0 * wall_phi[:, None])
    return wall_phi, wall_R, wall_Z


def synthetic_wall_heat_footprint():
    """Return a toy wall heat footprint with two strike-line bands."""

    wall_phi, wall_R, wall_Z = synthetic_wall_geometry()
    n_hits = 900
    t = np.linspace(0.0, 1.0, n_hits, endpoint=False)
    phi_a = 2.0 * np.pi * t
    theta_a = 0.32 * np.pi + 0.18 * np.sin(3.0 * phi_a)
    phi_b = 2.0 * np.pi * ((t + 0.37) % 1.0)
    theta_b = 1.18 * np.pi + 0.15 * np.sin(2.0 * phi_b + 0.5)
    R_a = 3.05 + 0.68 * np.cos(theta_a) + 0.05 * np.cos(2.0 * phi_a)
    Z_a = 0.78 * np.sin(theta_a) + 0.035 * np.sin(3.0 * phi_a)
    R_b = 3.05 + 0.68 * np.cos(theta_b) + 0.05 * np.cos(2.0 * phi_b)
    Z_b = 0.78 * np.sin(theta_b) + 0.035 * np.sin(3.0 * phi_b)
    hit_R = np.concatenate([R_a, R_b])
    hit_Z = np.concatenate([Z_a, Z_b])
    hit_phi = np.concatenate([phi_a, phi_b])
    weights = np.concatenate([
        1.0 + 0.55 * np.cos(phi_a - 0.4) ** 2,
        0.55 + 0.35 * np.sin(phi_b + 0.7) ** 2,
    ])
    footprint = wall_heat_footprint_from_hits(
        hit_R,
        hit_Z,
        hit_phi,
        wall_phi,
        wall_R,
        wall_Z,
        weights=weights,
        n_phi_bins=48,
        n_s_bins=96,
        field_period=2.0 * np.pi,
    )
    return footprint, wall_phi, wall_R, wall_Z


def write_summary(path: Path, control_result, footprint) -> Path:
    """Write a short text summary next to the generated figures."""

    lines = [
        "Synthetic boundary-design report",
        "",
        format_boundary_topology_control_summary(control_result),
    ]
    lines.extend(
        [
            f"wall heat total weight: {float(np.sum(footprint.heat)):.6g}",
            f"wall heat peak bin: {float(np.nanmax(footprint.heat)):.6g}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("~/MCFdata/pyna_synthetic/boundary_design_report"),
        help="Output directory for synthetic figures.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args(argv)

    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    topology_payload = synthetic_poincare_topology_payload()
    fig, _axes, _classification = plot_poincare_topology_payload_report(
        topology_payload,
        growth_threshold=0.05,
        recurrence_threshold=0.02,
        out_path=out_dir / "synthetic_poincare_topology_report.png",
        save_dpi=args.dpi,
        title="Synthetic Poincare topology response",
    )
    _close(fig)

    control_result = synthetic_boundary_control_result()
    fig, _axes = plot_boundary_response_optimization_history(
        control_result.optimization,
        out_path=out_dir / "synthetic_boundary_optimization_history.png",
        save_dpi=args.dpi,
        title="Synthetic boundary response optimization",
    )
    _close(fig)

    footprint, wall_phi, wall_R, wall_Z = synthetic_wall_heat_footprint()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
    plot_wall_heat_footprint(footprint, ax=ax, title="Synthetic wall heat footprint")
    fig.savefig(out_dir / "synthetic_wall_heat_footprint.png", dpi=args.dpi, bbox_inches="tight")
    _close(fig)

    surface = wall_surface_heat_from_footprint(footprint, wall_phi, wall_R, wall_Z, sigma_phi=0.35, sigma_s=0.025)
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    plot_wall_heat_camera_surface(
        surface,
        ax=ax,
        camera_position=camera_xyz_from_cylindrical(6.5, 1.25, 0.35 * np.pi),
        camera_target=camera_xyz_from_cylindrical(3.05, 0.0, 0.35 * np.pi),
        colorbar=True,
        title="Synthetic wall heat camera view",
    )
    fig.savefig(out_dir / "synthetic_wall_heat_camera.png", dpi=args.dpi, bbox_inches="tight")
    _close(fig)

    write_summary(out_dir / "synthetic_boundary_design_summary.txt", control_result, footprint)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
