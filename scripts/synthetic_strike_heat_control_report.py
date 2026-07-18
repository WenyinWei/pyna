#!/usr/bin/env python3
"""Build a public synthetic manifold-to-strike heat-control audit."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyna.plot import WallSurfaceHeat, plot_wall_heat_camera_views
from pyna.toroidal.control import (
    FusionSCWallSurfaceSpec,
    finite_difference_boundary_response_system,
    manifold_strike_seed_bundles,
    trace_wall_strikes_field,
    wall_heat_flux_metrics,
    wall_heat_flux_observables,
    wall_heat_state_from_strikes,
)
from pyna.toroidal.geometry import project_points_to_toroidal_surface


TWOPI = 2.0 * np.pi
R0 = 1.8
WALL_MINOR = 0.56


def _wall() -> FusionSCWallSurfaceSpec:
    phi = np.linspace(0.0, TWOPI, 64, endpoint=False)
    theta = np.linspace(0.0, TWOPI, 160, endpoint=False)
    R = np.broadcast_to(R0 + WALL_MINOR * np.cos(theta), (phi.size, theta.size)).copy()
    Z = np.broadcast_to(WALL_MINOR * np.sin(theta), (phi.size, theta.size)).copy()
    return FusionSCWallSurfaceSpec(
        phi_values=phi,
        R=R,
        Z=Z,
        toroidal_period=TWOPI,
    )


def _manifold_payload() -> dict[str, object]:
    distance = np.geomspace(2.0e-4, 2.0e-2, 24)
    side = np.repeat(np.asarray([-1.0, 1.0]), distance.size)
    radius = np.tile(distance, 2)
    order = np.tile(np.arange(distance.size), 2)
    return {
        "manifold_origin_label": "X0",
        "origin_phi": 0.0,
        "manifold_field_period": TWOPI,
        "manifold_field_period_source": "synthetic full return",
        "u_seed_R": R0 + 0.12 + 0.15 * side * radius,
        "u_seed_Z": side * radius,
        "u_seed_distance": radius,
        "u_seed_side": side,
        "u_seed_order": order,
        "s_seed_R": R0 + 0.12 + side * radius,
        "s_seed_Z": 0.12 * side + 0.15 * side * radius,
        "s_seed_distance": radius,
        "s_seed_side": side,
        "s_seed_order": order,
        "orbit_id": 0,
        "point_index": 0,
    }


def _power_weights(context):
    coordinate = np.asarray(context["source_coordinate"], dtype=float)
    profile = np.exp(-coordinate / 0.008)
    return {
        "weights": 0.25 * profile / np.sum(profile),
        "weight_kind": "power",
        "quantitative": True,
        "provenance": "synthetic normalized flux-tube power",
    }


def _trace_function(controls):
    commands = np.asarray(controls, dtype=float)

    def trace(
        _field,
        R,
        Z,
        phi_start,
        _max_turns,
        _DPhi,
        _wall_phi,
        _wall_R,
        _wall_Z,
        *,
        extend_phi,
        direction,
    ):
        del extend_phi
        R = np.asarray(R, dtype=float)
        Z = np.asarray(Z, dtype=float)
        n_seed = R.size
        coordinate = np.linspace(-1.0, 1.0, n_seed)
        trace_sign = 1.0 if direction == "+" else -1.0
        branch_side = np.sign(np.mean(Z))
        if branch_side == 0.0:
            branch_side = np.sign(np.mean(R) - (R0 + 0.12)) or 1.0
        center_theta = 1.10 + 0.14 * branch_side + 0.05 * trace_sign + 0.58 * commands[0]
        half_span = 0.08 + 0.32 * (commands[1] + 0.8) / 1.6
        theta_hit = center_theta + half_span * coordinate
        phi_hit = phi_start + trace_sign * (0.65 + 0.22 * coordinate + 0.10 * commands[0])
        hit_R = R0 + WALL_MINOR * np.cos(theta_hit)
        hit_Z = WALL_MINOR * np.sin(theta_hit)
        hit = np.column_stack((hit_R, hit_Z, phi_hit))
        suffix = "plus" if direction == "+" else "minus"
        return {
            f"Lc_{suffix}": 8.0 + 5.0 * (coordinate + 1.0),
            f"hit_{suffix}": hit,
            f"term_{suffix}": np.ones(n_seed, dtype=int),
        }

    return trace


def _evaluate(controls, wall, bundles):
    strikes = trace_wall_strikes_field(
        SimpleNamespace(controls=np.asarray(controls, dtype=float)),
        bundles,
        wall,
        max_turns=40,
        DPhi=0.01,
        trace_function=_trace_function(controls),
    )
    state = wall_heat_state_from_strikes(
        strikes,
        wall,
        phi_edges=np.linspace(0.0, TWOPI, 65),
        s_edges=np.linspace(0.0, 1.0, 97),
    )
    rows = wall_heat_flux_observables(
        state.heat,
        phi_values=state.phi_values,
        s_values=state.s_values,
        cell_areas=state.cell_areas,
        quantities=("centroid_s", "rms_width_s", "peak_flux"),
        weights=(45.0, 70.0, 0.002),
        prefix="strike",
        metadata={"source": "synthetic X-manifold first-wall hits"},
    )
    return state, strikes, rows


def _wall_surface(state) -> WallSurfaceHeat:
    phi = np.asarray(state.phi_values, dtype=float)
    s = np.asarray(state.s_values, dtype=float)
    theta = TWOPI * s
    wall_R = np.broadcast_to(R0 + WALL_MINOR * np.cos(theta), state.heat.shape).copy()
    wall_Z = np.broadcast_to(WALL_MINOR * np.sin(theta), state.heat.shape).copy()
    wall_xyz = np.stack(
        (
            wall_R * np.cos(phi[:, None]),
            wall_R * np.sin(phi[:, None]),
            wall_Z,
        ),
        axis=-1,
    )
    return WallSurfaceHeat(
        heat=state.heat * state.cell_areas,
        heat_flux=state.heat,
        area=state.cell_areas,
        wall_phi=phi,
        wall_R=wall_R,
        wall_Z=wall_Z,
        wall_xyz=wall_xyz,
        section_s=np.broadcast_to(s[None, :], state.heat.shape).copy(),
    )


def _strike_coordinates(strikes, wall):
    R = np.concatenate([row.R for row in strikes])
    Z = np.concatenate([row.Z for row in strikes])
    phi = np.concatenate([row.phi for row in strikes])
    projection = project_points_to_toroidal_surface(R, Z, phi, wall)
    labels = np.concatenate(
        [np.full(row.R.size, index, dtype=int) for index, row in enumerate(strikes)]
    )
    return projection.phi, projection.s, labels, R, Z


def _plot_report(baseline, controlled, target, solve, wall, bundles, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    baseline_state, baseline_strikes, baseline_rows = baseline
    controlled_state, controlled_strikes, controlled_rows = controlled
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.0), constrained_layout=True)

    ax = axes[0, 0]
    for bundle in bundles:
        color = "#d55e00" if "unstable" in bundle.label else "#0072b2"
        ax.plot(bundle.R, bundle.Z, color=color, linewidth=1.2, alpha=0.86)
        ax.scatter(bundle.R[::4], bundle.Z[::4], s=10, color=color)
    ax.scatter([R0 + 0.12], [0.0], marker="x", s=58, color="#c43b4d", label="X point")
    theta = np.linspace(0.0, TWOPI, 300)
    ax.plot(R0 + WALL_MINOR * np.cos(theta), WALL_MINOR * np.sin(theta), color="0.25", linewidth=1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("local stable/unstable seed branches")
    ax.legend(frameon=False, fontsize=8)

    for axis, strikes, title in (
        (axes[0, 1], baseline_strikes, "baseline first-wall strikes"),
        (axes[0, 2], controlled_strikes, "controlled first-wall strikes"),
    ):
        phi, s, branch, _R, _Z = _strike_coordinates(strikes, wall)
        axis.scatter(np.mod(phi, TWOPI), s, c=branch, cmap="tab10", s=10, alpha=0.82)
        axis.axhspan(target["s_min"], target["s_max"], color="#2a9d8f", alpha=0.14)
        axis.set_xlim(0.0, TWOPI)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("wall phi [rad]")
        axis.set_ylabel("wall arclength fraction")
        axis.set_title(title)

    positive = np.concatenate(
        (
            baseline_state.heat[baseline_state.heat > 0.0],
            controlled_state.heat[controlled_state.heat > 0.0],
        )
    )
    norm = LogNorm(vmin=max(float(np.percentile(positive, 5.0)), 1.0e-12), vmax=float(np.max(positive)))
    heat_mesh = None
    for axis, state, title in (
        (axes[1, 0], baseline_state, "baseline strike heat flux"),
        (axes[1, 1], controlled_state, "controlled strike heat flux"),
    ):
        phi_edges = np.linspace(0.0, TWOPI, state.heat.shape[0] + 1)
        s_edges = np.linspace(0.0, 1.0, state.heat.shape[1] + 1)
        heat_mesh = axis.pcolormesh(phi_edges, s_edges, state.heat.T, cmap="inferno", norm=norm, shading="auto")
        axis.set_xlabel("wall phi [rad]")
        axis.set_ylabel("wall arclength fraction")
        axis.set_title(title)
    fig.colorbar(heat_mesh, ax=[axes[1, 0], axes[1, 1]], label="synthetic deposited heat flux")

    ax = axes[1, 2]
    labels = ("centroid s", "RMS width s", "peak flux")
    current = np.asarray(baseline_rows.values, dtype=float)
    actual = np.asarray(controlled_rows.values, dtype=float)
    target_values = np.asarray(target["values"], dtype=float)
    scale = np.maximum(np.abs(current), 1.0e-12)
    x = np.arange(3)
    ax.bar(x - 0.25, current / scale, width=0.25, color="0.45", label="baseline")
    ax.bar(x, target_values / scale, width=0.25, color="#2a9d8f", label="target")
    ax.bar(x + 0.25, actual / scale, width=0.25, color="#e76f51", label="nonlinear check")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("value / baseline")
    ax.set_title(f"bounded solve, cond={solve.diagnostics.condition_number:.2f}")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color="0.88", linewidth=0.6)

    fig.suptitle("Synthetic active strike-line heat control from X-point manifolds", fontsize=14)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    out = Path("~/MCFdata/pyna/strike_heat_control").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    wall = _wall()
    bundles = manifold_strike_seed_bundles([_manifold_payload()], weight_builder=_power_weights)
    baseline = _evaluate((0.0, 0.0), wall, bundles)
    step = 0.05
    plus = [_evaluate((step, 0.0), wall, bundles), _evaluate((0.0, step), wall, bundles)]
    minus = [_evaluate((-step, 0.0), wall, bundles), _evaluate((0.0, -step), wall, bundles)]
    system = finite_difference_boundary_response_system(
        baseline[2],
        [row[2] for row in plus],
        [row[2] for row in minus],
        steps=step,
        control_labels=("strike_phase_trim", "strike_width_trim"),
        metadata={"forward_model": "synthetic X-manifold first-wall heat"},
    )
    current = np.asarray(baseline[2].values, dtype=float)
    target_values = np.asarray(
        [current[0] + 0.045, current[1] * 1.35, current[2] * 0.78],
        dtype=float,
    )
    solve = system.solve(
        dict(zip(system.labels, target_values)),
        bounds=((-0.8, -0.8), (0.8, 0.8)),
        regularization=2.0e-4,
        control_scale=(1.0, 1.0),
    )
    controlled = _evaluate(solve.controls, wall, bundles)
    target = {
        "values": target_values,
        "s_min": float(target_values[0] - 1.5 * target_values[1]),
        "s_max": float(target_values[0] + 1.5 * target_values[1]),
    }
    _plot_report(
        baseline,
        controlled,
        target,
        solve,
        wall,
        bundles,
        out / "synthetic_manifold_strike_heat_control.png",
    )

    hit_R = np.concatenate([row.R for row in controlled[1]])
    hit_Z = np.concatenate([row.Z for row in controlled[1]])
    hit_phi = np.concatenate([row.phi for row in controlled[1]])
    camera_fig, _axes, _collections = plot_wall_heat_camera_views(
        _wall_surface(controlled[0]),
        collision_hits=(hit_R, hit_Z, hit_phi),
        value="heat_flux",
        log_scale=True,
        colorbar_label="synthetic manifold-guided heat flux",
    )
    camera_fig.suptitle(
        "Controlled strike heat on a transparent synthetic wall",
        fontsize=14,
        color="#e8edf4",
    )
    camera_fig.savefig(out / "synthetic_manifold_strike_heat_camera.png", dpi=220, bbox_inches="tight")

    baseline_metrics = wall_heat_flux_metrics(
        baseline[0].heat,
        phi_values=baseline[0].phi_values,
        s_values=baseline[0].s_values,
        cell_areas=baseline[0].cell_areas,
    )
    controlled_metrics = wall_heat_flux_metrics(
        controlled[0].heat,
        phi_values=controlled[0].phi_values,
        s_values=controlled[0].s_values,
        cell_areas=controlled[0].cell_areas,
    )
    summary = {
        "schema": "pyna_synthetic_manifold_strike_heat_control_v1",
        "controls": solve.controls.tolist(),
        "control_labels": list(system.control_labels),
        "response_condition_number": system.diagnostics.condition_number,
        "response_rank": system.diagnostics.rank,
        "target": dict(zip(system.labels, target_values.tolist())),
        "predicted": dict(zip(system.labels, solve.predicted.tolist())),
        "actual": dict(zip(system.labels, controlled[2].values.tolist())),
        "baseline_metrics": {
            "centroid_s": baseline_metrics.centroid_s,
            "rms_width_s": baseline_metrics.rms_width_s,
            "peak_flux": baseline_metrics.peak_flux,
        },
        "controlled_metrics": {
            "centroid_s": controlled_metrics.centroid_s,
            "rms_width_s": controlled_metrics.rms_width_s,
            "peak_flux": controlled_metrics.peak_flux,
        },
        "power_audit": dict(controlled[0].metadata),
        "physics_scope": "public synthetic geometry and synthetic flux-tube powers",
    }
    (out / "synthetic_manifold_strike_heat_control_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"outputs: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
