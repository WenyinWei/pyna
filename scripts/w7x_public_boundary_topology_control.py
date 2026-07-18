#!/usr/bin/env python3
"""Run a public W7-X virtual-dipole boundary topology control benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyna.plot import (
    plot_boundary_response_matrix_audit,
    plot_boundary_response_optimization_history,
    plot_boundary_topology_control_audit,
)
from pyna.toroidal.coils import (
    boundary_dipole_local_actuator_array_from_surface,
    boundary_dipole_mode_actuator_array_from_surface,
    stack_boundary_dipole_actuator_arrays,
)
from pyna.toroidal.control import (
    BoundaryHeatTargetRegion,
    BoundaryPlasmaResponseInput,
    BoundaryTopologyCaseBackend,
    BoundaryTopologyObservableSpec,
    ReducedSpectralHeatModel,
    boundary_topology_case_observable_builder,
    build_boundary_dipole_spectrum_library,
    format_boundary_topology_control_summary,
    make_boundary_topology_control_problem,
    solve_boundary_topology_control,
    vmec_boundary_topology_case_from_wout,
    wall_heat_flux_metrics,
)


DEFAULT_WOUT = Path("~/MCFdata/W7X_public/stagextender_beta1/wout_std_scp00_beta1.nc")
DEFAULT_OUT = Path("~/MCFdata/W7X_public/boundary_topology_control")
MODES = ((8, 7), (9, 8))


def _request(controls, labels):
    return BoundaryPlasmaResponseInput(controls=np.asarray(controls, dtype=float), control_labels=labels)


def _screened_response(case, request, vacuum_tilde):
    del request
    radial = case.radial_labels[None, :, None]
    gain = 0.06 + 0.94 * radial**4
    phase_lag = 0.08 * radial
    return {
        "tilde_b1": vacuum_tilde * gain * np.exp(1j * phase_lag),
        "metadata": {
            "response_model": "screened_linear_response_surrogate",
            "quantitative_mhd": False,
        },
    }


def _observable_weights(target_rows):
    index = {label: idx for idx, label in enumerate(target_rows.labels)}
    island_weights = []
    for m, n in MODES:
        for quantity, floor in (
            ("half_width", 0.015),
            ("coefficient_real", 2.0e-5),
            ("coefficient_imag", 2.0e-5),
        ):
            value = abs(float(target_rows.values[index[f"island.m{m}.n{n}.{quantity}"]]))
            island_weights.append(1.0 / max(value, floor) ** 2)
    chaos_value = abs(float(target_rows.values[index["chaos.edge_overlap"]]))
    return {
        "island": np.asarray(island_weights, dtype=float),
        "chaos": np.asarray([1.0 / max(chaos_value, 0.2) ** 2], dtype=float),
        "heat": np.asarray([
            1.0 / 0.5**2,
            1.0 / 0.6**2,
            1.0 / 0.03**2,
            1.0 / 0.015**2,
        ]),
    }


def _state_payload(state):
    heat_metrics = None
    if state.heat is not None:
        metrics = wall_heat_flux_metrics(
            state.heat.heat,
            phi_values=state.heat.phi_values,
            s_values=state.heat.s_values,
            cell_areas=state.heat.cell_areas,
        )
        heat_metrics = {name: float(getattr(metrics, name)) for name in metrics.__dataclass_fields__}
    return {
        "chains": [
            {
                "m": int(chain.m),
                "n": int(chain.n),
                "radial_label": float(chain.radial_label),
                "coefficient_real": float(np.real(chain.coefficient)),
                "coefficient_imag": float(np.imag(chain.coefficient)),
                "phase": float(np.angle(chain.coefficient)),
                "half_width": float(chain.half_width),
            }
            for chain in state.chains
        ],
        "chaotic_intervals": [
            {
                "inner": float(interval.inner),
                "outer": float(interval.outer),
                "width": float(interval.width),
                "max_sigma": float(interval.max_sigma),
            }
            for interval in state.chaotic_intervals
        ],
        "heat_metrics": heat_metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wout", type=Path, default=DEFAULT_WOUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-radial", type=int, default=10)
    parser.add_argument("--n-phi", type=int, default=32)
    parser.add_argument("--n-theta", type=int, default=64)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--vacuum", action="store_true", help="Disable the documented screened-response surrogate.")
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = args.out_dir.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    case = vmec_boundary_topology_case_from_wout(
        args.wout.expanduser(),
        name="W7-X public",
        n_radial=args.n_radial,
        n_phi=args.n_phi,
        n_theta=args.n_theta,
    )
    phi_indices = range(0, case.phi_vals.size, 2)
    theta_stride = max(1, case.theta_vals.size // 16)
    theta_indices = range(0, case.theta_vals.size, theta_stride)
    mode_array = boundary_dipole_mode_actuator_array_from_surface(
        case.R_surf,
        case.Z_surf,
        case.phi_vals,
        case.theta_vals,
        MODES,
        phi_indices=phi_indices,
        theta_indices=theta_indices,
        radius=0.07,
        unit_moment=300.0,
        clearance=0.18,
        lower_bound=-1.0,
        upper_bound=1.0,
    )
    trim_array = boundary_dipole_local_actuator_array_from_surface(
        case.R_surf,
        case.Z_surf,
        case.phi_vals,
        case.theta_vals,
        sites=((0, case.theta_vals.size // 4), (0, 3 * case.theta_vals.size // 4)),
        radius=0.06,
        unit_moment=30.0,
        clearance=0.16,
        lower_bound=-0.7,
        upper_bound=0.7,
    )
    actuators = stack_boundary_dipole_actuator_arrays(
        (mode_array, trim_array),
        metadata={"case": "W7-X public virtual dipole benchmark"},
    )

    def progress(index, total, label):
        print(f"dipole response {index}/{total}: {label}")

    library = build_boundary_dipole_spectrum_library(
        case,
        actuators,
        m_max=12,
        n_max=9,
        progress=progress,
    )
    heat_model = ReducedSpectralHeatModel(
        phi_values=np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False),
        s_values=np.linspace(0.0, 1.0, 120),
        base_total_power=10.0,
        base_center_s=0.55,
        base_sigma_s=0.047,
        phase_excursion_s=0.105,
        chaos_broadening=0.28,
        control_center_s={
            trim_array.control_labels[0]: 0.09,
            trim_array.control_labels[1]: -0.09,
        },
        control_sigma_s={
            trim_array.control_labels[0]: 0.012,
            trim_array.control_labels[1]: -0.010,
        },
        control_power_fraction={
            trim_array.control_labels[0]: 0.08,
            trim_array.control_labels[1]: -0.05,
        },
    )
    feedback = None if args.vacuum else _screened_response
    preliminary_spec = BoundaryTopologyObservableSpec(
        resonant_modes=MODES,
        chaos_regions=((0.52, 1.0),),
        chaos_labels=("edge_overlap",),
        heat_regions=(BoundaryHeatTargetRegion("target_band", (0.52, 0.72), weight=100.0),),
        core_radial_max=0.42,
        core_field_weights=(1.0e6, 1.0e5),
    )
    backend = BoundaryTopologyCaseBackend(
        library=library,
        n_values=(7, 8),
        m_values={7: (8,), 8: (9,)},
        sigma_threshold=0.4,
        heat_model=heat_model,
        plasma_feedback=feedback,
    )
    desired_controls = np.array([0.35, -0.18, 0.30, 0.16, 0.35, -0.22])
    desired_snapshot = backend.evaluate(_request(desired_controls, actuators.control_labels))
    preliminary_rows = boundary_topology_case_observable_builder(preliminary_spec)(
        desired_snapshot,
        _request(desired_controls, actuators.control_labels),
    )
    weights = _observable_weights(preliminary_rows)
    observable_spec = BoundaryTopologyObservableSpec(
        resonant_modes=MODES,
        resonant_weights=weights["island"],
        chaos_regions=((0.52, 1.0),),
        chaos_labels=("edge_overlap",),
        chaos_weights=weights["chaos"],
        heat_weights=weights["heat"],
        heat_regions=(BoundaryHeatTargetRegion("target_band", (0.52, 0.72), weight=100.0),),
        core_radial_max=0.42,
        core_field_weights=(1.0e6, 1.0e5),
    )
    desired_rows = boundary_topology_case_observable_builder(observable_spec)(
        desired_snapshot,
        _request(desired_controls, actuators.control_labels),
    )
    target = {
        label: float(value)
        for label, value in zip(desired_rows.labels, desired_rows.values)
        if not label.startswith("core.")
    }
    problem = make_boundary_topology_control_problem(
        library,
        observable_spec,
        target,
        initial_controls=np.zeros(len(actuators.actuators)),
        n_values=(7, 8),
        m_values={7: (8,), 8: (9,)},
        heat_model=heat_model,
        plasma_feedback=feedback,
        sigma_threshold=0.4,
        steps=0.08,
        n_iterations=7,
        regularization=2.0e-3,
        line_search=(1.0, 0.5, 0.25, 0.125, 0.0625),
        convergence_tolerance=1.0e-5,
        metadata={"benchmark": "W7-X public virtual dipole boundary control"},
    )
    result = solve_boundary_topology_control(problem, top_n_residuals=20)
    initial_state, _ = problem.backend.forward_state(_request(problem.initial_controls, actuators.control_labels))
    final_state, _ = problem.backend.forward_state(_request(result.final_controls, actuators.control_labels))

    fig, _axes = plot_boundary_topology_control_audit(
        case,
        actuators,
        initial_state,
        final_state,
        result=result,
        modes=MODES,
        out_path=out / "w7x_public_boundary_topology_control_audit.png",
        save_dpi=args.dpi,
        title="W7-X public virtual-dipole boundary topology and heat control",
    )
    plt.close(fig)
    if result.final_system is not None:
        fig, _axes = plot_boundary_response_matrix_audit(
            result.final_system,
            out_path=out / "w7x_public_boundary_response_matrix_audit.png",
            save_dpi=args.dpi,
            title="W7-X public joint topology/heat response matrix",
        )
        plt.close(fig)
    fig, _axes = plot_boundary_response_optimization_history(
        result.optimization,
        out_path=out / "w7x_public_boundary_optimization_history.png",
        save_dpi=args.dpi,
        title="W7-X public virtual-dipole nonlinear optimization history",
    )
    plt.close(fig)

    text_summary = "\n".join(
        [
            "W7-X public virtual-dipole boundary control benchmark",
            "response: screened linear surrogate (not quantitative MHD)",
            "heat: reduced spectral diffusion surrogate (not quantitative transport)",
            "overlap layer: sigma >= 0.4 screening threshold; strong Chirikov overlap requires sigma >= 1",
            "",
            format_boundary_topology_control_summary(result, top_n_residuals=12),
        ]
    )
    (out / "w7x_public_boundary_topology_control_summary.txt").write_text(text_summary + "\n", encoding="utf-8")
    payload = {
        "schema": "pyna_w7x_public_boundary_topology_control_v1",
        "case": "W7-X public",
        "source_id": case.metadata.get("source_id"),
        "response_model": "vacuum" if args.vacuum else "screened_linear_response_surrogate",
        "response_model_quantitative_mhd": False,
        "heat_model": "reduced_spectral_diffusive_heat",
        "heat_model_quantitative_transport": False,
        "overlap_sigma_threshold": 0.4,
        "overlap_interpretation": "weak-overlap screening layer; strong Chirikov overlap requires sigma >= 1 and nonlinear DPk verification",
        "target_kind": "feasible inverse benchmark generated from a hidden bounded dipole command vector",
        "coordinate_system": case.coordinate_system,
        "radial_coordinate": case.radial_coordinate,
        "actuator_waveform": "dipole currents cos(m*theta_PEST-n0*phi) and sin(m*theta_PEST-n0*phi)",
        "nardon_fourier_basis": "exp(i*(m*theta_PEST+n_N*phi))",
        "positive_q_resonant_branch": "n_N=-n0",
        "control_labels": list(actuators.control_labels),
        "desired_controls": desired_controls.tolist(),
        "final_controls": result.final_controls.tolist(),
        "initial_residual_norm": float(result.validation.initial_weighted_residual_norm),
        "final_residual_norm": float(result.validation.final_weighted_residual_norm),
        "residual_reduction_fraction": float(result.validation.residual_reduction_fraction),
        "matrix_rank": None if result.final_system is None else int(result.final_system.diagnostics.rank),
        "matrix_condition_number": None if result.final_system is None else float(result.final_system.diagnostics.condition_number),
        "initial_state": _state_payload(initial_state),
        "final_state": _state_payload(final_state),
    }
    (out / "w7x_public_boundary_topology_control_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        out / "w7x_public_boundary_topology_control_state.npz",
        initial_controls=np.asarray(problem.initial_controls),
        desired_controls=desired_controls,
        final_controls=np.asarray(result.final_controls),
        heat_phi=initial_state.heat.phi_values,
        heat_s=initial_state.heat.s_values,
        initial_heat=initial_state.heat.heat,
        final_heat=final_state.heat.heat,
    )
    print(text_summary)
    print(f"outputs: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
