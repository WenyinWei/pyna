#!/usr/bin/env python3
"""Control the W7-X edge island chain with the two measured vacuum fields."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import pickle
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyna.plot import PoincareCurvedIslandBar, draw_poincare_curved_island_bars
from pyna.topo._monodromy_classification import classify_monodromy_2x2
from pyna.toroidal.control.boundary_field_basis import (
    CylindricalGridFieldControlBasis,
    CylindricalGridFieldCandidate,
    boundary_field_actuator_array_from_grid_fields,
    cylindrical_vector_field_from_array,
    load_cylindrical_vector_field_npz,
)
from pyna.toroidal.control.boundary_perturbation_candidates import (
    perturbation_candidate_nardon_response,
)
from pyna.toroidal.control.boundary_topology_cases import (
    BoundaryTopologyObservableSpec,
    build_boundary_perturbation_spectrum_library,
    extend_boundary_topology_case_to_resonance,
    make_boundary_topology_control_problem,
    vmec_boundary_topology_case_from_wout,
)
from pyna.toroidal.control.boundary_topology_control import solve_boundary_topology_control
from pyna.toroidal.control.boundary_topology_design import boundary_dpk_growth_metrics
from pyna.toroidal.control.topoquest_fpt import (
    TopoquestFPTBetaRampSpec,
    TopoquestFPTCachedResponseBasis,
    TopoquestFPTPlasmaFeedbackAdapter,
    diagnose_topoquest_fpt_capability,
)
from pyna.toroidal.flt.numba_poincare import (
    find_fixed_points_batch_span_field,
    trace_orbit_along_phi_field,
    trace_poincare_batch_field,
)
from pyna.toroidal.flt.island_chain import (
    BoundaryIslandFixedPoint,
    trace_fixed_point_manifolds_field,
)
from pyna.toroidal.perturbation_spectrum import (
    analyze_resonant_island_chains,
    radial_perturbation_component,
    surface_field_alignment_diagnostics,
)


TWOPI = 2.0 * np.pi
MODE = (5, 5)
DPHI = (TWOPI / 5.0) / 256.0
DEFAULT_ROOT = Path("~/MCFdata/w7x")
DEFAULT_FPT = DEFAULT_ROOT / "w7x_prefect_fullres_beta002_divJbdy_nc5em12_equil_20260616_v1/w7x_beta_final_state.npz"


def _source_id(path: Path, length: int = 12) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def _grid_field_from_npy(path: Path, reference, name: str):
    values = np.load(path, allow_pickle=False)
    return cylindrical_vector_field_from_array(
        values,
        reference.R,
        reference.Z,
        reference.Phi,
        component_order=("BR", "BZ", "BPhi"),
        nfp=reference.nfp,
        name=name,
    )


def _load_fpt_increment(path: Path | None, reference):
    if path is None or not path.exists():
        return None, None
    with np.load(path, allow_pickle=False) as data:
        required = ("R", "Z", "Phi", "delta_B_R", "delta_B_Z", "delta_B_Phi")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"FPT state is missing arrays: {missing}")
        values = np.stack(
            [data["delta_B_R"], data["delta_B_Z"], data["delta_B_Phi"]],
            axis=-1,
        )
        n_fp = int(np.asarray(data["n_fp"]).item()) if "n_fp" in data else reference.nfp
        beta = float(np.asarray(data["beta"]).item()) if "beta" in data else float("nan")
        accepted = bool(np.asarray(data["accepted"]).item()) if "accepted" in data else False
        field = cylindrical_vector_field_from_array(
            values,
            data["R"],
            data["Z"],
            data["Phi"],
            component_order=("BR", "BZ", "BPhi"),
            nfp=n_fp,
            name="FPT pressure-driven plasma increment",
        )
    # Grid arithmetic is also the strict compatibility check.
    _ = reference + field
    return field, {
        "source_id": _source_id(path),
        "beta": beta,
        "accepted": accepted,
        "production_ready": False,
        "response_scope": "pressure_driven_background_only",
        "external_control_screening_solved": False,
    }


def _load_healed_cache(path: Path, topoquest_root: Path):
    helper = topoquest_root / "scripts/w7x"
    for item in (helper, topoquest_root):
        if str(item) not in sys.path:
            sys.path.insert(0, str(item))
    with path.open("rb") as handle:
        return pickle.load(handle)


def _request_chain(state):
    rows = [chain for chain in state.chains if (int(chain.m), int(chain.n)) == MODE]
    if len(rows) != 1:
        raise RuntimeError(f"expected one {MODE} resonant chain, found {len(rows)}")
    return rows[0]


def _response_mode_coefficient(response, case) -> complex:
    chains = analyze_resonant_island_chains(
        response.spectrum,
        case.q_profile,
        n=MODE[1],
        radial_labels=case.radial_labels,
        m_values=(MODE[0],),
    )
    if len(chains) != 1:
        raise RuntimeError(f"expected one {MODE} response chain, found {len(chains)}")
    return complex(chains[0].coefficient)


def _complex_payload(value: complex) -> dict[str, float]:
    coefficient = complex(value)
    return {
        "real": float(coefficient.real),
        "imag": float(coefficient.imag),
        "abs": float(abs(coefficient)),
        "phase_deg": float(np.degrees(np.angle(coefficient))),
    }


def _nardon_phase_shift_deg(coefficient: complex, reference: complex, m: int) -> float:
    """Return the fixed-phi O-phase prediction ``-delta arg(b)/m``."""

    if int(m) <= 0 or abs(complex(reference)) == 0.0:
        raise ValueError("m and the reference coefficient must be nonzero")
    coefficient_phase_shift = float(np.angle(complex(coefficient) / complex(reference)))
    return float(-np.degrees(coefficient_phase_shift) / int(m))


def _phase_closure_error_deg(newton_shift: float, predicted_shift: float, m: int) -> float:
    """Return a branch-wrapped Newton-minus-Nardon phase error."""

    if int(m) <= 0:
        raise ValueError("m must be positive")
    return float(
        np.degrees(
            np.angle(
                np.exp(
                    1j
                    * int(m)
                    * np.deg2rad(float(newton_shift) - float(predicted_shift))
                )
            )
            / int(m)
        )
    )


def _phase_b0_alignment_gate(
    alignment,
    *,
    radial_ratio_limit: float,
    iota_rms_limit: float,
) -> dict[str, object]:
    """Audit whether the sampled field can share the healed ``B0`` phase chart."""

    radial_limit = float(radial_ratio_limit)
    iota_limit = float(iota_rms_limit)
    if not np.isfinite(radial_limit) or radial_limit <= 0.0:
        raise ValueError("radial_ratio_limit must be positive and finite")
    if not np.isfinite(iota_limit) or iota_limit <= 0.0:
        raise ValueError("iota_rms_limit must be positive and finite")
    radial_rms = float(alignment.edge_radial_ratio_rms)
    iota_rms = float(alignment.iota_profile_error_rms)
    radial_passed = bool(np.isfinite(radial_rms) and radial_rms <= radial_limit)
    iota_passed = bool(np.isfinite(iota_rms) and iota_rms <= iota_limit)
    return {
        "edge_radial_ratio_rms": radial_rms,
        "edge_radial_ratio_limit": radial_limit,
        "edge_radial_ratio_passed": radial_passed,
        "iota_profile_error_rms": iota_rms,
        "iota_profile_error_limit": iota_limit,
        "iota_profile_passed": iota_passed,
        "passed": radial_passed and iota_passed,
    }


def _constant_spectral_amplitude_controls(
    coefficient_phase_shift_deg: float,
    reference_coefficient: complex,
    response_matrix,
) -> np.ndarray:
    """Return minimum-norm controls rotating a coefficient at fixed amplitude."""

    reference = complex(reference_coefficient)
    if abs(reference) == 0.0:
        raise ValueError("reference_coefficient must be nonzero")
    matrix = np.asarray(response_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("response_matrix must be a finite 2 by n_controls array")
    if np.linalg.matrix_rank(matrix) < 2:
        raise ValueError("response_matrix must span the resonant complex plane")
    target = abs(reference) * np.exp(
        1j * (np.angle(reference) + np.deg2rad(float(coefficient_phase_shift_deg)))
    )
    rhs = np.asarray([target.real - reference.real, target.imag - reference.imag])
    controls, _residuals, _rank, _singular_values = np.linalg.lstsq(matrix, rhs, rcond=None)
    return np.asarray(controls, dtype=float)


def _calibrate_newton_chain_phase_target(
    target_shift_deg: float,
    reference_coefficient: complex,
    response_matrix,
    control_bounds,
    phase_evaluator,
    *,
    coefficient_phase_limit_deg: float = 30.0,
    feasibility_samples: int = 1201,
    phase_tolerance_deg: float = 0.02,
    max_iterations: int = 16,
) -> dict[str, object]:
    """Calibrate an O-chain phase target along a constant-width spectral path.

    ``phase_evaluator(controls)`` is authoritative: it must return the Newton
    O-chain phase shift in degrees.  The Nardon coefficient only parameterizes
    a bounded path that preserves ``|tilde b^1_mn|``; it is not used as the
    phase sensor.
    """

    target = float(target_shift_deg)
    if not np.isfinite(target) or target == 0.0:
        raise ValueError("target_shift_deg must be finite and nonzero")
    if not callable(phase_evaluator):
        raise TypeError("phase_evaluator must be callable")
    matrix = np.asarray(response_matrix, dtype=float)
    bounds = np.asarray(control_bounds, dtype=float)
    if bounds.shape != (matrix.shape[1], 2):
        raise ValueError("control_bounds must have shape (n_controls, 2)")
    if np.any(~np.isfinite(bounds)) or np.any(bounds[:, 1] < bounds[:, 0]):
        raise ValueError("control_bounds must be finite lower/upper pairs")
    limit = float(coefficient_phase_limit_deg)
    samples = int(feasibility_samples)
    if not np.isfinite(limit) or limit <= 0.0 or samples < 3:
        raise ValueError("coefficient phase scan must have a positive limit and at least three samples")

    direction = 1.0 if target > 0.0 else -1.0
    phase_grid = direction * np.linspace(0.0, limit, samples)
    feasible: list[tuple[float, np.ndarray]] = []
    for phase_deg in phase_grid:
        controls = _constant_spectral_amplitude_controls(
            phase_deg,
            reference_coefficient,
            matrix,
        )
        if np.all(controls >= bounds[:, 0] - 1.0e-12) and np.all(
            controls <= bounds[:, 1] + 1.0e-12
        ):
            feasible.append((float(phase_deg), controls))
        elif feasible:
            break
    if len(feasible) < 2:
        raise RuntimeError("no nonzero constant-amplitude phase path lies inside control bounds")

    evaluations: dict[tuple[float, ...], float] = {}

    def evaluate(controls: np.ndarray) -> float:
        key = tuple(float(value) for value in np.round(controls, decimals=13))
        if key not in evaluations:
            value = float(phase_evaluator(np.asarray(controls, dtype=float)))
            if not np.isfinite(value):
                raise ValueError("phase_evaluator returned a non-finite value")
            evaluations[key] = value
        return evaluations[key]

    phase_lo, controls_lo = feasible[0]
    phase_hi, controls_hi = feasible[-1]
    value_lo = evaluate(controls_lo)
    value_hi = evaluate(controls_hi)
    residual_lo = value_lo - target
    residual_hi = value_hi - target
    if residual_lo * residual_hi > 0.0:
        raise RuntimeError(
            "Newton phase target is outside the bounded constant-amplitude response range: "
            f"target={target:+.6g} deg, endpoint={value_hi:+.6g} deg"
        )

    best = (abs(residual_lo), phase_lo, controls_lo, value_lo)
    if abs(residual_hi) < best[0]:
        best = (abs(residual_hi), phase_hi, controls_hi, value_hi)
    iterations = 0
    while iterations < int(max_iterations) and best[0] > float(phase_tolerance_deg):
        iterations += 1
        phase_mid = 0.5 * (phase_lo + phase_hi)
        controls_mid = _constant_spectral_amplitude_controls(
            phase_mid,
            reference_coefficient,
            matrix,
        )
        value_mid = evaluate(controls_mid)
        residual_mid = value_mid - target
        if abs(residual_mid) < best[0]:
            best = (abs(residual_mid), phase_mid, controls_mid, value_mid)
        if residual_lo * residual_mid <= 0.0:
            phase_hi, controls_hi, value_hi, residual_hi = (
                phase_mid,
                controls_mid,
                value_mid,
                residual_mid,
            )
        else:
            phase_lo, controls_lo, value_lo, residual_lo = (
                phase_mid,
                controls_mid,
                value_mid,
                residual_mid,
            )

    _error, coefficient_phase, controls, achieved = best
    coefficient = complex(reference_coefficient) + complex(
        *(matrix @ np.asarray(controls, dtype=float))
    )
    relative_amplitude_error = float(
        abs(abs(coefficient) - abs(complex(reference_coefficient)))
        / abs(complex(reference_coefficient))
    )
    return {
        "target_shift_deg": target,
        "achieved_shift_deg": float(achieved),
        "error_deg": float(achieved - target),
        "controls": np.asarray(controls, dtype=float),
        "coefficient_phase_shift_deg": float(coefficient_phase),
        "relative_spectral_amplitude_error": relative_amplitude_error,
        "phase_tolerance_deg": float(phase_tolerance_deg),
        "iterations": int(iterations),
        "evaluations": int(len(evaluations)),
        "accepted": bool(abs(achieved - target) <= float(phase_tolerance_deg)),
        "sensor": "Newton O-chain Arg(mean(exp(i*m*theta_O)))/m",
        "path_constraint": "constant |tilde b^1_mn| in the measured control-field span",
    }


def _solve_target(library, feedback, target_coefficient: complex, *, name: str):
    spec = BoundaryTopologyObservableSpec(
        resonant_modes=(MODE,),
        resonant_quantities=("coefficient_real", "coefficient_imag"),
        resonant_weights=(2.5e9, 2.5e9),
        core_radial_max=0.42,
        core_field_weights=(2.0e5, 5.0e4),
        metadata={"target_name": name},
    )
    problem = make_boundary_topology_control_problem(
        library,
        spec,
        {
            "island.m5.n5.coefficient_real": float(np.real(target_coefficient)),
            "island.m5.n5.coefficient_imag": float(np.imag(target_coefficient)),
        },
        initial_controls=(0.0, 0.0),
        n_values=(5,),
        m_values=(5,),
        plasma_feedback=feedback,
        steps=(0.004, 0.004),
        n_iterations=4,
        regularization=1.0e-7,
        line_search=(1.0, 0.5, 0.25, 0.125),
        convergence_tolerance=2.0e-8,
        target_zero_prefixes=(),
        target_preserve_initial_prefixes=("core.",),
        metadata={"target_name": name, "mode": MODE},
    )
    result = solve_boundary_topology_control(problem)
    state, _ = problem.backend.forward_state(
        problem.backend.library.case_request(result.final_controls)
        if hasattr(problem.backend.library, "case_request")
        else _response_request(result.final_controls, library.control_labels)
    )
    return result, state


def _response_request(controls, labels):
    from pyna.toroidal.control.boundary_plasma_response import BoundaryPlasmaResponseInput

    return BoundaryPlasmaResponseInput(
        controls=np.asarray(controls, dtype=float),
        control_labels=tuple(labels),
    )


def _continuation_fixed_points(field_basis, fpt_field, controls, cache, *, steps: int = 10):
    section = cache["tc_sec"][0.0]
    rows = tuple(section["O"]) + tuple(section["X"])
    R = np.asarray([row[0] for row in rows], dtype=float)
    Z = np.asarray([row[1] for row in rows], dtype=float)
    residual = np.full(R.size, np.nan)
    ptype = np.full(R.size, -1)
    DPm = np.full((R.size, 2, 2), np.nan)
    for alpha in np.linspace(0.0, 1.0, max(2, int(steps))):
        field = field_basis.total_field(alpha * np.asarray(controls, dtype=float))
        if fpt_field is not None:
            field = field + fpt_field
        output = find_fixed_points_batch_span_field(
            field,
            R,
            Z,
            0.0,
            TWOPI,
            DPHI,
            fd_eps=1.0e-4,
            max_iter=100,
            tol=1.0e-11,
            n_threads=-1,
        )
        R_new, Z_new, residual, converged, DPm_flat, _eig_r, _eig_i, ptype = output
        converged = np.asarray(converged, dtype=bool)
        if not np.all(converged):
            failed = np.nonzero(~converged)[0].tolist()
            raise RuntimeError(f"fixed-point continuation failed for indices {failed} at alpha={alpha:.3f}")
        R = np.asarray(R_new, dtype=float)
        Z = np.asarray(Z_new, dtype=float)
        DPm = np.asarray(DPm_flat, dtype=float).reshape(-1, 2, 2)

    axis_R = np.asarray([cache["R_AX"][0]], dtype=float)
    axis_Z = np.asarray([cache["Z_AX"][0]], dtype=float)
    field = field_basis.total_field(np.asarray(controls, dtype=float))
    if fpt_field is not None:
        field = field + fpt_field
    axis_out = find_fixed_points_batch_span_field(
        field,
        axis_R,
        axis_Z,
        0.0,
        TWOPI,
        DPHI,
        max_iter=100,
        tol=1.0e-11,
        n_threads=1,
    )
    if not bool(np.asarray(axis_out[3])[0]):
        raise RuntimeError("magnetic-axis Newton solve failed")
    axis = (float(axis_out[0][0]), float(axis_out[1][0]))
    kinds = tuple(str(classify_monodromy_2x2(matrix).kind).upper() for matrix in DPm)
    if any(kind not in {"O", "X"} for kind in kinds):
        raise RuntimeError(f"final monodromy classification is not O/X: {kinds}")
    coordinates = np.column_stack([R, Z])
    if coordinates.shape[0] > 1:
        pairwise = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
        np.fill_diagonal(pairwise, np.inf)
        if float(np.min(pairwise)) < 1.0e-6:
            raise RuntimeError("multiple Newton continuation seeds converged to the same fixed point")
    kind_index = {"O": 0, "X": 0}
    fixed_points = []
    for index, (r, z, kind, res, pt, matrix) in enumerate(
        zip(R, Z, kinds, residual, ptype, DPm)
    ):
        branch = kind_index[kind]
        kind_index[kind] += 1
        fixed_points.append(
            {
                "R": float(r),
                "Z": float(z),
                "kind": kind,
                "residual": float(res),
                "converged": True,
                "point_type": int(pt),
                "DPm": matrix,
                "mode": MODE,
                "branch": branch,
                "seed_index": index,
            }
        )
    return field, tuple(fixed_points), axis


def _healed_phase(
    fixed_points,
    case,
    radial_label: float,
    *,
    return_diagnostics: bool = False,
):
    """Invert Newton O points into the full PEST surface stack used by the spectrum."""

    from pyna.topo import InnerFourierSection

    theta = np.asarray(case.theta_vals, dtype=float)
    radial = np.asarray(case.radial_labels, dtype=float)
    phi_index = int(np.argmin(np.abs(np.asarray(case.phi_vals, dtype=float))))
    R_stack = np.asarray(case.R_surf[phi_index], dtype=float)
    Z_stack = np.asarray(case.Z_surf[phi_index], dtype=float)
    root = float(radial_label)
    axis = np.asarray(case.core_reference.axis, dtype=float)
    section = InnerFourierSection.from_poincare_surfaces(
        phi_ref=float(case.phi_vals[phi_index]),
        R_ax=float(axis[0]),
        Z_ax=float(axis[1]),
        r_norms=radial,
        R_surf=R_stack,
        Z_surf=Z_stack,
        theta_arr=theta,
        n_fourier=min(12, max(1, (theta.size - 1) // 2)),
    )
    points = np.asarray([[item["R"], item["Z"]] for item in fixed_points if item["kind"] == "O"])
    coordinates = np.asarray(
        [section.project(float(R), float(Z), r_init=root) for R, Z in points],
        dtype=float,
    )
    s_O = coordinates[:, 0]
    theta_O = np.mod(coordinates[:, 1], TWOPI)
    mapped = np.asarray([section.eval_RZ(float(s), float(t)) for s, t in coordinates])
    residual = np.linalg.norm(mapped - points, axis=1)
    phase = float(np.angle(np.mean(np.exp(1j * MODE[0] * theta_O))) / MODE[0])
    lower_extrapolation = np.maximum(float(radial[0]) - s_O, 0.0)
    upper_extrapolation = np.maximum(s_O - float(radial[-1]), 0.0)
    extrapolation = lower_extrapolation + upper_extrapolation
    diagnostics = {
        "radial_domain": (float(radial[0]), float(radial[-1])),
        "radial_in_domain": extrapolation <= 1.0e-10,
        "all_radial_in_domain": bool(np.all(extrapolation <= 1.0e-10)),
        "max_radial_extrapolation": float(np.max(extrapolation)),
        "chain_coherence": float(abs(np.mean(np.exp(1j * MODE[0] * theta_O)))),
        "projection_extrapolates": bool(np.any(extrapolation > 1.0e-10)),
    }
    values = (phase, theta_O, s_O - root, residual)
    return values + (diagnostics,) if return_diagnostics else values


def _trace_manifolds(field, fixed_points, *, n_turns: int):
    x_points = []
    for item in fixed_points:
        if item["kind"] != "X":
            continue
        matrix = np.asarray(item["DPm"], dtype=float)
        point_index = int(item["branch"])
        x_points.append(
            BoundaryIslandFixedPoint(
                phi=0.0,
                R=float(item["R"]),
                Z=float(item["Z"]),
                map_power=MODE[0],
                kind="X",
                DPm=matrix,
                residual=float(item["residual"]),
                eigenvalues=np.linalg.eigvals(matrix),
                seed_R=float(item["R"]),
                seed_Z=float(item["Z"]),
                point_index=point_index,
                map_span=TWOPI / MODE[0],
                metadata={
                    "map_power": MODE[0],
                    "field_period": TWOPI / MODE[0],
                    "base_map_span": TWOPI / MODE[0],
                    "monodromy_map_span": TWOPI,
                    "monodromy_field_period": TWOPI,
                    "monodromy_local": True,
                    "point_index": point_index,
                    "orbit_id": 0,
                },
            )
        )
    manifolds = trace_fixed_point_manifolds_field(
        field,
        x_points,
        phi_section=0.0,
        N_turns=int(n_turns),
        map_span=TWOPI,
        DPhi=DPHI,
        eps_min=2.0e-6,
        eps_max=8.0e-4,
        n_eps=18,
        RZlimit=(4.45, 6.45, -1.08, 1.08),
        include_arclength=True,
        require_local_monodromy=True,
        refine_stable_inverse_anchor=True,
    )
    if len(manifolds) != len(x_points):
        raise RuntimeError(
            f"traced {len(manifolds)} manifolds for {len(x_points)} hyperbolic X points"
        )
    return tuple(manifolds)


def _poincare_and_dpk(field, *, n_seeds: int, n_turns: int, dpk_turns: int):
    R_seed = np.linspace(5.45, 6.38, int(n_seeds))
    Z_seed = np.zeros_like(R_seed)
    wall_R = np.asarray([4.01, 6.99, 6.99, 4.01, 4.01])
    wall_Z = np.asarray([-1.49, -1.49, 1.49, 1.49, -1.49])
    counts, R_raw, Z_raw = trace_poincare_batch_field(
        field,
        R_seed,
        Z_seed,
        0.0,
        int(n_turns),
        DPHI,
        wall_R,
        wall_Z,
    )
    counts = np.asarray(counts, dtype=int)
    R_raw = np.asarray(R_raw, dtype=float)
    Z_raw = np.asarray(Z_raw, dtype=float)
    R_flat = np.concatenate(
        [R_raw[index * int(n_turns) : index * int(n_turns) + int(count)] for index, count in enumerate(counts)]
    )
    Z_flat = np.concatenate(
        [Z_raw[index * int(n_turns) : index * int(n_turns) + int(count)] for index, count in enumerate(counts)]
    )
    point_growth = []
    point_svd_growth = []
    point_recurrent = []
    point_radial = []
    for index, (r0, z0, count) in enumerate(zip(R_seed, Z_seed, counts)):
        growth = np.nan
        svd_growth = np.nan
        recurrent = 0.0
        raw = trace_orbit_along_phi_field(
            field,
            float(r0),
            float(z0),
            0.0,
            float(dpk_turns) * TWOPI,
            DPHI,
            dphi_out=TWOPI,
            m_turns_DPm=1,
            fd_eps=1.0e-4,
        )
        local = np.asarray(raw[3], dtype=float).reshape(-1, 2, 2)
        alive = np.asarray(raw[4], dtype=bool)
        usable = min(int(dpk_turns), max(0, int(np.count_nonzero(alive)) - 1), local.shape[0])
        if usable > 0:
            product = np.eye(2)
            cumulative = []
            for matrix in local[:usable]:
                if not np.all(np.isfinite(matrix)):
                    break
                product = matrix @ product
                if not np.all(np.isfinite(product)):
                    break
                cumulative.append(product.copy())
            if cumulative:
                matrices = np.asarray(cumulative, dtype=float)
                eig_abs = np.asarray([np.abs(np.linalg.eigvals(matrix)) for matrix in matrices])
                metrics = boundary_dpk_growth_metrics(
                    {
                        "k": np.arange(1, matrices.shape[0] + 1, dtype=float),
                        "DPk": matrices,
                        "eig_abs": eig_abs,
                        "alive": np.ones(matrices.shape[0], dtype=int),
                    },
                    return_period=1.0,
                    recurrence_max_k=matrices.shape[0],
                )
                # Spectral recurrence and SVD growth answer different
                # questions for a twist map: eigenvalues can return close to
                # the unit circle while nonnormal shear keeps the SVD above
                # one.  Preserve both instead of requiring both to vanish.
                recurrent = float(metrics.spectral_recurrence_min <= 0.02)
                growth = float(metrics.eigenvalue_ftle)
                svd_growth = float(metrics.ftle)
        point_growth.extend([growth] * int(count))
        point_svd_growth.extend([svd_growth] * int(count))
        point_recurrent.extend([recurrent] * int(count))
        point_radial.extend([index / max(len(R_seed) - 1, 1)] * int(count))
    return {
        "R": np.asarray(R_flat, dtype=float),
        "Z": np.asarray(Z_flat, dtype=float),
        "counts": counts,
        "point_growth": np.asarray(point_growth, dtype=float),
        "point_svd_growth": np.asarray(point_svd_growth, dtype=float),
        "point_recurrent_surface": np.asarray(point_recurrent, dtype=float),
        "radial_label": np.asarray(point_radial, dtype=float),
    }


def _curved_bars(case, chain, fixed_points):
    phi_index = int(np.argmin(np.abs(case.phi_vals)))
    s = np.asarray(case.radial_labels, dtype=float)
    root = float(chain.radial_label)
    R = np.asarray(case.R_surf[phi_index], dtype=float)
    Z = np.asarray(case.Z_surf[phi_index], dtype=float)
    R_res = np.asarray([np.interp(root, s, R[:, index]) for index in range(R.shape[1])])
    Z_res = np.asarray([np.interp(root, s, Z[:, index]) for index in range(Z.shape[1])])
    s_path = np.linspace(
        max(float(s[0]), root - float(chain.half_width)),
        min(float(s[-1]), root + float(chain.half_width)),
        129,
    )
    bars = []
    for branch, point in enumerate(item for item in fixed_points if item["kind"] == "O"):
        index = int(np.argmin((R_res - point["R"]) ** 2 + (Z_res - point["Z"]) ** 2))
        bars.append(
            PoincareCurvedIslandBar(
                R_path=np.interp(s_path, s, R[:, index]),
                Z_path=np.interp(s_path, s, Z[:, index]),
                mode_m=MODE[0],
                mode_n=MODE[1],
                radial_label=root,
                half_width=float(chain.half_width),
                amplitude=float(chain.b_res),
                phase=float(np.angle(chain.coefficient)),
                kind="O",
                branch=branch,
                colormap="viridis",
                label="Nardon width on healed constant-theta line" if branch == 0 else None,
                source="healed B0 plus measured control-field spectrum",
            )
        )
    return tuple(bars)


def _plot_poincare_cases(
    cases,
    out_path: Path,
    *,
    dpi: int,
    phase_validation_passed: bool,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.2), sharex=True, sharey=True, constrained_layout=True)
    finite_growth = np.concatenate(
        [item["trace"]["point_growth"][np.isfinite(item["trace"]["point_growth"])] for item in cases]
    )
    vmax = max(float(np.percentile(finite_growth, 98.0)) if finite_growth.size else 1.0, 1.0e-6)
    norm = Normalize(vmin=0.0, vmax=vmax)
    for ax, item in zip(axes.ravel(), cases):
        trace = item["trace"]
        growth = np.maximum(trace["point_growth"], 0.0)
        ax.scatter(trace["R"], trace["Z"], c=growth, cmap="magma", norm=norm, s=1.0, alpha=0.70, linewidths=0)
        O = np.asarray([[row["R"], row["Z"]] for row in item["fixed_points"] if row["kind"] == "O"])
        X = np.asarray([[row["R"], row["Z"]] for row in item["fixed_points"] if row["kind"] == "X"])
        ax.scatter(O[:, 0], O[:, 1], s=44, facecolors="none", edgecolors="#1b9e77", linewidths=1.5, label="Newton O")
        ax.scatter(X[:, 0], X[:, 1], s=42, marker="x", c="#d73027", linewidths=1.4, label="Newton X")
        for manifold_index, manifold in enumerate(item["manifolds"]):
            for code, color, label in (
                ("u", "#d55e00", "unstable manifold"),
                ("s", "#0072b2", "stable manifold"),
            ):
                sides = np.asarray(manifold[f"{code}_point_side"], dtype=float)
                for side, linestyle in ((-1.0, "--"), (1.0, "-")):
                    selected = sides == side
                    ax.plot(
                        np.asarray(manifold[f"{code}_R"])[selected],
                        np.asarray(manifold[f"{code}_Z"])[selected],
                        color=color,
                        linewidth=0.55,
                        linestyle=linestyle,
                        alpha=0.72,
                        label=label if manifold_index == 0 and side > 0.0 else None,
                        zorder=2,
                    )
        draw_poincare_curved_island_bars(
            ax,
            item["bars"],
            linewidth=2.6,
            endpoint_markers=True,
            show_labels=True,
            mode_colormaps={MODE: "viridis"},
        )
        ax.set_title(
            f"{item['label']}\n"
            f"u=({item['controls'][0]:+.4f}, {item['controls'][1]:+.4f}), "
            f"w_s={item['chain'].half_width:.4f}\n"
            f"phase shift [deg]: Nardon {item['nardon_phase_shift_deg']:+.2f}, "
            f"Newton {item['phase_shift_deg']:+.2f}",
            fontsize=9.5,
            pad=5.0,
        )
        ax.set_aspect("equal")
        ax.set_xlim(4.45, 6.45)
        ax.set_ylim(-1.08, 1.08)
        ax.grid(color="0.90", linewidth=0.45)
    axes[0, 0].legend(loc="lower left", frameon=False, fontsize=8)
    axes[1, 0].set_xlabel("R [m]")
    axes[1, 1].set_xlabel("R [m]")
    axes[0, 0].set_ylabel("Z [m]")
    axes[1, 0].set_ylabel("Z [m]")
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap="magma"), ax=axes, shrink=0.82, pad=0.02)
    cbar.set_label(r"DP$^k$ eigenvalue growth per toroidal turn")
    status = (
        "phase chart validated"
        if phase_validation_passed
        else "exploratory periodic-point displacement; phase B0/chart gate failed"
    )
    fig.suptitle(
        f"W7-X actual control-coil boundary-island nonlinear verification\n{status}",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _plot_response(
    cases,
    c0,
    c1,
    c2,
    diagnostics,
    normal_alignment,
    phase_b0_alignment,
    out_path: Path,
    *,
    dpi: int,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)
    ax = axes[0, 0]
    scale = 1.0e6
    ax.axhline(0.0, color="0.8", linewidth=0.7)
    ax.axvline(0.0, color="0.8", linewidth=0.7)
    ax.arrow(scale * c0.real, scale * c0.imag, scale * c1.real, scale * c1.imag, color="#0072b2", width=0.35, length_includes_head=True, label="control 1 unit column")
    ax.arrow(scale * c0.real, scale * c0.imag, scale * c2.real, scale * c2.imag, color="#d55e00", width=0.35, length_includes_head=True, label="control 2 unit column")
    for item, color in zip(cases, ("#333333", "#009e73", "#cc79a7", "#e69f00")):
        coeff = item["chain"].coefficient
        ax.scatter(scale * coeff.real, scale * coeff.imag, s=55, color=color, label=item["label"], zorder=4)
    ax.set_xlabel(r"Re $\tilde b^1_{5,-5}$ [$10^{-6}$]")
    ax.set_ylabel(r"Im $\tilde b^1_{5,-5}$ [$10^{-6}$]")
    ax.set_title("Measured control-field span in resonant complex plane")
    ax.legend(fontsize=7, frameon=False, loc="best")
    ax.grid(color="0.91", linewidth=0.5)

    labels = [item["label"] for item in cases]
    x = np.arange(len(cases))
    axes[0, 1].bar(x, [item["chain"].half_width for item in cases], color=("#777777", "#009e73", "#cc79a7", "#e69f00"))
    axes[0, 1].set_xticks(x, labels, rotation=20, ha="right")
    axes[0, 1].set_ylabel("Nardon half-width in s")
    axes[0, 1].set_title("Predicted boundary island width")
    axes[0, 1].grid(axis="y", color="0.91", linewidth=0.5)

    matrix = diagnostics["matrix"]
    image = axes[1, 0].imshow(matrix, aspect="auto", cmap="RdBu_r")
    axes[1, 0].set_xticks((0, 1), ("control 1", "control 2"))
    axes[1, 0].set_yticks((0, 1), ("Re b", "Im b"))
    axes[1, 0].set_title(f"Linear resonant response, cond={diagnostics['condition_number']:.3f}")
    fig.colorbar(image, ax=axes[1, 0], shrink=0.78)

    axis_shift = np.asarray([item["axis_shift_mm"] for item in cases])
    phase = np.asarray([item["phase_shift_deg"] for item in cases])
    nardon_phase = np.asarray([item["nardon_phase_shift_deg"] for item in cases])
    manifold_expansion = np.asarray(
        [np.mean([row["unstable_expansion"] for row in item["manifolds"]]) for item in cases]
    )
    width = 0.36
    axes[1, 1].bar(x - width / 2, axis_shift, width, color="#56b4e9", label="axis shift [mm]")
    axes[1, 1].bar(x + width / 2, phase, width, color="#e69f00", label="Newton phase shift [deg]")
    axes[1, 1].plot(x, nardon_phase, color="#0072b2", marker="x", linestyle="--", label="Nardon -darg(b)/m [deg]")
    axes[1, 1].axhline(0.0, color="0.75", linewidth=0.7)
    axes[1, 1].set_xticks(x, labels, rotation=20, ha="right")
    phase_b0_status = "PASS" if phase_b0_alignment["passed"] else "FAIL"
    axes[1, 1].set_title(
        "Nonlinear core/topology audit\n"
        f"normal fraction RMS={normal_alignment['edge_rms']:.3e}; "
        f"phase-B0 {phase_b0_status}: |B^s/B^phi|={phase_b0_alignment['edge_radial_ratio_rms']:.3f}, "
        f"iota RMS={phase_b0_alignment['iota_profile_error_rms']:.3f}",
        fontsize=9.5,
    )
    expansion_axis = axes[1, 1].twinx()
    expansion_axis.plot(
        x,
        manifold_expansion,
        color="#8f1d5b",
        marker="D",
        linewidth=1.4,
        label="mean X unstable multiplier",
    )
    expansion_axis.set_ylabel("full-return unstable multiplier")
    handles, legend_labels = axes[1, 1].get_legend_handles_labels()
    extra_handles, extra_labels = expansion_axis.get_legend_handles_labels()
    axes[1, 1].legend(
        handles + extra_handles,
        legend_labels + extra_labels,
        fontsize=8,
        frameon=False,
    )
    axes[1, 1].grid(axis="y", color="0.91", linewidth=0.5)
    fig.suptitle("W7-X actual two-coil boundary topology control audit", fontsize=14)
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _save_manifold_state(cases, out_path: Path) -> None:
    R_parts = []
    Z_parts = []
    case_parts = []
    x_parts = []
    stability_parts = []
    side_parts = []
    generation_parts = []
    origins = []
    expansions = []
    for case_index, item in enumerate(cases):
        for x_index, manifold in enumerate(item["manifolds"]):
            origins.append(
                (case_index, x_index, manifold["origin_R"], manifold["origin_Z"])
            )
            expansions.append(
                (
                    case_index,
                    x_index,
                    manifold["unstable_expansion"],
                    manifold["stable_backward_expansion"],
                )
            )
            for code, stability in (("u", 1), ("s", -1)):
                R_values = np.asarray(manifold[f"{code}_R"], dtype=float)
                n_points = R_values.size
                R_parts.append(R_values)
                Z_parts.append(np.asarray(manifold[f"{code}_Z"], dtype=float))
                case_parts.append(np.full(n_points, case_index, dtype=int))
                x_parts.append(np.full(n_points, x_index, dtype=int))
                stability_parts.append(np.full(n_points, stability, dtype=int))
                side_parts.append(np.asarray(manifold[f"{code}_point_side"], dtype=int))
                generation_parts.append(np.asarray(manifold[f"{code}_generation"], dtype=int))
    np.savez_compressed(
        out_path,
        R=np.concatenate(R_parts),
        Z=np.concatenate(Z_parts),
        case_index=np.concatenate(case_parts),
        x_index=np.concatenate(x_parts),
        stability=np.concatenate(stability_parts),
        side=np.concatenate(side_parts),
        generation=np.concatenate(generation_parts),
        origins=np.asarray(origins, dtype=float),
        expansions=np.asarray(expansions, dtype=float),
        case_labels=np.asarray([item["label"] for item in cases]),
        stability_convention=np.asarray("+1 unstable, -1 stable"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--standard", type=Path, default=None)
    parser.add_argument("--control-1", type=Path, default=None)
    parser.add_argument("--control-2", type=Path, default=None)
    parser.add_argument("--wout", type=Path, default=None)
    parser.add_argument("--fpt-state", type=Path, default=DEFAULT_FPT)
    parser.add_argument("--no-fpt", action="store_true")
    parser.add_argument(
        "--allow-exploratory-fpt",
        action="store_true",
        help="Explicitly allow a non-production pressure beta-ramp background response.",
    )
    parser.add_argument("--healed-cache", type=Path, default=None)
    parser.add_argument("--topoquest-root", type=Path, default=Path("~/repos/topoquest"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n-phi", type=int, default=320)
    parser.add_argument("--n-theta", type=int, default=256)
    parser.add_argument("--n-radial", type=int, default=18)
    parser.add_argument("--poincare-turns", type=int, default=260)
    parser.add_argument("--poincare-seeds", type=int, default=34)
    parser.add_argument("--dpk-turns", type=int, default=20)
    parser.add_argument("--manifold-turns", type=int, default=18)
    parser.add_argument("--dpi", type=int, default=230)
    parser.add_argument("--alignment-limit", type=float, default=0.025)
    parser.add_argument("--phase-b0-radial-ratio-limit", type=float, default=0.05)
    parser.add_argument("--phase-b0-iota-rms-limit", type=float, default=0.02)
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")

    root = args.data_root.expanduser()
    standard_path = (args.standard or root / "w7x-op21-standard.npz").expanduser()
    control_paths = (
        (args.control_1 or root / "w7x-op21-controlCoils1.npy").expanduser(),
        (args.control_2 or root / "w7x-op21-controlCoils2.npy").expanduser(),
    )
    wout_path = (args.wout or root / "vmecpp_w7x_beta_ramp_refs_20260607/vacuum/wout_w7x_vacuum.nc").expanduser()
    cache_path = (args.healed_cache or root / "healed_scaffold_cache.pkl").expanduser()
    out = (args.out_dir or root / "actual_control_coil_boundary_topology").expanduser()
    out.mkdir(parents=True, exist_ok=True)

    standard = load_cylindrical_vector_field_npz(standard_path, name="W7-X standard vacuum field")
    control_fields = tuple(
        _grid_field_from_npy(path, standard, f"W7-X measured control field {index}")
        for index, path in enumerate(control_paths, start=1)
    )
    actuators = boundary_field_actuator_array_from_grid_fields(
        control_fields,
        labels=("control_coil_1", "control_coil_2"),
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
        metadata={
            "response_kind": "measured_w7x_control_coil_vacuum_fields",
            "control_units": "dimensionless native-field multiplier; hardware current calibration unavailable",
        },
    )
    field_basis = CylindricalGridFieldControlBasis(standard, actuators)
    fpt_path = None if args.no_fpt else args.fpt_state.expanduser()
    fpt_field, fpt_audit = _load_fpt_increment(fpt_path, standard)
    if fpt_audit is not None and not bool(fpt_audit["production_ready"]):
        if not args.allow_exploratory_fpt:
            raise RuntimeError(
                "the cached FPT response is not production-ready; pass "
                "--allow-exploratory-fpt for an explicitly exploratory run"
            )

    base_case = vmec_boundary_topology_case_from_wout(
        wout_path,
        name="W7-X public",
        n_radial=args.n_radial,
        radial_min=0.15,
        n_phi=args.n_phi,
        n_theta=args.n_theta,
        metadata={"role": "independent healed integrable B0"},
    )
    alignment = surface_field_alignment_diagnostics(
        standard.R,
        standard.Z,
        standard.Phi,
        standard.BR,
        standard.BPhi,
        standard.BZ,
        base_case.R_surf,
        base_case.Z_surf,
        base_case.phi_vals,
        base_case.theta_vals,
        base_case.radial_labels,
        nfp=standard.nfp,
        iota_profile=base_case.iota_profile,
    )
    phi_grid = np.broadcast_to(base_case.phi_vals[:, None, None], base_case.R_surf.shape)
    sampled_BR, sampled_BZ, sampled_BPhi = standard.interpolate_at(
        base_case.R_surf,
        base_case.Z_surf,
        phi_grid,
    )
    normal_field = radial_perturbation_component(
        base_case.R_surf,
        base_case.Z_surf,
        base_case.phi_vals,
        base_case.theta_vals,
        sampled_BR,
        sampled_BZ,
        sampled_BPhi,
    )
    field_magnitude = np.sqrt(sampled_BR**2 + sampled_BZ**2 + sampled_BPhi**2)
    normal_fraction = np.abs(normal_field) / np.maximum(field_magnitude, 1.0e-300)
    normal_alignment = {
        "global_rms": float(np.sqrt(np.mean(normal_fraction**2))),
        "edge_rms": float(np.sqrt(np.mean(normal_fraction[:, -2:, :] ** 2))),
        "edge_p95": float(np.percentile(normal_fraction[:, -2:, :], 95.0)),
    }
    if normal_alignment["edge_rms"] > float(args.alignment_limit):
        raise RuntimeError(
            "total-field/healed-surface normal-fraction gate failed: "
            f"edge RMS={normal_alignment['edge_rms']:.6g} "
            f"> limit={args.alignment_limit:.6g}"
        )
    phase_b0_alignment = _phase_b0_alignment_gate(
        alignment,
        radial_ratio_limit=args.phase_b0_radial_ratio_limit,
        iota_rms_limit=args.phase_b0_iota_rms_limit,
    )
    case = extend_boundary_topology_case_to_resonance(
        base_case,
        m=MODE[0],
        n=MODE[1],
        fit_points=8,
        n_extra=5,
        outer_margin=0.025,
        max_extension=0.12,
    )

    standard_response = perturbation_candidate_nardon_response(
        CylindricalGridFieldCandidate(standard),
        case.R_surf,
        case.Z_surf,
        case.phi_vals,
        case.theta_vals,
        case.radial_labels,
        denominator_B3=case.denominator_B3,
        m_max=8,
        n_max=10,
        metadata={
            "decomposition": "standard total field radial projection on independent healed B0",
            "alignment_gate_passed": True,
            "phase_b0_compatibility_passed": bool(phase_b0_alignment["passed"]),
        },
    )
    library = build_boundary_perturbation_spectrum_library(
        case,
        actuators,
        m_max=8,
        n_max=10,
        base_tilde_b1=standard_response.tilde_b1,
        metadata={"response_kind": "actual_w7x_two_control_field_basis"},
    )

    feedback = None
    fpt_response = None
    if fpt_field is not None:
        fpt_response = perturbation_candidate_nardon_response(
            CylindricalGridFieldCandidate(fpt_field),
            case.R_surf,
            case.Z_surf,
            case.phi_vals,
            case.theta_vals,
            case.radial_labels,
            denominator_B3=case.denominator_B3,
            m_max=8,
            n_max=10,
            metadata={"response_kind": "FPT pressure beta-ramp background increment"},
        )
        cached = TopoquestFPTCachedResponseBasis(
            control_labels=library.control_labels,
            tilde_b1_basis=np.zeros((len(library.control_labels),) + case.R_surf.shape, dtype=complex),
            base_tilde_b1=fpt_response.tilde_b1,
            beta=float(fpt_audit["beta"]),
            response_case=case,
            converged=bool(fpt_audit["accepted"]),
            production_ready=False,
            readiness={"accepted": False, "status": "exploratory FPT beta-ramp predictor"},
            metadata={
                "response_scope": "pressure_driven_background_only",
                "external_control_screening_solved": False,
            },
        )
        feedback = TopoquestFPTPlasmaFeedbackAdapter(
            spec=TopoquestFPTBetaRampSpec(
                beta_values=(float(fpt_audit["beta"]),),
                require_convergence=False,
                require_production_readiness=not args.allow_exploratory_fpt,
                case_alias="W7-X public",
            ),
            response_basis=cached,
        )

    baseline_problem = make_boundary_topology_control_problem(
        library,
        BoundaryTopologyObservableSpec(
            resonant_modes=(MODE,),
            resonant_quantities=("coefficient_real", "coefficient_imag"),
            core_radial_max=0.42,
        ),
        {},
        n_values=(5,),
        m_values=(5,),
        plasma_feedback=feedback,
        n_iterations=0,
        target_zero_prefixes=(),
    )
    baseline_state, _ = baseline_problem.backend.forward_state(_response_request((0.0, 0.0), library.control_labels))
    baseline_chain = _request_chain(baseline_state)
    c0 = complex(baseline_chain.coefficient)
    c1 = _response_mode_coefficient(library.responses[0], case)
    c2 = _response_mode_coefficient(library.responses[1], case)
    matrix = np.asarray([[c1.real, c2.real], [c1.imag, c2.imag]], dtype=float)
    response_diagnostics = {
        "matrix": matrix,
        "singular_values": np.linalg.svd(matrix, compute_uv=False),
        "condition_number": float(np.linalg.cond(matrix)),
    }

    cache = _load_healed_cache(cache_path, args.topoquest_root.expanduser())
    fixed_point_evaluations: dict[tuple[float, ...], dict[str, object]] = {}

    def fixed_point_evaluation(controls, radial_label: float) -> dict[str, object]:
        commands = np.asarray(controls, dtype=float).ravel()
        key = tuple(float(value) for value in np.round(commands, decimals=13)) + (
            float(radial_label),
        )
        if key not in fixed_point_evaluations:
            field, fixed_points, axis = _continuation_fixed_points(
                field_basis,
                fpt_field,
                commands,
                cache,
            )
            phase, theta_O, radial_shift, projection_residual, phase_diagnostics = _healed_phase(
                fixed_points,
                case,
                radial_label,
                return_diagnostics=True,
            )
            fixed_point_evaluations[key] = {
                "field": field,
                "fixed_points": fixed_points,
                "axis": axis,
                "phase": phase,
                "theta_O": theta_O,
                "radial_shift": radial_shift,
                "projection_residual": projection_residual,
                "phase_diagnostics": phase_diagnostics,
            }
        return fixed_point_evaluations[key]

    baseline_fixed = fixed_point_evaluation(np.zeros(2), baseline_chain.radial_label)
    baseline_phase = float(baseline_fixed["phase"])
    baseline_axis = tuple(baseline_fixed["axis"])

    def newton_phase_shift(controls) -> float:
        evaluated = fixed_point_evaluation(controls, baseline_chain.radial_label)
        phase = float(evaluated["phase"])
        return float(
            np.degrees(
                np.angle(np.exp(1j * MODE[0] * (phase - baseline_phase)))
                / MODE[0]
            )
        )

    direct_phase_calibration = _calibrate_newton_chain_phase_target(
        2.5,
        c0,
        matrix,
        ((-1.0, 1.0), (-1.0, 1.0)),
        newton_phase_shift,
        phase_tolerance_deg=0.02,
    )
    if not bool(direct_phase_calibration["accepted"]):
        raise RuntimeError("bounded direct Newton O-chain phase calibration failed")
    direct_phase_controls = np.asarray(direct_phase_calibration["controls"], dtype=float)
    direct_phase_state, _ = baseline_problem.backend.forward_state(
        _response_request(direct_phase_controls, library.control_labels)
    )

    broad_target = 1.69 * c0
    targets = (
        ("spectral suppression", 0.0j),
        ("predicted width x1.3", broad_target),
    )
    solved = []
    for label, coefficient in targets:
        result, state = _solve_target(library, feedback, coefficient, name=label)
        solved.append((label, result, state))

    raw_cases = [
        ("baseline", np.zeros(2), baseline_state, None, {"kind": "reference"}),
        (
            "Newton O phase +2.50 deg",
            direct_phase_controls,
            direct_phase_state,
            None,
            {
                "kind": "direct_newton_chain_phase",
                "target_shift_deg": float(direct_phase_calibration["target_shift_deg"]),
            },
        ),
    ] + [
        (label, result.final_controls, state, result, {"kind": "spectral"})
        for label, result, state in solved
    ]
    case_rows = []
    for label, controls, state, result, objective in raw_cases:
        chain = _request_chain(state)
        evaluated = fixed_point_evaluation(controls, chain.radial_label)
        field = evaluated["field"]
        fixed_points = evaluated["fixed_points"]
        axis = evaluated["axis"]
        phase = float(evaluated["phase"])
        theta_O = np.asarray(evaluated["theta_O"], dtype=float)
        phase_radial_shift = np.asarray(evaluated["radial_shift"], dtype=float)
        phase_projection_residual = np.asarray(
            evaluated["projection_residual"],
            dtype=float,
        )
        phase_diagnostics = dict(evaluated["phase_diagnostics"])
        phase_shift = float(np.degrees(np.angle(np.exp(1j * MODE[0] * (phase - baseline_phase))) / MODE[0]))
        nardon_phase_shift = _nardon_phase_shift_deg(
            chain.coefficient,
            c0,
            MODE[0],
        )
        phase_closure_error = _phase_closure_error_deg(
            phase_shift,
            nardon_phase_shift,
            MODE[0],
        )
        axis_shift_mm = 1000.0 * float(np.hypot(axis[0] - baseline_axis[0], axis[1] - baseline_axis[1]))
        trace = _poincare_and_dpk(
            field,
            n_seeds=args.poincare_seeds,
            n_turns=args.poincare_turns,
            dpk_turns=args.dpk_turns,
        )
        manifolds = _trace_manifolds(field, fixed_points, n_turns=args.manifold_turns)
        case_rows.append(
            {
                "label": label,
                "controls": np.asarray(controls, dtype=float),
                "chain": chain,
                "fixed_points": fixed_points,
                "axis": axis,
                "axis_shift_mm": axis_shift_mm,
                "phase": phase,
                "phase_shift_deg": phase_shift,
                "nardon_phase_shift_deg": nardon_phase_shift,
                "phase_closure_error_deg": phase_closure_error,
                "phase_radial_shift_rms_s": float(np.sqrt(np.mean(phase_radial_shift**2))),
                "phase_radial_shift_max_abs_s": float(np.max(np.abs(phase_radial_shift))),
                "phase_projection_rms_m": float(np.sqrt(np.mean(phase_projection_residual**2))),
                "phase_projection_max_m": float(np.max(phase_projection_residual)),
                "phase_coordinate_domain_valid": bool(
                    phase_diagnostics["all_radial_in_domain"]
                ),
                "phase_coordinate_max_extrapolation_s": float(
                    phase_diagnostics["max_radial_extrapolation"]
                ),
                "phase_chain_coherence": float(phase_diagnostics["chain_coherence"]),
                "theta_O": theta_O,
                "trace": trace,
                "manifolds": manifolds,
                "bars": _curved_bars(case, chain, fixed_points),
                "result": result,
                "objective": dict(objective),
            }
        )

    standard_coefficient = _response_mode_coefficient(standard_response, case)
    fpt_coefficient = 0.0j if fpt_response is None else _response_mode_coefficient(fpt_response, case)
    phase_closure_tolerance_deg = 0.75
    phase_projection_tolerance_m = 0.002
    phase_radial_shift_tolerance_s = 0.05
    max_phase_closure_error_deg = float(
        max(abs(item["phase_closure_error_deg"]) for item in case_rows)
    )
    max_phase_projection_rms_m = float(
        max(item["phase_projection_rms_m"] for item in case_rows)
    )
    max_phase_radial_shift_rms_s = float(
        max(item["phase_radial_shift_rms_s"] for item in case_rows)
    )
    max_phase_coordinate_extrapolation_s = float(
        max(item["phase_coordinate_max_extrapolation_s"] for item in case_rows)
    )
    phase_coordinate_domain_passed = bool(
        all(item["phase_coordinate_domain_valid"] for item in case_rows)
    )
    phase_closure_passed = max_phase_closure_error_deg <= phase_closure_tolerance_deg
    phase_projection_passed = max_phase_projection_rms_m <= phase_projection_tolerance_m
    phase_radial_alignment_passed = (
        max_phase_radial_shift_rms_s <= phase_radial_shift_tolerance_s
    )
    direct_phase_case = next(
        item
        for item in case_rows
        if item["objective"].get("kind") == "direct_newton_chain_phase"
    )
    direct_phase_width_relative_error = float(
        abs(float(direct_phase_case["chain"].half_width) - float(baseline_chain.half_width))
        / max(abs(float(baseline_chain.half_width)), 1.0e-300)
    )
    direct_phase_width_tolerance = 0.01
    direct_phase_coherence_tolerance = 0.90
    direct_phase_target_reached = bool(
        direct_phase_calibration["accepted"]
        and direct_phase_width_relative_error <= direct_phase_width_tolerance
    )
    direct_phase_control_passed = bool(
        direct_phase_target_reached
        and phase_b0_alignment["passed"]
        and direct_phase_case["phase_coordinate_domain_valid"]
        and direct_phase_case["phase_radial_shift_rms_s"]
        <= phase_radial_shift_tolerance_s
        and direct_phase_case["phase_chain_coherence"]
        >= direct_phase_coherence_tolerance
    )
    _plot_poincare_cases(
        case_rows,
        out / "w7x_actual_control_coil_poincare_dpk.png",
        dpi=args.dpi,
        phase_validation_passed=bool(
            phase_b0_alignment["passed"]
            and phase_coordinate_domain_passed
            and phase_radial_alignment_passed
        ),
    )
    _plot_response(
        case_rows,
        c0,
        c1,
        c2,
        response_diagnostics,
        normal_alignment,
        phase_b0_alignment,
        out / "w7x_actual_control_coil_response_audit.png",
        dpi=args.dpi,
    )

    payload = {
        "schema": "pyna_w7x_actual_control_coil_topology_v3",
        "case": "W7-X public",
        "mode_convention": {
            "physical_resonance": [5, 5],
            "q": "m/n",
            "iota": "1/q",
            "nardon_coefficient": [5, -5],
            "nardon_fourier_basis": "exp(i*(m*theta_PEST+n_N*phi))",
            "positive_q_resonant_phase": "m*theta_PEST-n0*phi",
            "positive_q_resonant_branch": "n_N=-n0",
        },
        "sources": {
            "standard_field_id": _source_id(standard_path),
            "control_field_ids": [_source_id(path) for path in control_paths],
            "healed_b0_id": case.metadata.get("source_id"),
            "fpt_state": fpt_audit,
            "exploratory_fpt_override": bool(
                fpt_audit is not None and args.allow_exploratory_fpt
            ),
        },
        "alignment": {
            "total_field_normal_fraction_global_rms": normal_alignment["global_rms"],
            "total_field_normal_fraction_edge_rms": normal_alignment["edge_rms"],
            "total_field_normal_fraction_edge_p95": normal_alignment["edge_p95"],
            "contravariant_radial_ratio_global_rms": alignment.global_radial_ratio_rms,
            "contravariant_radial_ratio_edge_rms": alignment.edge_radial_ratio_rms,
            "iota_profile_error_rms": alignment.iota_profile_error_rms,
            "iota_profile_sign_flipped_error_rms": alignment.iota_profile_sign_flipped_error_rms,
            "normal_fraction_gate_limit": float(args.alignment_limit),
            "normal_fraction_gate_passed": True,
            "phase_b0_edge_radial_ratio_limit": phase_b0_alignment[
                "edge_radial_ratio_limit"
            ],
            "phase_b0_edge_radial_ratio_passed": phase_b0_alignment[
                "edge_radial_ratio_passed"
            ],
            "phase_b0_iota_rms_limit": phase_b0_alignment[
                "iota_profile_error_limit"
            ],
            "phase_b0_iota_passed": phase_b0_alignment["iota_profile_passed"],
            "gate_passed": phase_b0_alignment["passed"],
        },
        "resonance_extension": case.metadata["edge_resonance_extension"],
        "fpt_capability": diagnose_topoquest_fpt_capability().as_dict(),
        "control_calibration": {
            "units": "dimensionless native-field multiplier",
            "bounds": [[-1.0, 1.0], [-1.0, 1.0]],
            "hardware_current_calibration_available": False,
        },
        "response_matrix": {
            "real_imag_by_control": matrix.tolist(),
            "singular_values": response_diagnostics["singular_values"].tolist(),
            "condition_number": response_diagnostics["condition_number"],
        },
        "phase_convention_audit": {
            "newton_coordinate": "full healed PEST surface stack used by the Nardon spectrum",
            "prediction": "delta_theta_star=-delta_arg_b/m at fixed phi",
            "closure_tolerance_deg": phase_closure_tolerance_deg,
            "projection_rms_tolerance_m": phase_projection_tolerance_m,
            "radial_shift_rms_tolerance_s": phase_radial_shift_tolerance_s,
            "max_closure_error_deg": max_phase_closure_error_deg,
            "max_projection_rms_m": max_phase_projection_rms_m,
            "max_radial_shift_rms_s": max_phase_radial_shift_rms_s,
            "max_coordinate_extrapolation_s": max_phase_coordinate_extrapolation_s,
            "closure_passed": phase_closure_passed,
            "projection_passed": phase_projection_passed,
            "radial_alignment_passed": phase_radial_alignment_passed,
            "coordinate_domain_passed": phase_coordinate_domain_passed,
            "same_source_b0_alignment_passed": phase_b0_alignment["passed"],
            "projection_residual_warning": "small residuals from an extrapolating spline do not prove coordinate-domain validity",
        },
        "direct_newton_phase_control": {
            "target_shift_deg": float(direct_phase_calibration["target_shift_deg"]),
            "achieved_shift_deg": float(direct_phase_calibration["achieved_shift_deg"]),
            "error_deg": float(direct_phase_calibration["error_deg"]),
            "controls": np.asarray(direct_phase_calibration["controls"], dtype=float).tolist(),
            "coefficient_phase_shift_deg": float(
                direct_phase_calibration["coefficient_phase_shift_deg"]
            ),
            "relative_spectral_amplitude_error": float(
                direct_phase_calibration["relative_spectral_amplitude_error"]
            ),
            "nardon_half_width_relative_error": direct_phase_width_relative_error,
            "phase_tolerance_deg": float(direct_phase_calibration["phase_tolerance_deg"]),
            "width_relative_tolerance": direct_phase_width_tolerance,
            "chain_coherence": float(direct_phase_case["phase_chain_coherence"]),
            "chain_coherence_tolerance": direct_phase_coherence_tolerance,
            "coordinate_domain_valid": bool(
                direct_phase_case["phase_coordinate_domain_valid"]
            ),
            "coordinate_max_extrapolation_s": float(
                direct_phase_case["phase_coordinate_max_extrapolation_s"]
            ),
            "radial_shift_rms_s": float(
                direct_phase_case["phase_radial_shift_rms_s"]
            ),
            "evaluations": int(direct_phase_calibration["evaluations"]),
            "iterations": int(direct_phase_calibration["iterations"]),
            "sensor": str(direct_phase_calibration["sensor"]),
            "path_constraint": str(direct_phase_calibration["path_constraint"]),
            "target_reached": direct_phase_target_reached,
            "physically_validated": direct_phase_control_passed,
        },
        "resonant_decomposition": {
            "standard_field_on_independent_healed_b0": _complex_payload(standard_coefficient),
            "pressure_beta_ramp_increment": None
            if fpt_response is None
            else _complex_payload(fpt_coefficient),
            "control_columns": [_complex_payload(c1), _complex_payload(c2)],
            "baseline_sum_residual_abs": float(
                abs(c0 - standard_coefficient - fpt_coefficient)
            ),
        },
        "cases": [
            {
                "label": item["label"],
                "objective": item["objective"],
                "controls": item["controls"].tolist(),
                "coefficient_real": float(np.real(item["chain"].coefficient)),
                "coefficient_imag": float(np.imag(item["chain"].coefficient)),
                "coefficient_abs": float(abs(item["chain"].coefficient)),
                "coefficient_phase_deg": float(np.degrees(np.angle(item["chain"].coefficient))),
                "nardon_half_width_s": float(item["chain"].half_width),
                "nardon_phase_shift_deg": float(item["nardon_phase_shift_deg"]),
                "newton_phase_shift_deg": float(item["phase_shift_deg"]),
                "phase_closure_error_deg": float(item["phase_closure_error_deg"]),
                "phase_radial_shift_rms_s": float(item["phase_radial_shift_rms_s"]),
                "phase_radial_shift_max_abs_s": float(item["phase_radial_shift_max_abs_s"]),
                "phase_projection_rms_m": float(item["phase_projection_rms_m"]),
                "phase_projection_max_m": float(item["phase_projection_max_m"]),
                "phase_coordinate_domain_valid": bool(
                    item["phase_coordinate_domain_valid"]
                ),
                "phase_coordinate_max_extrapolation_s": float(
                    item["phase_coordinate_max_extrapolation_s"]
                ),
                "phase_chain_coherence": float(item["phase_chain_coherence"]),
                "magnetic_axis_shift_mm": float(item["axis_shift_mm"]),
                "fixed_point_max_residual": float(max(row["residual"] for row in item["fixed_points"])),
                "manifold_x_count": int(len(item["manifolds"])),
                "manifold_unstable_expansion_mean": float(
                    np.mean([row["unstable_expansion"] for row in item["manifolds"]])
                ),
                "manifold_stable_anchor_max_residual": float(
                    np.nanmax(
                        [row["stable_inverse_anchor_residual"] for row in item["manifolds"]]
                    )
                ),
                "dpk_growth_p95": float(np.nanpercentile(item["trace"]["point_growth"], 95.0)),
                "dpk_svd_growth_p95": float(
                    np.nanpercentile(item["trace"]["point_svd_growth"], 95.0)
                ),
                "dpk_spectral_recurrence_fraction": float(
                    np.nanmean(item["trace"]["point_recurrent_surface"])
                ),
                "optimizer_residual_reduction": None
                if item["result"] is None
                else float(item["result"].validation.residual_reduction_fraction),
            }
            for item in case_rows
        ],
        "interpretation": {
            "nardon_width_geometry": "linear healed-edge continuation; nonlinear Poincare/Newton verification required",
            "phase": "the measured Nardon control span parameterizes a constant-width path; Newton O-chain phase is the nonlinear feedback sensor, but physical validation requires every point to remain inside a same-source healed coordinate domain",
            "dpk": "eigenvalue growth, SVD growth, and spectral recurrence are reported separately",
            "fpt": "pressure-driven beta background increment only; control-specific plasma screening not solved in cached run",
            "heat": "reported by the separate fusionsc field-line diffusion verification",
        },
        "production_readiness": {
            "accepted": False,
            "exploratory_demonstrated": (
                ["bounded Newton O-chain displacement target reached at fixed spectral island width"]
                if direct_phase_target_reached
                else []
            ),
            "validated": [
                "actual two-control vacuum-field span",
                "bounded resonant coefficient control",
                "Newton fixed-point continuation",
                "Poincare and DPk nonlinear audit",
                "full-return stable and unstable manifold tracing",
                "magnetic-axis displacement audit",
            ]
            + (
                ["bounded direct Newton O-chain phase control at fixed spectral island width"]
                if direct_phase_control_passed
                else []
            )
            + (
                ["Nardon/Newton phase closure in a shared PEST chart"]
                if phase_closure_passed
                and phase_projection_passed
                and phase_radial_alignment_passed
                and phase_coordinate_domain_passed
                and phase_b0_alignment["passed"]
                else []
            ),
            "blockers": [
                "hardware current calibration unavailable",
                "resonant surface uses a bounded healed-edge continuation",
                "control-specific self-consistent plasma screening not solved",
                "physical island-envelope width is not yet isolated from unique inner/outer manifold crossings",
            ]
            + ([] if direct_phase_target_reached else ["direct Newton O-chain displacement target failed its width/phase gates"])
            + ([] if phase_closure_passed else ["Nardon/Newton phase closure exceeds tolerance"])
            + ([] if phase_projection_passed else ["Newton O points do not invert accurately in the healed PEST surface stack"])
            + ([] if phase_radial_alignment_passed else ["Newton O points are not radially aligned with the Nardon resonant B0 surface"])
            + ([] if phase_coordinate_domain_passed else ["Newton O points lie outside the validated healed radial coordinate domain"])
            + ([] if phase_b0_alignment["passed"] else ["standard field and independent healed B0 fail the contravariant-radial/iota compatibility gate"]),
        },
    }
    (out / "w7x_actual_control_coil_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        out / "w7x_actual_control_coil_state.npz",
        controls=np.stack([item["controls"] for item in case_rows]),
        coefficient=np.asarray([item["chain"].coefficient for item in case_rows]),
        half_width_s=np.asarray([item["chain"].half_width for item in case_rows]),
        nardon_phase_shift_deg=np.asarray([item["nardon_phase_shift_deg"] for item in case_rows]),
        newton_phase_shift_deg=np.asarray([item["phase_shift_deg"] for item in case_rows]),
        phase_closure_error_deg=np.asarray([item["phase_closure_error_deg"] for item in case_rows]),
        phase_radial_shift_rms_s=np.asarray([item["phase_radial_shift_rms_s"] for item in case_rows]),
        phase_projection_rms_m=np.asarray([item["phase_projection_rms_m"] for item in case_rows]),
        axis_shift_mm=np.asarray([item["axis_shift_mm"] for item in case_rows]),
        labels=np.asarray([item["label"] for item in case_rows]),
    )
    _save_manifold_state(
        case_rows,
        out / "w7x_actual_control_coil_manifolds.npz",
    )
    print(json.dumps(payload["cases"], indent=2))
    print(f"outputs: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
