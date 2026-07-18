import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from pyna.toroidal.control.boundary_perturbation_candidates import (
    perturbation_candidate_nardon_response,
)
from pyna.toroidal.control.boundary_plasma_response import BoundaryPlasmaResponseInput
from pyna.toroidal.control.boundary_topology_cases import (
    BoundaryTopologyPlasmaFeedback,
    boundary_topology_case_from_arrays,
)
from pyna.toroidal.control import topoquest_fpt
from pyna.toroidal.control.topoquest_fpt import (
    TopoquestFEMFPTNeoclassicalRunner,
    TopoquestFPTBetaRampResult,
    TopoquestFPTBetaRampSpec,
    TopoquestFPTCachedResponseBasis,
    TopoquestFPTFieldPeriodRunner,
    TopoquestFPTPlasmaFeedbackAdapter,
    TopoquestFPTUnavailableError,
    TopoquestFPTVacuumControlState,
)


def _case(*, radial_count=3, radial_shift=0.0):
    phi = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    radial = np.linspace(0.25, 0.95, radial_count)
    PP, SS, TT = np.meshgrid(phi, radial, theta, indexing="ij")
    minor = 0.24 * np.sqrt(SS)
    R = 1.7 + radial_shift + minor * (1.0 + 0.03 * np.cos(2.0 * PP)) * np.cos(TT)
    Z = 0.82 * minor * np.sin(TT) + 0.012 * SS * np.sin(2.0 * PP)
    return boundary_topology_case_from_arrays(
        name="private stellarator",
        R_surf=R,
        Z_surf=Z,
        phi_vals=phi,
        theta_vals=theta,
        radial_labels=radial,
        iota_profile=np.linspace(0.55, 0.65, radial_count),
        denominator_B3=-0.75 * np.ones_like(R),
        nfp=2,
    )


def _request(controls=(0.2, -0.1), *, vacuum_delta_field=None):
    return BoundaryPlasmaResponseInput(
        controls=np.asarray(controls, dtype=float),
        control_labels=("upper", "lower"),
        vacuum_delta_field=vacuum_delta_field,
    )


class _SurfaceDeltaField:
    def B_at(self, R, Z, phi):
        R = np.asarray(R, dtype=float)
        Z = np.asarray(Z, dtype=float)
        phi = np.asarray(phi, dtype=float)
        return (
            2.0e-4 * np.cos(phi) + 1.0e-4 * Z,
            -1.5e-4 * np.sin(phi) + 0.5e-4 * (R - np.mean(R)),
            0.25e-4 * np.ones_like(R),
        )


class _RecordingRunner:
    def __init__(self, result_factory):
        self.result_factory = result_factory
        self.calls = []

    def __call__(self, spec, state):
        self.calls.append((spec, state))
        return self.result_factory(spec, state)


def test_beta_ramp_spec_and_result_validate_screening_contracts():
    spec = TopoquestFPTBetaRampSpec(beta_values=[0.0, 0.01, 0.02], response_beta=0.01)

    assert spec.beta_values == (0.0, 0.01, 0.02)
    assert spec.response_index == 1
    assert spec.case_alias == "private stellarator"
    with pytest.raises(ValueError, match="strictly increasing"):
        TopoquestFPTBetaRampSpec(beta_values=[0.0, 0.02, 0.01])
    with pytest.raises(ValueError, match="response_beta"):
        TopoquestFPTBetaRampSpec(beta_values=[0.0, 0.02], response_beta=0.01)
    with pytest.raises(ValueError, match="alias"):
        TopoquestFPTBetaRampSpec(beta_values=[0.0], case_alias="private/data")
    with pytest.raises(ValueError, match="exactly one"):
        TopoquestFPTBetaRampResult(
            beta=0.01,
            converged=True,
            production_ready=True,
        )
    with pytest.raises(ValueError, match="exactly one"):
        TopoquestFPTBetaRampResult(
            beta=0.01,
            converged=True,
            production_ready=True,
            tilde_b1=np.zeros((2, 2, 2), dtype=complex),
            delta_field=_SurfaceDeltaField(),
        )


def test_feedback_adapter_invokes_fake_runner_and_adds_direct_plasma_increment():
    case = _case()
    vacuum_field = _SurfaceDeltaField()
    request = _request(vacuum_delta_field=vacuum_field)
    vacuum = (1.0e-4 + 2.0e-5j) * np.ones_like(case.R_surf, dtype=complex)
    plasma = (3.0e-5 - 1.0e-5j) * np.ones_like(vacuum)
    replacement = _case(radial_shift=1.0e-3)
    runner = _RecordingRunner(
        lambda spec, state: TopoquestFPTBetaRampResult(
            beta=spec.response_beta,
            converged=True,
            production_ready=True,
            tilde_b1=plasma,
            response_case=replacement,
            readiness={"accepted": True, "mesh_gate": "passed"},
            metadata={"solver": "fake_fpt", "source_path": "not-public"},
        )
    )
    adapter = TopoquestFPTPlasmaFeedbackAdapter(
        spec=TopoquestFPTBetaRampSpec(beta_values=[0.0, 0.02]),
        runner=runner,
    )

    feedback = adapter(case, request, vacuum)

    projected_vacuum = perturbation_candidate_nardon_response(
        vacuum_field,
        replacement.R_surf,
        replacement.Z_surf,
        replacement.phi_vals,
        replacement.theta_vals,
        replacement.radial_labels,
        denominator_B3=replacement.denominator_B3,
    ).tilde_b1
    assert isinstance(feedback, BoundaryTopologyPlasmaFeedback)
    assert feedback.response_case is replacement
    np.testing.assert_allclose(feedback.tilde_b1, projected_vacuum + plasma)
    assert not np.allclose(projected_vacuum, vacuum)
    assert feedback.vacuum_delta_field is vacuum_field
    assert feedback.plasma_delta_field is None
    assert len(runner.calls) == 1
    assert runner.calls[0][1].request is request
    np.testing.assert_allclose(runner.calls[0][1].vacuum_tilde_b1, vacuum)
    assert feedback.metadata["beta"] == pytest.approx(0.02)
    assert feedback.metadata["readiness"]["accepted"] is True
    assert feedback.metadata["production_ready"] is True
    assert feedback.metadata["converged"] is True
    assert feedback.metadata["source_path"] == "<redacted>"
    assert (
        feedback.metadata["vacuum_response_representation"]
        == "cylindrical_delta_field_reprojected_on_response_case"
    )


def test_same_shape_replacement_without_vacuum_field_cannot_reuse_stale_samples():
    case = _case()
    replacement = _case(radial_shift=2.0e-3)
    vacuum = np.ones_like(case.R_surf, dtype=complex)
    runner = _RecordingRunner(
        lambda spec, state: TopoquestFPTBetaRampResult(
            beta=spec.response_beta,
            converged=True,
            production_ready=True,
            tilde_b1=np.zeros_like(replacement.R_surf, dtype=complex),
            response_case=replacement,
        )
    )
    adapter = TopoquestFPTPlasmaFeedbackAdapter(
        spec=TopoquestFPTBetaRampSpec(beta_values=[0.01]),
        runner=runner,
    )

    with pytest.raises(ValueError, match="sampling surfaces.*vacuum_delta_field"):
        adapter(case, _request(), vacuum)


def test_feedback_adapter_preserves_authoritative_total_field():
    case = _case()
    total_field = object()
    runner = _RecordingRunner(
        lambda spec, state: TopoquestFPTBetaRampResult(
            beta=spec.response_beta,
            converged=True,
            production_ready=True,
            tilde_b1=np.zeros_like(case.R_surf, dtype=complex),
            total_field=total_field,
        )
    )
    adapter = TopoquestFPTPlasmaFeedbackAdapter(
        spec=TopoquestFPTBetaRampSpec(beta_values=[0.01]),
        runner=runner,
        add_vacuum_tilde_b1=False,
    )

    feedback = adapter(case, _request(), np.zeros_like(case.R_surf, dtype=complex))

    assert feedback.total_field is total_field
    assert feedback.metadata["vacuum_response_representation"] == "not_included"
    assert feedback.metadata["spectrum_delta_components"] == ("plasma_delta_field",)


def test_feedback_adapter_projects_cylindrical_plasma_increment_with_nardon_helper():
    case = _case()
    field = _SurfaceDeltaField()
    request = _request(vacuum_delta_field=field)
    vacuum = np.zeros_like(case.R_surf, dtype=complex)
    expected = perturbation_candidate_nardon_response(
        field,
        case.R_surf,
        case.Z_surf,
        case.phi_vals,
        case.theta_vals,
        case.radial_labels,
        denominator_B3=case.denominator_B3,
    ).tilde_b1
    runner = _RecordingRunner(
        lambda spec, state: {
            "beta": spec.response_beta,
            "converged": True,
            "ready": False,
            "cylindrical_delta_field": field,
            "readiness": {"accepted": False, "issues": ("screening-only mesh",)},
        }
    )
    adapter = TopoquestFPTPlasmaFeedbackAdapter(
        spec=TopoquestFPTBetaRampSpec(beta_values=[0.01], require_production_readiness=False),
        runner=runner,
    )

    feedback = adapter(case, request, vacuum)

    np.testing.assert_allclose(feedback.tilde_b1, expected)
    assert feedback.vacuum_delta_field is field
    assert feedback.plasma_delta_field is field
    assert feedback.metadata["plasma_response_representation"] == "cylindrical_delta_field"
    assert feedback.metadata["production_ready"] is False
    assert feedback.metadata["readiness"]["issues"] == ("screening-only mesh",)


def test_mapping_runner_promotes_generation_metadata_into_production_feedback():
    generation_id = "sha256:" + "c" * 64
    case = _case()
    vacuum = np.zeros_like(case.R_surf, dtype=complex)
    runner = _RecordingRunner(
        lambda spec, state: {
            "beta": spec.response_beta,
            "converged": True,
            "production_ready": True,
            "tilde_b1": np.zeros_like(state.vacuum_tilde_b1),
            "readiness": {"accepted": True},
            "metadata": {
                "generation_id": generation_id,
                "runner_kind": "mapping",
            },
        }
    )
    spec = TopoquestFPTBetaRampSpec(
        beta_values=[0.01],
        require_production_readiness=True,
        generation_id=generation_id,
    )

    feedback = TopoquestFPTPlasmaFeedbackAdapter(
        spec=spec,
        runner=runner,
    )(case, _request(), vacuum)

    assert feedback.metadata["generation_id"] == generation_id
    assert feedback.metadata["runner_kind"] == "mapping"

    unbound = _RecordingRunner(
        lambda _spec, state: {
            "beta": 0.01,
            "converged": True,
            "production_ready": True,
            "tilde_b1": np.zeros_like(state.vacuum_tilde_b1),
            "readiness": {"accepted": True},
        }
    )
    with pytest.raises(RuntimeError, match="generation does not match"):
        TopoquestFPTPlasmaFeedbackAdapter(spec=spec, runner=unbound)(
            case, _request(), vacuum
        )


def test_live_generation_resolution_fails_closed_without_matching_echo():
    generation_id = "sha256:" + "9" * 64
    spec = TopoquestFPTBetaRampSpec(
        beta_values=[0.01],
        require_production_readiness=True,
        generation_id=generation_id,
    )

    with pytest.raises(RuntimeError, match="did not echo"):
        topoquest_fpt._live_result_generation_id(
            spec,
            SimpleNamespace(metadata={}),
            label="synthetic live FPT result",
        )
    with pytest.raises(ValueError, match="does not match"):
        topoquest_fpt._live_result_generation_id(
            spec,
            SimpleNamespace(
                metadata={"generation_id": "sha256:" + "8" * 64}
            ),
            label="synthetic live FPT result",
        )


def test_feedback_adapter_enforces_convergence_and_readiness_gates():
    case = _case()
    request = _request()
    vacuum = np.zeros_like(case.R_surf, dtype=complex)

    not_converged = _RecordingRunner(
        lambda spec, state: TopoquestFPTBetaRampResult(
            beta=spec.response_beta,
            converged=False,
            production_ready=True,
            tilde_b1=np.zeros_like(vacuum),
        )
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        TopoquestFPTPlasmaFeedbackAdapter(
            spec=TopoquestFPTBetaRampSpec(beta_values=[0.01]),
            runner=not_converged,
        )(case, request, vacuum)

    not_ready = _RecordingRunner(
        lambda spec, state: TopoquestFPTBetaRampResult(
            beta=spec.response_beta,
            converged=True,
            production_ready=False,
            tilde_b1=np.zeros_like(vacuum),
        )
    )
    with pytest.raises(RuntimeError, match="not production-ready"):
        TopoquestFPTPlasmaFeedbackAdapter(
            spec=TopoquestFPTBetaRampSpec(
                beta_values=[0.01],
                require_production_readiness=True,
            ),
            runner=not_ready,
        )(case, request, vacuum)


def test_cached_response_basis_is_a_dependency_free_callable_runner():
    case = _case()
    request = _request(controls=(0.25, -0.5))
    vacuum = 2.0e-5 * np.ones_like(case.R_surf, dtype=complex)
    basis_arrays = np.stack(
        [
            (1.0e-4 + 2.0e-5j) * np.ones_like(vacuum),
            (-0.4e-4 + 1.0e-5j) * np.ones_like(vacuum),
        ]
    )
    base = 0.2e-4 * np.ones_like(vacuum)
    basis = TopoquestFPTCachedResponseBasis(
        control_labels=request.control_labels,
        tilde_b1_basis=basis_arrays,
        base_tilde_b1=base,
        beta=0.015,
        response_case=case,
        readiness={"accepted": True},
    )
    adapter = TopoquestFPTPlasmaFeedbackAdapter(
        spec=TopoquestFPTBetaRampSpec(beta_values=[0.0, 0.015]),
        response_basis=basis,
    )

    feedback = adapter(case, request, vacuum)

    expected_plasma = base + request.controls[0] * basis_arrays[0] + request.controls[1] * basis_arrays[1]
    np.testing.assert_allclose(feedback.tilde_b1, vacuum + expected_plasma)
    assert feedback.metadata["response_source"] == "cached_response_basis"
    assert feedback.metadata["beta"] == pytest.approx(0.015)


def test_production_cached_response_requires_matching_generation_id():
    generation_id = "sha256:" + "a" * 64
    case = _case()
    request = _request(controls=(0.25, -0.5))
    vacuum = 2.0e-5 * np.ones_like(case.R_surf, dtype=complex)
    basis = TopoquestFPTCachedResponseBasis(
        control_labels=request.control_labels,
        tilde_b1_basis=np.zeros((2,) + vacuum.shape, dtype=complex),
        beta=0.015,
        production_ready=True,
        generation_id=generation_id,
        readiness={"accepted": True},
    )

    feedback = TopoquestFPTPlasmaFeedbackAdapter(
        spec=TopoquestFPTBetaRampSpec(
            beta_values=[0.015],
            require_production_readiness=True,
            generation_id=generation_id,
        ),
        response_basis=basis,
    )(case, request, vacuum)

    assert feedback.metadata["generation_id"] == generation_id

    with pytest.raises(ValueError, match="generation"):
        TopoquestFPTPlasmaFeedbackAdapter(
            spec=TopoquestFPTBetaRampSpec(
                beta_values=[0.015],
                require_production_readiness=True,
                generation_id="sha256:" + "b" * 64,
            ),
            response_basis=basis,
        )(case, request, vacuum)


def test_cached_basis_promotes_metadata_generation_and_rejects_conflicts():
    generation_id = "sha256:" + "d" * 64
    case = _case()
    request = _request()
    vacuum = np.zeros_like(case.R_surf, dtype=complex)
    basis = TopoquestFPTCachedResponseBasis(
        control_labels=request.control_labels,
        tilde_b1_basis=np.zeros((2,) + vacuum.shape, dtype=complex),
        beta=0.015,
        production_ready=True,
        readiness={"accepted": True},
        metadata={"generation_id": generation_id},
    )

    feedback = TopoquestFPTPlasmaFeedbackAdapter(
        spec=TopoquestFPTBetaRampSpec(
            beta_values=[0.015],
            require_production_readiness=True,
            generation_id=generation_id,
        ),
        response_basis=basis,
    )(case, request, vacuum)

    assert basis.generation_id == generation_id
    assert feedback.metadata["generation_id"] == generation_id
    with pytest.raises(ValueError, match="disagrees"):
        TopoquestFPTCachedResponseBasis(
            control_labels=request.control_labels,
            tilde_b1_basis=np.zeros((2,) + vacuum.shape, dtype=complex),
            beta=0.015,
            generation_id=generation_id,
            metadata={"generation_id": "sha256:" + "e" * 64},
        )


def test_production_cached_response_rejects_unbound_or_malformed_generation():
    case = _case()
    request = _request(controls=(0.25, -0.5))
    vacuum = 2.0e-5 * np.ones_like(case.R_surf, dtype=complex)
    basis = TopoquestFPTCachedResponseBasis(
        control_labels=request.control_labels,
        tilde_b1_basis=np.zeros((2,) + vacuum.shape, dtype=complex),
        beta=0.015,
        production_ready=True,
        readiness={"accepted": True},
    )

    with pytest.raises(RuntimeError, match="bound generation_id"):
        TopoquestFPTPlasmaFeedbackAdapter(
            spec=TopoquestFPTBetaRampSpec(
                beta_values=[0.015],
                require_production_readiness=True,
            ),
            response_basis=basis,
        )(case, request, vacuum)

    with pytest.raises(ValueError, match="full lowercase sha256"):
        TopoquestFPTBetaRampSpec(
            beta_values=[0.015],
            generation_id="4dec7ac708d",
        )


def test_capability_diagnostic_names_missing_petsc_and_dolfinx_without_fallback(monkeypatch):
    availability = {"topoquest": True, "petsc4py": False, "dolfinx": False}
    monkeypatch.setattr(
        topoquest_fpt.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(name=name) if availability[name] else None,
    )

    capability = topoquest_fpt.diagnose_topoquest_fpt_capability()

    assert capability.available is False
    assert capability.current_process_available is False
    assert capability.missing_dependencies == ("petsc4py", "dolfinx")
    assert "petsc4py" in capability.message
    assert "dolfinx" in capability.message
    assert "No scalar-factor fallback" in capability.message
    with pytest.raises(TopoquestFPTUnavailableError, match="petsc4py, dolfinx"):
        topoquest_fpt.require_topoquest_fpt_capability()


def test_neoclassical_capability_requires_aletheia_without_changing_base_fpt(monkeypatch):
    availability = {
        "topoquest": True,
        "petsc4py": True,
        "dolfinx": True,
        "aletheia": False,
    }
    monkeypatch.setattr(
        topoquest_fpt.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(name=name) if availability[name] else None,
    )

    base = topoquest_fpt.diagnose_topoquest_fpt_capability()
    neoclassical = topoquest_fpt.diagnose_topoquest_neoclassical_fpt_capability()

    assert base.available is True
    assert base.current_process_available is True
    assert neoclassical.available is False
    assert neoclassical.current_process_available is False
    assert neoclassical.missing_dependencies == ("aletheia",)
    assert neoclassical.as_dict()["aletheia_available"] is False
    with pytest.raises(TopoquestFPTUnavailableError, match="aletheia"):
        topoquest_fpt.require_topoquest_neoclassical_fpt_capability()


def _external_probe_payload(*, aletheia=True, petsc_cuda=True):
    return {
        "modules": {
            "topoquest": True,
            "petsc4py": True,
            "dolfinx": True,
            "ufl": True,
            "basix": True,
            "aletheia": bool(aletheia),
        },
        "versions": {
            "topoquest": "0.11",
            "petsc4py": "3.25",
            "dolfinx": "0.10",
            "ufl": "2025.2",
            "basix": "0.10",
            "aletheia": "0.1" if aletheia else None,
            "petsc": "3.25.1",
        },
        "production_entrypoint": True,
        "neoclassical_entrypoint": True,
        "petsc_cuda": bool(petsc_cuda),
    }


def test_external_runtime_probe_is_path_free_and_does_not_change_current_capability(
    monkeypatch,
    tmp_path,
):
    python = tmp_path / "venv" / "bin" / "python3"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    python.chmod(0o755)
    availability = {"topoquest": True, "petsc4py": False, "dolfinx": False}
    monkeypatch.setattr(
        topoquest_fpt.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(name=name) if availability[name] else None,
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(_external_probe_payload()) + "\n",
            stderr="",
        )

    monkeypatch.setattr(topoquest_fpt.subprocess, "run", fake_run)
    current = topoquest_fpt.diagnose_topoquest_fpt_capability()
    external = topoquest_fpt.diagnose_topoquest_fpt_external_runtime(
        python,
        runtime_alias="FEM runtime",
    )

    assert current.available is False
    assert external.available is True
    assert external.petsc_cuda_available is True
    assert external.source == "explicit_python"
    assert calls[0][0][0] == str(python)
    assert calls[0][1]["env"]["PETSC_DIR"] == str(tmp_path / "petsc")
    report = external.as_dict()
    assert report["runtime_alias"] == "FEM runtime"
    assert str(tmp_path) not in repr(report)
    with pytest.raises(TopoquestFPTUnavailableError, match="petsc4py, dolfinx"):
        topoquest_fpt.require_topoquest_fpt_capability()


def test_neoclassical_external_runtime_requires_aletheia(monkeypatch, tmp_path):
    python = tmp_path / "python3"
    python.write_text("", encoding="utf-8")
    python.chmod(0o755)
    monkeypatch.setattr(
        topoquest_fpt.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(_external_probe_payload(aletheia=False)) + "\n",
            stderr="",
        ),
    )

    base = topoquest_fpt.diagnose_topoquest_fpt_external_runtime(python)
    neoclassical = topoquest_fpt.diagnose_topoquest_neoclassical_fpt_external_runtime(
        python
    )

    assert base.available is True
    assert neoclassical.available is False
    assert neoclassical.missing_dependencies == ("aletheia",)


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (SimpleNamespace(returncode=0, stdout="not-json\n", stderr=""), "JSONDecodeError"),
        (SimpleNamespace(returncode=7, stdout="", stderr="failed"), "ExternalRuntimeProbeExit"),
    ],
)
def test_external_runtime_probe_reports_structured_failures(
    monkeypatch,
    tmp_path,
    failure,
    expected,
):
    python = tmp_path / "python3"
    python.write_text("", encoding="utf-8")
    python.chmod(0o755)
    monkeypatch.setattr(topoquest_fpt.subprocess, "run", lambda *_a, **_k: failure)

    result = topoquest_fpt.diagnose_topoquest_fpt_external_runtime(python)

    assert result.available is False
    assert result.probe_succeeded is False
    assert result.error_type == expected


def test_external_runtime_probe_reports_timeout(monkeypatch, tmp_path):
    python = tmp_path / "python3"
    python.write_text("", encoding="utf-8")
    python.chmod(0o755)

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("probe", 0.01)

    monkeypatch.setattr(topoquest_fpt.subprocess, "run", timeout)

    result = topoquest_fpt.diagnose_topoquest_fpt_external_runtime(
        python,
        timeout=0.01,
    )

    assert result.available is False
    assert result.error_type == "TimeoutExpired"


@pytest.mark.parametrize("coupled", [False, True])
def test_lazy_field_period_runner_calls_public_topoquest_entrypoint_and_samples_sections(
    monkeypatch,
    coupled,
):
    generation_id = "sha256:" + "f" * 64
    case = _case()
    vacuum_field = _SurfaceDeltaField()
    request = _request(vacuum_delta_field=vacuum_field)
    request = BoundaryPlasmaResponseInput(
        controls=request.controls,
        control_labels=request.control_labels,
        baseline_field=object(),
        vacuum_delta_field=vacuum_field,
    )
    state = TopoquestFPTVacuumControlState(
        case=case,
        request=request,
        vacuum_tilde_b1=np.zeros_like(case.R_surf, dtype=complex),
        vacuum_delta_field=vacuum_field,
    )
    plan = SimpleNamespace(
        n_fp=2,
        sections=(SimpleNamespace(phi=0.0), SimpleNamespace(phi=0.5 * np.pi)),
    )
    contexts = (object(), object())
    calls = []

    def fake_run(plan_arg, contexts_arg, **kwargs):
        calls.append((plan_arg, contexts_arg, kwargs))
        if coupled:
            ramp = SimpleNamespace(
                beta_values=(0.0, 0.02),
                converged=True,
                metadata=dict(kwargs["metadata"]),
                steps=tuple(
                    SimpleNamespace(solve=SimpleNamespace(beta_index=beta_index))
                    for beta_index in range(2)
                ),
            )
        else:
            sections = []
            for section_index in range(2):
                steps = tuple(
                    SimpleNamespace(solve=SimpleNamespace(section=section_index, beta_index=beta_index))
                    for beta_index in range(2)
                )
                sections.append(SimpleNamespace(result=SimpleNamespace(steps=steps)))
            ramp = SimpleNamespace(
                beta_values=(0.0, 0.02),
                converged=True,
                metadata=dict(kwargs["metadata"]),
                sections=tuple(sections),
            )
        readiness = SimpleNamespace(
            accepted=True,
            issues=(),
            as_dict=lambda: {"accepted": True, "mesh_gate": "passed"},
        )
        return SimpleNamespace(ramp=ramp, production_readiness=readiness)

    def fake_sample(solve, points):
        n_points = np.asarray(points).shape[0]
        scale = 10.0 * (solve.section + 1) + solve.beta_index
        return SimpleNamespace(
            dB_R=np.full(n_points, scale),
            dB_Z=np.full(n_points, scale + 1.0),
            dB_Phi=np.full(n_points, scale + 2.0),
        )

    def fake_sample_coupled(solve, points_by_section):
        return {
            section_index: fake_sample(
                SimpleNamespace(section=section_index, beta_index=solve.beta_index),
                points,
            )
            for section_index, points in points_by_section.items()
        }

    modules = {
        "topoquest.mesh.field_adapter": SimpleNamespace(
            run_fpt_field_period_beta_ramp_from_cylindrical_fields=fake_run
        ),
        "topoquest.mesh.fpt_solution": SimpleNamespace(
            sample_fpt_solution_at_points=fake_sample,
            sample_coupled_field_period_solution_at_points_by_section=fake_sample_coupled,
        ),
    }
    monkeypatch.setattr(topoquest_fpt, "require_topoquest_fpt_capability", lambda: None)
    monkeypatch.setattr(topoquest_fpt.importlib, "import_module", lambda name: modules[name])
    runner = TopoquestFPTFieldPeriodRunner(
        plan=plan,
        section_contexts=contexts,
        J0=object(),
        runner_kwargs={"coupled": coupled},
    )

    result = runner(
        TopoquestFPTBetaRampSpec(
            beta_values=[0.0, 0.02],
            require_production_readiness=True,
            generation_id=generation_id,
        ),
        state,
    )

    assert result.beta == pytest.approx(0.02)
    assert result.converged is True
    assert result.production_ready is True
    assert result.generation_id == generation_id
    assert result.metadata["generation_id"] == generation_id
    assert result.readiness["mesh_gate"] == "passed"
    assert calls[0][0] is plan
    assert calls[0][1] == contexts
    assert calls[0][2]["delta_B_ext"] is vacuum_field
    assert calls[0][2]["metadata"] == {
        "case_alias": "private stellarator",
        "generation_id": generation_id,
    }
    assert calls[0][2]["coupled"] is coupled
    # Full-torus phi slices repeat the two solved sections for n_fp=2.
    np.testing.assert_allclose(result.delta_field.delta_BR[:, 0, 0], [11.0, 21.0, 11.0, 21.0])
    np.testing.assert_allclose(result.delta_field.delta_BZ[:, 0, 0], [12.0, 22.0, 12.0, 22.0])
    np.testing.assert_allclose(result.delta_field.delta_BPhi[:, 0, 0], [13.0, 23.0, 13.0, 23.0])


def test_rectangular_fem_runner_preserves_full_neoclassical_factory_contract(monkeypatch):
    generation_id = "sha256:" + "1" * 64
    case = _case()
    vacuum_field = _SurfaceDeltaField()
    request = BoundaryPlasmaResponseInput(
        controls=np.asarray([0.2, -0.1]),
        control_labels=("upper", "lower"),
        baseline_field=object(),
        baseline_equilibrium=object(),
        vacuum_delta_field=vacuum_field,
    )
    state = TopoquestFPTVacuumControlState(
        case=case,
        request=request,
        vacuum_tilde_b1=np.zeros_like(case.R_surf, dtype=complex),
        vacuum_delta_field=vacuum_field,
    )
    wall = np.asarray([[0.0, -1.0], [3.0, -1.0], [3.0, 1.0], [0.0, 1.0]])
    plan = SimpleNamespace(
        n_fp=2,
        sections=(
            SimpleNamespace(phi=0.0, wall_curve=wall),
            SimpleNamespace(phi=0.5 * np.pi, wall_curve=wall),
            SimpleNamespace(phi=0.75 * np.pi, wall_curve=wall),
        ),
    )
    def closure_factory(beta, previous, problems, section_assemblies):
        return beta, previous, problems, section_assemblies
    calls = {}

    def coefficient_payload(plan_arg, **kwargs):
        calls["coefficient"] = (plan_arg, kwargs)
        return SimpleNamespace(
            payload={"coefficient": "payload"},
            audits=tuple(
                SimpleNamespace(
                    section_index=index,
                    phi=section.phi,
                    n_points=len(kwargs["points_by_section"][index]),
                    point_source=kwargs["point_source"],
                )
                for index, section in enumerate(plan.sections)
            ),
        )

    def residual_payload(plan_arg, **kwargs):
        calls["residual"] = (plan_arg, kwargs)
        return SimpleNamespace(
            payload={"residual": "payload"},
            audits=tuple(
                SimpleNamespace(
                    section_index=index,
                    phi=section.phi,
                    n_points=len(kwargs["points_by_section"][index]),
                    point_source=kwargs["point_source"],
                )
                for index, section in enumerate(plan.sections)
            ),
        )

    def run_production(plan_arg, paths, **kwargs):
        calls["production"] = (plan_arg, paths, kwargs)
        context_build = SimpleNamespace(contexts=(object(), object(), object()))
        kwargs["coefficient_payload"](context_build)
        kwargs["residual_payload"](context_build)
        steps = tuple(
            SimpleNamespace(
                beta=beta,
                solve=SimpleNamespace(beta_index=index),
                assembly=SimpleNamespace(
                    diagnostics={
                        "parallel_current_closure": {
                            "enabled": True,
                            "n_rows": 6,
                        }
                    }
                ),
            )
            for index, beta in enumerate((0.0, 0.02))
        )
        return SimpleNamespace(
            ran=True,
            skipped_reason=None,
            ramp=SimpleNamespace(steps=steps, converged=True),
            context_build=context_build,
            metadata={
                "mesh_phase_validated": True,
                "mesh_phase_audit": ({"order": 0}, {"order": 1}, {"order": 2}),
                "generation_id": kwargs["metadata"]["generation_id"],
            },
        )

    def sample_coupled(solve, points_by_section):
        return {
            section_index: SimpleNamespace(
                dB_R=np.full(np.asarray(points).shape[0], 10.0 * (section_index + 1) + solve.beta_index),
                dB_Z=np.full(np.asarray(points).shape[0], 20.0 * (section_index + 1) + solve.beta_index),
                dB_Phi=np.full(np.asarray(points).shape[0], 30.0 * (section_index + 1) + solve.beta_index),
            )
            for section_index, points in points_by_section.items()
        }

    modules = {
        "topoquest.mesh.field_adapter": SimpleNamespace(
            fpt_coefficient_payload_from_cylindrical_fields=coefficient_payload,
            fpt_external_field_residual_payload_from_cylindrical_fields=residual_payload,
            points_by_section_from_contexts=lambda contexts: {
                index: np.asarray([[1.0 + index, 0.0]])
                for index, _context in enumerate(contexts)
            },
        ),
        "topoquest.solvers.fem_fpt.production": SimpleNamespace(
            run_rectangular_fem_fpt_msh_payloads=run_production,
        ),
        "topoquest.mesh.fpt_solution": SimpleNamespace(
            sample_coupled_field_period_solution_at_points_by_section=sample_coupled,
        ),
        "topoquest.mesh.stellarator_plan": SimpleNamespace(
            points_in_polygon_rz=lambda points, polygon: np.ones(len(points), dtype=bool),
        ),
    }
    monkeypatch.setattr(topoquest_fpt, "require_topoquest_neoclassical_fpt_capability", lambda: None)
    monkeypatch.setattr(topoquest_fpt.importlib, "import_module", lambda name: modules[name])
    runner = TopoquestFEMFPTNeoclassicalRunner(
        plan=plan,
        msh_paths=("section0.msh", "section1.msh", "section2.msh"),
        parallel_current_closure_factory=closure_factory,
        J0=object(),
        response_case=case,
        readiness_evaluator=lambda raw: {"accepted": raw.ramp.converged, "gate": "synthetic"},
    )

    result = runner(
        TopoquestFPTBetaRampSpec(
            beta_values=(0.0, 0.02),
            require_production_readiness=True,
            generation_id=generation_id,
        ),
        state,
    )

    assert result.converged is True
    assert result.production_ready is True
    assert result.generation_id == generation_id
    assert result.metadata["generation_id"] == generation_id
    assert result.readiness["neoclassical_closure_refreshed_per_beta"] is True
    assert result.readiness["gate"] == "synthetic"
    assert calls["coefficient"][0] is plan
    assert calls["coefficient"][1]["point_source"] == "context_vertices"
    assert calls["residual"][1]["delta_B_ext"] is vacuum_field
    assert calls["production"][1] == ("section0.msh", "section1.msh", "section2.msh")
    assert calls["production"][2]["parallel_current_closure_factory"] is closure_factory
    assert calls["production"][2]["metadata"]["generation_id"] == generation_id
    assert calls["production"][2]["metadata"]["plasma_closure"].endswith("per_beta")
    np.testing.assert_allclose(result.delta_field.delta_BR[:, 0, 0], [11.0, 21.0, 11.0, 21.0])
    assert result.delta_field.sampling_audit["method"] == "periodic_linear_between_fem_sections"
    assert all(row["point_source"] == "context_vertices" for row in result.metadata["coefficient_sampling"])


def test_rectangular_fem_runner_does_not_invent_production_readiness(monkeypatch):
    case = _case()
    wall = np.asarray([[0.0, -1.0], [3.0, -1.0], [3.0, 1.0], [0.0, 1.0]])
    plan = SimpleNamespace(
        n_fp=2,
        sections=(
            SimpleNamespace(phi=0.0, wall_curve=wall),
            SimpleNamespace(phi=0.5 * np.pi, wall_curve=wall),
            SimpleNamespace(phi=0.75 * np.pi, wall_curve=wall),
        ),
    )
    request = BoundaryPlasmaResponseInput(
        controls=np.zeros(2),
        control_labels=("upper", "lower"),
        baseline_field=object(),
        vacuum_delta_field=_SurfaceDeltaField(),
    )
    state = TopoquestFPTVacuumControlState(
        case=case,
        request=request,
        vacuum_tilde_b1=np.zeros_like(case.R_surf, dtype=complex),
        vacuum_delta_field=request.vacuum_delta_field,
    )
    def sampled_payload(*args, **kwargs):
        return SimpleNamespace(payload={}, audits=())

    def run_production(*args, **kwargs):
        context_build = SimpleNamespace(contexts=(object(), object(), object()))
        kwargs["coefficient_payload"](context_build)
        kwargs["residual_payload"](context_build)
        return SimpleNamespace(
            ran=True,
            ramp=SimpleNamespace(
                steps=(
                    SimpleNamespace(
                        beta=0.01,
                        solve=object(),
                        assembly=SimpleNamespace(
                            diagnostics={
                                "parallel_current_closure": {
                                    "enabled": True,
                                    "n_rows": 3,
                                }
                            }
                        ),
                    ),
                ),
                converged=True,
            ),
            context_build=context_build,
            metadata={"mesh_phase_validated": True, "mesh_phase_audit": ()},
        )

    modules = {
        "topoquest.mesh.field_adapter": SimpleNamespace(
            fpt_coefficient_payload_from_cylindrical_fields=sampled_payload,
            fpt_external_field_residual_payload_from_cylindrical_fields=sampled_payload,
            points_by_section_from_contexts=lambda contexts: {
                index: np.asarray([[1.0 + index, 0.0]])
                for index, _context in enumerate(contexts)
            },
        ),
        "topoquest.solvers.fem_fpt.production": SimpleNamespace(
            run_rectangular_fem_fpt_msh_payloads=run_production,
        ),
        "topoquest.mesh.fpt_solution": SimpleNamespace(
            sample_coupled_field_period_solution_at_points_by_section=lambda solve, points: {
                index: SimpleNamespace(
                    dB_R=np.zeros(np.asarray(values).shape[0]),
                    dB_Z=np.zeros(np.asarray(values).shape[0]),
                    dB_Phi=np.zeros(np.asarray(values).shape[0]),
                )
                for index, values in points.items()
            }
        ),
        "topoquest.mesh.stellarator_plan": SimpleNamespace(
            points_in_polygon_rz=lambda points, polygon: np.ones(len(points), dtype=bool),
        ),
    }
    monkeypatch.setattr(topoquest_fpt, "require_topoquest_neoclassical_fpt_capability", lambda: None)
    monkeypatch.setattr(topoquest_fpt.importlib, "import_module", lambda name: modules[name])
    runner = TopoquestFEMFPTNeoclassicalRunner(
        plan=plan,
        msh_paths=("a.msh", "b.msh", "c.msh"),
        parallel_current_closure_factory=lambda *args: object(),
        J0=object(),
    )

    result = runner(TopoquestFPTBetaRampSpec(beta_values=(0.01,)), state)

    assert result.production_ready is None
    assert result.readiness["production_gate_supplied"] is False


def test_rectangular_fem_runner_requires_three_field_period_sections():
    plan = SimpleNamespace(
        n_fp=2,
        sections=(SimpleNamespace(phi=0.0), SimpleNamespace(phi=0.5 * np.pi)),
    )

    with pytest.raises(ValueError, match="at least three sections"):
        TopoquestFEMFPTNeoclassicalRunner(
            plan=plan,
            msh_paths=("a.msh", "b.msh"),
            parallel_current_closure_factory=lambda *args: object(),
            J0=object(),
        )


def test_surface_response_uses_periodic_linear_section_interpolation():
    plan = SimpleNamespace(
        n_fp=2,
        sections=(
            SimpleNamespace(phi=0.0),
            SimpleNamespace(phi=0.5 * np.pi),
            SimpleNamespace(phi=0.75 * np.pi),
        ),
    )

    lower, upper, weight, span = topoquest_fpt._periodic_section_interpolation(
        plan,
        np.asarray([0.25 * np.pi, 0.875 * np.pi]),
    )

    np.testing.assert_array_equal(lower, [0, 2])
    np.testing.assert_array_equal(upper, [1, 0])
    np.testing.assert_allclose(weight, [0.5, 0.5])
    np.testing.assert_allclose(span, [0.5 * np.pi, 0.25 * np.pi])
