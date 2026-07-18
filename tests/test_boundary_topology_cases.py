import numpy as np
import pytest

from pyna.fields import VectorFieldCylind
from pyna.plot.j_streamlines import GriddedPestVectorField
from pyna.toroidal.coils.boundary_local import boundary_dipole_mode_actuator_array_from_surface
from pyna.toroidal.control.boundary_plasma_response import BoundaryPlasmaResponseInput
from pyna.toroidal.control.boundary_topology_cases import (
    BoundaryHeatTargetRegion,
    BoundaryTopologyCaseBackend,
    BoundaryTopologyObservableSpec,
    boundary_topology_case_from_arrays,
    boundary_topology_case_observable_builder,
    build_boundary_dipole_spectrum_library,
    load_boundary_topology_case_npz,
    make_boundary_topology_control_problem,
    vmec_boundary_topology_case_from_wout,
)
from pyna.toroidal.control.boundary_topology_design import boundary_response_observables
from pyna.toroidal.control.heat_contracts import BoundaryTopologyHeatState
from pyna.toroidal.control.reduced_heat import ReducedSpectralHeatModel
from pyna.toroidal.control.boundary_field_basis import (
    boundary_field_actuator_array_from_grid_fields,
    cylindrical_vector_field_from_array,
)
from pyna.toroidal.control.boundary_perturbation_candidates import (
    perturbation_candidate_nardon_response,
)


def _synthetic_case(n_phi=6, n_radial=4, n_theta=24, *, nfp=2):
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    radial = np.linspace(0.2, 0.95, n_radial)
    PP, SS, TT = np.meshgrid(phi, radial, theta, indexing="ij")
    minor = 0.34 * np.sqrt(SS)
    R = 1.8 + minor * (1.0 + 0.04 * np.cos(2.0 * PP)) * np.cos(TT)
    Z = 0.86 * minor * np.sin(TT) + 0.025 * SS * np.sin(2.0 * PP)
    iota = np.linspace(0.62, 0.72, n_radial)
    return boundary_topology_case_from_arrays(
        name="synthetic stellarator",
        R_surf=R,
        Z_surf=Z,
        phi_vals=phi,
        theta_vals=theta,
        radial_labels=radial,
        iota_profile=iota,
        denominator_B3=-0.8 * np.ones_like(R),
        nfp=nfp,
        metadata={"source_kind": "synthetic"},
    )


def _small_response_library():
    case = _synthetic_case()
    array = boundary_dipole_mode_actuator_array_from_surface(
        case.R_surf,
        case.Z_surf,
        case.phi_vals,
        case.theta_vals,
        modes=[(3, 2)],
        phi_indices=[0, 2, 4],
        theta_indices=[0, 6, 12, 18],
        radius=0.035,
        unit_moment=60.0,
        clearance=0.12,
    )
    library = build_boundary_dipole_spectrum_library(case, array, m_max=5, n_max=3)
    return case, array, library


class _SentinelField:
    def __init__(self, name, components):
        self.name = name
        self.components = tuple(float(value) for value in components)

    def B_at(self, R, Z, phi):
        shape = np.broadcast_shapes(np.shape(R), np.shape(Z), np.shape(phi))
        return tuple(np.full(shape, value, dtype=float) for value in self.components)


def _grid_field(*, nfp, scale=1.0):
    R = np.linspace(1.0, 2.6, 7)
    Z = np.linspace(-0.8, 0.8, 6)
    phi = np.linspace(0.0, 2.0 * np.pi / nfp, 6, endpoint=False)
    RR, ZZ, PP = np.meshgrid(R, Z, phi, indexing="ij")
    values = np.stack(
        (
            scale * 0.02 * RR * np.cos(nfp * PP),
            scale * 0.01 * ZZ,
            scale * (1.0 + 0.01 * np.sin(nfp * PP)),
        ),
        axis=-1,
    )
    return cylindrical_vector_field_from_array(
        values,
        R,
        Z,
        phi,
        component_order=("BR", "BZ", "BPhi"),
        nfp=nfp,
    )


def test_boundary_dipole_spectrum_library_combines_unit_responses_linearly():
    _case, _array, library = _small_response_library()
    controls = np.array([0.3, -0.2])

    combined = library.combined_tilde_b1(controls)
    expected = controls[0] * library.responses[0].tilde_b1 + controls[1] * library.responses[1].tilde_b1
    spectrum = library.combined_spectrum(controls)

    np.testing.assert_allclose(combined, expected)
    np.testing.assert_allclose(spectrum.dBr_grid, np.moveaxis(combined, 1, 0))
    assert spectrum.radial_labels == pytest.approx(library.case.radial_labels)
    assert spectrum.metadata["coordinate_system"] == "PEST"


def test_grid_actuator_field_period_must_match_case_before_sampling():
    case = _synthetic_case(nfp=1)
    actuators = boundary_field_actuator_array_from_grid_fields(
        (_grid_field(nfp=2, scale=0.01),),
        labels=("nfp2",),
    )

    with pytest.raises(ValueError, match=r"nfp=2.*case\.nfp=1"):
        build_boundary_dipole_spectrum_library(case, actuators)


def test_grid_response_library_propagates_surface_b0_and_delta_signatures():
    background = _grid_field(nfp=2)
    delta = _grid_field(nfp=2, scale=0.02)
    phi = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    radial = np.linspace(0.25, 0.9, 3)
    PP, SS, TT = np.meshgrid(phi, radial, theta, indexing="ij")
    R = 1.8 + 0.25 * np.sqrt(SS) * np.cos(TT)
    Z = 0.3 * np.sqrt(SS) * np.sin(TT)
    denominator_B_phi = background.interpolate_at(R, Z, PP)[2]
    case = boundary_topology_case_from_arrays(
        name="signed grid case",
        R_surf=R,
        Z_surf=Z,
        phi_vals=phi,
        theta_vals=theta,
        radial_labels=radial,
        iota_profile=np.linspace(0.6, 0.7, radial.size),
        denominator_B_phi=denominator_B_phi,
        nfp=2,
        background_field=background,
    )
    actuators = boundary_field_actuator_array_from_grid_fields(
        (delta,),
        labels=("grid_delta",),
    )

    library = build_boundary_dipole_spectrum_library(case, actuators)
    response = library.responses[0]
    combined = library.combined_spectrum([0.4])

    assert response.spectrum.surface_signature is not None
    assert response.spectrum.background_field_signature is not None
    assert response.spectrum.delta_field_signature is not None
    assert (
        response.spectrum.surface_signature["background_field_signature"]
        == response.spectrum.background_field_signature
    )
    assert combined.surface_signature == response.spectrum.surface_signature
    assert combined.background_field_signature == response.spectrum.background_field_signature
    assert combined.delta_field_signature is not None


def test_case_backend_exposes_joint_island_chaos_heat_and_core_rows():
    case, array, library = _small_response_library()
    heat_model = ReducedSpectralHeatModel(
        phi_values=np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False),
        s_values=np.linspace(0.0, 1.0, 64),
        control_center_s={array.control_labels[1]: 0.08},
    )
    backend = BoundaryTopologyCaseBackend(
        library=library,
        n_values=(2,),
        m_values=(3,),
        sigma_threshold=0.0,
        heat_model=heat_model,
    )
    request = BoundaryPlasmaResponseInput(
        controls=np.array([0.25, -0.15]),
        control_labels=array.control_labels,
    )
    snapshot = backend.evaluate(request)
    spec = BoundaryTopologyObservableSpec(
        resonant_modes=((3, 2),),
        chaos_regions=((0.2, 0.95),),
        chaos_labels=("edge",),
        heat_regions=(BoundaryHeatTargetRegion("outer", (0.55, 0.8), weight=2.0),),
        core_radial_max=0.45,
    )

    rows = boundary_topology_case_observable_builder(spec)(snapshot, request)

    assert "island.m3.n2.half_width" in rows.labels
    assert "island.m3.n2.coefficient_real" in rows.labels
    assert "chaos.edge" in rows.labels
    assert "heat.total_power" in rows.labels
    assert "heat.rms_width_s" in rows.labels
    assert "heat_region.outer" in rows.labels
    assert "core.field.radial_leakage_rms" in rows.labels
    assert np.all(np.isfinite(rows.values))
    state = snapshot.metadata["boundary_topology_state"]
    assert state.heat is not None
    assert state.spectrum.metadata["plasma_response_model"] == "vacuum"


def test_feedback_fields_are_consistent_across_spectrum_heat_and_nonlinear_callbacks():
    case, array, library = _small_response_library()
    background = _SentinelField("B0", (0.0, 0.0, 1.0))
    vacuum_delta = _SentinelField("deltaB_vac", (1.2e-4, -0.4e-4, 0.2e-4))
    plasma_delta = _SentinelField("deltaB_plasma", (-0.3e-4, 0.8e-4, 0.1e-4))
    total = _SentinelField("B_total", (0.9e-4, 0.4e-4, 1.0 + 0.3e-4))
    vacuum_response = perturbation_candidate_nardon_response(
        vacuum_delta,
        case.R_surf,
        case.Z_surf,
        case.phi_vals,
        case.theta_vals,
        case.radial_labels,
        denominator_B3=case.denominator_B3,
    ).tilde_b1
    plasma_response = perturbation_candidate_nardon_response(
        plasma_delta,
        case.R_surf,
        case.Z_surf,
        case.phi_vals,
        case.theta_vals,
        case.radial_labels,
        denominator_B3=case.denominator_B3,
    ).tilde_b1
    authoritative_tilde = vacuum_response + plasma_response
    captured = {}

    def feedback(_case, _request, _vacuum_tilde):
        return {
            "tilde_b1": authoritative_tilde,
            "background_field": background,
            "vacuum_delta_field": vacuum_delta,
            "plasma_delta_field": plasma_delta,
            "total_field": total,
            "metadata": {
                "response_model": "sentinel_plasma",
                "spectrum_delta_components": (
                    "vacuum_delta_field",
                    "plasma_delta_field",
                ),
            },
        }

    class RecordingHeatModel:
        def evaluate(self, response_case, request, spectrum, chains, intervals):
            del chains, intervals
            captured["heat_case"] = response_case
            captured["heat_request"] = request
            captured["heat_spectrum"] = spectrum
            return BoundaryTopologyHeatState(
                heat=np.ones((2, 3)),
                phi_values=np.array([0.0, np.pi / 2.0]),
                s_values=np.array([0.0, 0.5, 1.0]),
                metadata={"field_period": np.pi},
            )

    def nonlinear_builder(snapshot, request):
        captured["nonlinear_snapshot"] = snapshot
        captured["nonlinear_request"] = request
        return boundary_response_observables(
            ("field_consistent",),
            (1.0,),
            prefix="sentinel",
        )

    problem = make_boundary_topology_control_problem(
        library,
        BoundaryTopologyObservableSpec(
            resonant_modes=((3, 2),),
            resonant_quantities=("coefficient_real",),
            heat_quantities=("total_power",),
        ),
        {"island.m3.n2.coefficient_real": 0.0},
        extra_observable_builders=(nonlinear_builder,),
        initial_controls=[0.2, -0.1],
        n_values=(2,),
        m_values=(3,),
        heat_model=RecordingHeatModel(),
        plasma_feedback=feedback,
    )

    rows = problem.evaluator()(problem.initial_controls)

    assert "sentinel.field_consistent" in rows.labels
    heat_request = captured["heat_request"]
    snapshot = captured["nonlinear_snapshot"]
    state = snapshot.metadata["boundary_topology_state"]
    np.testing.assert_allclose(state.spectrum.dBr_grid, np.moveaxis(authoritative_tilde, 1, 0))
    assert captured["heat_spectrum"] is state.spectrum
    assert heat_request.total_field is total
    assert heat_request.background_field is background
    assert heat_request.baseline_field is background
    assert heat_request.vacuum_delta_field is vacuum_delta
    assert heat_request.plasma_delta_field is plasma_delta
    assert snapshot.total_field is total
    assert snapshot.background_field is background
    assert snapshot.delta_field is heat_request.delta_field
    assert snapshot.vacuum_delta_field is vacuum_delta
    assert snapshot.plasma_delta_field is plasma_delta
    assert snapshot.field_context is heat_request
    assert snapshot.metadata["vacuum_delta_field"] is vacuum_delta
    assert snapshot.metadata["plasma_delta_field"] is plasma_delta
    assert snapshot.metadata["field_contract"]["authoritative_field"] == "total_field"
    assert captured["nonlinear_request"].control_labels == array.control_labels
    BR, BZ, BPhi = snapshot.delta_field.B_at(1.7, 0.0, 0.0)
    assert float(BR) == pytest.approx(vacuum_delta.components[0] + plasma_delta.components[0])
    assert float(BZ) == pytest.approx(vacuum_delta.components[1] + plasma_delta.components[1])
    assert float(BPhi) == pytest.approx(vacuum_delta.components[2] + plasma_delta.components[2])


def test_case_heat_observables_use_one_field_period_for_periodic_phi_moments():
    _case, array, library = _small_response_library()
    phi = np.arange(8, dtype=float) * np.pi / 8.0
    heat = np.zeros((phi.size, 2), dtype=float)
    heat[[0, -1], 0] = 1.0

    class OnePeriodHeatModel:
        def evaluate(self, case, request, spectrum, chains, intervals):
            del case, request, spectrum, chains, intervals
            return BoundaryTopologyHeatState(
                heat=heat,
                phi_values=phi,
                s_values=np.array([0.25, 0.75]),
                metadata={"field_period": np.pi},
            )

    backend = BoundaryTopologyCaseBackend(
        library=library,
        n_values=(2,),
        m_values=(3,),
        heat_model=OnePeriodHeatModel(),
    )
    request = _request_for_test([0.1, -0.05], array.control_labels)
    snapshot = backend.evaluate(request)
    rows = boundary_topology_case_observable_builder(
        BoundaryTopologyObservableSpec(
            heat_quantities=("centroid_phi", "rms_width_phi"),
        )
    )(snapshot, request)
    values = dict(zip(rows.labels, rows.values))

    assert values["heat.centroid_phi"] == pytest.approx(15.0 * np.pi / 16.0)
    assert values["heat.rms_width_phi"] == pytest.approx(np.pi / 16.0)


def test_case_problem_factory_keeps_actuator_labels_bounds_and_response_hook():
    _case, array, library = _small_response_library()
    observable_spec = BoundaryTopologyObservableSpec(
        resonant_modes=((3, 2),),
        resonant_quantities=("coefficient_real", "coefficient_imag"),
        core_radial_max=0.45,
    )

    def feedback(case, request, vacuum_tilde):
        del case, request
        return {"tilde_b1": 1.5 * vacuum_tilde, "metadata": {"response_model": "test_linear_mhd"}}

    def extra_builder(snapshot, request):
        del snapshot
        return boundary_response_observables(
            ("svd_growth",),
            (float(np.linalg.norm(request.controls)),),
            prefix="dpk",
        )

    problem = make_boundary_topology_control_problem(
        library,
        observable_spec,
        {
            "island.m3.n2.coefficient_real": 0.0,
            "island.m3.n2.coefficient_imag": 0.0,
        },
        extra_observable_builders=(extra_builder,),
        initial_controls=[0.1, -0.05],
        n_values=(2,),
        m_values=(3,),
        plasma_feedback=feedback,
        steps=0.02,
        n_iterations=1,
    )
    rows = problem.evaluator()(problem.initial_controls)
    system = problem.linearize()

    assert problem.control_labels == array.control_labels
    assert problem.control_bounds == array.control_bounds
    assert system.control_labels == array.control_labels
    assert system.response_matrix.shape == (len(rows.labels), len(array.control_labels))
    assert "core.field.radial_leakage_max" in rows.labels
    assert "dpk.svd_growth" in rows.labels


def test_plasma_feedback_can_replace_healed_response_case():
    case, array, library = _small_response_library()
    response_background = _SentinelField("replacement_B0", (0.0, 0.0, 0.9))
    response_equilibrium = object()
    response_case = boundary_topology_case_from_arrays(
        name=case.name,
        R_surf=case.R_surf + 1.0e-3 * case.radial_labels[None, :, None],
        Z_surf=case.Z_surf,
        phi_vals=case.phi_vals,
        theta_vals=case.theta_vals,
        radial_labels=case.radial_labels,
        iota_profile=case.iota_profile + 2.0e-3,
        denominator_B3=case.denominator_B3,
        nfp=case.nfp,
        background_field=response_background,
        equilibrium=response_equilibrium,
        core_reference={"scalars": {"replacement_core": 2.0}},
    )

    def feedback(_case, _request, vacuum_tilde):
        return {
            "tilde_b1": 0.8 * vacuum_tilde,
            "response_case": response_case,
            "metadata": {"response_model": "free_boundary_test"},
        }

    backend = BoundaryTopologyCaseBackend(
        library=library,
        n_values=(2,),
        m_values=(3,),
        plasma_feedback=feedback,
    )
    state, coerced_feedback = backend.forward_state(
        _request_for_test([0.2, -0.1], array.control_labels)
    )

    assert state.response_case is response_case
    np.testing.assert_allclose(state.spectrum.radial_labels, response_case.radial_labels)
    assert state.spectrum.metadata["plasma_response_model"] == "free_boundary_test"
    assert coerced_feedback.background_field is response_background
    assert coerced_feedback.equilibrium is response_equilibrium
    assert coerced_feedback.core is response_case.core_reference


def _write_minimal_vmec_wout(path, *, nfp=2):
    netcdf4 = pytest.importorskip("netCDF4")
    ns = 6
    xm = np.array([0.0, 1.0, 1.0])
    xn = np.array([0.0, 0.0, float(nfp)])
    s = np.linspace(0.0, 1.0, ns)
    with netcdf4.Dataset(path, "w") as dataset:
        dataset.createDimension("radius", ns)
        dataset.createDimension("mn_mode", xm.size)
        dataset.createDimension("mn_mode_nyq", xm.size)
        dataset.createDimension("axis_mode", 2)
        dataset.createVariable("nfp", "i4")[:] = int(nfp)
        dataset.createVariable("ns", "i4")[:] = ns
        dataset.createVariable("lasym__logical__", "i4")[:] = 0
        dataset.createVariable("xm", "f8", ("mn_mode",))[:] = xm
        dataset.createVariable("xn", "f8", ("mn_mode",))[:] = xn
        dataset.createVariable("xm_nyq", "f8", ("mn_mode_nyq",))[:] = xm
        dataset.createVariable("xn_nyq", "f8", ("mn_mode_nyq",))[:] = xn
        rmnc = np.zeros((ns, xm.size))
        zmns = np.zeros_like(rmnc)
        lmns = np.zeros_like(rmnc)
        bsup = np.zeros_like(rmnc)
        rmnc[:, 0] = 2.0
        rmnc[:, 1] = 0.32 * np.sqrt(s)
        rmnc[:, 2] = 0.02 * s
        zmns[:, 1] = 0.27 * np.sqrt(s)
        zmns[:, 2] = 0.015 * s
        lmns[1:, 2] = 0.025 * s[1:]
        bsup[1:, 0] = -0.7
        bsup[1:, 2] = 0.01 * s[1:]
        dataset.createVariable("rmnc", "f8", ("radius", "mn_mode"))[:] = rmnc
        dataset.createVariable("zmns", "f8", ("radius", "mn_mode"))[:] = zmns
        dataset.createVariable("lmns", "f8", ("radius", "mn_mode"))[:] = lmns
        dataset.createVariable("bsupvmnc", "f8", ("radius", "mn_mode_nyq"))[:] = bsup
        dataset.createVariable("iotaf", "f8", ("radius",))[:] = 0.55 + 0.08 * s
        dataset.createVariable("raxis_cc", "f8", ("axis_mode",))[:] = [2.0, 0.01]
        dataset.createVariable("zaxis_cs", "f8", ("axis_mode",))[:] = [0.0, 0.01]


def test_vmec_case_loader_reconstructs_pest_surfaces_and_signed_b3(tmp_path):
    path = tmp_path / "public_case.nc"
    _write_minimal_vmec_wout(path)

    case = vmec_boundary_topology_case_from_wout(
        path,
        name="public stellarator",
        radial_labels=[0.2, 0.6, 1.0],
        n_phi=8,
        n_theta=24,
    )

    assert case.R_surf.shape == (8, 3, 24)
    assert case.nfp == 2
    assert case.coordinate_system == "PEST"
    assert np.all(case.denominator_B3 < 0.0)
    np.testing.assert_allclose(case.q_profile, 1.0 / case.iota_profile)
    assert case.metadata["theta_relation"] == "theta_PEST=theta_VMEC+lambda"
    assert case.metadata["pest_inversion_max_residual"] < 1.0e-10
    assert "source_path" not in case.metadata


def test_vmec_case_loader_builds_native_field_period_directly_and_propagates_nfp(tmp_path):
    path = tmp_path / "nfp5_case.nc"
    nfp = 5
    n_phi_per_period = 7
    _write_minimal_vmec_wout(path, nfp=nfp)

    full = vmec_boundary_topology_case_from_wout(
        path,
        radial_labels=[0.2, 0.6, 1.0],
        n_phi=nfp * n_phi_per_period,
        n_theta=24,
    )
    native = vmec_boundary_topology_case_from_wout(
        path,
        radial_labels=[0.2, 0.6, 1.0],
        n_phi_per_period=n_phi_per_period,
        n_theta=24,
    )

    assert native.R_surf.shape == (n_phi_per_period, 3, 24)
    assert full.R_surf.shape[0] == nfp * native.R_surf.shape[0]
    np.testing.assert_allclose(native.phi_vals, full.phi_vals[:n_phi_per_period], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(native.R_surf, full.R_surf[:n_phi_per_period], atol=2.0e-13, rtol=2.0e-13)
    np.testing.assert_allclose(native.Z_surf, full.Z_surf[:n_phi_per_period], atol=2.0e-13, rtol=2.0e-13)
    np.testing.assert_allclose(
        native.denominator_B3,
        full.denominator_B3[:n_phi_per_period],
        atol=2.0e-13,
        rtol=2.0e-13,
    )
    assert native.phi_vals[-1] < 2.0 * np.pi / nfp
    assert native.metadata["toroidal_domain"] == "native_field_period"
    assert native.metadata["native_field_period_sampling"] is True

    pest = native.to_smooth_pest_coordinates()
    assert pest.nfp == nfp
    assert pest.period == pytest.approx(2.0 * np.pi / nfp)
    assert pest.stores_one_field_period is True
    shape = pest.R_surf.shape
    pest_field = GriddedPestVectorField.from_pest_coordinates(
        pest,
        JR=np.zeros(shape),
        JZ=np.zeros(shape),
        JPhi=np.ones(shape),
    )
    assert pest_field.nfp == nfp
    assert pest_field.field_period_rad == pytest.approx(2.0 * np.pi / nfp)

    R_grid = np.linspace(5.0, 6.0, 3)
    Z_grid = np.linspace(-0.5, 0.5, 3)
    values_shape = (R_grid.size, Z_grid.size, n_phi_per_period)
    vector = VectorFieldCylind(
        R_grid,
        Z_grid,
        Phi=pest.phi_vals,
        BR=np.zeros(values_shape),
        BZ=np.zeros(values_shape),
        BPhi=np.ones(values_shape),
        nfp=pest.nfp,
    )
    assert vector.nfp == nfp
    assert vector.Phi.size == n_phi_per_period
    np.testing.assert_allclose(
        vector(np.array([[5.5, 0.0, 0.03], [5.5, 0.0, 0.03 + 2.0 * np.pi / nfp]])),
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
    )


def test_private_npz_case_loader_keeps_source_path_out_of_metadata(tmp_path):
    source_case = _synthetic_case()
    path = tmp_path / "case_bundle.npz"
    np.savez_compressed(
        path,
        R_surf=source_case.R_surf,
        Z_surf=source_case.Z_surf,
        phi_vals=source_case.phi_vals,
        theta_vals=source_case.theta_vals,
        radial_labels=source_case.radial_labels,
        iota_profile=source_case.iota_profile,
        q_profile=source_case.q_profile,
        denominator_B3=source_case.denominator_B3,
        nfp=np.array(source_case.nfp),
    )

    case = load_boundary_topology_case_npz(path)

    assert case.name == "private stellarator"
    assert case.metadata["source_kind"] == "npz_case_bundle"
    assert len(case.metadata["source_id"]) == 12
    assert "source_path" not in case.metadata
    np.testing.assert_allclose(case.R_surf, source_case.R_surf)


def test_boundary_topology_case_audit_plots_run_headless():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pyna.plot.boundary_topology_case import (
        plot_boundary_response_matrix_audit,
        plot_boundary_topology_control_audit,
    )

    case, array, library = _small_response_library()
    heat_model = ReducedSpectralHeatModel(
        phi_values=np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False),
        s_values=np.linspace(0.0, 1.0, 48),
    )
    backend = BoundaryTopologyCaseBackend(
        library=library,
        n_values=(2,),
        m_values=(3,),
        sigma_threshold=0.0,
        heat_model=heat_model,
    )
    initial_request = _request_for_test([0.05, -0.02], array.control_labels)
    final_request = _request_for_test([0.2, 0.1], array.control_labels)
    initial_state, _ = backend.forward_state(initial_request)
    final_state, _ = backend.forward_state(final_request)
    observable_spec = BoundaryTopologyObservableSpec(
        resonant_modes=((3, 2),),
        core_radial_max=0.45,
    )
    problem = make_boundary_topology_control_problem(
        library,
        observable_spec,
        {
            "island.m3.n2.half_width": 0.01,
            "island.m3.n2.coefficient_real": 0.0,
            "island.m3.n2.coefficient_imag": 0.0,
        },
        initial_controls=[0.05, -0.02],
        n_values=(2,),
        m_values=(3,),
        steps=0.02,
        n_iterations=1,
    )
    system = problem.linearize()

    fig, axes = plot_boundary_topology_control_audit(
        case,
        array,
        initial_state,
        final_state,
        modes=((3, 2),),
    )
    assert axes.shape == (2, 3)
    plt.close(fig)
    fig, axes = plot_boundary_response_matrix_audit(system)
    assert axes.shape == (3,)
    plt.close(fig)


def _request_for_test(controls, labels):
    return BoundaryPlasmaResponseInput(controls=np.asarray(controls, dtype=float), control_labels=labels)
