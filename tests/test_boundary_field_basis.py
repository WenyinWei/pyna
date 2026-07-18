import numpy as np

from pyna.toroidal.control.boundary_field_basis import (
    CylindricalGridFieldControlBasis,
    boundary_field_actuator_array_from_grid_fields,
    cylindrical_vector_field_from_array,
    load_cylindrical_vector_field_npz,
)
from pyna.toroidal.control.boundary_topology_cases import (
    BoundaryTopologyObservableSpec,
    boundary_topology_case_from_arrays,
    build_boundary_perturbation_spectrum_library,
    extend_boundary_topology_case_to_resonance,
    make_boundary_topology_control_problem,
)


def _grid_field(scale=1.0, *, nfp=2):
    R = np.linspace(1.0, 2.0, 9)
    Z = np.linspace(-0.5, 0.5, 8)
    phi = np.linspace(0.0, np.pi, 7, endpoint=False)
    RR, ZZ, PP = np.meshgrid(R, Z, phi, indexing="ij")
    values = np.stack(
        [
            scale * (0.02 * RR * np.cos(2.0 * PP)),
            scale * (0.01 * np.sin(2.0 * PP) + 0.005 * ZZ),
            scale * np.ones_like(RR),
        ],
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


def test_metadata_rich_npz_loader_obeys_explicit_component_order(tmp_path):
    source = _grid_field()
    packed = np.stack([source.BR, source.BZ, source.BPhi], axis=-1)
    path = tmp_path / "field.npz"
    np.savez(
        path,
        R=source.R,
        Z=source.Z,
        Phi=source.Phi,
        field=packed,
        component_order=np.array(["B_R", "B_Z", "B_Phi"]),
        n_fp=np.array(2),
    )

    loaded = load_cylindrical_vector_field_npz(path)

    np.testing.assert_allclose(loaded.BR, source.BR)
    np.testing.assert_allclose(loaded.BZ, source.BZ)
    np.testing.assert_allclose(loaded.BPhi, source.BPhi)
    assert loaded.nfp == 2


def test_grid_control_basis_combines_actual_field_columns_without_losing_nfp():
    background = _grid_field(2.0)
    column_1 = _grid_field(0.1)
    column_2 = _grid_field(-0.04)
    actuators = boundary_field_actuator_array_from_grid_fields(
        (column_1, column_2),
        labels=("control_1", "control_2"),
        bounds=((-2.0, 2.0), (-3.0, 3.0)),
    )
    basis = CylindricalGridFieldControlBasis(background, actuators)

    delta = basis.delta_field([0.5, -0.25])
    total = basis.total_field([0.5, -0.25])

    np.testing.assert_allclose(delta.BR, 0.5 * column_1.BR - 0.25 * column_2.BR)
    np.testing.assert_allclose(total.BPhi, background.BPhi + delta.BPhi)
    assert total.nfp == 2
    assert basis.control_labels == ("control_1", "control_2")
    assert basis.control_bounds["control_2"] == (-3.0, 3.0)


def test_grid_field_columns_use_the_generic_nardon_response_library():
    background = _grid_field(2.0)
    column_1 = _grid_field(0.03)
    column_2 = _grid_field(-0.02)
    actuators = boundary_field_actuator_array_from_grid_fields(
        (column_1, column_2),
        labels=("control_1", "control_2"),
        metadata={"response_kind": "measured_vacuum_control_fields"},
    )
    phi = background.Phi
    theta = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    radial = np.linspace(0.2, 0.9, 4)
    PP, SS, TT = np.meshgrid(phi, radial, theta, indexing="ij")
    R = 1.5 + 0.25 * np.sqrt(SS) * np.cos(TT)
    Z = 0.32 * np.sqrt(SS) * np.sin(TT)
    Bphi = background.interpolate_at(R, Z, PP)[2]
    case = boundary_topology_case_from_arrays(
        name="public test case",
        R_surf=R,
        Z_surf=Z,
        phi_vals=phi,
        theta_vals=theta,
        radial_labels=radial,
        iota_profile=np.linspace(0.62, 0.72, radial.size),
        denominator_B_phi=Bphi,
        nfp=2,
    )

    library = build_boundary_perturbation_spectrum_library(
        case,
        actuators,
        m_max=4,
        n_max=3,
    )

    expected = 0.4 * library.responses[0].tilde_b1 - 0.1 * library.responses[1].tilde_b1
    np.testing.assert_allclose(library.combined_tilde_b1([0.4, -0.1]), expected)
    assert library.combined_spectrum([0.4, -0.1]).metadata["response_kind"] == "measured_vacuum_control_fields"

    problem = make_boundary_topology_control_problem(
        library,
        BoundaryTopologyObservableSpec(
            resonant_modes=((3, 2),),
            resonant_quantities=("coefficient_real", "coefficient_imag"),
        ),
        {
            "island.m3.n2.coefficient_real": 0.0,
            "island.m3.n2.coefficient_imag": 0.0,
        },
        n_values=(2,),
        m_values=(3,),
        target_zero_prefixes=(),
        target_preserve_initial_prefixes=("core.",),
        n_iterations=1,
    )
    assert problem.target_zero_prefixes == ()
    assert problem.target_preserve_initial_prefixes == ("core.",)


def test_healed_edge_extension_makes_nearby_resonance_explicit_and_bounded():
    phi = np.linspace(0.0, np.pi, 4, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    radial = np.linspace(0.5, 1.0, 6)
    PP, SS, TT = np.meshgrid(phi, radial, theta, indexing="ij")
    case = boundary_topology_case_from_arrays(
        name="public test case",
        R_surf=1.6 + 0.25 * np.sqrt(SS) * np.cos(TT),
        Z_surf=0.3 * np.sqrt(SS) * np.sin(TT),
        phi_vals=phi,
        theta_vals=theta,
        radial_labels=radial,
        iota_profile=1.0 / (1.12 - 0.1 * radial),
        denominator_B3=-np.ones_like(PP),
        nfp=2,
    )

    extended = extend_boundary_topology_case_to_resonance(
        case,
        m=1,
        n=1,
        outer_margin=0.01,
        max_extension=0.25,
    )

    assert extended.radial_labels[-1] > 1.0
    assert np.min(extended.q_profile) < 1.0 < np.max(extended.q_profile)
    audit = extended.metadata["edge_resonance_extension"]
    assert audit["model"] == "linear_healed_edge_continuation"
    assert audit["quantitative_geometry"] is False


def test_case_records_negative_toroidal_orientation_q_iota_relation():
    phi = np.linspace(0.0, np.pi, 4, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    radial = np.linspace(0.2, 0.9, 4)
    PP, SS, TT = np.meshgrid(phi, radial, theta, indexing="ij")
    iota = -np.linspace(0.3, 0.4, radial.size)

    case = boundary_topology_case_from_arrays(
        name="private stellarator",
        R_surf=1.5 + 0.2 * np.sqrt(SS) * np.cos(TT),
        Z_surf=0.25 * np.sqrt(SS) * np.sin(TT),
        phi_vals=phi,
        theta_vals=theta,
        radial_labels=radial,
        iota_profile=iota,
        q_iota_sign=-1,
        denominator_B3=-np.ones_like(PP),
        nfp=2,
    )

    np.testing.assert_allclose(case.q_profile, -1.0 / iota)
    assert case.q_iota_sign == -1
    with np.testing.assert_raises_regex(ValueError, "q=q_iota_sign/iota"):
        boundary_topology_case_from_arrays(
            name="private stellarator",
            R_surf=case.R_surf,
            Z_surf=case.Z_surf,
            phi_vals=phi,
            theta_vals=theta,
            radial_labels=radial,
            iota_profile=iota,
            q_profile=1.0 / iota,
            q_iota_sign=-1,
            denominator_B3=-np.ones_like(PP),
            nfp=2,
        )
