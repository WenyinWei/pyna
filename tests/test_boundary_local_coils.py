import numpy as np
import pytest

from pyna.toroidal.coils.boundary_local import (
    BoundaryDipoleActuatorArray,
    BoundaryDipoleCoilSpec,
    BoundaryLoopCoilSpec,
    boundary_dipole_local_actuator_array_from_surface,
    boundary_dipole_mode_actuator_array_from_surface,
    boundary_loop_coil_from_section,
    boundary_loop_coil_specs_from_surface,
    boundary_loop_coil_superposition,
    section_boundary_outward_normal,
    stack_boundary_dipole_actuator_arrays,
)


def _boundary_circle(n_theta=17, *, axis_R=1.0, axis_Z=0.0, radius=0.25):
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    return theta, axis_R + radius * np.cos(theta), axis_Z + radius * np.sin(theta)


def _boundary_surface(n_phi=5, n_theta=11):
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    PP, TT = np.meshgrid(phi, theta, indexing="ij")
    R = 1.0 + (0.22 + 0.02 * np.cos(2.0 * PP)) * np.cos(TT)
    Z = 0.18 * np.sin(TT) + 0.015 * np.sin(PP)
    return R, Z, phi, theta


def test_boundary_loop_coil_from_section_is_tangent_to_boundary_surface():
    theta, R, Z = _boundary_circle()
    phi = 0.4
    spec = boundary_loop_coil_from_section(
        R,
        Z,
        index=0,
        phi=phi,
        radius=0.04,
        current=2.5,
        clearance=0.03,
        axis_R=1.0,
        axis_Z=0.0,
    )

    assert isinstance(spec, BoundaryLoopCoilSpec)
    assert spec.anchor_R == pytest.approx(R[0])
    assert spec.anchor_Z == pytest.approx(Z[0])
    assert spec.radius == pytest.approx(0.04)
    assert spec.current == pytest.approx(2.5)
    assert np.linalg.norm(spec.normal_xyz) == pytest.approx(1.0)

    tangent_rz = np.array([R[1] - R[-1], Z[1] - Z[-1]])
    normal_rz = np.array([
        np.dot(spec.normal_xyz, [np.cos(phi), np.sin(phi), 0.0]),
        spec.normal_xyz[2],
    ])
    assert abs(float(np.dot(tangent_rz, normal_rz))) < 1.0e-12

    field = spec.to_field()
    BR, BZ, BPhi = field.B_at(1.15, 0.02, phi)
    assert np.all(np.isfinite(BR))
    assert np.all(np.isfinite(BZ))
    assert np.all(np.isfinite(BPhi))
    assert field.divergence_free() is True


def test_section_boundary_outward_normal_uses_axis_reference():
    _theta, R, Z = _boundary_circle()
    normal = section_boundary_outward_normal(R, Z, 0, phi=0.0, axis_R=1.0, axis_Z=0.0)

    np.testing.assert_allclose(normal, [1.0, 0.0, 0.0], atol=2.0e-2)


def test_boundary_loop_coil_specs_from_surface_phase_currents_on_non_square_grid():
    R, Z, phi, theta = _boundary_surface(n_phi=5, n_theta=13)
    specs = boundary_loop_coil_specs_from_surface(
        R,
        Z,
        phi,
        theta,
        phi_indices=[0, 3],
        theta_indices=[1, 4, 9],
        radius=0.035,
        current=10.0,
        clearance=0.015,
        mode_m=2,
        mode_n=1,
        phase=0.3,
    )

    assert len(specs) == 6
    for spec in specs:
        phase = 2 * float(spec.anchor_theta) - float(spec.anchor_phi) + 0.3
        assert spec.current == pytest.approx(10.0 * np.cos(phase))
        assert spec.metadata["mode_m"] == 2
        assert spec.metadata["mode_n"] == 1
        assert np.linalg.norm(spec.normal_xyz) == pytest.approx(1.0)

    field = boundary_loop_coil_superposition(specs)
    assert field.divergence_free() is True
    BR, BZ, BPhi = field.B_at(
        np.array([1.12, 0.92]),
        np.array([0.03, -0.04]),
        np.array([0.1, 1.3]),
    )
    assert BR.shape == (2,)
    assert BZ.shape == (2,)
    assert BPhi.shape == (2,)
    assert np.all(np.isfinite(BR + BZ + BPhi))


def test_boundary_loop_coil_specs_require_mode_pair():
    R, Z, phi, theta = _boundary_surface(n_phi=4, n_theta=9)

    with pytest.raises(ValueError, match="mode_m and mode_n"):
        boundary_loop_coil_specs_from_surface(
            R,
            Z,
            phi,
            theta,
            theta_indices=[0],
            radius=0.04,
            mode_m=2,
        )


def test_boundary_dipole_spec_uses_magnetic_moment_and_exact_loop_field():
    spec = BoundaryDipoleCoilSpec(
        center_xyz=(1.25, 0.0, 0.0),
        normal_xyz=(1.0, 0.0, 0.0),
        radius=0.05,
        magnetic_moment=25.0,
    )

    assert spec.current == pytest.approx(25.0 / (np.pi * 0.05**2))
    assert spec.to_loop_spec().current == pytest.approx(spec.current)
    assert spec.to_loop_spec().metadata["actuator_model"] == "finite_circular_dipole"

    point = (1.0, 0.02, 0.0)
    field = spec.to_field().B_at(*point)
    doubled = spec.scaled(2.0).to_field().B_at(*point)
    for value, value_2 in zip(field, doubled):
        np.testing.assert_allclose(value_2, 2.0 * value)


def test_boundary_dipole_mode_array_has_label_safe_cosine_and_sine_columns():
    R, Z, phi, theta = _boundary_surface(n_phi=6, n_theta=12)
    array = boundary_dipole_mode_actuator_array_from_surface(
        R,
        Z,
        phi,
        theta,
        modes=[(3, 2)],
        phi_indices=[0, 2, 4],
        theta_indices=[0, 3, 6, 9],
        radius=0.03,
        unit_moment=100.0,
        clearance=0.02,
    )

    assert isinstance(array, BoundaryDipoleActuatorArray)
    assert array.control_labels == ("dipole.m3.n2.cos", "dipole.m3.n2.sin")
    assert array.control_bounds["dipole.m3.n2.cos"] == (-1.0, 1.0)
    assert array.metadata["nardon_basis"] == "exp(i*(m*theta+n_N*phi))"
    assert array.metadata["nardon_mapping"] == "n_N=-n"
    cosine, sine = array.actuators
    assert cosine.metadata["nardon_n"] == -2
    assert sine.metadata["nardon_n"] == -2
    assert len(cosine.dipoles) == len(sine.dipoles) == 12
    for dipole in cosine.dipoles:
        phase = 3.0 * float(dipole.anchor_theta) - 2.0 * float(dipole.anchor_phi)
        assert dipole.magnetic_moment == pytest.approx(100.0 * np.cos(phase))
        assert dipole.metadata["nardon_n"] == -2
    for dipole in sine.dipoles:
        phase = 3.0 * float(dipole.anchor_theta) - 2.0 * float(dipole.anchor_phi)
        assert dipole.magnetic_moment == pytest.approx(100.0 * np.sin(phase))


def test_local_and_mode_dipole_arrays_stack_without_column_reordering():
    R, Z, phi, theta = _boundary_surface(n_phi=5, n_theta=10)
    modes = boundary_dipole_mode_actuator_array_from_surface(
        R,
        Z,
        phi,
        theta,
        modes=[(2, 1)],
        phi_indices=[0, 2],
        theta_indices=[0, 5],
        radius=0.025,
        unit_moment=20.0,
        include_sine=False,
    )
    trims = boundary_dipole_local_actuator_array_from_surface(
        R,
        Z,
        phi,
        theta,
        sites=[(1, 2), (4, 7)],
        radius=0.02,
        unit_moment=10.0,
    )

    array = stack_boundary_dipole_actuator_arrays([modes, trims])

    assert array.control_labels == modes.control_labels + trims.control_labels
    field = array.field([0.2, -0.1, 0.3])
    BR, BZ, BPhi = field.B_at(np.array([0.9, 1.1]), np.array([0.0, 0.03]), np.array([0.2, 1.0]))
    assert np.all(np.isfinite(BR + BZ + BPhi))
