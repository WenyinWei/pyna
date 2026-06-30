import numpy as np
import pytest

from pyna.dynamics import (
    BrownianMotion,
    CallableFlow,
    CallableMap,
    GeometricBrownianMotion,
    HamiltonianSystem,
    ItoSDE,
    NBodySystem,
    SeparableHamiltonianSystem,
    finite_difference_jacobian,
    fixed_point_eigenspaces,
)
from pyna.topo.core import LinearStabilityData, Orbit, PeriodicOrbit, Trajectory
from pyna.topo.section import HyperplaneSection


def test_finite_difference_jacobian_linear_map():
    matrix = np.array([[2.0, -1.0], [0.5, 3.0]])
    jac = finite_difference_jacobian(lambda x, t: matrix @ x, np.array([0.2, -0.3]))
    np.testing.assert_allclose(jac, matrix, atol=1e-8)


def test_callable_flow_harmonic_oscillator_trajectory():
    flow = CallableFlow(
        lambda x, t: np.array([x[1], -x[0]]),
        dim=2,
        coordinate_names=("q", "p"),
        label="harmonic oscillator",
    )
    sol = flow.trajectory([1.0, 0.0], (0.0, 2.0 * np.pi), dt=0.01)
    assert isinstance(sol, Trajectory)
    np.testing.assert_allclose(sol.final, np.array([1.0, 0.0]), atol=5e-4)
    assert sol.y.shape[1] == 2
    assert flow.phase_space.symplectic is False


def test_hamiltonian_system_vector_field_and_energy():
    system = HamiltonianSystem(
        lambda x, t: 0.5 * (x[0] ** 2 + x[1] ** 2),
        dof=1,
        gradient=lambda x, t: np.array([x[0], x[1]]),
    )
    np.testing.assert_allclose(system.vector_field(np.array([2.0, 3.0])), np.array([3.0, -2.0]))
    assert system.energy([2.0, 3.0]) == pytest.approx(6.5)
    assert system.phase_space.symplectic is True
    traj = system.trajectory([1.0, 0.0], (0.0, np.pi), dt=0.02)
    assert isinstance(traj, Trajectory)
    assert traj.ambient_dim == 2


def test_separable_hamiltonian_velocity_verlet_nearly_conserves_energy():
    system = SeparableHamiltonianSystem(
        kinetic=lambda p, t: 0.5 * np.dot(p, p),
        potential=lambda q, t: 0.5 * np.dot(q, q),
        grad_kinetic=lambda p, t: p,
        grad_potential=lambda q, t: q,
        dof=1,
    )
    x = np.array([1.0, 0.0])
    e0 = system.energy(x)
    for k in range(200):
        x = system.step_velocity_verlet(x, 0.02, t=0.02 * k)
    assert abs(system.energy(x) - e0) < 1e-4


def test_nbody_gravity_and_coulomb_accelerations_have_expected_direction():
    positions = np.array([[-1.0, 0.0], [1.0, 0.0]])
    velocities = np.zeros_like(positions)

    gravity = NBodySystem([1.0, 1.0], spatial_dim=2, interaction="gravity", coupling=1.0)
    state = gravity.pack_state(positions, velocities)
    acc = gravity.accelerations(positions)
    np.testing.assert_allclose(acc, np.array([[0.25, 0.0], [-0.25, 0.0]]))
    assert gravity.total_energy(state) == pytest.approx(-0.5)

    coulomb = NBodySystem(
        [1.0, 1.0],
        spatial_dim=2,
        interaction="electromagnetic",
        coupling=1.0,
        charges=[1.0, 1.0],
    )
    acc_c = coulomb.accelerations(positions)
    np.testing.assert_allclose(acc_c, np.array([[-0.25, 0.0], [0.25, 0.0]]))
    traj = gravity.trajectory(state, (0.0, 0.1), dt=0.01)
    assert isinstance(traj, Trajectory)
    assert traj.ambient_dim == 8


def test_callable_map_jacobian_lyapunov_and_eigenspaces():
    matrix = np.array([[2.0, 0.0], [0.0, 0.5]])
    cmap = CallableMap(lambda x: matrix @ x, dim=2)
    np.testing.assert_allclose(cmap.jacobian([1.0, 1.0]), matrix, atol=1e-8)
    exponents = np.sort(cmap.lyapunov_spectrum([1.0, 1.0], 8))
    np.testing.assert_allclose(exponents, np.sort(np.log([2.0, 0.5])), atol=1e-8)
    spaces = fixed_point_eigenspaces(cmap, [0.0, 0.0])
    assert set(spaces["stable"].tolist()) == {1}
    assert set(spaces["unstable"].tolist()) == {0}
    orbit = cmap.orbit_geometry([1.0, 1.0], 3)
    assert isinstance(orbit, Orbit)
    assert orbit.n_samples == 4
    point = cmap.section_point([0.0, 0.0], section_label="fixed")
    assert point.stability_data is not None
    po = cmap.periodic_orbit([[0.0, 0.0]], section_label="fixed")
    assert isinstance(po, PeriodicOrbit)
    assert po.period == 1


def test_generic_cycle_section_cut_uses_core_periodic_orbit():
    flow = CallableFlow(lambda x, t: np.array([1.0, 0.0]), dim=2)
    traj = flow.trajectory([0.0, 0.0], (0.0, 2.0), dt=0.25)
    from pyna.topo.core import Cycle

    cycle = Cycle(trajectory=traj, period_value=2.0)
    sec = HyperplaneSection(np.array([1.0, 0.0]), 1.0, phase_dim=2)
    po = cycle.section_cut(sec)
    assert isinstance(po, PeriodicOrbit)
    assert po.period == 1
    np.testing.assert_allclose(po.points[0].metadata["ambient_state"], np.array([1.0, 0.0]))


def test_topo_exports_keep_toroidal_defaults_and_core_aliases():
    import pyna.topo as topo
    from pyna.topo import core, toroidal

    assert topo.Tube is toroidal.Tube
    assert topo.Cycle is toroidal.Cycle
    assert topo.CoreTube is core.Tube
    assert topo.CoreCycle is core.Cycle
    assert topo.CoreTrajectory is core.Trajectory
    assert topo.Trajectory is core.Trajectory


def test_generic_invariants_fixed_point_uses_core_stability_data():
    from pyna.topo.invariants import FixedPoint

    stability = LinearStabilityData(jacobian=np.diag([2.0, 0.5, 1.0]))
    fp = FixedPoint(
        state=np.array([1.0, 2.0, 3.0]),
        stability=stability,
        section_value=0.25,
        section_label="sigma",
    )
    point = fp.as_section_point()
    orbit = fp.as_orbit()
    assert point.stability_data is stability
    assert orbit.stability_data is stability
    assert orbit.representative_state.shape == (3,)


def test_ito_sde_euler_maruyama_accepts_deterministic_increments():
    sde = ItoSDE(
        drift=lambda x, t: np.ones(1),
        diffusion=lambda x, t: np.ones((1, 1)),
        dim=1,
        brownian_dim=1,
    )
    sol = sde.euler_maruyama([0.0], (0.0, 1.0), dt=0.1, dW=np.zeros((10, 1)))
    assert isinstance(sol, Trajectory)
    assert sol.final[0] == pytest.approx(1.0)


def test_brownian_motion_moments_and_geometric_brownian_growth():
    brownian = BrownianMotion(dim=2, diffusion=0.3, drift=[0.1, -0.2])
    np.testing.assert_allclose(brownian.mean([1.0, 2.0], 4.0), np.array([1.4, 1.2]))
    np.testing.assert_allclose(brownian.variance(4.0), np.array([0.36, 0.36]))

    gbm = GeometricBrownianMotion(mu=[0.08], sigma=[0.2])
    np.testing.assert_allclose(gbm.expected_log_growth(), np.array([0.06]))
    np.testing.assert_allclose(gbm.mean([100.0], 0.0), np.array([100.0]))
