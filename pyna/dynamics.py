"""General-purpose dynamical systems for pyna.

This module complements the magnetic-confinement focused ``pyna.topo`` stack
with small, reusable building blocks for broad dynamical-systems work:
callable flows, Hamiltonian systems, pairwise N-body systems, arbitrary
finite-dimensional maps, and Ito SDEs.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from pyna.topo.dynamics import ContinuousFlow, DiscreteMap, HamiltonianFlow, PhaseSpace
from pyna.topo.core import (
    LinearStabilityData,
    Orbit,
    PeriodicOrbit,
    SectionPoint,
    Trajectory,
)
from pyna.topo.factories import (
    DynamicalSystemFactory,
    GeometryFactory,
    PoincareMapFactory,
    Registry,
)
from pyna.topo.workflow import TopologyWorkflow, make_poincare_map, orbit_from_map, section_cut


ArrayFunc = Callable[[np.ndarray, float], np.ndarray]
ScalarFunc = Callable[[np.ndarray, float], float]


class TimeSeriesSolution(Trajectory):
    """Sampled continuous-time solution.

    This is a small compatibility name for :class:`pyna.topo.core.Trajectory`.
    It keeps the familiar ``t``, ``y`` and ``final`` properties while living in
    the same geometry hierarchy as ``Cycle`` and ``Tube``.
    """

    def __init__(
        self,
        t: Sequence[float],
        y: Sequence[Sequence[float]],
        *,
        time_name: str = "t",
        coordinate_names: Optional[Sequence[str]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        super().__init__(
            states=np.asarray(y, dtype=float),
            times=np.asarray(t, dtype=float),
            time_name=time_name,
            coordinate_names=tuple(coordinate_names) if coordinate_names is not None else None,
            metadata=dict(metadata or {}),
        )


def _as_state(x: Sequence[float], dim: Optional[int] = None, name: str = "state") -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional state vector.")
    if dim is not None and arr.size != dim:
        raise ValueError(f"{name} has length {arr.size}; expected {dim}.")
    return arr


def _coordinate_names(prefix: str, dim: int) -> Tuple[str, ...]:
    return tuple(f"{prefix}{i}" for i in range(dim))


def _central_gradient(func: ScalarFunc, x: np.ndarray, t: float, eps: float) -> np.ndarray:
    grad = np.empty_like(x, dtype=float)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (float(func(xp, t)) - float(func(xm, t))) / (2.0 * eps)
    return grad


def finite_difference_jacobian(
    func: ArrayFunc,
    x: Sequence[float],
    *,
    t: float = 0.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Central finite-difference Jacobian of ``func(x, t)``.

    Parameters
    ----------
    func : callable
        Vector-valued function with signature ``func(x, t)``.
    x : array_like, shape (dim,)
        Evaluation point.
    t : float, optional
        Time/iteration parameter passed through to ``func``.
    eps : float, optional
        Finite-difference half step.

    Returns
    -------
    ndarray, shape (value_dim, dim)
        Numerical Jacobian.
    """

    x0 = _as_state(x, name="x")
    y0 = np.asarray(func(x0.copy(), float(t)), dtype=float).reshape(-1)
    jac = np.empty((y0.size, x0.size), dtype=float)
    h = float(eps)
    if h <= 0.0:
        raise ValueError("eps must be positive.")
    for i in range(x0.size):
        xp = x0.copy()
        xm = x0.copy()
        xp[i] += h
        xm[i] -= h
        jac[:, i] = (
            np.asarray(func(xp, float(t)), dtype=float).reshape(-1)
            - np.asarray(func(xm, float(t)), dtype=float).reshape(-1)
        ) / (2.0 * h)
    return jac


def _integrate_continuous_flow(
    flow: ContinuousFlow,
    x0: Sequence[float],
    t_span: Tuple[float, float],
    *,
    dt: Optional[float] = None,
    t_eval: Optional[Sequence[float]] = None,
    metadata: Optional[dict] = None,
) -> TimeSeriesSolution:
    from pyna.topo._rk4 import rk4_integrate

    x0_arr = _as_state(x0, flow.dim, name="x0")

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return flow.vector_field(np.asarray(y), float(t))

    sol = rk4_integrate(rhs, t_span, x0_arr, max_step=dt, t_eval=t_eval)
    return TimeSeriesSolution(
        t=np.asarray(sol.t, dtype=float),
        y=np.asarray(sol.y, dtype=float).T,
        coordinate_names=flow.phase_space.coordinate_names or None,
        metadata={
            "system": flow.__class__.__name__,
            "phase_space": flow.phase_space.label,
            **dict(metadata or {}),
        },
    )


class CallableFlow(ContinuousFlow):
    """Continuous flow backed by a Python callable.

    The callable uses pyna's state-first convention: ``rhs(x, t)`` returns
    ``dx/dt`` as a one-dimensional array of length ``dim``.
    """

    def __init__(
        self,
        rhs: ArrayFunc,
        dim: int,
        *,
        coordinate_names: Optional[Sequence[str]] = None,
        label: Optional[str] = None,
    ) -> None:
        self._rhs = rhs
        self._phase_space = PhaseSpace(
            dim=int(dim),
            coordinate_names=tuple(coordinate_names or _coordinate_names("x", int(dim))),
            label=label or "callable flow",
        )

    @property
    def phase_space(self) -> PhaseSpace:
        return self._phase_space

    def vector_field(self, x: np.ndarray, t: float = 0.0) -> np.ndarray:
        x_arr = _as_state(x, self.dim, name="x")
        out = np.asarray(self._rhs(x_arr, float(t)), dtype=float).reshape(-1)
        if out.size != self.dim:
            raise ValueError(f"rhs returned length {out.size}; expected {self.dim}.")
        return out

    def trajectory(
        self,
        x0: Sequence[float],
        t_span: Tuple[float, float],
        *,
        dt: Optional[float] = None,
        t_eval: Optional[Sequence[float]] = None,
    ) -> TimeSeriesSolution:
        """Integrate and return a :class:`pyna.topo.core.Trajectory` subclass."""

        return _integrate_continuous_flow(self, x0, t_span, dt=dt, t_eval=t_eval)

    solve = trajectory


class HamiltonianSystem(HamiltonianFlow):
    """Hamiltonian flow on canonical coordinates ``x = (q, p)``.

    ``hamiltonian(x, t)`` returns ``H(q, p, t)``.  If ``gradient`` is supplied,
    it should return ``dH/dx``; otherwise a central finite difference is used.
    """

    def __init__(
        self,
        hamiltonian: ScalarFunc,
        *,
        dof: Optional[int] = None,
        dim: Optional[int] = None,
        gradient: Optional[ArrayFunc] = None,
        coordinate_names: Optional[Sequence[str]] = None,
        label: Optional[str] = None,
        fd_eps: float = 1e-6,
    ) -> None:
        if dim is None:
            if dof is None:
                raise ValueError("Provide either dof or dim.")
            dim = 2 * int(dof)
        if int(dim) % 2 != 0:
            raise ValueError("Hamiltonian phase-space dimension must be even.")
        self._hamiltonian = hamiltonian
        self._gradient = gradient
        self._fd_eps = float(fd_eps)
        n = int(dim) // 2
        default_names = tuple(f"q{i}" for i in range(n)) + tuple(f"p{i}" for i in range(n))
        self._phase_space = PhaseSpace(
            dim=int(dim),
            coordinate_names=tuple(coordinate_names or default_names),
            symplectic=True,
            label=label or "Hamiltonian system",
        )

    @property
    def phase_space(self) -> PhaseSpace:
        return self._phase_space

    def hamiltonian(self, x: np.ndarray, t: float = 0.0) -> float:
        return float(self._hamiltonian(_as_state(x, self.dim, name="x"), float(t)))

    def gradient(self, x: Sequence[float], t: float = 0.0) -> np.ndarray:
        x_arr = _as_state(x, self.dim, name="x")
        if self._gradient is not None:
            grad = np.asarray(self._gradient(x_arr, float(t)), dtype=float).reshape(-1)
        else:
            grad = _central_gradient(self._hamiltonian, x_arr, float(t), self._fd_eps)
        if grad.size != self.dim:
            raise ValueError(f"gradient returned length {grad.size}; expected {self.dim}.")
        return grad

    def vector_field(self, x: np.ndarray, t: float = 0.0) -> np.ndarray:
        dH = self.gradient(x, t)
        n = self.dim // 2
        out = np.empty(self.dim, dtype=float)
        out[:n] = dH[n:]
        out[n:] = -dH[:n]
        return out

    def energy(self, x: Sequence[float], t: float = 0.0) -> float:
        """Return the Hamiltonian value."""

        return self.hamiltonian(np.asarray(x, dtype=float), t)

    def trajectory(
        self,
        x0: Sequence[float],
        t_span: Tuple[float, float],
        *,
        dt: Optional[float] = None,
        t_eval: Optional[Sequence[float]] = None,
    ) -> TimeSeriesSolution:
        """Integrate and return a generic sampled phase-space trajectory."""

        return _integrate_continuous_flow(self, x0, t_span, dt=dt, t_eval=t_eval)


class SeparableHamiltonianSystem(HamiltonianSystem):
    """Hamiltonian system ``H(q, p) = T(p) + V(q)``.

    The separable form enables the explicit velocity-Verlet step
    ``step_velocity_verlet`` in addition to generic RK4 integration.
    """

    def __init__(
        self,
        kinetic: ScalarFunc,
        potential: ScalarFunc,
        *,
        dof: int,
        grad_kinetic: Optional[ArrayFunc] = None,
        grad_potential: Optional[ArrayFunc] = None,
        coordinate_names: Optional[Sequence[str]] = None,
        label: Optional[str] = None,
        fd_eps: float = 1e-6,
    ) -> None:
        self._kinetic = kinetic
        self._potential = potential
        self._grad_kinetic = grad_kinetic
        self._grad_potential = grad_potential
        self._dof = int(dof)

        def h(x: np.ndarray, t: float) -> float:
            q, p = self.split_state(x)
            return float(self._kinetic(p, t) + self._potential(q, t))

        super().__init__(
            h,
            dof=dof,
            coordinate_names=coordinate_names,
            label=label or "separable Hamiltonian system",
            fd_eps=fd_eps,
        )

    def split_state(self, x: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        x_arr = _as_state(x, 2 * self._dof, name="x")
        return x_arr[: self._dof], x_arr[self._dof :]

    def _grad_t(self, p: np.ndarray, t: float) -> np.ndarray:
        if self._grad_kinetic is not None:
            grad = np.asarray(self._grad_kinetic(p, float(t)), dtype=float).reshape(-1)
        else:
            grad = _central_gradient(self._kinetic, p, float(t), self._fd_eps)
        if grad.size != self._dof:
            raise ValueError(f"grad_kinetic returned length {grad.size}; expected {self._dof}.")
        return grad

    def _grad_v(self, q: np.ndarray, t: float) -> np.ndarray:
        if self._grad_potential is not None:
            grad = np.asarray(self._grad_potential(q, float(t)), dtype=float).reshape(-1)
        else:
            grad = _central_gradient(self._potential, q, float(t), self._fd_eps)
        if grad.size != self._dof:
            raise ValueError(f"grad_potential returned length {grad.size}; expected {self._dof}.")
        return grad

    def vector_field(self, x: np.ndarray, t: float = 0.0) -> np.ndarray:
        q, p = self.split_state(x)
        return np.concatenate([self._grad_t(p, t), -self._grad_v(q, t)])

    def step_velocity_verlet(self, x: Sequence[float], dt: float, t: float = 0.0) -> np.ndarray:
        """One second-order symplectic velocity-Verlet step."""

        q, p = self.split_state(x)
        h = float(dt)
        p_half = p - 0.5 * h * self._grad_v(q, t)
        q_new = q + h * self._grad_t(p_half, t + 0.5 * h)
        p_new = p_half - 0.5 * h * self._grad_v(q_new, t + h)
        return np.concatenate([q_new, p_new])


class NBodySystem(ContinuousFlow):
    """Pairwise N-body system for gravity or electrostatic interactions.

    State vectors are flattened as ``[positions.ravel(), velocities.ravel()]``.
    Use :meth:`pack_state` and :meth:`unpack_state` to convert between flattened
    arrays and ``(n_bodies, spatial_dim)`` position/velocity arrays.
    """

    def __init__(
        self,
        masses: Sequence[float],
        *,
        spatial_dim: int = 3,
        interaction: str = "gravity",
        coupling: float = 1.0,
        charges: Optional[Sequence[float]] = None,
        softening: float = 0.0,
        label: Optional[str] = None,
    ) -> None:
        masses_arr = np.asarray(masses, dtype=float).reshape(-1)
        if masses_arr.size < 2:
            raise ValueError("NBodySystem requires at least two bodies.")
        if np.any(masses_arr <= 0.0):
            raise ValueError("All masses must be positive.")
        mode = interaction.lower()
        if mode == "electromagnetic":
            mode = "electrostatic"
        if mode not in {"gravity", "coulomb", "electrostatic"}:
            raise ValueError("interaction must be 'gravity', 'coulomb', or 'electrostatic'.")
        if charges is None:
            charges_arr = np.ones_like(masses_arr)
        else:
            charges_arr = np.asarray(charges, dtype=float).reshape(-1)
            if charges_arr.shape != masses_arr.shape:
                raise ValueError("charges must have the same length as masses.")
        self.masses = masses_arr
        self.charges = charges_arr
        self.spatial_dim = int(spatial_dim)
        self.interaction = mode
        self.coupling = float(coupling)
        self.softening = float(softening)
        dim = 2 * self.masses.size * self.spatial_dim
        axes = [f"x{k}" for k in range(self.spatial_dim)]
        names = tuple(
            f"q{i}_{axis}" for i in range(self.masses.size) for axis in axes
        ) + tuple(f"v{i}_{axis}" for i in range(self.masses.size) for axis in axes)
        self._phase_space = PhaseSpace(dim=dim, coordinate_names=names, label=label or f"{mode} N-body")

    @property
    def phase_space(self) -> PhaseSpace:
        return self._phase_space

    @property
    def n_bodies(self) -> int:
        return int(self.masses.size)

    def pack_state(self, positions: Sequence[Sequence[float]], velocities: Sequence[Sequence[float]]) -> np.ndarray:
        pos = np.asarray(positions, dtype=float)
        vel = np.asarray(velocities, dtype=float)
        expected = (self.n_bodies, self.spatial_dim)
        if pos.shape != expected or vel.shape != expected:
            raise ValueError(f"positions and velocities must both have shape {expected}.")
        return np.concatenate([pos.reshape(-1), vel.reshape(-1)])

    def unpack_state(self, x: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        x_arr = _as_state(x, self.dim, name="x")
        half = self.n_bodies * self.spatial_dim
        pos = x_arr[:half].reshape(self.n_bodies, self.spatial_dim)
        vel = x_arr[half:].reshape(self.n_bodies, self.spatial_dim)
        return pos, vel

    def accelerations(self, positions: Sequence[Sequence[float]]) -> np.ndarray:
        pos = np.asarray(positions, dtype=float)
        expected = (self.n_bodies, self.spatial_dim)
        if pos.shape != expected:
            raise ValueError(f"positions must have shape {expected}.")
        acc = np.zeros_like(pos, dtype=float)
        eps2 = self.softening * self.softening
        for i in range(self.n_bodies - 1):
            for j in range(i + 1, self.n_bodies):
                rij = pos[j] - pos[i]
                r2 = float(np.dot(rij, rij) + eps2)
                if r2 <= 0.0:
                    raise ValueError("Two bodies occupy the same position; use softening > 0.")
                inv_r3 = r2 ** -1.5
                if self.interaction == "gravity":
                    acc[i] += self.coupling * self.masses[j] * rij * inv_r3
                    acc[j] -= self.coupling * self.masses[i] * rij * inv_r3
                else:
                    force_i = -self.coupling * self.charges[i] * self.charges[j] * rij * inv_r3
                    acc[i] += force_i / self.masses[i]
                    acc[j] -= force_i / self.masses[j]
        return acc

    def vector_field(self, x: np.ndarray, t: float = 0.0) -> np.ndarray:
        positions, velocities = self.unpack_state(x)
        return np.concatenate([velocities.reshape(-1), self.accelerations(positions).reshape(-1)])

    def trajectory(
        self,
        x0: Sequence[float],
        t_span: Tuple[float, float],
        *,
        dt: Optional[float] = None,
        t_eval: Optional[Sequence[float]] = None,
    ) -> TimeSeriesSolution:
        """Integrate and return a generic sampled N-body trajectory."""

        return _integrate_continuous_flow(self, x0, t_span, dt=dt, t_eval=t_eval)

    def total_energy(self, x: Sequence[float]) -> float:
        positions, velocities = self.unpack_state(x)
        kinetic = 0.5 * float(np.sum(self.masses[:, None] * velocities * velocities))
        potential = 0.0
        eps2 = self.softening * self.softening
        for i in range(self.n_bodies - 1):
            for j in range(i + 1, self.n_bodies):
                rij = positions[j] - positions[i]
                r = float(np.sqrt(np.dot(rij, rij) + eps2))
                if self.interaction == "gravity":
                    potential -= self.coupling * self.masses[i] * self.masses[j] / r
                else:
                    potential += self.coupling * self.charges[i] * self.charges[j] / r
        return kinetic + potential


class CallableMap(DiscreteMap):
    """Finite-dimensional discrete map backed by ``step_func(x)``."""

    def __init__(
        self,
        step_func: Callable[[np.ndarray], np.ndarray],
        dim: int,
        *,
        coordinate_names: Optional[Sequence[str]] = None,
        label: Optional[str] = None,
    ) -> None:
        self._step_func = step_func
        self._phase_space = PhaseSpace(
            dim=int(dim),
            coordinate_names=tuple(coordinate_names or _coordinate_names("x", int(dim))),
            label=label or "callable map",
        )

    @property
    def phase_space(self) -> PhaseSpace:
        return self._phase_space

    def step(self, x: np.ndarray) -> np.ndarray:
        x_arr = _as_state(x, self.dim, name="x")
        out = np.asarray(self._step_func(x_arr), dtype=float).reshape(-1)
        if out.size != self.dim:
            raise ValueError(f"step_func returned length {out.size}; expected {self.dim}.")
        return out

    def jacobian(self, x: Sequence[float], *, eps: float = 1e-6) -> np.ndarray:
        return finite_difference_jacobian(lambda y, _t: self.step(y), x, eps=eps)

    def orbit_geometry(self, x0: Sequence[float], n_iter: int) -> Orbit:
        """Return a finite sampled :class:`pyna.topo.core.Orbit`."""

        states = super().orbit(np.asarray(x0, dtype=float), n_iter)
        return Orbit(
            states=states,
            steps=np.arange(states.shape[0]),
            coordinate_names=self.phase_space.coordinate_names or None,
            metadata={"map": self.__class__.__name__, "phase_space": self.phase_space.label},
        )

    def fixed_point_residual(self, x: Sequence[float]) -> np.ndarray:
        x_arr = _as_state(x, self.dim, name="x")
        return self.step(x_arr) - x_arr

    def section_point(
        self,
        x: Sequence[float],
        *,
        section_value: Optional[float] = None,
        section_label: Optional[str] = None,
        eps: float = 1e-6,
    ) -> SectionPoint:
        """Represent a map point as a generic section point with stability data."""

        x_arr = _as_state(x, self.dim, name="x")
        jac = self.jacobian(x_arr, eps=eps)
        return SectionPoint(
            state=x_arr.copy(),
            section_value=section_value,
            section_label=section_label,
            stability_data=LinearStabilityData(jacobian=jac),
            metadata={"coordinate_names": self.phase_space.coordinate_names},
        )

    def periodic_orbit(
        self,
        points: Sequence[Sequence[float]],
        *,
        section_value: Optional[float] = None,
        section_label: Optional[str] = None,
        eps: float = 1e-6,
    ) -> PeriodicOrbit:
        """Build a generic :class:`PeriodicOrbit` from map points."""

        section_points = [
            self.section_point(
                pt,
                section_value=section_value,
                section_label=section_label,
                eps=eps,
            )
            for pt in points
        ]
        representative = section_points[0].state.copy() if section_points else None
        stability = section_points[0].stability_data if section_points else None
        return PeriodicOrbit(
            points=section_points,
            period=len(section_points),
            stability_data=stability,
            representative_state=representative,
            orbit_trace=Orbit(
                states=np.asarray(points, dtype=float),
                steps=np.arange(len(points)),
                coordinate_names=self.phase_space.coordinate_names or None,
            ) if section_points else None,
            metadata={"map": self.__class__.__name__},
        )

    def lyapunov_spectrum(
        self,
        x0: Sequence[float],
        n_iter: int,
        *,
        discard: int = 0,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Estimate Lyapunov exponents with a QR tangent iteration."""

        if n_iter <= 0:
            raise ValueError("n_iter must be positive.")
        if discard < 0 or discard >= n_iter:
            raise ValueError("discard must satisfy 0 <= discard < n_iter.")
        x = _as_state(x0, self.dim, name="x0").copy()
        q = np.eye(self.dim)
        logs = np.zeros(self.dim, dtype=float)
        count = 0
        for k in range(n_iter):
            jac = self.jacobian(x, eps=eps)
            q, r = np.linalg.qr(jac @ q)
            diag = np.abs(np.diag(r))
            x = self.step(x)
            if k >= discard:
                logs += np.log(diag + 1e-300)
                count += 1
        return logs / max(count, 1)


def fixed_point_eigenspaces(
    map_obj: CallableMap,
    point: Sequence[float],
    *,
    eps: float = 1e-6,
    unit_tol: float = 1e-8,
) -> dict:
    """Classify tangent eigenspaces of a map fixed point."""

    jac = map_obj.jacobian(point, eps=eps)
    eigenvalues, eigenvectors = np.linalg.eig(jac)
    moduli = np.abs(eigenvalues)
    stable = np.where(moduli < 1.0 - unit_tol)[0]
    unstable = np.where(moduli > 1.0 + unit_tol)[0]
    center = np.where(np.abs(moduli - 1.0) <= unit_tol)[0]
    return {
        "jacobian": jac,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "stable": stable,
        "unstable": unstable,
        "center": center,
    }


class ItoSDE:
    """Ito stochastic differential equation.

    The equation is ``dX = drift(X, t) dt + diffusion(X, t) dW``.  Diffusion
    may return a scalar, a vector, or a ``(dim, brownian_dim)`` matrix.
    """

    def __init__(
        self,
        drift: ArrayFunc,
        diffusion: Callable[[np.ndarray, float], np.ndarray],
        dim: int,
        *,
        brownian_dim: Optional[int] = None,
        coordinate_names: Optional[Sequence[str]] = None,
        label: Optional[str] = None,
    ) -> None:
        self._drift = drift
        self._diffusion = diffusion
        self.dim = int(dim)
        self.brownian_dim = int(brownian_dim if brownian_dim is not None else dim)
        self.phase_space = PhaseSpace(
            dim=self.dim,
            coordinate_names=tuple(coordinate_names or _coordinate_names("x", self.dim)),
            label=label or "Ito SDE",
        )

    def drift(self, x: Sequence[float], t: float = 0.0) -> np.ndarray:
        x_arr = _as_state(x, self.dim, name="x")
        out = np.asarray(self._drift(x_arr, float(t)), dtype=float).reshape(-1)
        if out.size != self.dim:
            raise ValueError(f"drift returned length {out.size}; expected {self.dim}.")
        return out

    def diffusion_matrix(self, x: Sequence[float], t: float = 0.0) -> np.ndarray:
        x_arr = _as_state(x, self.dim, name="x")
        value = np.asarray(self._diffusion(x_arr, float(t)), dtype=float)
        if value.ndim == 0:
            return float(value) * np.eye(self.dim, self.brownian_dim)
        if value.ndim == 1:
            if value.size != self.dim:
                raise ValueError(f"diffusion vector has length {value.size}; expected {self.dim}.")
            if self.brownian_dim == 1:
                return value.reshape(self.dim, 1)
            if self.brownian_dim == self.dim:
                return np.diag(value)
        if value.shape != (self.dim, self.brownian_dim):
            raise ValueError(
                "diffusion must be scalar, length-dim vector, or "
                f"shape {(self.dim, self.brownian_dim)}; got {value.shape}."
            )
        return value

    def euler_maruyama(
        self,
        x0: Sequence[float],
        t_span: Tuple[float, float],
        *,
        dt: float,
        rng: Optional[object] = None,
        dW: Optional[np.ndarray] = None,
    ) -> TimeSeriesSolution:
        """Integrate one sample path with Euler-Maruyama."""

        t0, tf = float(t_span[0]), float(t_span[1])
        if tf < t0:
            raise ValueError("ItoSDE.euler_maruyama only supports forward time.")
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        x = _as_state(x0, self.dim, name="x0").copy()
        n_steps = max(int(np.ceil((tf - t0) / float(dt))), 1)
        h = (tf - t0) / n_steps
        times = t0 + h * np.arange(n_steps + 1)
        states = np.empty((n_steps + 1, self.dim), dtype=float)
        states[0] = x
        if dW is None:
            if isinstance(rng, np.random.Generator):
                generator = rng
            else:
                generator = np.random.default_rng(rng)
            increments = np.sqrt(h) * generator.normal(size=(n_steps, self.brownian_dim))
        else:
            increments = np.asarray(dW, dtype=float)
            if increments.shape != (n_steps, self.brownian_dim):
                raise ValueError(f"dW must have shape {(n_steps, self.brownian_dim)}.")
        for k in range(n_steps):
            t = times[k]
            g = self.diffusion_matrix(x, t)
            x = x + self.drift(x, t) * h + g @ increments[k]
            states[k + 1] = x
        return TimeSeriesSolution(t=times, y=states)

    def sample_paths(
        self,
        x0: Sequence[float],
        t_span: Tuple[float, float],
        *,
        dt: float,
        n_paths: int,
        rng: Optional[object] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(t, paths)`` with paths shape ``(n_paths, n_times, dim)``."""

        if n_paths <= 0:
            raise ValueError("n_paths must be positive.")
        generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        paths = []
        t = None
        for _ in range(int(n_paths)):
            sol = self.euler_maruyama(x0, t_span, dt=dt, rng=generator)
            t = sol.t
            paths.append(sol.y)
        return np.asarray(t, dtype=float), np.stack(paths, axis=0)


class BrownianMotion(ItoSDE):
    """Brownian motion with optional constant drift."""

    def __init__(
        self,
        *,
        dim: int = 1,
        diffusion: float = 1.0,
        drift: Optional[Sequence[float]] = None,
    ) -> None:
        drift_vec = np.zeros(int(dim), dtype=float) if drift is None else _as_state(drift, int(dim), name="drift")
        sigma = float(diffusion)
        super().__init__(
            lambda x, t: drift_vec,
            lambda x, t: sigma,
            int(dim),
            brownian_dim=int(dim),
            label="Brownian motion",
        )
        self.diffusion = sigma
        self.drift_vector = drift_vec

    def mean(self, x0: Sequence[float], t: float) -> np.ndarray:
        return _as_state(x0, self.dim, name="x0") + self.drift_vector * float(t)

    def variance(self, t: float) -> np.ndarray:
        return np.full(self.dim, self.diffusion * self.diffusion * float(t), dtype=float)


class GeometricBrownianMotion(ItoSDE):
    """Independent geometric Brownian motions ``dS = mu*S dt + sigma*S dW``."""

    def __init__(self, mu: Sequence[float], sigma: Sequence[float]) -> None:
        mu_arr = np.asarray(mu, dtype=float).reshape(-1)
        sigma_arr = np.asarray(sigma, dtype=float).reshape(-1)
        if mu_arr.shape != sigma_arr.shape:
            raise ValueError("mu and sigma must have the same shape.")
        self.mu = mu_arr
        self.sigma = sigma_arr
        super().__init__(
            lambda x, t: self.mu * x,
            lambda x, t: self.sigma * x,
            dim=mu_arr.size,
            brownian_dim=mu_arr.size,
            coordinate_names=tuple(f"S{i}" for i in range(mu_arr.size)),
            label="geometric Brownian motion",
        )

    def expected_log_growth(self) -> np.ndarray:
        """Return the long-time drift of ``log(S_t)``."""

        return self.mu - 0.5 * self.sigma * self.sigma

    def mean(self, x0: Sequence[float], t: float) -> np.ndarray:
        return _as_state(x0, self.dim, name="x0") * np.exp(self.mu * float(t))


__all__ = [
    "TimeSeriesSolution",
    "CallableFlow",
    "HamiltonianSystem",
    "SeparableHamiltonianSystem",
    "NBodySystem",
    "CallableMap",
    "ItoSDE",
    "BrownianMotion",
    "GeometricBrownianMotion",
    "finite_difference_jacobian",
    "fixed_point_eigenspaces",
    "DynamicalSystemFactory",
    "GeometryFactory",
    "PoincareMapFactory",
    "Registry",
    "TopologyWorkflow",
    "make_poincare_map",
    "orbit_from_map",
    "section_cut",
]
