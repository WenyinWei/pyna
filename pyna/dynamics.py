"""General-purpose dynamical systems for pyna.

This module complements the magnetic-confinement focused ``pyna.topo`` stack
with small, reusable building blocks for broad dynamical-systems work:
callable flows, Hamiltonian systems, pairwise N-body systems, arbitrary
finite-dimensional maps, and Ito SDEs.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple

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


def cartesian_points_to_cylindrical_coords(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Convert Cartesian ``(x, y, z)`` points to ``VectorFieldCylind`` coordinates.

    The returned final dimension is ``(R, Z, phi)`` because that is the
    coordinate order used by :class:`pyna.fields.VectorFieldCylind`.
    """

    pts = np.asarray(points, dtype=float)
    if pts.shape[-1:] != (3,):
        raise ValueError("Cartesian points must have final dimension 3.")
    x = pts[..., 0]
    y = pts[..., 1]
    z = pts[..., 2]
    return np.stack([np.sqrt(x * x + y * y), z, np.arctan2(y, x)], axis=-1)


def cylindrical_vectors_to_cartesian(
    values: Sequence[Sequence[float]] | np.ndarray,
    phi: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Convert cylindrical vector components ``(V_R, V_Z, V_phi)`` to Cartesian.

    ``VectorFieldCylind`` stores physical components in ``(R, Z, phi)`` order,
    so the second component is copied to Cartesian ``z``.
    """

    vec = np.asarray(values, dtype=float)
    if vec.shape[-1:] != (3,):
        raise ValueError("Cylindrical vectors must have final dimension 3.")
    phi_arr = np.asarray(phi, dtype=float)
    vR = vec[..., 0]
    vZ = vec[..., 1]
    vPhi = vec[..., 2]
    cp = np.cos(phi_arr)
    sp = np.sin(phi_arr)
    return np.stack([vR * cp - vPhi * sp, vR * sp + vPhi * cp, vZ], axis=-1)


def vector_field_cylind_cartesian_rhs(
    field: Any,
    *,
    normalize: bool = False,
    min_speed: float = 0.0,
    min_radius: float = 1.0e-12,
) -> ArrayFunc:
    """Return a Cartesian RHS wrapper for a :class:`VectorFieldCylind`.

    The returned callable has the pyna convention ``rhs(x, t)``.  It accepts
    either a single Cartesian state ``(3,)`` or a batched array ``(..., 3)`` and
    returns Cartesian velocity components with the same leading shape.
    """

    from pyna.fields import VectorFieldCylind

    if not isinstance(field, VectorFieldCylind):
        raise TypeError("field must be a VectorFieldCylind.")
    speed_floor = float(min_speed)
    radius_floor = float(min_radius)

    def rhs(x: np.ndarray, t: float = 0.0) -> np.ndarray:
        del t
        pts = np.asarray(x, dtype=float)
        if pts.shape[-1:] != (3,):
            raise ValueError("VectorFieldCylind Cartesian RHS requires states with final dimension 3.")
        cyl = cartesian_points_to_cylindrical_coords(pts)
        values = np.asarray(field(cyl), dtype=float)
        if values.shape != pts.shape:
            raise ValueError(f"field returned shape {values.shape}; expected {pts.shape}.")
        vel = cylindrical_vectors_to_cartesian(values, cyl[..., 2])
        speed = np.linalg.norm(vel, axis=-1)
        safe = (
            np.isfinite(pts).all(axis=-1)
            & np.isfinite(vel).all(axis=-1)
            & np.isfinite(speed)
            & (speed > speed_floor)
            & (cyl[..., 0] > radius_floor)
        )
        out = np.full_like(vel, np.nan, dtype=float)
        if normalize:
            out[safe] = vel[safe] / speed[safe][..., None]
        else:
            out[safe] = vel[safe]
        return out

    return rhs


def vector_field_cartesian_rhs(
    field: Any,
    *,
    normalize: bool = False,
    min_speed: float = 0.0,
) -> ArrayFunc:
    """Return a checked Cartesian RHS wrapper for ``VectorFieldCartesian``."""

    from pyna.fields import VectorFieldCartesian

    if not isinstance(field, VectorFieldCartesian):
        raise TypeError("field must be a VectorFieldCartesian.")
    speed_floor = float(min_speed)

    def rhs(x: np.ndarray, t: float = 0.0) -> np.ndarray:
        del t
        pts = np.asarray(x, dtype=float)
        if pts.shape[-1:] != (3,):
            raise ValueError("VectorFieldCartesian RHS requires states with final dimension 3.")
        vel = np.asarray(field(pts), dtype=float)
        if vel.shape != pts.shape:
            raise ValueError(f"field returned shape {vel.shape}; expected {pts.shape}.")
        speed = np.linalg.norm(vel, axis=-1)
        safe = (
            np.isfinite(pts).all(axis=-1)
            & np.isfinite(vel).all(axis=-1)
            & np.isfinite(speed)
            & (speed > speed_floor)
        )
        out = np.full_like(vel, np.nan, dtype=float)
        if normalize:
            out[safe] = vel[safe] / speed[safe][..., None]
        else:
            out[safe] = vel[safe]
        return out

    return rhs


def _call_rhs(system: Any, x: np.ndarray, t: float) -> np.ndarray:
    if hasattr(system, "vector_field"):
        return np.asarray(system.vector_field(x, float(t)), dtype=float)
    try:
        return np.asarray(system(x, float(t)), dtype=float)
    except TypeError:
        return np.asarray(system(x), dtype=float)


def _rhs_for_cartesian_trace(
    system: Any,
    *,
    dim: int,
    normalize: bool,
    min_speed: float,
) -> ArrayFunc:
    from pyna.fields import VectorFieldCartesian, VectorFieldCylind

    if isinstance(system, VectorFieldCylind):
        if dim != 3:
            raise ValueError("VectorFieldCylind Cartesian tracing requires a 3-D Cartesian state.")
        return vector_field_cylind_cartesian_rhs(system, normalize=normalize, min_speed=min_speed)
    if isinstance(system, VectorFieldCartesian):
        if dim != 3:
            raise ValueError("VectorFieldCartesian tracing requires a 3-D Cartesian state.")
        return vector_field_cartesian_rhs(system, normalize=normalize, min_speed=min_speed)

    speed_floor = float(min_speed)

    def rhs(x: np.ndarray, t: float = 0.0) -> np.ndarray:
        x_arr = _as_state(x, dim, name="x")
        raw = np.asarray(_call_rhs(system, x_arr, float(t)), dtype=float).reshape(-1)
        if raw.size != dim:
            raise ValueError(f"RHS returned length {raw.size}; expected {dim}.")
        speed = float(np.linalg.norm(raw))
        if not (np.all(np.isfinite(raw)) and np.isfinite(speed) and speed > speed_floor):
            return np.full(dim, np.nan, dtype=float)
        if normalize:
            return raw / speed
        return raw

    return rhs


def _rk4_step_checked(rhs: ArrayFunc, x: np.ndarray, t: float, h: float) -> tuple[np.ndarray, bool]:
    k1 = np.asarray(rhs(x, t), dtype=float).reshape(-1)
    k2 = np.asarray(rhs(x + 0.5 * h * k1, t + 0.5 * h), dtype=float).reshape(-1)
    k3 = np.asarray(rhs(x + 0.5 * h * k2, t + 0.5 * h), dtype=float).reshape(-1)
    k4 = np.asarray(rhs(x + h * k3, t + h), dtype=float).reshape(-1)
    if not (
        k1.size == x.size
        and k2.size == x.size
        and k3.size == x.size
        and k4.size == x.size
        and np.all(np.isfinite(k1))
        and np.all(np.isfinite(k2))
        and np.all(np.isfinite(k3))
        and np.all(np.isfinite(k4))
    ):
        return np.full_like(x, np.nan, dtype=float), False
    next_x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return next_x, bool(np.all(np.isfinite(next_x)))


def trace_cartesian_trajectory(
    system: Any,
    x0: Sequence[float],
    *,
    s_span: Tuple[float, float] = (0.0, 1.0),
    step_size: Optional[float] = None,
    n_steps: Optional[int] = None,
    normalize: bool = False,
    min_speed: float = 0.0,
    stop_on_invalid: bool = True,
    coordinate_names: Optional[Sequence[str]] = None,
    parameter_name: str = "s",
    metadata: Optional[dict] = None,
) -> TimeSeriesSolution:
    """Trace a finite-dimensional Cartesian trajectory with fixed-step RK4.

    ``system`` may be a pyna continuous-flow object, a callable using either
    ``rhs(x, s)`` or ``rhs(x)``, or a :class:`pyna.fields.VectorFieldCylind`.
    For ``VectorFieldCylind`` inputs the state is Cartesian ``(x, y, z)`` and
    the cylindrical vector components are converted internally.  With
    ``normalize=True`` the independent variable is physical arclength along the
    vector field, which is the usual choice for streamline geometry.
    """

    x = _as_state(x0, name="x0").copy()
    dim = int(x.size)
    t0 = float(s_span[0])
    tf = float(s_span[1])
    span = abs(tf - t0)
    if n_steps is not None:
        steps = max(int(n_steps), 1)
    elif step_size is not None and float(step_size) > 0.0 and np.isfinite(float(step_size)):
        steps = max(int(np.ceil(span / float(step_size))), 1)
    else:
        steps = 200
    direction = 1.0 if tf >= t0 else -1.0
    h = direction * span / float(steps) if steps else 0.0
    rhs = _rhs_for_cartesian_trace(system, dim=dim, normalize=bool(normalize), min_speed=float(min_speed))

    times = [t0]
    states = [x.copy()]
    terminated_reason = "completed"
    t = t0
    for _ in range(steps):
        next_x, ok = _rk4_step_checked(rhs, x, t, h)
        next_t = t + h
        if not ok:
            terminated_reason = "invalid_rhs_or_state"
            if not stop_on_invalid:
                times.append(next_t)
                states.append(next_x)
            break
        times.append(next_t)
        states.append(next_x.copy())
        x = next_x
        t = next_t

    state_arr = np.asarray(states, dtype=float)
    finite_fraction = float(np.count_nonzero(np.isfinite(state_arr)) / max(state_arr.size, 1))
    names = tuple(coordinate_names or (("x", "y", "z") if dim == 3 else _coordinate_names("x", dim)))
    meta = {
        "trace_backend": "pyna.dynamics.fixed_step_rk4_cartesian",
        "system_type": type(system).__name__,
        "dimension": dim,
        "normalize": bool(normalize),
        "min_speed": float(min_speed),
        "requested_n_steps": int(steps),
        "completed_n_steps": max(len(times) - 1, 0),
        "step_size": float(abs(h)),
        "terminated_reason": terminated_reason,
        "finite_fraction": finite_fraction,
        **dict(metadata or {}),
    }
    return TimeSeriesSolution(
        t=np.asarray(times, dtype=float),
        y=state_arr,
        time_name=str(parameter_name),
        coordinate_names=names,
        metadata=meta,
    )


def trace_cartesian_streamlines(
    system: Any,
    seeds: Sequence[Sequence[float]] | np.ndarray,
    *,
    length: float,
    step_size: Optional[float] = None,
    n_steps: Optional[int] = None,
    bidirectional: bool = True,
    normalize: bool = True,
    min_speed: float = 0.0,
    stop_on_invalid: bool = True,
    coordinate_names: Optional[Sequence[str]] = None,
    metadata: Optional[dict] = None,
) -> list[TimeSeriesSolution]:
    """Trace multiple Cartesian streamlines from seed points.

    Returns one :class:`TimeSeriesSolution` per seed.  When ``bidirectional`` is
    true, each solution is ordered from negative to positive arclength with the
    seed at ``s=0``.
    """

    seed_arr = np.asarray(seeds, dtype=float)
    if seed_arr.ndim != 2:
        raise ValueError("seeds must have shape (n_seed, dim).")
    half_length = abs(float(length))
    curves: list[TimeSeriesSolution] = []
    for seed_index, seed in enumerate(seed_arr):
        per_seed_metadata = {"seed_index": int(seed_index), **dict(metadata or {})}
        if not bidirectional:
            curves.append(
                trace_cartesian_trajectory(
                    system,
                    seed,
                    s_span=(0.0, half_length),
                    step_size=step_size,
                    n_steps=n_steps,
                    normalize=normalize,
                    min_speed=min_speed,
                    stop_on_invalid=stop_on_invalid,
                    coordinate_names=coordinate_names,
                    parameter_name="s",
                    metadata=per_seed_metadata,
                )
            )
            continue
        backward = trace_cartesian_trajectory(
            system,
            seed,
            s_span=(0.0, -half_length),
            step_size=step_size,
            n_steps=n_steps,
            normalize=normalize,
            min_speed=min_speed,
            stop_on_invalid=stop_on_invalid,
            coordinate_names=coordinate_names,
            parameter_name="s",
            metadata=per_seed_metadata,
        )
        forward = trace_cartesian_trajectory(
            system,
            seed,
            s_span=(0.0, half_length),
            step_size=step_size,
            n_steps=n_steps,
            normalize=normalize,
            min_speed=min_speed,
            stop_on_invalid=stop_on_invalid,
            coordinate_names=coordinate_names,
            parameter_name="s",
            metadata=per_seed_metadata,
        )
        times = np.concatenate([backward.t[:0:-1], forward.t])
        states = np.concatenate([backward.y[:0:-1], forward.y], axis=0)
        curves.append(
            TimeSeriesSolution(
                t=times,
                y=states,
                time_name="s",
                coordinate_names=forward.coordinate_names,
                metadata={
                    **per_seed_metadata,
                    "trace_backend": "pyna.dynamics.fixed_step_rk4_cartesian",
                    "system_type": type(system).__name__,
                    "dimension": int(seed_arr.shape[1]),
                    "normalize": bool(normalize),
                    "min_speed": float(min_speed),
                    "bidirectional": True,
                    "backward_terminated_reason": backward.metadata.get("terminated_reason"),
                    "forward_terminated_reason": forward.metadata.get("terminated_reason"),
                    "finite_fraction": float(np.count_nonzero(np.isfinite(states)) / max(states.size, 1)),
                },
            )
        )
    return curves


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
    "cartesian_points_to_cylindrical_coords",
    "cylindrical_vectors_to_cartesian",
    "vector_field_cartesian_rhs",
    "vector_field_cylind_cartesian_rhs",
    "trace_cartesian_trajectory",
    "trace_cartesian_streamlines",
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
