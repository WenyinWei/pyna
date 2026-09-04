"""Implicitly differentiable periodic points of fixed-phi magnetic maps.

The nonlinear solve is used to locate a periodic point, but derivatives are
defined by the root equation ``P^m(x, params) - x = 0`` through
``jax.lax.custom_root``.  Consequently Stage-2 coil derivatives do not
backpropagate through the Newton iterations.

This optional module requires JAX and builds on
``pyna.toroidal.flt.jax_poincare``.  Hyperbolic points are the initial scope;
parabolic points require an augmented continuation system because ``M-I`` is
singular there.
"""

from __future__ import annotations

import operator
from typing import Any, NamedTuple

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "pyna.topo.jax_periodic requires JAX; "
        "install the optional 'pyna-chaos[jax]' dependencies"
    ) from exc

from pyna.toroidal.flt.jax_poincare import CartesianField, poincare_map


class PeriodicPointState(NamedTuple):
    """Diagnostics for one periodic point on a section.

    The root, residual, monodromy, trace, determinant, Greene residue,
    hyperbolicity margin, and eigenvalues remain in the JAX derivative path.
    Eigenvectors are stop-gradient outer-loop data because JAX does not define
    derivatives for non-symmetric eigenvectors and their branch choice is not
    smooth at degeneracies.
    """

    position: Any
    residual: Any
    residual_norm: Any
    monodromy: Any
    trace: Any
    determinant: Any
    greene_residue: Any
    hyperbolicity_margin: Any
    eigenvalues: Any
    eigenvectors: Any
    converged: Any


def _positive_integer(value: int, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _periodic_map(
    field_fn: CartesianField,
    params: Any,
    rz: Any,
    *,
    phi_span: Any,
    phi_start: Any,
    map_power: int,
    n_steps_per_span: int,
    bphi_floor: float,
) -> Any:
    return poincare_map(
        field_fn,
        params,
        rz,
        phi_start=phi_start,
        phi_span=jnp.asarray(phi_span) * map_power,
        n_steps=n_steps_per_span * map_power,
        bphi_floor=bphi_floor,
    )


def solve_periodic_point(
    field_fn: CartesianField,
    params: Any,
    initial_guess: Any,
    *,
    phi_span: Any,
    phi_start: Any = 0.0,
    map_power: int = 1,
    n_steps_per_span: int = 256,
    newton_iterations: int = 8,
    newton_damping: float = 1.0,
    bphi_floor: float = 0.0,
) -> Any:
    """Solve ``P^map_power(x, params) - x = 0`` with implicit derivatives.

    The returned point is differentiated as a root, independently of the
    finite number of Newton iterations.  Callers should inspect
    :func:`periodic_point_state` and reject an outer-loop update when the
    residual has not converged.
    """

    power = _positive_integer(map_power, "map_power")
    steps = _positive_integer(n_steps_per_span, "n_steps_per_span")
    iterations = _positive_integer(newton_iterations, "newton_iterations")
    if not 0.0 < newton_damping <= 1.0:
        raise ValueError("newton_damping must lie in (0, 1]")

    guess = jnp.asarray(initial_guess)
    if guess.shape != (2,):
        raise ValueError("initial_guess must have shape (2,)")

    def displacement(rz):
        return _periodic_map(
            field_fn,
            params,
            rz,
            phi_span=phi_span,
            phi_start=phi_start,
            map_power=power,
            n_steps_per_span=steps,
            bphi_floor=bphi_floor,
        ) - rz

    def newton_solve(function, starting_point):
        def newton_step(_, position):
            residual = function(position)
            jacobian = jax.jacfwd(function)(position)
            correction = jnp.linalg.solve(jacobian, residual)
            return position - newton_damping * correction

        return jax.lax.fori_loop(0, iterations, newton_step, starting_point)

    def tangent_solve(linear_function, right_hand_side):
        basis = jnp.eye(right_hand_side.shape[0], dtype=right_hand_side.dtype)
        matrix = jax.vmap(linear_function, out_axes=1)(basis)
        return jnp.linalg.solve(matrix, right_hand_side)

    return jax.lax.custom_root(
        displacement,
        guess,
        newton_solve,
        tangent_solve,
    )


def periodic_point_state(
    field_fn: CartesianField,
    params: Any,
    initial_guess: Any,
    *,
    phi_span: Any,
    phi_start: Any = 0.0,
    map_power: int = 1,
    n_steps_per_span: int = 256,
    newton_iterations: int = 8,
    newton_damping: float = 1.0,
    bphi_floor: float = 0.0,
    residual_tolerance: float = 1.0e-10,
) -> PeriodicPointState:
    """Solve a periodic point and return its map-based stability data."""

    if residual_tolerance < 0.0:
        raise ValueError("residual_tolerance must be non-negative")
    power = _positive_integer(map_power, "map_power")
    steps = _positive_integer(n_steps_per_span, "n_steps_per_span")
    position = solve_periodic_point(
        field_fn,
        params,
        initial_guess,
        phi_span=phi_span,
        phi_start=phi_start,
        map_power=power,
        n_steps_per_span=steps,
        newton_iterations=newton_iterations,
        newton_damping=newton_damping,
        bphi_floor=bphi_floor,
    )

    def mapped(rz):
        return _periodic_map(
            field_fn,
            params,
            rz,
            phi_span=phi_span,
            phi_start=phi_start,
            map_power=power,
            n_steps_per_span=steps,
            bphi_floor=bphi_floor,
        )

    mapped_position = mapped(position)
    residual = mapped_position - position
    residual_norm = jnp.linalg.norm(residual)
    monodromy = jax.jacfwd(mapped)(position)
    trace = jnp.trace(monodromy)
    determinant = jnp.linalg.det(monodromy)
    eigenvalues = jnp.linalg.eigvals(monodromy)
    _, eigenvectors = jnp.linalg.eig(jax.lax.stop_gradient(monodromy))
    return PeriodicPointState(
        position=position,
        residual=residual,
        residual_norm=residual_norm,
        monodromy=monodromy,
        trace=trace,
        determinant=determinant,
        greene_residue=(2.0 - trace) / 4.0,
        hyperbolicity_margin=jnp.abs(trace) - 2.0,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        converged=residual_norm <= residual_tolerance,
    )


__all__ = [
    "PeriodicPointState",
    "periodic_point_state",
    "solve_periodic_point",
]
