"""Implicitly differentiable local wall crossings of magnetic field lines.

The outer Cyna trace identifies the physical first-wall hit and supplies a
nearby toroidal-angle guess.  This optional-JAX module holds that discrete hit
choice fixed and solves the smooth event equation

``g(R(phi), Z(phi), phi, wall_params) = 0``.

Derivatives are defined by the implicit-function theorem, not by
backpropagating through Newton iterations.  A crossing that leaves its frozen
angle window, becomes tangent to the wall, or fails to converge must be
rejected by the outer loop.  Cyna remains authoritative for first-hit identity
and production wall geometry.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from typing import Any, NamedTuple

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "pyna.topo.jax_strike requires JAX; "
        "install the optional 'pyna-chaos[jax]' dependencies"
    ) from exc

from pyna.toroidal.flt.jax_poincare import CartesianField, poincare_map


WallLevelSet = Callable[[Any, Any, Any], Any]


class WallIntersectionState(NamedTuple):
    """Differentiable hit geometry and outer-loop validity diagnostics."""

    point_RZPhi: Any
    point_xyz: Any
    phi: Any
    wall_residual: Any
    transversality: Any
    phi_shift_from_guess: Any
    within_window: Any
    converged: Any


def _positive_integer(value: int, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _wall_residual(
    field_fn: CartesianField,
    wall_level_set_fn: WallLevelSet,
    params: Any,
    wall_params: Any,
    rz0: Any,
    phi_start: Any,
    phi: Any,
    *,
    n_steps: int,
    bphi_floor: float,
) -> Any:
    rz = poincare_map(
        field_fn,
        params,
        rz0,
        phi_start=phi_start,
        phi_span=phi - phi_start,
        n_steps=n_steps,
        bphi_floor=bphi_floor,
    )
    residual = jnp.asarray(wall_level_set_fn(rz, phi, wall_params))
    if residual.shape != ():
        raise ValueError("wall_level_set_fn must return a scalar")
    return residual


def solve_wall_intersection(
    field_fn: CartesianField,
    params: Any,
    rz0: Any,
    wall_level_set_fn: WallLevelSet,
    wall_params: Any,
    *,
    phi_start: Any,
    phi_guess: Any,
    n_steps: int = 256,
    newton_iterations: int = 8,
    newton_damping: float = 1.0,
    maximum_phi_shift: float | None = None,
    bphi_floor: float = 0.0,
) -> Any:
    """Solve one local wall event with an implicit derivative.

    ``phi_guess`` and ``maximum_phi_shift`` are frozen outer-loop data from a
    production first-wall hit.  ``rz0`` remains active so gradients include
    motion of a manifold seed or periodic X-line.  The fixed ``n_steps`` RK4
    discretization is differentiable even though its step size changes with
    the solved endpoint angle.
    """

    steps = _positive_integer(n_steps, "n_steps")
    iterations = _positive_integer(newton_iterations, "newton_iterations")
    if not callable(field_fn):
        raise TypeError("field_fn must be callable")
    if not callable(wall_level_set_fn):
        raise TypeError("wall_level_set_fn must be callable")
    if not 0.0 < newton_damping <= 1.0:
        raise ValueError("newton_damping must lie in (0, 1]")
    if bphi_floor < 0.0:
        raise ValueError("bphi_floor must be non-negative")
    if maximum_phi_shift is None:
        window = None
    else:
        window = float(maximum_phi_shift)
        if not math.isfinite(window) or window <= 0.0:
            raise ValueError("maximum_phi_shift must be positive and finite")

    initial_rz = jnp.asarray(rz0)
    if initial_rz.shape != (2,):
        raise ValueError("rz0 must have shape (2,)")
    start = jnp.asarray(phi_start, dtype=initial_rz.dtype)
    guess = jax.lax.stop_gradient(
        jnp.asarray(phi_guess, dtype=initial_rz.dtype)
    )
    if start.shape != () or guess.shape != ():
        raise ValueError("phi_start and phi_guess must be scalars")

    def residual(
        phi,
        active_params,
        active_wall_params,
        active_rz0,
        active_start,
    ):
        return _wall_residual(
            field_fn,
            wall_level_set_fn,
            active_params,
            active_wall_params,
            active_rz0,
            active_start,
            phi,
            n_steps=steps,
            bphi_floor=bphi_floor,
        )

    def newton_solve(
        active_params,
        active_wall_params,
        active_rz0,
        active_start,
    ):
        def newton_step(_, phi):
            value = residual(
                phi,
                active_params,
                active_wall_params,
                active_rz0,
                active_start,
            )
            derivative = jax.jacfwd(residual, argnums=0)(
                phi,
                active_params,
                active_wall_params,
                active_rz0,
                active_start,
            )
            candidate = phi - newton_damping * value / derivative
            if window is not None:
                candidate = jnp.clip(candidate, guess - window, guess + window)
            return candidate

        return jax.lax.fori_loop(0, iterations, newton_step, guess)

    @jax.custom_jvp
    def implicit_solve(
        active_params,
        active_wall_params,
        active_rz0,
        active_start,
    ):
        return newton_solve(
            active_params,
            active_wall_params,
            active_rz0,
            active_start,
        )

    @implicit_solve.defjvp
    def implicit_solve_jvp(primals, tangents):
        active_params, active_wall_params, active_rz0, active_start = primals
        (
            params_tangent,
            wall_params_tangent,
            rz0_tangent,
            start_tangent,
        ) = tangents
        phi = implicit_solve(
            active_params,
            active_wall_params,
            active_rz0,
            active_start,
        )
        derivative = jax.jacfwd(residual, argnums=0)(
            phi,
            active_params,
            active_wall_params,
            active_rz0,
            active_start,
        )

        def residual_at_fixed_phi(
            parameters,
            wall_parameters,
            seed_rz,
            start_phi,
        ):
            return residual(
                phi,
                parameters,
                wall_parameters,
                seed_rz,
                start_phi,
            )

        _, right_hand_side = jax.jvp(
            residual_at_fixed_phi,
            (active_params, active_wall_params, active_rz0, active_start),
            (
                params_tangent,
                wall_params_tangent,
                rz0_tangent,
                start_tangent,
            ),
        )
        return phi, -right_hand_side / derivative

    return implicit_solve(params, wall_params, initial_rz, start)


def wall_intersection_state(
    field_fn: CartesianField,
    params: Any,
    rz0: Any,
    wall_level_set_fn: WallLevelSet,
    wall_params: Any,
    *,
    phi_start: Any,
    phi_guess: Any,
    n_steps: int = 256,
    newton_iterations: int = 8,
    newton_damping: float = 1.0,
    maximum_phi_shift: float | None = None,
    bphi_floor: float = 0.0,
    residual_tolerance: float = 1.0e-10,
    minimum_abs_transversality: float = 1.0e-8,
) -> WallIntersectionState:
    """Return local hit geometry and convergence/transversality diagnostics."""

    residual_limit = float(residual_tolerance)
    transversality_floor = float(minimum_abs_transversality)
    if not math.isfinite(residual_limit) or residual_limit < 0.0:
        raise ValueError("residual_tolerance must be non-negative and finite")
    if not math.isfinite(transversality_floor) or transversality_floor < 0.0:
        raise ValueError(
            "minimum_abs_transversality must be non-negative and finite"
        )
    phi = solve_wall_intersection(
        field_fn,
        params,
        rz0,
        wall_level_set_fn,
        wall_params,
        phi_start=phi_start,
        phi_guess=phi_guess,
        n_steps=n_steps,
        newton_iterations=newton_iterations,
        newton_damping=newton_damping,
        maximum_phi_shift=maximum_phi_shift,
        bphi_floor=bphi_floor,
    )
    initial_rz = jnp.asarray(rz0)
    start = jnp.asarray(phi_start, dtype=initial_rz.dtype)
    rz = poincare_map(
        field_fn,
        params,
        initial_rz,
        phi_start=start,
        phi_span=phi - start,
        n_steps=_positive_integer(n_steps, "n_steps"),
        bphi_floor=bphi_floor,
    )
    residual_function = lambda active_phi: _wall_residual(
        field_fn,
        wall_level_set_fn,
        params,
        wall_params,
        initial_rz,
        start,
        active_phi,
        n_steps=_positive_integer(n_steps, "n_steps"),
        bphi_floor=bphi_floor,
    )
    wall_residual = residual_function(phi)
    transversality = jax.jacfwd(residual_function)(phi)
    guess = jax.lax.stop_gradient(jnp.asarray(phi_guess, dtype=phi.dtype))
    shift = phi - guess
    within_window = (
        jnp.asarray(True)
        if maximum_phi_shift is None
        else jnp.abs(shift) < float(maximum_phi_shift)
    )
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    point_xyz = jnp.stack((rz[0] * cos_phi, rz[0] * sin_phi, rz[1]))
    finite = (
        jnp.all(jnp.isfinite(rz))
        & jnp.isfinite(phi)
        & jnp.isfinite(wall_residual)
        & jnp.isfinite(transversality)
    )
    converged = (
        finite
        & within_window
        & (jnp.abs(wall_residual) <= residual_limit)
        & (jnp.abs(transversality) >= transversality_floor)
    )
    return WallIntersectionState(
        point_RZPhi=jnp.stack((rz[0], rz[1], phi)),
        point_xyz=point_xyz,
        phi=phi,
        wall_residual=wall_residual,
        transversality=transversality,
        phi_shift_from_guess=shift,
        within_window=within_window,
        converged=converged,
    )


__all__ = [
    "WallIntersectionState",
    "WallLevelSet",
    "solve_wall_intersection",
    "wall_intersection_state",
]
