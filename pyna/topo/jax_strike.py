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
from pyna.topo.jax_manifold import manifold_seed_segment
from pyna.topo.jax_periodic import PeriodicPointState, periodic_point_state
from pyna.topo.manifold_correspondence import ManifoldBranchReference
from pyna.topo.manifold_strike_contracts import LocalWallPlane, ManifoldStrikeMatch


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


class ManifoldStrikeState(NamedTuple):
    """Moving X-line, exact labelled seed, and local wall intersection."""

    periodic_point: PeriodicPointState
    seed_RZ: Any
    intersection: WallIntersectionState


def _positive_integer(value: int, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def local_wall_plane_level_set(rz: Any, phi: Any, plane_params: Any) -> Any:
    """Signed Cartesian distance to an outer-frozen local wall plane."""

    values = jnp.asarray(plane_params)
    if values.shape != (6,):
        raise ValueError("local wall plane parameters must have shape (6,)")
    point = values[:3]
    normal = values[3:]
    xyz = jnp.stack((rz[0] * jnp.cos(phi), rz[0] * jnp.sin(phi), rz[1]))
    return jnp.dot(xyz - point, normal)


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


def trace_manifold_strike(
    field_fn: CartesianField,
    params: Any,
    branch_reference: ManifoldBranchReference,
    strike_match: ManifoldStrikeMatch,
    wall_plane: LocalWallPlane,
    *,
    n_steps_per_span: int = 256,
    xline_newton_iterations: int = 8,
    xline_newton_damping: float = 1.0,
    wall_n_steps: int = 2048,
    wall_newton_iterations: int = 8,
    wall_newton_damping: float = 1.0,
    maximum_phi_shift: float,
    bphi_floor: float = 0.0,
    xline_residual_tolerance: float = 1.0e-10,
    wall_residual_tolerance: float = 1.0e-10,
    minimum_abs_transversality: float = 1.0e-8,
) -> ManifoldStrikeState:
    """Trace the exact production-labelled manifold seed to its local wall.

    Branch identity, eigendirection, seed distance/order, Cyna hit angle, and
    wall tangent plane are frozen outer state.  The periodic X-line, local seed
    origin, field-line trajectory, event angle, and hit point remain in the JAX
    derivative path.
    """

    if not isinstance(branch_reference, ManifoldBranchReference):
        raise TypeError("branch_reference must be ManifoldBranchReference")
    if not isinstance(strike_match, ManifoldStrikeMatch):
        raise TypeError("strike_match must be ManifoldStrikeMatch")
    if not isinstance(wall_plane, LocalWallPlane):
        raise TypeError("wall_plane must be LocalWallPlane")
    if strike_match.label != wall_plane.strike_label:
        raise ValueError("wall_plane belongs to a different strike label")
    if strike_match.label.branch_label != branch_reference.label:
        raise ValueError("strike_match belongs to a different manifold branch")
    map_direction = 1.0 if branch_reference.map_span > 0.0 else -1.0
    physical_sign = map_direction * (
        1.0 if branch_reference.stability == "unstable" else -1.0
    )
    expected_direction = "+" if physical_sign > 0.0 else "-"
    if strike_match.label.direction != expected_direction:
        raise ValueError("strike direction disagrees with the manifold branch")
    signed_reference_span = (
        wall_plane.wall_phi_rad - branch_reference.section_phi
    ) * physical_sign
    if signed_reference_span <= 0.0:
        raise ValueError("production wall hit lies opposite the branch trace direction")
    seed_index = branch_reference.seed_index(strike_match.label.seed_order)

    periodic = periodic_point_state(
        field_fn,
        params,
        branch_reference.anchor_RZ_m,
        phi_span=branch_reference.map_span,
        phi_start=branch_reference.section_phi,
        map_power=1,
        n_steps_per_span=n_steps_per_span,
        newton_iterations=xline_newton_iterations,
        newton_damping=xline_newton_damping,
        bphi_floor=bphi_floor,
        residual_tolerance=xline_residual_tolerance,
    )
    seeds = manifold_seed_segment(
        periodic.position,
        branch_reference.direction_RZ,
        branch_reference.seed_distances_m,
        side=branch_reference.seed_side,
    )
    seed = seeds[seed_index]
    intersection = wall_intersection_state(
        field_fn,
        params,
        seed,
        local_wall_plane_level_set,
        wall_plane.level_set_params,
        phi_start=branch_reference.section_phi,
        phi_guess=wall_plane.wall_phi_rad,
        n_steps=wall_n_steps,
        newton_iterations=wall_newton_iterations,
        newton_damping=wall_newton_damping,
        maximum_phi_shift=maximum_phi_shift,
        bphi_floor=bphi_floor,
        residual_tolerance=wall_residual_tolerance,
        minimum_abs_transversality=minimum_abs_transversality,
    )
    return ManifoldStrikeState(
        periodic_point=periodic,
        seed_RZ=seed,
        intersection=intersection,
    )


__all__ = [
    "ManifoldStrikeState",
    "WallIntersectionState",
    "WallLevelSet",
    "local_wall_plane_level_set",
    "solve_wall_intersection",
    "trace_manifold_strike",
    "wall_intersection_state",
]
