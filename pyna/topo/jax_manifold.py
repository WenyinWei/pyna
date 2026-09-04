"""Differentiable local invariant-manifold branches for magnetic maps.

This optional-JAX module is the inner-loop counterpart of PyNA's production
Cyna manifold tracer.  It preserves the same discrete-map conventions:
unstable branches advance the forward return map, stable branches advance the
backward return map, and one generation contains a complete labelled seed
segment.

Branch discovery remains an outer-loop operation.  Eigenvector selection,
orientation, seed spacing, side labels, and sample correspondence are frozen
with respect to differentiation.  Gradients pass through the periodic-point
position, the seed origins, and every field-line map evaluation.  Cyna remains
authoritative for wall-aware tracing and accepted-step validation.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "pyna.topo.jax_manifold requires JAX; "
        "install the optional 'pyna-chaos[jax]' dependencies"
    ) from exc

from pyna.toroidal.flt.jax_poincare import CartesianField, poincare_map


class HyperbolicDirections(NamedTuple):
    """Outer-frozen stable and unstable eigendata for one return map."""

    stable_eigenvalue: Any
    stable_direction: Any
    unstable_eigenvalue: Any
    unstable_direction: Any
    stable_backward_expansion: Any
    unstable_expansion: Any
    valid: Any


@dataclass(frozen=True)
class ManifoldParityReport:
    """Fixed-correspondence comparison between JAX and Cyna generations."""

    n_seeds: int
    n_generations: int
    complete: bool
    max_deviation: float
    rms_deviation: float
    absolute_tolerance: float
    accepted: bool


def _positive_integer(value: int, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _oriented_real_unit(vector: Any, reference: Any | None) -> Any:
    real_vector = jnp.real(vector)
    unit = real_vector / jnp.linalg.norm(real_vector)
    if reference is None:
        pivot = jnp.argmax(jnp.abs(unit))
        sign = jnp.where(unit[pivot] < 0.0, -1.0, 1.0)
    else:
        reference_array = jax.lax.stop_gradient(
            jnp.asarray(reference, dtype=real_vector.dtype)
        )
        if reference_array.shape != (2,):
            raise ValueError("reference directions must have shape (2,)")
        sign = jnp.where(jnp.dot(unit, reference_array) < 0.0, -1.0, 1.0)
    return sign * unit


def hyperbolic_directions(
    monodromy: Any,
    *,
    stable_reference: Any | None = None,
    unstable_reference: Any | None = None,
    imaginary_tolerance: float = 1.0e-8,
) -> HyperbolicDirections:
    """Select consistently oriented stable and unstable eigendirections.

    The returned eigendata is deliberately stop-gradient outer-loop state.
    Callers must inspect ``valid`` before constructing seeds.  Differentiable
    stability objectives should use ``pyna.topo.jax_periodic`` directly rather
    than differentiating through this branch-selection operation.
    """

    tolerance = float(imaginary_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("imaginary_tolerance must be non-negative and finite")

    matrix = jax.lax.stop_gradient(jnp.asarray(monodromy))
    if matrix.shape != (2, 2):
        raise ValueError("monodromy must have shape (2, 2)")
    eigenvalues, eigenvectors = jnp.linalg.eig(matrix)
    eigenvalues = jax.lax.stop_gradient(eigenvalues)
    eigenvectors = jax.lax.stop_gradient(eigenvectors)

    moduli = jnp.abs(eigenvalues)
    stable_index = jnp.argmin(moduli)
    unstable_index = jnp.argmax(moduli)
    stable_value = eigenvalues[stable_index]
    unstable_value = eigenvalues[unstable_index]
    stable_direction = _oriented_real_unit(
        eigenvectors[:, stable_index], stable_reference
    )
    unstable_direction = _oriented_real_unit(
        eigenvectors[:, unstable_index], unstable_reference
    )

    imaginary_scale = jnp.maximum(
        1.0, jnp.max(jnp.abs(jnp.real(eigenvalues)))
    )
    real_enough = jnp.max(jnp.abs(jnp.imag(eigenvalues))) <= (
        tolerance * imaginary_scale
    )
    finite = (
        jnp.all(jnp.isfinite(jnp.real(eigenvalues)))
        & jnp.all(jnp.isfinite(jnp.imag(eigenvalues)))
        & jnp.all(jnp.isfinite(stable_direction))
        & jnp.all(jnp.isfinite(unstable_direction))
    )
    stable_modulus = jnp.abs(stable_value)
    unstable_modulus = jnp.abs(unstable_value)
    valid = (
        finite
        & real_enough
        & (stable_modulus > 0.0)
        & (stable_modulus < 1.0)
        & (unstable_modulus > 1.0)
    )
    return HyperbolicDirections(
        stable_eigenvalue=jnp.real(stable_value),
        stable_direction=stable_direction,
        unstable_eigenvalue=jnp.real(unstable_value),
        unstable_direction=unstable_direction,
        stable_backward_expansion=1.0 / stable_modulus,
        unstable_expansion=unstable_modulus,
        valid=valid,
    )


def fundamental_segment_distances(
    expansion: Any,
    *,
    minimum_distance: float,
    maximum_distance: float,
    n_seeds: int,
) -> Any:
    """Build PyNA's production eigenvalue-geometric seed carousel.

    For expansion ``lambda`` and ``n_seeds=N``, adjacent distances have ratio
    ``lambda**(1/N)``.  Thus the image of the first seed follows the final seed
    in the next generation.  Expansion and the resulting distances are frozen
    outer-loop data.  A non-finite expansion or ``abs(expansion) <= 1`` yields
    NaN distances and must cause the outer candidate to be rejected.
    """

    count = _positive_integer(n_seeds, "n_seeds")
    minimum = float(minimum_distance)
    maximum = float(maximum_distance)
    if not np.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("minimum_distance must be positive and finite")
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("maximum_distance must be positive and finite")

    expansion_array = jax.lax.stop_gradient(jnp.abs(jnp.asarray(expansion)))
    dtype = jnp.result_type(expansion_array, minimum, maximum)
    expansion_array = expansion_array.astype(dtype)
    ratio = expansion_array ** (1.0 / count)
    last_factor = ratio ** max(0, count - 1)
    first = jnp.minimum(jnp.asarray(minimum, dtype=dtype), maximum / last_factor)
    distances = first * ratio ** jnp.arange(count, dtype=dtype)
    valid = jnp.isfinite(expansion_array) & (expansion_array > 1.0)
    return jax.lax.stop_gradient(jnp.where(valid, distances, jnp.nan))


def manifold_seed_segment(
    origin: Any,
    direction: Any,
    distances: Any,
    *,
    side: int,
) -> Any:
    """Create one signed fundamental segment adjacent to an X-line cut.

    ``origin`` remains differentiable, while direction and distances are
    frozen branch metadata supplied by the outer continuation loop.
    """

    if side not in (-1, 1):
        raise ValueError("side must be -1 or +1")
    origin_array = jnp.asarray(origin)
    direction_array = jax.lax.stop_gradient(
        jnp.asarray(direction, dtype=origin_array.dtype)
    )
    distance_array = jax.lax.stop_gradient(
        jnp.asarray(distances, dtype=origin_array.dtype)
    )
    if origin_array.shape != (2,) or direction_array.shape != (2,):
        raise ValueError("origin and direction must have shape (2,)")
    if distance_array.ndim != 1:
        raise ValueError("distances must be one-dimensional")
    unit_direction = direction_array / jnp.linalg.norm(direction_array)
    return origin_array + side * distance_array[:, None] * unit_direction


def trace_manifold_generations(
    field_fn: CartesianField,
    params: Any,
    seeds: Any,
    *,
    stability: str,
    phi_span: Any,
    phi_start: Any = 0.0,
    map_power: int = 1,
    n_generations: int = 1,
    n_steps_per_span: int = 256,
    bphi_floor: float = 0.0,
) -> Any:
    """Trace complete differentiable generations of ``W^u`` or ``W^s``.

    The result has shape ``(n_generations + 1, n_seeds, 2)``; index zero is
    the labelled seed segment.  ``phi_span`` denotes the forward base return
    span.  Unstable branches use the positive map and stable branches use its
    negative-time counterpart.  Reusing ``phi_start`` each generation assumes
    the magnetic field is periodic over ``map_power * phi_span``.
    """

    branch = str(stability).lower()
    if branch not in {"stable", "unstable"}:
        raise ValueError("stability must be 'stable' or 'unstable'")
    power = _positive_integer(map_power, "map_power")
    generations = _positive_integer(n_generations, "n_generations")
    steps = _positive_integer(n_steps_per_span, "n_steps_per_span")
    initial = jnp.asarray(seeds)
    if initial.ndim != 2 or initial.shape[1] != 2:
        raise ValueError("seeds must have shape (n_seeds, 2)")

    direction = 1.0 if branch == "unstable" else -1.0
    signed_span = direction * jnp.asarray(phi_span) * power
    integration_steps = steps * power

    def map_point(rz):
        return poincare_map(
            field_fn,
            params,
            rz,
            phi_start=phi_start,
            phi_span=signed_span,
            n_steps=integration_steps,
            bphi_floor=bphi_floor,
        )

    def next_generation(points, _):
        mapped = jax.vmap(map_point)(points)
        return mapped, mapped

    _, traced = jax.lax.scan(
        next_generation,
        initial,
        xs=None,
        length=generations,
    )
    return jnp.concatenate((initial[None, ...], traced), axis=0)


def compare_cyna_manifold_generations(
    jax_generations: Any,
    cyna_counts: Any,
    cyna_r_flat: Any,
    cyna_z_flat: Any,
    *,
    absolute_tolerance: float,
) -> ManifoldParityReport:
    """Compare labelled JAX generations with seed-major Cyna map output.

    Cyna's fixed-stride arrays exclude the initial seeds and store every
    generation for one seed contiguously.  This adapter performs no tracing;
    it is intentionally a narrow outer-loop validation boundary.
    """

    jax_points = np.asarray(jax_generations, dtype=float)
    if jax_points.ndim != 3 or jax_points.shape[2] != 2:
        raise ValueError(
            "jax_generations must have shape (n_generations + 1, n_seeds, 2)"
        )
    n_generations = jax_points.shape[0] - 1
    n_seeds = jax_points.shape[1]
    counts = np.asarray(cyna_counts, dtype=int).ravel()
    if counts.shape != (n_seeds,):
        raise ValueError("cyna_counts must have shape (n_seeds,)")

    expected_size = n_seeds * n_generations
    r_values = np.asarray(cyna_r_flat, dtype=float).ravel()
    z_values = np.asarray(cyna_z_flat, dtype=float).ravel()
    if r_values.size != expected_size or z_values.size != expected_size:
        raise ValueError("Cyna coordinate arrays have the wrong fixed-stride size")
    tolerance = float(absolute_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("absolute_tolerance must be non-negative and finite")

    complete = bool(np.all(counts == n_generations))
    cyna_points = np.stack(
        (
            r_values.reshape(n_seeds, n_generations).T,
            z_values.reshape(n_seeds, n_generations).T,
        ),
        axis=-1,
    )
    differences = np.linalg.norm(jax_points[1:] - cyna_points, axis=-1)
    finite = bool(np.all(np.isfinite(differences)))
    maximum = float(np.max(differences)) if differences.size else 0.0
    rms = (
        float(np.sqrt(np.mean(differences * differences)))
        if differences.size
        else 0.0
    )
    return ManifoldParityReport(
        n_seeds=n_seeds,
        n_generations=n_generations,
        complete=complete,
        max_deviation=maximum,
        rms_deviation=rms,
        absolute_tolerance=tolerance,
        accepted=complete and finite and maximum <= tolerance,
    )


__all__ = [
    "HyperbolicDirections",
    "ManifoldParityReport",
    "compare_cyna_manifold_generations",
    "fundamental_segment_distances",
    "hyperbolic_directions",
    "manifold_seed_segment",
    "trace_manifold_generations",
]
