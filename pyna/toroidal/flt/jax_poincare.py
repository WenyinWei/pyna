"""Differentiable fixed-toroidal-angle field-line maps.

This module is the optional-JAX counterpart of the production Cyna tracing
backend.  It is intentionally small and field-source agnostic: downstream
packages provide a Cartesian magnetic-field callable with signature
``field_fn(xyz, params) -> (Bx, By, Bz)``.  PyNA owns the conversion to the
fixed-phi field-line equations and their integration.

The implementation uses a fixed-step fourth-order Runge--Kutta scheme.  A
fixed discretization makes the map deterministic and gives JAX a discretely
exact derivative with respect to both the launch point and ``params``.  Cyna
remains the authoritative backend for high-throughput and wall-aware tracing.

Importing this module requires the optional ``pyna-chaos[jax]`` dependency.
The rest of PyNA does not import JAX.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "pyna.toroidal.flt.jax_poincare requires JAX; "
        "install the optional 'pyna-chaos[jax]' dependencies"
    ) from exc


CartesianField = Callable[[Any, Any], Any]


def cylindrical_field_components(
    field_fn: CartesianField,
    params: Any,
    rz: Any,
    phi: Any,
) -> Any:
    """Evaluate a Cartesian field as ``(B_R, B_Z, B_phi)``.

    Parameters
    ----------
    field_fn:
        Callable ``field_fn(xyz, params)`` returning physical Cartesian field
        components.
    params:
        Arbitrary JAX PyTree containing the active field parameters.
    rz:
        Poloidal position ``(R, Z)``.
    phi:
        Geometric toroidal angle in radians.
    """

    rz = jnp.asarray(rz)
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    xyz = jnp.stack((rz[0] * cos_phi, rz[0] * sin_phi, rz[1]))
    bxyz = jnp.asarray(field_fn(xyz, params))

    b_r = bxyz[0] * cos_phi + bxyz[1] * sin_phi
    b_phi = -bxyz[0] * sin_phi + bxyz[1] * cos_phi
    return jnp.stack((b_r, bxyz[2], b_phi))


def fixed_phi_rhs(
    phi: Any,
    rz: Any,
    params: Any,
    *,
    field_fn: CartesianField,
    bphi_floor: float = 0.0,
) -> Any:
    """Return ``d(R,Z)/dphi`` for a toroidally monotone field line.

    The physical cylindrical components obey

    ``dR/dphi = R * B_R / B_phi`` and
    ``dZ/dphi = R * B_Z / B_phi``.

    Values with ``abs(B_phi) <= bphi_floor`` produce NaNs.  This explicit
    failure is preferable to silently regularizing a map whose fixed-phi
    parameterization is invalid.
    """

    b_r, b_z, b_phi = cylindrical_field_components(field_fn, params, rz, phi)
    denominator = jnp.where(jnp.abs(b_phi) > bphi_floor, b_phi, jnp.nan)
    return rz[0] * jnp.stack((b_r, b_z)) / denominator


def poincare_map(
    field_fn: CartesianField,
    params: Any,
    rz0: Any,
    *,
    phi_span: Any,
    phi_start: Any = 0.0,
    n_steps: int = 256,
    bphi_floor: float = 0.0,
) -> Any:
    """Trace a differentiable fixed-phi map from ``rz0``.

    ``phi_span`` is explicit so callers choose whether one application means a
    field period, several field periods, or a full toroidal turn.  Negative
    spans provide the inverse-time map used for stable-manifold tracing.

    This function is compatible with ``jax.jit``, ``jax.vmap``, ``jax.jvp``,
    ``jax.vjp``, and the Jacobian transforms.  The field callable is normally
    closed over as a static Python value when applying ``jax.jit``.
    """

    try:
        steps = operator.index(n_steps)
    except TypeError as exc:
        raise TypeError("n_steps must be an integer") from exc
    if steps <= 0:
        raise ValueError("n_steps must be positive")
    if bphi_floor < 0.0:
        raise ValueError("bphi_floor must be non-negative")

    rz_initial = jnp.asarray(rz0)
    if rz_initial.shape != (2,):
        raise ValueError("rz0 must have shape (2,)")

    phi0 = jnp.asarray(phi_start, dtype=rz_initial.dtype)
    step_size = jnp.asarray(phi_span, dtype=rz_initial.dtype) / steps

    def rhs(phi: Any, rz: Any) -> Any:
        return fixed_phi_rhs(
            phi,
            rz,
            params,
            field_fn=field_fn,
            bphi_floor=bphi_floor,
        )

    def rk4_step(index: int, rz: Any) -> Any:
        phi = phi0 + step_size * index
        k1 = rhs(phi, rz)
        k2 = rhs(phi + 0.5 * step_size, rz + 0.5 * step_size * k1)
        k3 = rhs(phi + 0.5 * step_size, rz + 0.5 * step_size * k2)
        k4 = rhs(phi + step_size, rz + step_size * k3)
        return rz + (step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return jax.lax.fori_loop(0, steps, rk4_step, rz_initial)


def poincare_map_jacobian(
    field_fn: CartesianField,
    params: Any,
    rz0: Any,
    *,
    phi_span: Any,
    phi_start: Any = 0.0,
    n_steps: int = 256,
    bphi_floor: float = 0.0,
) -> Any:
    """Return the discrete monodromy ``D_x P`` at ``rz0``."""

    def map_from_initial_position(rz: Any) -> Any:
        return poincare_map(
            field_fn,
            params,
            rz,
            phi_span=phi_span,
            phi_start=phi_start,
            n_steps=n_steps,
            bphi_floor=bphi_floor,
        )

    return jax.jacfwd(map_from_initial_position)(jnp.asarray(rz0))


__all__ = [
    "CartesianField",
    "cylindrical_field_components",
    "fixed_phi_rhs",
    "poincare_map",
    "poincare_map_jacobian",
]
