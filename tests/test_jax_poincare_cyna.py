"""Parity checks between differentiable JAX and production Cyna maps."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from pyna.fields import VectorFieldCylind
from pyna.toroidal.flt import trace_map_batch_span_field
from pyna.toroidal.flt.jax_poincare import poincare_map


def _require_cyna():
    import pyna._cyna as cyna

    if not cyna.is_available() or cyna.VectorFieldCylind is None:
        pytest.skip("cyna VectorFieldCylind is unavailable")


def _cartesian_hyperbolic_field(xyz, rate):
    """B_phi=R with exact flow R'=a(R-1), Z'=-a Z."""

    x, y, z = xyz
    radius = jnp.sqrt(x * x + y * y)
    cos_phi = x / radius
    sin_phi = y / radius
    b_r = rate * (radius - 1.0)
    b_z = -rate * z
    b_phi = radius
    return jnp.array(
        [
            b_r * cos_phi - b_phi * sin_phi,
            b_r * sin_phi + b_phi * cos_phi,
            b_z,
        ]
    )


def _cylindrical_hyperbolic_field(rate):
    radius = np.linspace(0.7, 1.3, 33)
    height = np.linspace(-0.3, 0.3, 33)
    phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    rr, zz, _ = np.meshgrid(radius, height, phi, indexing="ij")
    return VectorFieldCylind(
        R=radius,
        Z=height,
        Phi=phi,
        BR=rate * (rr - 1.0),
        BZ=-rate * zz,
        BPhi=rr,
        nfp=1,
    )


@pytest.mark.parametrize(
    ("phi_start", "phi_span"),
    [
        (0.17, 0.8),
        (0.97, -0.8),
    ],
)
def test_jax_map_matches_cyna_for_identical_field_and_rk4_steps(
    phi_start,
    phi_span,
):
    _require_cyna()
    rate = 0.35
    step_size = 0.01
    n_steps = round(abs(phi_span) / step_size)
    seeds = jnp.array(
        [
            [0.86, -0.12],
            [1.04, 0.03],
            [1.17, 0.11],
        ]
    )

    jax_map = jax.jit(
        jax.vmap(
            lambda rz: poincare_map(
                _cartesian_hyperbolic_field,
                rate,
                rz,
                phi_start=phi_start,
                phi_span=phi_span,
                n_steps=n_steps,
            )
        )
    )(seeds)

    counts, cyna_r, cyna_z = trace_map_batch_span_field(
        _cylindrical_hyperbolic_field(rate),
        np.asarray(seeds[:, 0]),
        np.asarray(seeds[:, 1]),
        phi_start,
        phi_span,
        1,
        step_size,
        n_threads=1,
    )
    cyna_map = np.column_stack((cyna_r[: len(seeds)], cyna_z[: len(seeds)]))

    np.testing.assert_array_equal(counts, np.ones(len(seeds), dtype=int))
    np.testing.assert_allclose(jax_map, cyna_map, rtol=0.0, atol=2e-12)
