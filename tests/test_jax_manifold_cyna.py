"""Fixed-step parity checks between JAX and Cyna manifold generations."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from pyna.fields import VectorFieldCylind
from pyna.topo.jax_manifold import (
    compare_cyna_manifold_generations,
    trace_manifold_generations,
)
from pyna.toroidal.flt import trace_map_batch_span_field


RATE = 0.35
PHI_SPAN = 0.6
N_STEPS_PER_SPAN = 60
N_GENERATIONS = 3


def _require_cyna():
    import pyna._cyna as cyna

    if not cyna.is_available() or cyna.VectorFieldCylind is None:
        pytest.skip("cyna VectorFieldCylind is unavailable")


def _cartesian_field(xyz, rate):
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


def _cyna_field():
    radius = np.linspace(0.7, 1.5, 49)
    height = np.linspace(-0.4, 0.4, 49)
    phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    rr, zz, _ = np.meshgrid(radius, height, phi, indexing="ij")
    return VectorFieldCylind(
        R=radius,
        Z=height,
        Phi=phi,
        BR=RATE * (rr - 1.0),
        BZ=-RATE * zz,
        BPhi=rr,
        nfp=1,
    )


@pytest.mark.parametrize(
    ("stability", "seeds", "signed_span"),
    [
        ("unstable", jnp.array([[1.001, 0.0], [1.002, 0.0]]), PHI_SPAN),
        ("stable", jnp.array([[1.0, 0.001], [1.0, 0.002]]), -PHI_SPAN),
    ],
)
def test_jax_manifold_generations_match_cyna(stability, seeds, signed_span):
    _require_cyna()
    generations = trace_manifold_generations(
        _cartesian_field,
        RATE,
        seeds,
        stability=stability,
        phi_span=PHI_SPAN,
        n_generations=N_GENERATIONS,
        n_steps_per_span=N_STEPS_PER_SPAN,
    )
    counts, r_flat, z_flat = trace_map_batch_span_field(
        _cyna_field(),
        np.asarray(seeds[:, 0]),
        np.asarray(seeds[:, 1]),
        0.0,
        signed_span,
        N_GENERATIONS,
        abs(signed_span) / N_STEPS_PER_SPAN,
        n_threads=1,
    )
    cyna = np.stack(
        (
            np.asarray(r_flat).reshape(len(seeds), N_GENERATIONS).T,
            np.asarray(z_flat).reshape(len(seeds), N_GENERATIONS).T,
        ),
        axis=-1,
    )

    np.testing.assert_array_equal(counts, N_GENERATIONS)
    np.testing.assert_allclose(generations[1:], cyna, rtol=0.0, atol=3.0e-12)
    report = compare_cyna_manifold_generations(
        generations,
        counts,
        r_flat,
        z_flat,
        absolute_tolerance=3.0e-12,
    )
    assert report.complete
    assert report.accepted
    assert report.max_deviation <= report.absolute_tolerance

    rejected = compare_cyna_manifold_generations(
        generations,
        counts,
        np.asarray(r_flat) + 1.0e-5,
        z_flat,
        absolute_tolerance=3.0e-12,
    )
    assert not rejected.accepted

