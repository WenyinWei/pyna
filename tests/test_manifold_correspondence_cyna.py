"""End-to-end Cyna payload correspondence against JAX generations."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from pyna.fields import VectorFieldCylind
from pyna.topo.jax_manifold import (
    manifold_seed_segment,
    trace_manifold_generations,
)
from pyna.topo.manifold_correspondence import (
    compare_jax_manifold_branch,
    manifold_branch_reference_from_trace,
)
from pyna.topo.toroidal import FixedPoint
from pyna.toroidal.flt import trace_fixed_point_manifolds_field


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


@pytest.mark.parametrize("stability", ["unstable", "stable"])
def test_production_payload_labels_match_jax_branch(stability):
    _require_cyna()
    expansion = np.exp(RATE * PHI_SPAN)
    fixed_point = FixedPoint(
        phi=0.0,
        R=1.0,
        Z=0.0,
        kind="X",
        DPm=np.diag([expansion, 1.0 / expansion]),
    )
    fixed_point.metadata.update(
        {
            "orbit_id": 9,
            "map_order_index": 0,
            "monodromy_map_span": PHI_SPAN,
        }
    )
    payload = trace_fixed_point_manifolds_field(
        _cyna_field(),
        [fixed_point],
        phi_section=0.0,
        map_span=PHI_SPAN,
        N_turns=N_GENERATIONS,
        DPhi=PHI_SPAN / N_STEPS_PER_SPAN,
        seed_distances=np.array([1.0e-3, 2.0e-3]),
        refine_stable_inverse_anchor=False,
    )[0]
    branch = manifold_branch_reference_from_trace(
        payload,
        stability=stability,
        seed_side=1,
        n_generations=N_GENERATIONS,
    )
    seeds = manifold_seed_segment(
        jnp.asarray(branch.anchor_RZ_m),
        jnp.asarray(branch.direction_RZ),
        jnp.asarray(branch.seed_distances_m),
        side=branch.seed_side,
    )
    jax_generations = trace_manifold_generations(
        _cartesian_field,
        RATE,
        seeds,
        stability=stability,
        phi_span=PHI_SPAN,
        n_generations=N_GENERATIONS,
        n_steps_per_span=N_STEPS_PER_SPAN,
    )
    report = compare_jax_manifold_branch(
        branch,
        jax_generations,
        absolute_tolerance_m=3.0e-12,
    )

    assert branch.complete
    assert branch.origin_label == "9:P0"
    assert report.accepted
    assert report.n_compared == (N_GENERATIONS + 1) * 2
    assert report.max_deviation_m <= report.absolute_tolerance_m

