"""Production Cyna refresh of an exact manifold branch identity."""

from __future__ import annotations

import numpy as np
import pytest

from pyna.fields import VectorFieldCylind
from pyna.topo.manifold_correspondence import (
    manifold_branch_reference_from_trace,
)
from pyna.topo.manifold_refresh import (
    refresh_manifold_branch_reference_field,
)
from pyna.topo.toroidal import FixedPoint
from pyna.toroidal.flt import trace_fixed_point_manifolds_field


MAP_SPAN = 0.6
DPHI = 0.01
N_GENERATIONS = 3
SEED_DISTANCES = np.array([1.0e-3, 2.0e-3])
SEED_ORDERS = np.array([3, 7])


def _require_cyna():
    import pyna._cyna as cyna

    if not cyna.is_available() or cyna.VectorFieldCylind is None:
        pytest.skip("cyna VectorFieldCylind is unavailable")


def _field(rate, center_r=1.0, center_z=0.0):
    radius = np.linspace(0.7, 1.5, 49)
    height = np.linspace(-0.4, 0.4, 49)
    phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    rr, zz, _ = np.meshgrid(radius, height, phi, indexing="ij")
    return VectorFieldCylind(
        R=radius,
        Z=height,
        Phi=phi,
        BR=rate * (rr - center_r),
        BZ=-rate * (zz - center_z),
        BPhi=rr,
        nfp=1,
        axisymmetric=True,
    )


def _initial_branch(stability):
    rate = 0.35
    expansion = np.exp(rate * MAP_SPAN)
    fixed_point = FixedPoint(
        phi=0.0,
        R=1.0,
        Z=0.0,
        kind="X",
        DPm=np.diag([expansion, 1.0 / expansion]),
    )
    fixed_point.metadata.update(
        {
            "orbit_id": 12,
            "map_order_index": 0,
            "monodromy_map_span": MAP_SPAN,
        }
    )
    payload = trace_fixed_point_manifolds_field(
        _field(rate),
        [fixed_point],
        phi_section=0.0,
        map_span=MAP_SPAN,
        N_turns=N_GENERATIONS,
        DPhi=DPHI,
        seed_distances=SEED_DISTANCES,
        seed_orders=SEED_ORDERS,
        refine_stable_inverse_anchor=False,
    )[0]
    return manifold_branch_reference_from_trace(
        payload,
        stability=stability,
        seed_side=1,
        n_generations=N_GENERATIONS,
    )


@pytest.mark.parametrize("stability", ["unstable", "stable"])
def test_production_refresh_continues_identity_and_sparse_seed_orders(stability):
    _require_cyna()
    previous = _initial_branch(stability)
    candidate_center = np.array([1.0003, -0.0002])
    report = refresh_manifold_branch_reference_field(
        _field(0.36, *candidate_center),
        previous,
        maximum_anchor_displacement_m=2.0e-3,
        minimum_direction_alignment=0.95,
        DPhi=DPHI,
    )

    assert report.accepted
    assert report.rejection_reason is None
    assert report.candidate_branch is not None
    assert report.candidate_branch.label == previous.label
    assert report.candidate_branch.origin_label == "12:P0"
    np.testing.assert_array_equal(report.candidate_branch.seed_orders, SEED_ORDERS)
    np.testing.assert_allclose(
        report.candidate_branch.anchor_RZ_m,
        candidate_center,
        atol=2.0e-11,
    )
    assert report.anchor_displacement_m < report.maximum_anchor_displacement_m
    assert report.diagnostics["direction_alignment"] >= 0.95


def test_production_refresh_rejects_an_anchor_outside_the_trust_region():
    _require_cyna()
    previous = _initial_branch("unstable")
    report = refresh_manifold_branch_reference_field(
        _field(0.36, 1.01, 0.0),
        previous,
        maximum_anchor_displacement_m=1.0e-3,
        minimum_direction_alignment=0.95,
        DPhi=DPHI,
    )

    assert not report.accepted
    assert report.candidate_branch is None
    assert report.rejection_reason == "anchor_displacement_exceeds_limit"
    assert report.anchor_displacement_m > report.maximum_anchor_displacement_m


def test_production_refresh_rejects_loss_of_hyperbolicity():
    _require_cyna()
    previous = _initial_branch("unstable")
    report = refresh_manifold_branch_reference_field(
        _field(0.0),
        previous,
        maximum_anchor_displacement_m=1.0e-3,
        minimum_direction_alignment=0.95,
        DPhi=DPHI,
    )

    assert not report.accepted
    assert report.rejection_reason in {
        "fixed_point_continuation_failed",
        "fixed_point_lost_hyperbolicity",
    }
