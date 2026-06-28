import numpy as np

from pyna.topo._monodromy_classification import classify_monodromy_2x2, monodromy_kind
from pyna.topo.core import LinearStabilityData, Stability
from pyna.topo.toroidal import FixedPoint
from pyna.topo.toroidal._monodromy import MonodromyData
from pyna.toroidal.flt.island_chain import BoundaryIslandFixedPoint, fixed_points_by_section_payload


def test_trace_greater_than_two_complex_pair_is_unknown_not_x():
    mat = np.array([[1.1, 1.0], [-0.09, 1.1]])

    cls = classify_monodromy_2x2(mat)

    assert cls.trace > 2.0
    assert cls.discriminant < 0.0
    assert cls.kind == "U"
    assert monodromy_kind(mat) == "U"


def test_area_preserving_real_pair_is_x_and_rotation_is_o():
    x_mat = np.diag([1.2, 1.0 / 1.2])
    flip_x_mat = np.diag([-1.2, -1.0 / 1.2])
    theta = 0.3
    o_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    assert classify_monodromy_2x2(x_mat).kind == "X"
    assert np.trace(flip_x_mat) < -2.0
    assert classify_monodromy_2x2(flip_x_mat).kind == "X"
    assert classify_monodromy_2x2(o_mat).kind == "O"


def test_public_fixed_point_and_stability_objects_use_checked_classification():
    bad = np.array([[1.1, 1.0], [-0.09, 1.1]])

    fp = FixedPoint(phi=0.0, R=1.0, Z=0.0, DPm=bad)
    mono = MonodromyData(DPm=bad, eigenvalues=np.linalg.eigvals(bad))
    lin = LinearStabilityData(jacobian=bad)

    assert fp.kind == "U"
    assert mono.stability is Stability.UNKNOWN
    assert lin.classification is Stability.UNKNOWN


def test_boundary_payload_keeps_unknown_out_of_xo_buckets():
    fp = BoundaryIslandFixedPoint(
        phi=0.0,
        R=1.0,
        Z=0.0,
        map_power=3,
        kind="U",
        DPm=np.array([[1.1, 1.0], [-0.09, 1.1]]),
        residual=1e-12,
        eigenvalues=np.array([1.1 + 0.3j, 1.1 - 0.3j]),
        seed_R=1.0,
        seed_Z=0.0,
    )

    payload = fixed_points_by_section_payload([fp], [0.0])

    assert payload[0.0]["xpts"] == []
    assert payload[0.0]["opts"] == []
    assert len(payload[0.0]["unknown"]) == 1
