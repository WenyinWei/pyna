import numpy as np
import pytest

from pyna.toroidal.control.heat_contracts import BoundaryTopologyHeatState
from pyna.toroidal.control.strike_heat import (
    StrikeSeedBundle,
    WallStrikeSamples,
    island_strike_seed_bundles,
    manifold_strike_seed_bundles,
    sum_boundary_heat_states,
    trace_wall_strikes_field,
    wall_heat_state_from_strikes,
)
from pyna.toroidal.geometry import ToroidalWall, project_points_to_toroidal_surface


def _wall(n_phi=8, n_pol=32):
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, n_pol, endpoint=False)
    R = np.broadcast_to(2.0 + 0.4 * np.cos(theta), (n_phi, n_pol)).copy()
    Z = np.broadcast_to(0.4 * np.sin(theta), (n_phi, n_pol)).copy()
    return ToroidalWall(phi, R, Z)


def _manifold_payload(map_span=np.pi):
    return {
        "manifold_origin_label": "X0",
        "origin_phi": 0.25,
        "manifold_field_period": map_span,
        "manifold_field_period_source": "test",
        "chain_id": "chain0",
        "u_seed_R": np.array([1.91, 1.94, 2.06, 2.09]),
        "u_seed_Z": np.array([-0.03, -0.01, 0.01, 0.03]),
        "u_seed_distance": np.array([0.03, 0.01, 0.01, 0.03]),
        "u_seed_side": np.array([-1.0, -1.0, 1.0, 1.0]),
        "u_seed_order": np.array([2, 0, 0, 2]),
        "s_seed_R": np.array([1.97, 1.99, 2.01, 2.03]),
        "s_seed_Z": np.array([-0.09, -0.04, 0.04, 0.09]),
        "s_seed_distance": np.array([0.09, 0.04, 0.04, 0.09]),
        "s_seed_side": np.array([-1.0, -1.0, 1.0, 1.0]),
        "s_seed_order": np.array([3, 1, 1, 3]),
    }


def _power_strikes(scale=1.0):
    theta = np.array([0.35, 2.1])
    return WallStrikeSamples(
        label="strike",
        mode="chaotic_manifold",
        R=2.0 + 0.4 * np.cos(theta),
        Z=0.4 * np.sin(theta),
        phi=np.array([0.4, 4.2]),
        weights=scale * np.array([2.0, 3.0]),
        connection_length=np.array([7.0, 11.0]),
        seed_index=np.array([0, 2]),
        direction="+",
        unresolved_weight=scale * 5.0,
        weight_kind="power",
        metadata={"quantitative": True},
    )


def test_manifold_seed_bundles_preserve_four_branches_order_and_signed_directions():
    bundles = manifold_strike_seed_bundles([_manifold_payload(np.pi)])

    assert [bundle.label for bundle in bundles] == [
        "X0.unstable.minus",
        "X0.unstable.plus",
        "X0.stable.minus",
        "X0.stable.plus",
    ]
    assert [bundle.direction for bundle in bundles] == ["+", "+", "-", "-"]
    np.testing.assert_allclose(bundles[0].R, [1.91, 1.94])
    np.testing.assert_allclose(bundles[0].source_coordinate, [0.03, 0.01])
    np.testing.assert_array_equal(bundles[0].metadata["seed_order"], [2, 0])
    assert bundles[0].metadata["quantitative"] is False

    reversed_bundles = manifold_strike_seed_bundles([_manifold_payload(-np.pi)])
    assert [bundle.direction for bundle in reversed_bundles] == ["-", "-", "+", "+"]


def test_manifold_weight_builder_can_attach_quantitative_power_provenance():
    def weights(context):
        return {
            "weights": 0.5 + np.arange(context["R"].size),
            "weight_kind": "power",
            "quantitative": True,
            "provenance": "flux_tube_power_test",
        }

    bundles = manifold_strike_seed_bundles([_manifold_payload()], weight_builder=weights)

    assert all(bundle.weight_kind == "power" for bundle in bundles)
    assert all(bundle.metadata["quantitative"] for bundle in bundles)
    assert all(bundle.metadata["weight_provenance"] == "flux_tube_power_test" for bundle in bundles)


def test_regular_island_order_and_bidirectional_power_are_preserved():
    contour = {
        "label": "island0",
        "R": np.array([2.1, 2.2, 2.0]),
        "Z": np.array([0.0, 0.1, 0.2]),
        "phi": 0.3,
        "source_coordinate": np.array([0.0, 0.7, 1.9]),
        "weights": np.array([2.0, 4.0, 6.0]),
        "weight_kind": "power",
        "weight_provenance": "validated_flux_tubes",
        "closed": True,
        "dpk_regular": True,
        "quantitative": True,
    }

    bundles = island_strike_seed_bundles([contour])

    assert [bundle.direction for bundle in bundles] == ["+", "-"]
    assert [bundle.label for bundle in bundles] == ["island0.+", "island0.-"]
    for bundle in bundles:
        np.testing.assert_allclose(bundle.R, contour["R"])
        np.testing.assert_allclose(bundle.source_coordinate, contour["source_coordinate"])
        np.testing.assert_allclose(bundle.weights, [1.0, 2.0, 3.0])
        assert bundle.metadata["quantitative"] is True
    assert sum(np.sum(bundle.weights) for bundle in bundles) == pytest.approx(12.0)

    with pytest.raises(ValueError, match="closed=True"):
        island_strike_seed_bundles(
            [{**contour, "closed": False, "quantitative": True}]
        )


def test_unvalidated_regular_island_is_explicit_proxy():
    bundles = island_strike_seed_bundles(
        [{"R": [2.0, 2.1], "Z": [0.0, 0.1], "section_phi": 0.0, "direction": "+"}]
    )

    assert len(bundles) == 1
    assert bundles[0].metadata["quantitative"] is False
    assert "proxy" in bundles[0].metadata["topology_provenance"]
    assert bundles[0].weight_kind == "relative"


def test_continuous_projection_interpolates_phi_and_poloidal_segment():
    wall_phi = np.array([0.0, np.pi])
    base_R = np.array([2.5, 2.0, 1.5, 2.0])
    base_Z = np.array([0.0, 0.5, 0.0, -0.5])
    wall_R = np.stack([base_R, base_R + 0.2])
    wall_Z = np.stack([base_Z, base_Z])

    projection = project_points_to_toroidal_surface(
        [2.35],
        [0.25],
        [0.5 * np.pi],
        wall_phi,
        wall_R,
        wall_Z,
        field_period=2.0 * np.pi,
    )

    assert projection.phi_fraction[0] == pytest.approx(0.5)
    assert projection.segment_index[0] == 0
    assert projection.R[0] == pytest.approx(2.35)
    assert projection.Z[0] == pytest.approx(0.25)
    assert projection.s[0] == pytest.approx(0.125)
    assert projection.distance[0] == pytest.approx(0.0, abs=1.0e-14)
    np.testing.assert_allclose(np.linalg.norm(projection.normal, axis=1), 1.0)
    np.testing.assert_allclose(projection.xyz[0], [0.0, 2.35, 0.25], atol=1.0e-14)


def test_trace_bridge_preserves_seed_weight_mapping_and_unresolved_weight():
    bundle = StrikeSeedBundle(
        label="ordered",
        mode="chaotic_manifold",
        R=np.array([1.9, 2.0, 2.1, 2.2]),
        Z=np.zeros(4),
        phi=np.full(4, 0.2),
        direction="+",
        weights=np.array([10.0, 20.0, 30.0, 40.0]),
        weight_kind="power",
        source_coordinate=np.array([0.4, 0.1, 0.3, 0.2]),
    )
    calls = []

    def fake_trace(
        field,
        R,
        Z,
        phi_start,
        max_turns,
        DPhi,
        wall_phi,
        wall_R,
        wall_Z,
        *,
        extend_phi,
        direction,
    ):
        del field, Z, max_turns, DPhi, wall_phi, wall_R, wall_Z, extend_phi
        calls.append((phi_start, direction, R.copy()))
        return {
            "Lc_plus": np.array([4.0, 5.0, 6.0, 7.0]),
            "hit_plus": np.column_stack([R + 0.1, np.zeros(4), np.full(4, phi_start + 0.2)]),
            "term_plus": np.array([1, 2, 1, 3]),
        }

    strikes = trace_wall_strikes_field(
        object(),
        [bundle],
        _wall(),
        max_turns=12,
        DPhi=0.01,
        trace_function=fake_trace,
    )

    assert len(calls) == 1
    assert calls[0][1] == "+"
    np.testing.assert_array_equal(strikes[0].seed_index, [0, 2])
    np.testing.assert_allclose(strikes[0].weights, [10.0, 30.0])
    np.testing.assert_allclose(strikes[0].connection_length, [4.0, 6.0])
    assert strikes[0].unresolved_weight == pytest.approx(60.0)
    assert strikes[0].launched_weight == pytest.approx(100.0)


def test_power_heat_state_conserves_resolved_and_reports_unresolved_power():
    state = wall_heat_state_from_strikes(
        [_power_strikes()],
        _wall(),
        phi_edges=np.linspace(0.0, 2.0 * np.pi, 9),
        s_edges=np.linspace(0.0, 1.0, 17),
    )

    assert isinstance(state, BoundaryTopologyHeatState)
    assert float(np.sum(state.heat * state.cell_areas)) == pytest.approx(5.0)
    assert state.metadata["launched_power"] == pytest.approx(10.0)
    assert state.metadata["deposited_power"] == pytest.approx(5.0)
    assert state.metadata["unresolved_power"] == pytest.approx(5.0)
    assert state.metadata["resolved_power_fraction"] == pytest.approx(0.5)
    assert state.metadata["projection"] == "continuous_toroidal_section_poloidal_segment"


def test_relative_heat_requires_total_power_and_preserves_unresolved_fraction():
    power = _power_strikes()
    relative = WallStrikeSamples(
        label=power.label,
        mode=power.mode,
        R=power.R,
        Z=power.Z,
        phi=power.phi,
        weights=np.array([1.0, 1.0]),
        connection_length=power.connection_length,
        seed_index=power.seed_index,
        direction=power.direction,
        unresolved_weight=2.0,
        weight_kind="relative",
    )
    kwargs = {
        "phi_edges": np.linspace(0.0, 2.0 * np.pi, 9),
        "s_edges": np.linspace(0.0, 1.0, 17),
    }
    with pytest.raises(ValueError, match="require total_power"):
        wall_heat_state_from_strikes([relative], _wall(), **kwargs)

    state = wall_heat_state_from_strikes([relative], _wall(), total_power=20.0, **kwargs)
    assert float(np.sum(state.heat * state.cell_areas)) == pytest.approx(10.0)
    assert state.metadata["unresolved_power"] == pytest.approx(10.0)


def test_heat_state_sum_uses_flux_and_common_cell_areas():
    kwargs = {
        "wall": _wall(),
        "phi_edges": np.linspace(0.0, 2.0 * np.pi, 9),
        "s_edges": np.linspace(0.0, 1.0, 17),
    }
    first = wall_heat_state_from_strikes([_power_strikes(1.0)], **kwargs)
    second = wall_heat_state_from_strikes([_power_strikes(2.0)], **kwargs)

    summed = sum_boundary_heat_states([first, second])

    np.testing.assert_allclose(summed.heat, first.heat + second.heat)
    assert float(np.sum(summed.heat * summed.cell_areas)) == pytest.approx(15.0)
    assert summed.metadata["launched_power"] == pytest.approx(30.0)
    assert summed.metadata["unresolved_power"] == pytest.approx(15.0)
    assert summed.metadata["component_count"] == 2
