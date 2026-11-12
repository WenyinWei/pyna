"""Tests for quantitative production manifold heat-load validation."""

from __future__ import annotations

import numpy as np
import pytest

from pyna.topo.manifold_correspondence import ManifoldBranchReference
from pyna.topo.manifold_heat import validate_manifold_heat_load


class _AreaWall:
    def __init__(self, n_phi=8, n_pol=16):
        self.phi_values = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
        theta = np.linspace(0.0, 2.0 * np.pi, n_pol, endpoint=False)
        self.R = np.broadcast_to(
            2.0 + 0.4 * np.cos(theta),
            (n_phi, n_pol),
        ).copy()
        self.Z = np.broadcast_to(
            0.4 * np.sin(theta),
            (n_phi, n_pol),
        ).copy()
        self.field_period = 2.0 * np.pi

    def cell_areas(self, phi_edges, s_edges):
        return np.full((len(phi_edges) - 1, len(s_edges) - 1), 0.5)


def _branch():
    anchor = np.array([2.0, 0.0])
    distances = np.array([0.01, 0.02])
    return ManifoldBranchReference(
        origin_label="heat:X0",
        stability="unstable",
        seed_side=1,
        section_phi=0.0,
        map_span=np.pi,
        anchor_RZ_m=anchor,
        direction_RZ=np.array([1.0, 0.0]),
        expansion=1.2,
        orientation_reversing=False,
        seed_orders=np.array([4, 9]),
        seed_distances_m=distances,
        points_RZ_m=np.array(
            [[[anchor[0] + distance, anchor[1]] for distance in distances]]
        ),
        available=np.ones((1, 2), dtype=bool),
    )


def _fake_trace(*, term=(1, 1), theta=(0.0, 0.5 * np.pi)):
    terms = np.asarray(term, dtype=int)
    hit_theta = np.asarray(theta, dtype=float)

    def trace(
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
        del (
            field,
            Z,
            phi_start,
            max_turns,
            DPhi,
            wall_phi,
            wall_R,
            wall_Z,
            extend_phi,
        )
        assert direction == "+"
        assert len(R) == 2
        hit_phi = np.array([0.25 * np.pi, 0.75 * np.pi])
        hits = np.column_stack(
            (
                2.0 + 0.4 * np.cos(hit_theta),
                0.4 * np.sin(hit_theta),
                hit_phi,
            )
        )
        return {
            "Lc_plus": np.array([4.0, 6.0]),
            "hit_plus": hits,
            "term_plus": terms,
        }

    return trace


def _validate(**overrides):
    kwargs = {
        "field": object(),
        "branch_reference": _branch(),
        "wall": _AreaWall(),
        "strike_powers_W": np.array([4.0, 6.0]),
        "power_provenance": "validated flux-tube powers",
        "phi_edges": np.linspace(0.0, 2.0 * np.pi, 9),
        "s_edges": np.linspace(0.0, 1.0, 17),
        "max_turns": 8,
        "DPhi": 0.01,
        "maximum_heat_flux_W_m2": 13.0,
        "maximum_projection_distance_m": 1.0e-10,
        "trace_function": _fake_trace(),
    }
    kwargs.update(overrides)
    return validate_manifold_heat_load(**kwargs)


def test_production_heat_gate_preserves_labels_power_and_wall_planes():
    report = _validate()

    assert report.accepted
    assert report.rejection_reason is None
    assert report.all_labels_resolved
    assert report.seed_bundle.weight_kind == "power"
    assert report.seed_bundle.metadata["quantitative"] is True
    assert report.seed_bundle.metadata["weight_provenance"] == report.power_provenance
    np.testing.assert_array_equal(report.seed_orders, [4, 9])
    assert tuple(match.label.seed_order for match in report.strike_matches) == (4, 9)
    assert all(
        match.label == plane.strike_label
        for match, plane in zip(report.strike_matches, report.wall_planes, strict=True)
    )
    assert report.unresolved_power_W == pytest.approx(0.0)
    assert report.peak_heat_flux_W_m2 == pytest.approx(12.0)
    assert float(
        np.sum(report.heat_state.heat * report.heat_state.cell_areas)
    ) == pytest.approx(10.0)


def test_production_heat_gate_rejects_peak_and_unresolved_labels():
    peak = _validate(maximum_heat_flux_W_m2=10.0)
    assert not peak.accepted
    assert peak.rejection_reason == "production_peak_heat_flux_exceeds_limit"
    assert len(peak.strike_matches) == 2

    unresolved = _validate(trace_function=_fake_trace(term=(1, 2)))
    assert not unresolved.accepted
    assert unresolved.rejection_reason == "production_unresolved_power_exceeds_limit"
    assert not unresolved.all_labels_resolved
    assert unresolved.unresolved_power_W == pytest.approx(6.0)
    assert len(unresolved.strike_matches) == 0


def test_production_heat_gate_refreshes_exact_labels_under_hit_trust_limit():
    initial = _validate()
    refreshed = _validate(
        previous_matches=initial.strike_matches,
        maximum_hit_displacement_m=1.0e-10,
    )
    assert refreshed.accepted
    assert tuple(match.label for match in refreshed.strike_matches) == tuple(
        match.label for match in initial.strike_matches
    )

    moved = _validate(
        previous_matches=initial.strike_matches,
        maximum_hit_displacement_m=0.01,
        trace_function=_fake_trace(theta=(0.125 * np.pi, 0.625 * np.pi)),
    )
    assert not moved.accepted
    assert moved.rejection_reason == (
        "production_strike_refresh_rejected:"
        "strike_displacement_exceeds_limit"
    )


def test_production_heat_gate_requires_quantitative_provenance_and_trust_pair():
    with pytest.raises(ValueError, match="power_provenance"):
        _validate(power_provenance="")
    with pytest.raises(ValueError, match="one value per branch seed"):
        _validate(strike_powers_W=np.ones(3))
    with pytest.raises(ValueError, match="requires previous_matches"):
        _validate(maximum_hit_displacement_m=0.1)
    initial = _validate()
    with pytest.raises(ValueError, match="required with previous_matches"):
        _validate(previous_matches=initial.strike_matches)
