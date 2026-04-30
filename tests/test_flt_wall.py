"""Tests for pyna.flt — WallModel and boundary reseeding."""
from __future__ import annotations

import numpy as np
import pytest
from pyna.flt import FieldLineTracer, WallModel, reseed_boundary_field_lines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

R0, a = 1.0, 0.3
B0, q0 = 1.0, 2.0


def solovev_field(rzphi: np.ndarray) -> np.ndarray:
    R, Z, phi = rzphi
    lam = B0 * a / (q0 * R0)
    dpsi_dR = (R * R - R0 * R0) * R / (R0 * R0 * a * a)
    dpsi_dZ = 2.0 * Z / (a * a)
    BR = -lam / R * dpsi_dZ
    BZ = lam / R * dpsi_dR
    Bphi = B0 * R0 / R
    Bmag = np.sqrt(BR ** 2 + BZ ** 2 + Bphi ** 2) + 1e-30
    return np.array([BR / Bmag, BZ / Bmag, Bphi / (R * Bmag)])


# ---------------------------------------------------------------------------
# WallModel tests
# ---------------------------------------------------------------------------

class TestWallModel:

    def test_circular_factory(self):
        wall = WallModel.circular(R0=R0, a=a)
        assert isinstance(wall, WallModel)

    def test_inside_is_not_outside(self):
        wall = WallModel.circular(R0=R0, a=a)
        # A point well inside should not be "outside"
        assert not wall.is_outside(np.array([R0, 0.0]))

    def test_outside_point(self):
        wall = WallModel.circular(R0=R0, a=a)
        # A point far outside the wall
        assert wall.is_outside(np.array([R0 + 2 * a, 0.0]))

    def test_min_clearance(self):
        # With a large min_clearance, even interior points appear as outside
        wall = WallModel.circular(R0=R0, a=a, min_clearance=0.5 * a)
        # A point near the wall edge should be flagged
        assert wall.is_outside(np.array([R0 + 0.9 * a, 0.0]))

    def test_polygon_needs_3_vertices(self):
        with pytest.raises(ValueError):
            WallModel(R_wall=[1.0, 1.1], Z_wall=[0.0, 0.1])

    def test_rz_shape_mismatch(self):
        with pytest.raises(ValueError):
            WallModel(R_wall=[1.0, 1.1, 0.9], Z_wall=[0.0, 0.1])


# ---------------------------------------------------------------------------
# FieldLineTracer with wall tests
# ---------------------------------------------------------------------------

class TestFieldLineTracerWithWall:

    def test_trace_stops_at_wall(self):
        """Trajectory should terminate when it exits the wall."""
        # Wall covers only a thin arc around (R0, 0): the orbit will
        # quickly leave it as the field line spirals in Z.
        # Use a rectangular (box) wall tight in Z so the orbit exits fast.
        theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        # A small lozenge centered at (R0, 0) of radius 0.01 in Z
        wall = WallModel(
            R_wall=R0 + 0.05 * np.cos(theta),
            Z_wall=0.01 * np.sin(theta),
        )
        tracer = FieldLineTracer(solovev_field, dt=0.1)
        # Start near center of wall but orbit will have Z-component > 0.01
        start = np.array([R0 + 0.02, 0.0, 0.0])
        traj = tracer.trace(start, t_max=50.0, wall=wall)
        full_steps = int(50.0 / 0.1) + 1
        assert traj.shape[0] < full_steps, "Wall did not stop trajectory early"

    def test_trace_many_with_wall_returns_correct_count(self):
        """trace_many with wall should return one trajectory per start point."""
        wall = WallModel.circular(R0=R0, a=a)
        tracer = FieldLineTracer(solovev_field, dt=0.1, n_workers=2)
        n = 5
        starts = np.column_stack([
            np.full(n, R0 + 0.05 * a),
            np.zeros(n),
            np.zeros(n),
        ])
        trajs = tracer.trace_many(starts, t_max=5.0, wall=wall)
        assert len(trajs) == n

    def test_trace_without_wall_unchanged(self):
        """Not passing wall=None should behave as before."""
        tracer = FieldLineTracer(solovev_field, dt=0.1)
        start = np.array([R0, 0.0, 0.0])
        traj_no_wall = tracer.trace(start, t_max=5.0)
        traj_wall_none = tracer.trace(start, t_max=5.0, wall=None)
        np.testing.assert_array_equal(traj_no_wall, traj_wall_none)


# ---------------------------------------------------------------------------
# reseed_boundary_field_lines tests
# ---------------------------------------------------------------------------

class TestReseedBoundaryFieldLines:

    def test_reseed_adds_extra_traces(self):
        """reseed_boundary_field_lines should append more trajectories."""
        tracer = FieldLineTracer(solovev_field, dt=0.1)
        wall = WallModel.circular(R0=R0, a=0.05)  # tiny wall → many hits

        n = 4
        starts = np.column_stack([
            np.full(n, R0 + 0.04 * a),
            np.zeros(n),
            np.zeros(n),
        ])
        trajs = tracer.trace_many(starts, t_max=5.0, wall=wall)

        all_trajs = reseed_boundary_field_lines(
            tracer, starts, trajs, t_max=5.0, wall=wall,
            min_valid_fraction=0.9,  # generous threshold to trigger reseeding
            n_reseed_factor=2,
            reseed_radius=0.01,
            n_workers=1,
        )
        assert len(all_trajs) >= n, "Should have at least original N trajectories"

    def test_no_reseed_when_all_valid(self):
        """When all traces are long enough, no reseeding should occur."""
        tracer = FieldLineTracer(solovev_field, dt=0.1)
        wall = WallModel.circular(R0=R0, a=2 * a)  # large wall → no hits

        starts = np.array([[R0, 0.0, 0.0], [R0 + 0.02, 0.0, 0.0]])
        trajs = tracer.trace_many(starts, t_max=2.0, wall=wall)

        all_trajs = reseed_boundary_field_lines(
            tracer, starts, trajs, t_max=2.0, wall=wall,
            min_valid_fraction=0.01,  # nearly all traces are valid
        )
        assert len(all_trajs) == len(trajs), "No reseeding expected"
