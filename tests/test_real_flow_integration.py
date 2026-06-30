"""Integration test: new Cycle/Tube/TubeChain/IslandChain class hierarchy.

Runs in two modes:
1. Synthetic data verifies the topology interface.
2. A public analytic stellarator computes fixed points by integrating field lines
   and promotes the resulting O/X points into the workflow geometry layer.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyna.dynamics import CallableMap
from pyna.topo import TopologyWorkflow
from pyna.topo.core import IslandChain as CoreIslandChain
from pyna.topo.fixed_points import classify_fixed_point, find_periodic_orbit, poincare_map
from pyna.topo.toroidal import Cycle, FixedPoint, MonodromyData
from pyna.topo.toroidal import Island, IslandChain
from pyna.topo.toroidal import Tube, TubeChain
from pyna.toroidal.equilibrium.stellarator import simple_stellarator

# ---------------------------------------------------------------------------
# Helper: build synthetic period-10 island chain (m=10, n=3)
# ---------------------------------------------------------------------------

def _build_synthetic_tc():
    """Return (TubeChain, o_cycle, x_cycle) for synthetic m=10/n=3 data."""
    DPm_O = np.array([[np.cos(2*np.pi*0.3), -np.sin(2*np.pi*0.3)],
                      [np.sin(2*np.pi*0.3),  np.cos(2*np.pi*0.3)]])
    DPm_X = np.array([[np.exp(0.15), 0.0], [0.0, np.exp(-0.15)]])

    # 10 O-points and 10 X-points at phi=0
    o_fps = [
        FixedPoint(phi=0.0,
                   R=1.0 + 0.02 * np.cos(2 * np.pi * k / 10),
                   Z=0.05 * np.sin(2 * np.pi * k / 10),
                   DPm=DPm_O)
        for k in range(10)
    ]
    x_fps = [
        FixedPoint(phi=0.0,
                   R=1.0 + 0.03 * np.cos(2 * np.pi * (k + 0.5) / 10),
                   Z=0.06 * np.sin(2 * np.pi * (k + 0.5) / 10),
                   DPm=DPm_X)
        for k in range(10)
    ]

    o_cycle = Cycle(winding=(10, 3), sections={0.0: o_fps})
    x_cycle = Cycle(
        winding=(10, 3),
        sections={0.0: x_fps},
        monodromy=MonodromyData(DPm=DPm_X, eigenvalues=np.linalg.eigvals(DPm_X)),
    )

    tube = Tube(O_cycle=o_cycle, X_cycles=[x_cycle])
    tc = TubeChain(tubes=[tube])
    return tc, o_cycle, x_cycle


# ===========================================================================
# Part A: Synthetic-data tests (always run)
# ===========================================================================

class TestSyntheticFlow:
    """Verify the new class hierarchy with synthetic data (no field cache needed)."""

    def test_section_cut_returns_island_chain(self):
        tc, _, _ = _build_synthetic_tc()
        chain = tc.section_cut(0.0)
        assert isinstance(chain, IslandChain)

    def test_island_count(self):
        tc, _, _ = _build_synthetic_tc()
        chain = tc.section_cut(0.0)
        assert len(chain.islands) == 10, (
            f"Expected 10 islands, got {len(chain.islands)}"
        )

    def test_step_ring(self):
        tc, _, _ = _build_synthetic_tc()
        chain = tc.section_cut(0.0)
        # Forward ring: step() 10 times should return to the start
        start = chain.islands[0]
        current = start
        for _ in range(10):
            current = current.step()
        assert current is start, "step() ring did not close after 10 steps"

    def test_step_forward_sequential(self):
        tc, _, _ = _build_synthetic_tc()
        chain = tc.section_cut(0.0)
        assert chain.islands[0].step() is chain.islands[1]
        assert chain.islands[9].step() is chain.islands[0]

    def test_step_back_ring(self):
        tc, _, _ = _build_synthetic_tc()
        chain = tc.section_cut(0.0)
        start = chain.islands[0]
        current = start
        for _ in range(10):
            current = current.step_back()
        assert current is start, "step_back() ring did not close after 10 steps"

    def test_o_points_in_chain(self):
        tc, _, _ = _build_synthetic_tc()
        chain = tc.section_cut(0.0)
        assert len(chain.O_points) == 10

    def test_x_points_in_chain(self):
        tc, _, _ = _build_synthetic_tc()
        chain = tc.section_cut(0.0)
        # Each island has 1 X-point; 10 total
        assert len(chain.X_points) == 10

    def test_summary(self):
        tc, _, _ = _build_synthetic_tc()
        s = tc.summary()
        assert "TubeChain" in s

    def test_island_o_point_type(self):
        tc, _, _ = _build_synthetic_tc()
        chain = tc.section_cut(0.0)
        for isl in chain.islands:
            assert isinstance(isl.O_point, FixedPoint)

    def test_island_x_points_type(self):
        tc, _, _ = _build_synthetic_tc()
        chain = tc.section_cut(0.0)
        for isl in chain.islands:
            for xfp in isl.X_points:
                assert isinstance(xfp, FixedPoint)

    def test_tube_section_cut_direct(self):
        """Tube.section_cut() itself returns m Islands with ring links."""
        tc, o_cycle, x_cycle = _build_synthetic_tc()
        tube = tc.tubes[0]
        islands = tube.section_cut(0.0).islands
        assert len(islands) == 10
        assert islands[0].step() is islands[1]
        assert islands[9].step() is islands[0]

    def test_monodromy_stability(self):
        from pyna.topo.invariants import Stability
        _, _, x_cycle = _build_synthetic_tc()
        assert x_cycle.stability == Stability.HYPERBOLIC

    def test_monodromy_stability_o(self):
        from pyna.topo.invariants import Stability
        _, o_cycle, _ = _build_synthetic_tc()
        assert o_cycle.stability == Stability.ELLIPTIC

    def test_section_points_lookup(self):
        _, o_cycle, _ = _build_synthetic_tc()
        pts = o_cycle.section_points(0.0)
        assert len(pts) == 10

    def test_section_points_missing(self):
        _, o_cycle, _ = _build_synthetic_tc()
        pts = o_cycle.section_points(99.9)
        assert pts == []


# ===========================================================================
# Part B: Public analytic stellarator computation
# ===========================================================================

def _stellarator_field_func_2d(st):
    """Wrap StellaratorSimple.field_func into d(R,Z)/dphi."""

    def ff2d(R: float, Z: float, phi: float) -> np.ndarray:
        tangent = np.asarray(st.field_func(np.array([R, Z, phi])), dtype=float)
        dphi_ds = tangent[2]
        if abs(dphi_ds) < 1e-15:
            return np.zeros(2)
        return np.array([tangent[0] / dphi_ds, tangent[1] / dphi_ds])

    return ff2d


@pytest.fixture(scope="module")
def analytic_stellarator_fixed_points():
    st = simple_stellarator(
        R0=3.0,
        r0=0.3,
        B0=1.0,
        q0=1.1,
        q1=5.0,
        m_h=4,
        n_h=4,
        epsilon_h=0.05,
    )
    ff2d = _stellarator_field_func_2d(st)
    psi_res = st.resonant_psi(5, 4)[0]
    r_res = st.r0 * np.sqrt(psi_res)
    seed = np.array([st.R0 + r_res, 0.0])
    fps = find_periodic_orbit(
        ff2d,
        seed=seed,
        n_turns=4,
        r_scan=r_res * 0.6,
        n_scan=300,
        verbose=False,
    )
    records = []
    for fp in fps:
        kind, jac, det_jac = classify_fixed_point(fp, ff2d, n_turns=4)
        residual = float(np.linalg.norm(poincare_map(fp, ff2d, n_turns=4) - fp))
        records.append(
            {
                "kind": kind,
                "point": np.asarray(fp, dtype=float),
                "jacobian": jac,
                "det": det_jac,
                "residual": residual,
            }
        )
    return ff2d, records


class TestAnalyticStellaratorWorkflow:
    """Verify workflow geometry against an integrated analytic stellarator map."""

    def test_finds_closed_o_and_x_points(self, analytic_stellarator_fixed_points):
        _, records = analytic_stellarator_fixed_points
        kinds = [record["kind"] for record in records]

        assert len(records) >= 2
        assert "O" in kinds
        assert "X" in kinds
        assert max(record["residual"] for record in records) < 1e-8
        for record in records:
            assert abs(record["det"] - 1.0) < 0.02

    def test_workflow_promotes_integrated_fixed_points_to_island_chain(
        self, analytic_stellarator_fixed_points
    ):
        ff2d, records = analytic_stellarator_fixed_points
        o_record = next(record for record in records if record["kind"] == "O")
        x_record = next(record for record in records if record["kind"] == "X")

        return_map = CallableMap(
            lambda x: poincare_map(x, ff2d, n_turns=4),
            dim=2,
            coordinate_names=("R", "Z"),
            label="analytic stellarator P^4",
        )
        wf = TopologyWorkflow(closure_tol=1e-7)
        o_orbit = wf.periodic_orbit(
            [o_record["point"]],
            map_obj=return_map,
            stability_data=o_record["jacobian"],
            coordinate_names=("R", "Z"),
            metadata={"kind": o_record["kind"]},
        )
        x_orbit = wf.periodic_orbit(
            [x_record["point"]],
            map_obj=return_map,
            stability_data=x_record["jacobian"],
            coordinate_names=("R", "Z"),
            metadata={"kind": x_record["kind"]},
        )

        chain = wf.island_chain(
            [o_orbit],
            [x_orbit],
            proximity_tol=1.0,
            label="q=5/4 analytic stellarator",
        )

        assert isinstance(chain, CoreIslandChain)
        assert chain.n_islands == 1
        assert len(chain.O_points) == 1
        assert len(chain.X_points) == 1
        assert o_orbit.stability_data.classification.name == "ELLIPTIC"
        assert x_orbit.stability_data.classification.name == "HYPERBOLIC"
