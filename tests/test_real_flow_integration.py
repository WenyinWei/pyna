"""Integration test: new Cycle/Tube/TubeChain/IslandChain class hierarchy.

Runs in two modes:
1. Synthetic data (always runs) — verifies the interface without any real field.
2. Real field cache (skipped if D:\\haodata\\data\\bluestar_starting_config_field_cache_200x190x128.pkl
   does not exist) — verifies that the new hierarchy wires correctly with actual physics data.
"""

from __future__ import annotations

import os
import numpy as np
import pytest

from pyna.topo.toroidal_invariants import Cycle, FixedPoint, MonodromyData
from pyna.topo.toroidal_island import Island, IslandChain
from pyna.topo.toroidal_tube import Tube, TubeChain

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIELD_CACHE_PATH = r"D:\haodata\data\bluestar_starting_config_field_cache_200x190x128.pkl"
FP_PKL_PATH = r"D:\haodata\fixed_points_all_sections.pkl"

FIELD_CACHE_EXISTS = os.path.isfile(FIELD_CACHE_PATH)

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
    tc = TubeChain(O_cycles=[o_cycle], X_cycles=[x_cycle], tubes=[tube])
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
        islands = tube.section_cut(0.0)
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
# Part B: Real field-cache tests (skipped if cache absent)
# ===========================================================================

@pytest.mark.skipif(not FIELD_CACHE_EXISTS, reason=f"Field cache not found at {FIELD_CACHE_PATH}")
class TestRealFlowIntegration:
    """Integration test using the real bluestar field cache + fixed-point seeds."""

    @pytest.fixture(scope="class")
    def field_cache(self):
        import pickle
        with open(FIELD_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    @pytest.fixture(scope="class")
    def fixed_points_pkl(self):
        if not os.path.isfile(FP_PKL_PATH):
            pytest.skip(f"Fixed-point seed file not found: {FP_PKL_PATH}")
        import pickle
        with open(FP_PKL_PATH, "rb") as f:
            return pickle.load(f)

    @pytest.fixture(scope="class")
    def real_tc(self, field_cache, fixed_points_pkl):
        """Build TubeChain from real fixed-point data.

        The pkl has the structure:
            {phi: {'xpts': [(R, Z, DPm), ...], 'opts': [(R, Z, DPm), ...]}}
        We use all sections available, wrapping them into Cycle sections dicts.
        """
        pkl = fixed_points_pkl

        o_fps_by_phi: dict = {}
        x_fps_by_phi: dict = {}

        if isinstance(pkl, dict):
            for phi_key, val in pkl.items():
                phi = float(phi_key)
                if isinstance(val, dict):
                    for pt in val.get('opts', []):
                        R, Z = float(pt[0]), float(pt[1])
                        DPm = np.asarray(pt[2], dtype=float) if len(pt) > 2 else np.eye(2)
                        o_fps_by_phi.setdefault(phi, []).append(
                            FixedPoint(phi=phi, R=R, Z=Z, DPm=DPm)
                        )
                    for pt in val.get('xpts', []):
                        R, Z = float(pt[0]), float(pt[1])
                        DPm = np.asarray(pt[2], dtype=float) if len(pt) > 2 else np.eye(2)
                        x_fps_by_phi.setdefault(phi, []).append(
                            FixedPoint(phi=phi, R=R, Z=Z, DPm=DPm)
                        )

        if not o_fps_by_phi and not x_fps_by_phi:
            pytest.skip("Could not extract fixed points from pkl")

        # Use DPm from first available X-point for chain monodromy
        DPm_X_default = np.array([[np.exp(0.15), 0.0], [0.0, np.exp(-0.15)]])
        first_phi = sorted(x_fps_by_phi.keys())[0] if x_fps_by_phi else None
        if first_phi is not None and x_fps_by_phi[first_phi]:
            DPm_X = x_fps_by_phi[first_phi][0].DPm
        else:
            DPm_X = DPm_X_default

        o_cycle = Cycle(winding=(10, 3), sections=o_fps_by_phi)
        x_cycle = Cycle(
            winding=(10, 3),
            sections=x_fps_by_phi,
            monodromy=MonodromyData(DPm=DPm_X, eigenvalues=np.linalg.eigvals(DPm_X)),
        )

        tube = Tube(O_cycle=o_cycle, X_cycles=[x_cycle])
        tc = TubeChain(O_cycles=[o_cycle], X_cycles=[x_cycle], tubes=[tube])
        return tc

    def test_section_cut_real(self, real_tc):
        chain = real_tc.section_cut(0.0)
        assert isinstance(chain, IslandChain)

    def test_islands_nonempty_real(self, real_tc):
        chain = real_tc.section_cut(0.0)
        assert len(chain.islands) > 0, "Expected at least one island from real fixed points"

    def test_step_doesnt_crash_real(self, real_tc):
        chain = real_tc.section_cut(0.0)
        isl = chain.islands[0]
        _ = isl.step()  # must not raise

    def test_summary_real(self, real_tc):
        s = real_tc.summary()
        assert "TubeChain" in s
        print("\nReal TubeChain summary:", s)

    def test_print_chain_info_real(self, real_tc):
        chain = real_tc.section_cut(0.0)
        print(f"\nReal IslandChain: {len(chain.islands)} islands, "
              f"{len(chain.O_points)} O-points, {len(chain.X_points)} X-points")
        for i, isl in enumerate(chain.islands):
            print(f"  Island[{i}]: O=({isl.O_point.R:.4f}, {isl.O_point.Z:.4f}), "
                  f"X_points={len(isl.X_points)}")
