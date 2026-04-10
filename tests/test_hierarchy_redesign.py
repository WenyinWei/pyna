"""Tests for the new invariant-object hierarchy (Phase 1 redesign).

Tests for:
  - InvariantSet (base class, no longer abstract)
  - InvariantManifold (intermediate with intrinsic_dim)
  - SectionCuttable protocol
  - FixedPoint.coords generalization
  - MonodromyData.spectral_regularity()
  - regularity.py module functions
"""
from __future__ import annotations

import numpy as np
import pytest

from pyna.topo._base import (
    InvariantSet,
    InvariantManifold,
    SectionCuttable,
    InvariantObject,       # backward-compat alias
)
from pyna.topo.invariants import (
    FixedPoint,
    Cycle,
    MonodromyData,
    PeriodicOrbit,
    Stability,
    InvariantTorus,
    Island,
    IslandChain,
    StableManifold,
    UnstableManifold,
)
from pyna.topo.regularity import (
    spectral_regularity,
    spectral_regularity_single,
    classify_orbit,
    hessian_regularity,
    eigenvalue_evolution_from_sequence,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rotation_matrix(theta: float) -> np.ndarray:
    """2×2 rotation matrix."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _hyperbolic_matrix(lam: float = 3.0) -> np.ndarray:
    """2×2 hyperbolic (area-preserving) matrix."""
    return np.array([[lam, 0.0], [0.0, 1.0 / lam]])


# ═══════════════════════════════════════════════════════════════════════════════
# 1. InvariantSet / InvariantManifold / SectionCuttable
# ═══════════════════════════════════════════════════════════════════════════════

class TestInvariantSet:
    def test_instantiable(self):
        """InvariantSet is no longer abstract; can be instantiated."""
        s = InvariantSet()
        assert s.label is None
        assert s.ambient_dim is None

    def test_diagnostics_default(self):
        s = InvariantSet()
        d = s.diagnostics()
        assert d["invariant_type"] == "InvariantSet"

    def test_section_cut_raises(self):
        s = InvariantSet()
        with pytest.raises(NotImplementedError):
            s.section_cut()

    def test_backward_compat_alias(self):
        """InvariantObject is InvariantSet."""
        assert InvariantObject is InvariantSet


class TestInvariantManifold:
    def test_instantiable(self):
        m = InvariantManifold()
        assert m.intrinsic_dim is None
        assert m.codim is None

    def test_is_subclass(self):
        assert issubclass(InvariantManifold, InvariantSet)

    def test_codim_computation(self):
        """codim = ambient_dim - intrinsic_dim when both known."""

        class MockManifold(InvariantManifold):
            @property
            def ambient_dim(self):
                return 6

            @property
            def intrinsic_dim(self):
                return 2

        m = MockManifold()
        assert m.codim == 4


class TestSectionCuttable:
    def test_fixedpoint_is_section_cuttable(self):
        fp = FixedPoint(phi=0.0, R=1.5, Z=0.0, DPm=_rotation_matrix(0.4))
        assert isinstance(fp, SectionCuttable)

    def test_plain_invariant_set_is_not_section_cuttable(self):
        """A bare InvariantSet doesn't satisfy SectionCuttable (raises)."""
        s = InvariantSet()
        # SectionCuttable is a runtime_checkable Protocol.
        # InvariantSet has section_cut method signature but it raises.
        # The Protocol check is structural: it matches if the method exists.
        assert isinstance(s, SectionCuttable)  # structurally yes
        with pytest.raises(NotImplementedError):
            s.section_cut()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FixedPoint.coords generalization
# ═══════════════════════════════════════════════════════════════════════════════

class TestFixedPointCoords:
    def test_coords_from_RZ(self):
        """When coords=None, auto-built from (R, Z)."""
        fp = FixedPoint(phi=0.0, R=1.5, Z=-0.2, DPm=_rotation_matrix(0.4))
        np.testing.assert_allclose(fp.coords, [1.5, -0.2])

    def test_coords_explicit(self):
        """Explicit coords overrides R/Z."""
        fp = FixedPoint(DPm=np.eye(4), coords=np.array([1.0, 2.0, 3.0, 4.0]))
        assert len(fp.coords) == 4
        assert fp[2] == 3.0  # array-like indexing works for higher dims
        assert len(fp) == 4

    def test_coords_backfills_RZ(self):
        """When coords given, R/Z are back-filled from coords[:2]."""
        fp = FixedPoint(DPm=np.eye(2), coords=np.array([1.6, -0.1]))
        assert fp.R == pytest.approx(1.6)
        assert fp.Z == pytest.approx(-0.1)

    def test_section_angle_sync(self):
        """section_angle mirrors phi."""
        fp = FixedPoint(phi=0.5, R=1.0, Z=0.0, DPm=np.eye(2))
        assert fp.section_angle == pytest.approx(0.5)

    def test_coordinate_names(self):
        fp = FixedPoint(
            DPm=np.eye(3),
            coords=np.array([1.0, 2.0, 3.0]),
            coordinate_names=('q1', 'q2', 'q3'),
        )
        assert fp.coordinate_names == ('q1', 'q2', 'q3')

    def test_intrinsic_dim(self):
        """FixedPoint always has intrinsic_dim = 0."""
        fp = FixedPoint(phi=0.0, R=1.5, Z=0.0, DPm=np.eye(2))
        assert fp.intrinsic_dim == 0

    def test_is_invariant_manifold(self):
        """FixedPoint is an InvariantManifold."""
        fp = FixedPoint(phi=0.0, R=1.5, Z=0.0, DPm=np.eye(2))
        assert isinstance(fp, InvariantManifold)
        assert isinstance(fp, InvariantSet)

    def test_np_asarray(self):
        """np.asarray(fp) returns coords."""
        fp = FixedPoint(DPm=np.eye(3), coords=np.array([1.0, 2.0, 3.0]))
        arr = np.asarray(fp)
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Cycle intrinsic_dim
# ═══════════════════════════════════════════════════════════════════════════════

class TestCycleIntrinsicDim:
    def test_cycle_is_manifold(self):
        fp = FixedPoint(phi=0.0, R=1.0, Z=0.0, DPm=np.eye(2))
        c = Cycle(winding=(3, 1), sections={0.0: [fp]})
        assert isinstance(c, InvariantManifold)
        assert c.intrinsic_dim == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 4. InvariantTorus intrinsic_dim
# ═══════════════════════════════════════════════════════════════════════════════

class TestInvariantTorusIntrinsicDim:
    def test_1d_torus(self):
        t = InvariantTorus(rotation_vector=(0.35,), ambient_dim=2)
        assert t.intrinsic_dim == 1
        assert t.codim == 1

    def test_2d_torus(self):
        t = InvariantTorus(rotation_vector=(0.35, 0.25), ambient_dim=4)
        assert t.intrinsic_dim == 2
        assert t.codim == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 5. StableManifold / UnstableManifold intrinsic_dim
# ═══════════════════════════════════════════════════════════════════════════════

class TestManifoldIntrinsicDim:
    def test_stable_manifold_dim(self):
        DPm = _hyperbolic_matrix(3.0)
        mono = MonodromyData(DPm=DPm, eigenvalues=np.linalg.eigvals(DPm))
        fp = FixedPoint(phi=0.0, R=1.0, Z=0.0, DPm=DPm)
        cycle = Cycle(winding=(1, 0), sections={0.0: [fp]}, monodromy=mono)
        sm = StableManifold(cycle=cycle)
        # One eigenvalue 1/3 < 1 → stable dim = 1
        assert sm.intrinsic_dim == 1
        assert isinstance(sm, InvariantManifold)

    def test_unstable_manifold_dim(self):
        DPm = _hyperbolic_matrix(3.0)
        mono = MonodromyData(DPm=DPm, eigenvalues=np.linalg.eigvals(DPm))
        fp = FixedPoint(phi=0.0, R=1.0, Z=0.0, DPm=DPm)
        cycle = Cycle(winding=(1, 0), sections={0.0: [fp]}, monodromy=mono)
        um = UnstableManifold(cycle=cycle)
        # One eigenvalue 3 > 1 → unstable dim = 1
        assert um.intrinsic_dim == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MonodromyData.spectral_regularity()
# ═══════════════════════════════════════════════════════════════════════════════

class TestMonodromySpectralRegularity:
    def test_elliptic_single(self):
        """Elliptic orbit (eigenvalues on unit circle) → regularity ≈ 0."""
        DPm = _rotation_matrix(0.4)
        mono = MonodromyData(DPm=DPm, eigenvalues=np.linalg.eigvals(DPm))
        reg = mono.spectral_regularity()
        assert reg < 1e-10

    def test_hyperbolic_single(self):
        """Hyperbolic orbit → regularity > 0."""
        DPm = _hyperbolic_matrix(3.0)
        mono = MonodromyData(DPm=DPm, eigenvalues=np.linalg.eigvals(DPm))
        reg = mono.spectral_regularity()
        assert reg > 0.5  # log(3) ≈ 1.1

    def test_with_sequence(self):
        """With a full DPk sequence, regularity is averaged."""
        theta = 0.4
        # A regular orbit: all intermediate matrices are rotations
        DPk_seq = [_rotation_matrix(theta * k / 5) for k in range(1, 6)]
        DPm = _rotation_matrix(theta)
        mono = MonodromyData(DPm=DPm, eigenvalues=np.linalg.eigvals(DPm))
        reg = mono.spectral_regularity(DPk_seq)
        assert reg < 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# 7. regularity.py standalone functions
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegularityModule:
    def test_spectral_regularity_regular(self):
        """Sequence of rotation matrices → near-zero regularity."""
        theta = 0.3
        seq = [_rotation_matrix(theta * k / 10) for k in range(1, 11)]
        assert spectral_regularity(seq) < 1e-10

    def test_spectral_regularity_chaotic(self):
        """Growing eigenvalue matrices → large regularity."""
        seq = [_hyperbolic_matrix(1.0 + 0.5 * k) for k in range(1, 6)]
        assert spectral_regularity(seq) > 0.5

    def test_spectral_regularity_single_unit(self):
        eigs = np.array([np.exp(1j * 0.3), np.exp(-1j * 0.3)])
        assert spectral_regularity_single(eigs) < 1e-10

    def test_spectral_regularity_single_divergent(self):
        eigs = np.array([3.0, 1.0 / 3.0])
        assert spectral_regularity_single(eigs) > 1.0

    def test_classify_orbit_regular(self):
        """Eigenvalue moduli all near 1 → regular."""
        ev = np.ones((10, 2)) * 1.0
        assert classify_orbit(ev) == "resonant"  # exactly 1 → resonant

    def test_classify_orbit_chaotic(self):
        """Growing eigenvalue moduli → chaotic."""
        ev = np.array([[1.0 + 0.5 * k, 1.0 / (1.0 + 0.5 * k)] for k in range(10)])
        result = classify_orbit(ev)
        assert result in ("weakly_chaotic", "strongly_chaotic")

    def test_hessian_regularity(self):
        """Hessian regularity is just normalised Frobenius norm."""
        D2 = np.zeros((2, 2, 2))
        assert hessian_regularity(D2) == 0.0

        D2 = np.ones((2, 2, 2))
        assert hessian_regularity(D2) > 0.0

    def test_eigenvalue_evolution(self):
        """eigenvalue_evolution_from_sequence returns correct shape."""
        seq = [_rotation_matrix(0.1 * k) for k in range(1, 6)]
        ev = eigenvalue_evolution_from_sequence(seq)
        assert ev.shape == (5, 2)
        # All rotation matrices have |λ| = 1
        np.testing.assert_allclose(ev, 1.0, atol=1e-10)

    def test_empty_sequence(self):
        assert spectral_regularity([]) == 0.0
        ev = eigenvalue_evolution_from_sequence([])
        assert ev.shape == (0, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Backward compatibility: all existing skeleton classes still work
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackwardCompat:
    def test_island_is_invariant_object(self):
        """Island is still an InvariantObject (= InvariantSet)."""
        fp = FixedPoint(phi=0.0, R=1.0, Z=0.0, DPm=_rotation_matrix(0.4))
        isl = Island(O_orbit=PeriodicOrbit(points=[fp]))
        assert isinstance(isl, InvariantObject)
        assert isinstance(isl, InvariantSet)

    def test_island_chain_is_invariant_object(self):
        chain = IslandChain()
        assert isinstance(chain, InvariantObject)

    def test_fixedpoint_is_invariant_object(self):
        fp = FixedPoint(phi=0.0, R=1.0, Z=0.0, DPm=np.eye(2))
        assert isinstance(fp, InvariantObject)

    def test_existing_fp_api_unchanged(self):
        """R, Z, phi, kind, greene_residue all work."""
        DPm = _rotation_matrix(0.4)
        fp = FixedPoint(phi=0.5, R=1.5, Z=-0.1, DPm=DPm)
        assert fp.R == pytest.approx(1.5)
        assert fp.Z == pytest.approx(-0.1)
        assert fp.phi == pytest.approx(0.5)
        assert fp.kind == 'O'
        assert 0 < fp.greene_residue < 1
        assert fp[0] == pytest.approx(1.5)
        assert fp[1] == pytest.approx(-0.1)
        assert len(fp) == 2
