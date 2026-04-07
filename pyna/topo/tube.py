"""Continuous-time tube and tube-chain abstractions.

These classes provide the continuous-time counterparts of the discrete
Poincaré-map objects ``Island`` / ``IslandChain``.

Conceptual correspondence
-------------------------
- ``Tube`` wraps one continuous-time periodic orbit (typically represented in
  practice by an :class:`~pyna.topo.island_chain.IslandChainOrbit`).
  Cutting the tube by a Poincaré section produces one discrete fixed point;
  for an O-type tube this becomes an ``Island`` centre on that section.
- ``TubeChain`` is a collection of tubes belonging to the same m/n resonance.
  Cutting the full chain by a Poincaré section produces a discrete
  ``IslandChain``.
- ``ResonanceStructure`` bundles the O-type and X-type tube chains of one
  resonance so that section cuts, discrete chain views, and boundary anchors
  can be queried from one continuous-time object.

This layer is useful when the continuous-time connectivity is more robust than
per-section Newton searches: if one section misses an island, the raw 3-D tube
still exists and can be used to recover / debug the missing discrete point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd
from typing import Any, Callable, Dict, List, Optional, Sequence
import warnings

import numpy as np

from pyna.topo.island import Island, IslandChain
from pyna.topo.island_chain import ChainFixedPoint, IslandChainOrbit


@dataclass
class TubeCutPoint:
    """One section-cut point produced by slicing a ``Tube``.

    Attributes
    ----------
    tube_index : int
        Index of the parent tube inside its ``TubeChain``.
    phi : float
        Section angle [rad].
    R, Z : float
        Section-cut coordinates.
    kind : str or None
        O / X classification when known.
    fixed_point : ChainFixedPoint or None
        The discrete fixed-point object if this point comes from an actual
        section cut / refinement.  May be ``None`` for lightweight reconstructed
        points supplied only as coordinates.
    source : str
        ``'exact-cut'`` | ``'reconstructed-missing'`` | ``'reconstructed-duplicate'`` | ...
    raw_center : tuple or None
        Nearest raw continuous-time orbit point used as the local reconstruction/debug
        centre.
    """

    tube_index: int
    phi: float
    R: float
    Z: float
    kind: Optional[str] = None
    fixed_point: Optional[ChainFixedPoint] = None
    source: str = "exact-cut"
    raw_center: Optional[tuple[float, float]] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def greene_residue(self) -> float:
        if self.fixed_point is None:
            return float("nan")
        return float(self.fixed_point.greene_residue)

    def as_array(self) -> np.ndarray:
        return np.array([self.R, self.Z], dtype=float)


@dataclass
class SectionCut:
    """Structured section cut of a ``TubeChain`` at one Poincaré angle."""

    phi: float
    cut_points: List[TubeCutPoint]
    expected_tube_count: Optional[int] = None
    missing_tube_indices: List[int] = field(default_factory=list)
    duplicate_groups: List[List[int]] = field(default_factory=list)
    reconstructed_tube_indices: List[int] = field(default_factory=list)
    debug_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_cut_points(self) -> int:
        return len(self.cut_points)

    @property
    def unique_tube_indices(self) -> List[int]:
        return sorted({cp.tube_index for cp in self.cut_points})

    def fixed_points(self) -> List[ChainFixedPoint]:
        return [cp.fixed_point for cp in self.cut_points if cp.fixed_point is not None]

    def unique_points(self, dedup_tol: float = 1e-6) -> List[TubeCutPoint]:
        out: List[TubeCutPoint] = []
        for cp in self.cut_points:
            if not any(np.hypot(cp.R - keep.R, cp.Z - keep.Z) < dedup_tol for keep in out):
                out.append(cp)
        return out

    def diagnostics(self) -> Dict[str, Any]:
        return {
            'phi': float(self.phi),
            'n_cut_points': int(self.n_cut_points),
            'expected_tube_count': (None if self.expected_tube_count is None
                                    else int(self.expected_tube_count)),
            'missing_tube_indices': list(self.missing_tube_indices),
            'duplicate_groups': [list(g) for g in self.duplicate_groups],
            'reconstructed_tube_indices': list(self.reconstructed_tube_indices),
            'sources': [cp.source for cp in self.cut_points],
            'tube_indices': [int(cp.tube_index) for cp in self.cut_points],
        }

    def is_complete(self) -> bool:
        if self.missing_tube_indices:
            return False
        if self.duplicate_groups:
            return False
        if self.expected_tube_count is not None and len(self.unique_tube_indices) != self.expected_tube_count:
            return False
        return True

    def summary(self) -> str:
        diag = self.diagnostics()
        return (
            f"SectionCut(phi={self.phi:.6f}) n={diag['n_cut_points']} "
            f"expected={diag['expected_tube_count']} missing={diag['missing_tube_indices']} "
            f"duplicates={diag['duplicate_groups']} reconstructed={diag['reconstructed_tube_indices']}"
        )


SectionReconstructor = Callable[[float, "Tube", Sequence[TubeCutPoint], str], Any]


@dataclass
class Tube:
    """Continuous-time counterpart of one discrete island / fixed-point family."""

    orbit: IslandChainOrbit
    label: Optional[str] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_orbit(cls, orbit: IslandChainOrbit, label: Optional[str] = None) -> "Tube":
        return cls(orbit=orbit, label=label)

    @property
    def m(self) -> int:
        return self.orbit.m

    @property
    def n(self) -> int:
        return self.orbit.n

    @property
    def Np(self) -> int:
        return self.orbit.Np

    @property
    def seed_phi(self) -> float:
        return self.orbit.seed_phi

    @property
    def seed_RZ(self) -> tuple:
        return self.orbit.seed_RZ

    @property
    def section_phis(self) -> Optional[List[float]]:
        return self.orbit.section_phis

    @property
    def kind(self) -> Optional[str]:
        diag = self.orbit.diagnostics(self.section_phis)
        if diag['mixed_kind']:
            return None
        return diag['dominant_kind']

    def at_section(self, phi: float, tol: float = 1e-6) -> List[ChainFixedPoint]:
        return self.orbit.fixed_points_at_section(phi, tol=tol)

    def diagnostics(self, requested_phis: Optional[Sequence[float]] = None) -> Dict[str, Any]:
        diag = self.orbit.diagnostics(requested_phis=requested_phis)
        diag['label'] = self.label
        diag['tube_kind'] = self.kind
        return diag

    def summary(self) -> str:
        kind = self.kind or 'mixed'
        base = self.orbit.summary()
        return f"Tube(kind={kind}, label={self.label})\n{base}"

    def orbit_xyz(self) -> Optional[np.ndarray]:
        return self.orbit.orbit_xyz()

    def raw_point_near_section(self, phi: float) -> Optional[tuple[float, float]]:
        """Nearest raw propagated orbit point to the requested section angle."""
        if not self.orbit.orbit_debug_available:
            return None
        phi_arr = np.asarray(self.orbit.orbit_phi, dtype=float)
        R_arr = np.asarray(self.orbit.orbit_R, dtype=float)
        Z_arr = np.asarray(self.orbit.orbit_Z, dtype=float)
        mask = np.isfinite(phi_arr) & np.isfinite(R_arr) & np.isfinite(Z_arr)
        if self.orbit.orbit_alive is not None:
            mask &= np.asarray(self.orbit.orbit_alive, dtype=bool)
        if not np.any(mask):
            return None
        phi_use = phi_arr[mask]
        R_use = R_arr[mask]
        Z_use = Z_arr[mask]
        idx = int(np.argmin(np.abs(phi_use - float(phi))))
        return float(R_use[idx]), float(Z_use[idx])

    def section_cut_point(
        self,
        phi: float,
        *,
        tube_index: int = 0,
        tol: float = 1e-6,
    ) -> List[TubeCutPoint]:
        fps = self.at_section(phi, tol=tol)
        raw_center = self.raw_point_near_section(phi)
        return [
            TubeCutPoint(
                tube_index=int(tube_index),
                phi=float(phi),
                R=float(fp.R),
                Z=float(fp.Z),
                kind=fp.kind,
                fixed_point=fp,
                source="exact-cut",
                raw_center=raw_center,
            )
            for fp in fps
        ]

    def _coerce_reconstructed_candidate(
        self,
        candidate: Any,
        *,
        phi: float,
        tube_index: int,
        source: str,
    ) -> Optional[TubeCutPoint]:
        raw_center = self.raw_point_near_section(phi)
        if candidate is None:
            return None
        if isinstance(candidate, TubeCutPoint):
            cp = candidate
            cp.tube_index = int(tube_index)
            cp.phi = float(phi)
            if cp.raw_center is None:
                cp.raw_center = raw_center
            cp.source = source
            if cp.kind is None and cp.fixed_point is not None:
                cp.kind = cp.fixed_point.kind
            return cp
        if isinstance(candidate, ChainFixedPoint):
            return TubeCutPoint(
                tube_index=int(tube_index),
                phi=float(phi),
                R=float(candidate.R),
                Z=float(candidate.Z),
                kind=candidate.kind,
                fixed_point=candidate,
                source=source,
                raw_center=raw_center,
            )
        if isinstance(candidate, (tuple, list)) and len(candidate) >= 2:
            return TubeCutPoint(
                tube_index=int(tube_index),
                phi=float(phi),
                R=float(candidate[0]),
                Z=float(candidate[1]),
                kind=self.kind,
                fixed_point=None,
                source=source,
                raw_center=raw_center,
            )
        raise TypeError(f"Unsupported reconstruction candidate type: {type(candidate)!r}")

    def reconstruct_section_cut(
        self,
        phi: float,
        *,
        tube_index: int,
        section_reconstructor: Optional[SectionReconstructor],
        existing_points: Sequence[TubeCutPoint],
        reason: str,
    ) -> Optional[TubeCutPoint]:
        if section_reconstructor is None:
            return None
        candidate = section_reconstructor(float(phi), self, existing_points, reason)
        if isinstance(candidate, list):
            candidate = candidate[0] if candidate else None
        source = f"reconstructed-{reason}"
        return self._coerce_reconstructed_candidate(candidate, phi=phi, tube_index=tube_index, source=source)

    def to_island(
        self,
        phi: float,
        *,
        x_points: Optional[List[np.ndarray]] = None,
        level: int = 1,
        label: Optional[str] = None,
        tol: float = 1e-6,
    ) -> Island:
        """Cut the tube by a section and convert it to a discrete ``Island``."""
        fps = self.at_section(phi, tol=tol)
        if len(fps) != 1:
            raise ValueError(
                f"Tube.to_island expected exactly 1 section point at phi={phi:.6f}, got {len(fps)}"
            )
        fp = fps[0]
        isl = Island(
            period_n=self.m,
            O_point=np.array([fp.R, fp.Z], dtype=float),
            X_points=[] if x_points is None else [np.asarray(x, dtype=float) for x in x_points],
            halfwidth=float('nan'),
            level=level,
            label=label or self.label,
            debug_info={
                'tube_kind': self.kind,
                'phi_section': float(phi),
                'greene_residue': float(fp.greene_residue),
                'seed_RZ': tuple(float(v) for v in self.seed_RZ),
            },
        )
        return isl


@dataclass
class TubeChain:
    """Continuous-time counterpart of a discrete ``IslandChain``."""

    m: int
    n: int
    Np: int
    tubes: List[Tube] = field(default_factory=list)
    kind: Optional[str] = None
    label: Optional[str] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_orbits(
        cls,
        orbits: Sequence[IslandChainOrbit],
        *,
        expected_kind: Optional[str] = None,
        label: Optional[str] = None,
    ) -> "TubeChain":
        if not orbits:
            raise ValueError("TubeChain.from_orbits requires at least one orbit")
        m = orbits[0].m
        n = orbits[0].n
        Np = orbits[0].Np
        tubes = [Tube.from_orbit(orbit, label=f"tube[{i}]") for i, orbit in enumerate(orbits)]
        chain_kind = expected_kind
        if chain_kind is None:
            kinds = {tube.kind for tube in tubes}
            kinds.discard(None)
            chain_kind = next(iter(kinds)) if len(kinds) == 1 else None
        return cls(m=m, n=n, Np=Np, tubes=tubes, kind=chain_kind, label=label)

    @property
    def expected_n_tubes(self) -> int:
        return self.m // gcd(self.m, self.n)

    @property
    def n_tubes(self) -> int:
        return len(self.tubes)

    @staticmethod
    def _duplicate_groups(cut_points: Sequence[TubeCutPoint], dedup_tol: float) -> List[List[int]]:
        groups: List[List[int]] = []
        used = set()
        for i, cp in enumerate(cut_points):
            if i in used:
                continue
            group = [i]
            for j in range(i + 1, len(cut_points)):
                cp2 = cut_points[j]
                if np.hypot(cp.R - cp2.R, cp.Z - cp2.Z) < dedup_tol:
                    group.append(j)
            if len(group) > 1:
                groups.append(group)
                used.update(group)
        return groups

    def section_fixed_points(self, phi: float, tol: float = 1e-6) -> List[ChainFixedPoint]:
        fps: List[ChainFixedPoint] = []
        for tube in self.tubes:
            fps.extend(tube.at_section(phi, tol=tol))
        return fps

    def section_cut(
        self,
        phi: float,
        *,
        tol: float = 1e-6,
        dedup_tol: float = 1e-6,
        reconstruct: bool = False,
        section_reconstructor: Optional[SectionReconstructor] = None,
    ) -> SectionCut:
        cut_points: List[TubeCutPoint] = []
        missing: List[int] = []
        reconstructed: List[int] = []

        for idx, tube in enumerate(self.tubes):
            cps = tube.section_cut_point(phi, tube_index=idx, tol=tol)
            if cps:
                cut_points.extend(cps)
            else:
                missing.append(idx)

        duplicate_groups = self._duplicate_groups(cut_points, dedup_tol)

        if reconstruct and section_reconstructor is not None:
            # Reconstruct missing tubes first.
            for idx in list(missing):
                reconstructed_cp = self.tubes[idx].reconstruct_section_cut(
                    phi,
                    tube_index=idx,
                    section_reconstructor=section_reconstructor,
                    existing_points=cut_points,
                    reason="missing",
                )
                if reconstructed_cp is not None:
                    cut_points.append(reconstructed_cp)
                    reconstructed.append(idx)
                    missing.remove(idx)

            # Then reconstruct duplicate tubes (keep the first member of each duplicate group).
            duplicate_groups = self._duplicate_groups(cut_points, dedup_tol)
            for group in list(duplicate_groups):
                keep = group[0]
                for dup_idx in group[1:]:
                    cp_old = cut_points[dup_idx]
                    reconstructed_cp = self.tubes[cp_old.tube_index].reconstruct_section_cut(
                        phi,
                        tube_index=cp_old.tube_index,
                        section_reconstructor=section_reconstructor,
                        existing_points=[cp for k, cp in enumerate(cut_points) if k != dup_idx],
                        reason="duplicate",
                    )
                    if reconstructed_cp is not None:
                        cut_points[dup_idx] = reconstructed_cp
                        reconstructed.append(cp_old.tube_index)

            duplicate_groups = self._duplicate_groups(cut_points, dedup_tol)

        return SectionCut(
            phi=float(phi),
            cut_points=cut_points,
            expected_tube_count=int(self.expected_n_tubes),
            missing_tube_indices=missing,
            duplicate_groups=duplicate_groups,
            reconstructed_tube_indices=sorted(set(reconstructed)),
        )

    def raw_section_cut(
        self,
        phi: float,
        *,
        tol: float = 1e-6,
        dedup_tol: float = 1e-6,
    ) -> SectionCut:
        return self.section_cut(
            phi,
            tol=tol,
            dedup_tol=dedup_tol,
            reconstruct=False,
        )

    def reconstruct_section_cut(
        self,
        phi: float,
        *,
        tol: float = 1e-6,
        dedup_tol: float = 1e-6,
        section_reconstructor: Optional[SectionReconstructor] = None,
    ) -> SectionCut:
        return self.section_cut(
            phi,
            tol=tol,
            dedup_tol=dedup_tol,
            reconstruct=True,
            section_reconstructor=section_reconstructor,
        )

    def raw_section_view(
        self,
        phi: float,
        *,
        kind: Optional[str] = None,
        tol: float = 1e-6,
        dedup_tol: float = 1e-6,
    ):
        """Bridge-layer section view built directly from the raw section cut."""
        from pyna.topo.section_view import SectionViewBuilder
        return SectionViewBuilder.from_tubechain(
            self,
            phi,
            kind=kind,
            reconstruct=False,
            tol=tol,
            dedup_tol=dedup_tol,
        )

    def reconstruct_section_view(
        self,
        phi: float,
        *,
        kind: Optional[str] = None,
        tol: float = 1e-6,
        dedup_tol: float = 1e-6,
        section_reconstructor: Optional[SectionReconstructor] = None,
    ):
        """Bridge-layer section view reconstructed from tube identity + geometry."""
        from pyna.topo.section_view import SectionViewBuilder
        return SectionViewBuilder.from_tubechain(
            self,
            phi,
            kind=kind,
            reconstruct=True,
            tol=tol,
            dedup_tol=dedup_tol,
            section_reconstructor=section_reconstructor,
        )

    def diagnostics(
        self,
        requested_phis: Optional[Sequence[float]] = None,
        tol: float = 1e-6,
    ) -> Dict[str, Any]:
        tube_diags = [tube.diagnostics(requested_phis=requested_phis) for tube in self.tubes]
        per_section_counts: Dict[float, int] = {}
        if requested_phis is None:
            requested_phis = self.tubes[0].section_phis if self.tubes else []
        if requested_phis is None:
            requested_phis = []
        for phi in requested_phis:
            per_section_counts[float(phi)] = len(self.section_fixed_points(float(phi), tol=tol))
        return {
            'm': int(self.m),
            'n': int(self.n),
            'Np': int(self.Np),
            'kind': self.kind,
            'label': self.label,
            'expected_n_tubes': int(self.expected_n_tubes),
            'n_tubes': int(self.n_tubes),
            'complete': bool(self.n_tubes == self.expected_n_tubes),
            'section_counts': per_section_counts,
            'tube_kinds': [tube.kind for tube in self.tubes],
            'tube_diagnostics': tube_diags,
        }

    def summary(self) -> str:
        diag = self.diagnostics(self.tubes[0].section_phis if self.tubes else None)
        return (
            f"TubeChain(kind={self.kind}, label={self.label}, m={self.m}, n={self.n}, Np={self.Np}) "
            f"tubes={diag['n_tubes']}/{diag['expected_n_tubes']} section_counts={diag['section_counts']}"
        )

    def warn_if_incomplete(self, requested_phis: Optional[Sequence[float]] = None) -> None:
        diag = self.diagnostics(requested_phis=requested_phis)
        if not diag['complete']:
            warnings.warn(
                f"TubeChain incomplete for m/n={self.m}/{self.n}: "
                f"found {diag['n_tubes']} tubes, expected {diag['expected_n_tubes']}"
            )

    def to_island_chain(
        self,
        phi: float,
        *,
        x_tubechain: Optional["TubeChain"] = None,
        proximity_tol: float = 1.0,
        tol: float = 1e-6,
        reconstruct: bool = False,
        section_reconstructor: Optional[SectionReconstructor] = None,
        x_section_reconstructor: Optional[SectionReconstructor] = None,
    ) -> IslandChain:
        """Cut the continuous-time chain by one section and form ``IslandChain``."""
        sec = self.section_cut(phi, tol=tol, reconstruct=reconstruct, section_reconstructor=section_reconstructor)
        O_points = [cp.as_array() for cp in sec.unique_points()]
        X_points: List[np.ndarray] = []
        if x_tubechain is not None:
            x_sec = x_tubechain.section_cut(
                phi,
                tol=tol,
                reconstruct=reconstruct,
                section_reconstructor=x_section_reconstructor,
            )
            X_points = [cp.as_array() for cp in x_sec.unique_points()]
        chain = IslandChain.from_fixed_points(
            O_points=O_points,
            X_points=X_points,
            m=self.m,
            n=self.n,
            proximity_tol=proximity_tol,
        )
        chain.warn_if_incomplete(prefix="TubeChain.to_island_chain: ")
        return chain


@dataclass
class ResonanceStructure:
    """Continuous-time resonance object bundling O/X tube chains and section views."""

    m: int
    n: int
    Np: int
    o_tubechain: Optional[TubeChain] = None
    x_tubechain: Optional[TubeChain] = None
    label: Optional[str] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_orbits(
        cls,
        *,
        o_orbits: Optional[Sequence[IslandChainOrbit]] = None,
        x_orbits: Optional[Sequence[IslandChainOrbit]] = None,
        label: Optional[str] = None,
    ) -> "ResonanceStructure":
        first = None
        if o_orbits:
            first = o_orbits[0]
        elif x_orbits:
            first = x_orbits[0]
        if first is None:
            raise ValueError("ResonanceStructure.from_orbits requires o_orbits and/or x_orbits")
        o_chain = TubeChain.from_orbits(o_orbits, expected_kind='O', label='O-tubes') if o_orbits else None
        x_chain = TubeChain.from_orbits(x_orbits, expected_kind='X', label='X-tubes') if x_orbits else None
        return cls(
            m=first.m,
            n=first.n,
            Np=first.Np,
            o_tubechain=o_chain,
            x_tubechain=x_chain,
            label=label,
        )

    def section_cut(
        self,
        phi: float,
        *,
        tol: float = 1e-6,
        reconstruct: bool = False,
        o_section_reconstructor: Optional[SectionReconstructor] = None,
        x_section_reconstructor: Optional[SectionReconstructor] = None,
    ) -> Dict[str, Optional[SectionCut]]:
        return {
            'O': None if self.o_tubechain is None else self.o_tubechain.section_cut(
                phi, tol=tol, reconstruct=reconstruct, section_reconstructor=o_section_reconstructor,
            ),
            'X': None if self.x_tubechain is None else self.x_tubechain.section_cut(
                phi, tol=tol, reconstruct=reconstruct, section_reconstructor=x_section_reconstructor,
            ),
        }

    def boundary_anchor_points(
        self,
        phi: float,
        *,
        tol: float = 1e-6,
        reconstruct: bool = False,
        o_section_reconstructor: Optional[SectionReconstructor] = None,
        x_section_reconstructor: Optional[SectionReconstructor] = None,
    ) -> List[np.ndarray]:
        cuts = self.section_cut(
            phi,
            tol=tol,
            reconstruct=reconstruct,
            o_section_reconstructor=o_section_reconstructor,
            x_section_reconstructor=x_section_reconstructor,
        )
        anchors: List[np.ndarray] = []
        for key in ('O', 'X'):
            sec = cuts[key]
            if sec is None:
                continue
            anchors.extend([cp.as_array() for cp in sec.unique_points()])
        return anchors

    def to_island_chains(
        self,
        phi: float,
        *,
        proximity_tol: float = 1.0,
        tol: float = 1e-6,
        reconstruct: bool = False,
        o_section_reconstructor: Optional[SectionReconstructor] = None,
        x_section_reconstructor: Optional[SectionReconstructor] = None,
    ) -> Dict[str, Optional[IslandChain]]:
        o_chain = None
        x_chain = None
        if self.o_tubechain is not None:
            o_chain = self.o_tubechain.to_island_chain(
                phi,
                x_tubechain=self.x_tubechain,
                proximity_tol=proximity_tol,
                tol=tol,
                reconstruct=reconstruct,
                section_reconstructor=o_section_reconstructor,
                x_section_reconstructor=x_section_reconstructor,
            )
        if self.x_tubechain is not None:
            x_chain = self.x_tubechain.to_island_chain(
                phi,
                x_tubechain=None,
                proximity_tol=proximity_tol,
                tol=tol,
                reconstruct=reconstruct,
                section_reconstructor=x_section_reconstructor,
            )
        return {'O': o_chain, 'X': x_chain}

    def section_view(
        self,
        phi: float,
        *,
        kind: str = 'O',
        reconstruct: bool = False,
        tol: float = 1e-6,
        dedup_tol: float = 1e-6,
        o_section_reconstructor: Optional[SectionReconstructor] = None,
        x_section_reconstructor: Optional[SectionReconstructor] = None,
    ):
        """Return one bridge-layer section view for the requested kind."""
        if kind.upper() == 'O':
            if self.o_tubechain is None:
                return None
            if reconstruct:
                return self.o_tubechain.reconstruct_section_view(
                    phi, kind='O', tol=tol, dedup_tol=dedup_tol, section_reconstructor=o_section_reconstructor,
                )
            return self.o_tubechain.raw_section_view(phi, kind='O', tol=tol, dedup_tol=dedup_tol)
        if kind.upper() == 'X':
            if self.x_tubechain is None:
                return None
            if reconstruct:
                return self.x_tubechain.reconstruct_section_view(
                    phi, kind='X', tol=tol, dedup_tol=dedup_tol, section_reconstructor=x_section_reconstructor,
                )
            return self.x_tubechain.raw_section_view(phi, kind='X', tol=tol, dedup_tol=dedup_tol)
        raise ValueError(f"Unsupported kind={kind!r}; expected 'O' or 'X'")

    def diagnostics(self, requested_phis: Optional[Sequence[float]] = None) -> Dict[str, Any]:
        return {
            'm': int(self.m),
            'n': int(self.n),
            'Np': int(self.Np),
            'label': self.label,
            'O': None if self.o_tubechain is None else self.o_tubechain.diagnostics(requested_phis=requested_phis),
            'X': None if self.x_tubechain is None else self.x_tubechain.diagnostics(requested_phis=requested_phis),
        }

    def summary(self) -> str:
        return (
            f"ResonanceStructure(label={self.label}, m={self.m}, n={self.n}, Np={self.Np}, "
            f"has_O={self.o_tubechain is not None}, has_X={self.x_tubechain is not None})"
        )


__all__ = [
    "TubeCutPoint",
    "SectionCut",
    "Tube",
    "TubeChain",
    "ResonanceStructure",
]
