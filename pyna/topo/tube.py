"""Continuous-time tube and resonance-structure abstractions.

This module keeps the continuous-time side of the topology model:
- ``Tube``: one connected periodic orbit / tube
- ``TubeChain``: one resonance-family collection of tubes
- ``ResonanceStructure``: paired O/X tube chains for one resonance

Bridge-layer representations live in :mod:`pyna.topo.section_view`.
The older ``SectionCut`` staging class has been removed; low-level cut data is
kept as private dictionaries and is not exposed as a public API concept.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd
from typing import Any, Callable, Dict, List, Optional, Sequence, TYPE_CHECKING
import warnings

import numpy as np

from pyna.topo.island import Island, IslandChain
from pyna.topo.island_chain import ChainFixedPoint, IslandChainOrbit

if TYPE_CHECKING:
    from pyna.topo.section_view import SectionView


SectionReconstructor = Callable[[float, "Tube", Sequence["TubeCutPoint"], str], Any]


@dataclass
class TubeCutPoint:
    """One section-cut point produced by slicing a ``Tube``."""

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
        return f"Tube(kind={kind}, label={self.label})\n{self.orbit.summary()}"

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

    def section_view_points(
        self,
        phi: float,
        *,
        tube_index: int,
        tol: float = 1e-6,
    ) -> List[TubeCutPoint]:
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
            for fp in self.at_section(phi, tol=tol)
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

    def _reconstruct_cut_point(
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
        fps = self.at_section(phi, tol=tol)
        if len(fps) != 1:
            raise ValueError(
                f"Tube.to_island expected exactly 1 section point at phi={phi:.6f}, got {len(fps)}"
            )
        fp = fps[0]
        return Island(
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

    @staticmethod
    def _unique_cut_points(cut_points: Sequence[TubeCutPoint], dedup_tol: float = 1e-6) -> List[TubeCutPoint]:
        out: List[TubeCutPoint] = []
        for cp in cut_points:
            if not any(np.hypot(cp.R - keep.R, cp.Z - keep.Z) < dedup_tol for keep in out):
                out.append(cp)
        return out

    def section_fixed_points(self, phi: float, tol: float = 1e-6) -> List[ChainFixedPoint]:
        fps: List[ChainFixedPoint] = []
        for tube in self.tubes:
            fps.extend(tube.at_section(phi, tol=tol))
        return fps

    def _section_view_data(
        self,
        phi: float,
        *,
        tol: float = 1e-6,
        dedup_tol: float = 1e-6,
        reconstruct: bool = False,
        section_reconstructor: Optional[SectionReconstructor] = None,
    ) -> Dict[str, Any]:
        cut_points: List[TubeCutPoint] = []
        missing: List[int] = []
        reconstructed: List[int] = []

        for idx, tube in enumerate(self.tubes):
            cps = tube.section_view_points(phi, tube_index=idx, tol=tol)
            if cps:
                cut_points.extend(cps)
            else:
                missing.append(idx)

        duplicate_groups = self._duplicate_groups(cut_points, dedup_tol)

        if reconstruct and section_reconstructor is not None:
            for idx in list(missing):
                cp = self.tubes[idx]._reconstruct_cut_point(
                    phi,
                    tube_index=idx,
                    section_reconstructor=section_reconstructor,
                    existing_points=cut_points,
                    reason="missing",
                )
                if cp is not None:
                    cut_points.append(cp)
                    reconstructed.append(idx)
                    missing.remove(idx)

            duplicate_groups = self._duplicate_groups(cut_points, dedup_tol)
            for group in list(duplicate_groups):
                for dup_idx in group[1:]:
                    cp_old = cut_points[dup_idx]
                    cp = self.tubes[cp_old.tube_index]._reconstruct_cut_point(
                        phi,
                        tube_index=cp_old.tube_index,
                        section_reconstructor=section_reconstructor,
                        existing_points=[c for k, c in enumerate(cut_points) if k != dup_idx],
                        reason="duplicate",
                    )
                    if cp is not None:
                        cut_points[dup_idx] = cp
                        reconstructed.append(cp_old.tube_index)

            duplicate_groups = self._duplicate_groups(cut_points, dedup_tol)

        return {
            'phi': float(phi),
            'cut_points': cut_points,
            'expected_tube_count': int(self.expected_n_tubes),
            'missing_tube_indices': missing,
            'duplicate_groups': duplicate_groups,
            'reconstructed_tube_indices': sorted(set(reconstructed)),
            'debug_info': {},
        }

    def raw_section_view(
        self,
        phi: float,
        *,
        kind: Optional[str] = None,
        tol: float = 1e-6,
        dedup_tol: float = 1e-6,
    ) -> "SectionView":
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
    ) -> "SectionView":
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

    def diagnostics(self, requested_phis: Optional[Sequence[float]] = None, tol: float = 1e-6) -> Dict[str, Any]:
        tube_diags = [tube.diagnostics(requested_phis=requested_phis) for tube in self.tubes]
        if requested_phis is None:
            requested_phis = self.tubes[0].section_phis if self.tubes else []
        requested_phis = [] if requested_phis is None else requested_phis
        per_section_counts = {float(phi): len(self.section_fixed_points(float(phi), tol=tol)) for phi in requested_phis}
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
        view = (
            self.reconstruct_section_view(
                phi,
                kind=self.kind or 'O',
                tol=tol,
                dedup_tol=1e-6,
                section_reconstructor=section_reconstructor,
            )
            if reconstruct else
            self.raw_section_view(
                phi,
                kind=self.kind or 'O',
                tol=tol,
                dedup_tol=1e-6,
            )
        )
        x_view = None
        if x_tubechain is not None:
            x_view = (
                x_tubechain.reconstruct_section_view(
                    phi,
                    kind=x_tubechain.kind or 'X',
                    tol=tol,
                    dedup_tol=1e-6,
                    section_reconstructor=x_section_reconstructor,
                )
                if reconstruct else
                x_tubechain.raw_section_view(
                    phi,
                    kind=x_tubechain.kind or 'X',
                    tol=tol,
                    dedup_tol=1e-6,
                )
            )
        chain = view.to_island_chain(x_section_view=x_view, proximity_tol=proximity_tol, dedup_tol=1e-6)
        chain.warn_if_incomplete(prefix="TubeChain.to_island_chain: ")
        return chain

    def to_island_chain_connected(
        self,
        phi: float,
        *,
        x_tubechain: Optional["TubeChain"] = None,
        proximity_tol: float = 1.0,
        tol: float = 1e-6,
    ) -> IslandChain:
        """Build an IslandChain with full orbit connectivity wired.

        Each Island in the returned chain carries:
          - ``tube_chain`` back-reference to this TubeChain
          - ``resonance_index``: index of its parent Tube (0-based)
          - ``next()`` / ``last()``: adjacent Islands under P^1 / P^{-1}

        The next/last connectivity follows the orbit order of the Tubes in
        this TubeChain: Island from Tube[i] maps to Island from Tube[(i+1) % n].
        """
        chain = self.to_island_chain(phi, x_tubechain=x_tubechain,
                                      proximity_tol=proximity_tol, tol=tol)
        # Attach TubeChain reference and wire connectivity
        self._attach_chain_refs(chain, phi=phi)
        return chain

    def _attach_chain_refs(self, chain: IslandChain, phi: float) -> None:
        """Attach tube_chain back-refs and wire next/last connectivity to Islands.

        Called after building an IslandChain from this TubeChain.
        Matches each Island's O_point to the Tube it came from (by proximity),
        then sets tube_chain, resonance_index, and wires P^1 connectivity
        (Island from Tube[i] → Island from Tube[(i+1) % n]).
        """
        import numpy as np
        islands = chain.O_islands if hasattr(chain, 'O_islands') else []
        if not islands and hasattr(chain, 'islands'):
            islands = chain.islands
        if not islands:
            return

        # Match each Island to the Tube it came from via O_point proximity
        tube_fps = []
        for tube_idx, tube in enumerate(self.tubes):
            fps = tube.at_section(phi)
            for fp in fps:
                tube_fps.append((tube_idx, fp.R, fp.Z))

        for island in islands:
            if island is None:
                continue
            best_idx, best_dist = None, float('inf')
            R0, Z0 = float(island.O_point[0]), float(island.O_point[1])
            for tidx, R, Z in tube_fps:
                d = np.hypot(R - R0, Z - Z0)
                if d < best_dist:
                    best_dist, best_idx = d, tidx
            if best_idx is not None:
                island.tube_chain = self
                island.resonance_index = best_idx

        # Wire P^1 connectivity: sort islands by resonance_index and link them
        indexed = [(isl.resonance_index, isl)
                   for isl in islands
                   if isl is not None and isl.resonance_index is not None]
        if not indexed:
            return
        indexed.sort(key=lambda x: x[0])
        n = len(indexed)
        for k in range(n):
            curr = indexed[k][1]
            nxt  = indexed[(k + 1) % n][1]
            lst  = indexed[(k - 1) % n][1]
            curr._set_next(nxt)
            curr._set_last(lst)

    def _wire_island_connectivity(self) -> None:
        """Lazy connectivity wiring trigger (called by Island.next()/last()).

        No-op here: connectivity should be set at chain-build time via
        _attach_chain_refs. This hook allows Island.next() to ask the
        TubeChain to wire itself if it has not yet been done.
        """
        pass


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
        first = o_orbits[0] if o_orbits else (x_orbits[0] if x_orbits else None)
        if first is None:
            raise ValueError("ResonanceStructure.from_orbits requires o_orbits and/or x_orbits")
        return cls(
            m=first.m,
            n=first.n,
            Np=first.Np,
            o_tubechain=None if not o_orbits else TubeChain.from_orbits(o_orbits, expected_kind='O', label='O-tubes'),
            x_tubechain=None if not x_orbits else TubeChain.from_orbits(x_orbits, expected_kind='X', label='X-tubes'),
            label=label,
        )

    def boundary_anchor_points(
        self,
        phi: float,
        *,
        tol: float = 1e-6,
        reconstruct: bool = False,
        o_section_reconstructor: Optional[SectionReconstructor] = None,
        x_section_reconstructor: Optional[SectionReconstructor] = None,
    ) -> List[np.ndarray]:
        views = self.section_views(
            phi,
            tol=tol,
            reconstruct=reconstruct,
            o_section_reconstructor=o_section_reconstructor,
            x_section_reconstructor=x_section_reconstructor,
        )
        anchors: List[np.ndarray] = []
        for key in ('O', 'X'):
            view = views[key]
            if view is None:
                continue
            anchors.extend([pt.as_array() for pt in view.unique_points()])
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

    def section_views(
        self,
        phi: float,
        *,
        reconstruct: bool = False,
        tol: float = 1e-6,
        dedup_tol: float = 1e-6,
        o_section_reconstructor: Optional[SectionReconstructor] = None,
        x_section_reconstructor: Optional[SectionReconstructor] = None,
    ) -> Dict[str, Any]:
        return {
            'O': None if self.o_tubechain is None else (
                self.o_tubechain.reconstruct_section_view(
                    phi, kind='O', tol=tol, dedup_tol=dedup_tol,
                    section_reconstructor=o_section_reconstructor,
                ) if reconstruct else self.o_tubechain.raw_section_view(
                    phi, kind='O', tol=tol, dedup_tol=dedup_tol,
                )
            ),
            'X': None if self.x_tubechain is None else (
                self.x_tubechain.reconstruct_section_view(
                    phi, kind='X', tol=tol, dedup_tol=dedup_tol,
                    section_reconstructor=x_section_reconstructor,
                ) if reconstruct else self.x_tubechain.raw_section_view(
                    phi, kind='X', tol=tol, dedup_tol=dedup_tol,
                )
            ),
        }

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
    ) -> Optional["SectionView"]:
        views = self.section_views(
            phi,
            reconstruct=reconstruct,
            tol=tol,
            dedup_tol=dedup_tol,
            o_section_reconstructor=o_section_reconstructor,
            x_section_reconstructor=x_section_reconstructor,
        )
        kind_up = kind.upper()
        if kind_up not in ('O', 'X'):
            raise ValueError(f"Unsupported kind={kind!r}; expected 'O' or 'X'")
        return views[kind_up]

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


__all__ = ["TubeCutPoint", "Tube", "TubeChain", "ResonanceStructure"]
