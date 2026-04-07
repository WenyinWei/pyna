"""Bridge objects between continuous-time tube geometry and discrete map chains."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np

from pyna.topo.identity import ResonanceID, TubeID, IslandID
from pyna.topo.island import IslandChain

if TYPE_CHECKING:
    from pyna.topo.island_chain import ChainFixedPoint
    from pyna.topo.tube import TubeChain, _SectionCut, TubeCutPoint


@dataclass
class SectionViewPoint:
    """One point in a section view, with optional continuous/discrete identity."""

    phi: float
    R: float
    Z: float
    kind: Optional[str] = None
    tube_id: Optional[TubeID] = None
    island_id: Optional[IslandID] = None
    fixed_point: Optional["ChainFixedPoint"] = None
    source: str = "raw"
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
class SectionCorrespondence:
    """Structured correspondence between tube IDs and section-view points."""

    resonance_id: ResonanceID
    phi: float
    tube_to_point_indices: Dict[TubeID, List[int]] = field(default_factory=dict)
    island_to_point_indices: Dict[IslandID, List[int]] = field(default_factory=dict)
    missing_tube_ids: List[TubeID] = field(default_factory=list)
    duplicate_tube_ids: List[TubeID] = field(default_factory=list)
    reconstructed_tube_ids: List[TubeID] = field(default_factory=list)
    debug_info: Dict[str, Any] = field(default_factory=dict)

    def is_complete(self) -> bool:
        return not self.missing_tube_ids and not self.duplicate_tube_ids

    def diagnostics(self) -> Dict[str, Any]:
        return {
            'resonance': self.resonance_id.short_label(),
            'phi': float(self.phi),
            'missing_tube_ids': [tid.short_label() for tid in self.missing_tube_ids],
            'duplicate_tube_ids': [tid.short_label() for tid in self.duplicate_tube_ids],
            'reconstructed_tube_ids': [tid.short_label() for tid in self.reconstructed_tube_ids],
            'n_tubes_mapped': len(self.tube_to_point_indices),
            'n_islands_mapped': len(self.island_to_point_indices),
        }


@dataclass
class SectionView:
    """Bridge-layer representation of one resonance at one Poincaré section."""

    phi: float
    resonance_id: ResonanceID
    points: List[SectionViewPoint]
    kind: Optional[str] = None
    correspondence: Optional[SectionCorrespondence] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return len(self.points)

    def unique_points(self, dedup_tol: float = 1e-6) -> List[SectionViewPoint]:
        out: List[SectionViewPoint] = []
        for pt in self.points:
            if not any(np.hypot(pt.R - keep.R, pt.Z - keep.Z) < dedup_tol for keep in out):
                out.append(pt)
        return out

    def fixed_points(self, dedup_tol: float = 1e-6):
        """Return unique fixed-point objects carried by this section view."""
        fps = []
        seen = []
        for pt in self.unique_points(dedup_tol=dedup_tol):
            if pt.fixed_point is None:
                continue
            fp = pt.fixed_point
            if any(np.hypot(fp.R - R0, fp.Z - Z0) < dedup_tol for R0, Z0 in seen):
                continue
            seen.append((float(fp.R), float(fp.Z)))
            fps.append(fp)
        return fps

    def to_fixed_point_dict(self, dedup_tol: float = 1e-6) -> Dict[str, List[Any]]:
        """Return a legacy-style ``{'xpts': [...], 'opts': [...]}`` dict."""
        pts = self.fixed_points(dedup_tol=dedup_tol)
        kind = (self.kind or '').upper()
        if kind == 'X':
            return {'xpts': pts, 'opts': []}
        if kind == 'O':
            return {'xpts': [], 'opts': pts}
        xpts = [fp for fp in pts if getattr(fp, 'kind', None) == 'X']
        opts = [fp for fp in pts if getattr(fp, 'kind', None) == 'O']
        return {'xpts': xpts, 'opts': opts}

    def diagnostics(self) -> Dict[str, Any]:
        return {
            'phi': float(self.phi),
            'resonance': self.resonance_id.short_label(),
            'kind': self.kind,
            'n_points': int(self.n_points),
            'correspondence': None if self.correspondence is None else self.correspondence.diagnostics(),
        }

    def summary(self) -> str:
        diag = self.diagnostics()
        return (
            f"SectionView(phi={self.phi:.6f}, resonance={diag['resonance']}, kind={self.kind}, "
            f"n_points={self.n_points}, corr={diag['correspondence']})"
        )

    def to_island_chain(
        self,
        *,
        x_section_view: Optional["SectionView"] = None,
        proximity_tol: float = 1.0,
        dedup_tol: float = 1e-6,
    ) -> IslandChain:
        """Convert a bridge-layer section view into a discrete ``IslandChain``."""
        O_points = [pt.as_array() for pt in self.unique_points(dedup_tol=dedup_tol)]
        X_points: List[np.ndarray] = []
        if x_section_view is not None:
            X_points = [pt.as_array() for pt in x_section_view.unique_points(dedup_tol=dedup_tol)]
        chain = IslandChain.from_fixed_points(
            O_points=O_points,
            X_points=X_points,
            m=self.resonance_id.m,
            n=self.resonance_id.n,
            proximity_tol=proximity_tol,
        )
        return chain

    @classmethod
    def from_island_chain(
        cls,
        chain: IslandChain,
        phi: float,
        *,
        resonance_id: Optional[ResonanceID] = None,
        kind: Optional[str] = "O",
    ) -> "SectionView":
        """Construct a section view from a discrete ``IslandChain``."""
        if resonance_id is None:
            resonance_id = ResonanceID(m=chain.m, n=chain.n, Np=1)
        points: List[SectionViewPoint] = []
        tube_map: Dict[TubeID, List[int]] = {}
        island_map: Dict[IslandID, List[int]] = {}
        for idx, isl in enumerate(chain.islands):
            tube_id = TubeID(resonance=resonance_id, tube_index=idx, kind=kind)
            island_id = IslandID(resonance=resonance_id, phi=float(phi), island_index=idx, kind=kind)
            pt = SectionViewPoint(
                phi=float(phi),
                R=float(isl.O_point[0]),
                Z=float(isl.O_point[1]),
                kind=kind,
                tube_id=tube_id,
                island_id=island_id,
                source="projected-from-discrete",
            )
            points.append(pt)
            tube_map[tube_id] = [idx]
            island_map[island_id] = [idx]
        corr = SectionCorrespondence(
            resonance_id=resonance_id,
            phi=float(phi),
            tube_to_point_indices=tube_map,
            island_to_point_indices=island_map,
        )
        return cls(phi=float(phi), resonance_id=resonance_id, points=points, kind=kind, correspondence=corr)


@dataclass
class SectionViewBuilder:
    """Builder/adaptor between TubeChain section cuts and SectionView."""

    resonance_id: ResonanceID

    @classmethod
    def from_tubechain(
        cls,
        tubechain: "TubeChain",
        phi: float,
        *,
        kind: Optional[str] = None,
        reconstruct: bool = False,
        tol: float = 1e-6,
        dedup_tol: float = 1e-6,
        section_reconstructor=None,
    ) -> SectionView:
        resonance_id = ResonanceID(m=tubechain.m, n=tubechain.n, Np=tubechain.Np, label=tubechain.label)
        builder = cls(resonance_id=resonance_id)
        if reconstruct:
            cut = tubechain._reconstruct_section_cut(
                phi,
                tol=tol,
                dedup_tol=dedup_tol,
                section_reconstructor=section_reconstructor,
            )
        else:
            cut = tubechain._raw_section_cut(
                phi,
                tol=tol,
                dedup_tol=dedup_tol,
            )
        view_kind = kind or tubechain.kind
        return builder.from_section_cut(cut, kind=view_kind)

    def from_section_cut(self, cut: "_SectionCut", *, kind: Optional[str] = None) -> SectionView:
        points: List[SectionViewPoint] = []
        tube_map: Dict[TubeID, List[int]] = {}
        island_map: Dict[IslandID, List[int]] = {}
        duplicate_tube_ids: List[TubeID] = []
        reconstructed_tube_ids: List[TubeID] = []

        # Build point list with stable tube IDs.
        for idx, cp in enumerate(cut.cut_points):
            tube_id = TubeID(self.resonance_id, int(cp.tube_index), cp.kind or kind)
            pt = SectionViewPoint(
                phi=float(cut.phi),
                R=float(cp.R),
                Z=float(cp.Z),
                kind=cp.kind or kind,
                tube_id=tube_id,
                fixed_point=cp.fixed_point,
                source=cp.source,
                raw_center=cp.raw_center,
                debug_info=dict(cp.debug_info),
            )
            points.append(pt)
            tube_map.setdefault(tube_id, []).append(idx)
            if pt.source != "exact-cut":
                reconstructed_tube_ids.append(tube_id)

        # Assign island IDs by geometric ordering on the unique point set.
        ordered_unique = sorted(cut.unique_points(), key=lambda cp: (cp.R, cp.Z))
        for island_idx, cp in enumerate(ordered_unique):
            for j, pt in enumerate(points):
                if np.hypot(pt.R - cp.R, pt.Z - cp.Z) < 1e-12:
                    island_id = IslandID(self.resonance_id, float(cut.phi), island_idx, pt.kind)
                    pt.island_id = island_id
                    island_map.setdefault(island_id, []).append(j)

        missing_tube_ids = [TubeID(self.resonance_id, idx, kind) for idx in cut.missing_tube_indices]
        for grp in cut.duplicate_groups:
            if not grp:
                continue
            first = cut.cut_points[grp[0]]
            duplicate_tube_ids.append(TubeID(self.resonance_id, int(first.tube_index), first.kind or kind))

        corr = SectionCorrespondence(
            resonance_id=self.resonance_id,
            phi=float(cut.phi),
            tube_to_point_indices=tube_map,
            island_to_point_indices=island_map,
            missing_tube_ids=missing_tube_ids,
            duplicate_tube_ids=duplicate_tube_ids,
            reconstructed_tube_ids=sorted(set(reconstructed_tube_ids), key=lambda t: t.tube_index),
        )
        return SectionView(
            phi=float(cut.phi),
            resonance_id=self.resonance_id,
            points=points,
            kind=kind,
            correspondence=corr,
            debug_info=dict(cut.debug_info),
        )


__all__ = [
    "SectionViewPoint",
    "SectionCorrespondence",
    "SectionView",
    "SectionViewBuilder",
]
