"""toroidal._tube — TubeCutPoint, Tube, TubeChain."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import gcd
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING
import warnings

import numpy as np

from pyna.topo.core import Tube as _Tube, TubeChain as _TubeChain
from ._cycle import Cycle
from ._fixed_point import FixedPoint, MonodromyData
from ._island import Island, IslandChain, PeriodicOrbit

if TYPE_CHECKING:
    from pyna.topo.toroidal_section_view import SectionView

_SimplePoint = type('_SimplePoint', (), {
    '__init__': lambda self, R, Z: setattr(self, 'R', R) or setattr(self, 'Z', Z)
})

SectionReconstructor = Callable[[float, "Tube", Sequence["TubeCutPoint"], str], Any]


@dataclass
class TubeCutPoint:
    """One section-cut point produced by slicing a Tube."""

    tube_index: int
    phi: float
    R: float
    Z: float
    kind: Optional[str] = None
    fixed_point: Optional[FixedPoint] = None
    source: str = "exact-cut"
    raw_center: Optional[Tuple[float, float]] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def greene_residue(self) -> float:
        if self.fixed_point is None:
            return float("nan")
        return float(self.fixed_point.greene_residue)

    def as_array(self) -> np.ndarray:
        return np.array([self.R, self.Z], dtype=float)


@dataclass(eq=False)
class Tube(_Tube):
    """Toroidal magnetic island: a nested family of invariant tori."""

    _orbit_R:   Optional[np.ndarray] = field(default=None, repr=False)
    _orbit_Z:   Optional[np.ndarray] = field(default=None, repr=False)
    _orbit_phi: Optional[np.ndarray] = field(default=None, repr=False)
    _orbit_alive: Optional[np.ndarray] = field(default=None, repr=False)
    _tube_chain_ref: Optional["TubeChain"] = field(default=None, repr=False, init=False)

    @property
    def m(self) -> int:
        if isinstance(self.O_cycle, Cycle):
            return int(self.O_cycle.winding[0])
        return 0

    @property
    def n(self) -> int:
        if isinstance(self.O_cycle, Cycle) and len(self.O_cycle.winding) > 1:
            return int(self.O_cycle.winding[1])
        return 1

    @property
    def section_phis(self) -> List[float]:
        if isinstance(self.O_cycle, Cycle):
            return sorted(self.O_cycle.sections.keys())
        return []

    @property
    def seed_phi(self) -> float:
        phis = self.section_phis
        return float(phis[0]) if phis else 0.0

    @property
    def seed_RZ(self) -> Tuple[float, float]:
        if isinstance(self.O_cycle, Cycle):
            fps = self.O_cycle.section_points(self.seed_phi)
            if fps:
                return (float(fps[0].R), float(fps[0].Z))
        return (float('nan'), float('nan'))

    def at_section(self, phi: float, tol: float = 1e-6) -> List[FixedPoint]:
        if isinstance(self.O_cycle, Cycle):
            return self.O_cycle.section_points(phi, tol=tol)
        return []

    def x_at_section(self, phi: float, tol: float = 1e-6) -> List[FixedPoint]:
        result: List[FixedPoint] = []
        for xc in self.X_cycles:
            if isinstance(xc, Cycle):
                result.extend(xc.section_points(phi, tol=tol))
        return result

    @property
    def orbit_debug_available(self) -> bool:
        return self._orbit_R is not None and len(self._orbit_R) > 0

    def raw_point_near_section(self, phi: float) -> Optional[Tuple[float, float]]:
        if not self.orbit_debug_available:
            return None
        phi_arr = np.asarray(self._orbit_phi, dtype=float)
        R_arr   = np.asarray(self._orbit_R,   dtype=float)
        Z_arr   = np.asarray(self._orbit_Z,   dtype=float)
        mask = np.isfinite(phi_arr) & np.isfinite(R_arr) & np.isfinite(Z_arr)
        if self._orbit_alive is not None:
            mask &= np.asarray(self._orbit_alive, dtype=bool)
        if not np.any(mask):
            return None
        phi_use = phi_arr[mask]; R_use = R_arr[mask]; Z_use = Z_arr[mask]
        idx = int(np.argmin(np.abs(phi_use - float(phi))))
        return float(R_use[idx]), float(Z_use[idx])

    def section_islands(self, section, tol: float = 1e-6) -> List[Island]:
        from pyna.topo.section import ToroidalSection, coerce_section
        section = coerce_section(section)

        if isinstance(section, ToroidalSection):
            phi = float(section.phi)
            fps = self.at_section(phi, tol=tol)
            x_phi = phi
        else:
            raw_fps = self._general_section_fps(section)
            fps = [FixedPoint(phi=self.seed_phi, R=float(fp.R), Z=float(fp.Z),
                              DPm=np.eye(2), kind='O') for fp in raw_fps]
            x_phi = self.seed_phi

        if not fps:
            return []

        x_fps = self.x_at_section(x_phi, tol=tol)
        x_groups: List[List[FixedPoint]] = [[] for _ in fps]
        for xfp in x_fps:
            dists = [np.hypot(xfp.R - ofp.R, xfp.Z - ofp.Z) for ofp in fps]
            idx = int(np.argmin(dists))
            x_groups[idx].append(xfp)

        islands: List[Island] = []
        for fp, x_fp_list in zip(fps, x_groups):
            isl = Island(
                O_orbit=PeriodicOrbit(points=[fp]),
                X_orbits=[PeriodicOrbit(points=[xfp]) for xfp in x_fp_list],
                label=self.label,
            )
            isl.tube = self
            isl.section = section
            if self._tube_chain_ref is not None:
                isl.tube_chain = self._tube_chain_ref
            islands.append(isl)
        return islands

    def section_cut(self, section, tol: float = 1e-6) -> IslandChain:
        from pyna.topo.section import coerce_section
        islands = self.section_islands(coerce_section(section), tol=tol)
        chain = IslandChain(m=self.m, n=self.n, parent_tube=self, label=self.label,
                            metadata={'n_tubes_included': 1})
        for isl in islands:
            chain.add_island(isl)
        return chain

    def _general_section_fps(self, section) -> list:
        if not self.orbit_debug_available:
            return []
        R_arr   = np.asarray(self._orbit_R,   dtype=float)
        Z_arr   = np.asarray(self._orbit_Z,   dtype=float)
        phi_arr = np.asarray(self._orbit_phi, dtype=float)
        alive   = (np.asarray(self._orbit_alive, dtype=bool)
                   if self._orbit_alive is not None else np.ones(len(R_arr), dtype=bool))
        mask = alive & np.isfinite(R_arr) & np.isfinite(Z_arr)
        R_use = R_arr[mask]; Z_use = Z_arr[mask]; phi_use = phi_arr[mask]
        if len(R_use) < 2:
            return []
        crossings = []
        phase_dim = getattr(section, 'dim_phase', 2)
        for i in range(len(R_use) - 1):
            if phase_dim >= 3:
                pt_prev = np.array([R_use[i],   Z_use[i],   phi_use[i]])
                pt_curr = np.array([R_use[i+1], Z_use[i+1], phi_use[i+1]])
            else:
                pt_prev = np.array([R_use[i],   Z_use[i]])
                pt_curr = np.array([R_use[i+1], Z_use[i+1]])
            if hasattr(section, 'detect_crossing'):
                hit = section.detect_crossing(pt_prev, pt_curr)
                if hit is not None:
                    crossings.append(_SimplePoint(R=float(hit[0]), Z=float(hit[1])))
            elif hasattr(section, 'f'):
                try:
                    f_prev = section.f(pt_prev)
                    f_curr = section.f(pt_curr)
                    if f_prev * f_curr < 0:
                        t = abs(f_prev) / (abs(f_prev) + abs(f_curr) + 1e-30)
                        R_hit = R_use[i] + t * (R_use[i+1] - R_use[i])
                        Z_hit = Z_use[i] + t * (Z_use[i+1] - Z_use[i])
                        crossings.append(_SimplePoint(R=float(R_hit), Z=float(Z_hit)))
                except Exception:
                    pass
        return crossings

    def section_view_points(self, phi: float, *, tube_index: int = 0, tol: float = 1e-6) -> List[TubeCutPoint]:
        fps = self.at_section(phi, tol=tol)
        return [TubeCutPoint(tube_index=tube_index, phi=phi, R=fp.R, Z=fp.Z,
                             kind=fp.kind, fixed_point=fp) for fp in fps]

    def _reconstruct_cut_point(self, phi: float, *, tube_index: int = 0,
                               section_reconstructor: Optional[SectionReconstructor] = None,
                               existing_points: Optional[Sequence[TubeCutPoint]] = None,
                               reason: str = "missing") -> Optional[TubeCutPoint]:
        if section_reconstructor is None:
            return None
        try:
            return section_reconstructor(phi, self, list(existing_points or []), reason)
        except Exception:
            return None

    def diagnostics(self) -> Dict[str, Any]:
        return {
            'invariant_type': 'Tube', 'label': self.label,
            'm': self.m, 'n': self.n,
            'n_X_cycles': len(self.X_cycles),
            'has_skeleton': self.is_skeleton_complete,
            'section_phis': self.section_phis,
        }

    def summary(self) -> str:
        counts = {phi: len(self.at_section(phi)) for phi in self.section_phis[:3]}
        return f"Tube(label={self.label!r}, m={self.m}, n={self.n}, X_cycles={len(self.X_cycles)}, sections={counts})"

    @classmethod
    def from_orbit(cls, orbit: Any, label: str = "") -> "Tube":
        fps: List[FixedPoint] = []
        if hasattr(orbit, 'fixed_points'):
            fps = list(orbit.fixed_points)
        m = int(getattr(orbit, 'm', 1))
        n = int(getattr(orbit, 'n', 1))
        winding = (m, n)
        sections: Dict[float, List[FixedPoint]] = defaultdict(list)
        for fp in fps:
            sections[float(fp.phi)].append(fp)
        mono = fps[0].monodromy if fps else None
        cycle = Cycle(winding=winding, sections=dict(sections), monodromy=mono, ambient_dim=2)
        orb_R   = np.asarray(orbit.orbit_R,   dtype=float) if getattr(orbit, 'orbit_R',   None) is not None else None
        orb_Z   = np.asarray(orbit.orbit_Z,   dtype=float) if getattr(orbit, 'orbit_Z',   None) is not None else None
        orb_phi = np.asarray(orbit.orbit_phi, dtype=float) if getattr(orbit, 'orbit_phi', None) is not None else None
        orb_alive = (np.asarray(orbit.orbit_alive, dtype=bool)
                     if getattr(orbit, 'orbit_alive', None) is not None else None)
        return cls(O_cycle=cycle, X_cycles=[], label=label,
                   _orbit_R=orb_R, _orbit_Z=orb_Z, _orbit_phi=orb_phi, _orbit_alive=orb_alive)


@dataclass(eq=False)
class TubeChain(_TubeChain):
    """Toroidal resonance family: all Tubes sharing one rational surface."""

    @property
    def m(self) -> int:
        return self.tubes[0].m if self.tubes else 0

    @property
    def n(self) -> int:
        return self.tubes[0].n if self.tubes else 0

    @property
    def Np(self) -> int:
        return getattr(self.tubes[0], 'Np', 1) if self.tubes else 1

    @property
    def winding(self) -> Tuple[int, int]:
        return (self.m, self.n)

    @property
    def expected_n_tubes(self) -> int:
        g = gcd(self.m, self.n) if self.n > 0 else 1
        return self.m // g if g > 0 else self.m

    @property
    def x_tubes(self) -> List[Tube]:
        return [t for t in self.tubes if t.is_skeleton_complete]

    @property
    def O_tubes(self) -> List[Tube]:
        return list(self.tubes)

    @classmethod
    def from_XO_fixed_points(cls, x_fps: Sequence[FixedPoint], o_fps: Sequence[FixedPoint],
                             winding: Tuple[int, int], *, label: Optional[str] = None) -> "TubeChain":
        def _build_cycle(fps):
            if not fps:
                return None
            sections: Dict[float, List[FixedPoint]] = defaultdict(list)
            for fp in fps:
                sections[float(fp.phi)].append(fp)
            mono = fps[0].monodromy
            return Cycle(winding=winding, sections=dict(sections), monodromy=mono, ambient_dim=2)

        x_cyc = _build_cycle(list(x_fps))
        o_cyc = _build_cycle(list(o_fps))

        if o_cyc is None:
            if x_cyc is None:
                return cls(tubes=[], label=label)
            dummy_o = Cycle(winding=winding, sections={}, monodromy=None, ambient_dim=2)
            tube = Tube(O_cycle=dummy_o, X_cycles=[x_cyc], label="tube[0]")
            return cls(tubes=[tube], label=label)

        tube = Tube(O_cycle=o_cyc, X_cycles=[x_cyc] if x_cyc else [], label="tube[0]")
        tc = cls(tubes=[tube], label=label)
        tube._tube_chain_ref = tc
        return tc

    @classmethod
    def from_XO_orbits(cls, x_orbits, o_orbits, winding, *, label=None) -> "TubeChain":
        def _collect_fps(orbits):
            fps = []
            for orb in (orbits or []):
                if hasattr(orb, 'fixed_points'):
                    fps.extend(orb.fixed_points)
                elif hasattr(orb, 'sections'):
                    for pts in orb.sections.values():
                        fps.extend(pts if isinstance(pts, list) else [pts])
            return fps
        return cls.from_XO_fixed_points(x_fps=_collect_fps(x_orbits), o_fps=_collect_fps(o_orbits),
                                        winding=winding, label=label)

    @classmethod
    def from_orbits(cls, orbits, *, expected_kind=None, label=None) -> "TubeChain":
        if not orbits:
            raise ValueError("from_orbits requires at least one orbit")
        m = int(getattr(orbits[0], 'm', 1))
        n = int(getattr(orbits[0], 'n', 1))
        tubes = [Tube.from_orbit(orb, label=f"tube[{i}]") for i, orb in enumerate(orbits)]
        tc = cls(tubes=tubes, label=label)
        for t in tubes:
            t._tube_chain_ref = tc
        return tc

    def section_fixed_points(self, phi: float, tol: float = 1e-6) -> List[FixedPoint]:
        fps: List[FixedPoint] = []
        for tube in self.tubes:
            fps.extend(tube.at_section(phi, tol=tol))
        return fps

    def section_xpoints(self, phi: float, tol: float = 1e-6) -> List[FixedPoint]:
        fps: List[FixedPoint] = []
        for tube in self.tubes:
            fps.extend(tube.x_at_section(phi, tol=tol))
        return fps

    def section_opoints(self, phi: float, tol: float = 1e-6) -> List[FixedPoint]:
        return self.section_fixed_points(phi, tol=tol)

    @staticmethod
    def _duplicate_groups(cut_points, dedup_tol):
        groups = []
        used = set()
        for i, cp in enumerate(cut_points):
            if i in used:
                continue
            group = [i]
            for j in range(i + 1, len(cut_points)):
                if np.hypot(cp.R - cut_points[j].R, cp.Z - cut_points[j].Z) < dedup_tol:
                    group.append(j)
            if len(group) > 1:
                groups.append(group)
                used.update(group)
        return groups

    @staticmethod
    def _unique_cut_points(cut_points, dedup_tol=1e-6):
        out = []
        for cp in cut_points:
            if not any(np.hypot(cp.R - k.R, cp.Z - k.Z) < dedup_tol for k in out):
                out.append(cp)
        return out

    def _section_view_data(self, phi, *, tol=1e-6, dedup_tol=1e-6, reconstruct=False, section_reconstructor=None):
        cut_points = []
        missing = []
        reconstructed = []
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
                    phi, tube_index=idx, section_reconstructor=section_reconstructor,
                    existing_points=cut_points, reason="missing")
                if cp is not None:
                    cut_points.append(cp)
                    reconstructed.append(idx)
                    missing.remove(idx)
            duplicate_groups = self._duplicate_groups(cut_points, dedup_tol)
            for group in list(duplicate_groups):
                for dup_idx in group[1:]:
                    cp_old = cut_points[dup_idx]
                    cp = self.tubes[cp_old.tube_index]._reconstruct_cut_point(
                        phi, tube_index=cp_old.tube_index,
                        section_reconstructor=section_reconstructor,
                        existing_points=[c for k, c in enumerate(cut_points) if k != dup_idx],
                        reason="duplicate")
                    if cp is not None:
                        cut_points[dup_idx] = cp
                        reconstructed.append(cp_old.tube_index)
            duplicate_groups = self._duplicate_groups(cut_points, dedup_tol)
        return {'phi': float(phi), 'cut_points': cut_points, 'expected_tube_count': int(self.expected_n_tubes),
                'missing_tube_indices': missing, 'duplicate_groups': duplicate_groups,
                'reconstructed_tube_indices': sorted(set(reconstructed)), 'debug_info': {}}

    def raw_section_view(self, phi, *, kind=None, tol=1e-6, dedup_tol=1e-6) -> "SectionView":
        from pyna.topo.toroidal_section_view import SectionViewBuilder
        return SectionViewBuilder.from_tubechain(self, phi, kind=kind, reconstruct=False, tol=tol, dedup_tol=dedup_tol)

    def reconstruct_section_view(self, phi, *, kind=None, tol=1e-6, dedup_tol=1e-6,
                                 section_reconstructor=None) -> "SectionView":
        from pyna.topo.toroidal_section_view import SectionViewBuilder
        return SectionViewBuilder.from_tubechain(self, phi, kind=kind, reconstruct=True,
                                                  tol=tol, dedup_tol=dedup_tol,
                                                  section_reconstructor=section_reconstructor)

    def to_island_chain(self, phi, *, x_tubechain=None, proximity_tol=1.0, tol=1e-6,
                        reconstruct=False, section_reconstructor=None, x_section_reconstructor=None) -> IslandChain:
        view = (self.reconstruct_section_view(phi, kind='O', tol=tol, dedup_tol=1e-6,
                                              section_reconstructor=section_reconstructor)
                if reconstruct else self.raw_section_view(phi, kind='O', tol=tol, dedup_tol=1e-6))
        x_view = None
        if x_tubechain is not None:
            x_view = (x_tubechain.reconstruct_section_view(phi, kind='X', tol=tol, dedup_tol=1e-6,
                                                            section_reconstructor=x_section_reconstructor)
                      if reconstruct else x_tubechain.raw_section_view(phi, kind='X', tol=tol, dedup_tol=1e-6))
        chain = view.to_island_chain(x_section_view=x_view, proximity_tol=proximity_tol, dedup_tol=1e-6)
        chain.metadata['n_tubes_included'] = self.n_tubes
        chain.warn_if_incomplete(prefix="TubeChain.to_island_chain: ")
        return chain

    def to_island_chain_connected(self, phi, *, proximity_tol=1.0, tol=1e-6) -> IslandChain:
        chain = self.to_island_chain(phi, proximity_tol=proximity_tol, tol=tol)
        self._attach_chain_refs(chain, phi=phi)
        return chain

    def _attach_chain_refs(self, chain, phi):
        islands = chain.islands if hasattr(chain, 'islands') else []
        tube_fps = []
        for tube_idx, tube in enumerate(self.tubes):
            for fp in tube.at_section(phi):
                tube_fps.append((tube_idx, tube, fp.R, fp.Z))
        for island in islands:
            if island is None:
                continue
            best_idx, best_tube, best_dist = None, None, float('inf')
            R0, Z0 = float(island.O_point[0]), float(island.O_point[1])
            for tidx, t, R, Z in tube_fps:
                d = np.hypot(R - R0, Z - Z0)
                if d < best_dist:
                    best_dist, best_idx, best_tube = d, tidx, t
            if best_idx is not None:
                island.tube = best_tube
                island.tube_chain = self
                island.resonance_index = best_idx
        indexed = [(isl.resonance_index, isl) for isl in islands
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

    def section_cut(self, section, tol: float = 1e-6) -> IslandChain:
        from pyna.topo.section import ToroidalSection, coerce_section
        section = coerce_section(section)
        if isinstance(section, ToroidalSection):
            return self.to_island_chain_connected(section.phi, tol=tol)
        chain = IslandChain(m=self.m, n=self.n, parent_tube=self, label=self.label,
                            metadata={'n_tubes_included': self.n_tubes,
                                      'section_object': section.__class__.__name__})
        for tube_idx, tube in enumerate(self.tubes):
            subchain = tube.section_cut(section, tol=tol)
            for isl in subchain.islands:
                isl.tube = tube
                isl.tube_chain = self
                isl.resonance_index = tube_idx
                chain.add_island(isl)
        return chain

    def wire_skeletons(self, section_phi=0.0, proximity_tol=0.05):
        pass

    def wire_xo_refs(self, section_phi=0.0, proximity_tol=0.05):
        self.wire_skeletons(section_phi=section_phi, proximity_tol=proximity_tol)

    def diagnostics(self, requested_phis=None, tol=1e-6):
        if requested_phis is None:
            requested_phis = self.tubes[0].section_phis if self.tubes else []
        requested_phis = list(requested_phis)
        per_section = {float(phi): len(self.section_fixed_points(float(phi), tol=tol)) for phi in requested_phis}
        return {'m': int(self.m), 'n': int(self.n), 'Np': int(self.Np), 'label': self.label,
                'expected_n_tubes': int(self.expected_n_tubes), 'n_tubes': int(self.n_tubes),
                'complete': bool(self.n_tubes == self.expected_n_tubes), 'section_counts': per_section}

    def summary(self) -> str:
        if not self.tubes:
            return "TubeChain(empty)"
        diag = self.diagnostics(self.tubes[0].section_phis[:2] if self.tubes else [])
        return (f"TubeChain(label={self.label}, m={self.m}, n={self.n}) "
                f"tubes={diag['n_tubes']}/{diag['expected_n_tubes']} section_counts={diag['section_counts']}")

    def warn_if_incomplete(self, requested_phis=None):
        diag = self.diagnostics(requested_phis=requested_phis)
        if not diag['complete']:
            warnings.warn(f"TubeChain incomplete for m/n={self.m}/{self.n}: "
                          f"found {diag['n_tubes']} tubes, expected {diag['expected_n_tubes']}")
