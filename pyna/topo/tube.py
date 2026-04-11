"""tube.py — Tube, TubeChain: continuous-time resonance structures.

A ``Tube`` represents one magnetic island — a complete nested family of
invariant tori whose skeleton is:

  o_cycle  : the elliptic periodic orbit at the island core (type ``Cycle``)
  x_cycles : the hyperbolic periodic orbit(s) at the separatrix (List[Cycle])
             may be empty (e.g. limiter configurations)

A ``TubeChain`` collects the full resonance family (all independent Tubes)
for one rational surface q = m/n.

Design principles
-----------------
- ``Tube`` holds ``o_cycle`` and ``x_cycles`` directly as ``Cycle`` objects
  from ``pyna.topo.invariants``.  There is no intermediate ``IslandChainOrbit``
  wrapper.
- ``TubeChain`` is the authoritative container.  ``from_XO_fixed_points`` and
  ``from_XO_orbits`` are the preferred constructors.
- ``TubeCutPoint`` is a low-level bridge for the ``SectionView`` layer.

Note: ``IslandChainOrbit`` and ``ChainFixedPoint`` have been removed.
Use ``Cycle`` and ``FixedPoint`` from ``pyna.topo.invariants`` instead.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import gcd
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING
import warnings

import numpy as np

from pyna.topo.invariant import InvariantSet
from pyna.topo.island import Island, IslandChain
from pyna.topo.invariants import Cycle, FixedPoint, MonodromyData, PeriodicOrbit

if TYPE_CHECKING:
    from pyna.topo.section_view import SectionView


_SimplePoint = type('_SimplePoint', (), {'__init__': lambda self, R, Z: setattr(self, 'R', R) or setattr(self, 'Z', Z)})

SectionReconstructor = Callable[[float, "Tube", Sequence["TubeCutPoint"], str], Any]


# ---------------------------------------------------------------------------
# TubeCutPoint  (section-cut low-level data; unchanged)
# ---------------------------------------------------------------------------

@dataclass
class TubeCutPoint:
    """One section-cut point produced by slicing a ``Tube``."""

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


# ---------------------------------------------------------------------------
# Tube
# ---------------------------------------------------------------------------

@dataclass
class Tube(InvariantSet):
    """One magnetic island: a nested family of invariant tori.

    Skeleton
    --------
    o_cycle : Cycle
        The elliptic periodic orbit at the island core.  All invariant tori
        of this island surround this orbit.
    x_cycles : list of Cycle
        The hyperbolic periodic orbit(s) bounding the island (separatrix).
        May be empty (limiter or axis topology).

    A Tube may additionally carry an optional raw trajectory for approximate
    section cuts with non-toroidal sections:

    _orbit_R, _orbit_Z, _orbit_phi : array or None
        Raw field-line trajectory samples.  Used only by
        ``_general_section_fps`` for HyperplaneSection / ParametricSection.

    The ``label`` field is a human-readable tag (e.g. ``'X-tube[2]'``).

    Notes on physical interpretation
    ----------------------------------
    - Tokamak single-null:  o_cycle = magnetic axis, x_cycles = [X-point cycle]
    - Tokamak double-null:  o_cycle = axis, x_cycles = [lower-X, upper-X]
    - Limiter:              o_cycle = axis, x_cycles = []
    - W7X 5/5 island:       o_cycle = island O-orbit, x_cycles = []
      (the X-separatrix is itself an IslandChain/TubeChain, not a single Cycle)
    """

    o_cycle: Cycle
    x_cycles: List[Cycle] = field(default_factory=list)
    label: Optional[str] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)

    # Optional raw trajectory (for general-section cuts only)
    _orbit_R:   Optional[np.ndarray] = field(default=None, repr=False)
    _orbit_Z:   Optional[np.ndarray] = field(default=None, repr=False)
    _orbit_phi: Optional[np.ndarray] = field(default=None, repr=False)
    _orbit_alive: Optional[np.ndarray] = field(default=None, repr=False)

    # Back-reference to parent TubeChain (set by TubeChain after construction)
    _tube_chain_ref: Optional["TubeChain"] = field(default=None, repr=False, init=False)

    # ── Resonance numbers ─────────────────────────────────────────────────────

    @property
    def m(self) -> int:
        """Poloidal period (numerator of iota = n/m)."""
        return int(self.o_cycle.winding[0])

    @property
    def n(self) -> int:
        """Toroidal winding number (denominator of iota = n/m)."""
        return int(self.o_cycle.winding[1]) if len(self.o_cycle.winding) > 1 else 1

    @property
    def section_phis(self) -> List[float]:
        """Toroidal angles of all recorded Poincare sections."""
        return sorted(self.o_cycle.sections.keys())

    @property
    def seed_phi(self) -> float:
        """Smallest section angle (used as canonical seed)."""
        phis = self.section_phis
        return float(phis[0]) if phis else 0.0

    @property
    def seed_RZ(self) -> Tuple[float, float]:
        """(R, Z) of the first O-type fixed point at seed_phi."""
        fps = self.o_cycle.section_points(self.seed_phi)
        if fps:
            return (float(fps[0].R), float(fps[0].Z))
        return (float('nan'), float('nan'))

    # ── Skeleton ──────────────────────────────────────────────────────────────

    @property
    def is_skeleton_complete(self) -> bool:
        """True when at least one x_cycle is known."""
        return len(self.x_cycles) > 0

    # ── Section interface ─────────────────────────────────────────────────────

    def at_section(self, phi: float, tol: float = 1e-6) -> List[FixedPoint]:
        """Return O-type FixedPoints at section ``phi``."""
        return self.o_cycle.section_points(phi, tol=tol)

    def x_at_section(self, phi: float, tol: float = 1e-6) -> List[FixedPoint]:
        """Return X-type FixedPoints from all x_cycles at section ``phi``."""
        result: List[FixedPoint] = []
        for xc in self.x_cycles:
            result.extend(xc.section_points(phi, tol=tol))
        return result

    @property
    def orbit_debug_available(self) -> bool:
        return self._orbit_R is not None and len(self._orbit_R) > 0

    def raw_point_near_section(self, phi: float) -> Optional[Tuple[float, float]]:
        """Nearest raw trajectory point to section ``phi``.  Returns None if no trajectory."""
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

    def section_cut(self, section, tol: float = 1e-6) -> List[Island]:
        """Cut this Tube with a Section; return one Island per O-region found."""
        from pyna.topo.island import Island as _Island
        from pyna.topo.section import ToroidalSection

        if isinstance(section, (int, float)):
            section = ToroidalSection(float(section))

        if hasattr(section, 'phi'):
            phi = section.phi
            fps = self.at_section(phi, tol=tol)
        else:
            # General section: _SimplePoint objects, wrap in FixedPoint
            raw_fps = self._general_section_fps(section)
            fps = [
                fp if isinstance(fp, FixedPoint) else
                FixedPoint(phi=self.seed_phi, R=float(fp.R), Z=float(fp.Z),
                           DPm=np.eye(2), kind='O')
                for fp in raw_fps
            ]

        if not fps:
            return []

        x_fps = self.x_at_section(getattr(section, 'phi', self.seed_phi), tol=tol)

        islands = []
        for fp in fps:
            x_fp_list = [xfp if isinstance(xfp, FixedPoint) else
                         FixedPoint(phi=fp.phi, R=float(np.asarray(xfp)[0]),
                                    Z=float(np.asarray(xfp)[1]),
                                    DPm=np.array([[2.,0.],[0.,.5]]), kind='X')
                         for xfp in x_fps]
            isl = _Island(
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

    def _general_section_fps(self, section) -> list:
        """Approximate section crossings via raw trajectory scan (non-toroidal sections)."""
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
        for i in range(len(R_use) - 1):
            pt_prev = np.array([R_use[i],   Z_use[i],   phi_use[i]])
            pt_curr = np.array([R_use[i+1], Z_use[i+1], phi_use[i+1]])
            if hasattr(section, 'detect_crossing'):
                hit = section.detect_crossing(pt_prev, pt_curr)
                if hit is not None:
                    crossings.append(_SimplePoint(R=float(hit[0]), Z=float(hit[1])))
            elif hasattr(section, 'f'):
                try:
                    f_prev = section.f(np.array([R_use[i],   Z_use[i]]))
                    f_curr = section.f(np.array([R_use[i+1], Z_use[i+1]]))
                    if f_prev * f_curr < 0:
                        t = abs(f_prev) / (abs(f_prev) + abs(f_curr) + 1e-30)
                        R_c = R_use[i] + t * (R_use[i+1] - R_use[i])
                        Z_c = Z_use[i] + t * (Z_use[i+1] - Z_use[i])
                        crossings.append(_SimplePoint(R=float(R_c), Z=float(Z_c)))
                except Exception:
                    pass
        return crossings

    def to_island(
        self,
        phi: float,
        *,
        x_points: Optional[List[np.ndarray]] = None,
        level: int = 1,
        label: Optional[str] = None,
        tol: float = 1e-6,
    ) -> Island:
        """Return a single Island at section ``phi`` (expects exactly one O-point)."""
        fps = self.at_section(phi, tol=tol)
        if len(fps) != 1:
            raise ValueError(
                f"Tube.to_island expected exactly 1 O-point at phi={phi:.4f}, got {len(fps)}"
            )
        fp = fps[0]
        from pyna.topo.island import Island as _Island

        def _to_xfp(x):
            if isinstance(x, FixedPoint):
                return x
            arr = np.asarray(x, dtype=float).ravel()
            return FixedPoint(phi=fp.phi, R=float(arr[0]), Z=float(arr[1]),
                              DPm=np.array([[2.,0.],[0.,.5]]), kind='X')

        x_fp_list = [] if x_points is None else [_to_xfp(x) for x in x_points]
        return _Island(
            O_orbit=PeriodicOrbit(points=[fp]),
            X_orbits=[PeriodicOrbit(points=[xfp]) for xfp in x_fp_list],
            label=label or self.label,
        )

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
        if isinstance(candidate, FixedPoint):
            return TubeCutPoint(
                tube_index=int(tube_index), phi=float(phi),
                R=float(candidate.R), Z=float(candidate.Z),
                kind=candidate.kind, fixed_point=candidate,
                source=source, raw_center=raw_center,
            )
        if isinstance(candidate, (tuple, list)) and len(candidate) >= 2:
            return TubeCutPoint(
                tube_index=int(tube_index), phi=float(phi),
                R=float(candidate[0]), Z=float(candidate[1]),
                kind=None, fixed_point=None,
                source=source, raw_center=raw_center,
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
        return self._coerce_reconstructed_candidate(
            candidate, phi=phi, tube_index=tube_index, source=f"reconstructed-{reason}"
        )

    def diagnostics(self, requested_phis: Optional[Sequence[float]] = None) -> Dict[str, Any]:
        phis = list(requested_phis) if requested_phis is not None else self.section_phis
        per_phi = {phi: len(self.at_section(phi)) for phi in phis}
        return {
            'label': self.label,
            'm': self.m,
            'n': self.n,
            'section_counts': per_phi,
            'skeleton_complete': self.is_skeleton_complete,
            'n_x_cycles': len(self.x_cycles),
        }

    def summary(self) -> str:
        phis = self.section_phis
        counts = {phi: len(self.at_section(phi)) for phi in phis[:4]}
        return (
            f"Tube(label={self.label}, m={self.m}, n={self.n}, "
            f"x_cycles={len(self.x_cycles)}, sections={counts})"
        )

    # ── Legacy factory (bridges from old IslandChainOrbit-based code) ─────────

    @classmethod
    def from_orbit(cls, orbit: Any, label: Optional[str] = None) -> "Tube":
        """Build a Tube from a legacy orbit object (IslandChainOrbit duck-type).

        Accepts any object with:
          - ``.fixed_points`` or ``.sections``: FixedPoint data
          - ``.m``, ``.n``: resonance numbers
          - (optional) ``.orbit_R/.orbit_Z/.orbit_phi/.orbit_alive``: trajectory
        """
        # Collect fixed points from orbit
        fps: List[FixedPoint] = []
        if hasattr(orbit, 'fixed_points'):
            fps = list(orbit.fixed_points)
        elif hasattr(orbit, 'sections'):
            for pts in orbit.sections.values():
                fps.extend(pts if isinstance(pts, list) else [pts])

        # Build Cycle from fixed points grouped by phi
        sections: Dict[float, List[FixedPoint]] = defaultdict(list)
        for fp in fps:
            sections[float(fp.phi)].append(fp)

        m = int(getattr(orbit, 'm', 1))
        n = int(getattr(orbit, 'n', 1))
        winding = (m, n)
        mono = fps[0].monodromy if fps else None
        cycle = Cycle(winding=winding, sections=dict(sections), monodromy=mono, ambient_dim=2)

        # Optional trajectory
        orb_R   = np.asarray(orbit.orbit_R,   dtype=float) if getattr(orbit, 'orbit_R',   None) is not None else None
        orb_Z   = np.asarray(orbit.orbit_Z,   dtype=float) if getattr(orbit, 'orbit_Z',   None) is not None else None
        orb_phi = np.asarray(orbit.orbit_phi, dtype=float) if getattr(orbit, 'orbit_phi', None) is not None else None
        orb_alive = (np.asarray(orbit.orbit_alive, dtype=bool)
                     if getattr(orbit, 'orbit_alive', None) is not None else None)

        return cls(
            o_cycle=cycle, x_cycles=[],
            label=label,
            _orbit_R=orb_R, _orbit_Z=orb_Z,
            _orbit_phi=orb_phi, _orbit_alive=orb_alive,
        )


# ---------------------------------------------------------------------------
# TubeChain
# ---------------------------------------------------------------------------

@dataclass
class TubeChain(InvariantSet):
    """Resonance family: all Tubes sharing one rational surface q = m/n.

    Each ``Tube`` in ``tubes`` represents one independent periodic orbit
    of the resonance.  For m=10, n=3 (HAO): gcd=1 → one orbit, 10 section
    points → ``tubes`` has one Tube with 10 O-points and 10 X-points.

    Construction
    ------------
    Preferred::

        tc = TubeChain.from_XO_fixed_points(x_fps, o_fps, winding=(10, 3))

    Legacy duck-typed::

        tc = TubeChain.from_XO_orbits(x_orbits, o_orbits, winding=(10, 3))

    Attributes
    ----------
    tubes : list of Tube
        One entry per independent periodic orbit.
    label : str or None
    """

    tubes: List[Tube] = field(default_factory=list)
    label: Optional[str] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)

    # ── Resonance numbers (derived from first tube) ───────────────────────────

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
    def n_tubes(self) -> int:
        return len(self.tubes)

    @property
    def x_tubes(self) -> List[Tube]:
        """Tubes that have at least one x_cycle wired."""
        return [t for t in self.tubes if t.is_skeleton_complete]

    @property
    def o_tubes(self) -> List[Tube]:
        """All tubes (each has an o_cycle by definition)."""
        return list(self.tubes)

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_XO_fixed_points(
        cls,
        x_fps: Sequence[FixedPoint],
        o_fps: Sequence[FixedPoint],
        winding: Tuple[int, int],
        *,
        label: Optional[str] = None,
    ) -> "TubeChain":
        """Build a TubeChain from X and O FixedPoint lists.

        Internally groups fps by phi to build ``Cycle.sections``, constructs
        one Cycle per stability type, then pairs them into Tubes.

        Parameters
        ----------
        x_fps : sequence of FixedPoint
            Hyperbolic fixed points across all sections.
        o_fps : sequence of FixedPoint
            Elliptic fixed points across all sections.
        winding : (m, n)
            Resonance numbers.
        label : str, optional
        """
        def _build_cycle(fps: Sequence[FixedPoint]) -> Optional[Cycle]:
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
            # No O-points — create a degenerate chain with X-only tubes
            if x_cyc is None:
                return cls(tubes=[], label=label)
            dummy_o = Cycle(winding=winding, sections={}, monodromy=None, ambient_dim=2)
            tube = Tube(o_cycle=dummy_o, x_cycles=[x_cyc], label=f"tube[0]")
            return cls(tubes=[tube], label=label)

        tube = Tube(
            o_cycle=o_cyc,
            x_cycles=[x_cyc] if x_cyc else [],
            label=f"tube[0]",
        )
        tc = cls(tubes=[tube], label=label)
        tube._tube_chain_ref = tc
        return tc

    @classmethod
    def from_XO_orbits(
        cls,
        x_orbits: Sequence[Any],
        o_orbits: Sequence[Any],
        winding: Tuple[int, int],
        *,
        label: Optional[str] = None,
    ) -> "TubeChain":
        """Build a TubeChain from legacy orbit objects (duck-typed).

        Accepts objects with ``.fixed_points`` (List[FixedPoint]) or
        ``.sections`` (dict).  The winding numbers are taken from ``winding``
        rather than the orbit objects to avoid ambiguity.
        """
        def _collect_fps(orbits: Sequence[Any]) -> List[FixedPoint]:
            fps: List[FixedPoint] = []
            for orb in (orbits or []):
                if hasattr(orb, 'fixed_points'):
                    fps.extend(orb.fixed_points)
                elif hasattr(orb, 'sections'):
                    for pts in orb.sections.values():
                        fps.extend(pts if isinstance(pts, list) else [pts])
            return fps

        return cls.from_XO_fixed_points(
            x_fps=_collect_fps(x_orbits),
            o_fps=_collect_fps(o_orbits),
            winding=winding,
            label=label,
        )

    @classmethod
    def from_orbits(
        cls,
        orbits: Sequence[Any],
        *,
        expected_kind: Optional[str] = None,
        label: Optional[str] = None,
    ) -> "TubeChain":
        """Build a TubeChain from legacy IslandChainOrbit objects.

        Each orbit becomes one Tube.  Use ``from_XO_orbits`` for new code.
        """
        if not orbits:
            raise ValueError("from_orbits requires at least one orbit")
        m = int(getattr(orbits[0], 'm', 1))
        n = int(getattr(orbits[0], 'n', 1))
        tubes = [Tube.from_orbit(orb, label=f"tube[{i}]") for i, orb in enumerate(orbits)]
        tc = cls(tubes=tubes, label=label)
        for t in tubes:
            t._tube_chain_ref = tc
        return tc

    # ── Section access ────────────────────────────────────────────────────────

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

    # ── SectionView bridge ────────────────────────────────────────────────────

    @staticmethod
    def _duplicate_groups(cut_points: Sequence[TubeCutPoint], dedup_tol: float) -> List[List[int]]:
        groups: List[List[int]] = []
        used: set = set()
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
    def _unique_cut_points(cut_points: Sequence[TubeCutPoint], dedup_tol: float = 1e-6) -> List[TubeCutPoint]:
        out: List[TubeCutPoint] = []
        for cp in cut_points:
            if not any(np.hypot(cp.R - k.R, cp.Z - k.Z) < dedup_tol for k in out):
                out.append(cp)
        return out

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
                    phi, tube_index=idx,
                    section_reconstructor=section_reconstructor,
                    existing_points=cut_points, reason="missing",
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
                        phi, tube_index=cp_old.tube_index,
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
            self, phi, kind=kind, reconstruct=False, tol=tol, dedup_tol=dedup_tol,
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
            self, phi, kind=kind, reconstruct=True,
            tol=tol, dedup_tol=dedup_tol,
            section_reconstructor=section_reconstructor,
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
            self.reconstruct_section_view(phi, kind='O', tol=tol, dedup_tol=1e-6,
                                          section_reconstructor=section_reconstructor)
            if reconstruct else
            self.raw_section_view(phi, kind='O', tol=tol, dedup_tol=1e-6)
        )
        x_view = None
        if x_tubechain is not None:
            x_view = (
                x_tubechain.reconstruct_section_view(phi, kind='X', tol=tol, dedup_tol=1e-6,
                                                     section_reconstructor=x_section_reconstructor)
                if reconstruct else
                x_tubechain.raw_section_view(phi, kind='X', tol=tol, dedup_tol=1e-6)
            )
        chain = view.to_island_chain(x_section_view=x_view, proximity_tol=proximity_tol, dedup_tol=1e-6)
        chain.warn_if_incomplete(prefix="TubeChain.to_island_chain: ")
        return chain

    def to_island_chain_connected(
        self,
        phi: float,
        *,
        proximity_tol: float = 1.0,
        tol: float = 1e-6,
    ) -> IslandChain:
        chain = self.to_island_chain(phi, proximity_tol=proximity_tol, tol=tol)
        self._attach_chain_refs(chain, phi=phi)
        return chain

    def _attach_chain_refs(self, chain: IslandChain, phi: float) -> None:
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
        from pyna.topo.section import ToroidalSection
        if isinstance(section, (int, float)):
            section = ToroidalSection(float(section))
        if hasattr(section, 'phi'):
            return self.to_island_chain_connected(section.phi, tol=tol)
        raise NotImplementedError("TubeChain.section_cut supports ToroidalSection only.")

    def _wire_island_connectivity(self) -> None:
        pass

    def wire_skeletons(self, section_phi: float = 0.0, proximity_tol: float = 0.05) -> None:
        """Pair X-cycles to O-cycles within this chain by proximity."""
        pass  # For from_XO_fixed_points chains, skeleton is already wired.

    def wire_xo_refs(self, section_phi: float = 0.0, proximity_tol: float = 0.05) -> None:
        """Alias for wire_skeletons (backward compatibility)."""
        self.wire_skeletons(section_phi=section_phi, proximity_tol=proximity_tol)

    def diagnostics(self, requested_phis: Optional[Sequence[float]] = None, tol: float = 1e-6) -> Dict[str, Any]:
        if requested_phis is None:
            requested_phis = self.tubes[0].section_phis if self.tubes else []
        requested_phis = list(requested_phis)
        per_section = {float(phi): len(self.section_fixed_points(float(phi), tol=tol))
                       for phi in requested_phis}
        return {
            'm': int(self.m), 'n': int(self.n), 'Np': int(self.Np),
            'label': self.label,
            'expected_n_tubes': int(self.expected_n_tubes),
            'n_tubes': int(self.n_tubes),
            'complete': bool(self.n_tubes == self.expected_n_tubes),
            'section_counts': per_section,
        }

    def summary(self) -> str:
        diag = self.diagnostics(self.tubes[0].section_phis[:2] if self.tubes else [])
        return (
            f"TubeChain(label={self.label}, m={self.m}, n={self.n}) "
            f"tubes={diag['n_tubes']}/{diag['expected_n_tubes']} "
            f"section_counts={diag['section_counts']}"
        )

    def warn_if_incomplete(self, requested_phis: Optional[Sequence[float]] = None) -> None:
        diag = self.diagnostics(requested_phis=requested_phis)
        if not diag['complete']:
            warnings.warn(
                f"TubeChain incomplete for m/n={self.m}/{self.n}: "
                f"found {diag['n_tubes']} tubes, expected {diag['expected_n_tubes']}"
            )


__all__ = ["TubeCutPoint", "Tube", "TubeChain"]
