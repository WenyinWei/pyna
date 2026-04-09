"""pyna.topo.invariant -- InvariantObject class hierarchy.

Defines the formal abstract layer for invariant geometric objects of a
dynamical system (continuous or discrete).  This is Layer 1 in the pyna
architecture, sitting above the raw field-line data (Layer 0: PhaseSpace /
DynamicalSystem) and below the application-level topology analysis.

Class hierarchy
---------------
InvariantObject (ABC)
    PeriodicOrbit          -- elliptic or hyperbolic periodic orbit (replaces
                              IslandChainOrbit as a first-class object)
    InvariantTorus         -- KAM torus (non-resonant invariant surface)
    InvariantManifold (ABC)
        StableManifold     -- W^s of a PeriodicOrbit
        UnstableManifold   -- W^u of a PeriodicOrbit

MCF-specific
------------
MCFPoincareMap lives in pyna.topo.dynamics (concrete PoincareMap subclass).

Notes
-----
* Convention: resonance is always labelled m/n (m = toroidal turns,
  n = poloidal winding number).  Never use p/q in MCF context.
* PeriodicOrbit WRAPS IslandChainOrbit for backward compatibility; it does
  NOT replace it in the call graph.
* InvariantManifold.grow() delegates to CynaStableManifold /
  CynaUnstableManifold from pyna.topo.manifold_improve when cyna is
  available, falling back to the scipy-based version otherwise.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyna.topo.section import Section
    from pyna.topo.island import Island
    from pyna.topo.island_chain import IslandChainOrbit, ChainFixedPoint
    from pyna.topo.resonance import ResonanceNumber
    from pyna.topo.dynamics import PhaseSpace, PoincareMap


__all__ = [
    "InvariantObject",
    "PeriodicOrbit",
    "InvariantTorus",
    "InvariantManifold",
    "StableManifold",
    "UnstableManifold",
]


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class InvariantObject(ABC):
    """Abstract mixin interface for invariant geometric objects of a dynamical system.

    An invariant object O satisfies phi^t(O) <= O for all t (continuous flow)
    or P(O) = O (discrete map).

    Design: Pure interface -- no __init__, no state fields.
    Concrete subclasses (including dataclasses) provide their own fields
    and satisfy the interface via @property overrides.

    The invariant-object hierarchy (from smallest to largest):
      ChainFixedPoint  -- a 0-D invariant set (one fixed point of P^m)
      Island           -- one magnetic island (O-region, nested InvariantTori)
      IslandChain      -- a full resonance family of Islands
      Tube             -- continuous-time Island (3D invariant torus structure)
      TubeChain        -- continuous-time IslandChain
      InvariantTorus   -- a KAM torus (non-resonant)
      PeriodicOrbit    -- a periodic orbit (closed orbit, resonant)
      InvariantManifold -- stable/unstable manifold of a PeriodicOrbit

    Required interface (abstract):
      .label           @property -> str | None
      .section_cut(section) -> list
      .diagnostics()   -> dict

    Optional interface (with sensible defaults):
      .poincare_map    @property -> PoincareMap | None  (default: None)
      .phase_space     @property -> PhaseSpace | None   (default: via poincare_map)
    """

    # ── Abstract interface ────────────────────────────────────────────────────

    @property
    @abstractmethod
    def label(self) -> Optional[str]:
        """Human-readable identifier for this invariant object."""

    @abstractmethod
    def section_cut(self, section) -> list:
        """Return the intersection of this object with a Poincare section.

        Parameters
        ----------
        section : Section | float
            The Poincare section (ToroidalSection or phi value).

        Returns
        -------
        list
            Contents depend on the subclass:
            - PeriodicOrbit / Island -> list[Island]
            - InvariantTorus -> list[np.ndarray] of (R,Z) crossing arrays
            - InvariantManifold -> list[np.ndarray] of manifold branch points
            - TubeChain / IslandChain -> list[Island]
        """

    @abstractmethod
    def diagnostics(self) -> Dict[str, Any]:
        """Return a structured diagnostic/debug dict."""

    # ── Optional interface (concrete defaults) ────────────────────────────────

    @property
    def poincare_map(self):
        """The Poincare map this object lives in (PoincareMap | None).

        Override in subclasses that carry a poincare_map field.
        """
        return None

    @property
    def phase_space(self):
        """Phase space of the associated map (PhaseSpace | None)."""
        pm = self.poincare_map
        if pm is not None:
            return pm.phase_space
        return None

    # ── Repr ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(label={self.label!r})"


# ─────────────────────────────────────────────────────────────────────────────
# PeriodicOrbit
# ─────────────────────────────────────────────────────────────────────────────

class PeriodicOrbit(InvariantObject):
    """A periodic orbit of the Poincaré map: P^m(x*) = x*.

    This is the formal ``InvariantObject`` wrapper around
    :class:`~pyna.topo.island_chain.IslandChainOrbit`.  It provides:

    * A stable API at the ``InvariantObject`` level (section_cut, diagnostics).
    * Semantic properties: ``stability``, ``greene_residue``, ``resonance``.
    * Forward compatibility for future direct construction (not via orbit).

    Backward compatibility
    ----------------------
    The wrapped :class:`IslandChainOrbit` is accessible via ``.orbit``.
    All existing code that uses ``IslandChainOrbit`` continues to work.

    MCF convention
    --------------
    Resonance is labelled m/n (m = toroidal turns, n = poloidal winding).
    ``self.m`` = toroidal period, ``self.n`` = poloidal winding number.
    """

    def __init__(
        self,
        orbit: "IslandChainOrbit",
        *,
        label: Optional[str] = None,
        poincare_map=None,
    ):
        self._label = label
        self._poincare_map = poincare_map
        self._orbit = orbit

    # ── InvariantObject interface ─────────────────────────────────────────────

    @property
    def label(self) -> Optional[str]:
        return self._label

    @property
    def poincare_map(self):
        return self._poincare_map

    # ── Constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_island_chain_orbit(
        cls,
        orbit: "IslandChainOrbit",
        *,
        label: Optional[str] = None,
        poincare_map=None,
    ) -> "PeriodicOrbit":
        """Wrap an existing IslandChainOrbit as a PeriodicOrbit.

        Parameters
        ----------
        orbit : IslandChainOrbit
            The orbit to wrap.
        label : str, optional
            Human-readable label; defaults to ``'m/n orbit'``.
        poincare_map : MCFPoincareMap, optional
            The Poincaré map this orbit belongs to.
        """
        if label is None:
            label = f"{orbit.m}/{orbit.n} orbit"
        return cls(orbit=orbit, label=label, poincare_map=poincare_map)

    @classmethod
    def from_fixed_point(
        cls,
        R0: float,
        Z0: float,
        phi0: float,
        m: int,
        n: int,
        Np: int,
        *,
        field_cache: Optional[dict] = None,
        poincare_map=None,
        section_phis: Optional[Sequence[float]] = None,
        n_sections: int = 4,
        label: Optional[str] = None,
        DPhi: float = 0.05,
        refine: bool = True,
    ) -> "PeriodicOrbit":
        """Build a PeriodicOrbit from a known seed fixed point.

        Requires either ``field_cache`` (cyna fast path) or ``poincare_map``
        with a working ``field_cache`` attribute.

        Parameters
        ----------
        R0, Z0, phi0 : float
            Seed fixed-point (R, Z) coordinates and section angle [rad].
        m, n : int
            Toroidal period and poloidal winding number (MCF convention).
        Np : int
            Field toroidal periodicity.
        field_cache : dict, optional
            Cache dict with keys ``BR, BPhi, BZ, R_grid, Z_grid, Phi_grid``.
        poincare_map : MCFPoincareMap, optional
            If field_cache is None, uses poincare_map.field_cache.
        section_phis : list of float, optional
            Target Poincaré sections.
        n_sections : int
            Number of equally spaced sections (used when section_phis is None).
        label : str, optional
        DPhi : float
            cyna RK4 step [rad].
        refine : bool
            Newton-refine each section crossing.
        """
        from pyna.topo.island_chain import IslandChainOrbit

        fc = field_cache
        if fc is None and poincare_map is not None and hasattr(poincare_map, 'field_cache'):
            fc = poincare_map.field_cache

        if fc is not None:
            orbit = IslandChainOrbit.from_cyna_cache(
                R0, Z0, phi0, fc, Np, m, n,
                section_phis=section_phis,
                n_sections=n_sections,
                DPhi=DPhi,
                refine=refine,
            )
        else:
            raise ValueError(
                "PeriodicOrbit.from_fixed_point requires either field_cache or "
                "a poincare_map with a .field_cache attribute."
            )

        if label is None:
            label = f"{m}/{n} orbit"
        return cls(orbit=orbit, label=label, poincare_map=poincare_map)

    # ── Backward-compat orbit access ─────────────────────────────────────────

    @property
    def orbit(self) -> "IslandChainOrbit":
        """The wrapped IslandChainOrbit (backward compatibility)."""
        return self._orbit

    # ── Resonance properties ──────────────────────────────────────────────────

    @property
    def m(self) -> int:
        """Toroidal period (MCF convention)."""
        return self._orbit.m

    @property
    def n(self) -> int:
        """Poloidal winding number (MCF convention)."""
        return self._orbit.n

    @property
    def Np(self) -> int:
        """Field toroidal periodicity."""
        return self._orbit.Np

    @property
    def resonance(self) -> "ResonanceNumber":
        """ResonanceNumber(m, n) for this orbit."""
        from pyna.topo.resonance import ResonanceNumber
        return ResonanceNumber(self.m, self.n)

    # ── Stability ─────────────────────────────────────────────────────────────

    @property
    def fixed_points(self) -> list:
        """All ChainFixedPoint objects (across all sections)."""
        return list(self._orbit.fixed_points)

    @property
    def stability(self) -> str:
        """'X' (hyperbolic), 'O' (elliptic), or 'mixed'."""
        diag = self._orbit.diagnostics()
        if diag['mixed_kind']:
            return 'mixed'
        return diag['dominant_kind'] or 'unknown'

    @property
    def greene_residue(self) -> float:
        """Greene's residue R = (2 - Tr DPm) / 4 at the first available fixed point.

        R < 0  →  hyperbolic (X-point, island separatrix).
        0 < R < 1  →  elliptic (O-point, island centre).
        R = 0 or 1  →  parabolic / bifurcation.
        """
        fps = self._orbit.fixed_points
        if not fps:
            return float('nan')
        return float(fps[0].greene_residue)

    @property
    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues of DPm at the first available fixed point."""
        fps = self._orbit.fixed_points
        if not fps:
            return np.array([float('nan'), float('nan')])
        return fps[0].eigenvalues

    # ── InvariantObject interface ─────────────────────────────────────────────

    def section_cut(self, section) -> List["Island"]:
        """Cut this orbit at a Poincaré section → list of Islands.

        Delegates to the underlying IslandChainOrbit.  Each Island carries
        a back-reference (island.periodic_orbit = self).

        Parameters
        ----------
        section : ToroidalSection | float
            The section to cut at.

        Returns
        -------
        list of Island
        """
        from pyna.topo.section import ToroidalSection
        from pyna.topo.island import Island as _Island

        if isinstance(section, (int, float)):
            phi = float(section)
        elif hasattr(section, 'phi'):
            phi = float(section.phi)
        else:
            raise NotImplementedError(
                "PeriodicOrbit.section_cut only supports ToroidalSection (phi=const)."
            )

        fps = self._orbit.fixed_points_at_section(phi)
        islands = []
        for fp in fps:
            isl = _Island(
                period_n=self.m,
                O_point=np.array([float(fp.R), float(fp.Z)], dtype=float),
                X_points=[],
                halfwidth=float('nan'),
                label=self._label,
            )
            isl.periodic_orbit = self  # new back-reference at InvariantObject level
            islands.append(isl)
        return islands

    def diagnostics(self) -> Dict[str, Any]:
        d = self._orbit.diagnostics()
        d['invariant_type'] = 'PeriodicOrbit'
        d['stability'] = self.stability
        d['greene_residue'] = self.greene_residue
        d['resonance'] = str(self.resonance)
        return d

    def __repr__(self) -> str:
        return (f"PeriodicOrbit(m={self.m}, n={self.n}, "
                f"stability={self.stability!r}, label={self._label!r})")


# ─────────────────────────────────────────────────────────────────────────────
# InvariantTorus
# ─────────────────────────────────────────────────────────────────────────────

class InvariantTorus(InvariantObject):
    """A KAM (Kolmogorov–Arnold–Moser) invariant torus.

    A KAM torus is a non-resonant invariant surface in phase space.
    Field lines on a KAM torus fill it densely and never close.  The
    rotational transform ι is irrational.

    In contrast to :class:`PeriodicOrbit` (resonant, rational ι = n/m),
    InvariantTorus represents the generic case.

    Construction
    ------------
    Use :meth:`from_poincare_trace` to build from a Poincaré scatter cloud,
    or construct directly with pre-computed crossing arrays.

    Notes
    -----
    Full KAM analysis (rigorous existence, Greene's criterion, Birkhoff
    normal form) is future work.  This class is currently a thin data
    container with rotational-transform estimation.
    """

    def __init__(
        self,
        crossings: Dict[float, np.ndarray],
        *,
        rotational_transform: Optional[float] = None,
        label: Optional[str] = None,
        poincare_map=None,
    ):
        """
        Parameters
        ----------
        crossings : dict {phi: ndarray shape (N,2)}
            Poincaré crossing arrays (R, Z) at each section phi.
        rotational_transform : float, optional
            Estimated ι. If None, estimated from the first crossing array.
        label : str, optional
        poincare_map : PoincareMap, optional
        """
        super().__init__()  # pure mixin, no state in ABC
        self._label = label
        self._poincare_map = poincare_map
        self._crossings = {float(k): np.asarray(v, dtype=float)
                           for k, v in crossings.items()}
        self._iota = rotational_transform

    # ── InvariantObject interface ─────────────────────────────────────────────

    @property
    def label(self) -> Optional[str]:
        return self._label

    @property
    def poincare_map(self):
        return self._poincare_map

    # ── Constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_poincare_trace(
        cls,
        R0: float,
        Z0: float,
        phi0: float,
        n_turns: int,
        *,
        field_cache: Optional[dict] = None,
        poincare_map=None,
        section_phis: Optional[Sequence[float]] = None,
        Np: int = 1,
        label: Optional[str] = None,
    ) -> "InvariantTorus":
        """Build an InvariantTorus by tracing a field line (cyna fast path).

        Parameters
        ----------
        R0, Z0, phi0 : float
            Starting point.
        n_turns : int
            Number of Poincaré map iterations.
        field_cache : dict, optional
            cyna field cache (BR, BPhi, BZ, grids).
        poincare_map : MCFPoincareMap, optional
            Alternative source for field_cache.
        section_phis : list of float, optional
            Sections to record crossings at.  Defaults to [phi0].
        Np : int
            Field toroidal periodicity.
        label : str, optional
        """
        fc = field_cache
        if fc is None and poincare_map is not None and hasattr(poincare_map, 'field_cache'):
            fc = poincare_map.field_cache

        if section_phis is None:
            section_phis = [phi0]

        crossings: Dict[float, np.ndarray] = {}

        if fc is not None:
            try:
                import pyna._cyna as _cyna
                # Use trace_poincare_multi for multi-section tracing
                Phi_grid = fc['Phi_grid']
                if abs(Phi_grid[-1] - 2 * np.pi) > 1e-6:
                    Phi_ext = np.append(Phi_grid, 2 * np.pi)
                else:
                    Phi_ext = np.asarray(Phi_grid, dtype=np.float64)

                def _ext(a):
                    return np.concatenate([a, a[:, :, :1]], axis=2)

                BR_c   = np.ascontiguousarray(_ext(fc['BR']),   dtype=np.float64)
                BPhi_c = np.ascontiguousarray(_ext(fc['BPhi']), dtype=np.float64)
                BZ_c   = np.ascontiguousarray(_ext(fc['BZ']),   dtype=np.float64)
                Rg = np.ascontiguousarray(fc['R_grid'], dtype=np.float64)
                Zg = np.ascontiguousarray(fc['Z_grid'], dtype=np.float64)
                Pg = np.ascontiguousarray(Phi_ext, dtype=np.float64)

                for phi_s in section_phis:
                    result = _cyna.trace_poincare_batch(
                        np.array([R0], dtype=np.float64),
                        np.array([Z0], dtype=np.float64),
                        float(phi_s),
                        int(n_turns),
                        BR=BR_c, BPhi=BPhi_c, BZ=BZ_c,
                        R_grid=Rg, Z_grid=Zg, Phi_grid=Pg,
                    )
                    R_arr = np.asarray(result[0][0], dtype=float)
                    Z_arr = np.asarray(result[1][0], dtype=float)
                    mask = np.isfinite(R_arr) & np.isfinite(Z_arr)
                    pts = np.column_stack([R_arr[mask], Z_arr[mask]])
                    crossings[float(phi_s)] = pts
            except Exception:
                # Fallback: empty crossings
                for phi_s in section_phis:
                    crossings[float(phi_s)] = np.empty((0, 2), dtype=float)
        else:
            for phi_s in section_phis:
                crossings[float(phi_s)] = np.empty((0, 2), dtype=float)

        # Estimate rotational transform from first section
        iota = None
        if crossings:
            pts = next(iter(crossings.values()))
            if len(pts) > 2:
                from pyna.topo.poincare import rotational_transform_from_trajectory
                # Build a dummy (R, Z, phi) trajectory for ι estimation
                try:
                    phi_vals = np.full(len(pts), phi0)
                    traj = np.column_stack([pts[:, 0], pts[:, 1], phi_vals])
                    iota = rotational_transform_from_trajectory(traj)
                except Exception:
                    pass

        return cls(
            crossings=crossings,
            rotational_transform=iota,
            label=label or f"KAM torus (R0={R0:.3f}, Z0={Z0:.3f})",
            poincare_map=poincare_map,
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def rotational_transform(self) -> Optional[float]:
        """Estimated rotational transform ι (iota)."""
        return self._iota

    @property
    def safety_factor(self) -> Optional[float]:
        """Estimated safety factor q = 1/ι."""
        if self._iota is None or abs(self._iota) < 1e-12:
            return None
        return 1.0 / self._iota

    @property
    def is_resonant(self) -> bool:
        """False for a KAM torus (ι is irrational). True for rational surfaces."""
        return False

    @property
    def sections(self) -> List[float]:
        """Sorted list of section angles for which crossings are stored."""
        return sorted(self._crossings.keys())

    def crossing_array(self, phi: float) -> np.ndarray:
        """Return crossings at section phi as (N, 2) array (R, Z)."""
        for k, v in self._crossings.items():
            if abs(k - phi) < 1e-9:
                return v
        return np.empty((0, 2), dtype=float)

    # ── InvariantObject interface ─────────────────────────────────────────────

    def section_cut(self, section) -> List[np.ndarray]:
        """Return crossing array(s) at the section.

        Returns a list containing one (N, 2) ndarray of (R, Z) crossings.
        """
        if isinstance(section, (int, float)):
            phi = float(section)
        elif hasattr(section, 'phi'):
            phi = float(section.phi)
        else:
            raise NotImplementedError(
                "InvariantTorus.section_cut only supports ToroidalSection."
            )
        return [self.crossing_array(phi)]

    def diagnostics(self) -> Dict[str, Any]:
        return {
            'invariant_type': 'InvariantTorus',
            'label': self._label,
            'rotational_transform': self._iota,
            'safety_factor': self.safety_factor,
            'is_resonant': self.is_resonant,
            'n_sections': len(self._crossings),
            'crossing_counts': {phi: len(v) for phi, v in self._crossings.items()},
        }

    def __repr__(self) -> str:
        iota_str = f"{self._iota:.5f}" if self._iota is not None else 'None'
        return f"InvariantTorus(ι={iota_str}, label={self._label!r})"


# ─────────────────────────────────────────────────────────────────────────────
# InvariantManifold (abstract)
# ─────────────────────────────────────────────────────────────────────────────

class InvariantManifold(InvariantObject, ABC):
    """Abstract base for stable/unstable manifolds of a PeriodicOrbit.

    An invariant manifold W^{s/u}(x*) of a hyperbolic fixed point x*
    is the set of points that asymptotically approach x* under forward
    (stable) or backward (unstable) iteration of the map.

    In MCF: the separatrix manifolds of X-points define the island
    boundaries and govern chaotic transport in the edge region.

    Parameters
    ----------
    periodic_orbit : PeriodicOrbit
        The hyperbolic periodic orbit this manifold belongs to.
    branch : str
        ``'stable'`` or ``'unstable'``.
    """

    _VALID_BRANCHES = ('stable', 'unstable')

    def __init__(
        self,
        periodic_orbit: PeriodicOrbit,
        branch: str,
        *,
        label: Optional[str] = None,
        poincare_map=None,
    ):
        if branch not in self._VALID_BRANCHES:
            raise ValueError(f"branch must be 'stable' or 'unstable', got {branch!r}")
        pm = poincare_map or periodic_orbit.poincare_map
        self._label = label
        self._poincare_map = pm
        self._orbit = periodic_orbit
        self._branch = branch
        self._points: Optional[np.ndarray] = None  # cached grown manifold

    # ── InvariantObject interface ─────────────────────────────────────────────

    @property
    def label(self) -> Optional[str]:
        return self._label

    @property
    def poincare_map(self):
        return self._poincare_map

    @property
    def periodic_orbit(self) -> PeriodicOrbit:
        """The hyperbolic periodic orbit this manifold belongs to."""
        return self._orbit

    @property
    def branch(self) -> str:
        """'stable' or 'unstable'."""
        return self._branch

    @property
    def points(self) -> Optional[np.ndarray]:
        """Cached grown manifold points, shape (N, 2) or None if not yet grown."""
        return self._points

    # ── Abstract ─────────────────────────────────────────────────────────────

    @abstractmethod
    def grow(
        self,
        n_steps: int,
        eps: float,
        field_cache_or_map=None,
        *,
        phi_section: Optional[float] = None,
        n_turns: Optional[int] = None,
    ) -> np.ndarray:
        """Grow the manifold and return its points as (N, 2) array.

        Parameters
        ----------
        n_steps : int
            Number of growth steps (map iterations per branch point).
        eps : float
            Initial displacement along the eigenvector (metres).
        field_cache_or_map : dict | MCFPoincareMap, optional
            Field cache or poincare map for tracing.
        phi_section : float, optional
            Poincaré section angle for the X-point.
        n_turns : int, optional
            Period m of the orbit (for n-turn map step).

        Returns
        -------
        ndarray of shape (N, 2)
        """

    def section_cut(self, section) -> List[np.ndarray]:
        """Return grown manifold points at the section.

        Raises RuntimeError if manifold has not been grown yet.
        """
        if self._points is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.section_cut called before grow(). "
                "Call grow() first."
            )
        # For a Poincaré section the manifold points are already 2D (R,Z)
        return [self._points.copy()]

    def diagnostics(self) -> Dict[str, Any]:
        return {
            'invariant_type': self.__class__.__name__,
            'branch': self._branch,
            'label': self._label,
            'periodic_orbit': repr(self._orbit),
            'n_points': len(self._points) if self._points is not None else 0,
            'grown': self._points is not None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# StableManifold / UnstableManifold
# ─────────────────────────────────────────────────────────────────────────────

class StableManifold(InvariantManifold):
    """Stable manifold W^s of a hyperbolic periodic orbit.

    Points on W^s approach the orbit asymptotically under forward
    iteration of the Poincaré map.

    Implementation delegates to :class:`~pyna.topo.manifold_improve.CynaStableManifold`
    when cyna is available, otherwise falls back to the scipy version.
    """

    def __init__(
        self,
        periodic_orbit: PeriodicOrbit,
        *,
        label: Optional[str] = None,
        poincare_map=None,
    ):
        super().__init__(
            periodic_orbit,
            branch='stable',
            label=label or f"W^s({periodic_orbit.label})",
            poincare_map=poincare_map,
        )

    def grow(
        self,
        n_steps: int,
        eps: float,
        field_cache_or_map=None,
        *,
        phi_section: Optional[float] = None,
        n_turns: Optional[int] = None,
    ) -> np.ndarray:
        """Grow the stable manifold using cyna field-line tracing.

        Parameters
        ----------
        n_steps : int
            Number of manifold growth steps.
        eps : float
            Initial seed displacement [m] along the stable eigenvector.
        field_cache_or_map : dict | MCFPoincareMap, optional
            If None: tries poincare_map.field_cache.
        phi_section : float, optional
            Section angle of the X-point.
        n_turns : int, optional
            Defaults to periodic_orbit.m.

        Returns
        -------
        ndarray of shape (N, 2)  — (R, Z) manifold points
        """
        from pyna.topo.manifold_improve import CynaStableManifold, ScipyStableManifold

        fc, field_func = self._resolve_field(field_cache_or_map)
        x_point, DPm, phi_s = self._get_xpoint_info(phi_section)
        m = n_turns or self._orbit.m
        phi_span = (phi_s, phi_s + m * 2.0 * np.pi)

        if fc is not None:
            try:
                mf = CynaStableManifold(
                    x_point=x_point,
                    DPm=DPm,
                    field_cache=fc,
                    phi_section=phi_s,
                    n_turns=m,
                )
                pts = mf.run(
                    np.array([[x_point[0]], [x_point[1]]]),
                    np.array([[x_point[0]], [x_point[1]]]),
                    n_steps,
                    eps=eps,
                )
                self._points = np.asarray(pts, dtype=float).T
                return self._points
            except Exception:
                pass  # fallback to scipy

        if field_func is not None:
            mf = ScipyStableManifold(
                x_point=x_point,
                DPm=DPm,
                field_func=field_func,
                phi_span=phi_span,
            )
            pts = mf.grow(n_steps, eps=eps)
            self._points = np.asarray(pts, dtype=float)
            return self._points

        raise ValueError(
            "StableManifold.grow requires field_cache or a field_func. "
            "Pass field_cache_or_map or ensure poincare_map.field_cache is set."
        )

    def _resolve_field(self, field_cache_or_map):
        """Extract (field_cache, field_func) from input."""
        fc = None
        field_func = None
        if field_cache_or_map is None:
            pm = self._poincare_map
            if pm is not None and hasattr(pm, 'field_cache'):
                fc = pm.field_cache
        elif isinstance(field_cache_or_map, dict):
            fc = field_cache_or_map
        elif hasattr(field_cache_or_map, 'field_cache'):
            fc = field_cache_or_map.field_cache
        elif callable(field_cache_or_map):
            field_func = field_cache_or_map
        return fc, field_func

    def _get_xpoint_info(self, phi_section):
        """Extract X-point (R, Z), DPm, and phi from the periodic orbit."""
        fps = self._orbit.fixed_points
        x_fps = [fp for fp in fps if fp.kind == 'X']
        if not x_fps:
            # Use first available point
            x_fps = fps
        if not x_fps:
            raise ValueError("StableManifold: no fixed points found in PeriodicOrbit.")

        if phi_section is not None:
            # Pick the fixed point nearest to the requested section
            dists = [abs(((fp.phi - phi_section + np.pi) % (2 * np.pi)) - np.pi)
                     for fp in x_fps]
            fp = x_fps[int(np.argmin(dists))]
        else:
            fp = x_fps[0]

        x_point = np.array([fp.R, fp.Z], dtype=float)
        DPm = np.asarray(fp.DPm, dtype=float)
        phi_s = float(fp.phi)
        return x_point, DPm, phi_s


class UnstableManifold(InvariantManifold):
    """Unstable manifold W^u of a hyperbolic periodic orbit.

    Points on W^u approach the orbit asymptotically under backward
    (time-reversed) iteration of the Poincaré map.

    Implementation delegates to :class:`~pyna.topo.manifold_improve.CynaUnstableManifold`
    when cyna is available.
    """

    def __init__(
        self,
        periodic_orbit: PeriodicOrbit,
        *,
        label: Optional[str] = None,
        poincare_map=None,
    ):
        super().__init__(
            periodic_orbit,
            branch='unstable',
            label=label or f"W^u({periodic_orbit.label})",
            poincare_map=poincare_map,
        )

    def grow(
        self,
        n_steps: int,
        eps: float,
        field_cache_or_map=None,
        *,
        phi_section: Optional[float] = None,
        n_turns: Optional[int] = None,
    ) -> np.ndarray:
        """Grow the unstable manifold using cyna field-line tracing.

        See :meth:`StableManifold.grow` for parameter documentation.
        """
        from pyna.topo.manifold_improve import CynaUnstableManifold, ScipyUnstableManifold

        fc, field_func = self._resolve_field(field_cache_or_map)
        x_point, DPm, phi_s = self._get_xpoint_info(phi_section)
        m = n_turns or self._orbit.m
        phi_span = (phi_s, phi_s + m * 2.0 * np.pi)

        if fc is not None:
            try:
                mf = CynaUnstableManifold(
                    x_point=x_point,
                    DPm=DPm,
                    field_cache=fc,
                    phi_section=phi_s,
                    n_turns=m,
                )
                pts = mf.run(
                    np.array([[x_point[0]], [x_point[1]]]),
                    np.array([[x_point[0]], [x_point[1]]]),
                    n_steps,
                    eps=eps,
                )
                self._points = np.asarray(pts, dtype=float).T
                return self._points
            except Exception:
                pass  # fallback

        if field_func is not None:
            mf = ScipyUnstableManifold(
                x_point=x_point,
                DPm=DPm,
                field_func=field_func,
                phi_span=phi_span,
            )
            pts = mf.grow(n_steps, eps=eps)
            self._points = np.asarray(pts, dtype=float)
            return self._points

        raise ValueError(
            "UnstableManifold.grow requires field_cache or a field_func."
        )

    # Reuse _resolve_field and _get_xpoint_info from StableManifold pattern
    _resolve_field = StableManifold._resolve_field
    _get_xpoint_info = StableManifold._get_xpoint_info
