"""pyna.topo.invariant -- InvariantObject class hierarchy.

Defines the formal abstract layer for invariant geometric objects of a
dynamical system (continuous or discrete).  This is Layer 1 in the pyna
architecture, sitting above the raw field-line data (Layer 0: PhaseSpace /
DynamicalSystem) and below the application-level topology analysis.

Class hierarchy
---------------
InvariantObject (ABC)           [pyna.topo._base]
    PeriodicOrbit          -- periodic orbit (IS-A InvariantObject, was IslandChainOrbit)
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
* PeriodicOrbit is now the canonical class (formerly IslandChainOrbit).
  IslandChainOrbit is kept as a backward-compat alias.
* InvariantManifold.grow() delegates to CynaStableManifold /
  CynaUnstableManifold from pyna.topo.manifold_improve when cyna is
  available, falling back to the scipy-based version otherwise.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np

# InvariantObject lives in _base to avoid circular imports
from pyna.topo._base import InvariantObject


class PeriodicOrbit(InvariantObject):
    """Stub periodic orbit class (island_chain.py was removed)."""

    def __init__(self, label: str = "", poincare_map=None, m: int = 1, **kwargs):
        self._label = label
        self._poincare_map = poincare_map
        self.m = m
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def label(self) -> Optional[str]:
        return self._label

    @property
    def poincare_map(self):
        return self._poincare_map

    def section_cut(self, section) -> list:
        """Return fixed points at the given section.

        Delegates to ``fixed_points_at_section`` if available (IslandChainOrbit
        API), otherwise falls back to ``fixed_points`` filtered by phi.

        Parameters
        ----------
        section : Section | float
            A ToroidalSection or a toroidal angle (float).

        Returns
        -------
        list of ChainFixedPoint (or empty list if no data available)
        """
        phi = None
        if isinstance(section, (int, float)):
            phi = float(section)
        elif hasattr(section, 'phi'):
            phi = float(section.phi)

        # Preferred path: IslandChainOrbit.fixed_points_at_section
        if phi is not None and hasattr(self, 'fixed_points_at_section'):
            return self.fixed_points_at_section(phi)

        # Fallback: filter stored fixed_points list by phi proximity
        fps = getattr(self, 'fixed_points', None)
        if fps is None:
            return []
        if phi is None:
            return list(fps)
        tol = 1e-5
        return [fp for fp in fps if abs(getattr(fp, 'phi', 0.0) - phi) < tol]

    def diagnostics(self) -> Dict[str, Any]:
        return {
            'invariant_type': 'PeriodicOrbit',
            'label': self.label,
            'm': self.m,
        }


if TYPE_CHECKING:
    from pyna.topo.section import Section
    from pyna.topo.island import Island
    from pyna.topo.resonance import ResonanceNumber
    from pyna.topo.dynamics import PhaseSpace, PoincareMap
    FixedPoint = object  # removed with island_chain.py


__all__ = [
    "InvariantObject",
    "PeriodicOrbit",
    "InvariantTorus",
    "InvariantManifold",
    "StableManifold",
    "UnstableManifold",
]# ─────────────────────────────────────────────────────────────────────────────
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
        # Multi-section cache: {phi: ndarray (N,2)}.
        # Replaces the old single-section _points field.
        self._section_points: Dict[float, np.ndarray] = {}
        # Legacy alias: _points points at the most-recently grown section.
        self._points: Optional[np.ndarray] = None

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
        """Cached grown manifold points for the most recently grown section.

        For multi-section access use :meth:`points_at` or :meth:`section_cut`.
        """
        return self._points

    def points_at(self, phi: float, tol: float = 1e-9) -> Optional[np.ndarray]:
        """Return grown manifold points for section *phi*, or None if not grown.

        Parameters
        ----------
        phi : float
            Toroidal angle of the Poincaré section.
        tol : float
            Tolerance for phi lookup in the cache.

        Returns
        -------
        ndarray of shape (N, 2) or None
        """
        for k, v in self._section_points.items():
            if abs(k - phi) < tol:
                return v
        return None

    @property
    def grown_sections(self) -> List[float]:
        """Sorted list of toroidal angles for which the manifold has been grown."""
        return sorted(self._section_points.keys())

    def _cache_points(self, phi: float, pts: np.ndarray) -> None:
        """Store grown points for *phi* and update the legacy _points alias."""
        self._section_points[float(phi)] = pts
        self._points = pts  # legacy alias = most-recently grown section

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

        The manifold must have been grown at (or near) the requested section.
        Call :meth:`grow` with the desired ``phi_section`` first.

        Parameters
        ----------
        section : Section | float
            Poincaré section.  Accepts a float toroidal angle or any object
            with a ``.phi`` attribute (ToroidalSection).

        Returns
        -------
        list of one ndarray of shape (N, 2)

        Raises
        ------
        RuntimeError
            If the manifold has not been grown at any section yet.
        KeyError
            If the manifold has been grown, but not at the requested phi.
            Includes a hint listing the available sections.
        """
        # Resolve phi
        if isinstance(section, (int, float)):
            phi = float(section)
        elif hasattr(section, 'phi'):
            phi = float(section.phi)
        else:
            # Non-toroidal section: return last-grown points if available
            if self._points is None:
                raise RuntimeError(
                    f"{self.__class__.__name__}.section_cut called before grow()."
                )
            return [self._points.copy()]

        pts = self.points_at(phi)
        if pts is not None:
            return [pts.copy()]

        # Not in cache
        if not self._section_points:
            raise RuntimeError(
                f"{self.__class__.__name__}.section_cut called before grow(). "
                "Call grow(phi_section=<phi>) first."
            )
        available = ", ".join(f"{p:.4f}" for p in self.grown_sections)
        raise KeyError(
            f"{self.__class__.__name__}.section_cut: no data for phi={phi:.4f}. "
            f"Available sections: [{available}]. "
            "Call grow(phi_section=<phi>) to grow at a new section."
        )

    def diagnostics(self) -> Dict[str, Any]:
        return {
            'invariant_type': self.__class__.__name__,
            'branch': self._branch,
            'label': self._label,
            'periodic_orbit': repr(self._orbit),
            'grown_sections': self.grown_sections,
            'n_sections_grown': len(self._section_points),
            'n_points_per_section': {
                phi: len(pts) for phi, pts in self._section_points.items()
            },
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
                result = np.asarray(pts, dtype=float).T
                self._cache_points(phi_s, result)
                return result
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
            result = np.asarray(pts, dtype=float)
            self._cache_points(phi_s, result)
            return result

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
                result = np.asarray(pts, dtype=float).T
                self._cache_points(phi_s, result)
                return result
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
            result = np.asarray(pts, dtype=float)
            self._cache_points(phi_s, result)
            return result

        raise ValueError(
            "UnstableManifold.grow requires field_cache or a field_func."
        )

    # Reuse _resolve_field and _get_xpoint_info from StableManifold pattern
    _resolve_field = StableManifold._resolve_field
    _get_xpoint_info = StableManifold._get_xpoint_info
