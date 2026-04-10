"""pyna.topo.invariant -- InvariantObject class hierarchy.

Defines the formal abstract layer for invariant geometric objects of a
dynamical system (continuous or discrete).  This is Layer 1 in the pyna
architecture, sitting above the raw field-line data (Layer 0: PhaseSpace /
DynamicalSystem) and below the application-level topology analysis.

Class hierarchy
---------------
InvariantObject (ABC)           [pyna.topo._base]
    InvariantTorus         -- KAM torus (non-resonant invariant surface)

Note
----
* InvariantManifold (StableManifold / UnstableManifold) previously lived
  here as dead stub subclasses.  The real implementations are in
  ``pyna.topo.manifold_improve`` and are exported from ``pyna.topo.__init__``.
* ``PeriodicOrbit`` is now a first-class invariant-manifold class in
  ``pyna.topo.invariants``.  It represents a period-m orbit of a discrete
  map.  ``FixedPoint`` is a single-point convenience (period-1 case).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np

# InvariantObject lives in _base to avoid circular imports
from pyna.topo._base import InvariantObject, InvariantSet, InvariantManifold, SectionCuttable


if TYPE_CHECKING:
    from pyna.topo.section import Section


__all__ = [
    "InvariantSet",
    "InvariantManifold",
    "SectionCuttable",
    "InvariantObject",
    "InvariantTorus",
]


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
