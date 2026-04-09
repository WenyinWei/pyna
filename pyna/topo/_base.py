"""pyna.topo._base -- Abstract base class for invariant objects.

Extracted to a separate module to avoid circular imports between
island_chain.py and invariant.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class InvariantObject(ABC):
    """Abstract mixin interface for invariant geometric objects of a dynamical system.

    An invariant object O satisfies phi^t(O) <= O for all t (continuous flow)
    or P(O) = O (discrete map).

    Design: Pure interface -- no __init__, no state fields.
    Concrete subclasses (including dataclasses) provide their own fields
    and satisfy the interface via @property overrides.

    Required interface (abstract):
      .section_cut(section) -> list
      .diagnostics()   -> dict

    Optional interface (with sensible defaults):
      .label           @property -> str | None
      .poincare_map    @property -> PoincareMap | None  (default: None)
      .phase_space     @property -> PhaseSpace | None   (default: via poincare_map)
    """

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def section_cut(self, section) -> list:
        """Return the intersection of this object with a Poincare section."""

    @abstractmethod
    def diagnostics(self) -> Dict[str, Any]:
        """Return a structured diagnostic/debug dict."""

    # ── Optional interface (concrete defaults) ────────────────────────────────

    @property
    def label(self) -> Optional[str]:
        """Human-readable identifier. Override in subclasses."""
        return None

    @property
    def poincare_map(self):
        """The Poincare map this object lives in (PoincareMap | None)."""
        return None

    @property
    def phase_space(self):
        """Phase space of the associated map (PhaseSpace | None)."""
        pm = self.poincare_map
        if pm is not None:
            return pm.phase_space
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(label={self.label!r})"
