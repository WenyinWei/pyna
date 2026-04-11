"""pyna.topo._base -- Base classes for invariant objects of dynamical systems.

Extracted to a separate module to avoid circular imports between
island_chain.py and invariant.py.

Hierarchy
---------
InvariantSet           — root ABC for any invariant geometric object
  InvariantManifold    — invariant set with a well-defined intrinsic dimension
SectionCuttable        — mixin protocol for objects that can be sliced by a
                         Poincaré section

``InvariantSet`` is retained as a backward-compatible alias for
``InvariantSet`` so that existing subclass declarations keep working.
"""
from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Optional, Protocol, runtime_checkable


# ─────────────────────────────────────────────────────────────────────────────
# InvariantSet — root of the hierarchy
# ─────────────────────────────────────────────────────────────────────────────

class InvariantSet(ABC):
    """Abstract base for any invariant geometric object of a dynamical system.

    An invariant set *S* satisfies φ^t(S) ⊆ S for all t (continuous flow) or
    P(S) = S (discrete map).

    Design: Pure interface — no ``__init__``, no state fields.
    Concrete subclasses (including dataclasses) provide their own fields
    and satisfy the interface via ``@property`` overrides.

    ``section_cut`` and ``diagnostics`` are **no longer abstract**.  Not every
    invariant set can be meaningfully cut by a section (e.g. a KAM torus in a
    6N-dim N-body phase space may have no natural Poincaré section).  Use the
    :class:`SectionCuttable` mixin to mark objects that *can* be sectioned.

    Optional interface (all have sensible defaults):
      .label           @property -> str | None
      .ambient_dim     @property -> int | None
      .poincare_map    @property -> PoincareMap | None
      .phase_space     @property -> PhaseSpace | None
      .diagnostics()   -> dict
      .section_cut(s)  -> list     (raises NotImplementedError by default)
    """

    # ── Optional concrete interface ───────────────────────────────────────────

    @property
    def label(self) -> Optional[str]:
        """Human-readable identifier.  Override in subclasses."""
        return None

    @property
    def ambient_dim(self) -> Optional[int]:
        """Dimension of the ambient phase space (None if unknown)."""
        return None

    @property
    def poincare_map(self):
        """The Poincaré map this object lives in (PoincareMap | None)."""
        return None

    @property
    def phase_space(self):
        """Phase space of the associated map (PhaseSpace | None)."""
        pm = self.poincare_map
        if pm is not None:
            return pm.phase_space
        return None

    def diagnostics(self) -> Dict[str, Any]:
        """Return a structured diagnostic / debug dict.

        The default implementation returns a minimal dict with just the class
        name.  Subclasses are encouraged to override and add domain-specific
        information.
        """
        return {"invariant_type": self.__class__.__name__}

    def section_cut(self, section=None) -> list:
        """Return the intersection of this invariant with a Poincaré section.

        Raises ``NotImplementedError`` by default.  Subclasses that support
        sectioning should override (and ideally also derive from
        :class:`SectionCuttable`).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support section_cut(). "
            "Derive from SectionCuttable and override."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(label={self.label!r})"


# ── Backward-compatible alias ─────────────────────────────────────────────────
InvariantSet = InvariantSet
"""Backward-compatible alias.  Prefer ``InvariantSet`` in new code."""


# ─────────────────────────────────────────────────────────────────────────────
# SectionCuttable — mixin protocol
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class SectionCuttable(Protocol):
    """Protocol / mixin marking objects that can be sliced by a Poincaré section.

    Any class that implements ``section_cut(section) -> list`` satisfies this
    protocol.  It is deliberately **not** a subclass of InvariantSet so that
    it can be mixed in freely (e.g. ``class Tube(InvariantSet, SectionCuttable)``).

    Usage::

        if isinstance(obj, SectionCuttable):
            pts = obj.section_cut(my_section)
    """

    def section_cut(self, section=None) -> list: ...


# ─────────────────────────────────────────────────────────────────────────────
# InvariantManifold — intermediate layer with intrinsic dimension
# ─────────────────────────────────────────────────────────────────────────────

class InvariantManifold(InvariantSet):
    """An invariant set with a well-defined intrinsic dimension.

    ``InvariantManifold`` sits between the abstract :class:`InvariantSet` and
    the concrete geometry types (``FixedPoint``, ``PeriodicOrbit``,
    ``InvariantTorus``, ``StableManifold``, …).

    The key additional concept is *intrinsic dimension*:

    - ``FixedPoint``  →  intrinsic_dim = 0
    - ``Orbit`` / ``PeriodicOrbit``  →  intrinsic_dim = 0  (finite set of map points)
    - ``Trajectory`` / ``Cycle``  →  intrinsic_dim = 1  (continuous curve)
    - ``InvariantTorus`` of order *k*  →  intrinsic_dim = k
    - ``StableManifold`` of a hyperbolic orbit  →  intrinsic_dim = dim(E^s)

    The codimension is ``ambient_dim - intrinsic_dim`` when ``ambient_dim`` is
    known.

    Subclasses must set *intrinsic_dim* (as a class attribute, ``__init__``
    argument, or ``@property``).  The default is ``None`` (unknown).
    """

    @property
    def intrinsic_dim(self) -> Optional[int]:
        """Intrinsic dimension of the invariant manifold (None if unknown)."""
        return None

    @property
    def codim(self) -> Optional[int]:
        """Codimension = ambient_dim − intrinsic_dim (None if either is unknown)."""
        a = self.ambient_dim
        i = self.intrinsic_dim
        if a is not None and i is not None:
            return a - i
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory — continuous-time solution curve
# ─────────────────────────────────────────────────────────────────────────────

class Trajectory(InvariantManifold):
    """A continuous-time trajectory: a 1-D solution curve in phase space.

    A ``Trajectory`` is the image of a time-parametrised solution
    ``t ↦ φ^t(x₀)`` of a continuous-flow ODE.  It can be:

    - **Open** (finite-time integration, e.g. a field line between walls)
    - **Closed** (periodic orbit of the flow, a.k.a. :class:`Cycle`)

    Concrete subtypes
    -----------------
    - :class:`~pyna.topo.trajectory3d.Trajectory3D`
      — Generic 3-D trajectory stored as (param, coords) arrays.
      - :class:`~pyna.topo.trajectory3d.Trajectory3DToroidal`
        — Toroidal variant stored as (R, Z, phi) arrays.
    - :class:`~pyna.topo.invariants.Cycle`
      — Closed periodic orbit of a continuous flow, stored as a dict of
      Poincaré-section crossing lists plus an optional 3-D trajectory.

    ``intrinsic_dim = 1`` (a curve).
    """

    @property
    def intrinsic_dim(self) -> int:
        return 1


# ─────────────────────────────────────────────────────────────────────────────
# Orbit — periodic orbit of a discrete map
# ─────────────────────────────────────────────────────────────────────────────

class Orbit(InvariantManifold):
    """A periodic orbit of a **discrete map**: a finite set of phase-space points.

    Mathematically, an *m*-periodic orbit is a set ``{x₁, …, xₘ}`` satisfying
    ``P(xₖ) = xₖ₊₁`` (indices mod m) and ``P^m(xₖ) = xₖ`` for the map *P*.

    An ``Orbit`` is fundamentally different from a :class:`Trajectory`:

    - ``Trajectory`` — solution of a *continuous* ODE (a curve, dim=1)
    - ``Orbit``      — fixed-point set of a *discrete* map (finite points, dim=0)

    Concrete subtype
    ----------------
    - :class:`~pyna.topo.invariants.PeriodicOrbit`
      — Stores ``m`` :class:`FixedPoint` objects, one per map iteration.

    ``intrinsic_dim = 0`` (a finite set of isolated points in the section).
    """

    @property
    def intrinsic_dim(self) -> int:
        return 0
