"""pyna.topo.section -- Abstract Poincaré section and concrete implementations.

Design philosophy
=================
A *Section* is a codimension-1 hypersurface in phase space used to define
a Poincaré map.  The phase space may be arbitrary finite-dimensional (n-dim).

Concrete sections for MCF:
  ToroidalSection   φ = const  (dim_phase=2, [R,Z])
  HyperplaneSection a·x = c   (any dimension)

The Section abstraction decouples the dynamics from the geometry of the
cut, enabling the same Tube/Island machinery to work for:
  - 2D Poincaré maps in tokamak/stellarator (ToroidalSection)
  - Higher-dimensional maps (coupled oscillators, guiding-centre maps, …)

Relationship to existing classes
---------------------------------
  Tube.cut(section) → list[CutPoint]
  TubeChain.section_view(section) → SectionView
  Island.section: Section (back-reference set by SectionView)

All existing code that passes ``phi: float`` implicitly uses
``ToroidalSection(phi)``.  New code should use Section objects explicitly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Section(ABC):
    """Abstract Poincaré section (codimension-1 hypersurface in phase space).

    A Section is defined by the equation  f(x) = 0  for some smooth
    scalar function f.  The *transversality condition* requires that the
    flow vector is not tangent to the section at intersection points.

    Attributes
    ----------
    dim_phase : int
        Dimension of the full phase space.
    dim_section : int
        Dimension of the section (= dim_phase - 1).
    label : str or None
        Human-readable identifier (e.g. ``"φ=0"`` or ``"z=0"``).
    """

    @property
    @abstractmethod
    def dim_phase(self) -> int:
        """Dimension of the ambient phase space."""

    @property
    def dim_section(self) -> int:
        """Dimension of the section surface (= dim_phase - 1)."""
        return self.dim_phase - 1

    @abstractmethod
    def f(self, x: np.ndarray) -> float:
        """Defining function: section = {x | f(x) = 0}."""

    @abstractmethod
    def contains(self, x: np.ndarray, tol: float = 1e-8) -> bool:
        """Return True if point x is on this section (within tol)."""

    @abstractmethod
    def project(self, x: np.ndarray) -> np.ndarray:
        """Project phase-space point x → intrinsic section coordinates.

        Returns an array of shape (dim_section,).
        For ToroidalSection this returns (R, Z).
        """

    @abstractmethod
    def normal(self, x: np.ndarray) -> np.ndarray:
        """Outward normal vector to the section at x (shape: (dim_phase,))."""

    def is_transverse(self, x: np.ndarray, v: np.ndarray, tol: float = 1e-10) -> bool:
        """Check that flow vector v is transverse (not tangent) to the section."""
        return abs(float(np.dot(self.normal(x), v))) > tol

    def crossing_direction(self, x: np.ndarray, v: np.ndarray) -> int:
        """Return +1 if trajectory crosses section in positive normal direction, -1 otherwise."""
        return 1 if float(np.dot(self.normal(x), v)) > 0 else -1

    @property
    def label(self) -> Optional[str]:
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim_phase}, label={self.label!r})"


# ---------------------------------------------------------------------------
# Concrete sections
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToroidalSection(Section):
    """Poincaré section φ = const in toroidal (R, Z, φ) coordinates.

    The phase space is 2D after projecting out φ: coordinates are (R, Z).
    This is the standard section used in MCF field-line tracing.

    Parameters
    ----------
    phi : float
        Toroidal angle of the section [rad].
    """
    phi: float
    _label: Optional[str] = field(default=None, compare=False)

    @property
    def dim_phase(self) -> int:
        return 2   # (R, Z) after removing the cyclic φ coordinate

    @property
    def label(self) -> Optional[str]:
        if self._label is not None:
            return self._label
        from fractions import Fraction
        import math
        # Express phi as a multiple of π
        frac = Fraction(self.phi / math.pi).limit_denominator(16)
        if frac.numerator == 0:
            return r"φ=0"
        if frac.denominator == 1:
            return rf"φ={frac.numerator}π"
        return rf"φ={frac.numerator}π/{frac.denominator}"

    def f(self, x: np.ndarray) -> float:
        """Not meaningful for ToroidalSection (φ is external, not in (R,Z))."""
        raise NotImplementedError("ToroidalSection.f() is undefined in (R,Z) phase space.")

    def contains(self, x: np.ndarray, tol: float = 1e-8) -> bool:
        """Always True for points in (R,Z): they're on this section by definition."""
        return True

    def project(self, x: np.ndarray) -> np.ndarray:
        """Identity: (R, Z) are already the section coordinates."""
        return np.asarray(x, dtype=float)

    def normal(self, x: np.ndarray) -> np.ndarray:
        """The normal to a φ=const section in (R,Z) is undefined (external coordinate)."""
        raise NotImplementedError("ToroidalSection normal is in the φ-direction, external to (R,Z) space.")

    def __str__(self) -> str:
        return self.label or f"φ={self.phi:.4f}"


@dataclass(frozen=True)
class HyperplaneSection(Section):
    """Poincaré section defined by a·x = c (hyperplane in n-dim phase space).

    Parameters
    ----------
    normal_vec : array-like, shape (n,)
        Normal vector to the hyperplane (not necessarily unit).
    offset : float
        Value of a·x on the plane.
    phase_dim : int
        Dimension of the phase space.
    """
    normal_vec: np.ndarray
    offset: float
    phase_dim: int
    _label: Optional[str] = field(default=None, compare=False)

    def __post_init__(self):
        object.__setattr__(self, 'normal_vec', np.asarray(self.normal_vec, dtype=float))

    @property
    def dim_phase(self) -> int:
        return self.phase_dim

    @property
    def label(self) -> Optional[str]:
        return self._label

    def f(self, x: np.ndarray) -> float:
        return float(np.dot(self.normal_vec, x)) - self.offset

    def contains(self, x: np.ndarray, tol: float = 1e-8) -> bool:
        return abs(self.f(x)) < tol

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project x onto the hyperplane using Gram-Schmidt (removes normal component)."""
        n = self.normal_vec / (np.linalg.norm(self.normal_vec) + 1e-30)
        return np.asarray(x, dtype=float) - float(np.dot(n, x)) * n

    def normal(self, x: np.ndarray) -> np.ndarray:
        return self.normal_vec.copy()


@dataclass(frozen=True)
class ParametricSection(Section):
    """Poincaré section defined by an arbitrary scalar function f(x) = 0.

    Parameters
    ----------
    f_func : callable
        Scalar function defining the section: x → float.
    grad_func : callable
        Gradient of f_func: x → ndarray of shape (dim,).
    phase_dim : int
        Dimension of the phase space.
    project_func : callable, optional
        Custom projection x → section coordinates.  If None, identity is used.
    """
    f_func: Any          # callable: x → float
    grad_func: Any       # callable: x → ndarray
    phase_dim: int
    project_func: Any = field(default=None)   # callable: x → ndarray | None
    _label: Optional[str] = field(default=None, compare=False)

    @property
    def dim_phase(self) -> int:
        return self.phase_dim

    @property
    def label(self) -> Optional[str]:
        return self._label

    def f(self, x: np.ndarray) -> float:
        return float(self.f_func(x))

    def contains(self, x: np.ndarray, tol: float = 1e-8) -> bool:
        return abs(self.f(x)) < tol

    def project(self, x: np.ndarray) -> np.ndarray:
        if self.project_func is not None:
            return np.asarray(self.project_func(x), dtype=float)
        return np.asarray(x, dtype=float)

    def normal(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.grad_func(x), dtype=float)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def toroidal_sections(phi_list: Sequence[float]) -> list[ToroidalSection]:
    """Create a list of ToroidalSection objects from a list of phi values."""
    return [ToroidalSection(float(phi)) for phi in phi_list]


# Standard HAO/W7X sections
HAO_SECTIONS = toroidal_sections([0.0, __import__('math').pi/4,
                                   __import__('math').pi/2,
                                   3*__import__('math').pi/4])
