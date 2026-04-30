"""pyna.topo.section -- Abstract Poincaré section and concrete implementations.

Design philosophy
=================
A *Section* is a codimension-1 hypersurface in phase space used to define
a Poincaré map. The phase space may be arbitrary finite-dimensional.

Concrete sections for MCF:
  ToroidalSection   phi = const  (reduced section coords [R, Z])
  HyperplaneSection a·x = c      (any dimension)
  ParametricSection f(x) = 0     (any dimension, optionally bounded)

The abstraction decouples dynamics from cut geometry, so the same
Trajectory/Cycle/Tube machinery can work with infinite hyperplanes, bounded
patches of hyperplanes, curved finite surfaces, or arbitrary hypersurfaces.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence
from numbers import Real

import numpy as np


class Section(ABC):
    """Abstract Poincaré section (codimension-1 hypersurface in phase space)."""

    @property
    @abstractmethod
    def dim_phase(self) -> int:
        """Dimension of the ambient phase space."""

    @property
    def dim_section(self) -> int:
        return self.dim_phase - 1

    @abstractmethod
    def f(self, x: np.ndarray) -> float:
        """Defining function: section = {x | f(x) = 0}."""

    @abstractmethod
    def contains(self, x: np.ndarray, tol: float = 1e-8) -> bool:
        """Return True if point x is on this section (within tol)."""

    @abstractmethod
    def project(self, x: np.ndarray) -> np.ndarray:
        """Project phase-space point x to a section-level coordinate representation."""

    @abstractmethod
    def normal(self, x: np.ndarray) -> np.ndarray:
        """Normal vector to the section at x."""

    def accepts(self, x: np.ndarray) -> bool:
        """Optional finite-extent acceptance test for bounded sections."""
        return True

    def detect_crossing(self, x_prev: np.ndarray, x_curr: np.ndarray, tol: float = 1e-10) -> Optional[np.ndarray]:
        """Detect a segment crossing and return an interpolated intersection point.

        Default behaviour works for any section defined by a scalar function
        ``f(x)=0``. Subclasses may override when a more natural intersection
        calculation exists.
        """
        x_prev = np.asarray(x_prev, dtype=float)
        x_curr = np.asarray(x_curr, dtype=float)
        try:
            f_prev = self.f(x_prev)
            f_curr = self.f(x_curr)
        except NotImplementedError:
            return None

        if abs(f_prev) < tol and self.accepts(x_prev):
            return x_prev.copy()
        if abs(f_curr) < tol and self.accepts(x_curr):
            return x_curr.copy()
        if f_prev * f_curr > 0:
            return None

        denom = abs(f_prev) + abs(f_curr)
        t = 0.5 if denom < 1e-30 else abs(f_prev) / denom
        x_hit = x_prev + t * (x_curr - x_prev)
        if not self.accepts(x_hit):
            return None
        return x_hit

    def is_transverse(self, x: np.ndarray, v: np.ndarray, tol: float = 1e-10) -> bool:
        return abs(float(np.dot(self.normal(x), v))) > tol

    def crossing_direction(self, x: np.ndarray, v: np.ndarray) -> int:
        return 1 if float(np.dot(self.normal(x), v)) > 0 else -1

    @property
    def label(self) -> Optional[str]:
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim_phase}, label={self.label!r})"


@dataclass(frozen=True)
class ToroidalSection(Section):
    """Poincaré section phi = const in toroidal (R, Z, phi) coordinates."""

    phi: float
    _label: Optional[str] = field(default=None, compare=False)

    @property
    def dim_phase(self) -> int:
        return 2

    @property
    def label(self) -> Optional[str]:
        if self._label is not None:
            return self._label
        from fractions import Fraction
        import math
        frac = Fraction(self.phi / math.pi).limit_denominator(16)
        if frac.numerator == 0:
            return r"φ=0"
        if frac.denominator == 1:
            return rf"φ={frac.numerator}π"
        return rf"φ={frac.numerator}π/{frac.denominator}"

    def f(self, x: np.ndarray) -> float:
        raise NotImplementedError("ToroidalSection.f() is undefined in reduced (R,Z) coordinates.")

    def contains(self, x: np.ndarray, tol: float = 1e-8) -> bool:
        return True

    def project(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)

    def normal(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("ToroidalSection normal points along the eliminated phi direction.")

    def __str__(self) -> str:
        return self.label or f"φ={self.phi:.4f}"


@dataclass(frozen=True)
class HyperplaneSection(Section):
    """Poincaré section defined by a·x = c in n-dimensional phase space."""

    normal_vec: np.ndarray
    offset: float
    phase_dim: int
    accept_func: Optional[Callable[[np.ndarray], bool]] = field(default=None, compare=False)
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
        return abs(self.f(x)) < tol and self.accepts(x)

    def project(self, x: np.ndarray) -> np.ndarray:
        n = self.normal_vec / (np.linalg.norm(self.normal_vec) + 1e-30)
        return np.asarray(x, dtype=float) - float(np.dot(n, x)) * n

    def normal(self, x: np.ndarray) -> np.ndarray:
        return self.normal_vec.copy()

    def accepts(self, x: np.ndarray) -> bool:
        if self.accept_func is None:
            return True
        return bool(self.accept_func(np.asarray(x, dtype=float)))


@dataclass(frozen=True)
class ParametricSection(Section):
    """Poincaré section defined by an arbitrary scalar function f(x) = 0."""

    f_func: Any
    grad_func: Any
    phase_dim: int
    project_func: Any = field(default=None)
    accept_func: Any = field(default=None)
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
        return abs(self.f(x)) < tol and self.accepts(x)

    def project(self, x: np.ndarray) -> np.ndarray:
        if self.project_func is not None:
            return np.asarray(self.project_func(x), dtype=float)
        return np.asarray(x, dtype=float)

    def normal(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.grad_func(x), dtype=float)

    def accepts(self, x: np.ndarray) -> bool:
        if self.accept_func is None:
            return True
        return bool(self.accept_func(np.asarray(x, dtype=float)))


def coerce_section(section: Any) -> Section:
    """Normalize user input into a first-class :class:`Section`.

    Rules
    -----
    - ``float`` / ``int`` → :class:`ToroidalSection`
    - existing :class:`Section` instance → returned unchanged
    - anything else → ``TypeError``

    This intentionally rejects legacy duck-typed objects carrying a ``phi``
    attribute but not implementing the Section protocol.
    """
    if isinstance(section, Section):
        return section
    if isinstance(section, Real):
        return ToroidalSection(float(section))
    raise TypeError(
        "section must be a Section instance or numeric toroidal angle; "
        f"got {type(section)!r}"
    )



def coerce_toroidal_section(section: Any) -> ToroidalSection:
    """Normalize input into a concrete :class:`ToroidalSection`.

    Use this at toroidal-only boundaries where generic sections are not
    semantically meaningful.
    """
    section_obj = coerce_section(section)
    if isinstance(section_obj, ToroidalSection):
        return section_obj
    raise TypeError(
        "expected a ToroidalSection or numeric toroidal angle; "
        f"got {type(section_obj)!r}"
    )



def toroidal_sections(phi_list: Sequence[float]) -> list[ToroidalSection]:
    return [ToroidalSection(float(phi)) for phi in phi_list]


HAO_SECTIONS = toroidal_sections([0.0, __import__('math').pi/4,
                                  __import__('math').pi/2,
                                  3*__import__('math').pi/4])
