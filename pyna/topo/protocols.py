"""Structural protocols for pyna's topology and dynamics layers.

The concrete class hierarchy intentionally stays small:
``pyna.topo.core`` owns domain-agnostic geometry and ``pyna.topo.toroidal``
owns magnetic-confinement specializations.  Protocols in this module define
the structural contracts used by adapters, builders, bridges and factories.
They make extension points explicit without forcing third-party systems to
subclass pyna internals.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from pyna.topo.dynamics import PhaseSpace


@runtime_checkable
class FlowLike(Protocol):
    """Continuous-time system exposing a vector field."""

    @property
    def phase_space(self) -> PhaseSpace: ...

    def vector_field(self, x: np.ndarray, t: float = 0.0) -> np.ndarray: ...


@runtime_checkable
class MapLike(Protocol):
    """Discrete-time map exposing one-step iteration."""

    @property
    def phase_space(self) -> PhaseSpace: ...

    def step(self, x: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class JacobianProvider(Protocol):
    """Object that can provide a local derivative matrix."""

    def jacobian(self, x: Sequence[float], **kwargs: Any) -> np.ndarray: ...


@runtime_checkable
class SectionLike(Protocol):
    """Codimension-one section used for cuts and Poincare maps."""

    @property
    def dim_phase(self) -> int: ...

    @property
    def label(self) -> Optional[str]: ...

    def detect_crossing(
        self,
        x_prev: np.ndarray,
        x_curr: np.ndarray,
        tol: float = 1e-10,
    ) -> Optional[np.ndarray]: ...

    def project(self, x: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class StabilityClassifier(Protocol):
    """Callable linear-stability classifier."""

    def __call__(self, jacobian: np.ndarray, eigenvalues: Optional[np.ndarray] = None) -> Any: ...


__all__ = [
    "FlowLike",
    "MapLike",
    "JacobianProvider",
    "SectionLike",
    "StabilityClassifier",
]
