"""Core dynamical system abstractions for pyna.

This module defines the class hierarchy for dynamical systems,
from general abstract base classes down to physically motivated
special cases.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DynamicalSystem(ABC):
    """Abstract base for all dynamical systems.

    A dynamical system defines the evolution law for a state vector.
    Subclasses must implement :meth:`state_dim` and :meth:`__call__`.
    """

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of the state space."""
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the system (right-hand side or map step)."""
        raise NotImplementedError


class NonAutonomousDynamicalSystem(DynamicalSystem):
    """Non-autonomous dynamical system: dx/dt = f(x, t).

    Time appears explicitly on the right-hand side.  Geometric
    structures such as Poincaré maps depend on the phase of the
    driving and are therefore more complex to analyse than their
    autonomous counterparts.
    """

    pass


class AutonomousDynamicalSystem(DynamicalSystem):
    """Autonomous dynamical system: dx/dt = f(x).

    The right-hand side does not depend explicitly on time.
    Key consequences:

    * Poincaré section maps are well-defined without a time reference.
    * Invariant manifolds, fixed points, and KAM tori are
      time-independent geometric objects in state space.
    """

    pass


class VectorField(AutonomousDynamicalSystem):
    """Autonomous dynamical system defined by a smooth vector field.

    The vector field **is** the right-hand side f(x) of dx/dt = f(x).
    Subclasses specialise to a particular spatial dimension.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Spatial dimension of the vector field."""
        raise NotImplementedError

    @property
    def state_dim(self) -> int:  # satisfies DynamicalSystem contract
        return self.dim


class VectorField1D(VectorField):
    """One-dimensional vector field: dx/dt = f(x)."""

    @property
    def dim(self) -> int:
        return 1


class VectorField2D(VectorField):
    """Two-dimensional vector field: dx/dt = f(x), x ∈ ℝ².

    The 2-D case is where KAM theory is most transparent and
    area-preserving (Hamiltonian) maps are most widely studied.
    """

    @property
    def dim(self) -> int:
        return 2


class _LegacyVectorField3D(VectorField):
    """Three-dimensional vector field: dx/dt = f(x), x ∈ ℝ³.

    This is the most important special case for plasma physics:

    * Admits a 2-D Poincaré section map — the most widely used
      diagnostic tool in magnetic confinement topology.
    * Magnetic field-line tracing is integration of a 3-D vector
      field with the toroidal angle φ playing the role of 'time'.
    """

    @property
    def dim(self) -> int:
        return 3

    def poincare_map_2d(self, section_value: float, section_coord: int = 2) -> Any:
        """Return a 2-D Poincaré map by integrating until section crossing.

        Parameters
        ----------
        section_value:
            Value of the section coordinate at which intersections
            are recorded (e.g. φ = 0 for a toroidal Poincaré section).
        section_coord:
            Index (0, 1, or 2) of the coordinate used as the section.
            Default is 2 (toroidal angle φ in cylindrical coordinates).

        Returns
        -------
        A callable map object or array of intersection points,
        depending on the concrete implementation.
        """
        raise NotImplementedError


class VectorField4D(VectorField):
    """Four-dimensional vector field: dx/dt = f(x), x ∈ ℝ⁴.

    Relevant for relativistic charged-particle dynamics (guiding-centre
    equations with energy as a fourth coordinate) and certain
    Hamiltonian systems with two degrees of freedom.
    """

    @property
    def dim(self) -> int:
        return 4


class _LegacyVectorField3DAxiSymmetric(_LegacyVectorField3D):
    """Axisymmetric 3-D vector field (no φ dependence in cylindrical coords).

    Special case of :class:`VectorField3D` where the field components
    B_R, B_Z, B_φ depend only on (R, Z), not on the toroidal angle φ.

    Consequence for topology:
    * The Poincaré section map is identical for *any* choice of the
      toroidal section angle φ₀ — the field is invariant under rotation.
    * Flux surfaces are tori of revolution; their cross-sections in the
      (R, Z) plane are isolines of the poloidal flux ψ.
    """

    pass


# ── Canonical names point to pyna.fields.base ─────────────────────────────────
# CylindricalVectorField3D inherits from fields.base.VectorField3D, so
# isinstance(field, VectorField3D) works for all cylindrical fields.
from pyna.fields.base import (  # noqa: E402
    VectorField3D as VectorField3D,
    VectorField3D as VectorField3DAxiSymmetric,  # compat alias
    VectorField as _FieldsVectorField,
)

# Register fields.base classes as virtual subclasses of system.VectorField
# so isinstance(cylindrical_field, system.VectorField) returns True.
VectorField.register(_FieldsVectorField)