"""Abstract base classes for the pyna field hierarchy.

Design principles
-----------------
* A *field* is a function f: Domain → Codomain.
  - Domain dimension: 1, 2, 3, or 4.
  - Codomain rank: 0 (scalar), 1 (vector), 2 (rank-2 tensor).
* Fields carry a ``properties`` attribute (FieldProperty flag set) that
  downstream code can query before performing operations.
* Concrete classes implement ``__call__(coords)`` and ``domain_dim``.
* Differential operations (grad, div, curl, ...) live in ``diff_ops.py``
  and return new Field instances, propagating properties automatically.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np
from pyna.fields.properties import FieldProperty


class Field(ABC):
    """Abstract base for all fields.

    Parameters
    ----------
    properties : FieldProperty
        Mathematical properties of this field.
    name : str
        Human-readable name (used in repr and plot labels).
    units : str
        Physical units string (e.g. 'T', 'T/m', 'Pa').
    """

    def __init__(
        self,
        properties: FieldProperty = FieldProperty.NONE,
        name: str = "",
        units: str = "",
    ) -> None:
        self._properties = properties
        self.name = name
        self.units = units

    @property
    @abstractmethod
    def domain_dim(self) -> int:
        """Spatial dimension of the domain."""

    @property
    @abstractmethod
    def range_rank(self) -> int:
        """Tensor rank of the range: 0=scalar, 1=vector, 2=matrix, ..."""

    @property
    def properties(self) -> FieldProperty:
        return self._properties

    def has_property(self, prop: FieldProperty) -> bool:
        return bool(self._properties & prop)

    @abstractmethod
    def __call__(self, coords: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate field at given coordinates.

        Parameters
        ----------
        coords : ndarray, shape (..., domain_dim)
            Evaluation points.

        Returns
        -------
        ndarray
            Shape (...) for scalars, (..., domain_dim) for vectors,
            (..., domain_dim, domain_dim) for rank-2 tensors.
        """

    def __repr__(self) -> str:
        cls = type(self).__name__
        prop_str = str(self._properties) if self._properties else "none"
        return f"{cls}(name={self.name!r}, units={self.units!r}, properties={prop_str})"


# ── Dimension-specialised abstract classes ───────────────────────────────────

class ScalarField(Field):
    """Abstract scalar field (range rank = 0)."""
    @property
    def range_rank(self) -> int: return 0


class VectorField(Field):
    """Abstract vector field (range rank = 1)."""
    @property
    def range_rank(self) -> int: return 1


class TensorField(Field):
    """Abstract tensor field (range rank ≥ 2)."""


# 1-D through 4-D scalar fields
class ScalarField1D(ScalarField):
    @property
    def domain_dim(self) -> int: return 1


class ScalarField2D(ScalarField):
    @property
    def domain_dim(self) -> int: return 2


class ScalarField3D(ScalarField):
    @property
    def domain_dim(self) -> int: return 3


class ScalarField4D(ScalarField):
    @property
    def domain_dim(self) -> int: return 4


# 1-D through 4-D vector fields
class VectorField1D(VectorField):
    @property
    def domain_dim(self) -> int: return 1


class VectorField2D(VectorField):
    @property
    def domain_dim(self) -> int: return 2


class VectorField3D(VectorField):
    @property
    def domain_dim(self) -> int: return 3

    @property
    def dim(self) -> int:
        """Alias for domain_dim — compatibility with pyna.system.VectorField3D."""
        return self.domain_dim

    @property
    def state_dim(self) -> int:
        """Alias for domain_dim — satisfies DynamicalSystem-like contracts."""
        return self.domain_dim

    def poincare_map_2d(self, section_value: float, section_coord: int = 2) -> Any:
        """Return the 2-D Poincaré map at a given section. To be overridden."""
        raise NotImplementedError


class VectorField4D(VectorField):
    @property
    def domain_dim(self) -> int: return 4


# Rank-2 tensor field on 3-D domain
class TensorField3D_rank2(TensorField):
    """Rank-2 tensor field T_ij(R,Z,φ).

    Concrete arrays stored as shape (nR, nZ, nPhi, 3, 3) — spatial axes first,
    tensor indices last (PyTorch/JAX convention).
    """
    @property
    def domain_dim(self) -> int: return 3

    @property
    def range_rank(self) -> int: return 2
