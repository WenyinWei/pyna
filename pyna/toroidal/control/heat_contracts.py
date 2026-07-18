"""Small, backend-neutral contracts for boundary heat forward models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyna.toroidal.control.boundary_plasma_response import BoundaryPlasmaResponseInput
    from pyna.toroidal.control.boundary_topology_cases import BoundaryTopologyCase
    from pyna.toroidal.perturbation_spectrum import (
        ChaoticLayerInterval,
        RadialPerturbationFourierSpectrum,
        ResonantIslandChain,
    )


@dataclass(frozen=True)
class BoundaryTopologyHeatState:
    """Wall heat map and physical bin coordinates returned by a heat backend."""

    heat: np.ndarray
    phi_values: np.ndarray
    s_values: np.ndarray
    cell_areas: np.ndarray | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        heat = np.asarray(self.heat, dtype=float)
        phi = np.asarray(self.phi_values, dtype=float).ravel()
        s = np.asarray(self.s_values, dtype=float).ravel()
        if heat.ndim != 2 or heat.shape != (phi.size, s.size):
            raise ValueError("heat must have shape (len(phi_values), len(s_values))")
        area = None if self.cell_areas is None else np.asarray(self.cell_areas, dtype=float)
        if area is not None and area.shape != heat.shape:
            raise ValueError("cell_areas must match heat shape")
        object.__setattr__(self, "heat", heat)
        object.__setattr__(self, "phi_values", phi)
        object.__setattr__(self, "s_values", s)
        object.__setattr__(self, "cell_areas", area)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


class BoundaryTopologyHeatForwardModel(Protocol):
    """Protocol for reduced, traced, or transport-coupled wall heat backends."""

    def evaluate(
        self,
        case: "BoundaryTopologyCase",
        request: "BoundaryPlasmaResponseInput",
        spectrum: "RadialPerturbationFourierSpectrum",
        chains: Sequence["ResonantIslandChain"],
        intervals: Sequence["ChaoticLayerInterval"],
    ) -> BoundaryTopologyHeatState: ...


@dataclass(frozen=True)
class CallableBoundaryTopologyHeatForwardModel:
    """Wrap an external tracer or transport solver as a heat forward model."""

    evaluator: Callable[..., BoundaryTopologyHeatState | Mapping[str, object]]

    def evaluate(
        self,
        case: "BoundaryTopologyCase",
        request: "BoundaryPlasmaResponseInput",
        spectrum: "RadialPerturbationFourierSpectrum",
        chains: Sequence["ResonantIslandChain"],
        intervals: Sequence["ChaoticLayerInterval"],
    ) -> BoundaryTopologyHeatState:
        value = self.evaluator(case, request, spectrum, chains, intervals)
        if isinstance(value, BoundaryTopologyHeatState):
            return value
        if isinstance(value, Mapping):
            return BoundaryTopologyHeatState(**value)
        raise TypeError(
            "external heat evaluator must return BoundaryTopologyHeatState or a mapping"
        )


__all__ = [
    "BoundaryTopologyHeatForwardModel",
    "BoundaryTopologyHeatState",
    "CallableBoundaryTopologyHeatForwardModel",
]
