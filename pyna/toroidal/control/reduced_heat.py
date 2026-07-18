"""Lightweight spectral wall-heat surrogate for optimizer screening."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence, TYPE_CHECKING

import numpy as np

from pyna.toroidal.control.heat_contracts import BoundaryTopologyHeatState

if TYPE_CHECKING:
    from pyna.toroidal.control.boundary_plasma_response import BoundaryPlasmaResponseInput
    from pyna.toroidal.control.boundary_topology_cases import BoundaryTopologyCase
    from pyna.toroidal.perturbation_spectrum import (
        ChaoticLayerInterval,
        RadialPerturbationFourierSpectrum,
        ResonantIslandChain,
    )


TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class ReducedSpectralHeatModel:
    """Fast non-quantitative strike-footprint surrogate.

    This model is intended for screening and optimizer development.  Traced
    or transport-coupled backends implement the same heat-forward protocol for
    final verification.
    """

    phi_values: np.ndarray
    s_values: np.ndarray
    base_total_power: float = 1.0
    base_center_s: float = 0.55
    base_sigma_s: float = 0.055
    phase_excursion_s: float = 0.12
    toroidal_modulation: float = 0.15
    chaos_broadening: float = 0.35
    island_power_gain: float = 0.6
    chaos_power_gain: float = 0.8
    control_center_s: Mapping[str, float] = field(default_factory=dict)
    control_sigma_s: Mapping[str, float] = field(default_factory=dict)
    control_power_fraction: Mapping[str, float] = field(default_factory=dict)
    minimum_sigma_s: float = 1.0e-3

    def __post_init__(self) -> None:
        phi = np.asarray(self.phi_values, dtype=float).ravel()
        s = np.asarray(self.s_values, dtype=float).ravel()
        if phi.size < 2 or s.size < 3 or np.any(np.diff(s) <= 0.0):
            raise ValueError("heat model requires at least two phi bins and increasing s_values")
        if float(self.base_total_power) <= 0.0 or float(self.base_sigma_s) <= 0.0:
            raise ValueError("base_total_power and base_sigma_s must be positive")
        object.__setattr__(self, "phi_values", phi)
        object.__setattr__(self, "s_values", s)
        object.__setattr__(self, "control_center_s", dict(self.control_center_s or {}))
        object.__setattr__(self, "control_sigma_s", dict(self.control_sigma_s or {}))
        object.__setattr__(self, "control_power_fraction", dict(self.control_power_fraction or {}))

    @staticmethod
    def _control_term(
        mapping: Mapping[str, float],
        request: "BoundaryPlasmaResponseInput",
    ) -> float:
        values = {
            label: float(value)
            for label, value in zip(request.control_labels, request.controls)
        }
        return float(
            sum(
                float(coefficient) * values.get(str(label), 0.0)
                for label, coefficient in mapping.items()
            )
        )

    def evaluate(
        self,
        case: "BoundaryTopologyCase",
        request: "BoundaryPlasmaResponseInput",
        spectrum: "RadialPerturbationFourierSpectrum",
        chains: Sequence["ResonantIslandChain"],
        intervals: Sequence["ChaoticLayerInterval"],
    ) -> BoundaryTopologyHeatState:
        del spectrum
        chain_tuple = tuple(chains)
        interval_tuple = tuple(intervals)
        dominant = max(chain_tuple, key=lambda chain: float(chain.half_width), default=None)
        chaos_width = float(sum(max(0.0, interval.width) for interval in interval_tuple))
        island_width = float(sum(max(0.0, chain.half_width) for chain in chain_tuple))
        center_shift = self._control_term(self.control_center_s, request)
        sigma_shift = self._control_term(self.control_sigma_s, request)
        power_shift = self._control_term(self.control_power_fraction, request)
        sigma = max(
            float(self.minimum_sigma_s),
            float(self.base_sigma_s) + float(self.chaos_broadening) * chaos_width + sigma_shift,
        )
        total_power = float(self.base_total_power) * max(
            0.02,
            1.0
            + float(self.island_power_gain) * island_width
            + float(self.chaos_power_gain) * chaos_width
            + power_shift,
        )
        phase = 0.0 if dominant is None else float(np.angle(dominant.coefficient))
        toroidal_n = case.nfp if dominant is None else max(1, abs(int(dominant.n)))
        center = (
            float(self.base_center_s)
            + center_shift
            + float(self.phase_excursion_s) * np.sin(toroidal_n * self.phi_values + phase)
        )
        _, SS = np.meshgrid(self.phi_values, self.s_values, indexing="ij")
        heat = np.exp(-0.5 * ((SS - center[:, None]) / sigma) ** 2)
        heat *= np.maximum(
            0.05,
            1.0
            + float(self.toroidal_modulation)
            * np.cos(toroidal_n * self.phi_values + phase),
        )[:, None]
        ds = np.gradient(self.s_values)
        dphi = TWOPI / float(self.phi_values.size)
        area = np.broadcast_to(dphi * ds[None, :], heat.shape).copy()
        normalization = float(np.sum(heat * area))
        if normalization > 0.0:
            heat *= total_power / normalization
        return BoundaryTopologyHeatState(
            heat=heat,
            phi_values=self.phi_values,
            s_values=self.s_values,
            cell_areas=area,
            metadata={
                "model": "reduced_spectral_diffusive_heat",
                "quantitative_transport": False,
                "field_period": case.field_period,
                "dominant_mode": (
                    None if dominant is None else (int(dominant.m), int(dominant.n))
                ),
                "chaos_width": chaos_width,
                "strike_sigma_s": sigma,
            },
        )


__all__ = ["ReducedSpectralHeatModel"]
