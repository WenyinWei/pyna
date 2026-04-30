"""toroidal._monodromy — MonodromyData and spectral regularity helpers."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional

import numpy as np

from pyna.topo.core import Stability


@dataclass(eq=False)
class MonodromyData:
    """Monodromy (period-m Jacobian) data for a periodic orbit."""

    DPm: np.ndarray          # (d, d) monodromy matrix
    eigenvalues: np.ndarray  # eigenvalues of DPm

    @cached_property
    def trace(self) -> float:
        return float(np.trace(self.DPm))

    @cached_property
    def stability(self) -> Stability:
        tr = self.trace
        if abs(tr) < 2.0 - 1e-10:
            return Stability.ELLIPTIC
        elif abs(tr) > 2.0 + 1e-10:
            return Stability.HYPERBOLIC
        else:
            return Stability.PARABOLIC

    @cached_property
    def greene_residue(self) -> float:
        return (2.0 - self.trace) / 4.0

    @cached_property
    def stability_index(self) -> float:
        return self.trace / 2.0

    def spectral_regularity(
        self,
        DPk_sequence: Optional[List[np.ndarray]] = None,
        *,
        k_max: Optional[int] = None,
    ) -> float:
        """Spectral regularity index."""
        if DPk_sequence is not None and len(DPk_sequence) > 0:
            return _spectral_regularity_from_sequence(DPk_sequence)
        return _spectral_regularity_single(self.eigenvalues)


def _spectral_regularity_single(eigenvalues: np.ndarray) -> float:
    mods = np.abs(eigenvalues)
    mods = np.where(mods < 1e-30, 1e-30, mods)
    return float(np.max(np.abs(np.log(mods))))


def _spectral_regularity_from_sequence(DPk_sequence: List[np.ndarray]) -> float:
    m = len(DPk_sequence)
    if m == 0:
        return 0.0
    total = 0.0
    for DPk in DPk_sequence:
        eigs = np.linalg.eigvals(DPk)
        total += _spectral_regularity_single(eigs)
    return total / m
