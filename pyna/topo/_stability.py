"""Stability enum and MonodromyData — split from invariants.py."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property

import numpy as np


class Stability(Enum):
    ELLIPTIC = auto()
    HYPERBOLIC = auto()
    PARABOLIC = auto()
    UNKNOWN = auto()


@dataclass(eq=False)
class MonodromyData:
    DPm: np.ndarray          # 2×2 matrix
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
