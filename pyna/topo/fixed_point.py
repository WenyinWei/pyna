"""FixedPoint dataclass — split from invariants.py."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np

from pyna.topo._stability import MonodromyData, Stability
from pyna.topo.invariant import InvariantObject


@dataclass(eq=False)
class FixedPoint(InvariantObject):
    """A single Poincaré fixed point with full monodromy data."""
    phi: float
    R: float
    Z: float
    DPm: np.ndarray
    kind: str = ''                          # 'X' or 'O'; auto-derived when empty
    DX_pol_accum: Optional[np.ndarray] = None
    ambient_dim: Optional[int] = None

    def __post_init__(self):
        if not self.kind:
            tr = float(np.trace(self.DPm))
            self.kind = 'O' if abs(tr) < 2.0 - 1e-10 else 'X'

    @cached_property
    def monodromy(self) -> MonodromyData:
        eigs = np.linalg.eigvals(self.DPm)
        return MonodromyData(DPm=self.DPm, eigenvalues=eigs)

    @property
    def stability(self) -> Stability:
        return self.monodromy.stability

    @property
    def greene_residue(self) -> float:
        """Greene's residue = (2 - Tr(DPm)) / 4."""
        return float((2.0 - np.trace(self.DPm)) / 4.0)
