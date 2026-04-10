"""FixedPoint dataclass — split from invariants.py."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Optional

import numpy as np

from pyna.topo._stability import MonodromyData, Stability
from pyna.topo._base import InvariantObject


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

    # ── Array-like interface (backward compat with ndarray O_point) ───────────

    def __getitem__(self, idx: int) -> float:
        """fp[0] → R,  fp[1] → Z  (supports code that treats fp as a 2-vector)."""
        if idx == 0:
            return float(self.R)
        if idx == 1:
            return float(self.Z)
        raise IndexError(f"FixedPoint index {idx} out of range (0=R, 1=Z)")

    def __len__(self) -> int:
        return 2

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """np.asarray(fp) → array([R, Z])."""
        arr = np.array([self.R, self.Z])
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    # ── InvariantObject interface ─────────────────────────────────────────────

    def section_cut(self, section=None) -> list:
        """A FixedPoint is already a section-level object; return [self]."""
        return [self]

    def diagnostics(self) -> Dict[str, Any]:
        return {
            'invariant_type': 'FixedPoint',
            'phi': self.phi,
            'R': self.R,
            'Z': self.Z,
            'kind': self.kind,
            'greene_residue': self.greene_residue,
        }
