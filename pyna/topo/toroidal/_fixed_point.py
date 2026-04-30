"""toroidal._fixed_point — FixedPoint(SectionPoint)."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pyna.topo.core import SectionPoint as _SectionPoint, Stability
from ._monodromy import MonodromyData


@dataclass(eq=False)
class FixedPoint(_SectionPoint):
    """One point of a periodic orbit of a discrete map, with monodromy data.

    Inherits from core.SectionPoint and adds MCF-specific fields.
    """

    phi: float = 0.0
    R: float = 0.0
    Z: float = 0.0
    DPm: np.ndarray = field(default_factory=lambda: np.eye(2))
    kind: str = ''
    DX_pol_accum: Optional[np.ndarray] = None
    ambient_dim: Optional[int] = None
    coords: Optional[np.ndarray] = field(default=None, repr=False)
    coordinate_names: Optional[Tuple[str, ...]] = field(default=None, repr=False)
    section_angle: Optional[float] = field(default=None, repr=False)

    def __post_init__(self):
        # Sync coords ↔ (R, Z)
        if self.coords is None:
            self.coords = np.array([self.R, self.Z], dtype=float)
        else:
            self.coords = np.asarray(self.coords, dtype=float)
            if len(self.coords) >= 2 and self.R == 0.0 and self.Z == 0.0:
                self.R = float(self.coords[0])
                self.Z = float(self.coords[1])

        # Sync section_angle ↔ phi
        if self.section_angle is None:
            self.section_angle = self.phi
        elif self.phi == 0.0:
            self.phi = self.section_angle

        # Sync state ← coords
        if self.state is None or len(self.state) == 0:
            object.__setattr__(self, 'state', self.coords.copy())
        else:
            object.__setattr__(self, 'state', np.asarray(self.state, dtype=float))

        # Auto-derive kind from DPm
        if not self.kind:
            tr = float(np.trace(self.DPm))
            self.kind = 'O' if abs(tr) < 2.0 - 1e-10 else 'X'

    @property
    def intrinsic_dim(self) -> int:
        return 0

    @cached_property
    def monodromy(self) -> MonodromyData:
        eigs = np.linalg.eigvals(self.DPm)
        return MonodromyData(DPm=self.DPm, eigenvalues=eigs)

    @property
    def stability(self) -> Stability:
        return self.monodromy.stability

    @property
    def greene_residue(self) -> float:
        return float((2.0 - np.trace(self.DPm)) / 4.0)

    def __getitem__(self, idx: int) -> float:
        return float(self.coords[idx])

    def __len__(self) -> int:
        return len(self.coords)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        arr = self.coords.copy()
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    def section_cut(self, section=None) -> list:
        return [self]

    def diagnostics(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            'invariant_type': 'FixedPoint',
            'phi': self.phi,
            'kind': self.kind,
            'greene_residue': self.greene_residue,
            'coords': self.coords.tolist(),
        }
        if len(self.coords) >= 2:
            d['R'] = float(self.coords[0])
            d['Z'] = float(self.coords[1])
        return d

    def as_orbit(self) -> "PeriodicOrbit":
        from ._cycle import PeriodicOrbit
        return PeriodicOrbit(points=[self])
