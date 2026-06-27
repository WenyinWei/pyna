"""toroidal._cycle — PeriodicOrbit and Cycle toroidal specializations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pyna.topo.core import PeriodicOrbit as _PeriodicOrbit, Cycle as _Cycle, Stability
from ._monodromy import MonodromyData
from ._fixed_point import FixedPoint


@dataclass(eq=False)
class PeriodicOrbit(_PeriodicOrbit):
    """Toroidal periodic orbit of a discrete map.

    Inherits from core.PeriodicOrbit and adds monodromy support.
    """

    def __post_init__(self):
        super().__post_init__()

    @property
    def monodromy(self) -> Optional[MonodromyData]:
        if self.points:
            fp = self.points[0]
            if isinstance(fp, FixedPoint):
                return fp.monodromy
        return None

    def __getitem__(self, idx: int):
        return self.points[idx]

    def __len__(self) -> int:
        return len(self.points)


@dataclass(eq=False)
class Cycle(_Cycle):
    """Toroidal periodic orbit of a continuous flow.

    Inherits from core.Cycle and adds winding, sections dict, monodromy.
    """

    winding: Tuple[int, ...] = (1,)
    sections: Dict[float, List[FixedPoint]] = field(default_factory=dict)
    monodromy: Optional[MonodromyData] = None
    ambient_dim: Optional[int] = None

    @property
    def stability(self) -> Stability:
        if self.monodromy is not None:
            return self.monodromy.stability
        for fp in self.section_points():
            if isinstance(fp, FixedPoint):
                return fp.stability
        return Stability.UNKNOWN

    def section_points(self, phi: Optional[float] = None, tol: float = 1e-9) -> List[FixedPoint]:
        if not self.sections:
            return []
        if phi is not None:
            for k, v in self.sections.items():
                if abs(k - float(phi)) < tol:
                    return list(v)
            phis = np.array(list(self.sections.keys()))
            idx = int(np.argmin(np.abs(phis - float(phi))))
            if abs(phis[idx] - float(phi)) < tol:
                return list(self.sections[float(phis[idx])])
            return []
        result: List[FixedPoint] = []
        for v in self.sections.values():
            result.extend(v)
        return result

    def section_cut(self, section=None) -> PeriodicOrbit:
        if section is None:
            fps = self.section_points()
            return PeriodicOrbit(points=list(fps)) if fps else PeriodicOrbit(points=[])
        from pyna.topo.section import coerce_toroidal_section
        phi = float(coerce_toroidal_section(section).phi)
        fps = self.section_points(phi)
        return PeriodicOrbit(points=list(fps))

    def unstable_seeds(
        self,
        phi: Optional[float] = None,
        *,
        n_seeds: int = 8,
        init_length: float = 1e-4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        fps = self.section_points(phi)
        if not fps:
            raise ValueError("cannot generate unstable seeds without a section fixed point")
        fp = fps[0]
        DPm = self.monodromy.DPm if self.monodromy is not None else fp.DPm
        eigvals, eigvecs = np.linalg.eig(np.asarray(DPm, dtype=float))
        idx = int(np.argmax(np.abs(eigvals)))
        direction = np.real(eigvecs[:, idx])
        if np.linalg.norm(direction) < 1e-14:
            direction = np.imag(eigvecs[:, idx])
        if np.linalg.norm(direction) < 1e-14:
            direction = np.array([1.0, 0.0])
        direction = direction[:2] / np.linalg.norm(direction[:2])
        offsets = np.linspace(-float(init_length), float(init_length), int(n_seeds))
        return fp.R + offsets * direction[0], fp.Z + offsets * direction[1]

    def diagnostics(self) -> Dict[str, Any]:
        return {
            'invariant_type': 'Cycle',
            'winding': self.winding,
            'n_sections': len(self.sections),
            'total_points': sum(len(v) for v in self.sections.values()),
            'has_monodromy': self.monodromy is not None,
        }
