"""toroidal._manifold — StableManifold and UnstableManifold."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

from pyna.topo._base import InvariantManifold
from ._cycle import Cycle


@dataclass(eq=False)
class StableManifold(InvariantManifold):
    """Stable manifold of a hyperbolic cycle."""
    cycle: Cycle
    branches: List[Any] = field(default_factory=list)
    ambient_dim: Optional[int] = None

    @property
    def intrinsic_dim(self) -> Optional[int]:
        if self.cycle.monodromy is None:
            return None
        eigs = self.cycle.monodromy.eigenvalues
        return int(np.sum(np.abs(eigs) < 1.0 - 1e-10))

    def section_cut(self, section=None) -> list:
        return list(self.branches)

    def diagnostics(self) -> dict:
        return {'invariant_type': 'StableManifold', 'n_branches': len(self.branches),
                'intrinsic_dim': self.intrinsic_dim}


@dataclass(eq=False)
class UnstableManifold(InvariantManifold):
    """Unstable manifold of a hyperbolic cycle."""
    cycle: Cycle
    branches: List[Any] = field(default_factory=list)
    ambient_dim: Optional[int] = None

    @property
    def intrinsic_dim(self) -> Optional[int]:
        if self.cycle.monodromy is None:
            return None
        eigs = self.cycle.monodromy.eigenvalues
        return int(np.sum(np.abs(eigs) > 1.0 + 1e-10))

    def section_cut(self, section=None) -> list:
        return list(self.branches)

    def diagnostics(self) -> dict:
        return {'invariant_type': 'UnstableManifold', 'n_branches': len(self.branches),
                'intrinsic_dim': self.intrinsic_dim}
