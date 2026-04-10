"""InvariantTorus and _ToriMixin — split from invariants.py."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pyna.topo._base import InvariantObject


@dataclass(eq=False)
class InvariantTorus(InvariantObject):
    rotation_vector: Tuple[float, ...]
    ambient_dim: Optional[int] = None

    def section_cut(self, section: Any = None) -> "InvariantTorus":
        rv = self.rotation_vector[:-1] if len(self.rotation_vector) > 1 else self.rotation_vector
        adim = self.ambient_dim - 1 if self.ambient_dim else None
        return InvariantTorus(rotation_vector=rv, ambient_dim=adim)

    def diagnostics(self) -> Dict[str, Any]:
        return {
            'invariant_type': 'InvariantTorus',
            'rotation_vector': self.rotation_vector,
            'ambient_dim': self.ambient_dim,
        }


class _ToriMixin:
    """Mixin for objects that manage a radial stack of tori."""

    def __init__(self):
        self._tori: List[InvariantTorus] = []
        self._r_vals: List[float] = []

    def add_torus(self, torus: InvariantTorus, r: float):
        import bisect
        idx = bisect.bisect_left(self._r_vals, r)
        self._r_vals.insert(idx, r)
        self._tori.insert(idx, torus)
        if "rotation_profile" in self.__dict__:
            del self.__dict__["rotation_profile"]

    def _central_rotation_vector(self) -> Tuple[float, ...]:
        raise NotImplementedError

    @cached_property
    def rotation_profile(self):
        r_vals = [0.0] + list(self._r_vals)
        rv_list = [self._central_rotation_vector()] + [t.rotation_vector for t in self._tori]
        r_arr = np.array(r_vals)
        dim = len(rv_list[0])
        rv_arr = np.array([[rv[i] for rv in rv_list] for i in range(dim)])

        if len(r_arr) < 2:
            const_val = tuple(float(rv_list[0][i]) for i in range(dim))
            def profile_constant(r: float) -> Tuple[float, ...]:
                return const_val
            return profile_constant

        try:
            from scipy.interpolate import PchipInterpolator
            interps = [PchipInterpolator(r_arr, rv_arr[i]) for i in range(dim)]
        except ImportError:
            interps = [lambda x, i=i: np.interp(x, r_arr, rv_arr[i]) for i in range(dim)]

        def profile(r: float) -> Tuple[float, ...]:
            return tuple(float(interps[i](r)) for i in range(dim))

        return profile
