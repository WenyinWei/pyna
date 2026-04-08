"""pyna.topo.resonance -- ResonanceNumber for arbitrary-dimensional maps.

In MCF the resonance is labeled m/n (integers), but in a general
d-dimensional map the resonance condition is:

    n_1·ω_1 + n_2·ω_2 + ... + n_d·ω_d ≈ 0

where ω_i are the rotation numbers (or winding numbers) and n_i ∈ Z.

The tuple (n_1, n_2, ..., n_d) is the ResonanceNumber.

In the 2D MCF case:
    n = (m, n_pol)   where m = toroidal period, n_pol = poloidal winding
    (conventional MCF notation: the ratio q = m/n_pol)

Usage
-----
    r = ResonanceNumber(10, 3)          # m/n = 10/3 (HAO)
    r = ResonanceNumber(3, 2, 1)        # 3D map resonance
    r.dim                               # number of frequency components
    r.short_label()                     # "10/3"
"""
from __future__ import annotations

from math import gcd
from functools import reduce
from typing import Sequence


class ResonanceNumber(tuple):
    """Immutable tuple of integers (n_1, n_2, ..., n_d) labeling a resonance.

    Inherits from tuple for hashing and equality.

    In MCF contexts (d=2): ResonanceNumber(m, n_pol) where q = m/n_pol.
    In higher-dimensional contexts: ResonanceNumber(n_1, n_2, ..., n_d).

    The *reduced form* divides all components by their GCD.
    """

    def __new__(cls, *args: int) -> "ResonanceNumber":
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            args = tuple(args[0])
        return super().__new__(cls, (int(a) for a in args))

    @property
    def dim(self) -> int:
        """Number of frequency components."""
        return len(self)

    @property
    def gcd(self) -> int:
        """GCD of all components."""
        return reduce(gcd, (abs(x) for x in self), 0)

    def reduced(self) -> "ResonanceNumber":
        """Return the reduced form (divided by GCD)."""
        g = self.gcd
        if g == 0:
            return self
        return ResonanceNumber(*(x // g for x in self))

    # ── MCF convenience (d=2) ────────────────────────────────────────────────

    @property
    def m(self) -> int:
        """Toroidal period (first component). MCF convention."""
        return self[0]

    @property
    def n_pol(self) -> int:
        """Poloidal winding number (second component). MCF convention."""
        return self[1] if len(self) > 1 else 0

    @property
    def q(self) -> float:
        """Safety factor q = m / n_pol. MCF only (d=2)."""
        if len(self) < 2 or self[1] == 0:
            return float('inf')
        return self[0] / self[1]

    @property
    def n_islands_per_section(self) -> int:
        """Number of distinct island/orbit visits per Poincaré section.

        = m // gcd(m, n_pol) for d=2.
        In general, this is the order of the resonance in the map.
        For d=2: m // gcd(m, n_pol).
        """
        if len(self) < 2:
            return self[0]
        return self[0] // gcd(self[0], self[1])

    # ── Labels ───────────────────────────────────────────────────────────────

    def short_label(self) -> str:
        """Short human-readable label, e.g. '10/3' or '3:2:1'."""
        if len(self) == 2:
            return f"{self[0]}/{self[1]}"
        return ":".join(str(x) for x in self)

    def latex(self) -> str:
        """LaTeX representation, e.g. '$10/3$'."""
        return f"${self.short_label()}$"

    def __repr__(self) -> str:
        return f"ResonanceNumber{tuple(self)}"

    def __str__(self) -> str:
        return self.short_label()
