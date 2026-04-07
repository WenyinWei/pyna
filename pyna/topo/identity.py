"""Identity objects for resonance / tube / island entities.

These are lightweight, hashable IDs used to keep continuous-time and discrete-
time objects aligned without relying only on coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResonanceID:
    """Identity of a resonance family."""

    m: int
    n: int
    Np: int
    label: str | None = None

    @property
    def q(self) -> float:
        return self.m / self.n

    def short_label(self) -> str:
        return self.label or f"{self.m}/{self.n}@Np{self.Np}"


@dataclass(frozen=True, slots=True)
class TubeID:
    """Identity of one tube inside a resonance family."""

    resonance: ResonanceID
    tube_index: int
    kind: str | None = None

    def short_label(self) -> str:
        kind = self.kind or "?"
        return f"{kind}-tube[{self.tube_index}]@{self.resonance.short_label()}"


@dataclass(frozen=True, slots=True)
class IslandID:
    """Identity of one discrete island in one section view."""

    resonance: ResonanceID
    phi: float
    island_index: int
    kind: str | None = None

    def short_label(self) -> str:
        kind = self.kind or "?"
        return (
            f"{kind}-isl[{self.island_index}]"
            f"@phi={self.phi:.6f}@{self.resonance.short_label()}"
        )


__all__ = ["ResonanceID", "TubeID", "IslandID"]
