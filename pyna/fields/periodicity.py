"""Toroidal field-period metadata and angle normalization.

The number of field periods belongs to the grid/domain object.  High-level
callers should pass a field or coordinate object and let that object expose a
``ToroidalPeriodicity`` instance instead of repeatedly spelling
``2*pi / nfp`` at interpolation and backend boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


TWOPI = 2.0 * np.pi


def normalize_nfp(nfp: int = 1) -> int:
    """Return a validated number of physical field periods."""

    if isinstance(nfp, (bool, np.bool_)):
        raise ValueError("nfp must be a positive integer")
    try:
        periods = int(nfp)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("nfp must be a positive integer") from exc
    try:
        exact = float(periods) == float(nfp)
    except (TypeError, ValueError, OverflowError):
        exact = False
    if not exact or periods < 1:
        raise ValueError("nfp must be a positive integer")
    return periods


@dataclass(frozen=True)
class ToroidalPeriodicity:
    """Physical field symmetry and the toroidal period stored by one grid.

    ``nfp`` defines the physical field period ``2*pi/nfp``.  A stored domain
    may contain one or an integer number of physical field periods, up to the
    full torus.  This distinction keeps native-period and full-torus arrays
    interoperable without inferring a period from an endpoint=False span.
    """

    nfp: int = 1
    domain_period: float | None = None
    origin: float = 0.0

    def __post_init__(self) -> None:
        periods = normalize_nfp(self.nfp)
        origin = float(self.origin)
        if not np.isfinite(origin):
            raise ValueError("origin must be finite")
        field_period = TWOPI / float(periods)
        domain_period = field_period if self.domain_period is None else float(self.domain_period)
        if not np.isfinite(domain_period) or domain_period <= 0.0:
            raise ValueError("domain_period must be positive and finite")
        multiple = int(round(domain_period / field_period))
        if multiple < 1 or multiple > periods or not np.isclose(
            domain_period,
            multiple * field_period,
            rtol=1.0e-12,
            atol=1.0e-14,
        ):
            raise ValueError(
                "domain_period must contain an integer number of physical field periods"
            )
        object.__setattr__(self, "nfp", periods)
        object.__setattr__(self, "domain_period", multiple * field_period)
        object.__setattr__(self, "origin", origin)

    @property
    def field_period_rad(self) -> float:
        return TWOPI / float(self.nfp)

    @property
    def field_period(self) -> float:
        return self.field_period_rad

    @property
    def domain_period_count(self) -> int:
        return int(round(float(self.domain_period) / self.field_period_rad))

    @property
    def stores_one_field_period(self) -> bool:
        return self.domain_period_count == 1

    def wrap(self, phi: Any) -> np.ndarray:
        """Wrap angles into one physical field period."""

        period = self.field_period
        values = np.asarray(phi, dtype=float)
        return self.origin + np.mod(values - self.origin, period)

    def wrap_domain(self, phi: Any) -> np.ndarray:
        """Wrap angles into the angular domain stored by the grid."""

        values = np.asarray(phi, dtype=float)
        return self.origin + np.mod(values - self.origin, float(self.domain_period))

    def native_sample_count(
        self,
        domain_sample_count: int,
        *,
        endpoint_included: bool = False,
    ) -> int:
        """Return samples in one field period for a uniform toroidal grid."""

        sample_count = int(domain_sample_count)
        period_count = self.domain_period_count
        interval_count = sample_count - 1 if endpoint_included else sample_count
        if interval_count < 1 or interval_count % period_count:
            raise ValueError(
                "toroidal interval count must be divisible by domain_period_count"
            )
        native_intervals = interval_count // period_count
        return native_intervals + 1 if endpoint_included else native_intervals

    def endpoint(self, *, period_count: int | None = None) -> float:
        """Return the exact endpoint for a native-period multiple."""

        count = self.domain_period_count if period_count is None else int(period_count)
        if count < 1 or count > self.nfp:
            raise ValueError("period_count must lie between one and nfp")
        return self.origin + count * self.field_period_rad

    def as_dict(self) -> dict[str, float | int]:
        return {
            "nfp": int(self.nfp),
            "field_period": float(self.field_period),
            "domain_period": float(self.domain_period),
            "domain_period_count": int(self.domain_period_count),
            "origin": float(self.origin),
        }

    @classmethod
    def from_object(
        cls,
        source: object,
        *,
        domain_period: float | None = None,
        origin: float | None = None,
    ) -> "ToroidalPeriodicity":
        """Resolve periodicity from a field/cache/coordinate object."""

        existing = getattr(source, "periodicity", None)
        if isinstance(existing, cls) and domain_period is None and origin is None:
            return existing
        if isinstance(source, Mapping):
            get = source.get
        else:
            get = lambda name, default=None: getattr(source, name, default)
        # ``field_periods`` is accepted only when reading legacy payloads where
        # it historically (and ambiguously) meant nfp.  New objects and output
        # metadata never use that spelling.
        nfp = int(get("nfp", get("field_periods", 1)))
        resolved_period = domain_period
        if resolved_period is None:
            resolved_period = get("toroidal_period", None)
        if resolved_period is None:
            resolved_period = get("domain_period", None)
        if resolved_period is None:
            resolved_period = TWOPI / float(nfp)
        resolved_origin = float(get("phi_origin", 0.0) if origin is None else origin)
        return cls(nfp=nfp, domain_period=resolved_period, origin=resolved_origin)

    @classmethod
    def from_field_period(
        cls,
        field_period: float,
        *,
        domain_period: float | None = None,
        origin: float = 0.0,
    ) -> "ToroidalPeriodicity":
        """Construct periodicity from a physical field-period angle."""

        angle = abs(float(field_period))
        if not np.isfinite(angle) or angle <= 0.0:
            raise ValueError("field_period must be positive and finite")
        nfp = int(round(TWOPI / angle))
        if nfp < 1 or not np.isclose(
            angle,
            TWOPI / float(nfp),
            rtol=1.0e-12,
            atol=1.0e-14,
        ):
            raise ValueError("field_period must equal 2*pi/nfp for an integer nfp")
        return cls(nfp=nfp, domain_period=domain_period, origin=origin)


__all__ = ["ToroidalPeriodicity", "normalize_nfp"]
