"""Functional perturbation theory shift hierarchy for field-line topology.

This module is the public landing zone for perturbative shifts of orbits,
trajectories, cycles, invariant tori, and invariant manifolds under a global
magnetic-field change.  The C++ backend lives in :mod:`pyna._cyna`; this module
keeps the mathematical names and the field-cache convention visible.

For a toroidal field-line ODE

    dX/dphi = f(X, phi) = (R * BR / BPhi, R * BZ / BPhi),

the first-order shift under a full-field perturbation is

    d(delta_X)/dphi = A(X, phi) delta_X + delta_f(X, phi),

where A = partial f / partial (R, Z).  ``delta_X_pol`` is the zero-initial
particular solution.  ``delta_X_cyc`` is the periodic cycle displacement:

    delta_X_cyc(phi) = delta_X_pol(phi) + DP(phi) delta_X_cyc(phi0),
    (I - DP(T)) delta_X_cyc(phi0) = delta_X_pol(phi0 + T).

For field-period-symmetric stellarators, use ``T = 2*pi / Nfp`` for one
field-period cycle, so the returned ``delta_X_cyc`` should repeat every field
period.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from pyna._cyna.utils import prepare_field_cache

try:
    from pyna._cyna import compute_cycle_perturbation_shift as _cyna_cycle_shift
except ImportError:  # pragma: no cover - import guard for source-only installs
    _cyna_cycle_shift = None

__all__ = [
    "OrbitPerturbationShift",
    "TrajectoryPerturbationShift",
    "CyclePerturbationShift",
    "InvariantTorusPerturbationShift",
    "StableManifoldPerturbationShift",
    "compute_cycle_shift",
    "compute_cycle_shift_from_cache",
]


@dataclass(frozen=True)
class OrbitPerturbationShift:
    """First-order shift data along a traced field-line orbit."""

    R: np.ndarray
    Z: np.ndarray
    phi: np.ndarray
    DP: np.ndarray
    delta_X_pol: np.ndarray
    alive: np.ndarray


@dataclass(frozen=True)
class TrajectoryPerturbationShift(OrbitPerturbationShift):
    """Shift of an open trajectory with a specified initial displacement."""


@dataclass(frozen=True)
class CyclePerturbationShift(OrbitPerturbationShift):
    """Shift of a periodic cycle under a full magnetic-field change.

    ``delta_X_pol`` is the particular solution with zero initial displacement.
    ``delta_X_cyc`` is the periodic physical cycle shift.  For an Nfp-periodic
    stellarator axis cycle, pass ``phi_span=2*pi/Nfp`` and verify that
    ``delta_X_cyc[-1]`` matches ``delta_X_cyc[0]``.
    """

    delta_X_cyc: np.ndarray
    delta_X_cyc0: np.ndarray

    @property
    def periodic_residual(self) -> np.ndarray:
        """Return ``delta_X_cyc(end) - delta_X_cyc(start)``."""

        return self.delta_X_cyc[-1] - self.delta_X_cyc[0]


class InvariantTorusPerturbationShift:
    """Placeholder for invariant-torus shift solvers.

    The public class is reserved here so callers can discover the intended FPT
    hierarchy.  Fourier/KAM torus-shift implementations should land behind
    this name instead of creating ad-hoc modules.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Invariant torus FPT shift is not implemented yet.")


class StableManifoldPerturbationShift:
    """Placeholder for stable/unstable manifold shift solvers.

    Planned implementation:
    compute the X-cycle shift, perturb the DPm eigenvalue/eigenvector that
    defines the X-leg seed direction, deploy seed points with geometric spacing
    so one P^m image lands just beyond the last seed, advect each seed
    displacement along the field line, then report the normal component of the
    displaced manifold arc.  The placeholder prevents future agents from
    creating duplicate one-off stable-manifold perturbation APIs.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Stable manifold FPT shift is not implemented yet.")


def _cache_array(cache: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    arr = np.asarray(cache[key], dtype=np.float64)
    if key in {"BR", "BZ", "BPhi"} and arr.ndim == 3:
        arr = arr.ravel()
    return np.ascontiguousarray(arr, dtype=np.float64)


def compute_cycle_shift(
    R0: float,
    Z0: float,
    phi0: float,
    phi_span: float,
    base_field: Any,
    pert_field: Any,
    *,
    dphi_out: float = 0.01,
    DPhi: float = 0.01,
    fd_eps: float = 1e-4,
) -> CyclePerturbationShift:
    """Compute cycle FPT shift from cylindrical vector-field objects.

    ``base_field`` and ``pert_field`` may be :class:`VectorFieldCylind`
    instances, compatible field-like objects, or legacy field-cache dicts.
    """

    return compute_cycle_shift_from_cache(
        R0,
        Z0,
        phi0,
        phi_span,
        base_field,
        pert_field,
        dphi_out=dphi_out,
        DPhi=DPhi,
        fd_eps=fd_eps,
    )


def compute_cycle_shift_from_cache(
    R0: float,
    Z0: float,
    phi0: float,
    phi_span: float,
    base_field_cache: Any,
    pert_field_cache: Any,
    *,
    dphi_out: float = 0.01,
    DPhi: float = 0.01,
    fd_eps: float = 1e-4,
) -> CyclePerturbationShift:
    """Compute ``delta_X_pol`` and periodic ``delta_X_cyc`` using cyna.

    ``base_field_cache`` and ``pert_field_cache`` may be
    :class:`VectorFieldCylind` objects, compatible field-like objects, or
    legacy field-cache dicts containing
    ``BR, BZ, BPhi, R_grid, Z_grid, Phi_grid``.  Component order is always
    canonical ``BR, BZ, BPhi``.
    """

    if _cyna_cycle_shift is None:
        raise RuntimeError("pyna._cyna.compute_cycle_perturbation_shift is unavailable.")

    base_fc = prepare_field_cache(base_field_cache, extend_phi=True)
    pert_fc = prepare_field_cache(pert_field_cache, extend_phi=True)

    R, Z, phi, DP, dXpol, dXcyc, dXcyc0, alive = _cyna_cycle_shift(
        float(R0), float(Z0), float(phi0),
        float(phi_span), float(dphi_out), float(DPhi), float(fd_eps),
        _cache_array(base_fc, "BR"),
        _cache_array(base_fc, "BZ"),
        _cache_array(base_fc, "BPhi"),
        _cache_array(pert_fc, "BR"),
        _cache_array(pert_fc, "BZ"),
        _cache_array(pert_fc, "BPhi"),
        _cache_array(base_fc, "R_grid"),
        _cache_array(base_fc, "Z_grid"),
        _cache_array(base_fc, "Phi_grid"),
    )

    return CyclePerturbationShift(
        R=np.asarray(R),
        Z=np.asarray(Z),
        phi=np.asarray(phi),
        DP=np.asarray(DP).reshape((-1, 2, 2)),
        delta_X_pol=np.asarray(dXpol),
        delta_X_cyc=np.asarray(dXcyc),
        delta_X_cyc0=np.asarray(dXcyc0),
        alive=np.asarray(alive, dtype=bool),
    )
