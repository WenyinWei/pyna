"""healed_scaffold_3d.py
=================================
Field-line-transported 3D scaffold for healed magnetic coordinates.

This module lifts the "reference section + field-line tracing" workflow from
project scripts (for example ``topoquest/scripts/w7x/w7x_healed_scaffold.py``)
into reusable ``pyna.topo`` infrastructure.

The key idea is simple:

1. Build a reliable reference poloidal scaffold at one toroidal angle
   ``phi_ref``.
2. Sample that scaffold on a regular ``(r, theta)`` grid.
3. Transport those sample points to other toroidal sections by tracing the
   *actual* field lines.
4. Fit / store the transported surface family as a discrete 3D scaffold.

This is intentionally a first-step implementation:

- it works on a *discrete* set of toroidal sections ``phi_samples``;
- it stores transported points directly instead of enforcing a global variational
  reconstruction;
- it is meant to replace ad-hoc script glue and serve as the basis for a future
  continuous ``IslandHealedCoordMap3D``.

The API is designed so that existing section-wise Fourier maps can provide the
reference scaffold, while downstream code can obtain toroidally consistent
section cuts from one unified 3D object.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np


ArrayLike = np.ndarray
TraceFunction = Callable[[float, float, float, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]


def _wrap_angle(phi: float) -> float:
    """Wrap angle to [0, 2π)."""
    return float(phi % (2.0 * np.pi))


def _forward_span(phi_src: float, phi_tgt: float) -> float:
    """Forward toroidal span from ``phi_src`` to ``phi_tgt`` in [0, 2π)."""
    return float((phi_tgt - phi_src) % (2.0 * np.pi))


@dataclass
class TransportedSection:
    """A single transported toroidal section of the scaffold.

    Attributes
    ----------
    phi : float
        Toroidal angle of this section [rad].
    R : ndarray, shape (n_r, n_theta)
        Transported R coordinates.
    Z : ndarray, shape (n_r, n_theta)
        Transported Z coordinates.
    valid : ndarray, shape (n_r, n_theta)
        Boolean validity mask. ``False`` means tracing / intersection failed.
    """

    phi: float
    R: ArrayLike
    Z: ArrayLike
    valid: ArrayLike


@dataclass
class SectionCorrespondence:
    """Discrete transport correspondence from a reference section.

    This object makes the transport relation explicit:
    the point with indices ``(i_r, i_theta)`` at the reference section is mapped
    to the transported point ``(R[i_r, i_theta], Z[i_r, i_theta])`` at target
    toroidal angle ``phi``.
    """

    phi_ref: float
    phi: float
    r_levels: ArrayLike
    theta_levels: ArrayLike
    R: ArrayLike
    Z: ArrayLike
    valid: ArrayLike

    def coverage_fraction(self) -> float:
        """Return valid-point coverage fraction in [0, 1]."""
        return float(np.mean(self.valid)) if self.valid.size else 0.0

    def valid_counts_by_r(self) -> ArrayLike:
        """Return number of valid θ samples for each r level."""
        return np.sum(self.valid, axis=1)


class FieldLineScaffold3D:
    """Discrete 3D scaffold built by field-line transport from one reference section.

    Parameters
    ----------
    phi_ref : float
        Reference toroidal angle [rad].
    r_levels : ndarray, shape (n_r,)
        Radial sample levels on the reference section.
    theta_levels : ndarray, shape (n_theta,)
        Poloidal sample levels on the reference section.
    phi_samples : ndarray, shape (n_phi,)
        Toroidal sections where the scaffold is sampled.
    sections : list[TransportedSection]
        One transported section per ``phi_samples``.
    R_ref, Z_ref : ndarray, shape (n_r, n_theta)
        Reference-section coordinates.
    """

    def __init__(
        self,
        phi_ref: float,
        r_levels: ArrayLike,
        theta_levels: ArrayLike,
        phi_samples: ArrayLike,
        sections: Sequence[TransportedSection],
        R_ref: ArrayLike,
        Z_ref: ArrayLike,
    ):
        self.phi_ref = _wrap_angle(phi_ref)
        self.r_levels = np.asarray(r_levels, dtype=float)
        self.theta_levels = np.asarray(theta_levels, dtype=float)
        self.phi_samples = np.asarray([_wrap_angle(phi) for phi in phi_samples], dtype=float)
        self.sections = list(sections)
        self.R_ref = np.asarray(R_ref, dtype=float)
        self.Z_ref = np.asarray(Z_ref, dtype=float)

        if len(self.sections) != len(self.phi_samples):
            raise ValueError("len(sections) must equal len(phi_samples)")
        if self.R_ref.shape != (len(self.r_levels), len(self.theta_levels)):
            raise ValueError("R_ref shape must be (n_r, n_theta)")
        if self.Z_ref.shape != self.R_ref.shape:
            raise ValueError("Z_ref shape mismatch")

    @classmethod
    def from_reference_map(
        cls,
        reference_map,
        phi_ref: float,
        r_levels: Sequence[float],
        theta_levels: Sequence[float],
        phi_samples: Sequence[float],
        trace_func: TraceFunction,
        *,
        dphi_hint: float = 0.04,
        phi_hit_tol_factor: float = 5.0,
    ) -> "FieldLineScaffold3D":
        """Build scaffold from a reference section map and a tracing function.

        Parameters
        ----------
        reference_map : object
            Any object exposing ``eval_RZ(r, theta) -> (R, Z)`` on the reference
            toroidal section. ``InnerFourierSection`` is the intended first use.
        phi_ref : float
            Reference toroidal angle [rad].
        r_levels, theta_levels, phi_samples : sequence of float
            Discrete scaffold sampling levels.
        trace_func : callable
            Signature ``trace_func(R0, Z0, phi0, phi_span, dphi_out)`` returning
            ``(R_arr, Z_arr, phi_arr)`` along the traced field line.
        dphi_hint : float, optional
            Preferred φ output spacing for tracing.
        phi_hit_tol_factor : float, optional
            Accept transported hit if nearest traced sample lies within
            ``phi_hit_tol_factor * dphi_out`` of the target section.

        Returns
        -------
        FieldLineScaffold3D
        """
        phi_ref = _wrap_angle(phi_ref)
        r_levels = np.asarray(r_levels, dtype=float)
        theta_levels = np.asarray(theta_levels, dtype=float)
        phi_samples = np.asarray([_wrap_angle(phi) for phi in phi_samples], dtype=float)

        n_r = len(r_levels)
        n_theta = len(theta_levels)

        R_ref = np.empty((n_r, n_theta), dtype=float)
        Z_ref = np.empty((n_r, n_theta), dtype=float)
        for i, r in enumerate(r_levels):
            for j, theta in enumerate(theta_levels):
                R_ref[i, j], Z_ref[i, j] = reference_map.eval_RZ(float(r), float(theta))

        sections = []
        for phi_tgt in phi_samples:
            if abs(_forward_span(phi_ref, phi_tgt)) < 1e-12:
                valid = np.ones_like(R_ref, dtype=bool)
                sections.append(
                    TransportedSection(phi=float(phi_tgt), R=R_ref.copy(), Z=Z_ref.copy(), valid=valid)
                )
                continue

            R_tgt, Z_tgt, valid = trace_grid_to_phi(
                R_ref,
                Z_ref,
                phi_src=phi_ref,
                phi_tgt=float(phi_tgt),
                trace_func=trace_func,
                dphi_hint=dphi_hint,
                phi_hit_tol_factor=phi_hit_tol_factor,
            )
            sections.append(
                TransportedSection(phi=float(phi_tgt), R=R_tgt, Z=Z_tgt, valid=valid)
            )

        return cls(
            phi_ref=phi_ref,
            r_levels=r_levels,
            theta_levels=theta_levels,
            phi_samples=phi_samples,
            sections=sections,
            R_ref=R_ref,
            Z_ref=Z_ref,
        )

    def nearest_section_index(self, phi: float) -> int:
        """Return index of the nearest sampled toroidal section."""
        phi_w = _wrap_angle(phi)
        dphi = np.abs(np.angle(np.exp(1j * (self.phi_samples - phi_w))))
        return int(np.argmin(dphi))

    def section_at(self, phi: float) -> TransportedSection:
        """Return nearest discrete transported section to ``phi``."""
        return self.sections[self.nearest_section_index(phi)]

    def correspondence_at(self, phi: float) -> SectionCorrespondence:
        """Return explicit transport correspondence to nearest sampled section."""
        sec = self.section_at(phi)
        return SectionCorrespondence(
            phi_ref=self.phi_ref,
            phi=sec.phi,
            r_levels=self.r_levels,
            theta_levels=self.theta_levels,
            R=sec.R,
            Z=sec.Z,
            valid=sec.valid,
        )

    def sample_surface(self, r_index: int, phi: float) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Return sampled θ-ring at a given radial index and toroidal section."""
        sec = self.section_at(phi)
        return sec.R[r_index], sec.Z[r_index], sec.valid[r_index]

    def sampled_arrays(self) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Return stacked sampled arrays with shape (n_phi, n_r, n_theta)."""
        R = np.stack([sec.R for sec in self.sections], axis=0)
        Z = np.stack([sec.Z for sec in self.sections], axis=0)
        valid = np.stack([sec.valid for sec in self.sections], axis=0)
        return R, Z, valid


def trace_grid_to_phi(
    R_grid: ArrayLike,
    Z_grid: ArrayLike,
    *,
    phi_src: float,
    phi_tgt: float,
    trace_func: TraceFunction,
    dphi_hint: float = 0.04,
    phi_hit_tol_factor: float = 5.0,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Transport a 2D ``(r, θ)`` point grid from one toroidal section to another.

    Parameters
    ----------
    R_grid, Z_grid : ndarray, shape (n_r, n_theta)
        Source coordinates on the reference section.
    phi_src, phi_tgt : float
        Source and target toroidal angles [rad].
    trace_func : callable
        Signature ``trace_func(R0, Z0, phi0, phi_span, dphi_out)``.
    dphi_hint : float, optional
        Preferred tracing output spacing.
    phi_hit_tol_factor : float, optional
        Target-hit tolerance in units of ``dphi_out``.

    Returns
    -------
    R_tgt, Z_tgt, valid : ndarray
        Transported coordinates and validity mask, all shape ``(n_r, n_theta)``.
    """
    R_grid = np.asarray(R_grid, dtype=float)
    Z_grid = np.asarray(Z_grid, dtype=float)
    if R_grid.shape != Z_grid.shape:
        raise ValueError("R_grid and Z_grid must have the same shape")

    n_r, n_theta = R_grid.shape
    R_tgt = np.full((n_r, n_theta), np.nan, dtype=float)
    Z_tgt = np.full((n_r, n_theta), np.nan, dtype=float)
    valid = np.zeros((n_r, n_theta), dtype=bool)

    span = _forward_span(phi_src, phi_tgt)
    if span < 1e-12:
        return R_grid.copy(), Z_grid.copy(), np.ones_like(R_grid, dtype=bool)

    dphi_out = min(span / 30.0, dphi_hint) if span > 0 else dphi_hint
    phi_tgt_w = _wrap_angle(phi_tgt)
    phi_tol = phi_hit_tol_factor * dphi_out

    for i in range(n_r):
        for j in range(n_theta):
            R0 = float(R_grid[i, j])
            Z0 = float(Z_grid[i, j])
            if not (np.isfinite(R0) and np.isfinite(Z0)):
                continue

            R_arr, Z_arr, phi_arr = trace_func(R0, Z0, float(phi_src), float(span + 2.0 * dphi_out), float(dphi_out))
            if len(phi_arr) == 0:
                continue

            phi_mod = np.asarray(phi_arr, dtype=float) % (2.0 * np.pi)
            idx = int(np.argmin(np.abs(phi_mod - phi_tgt_w)))
            if abs(phi_mod[idx] - phi_tgt_w) < phi_tol:
                R_tgt[i, j] = float(R_arr[idx])
                Z_tgt[i, j] = float(Z_arr[idx])
                valid[i, j] = True

    return R_tgt, Z_tgt, valid


def trace_section_curve_to_phi(
    R_curve: Sequence[float],
    Z_curve: Sequence[float],
    *,
    phi_src: float,
    phi_tgt: float,
    trace_func: TraceFunction,
    dphi_hint: float = 0.04,
    phi_hit_tol_factor: float = 5.0,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Transport one poloidal curve from ``phi_src`` to ``phi_tgt``.

    This is a thin convenience wrapper over :func:`trace_grid_to_phi` for the
    common "trace a boundary / one r-level ring" use case.
    """
    R_curve = np.asarray(R_curve, dtype=float)[None, :]
    Z_curve = np.asarray(Z_curve, dtype=float)[None, :]
    R_t, Z_t, valid = trace_grid_to_phi(
        R_curve,
        Z_curve,
        phi_src=phi_src,
        phi_tgt=phi_tgt,
        trace_func=trace_func,
        dphi_hint=dphi_hint,
        phi_hit_tol_factor=phi_hit_tol_factor,
    )
    return R_t[0], Z_t[0], valid[0]


def trace_surface_family_to_sections(
    reference_map,
    phi_ref: float,
    r_levels: Sequence[float],
    theta_levels: Sequence[float],
    phi_samples: Sequence[float],
    trace_func: TraceFunction,
    *,
    dphi_hint: float = 0.04,
    phi_hit_tol_factor: float = 5.0,
) -> FieldLineScaffold3D:
    """Functional convenience wrapper for :meth:`FieldLineScaffold3D.from_reference_map`."""
    return FieldLineScaffold3D.from_reference_map(
        reference_map=reference_map,
        phi_ref=phi_ref,
        r_levels=r_levels,
        theta_levels=theta_levels,
        phi_samples=phi_samples,
        trace_func=trace_func,
        dphi_hint=dphi_hint,
        phi_hit_tol_factor=phi_hit_tol_factor,
    )
