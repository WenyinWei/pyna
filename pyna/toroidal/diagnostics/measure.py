"""Toroidal field-line diagnostic measurements.

Canonical toroidal implementation of field-line observables computed from
traced streamlines in cylindrical ``(R, Z, \\phi)`` coordinates.

This module provides toroidal field-line observables in cylindrical
coordinates.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def field_line_length(streamline_rzphi: np.ndarray) -> float:
    r"""Compute the arc length of a field-line streamline in 3-D.

    The streamline is given in cylindrical coordinates ``(R, Z, \\phi)`` and the
    arc-length element is

    .. math::

        \mathrm{d}\ell = \sqrt{\mathrm{d}R^2 + \mathrm{d}Z^2 + (R\,\mathrm{d}\phi)^2}.
    """
    rzphi = np.asarray(streamline_rzphi, dtype=float)
    R = rzphi[:, 0]
    Z = rzphi[:, 1]
    Phi = rzphi[:, 2]

    dR = np.diff(R)
    dZ = np.diff(Z)
    dPhi = np.diff(Phi)
    R_mid = 0.5 * (R[:-1] + R[1:])

    ds = np.sqrt(dR**2 + dZ**2 + (R_mid * dPhi) ** 2)
    return float(np.sum(ds))


def field_line_endpoints(streamline_rzphi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return copies of the start and end points of a streamline."""
    rzphi = np.asarray(streamline_rzphi, dtype=float)
    return rzphi[0, :].copy(), rzphi[-1, :].copy()


def field_line_min_psi(
    streamline_rzphi: np.ndarray,
    psi_norm_interpolator: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> float:
    """Compute the minimum normalised poloidal flux ``psi_norm`` along a field line."""
    rzphi = np.asarray(streamline_rzphi, dtype=float)
    R = rzphi[:, 0]
    Z = rzphi[:, 1]
    psi = psi_norm_interpolator(R, Z)
    return float(np.nanmin(psi))


__all__ = [
    "field_line_length",
    "field_line_endpoints",
    "field_line_min_psi",
]
