"""Field-line measurements on traced streamlines.

Provides functions to compute physical quantities from field-line
tracing results in cylindrical (R, Z, φ) coordinates.

Ported and extended from ``mhdpy.measure.FLT`` (Wenyin Wei, EAST/Tsinghua).
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


def field_line_length(streamline_rzphi: np.ndarray) -> float:
    r"""Compute the arc length of a field-line streamline in 3-D.

    The streamline is given in cylindrical coordinates (R, Z, φ).
    The arc-length element is:

    .. math::

        \mathrm{d}\ell = \sqrt{\mathrm{d}R^2 + \mathrm{d}Z^2 + (R\,\mathrm{d}\phi)^2}

    Parameters
    ----------
    streamline_rzphi:
        Array of shape ``(N, 3)`` where columns are (R, Z, φ).

    Returns
    -------
    float
        Total arc length of the field line (m).
    """
    rzphi = np.asarray(streamline_rzphi, dtype=float)
    R = rzphi[:, 0]
    Z = rzphi[:, 1]
    Phi = rzphi[:, 2]

    dR = np.diff(R)
    dZ = np.diff(Z)
    dPhi = np.diff(Phi)
    R_mid = 0.5 * (R[:-1] + R[1:])

    ds = np.sqrt(dR**2 + dZ**2 + (R_mid * dPhi)**2)
    return float(np.sum(ds))


def field_line_endpoints(
    streamline_rzphi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the start and end points of a streamline.

    Parameters
    ----------
    streamline_rzphi:
        Array of shape ``(N, 3)`` with columns (R, Z, φ).

    Returns
    -------
    (start, end) : tuple of ndarray
        Each is a 1-D array of length 3: ``(R, Z, φ)``.
    """
    rzphi = np.asarray(streamline_rzphi, dtype=float)
    return rzphi[0, :].copy(), rzphi[-1, :].copy()


def field_line_min_psi(
    streamline_rzphi: np.ndarray,
    psi_norm_interpolator: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> float:
    """Compute the minimum normalised poloidal flux ψ_norm along a field line.

    Useful for estimating how deeply a field line penetrates into the
    plasma (smaller ψ_norm → deeper penetration towards the axis).

    Parameters
    ----------
    streamline_rzphi:
        Array of shape ``(N, 3)`` with columns (R, Z, φ).
    psi_norm_interpolator:
        Callable ``f(R, Z) -> psi_norm`` that evaluates ψ_norm at
        arbitrary (R, Z) positions.

    Returns
    -------
    float
        Minimum ψ_norm value encountered along the field line.
    """
    rzphi = np.asarray(streamline_rzphi, dtype=float)
    R = rzphi[:, 0]
    Z = rzphi[:, 1]
    psi = psi_norm_interpolator(R, Z)
    return float(np.nanmin(psi))
