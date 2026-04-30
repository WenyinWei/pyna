"""Tokamak equilibrium classes for pyna.

Provides:

* :class:`EquilibriumAxisym` — abstract base for axisymmetric
  tokamak equilibria.
* :class:`EquilibriumTokamakCircularSynthetic` — analytic model
  suitable for software testing (no real experimental data required).
* :func:`time_linear_weighting` — linear interpolation in a
  time series of equilibrium snapshots.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class EquilibriumAxisym(ABC):
    """Abstract base class for axisymmetric tokamak equilibria.

    Subclasses must supply the poloidal flux surface coordinate
    ``psi_norm(R, Z)``, the safety factor profile ``q(S)``, and the
    flux-surface mesh ``(r_mesh, z_mesh)`` on a ``(S, TET)`` grid.

    Attributes
    ----------
    R_mesh, Z_mesh:
        1-D arrays of the R and Z mesh used to store the equilibrium.
    S:
        1-D array of effective minor-radius coordinates (0 at axis,
        1 at last closed flux surface).
    TET:
        1-D array of poloidal angles (radians, PEST-like convention).
    """

    @property
    @abstractmethod
    def R_mesh(self) -> np.ndarray:
        """1-D radial mesh (m)."""

    @property
    @abstractmethod
    def Z_mesh(self) -> np.ndarray:
        """1-D vertical mesh (m)."""

    @property
    @abstractmethod
    def S(self) -> np.ndarray:
        """Flux-surface labels in [0, 1]."""

    @property
    @abstractmethod
    def TET(self) -> np.ndarray:
        """Poloidal angles (rad)."""

    @abstractmethod
    def psi_norm(self, R: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Evaluate normalised poloidal flux ψ_norm at (R, Z).

        Parameters
        ----------
        R, Z:
            Coordinates (m).  Broadcast-compatible arrays.

        Returns
        -------
        ndarray
            ψ_norm values (0 at magnetic axis, 1 at LCFS).
        """

    @abstractmethod
    def q(self, S: np.ndarray) -> np.ndarray:
        """Safety factor q as a function of the flux label S.

        Parameters
        ----------
        S:
            Flux-surface labels (0–1).

        Returns
        -------
        ndarray
            Safety-factor values.
        """

    @abstractmethod
    def B_field(
        self,
        R: np.ndarray,
        Z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Equilibrium magnetic field components at (R, Z).

        Returns
        -------
        (BR, BZ, BPhi) : tuple of ndarray
        """


# ---------------------------------------------------------------------------
# Synthetic equilibrium
# ---------------------------------------------------------------------------

class EquilibriumTokamakCircularSynthetic(EquilibriumAxisym):
    r"""Analytic test equilibrium for a circular cross-section tokamak.

    Uses a simplified Solov'ev-like ψ profile:

    .. math::

        \psi(R,Z) = \frac{(R^2 - R_0^2)^2}{4 R_0^2 a^2}
                    + \frac{Z^2}{a^2}

    This equals 0 at the magnetic axis :math:`(R_0, 0)` and 1 at
    :math:`(R_0 + a, 0)` and :math:`(R_0, a)`.

    The parabolic safety-factor profile is

    .. math::

        q(S) = q_0 + (q_1 - q_0) S^2

    The magnetic field components are

    .. math::

        B_\phi &= B_0 R_0 / R \\
        B_R    &= -\frac{\lambda}{R} \frac{\partial\psi}{\partial Z} \\
        B_Z    &=  \frac{\lambda}{R} \frac{\partial\psi}{\partial R}

    where :math:`\lambda = B_0 a / (q_0 R_0)`.

    .. note::

        This is **not** an exact MHD equilibrium.  It is intended
        solely for algorithm testing and demonstration.

    Parameters
    ----------
    R0:
        Major radius (m).  Default 1.85 (EAST-like).
    a:
        Minor radius (m).  Default 0.45.
    B0:
        On-axis toroidal field (T).  Default 2.0.
    q0:
        Safety factor at the magnetic axis.  Default 1.5.
    q1:
        Safety factor at the LCFS.  Default 4.0.
    nR, nZ:
        Number of grid points along R and Z.
    nS, nTET:
        Number of flux surfaces and poloidal angles for the
        (S, TET) mesh.
    """

    def __init__(
        self,
        R0: float = 1.85,
        a: float = 0.45,
        B0: float = 2.0,
        q0: float = 1.5,
        q1: float = 4.0,
        nR: int = 200,
        nZ: int = 200,
        nS: int = 100,
        nTET: int = 64,
    ) -> None:
        self.R0 = R0
        self.a = a
        self.B0 = B0
        self.q0 = q0
        self.q1 = q1

        self._R_mesh = np.linspace(R0 - 1.1 * a, R0 + 1.1 * a, nR)
        self._Z_mesh = np.linspace(-1.1 * a, 1.1 * a, nZ)
        self._S = np.linspace(0.01, 1.0, nS)
        self._TET = np.linspace(0.0, 2 * np.pi, nTET, endpoint=False)

        # Precompute psi on the R-Z grid for interpolation
        RR, ZZ = np.meshgrid(self._R_mesh, self._Z_mesh, indexing="ij")
        self._psi_grid = self._solovev_psi(RR, ZZ)
        self._psi_interp = RegularGridInterpolator(
            (self._R_mesh, self._Z_mesh), self._psi_grid, method="linear",
            bounds_error=False, fill_value=np.nan,
        )

    # ------------------------------------------------------------------
    # EquilibriumAxisym interface
    # ------------------------------------------------------------------

    @property
    def R_mesh(self) -> np.ndarray:
        return self._R_mesh

    @property
    def Z_mesh(self) -> np.ndarray:
        return self._Z_mesh

    @property
    def S(self) -> np.ndarray:
        return self._S

    @property
    def TET(self) -> np.ndarray:
        return self._TET

    def psi_norm(self, R: np.ndarray, Z: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=float)
        Z = np.asarray(Z, dtype=float)
        pts = np.stack((R.ravel(), Z.ravel()), axis=-1)
        return self._psi_interp(pts).reshape(R.shape)

    def q(self, S: np.ndarray) -> np.ndarray:
        S = np.asarray(S, dtype=float)
        return self.q0 + (self.q1 - self.q0) * S**2

    def B_field(
        self,
        R: np.ndarray,
        Z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        R = np.asarray(R, dtype=float)
        Z = np.asarray(Z, dtype=float)
        return self._solovev_field(R, Z)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _solovev_psi(self, R: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Raw ψ (0 at axis, 1 at LCFS)."""
        return (R**2 - self.R0**2) ** 2 / (4 * self.R0**2 * self.a**2) + Z**2 / self.a**2

    def _solovev_field(
        self, R: np.ndarray, Z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lam = self.B0 * self.a / (self.q0 * self.R0)
        dpsi_dR = (R**2 - self.R0**2) * R / (self.R0**2 * self.a**2)
        dpsi_dZ = 2 * Z / self.a**2
        BR = -lam / R * dpsi_dZ
        BZ = lam / R * dpsi_dR
        BPhi = self.B0 * self.R0 / R
        return BR, BZ, BPhi


# ---------------------------------------------------------------------------
# Utility: time interpolation
# ---------------------------------------------------------------------------

def time_linear_weighting(
    tdataseries: np.ndarray,
    dataseries: np.ndarray,
    tpoint: float,
) -> np.ndarray:
    """Linearly interpolate a data series to a single time point.

    Parameters
    ----------
    tdataseries:
        1-D array of time stamps in ascending order.
    dataseries:
        Array of data values.  The first axis must correspond to
        ``tdataseries`` (shape ``(nT, ...)``).
    tpoint:
        Target time (same units as ``tdataseries``).

    Returns
    -------
    ndarray
        Interpolated data at ``tpoint``, shape ``dataseries.shape[1:]``.

    Examples
    --------
    >>> t = np.array([0.0, 1.0, 2.0])
    >>> d = np.array([[0.0], [1.0], [4.0]])
    >>> time_linear_weighting(t, d, 0.5)
    array([0.5])
    """
    tdataseries = np.asarray(tdataseries, dtype=float)
    dataseries = np.asarray(dataseries, dtype=float)

    # Find surrounding indices
    idx = np.searchsorted(tdataseries, tpoint, side="right")
    idx = np.clip(idx, 1, len(tdataseries) - 1)
    t0, t1 = tdataseries[idx - 1], tdataseries[idx]
    w1 = (tpoint - t0) / (t1 - t0)
    w0 = 1.0 - w1
    return w0 * dataseries[idx - 1] + w1 * dataseries[idx]
