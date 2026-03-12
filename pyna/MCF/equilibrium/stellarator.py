"""Analytic helical-ripple stellarator model.

Provides
--------
* :class:`SimpleStellarartor` — analytic stellarator with a helical-ripple
  perturbation and a linear safety-factor profile.
* :func:`simple_stellarator` — convenience factory.
"""
from __future__ import annotations

import numpy as np
from typing import List


class SimpleStellarartor:
    r"""Analytic single-helicity stellarator.

    The unperturbed equilibrium is a set of nested circular flux surfaces
    centred at (R0, 0).  The safety factor profile is linear in ψ:

    .. math::

        q(\psi) = q_0 + (q_1 - q_0) \psi

    A helical-ripple perturbation is added to the field-line ODE:

    .. math::

        \delta B_R = \epsilon_h B_0 \psi \cos(m_h \theta - n_h \phi)

    where :math:`\theta = \mathrm{atan2}(Z, R - R_0)`.

    Parameters
    ----------
    R0 : float
        Major radius (m).
    r0 : float
        Minor radius (m).
    B0 : float
        On-axis toroidal field (T).
    q0 : float
        Safety factor on axis.
    q1 : float
        Safety factor at LCFS (psi=1).
    m_h : int
        Helical poloidal mode number.
    n_h : int
        Helical toroidal mode number.
    epsilon_h : float
        Relative helical-ripple amplitude.
    """

    def __init__(
        self,
        R0: float = 3.0,
        r0: float = 0.3,
        B0: float = 1.0,
        q0: float = 1.1,
        q1: float = 5.0,
        m_h: int = 4,
        n_h: int = 4,
        epsilon_h: float = 0.05,
    ) -> None:
        self.R0 = float(R0)
        self.r0 = float(r0)
        self.B0 = float(B0)
        self.q0 = float(q0)
        self.q1 = float(q1)
        self.m_h = int(m_h)
        self.n_h = int(n_h)
        self.epsilon_h = float(epsilon_h)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def magnetic_axis(self) -> tuple[float, float]:
        """(R0, 0.0) — the toroidal magnetic axis."""
        return self.R0, 0.0

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def psi_ax(self, R, Z) -> np.ndarray:
        """Normalised ψ from magnetic axis = ((R-R0)² + Z²) / r0²."""
        R = np.asarray(R, dtype=float)
        Z = np.asarray(Z, dtype=float)
        return ((R - self.R0) ** 2 + Z ** 2) / self.r0 ** 2

    def q_of_psi(self, psi) -> np.ndarray:
        """Linear safety factor profile: q = q0 + (q1 - q0) * psi."""
        psi = np.asarray(psi, dtype=float)
        return self.q0 + (self.q1 - self.q0) * psi

    def resonant_psi(self, m: int, n: int) -> List[float]:
        """ψ values where q = m/n.

        For a linear profile q(ψ) = q0 + (q1-q0)ψ:
            ψ_res = (m/n - q0) / (q1 - q0)

        Returns an empty list if the resonance lies outside [0, 1].
        """
        q_target = m / n
        dq = self.q1 - self.q0
        if abs(dq) < 1e-12:
            return []
        psi_res = (q_target - self.q0) / dq
        if 0.0 < psi_res < 1.0:
            return [float(psi_res)]
        return []

    def field_func(self, rzphi_1d) -> np.ndarray:
        """Unit-tangent (dR, dZ, dphi) for the field-line ODE at (R, Z, phi).

        The field consists of:
        * Toroidal: B_phi = B0 R0 / R
        * Poloidal from q profile: B_pol = B_phi r / (q R)
        * Helical ripple perturbation: δBR (no δBphi)
        """
        rzphi_1d = np.asarray(rzphi_1d, dtype=float)
        R, Z, phi = rzphi_1d[0], rzphi_1d[1], rzphi_1d[2]

        theta = np.arctan2(Z, R - self.R0)
        psi = self.psi_ax(R, Z)
        q = float(self.q_of_psi(psi))

        # Cylindrical radius from axis
        r_minor = np.sqrt((R - self.R0) ** 2 + Z ** 2)

        # Equilibrium fields
        B_phi = self.B0 * self.R0 / R
        # Poloidal field: B_pol = B_phi * (r_minor / R) / q
        B_pol = B_phi * r_minor / (R * max(abs(q), 1e-3))
        # Direction of B_pol: perpendicular to radial direction in (R, Z) plane
        # Radial unit vector: (cos θ, sin θ), poloidal: (-sin θ, cos θ)
        if r_minor > 1e-10:
            BR0 = -B_pol * np.sin(theta)
            BZ0 =  B_pol * np.cos(theta)
        else:
            BR0 = 0.0
            BZ0 = 0.0

        # Helical ripple perturbation
        delta_BR = self.epsilon_h * self.B0 * psi * np.cos(self.m_h * theta - self.n_h * phi)

        BR_tot = BR0 + delta_BR
        BZ_tot = BZ0
        B_mag = np.sqrt(BR_tot ** 2 + BZ_tot ** 2 + B_phi ** 2) + 1e-30

        return np.array([
            BR_tot / B_mag,
            BZ_tot / B_mag,
            B_phi / (R * B_mag),
        ])

    def start_points_near_resonance(
        self,
        m: int,
        n: int,
        n_lines: int = 16,
        delta_psi: float = 0.06,
    ) -> np.ndarray:
        """Start points distributed around the q=m/n resonant surface.

        Parameters
        ----------
        m, n : int
            Mode numbers.
        n_lines : int
            Total number of field lines.
        delta_psi : float
            Half-width in ψ around the resonant surface.

        Returns
        -------
        ndarray of shape (n_lines, 3)
            Columns: (R, Z, phi=0).
        """
        psi_list = self.resonant_psi(m, n)
        if not psi_list:
            raise ValueError(f"q={m}/{n} surface not found in [0,1]")
        psi_res = psi_list[0]
        psi_arr = np.linspace(
            max(psi_res - delta_psi, 0.01),
            min(psi_res + delta_psi, 0.98),
            n_lines,
        )
        r_arr = np.sqrt(psi_arr) * self.r0
        pts = np.zeros((n_lines, 3))
        pts[:, 0] = self.R0 + r_arr   # start on midplane, outboard side
        pts[:, 1] = 0.0
        pts[:, 2] = 0.0
        return pts

    # ------------------------------------------------------------------
    # String representations
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return (
            f"SimpleStellarartor(R0={self.R0} m, r0={self.r0} m, "
            f"B0={self.B0} T, q=[{self.q0}, {self.q1}], "
            f"m_h={self.m_h}, n_h={self.n_h}, ε_h={self.epsilon_h})"
        )

    def __repr__(self) -> str:
        return (
            f"SimpleStellarartor(R0={self.R0!r}, r0={self.r0!r}, "
            f"B0={self.B0!r}, q0={self.q0!r}, q1={self.q1!r}, "
            f"m_h={self.m_h!r}, n_h={self.n_h!r}, "
            f"epsilon_h={self.epsilon_h!r})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def simple_stellarator(
    R0: float = 3.0,
    r0: float = 0.3,
    B0: float = 1.0,
    q0: float = 1.1,
    q1: float = 5.0,
    m_h: int = 4,
    n_h: int = 4,
    epsilon_h: float = 0.05,
) -> SimpleStellarartor:
    """Create a :class:`SimpleStellarartor` with keyword arguments."""
    return SimpleStellarartor(
        R0=R0, r0=r0, B0=B0,
        q0=q0, q1=q1,
        m_h=m_h, n_h=n_h,
        epsilon_h=epsilon_h,
    )
