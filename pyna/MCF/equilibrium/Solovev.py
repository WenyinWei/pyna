"""Solov'ev analytic tokamak equilibrium (Cerfon & Freidberg PoP 2010).

Provides
--------
* :class:`SolovevEquilibrium` — up-down symmetric Solov'ev equilibrium
  with shaped cross-section (elongation κ, triangularity δ).
* :func:`solovev_iter_like` — factory for a scaled ITER-like configuration.

References
----------
Cerfon & Freidberg, Phys. Plasmas 17, 032502 (2010).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar, brentq
from scipy.interpolate import UnivariateSpline
from functools import lru_cache


# ---------------------------------------------------------------------------
# Internal: C&F basis functions and their derivatives
# ---------------------------------------------------------------------------

def _basis_and_derivs(x: float, y: float):
    """Evaluate all 7 C&F homogeneous basis functions and their key
    partial derivatives at a single point (x, y) = (R/R0, Z/R0).

    Returns array of shape (7, 5):
        columns = [f, df/dx, df/dy, d²f/dx², d²f/dy²]
    """
    lx = np.log(x)
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    x6 = x4 * x2
    y2 = y * y
    y3 = y2 * y
    y4 = y2 * y2
    y5 = y4 * y
    y6 = y4 * y2

    out = np.zeros((7, 5))

    # f1 = 1
    out[0] = [1, 0, 0, 0, 0]

    # f2 = x²
    out[1] = [x2, 2 * x, 0, 2, 0]

    # f3 = y² - x²ln(x)
    out[2] = [
        y2 - x2 * lx,
        -x * (2 * lx + 1),
        2 * y,
        -(2 * lx + 3),
        2,
    ]

    # f4 = x⁴ - 4x²y²
    out[3] = [
        x4 - 4 * x2 * y2,
        4 * x3 - 8 * x * y2,
        -8 * x2 * y,
        12 * x2 - 8 * y2,
        -8 * x2,
    ]

    # f5 = 2y⁴ - 9y²x² + 3x⁴ln(x) - 12x²y²ln(x)
    f5 = 2 * y4 - 9 * y2 * x2 + 3 * x4 * lx - 12 * x2 * y2 * lx
    df5dx = -18 * x * y2 + 12 * x3 * lx + 3 * x3 - 24 * x * y2 * lx - 12 * x * y2
    df5dy = 8 * y3 - 18 * y * x2 - 24 * x2 * y * lx
    d2f5dxx = -54 * y2 + 21 * x2 + (36 * x2 - 24 * y2) * lx
    d2f5dyy = 24 * y2 - 18 * x2 - 24 * x2 * lx
    out[4] = [f5, df5dx, df5dy, d2f5dxx, d2f5dyy]

    # f6 = x⁶ - 12x⁴y² + 8x²y⁴
    f6 = x6 - 12 * x4 * y2 + 8 * x2 * y4
    df6dx = 6 * x5 - 48 * x3 * y2 + 16 * x * y4
    df6dy = -24 * x4 * y + 32 * x2 * y3
    d2f6dxx = 30 * x4 - 144 * x2 * y2 + 16 * y4
    d2f6dyy = -24 * x4 + 96 * x2 * y2
    out[5] = [f6, df6dx, df6dy, d2f6dxx, d2f6dyy]

    # f7 = 8y⁶ - 140y⁴x² + 75y²x⁴ - 15x⁶ln(x) + 180x⁴y²ln(x) - 120x²y⁴ln(x)
    f7 = (8 * y6 - 140 * y4 * x2 + 75 * y2 * x4
          - 15 * x6 * lx + 180 * x4 * y2 * lx - 120 * x2 * y4 * lx)
    df7dx = (-280 * y4 * x + 300 * y2 * x3
             - 90 * x5 * lx - 15 * x5
             + 720 * x3 * y2 * lx + 180 * x3 * y2
             - 240 * x * y4 * lx - 120 * x * y4)
    df7dy = (48 * y5 - 560 * y3 * x2 + 150 * y * x4
             + 360 * x4 * y * lx - 480 * x2 * y3 * lx)
    d2f7dxx = (-280 * y4 + 900 * y2 * x2
               - 450 * x4 * lx - 165 * x4
               + 2160 * x2 * y2 * lx + 1260 * x2 * y2
               - 240 * y4 * lx - 360 * y4)
    d2f7dyy = (240 * y4 - 1680 * y2 * x2 + 150 * x4
               + 360 * x4 * lx - 1440 * x2 * y2 * lx)
    out[6] = [f7, df7dx, df7dy, d2f7dxx, d2f7dyy]

    return out


def _particular_and_derivs(x: float, y: float, A: float):
    """Particular solution ψ_p = (1-A)x⁴/8 + A x²/2 and derivatives."""
    x2 = x * x
    x3 = x2 * x
    psi_p = (1 - A) * x2 * x2 / 8.0 + A * x2 / 2.0
    dp_dx = (1 - A) * x3 / 2.0 + A * x
    dp_dy = 0.0
    d2p_dxx = 3.0 * (1 - A) * x2 / 2.0 + A
    d2p_dyy = 0.0
    return np.array([psi_p, dp_dx, dp_dy, d2p_dxx, d2p_dyy])


def _solovev_coeffs(eps: float, kappa: float, delta: float, A: float = 0.0):
    """Solve C&F (2010) linear system for the up-down-symmetric case.

    Parameters
    ----------
    eps : float
        Inverse aspect ratio a / R0.
    kappa : float
        Elongation.
    delta : float
        Triangularity.
    A : float
        Free parameter controlling the partition between pressure and
        toroidal-current profiles (0 = pure-pressure Solov'ev).

    Returns
    -------
    c : ndarray, shape (7,)
        Coefficients of the 7 homogeneous basis functions.
    """
    alpha = np.arcsin(delta)

    # Three LCFS points (in normalised coords x = R/R0, y = Z/R0)
    x1, y1 = 1.0 + eps, 0.0          # outer equatorial
    x2, y2 = 1.0 - eps, 0.0          # inner equatorial
    x3, y3 = 1.0 - delta * eps, kappa * eps  # top

    # Curvature parameters (C&F eq. 15)
    N1 = -(1.0 + alpha) ** 2 / (kappa ** 2 * eps)
    N2 =  (1.0 - alpha) ** 2 / (kappa ** 2 * eps)
    N3 = -kappa / (eps * (1.0 - delta ** 2))

    # Build 7×7 matrix M and RHS vector b  (M c = b)
    M = np.zeros((7, 7))
    rhs = np.zeros(7)

    def row_from_combo(bd, combo_idx, combo_coeff):
        """bd: basis_and_derivs at a point; combo: (col_idx, deriv_idx, coeff)"""
        row = np.zeros(7)
        rhs_val = 0.0
        for (ci, di, cc) in combo_coeff:
            row += bd[:, di] * cc
        return row

    # Condition 1: ψ(x1, y1) = 0
    bd1 = _basis_and_derivs(x1, y1)
    pp1 = _particular_and_derivs(x1, y1, A)
    M[0] = bd1[:, 0]
    rhs[0] = -pp1[0]

    # Condition 2: ψ(x2, y2) = 0
    bd2 = _basis_and_derivs(x2, y2)
    pp2 = _particular_and_derivs(x2, y2, A)
    M[1] = bd2[:, 0]
    rhs[1] = -pp2[0]

    # Condition 3: ψ(x3, y3) = 0
    bd3 = _basis_and_derivs(x3, y3)
    pp3 = _particular_and_derivs(x3, y3, A)
    M[2] = bd3[:, 0]
    rhs[2] = -pp3[0]

    # Condition 4: ∂ψ/∂x(x3, y3) = 0  (top has horizontal tangent in LCFS)
    M[3] = bd3[:, 1]
    rhs[3] = -pp3[1]

    # Condition 5: ∂²ψ/∂y²(x1,y1) + N1 ∂ψ/∂x(x1,y1) = 0
    M[4] = bd1[:, 4] + N1 * bd1[:, 1]
    rhs[4] = -(pp1[4] + N1 * pp1[1])

    # Condition 6: ∂²ψ/∂y²(x2,y2) + N2 ∂ψ/∂x(x2,y2) = 0
    M[5] = bd2[:, 4] + N2 * bd2[:, 1]
    rhs[5] = -(pp2[4] + N2 * pp2[1])

    # Condition 7: ∂²ψ/∂x²(x3,y3) + N3 ∂ψ/∂y(x3,y3) = 0
    M[6] = bd3[:, 3] + N3 * bd3[:, 2]
    rhs[6] = -(pp3[3] + N3 * pp3[2])

    c = np.linalg.solve(M, rhs)
    return c


# ---------------------------------------------------------------------------
# Helper: evaluate C&F ψ and its gradients on arrays
# ---------------------------------------------------------------------------

def _eval_psi_cf(R_norm, Z_norm, c, A):
    """Evaluate the dimensionless C&F ψ (= 0 on LCFS) on array inputs."""
    x = np.asarray(R_norm, dtype=float)
    y = np.asarray(Z_norm, dtype=float)
    scalar = x.ndim == 0
    x = np.atleast_1d(x).ravel()
    y = np.atleast_1d(y).ravel()

    result = np.zeros_like(x)
    for k in range(len(x)):
        xk, yk = x[k], y[k]
        if xk <= 0:
            result[k] = np.nan
            continue
        lx = np.log(xk)
        x2 = xk * xk
        x4 = x2 * x2
        y2 = yk * yk
        y4 = y2 * y2
        y6 = y4 * y2
        x6 = x4 * x2

        f = np.array([
            1.0,
            x2,
            y2 - x2 * lx,
            x4 - 4 * x2 * y2,
            2 * y4 - 9 * y2 * x2 + 3 * x4 * lx - 12 * x2 * y2 * lx,
            x6 - 12 * x4 * y2 + 8 * x2 * y4,
            (8 * y6 - 140 * y4 * x2 + 75 * y2 * x4
             - 15 * x6 * lx + 180 * x4 * y2 * lx - 120 * x2 * y4 * lx),
        ])
        pp = (1 - A) * x4 / 8.0 + A * x2 / 2.0
        result[k] = pp + np.dot(c, f)

    if scalar:
        return float(result[0])
    return result.reshape(np.asarray(R_norm).shape)


def _eval_grad_cf(R_norm, Z_norm, c, A):
    """Return (∂ψ_CF/∂x, ∂ψ_CF/∂y) on array inputs."""
    x = np.asarray(R_norm, dtype=float)
    y = np.asarray(Z_norm, dtype=float)
    scalar = x.ndim == 0
    x = np.atleast_1d(x).ravel()
    y = np.atleast_1d(y).ravel()

    dpx = np.zeros_like(x)
    dpy = np.zeros_like(x)
    for k in range(len(x)):
        xk, yk = x[k], y[k]
        if xk <= 0:
            dpx[k] = dpy[k] = np.nan
            continue
        bd = _basis_and_derivs(xk, yk)
        pp = _particular_and_derivs(xk, yk, A)
        dpx[k] = pp[1] + np.dot(c, bd[:, 1])
        dpy[k] = pp[2] + np.dot(c, bd[:, 2])

    shape = np.asarray(R_norm).shape
    if scalar:
        return float(dpx[0]), float(dpy[0])
    return dpx.reshape(shape), dpy.reshape(shape)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SolovevEquilibrium:
    r"""Cerfon & Freidberg (2010) up-down-symmetric Solov'ev equilibrium.

    The poloidal flux function is

    .. math::

        \psi(R, Z) = B_0 R_0^2 \lambda \psi_\mathrm{CF}(R/R_0, Z/R_0)

    where ψ_CF is the dimensionless C&F solution (0 on LCFS) and λ is
    determined by the required safety factor q₀ at the magnetic axis.

    Parameters
    ----------
    R0 : float
        Major radius (m).
    a : float
        Minor radius (m).
    B0 : float
        On-axis toroidal field (T).
    kappa : float
        Elongation.
    delta : float
        Triangularity.
    q0 : float
        Safety factor at the magnetic axis.
    A : float
        C&F free parameter (0 = pure-pressure Solov'ev).
    """

    def __init__(
        self,
        R0: float = 6.2,
        a: float = 2.0,
        B0: float = 5.3,
        kappa: float = 1.7,
        delta: float = 0.33,
        q0: float = 1.5,
        A: float = 0.0,
    ) -> None:
        self.R0 = float(R0)
        self.a = float(a)
        self._B0 = float(B0)
        self.kappa = float(kappa)
        self.delta = float(delta)
        self.q0 = float(q0)
        self.A = float(A)

        eps = a / R0
        self._c = _solovev_coeffs(eps, kappa, delta, A)

        # Magnetic axis: ∇ψ_CF = 0, near (1, 0) in normalised coords
        self._x_ax, self._y_ax = self._find_axis()

        # Axis value of ψ_CF (positive, since ψ_CF > 0 inside)
        self._psi_cf_ax = _eval_psi_cf(self._x_ax, self._y_ax, self._c, self.A)

        # Scaling constant λ: set so that q(axis) = q0
        self._lam = self._compute_lambda()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def B0(self) -> float:
        """On-axis toroidal field (T)."""
        return self._B0

    @property
    def magnetic_axis(self) -> tuple[float, float]:
        """(R_axis, Z_axis) in metres."""
        return self._x_ax * self.R0, self._y_ax * self.R0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_axis(self) -> tuple[float, float]:
        """Locate the magnetic axis in normalised (x, y) space.

        The axis is where ψ_CF is minimal (most negative for standard C&F).
        Start near (1, 0).
        """
        from scipy.optimize import minimize

        def psi_cf_obj(xy):
            return float(_eval_psi_cf(xy[0], xy[1], self._c, self.A))

        result = minimize(psi_cf_obj, [1.0, 0.0], method='Nelder-Mead',
                          options={'xatol': 1e-9, 'fatol': 1e-12})
        return result.x[0], result.x[1]

    def _compute_lambda(self) -> float:
        """Determine λ so that q on axis = q0, using field-line integral."""
        # q ∝ 1/λ  →  compute q at λ=1 then scale
        q_at_lam1 = self._q_at_axis_lam1()
        if q_at_lam1 == 0.0 or not np.isfinite(q_at_lam1):
            return 1.0
        return q_at_lam1 / self.q0

    def _q_at_axis_lam1(self) -> float:
        """Numerically compute q at the magnetic axis for λ=1.

        Formula: q = (1/(2π λ)) ∮ dl_norm / (x |∇_xy ψ_CF|)
        At λ=1: q_lam1 = (1/2π) ∮ dl_norm / (x |∇_xy ψ_CF|)
        """
        # Use a small contour near the axis: psi_n = 0.01 (close to axis)
        psi_cf_target = 0.99 * self._psi_cf_ax   # close to axis (psi_n=0.01)
        x_ax = self._x_ax

        # Find dr on the outer midplane for this psi_CF value
        def f(dr):
            return _eval_psi_cf(x_ax + dr, 0.0, self._c, self.A) - psi_cf_target

        try:
            # psi_cf is negative inside, 0 on LCFS
            # f(0) = psi_cf_ax - 0.99*psi_cf_ax = 0.01*psi_cf_ax < 0
            # f(large_dr) = psi_cf(near_LCFS) - 0.99*psi_cf_ax > 0 (less negative - more negative)
            f_at_zero = f(1e-8)
            f_at_max = f(0.5 * self.a / self.R0)
            if f_at_zero * f_at_max >= 0:
                return self._q_near_axis_analytic()
            dr = brentq(f, 1e-8, 0.5 * self.a / self.R0)
        except Exception:
            return self._q_near_axis_analytic()

        # Trace an approximate ellipse, then project onto contour
        n_theta = 512
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        xc = x_ax + dr * np.cos(theta)
        yc = dr * self.kappa * np.sin(theta)

        # Newton-project onto ψ_CF = psi_cf_target
        for _ in range(8):
            psi_c = _eval_psi_cf(xc, yc, self._c, self.A)
            dpx, dpy = _eval_grad_cf(xc, yc, self._c, self.A)
            g2 = dpx ** 2 + dpy ** 2 + 1e-30
            step = (psi_c - psi_cf_target) / g2
            xc -= step * dpx
            yc -= step * dpy

        dpsi_dx, dpsi_dy = _eval_grad_cf(xc, yc, self._c, self.A)
        grad_norm = np.sqrt(dpsi_dx ** 2 + dpsi_dy ** 2) + 1e-30

        # Arc-length elements in normalised units
        dx = np.diff(xc, append=xc[0])
        dy = np.diff(yc, append=yc[0])
        dl_norm = np.sqrt(dx ** 2 + dy ** 2)

        # q_lam1 = (1/2π) ∮ dl_norm / (x |∇ψ_CF|)
        integrand = dl_norm / (xc * grad_norm)
        q_lam1 = float(np.sum(integrand) / (2 * np.pi))
        return q_lam1

    def _q_near_axis_analytic(self) -> float:
        """Analytic near-axis q estimate using second derivatives of ψ_CF."""
        x_ax = self._x_ax
        # Second derivatives at axis
        bd = _basis_and_derivs(x_ax, 0.0)
        pp = _particular_and_derivs(x_ax, 0.0, self.A)
        psi_xx = pp[3] + np.dot(self._c, bd[:, 3])
        psi_yy = pp[4] + np.dot(self._c, bd[:, 4])
        # Near axis: q ≈ 1 / (x_ax * √(psi_xx * psi_yy))
        # This comes from the harmonic oscillator formula for q near elliptic axis
        denom = x_ax * np.sqrt(abs(psi_xx) * abs(psi_yy))
        if denom < 1e-30:
            return 1.0
        return 1.0 / denom

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def psi(self, R, Z) -> np.ndarray:
        """Normalised poloidal flux ψ_norm (0 at axis, 1 at LCFS).

        Parameters
        ----------
        R, Z : array-like
            Cylindrical coordinates (m).
        """
        R = np.asarray(R, dtype=float)
        Z = np.asarray(Z, dtype=float)
        psi_cf = _eval_psi_cf(R / self.R0, Z / self.R0, self._c, self.A)
        # ψ_CF is negative inside (0 on LCFS, ψ_CF_ax < 0 at axis)
        # So ψ_norm = 1 - ψ_CF / ψ_CF_ax
        #  = 0 at axis (ψ_CF = ψ_CF_ax)
        #  = 1 at LCFS (ψ_CF = 0)
        psi_n = 1.0 - psi_cf / self._psi_cf_ax
        return psi_n

    def BR_BZ(self, R, Z) -> tuple[np.ndarray, np.ndarray]:
        """Radial and vertical magnetic field components (T)."""
        R = np.asarray(R, dtype=float)
        Z = np.asarray(Z, dtype=float)
        dpsi_dx, dpsi_dy = _eval_grad_cf(R / self.R0, Z / self.R0, self._c, self.A)
        # ψ_phys = λ B0 R0^2 ψ_CF
        # BR = -(1/R) ∂ψ_phys/∂Z = -(λ B0 R0 / R) ∂ψ_CF/∂y
        # BZ =  (1/R) ∂ψ_phys/∂R = +(λ B0 R0 / R) ∂ψ_CF/∂x
        scale = self._lam * self._B0 * self.R0 / R
        BR = -scale * dpsi_dy
        BZ =  scale * dpsi_dx
        return BR, BZ

    def Bphi(self, R) -> np.ndarray:
        """Toroidal field component (T) — vacuum approximation B0 R0 / R."""
        R = np.asarray(R, dtype=float)
        return self._B0 * self.R0 / R

    def psi_lcfs(self) -> float:
        """Return the physical poloidal flux ψ_phys at the LCFS (= 0 in C&F normalisation).

        In our normalisation ``psi()`` returns ψ_norm in [0,1], so the LCFS
        corresponds to ψ_norm = 1.  This convenience method returns that value.
        """
        return 1.0

    def flux_surface(
        self,
        psi_norm: float,
        n_theta: int = None,
        n_grid: int = 300,
    ) -> tuple:
        """Return (R, Z) arrays tracing the flux surface at normalised flux *psi_norm*.

        ``psi_norm = 0`` is the magnetic axis, ``psi_norm = 1`` is the LCFS.

        Uses matplotlib contour extraction on a fine (R, Z) grid.

        Parameters
        ----------
        psi_norm : float
            Normalised poloidal flux label in (0, 1].
        n_theta : int or None
            Not used (kept for API compatibility).
        n_grid : int
            Grid resolution for the contour search.

        Returns
        -------
        R_arr, Z_arr : ndarray
            Coordinate arrays of the flux-surface contour.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        psi_norm = float(np.clip(psi_norm, 1e-4, 1.0))

        R_min = self.R0 - 1.25 * self.a
        R_max = self.R0 + 1.25 * self.a
        Z_min = -1.25 * self.a * self.kappa
        Z_max = +1.25 * self.a * self.kappa

        R_grid = np.linspace(R_min, R_max, n_grid)
        Z_grid = np.linspace(Z_min, Z_max, n_grid)
        RR, ZZ = np.meshgrid(R_grid, Z_grid)

        PSI = self.psi(RR, ZZ)  # ψ_norm on grid

        fig, ax = plt.subplots()
        try:
            cs = ax.contour(RR, ZZ, PSI, levels=[psi_norm])
            paths = cs.get_paths()
            if not paths:
                raise ValueError(f"No contour found for psi_norm={psi_norm}")
            # Pick the longest path (the main flux surface)
            path = max(paths, key=lambda p: len(p.vertices))
            R_arr = path.vertices[:, 0]
            Z_arr = path.vertices[:, 1]
        finally:
            plt.close(fig)

        return R_arr, Z_arr

    def q_profile(self, psi_values, n_theta: int = 512) -> np.ndarray:
        """Safety factor profile by poloidal field-line integration.

        Parameters
        ----------
        psi_values : array-like
            ψ_norm values (0–1) at which to compute q.
        n_theta : int
            Number of poloidal angle steps per flux surface.

        Returns
        -------
        ndarray
            q values at each ψ_norm.
        """
        psi_values = np.asarray(psi_values, dtype=float)
        q_out = np.full_like(psi_values, np.nan)

        R_ax, Z_ax = self.magnetic_axis
        x_ax = R_ax / self.R0

        for i, psi_n in enumerate(psi_values):
            if psi_n <= 0.0 or psi_n >= 1.0:
                continue
            # Find the minor-radius contour on the midplane for this ψ
            psi_cf_target = (1.0 - psi_n) * self._psi_cf_ax
            # R > R_ax (outer midplane)
            def f_outer(dr):
                return _eval_psi_cf(x_ax + dr, 0.0, self._c, self.A) - psi_cf_target
            try:
                dr = brentq(f_outer, 1e-8, 0.9 * self.a / self.R0)
            except Exception:
                continue

            # Approximate flux contour as tilted ellipse — use Newton iteration
            # to trace the actual ψ=const contour
            theta_arr = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
            dth = 2 * np.pi / n_theta

            # Initial guess: circle of radius dr
            x_c = x_ax + dr * np.cos(theta_arr)
            y_c = (dr * self.kappa) * np.sin(theta_arr)

            # Newton-project each point onto the contour (1 iteration)
            for _ in range(5):
                psi_c = _eval_psi_cf(x_c, y_c, self._c, self.A)
                dpx, dpy = _eval_grad_cf(x_c, y_c, self._c, self.A)
                g2 = dpx ** 2 + dpy ** 2 + 1e-30
                step = (psi_c - psi_cf_target) / g2
                x_c -= step * dpx
                y_c -= step * dpy

            R_c = x_c * self.R0
            # Arc length element dl
            dx = np.diff(x_c, append=x_c[0])
            dy = np.diff(y_c, append=y_c[0])
            dl = np.sqrt(dx ** 2 + dy ** 2) * self.R0

            dpsi_dx, dpsi_dy = _eval_grad_cf(x_c, y_c, self._c, self.A)
            grad_cf_norm = np.sqrt(dpsi_dx ** 2 + dpsi_dy ** 2) + 1e-30

            # q = (1/(2π λ)) ∮ dl_norm / (x_c |∇ψ_CF|)
            # dl_norm = arc length in normalised units (divide by R0)
            dl_norm = dl / self.R0
            integrand2 = dl_norm / (x_c * grad_cf_norm)
            q_out[i] = float(np.sum(integrand2) / (2 * np.pi * self._lam))

        return q_out

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Grid helpers: J and p on R-Z grid
    # ------------------------------------------------------------------

    def J_grid(self, R_arr, Z_arr):
        """Compute current density J = ∇×B/μ₀ on R-Z grid.

        For an axisymmetric equilibrium::

            J_R   = -(1/μ₀) ∂Bphi/∂Z
            J_Z   = (1/μ₀) (1/R) ∂(R·Bphi)/∂R
            J_phi = (1/μ₀) (∂BR/∂Z − ∂BZ/∂R)

        Parameters
        ----------
        R_arr, Z_arr : array-like
            1D arrays of R and Z grid points.

        Returns
        -------
        JR, JZ, Jphi : ndarray, shape (nR, nZ)
            Current density components on meshgrid of R_arr × Z_arr.
        """
        mu0 = 4e-7 * np.pi
        R_arr = np.asarray(R_arr, dtype=float)
        Z_arr = np.asarray(Z_arr, dtype=float)
        dR = R_arr[1] - R_arr[0]
        dZ = Z_arr[1] - Z_arr[0]

        RR, ZZ = np.meshgrid(R_arr, Z_arr, indexing='ij')
        BR_2d, BZ_2d = self.BR_BZ(RR, ZZ)
        Bphi_2d = self.Bphi(RR)

        dBphi_dZ = np.gradient(Bphi_2d, dZ, axis=1)
        d_RBphi_dR = np.gradient(RR * Bphi_2d, dR, axis=0)
        dBR_dZ = np.gradient(BR_2d, dZ, axis=1)
        dBZ_dR = np.gradient(BZ_2d, dR, axis=0)

        JR   = -dBphi_dZ / mu0
        JZ   = d_RBphi_dR / (RR * mu0)
        Jphi = (dBR_dZ - dBZ_dR) / mu0
        return JR, JZ, Jphi

    def p_grid(self, R_arr, Z_arr):
        """Compute pressure p from force balance J×B = ∇p on R-Z grid.

        Integrates (J×B)_R along R from left to right, then (J×B)_Z
        along Z, and averages the four directions for better accuracy.

        Parameters
        ----------
        R_arr, Z_arr : array-like
            1D grid arrays.

        Returns
        -------
        p : ndarray, shape (nR, nZ)
            Pressure on the R-Z grid.
        """
        R_arr = np.asarray(R_arr, dtype=float)
        Z_arr = np.asarray(Z_arr, dtype=float)
        dR = R_arr[1] - R_arr[0]
        dZ = Z_arr[1] - Z_arr[0]

        nR, nZ = len(R_arr), len(Z_arr)
        JR, JZ, Jphi = self.J_grid(R_arr, Z_arr)
        RR, ZZ = np.meshgrid(R_arr, Z_arr, indexing='ij')
        BR_2d, BZ_2d = self.BR_BZ(RR, ZZ)
        Bphi_2d = self.Bphi(RR)

        # J×B components:
        # (J×B)_R = Jphi*BZ - JZ*Bphi
        # (J×B)_Z = JR*Bphi - Jphi*BR
        JxB_R = Jphi * BZ_2d - JZ * Bphi_2d
        JxB_Z = JR * Bphi_2d - Jphi * BR_2d

        # Integrate along R (two directions)
        p_R_fwd = np.zeros((nR, nZ))
        p_R_bwd = np.zeros((nR, nZ))
        for i in range(1, nR):
            p_R_fwd[i, :] = p_R_fwd[i - 1, :] + 0.5 * (JxB_R[i, :] + JxB_R[i - 1, :]) * dR
        for i in range(nR - 2, -1, -1):
            p_R_bwd[i, :] = p_R_bwd[i + 1, :] - 0.5 * (JxB_R[i + 1, :] + JxB_R[i, :]) * dR

        # Integrate along Z (two directions)
        p_Z_fwd = np.zeros((nR, nZ))
        p_Z_bwd = np.zeros((nR, nZ))
        for j in range(1, nZ):
            p_Z_fwd[:, j] = p_Z_fwd[:, j - 1] + 0.5 * (JxB_Z[:, j] + JxB_Z[:, j - 1]) * dZ
        for j in range(nZ - 2, -1, -1):
            p_Z_bwd[:, j] = p_Z_bwd[:, j + 1] - 0.5 * (JxB_Z[:, j + 1] + JxB_Z[:, j]) * dZ

        p = 0.25 * (p_R_fwd + p_R_bwd + p_Z_fwd + p_Z_bwd)
        # Shift so minimum is zero (pressure is a relative quantity in Solov'ev)
        p -= p.min()
        return p

    # X/O-point and flux surface methods
    # ------------------------------------------------------------------

    def find_xpoint(self, R0_guess=None, Z0_guess=None):
        """Find divertor X-point (lower single null for typical params).

        Returns
        -------
        (R_xpt, Z_xpt) : tuple of float
            X-point location in metres.
        """
        from scipy.optimize import minimize
        if R0_guess is None:
            R0_guess = self.R0 - 0.3 * self.a
        if Z0_guess is None:
            Z0_guess = -self.a * self.kappa * 0.95

        def Bpol2(rz):
            BR, BZ = self.BR_BZ(rz[0], rz[1])
            return float(BR ** 2 + BZ ** 2)

        res = minimize(Bpol2, [R0_guess, Z0_guess], method='Nelder-Mead',
                       options={'xatol': 1e-7, 'fatol': 1e-20, 'maxiter': 10000})
        return tuple(res.x)

    def find_opoint(self):
        """Find magnetic axis (O-point).

        Returns
        -------
        (R_axis, Z_axis) : tuple of float
            Magnetic axis in metres.
        """
        return self.magnetic_axis

    def psi_lcfs(self) -> float:
        """ψ_norm value at the last closed flux surface (= 1.0 by definition)."""
        return 1.0

    def resonant_psi(self, m: int, n: int, n_scan: int = 200) -> list:
        """Find ψ_norm values where q(ψ) = m/n.

        Parameters
        ----------
        m, n : int
            Poloidal and toroidal mode numbers; resonance at q = m/n.
        n_scan : int
            Number of points to scan in ψ_norm ∈ [0.02, 0.97].

        Returns
        -------
        list of float
            ψ_norm values where q = m/n (sorted ascending).
        """
        from scipy.optimize import brentq
        q_target = m / n
        psi_arr = np.linspace(0.02, 0.97, n_scan)
        q_arr = self.q_profile(psi_arr)
        results = []
        for i in range(len(psi_arr) - 1):
            if (q_arr[i] - q_target) * (q_arr[i + 1] - q_target) < 0:
                try:
                    psi_r = brentq(
                        lambda p: float(self.q_profile(np.array([p]))[0]) - q_target,
                        psi_arr[i], psi_arr[i + 1], xtol=1e-6,
                    )
                    results.append(float(psi_r))
                except Exception:
                    pass
        return results

    def flux_surface(self, psi_norm: float, n_pts: int = 300) -> tuple:
        """Return (R, Z) contour of the flux surface at normalised ψ.

        Parameters
        ----------
        psi_norm : float
            0 → magnetic axis, 1 → LCFS.
        n_pts : int
            Grid resolution for contour extraction.

        Returns
        -------
        R_arr, Z_arr : ndarray
            Coordinates of the flux-surface contour in metres.

        Notes
        -----
        Uses matplotlib contour extraction with the Agg (non-display) backend.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        R_arr = np.linspace(self.R0 - 1.3 * self.a, self.R0 + 1.3 * self.a, n_pts)
        Z_arr = np.linspace(-1.3 * self.a * self.kappa,
                             1.3 * self.a * self.kappa, n_pts)
        RR, ZZ = np.meshgrid(R_arr, Z_arr)
        PSI = self.psi(RR, ZZ)

        fig, ax = plt.subplots()
        cs = ax.contour(RR, ZZ, PSI, levels=[float(psi_norm)])
        # matplotlib >= 3.8 removed .collections; use .get_paths() directly
        if hasattr(cs, 'collections'):
            paths = cs.collections[0].get_paths()
        else:
            paths = cs.get_paths()
        plt.close(fig)

        if not paths:
            raise ValueError(f"No contour found for psi_norm={psi_norm}")

        # Return longest path (should be the main flux surface)
        path = max(paths, key=lambda p: len(p.vertices))
        return path.vertices[:, 0], path.vertices[:, 1]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _solovev_coeffs_single_null(
    eps: float,
    kappa: float,
    delta_u: float,
    delta_l: float,
    kappa_x: float,
    A: float = 0.0,
):
    """Solve the C&F boundary-value system for a **single-null** equilibrium.

    The up-down-symmetric formulation uses 3 LCFS points + 4 curvature/tangent
    conditions.  Here we replace the bottom-curvature condition with a
    requirement that the lower separatrix X-point (where B_pol = 0) lies on
    the LCFS (ψ = 0 in C&F normalisation).

    Parameters
    ----------
    eps : float
        Inverse aspect ratio a/R₀.
    kappa : float
        Upper elongation (height of the LCFS top above the midplane, in units
        of the minor radius).
    delta_u : float
        Upper triangularity.
    delta_l : float
        Lower triangularity (controls the horizontal position of the X-point).
    kappa_x : float
        Poloidal coordinate of the X-point in units of the minor radius,
        i.e. |Z_x| = kappa_x * a.
    A : float, optional
        C&F free parameter.  Default 0.

    Returns
    -------
    c : ndarray, shape (7,)
        Homogeneous-basis coefficients.
    xpt_norm : tuple (x_x, y_x)
        Normalised (R/R₀, Z/R₀) of the imposed X-point.
    """
    alpha_u = np.arcsin(delta_u)

    # LCFS boundary points (normalised)
    x1, y1 = 1.0 + eps, 0.0                           # outer equatorial
    x2, y2 = 1.0 - eps, 0.0                           # inner equatorial
    x3, y3 = 1.0 - delta_u * eps, kappa * eps         # upper top
    x_x   = 1.0 - delta_l * eps                       # X-point (lower)
    y_x   = -kappa_x * eps

    # Curvature parameters (C&F eq. 15) for the upper boundary
    N1 = -(1.0 + alpha_u) ** 2 / (kappa ** 2 * eps)
    N2 =  (1.0 - alpha_u) ** 2 / (kappa ** 2 * eps)

    M   = np.zeros((7, 7))
    rhs = np.zeros(7)

    bd1 = _basis_and_derivs(x1, y1);   pp1 = _particular_and_derivs(x1, y1, A)
    bd2 = _basis_and_derivs(x2, y2);   pp2 = _particular_and_derivs(x2, y2, A)
    bd3 = _basis_and_derivs(x3, y3);   pp3 = _particular_and_derivs(x3, y3, A)
    bdx = _basis_and_derivs(x_x, y_x); ppx = _particular_and_derivs(x_x, y_x, A)

    # 1 – ψ(outer equatorial) = 0
    M[0]  = bd1[:, 0];                  rhs[0]  = -pp1[0]
    # 2 – ψ(inner equatorial) = 0
    M[1]  = bd2[:, 0];                  rhs[1]  = -pp2[0]
    # 3 – ψ(upper top) = 0
    M[2]  = bd3[:, 0];                  rhs[2]  = -pp3[0]
    # 4 – ∂ψ/∂x(upper top) = 0  (horizontal tangent)
    M[3]  = bd3[:, 1];                  rhs[3]  = -pp3[1]
    # 5 – curvature at outer equatorial
    M[4]  = bd1[:, 4] + N1 * bd1[:, 1]; rhs[4] = -(pp1[4] + N1 * pp1[1])
    # 6 – curvature at inner equatorial
    M[5]  = bd2[:, 4] + N2 * bd2[:, 1]; rhs[5] = -(pp2[4] + N2 * pp2[1])
    # 7 – ψ(X-point) = 0  (X-point lies on the LCFS)
    M[6]  = bdx[:, 0];                  rhs[6]  = -ppx[0]

    c = np.linalg.solve(M, rhs)
    return c, (x_x, y_x)


def solovev_single_null(
    R0: float = 1.86,
    a: float = 0.595,
    B0: float = 5.3,
    kappa: float = 1.8,
    delta_u: float = 0.33,
    delta_l: float = 0.40,
    kappa_x: float = 1.5,
    q0: float = 1.5,
    A: float = 0.0,
) -> SolovevEquilibrium:
    """Create a **single-null divertor** Solov'ev equilibrium.

    The lower X-point is placed on the last closed flux surface by choosing
    boundary conditions that force ψ = 0 at an asymmetric lower point.  The
    Grad-Shafranov solve uses the same 7 C&F (2010) homogeneous basis
    functions, but the 7th boundary condition is replaced by ψ(X-point) = 0
    instead of the bottom-curvature condition used in the up-down-symmetric
    case.

    Parameters
    ----------
    R0 : float
        Major radius (m).  Default 1.86.
    a : float
        Minor radius (m).  Default 0.595.
    B0 : float
        On-axis toroidal field (T).  Default 5.3.
    kappa : float
        Upper elongation.  Default 1.8.
    delta_u : float
        Upper triangularity.  Default 0.33.
    delta_l : float
        Lower triangularity (X-point horizontal position).  Default 0.40.
    kappa_x : float
        |Z_x| / a of the lower X-point.  Default 1.5.
    q0 : float
        Safety factor at the magnetic axis.  Default 1.5.
    A : float
        C&F free parameter.  Default 0.

    Returns
    -------
    SolovevEquilibrium
        An equilibrium whose lower separatrix X-point satisfies B_pol ≈ 0
        and ψ_norm ≈ 1.

    Examples
    --------
    >>> eq = solovev_single_null()
    >>> R_x, Z_x = eq.find_xpoint()
    >>> print(f"X-point: R={R_x:.3f} m  Z={Z_x:.3f} m")
    """
    eps = a / R0
    c_sn, _xpt_norm = _solovev_coeffs_single_null(
        eps, kappa, delta_u, delta_l, kappa_x, A
    )

    # Build the equilibrium using the up-down-symmetric class (same basis)
    eq = SolovevEquilibrium(
        R0=R0, a=a, B0=B0,
        kappa=kappa, delta=delta_u,
        q0=q0, A=A,
    )

    # Inject the asymmetric coefficients
    eq._c = c_sn
    eq._x_ax, eq._y_ax = eq._find_axis()
    eq._psi_cf_ax = _eval_psi_cf(eq._x_ax, eq._y_ax, c_sn, A)
    eq._lam = eq._compute_lambda()

    return eq


def solovev_iter_like(scale: float = 1.0) -> SolovevEquilibrium:
    """Create a scaled ITER-like Solov'ev equilibrium.

    Parameters
    ----------
    scale : float
        Linear scale factor.  ``scale=1.0`` → full ITER size (R0≈6.2 m).
        ``scale=0.3`` → ~30% size (R0≈1.86 m, EAST-sized).

    Returns
    -------
    SolovevEquilibrium
    """
    return SolovevEquilibrium(
        R0=6.2 * scale,
        a=2.0 * scale,
        B0=5.3,
        kappa=1.7,
        delta=0.33,
        q0=1.5,
    )
