"""flux_surface.py
=================
Fourier-based magnetic flux surface representation, flux-surface coordinate map,
and X-point orbit tracing.

Classes
-------
FluxSurface
    A single r=const poloidal cross-section represented by Fourier series.
FluxSurfaceMap
    Continuous (r, θ, φ) coordinate system built from a set of FluxSurface objects.
XPointOrbit
    Three-dimensional periodic orbit of an X-point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# FluxSurface
# ---------------------------------------------------------------------------

@dataclass
class FluxSurface:
    """Single magnetic flux surface (poloidal cross-section, r=const).

    Parametrised by poloidal angle θ ∈ [0, 2π):

        R(θ) = Σ_n [R_cos[n] cos(nθ) + R_sin[n] sin(nθ)]
        Z(θ) = Σ_n [Z_cos[n] cos(nθ) + Z_sin[n] sin(nθ)]

    Attributes
    ----------
    r_norm : float
        Normalised flux-surface label (0 = axis, 1 = LCFS).
    phi : float
        Toroidal angle of this cross-section [rad].
    R_ax : float
        Magnetic axis R at this toroidal section [m].
    Z_ax : float
        Magnetic axis Z at this toroidal section [m].
    R_cos, R_sin, Z_cos, Z_sin : ndarray, shape (n_fourier+1,)
        Fourier coefficients.
    fit_residual : float
        RMS residual of the Fourier fit [m].
    """
    r_norm: float
    phi: float
    R_ax: float
    Z_ax: float
    R_cos: np.ndarray
    R_sin: np.ndarray
    Z_cos: np.ndarray
    Z_sin: np.ndarray
    fit_residual: float = 0.0

    # ------------------------------------------------------------------
    @classmethod
    def from_poincare(
        cls,
        R_pts: np.ndarray,
        Z_pts: np.ndarray,
        R_ax: float,
        Z_ax: float,
        r_norm: float,
        phi: float,
        n_fourier: int = 8,
    ) -> "FluxSurface":
        """Fit Fourier coefficients to a Poincaré scatter cloud.

        Parameters
        ----------
        R_pts, Z_pts : array-like
            Points on the flux surface in the (R, Z) plane.
        R_ax, Z_ax : float
            Magnetic axis position (used to compute poloidal angle).
        r_norm : float
            Normalised label for this surface.
        phi : float
            Toroidal angle of this section.
        n_fourier : int
            Number of Fourier harmonics (0..n_fourier).

        Returns
        -------
        FluxSurface
        """
        R_pts = np.asarray(R_pts, dtype=float)
        Z_pts = np.asarray(Z_pts, dtype=float)

        # Compute poloidal angle w.r.t. magnetic axis
        dR = R_pts - R_ax
        dZ = Z_pts - Z_ax
        theta = np.arctan2(dZ, dR)  # in (-π, π]

        # Sort by theta for a clean 1-to-1 mapping
        order = np.argsort(theta)
        theta_s = theta[order]
        R_s = R_pts[order]
        Z_s = Z_pts[order]

        # Build design matrix for least-squares Fourier fit
        ns = np.arange(n_fourier + 1)
        # columns: cos(0*θ), cos(1*θ), ..., cos(n*θ), sin(1*θ), ..., sin(n*θ)
        cos_mat = np.cos(np.outer(theta_s, ns))                     # (N, n+1)
        sin_mat = np.sin(np.outer(theta_s, ns[1:]))                 # (N, n)

        A = np.hstack([cos_mat, sin_mat])                           # (N, 2n+1)

        # Solve with least squares
        c_R, res_R, _, _ = np.linalg.lstsq(A, R_s, rcond=None)
        c_Z, res_Z, _, _ = np.linalg.lstsq(A, Z_s, rcond=None)

        R_cos = c_R[:n_fourier + 1]
        R_sin_full = np.zeros(n_fourier + 1)
        R_sin_full[1:] = c_R[n_fourier + 1:]

        Z_cos = c_Z[:n_fourier + 1]
        Z_sin_full = np.zeros(n_fourier + 1)
        Z_sin_full[1:] = c_Z[n_fourier + 1:]

        # RMS residual
        R_fit = A @ c_R
        Z_fit = A @ c_Z
        residual = float(np.sqrt(np.mean((R_s - R_fit) ** 2 + (Z_s - Z_fit) ** 2)))

        return cls(
            r_norm=r_norm,
            phi=phi,
            R_ax=R_ax,
            Z_ax=Z_ax,
            R_cos=R_cos,
            R_sin=R_sin_full,
            Z_cos=Z_cos,
            Z_sin=Z_sin_full,
            fit_residual=residual,
        )

    # ------------------------------------------------------------------
    def RZ(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate (R, Z) at given poloidal angles.

        Parameters
        ----------
        theta : array-like
            Poloidal angles [rad].

        Returns
        -------
        R, Z : ndarray
        """
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        ns = np.arange(len(self.R_cos))
        cos_mat = np.cos(np.outer(theta, ns))
        sin_mat = np.sin(np.outer(theta, ns))
        R = cos_mat @ self.R_cos + sin_mat @ self.R_sin
        Z = cos_mat @ self.Z_cos + sin_mat @ self.Z_sin
        return R, Z

    # ------------------------------------------------------------------
    def contains(self, R: float, Z: float) -> bool:
        """Test whether point (R, Z) is inside this flux surface.

        Uses the winding number algorithm on the Fourier-parametrised boundary.

        Parameters
        ----------
        R, Z : float
            Test point.

        Returns
        -------
        bool
        """
        theta = np.linspace(0, 2 * np.pi, 256, endpoint=False)
        Rc, Zc = self.RZ(theta)
        # Winding number via cross products
        dR = Rc - R
        dZ = Zc - Z
        winding = 0.0
        for i in range(len(theta)):
            j = (i + 1) % len(theta)
            cross = dR[i] * dZ[j] - dR[j] * dZ[i]
            dot = dR[i] * dR[j] + dZ[i] * dZ[j]
            winding += np.arctan2(cross, dot)
        return abs(winding) > np.pi  # |winding| ~ 2π for inside

    # ------------------------------------------------------------------
    def area(self) -> float:
        """Area enclosed by this flux surface using the shoelace formula.

        Returns
        -------
        float
            Area [m²].
        """
        theta = np.linspace(0, 2 * np.pi, 512, endpoint=False)
        R, Z = self.RZ(theta)
        # Shoelace
        return 0.5 * abs(np.dot(R, np.roll(Z, -1)) - np.dot(np.roll(R, -1), Z))


# ---------------------------------------------------------------------------
# FluxSurfaceMap
# ---------------------------------------------------------------------------

class FluxSurfaceMap:
    """Continuous (r, θ, φ) magnetic flux-surface coordinate system.

    Built from a collection of ``FluxSurface`` objects at discrete r values.
    Fourier coefficients are interpolated/extrapolated as cubic splines in r.

    Attributes
    ----------
    surfaces : list of FluxSurface
        Source surfaces (sorted by r_norm).
    r_nodes : ndarray
        r values of the spline knots.
    phi_ref : float
        Representative toroidal section (all surfaces should share the same φ).
    splines : dict
        CubicSpline objects for each Fourier coefficient.
    r_max : float
        Maximum r in the extrapolated domain.
    """

    def __init__(
        self,
        surfaces: List[FluxSurface],
        r_nodes: np.ndarray,
        splines: dict,
        phi_ref: float,
        R_ax: float,
        Z_ax: float,
        r_max: float,
    ):
        self.surfaces = surfaces
        self.r_nodes = r_nodes
        self.splines = splines
        self.phi_ref = phi_ref
        self.R_ax = R_ax
        self.Z_ax = Z_ax
        self.r_max = r_max

    # ------------------------------------------------------------------
    @classmethod
    def from_surfaces(
        cls,
        surfaces: List[FluxSurface],
        r_max_fit: float = 0.85,
        r_extrapolate_max: float = 2.0,
    ) -> "FluxSurfaceMap":
        """Build a FluxSurfaceMap from a list of FluxSurface objects.

        Surfaces with r_norm > r_max_fit are excluded from the spline fit
        (island healing: stochastic layer near rational surface).
        The spline is then extrapolated to r_extrapolate_max.

        Parameters
        ----------
        surfaces : list of FluxSurface
            At least 3 surfaces required for a cubic spline.
        r_max_fit : float
            Only surfaces with r_norm <= r_max_fit are used for fitting.
        r_extrapolate_max : float
            Maximum r for extrapolation (e.g. 2.0 covers coil positions).

        Returns
        -------
        FluxSurfaceMap
        """
        # Filter and sort
        fit_surfs = sorted(
            [s for s in surfaces if s.r_norm <= r_max_fit],
            key=lambda s: s.r_norm,
        )
        if len(fit_surfs) < 3:
            raise ValueError(
                f"Need at least 3 surfaces with r_norm <= {r_max_fit}, "
                f"got {len(fit_surfs)}"
            )

        r_nodes = np.array([s.r_norm for s in fit_surfs])

        n_fourier = len(fit_surfs[0].R_cos) - 1
        phi_ref = fit_surfs[0].phi
        R_ax = fit_surfs[0].R_ax
        Z_ax = fit_surfs[0].Z_ax

        # Build splines for each Fourier coefficient
        coeff_names = ["R_cos", "R_sin", "Z_cos", "Z_sin"]
        splines = {}
        for name in coeff_names:
            data = np.array([getattr(s, name) for s in fit_surfs])  # (n_surf, n_fourier+1)
            splines[name] = CubicSpline(r_nodes, data, extrapolate=True)

        # Store R_ax, Z_ax as functions of r (constant ≈ axis value, but spline if varies)
        splines["R_ax"] = CubicSpline(
            r_nodes, np.array([s.R_ax for s in fit_surfs]), extrapolate=True
        )
        splines["Z_ax"] = CubicSpline(
            r_nodes, np.array([s.Z_ax for s in fit_surfs]), extrapolate=True
        )

        return cls(
            surfaces=sorted(surfaces, key=lambda s: s.r_norm),
            r_nodes=r_nodes,
            splines=splines,
            phi_ref=phi_ref,
            R_ax=R_ax,
            Z_ax=Z_ax,
            r_max=r_extrapolate_max,
        )

    # ------------------------------------------------------------------
    def _fourier_coeffs_at_r(self, r: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate Fourier coefficients at arbitrary r (spline evaluation)."""
        R_cos = self.splines["R_cos"](r)
        R_sin = self.splines["R_sin"](r)
        Z_cos = self.splines["Z_cos"](r)
        Z_sin = self.splines["Z_sin"](r)
        return R_cos, R_sin, Z_cos, Z_sin

    def _RZ_at(self, r: float, theta: float) -> Tuple[float, float]:
        """Evaluate (R, Z) at (r, θ) using spline coefficients."""
        R_cos, R_sin, Z_cos, Z_sin = self._fourier_coeffs_at_r(r)
        ns = np.arange(len(R_cos))
        cos_v = np.cos(ns * theta)
        sin_v = np.sin(ns * theta)
        R = float(np.dot(cos_v, R_cos) + np.dot(sin_v, R_sin))
        Z = float(np.dot(cos_v, Z_cos) + np.dot(sin_v, Z_sin))
        return R, Z

    # ------------------------------------------------------------------
    def to_RZ(self, r: float, theta: float, phi: float) -> Tuple[float, float]:
        """Convert flux-surface coordinates (r, θ, φ) → (R, Z).

        Parameters
        ----------
        r : float
            Normalised flux-surface label.
        theta : float
            Poloidal angle [rad].
        phi : float
            Toroidal angle [rad] (used for future phi-dependent maps; currently
            the stored phi_ref is assumed representative for all sections).

        Returns
        -------
        R, Z : float
        """
        return self._RZ_at(r, theta)

    # ------------------------------------------------------------------
    def to_rtheta(
        self,
        R: float,
        Z: float,
        phi: float,
        r_init: float = 0.5,
    ) -> Tuple[float, float]:
        """Project a physical point (R, Z, φ) to flux-surface coordinates (r, θ).

        Uses scipy.optimize.minimize to find the (r, θ) that minimises the
        distance ‖(R_model(r,θ), Z_model(r,θ)) − (R, Z)‖.

        Parameters
        ----------
        R, Z : float
            Target physical coordinates [m].
        phi : float
            Toroidal angle (currently unused; future extension).
        r_init : float
            Initial guess for r.

        Returns
        -------
        r, theta : float
        """
        R_ax = float(self.splines["R_ax"](r_init))
        Z_ax = float(self.splines["Z_ax"](r_init))
        theta_init = float(np.arctan2(Z - Z_ax, R - R_ax))

        def objective(x):
            r_, t_ = x
            if r_ <= 0 or r_ > self.r_max:
                return 1e6
            Rm, Zm = self._RZ_at(r_, t_)
            return (Rm - R) ** 2 + (Zm - Z) ** 2

        res = minimize(
            objective,
            x0=[r_init, theta_init],
            method="Nelder-Mead",
            options={"xatol": 1e-7, "fatol": 1e-14, "maxiter": 5000},
        )
        r_opt, theta_opt = res.x
        # Wrap theta to [0, 2π)
        theta_opt = theta_opt % (2 * np.pi)
        return float(r_opt), float(theta_opt)

    # ------------------------------------------------------------------
    def project_points(
        self,
        R_arr: np.ndarray,
        Z_arr: np.ndarray,
        phi_arr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch projection of physical points to (r, θ).

        Parameters
        ----------
        R_arr, Z_arr, phi_arr : array-like
            Arrays of equal length.

        Returns
        -------
        r_arr, theta_arr : ndarray
        """
        R_arr = np.asarray(R_arr, dtype=float)
        Z_arr = np.asarray(Z_arr, dtype=float)
        phi_arr = np.asarray(phi_arr, dtype=float)
        n = len(R_arr)
        r_out = np.empty(n)
        theta_out = np.empty(n)
        for i in range(n):
            r_out[i], theta_out[i] = self.to_rtheta(R_arr[i], Z_arr[i], phi_arr[i])
        return r_out, theta_out


# ---------------------------------------------------------------------------
# XPointOrbit
# ---------------------------------------------------------------------------

@dataclass
class XPointOrbit:
    """Three-dimensional periodic orbit of a magnetic X-point.

    For an m/n island chain the X-point returns to its starting position
    after m toroidal turns.  When projected into the flux-surface coordinate
    system, the orbit traces out a curve θ_Xpt(φ).

    Attributes
    ----------
    phi_arr : ndarray
        Toroidal angles along the orbit [rad], 0 … 2π·m.
    R_arr, Z_arr : ndarray
        Physical coordinates along the orbit.
    period : int
        Number of toroidal turns (= m for m/n chain).
    r_arr : ndarray or None
        Normalised flux-surface label r(φ).
    theta_arr : ndarray or None
        Poloidal angle θ(φ) in the flux-surface map.
    """
    phi_arr: np.ndarray
    R_arr: np.ndarray
    Z_arr: np.ndarray
    period: int
    r_arr: Optional[np.ndarray] = None
    theta_arr: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    @classmethod
    def trace(
        cls,
        R_xpt: float,
        Z_xpt: float,
        phi0: float,
        period: int,
        field_cache: dict,
        dphi_out: float = 0.05,
    ) -> "XPointOrbit":
        """Trace the X-point orbit for ``period`` toroidal turns.

        Calls ``_cyna_trace_orbit`` from pyna.topo.topology_eval.

        Parameters
        ----------
        R_xpt, Z_xpt : float
            Starting (R, Z) of the X-point.
        phi0 : float
            Starting toroidal angle [rad].
        period : int
            Number of toroidal turns (e.g. 10 for m/n=10/3).
        field_cache : dict
            Dict with keys R_grid, Z_grid, Phi_grid, BR, BPhi, BZ.
        dphi_out : float
            Output step in φ [rad].

        Returns
        -------
        XPointOrbit
        """
        from pyna.topo.topology_eval import _cyna_trace_orbit, _FC

        fc = _FC(field_cache)
        phi_span = float(period) * 2.0 * np.pi

        R_t, Z_t, phi_t, _, alive_t = _cyna_trace_orbit(
            float(R_xpt), float(Z_xpt), float(phi0),
            phi_span, dphi_out, 0,
            dphi_out, 1e-4,
            fc.BR, fc.BPhi, fc.BZ,
            fc.Rg, fc.Zg, fc.Pg_ext,
        )
        alive = np.asarray(alive_t, dtype=bool)
        R_t = np.asarray(R_t)[alive]
        Z_t = np.asarray(Z_t)[alive]
        phi_t = np.asarray(phi_t)[alive]

        return cls(
            phi_arr=phi_t,
            R_arr=R_t,
            Z_arr=Z_t,
            period=period,
        )

    # ------------------------------------------------------------------
    def project_to_map(self, fmap: FluxSurfaceMap) -> "XPointOrbit":
        """Project the orbit into flux-surface coordinates.

        Fills ``r_arr`` and ``theta_arr`` by calling
        :meth:`FluxSurfaceMap.to_rtheta` for each point.

        Parameters
        ----------
        fmap : FluxSurfaceMap

        Returns
        -------
        XPointOrbit
            Self with r_arr and theta_arr populated.
        """
        n = len(self.phi_arr)
        r_arr = np.empty(n)
        theta_arr = np.empty(n)
        r_guess = 1.0
        for i in range(n):
            r, th = fmap.to_rtheta(self.R_arr[i], self.Z_arr[i], self.phi_arr[i], r_init=r_guess)
            r_arr[i] = r
            theta_arr[i] = th
            r_guess = r  # warm-start for next point

        return XPointOrbit(
            phi_arr=self.phi_arr,
            R_arr=self.R_arr,
            Z_arr=self.Z_arr,
            period=self.period,
            r_arr=r_arr,
            theta_arr=theta_arr,
        )

    # ------------------------------------------------------------------
    def theta_at_phi(self, phi: float) -> float:
        """Interpolate θ_Xpt at arbitrary φ (wrapped to [0, 2π)).

        Parameters
        ----------
        phi : float
            Toroidal angle [rad].

        Returns
        -------
        float
            θ_Xpt(φ) ∈ [0, 2π).

        Raises
        ------
        RuntimeError
            If ``project_to_map`` has not been called yet.
        """
        if self.theta_arr is None:
            raise RuntimeError("Call project_to_map() first to populate theta_arr.")
        itp = interp1d(
            self.phi_arr, self.theta_arr, kind="linear", fill_value="extrapolate"
        )
        return float(itp(phi)) % (2 * np.pi)
