"""island_healed_coords.py
==========================
# NOTE: IslandChainOrbit references in this file are duck-typed; no hard import needed.
Island-constrained healed flux-surface coordinate system.

Motivation
----------
Standard Fourier-spline extrapolation of flux surfaces into the island
region is physically wrong: it treats the separatrix layer as if it were
a smooth continuation of nested surfaces, whereas in reality the X- and
O-point orbits *define* the topology there.

This module constructs a globally smooth (r, θ, φ) coordinate system that:

1. Reproduces exact closed nested surfaces for r �?r_inner (from Poincaré
   data, typically r_inner �?0.82).
2. Anchors the coordinate θ to the **Ocyc orbit** at r = r_island
   (island centre �?θ_O = 0 convention, or any chosen θ_O) and to the
   **Xcyc orbit** at r = r_island, θ = θ_X.
3. Smoothly blends from the inner Fourier description into the island
   frame using a monotone radial blend, then continues outward for
   r > r_island (useful for projecting external coil positions).

The X/O ring orbits are ``IslandChainOrbit`` objects from
``pyna.topo.island_chain``.  Their continuous (R, Z, φ) trajectories
define, at each toroidal angle φ, a local reference frame in the poloidal
plane.

API
---
Build coordinate system::

    from pyna.topo.island_healed_coords import IslandHealedCoordMap

    coord = IslandHealedCoordMap.build(
        inner_surfaces,          # list[FluxSurface], r �?r_inner
        xring_orbit,             # IslandChainOrbit (X-type fixed points)
        oring_orbit,             # IslandChainOrbit (O-type fixed points)
        r_island=1.0,            # normalised r assigned to the island layer
        r_inner_fit=0.82,        # only surfaces �?this r used for inner fit
        n_fourier=8,
        blend_width=0.08,        # radial width of blend zone
    )

Project a coil position::

    r_c, theta_c = coord.to_rtheta(R_coil, Z_coil, phi_coil)
    eR, eZ = coord.grad_r_direction(r_c, theta_c, phi_coil)
    B_radial = eR * BR_coil + eZ * BZ_coil

Key design decisions
--------------------
* **θ origin**: at each φ, θ = 0 is defined by the Ocyc position
  (mapped from the Ocyc orbit via the inner Fourier map at r_island).
  θ = θ_X is defined by the Xcyc position.
* **Fourier frame**: for r �?r_inner we use the usual geometric θ
  (arctan2 from axis), fitted by Fourier series.  At r = r_island we
  anchor to the X/O frame.  Between r_inner and r_island we blend
  via a smooth sigmoid.
* **Toroidal dependence**: the inner Fourier surfaces are built at
  discrete φ sections and interpolated.  The X/O ring orbits provide
  continuous φ coverage via field-line tracing.

References
----------
* Hudson & Dewar (1997): ghost surfaces and the definition of KAM surfaces
  near island chains.
* Cary & Hanson (1986): magnetic coordinates in the presence of island
  chains.
* Boozer (2004): MHD equilibrium and stability, sec. 5.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline, interp1d, make_interp_spline
from scipy.optimize import minimize, brentq


# ---------------------------------------------------------------------------
# Helper: build Fourier design matrix
# ---------------------------------------------------------------------------

def _fourier_matrix(theta: np.ndarray, n_fourier: int) -> np.ndarray:
    """Design matrix [1, cos(θ), sin(θ), cos(2θ), sin(2θ), ...] shape (N, 2n+1)."""
    cols = [np.ones(len(theta))]
    for n in range(1, n_fourier + 1):
        cols.append(np.cos(n * theta))
        cols.append(np.sin(n * theta))
    return np.column_stack(cols)


def _fourier_eval(coeffs: np.ndarray, theta: float, n_fourier: int) -> float:
    """Evaluate Fourier series at scalar θ."""
    val = coeffs[0]
    for n in range(1, n_fourier + 1):
        val += coeffs[2 * n - 1] * np.cos(n * theta)
        val += coeffs[2 * n] * np.sin(n * theta)
    return float(val)


def _sigmoid_blend(r: float, r_low: float, r_high: float) -> float:
    """Smooth sigmoid from 0 at r_low to 1 at r_high (C-infinity)."""
    if r <= r_low:
        return 0.0
    if r >= r_high:
        return 1.0
    t = (r - r_low) / (r_high - r_low)
    # Use smoothstep-3: 6t^5 - 15t^4 + 10t^3
    return t * t * t * (t * (t * 6 - 15) + 10)


# ---------------------------------------------------------------------------
# InnerFourierSection: one phi-section of the inner Fourier surface map
# ---------------------------------------------------------------------------

@dataclass
class InnerFourierSection:
    """Fourier surface description at one toroidal section φ_ref.

    Stores CubicSpline(r) objects for each Fourier coefficient, built
    from inner Poincaré surfaces (r �?r_inner_fit).

    Attributes
    ----------
    phi_ref : float
        Toroidal angle of this section [rad].
    R_ax, Z_ax : float
        Magnetic axis position at φ_ref [m].
    r_nodes : ndarray
        Normalised r values of the spline nodes.
    splines_R, splines_Z : list of CubicSpline
        One spline per Fourier coefficient (length 2*n_fourier+1).
    n_fourier : int
    """
    phi_ref: float
    R_ax: float
    Z_ax: float
    r_nodes: np.ndarray
    splines_R: list   # list[CubicSpline]
    splines_Z: list   # list[CubicSpline]
    n_fourier: int

    # ------------------------------------------------------------------
    @classmethod
    def from_poincare_surfaces(
        cls,
        phi_ref: float,
        R_ax: float,
        Z_ax: float,
        r_norms: np.ndarray,
        R_surf: np.ndarray,   # shape (n_r, n_theta)
        Z_surf: np.ndarray,   # shape (n_r, n_theta)
        theta_arr: np.ndarray,
        n_fourier: int = 8,
    ) -> 'InnerFourierSection':
        """Build from pre-computed surface (R, Z) grids.

        Parameters
        ----------
        phi_ref : float
            Toroidal angle [rad].
        R_ax, Z_ax : float
            Magnetic axis position.
        r_norms : ndarray, shape (n_r,)
            Normalised r labels (0 = axis, 1 = LCFS).
        R_surf, Z_surf : ndarray, shape (n_r, n_theta)
            Surface coordinates sampled at ``theta_arr`` angles.
        theta_arr : ndarray, shape (n_theta,)
            Poloidal angles [rad].
        n_fourier : int
            Number of Fourier harmonics.
        """
        n_r = len(r_norms)
        n_coeff = 1 + 2 * n_fourier
        A = _fourier_matrix(theta_arr, n_fourier)          # (n_theta, n_coeff)
        A_pinv = np.linalg.pinv(A)                         # (n_coeff, n_theta)

        coeffs_R = np.zeros((n_r, n_coeff))
        coeffs_Z = np.zeros((n_r, n_coeff))
        for ir in range(n_r):
            coeffs_R[ir] = A_pinv @ R_surf[ir]
            coeffs_Z[ir] = A_pinv @ Z_surf[ir]

        # Axis constraint: r=0 �?single point (axis)
        # Only prepend r=0 if not already present in r_norms
        if r_norms[0] > 1e-6:
            r_ext = np.concatenate([[0.0], r_norms])
            c0_R = np.zeros(n_coeff); c0_R[0] = R_ax   # only DC term
            c0_Z = np.zeros(n_coeff); c0_Z[0] = Z_ax
            cR_ext = np.vstack([c0_R, coeffs_R])
            cZ_ext = np.vstack([c0_Z, coeffs_Z])
        else:
            r_ext = r_norms
            cR_ext = coeffs_R.copy(); cR_ext[0, 0] = R_ax
            cZ_ext = coeffs_Z.copy(); cZ_ext[0, 0] = Z_ax

        sp_R = [CubicSpline(r_ext, cR_ext[:, c], extrapolate=True) for c in range(n_coeff)]
        sp_Z = [CubicSpline(r_ext, cZ_ext[:, c], extrapolate=True) for c in range(n_coeff)]

        return cls(
            phi_ref=phi_ref,
            R_ax=R_ax,
            Z_ax=Z_ax,
            r_nodes=r_ext,
            splines_R=sp_R,
            splines_Z=sp_Z,
            n_fourier=n_fourier,
        )

    # ------------------------------------------------------------------
    def eval_RZ(self, r: float, theta: float) -> Tuple[float, float]:
        """Evaluate (R, Z) at (r, θ) using inner Fourier splines."""
        n = self.n_fourier
        cv = np.empty(1 + 2 * n)
        cv[0] = 1.0
        for k in range(1, n + 1):
            cv[2 * k - 1] = np.cos(k * theta)
            cv[2 * k]     = np.sin(k * theta)
        R = sum(self.splines_R[c](r) * cv[c] for c in range(1 + 2 * n))
        Z = sum(self.splines_Z[c](r) * cv[c] for c in range(1 + 2 * n))
        return float(R), float(Z)

    # ------------------------------------------------------------------
    def grad_r_direction(self, r: float, theta: float, dr: float = 0.025
                          ) -> Tuple[float, float]:
        """Unit vector ê_r = ∂(R,Z)/∂r / |∂(R,Z)/∂r| at (r, θ).

        Note: do NOT clamp r_hi to r_nodes[-1]; the spline already
        extrapolates, and clamping causes r_lo > r_hi for r > r_nodes[-1],
        which flips the gradient sign.
        """
        r_lo = max(1e-4, r - dr)
        r_hi = r + dr
        Rlo, Zlo = self.eval_RZ(r_lo, theta)
        Rhi, Zhi = self.eval_RZ(r_hi, theta)
        dR = (Rhi - Rlo) / (r_hi - r_lo)
        dZ = (Zhi - Zlo) / (r_hi - r_lo)
        nm = np.sqrt(dR ** 2 + dZ ** 2) + 1e-30
        return dR / nm, dZ / nm

    # ------------------------------------------------------------------
    @classmethod
    def from_full_grid_with_cycle_constraint(
        cls,
        phi_ref: float,
        R_ax: float,
        Z_ax: float,
        R_surf_row: np.ndarray,       # (n_r, n_theta) from npz, ALL r values 0..1
        Z_surf_row: np.ndarray,
        r_vals: np.ndarray,            # (n_r,) r values, 0..1
        theta_arr: np.ndarray,         # (n_theta,) theta values
        R_cycle: np.ndarray,           # X/O cycle positions R [m]
        Z_cycle: np.ndarray,           # X/O cycle positions Z [m]
        n_fourier: int = 8,
        cycle_weight: float = 200.0,
        r_geom_min: float = 0.05,
    ) -> 'InnerFourierSection':
        """Build from full r-grid (r=0..1) with r=1 anchored to X/O cycle points.

        This is the correct unified representation:
        - Same Fourier order n_fourier for ALL r values (no discontinuity)
        - Inner surfaces (r < 1): fit from Poincare data (exact, RMS~0mm)
        - r=1 surface: weighted least squares with X/O cycle points (weight=cycle_weight)
          and natural extrapolation elsewhere (weight=1)

        Parameters
        ----------
        phi_ref : float
            Toroidal section angle [rad].
        R_ax, Z_ax : float
            Magnetic axis position [m].
        R_surf_row, Z_surf_row : ndarray (n_r, n_theta)
            Full r-grid surface data from npz (includes r=1 row as poly extrap).
        r_vals : ndarray (n_r,)
            Normalised r values, should span 0..1 with uniform spacing.
        theta_arr : ndarray (n_theta,)
            Poloidal angle samples [rad].
        R_cycle, Z_cycle : ndarray
            X/O cycle (R, Z) positions at this phi section [m].
        n_fourier : int
            Fourier harmonics (same for all r).
        cycle_weight : float
            Weight for X/O cycle constraint points in least squares.
        r_geom_min : float
            Filter: exclude cycle points within r_geom_min of axis.
        """
        n_r = len(r_vals)
        n_theta = len(theta_arr)
        n_coeff = 1 + 2 * n_fourier

        A = _fourier_matrix(theta_arr, n_fourier)   # (n_theta, n_coeff)
        A_pinv = np.linalg.pinv(A)                  # (n_coeff, n_theta)

        # ── Fit all r rows with unified Fourier ───────────────────────────
        coeffs_R = np.zeros((n_r, n_coeff))
        coeffs_Z = np.zeros((n_r, n_coeff))
        for ir in range(n_r):
            coeffs_R[ir] = A_pinv @ R_surf_row[ir]
            coeffs_Z[ir] = A_pinv @ Z_surf_row[ir]

        # ── Replace r=1 (last) row with weighted fit to X/O cycle ────────
        R_cycle = np.asarray(R_cycle, dtype=float)
        Z_cycle = np.asarray(Z_cycle, dtype=float)
        r_geom = np.sqrt((R_cycle - R_ax)**2 + (Z_cycle - Z_ax)**2)
        mask = r_geom > r_geom_min
        R_c = R_cycle[mask]; Z_c = Z_cycle[mask]

        if len(R_c) >= 2:
            # Start from natural extrapolation (last row of R_surf_row)
            R_r1 = R_surf_row[-1].copy()
            Z_r1 = Z_surf_row[-1].copy()

            # Replace nearest-theta grid point for each cycle point
            replaced = set()
            theta_c = np.arctan2(Z_c - Z_ax, R_c - R_ax)
            for k in range(len(R_c)):
                ith = int(np.argmin(np.abs(theta_arr - theta_c[k])))
                R_r1[ith] = R_c[k]
                Z_r1[ith] = Z_c[k]
                replaced.add(ith)

            # Weighted least squares: cycle points weight=cycle_weight
            weights = np.ones(n_theta)
            for ith in replaced:
                weights[ith] = cycle_weight
            W = np.diag(np.sqrt(weights))
            A_w = W @ A
            c_R_r1, _, _, _ = np.linalg.lstsq(A_w, W @ R_r1, rcond=None)
            c_Z_r1, _, _, _ = np.linalg.lstsq(A_w, W @ Z_r1, rcond=None)
            coeffs_R[-1] = c_R_r1
            coeffs_Z[-1] = c_Z_r1

        # ── Axis constraint: r=0 → single point ──────────────────────────
        if r_vals[0] > 1e-6:
            r_ext = np.concatenate([[0.0], r_vals])
            c0_R = np.zeros(n_coeff); c0_R[0] = R_ax
            c0_Z = np.zeros(n_coeff); c0_Z[0] = Z_ax
            cR_ext = np.vstack([c0_R, coeffs_R])
            cZ_ext = np.vstack([c0_Z, coeffs_Z])
        else:
            r_ext = r_vals.copy()
            cR_ext = coeffs_R.copy(); cR_ext[0, 0] = R_ax
            cZ_ext = coeffs_Z.copy(); cZ_ext[0, 0] = Z_ax

        sp_R = [CubicSpline(r_ext, cR_ext[:, c], extrapolate=True) for c in range(n_coeff)]
        sp_Z = [CubicSpline(r_ext, cZ_ext[:, c], extrapolate=True) for c in range(n_coeff)]

        return cls(
            phi_ref=phi_ref, R_ax=R_ax, Z_ax=Z_ax,
            r_nodes=r_ext, splines_R=sp_R, splines_Z=sp_Z,
            n_fourier=n_fourier,
        )

    def with_all_cycle_points(
        self,
        R_cycle: np.ndarray,
        Z_cycle: np.ndarray,
        r_island: float = 1.0,
        r_geom_min: float = 0.05,
    ) -> 'InnerFourierSection':
        """Return a new section with r=r_island fitted to ALL X/O cycle points.

        The r=r_island surface is determined by Fourier least-squares fit to all
        cycle points.  The Fourier coefficients at r=r_island are then spliced into
        the spline using LINEAR interpolation between r_inner_last and r_island
        for each Fourier coefficient.  This avoids CubicSpline overshoot artifacts
        in the wide extrapolation gap between r~0.82 and r~1.0.

        Parameters
        ----------
        R_cycle, Z_cycle : ndarray, shape (N,)
            All X- and O-cycle positions [m].
        r_island : float
            Normalised r of the island separatrix (default 1.0).
        r_geom_min : float
            Minimum geometric minor radius filter (excludes near-axis artefacts).
        """
        R_cycle = np.asarray(R_cycle, dtype=float)
        Z_cycle = np.asarray(Z_cycle, dtype=float)

        r_geom = np.sqrt((R_cycle - self.R_ax)**2 + (Z_cycle - self.Z_ax)**2)
        mask = r_geom > r_geom_min
        R_c = R_cycle[mask]; Z_c = Z_cycle[mask]
        n_pts = len(R_c)
        if n_pts < 2:
            return self

        theta_c = np.arctan2(Z_c - self.Z_ax, R_c - self.R_ax)

        # Fourier order: ensure overdetermined (n_pts > n_coeff)
        n_f_fit = min(self.n_fourier, max(1, (n_pts - 1) // 2))
        n_coeff_fit = 1 + 2 * n_f_fit
        n_coeff_full = 1 + 2 * self.n_fourier

        A = _fourier_matrix(theta_c, n_f_fit)  # (n_pts, n_coeff_fit)

        if n_pts >= n_coeff_fit:
            c_R_fit, _, _, _ = np.linalg.lstsq(A, R_c, rcond=None)
            c_Z_fit, _, _, _ = np.linalg.lstsq(A, Z_c, rcond=None)
        else:
            c_R_fit = np.linalg.pinv(A) @ R_c
            c_Z_fit = np.linalg.pinv(A) @ Z_c

        # Full-length coefficient vectors at r=r_island (pad higher modes with
        # the inner-spline extrapolated values to avoid aliasing discontinuity)
        c_R_extrap = np.array([sp(r_island) for sp in self.splines_R])
        c_Z_extrap = np.array([sp(r_island) for sp in self.splines_Z])
        c_R_island = c_R_extrap.copy()
        c_Z_island = c_Z_extrap.copy()
        c_R_island[:n_coeff_fit] = c_R_fit
        c_Z_island[:n_coeff_fit] = c_Z_fit

        # ----------------------------------------------------------------
        # Build new splines using LINEAR interpolation in r for the
        # extrapolation gap between r_nodes[-1] and r_island.
        # CubicSpline with only one distant extra node produces spurious
        # ringing (high d2/dr2). Linear interpolation is monotone and
        # physically consistent for this boundary condition.
        # ----------------------------------------------------------------
        new_r_nodes = np.append(self.r_nodes, r_island)
        n_coeff_full = 1 + 2 * self.n_fourier

        sp_R_new = []
        sp_Z_new = []
        for c in range(n_coeff_full):
            vals_R_old = np.array([self.splines_R[c](r) for r in self.r_nodes])
            vals_Z_old = np.array([self.splines_Z[c](r) for r in self.r_nodes])
            vals_R_new = np.append(vals_R_old, c_R_island[c])
            vals_Z_new = np.append(vals_Z_old, c_Z_island[c])
            # Use linear interpolation only in the extrapolation segment
            # (r_nodes[-1] .. r_island) by building a piecewise CubicSpline
            # with clamped boundary: first derivative from inner spline at r_nodes[-1]
            r_last = self.r_nodes[-1]
            dR_dr_last = float(self.splines_R[c](r_last, 1))
            dZ_dr_last = float(self.splines_Z[c](r_last, 1))
            # Linear slope from last fitted node to island node
            dR_linear = (c_R_island[c] - vals_R_old[-1]) / (r_island - r_last)
            dZ_linear = (c_Z_island[c] - vals_Z_old[-1]) / (r_island - r_last)
            # Use the average of inner derivative and linear slope for smoothness
            bc_left_R  = 0.5 * (dR_dr_last + dR_linear)
            bc_right_R = dR_linear
            bc_left_Z  = 0.5 * (dZ_dr_last + dZ_linear)
            bc_right_Z = dZ_linear
            # Build clamped cubic for the extrapolation segment only
            sp_R_new.append(
                make_interp_spline(
                    new_r_nodes, vals_R_new, k=3,
                    bc_type=([(1, bc_left_R)], [(1, bc_right_R)]),
                )
            )
            sp_Z_new.append(
                make_interp_spline(
                    new_r_nodes, vals_Z_new, k=3,
                    bc_type=([(1, bc_left_Z)], [(1, bc_right_Z)]),
                )
            )

        return InnerFourierSection(
            phi_ref=self.phi_ref, R_ax=self.R_ax, Z_ax=self.Z_ax,
            r_nodes=new_r_nodes, splines_R=sp_R_new, splines_Z=sp_Z_new,
            n_fourier=self.n_fourier,
        )
    def with_island_constraint(
        self,
        R_x: float, Z_x: float,
        R_o: float, Z_o: float,
        r_island: float = 1.0,
    ) -> 'InnerFourierSection':
        """Convenience wrapper: call ``with_all_cycle_points`` with 2 points.

        Prefer ``with_all_cycle_points`` with all available cycle points.
        """
        return self.with_all_cycle_points(
            np.array([R_x, R_o]),
            np.array([Z_x, Z_o]),
            r_island=r_island,
        )
        """Return a new section with r=r_island constrained to pass through X/O cycle positions.

        The corrected r=r_island surface exactly passes through (R_x, Z_x) at theta_X
        and (R_o, Z_o) at theta_O, where theta_X/O are the geometric poloidal angles
        of the X/O cycle positions from the magnetic axis.

        The minimum-norm correction to the Fourier coefficients is computed
        so that the two point constraints are satisfied exactly while
        disturbing all other angles as little as possible.

        Parameters
        ----------
        R_x, Z_x : float
            X-cycle (R, Z) position at this toroidal section [m].
        R_o, Z_o : float
            O-cycle (R, Z) position at this toroidal section [m].
        r_island : float
            Normalised r at which to add the constraint node (default 1.0).

        Returns
        -------
        InnerFourierSection
            New section with an extra spline node at r=r_island anchored
            to the X/O cycle positions.
        """
        n_coeff = 1 + 2 * self.n_fourier

        # 1. Natural extrapolated Fourier coefficients at r=r_island
        c_R_extrap = np.array([sp(r_island) for sp in self.splines_R])
        c_Z_extrap = np.array([sp(r_island) for sp in self.splines_Z])

        # 2. Geometric theta of X/O cycle positions from magnetic axis
        theta_x = float(np.arctan2(Z_x - self.Z_ax, R_x - self.R_ax))
        theta_o = float(np.arctan2(Z_o - self.Z_ax, R_o - self.R_ax))

        # 3. Current surface prediction at theta_X, theta_O
        R_pred_x, Z_pred_x = self.eval_RZ(r_island, theta_x)
        R_pred_o, Z_pred_o = self.eval_RZ(r_island, theta_o)

        # 4. Corrections needed
        dR_x = R_x - R_pred_x
        dZ_x = Z_x - Z_pred_x
        dR_o = R_o - R_pred_o
        dZ_o = Z_o - Z_pred_o

        # 5. Fourier basis vectors at theta_X and theta_O
        def fourier_vec(theta):
            v = np.zeros(n_coeff)
            v[0] = 1.0
            for k in range(1, self.n_fourier + 1):
                v[2 * k - 1] = np.cos(k * theta)
                v[2 * k]     = np.sin(k * theta)
            return v

        f_x = fourier_vec(theta_x)
        f_o = fourier_vec(theta_o)

        # 6. Minimum-norm correction: delta_c = A^T (A A^T)^{-1} b
        #    where A = [f_x; f_o], b = [dR_x, dR_o] (and similarly for Z)
        A = np.vstack([f_x, f_o])   # (2, n_coeff)
        AAt = A @ A.T               # (2, 2)
        try:
            b_R = np.array([dR_x, dR_o])
            b_Z = np.array([dZ_x, dZ_o])
            delta_cR = A.T @ np.linalg.solve(AAt, b_R)
            delta_cZ = A.T @ np.linalg.solve(AAt, b_Z)
        except np.linalg.LinAlgError:
            # X and O at same theta: skip correction
            delta_cR = np.zeros(n_coeff)
            delta_cZ = np.zeros(n_coeff)

        c_R_new = c_R_extrap + delta_cR
        c_Z_new = c_Z_extrap + delta_cZ

        # 7. Rebuild splines with an extra node at r=r_island
        r_new = np.append(self.r_nodes, r_island)
        # Collect existing node values from current splines
        cR_old = np.array([[sp(r) for sp in self.splines_R] for r in self.r_nodes])  # (n_nodes, n_coeff)
        cZ_old = np.array([[sp(r) for sp in self.splines_Z] for r in self.r_nodes])
        cR_all = np.vstack([cR_old, c_R_new])  # (n_nodes+1, n_coeff)
        cZ_all = np.vstack([cZ_old, c_Z_new])

        sp_R_new = [CubicSpline(r_new, cR_all[:, c], extrapolate=True) for c in range(n_coeff)]
        sp_Z_new = [CubicSpline(r_new, cZ_all[:, c], extrapolate=True) for c in range(n_coeff)]

        return InnerFourierSection(
            phi_ref=self.phi_ref,
            R_ax=self.R_ax,
            Z_ax=self.Z_ax,
            r_nodes=r_new,
            splines_R=sp_R_new,
            splines_Z=sp_Z_new,
            n_fourier=self.n_fourier,
        )

    # ------------------------------------------------------------------
    def project(self, R_t: float, Z_t: float,
                r_init: float = 0.5) -> Tuple[float, float]:
        """Project (R_t, Z_t) to (r, θ) via Nelder-Mead."""
        th0 = np.arctan2(Z_t - self.Z_ax, R_t - self.R_ax)

        def obj(x):
            r_, t_ = x
            if r_ < 0 or r_ > 3.0:
                return 1e6
            Rm, Zm = self.eval_RZ(r_, t_)
            return (Rm - R_t) ** 2 + (Zm - Z_t) ** 2

        res = minimize(obj, [r_init, th0], method='Nelder-Mead',
                       options={'xatol': 1e-6, 'fatol': 1e-12, 'maxiter': 5000})
        r_opt, t_opt = res.x
        t_opt = (t_opt + np.pi) % (2 * np.pi) - np.pi
        return float(r_opt), float(t_opt)


# ---------------------------------------------------------------------------
# XOCycAnchor: Xcyc and Ocyc orbit anchor data at one phi section
# ---------------------------------------------------------------------------

@dataclass
class XOCycAnchor:
    """Xcyc and Ocyc positions and their inner-Fourier θ coordinates
    at one toroidal section.

    Attributes
    ----------
    phi : float
        Toroidal angle [rad].
    R_x, Z_x : float
        Xcyc (R, Z) at this section [m].
    R_o, Z_o : float
        Ocyc (R, Z) at this section [m].
    r_island : float
        Normalised r label assigned to the island layer.
    theta_x_inner : float
        Inner-Fourier θ of the Xcyc position (projected via
        ``InnerFourierSection.project``).
    theta_o_inner : float
        Inner-Fourier θ of the Ocyc position.
    """
    phi: float
    R_x: float
    Z_x: float
    R_o: float
    Z_o: float
    r_island: float
    theta_x_inner: float
    theta_o_inner: float


# ---------------------------------------------------------------------------
# IslandHealedCoordMap: the main public class
# ---------------------------------------------------------------------------

class IslandHealedCoordMap:
    """Island-constrained healed flux-surface coordinate map (r, θ, φ).

    The coordinate system has three zones:

    Zone I   (r �?r_inner_fit):
        Pure inner-Fourier surfaces.  θ is geometric arctan2 from axis.

    Zone II  (r_inner_fit < r �?r_island):
        Blend zone.  θ is linearly interpolated between the inner-Fourier
        θ_inner and the island-frame θ_island, controlled by a sigmoid.
        At r = r_island, the Ocyc is at θ = θ_O (default 0) and the
        Xcyc is at θ = θ_X.

    Zone III (r > r_island):
        Continued extrapolation using the island-frame reference.
        Useful for projecting external coil positions.

    The island frame at each (r, φ) is defined by the Ocyc position as
    origin and the Xcyc direction as a reference axis.

    Construction
    ------------
    Use ``IslandHealedCoordMap.build()``.

    Usage
    -----
    ::

        coord = IslandHealedCoordMap.build(
            phi_sections,            # list[float] in rad
            R_ax_arr, Z_ax_arr,      # magnetic axis at each phi_section
            r_norms,                 # (n_r,) inner surface r labels
            R_surf_4d, Z_surf_4d,    # (n_r, n_theta, n_phi_sections)
            theta_arr,               # (n_theta,) poloidal angles
            xcycle_anchors,          # list[XOCycAnchor]
            r_inner_fit=0.82,
            r_island=1.0,
            blend_width=0.12,
            n_fourier=8,
        )

        r_c, theta_c = coord.to_rtheta(R_coil, Z_coil, phi_coil)
        eR, eZ = coord.grad_r_direction(r_c, theta_c, phi_coil)
    """

    def __init__(
        self,
        sections: List[InnerFourierSection],
        anchors: List[XOCycAnchor],
        r_inner_fit: float,
        r_island: float,
        blend_width: float,
        theta_X: float,      # island-frame θ assigned to Xcyc
        theta_O: float,      # island-frame θ assigned to Ocyc (default 0)
    ):
        self._sections = sections
        self._anchors = anchors
        self._phi_sec = np.array([s.phi_ref for s in sections])
        self._phi_anc = np.array([a.phi for a in anchors])
        self.r_inner_fit = r_inner_fit
        self.r_island = r_island
        self.blend_width = blend_width
        self.theta_X = theta_X
        self.theta_O = theta_O

        # Build splines for X/O ring positions vs phi
        # (anchor phi assumed to cover [0, 2π] or [0, m*2π])
        phi_a = np.array([a.phi for a in anchors])

        def _make_spline(vals, unwrap=False):
            arr = np.asarray(vals, dtype=float)
            if unwrap:
                arr = np.unwrap(arr)
            if len(phi_a) == 1:
                # Constant interpolant �?return a callable that ignores phi
                _c = float(arr[0])
                return type('_Const', (), {'__call__': lambda self, x: _c})()
            if len(phi_a) == 2:
                return interp1d(phi_a, arr, kind='linear', fill_value='extrapolate')
            return CubicSpline(phi_a, arr, extrapolate=True)

        self._spline_Rx  = _make_spline([a.R_x for a in anchors])
        self._spline_Zx  = _make_spline([a.Z_x for a in anchors])
        self._spline_Ro  = _make_spline([a.R_o for a in anchors])
        self._spline_Zo  = _make_spline([a.Z_o for a in anchors])
        # Splines for inner-Fourier θ of X/O rings vs phi
        # Use np.unwrap to avoid angle-wrapping discontinuities that cause
        # CubicSpline to diverge near phi = 2*pi.
        self._spline_thx = _make_spline([a.theta_x_inner for a in anchors], unwrap=True)
        self._spline_tho = _make_spline([a.theta_o_inner for a in anchors], unwrap=True)

    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        phi_sections: List[float],
        R_ax_arr: np.ndarray,
        Z_ax_arr: np.ndarray,
        r_norms: np.ndarray,
        R_surf_3d: np.ndarray,   # (n_r, n_theta, n_phi_sections)
        Z_surf_3d: np.ndarray,
        theta_arr: np.ndarray,
        xcycle_anchors: List[XOCycAnchor],
        r_inner_fit: float = 0.82,
        r_island: float = 1.0,
        blend_width: float = 0.12,
        n_fourier: int = 8,
        theta_X: float = np.pi,    # island-frame θ of Xcyc (π = opposite O)
        theta_O: float = 0.0,      # island-frame θ of Ocyc
    ) -> 'IslandHealedCoordMap':
        """Build the coordinate map from Poincaré surface data + X/O ring orbits.

        Parameters
        ----------
        phi_sections : list of float
            Toroidal angles of Poincaré sections [rad].
        R_ax_arr, Z_ax_arr : ndarray, shape (n_phi_sections,)
            Magnetic axis positions.
        r_norms : ndarray, shape (n_r,)
            Normalised r of inner surfaces (should all be �?r_inner_fit).
        R_surf_3d, Z_surf_3d : ndarray, shape (n_r, n_theta, n_phi_sections)
            Surface (R, Z) grids.
        theta_arr : ndarray, shape (n_theta,)
            Poloidal angle samples.
        xcycle_anchors : list of XOCycAnchor
            Xcyc and Ocyc positions vs phi, projected to inner-Fourier
            coordinates.  Use ``make_xcyc_anchors()`` to build these from
            ``IslandChainOrbit`` objects.
        r_inner_fit : float
            Surfaces with r > r_inner_fit are excluded from inner spline fit.
        r_island : float
            Normalised r assigned to the island layer (X/O rings live here).
        blend_width : float
            Radial width of the smooth blend zone from inner to island frame.
        n_fourier : int
            Fourier harmonics.
        theta_X, theta_O : float
            Island-frame θ values assigned to X- and Ocycs.
        """
        # Filter to r �?r_inner_fit
        mask = r_norms <= r_inner_fit
        r_fit = r_norms[mask]

        sections = []
        for ip, phi in enumerate(phi_sections):
            R_s = R_surf_3d[mask, :, ip]   # (n_r_fit, n_theta)
            Z_s = Z_surf_3d[mask, :, ip]
            sec = InnerFourierSection.from_poincare_surfaces(
                phi_ref=float(phi),
                R_ax=float(R_ax_arr[ip]),
                Z_ax=float(Z_ax_arr[ip]),
                r_norms=r_fit,
                R_surf=R_s,
                Z_surf=Z_s,
                theta_arr=theta_arr,
                n_fourier=n_fourier,
            )
            sections.append(sec)

        return cls(
            sections=sections,
            anchors=xcycle_anchors,
            r_inner_fit=r_inner_fit,
            r_island=r_island,
            blend_width=blend_width,
            theta_X=theta_X,
            theta_O=theta_O,
        )

    # ------------------------------------------------------------------
    def _nearest_section(self, phi: float) -> Tuple[InnerFourierSection, int]:
        """Return the nearest inner-Fourier section to phi (Np=2 folded)."""
        phi_fold = phi % np.pi
        ip = int(np.argmin(np.abs(self._phi_sec % np.pi - phi_fold)))
        return self._sections[ip], ip

    # ------------------------------------------------------------------
    def _xoring_theta_at_phi(self, phi: float) -> Tuple[float, float]:
        """Inner-Fourier θ of Xcyc and Ocyc at given phi (interpolated)."""
        theta_x = float(self._spline_thx(phi % (2 * np.pi)))
        theta_o = float(self._spline_tho(phi % (2 * np.pi)))
        return theta_x, theta_o

    # ------------------------------------------------------------------
    def _island_frame_theta(
        self, theta_inner: float, theta_x_inner: float, theta_o_inner: float
    ) -> float:
        """Convert inner-Fourier θ to island-frame θ.

        The island frame is defined by a linear rescaling such that:
          θ_inner = θ_o_inner  �? island θ = self.theta_O   (Ocyc)
          θ_inner = θ_x_inner  �? island θ = self.theta_X   (Xcyc)

        This is a piecewise-linear wrapping on the circle.

        Parameters
        ----------
        theta_inner : float
            Inner-Fourier θ of the point to convert.
        theta_x_inner, theta_o_inner : float
            Inner-Fourier θ of Xcyc and Ocyc at this section.

        Returns
        -------
        float
            Island-frame θ �?(-π, π].
        """
        # Angular separation X→O going counter-clockwise (positive direction)
        dxo = (theta_o_inner - theta_x_inner + np.pi) % (2 * np.pi) - np.pi
        # Angular separation X→point
        dxp = (theta_inner - theta_x_inner + np.pi) % (2 * np.pi) - np.pi

        # Scale: map [θ_x_inner, θ_o_inner] �?[theta_X, theta_O]
        if abs(dxo) < 1e-6:
            return self.theta_X
        # Fraction along arc from X to O
        frac = dxp / dxo
        # Island frame: linear interpolation
        island_theta = self.theta_X + frac * (self.theta_O - self.theta_X)
        island_theta = (island_theta + np.pi) % (2 * np.pi) - np.pi
        return float(island_theta)

    # ------------------------------------------------------------------
    def to_rtheta(
        self, R: float, Z: float, phi: float, r_init: float = 0.9
    ) -> Tuple[float, float]:
        """Project physical (R, Z, φ) to healed coordinates (r, θ).

        The projection uses the nearest inner-Fourier section and blends
        the θ coordinate into the island frame for r near r_island.

        Parameters
        ----------
        R, Z : float
            Physical coordinates [m].
        phi : float
            Toroidal angle [rad].
        r_init : float
            Initial guess for r (should be near the expected location).

        Returns
        -------
        r, theta : float
            Healed coordinate (r in [0, �?, θ �?(-π, π]).
        """
        sec, _ = self._nearest_section(phi)
        r_raw, theta_inner = sec.project(R, Z, r_init=r_init)

        # Blend θ into island frame
        alpha = _sigmoid_blend(r_raw,
                               self.r_inner_fit - self.blend_width,
                               self.r_inner_fit + self.blend_width)
        if alpha < 1e-6:
            # Pure inner zone �?θ is geometric
            return float(r_raw), float(theta_inner)

        # Island-frame θ at this phi
        theta_x_inn, theta_o_inn = self._xoring_theta_at_phi(phi)
        theta_island = self._island_frame_theta(theta_inner, theta_x_inn, theta_o_inn)

        # Blended θ
        theta_blend = (1 - alpha) * theta_inner + alpha * theta_island
        theta_blend = (theta_blend + np.pi) % (2 * np.pi) - np.pi
        return float(r_raw), float(theta_blend)

    # ------------------------------------------------------------------
    def eval_RZ(
        self, r: float, theta: float, phi: float
    ) -> Tuple[float, float]:
        """Forward map (r, θ, φ) �?(R, Z).

        For r �?r_inner_fit: uses inner Fourier surfaces directly.
        For r > r_inner_fit: uses inner Fourier extrapolation (the
        island frame adjusts θ but not the spatial map �?use with caution
        far from the fit region).

        Parameters
        ----------
        r, theta : float
            Healed coordinates.
        phi : float
            Toroidal angle [rad].

        Returns
        -------
        R, Z : float
        """
        sec, _ = self._nearest_section(phi)
        # Invert island-frame blend to get inner-Fourier θ
        alpha = _sigmoid_blend(r,
                               self.r_inner_fit - self.blend_width,
                               self.r_inner_fit + self.blend_width)
        if alpha < 1e-6:
            theta_inner = theta
        else:
            theta_x_inn, theta_o_inn = self._xoring_theta_at_phi(phi)
            # Inverse of island_frame_theta: inner �?island was linear
            dxo = (theta_o_inn - theta_x_inn + np.pi) % (2 * np.pi) - np.pi
            dxo_out = self.theta_O - self.theta_X
            if abs(dxo_out) < 1e-6 or abs(dxo) < 1e-6:
                theta_inner = theta_x_inn
            else:
                frac = (theta - self.theta_X) / dxo_out
                theta_island_inv = theta_x_inn + frac * dxo
                # Blend back
                theta_inner = (1 - alpha) * theta + alpha * theta_island_inv
                theta_inner = (theta_inner + np.pi) % (2 * np.pi) - np.pi

        return sec.eval_RZ(r, theta_inner)

    # ------------------------------------------------------------------
    def grad_r_direction(
        self, r: float, theta: float, phi: float, dr: float = 0.025
    ) -> Tuple[float, float]:
        """Unit vector ê_r = �?R,Z)/∂r / |�?R,Z)/∂r| at (r, θ, φ).

        Uses finite differences in r with the forward map eval_RZ.

        Parameters
        ----------
        r, theta : float
            Healed coordinates.
        phi : float
            Toroidal angle [rad].
        dr : float
            Finite-difference step.

        Returns
        -------
        eR, eZ : float
            Components of the unit radial vector.
        """
        # Do NOT clamp r_hi to r_nodes[-1]: the spline extrapolates, and
        # clamping causes r_lo > r_hi when r > r_nodes[-1] -> wrong sign.
        r_lo = max(1e-3, r - dr)
        r_hi = r + dr
        Rlo, Zlo = self.eval_RZ(r_lo, theta, phi)
        Rhi, Zhi = self.eval_RZ(r_hi, theta, phi)
        dR = (Rhi - Rlo) / (r_hi - r_lo)
        dZ = (Zhi - Zlo) / (r_hi - r_lo)
        nm = np.sqrt(dR ** 2 + dZ ** 2) + 1e-30
        return dR / nm, dZ / nm


# ---------------------------------------------------------------------------
# Adapter: convert IslandChainOrbit objects to fp_by_section dict
# ---------------------------------------------------------------------------

def fp_by_section_from_orbits(
    xring_orbit,
    oring_orbit,
    phi_sections: Optional[List[float]] = None,
    phi_tol: float = 0.05,
) -> dict:
    """Convert X/O IslandChainOrbit objects to fp_by_section format.

    Returns
    -------
    dict
        Mapping phi -> {'xpts': [...], 'opts': [...]} where entries are
        ChainFixedPoint objects (already exposing .R / .Z).
    """
    xfps = list(getattr(xring_orbit, 'fixed_points', []))
    ofps = list(getattr(oring_orbit, 'fixed_points', []))

    if phi_sections is None:
        phis = sorted({float(fp.phi) % (2 * np.pi) for fp in xfps + ofps})
    else:
        phis = [float(phi) % (2 * np.pi) for phi in phi_sections]

    out = {}
    for phi in phis:
        xs = []
        os = []
        for fp in xfps:
            dphi = np.angle(np.exp(1j * ((float(fp.phi) % (2 * np.pi)) - phi)))
            if abs(dphi) <= phi_tol:
                xs.append(fp)
        for fp in ofps:
            dphi = np.angle(np.exp(1j * ((float(fp.phi) % (2 * np.pi)) - phi)))
            if abs(dphi) <= phi_tol:
                os.append(fp)
        out[phi] = {'xpts': xs, 'opts': os}
    return out


# ---------------------------------------------------------------------------
# Factory: build XOCycAnchor list from IslandChainOrbit objects
# ---------------------------------------------------------------------------

def make_xcyc_anchors(
    xring_orbit,   # IslandChainOrbit with kind='X' fixed points
    oring_orbit,   # IslandChainOrbit with kind='O' fixed points
    inner_sections: List[InnerFourierSection],
    r_island: float = 1.0,
    r_init_project: float = 0.95,
) -> List[XOCycAnchor]:
    """Build XOCycAnchor list by projecting X/O ring positions into inner
    Fourier sections.

    Each anchor corresponds to one (phi, Xcyc point, Ocyc point) triple
    taken from the chain orbit objects.

    Parameters
    ----------
    xring_orbit : IslandChainOrbit
        Chain orbit whose fixed points all have kind='X'.  Must cover the
        same phi range as ``oring_orbit``.
    oring_orbit : IslandChainOrbit
        Chain orbit whose fixed points all have kind='O'.
    inner_sections : list of InnerFourierSection
        Inner Fourier section objects (for projecting X/O ring positions).
    r_island : float
        Normalised r to assign to the island layer.
    r_init_project : float
        Initial r guess for the projection optimizer.

    Returns
    -------
    list of XOCycAnchor
        One anchor per (Xcyc fp, Ocyc fp) pair, matched by phi proximity.
    """
    phi_sec = np.array([s.phi_ref for s in inner_sections])

    anchors = []
    xfps = xring_orbit.fixed_points
    ofps = oring_orbit.fixed_points

    # Match Ocyc to nearest Xcyc by phi
    phi_xarr = np.array([fp.phi for fp in xfps])

    for ofp in ofps:
        # Find nearest Xcyc by phi
        iphi = int(np.argmin(np.abs(phi_xarr - ofp.phi)))
        xfp = xfps[iphi]

        phi_anchor = (xfp.phi + ofp.phi) / 2.0  # use midpoint phi

        # Find nearest inner section
        phi_fold = phi_anchor % np.pi
        ip = int(np.argmin(np.abs(phi_sec % np.pi - phi_fold)))
        sec = inner_sections[ip]

        # Project Xcyc into inner Fourier section
        _, theta_x_inn = sec.project(xfp.R, xfp.Z, r_init=r_init_project)
        # Project Ocyc into inner Fourier section
        _, theta_o_inn = sec.project(ofp.R, ofp.Z, r_init=r_init_project)

        anchors.append(XOCycAnchor(
            phi=float(phi_anchor),
            R_x=float(xfp.R), Z_x=float(xfp.Z),
            R_o=float(ofp.R), Z_o=float(ofp.Z),
            r_island=r_island,
            theta_x_inner=float(theta_x_inn),
            theta_o_inner=float(theta_o_inn),
        ))

    return sorted(anchors, key=lambda a: a.phi)


# ---------------------------------------------------------------------------
# Convenience: build from saved npz + IslandChainOrbit
# ---------------------------------------------------------------------------

def build_from_trajectory_npz(
    coords_npz: str,
    trajectory_npz: str | None = None,
    xring_orbit = None,
    oring_orbit = None,
    r_inner_fit: float = 0.82,
    r_island: float = 1.0,
    blend_width: float = 0.12,
    n_fourier: int = 8,
    theta_X: float = np.pi,
    theta_O: float = 0.0,
    n_anchors: int = 40,
    fp_by_section: dict | None = None,
    r_geom_min: float = 0.05,
    cycle_weight: float = 200.0,
) -> 'IslandHealedCoordMap':
    """Build IslandHealedCoordMap from flux-surface npz + X/O cycle trajectory npz.

    Parameters
    ----------
    fp_by_section : dict | None
        If provided, a dict mapping phi (float, rad) -> {'xpts': [...], 'opts': [...]}
        where each point is indexable as pt[0]=R, pt[1]=Z.  When given, ALL
        X/O cycle intersections at each Poincaré section are used as r=r_island
        constraints via an overdetermined Fourier least-squares fit, giving
        a more accurate representation of the full island separatrix.
        If None (default), only 2 points (Xcyc + Ocyc from trajectory) are used.
    r_geom_min : float
        Minimum geometric minor radius; cycle points closer to the axis are
        excluded as near-axis numerical artefacts.
    """
    # ── Load inner surface coords ─────────────────────────────────────────
    d = np.load(coords_npz)
    R_surf_3d = d['R_surf']        # (n_r, n_theta, n_phi)
    Z_surf_3d = d['Z_surf']
    r_vals    = d['r_vals']
    theta_arr = d['theta_vals']
    phi_vals  = d['phi_vals']
    R_AX      = d['R_AX']
    Z_AX      = d['Z_AX']

    n_phi = len(phi_vals)

    # If orbit objects are provided, convert them to per-section fixed-point dict.
    if fp_by_section is None and xring_orbit is not None and oring_orbit is not None:
        fp_by_section = fp_by_section_from_orbits(
            xring_orbit, oring_orbit, phi_sections=list(phi_vals), phi_tol=0.08
        )

    # ── Build sections using unified Fourier representation ───────────────
    # When fp_by_section is provided: use all X/O cycle points per section
    # to anchor r=1. Otherwise: 2-point fallback from trajectory.
    sections = []
    for ip in range(n_phi):
        # Collect X/O cycle points for this section
        R_cyc_ip = np.array([], dtype=float)
        Z_cyc_ip = np.array([], dtype=float)
        if fp_by_section is not None:
            phi_v = float(phi_vals[ip])
            best_phi = min(fp_by_section.keys(),
                           key=lambda k: abs(k % np.pi - phi_v % np.pi))
            sec_data = fp_by_section[best_phi]
            R_list, Z_list = [], []
            for xpt in sec_data.get('xpts', []):
                R_p = xpt[0] if not hasattr(xpt, 'R') else xpt.R
                Z_p = xpt[1] if not hasattr(xpt, 'R') else xpt.Z
                R_list.append(R_p); Z_list.append(Z_p)
            for opt in sec_data.get('opts', []):
                R_p = opt[0] if not hasattr(opt, 'R') else opt.R
                Z_p = opt[1] if not hasattr(opt, 'R') else opt.Z
                R_list.append(R_p); Z_list.append(Z_p)
            R_cyc_ip = np.array(R_list)
            Z_cyc_ip = np.array(Z_list)
        else:
            # 2-point fallback from trajectory
            phi_v = float(phi_vals[ip]) % (2 * np.pi)
            R_cyc_ip = np.array([float(_Rx_itp(phi_v)), float(_Ro_itp(phi_v))])
            Z_cyc_ip = np.array([float(_Zx_itp(phi_v)), float(_Zo_itp(phi_v))])

        sec = InnerFourierSection.from_full_grid_with_cycle_constraint(
            phi_ref=float(phi_vals[ip]),
            R_ax=float(R_AX[ip]),
            Z_ax=float(Z_AX[ip]),
            R_surf_row=R_surf_3d[:, :, ip],   # (n_r, n_theta)
            Z_surf_row=Z_surf_3d[:, :, ip],
            r_vals=r_vals,
            theta_arr=theta_arr,
            R_cycle=R_cyc_ip,
            Z_cycle=Z_cyc_ip,
            n_fourier=n_fourier,
            cycle_weight=cycle_weight,
            r_geom_min=r_geom_min,
        )
        sections.append(sec)

    # ── Load / derive continuous X/O ring trajectories ────────────────────
    from scipy.interpolate import interp1d as _interp1d

    if xring_orbit is not None and oring_orbit is not None:
        xfps = sorted(list(getattr(xring_orbit, 'fixed_points', [])), key=lambda fp: float(fp.phi))
        ofps = sorted(list(getattr(oring_orbit, 'fixed_points', [])), key=lambda fp: float(fp.phi))
        phi_x = np.array([float(fp.phi) for fp in xfps], dtype=float)
        R_x_traj = np.array([float(fp.R) for fp in xfps], dtype=float)
        Z_x_traj = np.array([float(fp.Z) for fp in xfps], dtype=float)
        phi_o = np.array([float(fp.phi) for fp in ofps], dtype=float)
        R_o_traj = np.array([float(fp.R) for fp in ofps], dtype=float)
        Z_o_traj = np.array([float(fp.Z) for fp in ofps], dtype=float)

        if len(phi_x) < 2 or len(phi_o) < 2:
            raise ValueError('Orbit path requires >=2 fixed points in both xring_orbit and oring_orbit.')

        _Rx_itp = _interp1d(phi_x, R_x_traj, kind='linear', fill_value='extrapolate')
        _Zx_itp = _interp1d(phi_x, Z_x_traj, kind='linear', fill_value='extrapolate')
        _Ro_itp = _interp1d(phi_o, R_o_traj, kind='linear', fill_value='extrapolate')
        _Zo_itp = _interp1d(phi_o, Z_o_traj, kind='linear', fill_value='extrapolate')
    else:
        if trajectory_npz is None:
            raise ValueError('Need either trajectory_npz or both xring_orbit and oring_orbit.')
        t = np.load(trajectory_npz)
        phi_traj = t['phi_arr']
        R_x_traj = t['R_x_arr']
        Z_x_traj = t['Z_x_arr']
        R_o_traj = t['R_o_arr']
        Z_o_traj = t['Z_o_arr']
        _Rx_itp = _interp1d(phi_traj, R_x_traj, kind='cubic', fill_value='extrapolate')
        _Zx_itp = _interp1d(phi_traj, Z_x_traj, kind='cubic', fill_value='extrapolate')
        _Ro_itp = _interp1d(phi_traj, R_o_traj, kind='cubic', fill_value='extrapolate')
        _Zo_itp = _interp1d(phi_traj, Z_o_traj, kind='cubic', fill_value='extrapolate')

    # ── Apply X/O cycle constraint to each section ────────────────────────
    # r=1 already applied in from_full_grid_with_cycle_constraint

    # ── Build anchors at n_anchors evenly spaced phi values ───────────────
    phi_sec_arr = np.array([s.phi_ref for s in sections])
    phi_anchor_arr = np.linspace(0, 2 * np.pi, n_anchors, endpoint=False)
    anchors = []

    for phi_a in phi_anchor_arr:
        R_x = float(_Rx_itp(phi_a))
        Z_x = float(_Zx_itp(phi_a))
        R_o = float(_Ro_itp(phi_a))
        Z_o = float(_Zo_itp(phi_a))

        # Nearest inner section (Np=2 fold)
        phi_fold = phi_a % np.pi
        ip = int(np.argmin(np.abs(phi_sec_arr % np.pi - phi_fold)))
        sec = sections[ip]

        # Use geometric theta (arctan2 from axis) as inner-Fourier theta.
        # This avoids Nelder-Mead local-minimum failures that corrupt the spline.
        # The inner Fourier sections use geometric theta as their poloidal
        # coordinate, so arctan2 from the axis IS the correct inner theta.
        theta_x_inn = float(np.arctan2(Z_x - sec.Z_ax, R_x - sec.R_ax))
        theta_o_inn = float(np.arctan2(Z_o - sec.Z_ax, R_o - sec.R_ax))

        anchors.append(XOCycAnchor(
            phi=float(phi_a),
            R_x=R_x, Z_x=Z_x,
            R_o=R_o, Z_o=Z_o,
            r_island=r_island,
            theta_x_inner=float(theta_x_inn),
            theta_o_inner=float(theta_o_inn),
        ))

    return IslandHealedCoordMap(
        sections=sections,
        anchors=anchors,
        r_inner_fit=r_inner_fit,
        r_island=r_island,
        blend_width=blend_width,
        theta_X=theta_X,
        theta_O=theta_O,
    )


def build_from_orbits(
    coords_npz: str,
    xring_orbit,
    oring_orbit,
    r_inner_fit: float = 0.82,
    r_island: float = 1.0,
    blend_width: float = 0.12,
    n_fourier: int = 8,
    theta_X: float = np.pi,
    theta_O: float = 0.0,
    n_anchors: int = 40,
    r_geom_min: float = 0.05,
    cycle_weight: float = 200.0,
) -> 'IslandHealedCoordMap':
    """High-level convenience wrapper: build healed map directly from orbit objects."""
    return build_from_trajectory_npz(
        coords_npz=coords_npz,
        trajectory_npz=None,
        xring_orbit=xring_orbit,
        oring_orbit=oring_orbit,
        r_inner_fit=r_inner_fit,
        r_island=r_island,
        blend_width=blend_width,
        n_fourier=n_fourier,
        theta_X=theta_X,
        theta_O=theta_O,
        n_anchors=n_anchors,
        fp_by_section=None,
        r_geom_min=r_geom_min,
        cycle_weight=cycle_weight,
    )


# ---------------------------------------------------------------------------
# Helper: extract orbit from semantic chain objects
# ---------------------------------------------------------------------------

def _extract_orbit_from_chain(chain, label: str = 'chain'):
    """Extract an ``IslandChainOrbit`` from a semantic chain object.

    Accepts either an ``IslandChainOrbit`` directly, or any object that
    has an ``.orbit`` attribute (e.g. ``IslandChain``).

    Parameters
    ----------
    chain : IslandChainOrbit or object with .orbit
        The orbit or chain to unwrap.
    label : str
        Human-readable label used in error messages.

    Returns
    -------
    IslandChainOrbit

    Raises
    ------
    TypeError
        If ``chain`` is None or neither an orbit nor a chain-with-orbit.
    AttributeError
        If the chain object has no ``.orbit`` attribute.
    ValueError
        If ``chain.orbit`` is None (no orbit has been attached yet).
    """
    if chain is None:
        raise TypeError(
            f"{label} must not be None. "
            "Provide an IslandChainOrbit or an IslandChain with a .orbit attached."
        )

    # Already an IslandChainOrbit: check for fixed_points duck-type
    if hasattr(chain, 'fixed_points'):
        return chain

    # Higher-level chain object with .orbit attribute
    if not hasattr(chain, 'orbit'):
        raise AttributeError(
            f"{label} has no 'orbit' attribute. "
            "Expected an IslandChainOrbit or an IslandChain with .orbit set. "
            f"Got: {type(chain).__name__}"
        )

    orbit = chain.orbit
    if orbit is None:
        raise ValueError(
            f"{label}.orbit is None — no IslandChainOrbit has been attached. "
            "Call chain.compute_orbit(...) or set chain.orbit before using "
            "build_from_island_chain()."
        )

    return orbit


def _split_orbit_by_kind(orbit):
    """Split a mixed IslandChainOrbit into (xring_orbit, oring_orbit) by kind.

    Each returned object is a lightweight namespace that mimics the
    ``IslandChainOrbit`` interface expected by ``build_from_orbits`` / 
    ``build_from_trajectory_npz`` (specifically: ``.fixed_points`` list).

    Parameters
    ----------
    orbit : IslandChainOrbit
        Orbit whose ``fixed_points`` contain both 'X' and 'O' type points.

    Returns
    -------
    xring_orbit, oring_orbit : SimpleNamespace
        Each has a ``.fixed_points`` attribute filtered to the respective kind.

    Raises
    ------
    ValueError
        If the orbit has no X-type or no O-type fixed points.
    """
    import types
    x_fps = [fp for fp in orbit.fixed_points if getattr(fp, 'kind', None) == 'X']
    o_fps = [fp for fp in orbit.fixed_points if getattr(fp, 'kind', None) == 'O']

    if not x_fps:
        raise ValueError(
            "The supplied orbit has no X-type fixed points (kind='X'). "
            "Cannot build healed coordinates without X-cycle information."
        )
    if not o_fps:
        raise ValueError(
            "The supplied orbit has no O-type fixed points (kind='O'). "
            "Cannot build healed coordinates without O-cycle information."
        )

    xring = types.SimpleNamespace(fixed_points=x_fps)
    oring = types.SimpleNamespace(fixed_points=o_fps)
    return xring, oring


# ---------------------------------------------------------------------------
# High-level entry point: build from semantic IslandChain objects
# ---------------------------------------------------------------------------

def build_from_island_chain(
    coords_npz: str,
    secondary_chain,
    primary_chain=None,
    r_inner_fit: float = 0.82,
    r_island: float = 1.0,
    blend_width: float = 0.12,
    n_fourier: int = 8,
    theta_X: float = np.pi,
    theta_O: float = 0.0,
    n_anchors: int = 40,
    r_geom_min: float = 0.05,
    cycle_weight: float = 200.0,
) -> 'IslandHealedCoordMap':
    """Build an ``IslandHealedCoordMap`` from semantic island chain objects.

    This is the recommended high-level entry point when working with
    ``IslandChain`` or ``IslandChainOrbit`` objects from ``pyna.topo.island``
    and ``pyna.topo.island_chain``.

    Parameters
    ----------
    coords_npz : str
        Path to the flux-surface coordinate npz file produced by the surface
        tracing pipeline (keys: R_surf, Z_surf, r_vals, theta_vals, phi_vals,
        R_AX, Z_AX).
    secondary_chain : IslandChain or IslandChainOrbit
        The secondary (island) chain that defines the X/O cycle geometry.
        Must have a ``.orbit`` attribute (``IslandChainOrbit``) attached, or
        be an ``IslandChainOrbit`` directly.  The orbit's ``fixed_points``
        must include both X-type (``kind='X'``) and O-type (``kind='O'``)
        points covering the full toroidal extent.

        Typical usage::

            from pyna.topo.toroidal import IslandChain
            chain = IslandChain(m=2, n=1, islands=[...])
            chain.orbit = my_orbit   # IslandChainOrbit with X and O fps
            coord_map = build_from_island_chain(coords_npz, secondary_chain=chain)

    primary_chain : IslandChain or IslandChainOrbit or None, optional
        The primary (background) chain.  Currently accepted but not used in
        the coordinate construction — reserved for future integration where
        the primary chain constrains the inner Fourier frame or sets the
        boundary condition at r=0.  Pass it now so call sites remain valid
        when this feature is implemented.
    r_inner_fit : float
        Maximum normalised r of the inner flux surfaces used for the Fourier
        spline fit (default 0.82).  Surfaces with r > r_inner_fit are excluded
        from the inner fit.
    r_island : float
        Normalised r assigned to the island separatrix layer (default 1.0).
        X/O cycle positions are pinned to this r value.
    blend_width : float
        Radial width of the smooth sigmoid blend zone from the inner Fourier
        frame to the island frame (default 0.12).
    n_fourier : int
        Number of Fourier harmonics used for the poloidal surface description
        (default 8).
    theta_X : float
        Island-frame poloidal angle assigned to the X-cycle (default π).
    theta_O : float
        Island-frame poloidal angle assigned to the O-cycle (default 0).
    n_anchors : int
        Number of evenly-spaced toroidal angles used to build the anchor
        interpolation table for the X/O ring positions (default 40).
    r_geom_min : float
        Minimum geometric minor radius; cycle points closer to the magnetic
        axis than this value are excluded as near-axis artefacts (default 0.05 m).
    cycle_weight : float
        Relative weight of cycle-point constraints in the Fourier least-squares
        fit at r=r_island (default 200).

    Returns
    -------
    IslandHealedCoordMap

    Raises
    ------
    TypeError
        If ``secondary_chain`` is None.
    AttributeError
        If ``secondary_chain`` has no ``.orbit`` attribute.
    ValueError
        If ``secondary_chain.orbit`` is None, or if the orbit contains no
        X-type or O-type fixed points.

    Notes
    -----
    Internally this function:

    1. Extracts an ``IslandChainOrbit`` from ``secondary_chain`` (via
       ``secondary_chain.orbit`` if needed).
    2. Splits the orbit's ``fixed_points`` into X-only and O-only subsets
       by the ``ChainFixedPoint.kind`` attribute.
    3. Delegates to :func:`build_from_orbits`.

    The ``primary_chain`` parameter is currently a no-op placeholder.
    Future versions will use it to enforce consistency between the primary
    reference surface and the inner Fourier frame.

    See Also
    --------
    build_from_orbits : Lower-level builder accepting separate X/O orbit objects.
    build_from_trajectory_npz : Lowest-level builder accepting raw npz arrays.
    """
    # ── Validate and extract secondary orbit ──────────────────────────────
    secondary_orbit = _extract_orbit_from_chain(secondary_chain, label='secondary_chain')
    xring_orbit, oring_orbit = _split_orbit_by_kind(secondary_orbit)

    # ── primary_chain: accepted, documented, not yet used ─────────────────
    # When primary_chain integration is implemented, extract its orbit here:
    #   primary_orbit = _extract_orbit_from_chain(primary_chain, label='primary_chain')
    # For now, we silently accept any value (including None).

    return build_from_orbits(
        coords_npz=coords_npz,
        xring_orbit=xring_orbit,
        oring_orbit=oring_orbit,
        r_inner_fit=r_inner_fit,
        r_island=r_island,
        blend_width=blend_width,
        n_fourier=n_fourier,
        theta_X=theta_X,
        theta_O=theta_O,
        n_anchors=n_anchors,
        r_geom_min=r_geom_min,
        cycle_weight=cycle_weight,
    )
