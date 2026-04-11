"""
pyna.topo.topology_analysis
============================
Facade for exploratory magnetic-field topology analysis.

Given a 3-D magnetic field in cylindrical coordinates (R, Z, φ) expressed as
a callable ``field_func(rzphi) -> [dR/ds, dZ/ds, dφ/ds]``, this module
provides a single high-level entry point:

    result = analyse_topology(field_func, ...)

that returns a :class:`TopologyReport` with:

1. **Chaos / regularity boundary** – FTLE-based partition of the Poincaré
   section into chaotic and regular zones.
2. **Rotational-transform (q) profile** – for each identified regular zone,
   a radial q-profile estimated from field-line winding numbers.
3. **Island-around-island hierarchy** – rational surfaces and their island
   chains nested to a configurable depth.
4. **Fixed-point catalogue** – O- and X-points on one or more Poincaré
   sections, together with the monodromy matrix eigenvalues/vectors of the
   m-th return map (DPᵐ).

Typical usage
-------------
>>> from pyna.topo.topology_analysis import analyse_topology
>>> report = analyse_topology(
...     field_func,
...     R_range=(1.2, 2.4),
...     Z_range=(-0.8, 0.8),
...     phi_sections=[0.0],
...     q_range=(1.0, 4.0),
...     n_ftle_pts=30,
...     ftle_turns=20,
...     island_m_max=6,
...     island_n_max=2,
...     island_depth=2,
... )
>>> report.summary()
>>> report.plot()
"""

from __future__ import annotations

import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from scipy.interpolate import UnivariateSpline

from pyna.topo.poincare import ToroidalSection, poincare_from_fieldlines
from pyna.topo.chaos import ftle_field, chaotic_boundary_estimate, chirikov_overlap
from pyna.topo.toroidal_cycle import find_cycle, ToroidalPeriodicOrbitTrace
from pyna.topo.toroidal_island import locate_all_rational_surfaces, island_halfwidth
from pyna.topo.variational import PoincareMapVariationalEquations
from pyna.topo._rk4 import rk4_integrate


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FixedPointInfo:
    """O- or X-point on a Poincaré section with monodromy data.

    Attributes
    ----------
    rzphi0 : ndarray, shape (3,)
        Location [R, Z, φ] of the fixed point (first pierce of the orbit).
    orbit_type : str
        ``'O'`` for elliptic (stable island centre) or ``'X'`` for hyperbolic.
    m, n : int
        Mode numbers defining the resonance q = m/n.
    period : int
        Orbit period in toroidal turns (= n for q = m/n convention).
    eigenvalues : ndarray, shape (2,)
        Eigenvalues of the monodromy matrix DPᵐ (m = period).
    eigenvectors : ndarray, shape (2, 2)
        Corresponding eigenvectors (columns).
    monodromy : ndarray, shape (2, 2)
        Full 2×2 monodromy matrix.
    stability_index : float
        k = (λ₁ + λ₂) / 2.  |k| < 1 → elliptic, |k| > 1 → hyperbolic.
    """
    rzphi0: np.ndarray
    orbit_type: str            # 'O' or 'X'
    m: int
    n: int
    period: int
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    monodromy: np.ndarray
    stability_index: float


@dataclass
class IslandChain:
    """One island chain at a rational surface.

    Attributes
    ----------
    m, n : int
        Resonance q = m/n.
    S_res : float
        Radial label of the resonant surface (S = √ψ_norm or winding-number
        based, depending on the equilibrium source).
    q_res : float
        Safety factor at the resonant surface.
    half_width : float
        Estimated island half-width in S (from Chirikov formula if available,
        else from Poincaré extraction).
    chirikov_sigma : Optional[float]
        Overlap parameter with the neighbouring chain (None if not computed).
    fixed_points : List[FixedPointInfo]
        O- and X-points found for this chain.
    children : List['IslandChain']
        Sub-islands nested inside this chain (island-around-island hierarchy).
    """
    m: int
    n: int
    S_res: float
    q_res: float
    half_width: float
    chirikov_sigma: Optional[float] = None
    fixed_points: List[FixedPointInfo] = field(default_factory=list)
    children: List["IslandChain"] = field(default_factory=list)


@dataclass
class RegularZone:
    """A connected regular (KAM) region in the Poincaré section.

    Attributes
    ----------
    label : str
        Human-readable label (e.g. ``'core'``, ``'secondary_q41'``).
    S_range : tuple (S_min, S_max)
        Radial extent in normalised flux-surface label.
    q_profile_S : ndarray
        Radial grid of S values for the q profile.
    q_profile : ndarray
        Safety-factor values on q_profile_S.
    island_chains : List[IslandChain]
        All island chains found inside this zone.
    """
    label: str
    S_range: Tuple[float, float]
    q_profile_S: np.ndarray
    q_profile: np.ndarray
    island_chains: List[IslandChain] = field(default_factory=list)


@dataclass
class TopologyReport:
    """Full topology analysis result.

    Attributes
    ----------
    field_func : callable
        The original field function.
    phi_sections : list of float
        Poincaré section angles used.
    ftle : ndarray or None
        FTLE field on the analysis grid (None if not computed).
    R_grid, Z_grid : ndarray
        Spatial grids for FTLE.
    chaotic_boundary_R, chaotic_boundary_Z : ndarray or None
        Approximate boundary of the chaotic zone.
    regular_zones : List[RegularZone]
        Identified regular zones (innermost first).
    all_fixed_points : List[FixedPointInfo]
        Flat list of every fixed point found across all sections.
    warnings : List[str]
        Non-fatal warnings accumulated during analysis.
    """
    field_func: Callable
    phi_sections: List[float]
    ftle: Optional[np.ndarray]
    R_grid: Optional[np.ndarray]
    Z_grid: Optional[np.ndarray]
    chaotic_boundary_R: Optional[np.ndarray]
    chaotic_boundary_Z: Optional[np.ndarray]
    regular_zones: List[RegularZone]
    all_fixed_points: List[FixedPointInfo]
    warnings: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience output
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable text summary."""
        lines = ["=" * 60, "Topology Analysis Report", "=" * 60]

        lines.append(f"\nPoincaré sections: φ = {self.phi_sections}")

        if self.chaotic_boundary_R is not None:
            lines.append(f"\nChaotic boundary: {len(self.chaotic_boundary_R)} points")
        else:
            lines.append("\nChaotic boundary: not computed")

        lines.append(f"\nRegular zones found: {len(self.regular_zones)}")
        for zone in self.regular_zones:
            lines.append(
                f"  [{zone.label}]  S ∈ [{zone.S_range[0]:.3f}, {zone.S_range[1]:.3f}]"
                f"  q ∈ [{zone.q_profile.min():.3f}, {zone.q_profile.max():.3f}]"
                f"  island chains: {len(zone.island_chains)}"
            )
            for chain in zone.island_chains:
                sigma_str = (f"  σ_Chirikov={chain.chirikov_sigma:.2f}"
                             if chain.chirikov_sigma is not None else "")
                fp_O = sum(1 for fp in chain.fixed_points if fp.orbit_type == 'O')
                fp_X = sum(1 for fp in chain.fixed_points if fp.orbit_type == 'X')
                lines.append(
                    f"    q={chain.m}/{chain.n}  S={chain.S_res:.3f}"
                    f"  w={chain.half_width:.4f}"
                    f"  O-pts={fp_O}  X-pts={fp_X}"
                    + sigma_str
                )
                for child in chain.children:
                    lines.append(
                        f"      ↳ sub-island q={child.m}/{child.n}"
                        f"  S={child.S_res:.3f}  w={child.half_width:.4f}"
                    )

        lines.append(f"\nTotal fixed points catalogued: {len(self.all_fixed_points)}")
        for fp in self.all_fixed_points:
            lam_str = ", ".join(f"{v:.4g}" for v in fp.eigenvalues)
            lines.append(
                f"  {fp.orbit_type}-pt q={fp.m}/{fp.n}"
                f"  R={fp.rzphi0[0]:.4f}  Z={fp.rzphi0[1]:.4f}"
                f"  λ=[{lam_str}]  k={fp.stability_index:.4g}"
            )

        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")

        s = "\n".join(lines)
        print(s)
        return s

    def plot(
        self,
        ax=None,
        phi_idx: int = 0,
        show_ftle: bool = True,
        show_fixedpoints: bool = True,
        show_manifolds: bool = False,
        n_turns_manifold: int = 6,
        figsize: Tuple = (7, 8),
    ):
        """Plot the Poincaré section with topology overlays.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            If None, a new figure is created.
        phi_idx : int
            Which section in ``phi_sections`` to visualise.
        show_ftle : bool
            Overlay the FTLE field as a background heatmap.
        show_fixedpoints : bool
            Mark O-points (circles) and X-points (crosses).
        show_manifolds : bool
            Grow and plot stable/unstable manifolds for each X-point.
        n_turns_manifold : int
            Number of map iterations for manifold growth.
        figsize : tuple
            Figure size if ax is None.

        Returns
        -------
        fig, ax
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        phi0 = self.phi_sections[phi_idx]

        # --- FTLE background ---
        if show_ftle and self.ftle is not None:
            ftle_clipped = np.clip(self.ftle, 1e-6, None)
            ax.pcolormesh(
                self.R_grid, self.Z_grid, ftle_clipped,
                norm=LogNorm(vmin=ftle_clipped.min(), vmax=ftle_clipped.max()),
                cmap="hot_r", alpha=0.55, rasterized=True, zorder=1,
            )

        # --- Chaotic boundary ---
        if self.chaotic_boundary_R is not None:
            ax.plot(
                self.chaotic_boundary_R, self.chaotic_boundary_Z,
                "w--", lw=1.0, alpha=0.7, label="Chaotic boundary", zorder=4,
            )

        # --- Fixed points ---
        if show_fixedpoints:
            for fp in self.all_fixed_points:
                R0, Z0 = fp.rzphi0[0], fp.rzphi0[1]
                if fp.orbit_type == "O":
                    ax.plot(R0, Z0, "o", color="limegreen",
                            ms=7, mew=1.5, zorder=8,
                            label=f"O q={fp.m}/{fp.n}" if fp == self.all_fixed_points[0] else "")
                else:
                    ax.plot(R0, Z0, "x", color="tomato",
                            ms=9, mew=2.0, zorder=8)
                    # Draw eigenvectors
                    if fp.eigenvectors is not None:
                        scale = 0.04
                        for k in range(2):
                            ev = fp.eigenvectors[:, k].real
                            ev /= (np.linalg.norm(ev) + 1e-30)
                            ax.annotate(
                                "", xy=(R0 + scale * ev[0], Z0 + scale * ev[1]),
                                xytext=(R0, Z0),
                                arrowprops=dict(arrowstyle="->", color="cyan", lw=1.2),
                                zorder=9,
                            )

        # --- Manifolds ---
        if show_manifolds:
            from pyna.topo.manifold_improve import StableManifold, UnstableManifold
            from pyna.topo.variational import _fd_jacobian

            def _field_func_2d(R, Z, phi, _ff=self.field_func):
                tang = _ff(np.array([R, Z, phi]))
                dphi = tang[2]
                if abs(dphi) < 1e-15:
                    return np.array([0.0, 0.0])
                return np.array([tang[0] / dphi, tang[1] / dphi])

            for fp in self.all_fixed_points:
                if fp.orbit_type != "X":
                    continue
                phi_span = (phi0, phi0 + 2 * np.pi * fp.period)
                sm = StableManifold(fp.rzphi0[:2], fp.monodromy,
                                    _field_func_2d, phi_span=phi_span)
                um = UnstableManifold(fp.rzphi0[:2], fp.monodromy,
                                      _field_func_2d, phi_span=phi_span)
                sm.grow(n_turns=n_turns_manifold, init_length=5e-5, n_init_pts=4,
                        both_sides=True, rtol=1e-9, atol=1e-12)
                um.grow(n_turns=n_turns_manifold, init_length=5e-5, n_init_pts=4,
                        both_sides=True, rtol=1e-9, atol=1e-12)
                for seg in sm.segments:
                    if len(seg) >= 2:
                        ax.plot(seg[:, 0], seg[:, 1], "-", color="steelblue",
                                lw=0.9, alpha=0.85, zorder=6)
                for seg in um.segments:
                    if len(seg) >= 2:
                        ax.plot(seg[:, 0], seg[:, 1], "-", color="darkorange",
                                lw=0.9, alpha=0.85, zorder=6)

        ax.set_aspect("equal")
        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(f"Poincaré section φ = {phi0:.3f}")
        return fig, ax


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _winding_number(
    field_func: Callable,
    R0: float,
    Z0: float,
    phi0: float,
    n_turns: int,
    dt: float = 0.05,
) -> Optional[float]:
    """Estimate safety factor q for a single field line by winding count."""
    rzphi = np.array([R0, Z0, phi0])
    total_dphi = 0.0
    total_dtheta = 0.0
    phi = phi0

    def aug(t, y):
        tang = np.asarray(field_func(y), dtype=float)
        return tang

    # Integrate for n_turns toroidal transits
    phi_end = phi0 + 2 * np.pi * n_turns
    try:
        sol = rk4_integrate(
            aug,
            (phi0, phi_end),
            rzphi,
            max_step=dt,
            dense_output=True,
        )
        if not sol.success:
            return None
    except Exception:
        return None

    # Count toroidal turns and poloidal angle advance
    phi_arr = sol.t
    R_arr = sol.y[0]
    Z_arr = sol.y[1]

    # Reference centre (first point serves as approximate axis)
    R_ref = R_arr.mean()
    Z_ref = Z_arr.mean()

    theta_arr = np.arctan2(Z_arr - Z_ref, R_arr - R_ref)
    dtheta = np.diff(np.unwrap(theta_arr))
    dphi = np.diff(phi_arr)

    valid = dphi > 0
    if not valid.any():
        return None

    q_est = np.sum(dphi[valid]) / (np.sum(dtheta[valid]) + 1e-30)
    return float(abs(q_est))


def _make_field_func_2d(field_func: Callable) -> Callable:
    """Wrap a unit-tangent field_func into (dR/dφ, dZ/dφ) form."""
    def ff2d(R: float, Z: float, phi: float) -> np.ndarray:
        tang = np.asarray(field_func(np.array([R, Z, phi])), dtype=float)
        dphi_ds = tang[2]
        if abs(dphi_ds) < 1e-15:
            return np.array([0.0, 0.0])
        return np.array([tang[0] / dphi_ds, tang[1] / dphi_ds])
    return ff2d


def _classify_and_collect_orbit(
    orbit: PeriodicOrbit,
    m: int,
    n: int,
    field_func_2d: Callable,
    phi0: float = 0.0,
    fd_eps: float = 1e-6,
) -> FixedPointInfo:
    """Compute monodromy and classify a PeriodicOrbit as O- or X-point."""
    phi_span = (phi0, phi0 + 2 * np.pi * orbit.period_n)
    vq = PoincareMapVariationalEquations(field_func_2d, fd_eps=fd_eps)
    try:
        M = vq.jacobian_matrix(orbit.rzphi0[:2], phi_span)
    except Exception as e:
        warnings.warn(f"Monodromy computation failed for q={m}/{n}: {e}")
        M = np.eye(2)

    eigvals, eigvecs = np.linalg.eig(M)
    k = float((eigvals[0] + eigvals[1]).real / 2.0)

    if abs(k) < 1.0:
        otype = "O"
    else:
        otype = "X"

    return FixedPointInfo(
        rzphi0=orbit.rzphi0.copy(),
        orbit_type=otype,
        m=m,
        n=n,
        period=orbit.period_n,
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        monodromy=M,
        stability_index=k,
    )


def _q_profile_from_fieldlines(
    field_func: Callable,
    R_axis: float,
    Z_axis: float,
    R_range: Tuple[float, float],
    phi0: float = 0.0,
    n_radial: int = 20,
    n_turns: int = 30,
    dt: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate q(S) by winding-number integration on the midplane."""
    R_vals = np.linspace(R_range[0], R_range[1], n_radial)
    S_vals = (R_vals - R_axis) / (R_range[1] - R_axis)  # rough S proxy
    S_vals = np.clip(S_vals, 0.01, 0.99)

    q_vals = []
    for R0 in R_vals:
        q = _winding_number(field_func, R0, Z_axis, phi0, n_turns=n_turns, dt=dt)
        q_vals.append(q if q is not None else np.nan)

    return S_vals, np.array(q_vals)


# ---------------------------------------------------------------------------
# Main facade
# ---------------------------------------------------------------------------

def analyse_topology(
    field_func: Callable,
    R_range: Tuple[float, float],
    Z_range: Tuple[float, float],
    phi_sections: List[float] = None,
    # Q-profile estimation
    R_axis: Optional[float] = None,
    Z_axis: float = 0.0,
    q_range: Tuple[float, float] = (1.0, 5.0),
    n_q_radial: int = 30,
    n_q_turns: int = 40,
    # FTLE / chaos detection
    compute_ftle: bool = True,
    n_ftle_pts: int = 24,
    ftle_turns: int = 20,
    ftle_threshold: Optional[float] = None,
    # Island chain detection
    island_m_max: int = 6,
    island_n_max: int = 2,
    island_depth: int = 2,
    n_island_seeds: int = 6,
    dt: float = 0.05,
    # Fixed-point solver
    find_fixed_points: bool = True,
    fp_tol: float = 1e-8,
    fp_max_iter: int = 60,
    fp_fd_eps: float = 1e-6,
    # Verbosity
    verbose: bool = True,
) -> TopologyReport:
    """Perform a full topology analysis of a 3-D magnetic field.

    Parameters
    ----------
    field_func : callable
        ``field_func(rzphi)`` → array-like ``[dR/ds, dZ/ds, dφ/ds]`` in
        cylindrical coordinates.  The parameter ``s`` may be arc length,
        toroidal angle, or time — the function only needs to return a vector
        tangent to the field line at every point.
    R_range : (R_min, R_max)
        Radial extent of the analysis domain (metres).
    Z_range : (Z_min, Z_max)
        Vertical extent of the analysis domain (metres).
    phi_sections : list of float, optional
        Toroidal angles (radians) at which Poincaré sections are taken.
        Default ``[0.0]``.
    R_axis : float, optional
        Major radius of the magnetic axis.  If None, estimated as the
        midpoint of R_range.
    Z_axis : float
        Vertical position of the magnetic axis.  Default 0.
    q_range : (q_min, q_max)
        Range of safety factors to consider.
    n_q_radial : int
        Number of radial points for q-profile estimation.
    n_q_turns : int
        Number of toroidal turns for winding-number estimation.
    compute_ftle : bool
        Whether to compute the FTLE field.  Can be slow for large grids.
    n_ftle_pts : int
        Grid resolution (n × n) for the FTLE computation.
    ftle_turns : int
        Number of toroidal turns for FTLE integration.
    ftle_threshold : float or None
        FTLE threshold for the chaotic-boundary extraction.  If None,
        automatically set to the 75th percentile of the FTLE field.
    island_m_max : int
        Maximum poloidal mode number to search for island chains.
    island_n_max : int
        Maximum toroidal mode number to search for island chains.
    island_depth : int
        Recursion depth for island-around-island hierarchy (1 = top-level
        only, 2 = one level of sub-islands, etc.).
    n_island_seeds : int
        Angular seeds per expected fixed point.
    dt : float
        Integration step for field-line tracing.
    find_fixed_points : bool
        Whether to run the fixed-point Newton solver.
    fp_tol : float
        Newton solver convergence tolerance.
    fp_max_iter : int
        Maximum Newton iterations.
    fp_fd_eps : float
        Finite-difference step for monodromy matrix computation.
    verbose : bool
        Print progress messages.

    Returns
    -------
    TopologyReport
    """
    if phi_sections is None:
        phi_sections = [0.0]

    if R_axis is None:
        R_axis = 0.5 * (R_range[0] + R_range[1])

    warn_list: List[str] = []
    phi0 = phi_sections[0]

    def _log(msg: str) -> None:
        if verbose:
            print(f"[topology] {msg}")

    # ------------------------------------------------------------------
    # Step 1 – FTLE and chaos / regularity boundary
    # ------------------------------------------------------------------
    ftle_arr: Optional[np.ndarray] = None
    R_grid: Optional[np.ndarray] = None
    Z_grid: Optional[np.ndarray] = None
    chaos_R: Optional[np.ndarray] = None
    chaos_Z: Optional[np.ndarray] = None

    if compute_ftle:
        _log(f"Computing FTLE on {n_ftle_pts}×{n_ftle_pts} grid "
             f"({ftle_turns} turns)...")
        R_1d = np.linspace(R_range[0], R_range[1], n_ftle_pts)
        Z_1d = np.linspace(Z_range[0], Z_range[1], n_ftle_pts)
        R_grid, Z_grid = np.meshgrid(R_1d, Z_1d, indexing='ij')
        try:
            ftle_arr = ftle_field(
                field_func, R_grid, Z_grid,
                phi0=phi0,
                t_max=ftle_turns * 2 * np.pi,
            )
            threshold_pct = 75.0 if ftle_threshold is None else None
            if ftle_threshold is not None:
                # convert absolute threshold to percentile
                threshold_pct = float(
                    100.0 * np.mean(ftle_arr[np.isfinite(ftle_arr)]
                                    <= ftle_threshold)
                )
            chaos_R, chaos_Z = chaotic_boundary_estimate(
                ftle_arr, R_grid, Z_grid,
                threshold_percentile=threshold_pct,
            )
            _log(f"FTLE done. Chaotic boundary threshold = {threshold:.4f}")
        except Exception as e:
            warn_list.append(f"FTLE computation failed: {e}")
            _log(f"FTLE failed: {e}")

    # ------------------------------------------------------------------
    # Step 2 – q-profile estimation for the main regular zone
    # ------------------------------------------------------------------
    _log(f"Estimating q-profile ({n_q_radial} radial pts, {n_q_turns} turns)...")
    R_inner = R_axis + 0.02 * (R_range[1] - R_axis)
    R_outer = R_axis + 0.92 * (R_range[1] - R_axis)
    S_vals, q_vals = _q_profile_from_fieldlines(
        field_func, R_axis, Z_axis,
        R_range=(R_inner, R_outer),
        phi0=phi0,
        n_radial=n_q_radial,
        n_turns=n_q_turns,
        dt=dt,
    )
    # Mask NaN and clip to q_range
    valid = np.isfinite(q_vals) & (q_vals >= q_range[0]) & (q_vals <= q_range[1])
    if valid.sum() < 4:
        warn_list.append("Fewer than 4 valid q values; q-profile may be unreliable.")
    S_valid = S_vals[valid]
    q_valid = q_vals[valid]

    main_zone = RegularZone(
        label="core",
        S_range=(float(S_valid.min()) if len(S_valid) else 0.0,
                 float(S_valid.max()) if len(S_valid) else 1.0),
        q_profile_S=S_valid,
        q_profile=q_valid,
    )
    _log(f"q range in core: [{q_valid.min():.3f}, {q_valid.max():.3f}]"
         if len(q_valid) else "q profile empty")

    # ------------------------------------------------------------------
    # Step 3 – Locate rational surfaces + island half-widths
    # ------------------------------------------------------------------
    _log(f"Locating rational surfaces (m≤{island_m_max}, n≤{island_n_max})...")

    if len(S_valid) >= 4:
        rational = locate_all_rational_surfaces(
            S_valid, q_valid,
            m_max=island_m_max,
            n_max=island_n_max,
        )
    else:
        rational = {}
        warn_list.append("Not enough valid q points to locate rational surfaces.")

    # Collect (m, n, S_res, q_res) tuples, sorted by S_res
    surface_list: List[Tuple[int, int, float, float]] = []
    for m_key, n_dict in rational.items():
        if m_key <= 0:
            continue
        for n_key, S_list in n_dict.items():
            for S_r in S_list:
                # interpolate q at S_r
                if len(S_valid) >= 2:
                    q_r = float(UnivariateSpline(S_valid, q_valid, s=0.01)(S_r))
                else:
                    q_r = float(m_key) / float(n_key)
                surface_list.append((int(m_key), int(n_key), float(S_r), q_r))
    surface_list.sort(key=lambda t: t[2])  # sort by S_res

    _log(f"Found {len(surface_list)} rational surfaces.")

    # ------------------------------------------------------------------
    # Step 4 – Fixed-point Newton solver + monodromy
    # ------------------------------------------------------------------
    ff2d = _make_field_func_2d(field_func)
    all_fixed_points: List[FixedPointInfo] = []
    island_chains: List[IslandChain] = []
    RZlimit = (R_range[0], R_range[1], Z_range[0], Z_range[1])

    for mn_idx, (m, n, S_r, q_r) in enumerate(surface_list):
        _log(f"  q={m}/{n}  S={S_r:.3f}  q_est={q_r:.4f}")

        half_w = 0.0
        fps_this: List[FixedPointInfo] = []

        if find_fixed_points:
            # Build a minimal "equilibrium-like" proxy for find_all_cycles
            class _Proxy:
                def __init__(self):
                    self.R0 = R_axis
                    self.r0 = R_range[1] - R_axis
                def resonant_psi(self, _m, _n):
                    return [S_r ** 2]  # S² ≈ ψ_norm proxy

            proxy = _Proxy()
            try:
                from pyna.topo.toroidal_cycle import find_all_cycles_near_resonance
                orbits = find_all_cycles_near_resonance(
                    field_func, proxy, m, n,
                    n_seeds=n_island_seeds,
                    dt=dt,
                    RZlimit=RZlimit,
                )
                _log(f"    → {len(orbits)} orbits found")
                for orb in orbits:
                    fp = _classify_and_collect_orbit(
                        orb, m, n, ff2d, phi0=phi0, fd_eps=fp_fd_eps
                    )
                    fps_this.append(fp)
                    all_fixed_points.append(fp)
            except Exception as e:
                warn_list.append(f"Fixed-point search failed for q={m}/{n}: {e}")
                _log(f"    → fixed-point search error: {e}")

        # Rough half-width from S spacing (fallback)
        half_w = (0.5 / (island_m_max + 1)) if half_w == 0.0 else half_w

        chain = IslandChain(
            m=m, n=n,
            S_res=S_r, q_res=q_r,
            half_width=half_w,
            fixed_points=fps_this,
        )
        island_chains.append(chain)
        main_zone.island_chains.append(chain)

    # ------------------------------------------------------------------
    # Step 5 – Chirikov overlap between adjacent chains
    # ------------------------------------------------------------------
    if len(island_chains) >= 2:
        positions = np.array([c.S_res for c in island_chains])
        widths = np.array([c.half_width for c in island_chains])
        try:
            sigma_arr = chirikov_overlap(widths, positions)
            for i, chain in enumerate(island_chains[:-1]):
                chain.chirikov_sigma = float(sigma_arr[i])
        except Exception as e:
            warn_list.append(f"Chirikov overlap computation failed: {e}")

    # ------------------------------------------------------------------
    # Step 6 – Island-around-island recursion (depth > 1)
    # ------------------------------------------------------------------
    if island_depth >= 2:
        for chain in island_chains:
            _log(f"  Sub-island search inside q={chain.m}/{chain.n}...")
            # Define a new q-profile inside the island
            r_island = chain.S_res * (R_range[1] - R_axis)
            R_sub_inner = R_axis + (chain.S_res - chain.half_width) * (R_range[1] - R_axis) + 0.01
            R_sub_outer = R_axis + (chain.S_res + chain.half_width) * (R_range[1] - R_axis) - 0.01
            if R_sub_inner >= R_sub_outer:
                continue
            S_sub, q_sub = _q_profile_from_fieldlines(
                field_func, R_axis, Z_axis,
                R_range=(R_sub_inner, R_sub_outer),
                phi0=phi0,
                n_radial=max(8, n_q_radial // 2),
                n_turns=n_q_turns,
                dt=dt,
            )
            valid_sub = np.isfinite(q_sub)
            if valid_sub.sum() < 4:
                continue
            sub_rational = locate_all_rational_surfaces(
                S_sub[valid_sub], q_sub[valid_sub],
                m_max=island_m_max // 2,
                n_max=island_n_max,
            )
            for m2, n2_dict in sub_rational.items():
                if m2 <= 0:
                    continue
                for n2, S2_list in n2_dict.items():
                    for S2 in S2_list:
                        child = IslandChain(
                            m=int(m2), n=int(n2),
                            S_res=float(S2), q_res=float(m2) / float(n2),
                            half_width=chain.half_width / 4.0,
                        )
                        chain.children.append(child)
            _log(f"    → {len(chain.children)} sub-islands found")

    # ------------------------------------------------------------------
    # Assemble and return
    # ------------------------------------------------------------------
    report = TopologyReport(
        field_func=field_func,
        phi_sections=phi_sections,
        ftle=ftle_arr,
        R_grid=R_grid,
        Z_grid=Z_grid,
        chaotic_boundary_R=chaos_R,
        chaotic_boundary_Z=chaos_Z,
        regular_zones=[main_zone],
        all_fixed_points=all_fixed_points,
        warnings=warn_list,
    )

    _log("Analysis complete.")
    return report
