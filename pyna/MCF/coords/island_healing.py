"""
Heal PEST flux-surface coordinates at boundary island chains.

When a device's boundary is defined by an m/n island chain (e.g. W7-X 5/5),
the standard PEST construction breaks down at the LCFS because field lines
no longer close on rational surfaces.  This module extends the PEST mesh
through the island region by placing the X/O ring at the normalised radius
r = S = 1.

Background
----------
In PEST (straight-field-line) coordinates (S, θ*, φ):

  S   = √(ψ_norm)   radial coordinate (0 = axis, 1 = boundary)
  θ*  is chosen so that  B · ∇θ* / B · ∇φ = q(S) on each flux surface

At a rational surface q = m/n the field lines satisfy::

    θ*(φ) = (n/m) · φ + θ*_0

so all m O-points and m X-points on a φ = const section appear at equally
spaced PEST angles, separated by π/m::

    O-points:  θ*_0,   θ*_0 + 2π/m,   …,  θ*_0 + 2π(m-1)/m
    X-points:  θ*_0 + π/m,  θ*_0 + 3π/m,  …,  θ*_0 + (2m-1)π/m

This identity provides a complete geometric constraint on the r = 1 shape.

Conceptual hierarchy
--------------------
Following the island-around-island picture adopted in ``pyna.topo``:

* **Level-1 (primary)** – the main nested flux-surface structure treated as
  a single-island chain (m = 1) whose O-point is the magnetic axis.  The
  standard PEST mesh covers r ∈ [0, r_sep) for this chain.
* **Level-2 (boundary)** – the edge island chain (e.g. 5/5) that breaks the
  primary LCFS.  Its X/O ring is placed at r = 1 by this module.

A chain can be connected (all flux tubes share a common separatrix, typical
for inner-region islands) or disconnected (independent flux tubes, typical
for W7-X 5/5 std. config.): both cases are handled because the r = 1 curve
is constructed purely from the 2m anchor-point positions and PEST angles,
with no assumption about field-line connectivity.

Algorithm
---------
1. Estimate θ*_0 (the reference PEST angle of the first O-point) by
   interpolating the geometric-angle ↔ θ* map from the outermost existing
   PEST surfaces.
2. Assign θ* to all 2m anchor points using the equally-spaced rule.
3. Build the r = 1 boundary curve R(θ*), Z(θ*) via a periodic cubic spline
   through the 2m anchor points.
4. Add ``n_heal`` new flux surfaces interpolated between the last good
   surface and r = 1.

Public API
----------
.. function:: assign_island_chain_pest_angles(island_chain, R_mesh, ...)
   Assign θ* to all X/O points; update Island.pest_theta in place.

.. function:: build_r1_boundary(island_chain, theta_O, theta_X, TET)
   Construct the r = 1 boundary curve.

.. function:: heal_pest_mesh_at_island_chain(S, TET, R_mesh, Z_mesh, ...)
   Main entry point: extend PEST mesh to r = 1.
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d, CubicSpline


# ---------------------------------------------------------------------------
# § 1  Low-level helpers: geometric angle ↔ PEST angle
# ---------------------------------------------------------------------------

def _geo_angle(R: float, Z: float, Rmaxis: float, Zmaxis: float) -> float:
    """Geometric poloidal angle of (R, Z) w.r.t. the magnetic axis [rad, (-π, π]]."""
    return float(np.arctan2(Z - Zmaxis, R - Rmaxis))


def _build_geo2pest(
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    TET: np.ndarray,
    Rmaxis: float,
    Zmaxis: float,
) -> interp1d:
    """Build a linear map  geometric angle → PEST angle θ*  on one surface.

    The map is extended for periodicity so that the input domain covers
    approximately (−2π, 4π).

    Parameters
    ----------
    R_surf, Z_surf : 1-D arrays, shape (ntheta,)
        Cylindrical coordinates along one PEST flux surface.
    TET : 1-D array, shape (ntheta,)
        Corresponding PEST poloidal angles [0, 2π] (duplicate endpoints OK).
    Rmaxis, Zmaxis : float
        Magnetic axis.

    Returns
    -------
    interp1d
        Callable  geo_angle [rad] → θ* [rad].
    """
    geo = np.arctan2(R_surf - Rmaxis, Z_surf - Zmaxis)  # not used below
    geo = np.arctan2(Z_surf - Zmaxis, R_surf - Rmaxis)

    order = np.argsort(geo)
    geo_s = geo[order]
    tet_s = TET[order]

    # Remove duplicates
    _, uidx = np.unique(geo_s, return_index=True)
    geo_s = geo_s[uidx]
    tet_s = tet_s[uidx]

    # Periodic extension: three copies spanning (−2π, 4π)
    geo_ext = np.r_[geo_s - 2*np.pi, geo_s, geo_s + 2*np.pi]
    tet_ext = np.r_[tet_s - 2*np.pi, tet_s, tet_s + 2*np.pi]

    return interp1d(geo_ext, tet_ext, kind='linear', fill_value='extrapolate')


def _estimate_pest_angle(
    R_pt: float,
    Z_pt: float,
    R_mesh: np.ndarray,
    Z_mesh: np.ndarray,
    TET: np.ndarray,
    Rmaxis: float,
    Zmaxis: float,
    n_avg: int = 3,
) -> float:
    """Estimate the PEST angle θ* of a point (R_pt, Z_pt) outside the mesh.

    Averages the estimate from the ``n_avg`` outermost PEST surfaces for
    robustness.

    Returns
    -------
    float
        θ* in [0, 2π).
    """
    ns = R_mesh.shape[0]
    n_use = min(n_avg, ns)
    geo_pt = _geo_angle(R_pt, Z_pt, Rmaxis, Zmaxis)

    estimates = []
    for i in range(ns - n_use, ns):
        f = _build_geo2pest(R_mesh[i], Z_mesh[i], TET, Rmaxis, Zmaxis)
        estimates.append(float(f(geo_pt)))

    arr = np.array(estimates)
    # Circular mean
    mean_val = float(np.arctan2(np.mean(np.sin(arr)), np.mean(np.cos(arr))))
    return mean_val % (2 * np.pi)


# ---------------------------------------------------------------------------
# § 2  Public: assign PEST angles to island X/O points
# ---------------------------------------------------------------------------

def assign_island_chain_pest_angles(
    island_chain,
    R_mesh: np.ndarray,
    Z_mesh: np.ndarray,
    TET: np.ndarray,
    Rmaxis: float,
    Zmaxis: float,
    n_avg: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign PEST angles θ* to all X- and O-points of a boundary island chain.

    Exploits the rational-surface identity  θ*(φ) = (n/m)·φ + θ*_0  which
    ensures that all 2m anchor points are equally spaced at π/m in θ*.

    The reference angle θ*_0 is determined by a circular-mean fit over the
    individual per-O-point estimates obtained from the existing PEST mesh.

    Each ``Island.pest_theta`` attribute is updated in-place.
    The results are also stored as ``island_chain.pest_theta_O`` and
    ``island_chain.pest_theta_X``.

    Parameters
    ----------
    island_chain : IslandChain
        Must have ``islands`` populated with ``O_point`` arrays.
        ``X_points`` are optional (used in :func:`build_r1_boundary`).
    R_mesh, Z_mesh : ndarray, shape (ns, ntheta)
        Existing PEST mesh (valid up to some S < 1).
    TET : ndarray, shape (ntheta,)
        PEST poloidal angle grid [0, 2π].
    Rmaxis, Zmaxis : float
        Magnetic axis.
    n_avg : int
        Number of outer PEST surfaces used for angle estimation.

    Returns
    -------
    theta_O : ndarray, shape (m,)
        PEST angles of O-points in [0, 2π), ascending.
    theta_X : ndarray, shape (m,)
        PEST angles of X-points in [0, 2π), ascending.
        theta_X[k] = (theta_O[k] + π/m) mod 2π.

    Raises
    ------
    ValueError
        If no O-points are found.
    """
    m = island_chain.m
    dphi = 2.0 * np.pi / m          # spacing between consecutive O-points
    half_dphi = np.pi / m            # O-to-X spacing

    O_points = [isl.O_point for isl in island_chain.islands
                if isl.O_point is not None]
    if not O_points:
        raise ValueError(
            "IslandChain.islands contains no O_points. "
            "Run fixed-point finding before calling this function."
        )

    # -------------------------------------------------------------------
    # Step 1: estimate θ* for each O-point individually
    # -------------------------------------------------------------------
    raw = np.array([
        _estimate_pest_angle(
            float(op[0]), float(op[1]),
            R_mesh, Z_mesh, TET, Rmaxis, Zmaxis, n_avg=n_avg,
        )
        for op in O_points
    ])

    # -------------------------------------------------------------------
    # Step 2: fold all estimates onto [0, 2π/m) and take circular mean
    #         to recover the reference angle θ*_0.
    # -------------------------------------------------------------------
    folded = raw % dphi
    # Map [0, dphi) → [0, 2π) for circular statistics, then map back
    phi_norm = folded * (2.0 * np.pi / dphi)
    theta_0_norm = float(
        np.arctan2(np.mean(np.sin(phi_norm)), np.mean(np.cos(phi_norm)))
    ) % (2.0 * np.pi)
    theta_0 = theta_0_norm * (dphi / (2.0 * np.pi))   # back to [0, dphi)

    # -------------------------------------------------------------------
    # Step 3: build the full rings
    # -------------------------------------------------------------------
    theta_O = np.sort(np.array([(theta_0 + k * dphi) % (2*np.pi) for k in range(m)]))
    theta_X = np.sort(np.array([(theta_0 + half_dphi + k * dphi) % (2*np.pi)
                                 for k in range(m)]))

    # -------------------------------------------------------------------
    # Step 4: update Island.pest_theta in place (match to nearest theta_O)
    # -------------------------------------------------------------------
    for isl in island_chain.islands:
        if isl.O_point is not None:
            tet_est = _estimate_pest_angle(
                float(isl.O_point[0]), float(isl.O_point[1]),
                R_mesh, Z_mesh, TET, Rmaxis, Zmaxis, n_avg=n_avg,
            )
            ang_diffs = np.abs(np.angle(np.exp(1j * (theta_O - tet_est))))
            isl.pest_theta = float(theta_O[int(np.argmin(ang_diffs))])

    # Store on chain object for convenience
    island_chain.pest_theta_O = theta_O
    island_chain.pest_theta_X = theta_X

    return theta_O, theta_X


# ---------------------------------------------------------------------------
# § 3  Build the r = 1 boundary curve
# ---------------------------------------------------------------------------

def build_r1_boundary(
    island_chain,
    theta_O: np.ndarray,
    theta_X: np.ndarray,
    TET_out: np.ndarray,
    spline_kind: str = 'cubic',
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct the r = 1 boundary curve R(θ*), Z(θ*).

    The boundary passes through:

    * O-points at  θ* = theta_O[k]
    * X-points at  θ* = theta_X[k]   (used when X_points are available)

    In between, a periodic spline is fitted through the 2m anchor points.
    For disconnected island chains (e.g. W7-X 5/5) the X-points provide
    the "saddle" shape between flux tubes; for connected chains they mark
    the separatrix corners.

    Parameters
    ----------
    island_chain : IslandChain
        ``islands[k].O_point`` gives the k-th O-point position.
        ``islands[k].X_points`` gives the neighbouring X-points.
    theta_O, theta_X : ndarray, shape (m,)
        PEST angles from :func:`assign_island_chain_pest_angles`.
    TET_out : ndarray, shape (ntheta,)
        PEST angle grid at which to evaluate the boundary.
    spline_kind : {'cubic', 'linear'}
        Interpolation order.  'cubic' gives a C² boundary; 'linear' is
        faster and safer when fewer than 4 anchors are available.

    Returns
    -------
    R_bdy, Z_bdy : ndarray, shape (ntheta,)
        Boundary curve in cylindrical (R, Z) coordinates.

    Raises
    ------
    ValueError
        If fewer than 2 anchor points are found.
    """
    m = island_chain.m

    # -------------------------------------------------------------------
    # Collect (theta*, R, Z) anchor points
    # -------------------------------------------------------------------
    theta_anc: list[float] = []
    R_anc:     list[float] = []
    Z_anc:     list[float] = []

    # --- O-points ---
    islands_with_O = [isl for isl in island_chain.islands if isl.O_point is not None]
    # Match each island to its theta_O slot
    for isl in islands_with_O:
        th = isl.pest_theta
        if th is None:
            # Fallback: use nearest theta_O
            ang_diff = np.abs(np.angle(
                np.exp(1j * (theta_O - _fallback_angle(isl.O_point, theta_O)))
            ))
            th = float(theta_O[int(np.argmin(ang_diff))])
        theta_anc.append(th)
        R_anc.append(float(isl.O_point[0]))
        Z_anc.append(float(isl.O_point[1]))

    # --- X-points ---
    all_xpts = []
    for isl in island_chain.islands:
        all_xpts.extend(isl.X_points)

    if all_xpts:
        # Match X-points to theta_X slots by closest geometric proximity
        used = set()
        for k, th_x in enumerate(theta_X):
            best_d = np.inf
            best_xp = None
            for j, xp in enumerate(all_xpts):
                if j in used:
                    continue
                d = float(np.linalg.norm(xp - _xpt_ref_for_angle(th_x, theta_O, islands_with_O)))
                # use angular distance from estimated θ* as sorting key
                if d < best_d:
                    best_d = d
                    best_xp = (j, xp)
            if best_xp is not None:
                j, xp = best_xp
                used.add(j)
                theta_anc.append(float(th_x))
                R_anc.append(float(xp[0]))
                Z_anc.append(float(xp[1]))

    if len(theta_anc) < 2:
        raise ValueError(
            f"Only {len(theta_anc)} anchor point(s) found for the r=1 boundary. "
            "Ensure IslandChain.islands has O_points (and ideally X_points)."
        )

    # -------------------------------------------------------------------
    # Sort anchors by θ* and deduplicate (tolerance = π/(4m))
    # -------------------------------------------------------------------
    order = np.argsort(theta_anc)
    theta_s = np.array(theta_anc)[order]
    R_s     = np.array(R_anc)[order]
    Z_s     = np.array(Z_anc)[order]

    dedup_tol = np.pi / (4 * max(island_chain.m, 1))
    keep = np.ones(len(theta_s), dtype=bool)
    for i in range(1, len(theta_s)):
        if theta_s[i] - theta_s[i - 1] < dedup_tol:
            keep[i] = False
    theta_a = theta_s[keep]
    R_a     = R_s[keep]
    Z_a     = Z_s[keep]

    if len(theta_a) < 2:
        raise ValueError(
            f"After deduplication only {len(theta_a)} anchor point(s) remain. "
            "Ensure IslandChain.islands has O_points and X_points are geometrically distinct."
        )

    # Append first point at θ + 2π for periodicity
    theta_per = np.r_[theta_a, theta_a[0] + 2*np.pi]
    R_per     = np.r_[R_a, R_a[0]]
    Z_per     = np.r_[Z_a, Z_a[0]]

    # -------------------------------------------------------------------
    # Build spline and evaluate on TET_out
    # -------------------------------------------------------------------
    n_anchors = len(theta_per)
    use_cubic = (spline_kind == 'cubic') and (n_anchors >= 4)

    if use_cubic:
        cs_R = CubicSpline(theta_per, R_per, bc_type='periodic' if R_per[0] == R_per[-1] else 'not-a-knot')
        cs_Z = CubicSpline(theta_per, Z_per, bc_type='periodic' if Z_per[0] == Z_per[-1] else 'not-a-knot')
        # Evaluate: wrap TET_out to [theta_per[0], theta_per[-1]]
        tet_eval = theta_a[0] + (TET_out - theta_a[0]) % (2*np.pi)
        R_bdy = cs_R(tet_eval)
        Z_bdy = cs_Z(tet_eval)
    else:
        f_R = interp1d(theta_per, R_per, kind='linear', fill_value='extrapolate')
        f_Z = interp1d(theta_per, Z_per, kind='linear', fill_value='extrapolate')
        tet_eval = theta_a[0] + (TET_out - theta_a[0]) % (2*np.pi)
        R_bdy = f_R(tet_eval)
        Z_bdy = f_Z(tet_eval)

    return R_bdy.astype(float), Z_bdy.astype(float)


def _fallback_angle(O_point, theta_O: np.ndarray) -> float:
    """Fallback: geometric index into theta_O (used when pest_theta is None)."""
    return float(theta_O[0])


def _xpt_ref_for_angle(
    th_x: float,
    theta_O: np.ndarray,
    islands: list,
) -> np.ndarray:
    """Reference position for X-point matching: midpoint between two adjacent O-points."""
    m = len(theta_O)
    # Find O-point just below th_x
    diffs = np.angle(np.exp(1j * (theta_O - th_x)))
    k_lo = int(np.argmax(diffs < 0) if np.any(diffs < 0) else 0)
    k_hi = (k_lo + 1) % m
    op_lo = np.asarray(islands[k_lo % len(islands)].O_point, dtype=float)
    op_hi = np.asarray(islands[k_hi % len(islands)].O_point, dtype=float)
    return 0.5 * (op_lo + op_hi)


# ---------------------------------------------------------------------------
# § 4  Main entry point: extend PEST mesh to r = 1
# ---------------------------------------------------------------------------

def heal_pest_mesh_at_island_chain(
    S: np.ndarray,
    TET: np.ndarray,
    R_mesh: np.ndarray,
    Z_mesh: np.ndarray,
    q_iS: np.ndarray,
    island_chain,
    Rmaxis: float,
    Zmaxis: float,
    *,
    n_heal: int = 20,
    S_last_good: Optional[float] = None,
    n_avg: int = 3,
    spline_kind: str = 'cubic',
    interp_radial: str = 'linear',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extend a PEST mesh to r = 1 by healing at a boundary island chain.

    The existing PEST mesh covers S ∈ [0, S_last_good].  This function adds
    ``n_heal`` new surfaces from S_last_good to S = 1, where the r = 1
    surface is defined by the X/O ring of the island chain.

    Parameters
    ----------
    S : ndarray, shape (ns,)
        Existing radial coordinate (S = √ψ_norm), ascending.
    TET : ndarray, shape (ntheta,)
        PEST poloidal angle grid [0, 2π].
    R_mesh, Z_mesh : ndarray, shape (ns, ntheta)
        Existing PEST mesh.
    q_iS : ndarray, shape (ns,)
        Safety factor on each surface.
    island_chain : IslandChain
        Boundary island chain.  Must have ``islands`` with ``O_point`` set.
        ``X_points`` are optional but strongly recommended for accuracy.
    Rmaxis, Zmaxis : float
        Magnetic axis.
    n_heal : int
        Number of new flux surfaces added between S_last_good and r = 1.
        Default 20.
    S_last_good : float or None
        Last reliable PEST surface.  Defaults to S[-1].
    n_avg : int
        Outer surfaces used for PEST angle estimation.
    spline_kind : {'cubic', 'linear'}
        Boundary-curve spline order (see :func:`build_r1_boundary`).
    interp_radial : {'linear', 'cubic'}
        Radial interpolation between S_last_good and r = 1.

    Returns
    -------
    S_out : ndarray, shape (ns + n_heal,)
        Extended radial grid.
    TET : ndarray (unchanged)
    R_out : ndarray, shape (ns + n_heal, ntheta)
    Z_out : ndarray, shape (ns + n_heal, ntheta)
    q_out : ndarray, shape (ns + n_heal,)

    Notes
    -----
    The safety factor at r = 1 is set to q = m/n (exact rational value).
    The q profile between S_last_good and 1 is linearly interpolated.

    After calling this function, ``island_chain.pest_theta_O``,
    ``island_chain.pest_theta_X``, and each ``Island.pest_theta`` are set.
    """
    if S_last_good is None:
        S_last_good = float(S[-1])

    # Index of last good surface (≤ S_last_good)
    idx_last = int(np.searchsorted(S, S_last_good, side='right')) - 1
    idx_last = max(0, min(idx_last, len(S) - 1))

    # ------------------------------------------------------------------
    # Step 1: assign PEST angles to island X/O points
    # ------------------------------------------------------------------
    theta_O, theta_X = assign_island_chain_pest_angles(
        island_chain,
        R_mesh[:idx_last+1],
        Z_mesh[:idx_last+1],
        TET,
        Rmaxis, Zmaxis,
        n_avg=n_avg,
    )

    # ------------------------------------------------------------------
    # Step 2: build the r = 1 boundary curve
    # ------------------------------------------------------------------
    R_bdy, Z_bdy = build_r1_boundary(
        island_chain, theta_O, theta_X, TET, spline_kind=spline_kind
    )

    # ------------------------------------------------------------------
    # Step 3: new radial grid from S_ref to 1
    # ------------------------------------------------------------------
    S_ref = float(S[idx_last])
    S_heal = np.linspace(S_ref, 1.0, n_heal + 2)[1:-1]   # excludes S_ref and 1
    S_r1   = np.array([1.0])
    S_new  = np.r_[S_heal, S_r1]                           # n_heal + 1 new surfaces

    # Reference surface (last good)
    R_ref = R_mesh[idx_last, :].copy()
    Z_ref = Z_mesh[idx_last, :].copy()

    # ------------------------------------------------------------------
    # Step 4: interpolate new surfaces
    # ------------------------------------------------------------------
    n_new = len(S_new)
    R_new = np.empty((n_new, len(TET)))
    Z_new = np.empty((n_new, len(TET)))

    if interp_radial == 'cubic' and n_new >= 4:
        # Use cubic interpolation in S for each θ* independently
        S_ctrl   = np.array([S_ref, 1.0])
        for j in range(len(TET)):
            cs_R = CubicSpline(S_ctrl, [R_ref[j], R_bdy[j]])
            cs_Z = CubicSpline(S_ctrl, [Z_ref[j], Z_bdy[j]])
            R_new[:, j] = cs_R(S_new)
            Z_new[:, j] = cs_Z(S_new)
    else:
        # Linear interpolation
        for i, s in enumerate(S_new):
            alpha = (s - S_ref) / (1.0 - S_ref)   # 0 at S_ref, 1 at r=1
            R_new[i, :] = (1.0 - alpha) * R_ref + alpha * R_bdy
            Z_new[i, :] = (1.0 - alpha) * Z_ref + alpha * Z_bdy

    # ------------------------------------------------------------------
    # Step 5: extend q profile linearly to q = m/n at r = 1
    # ------------------------------------------------------------------
    q_mn  = island_chain.m / island_chain.n
    q_ref = float(q_iS[idx_last])
    q_new = q_ref + (q_mn - q_ref) * (S_new - S_ref) / (1.0 - S_ref)

    # ------------------------------------------------------------------
    # Assemble output (keep original mesh up to and including idx_last)
    # ------------------------------------------------------------------
    S_out = np.r_[S[:idx_last+1], S_new]
    R_out = np.r_[R_mesh[:idx_last+1, :], R_new]
    Z_out = np.r_[Z_mesh[:idx_last+1, :], Z_new]
    q_out = np.r_[q_iS[:idx_last+1], q_new]

    return S_out, TET, R_out, Z_out, q_out
