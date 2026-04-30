"""
Heal PEST flux-surface coordinates at boundary island chains.

Physical background
-------------------
In PEST (straight-field-line) coordinates (S, θ*, φ):

  S   = √(ψ_norm)   radial coordinate (0 = axis, 1 = boundary)
  θ*  is chosen so that  B·∇θ* / B·∇φ = q(S) = const on each flux surface

At a rational surface q = m/n, the Poincaré map P (one toroidal turn) maps
each fixed point to another fixed point of the *same* chain.  Crucially:

  **The map shifts the PEST angle by exactly 2π·n/m per turn.**

So the k-th iterate P^k(x₀) has PEST angle:

    θ*(P^k(x₀)) = θ*(x₀) + k · (2π n/m)  mod 2π

This means the m O-points of an m/n chain, listed in **field-line traversal
order** (not geometric order), are equally spaced in θ* by 2πn/m.

When n and m are coprime (which is always true for a fundamental resonance),
the traversal order visits all m islands before returning to start.  The
sorted-in-θ* order of the O-points therefore has a gap of 2πn/m between
*consecutive traversal steps*, but a gap of 2π/m between *consecutive θ*
values* (since the traversal covers all m residues mod m exactly once).

For the boundary curve construction we need:
  - the m O-point positions R_k, Z_k in **ascending θ* order** (not
    traversal order), because the spline interpolates over θ* ∈ [0, 2π).
  - the m X-point positions interleaved, also in ascending θ* order.

The PEST angle of the k-th O-point in traversal order is:

    θ*_k = θ*_0 + k · (2π n/m)  mod 2π

Sorting these gives the spatial order along the boundary.

The correct procedure is therefore:
  1. Find ONE seed X- or O-point precisely (via find_fixed_points_batch).
  2. Propagate the orbit using IslandChainOrbit to get ALL m fixed points
     across all Poincaré sections with their correct DPm matrices.
  3. Assign PEST angles using the traversal-order formula.
  4. Sort by PEST angle to get spatial order.
  5. Build the periodic spline R(θ*), Z(θ*).

Hierarchy
---------
Level-1: primary nested flux surface structure (treated as m=1 chain).
Level-2: boundary island chain (e.g. 10/3) placed at r=1.

The boundary curve construction and PEST mesh extension work for any
(m, n) chain, connected or disconnected.
"""
from __future__ import annotations

import warnings
from math import gcd
from typing import Optional, Tuple, List

import numpy as np
from scipy.interpolate import interp1d, CubicSpline


# ---------------------------------------------------------------------------
# § 1  Geometric helpers
# ---------------------------------------------------------------------------

def _geo_angle(R: float, Z: float, Rmaxis: float, Zmaxis: float) -> float:
    """Geometric poloidal angle of (R, Z) w.r.t. the magnetic axis, in (-π, π]."""
    return float(np.arctan2(Z - Zmaxis, R - Rmaxis))


def _build_geo2pest(
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    TET: np.ndarray,
    Rmaxis: float,
    Zmaxis: float,
) -> interp1d:
    """Build a monotone map  geometric angle → θ*  on one PEST surface."""
    geo = np.arctan2(Z_surf - Zmaxis, R_surf - Rmaxis)
    order = np.argsort(geo)
    geo_s = geo[order]
    tet_s = TET[order]
    _, uidx = np.unique(geo_s, return_index=True)
    geo_s = geo_s[uidx]
    tet_s = tet_s[uidx]
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
    n_avg: int = 4,
) -> float:
    """Estimate θ* of a point outside the mesh by extrapolating from outer surfaces."""
    ns = R_mesh.shape[0]
    n_use = min(n_avg, ns)
    geo_pt = _geo_angle(R_pt, Z_pt, Rmaxis, Zmaxis)
    estimates = []
    for i in range(ns - n_use, ns):
        f = _build_geo2pest(R_mesh[i], Z_mesh[i], TET, Rmaxis, Zmaxis)
        estimates.append(float(f(geo_pt)))
    arr = np.array(estimates)
    mean_val = float(np.arctan2(np.mean(np.sin(arr)), np.mean(np.cos(arr))))
    return mean_val % (2 * np.pi)


# ---------------------------------------------------------------------------
# § 2  Core: traversal-order PEST angle assignment
# ---------------------------------------------------------------------------

def assign_pest_angles_from_orbit(
    island_chain_orbit,
    m: int,
    n: int,
    R_mesh: np.ndarray,
    Z_mesh: np.ndarray,
    TET: np.ndarray,
    Rmaxis: float,
    Zmaxis: float,
    phi_section: float = 0.0,
    n_avg: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assign PEST angles to all fixed points of an IslandChainOrbit.

    Each fixed point's θ* is estimated independently from its geometric
    position on the existing PEST mesh.  This avoids the traversal-order
    shift formula (θ*_seed + k·2πn/m), which is only valid when the list
    of fixed points is in field-line traversal order — a condition that
    ``at_section`` does NOT guarantee (points are typically sorted by R).
    Using the shift formula on a geometrically-ordered list causes each
    X/O pair to be assigned mismatched angles, producing the characteristic
    triangular kink (knot) pointing toward the magnetic axis in the
    boundary spline.

    Parameters
    ----------
    island_chain_orbit : IslandChainOrbit
        Must have fixed_points populated (from from_cyna_cache or
        from_single_fixedpoint).  Only fixed points at phi ≈ phi_section
        are used.
    m, n : int
        Toroidal / poloidal mode numbers (informational; not used for
        angle computation in this implementation).
    R_mesh, Z_mesh : ndarray, shape (ns, ntheta)
        Existing PEST mesh.
    TET : ndarray, shape (ntheta,)
        PEST angle grid [0, 2π].
    Rmaxis, Zmaxis : float
        Magnetic axis.
    phi_section : float
        Which Poincaré section to use [rad].
    n_avg : int
        Outer surfaces used for θ* estimation.

    Returns
    -------
    R_fps : ndarray, shape (m,)  — R positions of all m fixed points at phi_section
    Z_fps : ndarray, shape (m,)  — Z positions
    theta_fps : ndarray, shape (m,)  — PEST angles (one per point, from geometry)
    kinds : list of str  — 'X' or 'O' for each point
    """
    # Gather fixed points at this section
    fps_at_sec = island_chain_orbit.at_section(phi_section)
    if not fps_at_sec:
        raise ValueError(
            f"No fixed points found at phi_section={phi_section:.4f}. "
            "Check that section_phis includes this angle."
        )

    R_fps = np.array([fp.R for fp in fps_at_sec])
    Z_fps = np.array([fp.Z for fp in fps_at_sec])
    kinds = [fp.kind for fp in fps_at_sec]

    # Estimate θ* independently for each fixed point from its geometry.
    # This is safe regardless of the order in which at_section returns points.
    theta_fps = np.array([
        _estimate_pest_angle(
            float(R_fps[k]), float(Z_fps[k]),
            R_mesh, Z_mesh, TET, Rmaxis, Zmaxis, n_avg=n_avg,
        )
        for k in range(len(fps_at_sec))
    ])

    return R_fps, Z_fps, theta_fps, kinds


def assign_island_chain_pest_angles_from_orbit(
    island_chain_orbit,
    m: int,
    n: int,
    R_mesh: np.ndarray,
    Z_mesh: np.ndarray,
    TET: np.ndarray,
    Rmaxis: float,
    Zmaxis: float,
    phi_section: float = 0.0,
    n_avg: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full assignment: returns sorted O-point and X-point positions and angles.

    For an m/n chain the Poincaré map P maps each island to another island
    shifted by n positions in traversal order (i.e. shifted by 2πn/m in θ*).
    This function:
      1. Estimates θ* of the seed point.
      2. Propagates θ* to all m fixed points using the traversal-order formula.
      3. Separates X-points and O-points.
      4. Returns them sorted by ascending θ*.

    Returns
    -------
    R_O, Z_O, theta_O : O-point positions and PEST angles, sorted by θ*
    R_X, Z_X, theta_X : X-point positions and PEST angles, sorted by θ*
    """
    R_fps, Z_fps, theta_fps, kinds = assign_pest_angles_from_orbit(
        island_chain_orbit, m, n, R_mesh, Z_mesh, TET,
        Rmaxis, Zmaxis, phi_section=phi_section, n_avg=n_avg,
    )

    o_mask = np.array([k == 'O' for k in kinds])
    x_mask = np.array([k == 'X' for k in kinds])

    R_O = R_fps[o_mask]; Z_O = Z_fps[o_mask]; theta_O = theta_fps[o_mask]
    R_X = R_fps[x_mask]; Z_X = Z_fps[x_mask]; theta_X = theta_fps[x_mask]

    # Sort by PEST angle
    if len(theta_O) > 0:
        so = np.argsort(theta_O)
        R_O, Z_O, theta_O = R_O[so], Z_O[so], theta_O[so]
    if len(theta_X) > 0:
        sx = np.argsort(theta_X)
        R_X, Z_X, theta_X = R_X[sx], Z_X[sx], theta_X[sx]

    return R_O, Z_O, theta_O, R_X, Z_X, theta_X


# ---------------------------------------------------------------------------
# § 3  Legacy interface: assign PEST angles from IslandChain dataclass
#       (uses geometric-angle estimation, no orbit propagation)
#       Kept for compatibility but marked as approximate.
# ---------------------------------------------------------------------------

def assign_island_chain_pest_angles(
    island_chain,
    R_mesh: np.ndarray,
    Z_mesh: np.ndarray,
    TET: np.ndarray,
    Rmaxis: float,
    Zmaxis: float,
    n_avg: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign PEST angles to X/O points using geometric-angle estimation.

    .. note::
        This is the **approximate** interface for use when no
        ``IslandChainOrbit`` is available.  It does NOT account for the
        traversal-order shift (n positions per map step).  Use
        :func:`assign_island_chain_pest_angles_from_orbit` when possible.

    The reference angle θ*_0 is found by circular-mean folding of the
    per-O-point estimates.  O-points and X-points are then placed at
    equally-spaced θ* values starting from θ*_0.
    """
    m = island_chain.m
    n = island_chain.n
    dphi_O = 2.0 * np.pi / m
    half   = np.pi / m

    O_points = [isl.O_point for isl in island_chain.islands
                if isl.O_point is not None]
    if not O_points:
        raise ValueError("IslandChain has no O_points.")

    # Estimate θ* for each O-point
    raw = np.array([
        _estimate_pest_angle(float(op[0]), float(op[1]),
                             R_mesh, Z_mesh, TET, Rmaxis, Zmaxis, n_avg=n_avg)
        for op in O_points
    ])

    # Fold onto [0, 2π/m) and take circular mean → θ*_0
    folded   = raw % dphi_O
    phi_norm = folded * (2.0 * np.pi / dphi_O)
    theta_0_norm = float(np.arctan2(np.mean(np.sin(phi_norm)),
                                    np.mean(np.cos(phi_norm)))) % (2.0*np.pi)
    theta_0 = theta_0_norm * dphi_O / (2.0*np.pi)

    theta_O = np.sort([(theta_0 + k*dphi_O) % (2*np.pi) for k in range(m)])
    theta_X = np.sort([(theta_0 + half + k*dphi_O) % (2*np.pi) for k in range(m)])

    # Match each island's O_point to nearest theta_O slot
    for isl in island_chain.islands:
        if isl.O_point is None:
            continue
        tet_est = _estimate_pest_angle(
            float(isl.O_point[0]), float(isl.O_point[1]),
            R_mesh, Z_mesh, TET, Rmaxis, Zmaxis, n_avg=n_avg)
        diffs = np.abs(np.angle(np.exp(1j*(np.array(theta_O) - tet_est))))
        isl.pest_theta = float(theta_O[int(np.argmin(diffs))])

    island_chain.pest_theta_O = np.array(theta_O)
    island_chain.pest_theta_X = np.array(theta_X)
    return np.array(theta_O), np.array(theta_X)


# ---------------------------------------------------------------------------
# § 4  Build r = 1 boundary curve from sorted anchor points
# ---------------------------------------------------------------------------

def build_r1_boundary_from_anchors(
    theta_anc: np.ndarray,
    R_anc: np.ndarray,
    Z_anc: np.ndarray,
    TET_out: np.ndarray,
    spline_kind: str = 'cubic',
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the r=1 boundary curve from sorted (θ*, R, Z) anchor points.

    Anchors must be sorted in ascending θ* order.  A periodic cubic spline
    is fitted through the 2m (or m) points and evaluated on TET_out.

    Parameters
    ----------
    theta_anc, R_anc, Z_anc : ndarray, shape (N,)
        Anchor points sorted by θ* ∈ [0, 2π).
    TET_out : ndarray, shape (ntheta,)
        Output PEST angle grid.
    spline_kind : str
        'cubic' or 'linear'.

    Returns
    -------
    R_bdy, Z_bdy : ndarray, shape (ntheta,)
    """
    # Deduplicate (tolerance = π / (4 * N))
    N = len(theta_anc)
    if N < 2:
        raise ValueError(f"Need at least 2 anchor points, got {N}.")

    dedup_tol = np.pi / (4 * max(N, 1))
    keep = np.ones(N, dtype=bool)
    for i in range(1, N):
        if abs(theta_anc[i] - theta_anc[i-1]) < dedup_tol:
            keep[i] = False
    theta_a = theta_anc[keep]
    R_a     = R_anc[keep]
    Z_a     = Z_anc[keep]

    if len(theta_a) < 2:
        raise ValueError("Too few unique anchors after deduplication.")

    # Periodic closure
    theta_per = np.r_[theta_a, theta_a[0] + 2*np.pi]
    R_per     = np.r_[R_a, R_a[0]]
    Z_per     = np.r_[Z_a, Z_a[0]]

    # Wrap TET_out into [theta_a[0], theta_a[0] + 2π)
    t0 = theta_a[0]
    tet_eval = t0 + (TET_out - t0) % (2*np.pi)

    use_cubic = (spline_kind == 'cubic') and (len(theta_per) >= 4)
    if use_cubic:
        cs_R = CubicSpline(theta_per, R_per)
        cs_Z = CubicSpline(theta_per, Z_per)
        R_bdy = cs_R(tet_eval)
        Z_bdy = cs_Z(tet_eval)
    else:
        f_R = interp1d(theta_per, R_per, kind='linear', fill_value='extrapolate')
        f_Z = interp1d(theta_per, Z_per, kind='linear', fill_value='extrapolate')
        R_bdy = f_R(tet_eval)
        Z_bdy = f_Z(tet_eval)

    return R_bdy.astype(float), Z_bdy.astype(float)


def build_r1_boundary(
    island_chain,
    theta_O: np.ndarray,
    theta_X: np.ndarray,
    TET_out: np.ndarray,
    spline_kind: str = 'cubic',
) -> Tuple[np.ndarray, np.ndarray]:
    """Build r=1 boundary from IslandChain + pre-assigned θ* arrays.

    Merges O-point and X-point positions (if available) into a single
    sorted anchor list and calls :func:`build_r1_boundary_from_anchors`.

    .. warning::
        This uses the *approximate* θ* assignment (geometric angle).
        Prefer :func:`build_r1_boundary_from_orbit` when an
        IslandChainOrbit is available.
    """
    theta_anc, R_anc, Z_anc = [], [], []

    for isl in island_chain.islands:
        if isl.O_point is None:
            continue
        th = getattr(isl, 'pest_theta', None)
        if th is None:
            continue
        theta_anc.append(float(th))
        R_anc.append(float(isl.O_point[0]))
        Z_anc.append(float(isl.O_point[1]))

    # X-points: match to theta_X slots by nearest geometry
    all_xpts = []
    for isl in island_chain.islands:
        all_xpts.extend(isl.X_points)

    if all_xpts and len(theta_X) > 0:
        used = set()
        for k, th_x in enumerate(theta_X):
            best_d, best = np.inf, None
            for j, xp in enumerate(all_xpts):
                if j in used:
                    continue
                d = float(np.linalg.norm(np.array(xp) -
                          np.array([R_anc[k % len(R_anc)], Z_anc[k % len(Z_anc)]])))
                if d < best_d:
                    best_d, best = d, j
            if best is not None:
                used.add(best)
                theta_anc.append(float(th_x))
                R_anc.append(float(all_xpts[best][0]))
                Z_anc.append(float(all_xpts[best][1]))

    if len(theta_anc) < 2:
        raise ValueError(f"Only {len(theta_anc)} anchor points found.")

    order = np.argsort(theta_anc)
    return build_r1_boundary_from_anchors(
        np.array(theta_anc)[order],
        np.array(R_anc)[order],
        np.array(Z_anc)[order],
        TET_out, spline_kind=spline_kind,
    )


def build_r1_boundary_from_orbit(
    R_O: np.ndarray,
    Z_O: np.ndarray,
    theta_O: np.ndarray,
    R_X: np.ndarray,
    Z_X: np.ndarray,
    theta_X: np.ndarray,
    TET_out: np.ndarray,
    spline_kind: str = 'cubic',
) -> Tuple[np.ndarray, np.ndarray]:
    """Build r=1 boundary from orbit-propagated anchor points.

    Merges O-points and X-points (all in ascending θ* order) into a single
    sorted list and calls :func:`build_r1_boundary_from_anchors`.

    Parameters
    ----------
    R_O, Z_O, theta_O : ndarray — O-point positions + angles, sorted by θ*
    R_X, Z_X, theta_X : ndarray — X-point positions + angles, sorted by θ*
    TET_out : ndarray — output PEST angle grid
    spline_kind : str — 'cubic' or 'linear'
    """
    R_anc   = np.r_[R_O,     R_X]
    Z_anc   = np.r_[Z_O,     Z_X]
    theta_a = np.r_[theta_O, theta_X]

    order = np.argsort(theta_a)
    return build_r1_boundary_from_anchors(
        theta_a[order], R_anc[order], Z_anc[order],
        TET_out, spline_kind=spline_kind,
    )


# ---------------------------------------------------------------------------
# § 5  Main entry point: heal_pest_mesh_at_island_chain
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
    island_chain_orbit=None,
    phi_section: float = 0.0,
    n_heal: int = 20,
    S_last_good: Optional[float] = None,
    n_avg: int = 4,
    spline_kind: str = 'cubic',
    interp_radial: str = 'linear',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extend a PEST mesh to r=1 by healing at a boundary island chain.

    If ``island_chain_orbit`` is provided (recommended), uses the
    orbit-propagated fixed-point positions and the correct traversal-order
    PEST angle formula.  Otherwise falls back to the approximate geometric
    estimation from the legacy ``assign_island_chain_pest_angles``.

    Parameters
    ----------
    S, TET, R_mesh, Z_mesh, q_iS : existing PEST mesh
    island_chain : IslandChain
        Provides m, n, and (if no orbit given) O/X point positions.
    Rmaxis, Zmaxis : float — magnetic axis
    island_chain_orbit : IslandChainOrbit or None
        If given, used for precise fixed-point positions and θ* assignment.
    phi_section : float
        Which Poincaré section to use for orbit-based assignment [rad].
    n_heal : int
        Number of new flux surfaces added between S_last_good and r=1.
    S_last_good : float or None
        Upper boundary of the existing reliable mesh.  Defaults to S[-1].
    n_avg : int
        Outer surfaces used for θ* estimation.
    spline_kind : str — 'cubic' or 'linear' boundary spline.
    interp_radial : str — 'linear' or 'cubic' radial interpolation.

    Returns
    -------
    S_out, TET, R_out, Z_out, q_out : extended mesh arrays
    """
    if S_last_good is None:
        S_last_good = float(S[-1])
    idx_last = int(np.searchsorted(S, S_last_good, side='right')) - 1
    idx_last = max(0, min(idx_last, len(S) - 1))

    m = island_chain.m
    n = island_chain.n
    q_mn = m / n

    # ------------------------------------------------------------------
    # Step 1: get anchor positions in sorted θ* order
    # ------------------------------------------------------------------
    if island_chain_orbit is not None:
        R_O, Z_O, theta_O, R_X, Z_X, theta_X = \
            assign_island_chain_pest_angles_from_orbit(
                island_chain_orbit, m, n,
                R_mesh[:idx_last+1], Z_mesh[:idx_last+1], TET,
                Rmaxis, Zmaxis,
                phi_section=phi_section, n_avg=n_avg,
            )
        # Store on island_chain for downstream inspection
        island_chain.pest_theta_O = theta_O
        island_chain.pest_theta_X = theta_X

        # Build boundary curve
        R_bdy, Z_bdy = build_r1_boundary_from_orbit(
            R_O, Z_O, theta_O, R_X, Z_X, theta_X,
            TET, spline_kind=spline_kind,
        )
    else:
        # Legacy: approximate geometric assignment
        theta_O, theta_X = assign_island_chain_pest_angles(
            island_chain,
            R_mesh[:idx_last+1], Z_mesh[:idx_last+1], TET,
            Rmaxis, Zmaxis, n_avg=n_avg,
        )
        R_bdy, Z_bdy = build_r1_boundary(
            island_chain, theta_O, theta_X, TET, spline_kind=spline_kind,
        )

    # ------------------------------------------------------------------
    # Step 2: new radial grid
    # ------------------------------------------------------------------
    S_ref  = float(S[idx_last])
    S_heal = np.linspace(S_ref, 1.0, n_heal + 2)[1:-1]
    S_r1   = np.array([1.0])
    S_new  = np.r_[S_heal, S_r1]

    R_ref = R_mesh[idx_last, :].copy()
    Z_ref = Z_mesh[idx_last, :].copy()

    # ------------------------------------------------------------------
    # Step 3: interpolate new surfaces
    # ------------------------------------------------------------------
    n_new = len(S_new)
    R_new = np.empty((n_new, len(TET)))
    Z_new = np.empty((n_new, len(TET)))

    if interp_radial == 'cubic' and n_new >= 4:
        for j in range(len(TET)):
            cs_R = CubicSpline([S_ref, 1.0], [R_ref[j], R_bdy[j]])
            cs_Z = CubicSpline([S_ref, 1.0], [Z_ref[j], Z_bdy[j]])
            R_new[:, j] = cs_R(S_new)
            Z_new[:, j] = cs_Z(S_new)
    else:
        for i, s in enumerate(S_new):
            alpha = (s - S_ref) / (1.0 - S_ref)
            R_new[i, :] = (1.0 - alpha) * R_ref + alpha * R_bdy
            Z_new[i, :] = (1.0 - alpha) * Z_ref + alpha * Z_bdy

    # ------------------------------------------------------------------
    # Step 4: q profile
    # ------------------------------------------------------------------
    q_ref = float(q_iS[idx_last])
    q_new = q_ref + (q_mn - q_ref) * (S_new - S_ref) / (1.0 - S_ref)

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    S_out = np.r_[S[:idx_last+1], S_new]
    R_out = np.r_[R_mesh[:idx_last+1, :], R_new]
    Z_out = np.r_[Z_mesh[:idx_last+1, :], Z_new]
    q_out = np.r_[q_iS[:idx_last+1], q_new]

    return S_out, TET, R_out, Z_out, q_out
