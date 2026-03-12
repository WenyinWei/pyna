"""
PEST (Straight Field Line) coordinate system for tokamak equilibria.

This module provides tools for constructing PEST coordinates (S, θ*, φ) from
a numerical MHD equilibrium, and computing the associated metric tensors and
field components.

PEST coordinates:
    S    = sqrt(ψ_norm)  — radial-like coordinate (square root of normalized flux)
    θ*   = PEST poloidal angle (chosen so that B · ∇θ* / B · ∇φ = q(S) = const on flux surface)
    φ    = standard toroidal angle

References:
    J. Manickam et al., PEST code. Princeton Plasma Physics Laboratory.
"""

import warnings
import numpy as np
from scipy.integrate import solve_ivp
import scipy.interpolate as interp
from scipy.interpolate import interpn


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------

def build_PEST_mesh(
        R, Z, BR0, BZ0, BPhi0, psi_norm, Rmaxis, Zmaxis,
        ns=60, ntheta=181, bdry=None,
        solve_ivp_kwarg=None):
    """Build a PEST (straight field-line) coordinate mesh (S, θ*, φ).

    The algorithm seeds field lines from the midplane (Z = Zmaxis), at
    uniformly spaced radial positions from the magnetic axis to the LCFS,
    and integrates them in φ until they return to the midplane.  Each
    field-line traces one iso-S surface.  The PEST poloidal angle θ* is then
    proportional to the toroidal angle traversed along the field line, so
    that q(S) = Δφ / (2π) is the safety factor.

    Parameters
    ----------
    R, Z : 1D array_like
        Radial and vertical grid coordinates.
    BR0, BZ0, BPhi0 : 2D array_like, shape (nR, nZ)
        Background equilibrium field components on the (R, Z) grid.
    psi_norm : 2D array_like, shape (nR, nZ)
        Normalised poloidal flux ψ_norm (0 on axis, 1 on LCFS).
    Rmaxis, Zmaxis : float
        Magnetic axis position.
    ns : int, optional
        Number of radial (S) surfaces.  Default 60.
    ntheta : int, optional
        Number of poloidal (θ*) points per surface.  Default 181.
    bdry : array_like of shape (N, 2), optional
        (R, Z) boundary polygon.  If given the LCFS intersection is found
        via the *intersect* package rather than a spline root.
    solve_ivp_kwarg : dict, optional
        Keyword arguments forwarded to ``scipy.integrate.solve_ivp``.
        Defaults to ``{"method": "DOP853", "rtol": 1e-5, "atol": 1e-8}``.

    Returns
    -------
    S : ndarray, shape (ns,)
        Radial PEST coordinate values (S[0] = 0 on axis).
    TET : ndarray, shape (ntheta,)
        Poloidal PEST angle θ* from 0 to 2π (inclusive).
    R_mesh, Z_mesh : ndarray, shape (ns, ntheta)
        Cylindrical coordinates of the (S, θ*) mesh.
    q_iS : ndarray, shape (ns,)
        Safety factor q(S) for each surface (q[0] = NaN for axis).
    """
    if solve_ivp_kwarg is None:
        solve_ivp_kwarg = {"method": "DOP853", "rtol": 1e-5, "atol": 1e-8}

    R = np.asarray(R)
    Z = np.asarray(Z)

    R_mesh, Z_mesh = [np.empty((ns, ntheta)) for _ in range(2)]

    # --- Find LCFS intersection with midplane ---
    test_horizon_R = np.linspace(Rmaxis + 0.05, max(R) - 0.05, num=100)
    test_horizon_Z = Zmaxis + np.zeros_like(test_horizon_R)

    if bdry is None:
        from scipy.interpolate import RegularGridInterpolator, UnivariateSpline
        psi_interp = RegularGridInterpolator((R, Z), psi_norm)
        psi_on_midplane = psi_interp(
            np.stack((test_horizon_R, test_horizon_Z), axis=1))
        LCFS_R = UnivariateSpline(test_horizon_R, psi_on_midplane - 1.0).roots()[0]
        LCFS_Z = test_horizon_Z[0]
    else:
        from intersect import intersection
        LCFS_R, LCFS_Z = intersection(
            test_horizon_R, test_horizon_Z, bdry[:, 0], bdry[:, 1])
        LCFS_R, LCFS_Z = LCFS_R[0], LCFS_Z[0]

    # --- Seed points on midplane ---
    seed_R = np.linspace(Rmaxis, LCFS_R, endpoint=False, num=ns)[1:]
    fcflts_seeds = [np.array([r, Zmaxis]) for r in seed_R]

    # --- Build field-line RHS (integrate in φ, i.e. dR/dφ = R·BR/Bφ) ---
    Rg, Zg = np.meshgrid(R, Z)
    B_RZ = interp.LinearNDInterpolator(
        np.column_stack((Rg.T.flatten(), Zg.T.flatten())),
        np.column_stack(((R[:, None] * BR0 / BPhi0).flatten(),
                         (R[:, None] * BZ0 / BPhi0).flatten())))

    def hit_init_horizon(t, y):
        """Event: field line returns to midplane Z = Zmaxis (from below)."""
        if t < 0.05:
            return 0.05
        return y[1] - Zmaxis

    hit_init_horizon.terminal = True
    hit_init_horizon.direction = 1.0

    # Determine direction of integration
    if B_RZ(np.array([[LCFS_R, LCFS_Z]]))[0, 1] > 0:
        rhs = lambda t, y: B_RZ(y)[0, :]
        q_pos = True
    else:
        rhs = lambda t, y: -B_RZ(y)[0, :]
        q_pos = False

    # --- Field-line integration ---
    fcflts_sols = []
    for seed in fcflts_seeds:
        sol = solve_ivp(
            fun=rhs,
            t_span=[0.0, 3200 * np.pi],
            y0=seed,
            events=hit_init_horizon,
            dense_output=True,
            **solve_ivp_kwarg,
        )
        fcflts_sols.append(sol)

    # --- Safety factor q ---
    q_iS = np.empty(ns)
    q_iS[0] = np.nan
    for i, sol in enumerate(fcflts_sols):
        q_iS[i + 1] = sol.t_events[0][0] / (2 * np.pi)

    # --- Build (R, Z) mesh on PEST grid ---
    TET = np.linspace(0.0, 2 * np.pi, endpoint=True, num=ntheta)
    R_mesh[0, :] = Rmaxis
    Z_mesh[0, :] = Zmaxis

    for i, sol in enumerate(fcflts_sols):
        RZ_temp = sol.sol(q_iS[i + 1] * TET)
        R_mesh[i + 1, :] = RZ_temp[0, :]
        Z_mesh[i + 1, :] = RZ_temp[1, :]

    # --- Compute S = sqrt(ψ_norm) ---
    from scipy.interpolate import RegularGridInterpolator as _RGI
    _psi_interp_S = _RGI((R, Z), psi_norm, method='linear', bounds_error=False, fill_value=None)
    S = np.empty(ns)
    S[0] = 0.0
    for i, seed in enumerate(fcflts_seeds):
        psi_val = _psi_interp_S([[seed[0], seed[1]]])[0]
        if psi_val > 0:
            S[i + 1] = np.sqrt(psi_val)
        else:
            S[i + 1] = 0.0
            warnings.warn(
                f"sqrt(psi_norm) at iS={i+1} is non-positive — the seed may be "
                "too close to the magnetic axis.  "
                "Consider using S[1:], R_mesh[1:], Z_mesh[1:] as a workaround.",
                RuntimeWarning,
                stacklevel=2,
            )

    if not q_pos:
        q_iS *= -1.0

    return S, TET, R_mesh, Z_mesh, q_iS


def RZmesh_isoSTET(*args, **kwargs):
    """Deprecated alias for :func:`build_PEST_mesh`.

    .. deprecated::
        Use :func:`build_PEST_mesh` instead.
    """
    warnings.warn(
        "RZmesh_isoSTET is deprecated; use build_PEST_mesh instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_PEST_mesh(*args, **kwargs)


# ---------------------------------------------------------------------------
# Metric tensors
# ---------------------------------------------------------------------------

def g_i_g__i_from_STET_mesh(S, TET, R_mesh, Z_mesh):
    """Compute covariant basis vectors g_i and contravariant (dual) basis g^i.

    Given a PEST mesh (S, θ*, φ) parametrised by the cylindrical (R, Z)
    coordinates on each iso-S surface, this function evaluates the tangent
    basis vectors and their duals using central-difference numerical
    differentiation.

    Tangent (covariant) basis:
        g_1 = ∂_S   r = (∂R/∂S,   ∂Z/∂S)   in the (R, Z) plane
        g_2 = ∂_θ*  r = (∂R/∂θ*,  ∂Z/∂θ*)  in the (R, Z) plane
        g_3 = ∂_φ   r = R  ê_φ              (toroidal direction)

    Dual (contravariant) basis via the triple-product formula:
        g^1 = ∇S    = (g_2 × g_3) / [g_1, g_2, g_3]
        g^2 = ∇θ*   = (g_3 × g_1) / [g_1, g_2, g_3]
        g^3 = ∇φ    = (g_1 × g_2) / [g_1, g_2, g_3]

    In axisymmetry the poloidal cross-products reduce to 2-D rotations and
    [g_1, g_2, g_3] = sqrt(g) = (g_1 × g_2) · g_3 = -(g_1×g_2)_φ · R.

    Parameters
    ----------
    S : ndarray, shape (ns,)
    TET : ndarray, shape (ntheta,)
    R_mesh, Z_mesh : ndarray, shape (ns, ntheta)

    Returns
    -------
    g_1, g_2 : ndarray, shape (ns, ntheta, 2)
        Covariant basis in the (R, Z) plane.  The last axis is [R, Z].
        Boundary rows/columns are NaN (g_1) or periodic-wrapped (g_2).
    g_3 : callable
        ``g_3(R_arr)`` returns the magnitude of the toroidal basis vector,
        which equals R (the cylindrical radius).
    g__1, g__2 : ndarray, shape (ns, ntheta, 2)
        Contravariant basis in the (R, Z) plane.
    g__3 : callable
        ``g__3(R_arr)`` returns |g^3| = 1/R.
    """
    ns, ntheta = len(S), len(TET)

    # --- Covariant basis ---
    g_1 = np.empty((ns, ntheta, 2))  # [iS, itheta, R/Z]
    # Central differences in S (interior only)
    g_1[1:-1, :, 0] = (R_mesh[2:, :] - R_mesh[:-2, :]) / (S[2:] - S[:-2])[:, None]
    g_1[1:-1, :, 1] = (Z_mesh[2:, :] - Z_mesh[:-2, :]) / (S[2:] - S[:-2])[:, None]
    g_1[0, :, :] = np.nan  # undefined at the magnetic axis
    g_1[-1, :, :] = np.nan  # undefined at the LCFS boundary

    g_2 = np.empty((ns, ntheta, 2))  # [iS, itheta, R/Z]
    # Central differences in θ* (interior)
    g_2[:, 1:-1, 0] = (R_mesh[:, 2:] - R_mesh[:, :-2]) / (TET[2:] - TET[:-2])[None, :]
    g_2[:, 1:-1, 1] = (Z_mesh[:, 2:] - Z_mesh[:, :-2]) / (TET[2:] - TET[:-2])[None, :]
    # Periodic boundary: θ*=0 and θ*=2π are the same point
    dTET_wrap = -(TET[-2] - TET[1] - 2 * np.pi)
    g_2[:, 0, 0] = g_2[:, -1, 0] = (R_mesh[:, 1] - R_mesh[:, -2]) / dTET_wrap
    g_2[:, 0, 1] = g_2[:, -1, 1] = (Z_mesh[:, 1] - Z_mesh[:, -2]) / dTET_wrap

    # g_3 = R ê_φ  (magnitude only, since φ is the cyclic direction)
    g_3 = lambda R_arr: R_arr

    # --- Jacobian sqrt(g) = [g_1, g_2, g_3] = -(g_1 × g_2)_φ · R ---
    # In the (R, Z) plane: (g_1 × g_2)_φ = g_1R·g_2Z - g_2R·g_1Z
    g_123_prod = -(g_1[:, :, 0] * g_2[:, :, 1]
                   - g_2[:, :, 0] * g_1[:, :, 1]) * g_3(R_mesh)

    # --- Contravariant basis via cross-product formulae ---
    # g^1 = (g_2 × g_3) / sqrt(g)
    # In (R, Z):  g_2 × g_3 = R·(−g_2Z, g_2R)  (CCW rotation of g_2)
    g__1 = np.empty((ns, ntheta, 2))
    g__1[:, :, 0] = -g_2[:, :, 1]
    g__1[:, :, 1] =  g_2[:, :, 0]
    g__1 *= (g_3(R_mesh) / g_123_prod)[:, :, None]

    # g^2 = (g_3 × g_1) / sqrt(g)
    # In (R, Z):  g_3 × g_1 = R·(g_1Z, −g_1R)  (CW rotation of g_1)
    g__2 = np.empty((ns, ntheta, 2))
    g__2[:, :, 0] =  g_1[:, :, 1]
    g__2[:, :, 1] = -g_1[:, :, 0]
    g__2 *= (g_3(R_mesh) / g_123_prod)[:, :, None]

    # g^3 = ∇φ = ê_φ / R  →  |g^3| = 1/R
    g__3 = lambda R_arr: 1.0 / R_arr

    return g_1, g_2, g_3, g__1, g__2, g__3


# ---------------------------------------------------------------------------
# Field component projections
# ---------------------------------------------------------------------------

def counter_comp_of_a_field(B_pert, S, TET, R_mesh, Z_mesh):
    """Project a 3-D cylindrical vector field onto contravariant PEST components.

    Computes B^i such that  **B** = B^1 g_1 + B^2 g_2 + B^3 g_3, where
    B^i = **B** · g^i.

    Parameters
    ----------
    B_pert : CylindricalGridAxiVectorField or compatible
        The vector field to project.  Must expose attributes
        ``.R``, ``.Z``, ``.Phi``, ``.BR``, ``.BZ``, ``.BPhi``
        where BR, BZ, BPhi have shape (nR, nZ, nPhi).
    S : ndarray, shape (ns,)
    TET : ndarray, shape (ntheta,)
    R_mesh, Z_mesh : ndarray, shape (ns, ntheta)

    Returns
    -------
    B__1, B__2, B__3 : ndarray, shape (ns, ntheta, nPhi)
        Contravariant components B^S, B^θ*, B^φ.
    """
    g_1, g_2, g_3, g__1, g__2, g__3 = g_i_g__i_from_STET_mesh(S, TET, R_mesh, Z_mesh)

    R, Z, Phi = B_pert.R, B_pert.Z, B_pert.Phi
    BR_pert, BZ_pert, BPhi_pert = B_pert.BR, B_pert.BZ, B_pert.BPhi
    ns, ntheta, nPhi = len(S), len(TET), BPhi_pert.shape[2]

    # Interpolate field onto the (S, θ*, φ) mesh
    rzPhi_mesh = np.empty((ns, ntheta, nPhi, 3))
    rzPhi_mesh[:, :, :, 0] = R_mesh[:, :, None]
    rzPhi_mesh[:, :, :, 1] = Z_mesh[:, :, None]
    rzPhi_mesh[:, :, :, 2] = Phi[None, None, :]

    points = (R, Z, Phi)
    BR_on_mesh   = interpn(points, BR_pert,   rzPhi_mesh)
    BZ_on_mesh   = interpn(points, BZ_pert,   rzPhi_mesh)
    BPhi_on_mesh = interpn(points, BPhi_pert, rzPhi_mesh)

    # Project: B^i = B · g^i
    B__1 = BR_on_mesh * g__1[:, :, 0][:, :, None] + BZ_on_mesh * g__1[:, :, 1][:, :, None]
    B__2 = BR_on_mesh * g__2[:, :, 0][:, :, None] + BZ_on_mesh * g__2[:, :, 1][:, :, None]
    B__3 = BPhi_on_mesh * g__3(R_mesh)[:, :, None]

    return B__1, B__2, B__3


def co_comp_of_a_field(B_pert, S, TET, R_mesh, Z_mesh):
    """Project a 3-D cylindrical vector field onto covariant PEST components.

    Computes B_i such that  **B** = B_1 g^1 + B_2 g^2 + B_3 g^3, where
    B_i = **B** · g_i.

    Parameters
    ----------
    B_pert : CylindricalGridAxiVectorField or compatible
        The vector field to project.  See :func:`counter_comp_of_a_field`.
    S : ndarray, shape (ns,)
    TET : ndarray, shape (ntheta,)
    R_mesh, Z_mesh : ndarray, shape (ns, ntheta)

    Returns
    -------
    B_1, B_2, B_3 : ndarray, shape (ns, ntheta, nPhi)
        Covariant components B_S, B_θ*, B_φ.
    """
    g_1, g_2, g_3, g__1, g__2, g__3 = g_i_g__i_from_STET_mesh(S, TET, R_mesh, Z_mesh)

    R, Z, Phi = B_pert.R, B_pert.Z, B_pert.Phi
    BR_pert, BZ_pert, BPhi_pert = B_pert.BR, B_pert.BZ, B_pert.BPhi
    ns, ntheta, nPhi = len(S), len(TET), BPhi_pert.shape[2]

    rzPhi_mesh = np.empty((ns, ntheta, nPhi, 3))
    rzPhi_mesh[:, :, :, 0] = R_mesh[:, :, None]
    rzPhi_mesh[:, :, :, 1] = Z_mesh[:, :, None]
    rzPhi_mesh[:, :, :, 2] = Phi[None, None, :]

    points = (R, Z, Phi)
    BR_on_mesh   = interpn(points, BR_pert,   rzPhi_mesh)
    BZ_on_mesh   = interpn(points, BZ_pert,   rzPhi_mesh)
    BPhi_on_mesh = interpn(points, BPhi_pert, rzPhi_mesh)

    # Project: B_i = B · g_i
    B_1 = BR_on_mesh * g_1[:, :, 0][:, :, None] + BZ_on_mesh * g_1[:, :, 1][:, :, None]
    B_2 = BR_on_mesh * g_2[:, :, 0][:, :, None] + BZ_on_mesh * g_2[:, :, 1][:, :, None]
    B_3 = BPhi_on_mesh * g_3(R_mesh)[:, :, None]

    return B_1, B_2, B_3
