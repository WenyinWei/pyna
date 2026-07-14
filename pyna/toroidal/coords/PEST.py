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
from dataclasses import dataclass, field
from typing import Any, Mapping
import numpy as np
import scipy.interpolate as interp
from scipy.interpolate import interpn, RegularGridInterpolator
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class PestCoordinateProjection:
    """Inverse projection of cylindrical points onto a sampled 3-D PEST mesh."""

    rho: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    residual_distance: np.ndarray
    valid: np.ndarray
    nearest_radial_index: np.ndarray
    nearest_theta_index: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_summary_dict(self) -> dict[str, Any]:
        finite = np.asarray(self.residual_distance)[np.asarray(self.valid)]
        return {
            "schema": "pyna_pest_coordinate_projection_v1",
            "shape": list(np.asarray(self.rho).shape),
            "valid_count": int(np.count_nonzero(self.valid)),
            "valid_fraction": float(np.mean(self.valid)),
            "residual_distance_median": (
                float(np.median(finite)) if finite.size else None
            ),
            "residual_distance_p95": (
                float(np.percentile(finite, 95.0)) if finite.size else None
            ),
            "residual_distance_max": (
                float(np.max(finite)) if finite.size else None
            ),
            "metadata": dict(self.metadata),
        }


def project_cylindrical_points_to_pest(
    pest,
    R,
    Z,
    phi,
    *,
    max_distance: float | None = None,
    local_refinement: bool = True,
    workers: int = -1,
) -> PestCoordinateProjection:
    """Map cylindrical points to ``(rho, theta*, phi)`` on sampled PEST surfaces.

    ``pest`` must expose the :class:`SmoothPestCoordinates` layout
    ``(phi, rho, theta)``.  Each toroidal section is inverted jointly in
    ``(R,Z)`` using a nearest PEST node followed by one local Jacobian step.
    The returned residual is a physical distance in the same units as ``R``
    and ``Z``; callers should set ``max_distance`` for production gating.
    """

    R_surface = np.asarray(pest.R_surf, dtype=np.float64)
    Z_surface = np.asarray(pest.Z_surf, dtype=np.float64)
    rho_values = np.asarray(pest.rho_vals, dtype=np.float64)
    theta_values = np.asarray(pest.theta_vals, dtype=np.float64)
    phi_values = np.asarray(pest.phi_vals, dtype=np.float64)
    if R_surface.shape != Z_surface.shape or R_surface.ndim != 3:
        raise ValueError("PEST surfaces must have shape (phi, rho, theta)")
    if R_surface.shape != (
        phi_values.size,
        rho_values.size,
        theta_values.size,
    ):
        raise ValueError("PEST surface arrays do not match their coordinate axes")
    if rho_values.size < 3 or np.any(np.diff(rho_values) <= 0.0):
        raise ValueError("PEST rho values must contain at least three increasing values")
    if theta_values.size < 4:
        raise ValueError("PEST theta grid must contain at least four points")
    theta_step = 2.0 * np.pi / theta_values.size
    if not np.allclose(
        np.mod(theta_values - theta_values[0], 2.0 * np.pi),
        np.arange(theta_values.size) * theta_step,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError("PEST theta grid must be uniform and endpoint-excluded")
    if max_distance is not None and (
        not np.isfinite(max_distance) or float(max_distance) <= 0.0
    ):
        raise ValueError("max_distance must be finite and positive")

    R_points, Z_points, phi_points = np.broadcast_arrays(
        np.asarray(R, dtype=np.float64),
        np.asarray(Z, dtype=np.float64),
        np.asarray(phi, dtype=np.float64),
    )
    shape = R_points.shape
    flat_R = R_points.ravel()
    flat_Z = Z_points.ravel()
    flat_phi = phi_points.ravel()
    mapped_rho = np.full(flat_R.shape, np.nan, dtype=np.float64)
    mapped_theta = np.full(flat_R.shape, np.nan, dtype=np.float64)
    residual = np.full(flat_R.shape, np.nan, dtype=np.float64)
    nearest_rho = np.full(flat_R.shape, -1, dtype=np.int64)
    nearest_theta = np.full(flat_R.shape, -1, dtype=np.int64)
    full_period = float(getattr(pest, "period", 2.0 * np.pi) or 2.0 * np.pi)
    rho_step = float(np.median(np.diff(rho_values)))

    def periodic_slice(values: np.ndarray, angle: float) -> np.ndarray:
        u = np.mod(float(angle), full_period) * values.shape[0] / full_period
        i0 = int(np.floor(u)) % values.shape[0]
        fraction = u - np.floor(u)
        return (
            (1.0 - fraction) * values[i0]
            + fraction * values[(i0 + 1) % values.shape[0]]
        )

    finite_input = np.isfinite(flat_R) & np.isfinite(flat_Z) & np.isfinite(flat_phi)
    for angle in np.unique(flat_phi[finite_input]):
        selected = finite_input & np.isclose(
            flat_phi, angle, rtol=0.0, atol=1.0e-14
        )
        R_section = periodic_slice(R_surface, float(angle))
        Z_section = periodic_slice(Z_surface, float(angle))
        query = np.column_stack([flat_R[selected], flat_Z[selected]])
        tree = cKDTree(
            np.column_stack([R_section.ravel(), Z_section.ravel()])
        )
        _distance0, nearest = tree.query(query, k=1, workers=int(workers))
        irho = nearest // theta_values.size
        itheta = nearest % theta_values.size
        rho_current = rho_values[irho].copy()
        theta_current = theta_values[itheta].copy()

        def interpolator(values: np.ndarray) -> RegularGridInterpolator:
            extended = np.concatenate([values, values[:, :1]], axis=1)
            theta_extended = np.concatenate(
                [theta_values, [theta_values[0] + 2.0 * np.pi]]
            )
            return RegularGridInterpolator(
                (rho_values, theta_extended),
                extended,
                bounds_error=False,
                fill_value=None,
            )

        R_interp = interpolator(R_section)
        Z_interp = interpolator(Z_section)
        if local_refinement:
            R_rho = np.gradient(
                R_section, rho_values, axis=0, edge_order=2
            )
            Z_rho = np.gradient(
                Z_section, rho_values, axis=0, edge_order=2
            )
            R_theta = (
                np.roll(R_section, -1, axis=1)
                - np.roll(R_section, 1, axis=1)
            ) / (2.0 * theta_step)
            Z_theta = (
                np.roll(Z_section, -1, axis=1)
                - np.roll(Z_section, 1, axis=1)
            ) / (2.0 * theta_step)
            R_rho_interp = interpolator(R_rho)
            Z_rho_interp = interpolator(Z_rho)
            R_theta_interp = interpolator(R_theta)
            Z_theta_interp = interpolator(Z_theta)
            for _ in range(5):
                coordinates = np.column_stack(
                    [rho_current, np.mod(theta_current, 2.0 * np.pi)]
                )
                R_at = R_interp(coordinates)
                Z_at = Z_interp(coordinates)
                Rr = R_rho_interp(coordinates)
                Zr = Z_rho_interp(coordinates)
                Rt = R_theta_interp(coordinates)
                Zt = Z_theta_interp(coordinates)
                delta_R = query[:, 0] - R_at
                delta_Z = query[:, 1] - Z_at
                determinant = Rr * Zt - Rt * Zr
                delta_rho = np.divide(
                    delta_R * Zt - delta_Z * Rt,
                    determinant,
                    out=np.zeros_like(delta_R),
                    where=np.abs(determinant) > 1.0e-12,
                )
                delta_theta = np.divide(
                    Rr * delta_Z - Zr * delta_R,
                    determinant,
                    out=np.zeros_like(delta_R),
                    where=np.abs(determinant) > 1.0e-12,
                )
                rho_current = np.clip(
                    rho_current + np.clip(delta_rho, -rho_step, rho_step),
                    rho_values[0],
                    rho_values[-1],
                )
                theta_current = np.mod(
                    theta_current
                    + np.clip(delta_theta, -theta_step, theta_step),
                    2.0 * np.pi,
                )
        coordinates = np.column_stack([rho_current, theta_current])
        reconstructed_R = R_interp(coordinates)
        reconstructed_Z = Z_interp(coordinates)
        section_residual = np.hypot(
            query[:, 0] - reconstructed_R,
            query[:, 1] - reconstructed_Z,
        )
        target = np.flatnonzero(selected)
        mapped_rho[target] = rho_current
        mapped_theta[target] = theta_current
        residual[target] = section_residual
        nearest_rho[target] = irho
        nearest_theta[target] = itheta

    valid = (
        finite_input
        & np.isfinite(mapped_rho)
        & (mapped_rho >= rho_values[0])
        & (mapped_rho <= rho_values[-1])
    )
    if max_distance is not None:
        valid &= residual <= float(max_distance)
    metadata = {
        "method": "section_KD_tree_plus_local_PEST_Jacobian",
        "surface_layout": "phi_rho_theta",
        "local_refinement": bool(local_refinement),
        "max_distance": None if max_distance is None else float(max_distance),
        "source": getattr(pest, "source", None),
    }
    return PestCoordinateProjection(
        rho=mapped_rho.reshape(shape),
        theta=mapped_theta.reshape(shape),
        phi=np.mod(phi_points, full_period),
        residual_distance=residual.reshape(shape),
        valid=valid.reshape(shape),
        nearest_radial_index=nearest_rho.reshape(shape),
        nearest_theta_index=nearest_theta.reshape(shape),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------

def build_PEST_mesh(
        R, Z, BR0, BZ0, BPhi0, psi_norm, Rmaxis, Zmaxis,
        ns=60, ntheta=181, bdry=None,
        solve_ivp_kwarg=None,
        dt=0.01,
        max_q=30):
    """Build a PEST (straight field-line) coordinate mesh (S, θ*, φ).

    The algorithm seeds field lines from the midplane (Z = Zmaxis), at
    uniformly spaced radial positions from the magnetic axis to the LCFS,
    and traces them with :class:`pyna.flt.FieldLineTracer` until they
    return to the midplane.  Each field-line traces one iso-S surface.
    The PEST poloidal angle θ* is then proportional to the toroidal angle
    traversed along the field line, so that q(S) = Δφ / (2π) is the
    safety factor.

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
        .. deprecated::
            Ignored.  :class:`pyna.flt.FieldLineTracer` is now the sole
            entry point for field-line tracing.  Pass ``dt`` to control the
            arc-length step size instead.
    dt : float, optional
        Arc-length step size for :class:`pyna.flt.FieldLineTracer`.
        Default 0.01 (units match R, Z — typically metres).
    max_q : float, optional
        Expected maximum safety factor.  Sets the integration arc-length
        upper bound to ``max_q * 2π * LCFS_R``.  Increase for devices with
        very high edge q.  Default 30.

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
    if solve_ivp_kwarg is not None:
        warnings.warn(
            "solve_ivp_kwarg is deprecated and ignored; FieldLineTracer is "
            "now the sole entry point for field-line tracing.  "
            "Pass dt= to control the arc-length step size instead.",
            DeprecationWarning,
            stacklevel=2,
        )

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

    # --- Build FieldLineTracer field function (arc-length parameterisation) ---
    # f([R, Z, φ]) → [dR/dl, dZ/dl, dφ/dl]  (unit tangent in arc-length).
    # We use RegularGridInterpolators for each component so that |B| can be
    # computed and the unit tangent normalised correctly.
    from pyna.flt import FieldLineTracer
    _BR_rgi   = RegularGridInterpolator((R, Z), BR0,   method='linear',
                                        bounds_error=False, fill_value=None)
    _BZ_rgi   = RegularGridInterpolator((R, Z), BZ0,   method='linear',
                                        bounds_error=False, fill_value=None)
    _BPhi_rgi = RegularGridInterpolator((R, Z), BPhi0, method='linear',
                                        bounds_error=False, fill_value=None)

    # Determine the sign of Bφ at the LCFS midplane so that the integration
    # always proceeds in the direction of increasing φ (ensuring q > 0).
    _bphi_sign = 1.0 if float(_BPhi_rgi([[LCFS_R, LCFS_Z]])[0]) >= 0.0 else -1.0

    def _field_func(rzphi):
        """Unit tangent vector for arc-length field-line tracing.

        Always oriented so that dφ/dl > 0 (φ is monotonically increasing
        along the trajectory, giving q > 0 by construction).
        """
        r, z = rzphi[0], rzphi[1]
        pt = [[r, z]]
        br   = float(_BR_rgi(pt)[0])
        bz   = float(_BZ_rgi(pt)[0])
        bphi = float(_BPhi_rgi(pt)[0])
        bmag = np.sqrt(br**2 + bz**2 + bphi**2) + 1e-30
        s = _bphi_sign / bmag
        return [s * br, s * bz, s * bphi / (r + 1e-30)]

    tracer = FieldLineTracer(_field_func, dt=dt)

    # Upper-bound arc-length: covers safety factors up to max_q.
    # We use a step-wise trace with early exit at midplane return to avoid
    # integrating the full upper-bound length when q is small.
    t_max_flt = max_q * 2.0 * np.pi * LCFS_R

    # Minimum number of steps to skip before looking for the midplane return
    # (avoids re-triggering at the seed itself — mirrors the old t < 0.05 guard).
    skip_pts = max(5, int(0.05 * LCFS_R / dt))

    # Chunk size: trace in blocks of ~half-poloidal-turn; stop as soon as the
    # midplane return is detected.  This avoids allocating max_q full orbits.
    chunk_steps = max(50, int(np.pi * LCFS_R / dt))  # ≈ half toroidal turn
    chunk_len   = chunk_steps * dt

    # --- Field-line tracing via FieldLineTracer (with early midplane stop) ---
    # Each trace returns ndarray (N, 3) with columns [R, Z, φ].
    fcflts_trajs = []
    for seed in fcflts_seeds:
        start = np.array([seed[0], seed[1], 0.0])  # φ₀ = 0
        traj_chunks = []
        total_pts = 0
        found = False

        while total_pts * dt < t_max_flt:
            chunk = tracer.trace(start, chunk_len)
            if total_pts == 0:
                traj_chunks.append(chunk)
            else:
                traj_chunks.append(chunk[1:])  # avoid duplicate start point
            total_pts += len(chunk) - 1

            # Check for midplane return after skip_pts
            so_far = np.concatenate(traj_chunks, axis=0)
            if len(so_far) > skip_pts + 1:
                Z_rel = so_far[skip_pts:, 1] - Zmaxis
                cross = np.where((Z_rel[:-1] <= 0.0) & (Z_rel[1:] > 0.0))[0]
                if len(cross) > 0:
                    found = True
                    break

            start = chunk[-1].copy()  # continue from last point

        fcflts_trajs.append(np.concatenate(traj_chunks, axis=0))

    # --- Safety factor q and midplane-crossing detection ---
    # Find the first return to Z = Zmaxis (Z_rel: ≤0 → >0) after skip_pts.
    q_iS = np.empty(ns)
    q_iS[0] = np.nan

    for i, traj in enumerate(fcflts_trajs):
        n_pts = len(traj)
        if n_pts <= skip_pts + 1:
            warnings.warn(
                f"Field-line trace for iS={i+1} terminated before returning "
                "to the midplane — trajectory too short.  "
                "Try increasing max_q or decreasing dt.",
                RuntimeWarning,
                stacklevel=2,
            )
            q_iS[i + 1] = np.nan
            continue

        Z_rel = traj[skip_pts:, 1] - Zmaxis
        cross = np.where((Z_rel[:-1] <= 0.0) & (Z_rel[1:] > 0.0))[0]
        if len(cross) == 0:
            warnings.warn(
                f"No midplane return detected for iS={i+1}.  "
                "Try increasing max_q or decreasing dt.",
                RuntimeWarning,
                stacklevel=2,
            )
            q_iS[i + 1] = np.nan
            continue

        # Linear interpolation to find precise φ at the crossing.
        ci = cross[0] + skip_pts          # index in full trajectory
        dZ = traj[ci + 1, 1] - traj[ci, 1]
        frac = (Zmaxis - traj[ci, 1]) / dZ if abs(dZ) > 1e-30 else 0.0
        phi_cross = traj[ci, 2] + frac * (traj[ci + 1, 2] - traj[ci, 2])
        q_iS[i + 1] = phi_cross / (2.0 * np.pi)

    # --- Build (R, Z) mesh on PEST grid ---
    TET = np.linspace(0.0, 2 * np.pi, endpoint=True, num=ntheta)
    R_mesh[0, :] = Rmaxis
    Z_mesh[0, :] = Zmaxis

    for i, traj in enumerate(fcflts_trajs):
        if np.isnan(q_iS[i + 1]):
            R_mesh[i + 1, :] = np.nan
            Z_mesh[i + 1, :] = np.nan
            continue

        # φ-parameterised interpolation along the traced trajectory.
        # phi is monotonically increasing (ensured by _bphi_sign).
        phi_traj = traj[:, 2]
        phi_targets = q_iS[i + 1] * TET  # 0 → phi_cross
        R_mesh[i + 1, :] = np.interp(phi_targets, phi_traj, traj[:, 0])
        Z_mesh[i + 1, :] = np.interp(phi_targets, phi_traj, traj[:, 1])

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
