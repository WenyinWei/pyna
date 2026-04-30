"""Coordinate transforms for cylindrical/Cartesian conversions and
PEST flux-surface coordinates.

Ported and extended from ``mhdpy.coordinate`` (Wenyin Wei, EAST/Tsinghua).
All public functions are pure NumPy and have no tokamak-specific
hard-coded parameters.
"""
from __future__ import annotations

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# Cylindrical ↔ Cartesian
# ---------------------------------------------------------------------------

def rzphi_to_xyz(
    rzphi: ndarray,
    category: str = "coord",
    merge_return: bool = True,
) -> ndarray | tuple[ndarray, ndarray, ndarray]:
    """Convert cylindrical (R, Z, φ) to Cartesian (x, y, z).

    Parameters
    ----------
    rzphi:
        Array whose last axis contains (R, Z, φ) components.
        For ``category='coord'`` the shape is ``(..., 3)``.
        For ``category='field'`` the shape is ``(nR, nZ, nPhi, 3)``
        and φ is sampled uniformly on [0, 2π).
    category:
        ``'coord'`` — transform coordinate positions.
        ``'field'``  — rotate cylindrical vector-field components
                       into Cartesian components on a 3-D grid.
    merge_return:
        If ``True`` return a single stacked array; if ``False``
        return ``(x, y, z)`` separately.

    Returns
    -------
    ndarray or (ndarray, ndarray, ndarray)
        Cartesian representation with the same leading shape as
        ``rzphi`` and last axis of length 3 (when ``merge_return``).
    """
    if category == "coord":
        x = rzphi[..., 0] * np.cos(rzphi[..., 2])
        y = rzphi[..., 0] * np.sin(rzphi[..., 2])
        z = rzphi[..., 1]
    elif category == "field":
        nPhi = rzphi.shape[2]
        Phi = np.linspace(0, 2 * np.pi, nPhi)
        x = (rzphi[..., 0] * np.cos(Phi[None, None, :])
             - rzphi[..., 2] * np.sin(Phi[None, None, :]))
        y = (rzphi[..., 0] * np.sin(Phi[None, None, :])
             + rzphi[..., 2] * np.cos(Phi[None, None, :]))
        z = rzphi[..., 1]
    else:
        raise ValueError(
            "category must be 'coord' or 'field', got {!r}".format(category)
        )

    if merge_return:
        return np.stack((x, y, z), axis=-1)
    return x, y, z


def xyz_to_rzphi(
    xyz: ndarray,
    category: str = "coord",
    merge_return: bool = True,
) -> ndarray | tuple[ndarray, ndarray, ndarray]:
    """Convert Cartesian (x, y, z) to cylindrical (R, Z, φ).

    Parameters
    ----------
    xyz:
        Array whose last axis contains (x, y, z) components.
    category:
        ``'coord'`` — transform coordinate positions.
        ``'field'``  — rotate Cartesian vector-field components
                       into cylindrical components on a 3-D grid.
    merge_return:
        If ``True`` return a single stacked array; if ``False``
        return ``(R, Z, φ)`` separately.

    Returns
    -------
    ndarray or (ndarray, ndarray, ndarray)
    """
    r = np.sqrt(xyz[..., 0] ** 2 + xyz[..., 1] ** 2)
    z = xyz[..., 2]
    if category == "coord":
        phi = np.arctan2(xyz[..., 1], xyz[..., 0])
    elif category == "field":
        nPhi = xyz.shape[2]
        Phi = np.linspace(0, 2 * np.pi, nPhi)
        phi = (xyz[..., 1] * np.cos(Phi[None, None, :])
               + xyz[..., 0] * np.sin(Phi[None, None, :]))
    else:
        raise ValueError(
            "category must be 'coord' or 'field', got {!r}".format(category)
        )

    if merge_return:
        return np.stack((r, z, phi), axis=-1)
    return r, z, phi


def coord_system_change(
    coord_from: str,
    coord_to: str,
    r: ndarray,
    merge_return: bool = True,
) -> ndarray | tuple[ndarray, ndarray, ndarray]:
    """General coordinate-system transform dispatcher.

    Supported pairs: ``'XYZ'`` ↔ ``'RZPhi'``.

    Parameters
    ----------
    coord_from:
        Source coordinate system (``'XYZ'`` or ``'RZPhi'``).
    coord_to:
        Target coordinate system.
    r:
        Coordinate array (last axis is the 3-component vector).
    merge_return:
        Passed through to the underlying transform function.
    """
    if coord_from == coord_to:
        if merge_return:
            return r
        return r[..., 0], r[..., 1], r[..., 2]

    if coord_from == "XYZ":
        if coord_to == "RZPhi":
            return xyz_to_rzphi(r, merge_return=merge_return)
        raise ValueError(f"Transform XYZ → {coord_to!r} not implemented.")
    elif coord_from == "RZPhi":
        if coord_to == "XYZ":
            return rzphi_to_xyz(r, merge_return=merge_return)
        raise ValueError(f"Transform RZPhi → {coord_to!r} not implemented.")
    raise ValueError(f"Unknown source coordinate system {coord_from!r}.")


def coord_mirror(coord: str, r: ndarray, plane: str) -> ndarray:
    """Mirror coordinates about the specified plane.

    Parameters
    ----------
    coord:
        Coordinate system of ``r``: ``'XYZ'`` or ``'RZPhi'``.
    r:
        Coordinate array (last axis is the 3-component vector).
    plane:
        Mirror plane — currently only ``'xy'`` is supported.

    Returns
    -------
    ndarray
        Copy of ``r`` with the appropriate component negated.
    """
    r_new = r.copy()
    if plane == "xy":
        if coord == "XYZ":
            r_new[..., 2] *= -1.0
        elif coord == "RZPhi":
            r_new[..., 1] *= -1.0
        else:
            raise ValueError(f"Unknown coordinate system {coord!r}.")
    else:
        raise ValueError(f"Mirror plane {plane!r} is not implemented.")
    return r_new


# ---------------------------------------------------------------------------
# PEST / flux-surface coordinate utilities
# ---------------------------------------------------------------------------

def Jac_rz2stheta(
    S: ndarray,
    TET: ndarray,
    r_mesh: ndarray,
    z_mesh: ndarray,
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """Compute the Jacobian of the (R, Z) → (S, θ) mapping.

    Uses a centred finite-difference stencil on the 2-D mesh.

    Parameters
    ----------
    S:
        1-D array of flux-surface labels.
    TET:
        1-D array of poloidal angles (PEST angles).
    r_mesh:
        2-D array ``(nS, nTET)`` of R values on the (S, TET) grid.
    z_mesh:
        2-D array ``(nS, nTET)`` of Z values on the (S, TET) grid.

    Returns
    -------
    dRs, dZs, dRtheta, dZtheta : ndarray
        Partial derivatives of (R, Z) with respect to (S, θ),
        shape ``(nS, nTET)``.
    """
    assert S.ndim == 1 and TET.ndim == 1
    assert r_mesh.ndim == 2 and z_mesh.ndim == 2

    ds = np.roll(S, 1) - np.roll(S, -1)
    ds[0], ds[-1] = S[1] - S[0], S[-1] - S[-2]
    dsR = np.roll(r_mesh, 1, axis=0) - np.roll(r_mesh, -1, axis=0)
    dsZ = np.roll(z_mesh, 1, axis=0) - np.roll(z_mesh, -1, axis=0)
    dsR[0, :], dsZ[0, :] = r_mesh[1, :] - r_mesh[0, :], z_mesh[1, :] - z_mesh[0, :]
    dsR[-1, :], dsZ[-1, :] = r_mesh[-1, :] - r_mesh[-2, :], z_mesh[-1, :] - z_mesh[-2, :]
    dsR = dsR / ds[:, None]
    dsZ = dsZ / ds[:, None]

    dtheta = np.roll(TET, 1) - np.roll(TET, -1)
    dtheta[0], dtheta[-1] = TET[1] - TET[0], TET[-1] - TET[-2]
    dthetaR = np.roll(r_mesh, 1, axis=1) - np.roll(r_mesh, -1, axis=1)
    dthetaZ = np.roll(z_mesh, 1, axis=1) - np.roll(z_mesh, -1, axis=1)
    dthetaR[:, 0], dthetaZ[:, 0] = r_mesh[:, 1] - r_mesh[:, 0], z_mesh[:, 1] - z_mesh[:, 0]
    dthetaR[:, -1], dthetaZ[:, -1] = (r_mesh[:, -1] - r_mesh[:, -2],
                                       z_mesh[:, -1] - z_mesh[:, -2])
    dthetaR = dthetaR / dtheta[None, :]
    dthetaZ = dthetaZ / dtheta[None, :]

    det_kl = dsR * dthetaZ - dthetaR * dsZ
    dRs = dthetaZ / det_kl
    dZs = -dthetaR / det_kl
    dRtheta = -dsZ / det_kl
    dZtheta = dsR / det_kl
    return dRs, dZs, dRtheta, dZtheta


def calc_dRZdSTET_mesh(
    S: ndarray,
    TET: ndarray,
    r_mesh: ndarray,
    z_mesh: ndarray,
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """Compute (∂R/∂S, ∂R/∂θ, ∂Z/∂S, ∂Z/∂θ) on the (S, θ) mesh.

    Uses a 5-point centred finite-difference stencil.

    Parameters
    ----------
    S, TET:
        1-D coordinate arrays (equally spaced).
    r_mesh, z_mesh:
        2-D arrays of shape ``(nS, nTET)``.

    Returns
    -------
    dRdS_mesh, dRdTET_mesh, dZdS_mesh, dZdTET_mesh : ndarray
    """
    dTET = TET[1] - TET[0]
    dS = S[1] - S[0]

    def _d5(arr: ndarray, axis: int, h: float) -> ndarray:
        """5-point centred difference, with 2-point endpoints."""
        d = (
            -np.roll(arr, -2, axis=axis)
            + 8 * np.roll(arr, -1, axis=axis)
            - 8 * np.roll(arr,  1, axis=axis)
            +    np.roll(arr,  2, axis=axis)
        ) / (12 * h)
        return d

    dRdTET_mesh = _d5(r_mesh, axis=1, h=dTET)
    dZdTET_mesh = _d5(z_mesh, axis=1, h=dTET)
    dRdS_mesh = _d5(r_mesh, axis=0, h=dS)
    dZdS_mesh = _d5(z_mesh, axis=0, h=dS)

    # Fix boundary rows with 2-point difference
    dRdS_mesh[1, :] = (r_mesh[2, :] - r_mesh[0, :]) / (2 * dS)
    dRdS_mesh[-2, :] = (r_mesh[-1, :] - r_mesh[-3, :]) / (2 * dS)
    dZdS_mesh[0, :] = (z_mesh[1, :] - z_mesh[0, :]) / dS
    dZdS_mesh[-1, :] = (z_mesh[-1, :] - z_mesh[-2, :]) / dS

    return dRdS_mesh, dRdTET_mesh, dZdS_mesh, dZdTET_mesh


def RZ2STET(
    RZ: ndarray,
    S: ndarray,
    TET: ndarray,
    r_mesh: ndarray,
    z_mesh: ndarray,
    dRZdSTET_mesh: tuple | None = None,
) -> ndarray:
    """Convert (R, Z) positions to PEST (S, θ) coordinates.

    Uses a nearest-mesh-point seed followed by a first-order Newton
    correction, as in the original MHDpy implementation.

    Parameters
    ----------
    RZ:
        Array of shape ``(..., 2)`` containing (R, Z) positions.
    S, TET:
        1-D flux-surface and poloidal-angle coordinate arrays.
    r_mesh, z_mesh:
        2-D mesh arrays of shape ``(nS, nTET)``.
    dRZdSTET_mesh:
        Pre-computed derivative tuple from :func:`calc_dRZdSTET_mesh`.
        Pass to avoid recomputation when calling in a loop.

    Returns
    -------
    ndarray
        Array of shape ``(..., 2)`` containing ``(S, θ)`` for each
        input point.
    """
    from scipy.linalg import solve

    RZ_mesh = np.stack((r_mesh, z_mesh), axis=-1)
    STET = np.empty_like(RZ)
    (dRdS_mesh, dRdTET_mesh,
     dZdS_mesh, dZdTET_mesh) = (
        calc_dRZdSTET_mesh(S, TET, r_mesh, z_mesh)
        if dRZdSTET_mesh is None else dRZdSTET_mesh
    )

    for x in np.ndindex(RZ.shape[:-1]):
        bias = np.linalg.norm(RZ_mesh - RZ[x][None, None, :], axis=-1)
        idx = np.unravel_index(np.argmin(bias), bias.shape)
        r0, z0 = r_mesh[idx], z_mesh[idx]
        A = np.array([
            [dRdS_mesh[idx],   dRdTET_mesh[idx]],
            [dZdS_mesh[idx],   dZdTET_mesh[idx]],
        ])
        b_vec = np.array(RZ[x] - [r0, z0])
        ds, dtheta = solve(A, b_vec)
        STET[x] = [S[idx[0]] + ds, TET[idx[1]] + dtheta]

    return STET


def STET2RZ(
    STET: ndarray,
    S: ndarray,
    TET: ndarray,
    r_mesh: ndarray,
    z_mesh: ndarray,
) -> ndarray:
    """Convert PEST (S, θ) coordinates to (R, Z) positions.

    Uses bilinear interpolation on the regular (S, TET) grid.

    Parameters
    ----------
    STET:
        Array of shape ``(..., 2)`` containing ``(S, θ)``.
    S, TET, r_mesh, z_mesh:
        Grid definition as for :func:`RZ2STET`.

    Returns
    -------
    ndarray
        Array of shape ``(..., 2)`` containing ``(R, Z)``.
    """
    from scipy.interpolate import RegularGridInterpolator

    R_interp = RegularGridInterpolator((S, TET), r_mesh)
    Z_interp = RegularGridInterpolator((S, TET), z_mesh)
    grid_R = R_interp(STET)
    grid_Z = Z_interp(STET)
    return np.stack((grid_R, grid_Z), axis=-1)
