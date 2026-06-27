"""Boozer-like coordinate construction from nested toroidal surface grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyna.toroidal._periodic_grid import (
    TWOPI,
    prepare_surface_arrays,
    periodic_derivative,
    periodic_interp,
    strip_field_grid,
)


@dataclass(frozen=True)
class BoozerCoordinateMesh:
    """Boozer-like remap of nested surfaces built from straight-theta grids."""

    R_surf: np.ndarray
    Z_surf: np.ndarray
    theta_B: np.ndarray
    phi_B: np.ndarray
    radial_labels: np.ndarray
    theta_B_of_theta: np.ndarray
    lambda_B: np.ndarray
    jacobian: np.ndarray
    jacobian_B: np.ndarray
    B2_jacobian_B: np.ndarray | None


def _surface_jacobian(
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    radial_labels: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(radial_labels, dtype=np.float64)
    edge_order = 2 if labels.size >= 3 else 1
    dR_dr = np.gradient(R_surf, labels, axis=1, edge_order=edge_order)
    dZ_dr = np.gradient(Z_surf, labels, axis=1, edge_order=edge_order)
    dR_dtheta = periodic_derivative(R_surf, TWOPI, axis=2)
    dZ_dtheta = periodic_derivative(Z_surf, TWOPI, axis=2)
    return R_surf * (dR_dr * dZ_dtheta - dR_dtheta * dZ_dr)


def build_Boozer_coordinates(
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    *,
    radial_labels: np.ndarray | None = None,
    B_R: np.ndarray | None = None,
    B_Z: np.ndarray | None = None,
    B_phi: np.ndarray | None = None,
    B_abs: np.ndarray | None = None,
    n_theta: int | None = None,
    min_weight: float = 1.0e-30,
) -> BoozerCoordinateMesh:
    """Build a Boozer-like theta grid from nested straight-theta surfaces.

    The input surfaces are assumed to have already been constructed from
    Poincare traces and stitched onto a common straight-theta grid.  If magnetic
    field samples are supplied, the remap makes ``B**2 * sqrt(g_B)``
    approximately theta-independent on each ``(r, phi)`` strip.  Without field
    samples the same machinery equalises the geometric Jacobian.
    """

    R, Z, phi, theta = prepare_surface_arrays(R_surf, Z_surf, phi_vals, theta_vals)
    n_phi, n_r, n_theta_in = R.shape
    labels = (
        np.arange(n_r, dtype=np.float64)
        if radial_labels is None
        else np.asarray(radial_labels, dtype=np.float64)
    )
    if labels.shape != (n_r,):
        raise ValueError("radial_labels must have shape (n_r,)")
    if n_r < 2:
        raise ValueError("at least two radial surfaces are required to build Boozer coordinates")
    if np.any(np.diff(labels) <= 0.0):
        raise ValueError("radial_labels must be strictly increasing")

    jac = _surface_jacobian(R, Z, labels)
    abs_jac = np.maximum(np.abs(jac), float(min_weight))

    B2 = None
    if B_abs is not None:
        B2 = strip_field_grid(np.asarray(B_abs, dtype=np.float64), theta_vals, phi_vals) ** 2
    elif B_R is not None or B_Z is not None or B_phi is not None:
        if B_R is None or B_Z is None or B_phi is None:
            raise ValueError("Provide either B_abs or all of B_R, B_Z, and B_phi")
        BR = strip_field_grid(np.asarray(B_R, dtype=np.float64), theta_vals, phi_vals)
        BZ = strip_field_grid(np.asarray(B_Z, dtype=np.float64), theta_vals, phi_vals)
        BPhi = strip_field_grid(np.asarray(B_phi, dtype=np.float64), theta_vals, phi_vals)
        B2 = BR * BR + BZ * BZ + BPhi * BPhi
    if B2 is not None and B2.shape != R.shape:
        raise ValueError("field samples must have the same shape as R_surf/Z_surf")

    weight = abs_jac if B2 is None else np.maximum(B2 * abs_jac, float(min_weight))
    weight_sum = np.sum(weight, axis=2, keepdims=True)
    dtheta_B_dtheta = weight * float(n_theta_in) / np.maximum(weight_sum, float(min_weight))
    theta_B_of_theta = TWOPI * (np.cumsum(weight, axis=2) - weight) / np.maximum(
        weight_sum,
        float(min_weight),
    )
    lambda_B = theta_B_of_theta - theta[np.newaxis, np.newaxis, :]

    n_out = int(n_theta_in if n_theta is None else n_theta)
    theta_B = np.linspace(0.0, TWOPI, n_out, endpoint=False, dtype=np.float64)
    R_B = np.empty((n_phi, n_r, n_out), dtype=np.float64)
    Z_B = np.empty_like(R_B)
    for i_phi in range(n_phi):
        for i_r in range(n_r):
            R_B[i_phi, i_r] = periodic_interp(
                theta_B_of_theta[i_phi, i_r],
                R[i_phi, i_r],
                theta_B,
                TWOPI,
            )
            Z_B[i_phi, i_r] = periodic_interp(
                theta_B_of_theta[i_phi, i_r],
                Z[i_phi, i_r],
                theta_B,
                TWOPI,
            )

    jac_B = jac / np.maximum(dtheta_B_dtheta, float(min_weight))
    B2_jac_B = None if B2 is None else B2 * jac_B
    return BoozerCoordinateMesh(
        R_surf=R_B,
        Z_surf=Z_B,
        theta_B=theta_B,
        phi_B=phi,
        radial_labels=labels.copy(),
        theta_B_of_theta=theta_B_of_theta,
        lambda_B=lambda_B,
        jacobian=jac,
        jacobian_B=jac_B,
        B2_jacobian_B=B2_jac_B,
    )


__all__ = [
    "BoozerCoordinateMesh",
    "build_Boozer_coordinates",
]
