"""External coil systems for stellarator island control.

Provides:
- CoilSet: a collection of current-carrying coils
- StellaratorControlCoils: standard external island control coil geometry
- Biot_Savart_field: numerical Biot-Savart computation on a cylindrical grid

The Biot-Savart integration is parallelized over coils and grid points.
"""
from __future__ import annotations

import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os


MU0_OVER_4PI = 1e-7  # mu0 / (4*pi)  [T·m/A]


def Biot_Savart_field(coil_pts, coil_current, R_grid, Z_grid, Phi_grid=None):
    """Compute magnetic field from a current-carrying coil via Biot-Savart.

    Parameters
    ----------
    coil_pts : ndarray, shape (N, 3)
        XYZ coordinates of coil points (meters).
    coil_current : float
        Current in amperes (positive = right-hand rule along coil_pts direction).
    R_grid, Z_grid : ndarray, shape (nR, nZ) or 1D
        Cylindrical coordinate grid.
    Phi_grid : ndarray or None
        If None, compute on the (R, Z) plane at phi=0.
        If provided, compute on the full 3D (R, Z, Phi) grid.

    Returns
    -------
    BR, BZ, BPhi : ndarray
        Field components on the grid (Tesla).
    """
    coil_pts = np.asarray(coil_pts, dtype=float)
    R_grid = np.asarray(R_grid, dtype=float)
    Z_grid = np.asarray(Z_grid, dtype=float)

    if Phi_grid is None:
        # Evaluate at phi=0: X = R, Y = 0, Z = Z
        X_grid = R_grid
        Y_grid = np.zeros_like(R_grid)
        Z_grid_3d = Z_grid
        shape = R_grid.shape
    else:
        Phi_grid = np.asarray(Phi_grid, dtype=float)
        X_grid = R_grid * np.cos(Phi_grid)
        Y_grid = R_grid * np.sin(Phi_grid)
        Z_grid_3d = Z_grid
        shape = R_grid.shape

    # Flatten grid points
    Xf = X_grid.ravel()
    Yf = Y_grid.ravel()
    Zf = Z_grid_3d.ravel()
    Rf = R_grid.ravel()
    if Phi_grid is not None:
        Phif = Phi_grid.ravel()
    else:
        Phif = np.zeros_like(Rf)

    n_pts = len(Xf)
    Bx = np.zeros(n_pts)
    By = np.zeros(n_pts)
    Bz_out = np.zeros(n_pts)

    # Segment-wise Biot-Savart
    N_seg = len(coil_pts) - 1
    for i in range(N_seg):
        P1 = coil_pts[i]
        P2 = coil_pts[i + 1]
        dl = P2 - P1  # (3,)
        mid = 0.5 * (P1 + P2)

        # r = field_point - mid
        rx = Xf - mid[0]
        ry = Yf - mid[1]
        rz = Zf - mid[2]
        r_mag = np.sqrt(rx**2 + ry**2 + rz**2) + 1e-30

        # dl x r
        cx = dl[1] * rz - dl[2] * ry
        cy = dl[2] * rx - dl[0] * rz
        cz = dl[0] * ry - dl[1] * rx

        coeff = MU0_OVER_4PI * coil_current / r_mag**3
        Bx += coeff * cx
        By += coeff * cy
        Bz_out += coeff * cz

    # Convert Bx, By, Bz -> BR, BPhi, BZ
    cos_phi = np.cos(Phif)
    sin_phi = np.sin(Phif)
    BR = Bx * cos_phi + By * sin_phi
    BPhi = -Bx * sin_phi + By * cos_phi
    BZ_cyl = Bz_out

    return BR.reshape(shape), BZ_cyl.reshape(shape), BPhi.reshape(shape)


class CoilSet:
    """Collection of coils with individual currents.

    Each coil is defined by a list of 3D points (forming a closed loop).
    """

    def __init__(self):
        self.coils = []  # list of (pts_xyz, current)

    def add_coil(self, pts_xyz, current=1.0):
        """Add a coil defined by XYZ points."""
        self.coils.append((np.asarray(pts_xyz, dtype=float), float(current)))

    def field_on_grid(self, R_1d, Z_1d, phi_1d=None, n_workers=None):
        """Compute total field from all coils on a cylindrical grid.

        If phi_1d is None: compute on (R,Z) at phi=0.
        If phi_1d provided: full 3D computation (R, Z, Phi grid).

        Returns
        -------
        BR, BZ, BPhi : ndarray
            Shape (nR, nZ) if phi_1d is None, else (nR, nZ, nPhi).
        """
        R_1d = np.asarray(R_1d, dtype=float)
        Z_1d = np.asarray(Z_1d, dtype=float)

        if phi_1d is None:
            RR, ZZ = np.meshgrid(R_1d, Z_1d, indexing='ij')
            PP = None
        else:
            phi_1d = np.asarray(phi_1d, dtype=float)
            RR, ZZ, PP = np.meshgrid(R_1d, Z_1d, phi_1d, indexing='ij')

        shape = RR.shape
        BR_tot = np.zeros(shape)
        BZ_tot = np.zeros(shape)
        BP_tot = np.zeros(shape)

        if n_workers is None:
            n_workers = min(len(self.coils), os.cpu_count() or 1)

        def _compute_coil(coil_item):
            pts, current = coil_item
            return Biot_Savart_field(pts, current, RR, ZZ, PP)

        if n_workers > 1 and len(self.coils) > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(_compute_coil, c) for c in self.coils]
                for fut in futures:
                    br, bz, bp = fut.result()
                    BR_tot += br
                    BZ_tot += bz
                    BP_tot += bp
        else:
            for c in self.coils:
                br, bz, bp = _compute_coil(c)
                BR_tot += br
                BZ_tot += bz
                BP_tot += bp

        return BR_tot, BZ_tot, BP_tot

    def scale_currents(self, scale):
        """Scale all currents by a factor."""
        self.coils = [(pts, I * scale) for pts, I in self.coils]

    def set_current(self, idx, current):
        self.coils[idx] = (self.coils[idx][0], float(current))

    def get_currents(self):
        """Return array of all currents."""
        return np.array([I for _, I in self.coils])

    def set_currents(self, currents):
        """Set all currents from an array."""
        currents = np.asarray(currents)
        self.coils = [(pts, float(currents[i])) for i, (pts, _) in enumerate(self.coils)]

    def __len__(self):
        return len(self.coils)

    def __repr__(self):
        return f"CoilSet({len(self.coils)} coils)"


class StellaratorControlCoils(CoilSet):
    """Standard external island control coil array for a stellarator.

    Creates a set of saddle coils around the torus designed to produce
    a resonant (m_target, n_target) perturbation.

    The coils are phased as I_k = I0 * cos(n_target * phi_k) to produce
    a dominant (m=0, n=n_target) or combined (m_target, n_target) response.

    Parameters
    ----------
    R0, r_coil : float
        Major radius and coil minor radius (slightly outside plasma, m).
    N_coils : int
        Total number of saddle coils around the torus.
    m_target, n_target : int
        Target resonant mode to control.
    I0 : float
        Reference current (A).
    """

    def __init__(self, R0, r_coil, N_coils, m_target, n_target, I0=1.0):
        super().__init__()
        self.R0 = float(R0)
        self.r_coil = float(r_coil)
        self.m_target = int(m_target)
        self.n_target = int(n_target)
        self.I0 = float(I0)
        self._N_coils = N_coils
        self._build_coils(N_coils)

    def _build_coils(self, N_coils):
        """Build saddle coil geometry with resonant (m,n) phasing."""
        for k in range(N_coils):
            phi_k = 2 * np.pi * k / N_coils
            # Current amplitude for (n_target) resonant phasing
            I_k = self.I0 * np.cos(self.n_target * phi_k)
            pts = self._saddle_coil_pts(phi_k)
            self.add_coil(pts, I_k)

    def _saddle_coil_pts(self, phi_center, n_pts=5):
        """Points defining a rectangular saddle coil centered at phi_center.

        The coil is a closed rectangle in the (toroidal, poloidal) plane.
        """
        d_phi = np.pi / 8    # half-width in toroidal angle
        d_theta = np.pi / 4  # half-width in poloidal angle

        phi_vals = [
            phi_center - d_phi,
            phi_center + d_phi,
            phi_center + d_phi,
            phi_center - d_phi,
            phi_center - d_phi,
        ]
        theta_vals = [-d_theta, -d_theta, d_theta, d_theta, -d_theta]

        pts = []
        for phi_c, theta_c in zip(phi_vals, theta_vals):
            R_c = self.R0 + self.r_coil * np.cos(theta_c)
            Z_c = self.r_coil * np.sin(theta_c)
            X_c = R_c * np.cos(phi_c)
            Y_c = R_c * np.sin(phi_c)
            pts.append([X_c, Y_c, Z_c])
        return np.array(pts)

    def set_phase(self, phase_offset):
        """Re-phase all coil currents by adding a toroidal phase offset (rad)."""
        for k in range(len(self.coils)):
            phi_k = 2 * np.pi * k / self._N_coils
            I_k = self.I0 * np.cos(self.n_target * phi_k + phase_offset)
            self.set_current(k, I_k)

    def set_amplitude(self, I0_new):
        """Set a new reference amplitude for all coil currents."""
        ratio = I0_new / (self.I0 + 1e-30)
        self.scale_currents(ratio)
        self.I0 = float(I0_new)

    def __repr__(self):
        return (
            f"StellaratorControlCoils(R0={self.R0}, r_coil={self.r_coil}, "
            f"N_coils={self._N_coils}, m={self.m_target}, n={self.n_target}, "
            f"I0={self.I0} A)"
        )


# Backwards-compatibility alias
biot_savart_field = Biot_Savart_field
