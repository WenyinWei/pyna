"""Analytic magnetic field formulas for current loops and solenoids.

Ported from ``mhdpy.field.axisym`` (Wenyin Wei, EAST/Tsinghua).
All functions are pure NumPy/SciPy with no external data dependencies.
"""
from __future__ import annotations

import numpy as np


def BRBZ_induced_by_current_loop(
    a: float,
    Z_o: float,
    I: float,
    R: float | np.ndarray,
    Z: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Magnetic field (B_R, B_Z) of a single circular current loop.

    Uses the exact analytic formula based on complete elliptic integrals
    (Smythe 1989, §7.10; Schill 2003):

    .. math::

        B_R = \frac{\mu_0 I}{2\pi}
              \frac{Z - Z_o}{R\,\sqrt{(R+a)^2+(Z-Z_o)^2}}
              \left[-K(m) + \frac{a^2+R^2+(Z-Z_o)^2}{(a-R)^2+(Z-Z_o)^2}E(m)\right]

        B_Z = \frac{\mu_0 I}{2\pi}
              \frac{1}{\sqrt{(R+a)^2+(Z-Z_o)^2}}
              \left[K(m) + \frac{a^2-R^2-(Z-Z_o)^2}{(a-R)^2+(Z-Z_o)^2}E(m)\right]

    where :math:`m = 4aR / [(R+a)^2+(Z-Z_o)^2]`.

    Parameters
    ----------
    a:
        Loop radius (m).
    Z_o:
        Axial position of the loop (m).
    I:
        Current (A).  Positive current produces positive B_Z on axis.
    R:
        Radial coordinate(s) of the evaluation point(s) (m).
        Must be > 0.
    Z:
        Axial coordinate(s) of the evaluation point(s) (m).

    Returns
    -------
    (BR, BZ) : tuple of ndarray
        Radial and axial field components (T).

    References
    ----------
    * W. R. Smythe, *Static and Dynamic Electricity*, 3rd ed. (1989),
      Taylor & Francis, p. 291.
    * R. A. Schill Jr., *IEEE Trans. Magn.* 39, 961 (2003).
    """
    from scipy.constants import mu_0, pi
    from scipy.special import ellipk, ellipe

    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    Z_rel = Z - Z_o
    denom = (R + a) ** 2 + Z_rel ** 2
    m = 4 * a * R / denom
    coeff = mu_0 * I / (2 * pi) / np.sqrt(denom)
    d2 = (a - R) ** 2 + Z_rel ** 2
    BR = coeff * Z_rel / R * (-ellipk(m) + (a**2 + R**2 + Z_rel**2) / d2 * ellipe(m))
    BZ = coeff * (ellipk(m) + (a**2 - R**2 - Z_rel**2) / d2 * ellipe(m))
    return BR, BZ


def BRBZ_induced_by_thick_finitelen_solenoid(
    a: float,
    b: float,
    Z_solenoid_lowend: float,
    L: float,
    I: float,
    N: float,
    R: float | np.ndarray,
    Z: float | np.ndarray,
) -> tuple[float, float]:
    """Magnetic field of a thick finite-length solenoid.

    Uses the Labinac et al. (2006) formula based on Bessel functions
    and Struve functions.

    .. note::

        The integral may be numerically unstable for extreme aspect
        ratios.  Use with caution near the solenoid boundary.

    Parameters
    ----------
    a:
        Inner radius of the solenoid (m).
    b:
        Outer radius of the solenoid (m).
    Z_solenoid_lowend:
        Z coordinate of the lower end (m).
    L:
        Axial length of the solenoid (m).
    I:
        Current per wire (A).
    N:
        Total number of turns.
    R, Z:
        Evaluation point (m).  Scalar values only.

    Returns
    -------
    (BR, BZ) : tuple of float
        Field components at (R, Z) in Tesla.

    References
    ----------
    * V. Labinac, N. Erceg, D. Kotnik-Karuza, *Am. J. Phys.* 74,
      621 (2006).  https://doi.org/10.1119/1.2198885
    """
    from math import exp
    from scipy.constants import mu_0, pi
    from scipy.special import j0, j1, struve as H
    from scipy.integrate import quad

    R = float(R)
    Z = float(Z)

    # Handle field points below the lower end by symmetry
    if Z < Z_solenoid_lowend:
        Z_mid = Z_solenoid_lowend + L / 2
        BR, BZ = BRBZ_induced_by_thick_finitelen_solenoid(
            a, b, Z_solenoid_lowend, L, I, N, R, Z_mid + (Z_mid - Z)
        )
        return -BR, BZ

    Z = Z - Z_solenoid_lowend  # local Z from lower end

    B_inf = mu_0 * N * I / L  # infinite solenoid field

    def g(k: float) -> float:
        """Eq. (23) of Labinac (2006)."""
        ka, kb = k * a, k * b
        return (1 / ka) * (
            -j1(ka) * H(0, ka)
            + b / a * j1(kb) * H(0, kb)
            + j0(ka) * H(1, ka)
            - b / a * j0(kb) * H(1, kb)
        )

    def f(k: float) -> float:
        """Eq. (9) of Labinac (2006)."""
        if Z >= L:
            return exp(-k * (Z - L)) - exp(-k * Z)
        return 2.0 - exp(-k * (L - Z)) - exp(-k * Z)

    # Variable substitution k = x/(1-x) to map [0, ∞) → [0, 1)
    def integrand_BR(x: float) -> float:
        k = x / (1 - x)
        dk_dx = 1 / (1 - x) ** 2
        return j1(k * R) * (exp(-k * abs(Z - L)) - exp(-k * Z)) * g(k) * dk_dx

    def integrand_BZ(x: float) -> float:
        k = x / (1 - x)
        dk_dx = 1 / (1 - x) ** 2
        return j0(k * R) * f(k) * g(k) * dk_dx

    BR = B_inf * a**2 * pi / (4 * (b - a)) * quad(integrand_BR, 0, 1, limit=200)[0]
    BZ = B_inf * a**2 * pi / (4 * (b - a)) * quad(integrand_BZ, 0, 1, limit=200)[0]
    return BR, BZ


def BRBZ_induced_by_thick_finitelen_solenoid_multiprocessing(
    R: np.ndarray,
    Z: np.ndarray,
    Rc: float,
    Zc: float,
    dR: float,
    dZ: float,
    turn: int,
    I: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Parallel evaluation of :func:`BRBZ_induced_by_thick_finitelen_solenoid` on a grid.

    Parameters
    ----------
    R:
        1-D array of radial grid values (m).
    Z:
        1-D array of axial grid values (m).
    Rc, Zc:
        Centre position of the solenoid (m).
    dR, dZ:
        Half-width and half-height of the solenoid cross-section (m).
    turn:
        Number of turns.
    I:
        Current (A).

    Returns
    -------
    (BR_grid, BZ_grid) : tuple of ndarray
        Arrays of shape ``(len(R), len(Z))``.  Grid points inside
        the solenoid body are set to NaN.
    """
    from joblib import Parallel, delayed

    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    BR_o = np.zeros((len(R), len(Z)))
    BZ_o = np.zeros((len(R), len(Z)))

    def _task(iR: int, iZ: int, Rval: float, Zval: float):
        if Rc - dR <= Rval <= Rc + dR and Zc - dZ <= Zval <= Zc + dZ:
            return iR, iZ, np.nan, np.nan
        br, bz = BRBZ_induced_by_thick_finitelen_solenoid(
            Rc - dR / 2, Rc + dR / 2, Zc - dZ / 2, dZ, I, turn, Rval, Zval
        )
        return iR, iZ, br, bz

    tasks = [
        (iR, iZ, float(Rval), float(Zval))
        for iR, Rval in enumerate(R)
        for iZ, Zval in enumerate(Z)
    ]

    results = Parallel(n_jobs=-1, backend="loky", verbose=0)(
        delayed(_task)(*t) for t in tasks
    )

    for res in results:
        if res is not None:
            iR, iZ, br, bz = res
            BR_o[iR, iZ] = br
            BZ_o[iR, iZ] = bz

    return BR_o, BZ_o


from pyna.MCF.coils.base import VacuumCoilField


class AnalyticCircularCoilField(VacuumCoilField):
    """Vacuum field of a circular current loop, using exact analytic formula.

    The loop may be translated and tilted relative to the cylindrical
    coordinate system via a center position and normal vector.

    For an untilted loop (normal along Z), the exact Smythe formula is used.
    For a tilted loop, evaluation points are first transformed into the loop's
    local Cartesian frame, then the analytic formula is applied, and the result
    is rotated back to the lab frame.

    Parameters
    ----------
    radius : float
        Loop radius (m).
    center_xyz : array-like, shape (3,)
        Center of the loop in lab Cartesian (X, Y, Z) coordinates (m).
    normal_xyz : array-like, shape (3,)
        Unit normal vector of the loop plane in lab Cartesian coordinates.
        Defaults to (0, 0, 1) for a horizontal loop.
    current : float
        Loop current (A). Positive -> right-hand rule along the normal.
    """

    def __init__(
        self,
        radius: float,
        center_xyz,
        normal_xyz=(0.0, 0.0, 1.0),
        current: float = 1.0,
    ) -> None:
        self._a = float(radius)
        self._center = np.asarray(center_xyz, dtype=float)
        normal = np.asarray(normal_xyz, dtype=float)
        self._normal = normal / np.linalg.norm(normal)
        self._I = float(current)
        self._R_lab2loc, self._R_loc2lab = _build_rotation(self._normal)

    def B_at(self, R, Z, phi):
        R = np.asarray(R, dtype=float)
        Z = np.asarray(Z, dtype=float)
        phi = np.asarray(phi, dtype=float)
        shape = np.broadcast(R, Z, phi).shape
        X = (R * np.cos(phi)).ravel()
        Y = (R * np.sin(phi)).ravel()
        Zlab = Z.ravel() if Z.ndim > 0 else np.full(X.shape, float(Z))
        pts_lab = np.stack([X, Y, Zlab], axis=1) - self._center
        pts_loc = pts_lab @ self._R_lab2loc.T
        R_loc = np.sqrt(pts_loc[:, 0]**2 + pts_loc[:, 1]**2)
        Z_loc = pts_loc[:, 2]
        phi_loc = np.arctan2(pts_loc[:, 1], pts_loc[:, 0])
        BR_loc, BZ_loc = BRBZ_induced_by_current_loop(
            self._a, 0.0, self._I, R_loc, Z_loc
        )
        Bx_loc = BR_loc * np.cos(phi_loc)
        By_loc = BR_loc * np.sin(phi_loc)
        Bz_loc = BZ_loc
        B_loc = np.stack([Bx_loc, By_loc, Bz_loc], axis=1)
        B_lab = B_loc @ self._R_loc2lab.T
        phi_flat = phi.ravel()
        BR_lab = B_lab[:, 0] * np.cos(phi_flat) + B_lab[:, 1] * np.sin(phi_flat)
        Bp_lab = -B_lab[:, 0] * np.sin(phi_flat) + B_lab[:, 1] * np.cos(phi_flat)
        BZ_lab = B_lab[:, 2]
        return BR_lab.reshape(shape), BZ_lab.reshape(shape), Bp_lab.reshape(shape)

    def divergence_free(self) -> bool:
        return True


def _build_rotation(normal):
    """Build rotation matrices between lab frame and loop-local frame."""
    z = np.array([0.0, 0.0, 1.0])
    n = normal / np.linalg.norm(normal)
    cross = np.cross(z, n)
    sin_a = np.linalg.norm(cross)
    cos_a = np.dot(z, n)
    if sin_a < 1e-12:
        R = np.eye(3) if cos_a > 0 else np.diag([1.0, -1.0, -1.0])
        return R, R.T
    axis = cross / sin_a
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + sin_a * K + (1 - cos_a) * K @ K
    return R, R.T
