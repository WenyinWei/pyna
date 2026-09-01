"""Analytic magnetic field formulas for current loops and solenoids.

Ported from ``mhdpy.field.axisym`` (Wenyin Wei, EAST/Tsinghua).
The production rectangular winding-pack solver uses float64 CUDA when
available and a NumPy/SciPy threaded fallback otherwise.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec
import os

import numpy as np


MU0_VACUUM_H_M = 4.0e-7 * np.pi


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
    from scipy.special import ellipk, ellipe

    R, Z = np.broadcast_arrays(np.asarray(R, dtype=float), np.asarray(Z, dtype=float))
    Z_rel = Z - Z_o
    axis = np.abs(R) <= max(abs(float(a)), 1.0) * 1.0e-14
    R_safe = np.where(axis, max(abs(float(a)), 1.0) * 1.0e-14, R)
    denom = (R_safe + a) ** 2 + Z_rel ** 2
    m = 4 * a * R_safe / denom
    coeff = MU0_VACUUM_H_M * I / (2 * np.pi) / np.sqrt(denom)
    d2 = (a - R_safe) ** 2 + Z_rel ** 2
    BR = coeff * Z_rel / R_safe * (
        -ellipk(m) + (a**2 + R_safe**2 + Z_rel**2) / d2 * ellipe(m)
    )
    BZ = coeff * (ellipk(m) + (a**2 - R_safe**2 - Z_rel**2) / d2 * ellipe(m))
    if np.any(axis):
        BR = np.where(axis, 0.0, BR)
        BZ_axis = MU0_VACUUM_H_M * I * a**2 / (2.0 * (a**2 + Z_rel**2) ** 1.5)
        BZ = np.where(axis, BZ_axis, BZ)
    return BR, BZ


# DO NOT USE: retained only to document and reproduce the rejected PF1 archive.
def _BRBZ_induced_by_thick_finitelen_solenoid_legacy_nonconverged(
    a: float,
    b: float,
    Z_solenoid_lowend: float,
    L: float,
    I: float,
    N: float,
    R: float | np.ndarray,
    Z: float | np.ndarray,
) -> tuple[float, float]:
    """Rejected legacy Labinac-integral implementation; do not use.

    This implementation was inherited from MHDpy and is retained privately
    only for forensic reproduction.  Its 2025 substitution
    ``k = x / (1 - x)`` maps the infinite interval to ``x in [0, 1]`` and
    then calls ``quad(..., limit=200)``.  Inside the winding-pack Z interval,
    the BZ factor tends to the non-decaying constant 2 as ``k -> infinity``.
    The transformed endpoint therefore remains strongly oscillatory, exhausts
    the subdivision limit, and returns a false horizontal BZ-error band whose
    edges coincide with the two coil Z faces.  It must not generate vacuum
    field archives or production solver inputs.

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
        BR, BZ = _BRBZ_induced_by_thick_finitelen_solenoid_legacy_nonconverged(
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


def BRBZ_induced_by_rectangular_winding_pack_gauss_legendre(
    Rc: float,
    Zc: float,
    width_R: float,
    height_Z: float,
    turns: int,
    current_per_turn: float,
    R: float | np.ndarray,
    Z: float | np.ndarray,
    *,
    quadrature_order: int = 32,
    max_workers: int | None = None,
    backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Field of a uniformly filled rectangular PF winding pack.

    Tensor-product Gauss--Legendre quadrature area-averages exact circular-loop
    Biot--Savart fields.  The weighted filament currents sum to
    ``turns * current_per_turn``; no oscillatory infinite-interval integral is
    used.  ``backend='auto'`` uses the float64 CUDA analytic-loop kernel when
    CuPy and a CUDA device are available, otherwise CPU threads.

    The returned components are ordered ``(B_R, B_Z)``.  Positive current is
    physical ``+e_phi`` and produces positive ``B_Z`` on axis; this is directly
    compatible with the left-handed storage order ``(R, Z, phi)`` and performs
    no handedness-sensitive cross product.
    """
    Rc = float(Rc)
    Zc = float(Zc)
    width_R = float(width_R)
    height_Z = float(height_Z)
    turns = int(turns)
    current_per_turn = float(current_per_turn)
    order = int(quadrature_order)
    backend = str(backend).lower()
    if Rc <= 0.0 or width_R <= 0.0 or height_Z <= 0.0:
        raise ValueError("rectangular winding-pack dimensions must be positive")
    if turns <= 0 or order < 2:
        raise ValueError("turns and quadrature_order must be positive")
    if backend not in {"auto", "cpu", "cuda"}:
        raise ValueError("backend must be 'auto', 'cpu', or 'cuda'")

    R_target, Z_target = np.broadcast_arrays(
        np.asarray(R, dtype=float), np.asarray(Z, dtype=float)
    )
    if np.any(R_target < 0.0):
        raise ValueError("cylindrical target radius must be nonnegative")

    nodes, weights = np.polynomial.legendre.leggauss(order)
    source_R = Rc + 0.5 * width_R * nodes
    source_Z = Zc + 0.5 * height_Z * nodes
    area_weights = 0.5 * weights

    use_cuda = False
    accel = None
    if backend != "cpu":
        if find_spec("cupy") is not None:
            try:
                from pyna.toroidal.coils import accel

                use_cuda = bool(accel._CUPY_AVAILABLE)
                if use_cuda:
                    use_cuda = accel.cp.cuda.runtime.getDeviceCount() > 0
            except (ImportError, RuntimeError):
                use_cuda = False
        if backend == "cuda" and not use_cuda:
            raise RuntimeError("backend='cuda' requires CuPy and a CUDA device")
    if np.any(R_target == 0.0):
        use_cuda = False

    if use_cuda:
        source_radii = np.repeat(source_R, order)
        source_heights = np.tile(source_Z, order)
        source_currents = (
            turns
            * current_per_turn
            * np.repeat(area_weights, order)
            * np.tile(area_weights, order)
        )
        centers = np.column_stack(
            [
                np.zeros(order * order),
                np.zeros(order * order),
                source_heights,
            ]
        )
        normals = np.zeros((order * order, 3), dtype=float)
        normals[:, 2] = 1.0
        field_points = np.column_stack(
            [
                R_target.ravel(),
                np.zeros(R_target.size),
                Z_target.ravel(),
            ]
        )
        field = accel.analytic_coil_field_batched_gpu(
            centers,
            source_radii,
            normals,
            source_currents,
            field_points,
        )
        BR = field[:, 0].reshape(R_target.shape)
        BZ = field[:, 2].reshape(R_target.shape)
        if not np.all(np.isfinite(BR)) or not np.all(np.isfinite(BZ)):
            raise FloatingPointError(
                "rectangular winding-pack field is non-finite; target may intersect a quadrature filament"
            )
        return BR, BZ

    def radial_slice(index: int) -> tuple[np.ndarray, np.ndarray]:
        BR_slice = np.zeros(R_target.shape, dtype=float)
        BZ_slice = np.zeros(R_target.shape, dtype=float)
        for source_z, weight_z in zip(source_Z, area_weights):
            filament_current = (
                turns
                * current_per_turn
                * float(area_weights[index])
                * float(weight_z)
            )
            BR_loop, BZ_loop = BRBZ_induced_by_current_loop(
                float(source_R[index]),
                float(source_z),
                filament_current,
                R_target,
                Z_target,
            )
            BR_slice += BR_loop
            BZ_slice += BZ_loop
        return BR_slice, BZ_slice

    if max_workers is None:
        workers = min(order, 16, os.cpu_count() or 1)
    else:
        workers = int(max_workers)
        if workers == -1:
            workers = min(order, 16, os.cpu_count() or 1)
        elif workers < 1:
            raise ValueError("max_workers must be positive or -1")
        else:
            workers = min(workers, order)

    if workers == 1:
        slices = [radial_slice(index) for index in range(order)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            slices = list(executor.map(radial_slice, range(order)))

    BR = np.zeros(R_target.shape, dtype=float)
    BZ = np.zeros(R_target.shape, dtype=float)
    for BR_slice, BZ_slice in slices:
        BR += BR_slice
        BZ += BZ_slice
    if not np.all(np.isfinite(BR)) or not np.all(np.isfinite(BZ)):
        raise FloatingPointError(
            "rectangular winding-pack field is non-finite; target may intersect a quadrature filament"
        )
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
    *,
    quadrature_order: int = 32,
    max_workers: int | None = None,
    backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility name backed by the production Gauss--Legendre solver.

    The historical public signature is preserved, but this function no longer
    calls the rejected Labinac infinite-interval quadrature.  ``a`` and ``b``
    are the inner and outer winding-pack radii.
    """
    return BRBZ_induced_by_rectangular_winding_pack_gauss_legendre(
        0.5 * (float(a) + float(b)),
        float(Z_solenoid_lowend) + 0.5 * float(L),
        float(b) - float(a),
        float(L),
        int(N),
        float(I),
        R,
        Z,
        quadrature_order=quadrature_order,
        max_workers=max_workers,
        backend=backend,
    )


def BRBZ_induced_by_thick_finitelen_solenoid_multiprocessing(
    R: np.ndarray,
    Z: np.ndarray,
    Rc: float,
    Zc: float,
    dR: float,
    dZ: float,
    turn: int,
    I: float,
    *,
    quadrature_order: int = 32,
    n_jobs: int = -1,
    backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the production winding-pack field on an ``(R, Z)`` grid.

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
    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    BR_o, BZ_o = BRBZ_induced_by_rectangular_winding_pack_gauss_legendre(
        Rc,
        Zc,
        2.0 * dR,
        2.0 * dZ,
        turn,
        I,
        RR,
        ZZ,
        quadrature_order=quadrature_order,
        max_workers=n_jobs,
        backend=backend,
    )
    inside = (
        (RR >= Rc - dR)
        & (RR <= Rc + dR)
        & (ZZ >= Zc - dZ)
        & (ZZ <= Zc + dZ)
    )
    BR_o = np.where(inside, np.nan, BR_o)
    BZ_o = np.where(inside, np.nan, BZ_o)

    return BR_o, BZ_o


from pyna.toroidal.coils.base import CoilFieldVacuum


class CoilFieldAnalyticCircular(CoilFieldVacuum):
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


class CoilFieldAnalyticRectangularSection(CoilFieldVacuum):
    """Vacuum field of a tokamak PF coil with rectangular cross-section.

    Uses tensor-product Gauss--Legendre quadrature of exact circular-loop
    Biot--Savart fields over the uniformly filled winding pack.

    Parameters
    ----------
    Rc : float
        Radial center of the coil cross-section (m).
    Zc : float
        Axial center of the coil cross-section (m).
    dR : float
        Half-width of the cross-section in the R direction (m).
        The coil spans [Rc - dR, Rc + dR].
    dZ : float
        Half-height of the cross-section in the Z direction (m).
        The coil spans [Zc - dZ, Zc + dZ].
    turns : int
        Number of turns.
    current : float
        Current per turn (A). Positive current in the +phi direction
        produces positive B_Z on axis.

    For evaluation on a 2D (R, Z) grid, :meth:`B_at_grid` uses CUDA when
    available (CPU threads otherwise) and masks points inside the coil body.
    """

    def __init__(
        self,
        Rc: float,
        Zc: float,
        dR: float,
        dZ: float,
        turns: int,
        current: float = 1.0,
    ) -> None:
        self._Rc = float(Rc)
        self._Zc = float(Zc)
        self._dR = float(dR)
        self._dZ = float(dZ)
        self._turns = int(turns)
        self._I = float(current)

    @property
    def Rc(self) -> float:
        return self._Rc

    @property
    def Zc(self) -> float:
        return self._Zc

    @property
    def dR(self) -> float:
        return self._dR

    @property
    def dZ(self) -> float:
        return self._dZ

    @property
    def turns(self) -> int:
        return self._turns

    @property
    def current(self) -> float:
        return self._I

    @current.setter
    def current(self, value: float) -> None:
        self._I = float(value)

    def B_at(self, R, Z, phi):
        """Evaluate (B_R, B_Z, B_phi) at arbitrary (R, Z, phi) points.

        For axisymmetric coils, B_phi = 0 and the result does not depend on phi.

        Parameters
        ----------
        R, Z, phi : scalar or array-like, broadcast-compatible

        Returns
        -------
        (BR, BZ, Bphi) : tuple of ndarray
        """
        R_eval, Z_eval = np.broadcast_arrays(
            np.asarray(R, dtype=float), np.asarray(Z, dtype=float)
        )
        BR_out, BZ_out = BRBZ_induced_by_rectangular_winding_pack_gauss_legendre(
            self._Rc,
            self._Zc,
            2.0 * self._dR,
            2.0 * self._dZ,
            self._turns,
            self._I,
            R_eval,
            Z_eval,
            max_workers=1,
            backend="cpu",
        )
        return BR_out, BZ_out, np.zeros_like(BR_out)

    def B_at_grid(
        self,
        R: np.ndarray,
        Z: np.ndarray,
        *,
        n_jobs: int = -1,
    ) -> tuple:
        """Parallel evaluation on a 2D (R, Z) grid.

        Grid points inside the coil body are set to NaN.

        Parameters
        ----------
        R : 1-D array of radial grid values (m).
        Z : 1-D array of axial grid values (m).
        n_jobs : int
            CPU worker count; ``-1`` selects the automatic count.  Ignored
            when the CUDA backend is active.

        Returns
        -------
        (BR_grid, BZ_grid) : ndarray, shape (len(R), len(Z))
        """
        return BRBZ_induced_by_thick_finitelen_solenoid_multiprocessing(
            np.asarray(R, dtype=float),
            np.asarray(Z, dtype=float),
            self._Rc,
            self._Zc,
            self._dR,
            self._dZ,
            self._turns,
            self._I,
            n_jobs=n_jobs,
        )

    def divergence_free(self) -> bool:
        return True
