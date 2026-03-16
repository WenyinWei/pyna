
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
