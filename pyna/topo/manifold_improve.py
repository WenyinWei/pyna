"""
Stable and unstable manifold computation for hyperbolic fixed points.

This module supplements the existing ``pyna.topo.manifold`` module with
object-oriented :class:`StableManifold` and :class:`UnstableManifold` classes
that take an X-point and its monodromy matrix (from variational equations),
grow the manifold by iterating field-line integration, and can plot the
result.
"""

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _ManifoldBase:
    """Base class for stable/unstable manifolds of a hyperbolic fixed point.

    Parameters
    ----------
    x_point : array_like, shape (2,)
        Coordinates [R, Z] of the hyperbolic fixed point (X-point).
    monodromy : ndarray, shape (2, 2)
        Monodromy (Jacobian) matrix of the Poincaré map at the X-point,
        as returned by
        :meth:`~pyna.topo.variational.PoincareMapVariationalEquations.jacobian_matrix`.
    field_func : callable
        ``field_func(r, z, phi)`` → array_like of shape (2,) giving
        (dR/dφ, dZ/dφ) for field-line tracing.
    phi_span : tuple of float, optional
        One-turn integration range, e.g. (0, 2π).  Default (0, 2π).
    """

    # Subclasses set this to 'stable' or 'unstable'
    _branch = None

    def __init__(self, x_point, Jac, field_func,
                 phi_span=(0.0, 2 * np.pi)):
        self.x_point = np.asarray(x_point, dtype=float)
        self.Jac = np.asarray(Jac, dtype=float)
        self.field_func = field_func
        self.phi_span = phi_span

        # Compute eigenvalues / eigenvectors of the Jacobian matrix
        eigvals, eigvecs = np.linalg.eig(Jac)
        self._eigvals = eigvals
        self._eigvecs = eigvecs  # columns are eigenvectors

        # Select the eigenvector corresponding to this branch
        self._select_eigenvector()

        # Storage for grown manifold points
        self.segments: list[np.ndarray] = []  # each element: ndarray shape (N, 2)

    def _select_eigenvector(self):
        """Select the eigenvalue/eigenvector for this branch.

        Stable branch:   |λ| < 1
        Unstable branch: |λ| > 1
        """
        mods = np.abs(self._eigvals)
        if self._branch == 'stable':
            idx = np.argmin(mods)
        else:  # unstable
            idx = np.argmax(mods)

        lam = self._eigvals[idx]
        vec = self._eigvecs[:, idx].real

        if np.abs(lam) < 1e-14:
            raise RuntimeError("Zero eigenvalue found — monodromy matrix may be singular.")

        self._lam = lam
        self._evec = vec / np.linalg.norm(vec)

    def _integrate_fieldline(self, x0, phi_span, **kwargs):
        """Integrate a single field line from x0 over phi_span."""
        kw = dict(method="DOP853", rtol=1e-8, atol=1e-11, dense_output=False)
        kw.update(kwargs)
        sol = solve_ivp(
            fun=lambda phi, y: np.asarray(self.field_func(y[0], y[1], phi),
                                          dtype=float),
            t_span=phi_span,
            y0=np.asarray(x0, dtype=float),
            **kw,
        )
        return sol.y[:, -1]  # final position

    def grow(self, n_turns=20, init_length=1e-4, n_init_pts=5,
             both_sides=True, **solve_ivp_kwargs):
        """Grow the manifold by iterating the Poincaré map.

        Starting from a short initial segment along the eigenvector, each
        point is iterated forward (unstable) or backward (stable) under the
        Poincaré map for ``n_turns`` turns.

        Parameters
        ----------
        n_turns : int
            Number of map iterations.
        init_length : float
            Length of the initial perturbation segment.
        n_init_pts : int
            Number of seed points on the initial segment.
        both_sides : bool
            If True, also grow the manifold in the −eigenvector direction.
        **solve_ivp_kwargs :
            Forwarded to ``solve_ivp``.
        """
        self.segments = []
        signs = [1.0, -1.0] if both_sides else [1.0]

        phi_s, phi_e = self.phi_span
        if self._branch == 'stable':
            # Integrate backward (reverse φ span) so seeds converge to X-pt
            phi_span_iter = (phi_e, phi_s)
        else:
            phi_span_iter = (phi_s, phi_e)

        for sgn in signs:
            pts = np.empty((n_init_pts, 2))
            for k, eps in enumerate(
                    np.linspace(0, init_length, n_init_pts, endpoint=True)):
                pts[k] = self.x_point + sgn * eps * self._evec

            # Iterate the map
            seg_list = [pts.copy()]
            for _ in range(n_turns):
                new_pts = np.empty_like(pts)
                for k in range(n_init_pts):
                    new_pts[k] = self._integrate_fieldline(
                        pts[k], phi_span_iter, **solve_ivp_kwargs)
                pts = new_pts
                seg_list.append(pts.copy())

            # Flatten into a single (N, 2) array ordered along the manifold
            self.segments.append(np.vstack(seg_list))

        return self

    def plot(self, ax, **kwargs):
        """Plot the grown manifold on a matplotlib Axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        **kwargs :
            Forwarded to ``ax.plot`` (e.g. ``color``, ``lw``).
        """
        defaults = dict(lw=0.8)
        defaults.update(kwargs)
        for seg in self.segments:
            ax.plot(seg[:, 0], seg[:, 1], **defaults)
        return ax


# ---------------------------------------------------------------------------
# Concrete subclasses
# ---------------------------------------------------------------------------

class StableManifold(_ManifoldBase):
    """Stable manifold of a hyperbolic fixed point.

    The stable manifold consists of all trajectories that converge to the
    X-point as the map is iterated *forward*.  Equivalently, seeds on the
    stable eigenvector are integrated *backward* in φ.

    Inherits from :class:`_ManifoldBase`.  See its documentation for
    parameter details.
    """

    _branch = 'stable'


class UnstableManifold(_ManifoldBase):
    """Unstable manifold of a hyperbolic fixed point.

    The unstable manifold consists of all trajectories that diverge from the
    X-point as the map is iterated *forward*.  Seeds on the unstable
    eigenvector are integrated *forward* in φ.

    Inherits from :class:`_ManifoldBase`.  See its documentation for
    parameter details.
    """

    _branch = 'unstable'
