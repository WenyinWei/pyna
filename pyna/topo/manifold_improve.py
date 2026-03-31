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
# Warning class
# ---------------------------------------------------------------------------

class ManifoldWarning(UserWarning):
    """Warning for suspicious manifold growth behavior."""
    pass


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

    def __init__(self, x_point, DPm, field_func,
                 phi_span=(0.0, 2 * np.pi)):
        self.x_point = np.asarray(x_point, dtype=float)
        self.DPm = np.asarray(DPm, dtype=float)
        self.field_func = field_func
        self.phi_span = phi_span

        # Compute eigenvalues / eigenvectors of the full-period Poincare Jacobian
        eigvals, eigvecs = np.linalg.eig(DPm)
        self._eigvals = eigvals
        self._eigvecs = eigvecs  # columns are eigenvectors

        # Select the eigenvector corresponding to this branch
        self._select_eigenvector()

        # Run health check on the X-point
        self._verify_xpoint()

        # Storage for grown manifold points
        self.segments: list[np.ndarray] = []  # each element: ndarray shape (N, 2)

    @property
    def Jac(self):
        """Deprecated alias for DPm."""
        import warnings
        warnings.warn(
            "Stable/Unstable manifold .Jac is deprecated; use .DPm instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.DPm

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

    def _verify_xpoint(self, tol=1e-4):
        """Check the given x_point is actually a hyperbolic fixed point."""
        import warnings
        mods = np.abs(self._eigvals)
        if not (mods.max() > 1.0 + tol and mods.min() < 1.0 - tol):
            warnings.warn(
                f"X-point does not appear to be hyperbolic: |λ| = {mods}. "
                f"Manifold growth will likely produce straight lines. "
                f"Ensure n_turns matches the orbit period (e.g. n_turns=m for q=m/n (m = toroidal period)).",
                ManifoldWarning,
                stacklevel=3,
            )
        det = np.linalg.det(self.DPm)
        if abs(det - 1.0) > 0.01:
            warnings.warn(
                f"Monodromy matrix det={det:.4f} deviates from 1 (expected for area-preserving map). "
                f"Numerical integration may be inaccurate.",
                ManifoldWarning,
                stacklevel=3,
            )

    def _check_not_straight(self, segment, eigvec, turn_idx):
        """Warn if segment is suspiciously straight (manifold not curving)."""
        import warnings
        if len(segment) < 4:
            return
        diffs = np.diff(segment, axis=0)
        norms = np.linalg.norm(diffs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-15, 1e-15, norms)
        dirs = diffs / norms
        dots = np.einsum('ij,ij->i', dirs[:-1], dirs[1:])
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.degrees(np.arccos(dots))
        max_angle = angles.max() if len(angles) > 0 else 0.0
        if max_angle < 2.0 and turn_idx >= 3:
            warnings.warn(
                f"Manifold appears straight after {turn_idx} turns (max angle change {max_angle:.2f}°). "
                f"Check: (1) Is x_point a true fixed point? (2) Is phi_span correct for the orbit period? "
                f"(3) Is field_func returning dR/dphi not unit tangent?",
                ManifoldWarning,
                stacklevel=4,
            )

    def _check_step_size(self, segment, turn_idx):
        """Warn if manifold has large jumps (fold/cusp overshoot)."""
        import warnings
        if len(segment) < 3:
            return
        dists = np.linalg.norm(np.diff(segment, axis=0), axis=1)
        median_d = np.median(dists)
        if median_d < 1e-15:
            return
        bad = dists > 10.0 * median_d
        if bad.any():
            warnings.warn(
                f"Manifold segment has large jumps at turn {turn_idx} "
                f"(max/median = {dists.max()/median_d:.1f}x). "
                f"Consider smaller init_length or finer dt at folds.",
                ManifoldWarning,
                stacklevel=4,
            )

    def _check_self_intersection(self, segment, turn_idx):
        """Warn if a manifold branch appears to self-intersect."""
        import warnings
        if len(segment) < 6:
            return
        pts = segment
        n = len(pts)

        def _ccw(A, B, C):
            return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

        def _segments_intersect(p1, p2, p3, p4):
            return (_ccw(p1,p3,p4) != _ccw(p2,p3,p4) and
                    _ccw(p1,p2,p3) != _ccw(p1,p2,p4))

        found = False
        step = max(1, n // 40)
        idxs = list(range(0, n, step))
        for i in range(len(idxs)-1):
            for j in range(i+2, len(idxs)-1):
                if _segments_intersect(pts[idxs[i]], pts[idxs[i+1]],
                                       pts[idxs[j]], pts[idxs[j+1]]):
                    found = True
                    break
            if found:
                break

        if found:
            warnings.warn(
                f"Manifold branch appears to self-intersect at turn {turn_idx}. "
                f"This may indicate (1) overshoot at a fold, (2) wrong branch direction, "
                f"or (3) insufficient integration accuracy.",
                ManifoldWarning,
                stacklevel=4,
            )

    def _integrate_fieldline(self, x0, phi_span, **kwargs):
        """Integrate a single field line from x0 over phi_span."""
        kw = dict(method="DOP853", rtol=1e-8, atol=1e-11, dense_output=False)
        kw.update(kwargs)
        try:
            sol = solve_ivp(
                fun=lambda phi, y: np.asarray(self.field_func(y[0], y[1], phi),
                                              dtype=float),
                t_span=phi_span,
                y0=np.asarray(x0, dtype=float),
                **kw,
            )
        except (ValueError, FloatingPointError):
            return np.array([np.nan, np.nan])
        if not sol.success or not np.all(np.isfinite(sol.y[:, -1])):
            return np.array([np.nan, np.nan])
        return sol.y[:, -1]  # final position

    def grow(self, n_turns=20, init_length=1e-4, n_init_pts=5,
             both_sides=True, RZlimit=None, **solve_ivp_kwargs):
        """Grow the manifold using geometric-series seeding (carousel method).

        Seed points are placed at distances ``init_length * |λ|^k``
        (k = 0 … n_init_pts-1) along the eigenvector, so that after one
        Poincaré map iteration point *k* lands near the original position of
        point *k+1*.  This guarantees that ``np.vstack([gen0, gen1, …])``
        is monotone in arc-length without any post-sort.

        Points that escape outside *RZlimit* or fail to integrate are stored
        as NaN and masked when plotting — they do NOT abort the rest of the
        bundle.

        Parameters
        ----------
        n_turns : int
            Number of map iterations.
        init_length : float
            Distance of the *first* seed point from the X-point along the
            eigenvector (``init_length * |λ|^0``).
        n_init_pts : int
            Number of seed points.
        both_sides : bool
            Grow in both ±eigenvector directions.
        RZlimit : tuple (R_min, R_max, Z_min, Z_max) or None
            Points outside this box are treated as lost (replaced with NaN).
        **solve_ivp_kwargs :
            Forwarded to ``solve_ivp``.
        """
        self.segments = []
        signs = [1.0, -1.0] if both_sides else [1.0]

        phi_s, phi_e = self.phi_span
        if self._branch == 'stable':
            phi_span_iter = (phi_e, phi_s)   # integrate backward
        else:
            phi_span_iter = (phi_s, phi_e)   # integrate forward

        lam_abs = float(np.abs(self._lam))   # |λ_u| for unstable, |λ_s| for stable

        for sgn in signs:
            # Geometric-series seed points: distance = init_length * lam_abs^k
            pts = np.array([
                self.x_point + sgn * (init_length * lam_abs**k) * self._evec
                for k in range(n_init_pts)
            ])  # shape (n_init_pts, 2)

            all_gens = []

            for _turn in range(n_turns + 1):
                # Store current generation (may contain NaN for lost points)
                all_gens.append(pts.copy())

                if _turn == n_turns:
                    break

                # Advance each point independently; mask failures
                new_pts = np.full_like(pts, np.nan)
                for k in range(n_init_pts):
                    if np.any(np.isnan(pts[k])):
                        continue   # already lost
                    try:
                        p = self._integrate_fieldline(pts[k], phi_span_iter,
                                                      **solve_ivp_kwargs)
                        # Check wall
                        if RZlimit is not None:
                            Rm, RM, Zm, ZM = RZlimit
                            if not (Rm <= p[0] <= RM and Zm <= p[1] <= ZM):
                                continue   # lost — leave as NaN
                        new_pts[k] = p
                    except Exception:
                        pass   # leave as NaN
                pts = new_pts

            # Build monotone segment:
            #   gen0 fully: [pts[0], pts[1], ..., pts[n-1]]
            #   Then ONLY the last point (index n-1) from each subsequent generation.
            #
            # Why: after 1 map, gen1[k] ≈ gen0[k+1].  So gen0 already spans
            # the range [eps*lam^0 … eps*lam^{n-1}].  gen1[n-1] ≈ eps*lam^n is
            # the ONE new frontier point beyond gen0[-1].  gen1[0..n-2] would
            # approximate gen0[1..n-1] and if appended would reverse direction.
            seg_rows = []
            for gi, gen in enumerate(all_gens):
                if gi == 0:
                    for k in range(n_init_pts):
                        if not np.any(np.isnan(gen[k])):
                            seg_rows.append(gen[k])
                else:
                    k = n_init_pts - 1
                    if not np.any(np.isnan(gen[k])):
                        seg_rows.append(gen[k])

            if len(seg_rows) >= 2:
                seg = np.array(seg_rows)
                self._check_not_straight(seg, self._evec, n_turns)
                self._check_self_intersection(seg, n_turns)
                self.segments.append(seg)

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

class _AcceleratedManifoldBase(_ManifoldBase):
    """Manifold base that uses cyna C++ for single-step field-line integration."""

    def __init__(self, x_point, DPm, cache, wall, phi_section=0.0, DPhi=0.05, **kw):
        import numpy as np
        from scipy.interpolate import RegularGridInterpolator

        BR, BPhi_arr, BZ_arr = cache['BR'], cache['BPhi'], cache['BZ']
        R_grid, Z_grid, Phi_grid = cache['R_grid'], cache['Z_grid'], cache['Phi_grid']
        Phi_ext = np.append(Phi_grid, 2 * np.pi)
        _ext = lambda X: np.concatenate([X, X[:, :, :1]], axis=2)
        kwi = dict(method='linear', bounds_error=False, fill_value=np.nan)
        itp_BR   = RegularGridInterpolator((R_grid, Z_grid, Phi_ext), _ext(BR),     **kwi)
        itp_BPhi = RegularGridInterpolator((R_grid, Z_grid, Phi_ext), _ext(BPhi_arr), **kwi)
        itp_BZ   = RegularGridInterpolator((R_grid, Z_grid, Phi_ext), _ext(BZ_arr),  **kwi)

        def field_func_2d(R, Z, phi):
            phi_w = phi % (2 * np.pi)
            B_R   = float(itp_BR([[R, Z, phi_w]])[0])
            B_Phi = float(itp_BPhi([[R, Z, phi_w]])[0])
            B_Z   = float(itp_BZ([[R, Z, phi_w]])[0])
            if abs(B_Phi) < 1e-15:
                return np.array([0.0, 0.0])
            return np.array([B_R / B_Phi * R, B_Z / B_Phi * R])

        self._cache = cache
        self._wall = wall
        self._phi_section = phi_section
        self._DPhi = DPhi
        self._R_grid = np.ascontiguousarray(R_grid, dtype=np.float64)
        self._Z_grid = np.ascontiguousarray(Z_grid, dtype=np.float64)

        try:
            from pyna.MCF.flt import trace_poincare_batch_twall
            self._cyna_batch_twall = trace_poincare_batch_twall
        except ImportError:
            self._cyna_batch_twall = None
        try:
            from pyna.MCF.flt import trace_poincare_batch
            self._cyna_batch = trace_poincare_batch
        except ImportError:
            self._cyna_batch = None

        # Flat 1-D arrays (required by cyna C++ API)
        Phi_ext2 = np.append(Phi_grid, 2 * np.pi)
        self._Phi_ext  = np.ascontiguousarray(Phi_ext2, dtype=np.float64)
        self._BR_flat   = np.ascontiguousarray(_ext(BR),     dtype=np.float64).ravel()
        self._BPhi_flat = np.ascontiguousarray(_ext(BPhi_arr), dtype=np.float64).ravel()
        self._BZ_flat   = np.ascontiguousarray(_ext(BZ_arr),  dtype=np.float64).ravel()

        # Wall arrays
        self._wall_phi  = np.ascontiguousarray(wall._phi_centers, dtype=np.float64) if hasattr(wall, '_phi_centers') else None
        self._wall_R_all = np.ascontiguousarray(wall._R, dtype=np.float64) if hasattr(wall, '_R') else None
        self._wall_Z_all = np.ascontiguousarray(wall._Z, dtype=np.float64) if hasattr(wall, '_Z') else None
        wR, wZ = wall.get_section(phi_section)
        self._wall_R = np.ascontiguousarray(wR, dtype=np.float64)
        self._wall_Z = np.ascontiguousarray(wZ, dtype=np.float64)

        super().__init__(x_point, DPm, field_func_2d, **kw)

    def _integrate_fieldline(self, x0, phi_span, **kw):
        """Single field-line step using cyna batch (N_turns deduced from phi_span)."""
        import numpy as np
        phi_s, phi_e = phi_span
        forward = phi_e > phi_s
        if (self._cyna_batch_twall is None and self._cyna_batch is None) or not forward:
            return super()._integrate_fieldline(x0, phi_span, **kw)

        # Deduce number of toroidal turns from phi_span length
        N_turns = max(1, round(abs(phi_e - phi_s) / (2 * np.pi)))

        R_s = np.ascontiguousarray([float(x0[0])], dtype=np.float64)
        Z_s = np.ascontiguousarray([float(x0[1])], dtype=np.float64)

        if self._cyna_batch_twall is not None and self._wall_phi is not None:
            counts, R_flat, Z_flat = self._cyna_batch_twall(
                R_s, Z_s,
                float(self._phi_section), N_turns, float(self._DPhi),
                self._BR_flat, self._BPhi_flat, self._BZ_flat,
                self._R_grid, self._Z_grid, self._Phi_ext,
                self._wall_phi, self._wall_R_all, self._wall_Z_all,
            )
        else:
            counts, R_flat, Z_flat = self._cyna_batch(
                R_s, Z_s,
                float(self._phi_section), N_turns, float(self._DPhi),
                self._BR_flat, self._BPhi_flat, self._BZ_flat,
                self._R_grid, self._Z_grid, self._Phi_ext,
                self._wall_R, self._wall_Z,
            )
        # Fixed-stride output: seed 0 occupies slot 0 (1 turn requested)
        if int(counts[0]) < 1:
            return np.array([np.nan, np.nan])
        return np.array([R_flat[0], Z_flat[0]])


class StableManifold(_AcceleratedManifoldBase):
    """Stable manifold of a hyperbolic fixed point.

    Uses the cyna backend by default for forward one-turn integration and
    falls back to SciPy when accelerated integration is unavailable or a
    backward step is required.
    """
    _branch = 'stable'


class UnstableManifold(_AcceleratedManifoldBase):
    """Unstable manifold of a hyperbolic fixed point.

    Uses the cyna backend by default for forward one-turn integration and
    falls back to SciPy when accelerated integration is unavailable or a
    backward step is required.
    """
    _branch = 'unstable'


class ScipyStableManifold(_ManifoldBase):
    """Stable manifold of a hyperbolic fixed point.

    The stable manifold consists of all trajectories that converge to the
    X-point as the map is iterated *forward*.  Equivalently, seeds on the
    stable eigenvector are integrated *backward* in φ.

    Inherits from :class:`_ManifoldBase`.  See its documentation for
    parameter details.
    """

    _branch = 'stable'


class ScipyUnstableManifold(_ManifoldBase):
    """Unstable manifold of a hyperbolic fixed point.

    The unstable manifold consists of all trajectories that diverge from the
    X-point as the map is iterated *forward*.  Seeds on the unstable
    eigenvector are integrated *forward* in φ.

    Inherits from :class:`_ManifoldBase`.  See its documentation for
    parameter details.
    """

    _branch = 'unstable'


# Backward-compat aliases (old backend-explicit names)
CynaStableManifold = StableManifold
CynaUnstableManifold = UnstableManifold
