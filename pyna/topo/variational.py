"""
Variational equations for the Poincaré map and high-order tangent maps.

For a field-line / Hamiltonian system

    dX/dφ = f(X, φ),   X = (R, Z)

the *variational equations* describe how a small perturbation δX evolves:

    d(δXᵢ)/dφ = Σⱼ (∂fᵢ/∂Xⱼ) δXⱼ           (order 1, linearised flow)

The Jacobian (monodromy) matrix  M = ∂X(φ_end)/∂X(φ_start) is the
fundamental matrix of the linearised flow evaluated over one toroidal turn.

For an area-preserving Poincaré map, det(M) = 1.

Higher-order variational equations are formed by differentiating the ODE
and augmenting the state vector.  The Jacobian of f is approximated by
finite differences so that an arbitrary callable ``f(r, z, phi)`` can be
used without symbolic differentiation.

Classes
-------
PoincareMapVariationalEquations
    Encapsulates the variational ODE system and provides integration methods.
"""

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Finite-difference Jacobian and Hessian helpers
# ---------------------------------------------------------------------------

def _fd_jacobian(f, x, phi, eps=1e-6):
    """Compute the Jacobian ∂f/∂x by central finite differences.

    Parameters
    ----------
    f : callable
        ``f(r, z, phi)`` returning array_like of shape (2,).
    x : array_like, shape (2,)
        State vector [R, Z].
    phi : float
        Current toroidal angle.
    eps : float, optional
        Finite-difference step size.

    Returns
    -------
    J : ndarray, shape (2, 2)
        Jacobian matrix  J[i, j] = ∂fᵢ/∂xⱼ.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    J = np.empty((n, n))
    for j in range(n):
        xp = x.copy()
        xp[j] += eps
        xm = x.copy()
        xm[j] -= eps
        J[:, j] = (np.asarray(f(xp[0], xp[1], phi), dtype=float)
                   - np.asarray(f(xm[0], xm[1], phi), dtype=float)) / (2 * eps)
    return J


def _fd_hessian(f, x, phi, eps=1e-5):
    """Compute the Hessian tensor ∂²f/∂x² by central finite differences.

    Parameters
    ----------
    f : callable
        ``f(r, z, phi)`` returning array_like of shape (n,).
    x : array_like, shape (n,)
        State vector.
    phi : float
        Current toroidal angle.
    eps : float, optional
        Finite-difference step size.

    Returns
    -------
    H : ndarray, shape (n, n, n)
        Hessian tensor  H[i, j, k] = ∂²fᵢ / (∂xⱼ ∂xₖ).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    f0 = np.asarray(f(x[0], x[1], phi), dtype=float)
    H = np.empty((n, n, n))
    for j in range(n):
        for k in range(j, n):
            if j == k:
                xp = x.copy()
                xp[j] += eps
                xm = x.copy()
                xm[j] -= eps
                fp = np.asarray(f(xp[0], xp[1], phi), dtype=float)
                fm = np.asarray(f(xm[0], xm[1], phi), dtype=float)
                H[:, j, j] = (fp - 2 * f0 + fm) / eps**2
            else:
                xpp = x.copy()
                xpp[j] += eps
                xpp[k] += eps
                xpm = x.copy()
                xpm[j] += eps
                xpm[k] -= eps
                xmp = x.copy()
                xmp[j] -= eps
                xmp[k] += eps
                xmm = x.copy()
                xmm[j] -= eps
                xmm[k] -= eps
                fpp = np.asarray(f(xpp[0], xpp[1], phi), dtype=float)
                fpm = np.asarray(f(xpm[0], xpm[1], phi), dtype=float)
                fmp = np.asarray(f(xmp[0], xmp[1], phi), dtype=float)
                fmm = np.asarray(f(xmm[0], xmm[1], phi), dtype=float)
                val = (fpp - fpm - fmp + fmm) / (4 * eps**2)
                H[:, j, k] = val
                H[:, k, j] = val  # symmetry
    return H


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PoincareMapVariationalEquations:
    """Variational equations for the Poincaré map of a 2-D field-line system.

    The field-line ODE is

        dR/dφ = fR(R, Z, φ)
        dZ/dφ = fZ(R, Z, φ)

    The order-1 variational (linearised) equation for the Jacobian matrix
    M(φ) = ∂(R,Z)/∂(R₀,Z₀) reads

        dM/dφ = A(X(φ), φ) · M,   M(φ₀) = I,

    where  A_ij = ∂fᵢ/∂Xⱼ  is the Jacobian of the right-hand side.

    The order-2 variational equation for the second-derivative tensor
    T_{ijk}(φ) = ∂²Xᵢ/∂X₀ⱼ∂X₀ₖ reads

        dT_{ijk}/dφ = Σₗ A_{il} T_{ljk} + Σₗ,ₘ H_{ilm} M_{lj} M_{mk},
        T(φ₀) = 0,

    where H_{ilm} = ∂²fᵢ/∂Xₗ∂Xₘ is the Hessian of the right-hand side.

    Parameters
    ----------
    field_func : callable
        ``field_func(r, z, phi)`` → array_like of shape (2,) giving
        (dR/dφ, dZ/dφ).
    fd_eps : float, optional
        Finite-difference step for computing the Jacobian A.  Default 1e-6.
    fd_eps2 : float, optional
        Finite-difference step for computing the Hessian H (order-2 only).
        Default 1e-5.

    Examples
    --------
    >>> def f(r, z, phi):
    ...     return np.array([-z, r])          # circular motion placeholder
    >>> vq = PoincareMapVariationalEquations(f)
    >>> M = vq.jacobian_matrix([1.8, 0.0], [0.0, 2*np.pi])
    >>> np.linalg.det(M)   # ≈ 1 for area-preserving map
    """

    def __init__(self, field_func, fd_eps=1e-6, fd_eps2=1e-5):
        self.field_func = field_func
        self.fd_eps = fd_eps
        self.fd_eps2 = fd_eps2

    # ------------------------------------------------------------------
    # Internal ODE builders
    # ------------------------------------------------------------------

    def _augmented_rhs_order1(self, phi, state):
        """Augmented RHS for the order-1 variational system.

        State layout: [R, Z, M00, M10, M01, M11]
        (M is stored column-major: first column then second column)
        """
        x = state[:2]
        M = state[2:].reshape(2, 2)

        f_val = np.asarray(self.field_func(x[0], x[1], phi), dtype=float)
        A = _fd_jacobian(self.field_func, x, phi, self.fd_eps)

        dM = A @ M  # dM/dφ = A(x) · M
        return np.concatenate([f_val, dM.flatten()])

    def _augmented_rhs_order2(self, phi, state):
        """Augmented RHS for the order-1 + order-2 variational system.

        State layout: [R, Z,            (2)
                       M00,M10,M01,M11, (4, column-major)
                       T000,T100,       (8, T[:,j,k] column-major over jk)
                       T010,T110,
                       T001,T101,
                       T011,T111]

        The second-order ODE is:
            dT_{ijk}/dφ = Σₗ A_{il} T_{ljk} + Σₗ,ₘ H_{ilm} M_{lj} M_{mk}
        """
        x = state[:2]
        M = state[2:6].reshape(2, 2)
        T = state[6:].reshape(2, 2, 2)   # T[i,j,k]

        f_val = np.asarray(self.field_func(x[0], x[1], phi), dtype=float)
        A = _fd_jacobian(self.field_func, x, phi, self.fd_eps)
        H = _fd_hessian(self.field_func, x, phi, self.fd_eps2)

        dM = A @ M  # shape (2,2)

        # dT_{ijk}/dφ = Σₗ A_{il} T_{ljk} + Σₗ,ₘ H_{ilm} M_{lj} M_{mk}
        # First term: A @ T reshaped appropriately
        dT = np.einsum('il,ljk->ijk', A, T)
        # Second term: H_{ilm} M_{lj} M_{mk}  →  contract l,m
        dT += np.einsum('ilm,lj,mk->ijk', H, M, M)

        return np.concatenate([f_val, dM.flatten(), dT.flatten()])

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def jacobian_matrix(self, x0, phi_span, solve_ivp_kwargs=None):
        """Integrate the linearised equations to obtain the monodromy matrix.

        Parameters
        ----------
        x0 : array_like, shape (2,)
            Initial position [R₀, Z₀].
        phi_span : tuple of float
            (φ_start, φ_end) — integration range (one toroidal turn
            is typically 0 to 2π).
        solve_ivp_kwargs : dict, optional
            Extra keyword arguments forwarded to ``scipy.integrate.solve_ivp``.
            Defaults to ``{"method": "DOP853", "rtol": 1e-9, "atol": 1e-12}``.

        Returns
        -------
        M : ndarray, shape (2, 2)
            Monodromy (Jacobian / fundamental) matrix of the Poincaré map.
            For a Hamiltonian / area-preserving map, det(M) ≈ 1.
        """
        if solve_ivp_kwargs is None:
            solve_ivp_kwargs = {"method": "DOP853", "rtol": 1e-9, "atol": 1e-12}

        x0 = np.asarray(x0, dtype=float)
        # Initial condition: x₀ and M = I
        y0 = np.concatenate([x0, np.eye(2).flatten()])

        sol = solve_ivp(
            fun=self._augmented_rhs_order1,
            t_span=phi_span,
            y0=y0,
            **solve_ivp_kwargs,
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")

        M = sol.y[2:, -1].reshape(2, 2)
        return M

    def tangent_map(self, x0, phi_span, order=1, solve_ivp_kwargs=None):
        """Integrate variational equations up to the given order.

        Parameters
        ----------
        x0 : array_like, shape (2,)
            Initial position.
        phi_span : tuple of float
            (φ_start, φ_end).
        order : int, optional
            Order of the tangent map.  1 returns the Jacobian (monodromy)
            matrix.  2 additionally returns the second-derivative tensor.
            Default 1.
        solve_ivp_kwargs : dict, optional
            Forwarded to ``scipy.integrate.solve_ivp``.

        Returns
        -------
        x_end : ndarray, shape (2,)
            Final position after integration.
        M : ndarray, shape (2, 2)
            Order-1 monodromy matrix DP = ∂X(φ_end)/∂X(φ_start).
        T : ndarray, shape (2, 2, 2), only returned when order >= 2
            Order-2 second-derivative tensor
            T[i,j,k] = ∂²Xᵢ(φ_end) / (∂X₀ⱼ ∂X₀ₖ).
            Returned as the third element of the tuple when ``order=2``.

        Raises
        ------
        NotImplementedError
            If ``order`` > 2.
        """
        if order > 2:
            raise NotImplementedError(
                "Only order=1 and order=2 are currently implemented. "
                "Higher-order variational equations are a planned extension."
            )
        if solve_ivp_kwargs is None:
            solve_ivp_kwargs = {"method": "DOP853", "rtol": 1e-9, "atol": 1e-12}

        x0 = np.asarray(x0, dtype=float)

        if order == 1:
            y0 = np.concatenate([x0, np.eye(2).flatten()])
            sol = solve_ivp(
                fun=self._augmented_rhs_order1,
                t_span=phi_span,
                y0=y0,
                **solve_ivp_kwargs,
            )
            if not sol.success:
                raise RuntimeError(f"solve_ivp failed: {sol.message}")
            x_end = sol.y[:2, -1]
            M = sol.y[2:, -1].reshape(2, 2)
            return x_end, M

        # order == 2
        # State: [x(2), M(4), T(8)]  — T starts as zero tensor
        y0 = np.concatenate([x0, np.eye(2).flatten(), np.zeros(8)])
        sol = solve_ivp(
            fun=self._augmented_rhs_order2,
            t_span=phi_span,
            y0=y0,
            **solve_ivp_kwargs,
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")
        x_end = sol.y[:2, -1]
        M = sol.y[2:6, -1].reshape(2, 2)
        T = sol.y[6:, -1].reshape(2, 2, 2)
        return x_end, M, T
