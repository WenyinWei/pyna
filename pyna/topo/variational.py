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
# Finite-difference Jacobian helper
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
        xp = x.copy(); xp[j] += eps
        xm = x.copy(); xm[j] -= eps
        J[:, j] = (np.asarray(f(xp[0], xp[1], phi), dtype=float)
                   - np.asarray(f(xm[0], xm[1], phi), dtype=float)) / (2 * eps)
    return J


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

    Parameters
    ----------
    field_func : callable
        ``field_func(r, z, phi)`` → array_like of shape (2,) giving
        (dR/dφ, dZ/dφ).
    fd_eps : float, optional
        Finite-difference step for computing the Jacobian A.  Default 1e-6.

    Examples
    --------
    >>> def f(r, z, phi):
    ...     return np.array([-z, r])          # circular motion placeholder
    >>> vq = PoincareMapVariationalEquations(f)
    >>> M = vq.jacobian_matrix([1.8, 0.0], [0.0, 2*np.pi])
    >>> np.linalg.det(M)   # ≈ 1 for area-preserving map
    """

    def __init__(self, field_func, fd_eps=1e-6):
        self.field_func = field_func
        self.fd_eps = fd_eps

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

        Currently order=1 is fully implemented (returns the Jacobian matrix).
        Higher orders (order>1) are planned for future extension.

        Parameters
        ----------
        x0 : array_like, shape (2,)
            Initial position.
        phi_span : tuple of float
            (φ_start, φ_end).
        order : int, optional
            Order of the tangent map.  Only order=1 is supported.  Default 1.
        solve_ivp_kwargs : dict, optional
            Forwarded to ``scipy.integrate.solve_ivp``.

        Returns
        -------
        x_end : ndarray, shape (2,)
            Final position after integration.
        M : ndarray, shape (2, 2)
            Order-1 monodromy matrix.
        """
        if order != 1:
            raise NotImplementedError(
                "Only order=1 is currently implemented. "
                "Higher-order variational equations are a planned extension."
            )
        if solve_ivp_kwargs is None:
            solve_ivp_kwargs = {"method": "DOP853", "rtol": 1e-9, "atol": 1e-12}

        x0 = np.asarray(x0, dtype=float)
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
