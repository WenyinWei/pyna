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

Order-k tensor notation
-----------------------
The k-th order derivative tensor of the flow map is

    D^k P^m [i, j₁, …, jₖ] = ∂^k Xᵢ(φ_end) / (∂X₀ⱼ₁ … ∂X₀ⱼₖ)

For k=1 this is the monodromy matrix M (shape (2,2)).
For k=2 this is the tensor T (shape (2,2,2)).
For k=3 this is the tensor Q (shape (2,2,2,2)).

The ODE for the k-th order tensor U^(k) in multi-index notation is:

    dU^(k)_{i,J}/dφ = Σₗ A_{il} U^(k)_{l,J}
                    + Σ_{p=2}^{k} Σ_{partitions S_p of J into p parts}
                        F^(p)_{i,l₁,…,lₚ} Π_q U^(|S_q|)_{lq, S_q}

where F^(p) is the p-th derivative tensor of f w.r.t. x, and the sum
runs over all ways to partition the multi-index J into p non-empty parts.

In practice only k≤3 is implemented here; going beyond that is possible
but the number of terms grows rapidly.

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


def _fd_third_derivative(f, x, phi, eps=1e-4):
    """Compute the 3rd-order derivative tensor ∂³f/∂x³ by finite differences.

    Uses a combination of central-difference formulae.  The result is
    symmetric in the last three indices (j, k, l).

    Parameters
    ----------
    f : callable
        ``f(r, z, phi)`` returning array_like of shape (n,).
    x : array_like, shape (n,)
        State vector [R, Z].
    phi : float
        Current toroidal angle.
    eps : float, optional
        Finite-difference step size.  A larger step than the Hessian is
        recommended because third differences amplify round-off.

    Returns
    -------
    D3 : ndarray, shape (n, n, n, n)
        Third derivative tensor  D3[i, j, k, l] = ∂³fᵢ / (∂xⱼ ∂xₖ ∂xₗ).
        Symmetric in j, k, l.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    D3 = np.zeros((n, n, n, n))

    def _eval(dx):
        xd = x + dx
        return np.asarray(f(xd[0], xd[1], phi), dtype=float)

    for j in range(n):
        for k in range(j, n):
            for l in range(k, n):
                ej = np.zeros(n); ej[j] = eps
                ek = np.zeros(n); ek[k] = eps
                el = np.zeros(n); el[l] = eps

                if j == k == l:
                    # ∂³f/∂xⱼ³ ≈ (-f(x+2e)+2f(x+e)-2f(x-e)+f(x-2e)) / (2eps³)
                    val = (_eval(2*ej) - 2*_eval(ej) + 2*_eval(-ej) - _eval(-2*ej)) / (2 * eps**3)
                elif j == k:
                    # mixed ∂³f/(∂xⱼ²∂xₗ) using cross-central differences
                    # ≈ [f(x+ej+el) - 2f(x+el) + f(x-ej+el)
                    #   - f(x+ej-el) + 2f(x-el) - f(x-ej-el)] / (2*eps² * (2*eps))
                    val = (_eval(ej+el) - 2*_eval(el) + _eval(-ej+el)
                           - _eval(ej-el) + 2*_eval(-el) - _eval(-ej-el)) / (2 * eps**3)
                elif k == l:
                    val = (_eval(ek+ej) - 2*_eval(ej) + _eval(-ek+ej)
                           - _eval(ek-ej) + 2*_eval(-ej) - _eval(-ek-ej)) / (2 * eps**3)
                elif j == l:
                    val = (_eval(ej+ek) - 2*_eval(ek) + _eval(-ej+ek)
                           - _eval(ej-ek) + 2*_eval(-ek) - _eval(-ej-ek)) / (2 * eps**3)
                else:
                    # All distinct: use 8-point formula
                    val = (_eval(ej+ek+el) - _eval(ej+ek-el) - _eval(ej-ek+el)
                           + _eval(ej-ek-el) - _eval(-ej+ek+el) + _eval(-ej+ek-el)
                           + _eval(-ej-ek+el) - _eval(-ej-ek-el)) / (8 * eps**3)

                # Assign all permutations (symmetry in j,k,l)
                for pj, pk, pl in _permutations3(j, k, l):
                    D3[:, pj, pk, pl] = val
    return D3


def _permutations3(j, k, l):
    """Return the (up to 6) permutations of three indices."""
    seen = set()
    result = []
    for p in [(j, k, l), (j, l, k), (k, j, l), (k, l, j), (l, j, k), (l, k, j)]:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


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

    The order-3 variational equation for Q_{ijkl}(φ) = ∂³Xᵢ/∂X₀ⱼ∂X₀ₖ∂X₀ₗ reads

        dQ_{ijkl}/dφ = Σₐ A_{ia} Q_{ajkl}
                      + Σₐ,ᵦ H_{iaβ} [T_{ajk} M_{βl} + T_{ajl} M_{βk} + T_{akl} M_{βj}]
                      + Σₐ,ᵦ,γ D³f_{iaβγ} M_{aj} M_{βk} M_{γl}   (+ all permutations),
        Q(φ₀) = 0.

    Here D³f_{iaβγ} is the third-derivative tensor of f w.r.t. x.

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
    fd_eps3 : float, optional
        Finite-difference step for computing the third-derivative tensor
        D³f (order-3 only).  A slightly larger step than ``fd_eps2`` is
        recommended to keep round-off under control.  Default 1e-4.

    Examples
    --------
    >>> def f(r, z, phi):
    ...     return np.array([-z, r])          # circular motion placeholder
    >>> vq = PoincareMapVariationalEquations(f)
    >>> M = vq.jacobian_matrix([1.8, 0.0], [0.0, 2*np.pi])
    >>> np.linalg.det(M)   # ≈ 1 for area-preserving map
    """

    def __init__(self, field_func, fd_eps=1e-6, fd_eps2=1e-5, fd_eps3=1e-4):
        self.field_func = field_func
        self.fd_eps = fd_eps
        self.fd_eps2 = fd_eps2
        self.fd_eps3 = fd_eps3

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

    def _augmented_rhs_order3(self, phi, state):
        """Augmented RHS for the order-1 + order-2 + order-3 variational system.

        State layout: [x(2), M(4), T(8), Q(16)]

        The third-order ODE is:
            dQ_{ijkl}/dφ = Σₐ A_{ia} Q_{ajkl}
                         + Σₐ,ᵦ H_{iaβ} (T_{ajk} M_{βl} + T_{ajl} M_{βk} + T_{akl} M_{βj})
                         + Σₐ,ᵦ,γ D³f_{iaβγ} M_{aj} M_{βk} M_{γl}
                           (plus the 5 other permutations of (j,k,l))
        Since j,k,l are symmetric, the sum over permutations gives a factor-of-6
        term but only 1 unique term for distinct j,k,l.  We implement it
        symmetrically by summing over all 6 permutations and dividing by the
        number of equal permutations (handled implicitly via the einsum).
        """
        x = state[:2]
        M = state[2:6].reshape(2, 2)
        T = state[6:14].reshape(2, 2, 2)    # T[i,j,k]
        Q = state[14:].reshape(2, 2, 2, 2)  # Q[i,j,k,l]

        f_val = np.asarray(self.field_func(x[0], x[1], phi), dtype=float)
        A = _fd_jacobian(self.field_func, x, phi, self.fd_eps)
        H = _fd_hessian(self.field_func, x, phi, self.fd_eps2)
        D3 = _fd_third_derivative(self.field_func, x, phi, self.fd_eps3)

        dM = A @ M  # (2,2)

        # dT_{ijk} = Σₗ A_{il} T_{ljk} + Σₗ,ₘ H_{ilm} M_{lj} M_{mk}
        dT = np.einsum('il,ljk->ijk', A, T)
        dT += np.einsum('ilm,lj,mk->ijk', H, M, M)

        # dQ_{ijkl} = Σₐ A_{ia} Q_{ajkl}
        dQ = np.einsum('ia,ajkl->ijkl', A, Q)

        # + Σₐ,ᵦ H_{iaβ} (T_{ajk} M_{βl} + T_{ajl} M_{βk} + T_{akl} M_{βj})
        # Term 1: contract over (a,β), free (j,k), contribute to l
        dQ += np.einsum('iab,ajk,bl->ijkl', H, T, M)
        # Term 2: contract over (a,β), free (j,l), contribute to k
        dQ += np.einsum('iab,ajl,bk->ijkl', H, T, M)
        # Term 3: contract over (a,β), free (k,l), contribute to j
        dQ += np.einsum('iab,akl,bj->ijkl', H, T, M)

        # + Σₐ,ᵦ,γ D³f_{iaβγ} M_{aj} M_{βk} M_{γl}
        dQ += np.einsum('iabg,aj,bk,gl->ijkl', D3, M, M, M)

        return np.concatenate([f_val, dM.flatten(), dT.flatten(), dQ.flatten()])

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
            (φ_start, φ_end) — integration range. phi_span should span m full
            toroidal turns: phi_span=(0, m*2*pi) for the m-turn monodromy matrix.
            One toroidal turn is typically 0 to 2π.
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
            Order of the tangent map.

            * ``1`` — returns ``(x_end, M)``; M is the monodromy matrix
              D¹P = ∂X(φ_end)/∂X(φ_start), shape (2, 2).
            * ``2`` — returns ``(x_end, M, T)``; T is the second-derivative
              tensor D²P, shape (2, 2, 2).
            * ``3`` — returns ``(x_end, M, T, Q)``; Q is the third-derivative
              tensor D³P, shape (2, 2, 2, 2).  More expensive than order 2
              because it requires 3rd-order finite differences and 30 state
              variables.

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
        Q : ndarray, shape (2, 2, 2, 2), only returned when order >= 3
            Order-3 third-derivative tensor
            Q[i,j,k,l] = ∂³Xᵢ(φ_end) / (∂X₀ⱼ ∂X₀ₖ ∂X₀ₗ).

        Raises
        ------
        NotImplementedError
            If ``order`` > 3.
        """
        if order > 3:
            raise NotImplementedError(
                "Only order=1, 2, and 3 are currently implemented. "
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

        if order == 2:
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

        # order == 3
        # State: [x(2), M(4), T(8), Q(16)]  — T and Q start as zero
        y0 = np.concatenate([x0, np.eye(2).flatten(), np.zeros(8), np.zeros(16)])
        sol = solve_ivp(
            fun=self._augmented_rhs_order3,
            t_span=phi_span,
            y0=y0,
            **solve_ivp_kwargs,
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")
        x_end = sol.y[:2, -1]
        M = sol.y[2:6, -1].reshape(2, 2)
        T = sol.y[6:14, -1].reshape(2, 2, 2)
        Q = sol.y[14:, -1].reshape(2, 2, 2, 2)
        return x_end, M, T, Q
