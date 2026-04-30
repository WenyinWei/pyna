"""Monodromy matrix and variational equation analysis along field-line orbits.



The variational equations describe how a small displacement Î´X evolves:

    dDX_pol/dÏ = A(r,z,Ï) Â· DX_pol,   DX_pol(Ï0) = I



where A is the 2Ã2 Jacobian of the field direction:

    A_ij = ï¿½?RÂ·B_pol_i / BÏ) / âx_j



The monodromy matrix DPm = DX_pol(Ï_end), Ï_end = Ï0 + 2mÂ·Ï, gives the linearized m-turn PoincarÃ© map.

Eigenvalues of DPm: Î»âÂ·Î»â = 1 (area-preserving),

    |Î»| = 1 ï¿½?elliptic (O-point), Î» > 1 ï¿½?hyperbolic (X-point).



DPm evolution (Lie algebra / commutator equation):

    dDPm/dÏ = AÂ·DPm - DPmÂ·A

Used for computing how monodromy changes with perturbation.



Orbit shift under perturbation Î´B:

    dXcyc/dÏ = AÂ·Xcyc + Î´b_pol

    Î´b_pol_R = RÂ·Î´BR/BÏ - RÂ·BRÂ·Î´BÏ/BÏÂ²

    Î´b_pol_Z = RÂ·Î´BZ/BÏ - RÂ·BZÂ·Î´BÏ/BÏÂ²



Reference: W7-X Jacobian analysis notebook (Julia) ï¿½?ported to Python.

"""

from __future__ import annotations



import numpy as np

from pyna.topo._rk4 import rk4_integrate as solve_ivp  # replaced scipy for performance

from scipy.interpolate import interp1d

from typing import Callable, Optional

from dataclasses import dataclass





# ---------------------------------------------------------------------------

# Data class

# ---------------------------------------------------------------------------



@dataclass

class CycleVariationalData:

    """Full monodromy analysis result along a periodic orbit.



    Attributes

    ----------

    phi_arr : ndarray

        Toroidal angle array along orbit.

    trajectory : ndarray, shape (N, 2)

        (R, Z) trajectory.

    DX_pol_arr : ndarray, shape (N, 2, 2)

        Variational matrix DX_pol(Ï) in cylindrical (polar) coords at each Ï.

    DPm_arr : ndarray, shape (N, 2, 2)

        DPm matrix (commutator evolution).

    DPm : ndarray, shape (2, 2)

        Monodromy matrix DPm = DX_pol(Ï_end), Ï_end = Ï0 + 2mÂ·Ï.

    """

    phi_arr: np.ndarray

    trajectory: np.ndarray

    DX_pol_arr: np.ndarray

    DPm_arr: np.ndarray

    DPm: np.ndarray



    @property

    def eigenvalues(self) -> np.ndarray:

        """Eigenvalues of the monodromy matrix."""

        return np.linalg.eigvals(self.DPm)



    @property

    def stability_index(self) -> float:

        """Tr(DPm)/2 for a 2Ã2 symplectic map."""

        return float(np.trace(self.DPm) / 2.0)



    @property

    def Greene_residue(self) -> float:

        """Greene's residue R = (2 - Tr(DPm))/4. R<0: hyperbolic, 0<R<1: elliptic."""

        return float((2.0 - np.trace(self.DPm)) / 4.0)



    def DX_pol_at(self, phi: float) -> np.ndarray:

        """Interpolate variational matrix DX_pol at arbitrary Ï."""

        n = len(self.phi_arr)

        DX_pol_flat = self.DX_pol_arr.reshape(n, 4)

        out = np.zeros(4)

        for k in range(4):

            out[k] = float(interp1d(self.phi_arr, DX_pol_flat[:, k], kind='cubic')(phi))

        return out.reshape(2, 2)



    def DPm_at(self, phi: float) -> np.ndarray:

        """Interpolate DPm matrix at arbitrary Ï."""

        n = len(self.phi_arr)

        DPm_flat = self.DPm_arr.reshape(n, 4)

        out = np.zeros(4)

        for k in range(4):

            out[k] = float(interp1d(self.phi_arr, DPm_flat[:, k], kind='cubic')(phi))

        return out.reshape(2, 2)





# ---------------------------------------------------------------------------

# Build A matrix function

# ---------------------------------------------------------------------------



def build_A_matrix_func(field_func: Callable, eps: float = 1e-4) -> Callable:

    """Build A(r,z,phi) = Jacobian of (RÂ·BR/BÏ, RÂ·BZ/BÏ) w.r.t. (R, Z).



    Uses forward finite differences on field_func.



    Parameters

    ----------

    field_func : callable

        ``field_func(rzphi) ï¿½?(dR/dl, dZ/dl, dphi/dl)``.

    eps : float

        Finite-difference step size.



    Returns

    -------

    callable

        ``A_func(r, z, phi) ï¿½?ndarray shape (2, 2)``.

    """

    def _g(rzphi):

        """Ï-parameterized field direction g = (RÂ·BR/BÏ, RÂ·BZ/BÏ)."""

        f = np.asarray(field_func(rzphi), dtype=float)

        dphi_dl = f[2]

        if abs(dphi_dl) < 1e-30:

            return np.zeros(2)

        return np.array([f[0] / dphi_dl, f[1] / dphi_dl])



    def A_func(r: float, z: float, phi: float) -> np.ndarray:

        rzphi = np.array([r, z, phi])

        g0 = _g(rzphi)



        rzphi_R = np.array([r + eps, z, phi])

        gR = _g(rzphi_R)



        rzphi_Z = np.array([r, z + eps, phi])

        gZ = _g(rzphi_Z)



        A = np.array([

            [(gR[0] - g0[0]) / eps, (gZ[0] - g0[0]) / eps],

            [(gR[1] - g0[1]) / eps, (gZ[1] - g0[1]) / eps],

        ])

        return A



    return A_func





def build_delta_b_pol_func(

    field_func: Callable,

    delta_field_func: Callable,

) -> Callable:

    """Build Î´b_pol(r, z, phi) = perturbation forcing in the orbit equation.



    Î´b_pol_R = RÂ·Î´BR/BÏ - RÂ·BRÂ·Î´BÏ/BÏÂ²

    Î´b_pol_Z = RÂ·Î´BZ/BÏ - RÂ·BZÂ·Î´BÏ/BÏÂ²



    For a field_func returning (dR/dl, dZ/dl, dphi/dl), and similarly for

    delta_field_func, we have:

        BR/|B| = f[0], BZ/|B| = f[1], Bphi/(R|B|) = f[2]

    so Bphi = R * |B| * f[2], BR = |B| * f[0], etc.



    The ratio RÂ·BR/BÏ = f[0] / f[2] (cancels |B|).

    For perturbation Î´B, same ratios but cross terms:

        RÂ·Î´BR/BÏ - RÂ·BRÂ·Î´BÏ/BÏÂ² = Î´f[0]/f[2] - f[0]*Î´f[2]/f[2]Â²



    Parameters

    ----------

    field_func : callable

        Unperturbed field.

    delta_field_func : callable

        Perturbation field (same signature).



    Returns

    -------

    callable

        ``delta_b_pol(r, z, phi) ï¿½?ndarray shape (2,)``.

    """

    def delta_b_pol(r: float, z: float, phi: float) -> np.ndarray:

        rzphi = np.array([r, z, phi])

        f = np.asarray(field_func(rzphi), dtype=float)

        df = np.asarray(delta_field_func(rzphi), dtype=float)

        dphi_dl = f[2]

        if abs(dphi_dl) < 1e-30:

            return np.zeros(2)

        dbR = df[0] / dphi_dl - f[0] * df[2] / dphi_dl ** 2

        dbZ = df[1] / dphi_dl - f[1] * df[2] / dphi_dl ** 2

        return np.array([dbR, dbZ])



    return delta_b_pol





# ---------------------------------------------------------------------------

# Compute monodromy

# ---------------------------------------------------------------------------



def evolve_DPm_along_cycle(

    field_func: Callable,

    orbit,

    n_turns: Optional[int] = None,

    dt_output: float = 0.1,

    rtol: float = 1e-8,

    atol: float = 1e-9,

) -> CycleVariationalData:

    """Compute DPm and full DX_pol / DPm evolution along a periodic orbit.



    Here ``phi_end = phi0 + 2Ï * n_turns`` is the final cylindrical toroidal

    angle (``phi_e`` in the research-notebook notation).



    Integrates simultaneously (Ï-parameterized):

    1. The orbit trajectory (R(Ï), Z(Ï))

    2. The variational equation dDX_pol/dÏ = A(r,z,Ï)Â·DX_pol

    3. The DPm commutator equation dDPm/dÏ = AÂ·DPm - DPmÂ·A



    State vector layout (10 components):

        y[0:2]  = (R, Z)

        y[2:6]  = DX_pol flattened (row-major: 00, 01, 10, 11)

        y[6:10] = DPm flattened



    Parameters

    ----------

    field_func : callable

        ``field_func(rzphi) ï¿½?(dR/dl, dZ/dl, dphi/dl)``.

    orbit : ToroidalPeriodicOrbitTrace

        The periodic orbit to analyze.

    n_turns : int or None

        Number of turns. If None, uses orbit.period_n.

    dt_output : float

        Output spacing in Ï.

    rtol, atol : float

        ODE solver tolerances.



    Returns

    -------

    CycleVariationalData

    """

    from pyna.topo.toroidal_cycle import ToroidalPeriodicOrbitTrace  # avoid circular at import time



    if n_turns is None:

        n_turns = orbit.period_n



    R0, Z0, phi0 = float(orbit.rzphi0[0]), float(orbit.rzphi0[1]), float(orbit.rzphi0[2])

    phi_end = phi0 + n_turns * 2.0 * np.pi



    A_func = build_A_matrix_func(field_func)



    def _g(r, z, phi):

        rzphi = np.array([r, z, phi])

        f = np.asarray(field_func(rzphi), dtype=float)

        dphi_dl = f[2]

        if abs(dphi_dl) < 1e-30:

            return np.zeros(2)

        return np.array([f[0] / dphi_dl, f[1] / dphi_dl])



    def rhs_J_only(phi, y):

        """Phase 1: integrate orbit + DX_pol only (6 components)."""

        r, z = y[0], y[1]

        DX_pol = y[2:6].reshape(2, 2)

        drz = _g(r, z, phi)

        A = A_func(r, z, phi)

        dDX_pol = A @ DX_pol

        return np.concatenate([drz, dDX_pol.flatten()])



    def rhs(phi, y):

        """Phase 2: integrate orbit + DX_pol + DPm (10 components)."""

        r, z = y[0], y[1]

        DX_pol = y[2:6].reshape(2, 2)

        DPm = y[6:10].reshape(2, 2)



        # orbit velocity

        drz = _g(r, z, phi)



        A = A_func(r, z, phi)

        dDX_pol = A @ DX_pol

        dDPm = A @ DPm - DPm @ A



        return np.concatenate([drz, dDX_pol.flatten(), dDPm.flatten()])



    n_out = max(int((phi_end - phi0) / dt_output), 50)

    t_eval = np.linspace(phi0, phi_end, n_out)



    # Phase 1: integrate orbit + DX_pol to get DPm_init = DX_pol(phi_end).

    # DPm initial condition must be DX_pol(phi_end) (not I) ï¿½?see reference:

    #   dummy_Xcycle.ipynb cell 1 / W7X_Jac_change_under_perturbation.ipynb cell 43:

    #   DPm_ivp = solve_ivp(..., [0, 2m*pi], DX_pol_ivp.sol(2*m*pi), ...)

    #   i.e. DPm(phi0) = DX_pol(phi_end).

    y0_phase1 = np.zeros(6)

    y0_phase1[0], y0_phase1[1] = R0, Z0

    y0_phase1[2:6] = np.eye(2).flatten()  # DX_pol(phi0) = I



    sol_phase1 = solve_ivp(

        rhs_J_only,

        (phi0, phi_end),

        y0_phase1,

        t_eval=t_eval,

        rtol=rtol,

        atol=atol,

    )



    if not sol_phase1.success:

        raise RuntimeError(f"Monodromy phase-1 integration failed: {sol_phase1.message}")



    # DPm_init = DX_pol(phi_end) from Phase 1

    DPm_init = sol_phase1.y[2:6, -1].reshape(2, 2)



    # Phase 2: re-integrate with DPm(phi0) = DPm_init = DX_pol(phi_end)

    y0 = np.zeros(10)

    y0[0], y0[1] = R0, Z0

    y0[2:6] = np.eye(2).flatten()  # DX_pol(phi0) = I

    y0[6:10] = DPm_init.flatten()  # DPm(phi0) = DX_pol(phi_end) (NOT identity!)



    sol = solve_ivp(

        rhs,

        (phi0, phi_end),

        y0,

        t_eval=t_eval,

        rtol=rtol,

        atol=atol,

    )



    if not sol.success:

        raise RuntimeError(f"Monodromy integration failed: {sol.message}")



    phi_arr = sol.t

    traj = sol.y[:2].T                              # (N, 2)

    DX_pol_arr = sol.y[2:6].T.reshape(-1, 2, 2)    # (N, 2, 2)

    DPm_arr = sol.y[6:10].T.reshape(-1, 2, 2)      # (N, 2, 2)

    DPm_val = DX_pol_arr[-1]



    return CycleVariationalData(

        phi_arr=phi_arr,

        trajectory=traj,

        DX_pol_arr=DX_pol_arr,

        DPm_arr=DPm_arr,

        DPm=DPm_val,

    )





# ---------------------------------------------------------------------------

# Orbit shift under perturbation

# ---------------------------------------------------------------------------



def orbit_shift_under_perturbation(

    field_func: Callable,

    delta_field_func: Callable,

    orbit,

    monodromy_analysis: CycleVariationalData,

) -> np.ndarray:

    """Compute the orbit position shift under a perturbation Î´B.



    Solves the inhomogeneous variational equation:

        dXcyc/dÏ = AÂ·Xcyc + Î´b_pol(r(Ï), z(Ï), Ï)



    with periodic boundary condition. The initial condition is:

        Xcyc(Ï0) = (DPm - I)^{-1} Â· (-ï¿½?DX_polÂ·DX_pol^{-1}Â·Î´b dÏ)

    which is found by requiring Xcyc(Ï0 + 2Ïn) = Xcyc(Ï0).



    In practice: first integrate with Xcyc(Ï0)=0 to find the particular

    solution Xpart(Ï_end), then set Xcyc(Ï0) = (DPm - I)^{-1} Â· (-Xpart(Ï_end)).



    Parameters

    ----------

    field_func : callable

        Unperturbed field.

    delta_field_func : callable

        Perturbation field Î´B, same signature as field_func.

    orbit : ToroidalPeriodicOrbitTrace

    monodromy_analysis : CycleVariationalData



    Returns

    -------

    ndarray, shape (N, 2)

        Orbit displacement (Î´R(Ï), Î´Z(Ï)) along the orbit.

    """

    A_func = build_A_matrix_func(field_func)

    db_pol_func = build_delta_b_pol_func(field_func, delta_field_func)



    phi_arr = monodromy_analysis.phi_arr

    phi0 = phi_arr[0]

    phi_end = phi_arr[-1]

    R0, Z0 = float(orbit.rzphi0[0]), float(orbit.rzphi0[1])



    # Precompute trajectory interpolants

    traj = monodromy_analysis.trajectory

    r_interp = interp1d(phi_arr, traj[:, 0], kind='cubic')

    z_interp = interp1d(phi_arr, traj[:, 1], kind='cubic')



    def rhs_particular(phi, Xcyc):

        r = float(r_interp(phi))

        z = float(z_interp(phi))

        A = A_func(r, z, phi)

        db = db_pol_func(r, z, phi)

        return A @ Xcyc + db



    # Step 1: integrate with Xcyc(phi0) = 0 ï¿½?particular solution

    sol_part = solve_ivp(

        rhs_particular,

        (phi0, phi_end),

        [0.0, 0.0],

        t_eval=phi_arr,

    )



    DPm = monodromy_analysis.DPm

    Xpart_end = sol_part.y[:, -1]



    # Step 2: periodic BC: X(phi_end) = X(phi0)

    # DPm * X0 + Xpart_end = X0  ï¿½?(DPm - I) * X0 = -Xpart_end

    try:

        X0 = np.linalg.solve(DPm - np.eye(2), -Xpart_end)

    except np.linalg.LinAlgError:

        X0 = np.zeros(2)



    # Step 3: integrate again with correct IC

    sol_full = solve_ivp(

        rhs_particular,

        (phi0, phi_end),

        X0,

        t_eval=phi_arr,

    )



    return sol_full.y.T  # (N, 2)





# ---------------------------------------------------------------------------

# Monodromy change under perturbation

# ---------------------------------------------------------------------------



def monodromy_change_under_perturbation(

    orbit,

    monodromy_analysis: CycleVariationalData,

    orbit_shift: np.ndarray,

    delta_A_func: Callable,

) -> np.ndarray:

    """Compute how the monodromy matrix changes under perturbation Î´B.



    Î´DPm = â«_Ï0^{Ï_end} DX_pol(Ï_end)Â·DX_pol^{-1}(Ï)Â·Î´A_eff(Ï)Â·DX_pol(Ï) dÏ



    where Î´A_eff(Ï) = Î´A(r(Ï), z(Ï), Ï) accounts for the change in the

    A matrix due to the perturbation (both direct Î´A and shift of orbit).



    The integral is evaluated numerically using the trapezoidal rule.



    Parameters

    ----------

    orbit : ToroidalPeriodicOrbitTrace

    monodromy_analysis : CycleVariationalData

    orbit_shift : ndarray, shape (N, 2)

        Orbit displacement from orbit_shift_under_perturbation.

    delta_A_func : callable

        ``delta_A_func(r, z, phi) ï¿½?ndarray (2, 2)`` ï¿½?the change in A

        due to the perturbation Î´B (computed via build_A_matrix_func on Î´B).



    Returns

    -------

    ndarray, shape (2, 2)

        Î´DPm ï¿½?the change in the monodromy matrix.

    """

    phi_arr = monodromy_analysis.phi_arr

    DX_pol_arr = monodromy_analysis.DX_pol_arr

    traj = monodromy_analysis.trajectory

    DPm = monodromy_analysis.DPm



    integrand = np.zeros((len(phi_arr), 2, 2))



    for i, phi in enumerate(phi_arr):

        r, z = traj[i, 0], traj[i, 1]

        dr, dz = orbit_shift[i, 0], orbit_shift[i, 1]

        DX_pol_phi = DX_pol_arr[i]



        # DX_pol(Ï_end) Â· DX_pol^{-1}(Ï) = DPm Â· DX_pol^{-1}(Ï)

        try:

            DX_pol_inv = np.linalg.inv(DX_pol_phi)

        except np.linalg.LinAlgError:

            continue



        dA = delta_A_func(r, z, phi)



        # Contribution to Î´DPm integrand

        integrand[i] = DPm @ DX_pol_inv @ dA @ DX_pol_phi



    # Trapezoidal integration

    dphi = np.diff(phi_arr)

    dM = np.zeros((2, 2))

    for i in range(len(phi_arr) - 1):

        dM += 0.5 * dphi[i] * (integrand[i] + integrand[i + 1])



    return dM





# ---------------------------------------------------------------------------

# Second-order orbit variation

# ---------------------------------------------------------------------------



def second_order_orbit_variation(

    field_func: Callable,

    delta_field_func: Callable,

    orbit,

    monodromy_analysis: CycleVariationalData,

    first_order_shift: np.ndarray,

) -> np.ndarray:

    r"""Compute the second-order orbit position variation Î´Â²X(Ï).



    The second-order variational equation is



        d(Î´Â²Xï¿½?/dÏ = Î£ï¿½?A_{ij} Î´Â²Xï¿½?                    + Î£ï¿½?ï¿½?H_{ijk} Î´Xï¿½?Î´Xï¿½?                    + Î£ï¿½?Î´A_{ij} Î´Xï¿½?

    where:

        - A = âf/âX is the Jacobian of the unperturbed field direction,

        - H_{ijk} = âÂ²f_i/âX_jâX_k is the Hessian of f,

        - Î´A_{ij} = ï¿½?Î´f_i)/âX_j is the Jacobian of the perturbation Î´f,

        - Î´X is the first-order orbit shift (from

          :func:`orbit_shift_under_perturbation`).



    The periodic boundary condition is handled in the same way as in

    :func:`orbit_shift_under_perturbation`: integrate first with zero

    initial condition to find the particular solution, then use

    (DPm ï¿½?I) Î´Â²Xâ = âÎ´Â²X_particular(Ï_end).



    Parameters

    ----------

    field_func : callable

        Unperturbed field function.

    delta_field_func : callable

        First-order perturbation field Î´B (same signature as field_func).

    orbit : ToroidalPeriodicOrbitTrace

        The unperturbed periodic orbit.

    monodromy_analysis : CycleVariationalData

        Monodromy analysis of the unperturbed orbit (from

        :func:`evolve_DPm_along_cycle`).

    first_order_shift : ndarray, shape (N, 2)

        First-order orbit position shift Î´X(Ï), as returned by

        :func:`orbit_shift_under_perturbation`.



    Returns

    -------

    ndarray, shape (N, 2)

        Second-order orbit displacement Î´Â²X(Ï) = (Î´Â²R(Ï), Î´Â²Z(Ï)).



    Notes

    -----

    This function computes the *second-order* correction to the orbit

    position under the perturbation, following the expansion



        X = Xâ + Îµ Î´X + ÎµÂ² Î´Â²X / 2 + O(ÎµÂ³).



    The dependence on ÎµÂ² makes this term important when the first-order

    shift Î´X is large (e.g. near a separatrix or for strong perturbations).

    """

    from scipy.interpolate import interp1d as _interp1d



    A_func = build_A_matrix_func(field_func)

    # Build Jacobian of delta_field_func (Î´A)

    from pyna.topo.variational import _fd_jacobian as _fd_jac



    def delta_A_func(r: float, z: float, phi: float) -> np.ndarray:

        """Finite-difference Jacobian of Î´f w.r.t. (R, Z)."""

        def df(r_, z_, phi_):

            rzphi = np.array([r_, z_, phi_])

            f = np.asarray(field_func(rzphi), dtype=float)

            df_ = np.asarray(delta_field_func(rzphi), dtype=float)

            dphi_dl = f[2]

            if abs(dphi_dl) < 1e-30:

                return np.zeros(2)

            dbR = df_[0] / dphi_dl - f[0] * df_[2] / dphi_dl ** 2

            dbZ = df_[1] / dphi_dl - f[1] * df_[2] / dphi_dl ** 2

            return np.array([dbR, dbZ])

        return _fd_jac(df, np.array([r, z]), phi, 1e-6)



    phi_arr = monodromy_analysis.phi_arr

    phi0 = phi_arr[0]

    phi_end = phi_arr[-1]



    traj = monodromy_analysis.trajectory

    r_interp = _interp1d(phi_arr, traj[:, 0], kind='cubic')

    z_interp = _interp1d(phi_arr, traj[:, 1], kind='cubic')



    # Interpolate Î´X(Ï) ï¿½?the first-order shift

    dX_interp_0 = _interp1d(phi_arr, first_order_shift[:, 0], kind='cubic')

    dX_interp_1 = _interp1d(phi_arr, first_order_shift[:, 1], kind='cubic')



    # Compute Hessian of f along the orbit (expensive; evaluated on-the-fly)

    from pyna.topo.variational import _fd_hessian as _fd_hes



    def _hessian_f(r: float, z: float, phi: float) -> np.ndarray:

        def f2(r_, z_, phi_):

            rzphi = np.array([r_, z_, phi_])

            fv = np.asarray(field_func(rzphi), dtype=float)

            dphi_dl = fv[2]

            if abs(dphi_dl) < 1e-30:

                return np.zeros(2)

            return np.array([fv[0] / dphi_dl, fv[1] / dphi_dl])

        return _fd_hes(f2, np.array([r, z]), phi, 1e-5)



    def rhs_second_order(phi: float, d2X: np.ndarray) -> np.ndarray:

        r = float(r_interp(phi))

        z = float(z_interp(phi))

        dX = np.array([float(dX_interp_0(phi)), float(dX_interp_1(phi))])



        A = A_func(r, z, phi)

        H = _hessian_f(r, z, phi)          # shape (2, 2, 2)

        dA = delta_A_func(r, z, phi)        # shape (2, 2)



        # A @ Î´Â²X

        lhs1 = A @ d2X

        # Î£ï¿½?ï¿½?H_{ijk} Î´Xï¿½?Î´Xï¿½?  ï¿½?shape (2,)

        lhs2 = np.einsum('ijk,j,k->i', H, dX, dX)

        # Î£ï¿½?Î´A_{ij} Î´Xï¿½? ï¿½?shape (2,)

        lhs3 = dA @ dX



        return lhs1 + lhs2 + lhs3



    # Step 1: particular solution with Î´Â²X(Ï0) = 0

    from pyna.topo._rk4 import rk4_integrate as solve_ivp  # replaced scipy for performance as _solve_ivp

    sol_part = _solve_ivp(

        rhs_second_order, (phi0, phi_end), [0.0, 0.0], t_eval=phi_arr,

    )

    d2X_part_end = sol_part.y[:, -1]



    # Step 2: periodic BC:  (DPm ï¿½?I) dÂ²Xâ = âdÂ²X_part_end

    DPm = monodromy_analysis.DPm

    try:

        d2X0 = np.linalg.solve(DPm - np.eye(2), -d2X_part_end)

    except np.linalg.LinAlgError:

        d2X0 = np.zeros(2)



    # Step 3: full solution with correct IC

    sol_full = _solve_ivp(

        rhs_second_order, (phi0, phi_end), d2X0, t_eval=phi_arr,

    )

    return sol_full.y.T  # shape (N, 2)





def monodromy_matrix(

    R_xpt: float,

    Z_xpt: float,

    phi_section: float,

    m_turns: int,

    cache: dict,

    wall,

    fd_eps: float = 1e-4,

    DPhi: float = 0.05,

) -> "np.ndarray":

    """Compute the 2x2 monodromy matrix at (R_xpt, Z_xpt) using cyna C++.



    Uses finite differences: evaluates P^m at (R+/-eps, Z) and (R, Z+/-eps)

    in a single cyna batch call (5 seeds x m_turns, all in parallel).



    Parameters

    ----------

    R_xpt, Z_xpt : float

        X-point coordinates.

    phi_section : float

        Toroidal angle of the Poincare section [rad].

    m_turns : int

        Number of toroidal turns (m in q = m/n -- the orbit period).

    cache : dict

        Field cache dict with keys: 'BR', 'BPhi', 'BZ', 'R_grid', 'Z_grid', 'Phi_grid'.

    wall : WallGeometry

        Wall geometry object with ._phi_centers, ._R, ._Z attributes.

    fd_eps : float

        Finite difference step size [m].

    DPhi : float

        Step size for cyna RK4 integrator [rad].



    Returns

    -------

    DPm : ndarray, shape (2, 2)

        Monodromy matrix (Jacobian of P^m).

    """

    import numpy as np

    try:

        from pyna.toroidal.flt import trace_poincare_batch_twall

        _has_twall = True

    except ImportError:

        _has_twall = False

    try:

        from pyna.toroidal.flt import trace_poincare_batch

        _has_batch = True

    except ImportError:

        _has_batch = False



    if not _has_twall and not _has_batch:

        raise ImportError("pyna._cyna not available")



    # Extract grid

    BR, BPhi, BZ = cache['BR'], cache['BPhi'], cache['BZ']

    R_grid, Z_grid, Phi_grid = cache['R_grid'], cache['Z_grid'], cache['Phi_grid']



    # Extend Phi_grid for periodicity (matches FieldlineTracer convention)

    Phi_ext = np.append(Phi_grid, 2 * np.pi)

    BR_ext  = np.concatenate([BR,   BR[:, :, :1]],  axis=2)

    BPhi_ext= np.concatenate([BPhi, BPhi[:, :, :1]], axis=2)

    BZ_ext  = np.concatenate([BZ,   BZ[:, :, :1]],  axis=2)



    # Flatten to 1-D (required by cyna C++ API)

    BR_flat   = np.ascontiguousarray(BR_ext,   dtype=np.float64).ravel()

    BPhi_flat = np.ascontiguousarray(BPhi_ext, dtype=np.float64).ravel()

    BZ_flat   = np.ascontiguousarray(BZ_ext,   dtype=np.float64).ravel()

    R_g = np.ascontiguousarray(R_grid, dtype=np.float64)

    Z_g = np.ascontiguousarray(Z_grid, dtype=np.float64)

    Phi_g = np.ascontiguousarray(Phi_ext, dtype=np.float64)



    # 5 seeds: center, +-R, +-Z

    R_seeds = np.ascontiguousarray(

        [R_xpt, R_xpt + fd_eps, R_xpt - fd_eps, R_xpt,          R_xpt         ], dtype=np.float64)

    Z_seeds = np.ascontiguousarray(

        [Z_xpt, Z_xpt,          Z_xpt,          Z_xpt + fd_eps, Z_xpt - fd_eps], dtype=np.float64)



    # Output layout: fixed stride ï¿½?each seed gets exactly m_turns slots

    # counts[i] = actual crossings (<= m_turns if particle hits wall)

    if _has_twall and hasattr(wall, '_phi_centers'):

        counts, R_flat_out, Z_flat_out = trace_poincare_batch_twall(

            R_seeds, Z_seeds, float(phi_section), int(m_turns), float(DPhi),

            BR_flat, BPhi_flat, BZ_flat, R_g, Z_g, Phi_g,

            np.ascontiguousarray(wall._phi_centers, dtype=np.float64),

            np.ascontiguousarray(wall._R,           dtype=np.float64),

            np.ascontiguousarray(wall._Z,           dtype=np.float64),

        )

    elif _has_batch:

        wR, wZ = wall.get_section(phi_section)

        counts, R_flat_out, Z_flat_out = trace_poincare_batch(

            R_seeds, Z_seeds, float(phi_section), int(m_turns), float(DPhi),

            BR_flat, BPhi_flat, BZ_flat, R_g, Z_g, Phi_g,

            np.ascontiguousarray(wR, dtype=np.float64),

            np.ascontiguousarray(wZ, dtype=np.float64),

        )

    else:

        raise RuntimeError("No suitable cyna batch function found")



    # Fixed-stride output: base = i * m_turns

    # counts[i] = how many crossings actually occurred (0..m_turns)

    R_final = np.full(5, np.nan)

    Z_final = np.full(5, np.nan)

    for i in range(5):

        n = int(counts[i])

        if n >= m_turns:

            idx = i * m_turns + (m_turns - 1)

            R_final[i] = R_flat_out[idx]

            Z_final[i] = Z_flat_out[idx]



    # FD Jacobian of m-turn Poincare map: DPm = [[dR'/dR, dR'/dZ], [dZ'/dR, dZ'/dZ]]

    DPm = np.array([

        [(R_final[1] - R_final[2]) / (2 * fd_eps),

         (R_final[3] - R_final[4]) / (2 * fd_eps)],

        [(Z_final[1] - Z_final[2]) / (2 * fd_eps),

         (Z_final[3] - Z_final[4]) / (2 * fd_eps)],

    ])

    return DPm


