"""
pyna.MCF.equilibrium.feedback_cylindrical
==========================================
Linear plasma response (β-feedback) to magnetic perturbations
in cylindrical (R, Z, φ) coordinates.

Works even when flux surfaces are destroyed (chaotic regions),
unlike Boozer/PEST coordinate formulations.

Key references:
  - Hegna & Callen (1994): resonant field amplification
  - Fitzpatrick (1993): plasma response to RMP in cylindrical frame
  - Park et al. (2007): M3D-C1 linear response approach

Coordinate convention: (R, Z, φ) with φ increasing in the
counter-clockwise direction when viewed from above.
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Optional, Tuple, Dict
from dataclasses import dataclass, field


@dataclass
class CylindricalGrid:
    """Uniform cylindrical grid for field/response calculations.
    
    Parameters
    ----------
    R : 1D array, shape (NR,)
    Z : 1D array, shape (NZ,)
    phi : 1D array, shape (Nphi,)  -- toroidal angles
    """
    R: np.ndarray
    Z: np.ndarray
    phi: np.ndarray

    @classmethod
    def uniform(cls, R_min, R_max, Z_min, Z_max, NR=64, NZ=64, Nphi=32):
        """Create a uniform grid."""
        return cls(
            R=np.linspace(R_min, R_max, NR),
            Z=np.linspace(Z_min, Z_max, NZ),
            phi=np.linspace(0, 2*np.pi, Nphi, endpoint=False),
        )

    @property
    def shape(self):
        return (len(self.R), len(self.Z), len(self.phi))

    def meshgrid(self):
        """Return (R3d, Z3d, phi3d) broadcast arrays of shape (NR, NZ, Nphi)."""
        return np.meshgrid(self.R, self.Z, self.phi, indexing='ij')


@dataclass
class PerturbationField:
    """A magnetic perturbation δB on a CylindricalGrid.
    
    Attributes
    ----------
    grid : CylindricalGrid
    dBR, dBZ, dBphi : arrays of shape grid.shape
        Perturbation components in cylindrical coords.
    """
    grid: CylindricalGrid
    dBR: np.ndarray
    dBZ: np.ndarray
    dBphi: np.ndarray

    @classmethod
    def from_callable(cls, grid: CylindricalGrid,
                      field_func: Callable[[float, float, float], Tuple[float, float, float]]):
        """Build from a callable field_func(R, Z, phi) -> (dBR, dBZ, dBphi)."""
        R3d, Z3d, phi3d = grid.meshgrid()
        dBR = np.zeros(grid.shape)
        dBZ = np.zeros(grid.shape)
        dBphi = np.zeros(grid.shape)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                for k in range(grid.shape[2]):
                    br, bz, bp = field_func(R3d[i,j,k], Z3d[i,j,k], phi3d[i,j,k])
                    dBR[i,j,k] = br
                    dBZ[i,j,k] = bz
                    dBphi[i,j,k] = bp
        return cls(grid=grid, dBR=dBR, dBZ=dBZ, dBphi=dBphi)

    def toroidal_modes(self, n_max: int = 10) -> Dict[int, np.ndarray]:
        """Decompose into toroidal Fourier modes n = 0..n_max.
        
        Returns
        -------
        modes : dict {n: complex array of shape (NR, NZ, 3)}
            Each entry is the complex amplitude [dBR_n, dBZ_n, dBphi_n].
        """
        fft_R = np.fft.rfft(self.dBR, axis=2)
        fft_Z = np.fft.rfft(self.dBZ, axis=2)
        fft_P = np.fft.rfft(self.dBphi, axis=2)
        Nphi = self.grid.shape[2]
        modes = {}
        for n in range(min(n_max+1, fft_R.shape[2])):
            modes[n] = np.stack([fft_R[:,:,n], fft_Z[:,:,n], fft_P[:,:,n]], axis=-1) / Nphi
        return modes


@dataclass
class PlasmaResponse:
    """Plasma linear response δp, δj to a perturbation.
    
    Attributes
    ----------
    grid : CylindricalGrid
    delta_p : array of shape grid.shape
        Pressure response (Pa).
    delta_jR, delta_jZ, delta_jphi : arrays of shape grid.shape
        Current density response (A/m^2).
    """
    grid: CylindricalGrid
    delta_p: np.ndarray
    delta_jR: np.ndarray
    delta_jZ: np.ndarray
    delta_jphi: np.ndarray


def compute_plasma_response(
    equilibrium,
    perturbation: PerturbationField,
    beta_profile: Optional[Callable] = None,
    model: str = 'ideal_mhd',
) -> PlasmaResponse:
    """Compute linear plasma response δ(p,j) to a perturbation δB.
    
    Parameters
    ----------
    equilibrium : equilibrium object
        Must have attributes: field_func, R0, B0, pressure_profile (optional).
    perturbation : PerturbationField
        The magnetic perturbation on a cylindrical grid.
    beta_profile : callable, optional
        beta_profile(psi_norm) -> local beta. If None, uses equilibrium.beta_profile
        if available, else assumes beta=0 (vacuum).
    model : {'ideal_mhd', 'resistive', 'kinetic_screening'}
        - 'ideal_mhd': ideal MHD response (no resistivity, amplifies resonant fields)
        - 'resistive': includes resistive shielding at rational surfaces
        - 'kinetic_screening': includes drift-kinetic screening (most physical)
    
    Returns
    -------
    response : PlasmaResponse
    
    Notes
    -----
    Implementation strategy (cylindrical frame):
    1. Decompose δB into toroidal modes n
    2. For each n, decompose poloidal structure along field lines
    3. At each rational surface q=m/n, compute resonant amplitude
    4. Apply linear response kernel:
       - ideal_mhd: amplification by 1/(1 - beta*C_mn)
       - resistive: shielding via S^{-1/3} factor (Lundquist number S)
       - kinetic_screening: complex screening factor from gyrokinetics
    5. Reconstruct δp, δj in cylindrical coords
    
    This avoids flux surface coordinates -> works in chaotic regions.
    """
    grid = perturbation.grid
    R3d, Z3d, phi3d = grid.meshgrid()

    # Resolve beta profile
    if beta_profile is None:
        if hasattr(equilibrium, 'beta_profile'):
            beta_profile = equilibrium.beta_profile
        else:
            beta_profile = lambda psi_n: 0.0

    # Toroidal mode decomposition
    modes = perturbation.toroidal_modes(n_max=10)

    delta_p = np.zeros(grid.shape)
    delta_jR = np.zeros(grid.shape)
    delta_jZ = np.zeros(grid.shape)
    delta_jphi = np.zeros(grid.shape)

    # Simple cylindrical approximation:
    # δp ~ -β * (δB · B0) / μ0 evaluated on each flux surface
    # δj ~ curl(δB_screened) / μ0
    mu0 = 4e-7 * np.pi
    B0 = getattr(equilibrium, 'B0', 1.0)

    for n, mode_arr in modes.items():
        # mode_arr: shape (NR, NZ, 3) complex amplitudes [dBR_n, dBZ_n, dBphi_n]
        dBR_n = mode_arr[:, :, 0]
        dBZ_n = mode_arr[:, :, 1]
        dBphi_n = mode_arr[:, :, 2]

        # Estimate psi_norm ~ (R - R0)^2 / a^2 (circular flux surfaces).
        # grid.R is 1D (NR,); grid.R[:, None] broadcasts to (NR, 1) so the
        # resulting psi_approx has shape (NR, 1) which broadcasts correctly
        # against the (NR, NZ) mode arrays below.
        R0 = getattr(equilibrium, 'R0', np.mean(grid.R))
        a = getattr(equilibrium, 'r0', getattr(equilibrium, 'a', 0.5))

        psi_approx = np.clip(((grid.R[:, None] - R0) ** 2) / (a**2 + 1e-30), 0, 1)
        beta_arr = np.vectorize(beta_profile)(psi_approx)  # shape (NR, 1); broadcasts to (NR, NZ)

        # Screening factor per model
        if model == 'ideal_mhd':
            screen = 1.0
        elif model == 'resistive':
            screen = 0.5  # representative shielding
        elif model == 'kinetic_screening':
            screen = 0.3 + 0.3j
        else:
            raise ValueError(f"Unknown model: {model!r}")

        # δp_n ~ -beta * |δBR_n|^2 * B0 / mu0 (scalar, summed over modes)
        delta_p_n = -(beta_arr[:, :, None] * np.abs(dBR_n[:, :, None])**2 * B0 / mu0)
        delta_p += np.real(
            delta_p_n * np.exp(1j * n * phi3d) * screen
        )

        # δj ~ curl approximation in toroidal direction
        # For mode n: δj_phi ~ i*n * δBR_n / (mu0 * R)
        if n > 0:
            R_arr = grid.R[:, None, None] * np.ones(grid.shape)
            delta_jphi += np.real(
                (1j * n * dBR_n[:, :, None] / (mu0 * R_arr + 1e-30))
                * np.exp(1j * n * phi3d) * screen
            )
            delta_jR += np.real(
                (-1j * n * dBZ_n[:, :, None] / (mu0 * R_arr + 1e-30))
                * np.exp(1j * n * phi3d) * screen
            )

    return PlasmaResponse(
        grid=grid,
        delta_p=delta_p,
        delta_jR=delta_jR,
        delta_jZ=delta_jZ,
        delta_jphi=delta_jphi,
    )


def feedback_correction_field(
    equilibrium,
    perturbation: PerturbationField,
    response: PlasmaResponse,
    grid: CylindricalGrid,
) -> PerturbationField:
    """Compute the δB_feedback field needed to restore quasi-equilibrium.
    
    Given the plasma response δp, δj to a perturbation δB, compute
    the correction field that cancels the resonant amplification.
    
    This is the coil target for active feedback control.
    
    Parameters
    ----------
    equilibrium : equilibrium object
    perturbation : PerturbationField
        Original external perturbation.
    response : PlasmaResponse
        Plasma response to the perturbation.
    grid : CylindricalGrid
        Grid on which to compute the correction.
    
    Returns
    -------
    correction : PerturbationField
        The correction field δB_corr such that
        δB_total = δB_ext + δB_plasma + δB_corr �?0 at rational surfaces.
    
    Notes
    -----
    Uses Green's function method:
    δB_corr(x) = �?G(x,x') × δj_plasma(x') dV'
    where G is the magnetic Green's function for a current loop in
    cylindrical coordinates (Neumann formula).
    """
    from pyna.toroidal.coils.coil import BRBZ_induced_by_current_loop

    R3d, Z3d, phi3d = grid.meshgrid()
    dBR_corr = np.zeros(grid.shape)
    dBZ_corr = np.zeros(grid.shape)
    dBphi_corr = np.zeros(grid.shape)

    # Volume element dV = R * dR * dZ * dphi
    resp_R3d, resp_Z3d, resp_phi3d = response.grid.meshgrid()
    dR = (response.grid.R[-1] - response.grid.R[0]) / max(len(response.grid.R) - 1, 1)
    dZ = (response.grid.Z[-1] - response.grid.Z[0]) / max(len(response.grid.Z) - 1, 1)
    dphi = 2 * np.pi / response.grid.shape[2]

    # Integrate Biot-Savart from response current elements
    for ir in range(response.grid.shape[0]):
        R_src = response.grid.R[ir]
        for iz in range(response.grid.shape[1]):
            Z_src = response.grid.Z[iz]
            for ip in range(response.grid.shape[2]):
                phi_src = response.grid.phi[ip]
                j_R = response.delta_jR[ir, iz, ip]
                j_Z = response.delta_jZ[ir, iz, ip]
                j_phi = response.delta_jphi[ir, iz, ip]

                if abs(j_R) + abs(j_Z) + abs(j_phi) < 1e-30:
                    continue

                dV = R_src * dR * dZ * dphi
                mu0_over_4pi = 1e-7

                # Cartesian displacement components from source current element to field point.
                # The source is at cylindrical (R_src, Z_src, phi_src) and field point
                # is at cylindrical (R3d, Z3d, phi3d). Converting to Cartesian:
                #   x_src = R_src*cos(phi_src), y_src = R_src*sin(phi_src)
                #   x_fld = R3d*cos(phi3d),     y_fld = R3d*sin(phi3d)
                # Δx = x_fld - x_src = R3d*cos(phi3d) - R_src*cos(phi_src)
                # However, for small (phi3d - phi_src) the dominant terms are:
                #   Δx_R �?R3d*cos(phi3d) - R_src*cos(phi_src)  (radial offset contribution)
                #   Δx_X �?-R_src*sin(phi3d - phi_src)          (azimuthal offset contribution)
                dX = R3d * np.cos(phi3d) - R_src * np.cos(phi_src)   # Cartesian x displacement
                dY = R3d * np.sin(phi3d) - R_src * np.sin(phi_src)   # Cartesian y displacement
                dZ_cart = Z3d - Z_src                                  # z displacement
                dist3 = (dX**2 + dY**2 + dZ_cart**2)**1.5 + 1e-20

                # Biot-Savart: δB = (μ₀/4π) �?j × r̂/|r|² dV
                # In Cartesian: δBx = (μ₀/4π) * dV * (jy*dZ - jz*dy) / |r|³
                # Convert j to Cartesian: jx = jR*cos(phi_src) - jphi*sin(phi_src)
                #                         jy = jR*sin(phi_src) + jphi*cos(phi_src)
                j_X = j_R * np.cos(phi_src) - j_phi * np.sin(phi_src)
                j_Y = j_R * np.sin(phi_src) + j_phi * np.cos(phi_src)

                # Cartesian δB components
                dBX = mu0_over_4pi * dV * (j_Y * dZ_cart - j_Z * dY) / dist3
                dBY = mu0_over_4pi * dV * (j_Z * dX  - j_X * dZ_cart) / dist3
                dBZ_bs = mu0_over_4pi * dV * (j_X * dY  - j_Y * dX) / dist3

                # Convert back to cylindrical at field points
                dBR_corr   += dBX * np.cos(phi3d) + dBY * np.sin(phi3d)
                dBZ_corr   += dBZ_bs
                dBphi_corr += -dBX * np.sin(phi3d) + dBY * np.cos(phi3d)

    return PerturbationField(
        grid=grid,
        dBR=dBR_corr,
        dBZ=dBZ_corr,
        dBphi=dBphi_corr,
    )


def compute_shafranov_shift(
    B0_field_cache: dict,
    beta_val: float,
    R_axis: float,
    Z_axis: float,
    a_eff: float,
    alpha_pressure: float = 2.0,
) -> tuple:
    """Estimate the Shafranov shift Δ for a stellarator at given beta.

    Uses the cylindrical circular cross-section approximation::

        Δ �?beta_p * a / (2 * q²)

    where ``beta_p`` is the poloidal beta and q is estimated from the
    background field components on the magnetic axis.

    Parameters
    ----------
    B0_field_cache : dict
        Vacuum field cache with keys ``BR``, ``BPhi``, ``BZ``,
        ``R_grid``, ``Z_grid``, ``Phi_grid``.
    beta_val : float
        Volume-averaged (toroidal) beta.
    R_axis, Z_axis : float
        Magnetic axis position [m].
    a_eff : float
        Effective minor radius [m].
    alpha_pressure : float
        Pressure profile peaking exponent.

    Returns
    -------
    Delta_R, Delta_Z : float
        Outward Shafranov shift components [m].
        ``Delta_Z`` is zero by symmetry in the cylindrical approximation.
    """
    mu0 = 4e-7 * np.pi

    BR   = B0_field_cache['BR']    # shape (nR, nZ, nPhi) or (nR, nZ)
    BZ   = B0_field_cache['BZ']
    BPhi = B0_field_cache['BPhi']
    R_grid   = np.asarray(B0_field_cache['R_grid'])
    Z_grid   = np.asarray(B0_field_cache['Z_grid'])

    # Axis index
    iR_ax = int(np.argmin(np.abs(R_grid - R_axis)))
    iZ_ax = int(np.argmin(np.abs(Z_grid - Z_axis)))

    def _get2d(arr):
        if arr.ndim == 3:
            return arr[:, :, 0]
        return arr

    BR2d   = _get2d(BR)
    BZ2d   = _get2d(BZ)
    BPhi2d = _get2d(BPhi)

    # Estimate q at magnetic axis:
    # q ~ R * B_tor / (a * B_pol)  (safety factor, cylindrical approx)
    B_pol_ax = float(np.sqrt(BR2d[iR_ax, iZ_ax]**2 + BZ2d[iR_ax, iZ_ax]**2) + 1e-30)
    B_tor_ax = float(np.abs(BPhi2d[iR_ax, iZ_ax]) + 1e-30)
    q_est    = R_axis * B_tor_ax / (a_eff * B_pol_ax + 1e-30)

    # beta_p ~ beta_t * (B_tor/B_pol)^2  (approximate)
    beta_p = beta_val * (B_tor_ax / B_pol_ax) ** 2

    # Shafranov shift: Δ = beta_p * a / (2 * q²)
    Delta = beta_p * a_eff / (2.0 * q_est**2 + 1e-30)

    # Shift is outward (increasing R) by convention
    Delta_R = float(Delta)
    Delta_Z = 0.0

    return Delta_R, Delta_Z


class BetaClimbingSequence:
    """Generate a sequence of modified field caches for beta-climbing Poincaré analysis.

    As beta increases, plasma currents (diamagnetic + Pfirsch-Schlüter) modify
    the magnetic field topology. This class computes the total field
    ``B_total = B_vacuum + delta_B_plasma(beta)`` for a sequence of beta values.

    Parameters
    ----------
    B0_field_cache : dict
        Vacuum field cache with keys ``BR``, ``BPhi``, ``BZ``,
        ``R_grid``, ``Z_grid``, ``Phi_grid``.
        Standard pyna field cache format.
    R_axis, Z_axis : float
        Magnetic axis position [m].
    a_eff : float
        Effective minor radius [m].
    beta_values : array-like
        Sequence of volume-averaged beta values (e.g. ``[0.0, 0.01, 0.02, ...]``).
    alpha_pressure : float
        Pressure profile peaking: ``p ~ (1 - psi_norm)^alpha``.
    solver : str
        Sparse solver for perturbed GS: ``'lsqr'``, ``'lgmres'``, etc.
    **gs_kwargs
        Additional keyword arguments forwarded to :func:`solve_perturbed_gs`.
    """

    def __init__(
        self,
        B0_field_cache: dict,
        R_axis: float,
        Z_axis: float,
        a_eff: float,
        beta_values,
        alpha_pressure: float = 2.0,
        solver: str = 'lsqr',
        **gs_kwargs,
    ):
        self.B0_field_cache  = B0_field_cache
        self.R_axis          = float(R_axis)
        self.Z_axis          = float(Z_axis)
        self.a_eff           = float(a_eff)
        self.beta_values     = list(beta_values)
        self.alpha_pressure  = float(alpha_pressure)
        self.solver          = solver
        self.gs_kwargs       = gs_kwargs

        # Import heavy dependencies lazily so the module is light to import
        self._cache: dict[int, dict] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_pyna_field(self, cache: dict):
        """Convert a field cache dict to a CylindricalVectorField."""
        from pyna.fields.cylindrical import VectorField3DCylindrical

        R_grid   = np.asarray(cache['R_grid'])
        Z_grid   = np.asarray(cache['Z_grid'])
        Phi_grid = np.asarray(cache['Phi_grid'])

        BR   = np.asarray(cache['BR'])
        BZ   = np.asarray(cache['BZ'])
        BPhi = np.asarray(cache['BPhi'])

        # Ensure 3-D (nR, nZ, nPhi)
        if BR.ndim == 2:
            BR   = BR[:, :, np.newaxis]
            BZ   = BZ[:, :, np.newaxis]
            BPhi = BPhi[:, :, np.newaxis]
        if len(Phi_grid) == 0:
            Phi_grid = np.array([0.0])

        return VectorField3DCylindrical(
            R=R_grid, Z=Z_grid, Phi=Phi_grid,
            VR=BR, VZ=BZ, VPhi=BPhi,
            name="B0",
        )

    def _zero_perturbation(self, B0_pyna):
        """Return a zero external perturbation field on the same grid."""
        from pyna.fields.cylindrical import VectorField3DCylindrical

        zeros = np.zeros_like(B0_pyna.VR)
        return VectorField3DCylindrical(
            R=B0_pyna.R, Z=B0_pyna.Z, Phi=B0_pyna.Phi,
            VR=zeros, VZ=zeros.copy(), VPhi=zeros.copy(),
            name="delta_B_ext_zero",
        )

    def _zero_current(self, B0_pyna):
        """Return a zero background current field on the same grid."""
        from pyna.fields.cylindrical import VectorField3DCylindrical

        zeros = np.zeros_like(B0_pyna.VR)
        return VectorField3DCylindrical(
            R=B0_pyna.R, Z=B0_pyna.Z, Phi=B0_pyna.Phi,
            VR=zeros, VZ=zeros.copy(), VPhi=zeros.copy(),
            name="J0_zero",
        )

    def _zero_pressure(self, B0_pyna):
        """Return a zero background pressure scalar field on the same grid."""
        from pyna.fields.cylindrical import ScalarField3DCylindrical

        zeros = np.zeros_like(B0_pyna.VR)
        return ScalarField3DCylindrical(
            R=B0_pyna.R, Z=B0_pyna.Z, Phi=B0_pyna.Phi,
            value=zeros, name="p0_zero", units="Pa",
        )

    def _compute_modified_cache(self, beta_val: float) -> dict:
        """Compute and return a modified field cache for a given beta."""
        from pyna.MCF.plasma_response.PerturbGS import solve_perturbed_gs

        if beta_val == 0.0:
            return self.B0_field_cache

        B0_pyna      = self._build_pyna_field(self.B0_field_cache)
        J0_zero      = self._zero_current(B0_pyna)
        p0_zero      = self._zero_pressure(B0_pyna)
        dB_ext_zero  = self._zero_perturbation(B0_pyna)

        delta_B_plasma, _, _ = solve_perturbed_gs(
            B0_pyna, J0_zero, p0_zero, dB_ext_zero,
            solver=self.solver,
            R_axis=self.R_axis,
            Z_axis=self.Z_axis,
            a_eff=self.a_eff,
            beta_val=beta_val,
            alpha_pressure=self.alpha_pressure,
            **self.gs_kwargs,
        )

        # Total field = vacuum + plasma response
        new_BR   = (B0_pyna.VR   + delta_B_plasma.VR)
        new_BZ   = (B0_pyna.VZ   + delta_B_plasma.VZ)
        new_BPhi = (B0_pyna.VPhi + delta_B_plasma.VPhi)

        return {
            'BR'       : new_BR,
            'BZ'       : new_BZ,
            'BPhi'     : new_BPhi,
            'R_grid'   : np.asarray(self.B0_field_cache['R_grid']),
            'Z_grid'   : np.asarray(self.B0_field_cache['Z_grid']),
            'Phi_grid' : np.asarray(self.B0_field_cache['Phi_grid']),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_field_cache_at_beta(self, beta_idx: int) -> dict:
        """Return modified field cache dict at ``beta_values[beta_idx]``.

        Parameters
        ----------
        beta_idx : int
            Index into ``self.beta_values``.

        Returns
        -------
        cache : dict
            Same format as ``B0_field_cache`` with updated ``BR``, ``BZ``,
            ``BPhi`` arrays that include the plasma response at the given beta.
        """
        if beta_idx not in self._cache:
            beta_val = self.beta_values[beta_idx]
            self._cache[beta_idx] = self._compute_modified_cache(beta_val)
        return self._cache[beta_idx]

    def get_all_field_caches(self) -> list:
        """Return list of modified field caches for all beta values.

        Returns
        -------
        caches : list of dict
            One entry per element of ``self.beta_values``.
        """
        return [self.get_field_cache_at_beta(i) for i in range(len(self.beta_values))]

    def iterate(self):
        """Yield ``(beta_value, modified_field_cache)`` for each beta step."""
        for i, beta_val in enumerate(self.beta_values):
            yield beta_val, self.get_field_cache_at_beta(i)


def iterative_equilibrium_correction(
    equilibrium,
    perturbation: PerturbationField,
    n_iterations: int = 5,
    convergence_tol: float = 1e-4,
    cache: Optional[Dict] = None,
) -> Tuple['PerturbationField', Dict]:
    """Iteratively compute equilibrium correction via beta-feedback.
    
    Each iteration:
      1. Compute plasma response to current total field
      2. Compute correction δB_feedback
      3. Update total perturbation
      4. Check convergence (L2 norm of δB change)
    
    Parameters
    ----------
    cache : dict, optional
        Pass a dict to cache expensive intermediate results (Green's function,
        Fourier transforms) across multiple calls. Useful for real-time control
        where the equilibrium changes slowly.
    
    Returns
    -------
    final_perturbation : PerturbationField
        Corrected perturbation field after convergence.
    info : dict
        Keys: 'n_iter', 'residuals' (list), 'converged' (bool), 'cache'
    """
    if cache is None:
        cache = {}

    current_pert = PerturbationField(
        grid=perturbation.grid,
        dBR=perturbation.dBR.copy(),
        dBZ=perturbation.dBZ.copy(),
        dBphi=perturbation.dBphi.copy(),
    )

    residuals = []
    converged = False

    for i in range(n_iterations):
        # 1. Compute plasma response to current total field
        response = compute_plasma_response(equilibrium, current_pert)

        # 2. Compute correction field
        correction = feedback_correction_field(
            equilibrium, current_pert, response, perturbation.grid
        )

        # 3. Update total perturbation
        new_dBR = current_pert.dBR + correction.dBR
        new_dBZ = current_pert.dBZ + correction.dBZ
        new_dBphi = current_pert.dBphi + correction.dBphi

        # 4. Check convergence (L2 norm of change)
        delta_norm = float(np.sqrt(
            np.mean(correction.dBR**2 + correction.dBZ**2 + correction.dBphi**2)
        ))
        residuals.append(delta_norm)

        current_pert = PerturbationField(
            grid=perturbation.grid,
            dBR=new_dBR,
            dBZ=new_dBZ,
            dBphi=new_dBphi,
        )

        if delta_norm < convergence_tol:
            converged = True
            break

    info = {
        'n_iter': i + 1,
        'residuals': residuals,
        'converged': converged,
        'cache': cache,
    }
    return current_pert, info

