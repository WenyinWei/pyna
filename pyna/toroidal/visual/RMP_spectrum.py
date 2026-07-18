"""RMP Fourier spectrum analysis and island width visualization.

For a StellaratorSimple, compute resonant components analytically:
  - Resonant surface location from q(ψ) = m/n0
  - Island half-width via Rutherford formula
  - O-point phase from the RMP field structure

O-point Phase Convention
------------------------
For resonant component b_{m,-n} = |b|·exp(iφ) in the Fourier expansion
  δBψ = Σ b_{mn} exp(i(mθ* + nφ))

The fixed points of the Poincaré map (one toroidal turn) satisfy:
  δBψ = 0  →  mθ* − nφ + φ_mn = ±π/2

Stability analysis (q' > 0):
  O-point: mθ_O + φ_mn = −π/2  →  θ_O = (−π/2 − φ_mn)/m
  X-point: mθ_X + φ_mn = +π/2  →  θ_X = (+π/2 − φ_mn)/m

Reference: Rutherford (1973); Nardon (2007) thesis App. A;

General φ-section O/X-point formula
-------------------------------------
At an arbitrary toroidal angle φ, the m O-points lie at poloidal angles:

    θ_O^(k)(φ) = [nφ − π/2 − arg(b_{m,−n})] / m  +  2πk/m,   k = 0…m−1

and the m X-points at:

    θ_X^(k)(φ) = [nφ + π/2 − arg(b_{m,−n})] / m  +  2πk/m,   k = 0…m−1

All angles in radians; results should be taken mod 2π.
For reversed shear (q' < 0) swap O ↔ X.

References
----------
Boozer, Phys. Fluids B 3 (1991) — resonance condition
Rutherford, Phys. Fluids 16 (1973) — island width formula
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Union
import matplotlib.pyplot as plt

from .equilibrium import ISLAND_CMAPS


TWOPI = 2.0 * np.pi


def island_fixed_points(
    m: int,
    n: int,
    b_mn: complex,
    phi: Union[float, np.ndarray],
    q_prime_sign: int = 1,
) -> dict:
    """Return poloidal angles of all O-points and X-points at toroidal angle φ.

    For the resonant component b_{m,−n} of the RMP field, the island fixed
    points at an arbitrary toroidal cross-section φ satisfy:

        mθ* − nφ + arg(b_{m,−n}) = ±π/2   (mod 2π)

    For q' > 0 (normal shear):
        O-points: mθ_O^(k) = nφ − π/2 − arg(b)   →  k = 0 … m−1
        X-points: mθ_X^(k) = nφ + π/2 − arg(b)

    Parameters
    ----------
    m, n : int
        Poloidal and toroidal mode numbers.
    b_mn : complex
        Fourier coefficient b_{m,−n} from the RMP spectrum.
    phi : float or array_like
        Toroidal angle(s) φ in radians at which to evaluate (can be scalar or
        1-D array for a sweep over multiple sections).
    q_prime_sign : int
        +1 for normal shear (q' > 0, default), −1 for reversed shear.
        Reversed shear swaps O and X.

    Returns
    -------
    dict with keys:
        'phi'      : input toroidal angles, shape (N,)
        'theta_O'  : O-point poloidal angles, shape (N, m)  — each row has m O-points
        'theta_X'  : X-point poloidal angles, shape (N, m)
        'theta_O_deg', 'theta_X_deg' : same in degrees

    Examples
    --------
    >>> pts = island_fixed_points(m=2, n=1, b_mn=0.002+0j, phi=0.0)
    >>> print(np.degrees(pts['theta_O']))   # O-point angles at φ=0

    >>> phis = np.linspace(0, 2*np.pi, 100)
    >>> pts = island_fixed_points(m=2, n=1, b_mn=0.002+0j, phi=phis)
    >>> # pts['theta_O'] shape: (100, 2)  — 2 O-points per section
    """
    arg_b = np.angle(b_mn)
    phi = np.atleast_1d(np.asarray(phi, dtype=float))  # shape (N,)
    N = len(phi)

    # Base angle before distributing k branches
    # q' > 0: O-point at mθ = nφ − π/2 − arg(b)
    #         X-point at mθ = nφ + π/2 − arg(b)
    if q_prime_sign >= 0:
        base_O = n * phi - np.pi / 2 - arg_b   # shape (N,)
        base_X = n * phi + np.pi / 2 - arg_b
    else:
        base_O = n * phi + np.pi / 2 - arg_b   # swap for reversed shear
        base_X = n * phi - np.pi / 2 - arg_b

    # k branches: k = 0, 1, …, m−1  → add 2πk
    k = np.arange(m)  # shape (m,)
    theta_O = (base_O[:, None] + 2 * np.pi * k[None, :]) / m  # (N, m)
    theta_X = (base_X[:, None] + 2 * np.pi * k[None, :]) / m  # (N, m)

    # Normalize to [0, 2π)
    theta_O = theta_O % (2 * np.pi)
    theta_X = theta_X % (2 * np.pi)

    return {
        'phi':         phi,
        'theta_O':     theta_O,
        'theta_X':     theta_X,
        'theta_O_deg': np.degrees(theta_O),
        'theta_X_deg': np.degrees(theta_X),
    }


def radial_rmp_field_template(
    m: int,
    n: int,
    amplitude: float = 1.0,
    phase: float = 0.0,
    *,
    axis_R: float,
    axis_Z: float = 0.0,
):
    """Return a divergence-free circular-shell radial RMP field template.

    In local circular coordinates ``R = axis_R + r*cos(theta)`` and
    ``Z = axis_Z + r*sin(theta)``, the returned cylindrical field has

        delta B_r = amplitude * cos(m*theta - n*phi + phase)

    and compensating poloidal/toroidal components chosen so that
    ``div(delta B) = 0`` exactly in the circular-shell metric.  The helper is
    meant for resonant-surface test fields away from the magnetic axis.  Under
    the FFT convention used by :func:`find_resonant_components_analytic`, the
    analytic circular-surface coefficient ``b_{m,-n}`` has phase ``phase`` and
    magnitude ``amplitude/2``.  For ``m = 1`` a toroidal component is required
    to cancel the theta-independent part of ``div(delta B)``.
    """

    m_int = int(m)
    n_int = int(n)
    if m_int <= 0 or n_int <= 0:
        raise ValueError("m and n must be positive resonant mode numbers")
    amp = float(amplitude)
    phase0 = float(phase)
    center_R = float(axis_R)
    center_Z = float(axis_Z)

    def delta_B_func(R, Z, phi):
        R_arr = np.asarray(R, dtype=float)
        Z_arr = np.asarray(Z, dtype=float)
        phi_arr = np.asarray(phi, dtype=float)
        x = R_arr - center_R
        z = Z_arr - center_Z
        r_minor = np.hypot(x, z)
        theta = np.arctan2(z, x)
        phase_m = m_int * theta - n_int * phi_arr + phase0
        radial = amp * np.cos(phase_m)

        theta_plus = (m_int + 1) * theta - n_int * phi_arr + phase0
        if m_int == 1:
            poloidal_flux = -amp * (
                center_R * np.sin(phase_m)
                + 0.5 * r_minor * np.sin(theta_plus)
            )
            toroidal = amp * np.sin(-n_int * phi_arr + phase0) / float(n_int)
        else:
            theta_minus = (m_int - 1) * theta - n_int * phi_arr + phase0
            poloidal_flux = -amp * (
                center_R * np.sin(phase_m) / float(m_int)
                + r_minor * (
                    np.sin(theta_plus) / float(m_int + 1)
                    + np.sin(theta_minus) / float(m_int - 1)
                )
            )
            toroidal = np.zeros_like(radial, dtype=float)
        poloidal = poloidal_flux / np.maximum(R_arr, 1.0e-300)

        BR = radial * np.cos(theta) - poloidal * np.sin(theta)
        BZ = radial * np.sin(theta) + poloidal * np.cos(theta)
        return np.array([
            BR,
            BZ,
            toroidal + np.zeros_like(radial, dtype=float),
        ])

    delta_B_func.divergence_free = True
    return delta_B_func


def compose_magnetic_perturbations(*delta_B_funcs: Callable) -> Callable:
    """Return the linear superposition of several ``delta_B(R, Z, phi)`` callables."""

    funcs = tuple(delta_B_funcs)

    def delta_B_sum(R, Z, phi):
        shape = np.broadcast(np.asarray(R), np.asarray(Z), np.asarray(phi)).shape
        out = np.zeros((3,) + shape, dtype=float)
        for func in funcs:
            out = out + np.asarray(func(R, Z, phi), dtype=float)
        return out

    delta_B_sum.divergence_free = all(bool(getattr(func, "divergence_free", False)) for func in funcs)
    return delta_B_sum


@dataclass(frozen=True)
class DivergenceDiagnostic:
    """Numerical divergence check for a circular-shell perturbation field."""

    max_abs: float
    rms: float
    scale: float
    relative_max: float
    relative_rms: float


@dataclass(frozen=True)
class RMPnRMPModeRow:
    """One Fourier mode classified as resonant RMP or non-resonant magnetic perturbation."""

    m: int
    n: int
    coefficient: complex
    amplitude: float
    phase: float
    detuning: float
    kind: str

    @property
    def phase_deg(self) -> float:
        return float(np.degrees(self.phase))


@dataclass(frozen=True)
class NonResonantContributionRow:
    """One non-resonant mode contribution to the total field-line response."""

    m: int
    n: int
    detuning: float
    radial_velocity_coefficient: complex
    poloidal_velocity_coefficient: complex
    delta_r_coefficient: complex
    delta_theta_coefficient: complex
    radial_response_weight: float
    cumulative_fraction: float

    @property
    def phase_deg(self) -> float:
        return float(np.degrees(np.angle(self.delta_r_coefficient)))


@dataclass(frozen=True)
class NonResonantFieldlineResponse:
    """Total nRMP field-line response from all non-resonant Fourier modes.

    The response is not a single selected mode.  It is the full sum over every
    mode with ``m*iota+n`` outside the resonance tolerance.  Contribution rows
    are diagnostics for ranking and convergence, not a replacement for the
    complete deformation spectrum.
    """

    velocity: "FieldlineVelocitySpectrum"
    deformation: Any
    resonance_tol: float
    include_shear: bool
    regularise_eps: float
    nonresonant_mask: np.ndarray
    detuning: np.ndarray
    poloidal_velocity_coefficients: np.ndarray

    @property
    def n_total_modes(self) -> int:
        return int(self.velocity.radial_spectrum.m.size)

    @property
    def n_nonresonant_modes(self) -> int:
        return int(np.count_nonzero(self.nonresonant_mask))

    @property
    def n_resonant_modes(self) -> int:
        return int(self.n_total_modes - self.n_nonresonant_modes)

    @property
    def total_radial_response_l2(self) -> float:
        return float(np.sqrt(np.nansum(np.abs(self.deformation.delta_r) ** 2)))

    @property
    def max_radial_response(self) -> float:
        return float(np.nanmax(np.abs(self.deformation.delta_r))) if self.deformation.delta_r.size else 0.0

    def real_fields(
        self,
        theta: np.ndarray | None = None,
        phi: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the total ``delta_r`` and ``delta_theta`` on a grid."""

        theta_vals = self.velocity.theta if theta is None else np.asarray(theta, dtype=float)
        phi_vals = self.velocity.phi if phi is None else np.asarray(phi, dtype=float)
        theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals, indexing="xy")
        return (
            theta_grid,
            phi_grid,
            self.deformation.real_field_r(theta_grid, phi_grid),
            self.deformation.real_field_theta(theta_grid, phi_grid),
        )

    def contribution_rows(
        self,
        *,
        top: int | None = None,
        min_weight: float = 0.0,
    ) -> list[NonResonantContributionRow]:
        """Return non-resonant contribution diagnostics sorted by ``|delta_r_mn|``."""

        keep_idx = np.flatnonzero(self.nonresonant_mask)
        delta_r = np.asarray(self.deformation.delta_r, dtype=complex)
        delta_theta = np.asarray(self.deformation.delta_theta, dtype=complex)
        weights = np.abs(delta_r)
        total = float(np.nansum(weights * weights))
        order = sorted(
            [int(i) for i in range(weights.size) if np.isfinite(weights[i]) and weights[i] >= float(min_weight)],
            key=lambda i: float(weights[i]),
            reverse=True,
        )
        if top is not None:
            order = order[:int(top)]

        rows: list[NonResonantContributionRow] = []
        cumulative = 0.0
        for response_i in order:
            cumulative += float(weights[response_i] ** 2)
            source_i = int(keep_idx[response_i])
            rows.append(NonResonantContributionRow(
                m=int(self.velocity.radial_spectrum.m[source_i]),
                n=int(self.velocity.radial_spectrum.n[source_i]),
                detuning=float(self.detuning[source_i]),
                radial_velocity_coefficient=complex(self.velocity.radial_spectrum.dBr[source_i]),
                poloidal_velocity_coefficient=complex(self.poloidal_velocity_coefficients[source_i]),
                delta_r_coefficient=complex(delta_r[response_i]),
                delta_theta_coefficient=complex(delta_theta[response_i]),
                radial_response_weight=float(weights[response_i]),
                cumulative_fraction=float(cumulative / total) if total > 0.0 else 0.0,
            ))
        return rows

    def cumulative_contribution(self) -> tuple[np.ndarray, np.ndarray]:
        """Return sorted mode counts and cumulative ``|delta_r_mn|^2`` fraction."""

        weights = np.abs(np.asarray(self.deformation.delta_r, dtype=complex))
        weights = weights[np.isfinite(weights)]
        weights = np.sort(weights)[::-1]
        if weights.size == 0:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
        power = weights * weights
        total = float(np.sum(power))
        counts = np.arange(1, weights.size + 1, dtype=int)
        fraction = np.cumsum(power) / total if total > 0.0 else np.zeros_like(power, dtype=float)
        return counts, fraction


@dataclass(frozen=True)
class FieldlineVelocitySpectrum:
    """Field-line velocity spectra induced by a perturbation on a circular surface."""

    psi: float
    r_minor: float
    iota: float
    iota_prime: float
    theta: np.ndarray
    phi: np.ndarray
    radial_velocity: np.ndarray
    poloidal_velocity: np.ndarray
    radial_spectrum: Any
    poloidal_spectrum: Any

    def poloidal_coefficients_for_radial_modes(self) -> np.ndarray:
        """Return coefficients aligned by signed Nardon Fourier indices."""

        return np.asarray([
            self.poloidal_spectrum.nardon_mode_coefficient(int(m), int(nardon_n))
            for m, nardon_n in zip(self.radial_spectrum.m, self.radial_spectrum.nardon_n)
        ], dtype=complex)

    def split(self, resonance_tol: float = 1.0e-9):
        """Split the radial-velocity spectrum into RMP and nRMP modes."""

        return self.radial_spectrum.split(self.iota, resonance_tol=resonance_tol)

    def nonresonant_deformation(
        self,
        *,
        include_shear: bool = True,
        resonance_tol: float = 1.0e-9,
        regularise_eps: float = 0.0,
    ):
        """Compute total nRMP flux-surface deformation from all non-resonant modes."""

        return self.nonresonant_response(
            include_shear=include_shear,
            resonance_tol=resonance_tol,
            regularise_eps=regularise_eps,
        ).deformation

    def nonresonant_response(
        self,
        *,
        include_shear: bool = True,
        resonance_tol: float = 1.0e-9,
        regularise_eps: float = 0.0,
    ) -> NonResonantFieldlineResponse:
        """Compute the total nRMP response by summing every non-resonant mode."""

        from pyna.toroidal.torus_deformation import fieldline_deformation_spectrum

        dtheta = self.poloidal_coefficients_for_radial_modes()
        split = self.split(resonance_tol=resonance_tol)
        keep = split.nonresonant_mask
        detuning = np.asarray(self.radial_spectrum.m, dtype=float) * float(self.iota) + np.asarray(
            self.radial_spectrum.nardon_n, dtype=float
        )
        deformation = fieldline_deformation_spectrum(
            self.radial_spectrum.m[keep],
            self.radial_spectrum.nardon_n[keep],
            self.radial_spectrum.dBr[keep],
            dtheta[keep],
            iota=self.iota,
            iota_prime=self.iota_prime,
            include_shear=include_shear,
            resonance_tol=0.0,
            regularise_eps=regularise_eps,
        )
        return NonResonantFieldlineResponse(
            velocity=self,
            deformation=deformation,
            resonance_tol=float(resonance_tol),
            include_shear=bool(include_shear),
            regularise_eps=float(regularise_eps),
            nonresonant_mask=np.asarray(keep, dtype=bool),
            detuning=detuning,
            poloidal_velocity_coefficients=dtheta,
        )


def _periodic_central_diff(values: np.ndarray, axis_values: np.ndarray, axis: int) -> np.ndarray:
    step = float(np.asarray(axis_values, dtype=float)[1] - np.asarray(axis_values, dtype=float)[0])
    return (np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)) / (2.0 * step)


def circular_shell_divergence_diagnostic(
    delta_B_func: Callable,
    *,
    axis_R: float,
    axis_Z: float = 0.0,
    r_values: Sequence[float] | np.ndarray = (0.08, 0.12, 0.16, 0.20, 0.24, 0.28),
    n_theta: int = 256,
    n_phi: int = 256,
) -> DivergenceDiagnostic:
    """Numerically check ``div(delta_B)`` in circular-shell coordinates."""

    r = np.asarray(r_values, dtype=float)
    theta = np.linspace(0.0, TWOPI, int(n_theta), endpoint=False)
    phi = np.linspace(0.0, TWOPI, int(n_phi), endpoint=False)
    pp, rr, tt = np.meshgrid(phi, r, theta, indexing="ij")
    R = float(axis_R) + rr * np.cos(tt)
    Z = float(axis_Z) + rr * np.sin(tt)
    BR, BZ, Bphi = np.asarray(delta_B_func(R, Z, pp), dtype=float)
    Br = BR * np.cos(tt) + BZ * np.sin(tt)
    Btheta = -BR * np.sin(tt) + BZ * np.cos(tt)

    radial_flux = rr * R * Br
    poloidal_flux = R * Btheta
    toroidal_flux = rr * Bphi
    d_radial = np.gradient(radial_flux, r, axis=1, edge_order=2)
    d_poloidal = _periodic_central_diff(poloidal_flux, theta, axis=2)
    d_toroidal = _periodic_central_diff(toroidal_flux, phi, axis=0)
    div = (d_radial + d_poloidal + d_toroidal) / (rr * R)
    interior = div[:, 1:-1, :] if r.size > 2 else div
    amp = np.sqrt(BR * BR + BZ * BZ + Bphi * Bphi)
    scale = float(np.nanmax(amp) / max(float(np.nanmin(r)), 1.0e-300))
    scale = max(scale, 1.0e-300)
    max_abs = float(np.nanmax(np.abs(interior)))
    rms = float(np.sqrt(np.nanmean(np.abs(interior) ** 2)))
    return DivergenceDiagnostic(
        max_abs=max_abs,
        rms=rms,
        scale=scale,
        relative_max=max_abs / scale,
        relative_rms=rms / scale,
    )


def rmp_nrmp_mode_rows(
    spectrum: Any,
    iota: float,
    *,
    resonance_tol: float = 1.0e-9,
    top: int | None = 12,
    min_amplitude: float = 0.0,
    radial_index: int | None = None,
) -> list[RMPnRMPModeRow]:
    """Return a sorted table of RMP/nRMP Fourier modes from a radial spectrum."""

    coeffs = spectrum.dBr
    if np.ndim(coeffs) == 2:
        if radial_index is None:
            raise ValueError("radial_index is required for radial-stack spectra")
        coeffs = coeffs[int(radial_index)]
    rows: list[RMPnRMPModeRow] = []
    for m_val, n_val, coeff in zip(spectrum.m, spectrum.nardon_n, coeffs):
        coeff = complex(coeff)
        amp = abs(coeff)
        if amp < float(min_amplitude):
            continue
        detuning = float(int(m_val) * float(iota) + int(n_val))
        kind = "RMP" if abs(detuning) <= float(resonance_tol) else "nRMP"
        rows.append(RMPnRMPModeRow(
            m=int(m_val),
            n=int(n_val),
            coefficient=coeff,
            amplitude=float(amp),
            phase=float(np.angle(coeff)),
            detuning=detuning,
            kind=kind,
        ))
    rows.sort(key=lambda row: row.amplitude, reverse=True)
    return rows if top is None else rows[:int(top)]


def _iota_prime_radius(eq: Any, r_minor: float) -> float:
    r0 = float(getattr(eq, "r0"))
    psi = (float(r_minor) / r0) ** 2
    h = min(1.0e-4, max(1.0e-6, 0.25 * min(psi, 1.0 - psi))) if 0.0 < psi < 1.0 else 1.0e-5
    psi_lo = max(0.0, psi - h)
    psi_hi = min(1.0, psi + h)
    if psi_hi == psi_lo:
        return 0.0
    iota_lo = 1.0 / float(eq.q_of_psi(psi_lo))
    iota_hi = 1.0 / float(eq.q_of_psi(psi_hi))
    diota_dpsi = (iota_hi - iota_lo) / (psi_hi - psi_lo)
    return float(diota_dpsi * 2.0 * float(r_minor) / (r0 * r0))


def fieldline_velocity_spectrum_on_circular_surface(
    eq: Any,
    delta_B_func: Callable,
    psi: float,
    *,
    n_theta: int = 128,
    n_phi: int = 128,
    m_max: int = 8,
    n_max: int = 8,
    min_amplitude: float = 1.0e-12,
) -> FieldlineVelocitySpectrum:
    """Sample first-order field-line velocity spectra on one circular flux surface.

    The returned radial velocity is ``dr/dphi`` and the poloidal velocity is the
    perturbation to ``dtheta/dphi``.  A perturbation ``delta B_phi`` contributes
    ``-iota*delta B_phi/B_phi`` to the poloidal velocity through the toroidal
    denominator.
    """

    from pyna.toroidal.perturbation_spectrum import radial_perturbation_Fourier_spectrum

    psi_val = float(psi)
    r_minor = float(np.sqrt(max(psi_val, 0.0)) * float(getattr(eq, "r0")))
    iota = 1.0 / float(eq.q_of_psi(psi_val))
    theta = np.linspace(0.0, TWOPI, int(n_theta), endpoint=False)
    phi = np.linspace(0.0, TWOPI, int(n_phi), endpoint=False)
    TT, PP = np.meshgrid(theta, phi, indexing="xy")
    RR = float(getattr(eq, "R0")) + r_minor * np.cos(TT)
    ZZ = r_minor * np.sin(TT)
    Bphi0 = float(getattr(eq, "B0")) * float(getattr(eq, "R0")) / RR
    dBR, dBZ, dBphi = np.asarray(delta_B_func(RR, ZZ, PP), dtype=float)
    dBr = dBR * np.cos(TT) + dBZ * np.sin(TT)
    dBtheta = -dBR * np.sin(TT) + dBZ * np.cos(TT)
    radial_velocity = RR * dBr / Bphi0
    poloidal_velocity = RR * dBtheta / (max(r_minor, 1.0e-300) * Bphi0) - iota * dBphi / Bphi0
    radial_spec = radial_perturbation_Fourier_spectrum(
        radial_velocity,
        theta,
        phi,
        m_max=m_max,
        n_max=n_max,
        min_amplitude=min_amplitude,
    )
    poloidal_spec = radial_perturbation_Fourier_spectrum(
        poloidal_velocity,
        theta,
        phi,
        m_max=m_max,
        n_max=n_max,
        min_amplitude=min_amplitude,
    )
    return FieldlineVelocitySpectrum(
        psi=psi_val,
        r_minor=r_minor,
        iota=iota,
        iota_prime=_iota_prime_radius(eq, r_minor),
        theta=theta,
        phi=phi,
        radial_velocity=radial_velocity,
        poloidal_velocity=poloidal_velocity,
        radial_spectrum=radial_spec,
        poloidal_spectrum=poloidal_spec,
    )


def sample_stellarator_cylindrical_field(
    eq: Any,
    delta_B_func: Callable | None = None,
    *,
    nR: int = 128,
    nPhi: int = 128,
    lim_factor: float = 1.18,
    label: str = "sampled_stellarator_field",
):
    """Sample an analytic stellarator plus optional perturbation as ``VectorFieldCylind``."""

    from pyna.fields.cylindrical import VectorFieldCylind

    lim = float(lim_factor) * float(eq.r0)
    R_grid = np.linspace(float(eq.R0) - lim, float(eq.R0) + lim, int(nR))
    Z_grid = np.linspace(-lim, lim, int(nR))
    Phi_grid = np.linspace(0.0, TWOPI, int(nPhi), endpoint=False)
    RR, ZZ, PP = np.meshgrid(R_grid, Z_grid, Phi_grid, indexing="ij")

    theta = np.arctan2(ZZ, RR - float(eq.R0))
    psi = eq.psi_ax(RR, ZZ)
    q = eq.q_of_psi(psi)
    r_minor = np.hypot(RR - float(eq.R0), ZZ)
    Bphi = float(eq.B0) * float(eq.R0) / RR
    Bpol = Bphi * r_minor / (RR * np.maximum(np.abs(q), 1.0e-3))
    BR = np.where(r_minor > 1.0e-10, -Bpol * np.sin(theta), 0.0)
    BZ = np.where(r_minor > 1.0e-10, Bpol * np.cos(theta), 0.0)
    helical_BR = float(eq.epsilon_h) * float(eq.B0) * psi * np.cos(eq.m_h * theta - eq.n_h * PP)
    BR = BR + helical_BR
    if delta_B_func is not None:
        dBR, dBZ, dBphi = np.asarray(delta_B_func(RR, ZZ, PP), dtype=float)
        BR = BR + dBR
        BZ = BZ + dBZ
        Bphi = Bphi + dBphi
    return VectorFieldCylind(
        R_grid,
        Z_grid,
        Phi_grid,
        BR=BR,
        BZ=BZ,
        BPhi=Bphi,
        label=label,
    )


@dataclass
class ResonantComponent:
    """One resonant (m, n) Fourier component of the RMP field."""
    m: int
    n: int
    harmonic_order: int
    b_mn: complex
    psi_res: float
    q_res: float
    half_width_psi: float
    half_width_r: float
    opoint_theta: float    # first O-point at φ=0, in [0, 2π/m)
    xpoint_theta: float    # first X-point at φ=0, in [0, 2π/m)
    q_prime_sign: int = 1  # +1 normal shear, −1 reversed shear

    def fixed_points(self, phi: Union[float, np.ndarray]) -> dict:
        """O-points and X-points at arbitrary toroidal section(s) φ.

        Parameters
        ----------
        phi : float or array_like
            Toroidal angle(s) in radians.

        Returns
        -------
        dict with 'theta_O', 'theta_X' (shape (N, m)) and degree variants.

        Example
        -------
        >>> comp.fixed_points(0.0)['theta_O_deg']   # at φ=0
        >>> comp.fixed_points(np.linspace(0, 2*np.pi, 36))['theta_O']
        """
        return island_fixed_points(
            self.m, self.n, self.b_mn, phi, self.q_prime_sign
        )


@dataclass(frozen=True)
class FixedPointPhaseComparison:
    """RMP spectrum prediction compared with one cyna Newton fixed point."""

    predicted_kind: str
    branch: int
    predicted_theta: float
    predicted_R: float
    predicted_Z: float
    newton_kind: Optional[str]
    newton_R: float
    newton_Z: float
    residual: float
    converged: bool
    point_type: int
    theta_error: float
    helical_phase_error: float
    radial_error: float
    map_span: float
    phi: float
    m: Optional[int] = None
    n: Optional[int] = None

    @property
    def predicted_theta_deg(self) -> float:
        return float(np.degrees(self.predicted_theta))

    @property
    def theta_error_deg(self) -> float:
        return float(np.degrees(self.theta_error))

    @property
    def helical_phase_error_deg(self) -> float:
        return float(np.degrees(self.helical_phase_error))

    @property
    def radial_error_cm(self) -> float:
        return 100.0 * float(self.radial_error)


@dataclass(frozen=True)
class DeformedFixedPointProjection:
    """Projection of a Newton fixed point onto a deformed resonant surface."""

    predicted_kind: str
    branch: int
    predicted_theta: float
    projected_theta: float
    theta_error: float
    closest_R: float
    closest_Z: float
    distance: float
    phi: float

    @property
    def theta_error_deg(self) -> float:
        return float(np.degrees(self.theta_error))

    @property
    def distance_cm(self) -> float:
        return 100.0 * float(self.distance)


@dataclass(frozen=True)
class LogLogOrderFit:
    """Least-squares power-law fit ``y ~= exp(intercept) * x**slope``."""

    x: np.ndarray
    y: np.ndarray
    slope: float
    intercept: float
    expected: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "x", np.asarray(self.x, dtype=float))
        object.__setattr__(self, "y", np.asarray(self.y, dtype=float))

    def reference(self, *, exponent: float | None = None, anchor: int = 0) -> np.ndarray:
        """Return a power-law reference through one measured point."""

        if self.x.size == 0 or self.y.size == 0:
            return np.zeros(0, dtype=float)
        idx = int(anchor)
        power = self.expected if exponent is None else float(exponent)
        if power is None:
            power = self.slope
        return self.y[idx] * (self.x / self.x[idx]) ** float(power)


@dataclass(frozen=True)
class NonResonantResidualOrderScan:
    """Perturbation-order scan for a smooth nRMP deformation residual."""

    k: np.ndarray
    residual: np.ndarray
    residual_fit: LogLogOrderFit

    def __post_init__(self):
        object.__setattr__(self, "k", np.asarray(self.k, dtype=float))
        object.__setattr__(self, "residual", np.asarray(self.residual, dtype=float))

    @property
    def slope(self) -> float:
        return float(self.residual_fit.slope)


@dataclass(frozen=True)
class RMPAmplitudeOrderScan:
    """Amplitude-order scan for resonant coefficients and island widths."""

    k: np.ndarray
    b_abs: np.ndarray
    half_width: np.ndarray
    opoint_theta: np.ndarray
    b_fit: LogLogOrderFit
    width_fit: LogLogOrderFit

    def __post_init__(self):
        object.__setattr__(self, "k", np.asarray(self.k, dtype=float))
        object.__setattr__(self, "b_abs", np.asarray(self.b_abs, dtype=float))
        object.__setattr__(self, "half_width", np.asarray(self.half_width, dtype=float))
        object.__setattr__(self, "opoint_theta", np.asarray(self.opoint_theta, dtype=float))

    @property
    def phase_span_deg(self) -> float:
        return float(np.degrees(np.ptp(np.unwrap(self.opoint_theta)))) if self.opoint_theta.size else 0.0


@dataclass(frozen=True)
class RMPPhaseOrderScan:
    """Phase-control scan for ``arg(b_{m,-n})`` and X/O-point phases."""

    control: np.ndarray
    b_phase_shift: np.ndarray
    opoint_shift: np.ndarray
    exact_relation_residual: np.ndarray
    first_order_residual: np.ndarray
    b_phase_fit: LogLogOrderFit
    opoint_vs_b_phase_fit: LogLogOrderFit
    first_order_residual_fit: LogLogOrderFit

    def __post_init__(self):
        object.__setattr__(self, "control", np.asarray(self.control, dtype=float))
        object.__setattr__(self, "b_phase_shift", np.asarray(self.b_phase_shift, dtype=float))
        object.__setattr__(self, "opoint_shift", np.asarray(self.opoint_shift, dtype=float))
        object.__setattr__(
            self,
            "exact_relation_residual",
            np.asarray(self.exact_relation_residual, dtype=float),
        )
        object.__setattr__(
            self,
            "first_order_residual",
            np.asarray(self.first_order_residual, dtype=float),
        )

    @property
    def max_exact_relation_residual(self) -> float:
        return float(np.nanmax(self.exact_relation_residual)) if self.exact_relation_residual.size else 0.0


@dataclass(frozen=True)
class CoupledFixedPointSweep:
    """Fixed-point distance sweep for coupled RMP+nRMP fields."""

    k: np.ndarray
    raw_distance: np.ndarray
    superposed_distance: np.ndarray
    nearest_deformed_distance: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "k", np.asarray(self.k, dtype=float))
        object.__setattr__(self, "raw_distance", np.asarray(self.raw_distance, dtype=float))
        object.__setattr__(self, "superposed_distance", np.asarray(self.superposed_distance, dtype=float))
        object.__setattr__(
            self,
            "nearest_deformed_distance",
            np.asarray(self.nearest_deformed_distance, dtype=float),
        )


@dataclass(frozen=True)
class RMPResolutionConvergenceRow:
    """One grid-resolution row for resonant spectrum convergence."""

    n_theta: int
    n_phi: int
    relative_b_error: float
    phase_error: float
    relative_width_error: float
    deformation_metric: float | None = None

    @property
    def phase_error_deg(self) -> float:
        return float(np.degrees(self.phase_error))


@dataclass(frozen=True)
class RMPResolutionConvergenceScan:
    """Resolution convergence of a resonant-component extraction workflow."""

    rows: tuple[RMPResolutionConvergenceRow, ...]
    reference_n_theta: int
    reference_n_phi: int
    reference_component: Any


@dataclass(frozen=True)
class SurfaceMapResidual:
    """Residuals for one deformed invariant-surface map test."""

    alpha: np.ndarray
    residual: np.ndarray
    endpoint_state: np.ndarray
    predicted_state: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "alpha", np.asarray(self.alpha, dtype=float))
        object.__setattr__(self, "residual", np.asarray(self.residual, dtype=float))
        object.__setattr__(self, "endpoint_state", np.asarray(self.endpoint_state, dtype=float))
        object.__setattr__(self, "predicted_state", np.asarray(self.predicted_state, dtype=float))

    @property
    def max_residual(self) -> float:
        return float(np.nanmax(self.residual)) if self.residual.size else 0.0


def _wrap_to_pi(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Wrap angle(s) to [-pi, pi)."""

    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def loglog_order_fit(
    x: Sequence[float],
    y: Sequence[float],
    *,
    expected: float | None = None,
) -> LogLogOrderFit:
    """Fit a power-law slope on positive finite samples."""

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    keep = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr > 0.0) & (y_arr > 0.0)
    if np.count_nonzero(keep) < 2:
        raise ValueError("at least two positive finite samples are required for a log-log order fit")
    slope, intercept = np.polyfit(np.log(x_arr[keep]), np.log(y_arr[keep]), 1)
    return LogLogOrderFit(
        x=x_arr,
        y=y_arr,
        slope=float(slope),
        intercept=float(intercept),
        expected=None if expected is None else float(expected),
    )


def scan_nonresonant_residual_order(
    k_values: Sequence[float],
    residual_factory: Callable[[float], float],
    *,
    expected: float = 2.0,
) -> NonResonantResidualOrderScan:
    """Evaluate an nRMP residual over perturbation coefficients ``k``."""

    k = np.asarray(k_values, dtype=float)
    residual = np.asarray([float(residual_factory(float(k_val))) for k_val in k], dtype=float)
    return NonResonantResidualOrderScan(
        k=k,
        residual=residual,
        residual_fit=loglog_order_fit(k, residual, expected=expected),
    )


def scan_rmp_amplitude_order(
    k_values: Sequence[float],
    component_factory: Callable[[float], ResonantComponent],
) -> RMPAmplitudeOrderScan:
    """Scan resonant ``b_{m,-n}`` amplitude and island width versus coefficient ``k``."""

    k = np.asarray(k_values, dtype=float)
    components = [component_factory(float(k_val)) for k_val in k]
    b_abs = np.asarray([abs(component.b_mn) for component in components], dtype=float)
    half_width = np.asarray([float(component.half_width_r) for component in components], dtype=float)
    opoint_theta = np.unwrap(np.asarray([float(component.opoint_theta) for component in components], dtype=float))
    return RMPAmplitudeOrderScan(
        k=k,
        b_abs=b_abs,
        half_width=half_width,
        opoint_theta=opoint_theta,
        b_fit=loglog_order_fit(k, b_abs, expected=1.0),
        width_fit=loglog_order_fit(k, half_width, expected=0.5),
    )


def scan_rmp_phase_order(
    control_values: Sequence[float],
    component_factory: Callable[[float], ResonantComponent],
    *,
    base_component: ResonantComponent | None = None,
    mode_m: int | None = None,
    first_order_phase_coefficient: float = 1.0,
) -> RMPPhaseOrderScan:
    """Scan how X/O phase follows the resonant coefficient phase.

    ``component_factory(control)`` should return the resonant component after
    applying the requested phase-control parameter.  The exact first-order
    relation is ``m*Delta(theta_O)+Delta(arg b)=0``.
    """

    control = np.asarray(control_values, dtype=float)
    base = component_factory(0.0) if base_component is None else base_component
    m_val = int(base.m if mode_m is None else mode_m)
    if m_val == 0:
        raise ValueError("mode_m must be nonzero")

    b_phase_shift = []
    opoint_shift = []
    exact_residual = []
    first_order_residual = []
    for control_val in control:
        component = component_factory(float(control_val))
        darg_b = float(_wrap_to_pi(np.angle(component.b_mn / base.b_mn)))
        dtheta = float(_wrap_to_pi(component.opoint_theta - base.opoint_theta))
        b_phase_shift.append(abs(darg_b))
        opoint_shift.append(abs(dtheta))
        exact_residual.append(abs(float(_wrap_to_pi(m_val * dtheta + darg_b))))
        first_order_residual.append(
            abs(float(_wrap_to_pi(dtheta + float(first_order_phase_coefficient) * float(control_val) / m_val)))
        )

    b_phase = np.asarray(b_phase_shift, dtype=float)
    opoint = np.asarray(opoint_shift, dtype=float)
    first_res = np.asarray(first_order_residual, dtype=float)
    return RMPPhaseOrderScan(
        control=control,
        b_phase_shift=b_phase,
        opoint_shift=opoint,
        exact_relation_residual=np.asarray(exact_residual, dtype=float),
        first_order_residual=first_res,
        b_phase_fit=loglog_order_fit(control, b_phase, expected=1.0),
        opoint_vs_b_phase_fit=loglog_order_fit(b_phase, opoint, expected=1.0),
        first_order_residual_fit=loglog_order_fit(control, first_res, expected=2.0),
    )


def scan_coupled_fixed_point_sweep(
    k_values: Sequence[float],
    distance_factory: Callable[[float], Sequence[float]],
) -> CoupledFixedPointSweep:
    """Evaluate raw/superposed/deformed fixed-point distances over ``k``."""

    k = np.asarray(k_values, dtype=float)
    rows = np.asarray([distance_factory(float(k_val)) for k_val in k], dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 3:
        raise ValueError("distance_factory must return three distances: raw, superposed, nearest")
    return CoupledFixedPointSweep(
        k=k,
        raw_distance=rows[:, 0],
        superposed_distance=rows[:, 1],
        nearest_deformed_distance=rows[:, 2],
    )


def scan_rmp_resolution_convergence(
    grids: Sequence[tuple[int, int]],
    component_factory: Callable[[int, int], ResonantComponent],
    *,
    reference_grid: tuple[int, int] | None = None,
    deformation_metric_factory: Callable[[int, int], float] | None = None,
) -> RMPResolutionConvergenceScan:
    """Compare RMP component extraction across ``(n_theta, n_phi)`` grids."""

    grid_list = [(int(n_theta), int(n_phi)) for n_theta, n_phi in grids]
    if not grid_list:
        raise ValueError("at least one resolution grid is required")
    ref_grid = grid_list[-1] if reference_grid is None else (int(reference_grid[0]), int(reference_grid[1]))
    ref_component = component_factory(*ref_grid)
    ref_abs = max(abs(ref_component.b_mn), 1.0e-300)
    ref_width = max(abs(float(ref_component.half_width_r)), 1.0e-300)

    rows: list[RMPResolutionConvergenceRow] = []
    for n_theta, n_phi in grid_list:
        component = component_factory(n_theta, n_phi)
        metric = None if deformation_metric_factory is None else float(deformation_metric_factory(n_theta, n_phi))
        rows.append(RMPResolutionConvergenceRow(
            n_theta=n_theta,
            n_phi=n_phi,
            relative_b_error=float(abs(abs(component.b_mn) - abs(ref_component.b_mn)) / ref_abs),
            phase_error=float(abs(_wrap_to_pi(np.angle(component.b_mn / ref_component.b_mn)))),
            relative_width_error=float(abs(component.half_width_r - ref_component.half_width_r) / ref_width),
            deformation_metric=metric,
        ))
    return RMPResolutionConvergenceScan(
        rows=tuple(rows),
        reference_n_theta=ref_grid[0],
        reference_n_phi=ref_grid[1],
        reference_component=ref_component,
    )


def deformed_surface_map_residual(
    surface: Callable[[float, float], Sequence[float]],
    rhs: Callable[[float, np.ndarray], Sequence[float]],
    iota: float,
    *,
    alpha_values: Sequence[float] | None = None,
    phi0: float = 0.0,
    phi_span: float = TWOPI,
    state_to_cartesian: Callable[[Sequence[float], float], Sequence[float]] | None = None,
    solver_kwargs: dict | None = None,
) -> SurfaceMapResidual:
    """Measure one-turn residuals for a parameterized deformed surface."""

    from scipy.integrate import solve_ivp

    if alpha_values is None:
        alpha = np.linspace(0.0, TWOPI, 12, endpoint=False)
    else:
        alpha = np.asarray(alpha_values, dtype=float)
    kwargs = {
        "method": "DOP853",
        "rtol": 5.0e-10,
        "atol": 1.0e-12,
    }
    if solver_kwargs is not None:
        kwargs.update(dict(solver_kwargs))

    endpoint_rows = []
    predicted_rows = []
    residual = []
    phi1 = float(phi0) + float(phi_span)
    for alpha0 in alpha:
        y0 = np.asarray(surface(float(alpha0), float(phi0)), dtype=float).ravel()
        sol = solve_ivp(rhs, (float(phi0), phi1), y0, **kwargs)
        y_end = np.asarray(sol.y[:, -1], dtype=float)
        y_pred = np.asarray(surface(float(alpha0) + float(iota) * float(phi_span), phi1), dtype=float).ravel()
        endpoint_rows.append(y_end)
        predicted_rows.append(y_pred)
        if state_to_cartesian is None:
            lhs = y_end
            rhs_pred = y_pred
        else:
            lhs = np.asarray(state_to_cartesian(y_end, phi1), dtype=float).ravel()
            rhs_pred = np.asarray(state_to_cartesian(y_pred, phi1), dtype=float).ravel()
        residual.append(float(np.linalg.norm(lhs - rhs_pred)))

    return SurfaceMapResidual(
        alpha=alpha,
        residual=np.asarray(residual, dtype=float),
        endpoint_state=np.asarray(endpoint_rows, dtype=float),
        predicted_state=np.asarray(predicted_rows, dtype=float),
    )


def plot_perturbation_order_summary(
    *,
    nonresonant: NonResonantResidualOrderScan | None = None,
    rmp_amplitude: RMPAmplitudeOrderScan | None = None,
    rmp_phase: RMPPhaseOrderScan | None = None,
    coupling: CoupledFixedPointSweep | None = None,
    axes=None,
    residual_scale: float = 1.0,
    residual_label: str = "residual",
    coefficient_label: str = "perturbation coefficient k",
):
    """Plot a compact four-panel order-analysis summary."""

    if axes is None:
        fig, axes_arr = plt.subplots(2, 2, figsize=(12.4, 7.2), constrained_layout=True)
    else:
        axes_arr = np.asarray(axes)
        fig = axes_arr.ravel()[0].figure
    axes_arr = np.asarray(axes_arr).reshape(2, 2)

    ax = axes_arr[0, 0]
    if nonresonant is None:
        ax.axis("off")
    else:
        ax.loglog(nonresonant.k, nonresonant.residual * residual_scale, "o-", color="#16a34a", label="measured")
        ax.loglog(
            nonresonant.k,
            nonresonant.residual_fit.reference(exponent=2.0) * residual_scale,
            "--",
            color="0.35",
            label="slope 2",
        )
        ax.set_xlabel(coefficient_label)
        ax.set_ylabel(residual_label)
        ax.set_title(f"Non-resonant residual, k={nonresonant.slope:.2f}")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, which="both", alpha=0.25)

    ax = axes_arr[0, 1]
    if rmp_amplitude is None:
        ax.axis("off")
    else:
        k_norm = rmp_amplitude.k / rmp_amplitude.k[0]
        ax.loglog(
            k_norm,
            rmp_amplitude.b_abs / rmp_amplitude.b_abs[0],
            "o-",
            color="#2563eb",
            label=rf"$|b_{{m,-n}}|$, slope={rmp_amplitude.b_fit.slope:.2f}",
        )
        ax.loglog(
            k_norm,
            rmp_amplitude.half_width / rmp_amplitude.half_width[0],
            "s-",
            color="#dc2626",
            label=f"width, slope={rmp_amplitude.width_fit.slope:.2f}",
        )
        ax.loglog(k_norm, k_norm, "--", color="#2563eb", alpha=0.35)
        ax.loglog(k_norm, k_norm ** 0.5, "--", color="#dc2626", alpha=0.35)
        ax.set_xlabel("coefficient / first coefficient")
        ax.set_ylabel("normalised response")
        ax.set_title("Resonant amplitude orders")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, which="both", alpha=0.25)

    ax = axes_arr[1, 0]
    if rmp_phase is None:
        ax.axis("off")
    else:
        control_norm = rmp_phase.control / rmp_phase.control[0]
        ax.loglog(
            control_norm,
            rmp_phase.b_phase_shift / rmp_phase.b_phase_shift[0],
            "o-",
            color="#7c3aed",
            label=rf"$|\Delta\arg b|$, slope={rmp_phase.b_phase_fit.slope:.2f}",
        )
        ax.loglog(
            control_norm,
            rmp_phase.opoint_shift / rmp_phase.opoint_shift[0],
            "s-",
            color="#2563eb",
            label=rf"$|\Delta\theta_O|$, slope={loglog_order_fit(rmp_phase.control, rmp_phase.opoint_shift).slope:.2f}",
        )
        ax.loglog(
            control_norm,
            rmp_phase.first_order_residual / rmp_phase.first_order_residual[0],
            "^-",
            color="#dc2626",
            label=f"1st-order residual, slope={rmp_phase.first_order_residual_fit.slope:.2f}",
        )
        ax.loglog(control_norm, control_norm, "--", color="0.45", alpha=0.35)
        ax.loglog(control_norm, control_norm ** 2, "--", color="#dc2626", alpha=0.35)
        ax.set_xlabel("phase-control coefficient / first coefficient")
        ax.set_ylabel("normalised response")
        ax.set_title("X/O phase-control order")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(True, which="both", alpha=0.25)

    ax = axes_arr[1, 1]
    if coupling is None or coupling.k.size == 0:
        ax.axis("off")
    else:
        ax.plot(coupling.k, coupling.raw_distance, "o-", color="0.35", label="circular RMP seed")
        ax.plot(coupling.k, coupling.superposed_distance, "s-", color="#7c3aed", label="linear superposition")
        ax.plot(coupling.k, coupling.nearest_deformed_distance, "^-", color="#16a34a", label="nearest deformed section")
        ax.set_xlabel("nRMP coefficient k")
        ax.set_ylabel("max fixed-point distance")
        ax.set_title("Coupled-field diagnostic")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.25)

    return fig, axes_arr


def _magnetic_axis(eq: Any) -> tuple[float, float]:
    axis = getattr(eq, "magnetic_axis", None)
    if axis is not None:
        return float(axis[0]), float(axis[1])
    return float(getattr(eq, "R0")), 0.0


def rmp_fixed_point_seeds(component: ResonantComponent, eq: Any, phi: float = 0.0) -> list[dict]:
    """Return geometric R-Z seeds predicted by a resonant RMP component.

    The seeds are placed on the resonant surface ``psi_res`` using the same
    circular geometry convention as :func:`find_resonant_components_analytic`.
    They are intended as first guesses for Newton refinement, not as final
    fixed-point locations for finite-amplitude perturbations.
    """

    r0 = float(getattr(eq, "r0"))
    axis_R, axis_Z = _magnetic_axis(eq)
    r_res = float(np.sqrt(component.psi_res) * r0)
    pts = component.fixed_points(float(phi))

    seeds = []
    for kind, key in (("O", "theta_O"), ("X", "theta_X")):
        theta_values = np.asarray(pts[key], dtype=float)[0]
        for branch, theta in enumerate(theta_values):
            theta = float(theta)
            seeds.append({
                "predicted_kind": kind,
                "branch": int(branch),
                "theta": theta,
                "R": axis_R + r_res * np.cos(theta),
                "Z": axis_Z + r_res * np.sin(theta),
                "r_res": r_res,
            })
    return seeds


def rmp_closure_map_span(component: ResonantComponent) -> float:
    """Return the full-torus toroidal span needed to close an m/n island orbit."""

    m = abs(int(component.m))
    n = abs(int(component.n))
    if m <= 0 or n <= 0:
        raise ValueError("component.m and component.n must be positive nonzero integers")
    return 2.0 * np.pi * (m // int(np.gcd(m, n)))


def compare_cyna_fixed_points_for_component(
    field: Any,
    component: ResonantComponent,
    eq: Any,
    *,
    phi: float = 0.0,
    map_span: Optional[float] = None,
    DPhi: float = 0.025,
    fd_eps: float = 1.0e-4,
    max_iter: int = 60,
    tol: float = 1.0e-10,
    n_threads: int = -1,
    extend_phi: bool = True,
) -> list[FixedPointPhaseComparison]:
    """Refine RMP-predicted O/X seeds with cyna and compare phases.

    Parameters
    ----------
    field
        Cylindrical physical magnetic field components ``(BR, BZ, BPhi)`` on a
        :class:`~pyna.fields.VectorFieldCylind`-compatible grid.  cyna evolves
        ``dR/dphi = R*BR/BPhi`` and ``dZ/dphi = R*BZ/BPhi``.
    component
        Resonant RMP component that supplies the first-order O/X phase
        prediction.
    eq
        Equilibrium-like object with ``r0`` and either ``magnetic_axis`` or
        ``R0``.
    map_span
        Toroidal span for the fixed-point map.  If omitted, the helper uses the
        smallest full-torus closure span implied by ``q=m/n``.

    Returns
    -------
    list of :class:`FixedPointPhaseComparison`
        One row per predicted O/X seed.  ``theta_error`` is the wrapped
        geometric poloidal-angle error; ``helical_phase_error`` wraps
        ``m*theta_error``.
    """

    try:
        import pyna._cyna as _cyna
        from pyna._cyna.utils import prepare_field_cache
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError("cyna fixed-point comparison requires pyna._cyna") from exc

    if not _cyna.is_available() or getattr(_cyna, "find_fixed_points_batch_span", None) is None:
        raise ImportError("pyna._cyna.find_fixed_points_batch_span is unavailable. Build cyna first.")

    seeds = rmp_fixed_point_seeds(component, eq, phi=phi)
    R_seeds = np.ascontiguousarray([seed["R"] for seed in seeds], dtype=np.float64)
    Z_seeds = np.ascontiguousarray([seed["Z"] for seed in seeds], dtype=np.float64)

    if map_span is None:
        map_span = rmp_closure_map_span(component)
    map_span = float(map_span)

    cache = prepare_field_cache(field, extend_phi=extend_phi)
    out = _cyna.find_fixed_points_batch_span(
        R_seeds,
        Z_seeds,
        float(phi),
        map_span,
        float(DPhi),
        float(fd_eps),
        int(max_iter),
        float(tol),
        np.ravel(cache["BR"]),
        np.ravel(cache["BZ"]),
        np.ravel(cache["BPhi"]),
        cache["R_grid"],
        cache["Z_grid"],
        cache["Phi_grid"],
        int(n_threads),
        int(cache["nfp"]),
    )

    R_out, Z_out, residual, converged, _DPm, _eig_r, _eig_i, point_type = out
    axis_R, axis_Z = _magnetic_axis(eq)
    rows: list[FixedPointPhaseComparison] = []
    for i, seed in enumerate(seeds):
        conv = bool(converged[i])
        ptype = int(point_type[i])
        newton_kind = "X" if ptype == 1 else ("O" if ptype == 0 else None)
        Rn = float(R_out[i])
        Zn = float(Z_out[i])
        if conv and np.isfinite(Rn) and np.isfinite(Zn):
            theta_new = float(np.arctan2(Zn - axis_Z, Rn - axis_R) % (2.0 * np.pi))
            theta_error = float(_wrap_to_pi(theta_new - seed["theta"]))
            radial = float(np.hypot(Rn - axis_R, Zn - axis_Z))
            radial_error = radial - float(seed["r_res"])
        else:
            theta_error = float("nan")
            radial_error = float("nan")

        rows.append(FixedPointPhaseComparison(
            predicted_kind=str(seed["predicted_kind"]),
            branch=int(seed["branch"]),
            predicted_theta=float(seed["theta"]),
            predicted_R=float(seed["R"]),
            predicted_Z=float(seed["Z"]),
            newton_kind=newton_kind,
            newton_R=Rn,
            newton_Z=Zn,
            residual=float(residual[i]),
            converged=conv,
            point_type=ptype,
            theta_error=theta_error,
            helical_phase_error=float(_wrap_to_pi(component.m * theta_error))
            if np.isfinite(theta_error) else float("nan"),
            radial_error=radial_error,
            map_span=map_span,
            phi=float(phi),
            m=int(component.m),
            n=int(component.n),
        ))
    return rows


def deformed_circular_section_rz(
    eq: Any,
    r_minor: float,
    deformation: Any,
    theta: Union[float, np.ndarray],
    *,
    phi: float = 0.0,
    include_poloidal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a circular flux surface after applying a deformation spectrum.

    ``deformation`` is typically a
    :class:`pyna.toroidal.torus_deformation.TorusDeformationSpectrum`.
    ``delta_r`` changes the minor radius and, when ``include_poloidal`` is true,
    ``delta_theta`` changes the poloidal coordinate before mapping to ``(R, Z)``.
    """

    theta_arr = np.asarray(theta, dtype=float)
    axis_R, axis_Z = _magnetic_axis(eq)
    dr = deformation.section_r(theta_arr, float(phi))
    dtheta = deformation.section_theta(theta_arr, float(phi)) if include_poloidal else 0.0
    theta_phys = theta_arr + dtheta
    radius = float(r_minor) + dr
    return (
        axis_R + radius * np.cos(theta_phys),
        axis_Z + radius * np.sin(theta_phys),
    )


def project_fixed_points_to_deformed_surface(
    rows: list[FixedPointPhaseComparison],
    eq: Any,
    deformation: Any,
    *,
    r_minor: Optional[float] = None,
    phi: float = 0.0,
    theta_window: float = 0.35,
    include_poloidal: bool = True,
) -> list[DeformedFixedPointProjection]:
    """Project cyna Newton points onto a deformed resonant-surface section.

    The returned ``theta_error`` compares the best-fit deformed-surface
    coordinate against the original RMP spectrum prediction.  This separates
    apparent geometric phase shifts caused by smooth non-resonant surface
    deformation from residual periodic-orbit shifts.
    """

    from scipy.optimize import minimize_scalar

    axis_R, axis_Z = _magnetic_axis(eq)
    out: list[DeformedFixedPointProjection] = []
    for row in rows:
        if r_minor is None:
            r_row = float(np.hypot(row.predicted_R - axis_R, row.predicted_Z - axis_Z))
        else:
            r_row = float(r_minor)

        def objective(theta_val: float) -> float:
            Rm, Zm = deformed_circular_section_rz(
                eq,
                r_row,
                deformation,
                float(theta_val),
                phi=phi,
                include_poloidal=include_poloidal,
            )
            return float((float(Rm) - row.newton_R) ** 2 + (float(Zm) - row.newton_Z) ** 2)

        center = float(row.predicted_theta)
        result = minimize_scalar(
            objective,
            bounds=(center - float(theta_window), center + float(theta_window)),
            method="bounded",
        )
        theta_proj = float(result.x % (2.0 * np.pi))
        R_closest, Z_closest = deformed_circular_section_rz(
            eq,
            r_row,
            deformation,
            theta_proj,
            phi=phi,
            include_poloidal=include_poloidal,
        )
        out.append(DeformedFixedPointProjection(
            predicted_kind=row.predicted_kind,
            branch=row.branch,
            predicted_theta=center,
            projected_theta=theta_proj,
            theta_error=float(_wrap_to_pi(theta_proj - center)),
            closest_R=float(R_closest),
            closest_Z=float(Z_closest),
            distance=float(np.sqrt(max(float(result.fun), 0.0))),
            phi=float(phi),
        ))
    return out


def find_resonant_components_analytic(
    eq,
    delta_B_func,
    base_m: int,
    base_n: int,
    max_harmonic: int = 3,
    n_theta: int = 128,
    n_phi: int = 64,
    min_amplitude: float = 1e-8,
    verbose: bool = True,
) -> List[ResonantComponent]:
    """Find resonant RMP components using analytic surface sampling.

    For each harmonic k, finds the resonant surface ψ_res where
    q(ψ_res) = k*base_m / (k*base_n) = base_m/base_n, then computes
    the Fourier coefficient b_{km,-kn} by sampling the RMP on that surface.

    Works with StellaratorSimple's psi_ax / q_of_psi / resonant_psi API.
    """
    components = []

    for k in range(1, max_harmonic + 1):
        m_k = k * base_m
        n_k = k * base_n

        # Find resonant ψ: resonance condition q = m/n (mode numbers)
        # eq.resonant_psi(m, n) gives q = m/n
        # We want q = m_k/n_k, so call resonant_psi(m_k, n_k)
        psi_list = eq.resonant_psi(m_k, n_k)
        if not psi_list:
            if verbose:
                print(f"  k={k}: ({m_k},{n_k}) — no resonant surface in [0,1], skipping")
            continue

        psi_res = float(psi_list[0])
        r_res = np.sqrt(psi_res) * eq.r0
        q_res = float(eq.q_of_psi(psi_res))

        # Sample RMP on flux surface (circular approximation)
        theta_arr = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
        phi_arr   = np.linspace(0, 2*np.pi, n_phi,   endpoint=False)

        R_surf = eq.R0 + r_res * np.cos(theta_arr)
        Z_surf =         r_res * np.sin(theta_arr)

        # Compute δB^ψ on (theta, phi) grid
        # δB^ψ ≈ δBR * cos(θ) + δBZ * sin(θ)  (radial projection)
        dBpsi = np.zeros((n_theta, n_phi), dtype=complex)
        for j, phi in enumerate(phi_arr):
            for i in range(n_theta):
                db = delta_B_func(R_surf[i], Z_surf[i], phi)
                # Project onto outward radial direction
                dBpsi[i, j] = db[0] * np.cos(theta_arr[i]) + db[1] * np.sin(theta_arr[i])

        # Nardon forward transform: b_mn = integral b exp[-i(m theta+n phi)].
        b_fft = np.fft.fft2(dBpsi) / (n_theta * n_phi)
        m_freq = np.fft.fftfreq(n_theta, 1/n_theta).astype(int)
        n_freq = np.fft.fftfreq(n_phi,   1/n_phi).astype(int)

        # The resonant positive-q branch is Nardon's (m_k, -n_k).  A real
        # field has its conjugate at (-m_k, +n_k), not at (m_k, +n_k).
        m_idx_arr = np.where(m_freq == m_k)[0]
        n_idx_arr = np.where(n_freq == -n_k)[0]  # note: -n_k (conjugate convention)

        if len(m_idx_arr) == 0 or len(n_idx_arr) == 0:
            if verbose:
                print(f"  k={k}: ({m_k},{n_k}) — mode not in FFT grid, skipping")
            continue

        b_mn = b_fft[m_idx_arr[0], n_idx_arr[0]]

        if abs(b_mn) < min_amplitude:
            if verbose:
                print(f"  k={k}: ({m_k},{n_k}) — |b_mn|={abs(b_mn):.2e} below threshold")
            continue

        # dq/dψ at resonant surface (from linear profile: dq/dψ = q1 - q0)
        dq_dpsi = eq.q1 - eq.q0   # constant for linear profile

        if abs(dq_dpsi) < 1e-12:
            continue

        # Rutherford formula: w_ψ = 4 * sqrt(|b_mn| / (m * |dq/dψ|))
        half_width_psi = 4.0 * np.sqrt(abs(b_mn) / (m_k * abs(dq_dpsi) + 1e-30))

        # Convert to meters: r ≈ sqrt(ψ) * r0
        half_width_r = half_width_psi * eq.r0 / (2.0 * np.sqrt(max(psi_res, 0.01)))

        # Reuse the section-phase convention used by the public fixed-point API.
        phi_mn = np.angle(b_mn)
        q_prime_sign = 1 if (getattr(eq, 'q1', 1.0) - getattr(eq, 'q0', 0.5)) >= 0 else -1
        fixed_points = island_fixed_points(m_k, n_k, b_mn, 0.0, q_prime_sign=q_prime_sign)
        # Preserve this function's historical primary-branch range [0, 2*pi/m).
        opoint_theta = float(fixed_points["theta_O"][0, 0] % (TWOPI / m_k))
        xpoint_theta = float(fixed_points["theta_X"][0, 0] % (TWOPI / m_k))

        if verbose:
            print(f"  k={k}: ({m_k},{n_k}) ψ_res={psi_res:.3f} q_res={q_res:.3f} "
                  f"|b_mn|={abs(b_mn):.3e} phase_arg={np.degrees(phi_mn):.1f}° "
                  f"w_ψ={half_width_psi:.4f} ({half_width_r*100:.2f} cm) "
                  f"θ_O={np.degrees(opoint_theta):.1f}° θ_X={np.degrees(xpoint_theta):.1f}°")

        components.append(ResonantComponent(
            m=m_k, n=n_k,
            harmonic_order=k,
            b_mn=b_mn,
            psi_res=psi_res,
            q_res=q_res,
            half_width_psi=half_width_psi,
            half_width_r=half_width_r,
            opoint_theta=opoint_theta,
            xpoint_theta=xpoint_theta,
            q_prime_sign=q_prime_sign,
        ))

    return components


def plot_island_width_bars(
    ax,
    components: List[ResonantComponent],
    eq,
    phi_section: float = 0.0,
    colors: list = None,
    label_harmonics: bool = True,
) -> None:
    """Draw island width bars at O-point positions on R-Z cross-section.

    Parameters
    ----------
    phi_section : float
        Toroidal angle φ (radians) of this Poincaré cross-section.
        O/X-point angles are computed via the general formula at this φ.
    """
    if colors is None:
        colors = ISLAND_CMAPS

    R0 = eq.R0
    r0 = eq.r0

    for comp in components:
        color = colors[(comp.harmonic_order - 1) % len(colors)]
        r_res = np.sqrt(comp.psi_res) * r0

        # Use the general φ-aware formula
        pts = comp.fixed_points(phi_section)
        theta_O_all = pts['theta_O'][0]   # shape (m,)  — all m O-points
        theta_X_all = pts['theta_X'][0]   # shape (m,)

        for i_op in range(comp.m):
            theta_op = theta_O_all[i_op]
            theta_xp = theta_X_all[i_op]

            R_O = R0 + r_res * np.cos(theta_op)
            Z_O =      r_res * np.sin(theta_op)
            R_X = R0 + r_res * np.cos(theta_xp)
            Z_X =      r_res * np.sin(theta_xp)

            r_inner = max(0.01, r_res - comp.half_width_r)
            r_outer = r_res + comp.half_width_r

            R_in  = R0 + r_inner * np.cos(theta_op)
            Z_in  =      r_inner * np.sin(theta_op)
            R_out = R0 + r_outer * np.cos(theta_op)
            Z_out =      r_outer * np.sin(theta_op)

            # Island width bar at O-point
            ax.plot([R_in, R_out], [Z_in, Z_out],
                    color=color, linewidth=3.5, alpha=0.85,
                    solid_capstyle='round', zorder=5)
            ax.plot(R_O, Z_O, 'o', color=color, markersize=6, zorder=6)
            # X-point marker
            ax.plot(R_X, Z_X, 'x', color=color, markersize=7, markeredgewidth=1.5,
                    zorder=6, alpha=0.7)

        if label_harmonics:
            theta_op0 = theta_O_all[0]
            r_label = r_res + comp.half_width_r + 0.015
            R_label = R0 + r_label * np.cos(theta_op0)
            Z_label =      r_label * np.sin(theta_op0)
            ax.annotate(
                f'$({comp.m},{comp.n})$',
                xy=(R_label, Z_label),
                fontsize=7, color=color,
                ha='center', va='center', zorder=7,
                fontweight='bold',
            )


def rmp_section_layout(*args, **kwargs):
    """Create a compact analytic RMP section layout via :mod:`pyna.plot.rmp`."""

    from pyna.plot.rmp import create_rmp_section_layout

    return create_rmp_section_layout(*args, **kwargs)


def draw_rmp_overlays(*args, **kwargs):
    """Apply named analytic RMP section overlays via :mod:`pyna.plot.rmp`."""

    from pyna.plot.rmp import draw_rmp_section_overlays

    return draw_rmp_section_overlays(*args, **kwargs)


def plot_rmp_section(ax, R=None, Z=None, **kwargs):
    """Draw one analytic RMP section with optional modular overlays."""

    from pyna.plot.rmp import draw_rmp_resonance_section

    return draw_rmp_resonance_section(
        ax,
        [] if R is None else R,
        [] if Z is None else Z,
        **kwargs,
    )


def plot_rmp_sections(*args, **kwargs):
    """Draw a compact multi-section analytic RMP figure via :mod:`pyna.plot.rmp`."""

    from pyna.plot.rmp import plot_rmp_resonance_sections

    return plot_rmp_resonance_sections(*args, **kwargs)


# ---------------------------------------------------------------------------
# 2-D (m, n) Fourier spectrum heatmap utilities
# ---------------------------------------------------------------------------

def compute_mn_spectrum(
    delta_B_func,
    S: float,
    equilibrium,
    m_max: int = 6,
    n_max: int = 4,
    n_theta: int = 64,
    n_phi: int = 64,
    phi0: float = 0.0,
) -> np.ndarray:
    """Compute the 2-D (m, n) Fourier spectrum of delta_B on a flux surface.

    Samples the radial perturbation field delta_B^psi = delta_BR * cos(theta)
    + delta_BZ * sin(theta) on a flux surface at normalised label S, then
    returns a (2*m_max+1) x (2*n_max+1) array of complex Fourier amplitudes
    b_{m,n} for m in [-m_max, m_max] and n in [-n_max, n_max].

    Parameters
    ----------
    delta_B_func : callable
        ``(R, Z, phi) -> [dBR, dBZ, dBphi]``
    S : float
        Normalised flux label (r_minor / r0)^2.
    equilibrium :
        Provides ``R0``, ``r0``.
    m_max, n_max : int
        Maximum poloidal / toroidal mode numbers.
    n_theta, n_phi : int
        Sampling resolution in theta and phi.
    phi0 : float
        Starting toroidal angle (unused; sampling covers full [0, 2pi)).

    Returns
    -------
    b_mn : ndarray, shape (2*m_max+1, 2*n_max+1), complex
        b_mn[i, j] = amplitude for m = i - m_max, n = j - n_max.
    """
    R0, r0 = equilibrium.R0, equilibrium.r0
    r = np.sqrt(S) * r0

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi   = np.linspace(0, 2 * np.pi, n_phi,   endpoint=False)

    R_surf = R0 + r * np.cos(theta)
    Z_surf =      r * np.sin(theta)

    dBpsi = np.zeros((n_theta, n_phi), dtype=complex)
    for j_phi in range(n_phi):
        for i_th in range(n_theta):
            try:
                db = delta_B_func(R_surf[i_th], Z_surf[i_th], phi[j_phi])
                dBpsi[i_th, j_phi] = (
                    db[0] * np.cos(theta[i_th]) + db[1] * np.sin(theta[i_th])
                )
            except Exception:
                pass

    # Full 2-D DFT: b_{m,n} at fftfreq indices
    B_fft = np.fft.fft2(dBpsi) / (n_theta * n_phi)
    m_freq = np.fft.fftfreq(n_theta, 1 / n_theta).astype(int)
    n_freq = np.fft.fftfreq(n_phi,   1 / n_phi).astype(int)

    b_mn = np.zeros((2 * m_max + 1, 2 * n_max + 1), dtype=complex)
    for i, m in enumerate(range(-m_max, m_max + 1)):
        im = np.where(m_freq == m)[0]
        if not len(im):
            continue
        for j, n in enumerate(range(-n_max, n_max + 1)):
            jn = np.where(n_freq == n)[0]
            if not len(jn):
                continue
            b_mn[i, j] = B_fft[im[0], jn[0]]

    return b_mn


def plot_mn_heatmap(
    b_mn: np.ndarray,
    m_max: int = 6,
    n_max: int = 4,
    ax=None,
    log_scale: bool = True,
    title: str = r'$|\tilde{b}_{mn}|$ spectrum',
    cmap: str = 'hot_r',
    vmin: float = None,
    annotate: bool = True,
    highlight_modes: list = None,
) -> "tuple[plt.Figure, plt.Axes]":
    """Plot a (m, n) Fourier amplitude heatmap.

    Parameters
    ----------
    b_mn : ndarray, shape (2*m_max+1, 2*n_max+1)
        Complex Fourier amplitudes from ``compute_mn_spectrum``.
    m_max, n_max : int
        Must match the shape of b_mn.
    ax : matplotlib Axes or None
    log_scale : bool
        Use log10 colour scale (recommended for large dynamic range).
    title : str
    cmap : str
        Matplotlib colourmap name.
    vmin : float or None
        Minimum value for colour scale (log10 units if log_scale=True).
    annotate : bool
        Annotate each cell with its numeric value.
    highlight_modes : list of (m, n) tuples
        Draw a red box around these specific modes.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    amps = np.abs(b_mn)
    m_range = np.arange(-m_max, m_max + 1)
    n_range = np.arange(-n_max, n_max + 1)

    if log_scale:
        plot_data = np.log10(amps + 1e-30)
        cbar_label = r'$\log_{10}|\tilde{b}_{mn}|$'
        if vmin is None:
            vmin = plot_data.max() - 6  # show 6 decades
    else:
        plot_data = amps
        cbar_label = r'$|\tilde{b}_{mn}|$'
        if vmin is None:
            vmin = 0.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(n_range) + 1.5),
                                        max(4, 0.5 * len(m_range) + 1.5)))
    else:
        fig = ax.figure

    im = ax.imshow(
        plot_data,
        origin='lower',
        aspect='auto',
        cmap=cmap,
        vmin=vmin,
        vmax=plot_data.max(),
        extent=[-n_max - 0.5, n_max + 0.5, -m_max - 0.5, m_max + 0.5],
        interpolation='nearest',
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label(cbar_label, fontsize=9)

    ax.set_xlabel('n  (toroidal mode)', fontsize=10)
    ax.set_ylabel('m  (poloidal mode)',  fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(n_range)
    ax.set_yticks(m_range)
    ax.axvline(0, color='white', lw=0.5, alpha=0.4)
    ax.axhline(0, color='white', lw=0.5, alpha=0.4)

    if annotate:
        for i, m in enumerate(m_range):
            for j, n in enumerate(n_range):
                val = amps[i, j]
                if log_scale:
                    txt = f'{np.log10(val+1e-30):.1f}'
                else:
                    txt = f'{val:.1e}'
                ax.text(n, m, txt, ha='center', va='center',
                        fontsize=5.5, color='white' if plot_data[i, j] > (plot_data.max() + vmin) / 2 else 'black')

    if highlight_modes:
        for (hm, hn) in highlight_modes:
            if abs(hm) <= m_max and abs(hn) <= n_max:
                ax.add_patch(plt.Rectangle(
                    (hn - 0.5, hm - 0.5), 1, 1,
                    linewidth=2, edgecolor='red', facecolor='none', zorder=5,
                ))

    return fig, ax


def plot_mn_heatmap_radial(
    delta_B_func,
    equilibrium,
    S_values: np.ndarray,
    m_max: int = 4,
    n_max: int = 3,
    n_theta: int = 32,
    n_phi: int = 32,
    target_modes: list = None,
    fig_title: str = 'Fourier spectrum vs flux surface',
    cmap: str = 'hot_r',
) -> "tuple[plt.Figure, list]":
    """Plot one (m,n)-heatmap per flux surface, arranged in a row.

    For each S in S_values, compute the full (m,n) spectrum and plot
    a heatmap.  Useful for showing how the resonant structure varies
    radially across the plasma.

    Parameters
    ----------
    S_values : array_like
        Normalised flux labels at which to evaluate the spectrum.
    target_modes : list of (m,n) or None
        Highlight these modes with a red box in every panel.

    Returns
    -------
    fig, axes
    """
    S_values = np.atleast_1d(S_values)
    nS = len(S_values)

    fig, axes = plt.subplots(1, nS, figsize=(3.5 * nS, 3.5))
    if nS == 1:
        axes = [axes]

    for ax, S in zip(axes, S_values):
        b_mn = compute_mn_spectrum(
            delta_B_func, S, equilibrium,
            m_max=m_max, n_max=n_max,
            n_theta=n_theta, n_phi=n_phi,
        )
        plot_mn_heatmap(
            b_mn, m_max=m_max, n_max=n_max,
            ax=ax, log_scale=True,
            title=f'S={S:.2f}  (q={equilibrium.q_of_psi(S):.2f})',
            cmap=cmap,
            annotate=(nS <= 4),
            highlight_modes=target_modes,
        )

    fig.suptitle(fig_title, fontsize=12)
    plt.tight_layout()
    return fig, axes
