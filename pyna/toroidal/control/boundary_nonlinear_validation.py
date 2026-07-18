"""Validated nonlinear topology observables for boundary control.

This module consumes fixed points and manifold traces produced by the public
``pyna.toroidal.flt.island_chain`` API.  It deliberately does not search for,
refine, or trace invariant objects.  Full separatrix width is measured between
explicit inner/outer envelope crossings on transverse lines in physical
``(R, Z)`` metres.  Corresponding stable/unstable separation is retained as a
different homoclinic/lobe splitting quantity and is never called island width.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from hashlib import sha256
from typing import Any, Callable, Mapping, MutableMapping, Protocol, Sequence

import numpy as np

from pyna.toroidal.control.boundary_topology_design import (
    BoundaryResponseObservables,
    boundary_response_observables,
)


TWOPI = 2.0 * np.pi
_MISSING = object()


def periodic_phase_difference(
    phase: float,
    reference: float,
    period: float = TWOPI,
) -> float:
    """Return the principal signed difference ``phase - reference``.

    The result is periodic with ``period`` and lies in the half-open interval
    ``[-period / 2, period / 2)``.  For a chain phase returned by
    :func:`fixed_point_chain_phase`, pass ``period=2*pi/m``.
    """

    phase_f = float(phase)
    reference_f = float(reference)
    period_f = float(period)
    if not np.isfinite(phase_f) or not np.isfinite(reference_f):
        raise ValueError("phase and reference must be finite")
    if not np.isfinite(period_f) or period_f <= 0.0:
        raise ValueError("period must be positive and finite")
    return float((phase_f - reference_f + 0.5 * period_f) % period_f - 0.5 * period_f)


def fixed_point_chain_coherence(theta_O, m: int) -> float:
    """Return ``|mean(exp(i*m*theta_O))|`` for healed O-point angles."""

    harmonic = _positive_integer(m, "m")
    theta = np.asarray(theta_O, dtype=float).ravel()
    if theta.size == 0 or not np.all(np.isfinite(theta)):
        raise ValueError("theta_O must contain finite healed angles")
    coherence = float(abs(np.mean(np.exp(1j * harmonic * theta))))
    return float(np.clip(coherence, 0.0, 1.0))


def fixed_point_chain_helical_phasor(theta_O, phi_O, m: int, nardon_n: int) -> complex:
    """Return ``mean(exp(i*(m*theta* + n_N*phi)))`` for one O chain.

    This is the multi-section invariant that must be compared with Nardon's
    ``tilde b^1_(m,n_N)``.  For a positive-``q`` physical ``(m,n0)`` chain,
    pass ``nardon_n=-n0``.  The familiar single-section phase is the
    ``phi=0`` special case.
    """

    harmonic = _positive_integer(m, "m")
    n_index = int(nardon_n)
    if isinstance(nardon_n, (float, np.floating)) and float(nardon_n) != float(n_index):
        raise ValueError("nardon_n must be an integer")
    theta = np.asarray(theta_O, dtype=float).ravel()
    phi = np.asarray(phi_O, dtype=float).ravel()
    if theta.size == 0 or theta.shape != phi.shape:
        raise ValueError("theta_O and phi_O must be non-empty arrays with matching shape")
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(phi)):
        raise ValueError("theta_O and phi_O must be finite")
    return complex(np.mean(np.exp(1j * (harmonic * theta + n_index * phi))))


def fixed_point_chain_helical_phase(theta_O, phi_O, m: int, nardon_n: int) -> float:
    """Return the O-chain phase modulo ``2*pi/m`` from multiple sections."""

    harmonic = _positive_integer(m, "m")
    phasor = fixed_point_chain_helical_phasor(theta_O, phi_O, harmonic, nardon_n)
    if abs(phasor) <= 32.0 * np.finfo(float).eps:
        raise ValueError("helical chain phase is undefined for incoherent O-point samples")
    return float(np.angle(phasor) / harmonic)


def fixed_point_chain_helical_coherence(theta_O, phi_O, m: int, nardon_n: int) -> float:
    """Return the coherence of the multi-section resonant helical phasor."""

    coherence = abs(fixed_point_chain_helical_phasor(theta_O, phi_O, m, nardon_n))
    return float(np.clip(coherence, 0.0, 1.0))


def nardon_fixed_point_phase_closure_error(
    coefficient: complex,
    reference_coefficient: complex,
    chain_phasor: complex,
    reference_chain_phasor: complex,
    m: int,
) -> float:
    """Return the Newton/Nardon O-phase closure error in poloidal radians.

    With Nardon's basis ``exp(i*(m*theta* + n_N*phi))``, a resonant O chain
    satisfies ``delta arg(H_O) + delta arg(tilde_b^1_mn) = 0`` modulo
    ``2*pi``.  The principal error is divided by ``m`` and therefore lies in
    ``[-pi/m, pi/m)``.  Build ``chain_phasor`` with
    :func:`fixed_point_chain_helical_phasor`; for positive physical ``q=m/n0``
    that helper must receive ``nardon_n=-n0``.
    """

    harmonic = _positive_integer(m, "m")
    values = tuple(
        complex(value)
        for value in (
            coefficient,
            reference_coefficient,
            chain_phasor,
            reference_chain_phasor,
        )
    )
    if not all(np.isfinite(value.real) and np.isfinite(value.imag) for value in values):
        raise ValueError("coefficients and chain phasors must be finite")
    if any(abs(value) <= 32.0 * np.finfo(float).eps for value in values):
        raise ValueError("coefficients and coherent chain phasors must be nonzero")
    b, b0, helical, helical0 = values
    closure = (helical / helical0) * (b / b0)
    return float(np.angle(closure) / harmonic)


def fixed_point_chain_phase(theta_O, m: int) -> float:
    """Return the physical O-chain phase in healed ``theta``.

    The convention is

    ``phase = Arg(mean(exp(i*m*theta_O))) / m``.

    Thus the returned phase is defined modulo ``2*pi/m`` and its principal
    representative lies in ``[-pi/m, pi/m]``.  Downstream observables use
    ``sin(m*phase)`` and ``cos(m*phase)`` so they remain continuous when that
    principal representative wraps.
    """

    harmonic = _positive_integer(m, "m")
    theta = np.asarray(theta_O, dtype=float).ravel()
    if theta.size == 0 or not np.all(np.isfinite(theta)):
        raise ValueError("theta_O must contain finite healed angles")
    phasor = complex(np.mean(np.exp(1j * harmonic * theta)))
    if abs(phasor) <= 32.0 * np.finfo(float).eps:
        raise ValueError("chain phase is undefined for incoherent O-point angles")
    return float(np.angle(phasor) / harmonic)


def greene_residue(DPm) -> float:
    """Return Greene's residue ``(2 - trace(DP^m)) / 4`` for a 2-D map."""

    matrix = np.asarray(DPm, dtype=float)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError("DPm must be a finite 2 by 2 matrix")
    return float((2.0 - np.trace(matrix)) / 4.0)


def _positive_integer(value: int, name: str) -> int:
    integer = int(value)
    if isinstance(value, (float, np.floating)) and float(value) != float(integer):
        raise ValueError(f"{name} must be an integer")
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _nonempty_label(value: object, name: str = "label") -> str:
    label = str(value).strip()
    if not label:
        raise ValueError(f"{name} must be non-empty")
    return label


def _public_value(obj: object, name: str, default: object = _MISSING) -> object:
    if isinstance(obj, Mapping):
        if name in obj:
            return obj[name]
    elif hasattr(obj, name):
        return getattr(obj, name)
    if default is _MISSING:
        raise ValueError(f"fixed-point payload is missing {name!r}")
    return default


class HealedSurfaceSectionChart:
    """A healed-surface chart between section geometry and phase coordinates.

    The supplied callbacks always describe the geometric chart
    ``(s, theta*) <-> (R, Z)``, where ``s = sqrt(psi)`` and ``theta*`` is the
    healed (for example PEST) angle.  ``radial_phase`` selects the coordinate
    used by the map linearization:

    - ``"s"`` uses ``z = (s, theta*)``;
    - ``"psi"`` uses ``z = (psi=s**2, theta*)``.

    Merely using healed ``(s, theta*)`` coordinates does not make them
    canonical.  A chart declared canonical or symplectic defaults to ``psi``;
    an explicit canonical/symplectic claim with ``radial_phase="s"`` is
    rejected.  The coordinate choice and claim are always recorded in
    :attr:`metadata`.
    """

    def __init__(
        self,
        s_theta_to_x: Callable[[np.ndarray], np.ndarray],
        x_to_s_theta: Callable[[np.ndarray], np.ndarray],
        jacobian_s_theta: Callable[[np.ndarray], np.ndarray],
        *,
        radial_phase: str | None = None,
        theta_period: float = TWOPI,
        canonical: bool = False,
        symplectic: bool = False,
        name: str = "healed_surface_section",
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        callbacks = {
            "s_theta_to_x": s_theta_to_x,
            "x_to_s_theta": x_to_s_theta,
            "jacobian_s_theta": jacobian_s_theta,
        }
        for callback_name, callback in callbacks.items():
            if not callable(callback):
                raise TypeError(f"{callback_name} must be callable")
        canonical_flag = bool(canonical)
        symplectic_flag = bool(symplectic) or canonical_flag
        radial = (
            "psi"
            if radial_phase is None and (canonical_flag or symplectic_flag)
            else "s"
            if radial_phase is None
            else str(radial_phase).strip().lower()
        )
        if radial not in {"s", "psi"}:
            raise ValueError("radial_phase must be 's' or 'psi'")
        if radial == "s" and (canonical_flag or symplectic_flag):
            raise ValueError(
                "s=sqrt(psi) is not automatically a canonical radial momentum; "
                "canonical/symplectic charts must use radial_phase='psi'"
            )
        period = float(theta_period)
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("theta_period must be positive and finite")
        chart_name = str(name).strip()
        if not chart_name:
            raise ValueError("chart name must be non-empty")
        coordinate_choice = (
            "z=(s, theta*)" if radial == "s" else "z=(psi=s^2, theta*)"
        )
        chart_metadata = dict(metadata or {})
        supplied_choice = chart_metadata.get("coordinate_choice")
        if supplied_choice is not None and str(supplied_choice) != coordinate_choice:
            raise ValueError("metadata coordinate_choice conflicts with radial_phase")
        chart_metadata.update(
            {
                "coordinate_choice": coordinate_choice,
                "radial_phase": radial,
                "radial_definition": "s=sqrt(psi)",
                "canonical": canonical_flag,
                "symplectic": symplectic_flag,
                "theta_coordinate": "healed theta* (not geometric polar angle)",
            }
        )

        self._s_theta_to_x = s_theta_to_x
        self._x_to_s_theta = x_to_s_theta
        self._jacobian_s_theta = jacobian_s_theta
        self.radial_phase = radial
        self.theta_period = period
        self.canonical = canonical_flag
        self.symplectic = symplectic_flag
        self.name = chart_name
        self.coordinate_choice = coordinate_choice
        self.metadata = chart_metadata

    @staticmethod
    def _points(value, name: str) -> np.ndarray:
        points = np.asarray(value, dtype=float)
        if points.shape[-1:] != (2,) or not np.all(np.isfinite(points)):
            raise ValueError(f"{name} must be a finite array with trailing shape (2,)")
        return points

    @staticmethod
    def _mapped_points(callback, points: np.ndarray, name: str) -> np.ndarray:
        mapped = np.asarray(callback(points.copy()), dtype=float)
        if mapped.shape != points.shape or not np.all(np.isfinite(mapped)):
            raise ValueError(f"{name} must return a finite array with shape {points.shape}")
        return mapped

    def s_theta_to_x(self, s_theta) -> np.ndarray:
        """Map healed ``(s, theta*)`` coordinates to physical ``(R, Z)``."""

        coordinates = self._points(s_theta, "s_theta")
        if np.any(coordinates[..., 0] < 0.0):
            raise ValueError("healed radial coordinate s must be non-negative")
        canonical_theta = coordinates.copy()
        canonical_theta[..., 1] %= self.theta_period
        return self._mapped_points(
            self._s_theta_to_x,
            canonical_theta,
            "s_theta_to_x",
        )

    def x_to_s_theta(self, x_RZ) -> np.ndarray:
        """Map physical ``(R, Z)`` to healed ``(s, theta*)`` coordinates."""

        points = self._points(x_RZ, "x_RZ")
        coordinates = self._mapped_points(
            self._x_to_s_theta,
            points,
            "x_to_s_theta",
        )
        radial_tolerance = 64.0 * np.finfo(float).eps
        if np.any(coordinates[..., 0] < -radial_tolerance):
            raise ValueError("x_to_s_theta returned negative s")
        coordinates = coordinates.copy()
        coordinates[..., 0] = np.maximum(coordinates[..., 0], 0.0)
        coordinates[..., 1] %= self.theta_period
        return coordinates

    def jacobian_s_theta(self, s_theta) -> np.ndarray:
        """Return ``d(R,Z)/d(s,theta*)`` at one healed section point."""

        coordinates = self._points(s_theta, "s_theta")
        if coordinates.shape != (2,):
            raise ValueError("jacobian_s_theta requires one point with shape (2,)")
        if coordinates[0] < 0.0:
            raise ValueError("healed radial coordinate s must be non-negative")
        coordinates = coordinates.copy()
        coordinates[1] %= self.theta_period
        matrix = np.asarray(self._jacobian_s_theta(coordinates), dtype=float)
        if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
            raise ValueError("jacobian_s_theta must return a finite 2 by 2 matrix")
        return matrix

    def z_to_x(self, z) -> np.ndarray:
        """Map selected phase coordinates ``z`` to physical ``(R, Z)``."""

        phase_coordinates = self._points(z, "z")
        s_theta = phase_coordinates.copy()
        if self.radial_phase == "psi":
            radial_tolerance = 64.0 * np.finfo(float).eps
            if np.any(s_theta[..., 0] < -radial_tolerance):
                raise ValueError("phase coordinate psi must be non-negative")
            s_theta[..., 0] = np.sqrt(np.maximum(s_theta[..., 0], 0.0))
        return self.s_theta_to_x(s_theta)

    def x_to_z(self, x_RZ) -> np.ndarray:
        """Map physical ``(R, Z)`` to the selected phase coordinates ``z``."""

        s_theta = self.x_to_s_theta(x_RZ)
        phase_coordinates = s_theta.copy()
        if self.radial_phase == "psi":
            phase_coordinates[..., 0] = np.square(phase_coordinates[..., 0])
        return phase_coordinates

    def jacobian(self, z) -> np.ndarray:
        """Return ``J=d(R,Z)/dz`` for the selected phase coordinates."""

        phase_coordinates = self._points(z, "z")
        if phase_coordinates.shape != (2,):
            raise ValueError("jacobian requires one point with shape (2,)")
        s_theta = phase_coordinates.copy()
        radial_chain = 1.0
        if self.radial_phase == "psi":
            psi = float(phase_coordinates[0])
            if psi <= 0.0:
                raise ValueError("the (psi, theta*) chart Jacobian is singular at psi=0")
            s_theta[0] = np.sqrt(psi)
            radial_chain = 0.5 / float(s_theta[0])
        J_s_theta = self.jacobian_s_theta(s_theta)
        return J_s_theta @ np.diag([radial_chain, 1.0])


@dataclass(frozen=True)
class PeriodicPointPhaseSpaceResponse:
    """First-order periodic-point response in an explicit healed chart.

    ``delta_z`` is the phase-coordinate solution of
    ``(I - DPk_z) delta_z = deltaPk_z``.  It is canonical only when the chart
    explicitly says so.  ``geometric_displacement_RZ_m`` is a mapped physical
    tangent-space physical displacement and must not be relabelled as a phase
    response.  The finite chart endpoint is retained separately because it
    includes second-order coordinate effects.
    """

    kind: str | None
    map_power: int | None
    radial_phase: str
    coordinate_choice: str
    canonical: bool
    symplectic: bool
    z0: np.ndarray
    z1: np.ndarray
    x0_RZ_m: np.ndarray
    x1_RZ_m: np.ndarray
    mapped_phase_endpoint_RZ_m: np.ndarray
    chart_jacobian: np.ndarray
    DPk_x: np.ndarray
    deltaPk_x: np.ndarray
    DPk_z: np.ndarray
    deltaPk_z: np.ndarray
    delta_z: np.ndarray
    s0: float
    theta_star0: float
    delta_s: float
    delta_psi: float
    delta_theta_star: float
    delta_theta_star_wrapped: float
    theta_star1_wrapped: float
    geometric_displacement_RZ_m: np.ndarray
    mapped_endpoint_displacement_RZ_m: np.ndarray
    singular_values: np.ndarray
    svd_rank: int
    svd_rcond: float
    solve_condition_number: float
    chart_condition_number: float
    residual_norm: float
    relative_residual: float
    regularized: bool
    status: str
    phase_space_valid: bool
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)

    @property
    def delta_radial_phase(self) -> float:
        """Return ``delta s`` or ``delta psi`` according to ``radial_phase``."""

        return float(self.delta_z[0])

    @property
    def geometric_dR_m(self) -> float:
        return float(self.geometric_displacement_RZ_m[0])

    @property
    def geometric_dZ_m(self) -> float:
        return float(self.geometric_displacement_RZ_m[1])


def _finite_vector2(value, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite length-2 vector")
    return vector


def _finite_matrix2(value, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 2 by 2 matrix")
    return matrix


def solve_periodic_point_phase_response(
    DPk_x,
    deltaPk_x,
    chart: HealedSurfaceSectionChart,
    *,
    x0_RZ_m=None,
    z0=None,
    kind: str | None = None,
    map_power: int | None = None,
    svd_rcond: float = 1.0e-12,
    condition_limit: float = 1.0e10,
    residual_tolerance: float = 1.0e-9,
) -> PeriodicPointPhaseSpaceResponse:
    """Solve a periodic-point response in an explicit healed phase chart.

    With ``x=(R,Z)`` and the local chart Jacobian ``J=dx/dz``, this performs

    ``DPk_z = J^-1 DPk_x J`` and ``deltaPk_z = J^-1 deltaPk_x``,

    then solves ``(I-DPk_z) delta_z = deltaPk_z`` using truncated SVD.  No
    geometric polar angle is inferred; callers must provide the healed chart.
    """

    if not isinstance(chart, HealedSurfaceSectionChart):
        raise TypeError("chart must be a HealedSurfaceSectionChart")
    matrix_x = _finite_matrix2(DPk_x, "DPk_x")
    forcing_x = _finite_vector2(deltaPk_x, "deltaPk_x")
    rcond = float(svd_rcond)
    cond_limit = float(condition_limit)
    residual_limit = float(residual_tolerance)
    if not np.isfinite(rcond) or rcond <= 0.0 or rcond >= 1.0:
        raise ValueError("svd_rcond must lie strictly between zero and one")
    if not np.isfinite(cond_limit) or cond_limit <= 1.0:
        raise ValueError("condition_limit must be finite and greater than one")
    if not np.isfinite(residual_limit) or residual_limit < 0.0:
        raise ValueError("residual_tolerance must be finite and non-negative")

    if z0 is None and x0_RZ_m is None:
        raise ValueError("provide z0 or x0_RZ_m")
    phase0 = None if z0 is None else _finite_vector2(z0, "z0")
    physical0 = None if x0_RZ_m is None else _finite_vector2(x0_RZ_m, "x0_RZ_m")
    if phase0 is None:
        phase0 = np.asarray(chart.x_to_z(physical0), dtype=float)
    if physical0 is None:
        physical0 = np.asarray(chart.z_to_x(phase0), dtype=float)
    mapped0 = np.asarray(chart.z_to_x(phase0), dtype=float)
    consistency_scale = max(1.0, float(np.linalg.norm(physical0)))
    if not np.allclose(mapped0, physical0, rtol=1.0e-9, atol=1.0e-10 * consistency_scale):
        raise ValueError("z0 and x0_RZ_m are inconsistent with the healed chart")

    phase0 = phase0.copy()
    phase0[1] %= chart.theta_period
    physical0 = mapped0
    J = _finite_matrix2(chart.jacobian(phase0), "chart Jacobian")
    chart_singular_values = np.linalg.svd(J, compute_uv=False)
    chart_condition = (
        np.inf
        if chart_singular_values[-1] == 0.0
        else float(chart_singular_values[0] / chart_singular_values[-1])
    )
    chart_cutoff = 2.0 * np.finfo(float).eps * float(chart_singular_values[0])
    if chart_singular_values[-1] <= chart_cutoff:
        raise ValueError("healed chart Jacobian is singular at the periodic point")

    matrix_z = np.linalg.solve(J, matrix_x @ J)
    forcing_z = np.linalg.solve(J, forcing_x)
    response_matrix = np.eye(2) - matrix_z
    U, singular_values, Vh = np.linalg.svd(response_matrix, full_matrices=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    cutoff = rcond * largest
    retained = singular_values > cutoff
    rank = int(np.count_nonzero(retained))
    inverse_singular = np.zeros_like(singular_values)
    inverse_singular[retained] = 1.0 / singular_values[retained]
    delta_z = Vh.T @ (inverse_singular * (U.T @ forcing_z))
    solve_condition = (
        np.inf
        if singular_values[-1] == 0.0
        else float(singular_values[0] / singular_values[-1])
    )
    residual = response_matrix @ delta_z - forcing_z
    residual_norm = float(np.linalg.norm(residual))
    forcing_norm = float(np.linalg.norm(forcing_z))
    relative_residual = (
        residual_norm if forcing_norm == 0.0 else residual_norm / forcing_norm
    )

    theta0 = float(phase0[1])
    delta_theta = float(delta_z[1])
    theta1_wrapped = float((theta0 + delta_theta) % chart.theta_period)
    delta_theta_wrapped = periodic_phase_difference(
        theta1_wrapped,
        theta0,
        chart.theta_period,
    )
    phase1 = phase0 + delta_z
    phase1[1] = theta1_wrapped
    mapped_phase_endpoint = np.asarray(chart.z_to_x(phase1), dtype=float)
    geometric_displacement = J @ delta_z
    physical1 = physical0 + geometric_displacement
    mapped_endpoint_displacement = mapped_phase_endpoint - physical0

    radial0 = float(phase0[0])
    if chart.radial_phase == "psi":
        s0 = float(np.sqrt(radial0))
        delta_psi = float(delta_z[0])
        delta_s = float(delta_psi / (2.0 * s0))
    else:
        s0 = radial0
        delta_s = float(delta_z[0])
        delta_psi = float(2.0 * s0 * delta_s)

    if chart_condition > cond_limit:
        status = "chart_ill_conditioned"
    elif rank < 2:
        status = "rank_deficient"
    elif solve_condition > cond_limit:
        status = "ill_conditioned"
    elif relative_residual > residual_limit:
        status = "residual_too_large"
    else:
        status = "valid"
    valid = status == "valid"
    point_kind = None if kind is None else str(kind).strip().upper()
    if point_kind not in {None, "O", "X"}:
        raise ValueError("kind must be 'O', 'X', or None")
    power = None if map_power is None else _positive_integer(map_power, "map_power")
    result_metadata = dict(chart.metadata)
    result_metadata.update(
        {
            "linearization": "phase-space periodic-point response",
            "equation": "(I-DPk_z) delta_z = deltaPk_z",
            "transform": "DPk_z=J^-1 DPk_x J; deltaPk_z=J^-1 deltaPk_x",
            "solver": "truncated_svd",
            "geometric_displacement_semantics": (
                "linear tangent push-forward J delta_z; not a phase response"
            ),
            "mapped_endpoint_semantics": (
                "finite chart endpoint, including higher-order coordinate effects"
            ),
            "delta_s_semantics": (
                "first-order radial response; derived as delta_psi/(2*s0) "
                "when radial_phase='psi'"
            ),
        }
    )
    return PeriodicPointPhaseSpaceResponse(
        kind=point_kind,
        map_power=power,
        radial_phase=chart.radial_phase,
        coordinate_choice=chart.coordinate_choice,
        canonical=chart.canonical,
        symplectic=chart.symplectic,
        z0=phase0.copy(),
        z1=phase1.copy(),
        x0_RZ_m=physical0.copy(),
        x1_RZ_m=physical1.copy(),
        mapped_phase_endpoint_RZ_m=mapped_phase_endpoint.copy(),
        chart_jacobian=J.copy(),
        DPk_x=matrix_x.copy(),
        deltaPk_x=forcing_x.copy(),
        DPk_z=matrix_z.copy(),
        deltaPk_z=forcing_z.copy(),
        delta_z=delta_z.copy(),
        s0=s0,
        theta_star0=theta0,
        delta_s=delta_s,
        delta_psi=delta_psi,
        delta_theta_star=delta_theta,
        delta_theta_star_wrapped=delta_theta_wrapped,
        theta_star1_wrapped=theta1_wrapped,
        geometric_displacement_RZ_m=geometric_displacement.copy(),
        mapped_endpoint_displacement_RZ_m=mapped_endpoint_displacement.copy(),
        singular_values=singular_values.copy(),
        svd_rank=rank,
        svd_rcond=rcond,
        solve_condition_number=solve_condition,
        chart_condition_number=chart_condition,
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        regularized=rank < 2,
        status=status,
        phase_space_valid=valid,
        metadata=result_metadata,
    )


@dataclass(frozen=True)
class NewtonFixedPointState:
    """Validated O/X fixed point returned by a Newton ``P^m`` solve.

    ``R`` and ``Z`` are physical metres.  ``healed_theta`` is the angle on the
    healed reference surface and is required for O points used in a chain.
    """

    kind: str
    R: float
    Z: float
    map_power: int
    DPm: np.ndarray
    residual: float
    converged: bool = True
    residual_tolerance: float = 1.0e-8
    classification_tolerance: float = 1.0e-8
    healed_theta: float | None = None
    section_phi: float | None = None
    iterations: int | None = None
    label: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        kind = str(self.kind).upper()
        if kind not in {"O", "X"}:
            raise ValueError("fixed-point kind must be 'O' or 'X'")
        R = float(self.R)
        Z = float(self.Z)
        residual = float(self.residual)
        if not np.isfinite(R) or not np.isfinite(Z):
            raise ValueError("fixed-point R and Z must be finite physical metres")
        if not np.isfinite(residual) or residual < 0.0:
            raise ValueError("fixed-point residual must be finite and non-negative")
        residual_tolerance = float(self.residual_tolerance)
        if not np.isfinite(residual_tolerance) or residual_tolerance <= 0.0:
            raise ValueError("residual_tolerance must be positive and finite")
        if not bool(self.converged):
            raise ValueError("validated Newton fixed points must be converged")
        if residual > residual_tolerance:
            raise ValueError("fixed-point residual exceeds residual_tolerance")
        matrix = np.asarray(self.DPm, dtype=float)
        if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
            raise ValueError("fixed-point DPm must be a finite 2 by 2 matrix")
        classification_tolerance = float(self.classification_tolerance)
        if not np.isfinite(classification_tolerance) or classification_tolerance < 0.0:
            raise ValueError("classification_tolerance must be finite and non-negative")
        trace = float(np.trace(matrix))
        determinant = float(np.linalg.det(matrix))
        discriminant = trace * trace - 4.0 * determinant
        discriminant_tolerance = classification_tolerance * max(
            1.0,
            trace * trace,
            abs(determinant),
        )
        if kind == "O" and discriminant >= -discriminant_tolerance:
            raise ValueError("O point DPm must have an elliptic complex-conjugate eigenpair")
        if kind == "X" and discriminant <= discriminant_tolerance:
            raise ValueError("X point DPm must have a real hyperbolic eigenpair")
        map_power = _positive_integer(self.map_power, "map_power")
        theta = None if self.healed_theta is None else float(self.healed_theta) % TWOPI
        if theta is not None and not np.isfinite(theta):
            raise ValueError("healed_theta must be finite")
        section_phi = None if self.section_phi is None else float(self.section_phi)
        if section_phi is not None and not np.isfinite(section_phi):
            raise ValueError("section_phi must be finite")
        iterations = None if self.iterations is None else int(self.iterations)
        if iterations is not None and iterations < 0:
            raise ValueError("iterations must be non-negative")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "Z", Z)
        object.__setattr__(self, "map_power", map_power)
        object.__setattr__(self, "DPm", matrix.copy())
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "residual_tolerance", residual_tolerance)
        object.__setattr__(self, "classification_tolerance", classification_tolerance)
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "healed_theta", theta)
        object.__setattr__(self, "section_phi", section_phi)
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "label", str(self.label).strip())
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def position_RZ_m(self) -> np.ndarray:
        """Return the physical section position ``[R, Z]`` in metres."""

        return np.asarray([self.R, self.Z], dtype=float)

    @property
    def residue(self) -> float:
        """Greene residue computed from this point's complete ``DP^m``."""

        return greene_residue(self.DPm)

    @classmethod
    def from_fixed_point(
        cls,
        fixed_point: object,
        *,
        healed_theta: float | None = None,
        converged: bool | None = None,
        label: str | None = None,
        residual_tolerance: float | None = None,
        classification_tolerance: float = 1.0e-8,
    ) -> "NewtonFixedPointState":
        """Adapt a public island-chain fixed-point object or mapping."""

        metadata = dict(_public_value(fixed_point, "metadata", {}) or {})
        theta = healed_theta
        if theta is None:
            theta = _public_value(fixed_point, "healed_theta", metadata.get("healed_theta"))
        did_converge = converged
        if did_converge is None:
            did_converge = bool(_public_value(fixed_point, "converged", True))
        point_label = label
        if point_label is None:
            point_label = str(_public_value(fixed_point, "label", metadata.get("label", "")))
        if not point_label and metadata.get("point_index") is not None:
            point_label = f"{str(_public_value(fixed_point, 'kind')).lower()}{int(metadata['point_index'])}"
        section_phi = _public_value(fixed_point, "section_phi", None)
        if section_phi is None:
            section_phi = _public_value(fixed_point, "phi", metadata.get("section_phi"))
        tolerance = residual_tolerance
        if tolerance is None:
            tolerance = metadata.get("residual_tolerance", metadata.get("residual_tol", 1.0e-8))
        return cls(
            kind=str(_public_value(fixed_point, "kind")),
            R=float(_public_value(fixed_point, "R")),
            Z=float(_public_value(fixed_point, "Z")),
            map_power=int(_public_value(fixed_point, "map_power", metadata.get("map_power"))),
            DPm=np.asarray(_public_value(fixed_point, "DPm"), dtype=float),
            residual=float(_public_value(fixed_point, "residual", metadata.get("residual", np.inf))),
            residual_tolerance=float(tolerance),
            classification_tolerance=classification_tolerance,
            converged=bool(did_converge),
            healed_theta=None if theta is None else float(theta),
            section_phi=section_phi,
            iterations=_public_value(fixed_point, "iterations", metadata.get("iterations")),
            label=point_label,
            metadata=metadata,
        )


@dataclass(frozen=True)
class FixedPointChainValidation:
    """Validated resonant chain assembled from Newton O/X fixed points."""

    label: str
    m: int
    n: int | None
    fixed_points: Sequence[NewtonFixedPointState]
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        label = _nonempty_label(self.label)
        harmonic = _positive_integer(self.m, "m")
        n = None if self.n is None else int(self.n)
        metadata = dict(self.metadata or {})
        nardon_n_raw = metadata.get("nardon_n")
        if nardon_n_raw is not None:
            nardon_n = int(nardon_n_raw)
            if isinstance(nardon_n_raw, (float, np.floating)) and float(
                nardon_n_raw
            ) != float(nardon_n):
                raise ValueError("metadata nardon_n must be an integer")
            metadata["nardon_n"] = nardon_n
        points = tuple(self.fixed_points)
        if not points or not all(isinstance(point, NewtonFixedPointState) for point in points):
            raise ValueError("fixed_points must contain validated NewtonFixedPointState objects")
        if not any(point.kind == "O" for point in points) or not any(point.kind == "X" for point in points):
            raise ValueError("a validated chain requires at least one O point and one X point")
        if any(point.map_power != harmonic for point in points):
            raise ValueError("each fixed-point map_power must equal chain m")
        if any(not point.converged for point in points):
            raise ValueError("a validated chain cannot contain unconverged Newton points")
        theta_O = [point.healed_theta for point in points if point.kind == "O"]
        if any(theta is None for theta in theta_O):
            raise ValueError("every O point requires healed_theta for chain phase")
        phi_O = [point.section_phi for point in points if point.kind == "O"]
        has_phi = [value is not None for value in phi_O]
        if any(has_phi) and not all(has_phi):
            raise ValueError("O points must either all provide section_phi or all omit it")
        spans_sections = False
        if all(has_phi):
            phi_values = np.asarray(phi_O, dtype=float)
            spans_sections = any(
                abs(periodic_phase_difference(value, phi_values[0])) > 1.0e-10
                for value in phi_values[1:]
            )
            if spans_sections and nardon_n_raw is None:
                raise ValueError(
                    "multi-section O chains require metadata['nardon_n']; "
                    "for positive physical q=m/n0 use nardon_n=-n0"
                )
        if all(has_phi) and nardon_n_raw is not None:
            fixed_point_chain_helical_phase(theta_O, phi_O, harmonic, metadata["nardon_n"])
            metadata["phase_convention"] = "Arg(mean(exp(i*(m*theta*+n_N*phi))))/m"
        else:
            fixed_point_chain_phase(theta_O, harmonic)
            metadata["phase_convention"] = "Arg(mean(exp(i*m*theta*)))/m"
        metadata["multi_section"] = spans_sections
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "m", harmonic)
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "fixed_points", points)
        object.__setattr__(self, "metadata", metadata)

    @property
    def o_points(self) -> tuple[NewtonFixedPointState, ...]:
        return tuple(point for point in self.fixed_points if point.kind == "O")

    @property
    def x_points(self) -> tuple[NewtonFixedPointState, ...]:
        return tuple(point for point in self.fixed_points if point.kind == "X")

    @property
    def theta_O(self) -> np.ndarray:
        return np.asarray([point.healed_theta for point in self.o_points], dtype=float)

    @property
    def phi_O(self) -> np.ndarray | None:
        values = tuple(point.section_phi for point in self.o_points)
        if not values or all(value is None for value in values):
            return None
        return np.asarray(values, dtype=float)

    @property
    def nardon_n(self) -> int | None:
        value = self.metadata.get("nardon_n")
        return None if value is None else int(value)

    @property
    def helical_phasor(self) -> complex | None:
        phi = self.phi_O
        if phi is None or self.nardon_n is None:
            return None
        return fixed_point_chain_helical_phasor(
            self.theta_O,
            phi,
            self.m,
            self.nardon_n,
        )

    @property
    def resonant_phase(self) -> float:
        phasor = self.helical_phasor
        if phasor is not None:
            return float(np.angle(phasor))
        return float(self.m * fixed_point_chain_phase(self.theta_O, self.m))

    @property
    def phase(self) -> float:
        return float(self.resonant_phase / self.m)

    @property
    def phase_period(self) -> float:
        return float(TWOPI / self.m)

    @property
    def coherence(self) -> float:
        phasor = self.helical_phasor
        if phasor is not None:
            return float(np.clip(abs(phasor), 0.0, 1.0))
        return fixed_point_chain_coherence(self.theta_O, self.m)

    @property
    def residues(self) -> np.ndarray:
        return np.asarray([point.residue for point in self.fixed_points], dtype=float)


IslandChainValidation = FixedPointChainValidation


@dataclass(frozen=True)
class MagneticAxisCoreShift:
    """Physical displacement between registered reference/current core data."""

    axis_reference_RZ_m: np.ndarray | None = None
    axis_current_RZ_m: np.ndarray | None = None
    core_reference_RZ_m: np.ndarray | None = None
    core_current_RZ_m: np.ndarray | None = None
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        axis_ref, axis_cur = _coordinate_pair(
            self.axis_reference_RZ_m,
            self.axis_current_RZ_m,
            "axis",
            exact_vector=True,
        )
        core_ref, core_cur = _coordinate_pair(
            self.core_reference_RZ_m,
            self.core_current_RZ_m,
            "core",
            exact_vector=False,
        )
        if axis_ref is None and core_ref is None:
            raise ValueError("axis or registered core coordinates are required")
        object.__setattr__(self, "axis_reference_RZ_m", axis_ref)
        object.__setattr__(self, "axis_current_RZ_m", axis_cur)
        object.__setattr__(self, "core_reference_RZ_m", core_ref)
        object.__setattr__(self, "core_current_RZ_m", core_cur)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def axis_shift_vector_m(self) -> np.ndarray | None:
        if self.axis_reference_RZ_m is None:
            return None
        return self.axis_current_RZ_m - self.axis_reference_RZ_m

    @property
    def axis_shift_m(self) -> float | None:
        vector = self.axis_shift_vector_m
        return None if vector is None else float(np.linalg.norm(vector))

    @property
    def core_displacement_RZ_m(self) -> np.ndarray | None:
        if self.core_reference_RZ_m is None:
            return None
        return self.core_current_RZ_m - self.core_reference_RZ_m

    @property
    def core_shift_rms_m(self) -> float | None:
        displacement = self.core_displacement_RZ_m
        if displacement is None:
            return None
        norms = np.linalg.norm(displacement.reshape(-1, 2), axis=1)
        return float(np.sqrt(np.mean(norms**2)))

    @property
    def core_shift_max_m(self) -> float | None:
        displacement = self.core_displacement_RZ_m
        if displacement is None:
            return None
        return float(np.max(np.linalg.norm(displacement.reshape(-1, 2), axis=1)))

    @classmethod
    def from_core_snapshots(
        cls,
        reference: object,
        current: object,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> "MagneticAxisCoreShift":
        """Build shifts from public core-snapshot attributes or mappings."""

        axis_ref = _public_value(reference, "axis", None)
        axis_cur = _public_value(current, "axis", None)
        ref_R = _public_value(reference, "surface_R", None)
        ref_Z = _public_value(reference, "surface_Z", None)
        cur_R = _public_value(current, "surface_R", None)
        cur_Z = _public_value(current, "surface_Z", None)
        core_ref = None if ref_R is None or ref_Z is None else np.stack([ref_R, ref_Z], axis=-1)
        core_cur = None if cur_R is None or cur_Z is None else np.stack([cur_R, cur_Z], axis=-1)
        return cls(
            axis_reference_RZ_m=axis_ref,
            axis_current_RZ_m=axis_cur,
            core_reference_RZ_m=core_ref,
            core_current_RZ_m=core_cur,
            metadata={} if metadata is None else metadata,
        )


def _coordinate_pair(reference, current, name: str, *, exact_vector: bool):
    if (reference is None) != (current is None):
        raise ValueError(f"{name} reference and current coordinates must be supplied together")
    if reference is None:
        return None, None
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.shape != cur.shape:
        raise ValueError(f"{name} reference and current coordinates must have matching shapes")
    if exact_vector:
        if ref.shape != (2,):
            raise ValueError("axis coordinates must have shape (2,)")
    elif ref.ndim < 1 or ref.shape[-1] != 2 or ref.size == 0:
        raise ValueError("core coordinates must have non-empty shape (..., 2)")
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(cur)):
        raise ValueError(f"{name} coordinates must be finite physical metres")
    return ref.copy(), cur.copy()


@dataclass(frozen=True)
class ManifoldTraceProvenance:
    """Resolution inputs retained from an external manifold trace."""

    source: str = "pyna.toroidal.flt.trace_fixed_point_manifolds_field"
    n_turns: int | None = None
    integration_step: float | None = None
    seed_distances_m: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        source = _nonempty_label(self.source, "source")
        n_turns = None if self.n_turns is None else _positive_integer(self.n_turns, "n_turns")
        step = None if self.integration_step is None else float(self.integration_step)
        if step is not None and (not np.isfinite(step) or step <= 0.0):
            raise ValueError("integration_step must be positive and finite")
        distances = np.asarray(
            [] if self.seed_distances_m is None else self.seed_distances_m,
            dtype=float,
        ).ravel()
        if distances.size and (not np.all(np.isfinite(distances)) or np.any(distances <= 0.0)):
            raise ValueError("seed_distances_m must be positive finite metres")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "n_turns", n_turns)
        object.__setattr__(self, "integration_step", step)
        object.__setattr__(self, "seed_distances_m", distances.copy())
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class ManifoldBranchSamples:
    """One side of a traced stable or unstable manifold in physical metres."""

    label: str
    stability: str
    points_RZ_m: np.ndarray
    side: int | None = None
    section_phi: float | None = None
    provenance: ManifoldTraceProvenance = field(default_factory=ManifoldTraceProvenance)
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        label = _nonempty_label(self.label)
        stability = str(self.stability).lower()
        if stability not in {"stable", "unstable"}:
            raise ValueError("stability must be 'stable' or 'unstable'")
        points = np.asarray(self.points_RZ_m, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
            raise ValueError("points_RZ_m must have shape (n, 2) with n >= 2")
        if not np.all(np.isfinite(points)):
            raise ValueError("manifold samples must be finite physical metres")
        side = None if self.side is None else int(self.side)
        if side not in {None, -1, 1}:
            raise ValueError("side must be -1, +1, or None")
        section_phi = None if self.section_phi is None else float(self.section_phi)
        if section_phi is not None and not np.isfinite(section_phi):
            raise ValueError("section_phi must be finite")
        if not isinstance(self.provenance, ManifoldTraceProvenance):
            raise TypeError("provenance must be ManifoldTraceProvenance")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "stability", stability)
        object.__setattr__(self, "points_RZ_m", points.copy())
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "section_phi", section_phi)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def segment_lengths_m(self) -> np.ndarray:
        return np.linalg.norm(np.diff(self.points_RZ_m, axis=0), axis=1)

    @property
    def median_segment_length_m(self) -> float:
        return float(np.median(self.segment_lengths_m))

    @property
    def max_segment_length_m(self) -> float:
        return float(np.max(self.segment_lengths_m))


def manifold_branches_from_trace(
    payload: Mapping[str, object],
    *,
    n_turns: int | None = None,
    integration_step: float | None = None,
    source: str = "pyna.toroidal.flt.trace_fixed_point_manifolds_field",
) -> tuple[ManifoldBranchSamples, ...]:
    """Adapt one public island-chain manifold payload into side branches.

    The public tracer's ``s_*`` and ``u_*`` arrays are consumed as-is.  The
    accompanying ``*_point_side`` arrays are used to prevent segments from
    joining the two physical sides of an X point.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a public manifold mapping")
    origin_label = payload.get("manifold_origin_label")
    if origin_label is None:
        origin_label = f"orbit{payload.get('orbit_id', 'unknown')}.point{payload.get('point_index', 'unknown')}"
    branches: list[ManifoldBranchSamples] = []
    for code, stability in (("s", "stable"), ("u", "unstable")):
        R = np.asarray(payload.get(f"{code}_R", []), dtype=float).ravel()
        Z = np.asarray(payload.get(f"{code}_Z", []), dtype=float).ravel()
        if R.size != Z.size:
            raise ValueError(f"{code}_R and {code}_Z must have matching lengths")
        if R.size == 0:
            continue
        raw_sides = payload.get(f"{code}_point_side")
        if raw_sides is None:
            sides = np.ones(R.size, dtype=int)
            side_values: tuple[int | None, ...] = (None,)
        else:
            sides = np.asarray(raw_sides, dtype=float).ravel()
            if sides.size != R.size or not np.all(np.isfinite(sides)):
                raise ValueError(f"{code}_point_side must match traced points")
            side_values = tuple(int(value) for value in sorted(set(np.sign(sides).astype(int))) if value != 0)
        seed_distances = np.asarray(payload.get(f"{code}_seed_distance", []), dtype=float).ravel()
        seed_sides = np.asarray(payload.get(f"{code}_seed_side", []), dtype=float).ravel()
        for side in side_values:
            mask = np.ones(R.size, dtype=bool) if side is None else np.sign(sides).astype(int) == side
            points = np.column_stack([R[mask], Z[mask]])
            if points.shape[0] < 2:
                continue
            branch_seed_distances = seed_distances
            if side is not None and seed_sides.size == seed_distances.size:
                branch_seed_distances = seed_distances[np.sign(seed_sides).astype(int) == side]
            generation = np.asarray(payload.get(f"{code}_generation", []), dtype=int).ravel()
            trace_metadata = {
                "seed_spacing": payload.get("seed_spacing"),
                "seed_ratio": payload.get(f"{code}_seed_ratio"),
                "map_span": payload.get("manifold_field_period"),
                "map_span_source": payload.get("manifold_field_period_source"),
                "max_generation": int(np.max(generation[mask])) if generation.size == R.size else None,
            }
            provenance = ManifoldTraceProvenance(
                source=source,
                n_turns=n_turns,
                integration_step=integration_step,
                seed_distances_m=branch_seed_distances,
                metadata=trace_metadata,
            )
            side_name = "all" if side is None else "minus" if side < 0 else "plus"
            branches.append(
                ManifoldBranchSamples(
                    label=f"{origin_label}.{stability}.{side_name}",
                    stability=stability,
                    points_RZ_m=points,
                    side=side,
                    section_phi=payload.get("origin_phi"),
                    provenance=provenance,
                    metadata={
                        "chain_id": payload.get("chain_id"),
                        "orbit_id": payload.get("orbit_id"),
                        "point_index": payload.get("point_index"),
                    },
                )
            )
    return tuple(branches)


@dataclass(frozen=True)
class LocalManifoldMeasurementLine:
    """A bounded local radial/normal line in a physical ``(R, Z)`` section."""

    label: str
    origin_RZ_m: np.ndarray
    direction_RZ: np.ndarray
    offset_bounds_m: tuple[float, float] = (-np.inf, np.inf)
    reference_offset_m: float = 0.0
    kind: str = "boundary_normal"
    section_phi: float | None = None
    minimum_crossing_sine: float = 1.0e-3
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        label = _nonempty_label(self.label)
        origin = np.asarray(self.origin_RZ_m, dtype=float).ravel()
        direction = np.asarray(self.direction_RZ, dtype=float).ravel()
        if origin.shape != (2,) or not np.all(np.isfinite(origin)):
            raise ValueError("origin_RZ_m must be a finite physical (R, Z) point")
        if direction.shape != (2,) or not np.all(np.isfinite(direction)):
            raise ValueError("direction_RZ must be a finite 2-vector")
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise ValueError("direction_RZ must be non-zero")
        bounds = tuple(float(value) for value in self.offset_bounds_m)
        if len(bounds) != 2 or np.isnan(bounds[0]) or np.isnan(bounds[1]) or bounds[1] <= bounds[0]:
            raise ValueError("offset_bounds_m must be an ordered pair")
        reference = float(self.reference_offset_m)
        if not np.isfinite(reference) or not (bounds[0] <= reference <= bounds[1]):
            raise ValueError("reference_offset_m must be finite and inside offset_bounds_m")
        kind = str(self.kind).lower()
        if kind not in {"boundary_normal", "radial", "custom"}:
            raise ValueError("line kind must be boundary_normal, radial, or custom")
        section_phi = None if self.section_phi is None else float(self.section_phi)
        if section_phi is not None and not np.isfinite(section_phi):
            raise ValueError("section_phi must be finite")
        minimum_sine = float(self.minimum_crossing_sine)
        if not np.isfinite(minimum_sine) or not 0.0 <= minimum_sine <= 1.0:
            raise ValueError("minimum_crossing_sine must lie in [0, 1]")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "origin_RZ_m", origin.copy())
        object.__setattr__(self, "direction_RZ", direction / norm)
        object.__setattr__(self, "offset_bounds_m", bounds)
        object.__setattr__(self, "reference_offset_m", reference)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "section_phi", section_phi)
        object.__setattr__(self, "minimum_crossing_sine", minimum_sine)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


LocalMeasurementLine = LocalManifoldMeasurementLine


@dataclass(frozen=True)
class PolylineLineIntersection:
    """One isolated intersection of a polyline and a measurement line."""

    point_RZ_m: np.ndarray
    offset_m: float
    segment_index: int
    segment_fraction: float
    crossing_sine: float

    def __post_init__(self) -> None:
        point = np.asarray(self.point_RZ_m, dtype=float).ravel()
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("intersection point must be finite physical metres")
        offset = float(self.offset_m)
        fraction = float(self.segment_fraction)
        crossing_sine = float(self.crossing_sine)
        if not np.isfinite(offset) or not np.isfinite(fraction) or not np.isfinite(crossing_sine):
            raise ValueError("intersection parameters must be finite")
        if int(self.segment_index) < 0 or not (-1.0e-12 <= fraction <= 1.0 + 1.0e-12):
            raise ValueError("invalid polyline segment intersection")
        if not 0.0 <= crossing_sine <= 1.0 + 1.0e-12:
            raise ValueError("crossing_sine must lie in [0, 1]")
        object.__setattr__(self, "point_RZ_m", point.copy())
        object.__setattr__(self, "offset_m", offset)
        object.__setattr__(self, "segment_index", int(self.segment_index))
        object.__setattr__(self, "segment_fraction", float(np.clip(fraction, 0.0, 1.0)))
        object.__setattr__(self, "crossing_sine", float(np.clip(crossing_sine, 0.0, 1.0)))


def polyline_line_intersections(
    points_RZ_m,
    line: LocalManifoldMeasurementLine,
    *,
    atol_m: float = 1.0e-12,
) -> tuple[PolylineLineIntersection, ...]:
    """Return isolated intersections with a bounded infinite-direction line.

    Collinear segments are not isolated crossings and are therefore omitted.
    A crossing at a shared polyline vertex is deduplicated.
    """

    if not isinstance(line, LocalManifoldMeasurementLine):
        raise TypeError("line must be LocalManifoldMeasurementLine")
    points = np.asarray(points_RZ_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError("points_RZ_m must have shape (n, 2) with n >= 2")
    if not np.all(np.isfinite(points)):
        raise ValueError("polyline points must be finite physical metres")
    atol = float(atol_m)
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol_m must be finite and non-negative")
    origin = line.origin_RZ_m
    direction = line.direction_RZ
    lo, hi = line.offset_bounds_m
    intersections: list[PolylineLineIntersection] = []
    for index, (start, stop) in enumerate(zip(points[:-1], points[1:])):
        segment = stop - start
        length = float(np.linalg.norm(segment))
        if length <= atol:
            continue
        denominator = _cross2(segment, direction)
        crossing_sine = abs(denominator) / length
        if crossing_sine < line.minimum_crossing_sine:
            continue
        if abs(denominator) <= max(atol, np.finfo(float).eps * length * 16.0):
            continue
        delta = origin - start
        fraction = _cross2(delta, direction) / denominator
        offset = _cross2(delta, segment) / denominator
        fraction_tol = atol / max(length, np.finfo(float).tiny)
        if fraction < -fraction_tol or fraction > 1.0 + fraction_tol:
            continue
        if offset < lo - atol or offset > hi + atol:
            continue
        fraction = float(np.clip(fraction, 0.0, 1.0))
        offset = float(np.clip(offset, lo, hi)) if np.isfinite(lo) and np.isfinite(hi) else float(offset)
        point = origin + offset * direction
        duplicate = any(
            abs(item.offset_m - offset) <= atol
            and np.linalg.norm(item.point_RZ_m - point) <= atol
            for item in intersections
        )
        if not duplicate:
            intersections.append(
                PolylineLineIntersection(
                    point_RZ_m=point,
                    offset_m=offset,
                    segment_index=index,
                    segment_fraction=fraction,
                    crossing_sine=crossing_sine,
                )
            )
    return tuple(sorted(intersections, key=lambda item: (item.offset_m, item.segment_index)))


def manifold_line_intersections(
    branch: ManifoldBranchSamples,
    line: LocalManifoldMeasurementLine,
    *,
    atol_m: float = 1.0e-12,
) -> tuple[PolylineLineIntersection, ...]:
    """Intersect a validated manifold branch with a local measurement line."""

    if not isinstance(branch, ManifoldBranchSamples):
        raise TypeError("branch must be ManifoldBranchSamples")
    if branch.section_phi is not None and line.section_phi is not None:
        if abs(periodic_phase_difference(branch.section_phi, line.section_phi)) > 1.0e-10:
            raise ValueError("manifold branch and measurement line must share a toroidal section")
    return polyline_line_intersections(branch.points_RZ_m, line, atol_m=atol_m)


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


@dataclass(frozen=True)
class StableUnstableBranchPair:
    """Corresponding W^s/W^u branches used to measure lobe splitting."""

    label: str
    stable: ManifoldBranchSamples
    unstable: ManifoldBranchSamples

    def __post_init__(self) -> None:
        label = _nonempty_label(self.label)
        if not isinstance(self.stable, ManifoldBranchSamples) or self.stable.stability != "stable":
            raise ValueError("stable must be a stable ManifoldBranchSamples")
        if not isinstance(self.unstable, ManifoldBranchSamples) or self.unstable.stability != "unstable":
            raise ValueError("unstable must be an unstable ManifoldBranchSamples")
        _validate_common_section(self.stable, self.unstable)
        object.__setattr__(self, "label", label)


@dataclass(frozen=True)
class StableUnstableSplittingSample:
    """One transverse homoclinic/lobe splitting measurement in metres."""

    line_label: str
    stable_offset_m: float
    unstable_offset_m: float
    splitting_m: float
    stable_crossing_sine: float = 1.0
    unstable_crossing_sine: float = 1.0

    def __post_init__(self) -> None:
        label = _nonempty_label(self.line_label, "line_label")
        stable = float(self.stable_offset_m)
        unstable = float(self.unstable_offset_m)
        splitting = float(self.splitting_m)
        stable_sine = float(self.stable_crossing_sine)
        unstable_sine = float(self.unstable_crossing_sine)
        if not np.all(np.isfinite([stable, unstable, splitting])) or splitting < 0.0:
            raise ValueError("stable/unstable splitting samples must be finite physical metres")
        if not 0.0 <= stable_sine <= 1.0 or not 0.0 <= unstable_sine <= 1.0:
            raise ValueError("crossing sines must lie in [0, 1]")
        if not np.isclose(splitting, abs(unstable - stable), rtol=1.0e-12, atol=1.0e-14):
            raise ValueError("splitting_m must equal abs(unstable_offset_m - stable_offset_m)")
        object.__setattr__(self, "line_label", label)
        object.__setattr__(self, "stable_offset_m", stable)
        object.__setattr__(self, "unstable_offset_m", unstable)
        object.__setattr__(self, "splitting_m", splitting)
        object.__setattr__(self, "stable_crossing_sine", stable_sine)
        object.__setattr__(self, "unstable_crossing_sine", unstable_sine)


@dataclass(frozen=True)
class StableUnstableBranchSplitting:
    """Median/MAD splitting summary for one W^s/W^u branch pair."""

    label: str
    samples: Sequence[StableUnstableSplittingSample]

    def __post_init__(self) -> None:
        label = _nonempty_label(self.label)
        samples = tuple(self.samples)
        if not samples or not all(isinstance(item, StableUnstableSplittingSample) for item in samples):
            raise ValueError("branch splitting requires StableUnstableSplittingSample objects")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "samples", samples)

    @property
    def splittings_m(self) -> np.ndarray:
        return np.asarray([sample.splitting_m for sample in self.samples], dtype=float)

    @property
    def median_m(self) -> float:
        return float(np.median(self.splittings_m))

    @property
    def spread_m(self) -> float:
        return float(np.median(np.abs(self.splittings_m - self.median_m)))


@dataclass(frozen=True)
class StableUnstableSplittingMetric:
    """W^s/W^u lobe splitting, explicitly not full island width."""

    label: str
    branches: Sequence[StableUnstableBranchSplitting]
    resolution_provenance: Mapping[str, object]
    rejected_measurements: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        label = _nonempty_label(self.label)
        branches = tuple(self.branches)
        if not branches or not all(isinstance(item, StableUnstableBranchSplitting) for item in branches):
            raise ValueError("splitting metric requires stable/unstable branch summaries")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "branches", branches)
        _set_metric_mappings(self)

    @property
    def splittings_m(self) -> np.ndarray:
        return np.concatenate([branch.splittings_m for branch in self.branches])

    @property
    def median_m(self) -> float:
        return float(np.median(self.splittings_m))

    @property
    def spread_m(self) -> float:
        return float(np.median(np.abs(self.splittings_m - self.median_m)))

    @property
    def branch_medians_m(self) -> dict[str, float]:
        return {branch.label: branch.median_m for branch in self.branches}

    @property
    def branch_spreads_m(self) -> dict[str, float]:
        return {branch.label: branch.spread_m for branch in self.branches}


@dataclass(frozen=True)
class SeparatrixEnvelopeBranchPair:
    """Explicit inner/outer branches defining the full island envelope."""

    label: str
    inner: ManifoldBranchSamples
    outer: ManifoldBranchSamples
    inner_reference_offset_m: float | None = None
    outer_reference_offset_m: float | None = None

    def __post_init__(self) -> None:
        label = _nonempty_label(self.label)
        if not isinstance(self.inner, ManifoldBranchSamples):
            raise TypeError("inner must be ManifoldBranchSamples")
        if not isinstance(self.outer, ManifoldBranchSamples):
            raise TypeError("outer must be ManifoldBranchSamples")
        _validate_common_section(self.inner, self.outer)
        inner_reference = _optional_finite(self.inner_reference_offset_m, "inner_reference_offset_m")
        outer_reference = _optional_finite(self.outer_reference_offset_m, "outer_reference_offset_m")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "inner_reference_offset_m", inner_reference)
        object.__setattr__(self, "outer_reference_offset_m", outer_reference)


@dataclass(frozen=True)
class IslandEnvelopeWidthSample:
    """One ordered inner-to-outer separatrix-envelope width in metres."""

    line_label: str
    inner_offset_m: float
    outer_offset_m: float
    full_width_m: float
    inner_crossing_sine: float = 1.0
    outer_crossing_sine: float = 1.0

    def __post_init__(self) -> None:
        label = _nonempty_label(self.line_label, "line_label")
        inner = float(self.inner_offset_m)
        outer = float(self.outer_offset_m)
        width = float(self.full_width_m)
        inner_sine = float(self.inner_crossing_sine)
        outer_sine = float(self.outer_crossing_sine)
        if not np.all(np.isfinite([inner, outer, width])) or width < 0.0:
            raise ValueError("island-envelope width samples must be finite physical metres")
        if not 0.0 <= inner_sine <= 1.0 or not 0.0 <= outer_sine <= 1.0:
            raise ValueError("crossing sines must lie in [0, 1]")
        if not np.isclose(width, outer - inner, rtol=1.0e-12, atol=1.0e-14):
            raise ValueError("full_width_m must equal outer_offset_m - inner_offset_m")
        object.__setattr__(self, "line_label", label)
        object.__setattr__(self, "inner_offset_m", inner)
        object.__setattr__(self, "outer_offset_m", outer)
        object.__setattr__(self, "full_width_m", width)
        object.__setattr__(self, "inner_crossing_sine", inner_sine)
        object.__setattr__(self, "outer_crossing_sine", outer_sine)


@dataclass(frozen=True)
class IslandEnvelopeBranchWidth:
    """Median/MAD full-width summary for one inner/outer branch pair."""

    label: str
    samples: Sequence[IslandEnvelopeWidthSample]

    def __post_init__(self) -> None:
        label = _nonempty_label(self.label)
        samples = tuple(self.samples)
        if not samples or not all(isinstance(item, IslandEnvelopeWidthSample) for item in samples):
            raise ValueError("envelope width requires IslandEnvelopeWidthSample objects")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "samples", samples)

    @property
    def widths_m(self) -> np.ndarray:
        return np.asarray([sample.full_width_m for sample in self.samples], dtype=float)

    @property
    def median_m(self) -> float:
        return float(np.median(self.widths_m))

    @property
    def spread_m(self) -> float:
        return float(np.median(np.abs(self.widths_m - self.median_m)))


@dataclass(frozen=True)
class IslandEnvelopeWidthMetric:
    """Full inner-to-outer island separatrix width with provenance."""

    label: str
    branches: Sequence[IslandEnvelopeBranchWidth]
    resolution_provenance: Mapping[str, object]
    rejected_measurements: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        label = _nonempty_label(self.label)
        branches = tuple(self.branches)
        if not branches or not all(isinstance(item, IslandEnvelopeBranchWidth) for item in branches):
            raise ValueError("envelope metric requires inner/outer branch summaries")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "branches", branches)
        _set_metric_mappings(self)

    @property
    def widths_m(self) -> np.ndarray:
        return np.concatenate([branch.widths_m for branch in self.branches])

    @property
    def median_m(self) -> float:
        return float(np.median(self.widths_m))

    @property
    def spread_m(self) -> float:
        return float(np.median(np.abs(self.widths_m - self.median_m)))

    @property
    def branch_medians_m(self) -> dict[str, float]:
        return {branch.label: branch.median_m for branch in self.branches}

    @property
    def branch_spreads_m(self) -> dict[str, float]:
        return {branch.label: branch.spread_m for branch in self.branches}


def stable_unstable_splitting_from_manifolds(
    branch_pairs: Sequence[StableUnstableBranchPair],
    measurement_lines: Sequence[LocalManifoldMeasurementLine],
    *,
    label: str = "lobe",
    selection: str = "unique",
    require_all: bool = False,
    intersection_atol_m: float = 1.0e-12,
    metadata: Mapping[str, object] | None = None,
) -> StableUnstableSplittingMetric:
    """Measure W^s/W^u splitting along explicit physical transverse lines.

    This is a homoclinic/lobe or chaotic-layer splitting metric.  Coincident
    stable and unstable branches give zero even when the integrable island has
    nonzero full width.  It must not be interpreted as island width.
    """

    pairs, lines, selection_name, atol = _validate_transverse_inputs(
        branch_pairs,
        measurement_lines,
        StableUnstableBranchPair,
        selection,
        intersection_atol_m,
    )
    rejected: dict[str, str] = {}
    branch_results: list[StableUnstableBranchSplitting] = []
    trace_resolution: dict[str, object] = {}
    for pair in pairs:
        samples: list[StableUnstableSplittingSample] = []
        trace_resolution[pair.label] = {
            "stable": _branch_resolution(pair.stable),
            "unstable": _branch_resolution(pair.unstable),
        }
        for line in lines:
            stable_hits = manifold_line_intersections(pair.stable, line, atol_m=atol)
            unstable_hits = manifold_line_intersections(pair.unstable, line, atol_m=atol)
            stable_hit = _select_intersection(stable_hits, line, selection_name, atol)
            unstable_hit = _select_intersection(unstable_hits, line, selection_name, atol)
            key = f"{pair.label}:{line.label}"
            if stable_hit is None or unstable_hit is None:
                rejected[key] = f"stable_hits={len(stable_hits)}, unstable_hits={len(unstable_hits)}"
                if require_all:
                    raise ValueError(f"measurement {key!r} does not isolate both W^s/W^u crossings")
                continue
            samples.append(
                StableUnstableSplittingSample(
                    line_label=line.label,
                    stable_offset_m=stable_hit.offset_m,
                    unstable_offset_m=unstable_hit.offset_m,
                    splitting_m=abs(unstable_hit.offset_m - stable_hit.offset_m),
                    stable_crossing_sine=stable_hit.crossing_sine,
                    unstable_crossing_sine=unstable_hit.crossing_sine,
                )
            )
        if samples:
            branch_results.append(StableUnstableBranchSplitting(pair.label, samples))
    if not branch_results:
        raise ValueError("no measurement line produced both stable and unstable crossings")
    provenance = _transverse_provenance(
        algorithm="piecewise_linear_stable_unstable_splitting",
        quantity_definition="abs(s_unstable - s_stable) in physical metres; not island width",
        pairs=pairs,
        lines=lines,
        branch_results=branch_results,
        trace_resolution=trace_resolution,
        selection=selection_name,
        atol=atol,
    )
    return StableUnstableSplittingMetric(
        label=label,
        branches=branch_results,
        resolution_provenance=provenance,
        rejected_measurements=rejected,
        metadata={} if metadata is None else metadata,
    )


def separatrix_width_from_manifolds(
    branch_pairs: Sequence[SeparatrixEnvelopeBranchPair],
    measurement_lines: Sequence[LocalManifoldMeasurementLine],
    *,
    label: str = "separatrix",
    selection: str = "unique",
    require_all: bool = False,
    intersection_atol_m: float = 1.0e-12,
    metadata: Mapping[str, object] | None = None,
) -> IslandEnvelopeWidthMetric:
    """Measure full inner-to-outer island-envelope width in physical metres.

    Each pair explicitly identifies inner and outer separatrix branches; their
    stability may be the same or different.  Along an outward-oriented unit
    radial/normal line, an accepted width is ``s_outer - s_inner``.  Optional
    pair-specific reference offsets disambiguate multiple crossings when
    ``selection='nearest'``.  W^s/W^u splitting is a separate observable, and
    neither O-X distance nor Greene residue is accepted as a width surrogate.
    """

    pairs, lines, selection_name, atol = _validate_transverse_inputs(
        branch_pairs,
        measurement_lines,
        SeparatrixEnvelopeBranchPair,
        selection,
        intersection_atol_m,
    )
    rejected: dict[str, str] = {}
    branch_results: list[IslandEnvelopeBranchWidth] = []
    trace_resolution: dict[str, object] = {}
    for pair in pairs:
        samples: list[IslandEnvelopeWidthSample] = []
        trace_resolution[pair.label] = {
            "inner": _branch_resolution(pair.inner),
            "outer": _branch_resolution(pair.outer),
            "inner_reference_offset_m": pair.inner_reference_offset_m,
            "outer_reference_offset_m": pair.outer_reference_offset_m,
        }
        for line in lines:
            inner_hits = manifold_line_intersections(pair.inner, line, atol_m=atol)
            outer_hits = manifold_line_intersections(pair.outer, line, atol_m=atol)
            inner_hit = _select_intersection(
                inner_hits,
                line,
                selection_name,
                atol,
                reference_offset_m=pair.inner_reference_offset_m,
            )
            outer_hit = _select_intersection(
                outer_hits,
                line,
                selection_name,
                atol,
                reference_offset_m=pair.outer_reference_offset_m,
            )
            key = f"{pair.label}:{line.label}"
            if inner_hit is None or outer_hit is None:
                rejected[key] = f"inner_hits={len(inner_hits)}, outer_hits={len(outer_hits)}"
                if require_all:
                    raise ValueError(f"measurement {key!r} does not isolate inner/outer crossings")
                continue
            if outer_hit.offset_m < inner_hit.offset_m - atol:
                rejected[key] = "outer crossing precedes inner crossing along the oriented line"
                if require_all:
                    raise ValueError(f"measurement {key!r} has reversed inner/outer crossings")
                continue
            width = max(0.0, outer_hit.offset_m - inner_hit.offset_m)
            samples.append(
                IslandEnvelopeWidthSample(
                    line_label=line.label,
                    inner_offset_m=inner_hit.offset_m,
                    outer_offset_m=outer_hit.offset_m,
                    full_width_m=width,
                    inner_crossing_sine=inner_hit.crossing_sine,
                    outer_crossing_sine=outer_hit.crossing_sine,
                )
            )
        if samples:
            branch_results.append(IslandEnvelopeBranchWidth(pair.label, samples))
    if not branch_results:
        raise ValueError("no measurement line produced ordered inner and outer envelope crossings")
    provenance = _transverse_provenance(
        algorithm="piecewise_linear_inner_outer_separatrix_envelope",
        quantity_definition="s_outer - s_inner in physical metres; full island envelope width",
        pairs=pairs,
        lines=lines,
        branch_results=branch_results,
        trace_resolution=trace_resolution,
        selection=selection_name,
        atol=atol,
    )
    provenance["forbidden_width_surrogates"] = ("O-X distance", "Greene residue")
    return IslandEnvelopeWidthMetric(
        label=label,
        branches=branch_results,
        resolution_provenance=provenance,
        rejected_measurements=rejected,
        metadata={} if metadata is None else metadata,
    )


def _validate_transverse_inputs(branch_pairs, measurement_lines, pair_type, selection, atol):
    pairs = tuple(branch_pairs)
    lines = tuple(measurement_lines)
    if not pairs or not all(isinstance(pair, pair_type) for pair in pairs):
        raise ValueError(f"branch_pairs must contain at least one {pair_type.__name__}")
    if not lines or not all(isinstance(line, LocalManifoldMeasurementLine) for line in lines):
        raise ValueError("measurement_lines must contain at least one local line")
    if len({pair.label for pair in pairs}) != len(pairs):
        raise ValueError("branch-pair labels must be unique")
    if len({line.label for line in lines}) != len(lines):
        raise ValueError("measurement-line labels must be unique")
    selection_name = str(selection).lower()
    if selection_name not in {"unique", "nearest"}:
        raise ValueError("selection must be 'unique' or 'nearest'")
    atol_f = float(atol)
    if not np.isfinite(atol_f) or atol_f < 0.0:
        raise ValueError("intersection_atol_m must be finite and non-negative")
    return pairs, lines, selection_name, atol_f


def _select_intersection(
    intersections,
    line,
    selection: str,
    atol: float,
    *,
    reference_offset_m: float | None = None,
):
    if len(intersections) == 1:
        return intersections[0]
    if selection == "unique" or not intersections:
        return None
    reference = line.reference_offset_m if reference_offset_m is None else reference_offset_m
    ordered = sorted(intersections, key=lambda item: abs(item.offset_m - reference))
    if len(ordered) > 1:
        first = abs(ordered[0].offset_m - reference)
        second = abs(ordered[1].offset_m - reference)
        if abs(first - second) <= atol:
            return None
    return ordered[0]


def _transverse_provenance(
    *,
    algorithm,
    quantity_definition,
    pairs,
    lines,
    branch_results,
    trace_resolution,
    selection,
    atol,
):
    crossing_sines = [
        float(value)
        for branch in branch_results
        for sample in branch.samples
        for name, value in vars(sample).items()
        if name.endswith("_crossing_sine")
    ]
    return {
        "algorithm": algorithm,
        "quantity_definition": quantity_definition,
        "spread_definition": "median_absolute_deviation",
        "selection": selection,
        "intersection_atol_m": atol,
        "branch_pairs_requested": len(pairs),
        "measurement_lines_requested": len(lines),
        "accepted_samples": int(sum(len(branch.samples) for branch in branch_results)),
        "minimum_accepted_crossing_sine": min(crossing_sines),
        "branch_summary_m": {
            branch.label: {"median": branch.median_m, "spread": branch.spread_m}
            for branch in branch_results
        },
        "measurement_lines": {
            line.label: {
                "kind": line.kind,
                "origin_RZ_m": line.origin_RZ_m.copy(),
                "direction_RZ": line.direction_RZ.copy(),
                "offset_bounds_m": line.offset_bounds_m,
                "reference_offset_m": line.reference_offset_m,
                "section_phi": line.section_phi,
                "minimum_crossing_sine": line.minimum_crossing_sine,
            }
            for line in lines
        },
        "trace_resolution": trace_resolution,
    }


def _set_metric_mappings(metric) -> None:
    object.__setattr__(metric, "resolution_provenance", dict(metric.resolution_provenance or {}))
    object.__setattr__(
        metric,
        "rejected_measurements",
        {str(key): str(value) for key, value in dict(metric.rejected_measurements or {}).items()},
    )
    object.__setattr__(metric, "metadata", dict(metric.metadata or {}))


def _validate_common_section(first: ManifoldBranchSamples, second: ManifoldBranchSamples) -> None:
    if first.section_phi is None or second.section_phi is None:
        return
    if abs(periodic_phase_difference(first.section_phi, second.section_phi)) > 1.0e-10:
        raise ValueError("paired manifold branches must lie on the same toroidal section")


def _optional_finite(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _branch_resolution(branch: ManifoldBranchSamples) -> dict[str, object]:
    provenance = branch.provenance
    return {
        "label": branch.label,
        "stability": branch.stability,
        "side": branch.side,
        "section_phi": branch.section_phi,
        "source": provenance.source,
        "n_points": int(branch.points_RZ_m.shape[0]),
        "median_segment_length_m": branch.median_segment_length_m,
        "max_segment_length_m": branch.max_segment_length_m,
        "n_turns": provenance.n_turns,
        "integration_step": provenance.integration_step,
        "seed_distances_m": provenance.seed_distances_m.copy(),
        "trace_metadata": dict(provenance.metadata),
    }


@dataclass(frozen=True)
class DPKGrowthValidation:
    """Thin validator for one cumulative ``DP^k`` matrix.

    This record intentionally does not reproduce the history-based FTLE,
    recurrence, termination, or classification semantics of the existing
    public ``boundary_dpk_growth_metrics`` function.
    """

    label: str
    k: int
    DPk: np.ndarray
    alive_fraction: float | None = None
    termination_iteration: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)
    singular_values: np.ndarray = field(init=False, compare=False)
    eigenvalue_magnitudes: np.ndarray = field(init=False, compare=False)

    def __post_init__(self) -> None:
        label = _nonempty_label(self.label)
        k = _positive_integer(self.k, "k")
        matrix = np.asarray(self.DPk, dtype=float)
        if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
            raise ValueError("DPk must be a finite 2 by 2 matrix")
        singular = np.linalg.svd(matrix, compute_uv=False)
        eigen_magnitudes = np.abs(np.linalg.eigvals(matrix))
        if not np.all(np.isfinite(singular)) or not np.all(np.isfinite(eigen_magnitudes)):
            raise ValueError("DPk growth factors must be finite")
        alive = None if self.alive_fraction is None else float(self.alive_fraction)
        if alive is not None and (not np.isfinite(alive) or not 0.0 <= alive <= 1.0):
            raise ValueError("alive_fraction must lie in [0, 1]")
        termination = None if self.termination_iteration is None else int(self.termination_iteration)
        if termination is not None and termination < 0:
            raise ValueError("termination_iteration must be non-negative")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "DPk", matrix.copy())
        object.__setattr__(self, "alive_fraction", alive)
        object.__setattr__(self, "termination_iteration", termination)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        object.__setattr__(self, "singular_values", singular.copy())
        object.__setattr__(self, "eigenvalue_magnitudes", eigen_magnitudes.copy())

    @property
    def max_singular(self) -> float:
        return float(np.max(self.singular_values))

    @property
    def spectral_radius(self) -> float:
        return float(np.max(self.eigenvalue_magnitudes))

    @property
    def svd_growth_factor(self) -> float:
        return float(self.max_singular ** (1.0 / self.k))

    @property
    def eigen_growth_factor(self) -> float:
        return float(self.spectral_radius ** (1.0 / self.k))

    @property
    def svd_log_growth_per_iter(self) -> float:
        return float(np.log(max(self.max_singular, np.finfo(float).tiny)) / self.k)

    @property
    def eigen_log_growth_per_iter(self) -> float:
        return float(np.log(max(self.spectral_radius, np.finfo(float).tiny)) / self.k)

    @property
    def open_loss_fraction(self) -> float | None:
        return None if self.alive_fraction is None else float(1.0 - self.alive_fraction)


@dataclass(frozen=True)
class BoundaryNonlinearValidationState:
    """Complete validated nonlinear state for one boundary-control command."""

    chains: Sequence[FixedPointChainValidation] = field(default_factory=tuple)
    axis_core_shift: MagneticAxisCoreShift | None = None
    manifold_branches: Sequence[ManifoldBranchSamples] = field(default_factory=tuple)
    stable_unstable_splittings: Sequence[StableUnstableSplittingMetric] = field(default_factory=tuple)
    separatrix_widths: Sequence[IslandEnvelopeWidthMetric] = field(default_factory=tuple)
    dpk_growth: Sequence[DPKGrowthValidation] = field(default_factory=tuple)
    wall_metrics: Mapping[str, float] = field(default_factory=dict)
    heat_metrics: Mapping[str, float] = field(default_factory=dict)
    wall_metric_units: Mapping[str, str] = field(default_factory=dict)
    heat_metric_units: Mapping[str, str] = field(default_factory=dict)
    open_loss_fraction: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        chains = _typed_tuple(self.chains, FixedPointChainValidation, "chains")
        branches = _typed_tuple(self.manifold_branches, ManifoldBranchSamples, "manifold_branches")
        splittings = _typed_tuple(
            self.stable_unstable_splittings,
            StableUnstableSplittingMetric,
            "stable_unstable_splittings",
        )
        widths = _typed_tuple(self.separatrix_widths, IslandEnvelopeWidthMetric, "separatrix_widths")
        growth = _typed_tuple(self.dpk_growth, DPKGrowthValidation, "dpk_growth")
        if self.axis_core_shift is not None and not isinstance(self.axis_core_shift, MagneticAxisCoreShift):
            raise TypeError("axis_core_shift must be MagneticAxisCoreShift")
        for name, values in (
            ("chain", [item.label for item in chains]),
            ("manifold branch", [item.label for item in branches]),
            ("stable/unstable splitting", [item.label for item in splittings]),
            ("island envelope width", [item.label for item in widths]),
            ("DPk growth", [item.label for item in growth]),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{name} labels must be unique")
        wall = _validated_metric_mapping(self.wall_metrics, "wall_metrics")
        heat = _validated_metric_mapping(self.heat_metrics, "heat_metrics")
        wall_units = _validated_metric_units(self.wall_metric_units, wall, "wall_metric_units")
        heat_units = _validated_metric_units(self.heat_metric_units, heat, "heat_metric_units")
        loss = None if self.open_loss_fraction is None else float(self.open_loss_fraction)
        if loss is not None and (not np.isfinite(loss) or not 0.0 <= loss <= 1.0):
            raise ValueError("open_loss_fraction must lie in [0, 1]")
        object.__setattr__(self, "chains", chains)
        object.__setattr__(self, "manifold_branches", branches)
        object.__setattr__(self, "stable_unstable_splittings", splittings)
        object.__setattr__(self, "separatrix_widths", widths)
        object.__setattr__(self, "dpk_growth", growth)
        object.__setattr__(self, "wall_metrics", wall)
        object.__setattr__(self, "heat_metrics", heat)
        object.__setattr__(self, "wall_metric_units", wall_units)
        object.__setattr__(self, "heat_metric_units", heat_units)
        object.__setattr__(self, "open_loss_fraction", loss)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def fixed_points(self) -> tuple[NewtonFixedPointState, ...]:
        return tuple(point for chain in self.chains for point in chain.fixed_points)

    def to_observables(
        self,
        *,
        weights=None,
        prefix: str | None = "nonlinear",
        metadata: Mapping[str, object] | None = None,
    ) -> BoundaryResponseObservables:
        """Convert this state to named boundary-response rows."""

        return boundary_nonlinear_validation_observables(
            self,
            weights=weights,
            prefix=prefix,
            metadata=metadata,
        )


def _typed_tuple(values, expected_type, name: str):
    result = tuple(values)
    if not all(isinstance(item, expected_type) for item in result):
        raise TypeError(f"{name} must contain {expected_type.__name__} objects")
    return result


def _validated_metric_mapping(values, name: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for raw_label, raw_value in dict(values or {}).items():
        label = _nonempty_label(raw_label, f"{name} label")
        if label in result:
            raise ValueError(f"{name} labels must remain unique after string conversion")
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError(f"{name} values must be finite")
        result[label] = value
    return result


def _validated_metric_units(values, metrics: Mapping[str, float], name: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_label, raw_unit in dict(values or {}).items():
        label = _nonempty_label(raw_label, f"{name} label")
        if label not in metrics:
            raise ValueError(f"{name} contains no matching metric for {label!r}")
        result[label] = _nonempty_label(raw_unit, f"{name} unit")
    return result


def boundary_nonlinear_validation_observables(
    state: BoundaryNonlinearValidationState,
    *,
    weights=None,
    prefix: str | None = "nonlinear",
    metadata: Mapping[str, object] | None = None,
) -> BoundaryResponseObservables:
    """Convert validated nonlinear metrics to stable labelled control rows."""

    if not isinstance(state, BoundaryNonlinearValidationState):
        raise TypeError("state must be BoundaryNonlinearValidationState")
    labels: list[str] = []
    values: list[float] = []
    units: dict[str, str] = {}

    def append(label: str, value: float, unit: str = "1") -> None:
        labels.append(label)
        values.append(float(value))
        units[label] = unit

    for chain in state.chains:
        base = f"chain.{chain.label}"
        resonant_phase = chain.resonant_phase
        append(f"{base}.phase_sin", np.sin(resonant_phase))
        append(f"{base}.phase_cos", np.cos(resonant_phase))
        append(f"{base}.coherence", chain.coherence)
        for kind in ("O", "X"):
            residues = [point.residue for point in chain.fixed_points if point.kind == kind]
            append(f"{base}.residue.{kind.lower()}_median", np.median(residues))

    for splitting in state.stable_unstable_splittings:
        base = f"manifold.{splitting.label}"
        append(f"{base}.stable_unstable_splitting_m", splitting.median_m, "m")
        append(f"{base}.stable_unstable_splitting_spread_m", splitting.spread_m, "m")

    for width in state.separatrix_widths:
        base = f"separatrix.{width.label}"
        append(f"{base}.island_envelope_full_width_m", width.median_m, "m")
        append(f"{base}.island_envelope_full_width_spread_m", width.spread_m, "m")

    shift = state.axis_core_shift
    if shift is not None:
        if shift.axis_shift_m is not None:
            append("axis.shift_m", shift.axis_shift_m, "m")
        if shift.core_shift_rms_m is not None:
            append("core.shift_rms_m", shift.core_shift_rms_m, "m")
            append("core.shift_max_m", shift.core_shift_max_m, "m")

    for growth in state.dpk_growth:
        base = f"dpk.{growth.label}"
        append(f"{base}.svd_growth_factor", growth.svd_growth_factor)
        append(f"{base}.eigen_growth_factor", growth.eigen_growth_factor)
        if growth.open_loss_fraction is not None:
            append(f"{base}.open_loss_fraction", growth.open_loss_fraction)

    for label, value in state.wall_metrics.items():
        append(f"wall.{label}", value, state.wall_metric_units.get(label, "unspecified"))
    for label, value in state.heat_metrics.items():
        append(f"heat.{label}", value, state.heat_metric_units.get(label, "unspecified"))
    if state.open_loss_fraction is not None:
        append("transport.open_loss_fraction", state.open_loss_fraction)

    if len(set(labels)) != len(labels):
        raise ValueError("nonlinear observable labels must be unique")
    prefix_text = None if prefix is None else str(prefix).strip(".")
    full_labels = labels if not prefix_text else [f"{prefix_text}.{label}" for label in labels]
    resolved_weights = _observable_weights(weights, labels, full_labels)
    md: dict[str, object] = {
        "phase_convention": "per-chain; multi-section chains require Nardon exp(i*(m*theta*+n_N*phi))",
        "chain_phase_conventions": {
            chain.label: chain.metadata["phase_convention"] for chain in state.chains
        },
        "phase_observables": "sin(m*phase), cos(m*phase)",
        "stable_unstable_splitting_definition": "W^s/W^u lobe splitting, not island width",
        "island_envelope_width_definition": "ordered inner-to-outer separatrix crossings in metres",
        "units": {
            (label if not prefix_text else f"{prefix_text}.{label}"): unit
            for label, unit in units.items()
        },
        "stable_unstable_splitting_resolution": {
            splitting.label: dict(splitting.resolution_provenance)
            for splitting in state.stable_unstable_splittings
        },
        "island_envelope_resolution": {
            width.label: dict(width.resolution_provenance) for width in state.separatrix_widths
        },
        **dict(state.metadata),
    }
    if metadata:
        md.update(dict(metadata))
    return boundary_response_observables(
        labels,
        values,
        weights=resolved_weights,
        prefix=prefix_text,
        metadata=md,
    )


nonlinear_validation_observables = boundary_nonlinear_validation_observables


def _observable_weights(weights, local_labels, full_labels) -> np.ndarray:
    if weights is None:
        return np.ones(len(local_labels), dtype=float)
    if isinstance(weights, Mapping):
        return np.asarray(
            [
                float(weights[full])
                if full in weights
                else float(weights[local])
                if local in weights
                else 1.0
                for local, full in zip(local_labels, full_labels)
            ],
            dtype=float,
        )
    return np.broadcast_to(np.asarray(weights, dtype=float), (len(local_labels),)).copy()


def content_fingerprint(value: object) -> str:
    """Return a deterministic SHA-256 digest of full public content.

    NumPy arrays are hashed in full, including dtype and shape.  Objects with
    unsupported opaque state should use an injected content-key builder rather
    than an identity- or repr-based cache key.
    """

    digest = sha256()
    _update_content_digest(digest, value, set())
    return digest.hexdigest()


def _update_content_digest(digest, value: object, active: set[int]) -> None:
    if value is None:
        digest.update(b"none;")
        return
    if isinstance(value, (bool, np.bool_)):
        digest.update(b"bool:1;" if bool(value) else b"bool:0;")
        return
    if isinstance(value, (int, np.integer)):
        digest.update(f"int:{int(value)};".encode("ascii"))
        return
    if isinstance(value, (float, np.floating)):
        digest.update(b"float:")
        digest.update(np.asarray(float(value), dtype=np.float64).tobytes())
        return
    if isinstance(value, (complex, np.complexfloating)):
        digest.update(b"complex:")
        digest.update(np.asarray(complex(value), dtype=np.complex128).tobytes())
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(f"str:{len(encoded)}:".encode("ascii"))
        digest.update(encoded)
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        digest.update(f"bytes:{len(raw)}:".encode("ascii"))
        digest.update(raw)
        return
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        digest.update(f"array:{array.dtype.str}:{array.shape}:".encode("ascii"))
        if array.dtype.hasobject:
            for item in array.ravel(order="C"):
                _update_content_digest(digest, item, active)
        else:
            digest.update(np.ascontiguousarray(array).tobytes())
        return

    identity = id(value)
    if identity in active:
        digest.update(b"cycle;")
        return
    active.add(identity)
    try:
        if is_dataclass(value) and not isinstance(value, type):
            digest.update(f"dataclass:{type(value).__module__}.{type(value).__qualname__}:".encode("utf-8"))
            for item in fields(value):
                digest.update(item.name.encode("utf-8"))
                _update_content_digest(digest, getattr(value, item.name), active)
            return
        if isinstance(value, Mapping):
            digest.update(b"mapping:")
            ordered = sorted(value.items(), key=lambda item: content_fingerprint(item[0]))
            for key, item in ordered:
                _update_content_digest(digest, key, active)
                _update_content_digest(digest, item, active)
            return
        if isinstance(value, (tuple, list)):
            digest.update(f"sequence:{len(value)}:".encode("ascii"))
            for item in value:
                _update_content_digest(digest, item, active)
            return
        if isinstance(value, (set, frozenset)):
            digest.update(b"set:")
            for item_digest in sorted(content_fingerprint(item) for item in value):
                digest.update(item_digest.encode("ascii"))
            return
        if callable(value):
            module = getattr(value, "__module__", type(value).__module__)
            qualname = getattr(value, "__qualname__", type(value).__qualname__)
            digest.update(f"callable:{module}.{qualname};".encode("utf-8"))
            return
        public_state = None
        if hasattr(value, "__dict__"):
            public_state = {
                key: item for key, item in vars(value).items() if not str(key).startswith("_")
            }
        if public_state:
            digest.update(f"object:{type(value).__module__}.{type(value).__qualname__}:".encode("utf-8"))
            _update_content_digest(digest, public_state, active)
            return
        raise TypeError(
            f"cannot derive a content key for opaque {type(value).__name__}; "
            "inject content_key_builder"
        )
    finally:
        active.remove(identity)


def boundary_snapshot_content_key(snapshot: object) -> str:
    """Return a content, not identity, cache key for a response snapshot."""

    metadata = _public_value(snapshot, "metadata", {})
    if isinstance(metadata, Mapping):
        for name in ("nonlinear_validation_content_key", "content_key"):
            if name in metadata:
                return content_fingerprint(("explicit_snapshot_content", metadata[name]))
    return content_fingerprint(snapshot)


def boundary_control_key(request: object) -> str:
    """Return a label-stable key for controls and evaluator-visible context."""

    controls = np.asarray(_public_value(request, "controls"), dtype=float).ravel()
    labels = tuple(str(label) for label in _public_value(request, "control_labels"))
    if controls.size != len(labels) or len(set(labels)) != len(labels):
        raise ValueError("request controls and unique control_labels must align")
    if not np.all(np.isfinite(controls)):
        raise ValueError("request controls must be finite")
    ordered = sorted(zip(labels, controls), key=lambda item: item[0])
    metadata = _public_value(request, "metadata", {})
    return content_fingerprint(
        {
            "controls_by_label": tuple((label, float(value)) for label, value in ordered),
            "baseline_equilibrium": _public_value(request, "baseline_equilibrium", None),
            "baseline_field": _public_value(request, "baseline_field", None),
            "vacuum_delta_field": _public_value(request, "vacuum_delta_field", None),
            "metadata": metadata,
        }
    )


@dataclass(frozen=True)
class NonlinearValidationCacheKey:
    """Separate evaluator, snapshot-content, and control-command key parts."""

    evaluator: str
    content: str
    control: str


class NonlinearValidationEvaluator(Protocol):
    """Expensive evaluator contract used by the observable callback."""

    def __call__(self, snapshot: object, request: object) -> BoundaryNonlinearValidationState:
        ...


@dataclass
class BoundaryNonlinearObservableBuilder:
    """Callable ``extra_observable_builders(snapshot, request)`` adapter."""

    evaluator: object
    cache: MutableMapping[NonlinearValidationCacheKey, BoundaryNonlinearValidationState] | None = None
    content_key_builder: Callable[[object], object] = boundary_snapshot_content_key
    control_key_builder: Callable[[object], object] = boundary_control_key
    evaluator_key: str | None = None
    weights: object = None
    prefix: str | None = "nonlinear"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        evaluate_method = getattr(self.evaluator, "evaluate", None)
        if not callable(self.evaluator) and not callable(evaluate_method):
            raise TypeError("evaluator must be callable or expose evaluate(snapshot, request)")
        if self.cache is not None and not (
            hasattr(self.cache, "__getitem__") and hasattr(self.cache, "__setitem__")
        ):
            raise TypeError("cache must be a mutable mapping or None")
        if not callable(self.content_key_builder) or not callable(self.control_key_builder):
            raise TypeError("content/control key builders must be callable")
        if self.evaluator_key is None:
            module = getattr(self.evaluator, "__module__", type(self.evaluator).__module__)
            qualname = getattr(self.evaluator, "__qualname__", type(self.evaluator).__qualname__)
            self.evaluator_key = f"{module}.{qualname}@{id(self.evaluator):x}"
        else:
            self.evaluator_key = _nonempty_label(self.evaluator_key, "evaluator_key")
        self.metadata = dict(self.metadata or {})

    def cache_key(self, snapshot: object, request: object) -> NonlinearValidationCacheKey:
        """Build the evaluator/content/control key for one callback request."""

        return NonlinearValidationCacheKey(
            evaluator=str(self.evaluator_key),
            content=content_fingerprint(self.content_key_builder(snapshot)),
            control=content_fingerprint(self.control_key_builder(request)),
        )

    def __call__(self, snapshot: object, request: object) -> BoundaryResponseObservables:
        state: BoundaryNonlinearValidationState | None = None
        key = None
        if self.cache is not None:
            key = self.cache_key(snapshot, request)
            try:
                state = self.cache[key]
            except KeyError:
                state = None
            if state is not None and not isinstance(state, BoundaryNonlinearValidationState):
                raise TypeError("cached evaluator result must be BoundaryNonlinearValidationState")
        if state is None:
            if callable(self.evaluator):
                state = self.evaluator(snapshot, request)
            else:
                state = self.evaluator.evaluate(snapshot, request)
            if not isinstance(state, BoundaryNonlinearValidationState):
                raise TypeError("evaluator must return BoundaryNonlinearValidationState")
            if self.cache is not None and key is not None:
                self.cache[key] = state
        return boundary_nonlinear_validation_observables(
            state,
            weights=self.weights,
            prefix=self.prefix,
            metadata=self.metadata,
        )


def boundary_nonlinear_observable_builder(
    evaluator: object,
    *,
    cache: MutableMapping[NonlinearValidationCacheKey, BoundaryNonlinearValidationState] | None = None,
    content_key_builder: Callable[[object], object] = boundary_snapshot_content_key,
    control_key_builder: Callable[[object], object] = boundary_control_key,
    evaluator_key: str | None = None,
    weights=None,
    prefix: str | None = "nonlinear",
    metadata: Mapping[str, object] | None = None,
) -> BoundaryNonlinearObservableBuilder:
    """Build a cached callback compatible with ``extra_observable_builders``.

    The default evaluator namespace is instance-scoped to prevent collisions.
    Supply ``evaluator_key`` only when equivalent evaluator instances should
    deliberately share cached nonlinear states.
    """

    return BoundaryNonlinearObservableBuilder(
        evaluator=evaluator,
        cache=cache,
        content_key_builder=content_key_builder,
        control_key_builder=control_key_builder,
        evaluator_key=evaluator_key,
        weights=weights,
        prefix=prefix,
        metadata={} if metadata is None else metadata,
    )


make_boundary_nonlinear_observable_builder = boundary_nonlinear_observable_builder


__all__ = [
    "BoundaryNonlinearObservableBuilder",
    "BoundaryNonlinearValidationState",
    "DPKGrowthValidation",
    "FixedPointChainValidation",
    "HealedSurfaceSectionChart",
    "IslandEnvelopeBranchWidth",
    "IslandEnvelopeWidthMetric",
    "IslandEnvelopeWidthSample",
    "IslandChainValidation",
    "LocalManifoldMeasurementLine",
    "LocalMeasurementLine",
    "MagneticAxisCoreShift",
    "ManifoldBranchSamples",
    "ManifoldTraceProvenance",
    "NewtonFixedPointState",
    "NonlinearValidationCacheKey",
    "NonlinearValidationEvaluator",
    "PolylineLineIntersection",
    "PeriodicPointPhaseSpaceResponse",
    "SeparatrixEnvelopeBranchPair",
    "StableUnstableBranchPair",
    "StableUnstableBranchSplitting",
    "StableUnstableSplittingMetric",
    "StableUnstableSplittingSample",
    "boundary_control_key",
    "boundary_nonlinear_observable_builder",
    "boundary_nonlinear_validation_observables",
    "boundary_snapshot_content_key",
    "content_fingerprint",
    "fixed_point_chain_coherence",
    "fixed_point_chain_helical_coherence",
    "fixed_point_chain_helical_phase",
    "fixed_point_chain_helical_phasor",
    "fixed_point_chain_phase",
    "greene_residue",
    "make_boundary_nonlinear_observable_builder",
    "manifold_branches_from_trace",
    "manifold_line_intersections",
    "nardon_fixed_point_phase_closure_error",
    "nonlinear_validation_observables",
    "periodic_phase_difference",
    "polyline_line_intersections",
    "separatrix_width_from_manifolds",
    "solve_periodic_point_phase_response",
    "stable_unstable_splitting_from_manifolds",
]
