"""C++ physical-tau tracing for axisymmetric closed field lines.

The Python layer only normalizes a :class:`~pyna.fields.VectorFieldCylind`
and freezes the arrays returned by cyna.  Integration and directed section
event location are performed by the compiled implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyna._cyna import (
    trace_axisym_tau_closed_orbit as _cyna_trace_axisym_tau_closed_orbit,
    trace_axisym_tau_closed_orbit_jvp as _cyna_trace_axisym_tau_closed_orbit_jvp,
)
from pyna.fields import as_vector_field_cylindrical
from pyna.toroidal.flt.numba_poincare import vector_field_cylind_from_field


AXISYM_TAU_RETURN_BRANCH = "first_same_direction_return_after_opposite_crossing"


@dataclass(frozen=True, eq=False)
class AxisymTauBExtremumEvent:
    certified: bool
    tau_m_per_T: float
    position_R_Z_phi: NDArray[np.float64]
    B_T: float
    d2_B2_dtau2_T4_per_m2: float
    interval_index: int
    interval_fraction: float

    def __post_init__(self) -> None:
        position = np.ascontiguousarray(
            self.position_R_Z_phi, dtype=np.float64
        ).reshape(-1)
        if position.shape != (3,):
            raise ValueError("extremum position must have shape (3,)")
        if self.certified and (
            not np.all(np.isfinite(position))
            or not all(
                math.isfinite(value)
                for value in (
                    self.tau_m_per_T,
                    self.B_T,
                    self.d2_B2_dtau2_T4_per_m2,
                    self.interval_fraction,
                )
            )
            or self.tau_m_per_T < 0.0
            or self.B_T <= 0.0
            or self.interval_index < 0
            or not 0.0 < self.interval_fraction < 1.0
            or self.d2_B2_dtau2_T4_per_m2 == 0.0
        ):
            raise ValueError("certified continuous B extremum is invalid")
        position.setflags(write=False)
        object.__setattr__(self, "position_R_Z_phi", position)


@dataclass(frozen=True, eq=False)
class AxisymTauBExtremumEventJVP:
    event: AxisymTauBExtremumEvent
    certified: bool
    tau_jvp_m_per_T: float
    position_R_Z_phi_jvp: NDArray[np.float64]
    B_jvp_T: float

    def __post_init__(self) -> None:
        if not isinstance(self.event, AxisymTauBExtremumEvent):
            raise TypeError("event must be an AxisymTauBExtremumEvent")
        position_jvp = np.ascontiguousarray(
            self.position_R_Z_phi_jvp, dtype=np.float64
        ).reshape(-1)
        if position_jvp.shape != (3,):
            raise ValueError("extremum position JVP must have shape (3,)")
        if self.certified and (
            not self.event.certified
            or not np.all(np.isfinite(position_jvp))
            or not math.isfinite(self.tau_jvp_m_per_T)
            or not math.isfinite(self.B_jvp_T)
        ):
            raise ValueError("certified continuous B extremum JVP is invalid")
        position_jvp.setflags(write=False)
        object.__setattr__(self, "position_R_Z_phi_jvp", position_jvp)


@dataclass(frozen=True, eq=False)
class AxisymTauOrbit:
    """One cyna axisymmetric directed-return trace and its quadrature."""

    seed_R_m: float
    section_Z_m: float
    status: str
    status_code: int
    tau: NDArray[np.float64]
    path_R_Z_phi: NDArray[np.float64]
    weights_B2_dtau: NDArray[np.float64]
    period_tau_m_per_T: float
    return_defect_m: float
    gauge_measure_B2_dtau_T_m: float
    phi_advance_rad: float
    opposite_section_crossed: bool
    accepted_steps: int
    production_cyna_executed: bool
    B_minimum_event: AxisymTauBExtremumEvent
    B_maximum_event: AxisymTauBExtremumEvent

    def __post_init__(self) -> None:
        tau = np.ascontiguousarray(self.tau, dtype=np.float64).reshape(-1)
        path = np.ascontiguousarray(self.path_R_Z_phi, dtype=np.float64)
        weights = np.ascontiguousarray(
            self.weights_B2_dtau, dtype=np.float64
        ).reshape(-1)
        if path.shape != (tau.size, 3):
            raise ValueError("path_R_Z_phi must have shape (tau.size, 3)")
        if weights.size not in (0, tau.size):
            raise ValueError("weights_B2_dtau must be empty or match tau")
        if not self.production_cyna_executed:
            raise ValueError("axisymmetric tau result did not originate in cyna")
        if not isinstance(self.B_minimum_event, AxisymTauBExtremumEvent) or not isinstance(
            self.B_maximum_event, AxisymTauBExtremumEvent
        ):
            raise TypeError("B extrema must be typed cyna events")
        if self.closed:
            if (
                tau.size < 3
                or weights.size != tau.size
                or not np.all(np.isfinite(tau))
                or not np.all(np.isfinite(path))
                or not np.all(np.isfinite(weights))
                or np.any(np.diff(tau) <= 0.0)
                or np.any(weights < 0.0)
            ):
                raise ValueError("closed cyna tau orbit arrays are invalid")
            if not np.isclose(
                path[0, 1], self.section_Z_m, rtol=0.0, atol=1.0e-13
            ) or not np.isclose(
                path[-1, 1], self.section_Z_m, rtol=0.0, atol=1.0e-13
            ):
                raise ValueError("closed orbit endpoints must lie on section_Z_m")
            measure = float(np.sum(weights))
            if not np.isclose(
                measure,
                self.gauge_measure_B2_dtau_T_m,
                rtol=2.0e-13,
                atol=1.0e-14,
            ):
                raise ValueError("B^2 dtau weights do not reproduce their measure")
            if (
                not math.isfinite(self.period_tau_m_per_T)
                or self.period_tau_m_per_T <= 0.0
                or not math.isfinite(self.return_defect_m)
                or self.return_defect_m < 0.0
                or not math.isfinite(self.gauge_measure_B2_dtau_T_m)
                or self.gauge_measure_B2_dtau_T_m <= 0.0
                or not math.isfinite(self.phi_advance_rad)
                or not self.opposite_section_crossed
            ):
                raise ValueError("closed cyna tau orbit diagnostics are invalid")
            if not np.isclose(
                tau[-1], self.period_tau_m_per_T, rtol=0.0, atol=1.0e-14
            ):
                raise ValueError("tau endpoint and period_tau_m_per_T disagree")
        elif tau.size or path.shape[0] or weights.size:
            raise ValueError("failed cyna tau traces must not expose partial quadrature")

        tau.setflags(write=False)
        path.setflags(write=False)
        weights.setflags(write=False)
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "path_R_Z_phi", path)
        object.__setattr__(self, "weights_B2_dtau", weights)

    @property
    def closed(self) -> bool:
        """Whether cyna certified the full directed first return."""

        return self.status == "closed" and self.status_code == 0

    @property
    def normalized_gauge_weights(self) -> NDArray[np.float64]:
        """Return the normalized ``B^2 d tau`` nodal weights."""

        if not self.closed:
            raise RuntimeError(f"axisymmetric tau orbit failed: {self.status}")
        normalized = self.weights_B2_dtau / self.gauge_measure_B2_dtau_T_m
        normalized.setflags(write=False)
        return normalized


@dataclass(frozen=True, eq=False)
class AxisymTauOrbitJVP:
    orbit: AxisymTauOrbit
    seed_R_jvp_m: float
    section_Z_jvp_m: float
    phi_start_jvp_rad: float
    return_direction: int
    return_branch: str
    tau_jvp_m_per_T: NDArray[np.float64]
    path_R_Z_phi_jvp: NDArray[np.float64]
    weights_B2_dtau_jvp: NDArray[np.float64]
    period_tau_jvp_m_per_T: float
    gauge_measure_B2_dtau_jvp: float
    phi_advance_jvp_rad: float
    return_displacement_R_Z_jvp_m: NDArray[np.float64]
    event_fraction: float
    event_fraction_jvp: float
    phase_parameterization: str
    production_cyna_tangent_executed: bool
    B_minimum_event_jvp: AxisymTauBExtremumEventJVP
    B_maximum_event_jvp: AxisymTauBExtremumEventJVP

    def __post_init__(self) -> None:
        if not isinstance(self.orbit, AxisymTauOrbit):
            raise TypeError("orbit must be an AxisymTauOrbit")
        tau_jvp = np.ascontiguousarray(
            self.tau_jvp_m_per_T, dtype=np.float64
        ).reshape(-1)
        path_jvp = np.ascontiguousarray(
            self.path_R_Z_phi_jvp, dtype=np.float64
        )
        weights_jvp = np.ascontiguousarray(
            self.weights_B2_dtau_jvp, dtype=np.float64
        ).reshape(-1)
        return_jvp = np.ascontiguousarray(
            self.return_displacement_R_Z_jvp_m, dtype=np.float64
        ).reshape(-1)
        if path_jvp.shape != (tau_jvp.size, 3):
            raise ValueError("path_R_Z_phi_jvp must have shape (tau_jvp.size, 3)")
        if weights_jvp.size not in (0, tau_jvp.size):
            raise ValueError("weights_B2_dtau_jvp must be empty or match tau_jvp")
        if return_jvp.shape != (2,):
            raise ValueError("return_displacement_R_Z_jvp_m must have shape (2,)")
        if not self.production_cyna_tangent_executed:
            raise ValueError("axisymmetric tau JVP did not originate in cyna")
        if self.B_minimum_event_jvp.event is not self.orbit.B_minimum_event or (
            self.B_maximum_event_jvp.event is not self.orbit.B_maximum_event
        ):
            raise ValueError("B-extremum tangent event identity changed")
        if self.return_direction not in (-1, 0, 1):
            raise ValueError("return_direction must be -1, 0, or +1")
        if self.return_branch != AXISYM_TAU_RETURN_BRANCH:
            raise ValueError("cyna returned an unknown directed-return branch")
        if self.phase_parameterization != "baseline_normalized_tau":
            raise ValueError("cyna returned an unknown tau JVP parameterization")
        if self.closed:
            if self.return_direction not in (-1, 1):
                raise ValueError("closed tau JVP has no directed-section identity")
            count = self.orbit.tau.size
            if (
                tau_jvp.size != count
                or weights_jvp.size != count
                or not np.all(np.isfinite(tau_jvp))
                or not np.all(np.isfinite(path_jvp))
                or not np.all(np.isfinite(weights_jvp))
                or not np.all(np.isfinite(return_jvp))
                or not all(
                    math.isfinite(value)
                    for value in (
                        self.period_tau_jvp_m_per_T,
                        self.gauge_measure_B2_dtau_jvp,
                        self.phi_advance_jvp_rad,
                        self.event_fraction,
                        self.event_fraction_jvp,
                    )
                )
            ):
                raise ValueError("closed cyna tau JVP arrays are invalid")
            phase = self.orbit.tau / self.orbit.period_tau_m_per_T
            expected_tau_jvp = phase * self.period_tau_jvp_m_per_T
            if not np.allclose(
                tau_jvp, expected_tau_jvp, rtol=2.0e-14, atol=2.0e-15
            ):
                raise ValueError("tau JVP is not parameterized on baseline phase")
            scale = max(float(np.sum(np.abs(weights_jvp))), 1.0)
            if not np.isclose(
                np.sum(weights_jvp),
                self.gauge_measure_B2_dtau_jvp,
                rtol=0.0,
                atol=256.0 * np.finfo(np.float64).eps * scale,
            ):
                raise ValueError("weight JVP does not reproduce measure JVP")
            if not np.isclose(
                path_jvp[0, 1], self.section_Z_jvp_m, rtol=0.0, atol=1.0e-13
            ) or not np.isclose(
                path_jvp[-1, 1], self.section_Z_jvp_m, rtol=0.0, atol=1.0e-13
            ):
                raise ValueError("tau JVP endpoints do not follow the moving section")
            expected_return = path_jvp[-1, :2] - np.asarray(
                (self.seed_R_jvp_m, self.section_Z_jvp_m)
            )
            if not np.allclose(return_jvp, expected_return, rtol=0.0, atol=1.0e-13):
                raise ValueError("return displacement JVP is inconsistent with path")
        elif tau_jvp.size or path_jvp.shape[0] or weights_jvp.size:
            raise ValueError("failed cyna tau JVPs must not expose partial arrays")

        for value in (tau_jvp, path_jvp, weights_jvp, return_jvp):
            value.setflags(write=False)
        object.__setattr__(self, "tau_jvp_m_per_T", tau_jvp)
        object.__setattr__(self, "path_R_Z_phi_jvp", path_jvp)
        object.__setattr__(self, "weights_B2_dtau_jvp", weights_jvp)
        object.__setattr__(self, "return_displacement_R_Z_jvp_m", return_jvp)

    @property
    def closed(self) -> bool:
        return self.orbit.closed

    @property
    def normalized_gauge_weights_jvp(self) -> NDArray[np.float64]:
        if not self.closed:
            raise RuntimeError(f"axisymmetric tau JVP failed: {self.orbit.status}")
        measure = self.orbit.gauge_measure_B2_dtau_T_m
        result = (
            self.weights_B2_dtau_jvp / measure
            - self.orbit.weights_B2_dtau
            * self.gauge_measure_B2_dtau_jvp
            / (measure * measure)
        )
        result.setflags(write=False)
        return result


def _orbit_from_cyna_raw(
    raw: Any, *, seed_R_m: float, section_Z_m: float
) -> AxisymTauOrbit:
    def extremum(name: str) -> AxisymTauBExtremumEvent:
        value = raw[name]
        return AxisymTauBExtremumEvent(
            certified=bool(value["certified"]),
            tau_m_per_T=float(value["tau"]),
            position_R_Z_phi=np.asarray(
                value["position_R_Z_phi"], dtype=np.float64
            ),
            B_T=float(value["B"]),
            d2_B2_dtau2_T4_per_m2=float(value["d2_B2_dtau2"]),
            interval_index=int(value["interval_index"]),
            interval_fraction=float(value["interval_fraction"]),
        )

    return AxisymTauOrbit(
        seed_R_m=seed_R_m,
        section_Z_m=section_Z_m,
        status=str(raw["status"]),
        status_code=int(raw["status_code"]),
        tau=np.asarray(raw["tau"], dtype=np.float64),
        path_R_Z_phi=np.asarray(raw["path_R_Z_phi"], dtype=np.float64),
        weights_B2_dtau=np.asarray(raw["weights_B2_dtau"], dtype=np.float64),
        period_tau_m_per_T=float(raw["period_tau"]),
        return_defect_m=float(raw["return_defect"]),
        gauge_measure_B2_dtau_T_m=float(raw["gauge_measure_B2_dtau"]),
        phi_advance_rad=float(raw["phi_advance"]),
        opposite_section_crossed=bool(raw["opposite_section_crossed"]),
        accepted_steps=int(raw["accepted_steps"]),
        production_cyna_executed=bool(raw["production_cyna_executed"]),
        B_minimum_event=extremum("B_minimum_event"),
        B_maximum_event=extremum("B_maximum_event"),
    )


def trace_axisym_tau_closed_orbit(
    field: Any,
    *,
    seed_R_m: float,
    section_Z_m: float,
    step_tau_m_per_T: float,
    maximum_tau_m_per_T: float,
    closure_tolerance_m: float,
    minimum_seed_poloidal_field_T: float = 1.0e-9,
    phi_start_rad: float = 0.0,
) -> AxisymTauOrbit:
    r"""Trace one full axisymmetric orbit using physical ``tau`` in cyna.

    The compiled equations use cylindrical component order ``(R, Z, Phi)``:

    ``dR/dtau = BR``, ``dZ/dtau = BZ``, ``dphi/dtau = BPhi/R``.

    A successful result has crossed the horizontal section once in the
    opposite direction before its same-direction first return.  Expected
    physical failures (outside domain, weak poloidal field, or no certified
    return) are reported through ``result.status`` with ``result.closed``
    false; malformed inputs raise ``ValueError``.
    """

    if _cyna_trace_axisym_tau_closed_orbit is None:
        raise ImportError(
            "pyna._cyna.trace_axisym_tau_closed_orbit is unavailable; rebuild cyna"
        )
    seed_R = float(seed_R_m)
    section_Z = float(section_Z_m)
    step_tau = float(step_tau_m_per_T)
    maximum_tau = float(maximum_tau_m_per_T)
    closure_tolerance = float(closure_tolerance_m)
    minimum_seed_Bp = float(minimum_seed_poloidal_field_T)
    phi_start = float(phi_start_rad)
    if not all(math.isfinite(value) for value in (seed_R, section_Z, phi_start)):
        raise ValueError("seed_R_m, section_Z_m, and phi_start_rad must be finite")
    if (
        not math.isfinite(step_tau)
        or step_tau <= 0.0
        or not math.isfinite(maximum_tau)
        or maximum_tau <= step_tau
    ):
        raise ValueError(
            "step_tau_m_per_T must be positive and maximum_tau_m_per_T "
            "must exceed it"
        )
    if not math.isfinite(closure_tolerance) or closure_tolerance <= 0.0:
        raise ValueError("closure_tolerance_m must be finite and positive")
    if not math.isfinite(minimum_seed_Bp) or minimum_seed_Bp < 0.0:
        raise ValueError(
            "minimum_seed_poloidal_field_T must be finite and nonnegative"
        )
    normalized = as_vector_field_cylindrical(field)
    if not normalized.is_axisymmetric:
        raise ValueError(
            "trace_axisym_tau_closed_orbit requires a field explicitly marked "
            "axisymmetric"
        )
    cyna_field = vector_field_cylind_from_field(normalized, extend_phi=True)
    raw = _cyna_trace_axisym_tau_closed_orbit(
        cyna_field,
        seed_R=seed_R,
        section_Z=section_Z,
        step_tau=step_tau,
        maximum_tau=maximum_tau,
        closure_tolerance=closure_tolerance,
        minimum_seed_poloidal_field=minimum_seed_Bp,
        phi_start=phi_start,
    )
    if tuple(raw.get("component_order", ())) != ("R", "Z", "Phi"):
        raise RuntimeError("cyna tau tracer returned an invalid component order")
    return _orbit_from_cyna_raw(raw, seed_R_m=seed_R, section_Z_m=section_Z)


def trace_axisym_tau_closed_orbit_jvp(
    field: Any,
    field_direction: Any,
    *,
    seed_R_m: float,
    section_Z_m: float,
    step_tau_m_per_T: float,
    maximum_tau_m_per_T: float,
    closure_tolerance_m: float,
    return_direction: int,
    return_branch: str,
    minimum_seed_poloidal_field_T: float = 1.0e-9,
    phi_start_rad: float = 0.0,
    seed_R_jvp_m: float = 0.0,
    section_Z_jvp_m: float = 0.0,
    phi_start_jvp_rad: float = 0.0,
) -> AxisymTauOrbitJVP:
    """Trace a directed axisymmetric orbit and its compiled tangent JVP."""

    if _cyna_trace_axisym_tau_closed_orbit_jvp is None:
        raise ImportError(
            "pyna._cyna.trace_axisym_tau_closed_orbit_jvp is unavailable; rebuild cyna"
        )
    branch = str(return_branch)
    if branch != AXISYM_TAU_RETURN_BRANCH:
        raise ValueError(
            f"return_branch must be {AXISYM_TAU_RETURN_BRANCH!r}"
        )
    direction_sign = int(return_direction)
    if direction_sign not in (-1, 1) or direction_sign != return_direction:
        raise ValueError("return_direction must be exactly -1 or +1")

    seed_R = float(seed_R_m)
    section_Z = float(section_Z_m)
    step_tau = float(step_tau_m_per_T)
    maximum_tau = float(maximum_tau_m_per_T)
    closure_tolerance = float(closure_tolerance_m)
    minimum_seed_Bp = float(minimum_seed_poloidal_field_T)
    phi_start = float(phi_start_rad)
    seed_R_jvp = float(seed_R_jvp_m)
    section_Z_jvp = float(section_Z_jvp_m)
    phi_start_jvp = float(phi_start_jvp_rad)
    if not all(
        math.isfinite(value)
        for value in (
            seed_R,
            section_Z,
            phi_start,
            seed_R_jvp,
            section_Z_jvp,
            phi_start_jvp,
        )
    ):
        raise ValueError("tau JVP seed, section, and directions must be finite")
    if (
        not math.isfinite(step_tau)
        or step_tau <= 0.0
        or not math.isfinite(maximum_tau)
        or maximum_tau <= step_tau
    ):
        raise ValueError(
            "step_tau_m_per_T must be positive and maximum_tau_m_per_T must exceed it"
        )
    if not math.isfinite(closure_tolerance) or closure_tolerance <= 0.0:
        raise ValueError("closure_tolerance_m must be finite and positive")
    if not math.isfinite(minimum_seed_Bp) or minimum_seed_Bp < 0.0:
        raise ValueError(
            "minimum_seed_poloidal_field_T must be finite and nonnegative"
        )

    normalized = as_vector_field_cylindrical(field)
    normalized_direction = as_vector_field_cylindrical(field_direction)
    if not normalized.is_axisymmetric or not normalized_direction.is_axisymmetric:
        raise ValueError("tau orbit JVP requires two explicitly axisymmetric fields")
    if tuple(normalized.component_order) != ("R", "Z", "Phi") or tuple(
        normalized_direction.component_order
    ) != ("R", "Z", "Phi"):
        raise ValueError("pyna field component order must be (R,Z,Phi)")
    if int(normalized.nfp) != int(normalized_direction.nfp) or any(
        not np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in (
            (normalized.R_arr, normalized_direction.R_arr),
            (normalized.Z_arr, normalized_direction.Z_arr),
            (normalized.Phi, normalized_direction.Phi),
        )
    ):
        raise ValueError("field and field_direction must use the same grid and nfp")

    cyna_field = vector_field_cylind_from_field(normalized, extend_phi=True)
    cyna_direction = vector_field_cylind_from_field(
        normalized_direction, extend_phi=True
    )
    raw = _cyna_trace_axisym_tau_closed_orbit_jvp(
        cyna_field,
        cyna_direction,
        seed_R=seed_R,
        section_Z=section_Z,
        step_tau=step_tau,
        maximum_tau=maximum_tau,
        closure_tolerance=closure_tolerance,
        return_direction=direction_sign,
        minimum_seed_poloidal_field=minimum_seed_Bp,
        phi_start=phi_start,
        seed_R_jvp=seed_R_jvp,
        section_Z_jvp=section_Z_jvp,
        phi_start_jvp=phi_start_jvp,
    )
    if tuple(raw.get("component_order", ())) != ("R", "Z", "Phi"):
        raise RuntimeError("cyna tau JVP returned an invalid component order")
    if str(raw.get("return_branch", "")) != branch:
        raise RuntimeError("cyna tau JVP changed the directed-return branch")
    if bool(raw.get("closed", False)) and int(raw["return_direction"]) != direction_sign:
        raise RuntimeError("cyna tau JVP changed the directed-section direction")

    orbit = _orbit_from_cyna_raw(raw, seed_R_m=seed_R, section_Z_m=section_Z)
    def extremum_jvp(
        name: str, event: AxisymTauBExtremumEvent
    ) -> AxisymTauBExtremumEventJVP:
        value = raw[name]
        return AxisymTauBExtremumEventJVP(
            event=event,
            certified=bool(value["tangent_certified"]),
            tau_jvp_m_per_T=float(value["tau_jvp"]),
            position_R_Z_phi_jvp=np.asarray(
                value["position_R_Z_phi_jvp"], dtype=np.float64
            ),
            B_jvp_T=float(value["B_jvp"]),
        )

    return AxisymTauOrbitJVP(
        orbit=orbit,
        seed_R_jvp_m=seed_R_jvp,
        section_Z_jvp_m=section_Z_jvp,
        phi_start_jvp_rad=phi_start_jvp,
        return_direction=int(raw["return_direction"]),
        return_branch=str(raw["return_branch"]),
        tau_jvp_m_per_T=np.asarray(raw["tau_jvp"], dtype=np.float64),
        path_R_Z_phi_jvp=np.asarray(raw["path_R_Z_phi_jvp"], dtype=np.float64),
        weights_B2_dtau_jvp=np.asarray(
            raw["weights_B2_dtau_jvp"], dtype=np.float64
        ),
        period_tau_jvp_m_per_T=float(raw["period_tau_jvp"]),
        gauge_measure_B2_dtau_jvp=float(raw["gauge_measure_B2_dtau_jvp"]),
        phi_advance_jvp_rad=float(raw["phi_advance_jvp"]),
        return_displacement_R_Z_jvp_m=np.asarray(
            raw["return_displacement_R_Z_jvp"], dtype=np.float64
        ),
        event_fraction=float(raw["event_fraction"]),
        event_fraction_jvp=float(raw["event_fraction_jvp"]),
        phase_parameterization=str(raw["phase_parameterization"]),
        production_cyna_tangent_executed=bool(
            raw["production_cyna_tangent_executed"]
        ),
        B_minimum_event_jvp=extremum_jvp(
            "B_minimum_event", orbit.B_minimum_event
        ),
        B_maximum_event_jvp=extremum_jvp(
            "B_maximum_event", orbit.B_maximum_event
        ),
    )


__all__ = [
    "AXISYM_TAU_RETURN_BRANCH",
    "AxisymTauBExtremumEvent",
    "AxisymTauBExtremumEventJVP",
    "AxisymTauOrbit",
    "AxisymTauOrbitJVP",
    "trace_axisym_tau_closed_orbit",
    "trace_axisym_tau_closed_orbit_jvp",
]
