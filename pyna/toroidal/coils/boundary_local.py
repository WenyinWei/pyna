"""Local boundary-oriented circular loop perturbation coils."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from pyna.toroidal.coils.base import CoilFieldSuperposition
from pyna.toroidal.coils.coil import CoilFieldAnalyticCircular


TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class BoundaryLoopCoilSpec:
    """Geometry and current for a circular loop tangent to a boundary surface."""

    center_xyz: tuple[float, float, float]
    normal_xyz: tuple[float, float, float]
    radius: float
    current: float
    anchor_R: float | None = None
    anchor_Z: float | None = None
    anchor_phi: float | None = None
    anchor_theta: float | None = None
    clearance: float = 0.0
    metadata: dict = field(default_factory=dict, compare=False, repr=False)

    def to_field(self) -> CoilFieldAnalyticCircular:
        """Return the analytic vacuum field for this loop."""

        return CoilFieldAnalyticCircular(
            self.radius,
            center_xyz=self.center_xyz,
            normal_xyz=self.normal_xyz,
            current=self.current,
        )


@dataclass(frozen=True)
class BoundaryDipoleCoilSpec:
    """Small circular-loop realization of a boundary magnetic dipole.

    ``magnetic_moment`` is the signed loop moment in A m^2 along
    ``normal_xyz``.  The field is evaluated with the exact circular-loop
    Biot-Savart solution rather than the point-dipole approximation, which
    remains finite and geometry-aware near the actuator aperture.
    """

    center_xyz: tuple[float, float, float]
    normal_xyz: tuple[float, float, float]
    radius: float
    magnetic_moment: float
    anchor_R: float | None = None
    anchor_Z: float | None = None
    anchor_phi: float | None = None
    anchor_theta: float | None = None
    clearance: float = 0.0
    metadata: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        center = np.asarray(self.center_xyz, dtype=float).reshape(3)
        normal = _unit(self.normal_xyz, name="normal_xyz")
        radius = float(self.radius)
        moment = float(self.magnetic_moment)
        if not np.all(np.isfinite(center)):
            raise ValueError("center_xyz must be finite")
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be positive and finite")
        if not np.isfinite(moment):
            raise ValueError("magnetic_moment must be finite")
        object.__setattr__(self, "center_xyz", tuple(float(v) for v in center))
        object.__setattr__(self, "normal_xyz", tuple(float(v) for v in normal))
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "magnetic_moment", moment)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def current(self) -> float:
        """Equivalent single-turn loop current in amperes."""

        return float(self.magnetic_moment / (np.pi * self.radius**2))

    def scaled(self, factor: float) -> "BoundaryDipoleCoilSpec":
        """Return a dipole with its signed magnetic moment scaled."""

        return BoundaryDipoleCoilSpec(
            center_xyz=self.center_xyz,
            normal_xyz=self.normal_xyz,
            radius=self.radius,
            magnetic_moment=self.magnetic_moment * float(factor),
            anchor_R=self.anchor_R,
            anchor_Z=self.anchor_Z,
            anchor_phi=self.anchor_phi,
            anchor_theta=self.anchor_theta,
            clearance=self.clearance,
            metadata=self.metadata,
        )

    def to_loop_spec(self) -> BoundaryLoopCoilSpec:
        """Return the equivalent exact circular-loop specification."""

        metadata = dict(self.metadata)
        metadata.update({
            "actuator_model": "finite_circular_dipole",
            "magnetic_moment_A_m2": float(self.magnetic_moment),
        })
        return BoundaryLoopCoilSpec(
            center_xyz=self.center_xyz,
            normal_xyz=self.normal_xyz,
            radius=self.radius,
            current=self.current,
            anchor_R=self.anchor_R,
            anchor_Z=self.anchor_Z,
            anchor_phi=self.anchor_phi,
            anchor_theta=self.anchor_theta,
            clearance=self.clearance,
            metadata=metadata,
        )

    def to_field(self) -> CoilFieldAnalyticCircular:
        """Return the exact finite-loop vacuum field."""

        return self.to_loop_spec().to_field()


@dataclass(frozen=True)
class BoundaryDipoleActuatorSpec:
    """One independently commanded column made from one or more dipoles."""

    label: str
    dipoles: tuple[BoundaryDipoleCoilSpec, ...]
    lower_bound: float = -1.0
    upper_bound: float = 1.0
    control_scale: float = 1.0
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        label = str(self.label)
        dipoles = tuple(self.dipoles)
        lower = float(self.lower_bound)
        upper = float(self.upper_bound)
        scale = float(self.control_scale)
        if not label:
            raise ValueError("actuator label must not be empty")
        if not dipoles:
            raise ValueError("an actuator must contain at least one dipole")
        if not all(isinstance(spec, BoundaryDipoleCoilSpec) for spec in dipoles):
            raise TypeError("dipoles must contain BoundaryDipoleCoilSpec objects")
        if not np.isfinite(lower) or not np.isfinite(upper) or upper < lower:
            raise ValueError("actuator bounds must be finite with upper >= lower")
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("control_scale must be positive and finite")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "dipoles", dipoles)
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "control_scale", scale)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def field(self, command: float = 1.0) -> CoilFieldSuperposition:
        """Return this actuator's exact finite-loop field at ``command``."""

        amplitude = float(command)
        return CoilFieldSuperposition([spec.scaled(amplitude).to_field() for spec in self.dipoles])


@dataclass(frozen=True)
class BoundaryDipoleActuatorArray:
    """Label-safe collection of independently commanded dipole columns."""

    actuators: tuple[BoundaryDipoleActuatorSpec, ...]
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        actuators = tuple(self.actuators)
        if not actuators:
            raise ValueError("at least one dipole actuator is required")
        if not all(isinstance(spec, BoundaryDipoleActuatorSpec) for spec in actuators):
            raise TypeError("actuators must contain BoundaryDipoleActuatorSpec objects")
        labels = tuple(spec.label for spec in actuators)
        if len(set(labels)) != len(labels):
            raise ValueError("dipole actuator labels must be unique")
        object.__setattr__(self, "actuators", actuators)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def control_labels(self) -> tuple[str, ...]:
        """Return stable response-matrix column labels."""

        return tuple(spec.label for spec in self.actuators)

    @property
    def control_bounds(self) -> dict[str, tuple[float, float]]:
        """Return absolute command bounds keyed by actuator label."""

        return {spec.label: (spec.lower_bound, spec.upper_bound) for spec in self.actuators}

    @property
    def control_scale(self) -> np.ndarray:
        """Return actuator scaling values in response-matrix column order."""

        return np.asarray([spec.control_scale for spec in self.actuators], dtype=float)

    def actuator(self, label: str) -> BoundaryDipoleActuatorSpec:
        """Return one actuator by label."""

        key = str(label)
        for spec in self.actuators:
            if spec.label == key:
                return spec
        raise KeyError(key)

    def unit_fields(self) -> tuple[CoilFieldSuperposition, ...]:
        """Return one unit-command field for each control column."""

        return tuple(spec.field(1.0) for spec in self.actuators)

    def field(self, controls: Sequence[float]) -> CoilFieldSuperposition:
        """Return the linear superposition for a complete command vector."""

        commands = np.asarray(controls, dtype=float).ravel()
        if commands.size != len(self.actuators):
            raise ValueError("controls length must match dipole actuator count")
        fields = []
        for command, actuator in zip(commands, self.actuators):
            if float(command) != 0.0:
                fields.extend(spec.scaled(float(command)).to_field() for spec in actuator.dipoles)
        return CoilFieldSuperposition(fields)


def _unit(value: Sequence[float], *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(3)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} must have nonzero finite norm")
    return arr / norm


def _cylindrical_basis(phi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = float(np.cos(phi))
    s = float(np.sin(phi))
    e_R = np.array([c, s, 0.0], dtype=float)
    e_phi = np.array([-s, c, 0.0], dtype=float)
    e_Z = np.array([0.0, 0.0, 1.0], dtype=float)
    return e_R, e_phi, e_Z


def cylindrical_point_xyz(R: float, Z: float, phi: float) -> np.ndarray:
    """Return Cartesian coordinates for one cylindrical point."""

    return np.array([
        float(R) * np.cos(float(phi)),
        float(R) * np.sin(float(phi)),
        float(Z),
    ], dtype=float)


def cylindrical_vector_xyz(v_R: float, v_phi: float, v_Z: float, phi: float) -> np.ndarray:
    """Return a Cartesian vector from cylindrical physical components."""

    e_R, e_phi, e_Z = _cylindrical_basis(float(phi))
    return float(v_R) * e_R + float(v_phi) * e_phi + float(v_Z) * e_Z


def section_boundary_outward_normal(
    boundary_R: Sequence[float],
    boundary_Z: Sequence[float],
    index: int,
    *,
    phi: float = 0.0,
    axis_R: float | None = None,
    axis_Z: float | None = None,
) -> np.ndarray:
    """Estimate the outward surface normal from a closed poloidal boundary cut.

    The loop plane should be locally parallel to the magnetic boundary surface,
    so its normal is aligned with the boundary-surface normal.  For one
    toroidal cut the surface tangent plane is spanned by the toroidal direction
    and the poloidal boundary tangent.
    """

    R = np.asarray(boundary_R, dtype=float).ravel()
    Z = np.asarray(boundary_Z, dtype=float).ravel()
    if R.size != Z.size or R.size < 3:
        raise ValueError("boundary_R and boundary_Z must have the same length >= 3")
    idx = int(index) % R.size
    prev_i = (idx - 1) % R.size
    next_i = (idx + 1) % R.size
    dR = float(R[next_i] - R[prev_i])
    dZ = float(Z[next_i] - Z[prev_i])
    normal_rz = np.array([dZ, -dR], dtype=float)
    norm = float(np.linalg.norm(normal_rz))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("boundary tangent is degenerate at the requested index")
    normal_rz /= norm

    ref_R = float(np.nanmean(R)) if axis_R is None else float(axis_R)
    ref_Z = float(np.nanmean(Z)) if axis_Z is None else float(axis_Z)
    outward = np.array([float(R[idx]) - ref_R, float(Z[idx]) - ref_Z], dtype=float)
    if float(np.dot(normal_rz, outward)) < 0.0:
        normal_rz = -normal_rz
    return _unit(cylindrical_vector_xyz(normal_rz[0], 0.0, normal_rz[1], phi), name="normal")


def boundary_loop_coil_from_section(
    boundary_R: Sequence[float],
    boundary_Z: Sequence[float],
    *,
    index: int,
    phi: float,
    radius: float,
    current: float = 1.0,
    clearance: float = 0.0,
    axis_R: float | None = None,
    axis_Z: float | None = None,
    metadata: dict | None = None,
) -> BoundaryLoopCoilSpec:
    """Build one boundary-local circular loop from a poloidal section."""

    R = np.asarray(boundary_R, dtype=float).ravel()
    Z = np.asarray(boundary_Z, dtype=float).ravel()
    if R.size != Z.size or R.size < 3:
        raise ValueError("boundary_R and boundary_Z must have the same length >= 3")
    idx = int(index) % R.size
    normal = section_boundary_outward_normal(
        R,
        Z,
        idx,
        phi=float(phi),
        axis_R=axis_R,
        axis_Z=axis_Z,
    )
    anchor = cylindrical_point_xyz(float(R[idx]), float(Z[idx]), float(phi))
    center = anchor + float(clearance) * normal
    return BoundaryLoopCoilSpec(
        center_xyz=tuple(float(v) for v in center),
        normal_xyz=tuple(float(v) for v in normal),
        radius=float(radius),
        current=float(current),
        anchor_R=float(R[idx]),
        anchor_Z=float(Z[idx]),
        anchor_phi=float(phi),
        anchor_theta=None,
        clearance=float(clearance),
        metadata={} if metadata is None else dict(metadata),
    )


def _prepare_boundary_surface(R_surf: np.ndarray, Z_surf: np.ndarray, radial_index: int) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(R_surf, dtype=float)
    Z = np.asarray(Z_surf, dtype=float)
    if R.shape != Z.shape:
        raise ValueError("R_surf and Z_surf must have matching shapes")
    if R.ndim == 2:
        return R, Z
    if R.ndim != 3:
        raise ValueError("surface arrays must have shape (n_phi, n_theta) or (n_phi, n_r, n_theta)")
    ridx = int(radial_index)
    return R[:, ridx, :], Z[:, ridx, :]


def _as_index_array(values: Iterable[int] | None, size: int) -> np.ndarray:
    if values is None:
        return np.arange(int(size), dtype=int)
    idx = np.asarray(list(values), dtype=int).ravel()
    if idx.size == 0:
        raise ValueError("index selections must not be empty")
    return np.mod(idx, int(size))


def boundary_loop_coil_specs_from_surface(
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    phi_vals: Sequence[float],
    theta_vals: Sequence[float],
    *,
    radial_index: int = -1,
    phi_indices: Iterable[int] | None = None,
    theta_indices: Iterable[int] | None = None,
    radius: float,
    current: float = 1.0,
    clearance: float = 0.0,
    mode_m: int | None = None,
    mode_n: int | None = None,
    phase: float = 0.0,
) -> list[BoundaryLoopCoilSpec]:
    """Build a phased array of local circular loops on a boundary surface.

    When ``mode_m`` and ``mode_n`` are supplied, loop currents follow
    ``current * cos(m * theta - n0 * phi + phase)``.  Here ``mode_n`` is the
    signed historical waveform label ``n0``; its positive-``q`` Nardon
    coefficient has signed index ``n_N=-n0`` in
    ``exp(i*(m*theta+n_N*phi))``.
    """

    R_bdy, Z_bdy = _prepare_boundary_surface(R_surf, Z_surf, radial_index)
    phi = np.asarray(phi_vals, dtype=float).ravel()
    theta = np.asarray(theta_vals, dtype=float).ravel()
    if R_bdy.shape != (phi.size, theta.size):
        raise ValueError("boundary surface shape must be (len(phi_vals), len(theta_vals))")
    pidx = _as_index_array(phi_indices, phi.size)
    tidx = _as_index_array(theta_indices, theta.size)
    if (mode_m is None) ^ (mode_n is None):
        raise ValueError("mode_m and mode_n must be supplied together")

    specs: list[BoundaryLoopCoilSpec] = []
    for ip in pidx:
        ref_R = float(np.nanmean(R_bdy[int(ip)]))
        ref_Z = float(np.nanmean(Z_bdy[int(ip)]))
        for it in tidx:
            theta_value = float(theta[int(it)])
            phi_value = float(phi[int(ip)])
            loop_current = float(current)
            metadata: dict[str, float | int] = {
                "phi_index": int(ip),
                "theta_index": int(it),
            }
            if mode_m is not None and mode_n is not None:
                loop_phase = float(mode_m) * theta_value - float(mode_n) * phi_value + float(phase)
                loop_current *= float(np.cos(loop_phase))
                metadata.update({
                    "mode_m": int(mode_m),
                    "mode_n": int(mode_n),
                    "nardon_n": -int(mode_n),
                    "mode_phase": float(phase),
                    "local_phase": loop_phase,
                })
            spec = boundary_loop_coil_from_section(
                R_bdy[int(ip)],
                Z_bdy[int(ip)],
                index=int(it),
                phi=phi_value,
                radius=float(radius),
                current=loop_current,
                clearance=float(clearance),
                axis_R=ref_R,
                axis_Z=ref_Z,
                metadata=metadata,
            )
            specs.append(
                BoundaryLoopCoilSpec(
                    center_xyz=spec.center_xyz,
                    normal_xyz=spec.normal_xyz,
                    radius=spec.radius,
                    current=spec.current,
                    anchor_R=spec.anchor_R,
                    anchor_Z=spec.anchor_Z,
                    anchor_phi=spec.anchor_phi,
                    anchor_theta=theta_value,
                    clearance=spec.clearance,
                    metadata=spec.metadata,
                )
            )
    return specs


def _dipole_from_loop_spec(
    loop: BoundaryLoopCoilSpec,
    *,
    magnetic_moment: float,
    metadata: Mapping[str, object] | None = None,
) -> BoundaryDipoleCoilSpec:
    md = dict(loop.metadata)
    if metadata:
        md.update(dict(metadata))
    return BoundaryDipoleCoilSpec(
        center_xyz=loop.center_xyz,
        normal_xyz=loop.normal_xyz,
        radius=loop.radius,
        magnetic_moment=float(magnetic_moment),
        anchor_R=loop.anchor_R,
        anchor_Z=loop.anchor_Z,
        anchor_phi=loop.anchor_phi,
        anchor_theta=loop.anchor_theta,
        clearance=loop.clearance,
        metadata=md,
    )


def boundary_dipole_coil_from_section(
    boundary_R: Sequence[float],
    boundary_Z: Sequence[float],
    *,
    index: int,
    phi: float,
    radius: float,
    magnetic_moment: float,
    clearance: float = 0.0,
    axis_R: float | None = None,
    axis_Z: float | None = None,
    metadata: Mapping[str, object] | None = None,
) -> BoundaryDipoleCoilSpec:
    """Build one outward-normal finite-loop dipole on a boundary section."""

    loop = boundary_loop_coil_from_section(
        boundary_R,
        boundary_Z,
        index=index,
        phi=phi,
        radius=radius,
        current=1.0,
        clearance=clearance,
        axis_R=axis_R,
        axis_Z=axis_Z,
        metadata={} if metadata is None else dict(metadata),
    )
    return _dipole_from_loop_spec(loop, magnetic_moment=float(magnetic_moment))


def boundary_dipole_mode_actuator_array_from_surface(
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    phi_vals: Sequence[float],
    theta_vals: Sequence[float],
    modes: Sequence[tuple[int, int]],
    *,
    radial_index: int = -1,
    phi_indices: Iterable[int] | None = None,
    theta_indices: Iterable[int] | None = None,
    radius: float,
    unit_moment: float,
    clearance: float = 0.0,
    include_sine: bool = True,
    lower_bound: float = -1.0,
    upper_bound: float = 1.0,
    control_scale: float = 1.0,
    label_prefix: str = "dipole",
) -> BoundaryDipoleActuatorArray:
    """Build helical cosine/sine dipole-current basis columns.

    For each historical waveform label ``(m,n0)``, the cosine column follows
    ``cos(m * theta - n0 * phi)`` and the optional sine column follows
    ``sin(m * theta - n0 * phi)``.  Its positive-``q`` Nardon index is
    ``n_N=-n0``.  The command is dimensionless; a unit command gives each
    finite loop the signed moment ``unit_moment * basis``.
    """

    mode_tuple = tuple((int(m), int(n)) for m, n in modes)
    if not mode_tuple:
        raise ValueError("modes must contain at least one (m, n) pair")
    if len(set(mode_tuple)) != len(mode_tuple):
        raise ValueError("modes must be unique")
    if not np.isfinite(float(unit_moment)) or float(unit_moment) == 0.0:
        raise ValueError("unit_moment must be finite and nonzero")

    actuators: list[BoundaryDipoleActuatorSpec] = []
    basis_defs = (("cos", 0.0), ("sin", -0.5 * np.pi)) if include_sine else (("cos", 0.0),)
    for m_int, n_int in mode_tuple:
        for basis_name, phase in basis_defs:
            loops = boundary_loop_coil_specs_from_surface(
                R_surf,
                Z_surf,
                phi_vals,
                theta_vals,
                radial_index=radial_index,
                phi_indices=phi_indices,
                theta_indices=theta_indices,
                radius=radius,
                current=1.0,
                clearance=clearance,
                mode_m=m_int,
                mode_n=n_int,
                phase=phase,
            )
            label = f"{str(label_prefix)}.m{m_int}.n{n_int}.{basis_name}"
            dipoles = tuple(
                _dipole_from_loop_spec(
                    loop,
                    magnetic_moment=float(unit_moment) * float(loop.current),
                    metadata={
                        "actuator_label": label,
                        "basis": basis_name,
                        "fourier_convention": "cos(m*theta-n*phi+phase)",
                        "nardon_basis": "exp(i*(m*theta+n_N*phi))",
                        "nardon_n": -n_int,
                    },
                )
                for loop in loops
            )
            actuators.append(
                BoundaryDipoleActuatorSpec(
                    label=label,
                    dipoles=dipoles,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    control_scale=control_scale,
                    metadata={
                        "kind": "helical_mode",
                        "m": m_int,
                        "n": n_int,
                        "nardon_n": -n_int,
                        "basis": basis_name,
                        "phase": float(phase),
                        "unit_moment_A_m2": float(unit_moment),
                    },
                )
            )
    return BoundaryDipoleActuatorArray(
        tuple(actuators),
        metadata={
            "kind": "boundary_dipole_mode_array",
            "fourier_convention": "cos(m*theta-n*phi+phase)",
            "nardon_basis": "exp(i*(m*theta+n_N*phi))",
            "nardon_mapping": "n_N=-n",
        },
    )


def boundary_dipole_local_actuator_array_from_surface(
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    phi_vals: Sequence[float],
    theta_vals: Sequence[float],
    sites: Sequence[tuple[int, int]],
    *,
    radial_index: int = -1,
    radius: float,
    unit_moment: float,
    clearance: float = 0.0,
    lower_bound: float = -1.0,
    upper_bound: float = 1.0,
    control_scale: float = 1.0,
    label_prefix: str = "dipole.trim",
) -> BoundaryDipoleActuatorArray:
    """Build one independent local dipole command at each selected surface site."""

    R_bdy, Z_bdy = _prepare_boundary_surface(R_surf, Z_surf, radial_index)
    phi = np.asarray(phi_vals, dtype=float).ravel()
    theta = np.asarray(theta_vals, dtype=float).ravel()
    if R_bdy.shape != (phi.size, theta.size):
        raise ValueError("boundary surface shape must be (len(phi_vals), len(theta_vals))")
    site_tuple = tuple((int(ip) % phi.size, int(it) % theta.size) for ip, it in sites)
    if not site_tuple:
        raise ValueError("sites must contain at least one (phi_index, theta_index) pair")
    if len(set(site_tuple)) != len(site_tuple):
        raise ValueError("local dipole sites must be unique")

    actuators = []
    for ip, it in site_tuple:
        label = f"{str(label_prefix)}.p{ip:03d}.t{it:03d}"
        dipole = boundary_dipole_coil_from_section(
            R_bdy[ip],
            Z_bdy[ip],
            index=it,
            phi=float(phi[ip]),
            radius=radius,
            magnetic_moment=unit_moment,
            clearance=clearance,
            axis_R=float(np.nanmean(R_bdy[ip])),
            axis_Z=float(np.nanmean(Z_bdy[ip])),
            metadata={
                "actuator_label": label,
                "phi_index": ip,
                "theta_index": it,
                "kind": "local_trim",
            },
        )
        actuators.append(
            BoundaryDipoleActuatorSpec(
                label=label,
                dipoles=(dipole,),
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                control_scale=control_scale,
                metadata={
                    "kind": "local_trim",
                    "phi_index": ip,
                    "theta_index": it,
                    "unit_moment_A_m2": float(unit_moment),
                },
            )
        )
    return BoundaryDipoleActuatorArray(tuple(actuators), metadata={"kind": "boundary_dipole_local_array"})


def stack_boundary_dipole_actuator_arrays(
    arrays: Sequence[BoundaryDipoleActuatorArray],
    *,
    metadata: Mapping[str, object] | None = None,
) -> BoundaryDipoleActuatorArray:
    """Stack actuator arrays while preserving their stable column labels."""

    array_tuple = tuple(arrays)
    if not array_tuple:
        raise ValueError("at least one actuator array is required")
    actuators = tuple(spec for array in array_tuple for spec in array.actuators)
    md: dict[str, object] = {}
    for array in array_tuple:
        md.update(dict(array.metadata))
    if metadata:
        md.update(dict(metadata))
    md["kind"] = "stacked_boundary_dipole_array"
    return BoundaryDipoleActuatorArray(actuators, metadata=md)


def boundary_loop_coil_field(spec: BoundaryLoopCoilSpec) -> CoilFieldAnalyticCircular:
    """Return a coil field from one boundary-loop spec."""

    return spec.to_field()


def boundary_loop_coil_superposition(specs: Sequence[BoundaryLoopCoilSpec]) -> CoilFieldSuperposition:
    """Return a divergence-free superposition of boundary-loop coil fields."""

    return CoilFieldSuperposition([spec.to_field() for spec in specs])


__all__ = [
    "BoundaryDipoleActuatorArray",
    "BoundaryDipoleActuatorSpec",
    "BoundaryDipoleCoilSpec",
    "BoundaryLoopCoilSpec",
    "boundary_dipole_coil_from_section",
    "boundary_dipole_local_actuator_array_from_surface",
    "boundary_dipole_mode_actuator_array_from_surface",
    "boundary_loop_coil_field",
    "boundary_loop_coil_from_section",
    "boundary_loop_coil_specs_from_surface",
    "boundary_loop_coil_superposition",
    "cylindrical_point_xyz",
    "cylindrical_vector_xyz",
    "section_boundary_outward_normal",
    "stack_boundary_dipole_actuator_arrays",
]
