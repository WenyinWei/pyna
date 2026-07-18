"""Optional FusionSC field-line diffusion backend for wall heat loads.

The module imports FusionSC only when a model is evaluated.  Fields and
points use cylindrical coordinates at the pyna boundary, while the adapter
enforces the Cartesian and component-order contracts of the installed
FusionSC API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import inspect
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING

import numpy as np

from pyna.fields.cylindrical import as_vector_field_cylindrical
from pyna.fields.periodicity import ToroidalPeriodicity, normalize_nfp
from pyna.toroidal.control.heat_contracts import BoundaryTopologyHeatState
from pyna.toroidal.control.heat_distribution import wall_heat_footprint_from_fusionsc_trace
from pyna.toroidal.control.strike_heat import StrikeSeedBundle, sum_boundary_heat_states

if TYPE_CHECKING:
    from pyna.toroidal.control.boundary_plasma_response import BoundaryPlasmaResponseInput
    from pyna.toroidal.control.boundary_topology_cases import BoundaryTopologyCase
    from pyna.toroidal.perturbation_spectrum import (
        ChaoticLayerInterval,
        RadialPerturbationFourierSpectrum,
        ResonantIslandChain,
    )


TWOPI = 2.0 * np.pi


class FusionSCBackendUnavailableError(ImportError):
    """Raised when the optional FusionSC dependency cannot be imported."""


class FusionSCTraceError(RuntimeError):
    """Raised when a diffusive trace fails or cannot prove wall deposition."""


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _nonnegative_integer(value: int, name: str) -> int:
    result = int(value)
    if result != value or result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _grid_dimension(grid: Any, name: str) -> int | None:
    if isinstance(grid, Mapping):
        value = grid.get(name)
    else:
        value = getattr(grid, name, None)
    if value is None:
        return None
    return int(value)


@dataclass(frozen=True)
class FusionSCComputedField:
    """A cylindrical field sampled on a FusionSC toroidal grid.

    Each component has shape ``(n_phi, n_z, n_r)``.  Keeping the components
    named avoids relying on an ambiguous caller-provided final axis.
    """

    grid: Any
    B_phi: np.ndarray
    B_z: np.ndarray
    B_R: np.ndarray

    def __post_init__(self) -> None:
        if self.grid is None:
            raise ValueError("grid is required")
        B_phi = np.asarray(self.B_phi, dtype=float)
        B_z = np.asarray(self.B_z, dtype=float)
        B_R = np.asarray(self.B_R, dtype=float)
        if B_phi.ndim != 3 or B_z.shape != B_phi.shape or B_R.shape != B_phi.shape:
            raise ValueError("field components must share shape (n_phi, n_z, n_r)")
        if 0 in B_phi.shape or not all(np.all(np.isfinite(value)) for value in (B_phi, B_z, B_R)):
            raise ValueError("field components must be non-empty and finite")
        expected = {
            "nPhi": B_phi.shape[0],
            "nZ": B_phi.shape[1],
            "nR": B_phi.shape[2],
        }
        for name, size in expected.items():
            grid_size = _grid_dimension(self.grid, name)
            if grid_size is not None and grid_size != size:
                raise ValueError(
                    f"grid {name}={grid_size} does not match field axis size {size}"
                )
        object.__setattr__(self, "B_phi", B_phi)
        object.__setattr__(self, "B_z", B_z)
        object.__setattr__(self, "B_R", B_R)

    @property
    def tensor(self) -> np.ndarray:
        """Return ``[phi, z, r, (Bphi, Bz, Br)]`` for FusionSC."""

        return np.stack((self.B_phi, self.B_z, self.B_R), axis=-1)


def fusionsc_computed_field_from_cylindrical(field: object) -> FusionSCComputedField:
    """Convert a pyna cylindrical grid field to FusionSC's named grid tensor."""

    normalized = as_vector_field_cylindrical(field)
    arrays = normalized.cyna_arrays(extend_phi=False)
    R = np.asarray(arrays.R_grid, dtype=float).ravel()
    Z = np.asarray(arrays.Z_grid, dtype=float).ravel()
    phi = np.asarray(arrays.Phi_grid, dtype=float).ravel()
    BR = np.asarray(arrays.BR, dtype=float)
    BZ = np.asarray(arrays.BZ, dtype=float)
    BPhi = np.asarray(arrays.BPhi, dtype=float)
    if BR.shape != (R.size, Z.size, phi.size) or BZ.shape != BR.shape or BPhi.shape != BR.shape:
        raise ValueError("cylindrical field components must have shape (nR, nZ, nPhi)")
    n_sym = arrays.nfp
    period = arrays.field_period
    if phi.size >= 2 and np.isclose(phi[-1] - phi[0], period, rtol=0.0, atol=1.0e-10):
        if not all(np.allclose(value[..., -1], value[..., 0], rtol=1.0e-10, atol=1.0e-12) for value in (BR, BZ, BPhi)):
            raise ValueError("closed cylindrical phi endpoint does not match the first field slice")
        phi = phi[:-1]
        BR, BZ, BPhi = BR[..., :-1], BZ[..., :-1], BPhi[..., :-1]
    if phi.size < 2 or not np.isclose(phi[0], 0.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("FusionSC computed fields require an endpoint-false phi grid starting at zero")
    expected_phi = np.linspace(0.0, period, phi.size, endpoint=False)
    if not np.allclose(phi, expected_phi, rtol=0.0, atol=1.0e-10):
        raise ValueError("cylindrical phi grid must be uniform over one field period")
    grid = {
        "rMin": float(R[0]),
        "rMax": float(R[-1]),
        "zMin": float(Z[0]),
        "zMax": float(Z[-1]),
        "nSym": n_sym,
        "nR": int(R.size),
        "nZ": int(Z.size),
        "nPhi": int(phi.size),
    }
    return FusionSCComputedField(
        grid=grid,
        B_phi=np.transpose(BPhi, (2, 1, 0)),
        B_z=np.transpose(BZ, (2, 1, 0)),
        B_R=np.transpose(BR, (2, 1, 0)),
    )


def _mapping_value(value: Mapping[str, object], names: Sequence[str]) -> object:
    for name in names:
        if name in value:
            return value[name]
    joined = ", ".join(names)
    raise KeyError(f"computed field mapping requires one of: {joined}")


def _coerce_computed_field(value: object) -> FusionSCComputedField:
    if isinstance(value, FusionSCComputedField):
        return value
    if isinstance(value, Mapping):
        try:
            return FusionSCComputedField(
                grid=value["grid"],
                B_phi=_mapping_value(value, ("B_phi", "Bphi")),
                B_z=_mapping_value(value, ("B_z", "Bz")),
                B_R=_mapping_value(value, ("B_R", "B_r", "Br")),
            )
        except KeyError as exc:
            raise TypeError(
                "field_builder mapping must provide grid, B_phi/Bphi, B_z/Bz, and B_R/Br"
            ) from exc
    raise TypeError("field_builder must return FusionSCComputedField or a named component mapping")


@dataclass(frozen=True)
class FusionSCWallSurfaceSpec:
    """A closed three-dimensional wall sampled as ``(phi, poloidal)``.

    ``phi_values`` omit the repeated periodic endpoint.  ``R`` and ``Z`` are
    converted at every section to a Cartesian quad mesh; no axisymmetric wall
    approximation is made.
    """

    phi_values: np.ndarray
    R: np.ndarray
    Z: np.ndarray
    toroidal_period: float = TWOPI
    wrap_phi: bool = True
    wrap_poloidal: bool = True
    geometry_grid: Any = None
    geometry_grid_shape: tuple[int, int, int] = (48, 48, 48)
    geometry_grid_padding: float = 0.02

    def __post_init__(self) -> None:
        phi = np.asarray(self.phi_values, dtype=float).ravel()
        R = np.asarray(self.R, dtype=float)
        Z = np.asarray(self.Z, dtype=float)
        period = _finite_positive(self.toroidal_period, "toroidal_period")
        if R.shape != Z.shape or R.ndim != 2:
            raise ValueError("wall R and Z must share shape (n_phi, n_poloidal)")
        if R.shape[0] != phi.size or phi.size < 3 or R.shape[1] < 3:
            raise ValueError("wall requires at least three toroidal and poloidal vertices")
        if not np.all(np.isfinite(phi)) or np.any(np.diff(phi) <= 0.0):
            raise ValueError("wall phi_values must be finite and strictly increasing")
        if phi[-1] - phi[0] >= period * (1.0 - 1.0e-12):
            raise ValueError("wall phi_values must omit the repeated periodic endpoint")
        if not np.all(np.isfinite(R)) or not np.all(np.isfinite(Z)) or np.any(R <= 0.0):
            raise ValueError("wall R and Z must be finite and wall R must be positive")
        segment_lengths = np.hypot(np.roll(R, -1, axis=1) - R, np.roll(Z, -1, axis=1) - Z)
        if np.any(segment_lengths <= 0.0):
            raise ValueError("wall poloidal vertices must define non-zero closed segments")
        shape_values = tuple(self.geometry_grid_shape)
        if len(shape_values) != 3:
            raise ValueError("geometry_grid_shape must contain (n_x, n_y, n_z)")
        grid_shape = tuple(_nonnegative_integer(value, "geometry_grid_shape") for value in shape_values)
        if any(value < 2 for value in grid_shape):
            raise ValueError("geometry_grid_shape entries must be at least two")
        padding = float(self.geometry_grid_padding)
        if not np.isfinite(padding) or padding < 0.0:
            raise ValueError("geometry_grid_padding must be finite and non-negative")
        object.__setattr__(self, "phi_values", phi)
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "Z", Z)
        object.__setattr__(self, "toroidal_period", period)
        object.__setattr__(self, "wrap_phi", bool(self.wrap_phi))
        object.__setattr__(self, "wrap_poloidal", bool(self.wrap_poloidal))
        object.__setattr__(self, "geometry_grid_shape", grid_shape)
        object.__setattr__(self, "geometry_grid_padding", padding)

    @property
    def cartesian_vertices(self) -> np.ndarray:
        """Return mesh vertices with FusionSC shape ``(3, n_phi, n_pol)``."""

        cos_phi = np.cos(self.phi_values)[:, None]
        sin_phi = np.sin(self.phi_values)[:, None]
        return np.stack((self.R * cos_phi, self.R * sin_phi, self.Z), axis=0)

    def trace_geometry_grid(self) -> Any:
        """Return a supplied or bounding-box Cartesian geometry index grid."""

        if self.geometry_grid is not None:
            return self.geometry_grid
        vertices = self.cartesian_vertices.reshape(3, -1)
        lower = np.min(vertices, axis=1)
        upper = np.max(vertices, axis=1)
        span = upper - lower
        scale = max(float(np.max(span)), 1.0)
        padding = np.maximum(float(self.geometry_grid_padding) * span, 1.0e-6 * scale)
        lower = lower - padding
        upper = upper + padding
        n_x, n_y, n_z = self.geometry_grid_shape
        return {
            "xMin": float(lower[0]),
            "xMax": float(upper[0]),
            "yMin": float(lower[1]),
            "yMax": float(upper[1]),
            "zMin": float(lower[2]),
            "zMax": float(upper[2]),
            "nX": n_x,
            "nY": n_y,
            "nZ": n_z,
        }

    def _sections_at_s(self, s_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        target = np.asarray(s_values, dtype=float)
        R_out = np.empty((self.phi_values.size, target.size), dtype=float)
        Z_out = np.empty_like(R_out)
        for index, (section_R, section_Z) in enumerate(zip(self.R, self.Z)):
            lengths = np.hypot(
                np.roll(section_R, -1) - section_R,
                np.roll(section_Z, -1) - section_Z,
            )
            total = float(np.sum(lengths))
            starts = np.concatenate(([0.0], np.cumsum(lengths[:-1])))
            nodes = (starts + 0.5 * lengths) / total
            nodes = np.concatenate(([nodes[-1] - 1.0], nodes, [nodes[0] + 1.0]))
            R_nodes = np.concatenate((section_R[-1:], section_R, section_R[:1]))
            Z_nodes = np.concatenate((section_Z[-1:], section_Z, section_Z[:1]))
            R_out[index] = np.interp(target, nodes, R_nodes)
            Z_out[index] = np.interp(target, nodes, Z_nodes)
        return R_out, Z_out

    def _interpolate_phi(self, values: np.ndarray, phi_values: np.ndarray) -> np.ndarray:
        phi0 = float(self.phi_values[0])
        query = phi0 + np.mod(np.asarray(phi_values, dtype=float) - phi0, self.toroidal_period)
        source_phi = np.concatenate((self.phi_values, [phi0 + self.toroidal_period]))
        source_values = np.concatenate((values, values[:1]), axis=0)
        upper = np.searchsorted(source_phi, query, side="right")
        upper = np.clip(upper, 1, self.phi_values.size)
        lower = upper - 1
        span = source_phi[upper] - source_phi[lower]
        alpha = (query - source_phi[lower]) / span
        return source_values[lower] * (1.0 - alpha[:, None]) + source_values[upper] * alpha[:, None]

    def cell_areas(self, phi_edges: Sequence[float], s_edges: Sequence[float]) -> np.ndarray:
        """Approximate physical wall area in requested ``(phi, s)`` cells."""

        phi = np.asarray(phi_edges, dtype=float).ravel()
        s = np.asarray(s_edges, dtype=float).ravel()
        if phi.size < 2 or s.size < 2 or np.any(np.diff(phi) <= 0.0) or np.any(np.diff(s) <= 0.0):
            raise ValueError("phi_edges and s_edges must be strictly increasing")
        if phi[-1] - phi[0] > self.toroidal_period * (1.0 + 1.0e-12):
            raise ValueError("heat-map phi extent exceeds the wall toroidal period")
        section_R, section_Z = self._sections_at_s(s)
        R = self._interpolate_phi(section_R, phi)
        Z = self._interpolate_phi(section_Z, phi)
        vertices = np.stack((R * np.cos(phi)[:, None], R * np.sin(phi)[:, None], Z), axis=-1)
        p00 = vertices[:-1, :-1]
        p10 = vertices[1:, :-1]
        p11 = vertices[1:, 1:]
        p01 = vertices[:-1, 1:]
        first = 0.5 * np.linalg.norm(np.cross(p10 - p00, p11 - p00), axis=-1)
        second = 0.5 * np.linalg.norm(np.cross(p11 - p00, p01 - p00), axis=-1)
        areas = first + second
        if not np.all(np.isfinite(areas)) or np.any(areas <= 0.0):
            raise ValueError("wall geometry produces non-positive heat-map cell areas")
        return areas


def load_fusionsc_rz_section_wall(
    path: str | Path,
    *,
    full_torus: bool = True,
    geometry_grid_shape: tuple[int, int, int] = (48, 48, 48),
    geometry_grid_padding: float = 0.02,
) -> FusionSCWallSurfaceSpec:
    """Load a headered R-Z wall with ``nfp n_poloidal n_phi`` first line.

    The body is section-major and contains two columns, R and Z.  Toroidal
    angles are implicit, uniformly sample one field period, and omit the
    periodic endpoint.  Full-torus tiling is the default because a Cartesian
    collision mesh cannot identify opposite field-period faces.
    """

    source = Path(path).expanduser()
    with source.open("r", encoding="utf-8") as handle:
        header = handle.readline().split()
    if len(header) != 3:
        raise ValueError("R-Z wall header must contain nfp n_poloidal n_phi")
    try:
        nfp, n_poloidal, n_phi = (int(value) for value in header)
    except ValueError as exc:
        raise ValueError("R-Z wall header entries must be integers") from exc
    nfp = normalize_nfp(nfp)
    if n_poloidal < 3 or n_phi < 3:
        raise ValueError("R-Z wall header dimensions are invalid")
    data = np.loadtxt(source, skiprows=1, dtype=float)
    if data.shape != (n_phi * n_poloidal, 2):
        raise ValueError(
            "R-Z wall body must contain exactly n_phi*n_poloidal rows and two columns"
        )
    sections = data.reshape(n_phi, n_poloidal, 2)
    period = ToroidalPeriodicity(nfp).field_period
    phi_one = np.linspace(0.0, period, n_phi, endpoint=False)
    if full_torus and nfp > 1:
        phi = np.concatenate([phi_one + index * period for index in range(nfp)])
        R = np.concatenate([sections[:, :, 0]] * nfp, axis=0)
        Z = np.concatenate([sections[:, :, 1]] * nfp, axis=0)
        toroidal_period = TWOPI
    else:
        phi = phi_one
        R = sections[:, :, 0]
        Z = sections[:, :, 1]
        toroidal_period = period
    return FusionSCWallSurfaceSpec(
        phi_values=phi,
        R=R,
        Z=Z,
        toroidal_period=toroidal_period,
        wrap_phi=True,
        wrap_poloidal=True,
        geometry_grid_shape=geometry_grid_shape,
        geometry_grid_padding=geometry_grid_padding,
    )


@dataclass(frozen=True)
class FusionSCSeedSpec:
    """Weighted field-line starts expressed as cylindrical ``(R, Z, phi)``."""

    R: np.ndarray
    Z: np.ndarray
    phi: np.ndarray
    weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        try:
            R, Z, phi = np.broadcast_arrays(
                np.asarray(self.R, dtype=float),
                np.asarray(self.Z, dtype=float),
                np.asarray(self.phi, dtype=float),
            )
        except ValueError as exc:
            raise ValueError("seed R, Z, and phi must be broadcast-compatible") from exc
        seed_shape = R.shape
        if R.size == 0 or not np.all(np.isfinite(R)) or np.any(R <= 0.0):
            raise ValueError("seed R must be non-empty, positive, and finite")
        if not np.all(np.isfinite(Z)) or not np.all(np.isfinite(phi)):
            raise ValueError("seed Z and phi must be finite")
        if self.weights is None:
            weights = np.ones(R.size, dtype=float)
        else:
            try:
                weights = np.broadcast_to(np.asarray(self.weights, dtype=float), seed_shape).copy()
            except ValueError as exc:
                raise ValueError("seed weights must match the broadcast seed shape") from exc
        R = R.ravel()
        Z = Z.ravel()
        phi = phi.ravel()
        weights = weights.ravel()
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
            raise ValueError("seed weights must be finite, non-negative, and have positive sum")
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "Z", Z)
        object.__setattr__(self, "phi", phi)
        object.__setattr__(self, "weights", weights)

    @property
    def cartesian_points(self) -> np.ndarray:
        """Return starts with FusionSC shape ``(3, n_seed)`` in xyz order."""

        return np.stack((self.R * np.cos(self.phi), self.R * np.sin(self.phi), self.Z), axis=0)

    def power_weights(self, total_power: float) -> np.ndarray:
        total = _finite_positive(total_power, "total_power")
        return np.asarray(self.weights, dtype=float) * (total / float(np.sum(self.weights)))


@dataclass(frozen=True)
class FusionSCTransportSpec:
    """Validated arguments for FusionSC's diffusive tracing model.

    Exactly one perpendicular model and one parallel model are mandatory.
    Consequently this spec cannot represent a ballistic trace.
    """

    isotropic_diffusion_coefficient: float | None = None
    rz_diffusion_coefficient: float | None = None
    parallel_convection_velocity: float | None = None
    parallel_diffusion_coefficient: float | None = None
    mean_free_path: float = 1.0
    mean_free_path_growth: float = 0.0
    distance_limit: float = 10000.0
    turn_limit: int = 0
    step_limit: int = 0
    step_size: float = 0.001
    collision_limit: int = 1
    direction: str = "forward"
    ignore_collisions_before: float = 0.0
    allow_reversal: bool = False

    def __post_init__(self) -> None:
        perpendicular = (
            self.isotropic_diffusion_coefficient,
            self.rz_diffusion_coefficient,
        )
        parallel = (
            self.parallel_convection_velocity,
            self.parallel_diffusion_coefficient,
        )
        if sum(value is not None for value in perpendicular) != 1:
            raise ValueError(
                "exactly one of isotropic_diffusion_coefficient and "
                "rz_diffusion_coefficient is required"
            )
        if sum(value is not None for value in parallel) != 1:
            raise ValueError(
                "exactly one of parallel_convection_velocity and "
                "parallel_diffusion_coefficient is required"
            )
        for name in (
            "isotropic_diffusion_coefficient",
            "rz_diffusion_coefficient",
            "parallel_convection_velocity",
            "parallel_diffusion_coefficient",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_positive(value, name))
        object.__setattr__(self, "mean_free_path", _finite_positive(self.mean_free_path, "mean_free_path"))
        growth = float(self.mean_free_path_growth)
        if not np.isfinite(growth) or growth < 0.0:
            raise ValueError("mean_free_path_growth must be finite and non-negative")
        object.__setattr__(self, "mean_free_path_growth", growth)
        object.__setattr__(self, "distance_limit", _finite_positive(self.distance_limit, "distance_limit"))
        object.__setattr__(self, "step_size", _finite_positive(self.step_size, "step_size"))
        turn_limit = _nonnegative_integer(self.turn_limit, "turn_limit")
        step_limit = _nonnegative_integer(self.step_limit, "step_limit")
        collision_limit = _nonnegative_integer(self.collision_limit, "collision_limit")
        if collision_limit != 1:
            raise ValueError("collision_limit must be one so the first wall hit terminates tracing")
        direction = str(self.direction)
        if direction not in {"forward", "backward", "cw", "ccw"}:
            raise ValueError("direction must be forward, backward, cw, or ccw")
        ignore = float(self.ignore_collisions_before)
        if not np.isfinite(ignore) or ignore < 0.0:
            raise ValueError("ignore_collisions_before must be finite and non-negative")
        object.__setattr__(self, "turn_limit", turn_limit)
        object.__setattr__(self, "step_limit", step_limit)
        object.__setattr__(self, "collision_limit", collision_limit)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "ignore_collisions_before", ignore)
        object.__setattr__(self, "allow_reversal", bool(self.allow_reversal))

    @property
    def perpendicular_model(self) -> tuple[str, float]:
        if self.isotropic_diffusion_coefficient is not None:
            return "isotropic", float(self.isotropic_diffusion_coefficient)
        return "rz", float(self.rz_diffusion_coefficient)

    @property
    def parallel_model(self) -> tuple[str, float]:
        if self.parallel_convection_velocity is not None:
            return "convection", float(self.parallel_convection_velocity)
        return "diffusion", float(self.parallel_diffusion_coefficient)

    def trace_kwargs(self) -> dict[str, object]:
        """Return current FusionSC ``flt.trace`` keyword names."""

        result: dict[str, object] = {
            "distanceLimit": self.distance_limit,
            "turnLimit": self.turn_limit,
            "stepLimit": self.step_limit,
            "stepSize": self.step_size,
            "collisionLimit": self.collision_limit,
            "meanFreePath": self.mean_free_path,
            "meanFreePathGrowth": self.mean_free_path_growth,
            "direction": self.direction,
            "resultFormat": "dict",
            "ignoreCollisionsBefore": self.ignore_collisions_before,
            "allowReversal": self.allow_reversal,
        }
        if self.isotropic_diffusion_coefficient is not None:
            result["isotropicDiffusionCoefficient"] = self.isotropic_diffusion_coefficient
        else:
            result["rzDiffusionCoefficient"] = self.rz_diffusion_coefficient
        if self.parallel_convection_velocity is not None:
            result["parallelConvectionVelocity"] = self.parallel_convection_velocity
        else:
            result["parallelDiffusionCoefficient"] = self.parallel_diffusion_coefficient
        return result


def _load_fusionsc() -> Any:
    try:
        return importlib.import_module("fusionsc")
    except ImportError as exc:
        raise FusionSCBackendUnavailableError(
            "FusionSC is required for FusionSCFieldLineDiffusionHeatModel; "
            "no ballistic fallback is available"
        ) from exc


def _resolve_spec(
    value: object,
    case: "BoundaryTopologyCase",
    request: "BoundaryPlasmaResponseInput",
    expected_type: type,
    name: str,
) -> object:
    resolved = value(case, request) if callable(value) else value
    if not isinstance(resolved, expected_type):
        raise TypeError(f"{name} must resolve to {expected_type.__name__}")
    return resolved


def _is_collision_reason(value: object) -> bool:
    raw = getattr(value, "raw", None)
    if raw is not None:
        return int(raw) == 7
    if isinstance(value, (int, np.integer)):
        return int(value) == 7
    text = str(value).strip().lower().replace("_", "").replace("-", "")
    return text in {"7", "collision", "collisionlimit"}


def _validated_trace_result(result: object, n_seeds: int) -> tuple[Mapping[str, object], np.ndarray]:
    if not isinstance(result, Mapping):
        raise FusionSCTraceError("FusionSC trace must return resultFormat='dict'")
    if "endPoints" not in result:
        raise FusionSCTraceError("FusionSC trace result is missing Cartesian endPoints")
    endpoints = np.asarray(result["endPoints"], dtype=float)
    if endpoints.ndim < 2 or endpoints.shape[0] < 3 or endpoints[0].size != n_seeds:
        raise FusionSCTraceError("FusionSC endPoints do not match the launched seed count")
    if "stopReasons" not in result:
        raise FusionSCTraceError(
            "FusionSC trace result is missing stopReasons; wall collisions cannot be proven"
        )
    reasons = np.asarray(result["stopReasons"], dtype=object).ravel()
    if reasons.size != n_seeds:
        raise FusionSCTraceError("FusionSC stopReasons do not match the launched seed count")
    collision_mask = np.fromiter((_is_collision_reason(value) for value in reasons), dtype=bool)
    if not np.any(collision_mask):
        failed = sorted({str(value) for value in reasons})
        raise FusionSCTraceError(
            "no diffusive trace reached the wall; stop reasons: " + ", ".join(failed)
        )
    endpoint_flat = endpoints.reshape(endpoints.shape[0], -1)
    if not np.all(np.isfinite(endpoint_flat[:3, collision_mask])):
        raise FusionSCTraceError("FusionSC returned non-finite wall-collision endpoints")
    return result, collision_mask


def _callable_name(value: Callable[..., object]) -> str:
    module = getattr(value, "__module__", "")
    name = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))
    return f"{module}.{name}" if module else str(name)


@dataclass(frozen=True)
class FusionSCFieldLineDiffusionHeatModel:
    """FusionSC implementation of ``BoundaryTopologyHeatForwardModel``.

    ``field_builder(case, request)`` returns :class:`FusionSCComputedField` or
    an equivalent named mapping.  ``wall`` and ``seeds`` may be static specs or
    callables with the same ``(case, request)`` signature.
    """

    field_builder: Callable[[object, object], object]
    wall: FusionSCWallSurfaceSpec | Callable[[object, object], FusionSCWallSurfaceSpec]
    seeds: FusionSCSeedSpec | Callable[[object, object], FusionSCSeedSpec]
    transport: FusionSCTransportSpec
    total_power: float = 1.0
    n_phi_bins: int | None = None
    n_s_bins: int = 160
    field_period: float | None = None
    minimum_collision_fraction: float = 0.5
    renormalize_deposited_power: bool = False
    trace_function: Callable[..., object] | None = field(default=None, repr=False, compare=False)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not callable(self.field_builder):
            raise TypeError("field_builder must be callable")
        if not isinstance(self.wall, FusionSCWallSurfaceSpec) and not callable(self.wall):
            raise TypeError("wall must be FusionSCWallSurfaceSpec or a callable")
        if not isinstance(self.seeds, FusionSCSeedSpec) and not callable(self.seeds):
            raise TypeError("seeds must be FusionSCSeedSpec or a callable")
        if not isinstance(self.transport, FusionSCTransportSpec):
            raise TypeError("transport must be FusionSCTransportSpec")
        total_power = _finite_positive(self.total_power, "total_power")
        n_phi = None if self.n_phi_bins is None else _nonnegative_integer(self.n_phi_bins, "n_phi_bins")
        n_s = _nonnegative_integer(self.n_s_bins, "n_s_bins")
        if n_phi is not None and n_phi < 1:
            raise ValueError("n_phi_bins must be at least one")
        if n_s < 1:
            raise ValueError("n_s_bins must be at least one")
        period = None if self.field_period is None else _finite_positive(self.field_period, "field_period")
        minimum_fraction = float(self.minimum_collision_fraction)
        if not np.isfinite(minimum_fraction) or not 0.0 < minimum_fraction <= 1.0:
            raise ValueError("minimum_collision_fraction must lie in (0, 1]")
        if self.trace_function is not None and not callable(self.trace_function):
            raise TypeError("trace_function must be callable")
        object.__setattr__(self, "total_power", total_power)
        object.__setattr__(self, "n_phi_bins", n_phi)
        object.__setattr__(self, "n_s_bins", n_s)
        object.__setattr__(self, "field_period", period)
        object.__setattr__(self, "minimum_collision_fraction", minimum_fraction)
        object.__setattr__(self, "renormalize_deposited_power", bool(self.renormalize_deposited_power))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def evaluate(
        self,
        case: "BoundaryTopologyCase",
        request: "BoundaryPlasmaResponseInput",
        spectrum: "RadialPerturbationFourierSpectrum",
        chains: Sequence["ResonantIslandChain"],
        intervals: Sequence["ChaoticLayerInterval"],
    ) -> BoundaryTopologyHeatState:
        del spectrum, chains, intervals
        wall = _resolve_spec(self.wall, case, request, FusionSCWallSurfaceSpec, "wall")
        seeds = _resolve_spec(self.seeds, case, request, FusionSCSeedSpec, "seeds")
        assert isinstance(wall, FusionSCWallSurfaceSpec)
        assert isinstance(seeds, FusionSCSeedSpec)
        fusionsc = _load_fusionsc()
        field_data = _coerce_computed_field(self.field_builder(case, request))

        period = wall.toroidal_period if self.field_period is None else self.field_period
        if period > wall.toroidal_period * (1.0 + 1.0e-12):
            raise ValueError("field_period cannot exceed wall.toroidal_period")
        n_phi_bins = wall.phi_values.size if self.n_phi_bins is None else self.n_phi_bins

        try:
            config = fusionsc.magnetics.MagneticConfig.fromComputed(
                field_data.grid,
                field_data.tensor,
            )
            geometry = fusionsc.geometry.Geometry.quadMesh(
                wall.cartesian_vertices,
                wrapU=wall.wrap_phi,
                wrapV=wall.wrap_poloidal,
            )
        except Exception as exc:
            raise FusionSCTraceError("failed to construct the FusionSC field or wall geometry") from exc

        trace = fusionsc.flt.trace if self.trace_function is None else self.trace_function
        trace_kwargs = self.transport.trace_kwargs()
        trace_kwargs["geometryGrid"] = wall.trace_geometry_grid()
        try:
            trace_result = trace(
                seeds.cartesian_points,
                config,
                geometry=geometry,
                **trace_kwargs,
            )
        except Exception as exc:
            raise FusionSCTraceError(
                "FusionSC diffusive trace failed; ballistic fallback is disabled"
            ) from exc
        if inspect.isawaitable(trace_result):
            raise FusionSCTraceError("trace_function must complete synchronously")
        trace_result, collision_mask = _validated_trace_result(trace_result, seeds.R.size)
        launched_power = seeds.power_weights(self.total_power)
        collision_count_fraction = float(np.mean(collision_mask))
        collision_power = float(np.sum(launched_power[collision_mask]))
        collision_power_fraction = collision_power / float(self.total_power)
        if collision_power_fraction < self.minimum_collision_fraction:
            raise FusionSCTraceError(
                "diffusive wall-collision power fraction "
                f"{collision_power_fraction:.6g} is below minimum_collision_fraction "
                f"{self.minimum_collision_fraction:.6g}"
            )
        endpoints = np.asarray(trace_result["endPoints"], dtype=float).reshape(-1, seeds.R.size)
        collision_result = {"endPoints": endpoints[:, collision_mask]}

        footprint = wall_heat_footprint_from_fusionsc_trace(
            collision_result,
            wall.phi_values,
            wall.R,
            wall.Z,
            weights=launched_power[collision_mask],
            n_phi_bins=n_phi_bins,
            n_s_bins=self.n_s_bins,
            field_period=period,
        )
        deposited = float(np.sum(footprint.heat))
        if not np.isfinite(deposited) or deposited <= 0.0:
            raise FusionSCTraceError("diffusive trace produced no finite wall-deposited power")
        areas = wall.cell_areas(footprint.phi_edges, footprint.s_edges)
        cell_power = np.asarray(footprint.heat, dtype=float)
        target_deposited_power = collision_power
        if self.renormalize_deposited_power:
            target_deposited_power = float(self.total_power)
            cell_power *= target_deposited_power / deposited
        heat = cell_power / areas
        integrated = float(np.sum(heat * areas))
        if not np.isfinite(integrated) or integrated <= 0.0:
            raise FusionSCTraceError("wall heat normalization failed")
        heat *= target_deposited_power / integrated

        perpendicular_name, perpendicular_value = self.transport.perpendicular_model
        parallel_name, parallel_value = self.transport.parallel_model
        metadata = dict(self.metadata)
        metadata.update(
            {
                "model": "fusionsc_field_line_diffusion",
                "backend": "fusionsc.flt.trace",
                "trace_callable": _callable_name(trace),
                "trace_model": "diffusive",
                "diffusive_trace": True,
                "actual_diffusive_trace": True,
                "fusionsc_trace_invoked": True,
                "ballistic_fallback": False,
                "quantitative_transport": True,
                "perpendicular_diffusion_model": perpendicular_name,
                "perpendicular_diffusion_coefficient": perpendicular_value,
                "parallel_transport_model": parallel_name,
                "parallel_transport_coefficient": parallel_value,
                "field_tensor_axes": ("phi", "z", "r", "component"),
                "field_component_order": ("Bphi", "Bz", "Br"),
                "trace_coordinates": "cartesian_xyz",
                "endpoint_coordinates": "cartesian_xyz_converted_to_cylindrical",
                "wall_geometry": "full_3d_quad_mesh",
                "wall_collision_verified": True,
                "launched_seed_count": int(seeds.R.size),
                "wall_collision_count": int(np.count_nonzero(collision_mask)),
                "wall_collision_count_fraction": collision_count_fraction,
                "wall_collision_power_fraction": collision_power_fraction,
                "minimum_collision_fraction": float(self.minimum_collision_fraction),
                "field_period": float(period),
                "launched_power": float(self.total_power),
                "deposited_power": float(target_deposited_power),
                "unresolved_power": float(self.total_power - collision_power),
                "renormalize_deposited_power": bool(self.renormalize_deposited_power),
                "normalization": "sum(heat * cell_areas) == deposited_power",
            }
        )
        return BoundaryTopologyHeatState(
            heat=heat,
            phi_values=footprint.phi_centers,
            s_values=footprint.s_centers,
            cell_areas=areas,
            metadata=metadata,
        )


def _validated_topology_seed_bundles(value: object) -> tuple[StrikeSeedBundle, ...]:
    if isinstance(value, StrikeSeedBundle):
        raise TypeError("seed_bundles must be a sequence of StrikeSeedBundle values")
    try:
        bundles = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("seed_bundles must resolve to a StrikeSeedBundle sequence") from exc
    if not bundles:
        raise ValueError("seed_bundles must not be empty")
    if not all(isinstance(bundle, StrikeSeedBundle) for bundle in bundles):
        raise TypeError("seed_bundles must contain only StrikeSeedBundle values")
    weight_kinds = {bundle.weight_kind for bundle in bundles}
    if len(weight_kinds) != 1:
        raise ValueError("relative and power topology seed weights cannot be mixed")
    return bundles


def _directional_power_allocation(
    bundles: Sequence[StrikeSeedBundle],
    total_power: float | None,
) -> tuple[str, float, dict[str, float]]:
    weight_kind = bundles[0].weight_kind
    directional_weight = {
        direction: float(
            sum(
                float(np.sum(bundle.weights))
                for bundle in bundles
                if bundle.direction == direction
            )
        )
        for direction in ("+", "-")
    }
    launched_weight = float(sum(directional_weight.values()))
    if not np.isfinite(launched_weight) or launched_weight <= 0.0:
        raise ValueError("topology seed bundles contain no finite positive launched weight")

    if weight_kind == "relative":
        if total_power is None:
            raise ValueError("relative topology seed weights require total_power")
        launched_power = _finite_positive(total_power, "total_power")
        scale = launched_power / launched_weight
        allocation = {
            direction: weight * scale
            for direction, weight in directional_weight.items()
            if weight > 0.0
        }
    else:
        launched_power = launched_weight
        if total_power is not None and not np.isclose(
            float(total_power), launched_power, rtol=1.0e-10, atol=1.0e-12
        ):
            raise ValueError("total_power cannot renormalize absolute topology seed powers")
        allocation = {
            direction: weight
            for direction, weight in directional_weight.items()
            if weight > 0.0
        }
    return weight_kind, launched_power, allocation


def _topology_seed_spec(bundles: Sequence[StrikeSeedBundle]) -> FusionSCSeedSpec:
    return FusionSCSeedSpec(
        R=np.concatenate([bundle.R for bundle in bundles]),
        Z=np.concatenate([bundle.Z for bundle in bundles]),
        phi=np.concatenate([bundle.phi for bundle in bundles]),
        weights=np.concatenate([bundle.weights for bundle in bundles]),
    )


def _heat_state_power_accounting(
    state: BoundaryTopologyHeatState,
    *,
    expected_launched_power: float,
) -> tuple[float, float, float]:
    area = None if state.cell_areas is None else np.asarray(state.cell_areas, dtype=float)
    deposited = float(np.sum(state.heat) if area is None else np.sum(state.heat * area))
    metadata = dict(state.metadata)
    try:
        launched = float(metadata["launched_power"])
        unresolved = float(metadata["unresolved_power"])
        reported_deposited = float(metadata["deposited_power"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FusionSCTraceError(
            "topology-guided FusionSC component lacks complete power accounting"
        ) from exc
    values = np.asarray((launched, deposited, unresolved, reported_deposited), dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise FusionSCTraceError("topology-guided FusionSC power accounting is invalid")
    tolerance = max(1.0e-14, 1.0e-10 * expected_launched_power)
    if not np.isclose(launched, expected_launched_power, rtol=1.0e-10, atol=tolerance):
        raise FusionSCTraceError("FusionSC component launched power differs from its allocation")
    if not np.isclose(deposited, reported_deposited, rtol=1.0e-10, atol=tolerance):
        raise FusionSCTraceError("FusionSC component heat integral differs from deposited power")
    if not np.isclose(deposited + unresolved, launched, rtol=1.0e-10, atol=tolerance):
        raise FusionSCTraceError(
            "FusionSC component deposited and unresolved powers do not conserve launched power"
        )
    return launched, deposited, unresolved


@dataclass(frozen=True)
class FusionSCTopologyGuidedHeatModel:
    """Run FusionSC diffusion from topology-derived, direction-aware seeds.

    ``seed_bundles`` may be static or a callable receiving
    ``(case, request, spectrum, chains, intervals)``.  Bundles are grouped by
    their explicit physical ``+phi``/``-phi`` direction.  Each used direction
    must have its own :class:`FusionSCTransportSpec`; the spec's FusionSC
    ``direction`` value is passed through unchanged, so no field-orientation
    convention is inferred here.

    Direction entries in ``transport_provenance_by_direction`` default to
    non-quantitative.  Set an entry's ``quantitative`` flag only when its
    coefficients and field-direction mapping have been validated for the
    case.  The combined heat state is quantitative only when topology,
    weights, and every used transport entry are all quantitative.
    """

    field_builder: Callable[[object, object], object]
    wall: FusionSCWallSurfaceSpec | Callable[[object, object], FusionSCWallSurfaceSpec]
    seed_bundles: Sequence[StrikeSeedBundle] | Callable[
        [
            "BoundaryTopologyCase",
            "BoundaryPlasmaResponseInput",
            "RadialPerturbationFourierSpectrum",
            Sequence["ResonantIslandChain"],
            Sequence["ChaoticLayerInterval"],
        ],
        Sequence[StrikeSeedBundle],
    ]
    transport_by_direction: Mapping[str, FusionSCTransportSpec]
    total_power: float | None = None
    transport_provenance_by_direction: Mapping[str, Mapping[str, object]] = field(
        default_factory=dict
    )
    n_phi_bins: int | None = None
    n_s_bins: int = 160
    field_period: float | None = None
    minimum_collision_fraction: float = 0.5
    trace_function: Callable[..., object] | None = field(default=None, repr=False, compare=False)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not callable(self.field_builder):
            raise TypeError("field_builder must be callable")
        if not isinstance(self.wall, FusionSCWallSurfaceSpec) and not callable(self.wall):
            raise TypeError("wall must be FusionSCWallSurfaceSpec or a callable")
        static_allocation: dict[str, float] | None = None
        if not callable(self.seed_bundles):
            bundles = _validated_topology_seed_bundles(self.seed_bundles)
            _, _, static_allocation = _directional_power_allocation(
                bundles, self.total_power
            )
            object.__setattr__(self, "seed_bundles", bundles)

        transports = dict(self.transport_by_direction)
        invalid_directions = set(transports) - {"+", "-"}
        if invalid_directions:
            raise ValueError("transport_by_direction keys must be explicit '+' or '-'")
        if not transports:
            raise ValueError("transport_by_direction must not be empty")
        if not all(isinstance(value, FusionSCTransportSpec) for value in transports.values()):
            raise TypeError("transport_by_direction values must be FusionSCTransportSpec")
        if static_allocation is not None:
            missing = set(static_allocation) - set(transports)
            if missing:
                directions = ", ".join(sorted(missing))
                raise ValueError(
                    "missing FusionSC transport mapping for phi direction(s): "
                    f"{directions}"
                )

        provenance: dict[str, dict[str, object]] = {}
        for direction, value in dict(self.transport_provenance_by_direction).items():
            if direction not in {"+", "-"}:
                raise ValueError(
                    "transport_provenance_by_direction keys must be explicit '+' or '-'"
                )
            if direction not in transports:
                raise ValueError("transport provenance requires a matching transport spec")
            if not isinstance(value, Mapping):
                raise TypeError("transport provenance entries must be mappings")
            provenance[direction] = dict(value)

        total = (
            None
            if self.total_power is None
            else _finite_positive(self.total_power, "total_power")
        )
        n_phi = None if self.n_phi_bins is None else _nonnegative_integer(
            self.n_phi_bins, "n_phi_bins"
        )
        n_s = _nonnegative_integer(self.n_s_bins, "n_s_bins")
        if n_phi is not None and n_phi < 1:
            raise ValueError("n_phi_bins must be at least one")
        if n_s < 1:
            raise ValueError("n_s_bins must be at least one")
        period = None if self.field_period is None else _finite_positive(
            self.field_period, "field_period"
        )
        minimum_fraction = float(self.minimum_collision_fraction)
        if not np.isfinite(minimum_fraction) or not 0.0 < minimum_fraction <= 1.0:
            raise ValueError("minimum_collision_fraction must lie in (0, 1]")
        if self.trace_function is not None and not callable(self.trace_function):
            raise TypeError("trace_function must be callable")

        object.__setattr__(self, "transport_by_direction", transports)
        object.__setattr__(self, "transport_provenance_by_direction", provenance)
        object.__setattr__(self, "total_power", total)
        object.__setattr__(self, "n_phi_bins", n_phi)
        object.__setattr__(self, "n_s_bins", n_s)
        object.__setattr__(self, "field_period", period)
        object.__setattr__(self, "minimum_collision_fraction", minimum_fraction)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def _resolve_seed_bundles(
        self,
        case: "BoundaryTopologyCase",
        request: "BoundaryPlasmaResponseInput",
        spectrum: "RadialPerturbationFourierSpectrum",
        chains: Sequence["ResonantIslandChain"],
        intervals: Sequence["ChaoticLayerInterval"],
    ) -> tuple[StrikeSeedBundle, ...]:
        value = self.seed_bundles
        if callable(value):
            value = value(case, request, spectrum, chains, intervals)
        return _validated_topology_seed_bundles(value)

    def evaluate(
        self,
        case: "BoundaryTopologyCase",
        request: "BoundaryPlasmaResponseInput",
        spectrum: "RadialPerturbationFourierSpectrum",
        chains: Sequence["ResonantIslandChain"],
        intervals: Sequence["ChaoticLayerInterval"],
    ) -> BoundaryTopologyHeatState:
        bundles = self._resolve_seed_bundles(case, request, spectrum, chains, intervals)
        weight_kind, launched_power, allocation = _directional_power_allocation(
            bundles, self.total_power
        )
        missing = set(allocation) - set(self.transport_by_direction)
        if missing:
            directions = ", ".join(sorted(missing))
            raise ValueError(
                "missing FusionSC transport mapping for phi direction(s): "
                f"{directions}"
            )
        resolved_wall = _resolve_spec(
            self.wall,
            case,
            request,
            FusionSCWallSurfaceSpec,
            "wall",
        )
        resolved_field = _coerce_computed_field(self.field_builder(case, request))

        def resolved_field_builder(_case: object, _request: object) -> FusionSCComputedField:
            return resolved_field

        states: list[BoundaryTopologyHeatState] = []
        component_provenance: dict[str, dict[str, object]] = {}
        topology_geometry_quantitative = all(
            bool(
                bundle.metadata.get(
                    "topology_quantitative", bundle.metadata.get("quantitative", False)
                )
            )
            for bundle in bundles
        )
        topology_weights_quantitative = all(
            bool(
                bundle.metadata.get(
                    "weight_quantitative", bundle.metadata.get("quantitative", False)
                )
            )
            for bundle in bundles
        )
        topology_provenance = tuple(
            {
                "label": bundle.label,
                "mode": bundle.mode,
                "phi_direction": bundle.direction,
                "seed_count": int(bundle.R.size),
                "topology_provenance": bundle.metadata.get(
                    "topology_provenance", "caller_supplied"
                ),
                "weight_provenance": bundle.metadata.get(
                    "weight_provenance", "caller_supplied"
                ),
                "topology_quantitative": bool(
                    bundle.metadata.get(
                        "topology_quantitative", bundle.metadata.get("quantitative", False)
                    )
                ),
                "weight_quantitative": bool(
                    bundle.metadata.get(
                        "weight_quantitative", bundle.metadata.get("quantitative", False)
                    )
                ),
            }
            for bundle in bundles
        )
        transport_quantitative = True

        for direction in ("+", "-"):
            if direction not in allocation:
                continue
            direction_bundles = tuple(
                bundle for bundle in bundles if bundle.direction == direction
            )
            transport = self.transport_by_direction[direction]
            transport_metadata = dict(
                self.transport_provenance_by_direction.get(direction, {})
            )
            direction_transport_quantitative = bool(
                transport_metadata.pop("quantitative", False)
            )
            transport_provenance = str(
                transport_metadata.pop("provenance", "caller_supplied_transport_spec")
            ).strip()
            if not transport_provenance:
                raise ValueError("transport provenance must not be empty")
            transport_quantitative = (
                transport_quantitative and direction_transport_quantitative
            )
            component_quantitative = all(
                bool(
                    bundle.metadata.get(
                        "topology_quantitative", bundle.metadata.get("quantitative", False)
                    )
                )
                and bool(
                    bundle.metadata.get(
                        "weight_quantitative", bundle.metadata.get("quantitative", False)
                    )
                )
                for bundle in direction_bundles
            ) and direction_transport_quantitative
            labels = tuple(bundle.label for bundle in direction_bundles)
            component_metadata = {
                "topology_guided_component": True,
                "phi_direction": direction,
                "topology_source_labels": labels,
                "topology_modes": tuple(
                    dict.fromkeys(bundle.mode for bundle in direction_bundles)
                ),
                "topology_weight_kind": weight_kind,
                "transport_provenance": transport_provenance,
                "transport_provenance_quantitative": direction_transport_quantitative,
                "quantitative": component_quantitative,
            }
            model = FusionSCFieldLineDiffusionHeatModel(
                field_builder=resolved_field_builder,
                wall=resolved_wall,
                seeds=_topology_seed_spec(direction_bundles),
                transport=transport,
                total_power=allocation[direction],
                n_phi_bins=self.n_phi_bins,
                n_s_bins=self.n_s_bins,
                field_period=self.field_period,
                minimum_collision_fraction=self.minimum_collision_fraction,
                renormalize_deposited_power=False,
                trace_function=self.trace_function,
                metadata=component_metadata,
            )
            state = model.evaluate(case, request, spectrum, chains, intervals)
            component_launched, component_deposited, component_unresolved = (
                _heat_state_power_accounting(
                    state,
                    expected_launched_power=allocation[direction],
                )
            )
            perpendicular_name, perpendicular_value = transport.perpendicular_model
            parallel_name, parallel_value = transport.parallel_model
            component_provenance[direction] = {
                **transport_metadata,
                "source_labels": labels,
                "seed_count": int(sum(bundle.R.size for bundle in direction_bundles)),
                "allocated_power": float(allocation[direction]),
                "launched_power": component_launched,
                "deposited_power": component_deposited,
                "unresolved_power": component_unresolved,
                "fusionsc_trace_direction": transport.direction,
                "perpendicular_model": perpendicular_name,
                "perpendicular_coefficient": perpendicular_value,
                "parallel_model": parallel_name,
                "parallel_coefficient": parallel_value,
                "provenance": transport_provenance,
                "quantitative": direction_transport_quantitative,
            }
            states.append(state)

        combined = sum_boundary_heat_states(states)
        combined_launched, combined_deposited, combined_unresolved = (
            _heat_state_power_accounting(
                combined,
                expected_launched_power=launched_power,
            )
        )
        quantitative = (
            topology_geometry_quantitative
            and topology_weights_quantitative
            and transport_quantitative
        )
        metadata = dict(self.metadata)
        metadata.update(combined.metadata)
        metadata.update(
            {
                "model": "fusionsc_topology_guided_heat",
                "backend": "fusionsc.flt.trace",
                "topology_guided": True,
                "topology_seed_bundle_count": len(bundles),
                "topology_seed_count": int(sum(bundle.R.size for bundle in bundles)),
                "topology_source_labels": tuple(bundle.label for bundle in bundles),
                "topology_seed_provenance": topology_provenance,
                "topology_modes": tuple(dict.fromkeys(bundle.mode for bundle in bundles)),
                "topology_weight_kind": weight_kind,
                "topology_geometry_quantitative": topology_geometry_quantitative,
                "topology_weights_quantitative": topology_weights_quantitative,
                "transport_quantitative": transport_quantitative,
                "quantitative_transport": transport_quantitative,
                "quantitative": quantitative,
                "proxy": not quantitative,
                "phi_direction_dispatch": component_provenance,
                "launched_power": combined_launched,
                "deposited_power": combined_deposited,
                "unresolved_power": combined_unresolved,
                "resolved_power_fraction": combined_deposited / combined_launched,
                "wall_collision_power_fraction": combined_deposited / combined_launched,
                "normalization": "sum(heat * cell_areas) + unresolved_power == launched_power",
            }
        )
        return BoundaryTopologyHeatState(
            heat=combined.heat,
            phi_values=combined.phi_values,
            s_values=combined.s_values,
            cell_areas=combined.cell_areas,
            metadata=metadata,
        )


@dataclass(frozen=True)
class FusionSCEnsembleHeatModel:
    """Average repeated stochastic FusionSC heat evaluations.

    FusionSC's diffusive tracer does not currently expose a random seed.  A
    single trace must therefore not be treated as a deterministic response
    column.  This wrapper accepts either the direct or topology-guided
    diffusion model, evaluates it repeatedly on the same heat grid, and
    retains standard errors and power-accounting uncertainty in metadata.
    """

    model: FusionSCFieldLineDiffusionHeatModel | FusionSCTopologyGuidedHeatModel
    repeats: int = 5
    minimum_successful_repeats: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(
            self.model,
            (FusionSCFieldLineDiffusionHeatModel, FusionSCTopologyGuidedHeatModel),
        ):
            raise TypeError(
                "model must be FusionSCFieldLineDiffusionHeatModel or "
                "FusionSCTopologyGuidedHeatModel"
            )
        repeats = _nonnegative_integer(self.repeats, "repeats")
        if repeats < 2:
            raise ValueError("repeats must be at least two to estimate stochastic uncertainty")
        minimum = repeats if self.minimum_successful_repeats is None else _nonnegative_integer(
            self.minimum_successful_repeats,
            "minimum_successful_repeats",
        )
        if minimum < 2 or minimum > repeats:
            raise ValueError("minimum_successful_repeats must lie in [2, repeats]")
        object.__setattr__(self, "repeats", repeats)
        object.__setattr__(self, "minimum_successful_repeats", minimum)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def evaluate(
        self,
        case: "BoundaryTopologyCase",
        request: "BoundaryPlasmaResponseInput",
        spectrum: "RadialPerturbationFourierSpectrum",
        chains: Sequence["ResonantIslandChain"],
        intervals: Sequence["ChaoticLayerInterval"],
    ) -> BoundaryTopologyHeatState:
        states: list[BoundaryTopologyHeatState] = []
        failures: dict[str, int] = {}
        for _repeat in range(self.repeats):
            try:
                states.append(self.model.evaluate(case, request, spectrum, chains, intervals))
            except FusionSCTraceError as exc:
                name = type(exc).__name__
                failures[name] = failures.get(name, 0) + 1
        if len(states) < int(self.minimum_successful_repeats):
            details = ", ".join(f"{name}={count}" for name, count in sorted(failures.items()))
            suffix = "" if not details else f"; failures: {details}"
            raise FusionSCTraceError(
                "FusionSC ensemble produced "
                f"{len(states)}/{self.repeats} successful traces, below "
                f"minimum_successful_repeats={self.minimum_successful_repeats}{suffix}"
            )

        reference = states[0]
        reference_area = None if reference.cell_areas is None else np.asarray(reference.cell_areas, dtype=float)
        for state in states[1:]:
            if state.heat.shape != reference.heat.shape:
                raise FusionSCTraceError("FusionSC ensemble heat grids have different shapes")
            if not np.allclose(state.phi_values, reference.phi_values, rtol=0.0, atol=1.0e-12):
                raise FusionSCTraceError("FusionSC ensemble phi grids do not match")
            if not np.allclose(state.s_values, reference.s_values, rtol=0.0, atol=1.0e-12):
                raise FusionSCTraceError("FusionSC ensemble wall-coordinate grids do not match")
            area = None if state.cell_areas is None else np.asarray(state.cell_areas, dtype=float)
            if (reference_area is None) != (area is None):
                raise FusionSCTraceError("FusionSC ensemble cell-area availability does not match")
            if reference_area is not None and not np.allclose(area, reference_area, rtol=1.0e-12, atol=1.0e-15):
                raise FusionSCTraceError("FusionSC ensemble cell areas do not match")

        heat_stack = np.stack([np.asarray(state.heat, dtype=float) for state in states])
        heat_mean = np.mean(heat_stack, axis=0)
        heat_std = np.std(heat_stack, axis=0, ddof=1)
        heat_sem = heat_std / np.sqrt(float(len(states)))
        if reference_area is None:
            deposited = np.sum(heat_stack, axis=(1, 2))
        else:
            deposited = np.sum(heat_stack * reference_area[None, :, :], axis=(1, 2))

        def _metadata_samples(name: str) -> np.ndarray:
            values = [dict(state.metadata).get(name, np.nan) for state in states]
            try:
                return np.asarray(values, dtype=float)
            except (TypeError, ValueError):
                return np.full(len(states), np.nan, dtype=float)

        collision_fraction = _metadata_samples("wall_collision_power_fraction")
        unresolved = _metadata_samples("unresolved_power")
        launched = _metadata_samples("launched_power")
        md = dict(reference.metadata)
        ensemble_model = (
            "fusionsc_topology_guided_heat_ensemble"
            if isinstance(self.model, FusionSCTopologyGuidedHeatModel)
            else "fusionsc_field_line_diffusion_ensemble"
        )
        md.update(self.metadata)
        md.update(
            {
                "model": ensemble_model,
                "actual_diffusive_trace": True,
                "ballistic_fallback": False,
                "stochastic_uncertainty_estimated": True,
                "ensemble_repeats_requested": int(self.repeats),
                "ensemble_repeats_successful": int(len(states)),
                "ensemble_repeats_failed": int(self.repeats - len(states)),
                "ensemble_failure_counts": failures,
                "minimum_successful_repeats": int(self.minimum_successful_repeats),
                "heat_flux_standard_deviation": heat_std,
                "heat_flux_standard_error": heat_sem,
                "deposited_power": float(np.mean(deposited)),
                "deposited_power_standard_deviation": float(np.std(deposited, ddof=1)),
                "deposited_power_standard_error": float(
                    np.std(deposited, ddof=1) / np.sqrt(float(len(states)))
                ),
                "wall_collision_power_fraction": float(np.nanmean(collision_fraction)),
                "wall_collision_power_fraction_standard_deviation": float(
                    np.nanstd(collision_fraction, ddof=1)
                ),
                "unresolved_power": float(np.nanmean(unresolved)),
                "unresolved_power_standard_deviation": float(np.nanstd(unresolved, ddof=1)),
                "launched_power": float(np.nanmean(launched)),
                "normalization": "sum(mean_heat * cell_areas) == mean deposited_power",
            }
        )
        return BoundaryTopologyHeatState(
            heat=heat_mean,
            phi_values=reference.phi_values,
            s_values=reference.s_values,
            cell_areas=reference_area,
            metadata=md,
        )


FusionSCFieldLineDiffusionModel = FusionSCFieldLineDiffusionHeatModel
FusionSCWallSurface = FusionSCWallSurfaceSpec


__all__ = [
    "FusionSCBackendUnavailableError",
    "FusionSCComputedField",
    "FusionSCEnsembleHeatModel",
    "FusionSCFieldLineDiffusionHeatModel",
    "FusionSCFieldLineDiffusionModel",
    "FusionSCSeedSpec",
    "FusionSCTopologyGuidedHeatModel",
    "FusionSCTraceError",
    "FusionSCTransportSpec",
    "FusionSCWallSurface",
    "FusionSCWallSurfaceSpec",
    "fusionsc_computed_field_from_cylindrical",
    "load_fusionsc_rz_section_wall",
]
