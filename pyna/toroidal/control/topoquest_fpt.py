"""Optional Topoquest FPT beta-ramp adapter for boundary screening.

The public contracts in this module depend only on pyna and NumPy.  Imports of
Topoquest, PETSc, and DOLFINx are deferred until the concrete field-period
runner is called, so fake runners and cached response bases remain usable in a
minimal pyna installation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from pyna.fields.periodicity import ToroidalPeriodicity
from pyna.toroidal.control.boundary_perturbation_candidates import (
    sample_perturbation_candidate_on_surfaces,
)
from pyna.toroidal.control.boundary_plasma_response import (
    BoundaryPlasmaResponseInput,
    CorePreservationSnapshot,
)
from pyna.toroidal.control.boundary_topology_cases import (
    BoundaryTopologyCase,
    BoundaryTopologyPlasmaFeedback,
)
from pyna.toroidal.perturbation_spectrum import nardon_radial_perturbation


_PATH_KEY_PARTS = ("path", "file", "dir", "root", "screenshot", "image", "figure")
_GENERATION_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _optional_generation_id(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    generation_id = str(value).strip()
    if not _GENERATION_ID_RE.fullmatch(generation_id):
        raise ValueError(f"{label} must be a full lowercase sha256 generation ID")
    return generation_id


def _generation_id_and_metadata(
    generation_id: Any,
    metadata: Mapping[str, Any] | None,
    *,
    label: str,
) -> tuple[str | None, dict[str, Any]]:
    """Resolve one canonical generation ID from a field and its metadata."""

    public = _public_metadata(metadata)
    explicit = _optional_generation_id(generation_id, label=f"{label}.generation_id")
    metadata_value = _optional_generation_id(
        public.get("generation_id"),
        label=f"{label}.metadata['generation_id']",
    )
    if explicit is not None and metadata_value is not None and explicit != metadata_value:
        raise ValueError(f"{label} generation_id disagrees with metadata generation_id")
    resolved = explicit if explicit is not None else metadata_value
    if resolved is not None:
        public["generation_id"] = resolved
    return resolved, public


def _live_result_generation_id(
    spec: "TopoquestFPTBetaRampSpec",
    *sources: Any,
    label: str,
) -> str | None:
    """Read and verify the generation echoed by a concrete Topoquest runner."""

    candidates: list[str] = []
    for index, source in enumerate(sources):
        if source is None:
            continue
        if isinstance(source, Mapping):
            direct = source.get("generation_id")
            metadata = source.get("metadata")
        else:
            direct = getattr(source, "generation_id", None)
            metadata = getattr(source, "metadata", None)
        direct_id = _optional_generation_id(
            direct,
            label=f"{label}.source[{index}].generation_id",
        )
        if direct_id is not None:
            candidates.append(direct_id)
        if isinstance(metadata, Mapping):
            metadata_id = _optional_generation_id(
                metadata.get("generation_id"),
                label=f"{label}.source[{index}].metadata['generation_id']",
            )
            if metadata_id is not None:
                candidates.append(metadata_id)
    unique = tuple(dict.fromkeys(candidates))
    if len(unique) > 1:
        raise ValueError(f"{label} returned conflicting generation IDs")
    returned = None if not unique else unique[0]
    if spec.generation_id is not None:
        if returned is None:
            raise RuntimeError(
                f"{label} did not echo the requested generation_id"
            )
        if returned != spec.generation_id:
            raise ValueError(
                f"{label} generation does not match the requested generation"
            )
    return returned


def _project_delta_field_on_case(field, case: BoundaryTopologyCase) -> np.ndarray:
    delta_BR, delta_BZ, delta_BPhi = sample_perturbation_candidate_on_surfaces(
        field,
        case.R_surf,
        case.Z_surf,
        case.phi_vals,
    )
    return np.asarray(
        nardon_radial_perturbation(
            case.R_surf,
            case.Z_surf,
            case.phi_vals,
            case.theta_vals,
            delta_BR,
            delta_BZ,
            delta_BPhi,
            case.radial_labels,
            denominator_B3=case.denominator_B3,
        ),
        dtype=complex,
    )


def _public_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Copy metadata while redacting path-like values from public feedback."""

    public: dict[str, Any] = {}
    for key, value in dict(metadata or {}).items():
        key_s = str(key)
        if any(part in key_s.lower() for part in _PATH_KEY_PARTS):
            public[key_s] = "<redacted>"
        else:
            public[key_s] = value
    return public


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


@dataclass(frozen=True)
class TopoquestFPTCapability:
    """Availability report for the concrete Topoquest FPT PDE runner."""

    topoquest_available: bool
    petsc4py_available: bool
    dolfinx_available: bool

    @property
    def missing_dependencies(self) -> tuple[str, ...]:
        rows = (
            ("topoquest", self.topoquest_available),
            ("petsc4py", self.petsc4py_available),
            ("dolfinx", self.dolfinx_available),
        )
        return tuple(name for name, available in rows if not available)

    @property
    def available(self) -> bool:
        """Whether the full PDE runner dependency stack is importable."""

        return not self.missing_dependencies

    @property
    def current_process_available(self) -> bool:
        """Whether the current interpreter can call the in-process runner."""

        return self.available

    @property
    def message(self) -> str:
        """Return a diagnostic suitable for logs or user-facing errors."""

        if self.available:
            return "Topoquest FPT PDE backend is available"
        missing = ", ".join(self.missing_dependencies)
        return (
            "Topoquest FPT PDE backend is unavailable; missing optional "
            f"dependencies: {missing}. No scalar-factor fallback is used."
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly capability report."""

        return {
            "available": self.available,
            "topoquest_available": bool(self.topoquest_available),
            "petsc4py_available": bool(self.petsc4py_available),
            "dolfinx_available": bool(self.dolfinx_available),
            "missing_dependencies": self.missing_dependencies,
            "message": self.message,
        }


@dataclass(frozen=True)
class TopoquestNeoclassicalFPTCapability:
    """Availability report for rectangular FPT plus 3-D transport closure."""

    fpt: TopoquestFPTCapability
    aletheia_available: bool

    @property
    def missing_dependencies(self) -> tuple[str, ...]:
        missing = list(self.fpt.missing_dependencies)
        if not self.aletheia_available:
            missing.append("aletheia")
        return tuple(missing)

    @property
    def available(self) -> bool:
        return not self.missing_dependencies

    @property
    def current_process_available(self) -> bool:
        """Whether the current interpreter can call the in-process runner."""

        return self.available

    @property
    def message(self) -> str:
        if self.available:
            return "Topoquest rectangular FPT with 3-D neoclassical closure is available"
        return (
            "Topoquest neoclassical FPT backend is unavailable; missing optional "
            f"dependencies: {', '.join(self.missing_dependencies)}. "
            "No vacuum or scalar-screening fallback is used."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.fpt.as_dict(),
            "available": self.available,
            "aletheia_available": bool(self.aletheia_available),
            "missing_dependencies": self.missing_dependencies,
            "message": self.message,
        }


class TopoquestFPTUnavailableError(ImportError):
    """Raised when the concrete PDE runner's optional stack is unavailable."""


@dataclass(frozen=True)
class TopoquestFPTExternalRuntime:
    """Result of probing an isolated FEniCSx interpreter.

    External availability never changes the in-process capability reports or
    permits an in-process runner call.  It only proves that a separately
    launched interpreter can import the production stack and PETSc CUDA.
    """

    runtime_alias: str
    source: str
    python_available: bool
    probe_succeeded: bool
    module_availability: Mapping[str, bool]
    production_entrypoint_available: bool
    neoclassical_entrypoint_available: bool | None
    petsc_cuda_available: bool | None
    require_neoclassical: bool
    versions: Mapping[str, str | None] = field(default_factory=dict)
    error_type: str | None = None

    def __post_init__(self) -> None:
        alias = str(self.runtime_alias).strip()
        if not alias:
            raise ValueError("runtime_alias must be non-empty")
        object.__setattr__(self, "runtime_alias", alias)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(
            self,
            "module_availability",
            {str(name): bool(value) for name, value in self.module_availability.items()},
        )
        object.__setattr__(
            self,
            "versions",
            {
                str(name): None if value is None else str(value)
                for name, value in self.versions.items()
            },
        )

    @property
    def required_modules(self) -> tuple[str, ...]:
        modules = ["topoquest", "petsc4py", "dolfinx", "ufl", "basix"]
        if self.require_neoclassical:
            modules.append("aletheia")
        return tuple(modules)

    @property
    def missing_dependencies(self) -> tuple[str, ...]:
        missing = [
            name
            for name in self.required_modules
            if not self.module_availability.get(name, False)
        ]
        if self.probe_succeeded and not bool(self.petsc_cuda_available):
            missing.append("petsc_cuda")
        if self.probe_succeeded and not self.production_entrypoint_available:
            missing.append("topoquest_fpt_entrypoint")
        if (
            self.require_neoclassical
            and self.probe_succeeded
            and not bool(self.neoclassical_entrypoint_available)
        ):
            missing.append("topoquest_neoclassical_entrypoint")
        return tuple(missing)

    @property
    def available(self) -> bool:
        return bool(
            self.python_available
            and self.probe_succeeded
            and not self.missing_dependencies
        )

    @property
    def message(self) -> str:
        if self.available:
            return f"{self.runtime_alias} is available with PETSc CUDA"
        if not self.python_available:
            return f"{self.runtime_alias} Python executable is unavailable"
        if not self.probe_succeeded:
            suffix = "" if self.error_type is None else f" ({self.error_type})"
            return f"{self.runtime_alias} probe failed{suffix}"
        return (
            f"{self.runtime_alias} is incomplete; missing: "
            f"{', '.join(self.missing_dependencies)}"
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a path-free external-runtime report."""

        return {
            "available": self.available,
            "runtime_alias": self.runtime_alias,
            "source": self.source,
            "python_available": bool(self.python_available),
            "probe_succeeded": bool(self.probe_succeeded),
            "module_availability": dict(self.module_availability),
            "production_entrypoint_available": bool(
                self.production_entrypoint_available
            ),
            "neoclassical_entrypoint_available": (
                None
                if self.neoclassical_entrypoint_available is None
                else bool(self.neoclassical_entrypoint_available)
            ),
            "petsc_cuda_available": (
                None
                if self.petsc_cuda_available is None
                else bool(self.petsc_cuda_available)
            ),
            "require_neoclassical": bool(self.require_neoclassical),
            "versions": dict(self.versions),
            "missing_dependencies": self.missing_dependencies,
            "error_type": self.error_type,
            "message": self.message,
        }


_EXTERNAL_RUNTIME_PROBE = r"""
import importlib
import json

names = ("topoquest", "petsc4py", "dolfinx", "ufl", "basix", "aletheia")
modules = {}
versions = {}
for name in names:
    try:
        module = importlib.import_module(name)
    except Exception:
        modules[name] = False
        versions[name] = None
    else:
        modules[name] = True
        versions[name] = getattr(module, "__version__", None)

production_entrypoint = False
neoclassical_entrypoint = False
try:
    from topoquest.solvers.fem_fpt import run_rectangular_fem_fpt_msh_payloads
    production_entrypoint = callable(run_rectangular_fem_fpt_msh_payloads)
except Exception:
    pass
try:
    from topoquest.solvers.fem_fpt import build_neoclassical_parallel_current_closure_factory
    neoclassical_entrypoint = callable(build_neoclassical_parallel_current_closure_factory)
except Exception:
    pass

petsc_cuda = False
if modules["petsc4py"]:
    try:
        import petsc4py
        petsc4py.init(["-use_gpu_aware_mpi", "0"])
        from petsc4py import PETSc
        petsc_cuda = bool(PETSc.Sys.hasExternalPackage("cuda"))
        versions["petsc"] = ".".join(str(value) for value in PETSc.Sys.getVersion())
    except Exception:
        petsc_cuda = False

print(json.dumps({
    "modules": modules,
    "versions": versions,
    "production_entrypoint": production_entrypoint,
    "neoclassical_entrypoint": neoclassical_entrypoint,
    "petsc_cuda": petsc_cuda,
}, sort_keys=True))
"""


def _prepend_environment_path(
    environment: dict[str, str],
    name: str,
    entries: Sequence[Path],
) -> None:
    values = [str(entry) for entry in entries if entry.exists()]
    existing = environment.get(name, "")
    if existing:
        values.append(existing)
    if values:
        environment[name] = os.pathsep.join(values)


def _external_runtime_environment(prefix: Path | None) -> dict[str, str]:
    environment = dict(os.environ)
    if prefix is None:
        return environment
    petsc = prefix / "petsc"
    cuda = Path(environment.get("CUDA_HOME", "/usr/local/cuda")).expanduser()
    environment["FENICSX_CUDA_PREFIX"] = str(prefix)
    environment["PETSC_DIR"] = str(petsc)
    environment.pop("PETSC_ARCH", None)
    environment.setdefault("CUDA_HOME", str(cuda))
    _prepend_environment_path(
        environment,
        "PATH",
        (petsc / "bin", cuda / "bin", prefix / "venv" / "bin"),
    )
    _prepend_environment_path(
        environment,
        "LD_LIBRARY_PATH",
        (
            prefix / "lib",
            petsc / "lib",
            cuda / "lib64",
            Path("/usr/lib/wsl/lib"),
        ),
    )
    _prepend_environment_path(environment, "PYTHONPATH", (petsc / "lib",))
    options = environment.get("PETSC_OPTIONS", "")
    if "-use_gpu_aware_mpi" not in options:
        environment["PETSC_OPTIONS"] = f"{options} -use_gpu_aware_mpi 0".strip()
    return environment


def _resolved_external_runtime(
    python_executable: str | Path | None,
    runtime_prefix: str | Path | None,
) -> tuple[str | None, Path | None, str]:
    explicit_prefix = (
        None if runtime_prefix is None else Path(runtime_prefix).expanduser()
    )
    if python_executable is not None:
        candidate = str(Path(python_executable).expanduser())
        source = "explicit_python"
        prefix = explicit_prefix
    elif os.environ.get("TOPOQUEST_FENICSX_PYTHON"):
        candidate = os.environ["TOPOQUEST_FENICSX_PYTHON"]
        source = "environment_python"
        prefix = explicit_prefix
    else:
        if explicit_prefix is not None:
            prefix = explicit_prefix
            source = "explicit_prefix"
        elif os.environ.get("FENICSX_CUDA_PREFIX"):
            prefix = Path(os.environ["FENICSX_CUDA_PREFIX"]).expanduser()
            source = "environment_prefix"
        else:
            prefix = Path("/opt/fenicsx-cuda")
            source = "default_prefix"
        candidates = (prefix / "venv" / "bin" / "python3", prefix / "venv" / "bin" / "python")
        candidate = str(next((value for value in candidates if value.exists()), candidates[0]))

    path = Path(candidate).expanduser()
    if path.is_file() and os.access(path, os.X_OK):
        resolved = str(path)
    elif os.sep not in candidate:
        resolved = shutil.which(candidate)
    else:
        resolved = None

    if prefix is None and resolved is not None:
        executable_path = Path(resolved).expanduser()
        if executable_path.parent.name == "bin" and executable_path.parent.parent.name == "venv":
            prefix = executable_path.parent.parent.parent
    return resolved, prefix, source


def _failed_external_runtime(
    *,
    runtime_alias: str,
    source: str,
    python_available: bool,
    require_neoclassical: bool,
    error_type: str,
) -> TopoquestFPTExternalRuntime:
    return TopoquestFPTExternalRuntime(
        runtime_alias=runtime_alias,
        source=source,
        python_available=python_available,
        probe_succeeded=False,
        module_availability={},
        production_entrypoint_available=False,
        neoclassical_entrypoint_available=(False if require_neoclassical else None),
        petsc_cuda_available=None,
        require_neoclassical=require_neoclassical,
        versions={},
        error_type=error_type,
    )


def diagnose_topoquest_fpt_external_runtime(
    python_executable: str | Path | None = None,
    *,
    runtime_prefix: str | Path | None = None,
    runtime_alias: str = "configured FEniCSx runtime",
    require_neoclassical: bool = False,
    timeout: float = 30.0,
) -> TopoquestFPTExternalRuntime:
    """Probe a separate FEniCSx process without changing ``sys.path``.

    Resolution order is an explicit Python executable, the
    ``TOPOQUEST_FENICSX_PYTHON`` environment variable, then the venv under an
    explicit/ambient/default FEniCSx prefix.  Prefix-based probes reproduce
    the PETSc paths needed by source-built installations without sourcing a
    shell script.
    """

    timeout_f = float(timeout)
    if not np.isfinite(timeout_f) or timeout_f <= 0.0:
        raise ValueError("timeout must be positive and finite")
    resolved, prefix, source = _resolved_external_runtime(
        python_executable,
        runtime_prefix,
    )
    if resolved is None:
        return _failed_external_runtime(
            runtime_alias=runtime_alias,
            source=source,
            python_available=False,
            require_neoclassical=bool(require_neoclassical),
            error_type="FileNotFoundError",
        )
    try:
        completed = subprocess.run(
            [resolved, "-c", _EXTERNAL_RUNTIME_PROBE],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_f,
            env=_external_runtime_environment(prefix),
        )
    except subprocess.TimeoutExpired:
        return _failed_external_runtime(
            runtime_alias=runtime_alias,
            source=source,
            python_available=True,
            require_neoclassical=bool(require_neoclassical),
            error_type="TimeoutExpired",
        )
    except OSError as exc:
        return _failed_external_runtime(
            runtime_alias=runtime_alias,
            source=source,
            python_available=False,
            require_neoclassical=bool(require_neoclassical),
            error_type=type(exc).__name__,
        )
    if completed.returncode != 0:
        return _failed_external_runtime(
            runtime_alias=runtime_alias,
            source=source,
            python_available=True,
            require_neoclassical=bool(require_neoclassical),
            error_type="ExternalRuntimeProbeExit",
        )
    try:
        output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
        payload = json.loads(output_lines[-1])
        modules = dict(payload["modules"])
        versions = dict(payload["versions"])
        production_entrypoint = bool(payload["production_entrypoint"])
        neoclassical_entrypoint = bool(payload["neoclassical_entrypoint"])
        petsc_cuda = bool(payload["petsc_cuda"])
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return _failed_external_runtime(
            runtime_alias=runtime_alias,
            source=source,
            python_available=True,
            require_neoclassical=bool(require_neoclassical),
            error_type=type(exc).__name__,
        )
    return TopoquestFPTExternalRuntime(
        runtime_alias=runtime_alias,
        source=source,
        python_available=True,
        probe_succeeded=True,
        module_availability=modules,
        production_entrypoint_available=production_entrypoint,
        neoclassical_entrypoint_available=(
            neoclassical_entrypoint if require_neoclassical else None
        ),
        petsc_cuda_available=petsc_cuda,
        require_neoclassical=bool(require_neoclassical),
        versions=versions,
        error_type=None,
    )


def diagnose_topoquest_neoclassical_fpt_external_runtime(
    python_executable: str | Path | None = None,
    **kwargs: Any,
) -> TopoquestFPTExternalRuntime:
    """Probe a separate runtime including the Aletheia closure entrypoint."""

    return diagnose_topoquest_fpt_external_runtime(
        python_executable,
        require_neoclassical=True,
        **kwargs,
    )


def diagnose_topoquest_fpt_capability() -> TopoquestFPTCapability:
    """Inspect the optional PDE stack without importing Topoquest modules."""

    return TopoquestFPTCapability(
        topoquest_available=_module_available("topoquest"),
        petsc4py_available=_module_available("petsc4py"),
        dolfinx_available=_module_available("dolfinx"),
    )


def require_topoquest_fpt_capability() -> TopoquestFPTCapability:
    """Return capability details or raise with every missing PDE dependency."""

    capability = diagnose_topoquest_fpt_capability()
    if not capability.available:
        raise TopoquestFPTUnavailableError(capability.message)
    return capability


def diagnose_topoquest_neoclassical_fpt_capability() -> TopoquestNeoclassicalFPTCapability:
    """Inspect the PDE and Aletheia stacks without importing either solver."""

    return TopoquestNeoclassicalFPTCapability(
        fpt=diagnose_topoquest_fpt_capability(),
        aletheia_available=_module_available("aletheia"),
    )


def require_topoquest_neoclassical_fpt_capability() -> TopoquestNeoclassicalFPTCapability:
    capability = diagnose_topoquest_neoclassical_fpt_capability()
    if not capability.available:
        raise TopoquestFPTUnavailableError(capability.message)
    return capability


@dataclass(frozen=True)
class TopoquestFPTBetaRampSpec:
    """Validated beta levels and acceptance gates for one screening solve."""

    beta_values: Sequence[float]
    response_beta: float | None = None
    require_convergence: bool = True
    require_production_readiness: bool = False
    generation_id: str | None = None
    case_alias: str = "private stellarator"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        betas = tuple(float(value) for value in self.beta_values)
        if not betas:
            raise ValueError("beta_values must contain at least one beta level")
        beta_array = np.asarray(betas, dtype=float)
        if not np.all(np.isfinite(beta_array)) or np.any(beta_array < 0.0):
            raise ValueError("beta_values must be finite and non-negative")
        if len(betas) > 1 and np.any(np.diff(beta_array) <= 0.0):
            raise ValueError("beta_values must be strictly increasing")
        selected = betas[-1] if self.response_beta is None else float(self.response_beta)
        if not np.isfinite(selected) or selected < 0.0:
            raise ValueError("response_beta must be finite and non-negative")
        matches = np.flatnonzero(np.isclose(beta_array, selected, rtol=1.0e-12, atol=1.0e-15))
        if matches.size != 1:
            raise ValueError("response_beta must identify exactly one beta_values entry")
        alias = str(self.case_alias).strip()
        if not alias:
            raise ValueError("case_alias must be non-empty")
        if "/" in alias or "\\" in alias:
            raise ValueError("case_alias must be an alias, not a filesystem path")
        generation_id, metadata = _generation_id_and_metadata(
            self.generation_id,
            self.metadata,
            label="spec",
        )
        object.__setattr__(self, "beta_values", betas)
        object.__setattr__(self, "response_beta", betas[int(matches[0])])
        object.__setattr__(self, "case_alias", alias)
        object.__setattr__(self, "generation_id", generation_id)
        object.__setattr__(self, "metadata", metadata)

    @property
    def response_index(self) -> int:
        """Index of the beta level selected for boundary feedback."""

        return next(
            index
            for index, beta in enumerate(self.beta_values)
            if np.isclose(beta, self.response_beta, rtol=1.0e-12, atol=1.0e-15)
        )


@dataclass(frozen=True)
class TopoquestFPTVacuumControlState:
    """Current vacuum control state supplied to an FPT screening runner."""

    case: BoundaryTopologyCase
    request: BoundaryPlasmaResponseInput
    vacuum_tilde_b1: np.ndarray
    vacuum_delta_field: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.case, BoundaryTopologyCase):
            raise TypeError("case must be a BoundaryTopologyCase")
        if not isinstance(self.request, BoundaryPlasmaResponseInput):
            raise TypeError("request must be a BoundaryPlasmaResponseInput")
        tilde = np.asarray(self.vacuum_tilde_b1, dtype=complex)
        if tilde.shape != self.case.R_surf.shape:
            raise ValueError("vacuum_tilde_b1 must match the case surface stack")
        if not np.all(np.isfinite(tilde.real)) or not np.all(np.isfinite(tilde.imag)):
            raise ValueError("vacuum_tilde_b1 must be finite")
        object.__setattr__(self, "vacuum_tilde_b1", tilde)


@dataclass(frozen=True)
class TopoquestFPTBetaRampResult:
    """Selected plasma increment and FPT acceptance diagnostics.

    Exactly one of ``tilde_b1`` and ``delta_field`` is required.  ``tilde_b1``
    is the complex plasma-only Nardon perturbation.  ``delta_field`` must expose
    ``B_at(R, Z, phi)`` and is projected by the feedback adapter.
    """

    beta: float
    converged: bool
    production_ready: bool | None
    generation_id: str | None = None
    tilde_b1: np.ndarray | None = None
    delta_field: Any = None
    response_case: BoundaryTopologyCase | None = None
    background_field: Any = None
    equilibrium: Any = None
    core: CorePreservationSnapshot | None = None
    readiness: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    raw_result: Any = field(default=None, repr=False, compare=False)
    total_field: Any = None

    def __post_init__(self) -> None:
        beta = float(self.beta)
        if not np.isfinite(beta) or beta < 0.0:
            raise ValueError("beta must be finite and non-negative")
        if (self.tilde_b1 is None) == (self.delta_field is None):
            raise ValueError("exactly one of tilde_b1 and delta_field is required")
        tilde = None
        if self.tilde_b1 is not None:
            tilde = np.asarray(self.tilde_b1, dtype=complex)
            if tilde.ndim != 3 or tilde.size == 0:
                raise ValueError("tilde_b1 must be a non-empty 3-D surface stack")
            if not np.all(np.isfinite(tilde.real)) or not np.all(np.isfinite(tilde.imag)):
                raise ValueError("tilde_b1 must be finite")
        if self.delta_field is not None and not hasattr(self.delta_field, "B_at"):
            raise TypeError("delta_field must expose B_at(R, Z, phi)")
        if self.response_case is not None and not isinstance(self.response_case, BoundaryTopologyCase):
            raise TypeError("response_case must be a BoundaryTopologyCase")
        if tilde is not None and self.response_case is not None and tilde.shape != self.response_case.R_surf.shape:
            raise ValueError("tilde_b1 must match response_case surfaces")
        readiness = _public_metadata(self.readiness)
        ready = None if self.production_ready is None else bool(self.production_ready)
        if ready is None and "accepted" in readiness:
            ready = bool(readiness["accepted"])
        if ready is not None and "accepted" in readiness and bool(readiness["accepted"]) != ready:
            raise ValueError("production_ready disagrees with readiness['accepted']")
        if ready is not None:
            readiness.setdefault("accepted", ready)
        generation_id, metadata = _generation_id_and_metadata(
            self.generation_id,
            self.metadata,
            label="result",
        )
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "production_ready", ready)
        object.__setattr__(self, "generation_id", generation_id)
        object.__setattr__(self, "tilde_b1", tilde)
        object.__setattr__(self, "readiness", readiness)
        object.__setattr__(self, "metadata", metadata)

    @property
    def ready(self) -> bool | None:
        """Alias for the production-readiness decision."""

        return self.production_ready


@runtime_checkable
class TopoquestFPTBetaRampRunner(Protocol):
    """Callable protocol implemented by live and cached FPT response sources."""

    def __call__(
        self,
        spec: TopoquestFPTBetaRampSpec,
        state: TopoquestFPTVacuumControlState,
    ) -> TopoquestFPTBetaRampResult | Mapping[str, Any]: ...


def _coerce_runner_result(
    value: TopoquestFPTBetaRampResult | Mapping[str, Any],
) -> TopoquestFPTBetaRampResult:
    if isinstance(value, TopoquestFPTBetaRampResult):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("FPT runner must return TopoquestFPTBetaRampResult or a mapping")
    data = dict(value)
    aliases = {
        "plasma_tilde_b1": "tilde_b1",
        "plasma_delta_field": "delta_field",
        "cylindrical_delta_field": "delta_field",
        "authoritative_total_field": "total_field",
        "ready": "production_ready",
    }
    for source, target in aliases.items():
        if source in data:
            if target in data:
                raise ValueError(f"runner result supplies both {source!r} and {target!r}")
            data[target] = data.pop(source)
    return TopoquestFPTBetaRampResult(**data)


@dataclass(frozen=True)
class TopoquestFPTCachedResponseBasis:
    """Cached linear plasma-response columns usable as an FPT runner."""

    control_labels: Sequence[str]
    tilde_b1_basis: np.ndarray
    beta: float
    base_tilde_b1: np.ndarray | None = None
    response_case: BoundaryTopologyCase | None = None
    converged: bool = True
    production_ready: bool | None = None
    generation_id: str | None = None
    readiness: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        labels = tuple(str(label) for label in self.control_labels)
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("control_labels must be non-empty and unique")
        basis = np.asarray(self.tilde_b1_basis, dtype=complex)
        if basis.ndim != 4 or basis.shape[0] != len(labels):
            raise ValueError("tilde_b1_basis must have shape (n_controls, n_phi, n_radial, n_theta)")
        if not np.all(np.isfinite(basis.real)) or not np.all(np.isfinite(basis.imag)):
            raise ValueError("tilde_b1_basis must be finite")
        base = np.zeros(basis.shape[1:], dtype=complex) if self.base_tilde_b1 is None else np.asarray(
            self.base_tilde_b1,
            dtype=complex,
        )
        if base.shape != basis.shape[1:]:
            raise ValueError("base_tilde_b1 must match one cached response column")
        if not np.all(np.isfinite(base.real)) or not np.all(np.isfinite(base.imag)):
            raise ValueError("base_tilde_b1 must be finite")
        beta = float(self.beta)
        if not np.isfinite(beta) or beta < 0.0:
            raise ValueError("beta must be finite and non-negative")
        if self.response_case is not None:
            if not isinstance(self.response_case, BoundaryTopologyCase):
                raise TypeError("response_case must be a BoundaryTopologyCase")
            if self.response_case.R_surf.shape != basis.shape[1:]:
                raise ValueError("cached responses must match response_case surfaces")
        generation_id, metadata = _generation_id_and_metadata(
            self.generation_id,
            self.metadata,
            label="cached_basis",
        )
        object.__setattr__(self, "control_labels", labels)
        object.__setattr__(self, "tilde_b1_basis", basis)
        object.__setattr__(self, "base_tilde_b1", base)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "generation_id", generation_id)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "readiness", _public_metadata(self.readiness))

    def __call__(
        self,
        spec: TopoquestFPTBetaRampSpec,
        state: TopoquestFPTVacuumControlState,
    ) -> TopoquestFPTBetaRampResult:
        if tuple(state.request.control_labels) != tuple(self.control_labels):
            raise ValueError("request control_labels do not match the cached FPT response basis")
        if not np.isclose(spec.response_beta, self.beta, rtol=1.0e-12, atol=1.0e-15):
            raise ValueError("cached FPT response beta does not match spec.response_beta")
        if spec.generation_id is not None and self.generation_id != spec.generation_id:
            raise ValueError("cached FPT response generation does not match the requested generation")
        if spec.require_production_readiness and (
            spec.generation_id is None or self.generation_id is None
        ):
            raise RuntimeError(
                "production cached FPT responses require a bound generation_id"
            )
        response_case = state.case if self.response_case is None else self.response_case
        plasma_tilde = np.asarray(self.base_tilde_b1, dtype=complex) + np.tensordot(
            state.request.controls,
            self.tilde_b1_basis,
            axes=(0, 0),
        )
        return TopoquestFPTBetaRampResult(
            beta=self.beta,
            converged=self.converged,
            production_ready=self.production_ready,
            generation_id=self.generation_id,
            tilde_b1=plasma_tilde,
            response_case=response_case,
            readiness=self.readiness,
            metadata={"response_source": "cached_response_basis", **dict(self.metadata)},
        )


@dataclass(frozen=True)
class TopoquestFPTPlasmaFeedbackAdapter:
    """Adapt a live FPT runner or cached basis to plasma-feedback call shape."""

    spec: TopoquestFPTBetaRampSpec
    runner: TopoquestFPTBetaRampRunner | None = None
    response_basis: TopoquestFPTCachedResponseBasis | None = None
    vacuum_delta_field_factory: Callable[[BoundaryTopologyCase, BoundaryPlasmaResponseInput], Any] | None = None
    add_vacuum_tilde_b1: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, TopoquestFPTBetaRampSpec):
            raise TypeError("spec must be a TopoquestFPTBetaRampSpec")
        if (self.runner is None) == (self.response_basis is None):
            raise ValueError("provide exactly one of runner and response_basis")
        source = self.runner if self.runner is not None else self.response_basis
        if not callable(source):
            raise TypeError("FPT response source must be callable")
        if self.vacuum_delta_field_factory is not None and not callable(self.vacuum_delta_field_factory):
            raise TypeError("vacuum_delta_field_factory must be callable")
        object.__setattr__(self, "metadata", _public_metadata(self.metadata))

    def __call__(
        self,
        case: BoundaryTopologyCase,
        request: BoundaryPlasmaResponseInput,
        vacuum_tilde_b1: np.ndarray,
    ) -> BoundaryTopologyPlasmaFeedback:
        vacuum_field = request.vacuum_delta_field
        vacuum_field_source = (
            "request.vacuum_delta_field" if vacuum_field is not None else "unavailable"
        )
        if vacuum_field is None and self.vacuum_delta_field_factory is not None:
            vacuum_field = self.vacuum_delta_field_factory(case, request)
            vacuum_field_source = "adapter.vacuum_delta_field_factory"
        state = TopoquestFPTVacuumControlState(
            case=case,
            request=request,
            vacuum_tilde_b1=vacuum_tilde_b1,
            vacuum_delta_field=vacuum_field,
        )
        source = self.runner if self.runner is not None else self.response_basis
        result = _coerce_runner_result(source(self.spec, state))
        if not np.isclose(result.beta, self.spec.response_beta, rtol=1.0e-12, atol=1.0e-15):
            raise ValueError("FPT result beta does not match spec.response_beta")
        if self.spec.require_convergence and not result.converged:
            raise RuntimeError("Topoquest FPT beta-ramp response did not converge")
        if self.spec.require_production_readiness and result.production_ready is not True:
            raise RuntimeError("Topoquest FPT beta-ramp response is not production-ready")
        if (
            self.spec.generation_id is not None
            and result.generation_id != self.spec.generation_id
        ):
            raise RuntimeError(
                "Topoquest FPT beta-ramp response generation does not match the request"
            )
        if self.spec.require_production_readiness and (
            self.spec.generation_id is None or result.generation_id is None
        ):
            raise RuntimeError(
                "production Topoquest FPT responses require a bound generation_id"
            )

        response_case = case if result.response_case is None else result.response_case
        same_sampling_case = _same_nardon_sampling_case(case, response_case)
        representation = "tilde_b1"
        if result.tilde_b1 is not None:
            plasma_tilde = np.asarray(result.tilde_b1, dtype=complex)
        else:
            representation = "cylindrical_delta_field"
            plasma_tilde = _project_delta_field_on_case(result.delta_field, response_case)
        if plasma_tilde.shape != response_case.R_surf.shape:
            raise ValueError("FPT plasma tilde_b1 must match response_case surfaces")

        if self.add_vacuum_tilde_b1:
            vacuum_on_response = self._vacuum_on_response_case(state, response_case)
            feedback_tilde = vacuum_on_response + plasma_tilde
            spectrum_components = ("vacuum_delta_field", "plasma_delta_field")
        else:
            feedback_tilde = plasma_tilde
            spectrum_components = ("plasma_delta_field",)

        readiness = dict(result.readiness)
        if result.production_ready is not None:
            readiness.setdefault("accepted", bool(result.production_ready))
        md = _public_metadata(result.metadata)
        md.update(self.metadata)
        md.update(
            {
                "response_model": "topoquest_fpt_pde_beta_ramp",
                "response_source": "runner" if self.runner is not None else "cached_response_basis",
                "case_alias": self.spec.case_alias,
                "beta": float(result.beta),
                "beta_values": tuple(float(beta) for beta in self.spec.beta_values),
                "readiness": readiness,
                "production_ready": result.production_ready,
                "converged": bool(result.converged),
                "generation_id": result.generation_id,
                "plasma_response_representation": representation,
                "vacuum_response_representation": (
                    "not_included"
                    if not self.add_vacuum_tilde_b1
                    else (
                        "cached_tilde_b1_same_surface"
                        if same_sampling_case
                        else "cylindrical_delta_field_reprojected_on_response_case"
                    )
                ),
                "spectrum_delta_components": spectrum_components,
                "tilde_b1_semantics": "deltaB_over_B0_on_response_case",
            }
        )
        replacement_case = response_case is not case
        if result.background_field is not None:
            background_field = result.background_field
            background_source = "runner_result.background_field"
        elif replacement_case:
            background_field = response_case.background_field
            background_source = "response_case.background_field"
        else:
            background_field = request.baseline_field
            background_source = "request.baseline_field"
            if background_field is None:
                background_field = response_case.background_field
                background_source = "response_case.background_field"
        if result.equilibrium is not None:
            equilibrium = result.equilibrium
            equilibrium_source = "runner_result.equilibrium"
        elif replacement_case:
            equilibrium = response_case.equilibrium
            equilibrium_source = "response_case.equilibrium"
        else:
            equilibrium = request.baseline_equilibrium
            equilibrium_source = "request.baseline_equilibrium"
            if equilibrium is None:
                equilibrium = response_case.equilibrium
                equilibrium_source = "response_case.equilibrium"
        md["field_component_sources"] = {
            "background_field": background_source,
            "vacuum_delta_field": vacuum_field_source,
            "plasma_delta_field": (
                "runner_result.delta_field"
                if result.delta_field is not None
                else "unavailable_tilde_b1_representation"
            ),
            "total_field": (
                "runner_result.total_field"
                if result.total_field is not None
                else "unavailable"
            ),
        }
        md["response_context_sources"] = {
            "core": (
                "runner_result.core"
                if result.core is not None
                else "response_case.core_reference"
            ),
            "equilibrium": equilibrium_source,
        }
        return BoundaryTopologyPlasmaFeedback(
            tilde_b1=feedback_tilde,
            response_case=response_case,
            core=response_case.core_reference if result.core is None else result.core,
            background_field=background_field,
            equilibrium=equilibrium,
            metadata=md,
            vacuum_delta_field=vacuum_field,
            plasma_delta_field=result.delta_field,
            total_field=result.total_field,
        )

    @staticmethod
    def _vacuum_on_response_case(
        state: TopoquestFPTVacuumControlState,
        response_case: BoundaryTopologyCase,
    ) -> np.ndarray:
        if _same_nardon_sampling_case(state.case, response_case):
            return np.asarray(state.vacuum_tilde_b1, dtype=complex)
        if state.vacuum_delta_field is None:
            raise ValueError(
                "replacement response_case changes the Nardon sampling surfaces; "
                "a vacuum_delta_field is required for reprojection"
            )
        return _project_delta_field_on_case(state.vacuum_delta_field, response_case)


def _same_nardon_sampling_case(
    source_case: BoundaryTopologyCase,
    response_case: BoundaryTopologyCase,
) -> bool:
    """Whether cached vacuum samples remain valid on ``response_case``."""

    if source_case is response_case:
        return True
    source_signature = dict(source_case.metadata or {}).get("surface_signature")
    response_signature = dict(response_case.metadata or {}).get("surface_signature")
    if (
        source_signature is not None
        and response_signature is not None
        and source_signature != response_signature
    ):
        return False
    if (
        source_case.coordinate_system != response_case.coordinate_system
        or source_case.radial_coordinate != response_case.radial_coordinate
        or source_case.nfp != response_case.nfp
    ):
        return False
    values = (
        (source_case.R_surf, response_case.R_surf),
        (source_case.Z_surf, response_case.Z_surf),
        (source_case.phi_vals, response_case.phi_vals),
        (source_case.theta_vals, response_case.theta_vals),
        (source_case.radial_labels, response_case.radial_labels),
        (source_case.denominator_B3, response_case.denominator_B3),
    )
    return all(
        np.asarray(left).shape == np.asarray(right).shape
        and np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in values
    )


@dataclass(frozen=True)
class FPTSurfaceCylindricalDeltaField:
    """Cylindrical FPT samples tied to one boundary-case surface stack."""

    R_surf: np.ndarray
    Z_surf: np.ndarray
    phi_vals: np.ndarray
    delta_BR: np.ndarray
    delta_BZ: np.ndarray
    delta_BPhi: np.ndarray
    sampling_audit: Mapping[str, Any] = field(default_factory=dict)

    @property
    def sampling_scope(self) -> str:
        """Declare that these samples are not a global field-line trace field."""

        return "response_surface_stack"

    def __post_init__(self) -> None:
        R = np.asarray(self.R_surf, dtype=float)
        Z = np.asarray(self.Z_surf, dtype=float)
        phi = np.asarray(self.phi_vals, dtype=float).ravel()
        components = tuple(
            np.asarray(value, dtype=float)
            for value in (self.delta_BR, self.delta_BZ, self.delta_BPhi)
        )
        if R.ndim != 3 or Z.shape != R.shape or R.shape[0] != phi.size:
            raise ValueError("surface samples must have shape (n_phi, n_radial, n_theta)")
        if any(value.shape != R.shape for value in components):
            raise ValueError("cylindrical delta components must match the surface stack")
        if not all(np.all(np.isfinite(value)) for value in (R, Z, phi, *components)):
            raise ValueError("surface field samples must be finite")
        object.__setattr__(self, "R_surf", R)
        object.__setattr__(self, "Z_surf", Z)
        object.__setattr__(self, "phi_vals", phi)
        object.__setattr__(self, "delta_BR", components[0])
        object.__setattr__(self, "delta_BZ", components[1])
        object.__setattr__(self, "delta_BPhi", components[2])
        object.__setattr__(self, "sampling_audit", _public_metadata(self.sampling_audit))

    def B_at(self, R: Any, Z: Any, phi: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return samples when queried on the associated surface coordinates."""

        r = np.asarray(R, dtype=float)
        z = np.asarray(Z, dtype=float)
        p = np.asarray(phi, dtype=float)
        expected_phi = np.broadcast_to(self.phi_vals[:, None, None], self.R_surf.shape)
        if r.shape != self.R_surf.shape or z.shape != self.Z_surf.shape or p.shape != expected_phi.shape:
            raise ValueError("FPT surface field must be sampled on its complete surface stack")
        if not np.allclose(r, self.R_surf) or not np.allclose(z, self.Z_surf):
            raise ValueError("FPT surface field query geometry does not match its sampled surfaces")
        phase_error = np.angle(np.exp(1j * (p - expected_phi)))
        if not np.allclose(phase_error, 0.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("FPT surface field query phi values do not match its sampled sections")
        return self.delta_BR.copy(), self.delta_BZ.copy(), self.delta_BPhi.copy()


def _readiness_payload(audit: Any) -> tuple[bool | None, dict[str, Any]]:
    if audit is None:
        return None, {}
    if isinstance(audit, Mapping):
        payload = _public_metadata(audit)
        accepted = payload.get("accepted")
    else:
        as_dict = getattr(audit, "as_dict", None)
        payload = _public_metadata(as_dict() if callable(as_dict) else {})
        accepted = getattr(audit, "accepted", payload.get("accepted"))
        issues = getattr(audit, "issues", None)
        if issues is not None:
            payload.setdefault("issues", tuple(str(issue) for issue in issues))
    ready = None if accepted is None else bool(accepted)
    if ready is not None:
        payload["accepted"] = ready
    return ready, payload


def _field_sample_audit_rows(result: Any) -> tuple[dict[str, Any], ...]:
    rows = []
    for audit in tuple(getattr(result, "audits", ()) if result is not None else ()):
        rows.append(
            {
                "section_index": int(getattr(audit, "section_index")),
                "phi_rad": float(getattr(audit, "phi")),
                "n_points": int(getattr(audit, "n_points")),
                "point_source": str(getattr(audit, "point_source")),
            }
        )
    return tuple(rows)


def _periodic_section_interpolation(
    plan: Any,
    phi_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sections = tuple(getattr(plan, "sections", ()))
    if len(sections) < 2:
        raise ValueError("Topoquest plan must contain at least two toroidal sections")
    periodicity = ToroidalPeriodicity(nfp=getattr(plan, "n_fp", 1))
    period = periodicity.field_period
    section_phi = periodicity.wrap(
        np.asarray([float(section.phi) for section in sections], dtype=float)
    )
    order = np.argsort(section_phi)
    sorted_phi = section_phi[order]
    gaps = np.diff(np.concatenate((sorted_phi, [sorted_phi[0] + period])))
    if np.any(gaps <= 1.0e-12):
        raise ValueError("Topoquest plan section phases must be unique within a field period")
    target = periodicity.wrap(np.asarray(phi_vals, dtype=float).ravel())
    position = np.searchsorted(sorted_phi, target, side="right")
    lower_sorted = (position - 1) % sorted_phi.size
    upper_sorted = position % sorted_phi.size
    lower_phi = sorted_phi[lower_sorted].copy()
    upper_phi = sorted_phi[upper_sorted].copy()
    lower_phi[position == 0] -= period
    upper_phi[position == sorted_phi.size] += period
    span = upper_phi - lower_phi
    weight = (target - lower_phi) / span
    if not np.all(np.isfinite(weight)) or np.any(weight < -1.0e-12) or np.any(weight > 1.0 + 1.0e-12):
        raise ValueError("invalid periodic section interpolation weights")
    return order[lower_sorted], order[upper_sorted], np.clip(weight, 0.0, 1.0), span


def _section_interpolation_groups(
    lower: np.ndarray,
    upper: np.ndarray,
    weight: np.ndarray,
) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for phi_index, (lower_index, upper_index, upper_weight) in enumerate(
        zip(lower, upper, weight)
    ):
        groups.setdefault(int(lower_index), []).append(int(phi_index))
        if float(upper_weight) > 1.0e-14:
            groups.setdefault(int(upper_index), []).append(int(phi_index))
    return groups


def _surface_points(case: BoundaryTopologyCase, phi_indices: Sequence[int]) -> np.ndarray:
    return np.concatenate(
        [
            np.column_stack(
                [case.R_surf[index].reshape(-1), case.Z_surf[index].reshape(-1)]
            )
            for index in phi_indices
        ],
        axis=0,
    )


def _section_sample_blocks(
    samples: Any,
    phi_indices: Sequence[int],
    shape_2d: tuple[int, int],
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    count = int(np.prod(shape_2d))
    components = (
        np.asarray(samples.dB_R, dtype=float).reshape(-1),
        np.asarray(samples.dB_Z, dtype=float).reshape(-1),
        np.asarray(samples.dB_Phi, dtype=float).reshape(-1),
    )
    expected = count * len(phi_indices)
    if any(values.size != expected for values in components):
        raise ValueError("Topoquest solution sampler returned an unexpected number of values")
    blocks: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for offset, phi_index in enumerate(phi_indices):
        start = offset * count
        stop = start + count
        blocks[int(phi_index)] = tuple(
            component[start:stop].reshape(shape_2d)
            for component in components
        )
    return blocks


def _interpolated_surface_field(
    plan: Any,
    case: BoundaryTopologyCase,
    blocks_by_section: Mapping[
        int,
        Mapping[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ],
    lower: np.ndarray,
    upper: np.ndarray,
    weight: np.ndarray,
    span: np.ndarray,
) -> FPTSurfaceCylindricalDeltaField:
    components = [np.empty_like(case.R_surf) for _ in range(3)]
    for phi_index, (lower_index, upper_index, upper_weight) in enumerate(
        zip(lower, upper, weight)
    ):
        low = blocks_by_section[int(lower_index)][int(phi_index)]
        w = float(upper_weight)
        if w <= 1.0e-14:
            blended = low
        else:
            high = blocks_by_section[int(upper_index)][int(phi_index)]
            blended = tuple((1.0 - w) * left + w * right for left, right in zip(low, high))
        for destination, values in zip(components, blended):
            destination[phi_index] = values
    nearest_distance = np.minimum(weight, 1.0 - weight) * span
    return FPTSurfaceCylindricalDeltaField(
        R_surf=case.R_surf,
        Z_surf=case.Z_surf,
        phi_vals=case.phi_vals,
        delta_BR=components[0],
        delta_BZ=components[1],
        delta_BPhi=components[2],
        sampling_audit={
            "method": "periodic_linear_between_fem_sections",
            "nfp": int(getattr(plan, "n_fp", 1)),
            "section_count": len(tuple(getattr(plan, "sections", ()))),
            "exact_section_slice_count": int(np.count_nonzero(weight <= 1.0e-14)),
            "target_slice_count": int(weight.size),
            "max_bracketing_section_span_rad": float(np.max(span)),
            "max_nearest_section_distance_rad": float(np.max(nearest_distance)),
        },
    )


def _validate_surface_sampling_domain(
    plan: Any,
    case: BoundaryTopologyCase,
    points_in_polygon: Callable[[Any, Any], np.ndarray],
) -> tuple[dict[str, Any], ...]:
    lower, upper, weight, _span = _periodic_section_interpolation(plan, case.phi_vals)
    groups = _section_interpolation_groups(lower, upper, weight)
    sections = tuple(getattr(plan, "sections", ()))
    audit = []
    for section_index, phi_indices in groups.items():
        wall = getattr(sections[section_index], "wall_curve", None)
        if wall is None:
            raise ValueError(
                f"Topoquest section {section_index} has no wall curve for sampling validation"
            )
        points = _surface_points(case, phi_indices)
        inside = np.asarray(points_in_polygon(points, wall), dtype=bool).reshape(-1)
        if inside.size != points.shape[0]:
            raise ValueError("wall-domain validator returned an unexpected mask shape")
        outside_count = int(np.count_nonzero(~inside))
        if outside_count:
            raise ValueError(
                "response surface contains points outside a bracketing FEM wall domain: "
                f"section={section_index}, outside_count={outside_count}"
            )
        audit.append(
            {
                "section_index": int(section_index),
                "target_slice_count": int(len(phi_indices)),
                "point_count": int(points.shape[0]),
                "outside_point_count": 0,
            }
        )
    return tuple(audit)


def _sample_coupled_fem_solution(
    plan: Any,
    solution_module: Any,
    ramp: Any,
    beta_index: int,
    case: BoundaryTopologyCase,
) -> FPTSurfaceCylindricalDeltaField:
    """Sample one coupled rectangular-FEM beta step on pyna surfaces."""

    steps = tuple(getattr(ramp, "steps", ()))
    if beta_index < 0 or beta_index >= len(steps):
        raise ValueError("Topoquest coupled ramp is missing the selected beta step")
    lower, upper, weight, span = _periodic_section_interpolation(plan, case.phi_vals)
    groups = _section_interpolation_groups(lower, upper, weight)
    shape_2d = tuple(case.R_surf.shape[1:])
    points_by_section = {
        section_index: _surface_points(case, phi_indices)
        for section_index, phi_indices in groups.items()
    }
    sampler = getattr(
        solution_module,
        "sample_coupled_field_period_solution_at_points_by_section",
    )
    samples_by_section = sampler(steps[beta_index].solve, points_by_section)
    blocks_by_section: dict[
        int,
        dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ] = {}
    for section_index, phi_indices in groups.items():
        if section_index not in samples_by_section:
            raise ValueError(f"coupled solution samples are missing section {section_index}")
        blocks_by_section[section_index] = _section_sample_blocks(
            samples_by_section[section_index],
            phi_indices,
            shape_2d,
        )
    return _interpolated_surface_field(
        plan,
        case,
        blocks_by_section,
        lower,
        upper,
        weight,
        span,
    )


@dataclass(frozen=True)
class TopoquestFPTFieldPeriodRunner:
    """Lazy bridge to Topoquest's cylindrical field-period FPT beta ramp."""

    plan: Any
    section_contexts: Sequence[Any]
    B0: Any = None
    J0: Any = None
    pressure_source: Any = 0.0
    response_case: BoundaryTopologyCase | None = None
    runner_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        contexts = tuple(self.section_contexts)
        sections = tuple(getattr(self.plan, "sections", ()))
        if not contexts or len(contexts) != len(sections):
            raise ValueError("section_contexts must contain one context per Topoquest plan section")
        if self.response_case is not None and not isinstance(self.response_case, BoundaryTopologyCase):
            raise TypeError("response_case must be a BoundaryTopologyCase")
        options = dict(self.runner_kwargs or {})
        reserved = {"beta_values", "B0", "J0", "delta_B_ext", "pressure_source", "metadata"}
        overlap = reserved.intersection(options)
        if overlap:
            raise ValueError(f"runner_kwargs cannot override reserved arguments: {sorted(overlap)}")
        object.__setattr__(self, "section_contexts", contexts)
        object.__setattr__(self, "runner_kwargs", options)

    def __call__(
        self,
        spec: TopoquestFPTBetaRampSpec,
        state: TopoquestFPTVacuumControlState,
    ) -> TopoquestFPTBetaRampResult:
        require_topoquest_fpt_capability()
        if state.vacuum_delta_field is None:
            raise ValueError("live Topoquest FPT solves require a cylindrical vacuum_delta_field")
        B0 = self.B0 if self.B0 is not None else state.request.baseline_field
        if B0 is None:
            B0 = state.case.background_field
        if B0 is None:
            raise ValueError("live Topoquest FPT solves require a baseline cylindrical B0 field")
        if self.J0 is None:
            raise ValueError("live Topoquest FPT solves require a cylindrical J0 field")
        try:
            field_adapter = importlib.import_module("topoquest.mesh.field_adapter")
            solution = importlib.import_module("topoquest.mesh.fpt_solution")
        except (ImportError, ModuleNotFoundError) as exc:
            capability = diagnose_topoquest_fpt_capability()
            raise TopoquestFPTUnavailableError(
                f"{capability.message} Topoquest import failed: {exc}"
            ) from exc

        run = getattr(field_adapter, "run_fpt_field_period_beta_ramp_from_cylindrical_fields")
        raw = run(
            self.plan,
            self.section_contexts,
            beta_values=spec.beta_values,
            B0=B0,
            J0=self.J0,
            delta_B_ext=state.vacuum_delta_field,
            pressure_source=self.pressure_source,
            metadata={"case_alias": spec.case_alias, **dict(spec.metadata)},
            **self.runner_kwargs,
        )
        ramp = getattr(raw, "ramp", None)
        if ramp is None:
            raise ValueError("Topoquest field adapter returned no beta-ramp result")
        ramp_betas = np.asarray(getattr(ramp, "beta_values", ()), dtype=float)
        matches = np.flatnonzero(
            np.isclose(ramp_betas, spec.response_beta, rtol=1.0e-12, atol=1.0e-15)
        )
        if matches.size != 1:
            raise ValueError("Topoquest beta-ramp output does not contain spec.response_beta")
        beta_index = int(matches[0])
        response_case = state.case if self.response_case is None else self.response_case
        sampled = self._sample_selected_solution(solution, ramp, beta_index, response_case)
        ready, readiness = _readiness_payload(getattr(raw, "production_readiness", None))
        generation_id = _live_result_generation_id(
            spec,
            raw,
            ramp,
            label="Topoquest field-period FPT result",
        )
        return TopoquestFPTBetaRampResult(
            beta=float(ramp_betas[beta_index]),
            converged=bool(getattr(ramp, "converged", False)),
            production_ready=ready,
            generation_id=generation_id,
            delta_field=sampled,
            response_case=response_case,
            background_field=B0,
            equilibrium=state.request.baseline_equilibrium,
            readiness=readiness,
            metadata={
                "runner": "topoquest_field_period_fpt",
                "coupled": not hasattr(ramp, "sections"),
            },
            raw_result=raw,
        )

    def _sample_selected_solution(
        self,
        solution_module: Any,
        ramp: Any,
        beta_index: int,
        case: BoundaryTopologyCase,
    ) -> FPTSurfaceCylindricalDeltaField:
        lower, upper, weight, span = _periodic_section_interpolation(
            self.plan,
            case.phi_vals,
        )
        groups = _section_interpolation_groups(lower, upper, weight)
        shape_2d = tuple(case.R_surf.shape[1:])
        points_by_section = {
            section_index: _surface_points(case, phi_indices)
            for section_index, phi_indices in groups.items()
        }
        if hasattr(ramp, "sections"):
            sections = tuple(ramp.sections)
            if len(sections) != len(tuple(getattr(self.plan, "sections", ()))):
                raise ValueError("Topoquest uncoupled ramp section count does not match the plan")
            sampler = getattr(solution_module, "sample_fpt_solution_at_points")
            blocks_by_section: dict[
                int,
                dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
            ] = {}
            for section_index, phi_indices in groups.items():
                section_ramp = sections[section_index].result
                solve = tuple(section_ramp.steps)[beta_index].solve
                samples = sampler(solve, points_by_section[section_index])
                blocks_by_section[section_index] = _section_sample_blocks(
                    samples,
                    phi_indices,
                    shape_2d,
                )
            return _interpolated_surface_field(
                self.plan,
                case,
                blocks_by_section,
                lower,
                upper,
                weight,
                span,
            )
        else:
            return _sample_coupled_fem_solution(
                self.plan,
                solution_module,
                ramp,
                beta_index,
                case,
            )


@dataclass(frozen=True)
class TopoquestFEMFPTNeoclassicalRunner:
    """Bridge pyna controls to Topoquest's rectangular FEM neoclassical ramp.

    The injected ``parallel_current_closure_factory`` has Topoquest's complete
    callback signature ``(beta, previous, problems, section_assemblies)``.
    Consequently the transport closure can be recomputed from the previous
    beta solution instead of being frozen into a cached response column.
    """

    plan: Any
    msh_paths: Sequence[Any]
    parallel_current_closure_factory: Callable[..., Any]
    B0: Any = None
    J0: Any = None
    pressure_source: Any = 0.0
    response_case: BoundaryTopologyCase | None = None
    coefficient_kwargs: Mapping[str, Any] = field(default_factory=dict)
    residual_kwargs: Mapping[str, Any] = field(default_factory=dict)
    production_kwargs: Mapping[str, Any] = field(default_factory=dict)
    readiness_evaluator: Callable[[Any], Any] | None = None

    def __post_init__(self) -> None:
        paths = tuple(self.msh_paths)
        sections = tuple(getattr(self.plan, "sections", ()))
        if not paths or len(paths) != len(sections):
            raise ValueError("msh_paths must contain one mesh per Topoquest plan section")
        if len(sections) < 3:
            raise ValueError("rectangular FEM field-period solves require at least three sections")
        if not callable(self.parallel_current_closure_factory):
            raise TypeError("parallel_current_closure_factory must be callable")
        if self.response_case is not None and not isinstance(self.response_case, BoundaryTopologyCase):
            raise TypeError("response_case must be a BoundaryTopologyCase")
        if self.readiness_evaluator is not None and not callable(self.readiness_evaluator):
            raise TypeError("readiness_evaluator must be callable")
        coefficient_kwargs = dict(self.coefficient_kwargs or {})
        residual_kwargs = dict(self.residual_kwargs or {})
        production_kwargs = dict(self.production_kwargs or {})
        reserved_coefficient = {"plan", "B0", "J0", "points_by_section", "point_source"}
        reserved_residual = {
            "plan",
            "J0",
            "delta_B_ext",
            "pressure_source",
            "points_by_section",
            "point_source",
        }
        reserved_production = {
            "plan",
            "msh_paths",
            "coefficient_payload",
            "residual_payload",
            "beta_values",
            "parallel_current_closure_factory",
            "metadata",
        }
        overlaps = (
            reserved_coefficient.intersection(coefficient_kwargs),
            reserved_residual.intersection(residual_kwargs),
            reserved_production.intersection(production_kwargs),
        )
        if any(overlaps):
            names = sorted(set().union(*overlaps))
            raise ValueError(f"runner kwargs cannot override reserved arguments: {names}")
        object.__setattr__(self, "msh_paths", paths)
        object.__setattr__(self, "coefficient_kwargs", coefficient_kwargs)
        object.__setattr__(self, "residual_kwargs", residual_kwargs)
        object.__setattr__(self, "production_kwargs", production_kwargs)

    def __call__(
        self,
        spec: TopoquestFPTBetaRampSpec,
        state: TopoquestFPTVacuumControlState,
    ) -> TopoquestFPTBetaRampResult:
        require_topoquest_neoclassical_fpt_capability()
        if state.vacuum_delta_field is None:
            raise ValueError("rectangular FEM FPT solves require a cylindrical vacuum_delta_field")
        B0 = self.B0 if self.B0 is not None else state.request.baseline_field
        if B0 is None:
            B0 = state.case.background_field
        if B0 is None:
            raise ValueError("rectangular FEM FPT solves require a baseline cylindrical B0 field")
        if self.J0 is None:
            raise ValueError("rectangular FEM FPT solves require a cylindrical J0 field")
        try:
            field_adapter = importlib.import_module("topoquest.mesh.field_adapter")
            production = importlib.import_module("topoquest.solvers.fem_fpt.production")
            solution = importlib.import_module("topoquest.mesh.fpt_solution")
            stellarator_plan = importlib.import_module("topoquest.mesh.stellarator_plan")
        except (ImportError, ModuleNotFoundError) as exc:
            capability = diagnose_topoquest_fpt_capability()
            raise TopoquestFPTUnavailableError(
                f"{capability.message} Topoquest rectangular FEM import failed: {exc}"
            ) from exc

        sample_results: dict[str, Any] = {}
        points_cache: dict[int, Any] | None = None

        def context_points(context_build: Any) -> dict[int, Any]:
            nonlocal points_cache
            if points_cache is None:
                points_cache = getattr(
                    field_adapter,
                    "points_by_section_from_contexts",
                )(context_build.contexts)
            return points_cache

        def coefficient_payload_factory(context_build: Any) -> Mapping[str, Any]:
            result = getattr(
                field_adapter,
                "fpt_coefficient_payload_from_cylindrical_fields",
            )(
                self.plan,
                B0=B0,
                J0=self.J0,
                points_by_section=context_points(context_build),
                point_source="context_vertices",
                **self.coefficient_kwargs,
            )
            sample_results["coefficient"] = result
            return result.payload

        def residual_payload_factory(context_build: Any) -> Mapping[str, Any]:
            result = getattr(
                field_adapter,
                "fpt_external_field_residual_payload_from_cylindrical_fields",
            )(
                self.plan,
                J0=self.J0,
                delta_B_ext=state.vacuum_delta_field,
                pressure_source=self.pressure_source,
                points_by_section=context_points(context_build),
                point_source="context_vertices",
                **self.residual_kwargs,
            )
            sample_results["residual"] = result
            return result.payload

        run = getattr(production, "run_rectangular_fem_fpt_msh_payloads")
        raw = run(
            self.plan,
            self.msh_paths,
            coefficient_payload=coefficient_payload_factory,
            residual_payload=residual_payload_factory,
            beta_values=spec.beta_values,
            parallel_current_closure_factory=self.parallel_current_closure_factory,
            metadata={
                "case_alias": spec.case_alias,
                "plasma_closure": "neoclassical_parallel_current_refreshed_per_beta",
                **dict(spec.metadata),
            },
            **self.production_kwargs,
        )
        if not bool(getattr(raw, "ran", True)):
            reason = getattr(raw, "skipped_reason", "unknown")
            raise RuntimeError(f"Topoquest rectangular FEM beta ramp did not run: {reason}")
        ramp = getattr(raw, "ramp", None)
        if ramp is None:
            raise ValueError("Topoquest rectangular FEM entry point returned no beta-ramp result")
        steps = tuple(getattr(ramp, "steps", ()))
        ramp_betas = np.asarray([float(getattr(step, "beta")) for step in steps], dtype=float)
        matches = np.flatnonzero(
            np.isclose(ramp_betas, spec.response_beta, rtol=1.0e-12, atol=1.0e-15)
        )
        if matches.size != 1:
            stopped = getattr(ramp, "stopped_reason", None)
            raise RuntimeError(
                "Topoquest rectangular FEM output does not contain "
                f"response_beta={spec.response_beta}; stopped_reason={stopped!r}, "
                f"available_beta_values={tuple(float(value) for value in ramp_betas)}"
            )
        beta_index = int(matches[0])
        selected_step = steps[beta_index]
        assembly = getattr(selected_step, "assembly", None)
        assembly_diagnostics = dict(getattr(assembly, "diagnostics", {}) or {})
        closure_audit = _public_metadata(
            assembly_diagnostics.get("parallel_current_closure", {})
        )
        if not bool(closure_audit.get("enabled", False)):
            raise RuntimeError(
                "Topoquest neoclassical closure factory produced no physical "
                "parallel-current rows at response_beta"
            )
        raw_metadata = dict(getattr(raw, "metadata", {}) or {})
        if raw_metadata.get("mesh_phase_validated") is not True:
            raise RuntimeError("Topoquest rectangular FEM result lacks a passed mesh phase audit")
        contexts = tuple(getattr(getattr(raw, "context_build", None), "contexts", ()))
        for context in contexts:
            comm = getattr(getattr(context, "mesh", None), "comm", None)
            size = getattr(comm, "size", None)
            if size is None and hasattr(comm, "Get_size"):
                size = comm.Get_size()
            if size is not None and int(size) > 1:
                raise NotImplementedError(
                    "surface response sampling currently requires single-rank FEM contexts"
                )
        response_case = state.case if self.response_case is None else self.response_case
        domain_audit = _validate_surface_sampling_domain(
            self.plan,
            response_case,
            getattr(stellarator_plan, "points_in_polygon_rz"),
        )
        sampled = _sample_coupled_fem_solution(
            self.plan,
            solution,
            ramp,
            beta_index,
            response_case,
        )
        ready: bool | None = None
        readiness: dict[str, Any] = {
            "solver_route": "topoquest.solvers.fem_fpt.rectangular_least_squares",
            "neoclassical_closure_refreshed_per_beta": True,
            "parallel_current_closure": closure_audit,
            "mesh_phase_validated": True,
            "context_vertex_payload_sampling": True,
            "surface_sampling_wall_domain_validated": True,
            "production_gate_supplied": self.readiness_evaluator is not None,
        }
        if self.readiness_evaluator is not None:
            ready, evaluated = _readiness_payload(self.readiness_evaluator(raw))
            readiness.update(evaluated)
        generation_id = _live_result_generation_id(
            spec,
            raw,
            ramp,
            selected_step,
            label="Topoquest rectangular FEM FPT result",
        )
        return TopoquestFPTBetaRampResult(
            beta=float(ramp_betas[beta_index]),
            converged=bool(getattr(ramp, "converged", False)),
            production_ready=ready,
            generation_id=generation_id,
            delta_field=sampled,
            response_case=response_case,
            background_field=B0,
            equilibrium=state.request.baseline_equilibrium,
            readiness=readiness,
            metadata={
                "runner": "topoquest_rectangular_fem_fpt_neoclassical",
                "solver_route": "rectangular_least_squares",
                "parallel_current_rows": "physical_FEM_weak_rows",
                "closure_refresh": "per_beta_with_previous_solution",
                "mesh_phase_audit": _public_metadata(
                    {"rows": raw_metadata.get("mesh_phase_audit", ())}
                ).get("rows", ()),
                "coefficient_sampling": _field_sample_audit_rows(
                    sample_results.get("coefficient")
                ),
                "residual_sampling": _field_sample_audit_rows(
                    sample_results.get("residual")
                ),
                "surface_solution_sampling": sampled.sampling_audit,
                "surface_sampling_domain_audit": domain_audit,
            },
            raw_result=raw,
        )


# Concise aliases for callers that do not need the provider name repeated.
FPTBetaRampScreeningSpec = TopoquestFPTBetaRampSpec
FPTBetaRampScreeningResult = TopoquestFPTBetaRampResult
FPTBetaRampRunner = TopoquestFPTBetaRampRunner
CachedTopoquestFPTResponseBasis = TopoquestFPTCachedResponseBasis
TopoquestFPTFeedbackAdapter = TopoquestFPTPlasmaFeedbackAdapter
topoquest_fpt_capability = diagnose_topoquest_fpt_capability


__all__ = [
    "CachedTopoquestFPTResponseBasis",
    "FPTBetaRampRunner",
    "FPTBetaRampScreeningResult",
    "FPTBetaRampScreeningSpec",
    "FPTSurfaceCylindricalDeltaField",
    "TopoquestFEMFPTNeoclassicalRunner",
    "TopoquestFPTBetaRampResult",
    "TopoquestFPTBetaRampRunner",
    "TopoquestFPTBetaRampSpec",
    "TopoquestFPTCachedResponseBasis",
    "TopoquestFPTCapability",
    "TopoquestFPTFeedbackAdapter",
    "TopoquestFPTExternalRuntime",
    "TopoquestFPTFieldPeriodRunner",
    "TopoquestFPTPlasmaFeedbackAdapter",
    "TopoquestFPTUnavailableError",
    "TopoquestFPTVacuumControlState",
    "TopoquestNeoclassicalFPTCapability",
    "diagnose_topoquest_fpt_capability",
    "diagnose_topoquest_fpt_external_runtime",
    "diagnose_topoquest_neoclassical_fpt_capability",
    "diagnose_topoquest_neoclassical_fpt_external_runtime",
    "require_topoquest_fpt_capability",
    "require_topoquest_neoclassical_fpt_capability",
    "topoquest_fpt_capability",
]
