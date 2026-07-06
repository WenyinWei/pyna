"""VMEC wout counterpart benchmarks.

This module compares a public VMEC ``wout`` file as read by several external
packages.  It is intentionally an adapter/diagnostics layer: the external
packages remain optional imports, and the public report avoids writing local
absolute paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


SCALAR_KEYS = (
    "ier_flag",
    "nfp",
    "ns",
    "mpol",
    "ntor",
    "mnmax",
    "mnmax_nyq",
    "betatotal",
    "betapol",
    "betator",
    "aspect",
    "Rmajor_p",
    "Aminor_p",
    "volume_p",
    "volavgB",
)

PROFILE_KEYS = ("iotaf", "iotas", "presf", "pres", "phi")
MODE_KEYS = ("rmnc", "zmns", "bmnc")
AXIS_KEYS = ("raxis_cc", "zaxis_cs")
VMEC_ARRAY_KEYS = PROFILE_KEYS + MODE_KEYS + AXIS_KEYS + ("xm", "xn", "xm_nyq", "xn_nyq")
GEOMETRY_KEYS = ("axis_R_phi", "axis_Z_phi", "lcfs_R_sections", "lcfs_Z_sections")


@dataclass
class CounterpartResult:
    """One reader/backend result.

    ``arrays`` is deliberately excluded from JSON output; it exists only for
    local comparisons and plotting.
    """

    name: str
    status: str
    runtime_s: float
    summary: dict[str, Any] = field(default_factory=dict)
    arrays: dict[str, NDArray[np.float64]] = field(default_factory=dict, repr=False)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def public_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "runtime_s": self.runtime_s,
            "summary": self.summary,
        }
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass
class VmecBenchmarkReport:
    """In-memory VMEC counterpart benchmark result."""

    source: dict[str, Any]
    readers: dict[str, CounterpartResult]
    comparisons: dict[str, Any]
    booz_xform: CounterpartResult | None = None
    desc_booz: CounterpartResult | None = None

    def public_payload(self) -> dict[str, Any]:
        payload = {
            "source": self.source,
            "readers": {key: result.public_payload() for key, result in self.readers.items()},
            "comparisons": self.comparisons,
        }
        if self.booz_xform is not None:
            payload["booz_xform"] = self.booz_xform.public_payload()
        if self.desc_booz is not None:
            payload["desc_booz"] = self.desc_booz.public_payload()
        return payload


def run_vmec_counterpart_benchmark(
    wout: str | Path,
    *,
    booz_surfaces: Sequence[int] = (20, 50, 80),
    mboz: int = 24,
    nboz: int = 24,
    run_booz_xform: bool = True,
    run_desc_booz: bool = False,
    desc_use_gpu: bool = False,
    include_local_path: bool = False,
    desc_booz_path: str | Path | None = None,
    geometry_phi: Sequence[float] | None = None,
    geometry_theta_count: int = 256,
    topoquest_root: str | Path | None = None,
) -> VmecBenchmarkReport:
    """Run a VMEC ``wout`` benchmark across installed counterpart packages."""

    if not desc_use_gpu:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    path = Path(wout).expanduser()
    source = public_wout_source(path, include_local_path=include_local_path)
    phi_values = _geometry_phi_values(path, geometry_phi)
    theta_count = int(geometry_theta_count)
    source["geometry_phi_rad"] = [float(x) for x in phi_values]
    source["geometry_theta_count"] = theta_count
    readers: dict[str, CounterpartResult] = {
        "netcdf": _safe_reader("netcdf", lambda: _read_netcdf_wout(path, geometry_phi=phi_values, geometry_theta_count=theta_count)),
        "pyna_native": _safe_reader("pyna_native", lambda: _read_pyna_native_wout(path, geometry_phi=phi_values, geometry_theta_count=theta_count)),
        "topoquest": _safe_reader(
            "topoquest",
            lambda: _read_topoquest_vmec_geometry(
                path,
                geometry_phi=phi_values,
                geometry_theta_count=theta_count,
                topoquest_root=topoquest_root,
            ),
        ),
        "simsopt": _safe_reader("simsopt", lambda: _read_simsopt_wout(path, geometry_phi=phi_values, geometry_theta_count=theta_count)),
        "vmecpp": _safe_reader("vmecpp", lambda: _read_vmecpp_wout(path, geometry_phi=phi_values, geometry_theta_count=theta_count)),
        "desc": _safe_reader("desc", lambda: _read_desc_wout(path, use_gpu=desc_use_gpu, geometry_phi=phi_values, geometry_theta_count=theta_count)),
    }
    comparisons = compare_counterpart_results(readers)

    booz_result = None
    if run_booz_xform:
        booz_result = _safe_reader(
            "booz_xform",
            lambda: _run_booz_xform(path, booz_surfaces=booz_surfaces, mboz=mboz, nboz=nboz),
        )

    desc_booz_result = None
    if run_desc_booz:
        out_path = None if desc_booz_path is None else Path(desc_booz_path).expanduser()
        desc_booz_result = _safe_reader(
            "desc_booz",
            lambda: _run_desc_booz(path, out_path=out_path, surfs=max(booz_surfaces) + 1, mboz=mboz, nboz=nboz, use_gpu=desc_use_gpu),
        )

    return VmecBenchmarkReport(
        source=source,
        readers=readers,
        comparisons=comparisons,
        booz_xform=booz_result,
        desc_booz=desc_booz_result,
    )


def write_vmec_benchmark_outputs(
    report: VmecBenchmarkReport,
    out_dir: str | Path,
    *,
    prefix: str = "vmec_counterpart_benchmark",
    make_plots: bool = True,
) -> dict[str, str]:
    """Write JSON/CSV and optional PNG summaries."""

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    json_path = out / f"{prefix}.json"
    json_path.write_text(json.dumps(report.public_payload(), indent=2, sort_keys=True), encoding="utf-8")
    paths["json"] = str(json_path)

    csv_path = out / f"{prefix}_scalar_errors.csv"
    _write_scalar_comparison_csv(report.comparisons, csv_path)
    paths["scalar_csv"] = str(csv_path)

    if make_plots:
        profile_path = out / f"{prefix}_profiles.png"
        plot_profile_summary(report, profile_path)
        paths["profile_png"] = str(profile_path)
        geometry_path = out / f"{prefix}_geometry_sections.png"
        plot_vmec_geometry_summary(report, geometry_path)
        paths["geometry_png"] = str(geometry_path)
        if report.booz_xform is not None and report.booz_xform.ok:
            booz_path = out / f"{prefix}_booz_xform_modes.png"
            plot_boozer_mode_summary(report.booz_xform, booz_path)
            paths["booz_xform_png"] = str(booz_path)

    return paths


def public_wout_source(path: Path, *, include_local_path: bool = False) -> dict[str, Any]:
    """Return path-safe source provenance for a local wout file."""

    p = path.expanduser()
    if not p.exists():
        raise FileNotFoundError(p)
    payload = {
        "basename": p.name,
        "size_bytes": p.stat().st_size,
        "sha256_16": _sha256_prefix(p, n=16),
    }
    if include_local_path:
        payload["local_path"] = str(p)
    return payload


def compare_counterpart_results(readers: Mapping[str, CounterpartResult]) -> dict[str, Any]:
    """Compare all successful readers against the direct netCDF reader."""

    ref = readers.get("netcdf")
    if ref is None or not ref.ok:
        return {"status": "no_reference"}

    out: dict[str, Any] = {"status": "ok", "reference": "netcdf", "readers": {}}
    for name, result in readers.items():
        if name == "netcdf":
            continue
        if not result.ok:
            out["readers"][name] = {"status": result.status, "error": result.error}
            continue
        out["readers"][name] = {
            "status": "ok",
            "scalars": _compare_scalar_summaries(ref.summary.get("scalars", {}), result.summary.get("scalars", {})),
            "profiles": _compare_profiles(ref.arrays, result.arrays),
            "modes": _compare_modes(ref.arrays, result.arrays, ref.summary),
            "geometry": _compare_geometry(ref.arrays, result.arrays),
        }
    return out


def plot_profile_summary(report: VmecBenchmarkReport, out_path: str | Path):
    """Plot iota and pressure profiles from all successful readers."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = {
        "netcdf": "black",
        "pyna_native": "tab:orange",
        "topoquest": "tab:purple",
        "simsopt": "tab:blue",
        "vmecpp": "tab:green",
        "desc": "tab:red",
    }
    for name, result in report.readers.items():
        if not result.ok:
            continue
        arrays = result.arrays
        x = _normalised_radius(arrays.get("phi"), arrays.get("iotaf"))
        iota = arrays.get("iotaf")
        if x is not None and iota is not None:
            axes[0].plot(x, np.asarray(iota, dtype=float), label=name, lw=1.5, color=colors.get(name))
        pressure = arrays.get("presf")
        if pressure is None:
            pressure = arrays.get("pres")
        xp = _normalised_radius(arrays.get("phi"), pressure)
        if xp is not None and pressure is not None:
            axes[1].plot(xp, np.asarray(pressure, dtype=float), label=name, lw=1.5, color=colors.get(name))
    axes[0].set_title("W7-X public wout iota")
    axes[0].set_xlabel("normalised toroidal flux")
    axes[0].set_ylabel("iota")
    axes[1].set_title("W7-X public wout pressure")
    axes[1].set_xlabel("normalised toroidal flux")
    axes[1].set_ylabel("pressure [Pa]")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return fig


def plot_vmec_geometry_summary(report: VmecBenchmarkReport, out_path: str | Path):
    """Plot VMEC axis and LCFS sections from pyna/topoquest/counterparts."""

    import matplotlib.pyplot as plt

    ok = [
        (name, result)
        for name, result in report.readers.items()
        if result.ok and "lcfs_R_sections" in result.arrays and "lcfs_Z_sections" in result.arrays
    ]
    if not ok:
        raise ValueError("no successful reader contains LCFS geometry arrays")

    ref_arrays = report.readers.get("netcdf").arrays if report.readers.get("netcdf") is not None else ok[0][1].arrays
    phi = np.asarray(ref_arrays.get("geometry_phi", np.arange(ok[0][1].arrays["lcfs_R_sections"].shape[0])), dtype=float)
    n_panels = min(4, int(ok[0][1].arrays["lcfs_R_sections"].shape[0]))
    ncols = 2 if n_panels > 1 else 1
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 5.0 * nrows), squeeze=False, constrained_layout=True)
    colors = {
        "netcdf": "black",
        "pyna_native": "tab:orange",
        "topoquest": "tab:purple",
        "simsopt": "tab:blue",
        "vmecpp": "tab:green",
        "desc": "tab:red",
    }
    linestyles = {
        "netcdf": "-",
        "pyna_native": "--",
        "topoquest": ":",
        "simsopt": "-.",
        "vmecpp": (0, (3, 1, 1, 1)),
        "desc": (0, (1, 1)),
    }
    for ax in axes.ravel()[n_panels:]:
        ax.axis("off")
    for i_sec, ax in enumerate(axes.ravel()[:n_panels]):
        for name, result in ok:
            R = np.asarray(result.arrays["lcfs_R_sections"], dtype=float)
            Z = np.asarray(result.arrays["lcfs_Z_sections"], dtype=float)
            if i_sec >= R.shape[0] or i_sec >= Z.shape[0]:
                continue
            ax.plot(
                R[i_sec],
                Z[i_sec],
                lw=1.35 if name in {"netcdf", "pyna_native", "topoquest"} else 1.0,
                alpha=0.9,
                color=colors.get(name),
                linestyle=linestyles.get(name, "-"),
                label=name,
            )
            axis_R = result.arrays.get("axis_R_phi")
            axis_Z = result.arrays.get("axis_Z_phi")
            if axis_R is not None and axis_Z is not None and i_sec < np.asarray(axis_R).size:
                ax.scatter(
                    [float(np.asarray(axis_R)[i_sec])],
                    [float(np.asarray(axis_Z)[i_sec])],
                    s=18,
                    color=colors.get(name),
                    marker="x",
                    linewidths=0.8,
                )
        title_phi = phi[i_sec] if i_sec < phi.size else float(i_sec)
        ax.set_title(f"LCFS section phi={title_phi:.4f} rad")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
    axes.ravel()[0].legend(loc="best", fontsize=8)
    fig.suptitle("W7-X public VMEC geometry: pyna/topoquest/counterparts", fontsize=12)
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return fig


def plot_boozer_mode_summary(result: CounterpartResult, out_path: str | Path):
    """Plot leading Boozer |B| mode amplitudes from a booz_xform result."""

    import matplotlib.pyplot as plt

    bmnc = result.arrays.get("bmnc_b")
    xm = result.arrays.get("xm_b")
    xn = result.arrays.get("xn_b")
    s = result.arrays.get("s_b")
    if bmnc is None or xm is None or xn is None or s is None:
        raise ValueError("booz_xform result does not contain bmnc_b/xm_b/xn_b/s_b arrays")
    bmnc = np.asarray(bmnc, dtype=float)
    amp = np.max(np.abs(bmnc), axis=1)
    order = np.argsort(amp)[::-1][:20]
    labels = [f"({int(xm[i])},{int(xn[i])})" for i in order]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].bar(np.arange(order.size), amp[order])
    axes[0].set_xticks(np.arange(order.size))
    axes[0].set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    axes[0].set_ylabel("max |B_mn| [T]")
    axes[0].set_title("Leading Boozer |B| modes")

    for idx in order[:8]:
        axes[1].plot(np.asarray(s, dtype=float), np.abs(bmnc[idx, :]), marker="o", label=f"({int(xm[idx])},{int(xn[idx])})")
    axes[1].set_xlabel("s")
    axes[1].set_ylabel("|B_mn| [T]")
    axes[1].set_title("Radial samples")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return fig


def _safe_reader(name: str, fn: Callable[[], tuple[dict[str, Any], dict[str, NDArray[np.float64]]]]) -> CounterpartResult:
    t0 = time.perf_counter()
    try:
        summary, arrays = fn()
        return CounterpartResult(name=name, status="ok", runtime_s=time.perf_counter() - t0, summary=summary, arrays=arrays)
    except Exception as exc:  # pragma: no cover - used for optional external backends.
        return CounterpartResult(
            name=name,
            status="failed",
            runtime_s=time.perf_counter() - t0,
            error=f"{type(exc).__name__}: {exc}",
        )


class _silence_native_output:
    """Temporarily redirect process stdout/stderr for native extension noise."""

    def __enter__(self):
        self._devnull = open(os.devnull, "w", encoding="utf-8")
        self._old_stdout = os.dup(1)
        self._old_stderr = os.dup(2)
        os.dup2(self._devnull.fileno(), 1)
        os.dup2(self._devnull.fileno(), 2)
        return self

    def __exit__(self, exc_type, exc, tb):
        os.dup2(self._old_stdout, 1)
        os.dup2(self._old_stderr, 2)
        os.close(self._old_stdout)
        os.close(self._old_stderr)
        self._devnull.close()
        return False


def _read_netcdf_wout(
    path: Path,
    *,
    geometry_phi: NDArray[np.float64] | None = None,
    geometry_theta_count: int = 256,
) -> tuple[dict[str, Any], dict[str, NDArray[np.float64]]]:
    from netCDF4 import Dataset

    arrays: dict[str, NDArray[np.float64]] = {}
    scalars: dict[str, Any] = {}
    with Dataset(str(path), "r") as ds:
        for key in SCALAR_KEYS:
            if key in ds.variables:
                scalars[key] = _json_scalar(ds.variables[key][...])
        for key in VMEC_ARRAY_KEYS:
            if key in ds.variables:
                arrays[key] = _as_float_array(ds.variables[key][...])
    _add_vmec_geometry_arrays(arrays, scalars, geometry_phi=geometry_phi, geometry_theta_count=geometry_theta_count)
    summary = _summary_from_scalars_arrays(scalars, arrays)
    summary["backend"] = "netCDF4 + pyna VMEC Fourier geometry reference"
    return summary, arrays


def _read_pyna_native_wout(
    path: Path,
    *,
    geometry_phi: NDArray[np.float64] | None,
    geometry_theta_count: int,
) -> tuple[dict[str, Any], dict[str, NDArray[np.float64]]]:
    summary, arrays = _read_netcdf_wout(path, geometry_phi=geometry_phi, geometry_theta_count=geometry_theta_count)
    summary["backend"] = "pyna.benchmark native VMEC Fourier evaluator"
    summary["note"] = "Direct VMEC wout read plus pyna-native axis/LCFS Fourier evaluation; included to compare our convention against topoquest and external packages."
    return summary, arrays


def _read_topoquest_vmec_geometry(
    path: Path,
    *,
    geometry_phi: NDArray[np.float64] | None,
    geometry_theta_count: int,
    topoquest_root: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, NDArray[np.float64]]]:
    _ensure_topoquest_import_path(topoquest_root)
    from topoquest.analysis.stellarator_mgrid import vmec_axis_from_wout, vmec_lcfs_from_wout

    phi_values = np.asarray(geometry_phi, dtype=np.float64)
    axis_R, axis_Z, nfp = vmec_axis_from_wout(path, phi_values)
    lcfs_R: list[NDArray[np.float64]] = []
    lcfs_Z: list[NDArray[np.float64]] = []
    for phi in phi_values:
        R, Z = vmec_lcfs_from_wout(path, float(phi), ntheta=int(geometry_theta_count))
        lcfs_R.append(np.asarray(R, dtype=np.float64))
        lcfs_Z.append(np.asarray(Z, dtype=np.float64))
    arrays = {
        "geometry_phi": np.ascontiguousarray(phi_values, dtype=np.float64),
        "geometry_theta": np.linspace(0.0, 2.0 * np.pi, int(geometry_theta_count), endpoint=True, dtype=np.float64),
        "axis_R_phi": np.ascontiguousarray(axis_R, dtype=np.float64),
        "axis_Z_phi": np.ascontiguousarray(axis_Z, dtype=np.float64),
        "lcfs_R_sections": np.ascontiguousarray(np.stack(lcfs_R), dtype=np.float64),
        "lcfs_Z_sections": np.ascontiguousarray(np.stack(lcfs_Z), dtype=np.float64),
    }
    summary = _summary_from_scalars_arrays({"nfp": int(nfp)}, arrays)
    summary["backend"] = "topoquest.analysis.stellarator_mgrid VMEC geometry helpers"
    return summary, arrays


def _read_simsopt_wout(
    path: Path,
    *,
    geometry_phi: NDArray[np.float64] | None = None,
    geometry_theta_count: int = 256,
) -> tuple[dict[str, Any], dict[str, NDArray[np.float64]]]:
    from simsopt.mhd import Vmec

    vmec = Vmec(str(path))
    wout = vmec.wout
    scalars = {key: _json_scalar(getattr(wout, key)) for key in SCALAR_KEYS if hasattr(wout, key)}
    arrays = {key: _as_float_array(getattr(wout, key)) for key in VMEC_ARRAY_KEYS if hasattr(wout, key)}
    _add_vmec_geometry_arrays(arrays, scalars, geometry_phi=geometry_phi, geometry_theta_count=geometry_theta_count)
    summary = _summary_from_scalars_arrays(scalars, arrays)
    summary["backend"] = "simsopt.mhd.Vmec"
    return summary, arrays


def _read_vmecpp_wout(
    path: Path,
    *,
    geometry_phi: NDArray[np.float64] | None = None,
    geometry_theta_count: int = 256,
) -> tuple[dict[str, Any], dict[str, NDArray[np.float64]]]:
    import vmecpp

    wout = vmecpp.VmecWOut.from_wout_file(path)
    scalars = {key: _json_scalar(getattr(wout, key)) for key in SCALAR_KEYS if hasattr(wout, key)}
    arrays = {key: _as_float_array(getattr(wout, key)) for key in VMEC_ARRAY_KEYS if hasattr(wout, key)}
    _add_vmec_geometry_arrays(arrays, scalars, geometry_phi=geometry_phi, geometry_theta_count=geometry_theta_count)
    summary = _summary_from_scalars_arrays(scalars, arrays)
    summary["backend"] = "vmecpp.VmecWOut"
    return summary, arrays


def _read_desc_wout(
    path: Path,
    *,
    use_gpu: bool = False,
    geometry_phi: NDArray[np.float64] | None = None,
    geometry_theta_count: int = 256,
) -> tuple[dict[str, Any], dict[str, NDArray[np.float64]]]:
    import desc

    if use_gpu:
        desc.set_device("gpu")
    else:
        desc.set_device("cpu")
    from desc.vmec import VMECIO

    payload = VMECIO.read_vmec_output(str(path))
    arrays = {
        key: _as_float_array(payload[key])
        for key in ("rmnc", "zmns", "lmns", "xm", "xn", "raxis_cc", "zaxis_cs")
        if key in payload
    }
    scalars = {}
    if "NFP" in payload:
        scalars["nfp"] = _json_scalar(payload["NFP"])
    _add_vmec_geometry_arrays(arrays, scalars, geometry_phi=geometry_phi, geometry_theta_count=geometry_theta_count)
    summary = _summary_from_scalars_arrays(scalars, arrays)
    summary["backend"] = "desc.vmec.VMECIO.read_vmec_output"
    summary["note"] = "DESC VMECIO.read_vmec_output exposes geometry-focused payloads, not full VMEC scalar/profile output."
    return summary, arrays


def _run_booz_xform(
    path: Path,
    *,
    booz_surfaces: Sequence[int],
    mboz: int,
    nboz: int,
) -> tuple[dict[str, Any], dict[str, NDArray[np.float64]]]:
    import booz_xform as bx

    b = bx.Booz_xform()
    with _silence_native_output():
        b.read_wout(str(path))
    b.compute_surfs = [int(x) for x in booz_surfaces]
    b.mboz = int(mboz)
    b.nboz = int(nboz)
    b.verbose = False
    with _silence_native_output():
        b.run()
    arrays = {
        "s_b": _as_float_array(b.s_b),
        "iota_b": _as_float_array(b.iota),
        "bmnc_b": _as_float_array(b.bmnc_b),
        "xm_b": _as_float_array(b.xm_b),
        "xn_b": _as_float_array(b.xn_b),
    }
    summary = {
        "backend": "booz_xform.Booz_xform",
        "mboz": int(b.mboz),
        "nboz": int(b.nboz),
        "mnboz": int(b.mnboz),
        "ns_in": int(b.ns_in),
        "ns_b": int(b.ns_b),
        "compute_surfs": [int(x) for x in b.compute_surfs],
        "s_b": _stats(arrays["s_b"]),
        "bmnc_b": _stats(arrays["bmnc_b"]),
        "leading_modes": _leading_boozer_modes(arrays, count=12),
    }
    return summary, arrays


def _run_desc_booz(
    path: Path,
    *,
    out_path: Path | None,
    surfs: int,
    mboz: int,
    nboz: int,
    use_gpu: bool = False,
) -> tuple[dict[str, Any], dict[str, NDArray[np.float64]]]:
    import desc

    if use_gpu:
        desc.set_device("gpu")
    else:
        desc.set_device("cpu")
    from desc.vmec import VMECIO
    from desc.vmec_utils import make_boozmn_output
    from netCDF4 import Dataset

    if out_path is None:
        out_path = Path.cwd() / "desc_boozmn_benchmark.nc"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eq = VMECIO.load(str(path))
    make_boozmn_output(eq, str(out_path), surfs=int(surfs), M_booz=int(mboz), N_booz=int(nboz), verbose=0)
    arrays: dict[str, NDArray[np.float64]] = {}
    with Dataset(str(out_path), "r") as ds:
        for key in ("bmnc_b", "ixm_b", "ixn_b", "iota_b", "phi_b"):
            if key in ds.variables:
                arrays[key] = _as_float_array(ds.variables[key][...])
    summary = {
        "backend": "desc.vmec_utils.make_boozmn_output",
        "output_basename": out_path.name,
        "surfs": int(surfs),
        "mboz": int(mboz),
        "nboz": int(nboz),
        "bmnc_b": _stats(arrays["bmnc_b"]) if "bmnc_b" in arrays else None,
    }
    return summary, arrays


def _summary_from_scalars_arrays(scalars: Mapping[str, Any], arrays: Mapping[str, NDArray[np.float64]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"scalars": dict(scalars), "profiles": {}, "modes": {}, "geometry": {}}
    for key in PROFILE_KEYS:
        if key in arrays:
            summary["profiles"][key] = _stats(arrays[key])
    for key in MODE_KEYS:
        if key in arrays:
            summary["modes"][key] = _stats(arrays[key])
    for key in ("xm", "xn", "xm_nyq", "xn_nyq"):
        if key in arrays:
            summary["modes"][key] = {
                "shape": list(np.asarray(arrays[key]).shape),
                "min": float(np.nanmin(arrays[key])) if arrays[key].size else None,
                "max": float(np.nanmax(arrays[key])) if arrays[key].size else None,
            }
    for key in AXIS_KEYS + GEOMETRY_KEYS:
        if key in arrays:
            summary["geometry"][key] = _stats(arrays[key])
    return summary


def _compare_scalar_summaries(reference: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, ref_value in reference.items():
        if key not in other:
            out[key] = {"status": "missing"}
            continue
        cmp_value = other[key]
        try:
            diff = float(cmp_value) - float(ref_value)
            denom = max(abs(float(ref_value)), abs(float(cmp_value)), 1.0e-300)
            out[key] = {"status": "ok", "reference": ref_value, "value": cmp_value, "abs_error": abs(diff), "rel_error": abs(diff) / denom}
        except (TypeError, ValueError):
            out[key] = {"status": "non_numeric", "reference": ref_value, "value": cmp_value, "match": ref_value == cmp_value}
    return out


def _compare_profiles(reference: Mapping[str, NDArray[np.float64]], other: Mapping[str, NDArray[np.float64]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in PROFILE_KEYS:
        if key not in reference:
            continue
        if key not in other:
            out[key] = {"status": "missing"}
            continue
        out[key] = _array_error(reference[key], other[key])
    return out


def _compare_modes(
    reference: Mapping[str, NDArray[np.float64]],
    other: Mapping[str, NDArray[np.float64]],
    ref_summary: Mapping[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ns = _int_from_summary(ref_summary, "ns")
    mnmax = _int_from_summary(ref_summary, "mnmax")
    mnmax_nyq = _int_from_summary(ref_summary, "mnmax_nyq")
    for key in MODE_KEYS:
        if key not in reference:
            continue
        if key not in other:
            out[key] = {"status": "missing"}
            continue
        mode_count = mnmax_nyq if key == "bmnc" else mnmax
        a = _normalise_vmec_mode_array(reference[key], ns=ns, mode_count=mode_count)
        b = _normalise_vmec_mode_array(other[key], ns=ns, mode_count=mode_count)
        out[key] = _array_error(a, b)
    return out


def _compare_geometry(reference: Mapping[str, NDArray[np.float64]], other: Mapping[str, NDArray[np.float64]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in GEOMETRY_KEYS:
        if key not in reference:
            continue
        if key not in other:
            out[key] = {"status": "missing"}
            continue
        out[key] = _array_error(reference[key], other[key])
    return out


def _normalise_vmec_mode_array(values: NDArray[np.float64], *, ns: int | None, mode_count: int | None) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        return arr
    if ns is not None and arr.shape[0] == ns:
        return arr
    if ns is not None and arr.shape[1] == ns:
        return arr.T
    if mode_count is not None and arr.shape[1] == mode_count:
        return arr
    if mode_count is not None and arr.shape[0] == mode_count:
        return arr.T
    return arr


def _normalise_surface_mode_array(values: NDArray[np.float64], *, mode_count: int | None) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"VMEC surface coefficient array must be 1-D or 2-D, got {arr.shape}")
    if mode_count is not None:
        if arr.shape[1] == mode_count:
            return arr
        if arr.shape[0] == mode_count:
            return arr.T
    return arr


def evaluate_vmec_axis_fourier(
    phi: Sequence[float] | NDArray[np.float64],
    raxis_cc: Sequence[float] | NDArray[np.float64],
    zaxis_cs: Sequence[float] | NDArray[np.float64],
    *,
    nfp: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate VMEC magnetic-axis Fourier coefficients using pyna convention."""

    phi_arr = np.asarray(phi, dtype=np.float64)
    rcc = np.asarray(raxis_cc, dtype=np.float64).ravel()
    zcs = np.asarray(zaxis_cs, dtype=np.float64).ravel()
    if rcc.shape != zcs.shape:
        raise ValueError(f"raxis_cc shape {rcc.shape} != zaxis_cs shape {zcs.shape}")
    modes = np.arange(rcc.size, dtype=np.float64)
    angles = modes[None, :] * int(nfp) * phi_arr.reshape(-1, 1)
    axis_R = np.sum(rcc[None, :] * np.cos(angles), axis=1)
    axis_Z = -np.sum(zcs[None, :] * np.sin(angles), axis=1)
    return np.ascontiguousarray(axis_R, dtype=np.float64), np.ascontiguousarray(axis_Z, dtype=np.float64)


def evaluate_vmec_surface_fourier(
    theta: Sequence[float] | NDArray[np.float64],
    phi: Sequence[float] | NDArray[np.float64],
    rmnc: Sequence[float] | NDArray[np.float64],
    zmns: Sequence[float] | NDArray[np.float64],
    xm: Sequence[float] | NDArray[np.float64],
    xn: Sequence[float] | NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate a single VMEC ``rmnc/zmns`` surface on ``(phi, theta)`` sections."""

    theta_arr = np.asarray(theta, dtype=np.float64).reshape(-1)
    phi_arr = np.asarray(phi, dtype=np.float64).reshape(-1)
    rm = np.asarray(rmnc, dtype=np.float64).reshape(-1)
    zm = np.asarray(zmns, dtype=np.float64).reshape(-1)
    m = np.asarray(xm, dtype=np.float64).reshape(-1)
    n = np.asarray(xn, dtype=np.float64).reshape(-1)
    if not (rm.shape == zm.shape == m.shape == n.shape):
        raise ValueError("rmnc, zmns, xm, and xn must have the same flattened shape")
    phase = theta_arr[None, :, None] * m[None, None, :] - phi_arr[:, None, None] * n[None, None, :]
    R = np.sum(rm[None, None, :] * np.cos(phase), axis=2)
    Z = np.sum(zm[None, None, :] * np.sin(phase), axis=2)
    return np.ascontiguousarray(R, dtype=np.float64), np.ascontiguousarray(Z, dtype=np.float64)


def _add_vmec_geometry_arrays(
    arrays: dict[str, NDArray[np.float64]],
    scalars: Mapping[str, Any],
    *,
    geometry_phi: NDArray[np.float64] | None,
    geometry_theta_count: int,
) -> None:
    if geometry_phi is None:
        return
    phi_values = np.asarray(geometry_phi, dtype=np.float64).reshape(-1)
    arrays["geometry_phi"] = np.ascontiguousarray(phi_values, dtype=np.float64)
    arrays["geometry_theta"] = np.linspace(0.0, 2.0 * np.pi, int(geometry_theta_count), endpoint=True, dtype=np.float64)

    nfp = _int_from_scalars(scalars, "nfp")
    if nfp is not None and all(key in arrays for key in AXIS_KEYS):
        arrays["axis_R_phi"], arrays["axis_Z_phi"] = evaluate_vmec_axis_fourier(
            phi_values,
            arrays["raxis_cc"],
            arrays["zaxis_cs"],
            nfp=nfp,
        )

    if not all(key in arrays for key in ("rmnc", "zmns", "xm", "xn")):
        return
    mode_count = int(np.asarray(arrays["xm"]).size)
    rm = _normalise_surface_mode_array(arrays["rmnc"], mode_count=mode_count)
    zm = _normalise_surface_mode_array(arrays["zmns"], mode_count=mode_count)
    if rm.shape != zm.shape:
        return
    surface_index = rm.shape[0] - 1
    arrays["lcfs_R_sections"], arrays["lcfs_Z_sections"] = evaluate_vmec_surface_fourier(
        arrays["geometry_theta"],
        phi_values,
        rm[surface_index],
        zm[surface_index],
        arrays["xm"],
        arrays["xn"],
    )


def _geometry_phi_values(path: Path, values: Sequence[float] | None) -> NDArray[np.float64]:
    if values is not None:
        arr = np.asarray(list(values), dtype=np.float64)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("geometry_phi must contain at least one toroidal angle")
        return np.ascontiguousarray(arr, dtype=np.float64)
    nfp = _read_wout_nfp(path)
    period = 2.0 * np.pi / max(int(nfp), 1)
    return np.ascontiguousarray(np.linspace(0.0, period, 5, endpoint=True, dtype=np.float64)[:-1])


def _read_wout_nfp(path: Path) -> int:
    from netCDF4 import Dataset

    with Dataset(str(path), "r") as ds:
        return int(ds.variables["nfp"][()])


def _ensure_topoquest_import_path(topoquest_root: str | Path | None = None) -> None:
    candidates: list[Path] = []
    if topoquest_root is not None:
        candidates.append(Path(topoquest_root).expanduser())
    env = os.environ.get("TOPOQUEST_ROOT")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(Path(__file__).resolve().parents[3] / "topoquest")
    for root in candidates:
        if root.exists() and str(root) not in sys.path:
            sys.path.insert(0, str(root))
            return


def _array_error(reference: NDArray[np.float64], other: NDArray[np.float64]) -> dict[str, Any]:
    a = np.asarray(reference, dtype=float)
    b = np.asarray(other, dtype=float)
    if a.shape != b.shape:
        return {"status": "shape_mismatch", "reference_shape": list(a.shape), "shape": list(b.shape)}
    diff = b - a
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        return {"status": "no_finite_samples", "shape": list(a.shape)}
    ref_norm = float(np.linalg.norm(a[finite].ravel()))
    diff_norm = float(np.linalg.norm(diff[finite].ravel()))
    return {
        "status": "ok",
        "shape": list(a.shape),
        "max_abs_error": float(np.nanmax(np.abs(diff[finite]))),
        "rms_abs_error": float(math.sqrt(np.nanmean(diff[finite] ** 2))),
        "rel_l2_error": diff_norm / max(ref_norm, 1.0e-300),
    }


def _leading_boozer_modes(arrays: Mapping[str, NDArray[np.float64]], *, count: int) -> list[dict[str, Any]]:
    bmnc = np.asarray(arrays["bmnc_b"], dtype=float)
    xm = np.asarray(arrays["xm_b"], dtype=float)
    xn = np.asarray(arrays["xn_b"], dtype=float)
    amp = np.max(np.abs(bmnc), axis=1)
    order = np.argsort(amp)[::-1][: int(count)]
    return [
        {"m": int(xm[idx]), "n": int(xn[idx]), "max_abs_Bmn": float(amp[idx])}
        for idx in order
    ]


def _stats(values: NDArray[np.float64]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"shape": list(arr.shape), "finite_count": 0}
    return {
        "shape": list(arr.shape),
        "finite_count": int(finite.size),
        "min": float(np.nanmin(finite)),
        "max": float(np.nanmax(finite)),
        "mean": float(np.nanmean(finite)),
        "rms": float(math.sqrt(np.nanmean(finite ** 2))),
    }


def _normalised_radius(phi: NDArray[np.float64] | None, values: NDArray[np.float64] | None) -> NDArray[np.float64] | None:
    if values is None:
        return None
    n = np.asarray(values).shape[0]
    if phi is None:
        return np.linspace(0.0, 1.0, n)
    flux = np.asarray(phi, dtype=float)
    if flux.shape[0] != n:
        return np.linspace(0.0, 1.0, n)
    scale = float(np.nanmax(np.abs(flux)))
    if scale <= 0.0:
        return np.linspace(0.0, 1.0, n)
    return np.abs(flux) / scale


def _int_from_summary(summary: Mapping[str, Any], key: str) -> int | None:
    try:
        value = summary.get("scalars", {}).get(key)
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_from_scalars(scalars: Mapping[str, Any], key: str) -> int | None:
    try:
        value = scalars.get(key)
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_scalar(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        if isinstance(item, (np.integer, int)):
            return int(item)
        if isinstance(item, (np.floating, float)):
            return float(item)
        if isinstance(item, (np.bool_, bool)):
            return bool(item)
        return item
    if arr.size == 1:
        return _json_scalar(arr.reshape(()))
    return value


def _as_float_array(value: Any) -> NDArray[np.float64]:
    arr = np.ma.asarray(value).filled(np.nan)
    return np.asarray(arr, dtype=np.float64)


def _sha256_prefix(path: Path, *, n: int) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def _write_scalar_comparison_csv(comparisons: Mapping[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for reader, payload in comparisons.get("readers", {}).items():
        for key, row in payload.get("scalars", {}).items():
            rows.append(
                {
                    "reader": reader,
                    "key": key,
                    "status": row.get("status"),
                    "reference": row.get("reference"),
                    "value": row.get("value"),
                    "abs_error": row.get("abs_error"),
                    "rel_error": row.get("rel_error"),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("reader", "key", "status", "reference", "value", "abs_error", "rel_error")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "CounterpartResult",
    "VmecBenchmarkReport",
    "compare_counterpart_results",
    "evaluate_vmec_axis_fourier",
    "evaluate_vmec_surface_fourier",
    "plot_boozer_mode_summary",
    "plot_profile_summary",
    "plot_vmec_geometry_summary",
    "public_wout_source",
    "run_vmec_counterpart_benchmark",
    "write_vmec_benchmark_outputs",
]
