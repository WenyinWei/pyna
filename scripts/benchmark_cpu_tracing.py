#!/usr/bin/env python3
"""Benchmark the CPU coil-field grid and Poincare tracing paths.

This script measures the end-to-end cost of a coil-set field-line trace on a
machine without an NVIDIA GPU:

1. build a cylindrical coil-field grid with the cyna C++ CPU kernel;
2. build the same grid with the test-only NumPy reference;
3. trace a 6-turn x 3-seed Poincare section with the C++ tracer using each
   grid.

The tracer itself is always the cyna C++ kernel (there is no production NumPy
tracer).  The comparison therefore isolates the grid-construction cost and
shows the end-to-end difference a NumPy-only user would see.

Usage::

    python scripts/benchmark_cpu_tracing.py
    python scripts/benchmark_cpu_tracing.py --repeats 5 --skip-numpy

The reference backend emits a RuntimeWarning by design; this benchmark
suppresses it because the warning contract is covered by the test suite.
"""

from __future__ import annotations

import argparse
import os
import platform
import time
import warnings
from dataclasses import dataclass

import numpy as np

from pyna._cyna import is_available
from pyna.toroidal.flt import build_coil_field_grid, trace_poincare_batch

GRID_CASES = ((32, 32, 32), (48, 48, 32), (64, 64, 48))


@dataclass
class BenchRow:
    shape: tuple[int, int, int]
    cpp_grid_s: float
    numpy_grid_s: float
    cpp_trace_s: float
    numpy_trace_s: float
    rel_field_error: float
    counts_ok: bool

    @property
    def cpp_e2e_s(self) -> float:
        return self.cpp_grid_s + self.cpp_trace_s

    @property
    def numpy_e2e_s(self) -> float:
        return self.numpy_grid_s + self.numpy_trace_s


def _coil_set(n_coils: int, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Closed poloidal coils distributed toroidally (benchmark geometry)."""

    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    coils = []
    for k in range(n_coils):
        phi = 2.0 * np.pi * k / n_coils
        major = 1.55 + 0.35 * np.cos(2.0 * phi)
        minor = 0.13 + 0.04 * np.sin(3.0 * phi)
        radius = major + minor * np.cos(theta)
        z = minor * np.sin(theta)
        coils.append(
            np.stack(
                [radius * np.cos(phi), radius * np.sin(phi), z],
                axis=-1,
            )
        )
    segments = np.stack(coils, axis=0)
    currents = np.full(len(segments), 1.0e5)
    return segments, currents


def _axes(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_r, n_z, n_phi = shape
    return (
        np.linspace(1.15, 2.45, n_r),
        np.linspace(-0.55, 0.55, n_z),
        np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False),
    )


def _build(segments, currents, axes, backend: str) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return build_coil_field_grid(
            segments,
            currents,
            axes[0],
            axes[1],
            axes[2],
            nfp=1,
            backend=backend,
        )


def _best_of(fn, repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def _trace(grid: dict, seeds_r: np.ndarray, seeds_z: np.ndarray, turns: int, dphi: float):
    return trace_poincare_batch(
        seeds_r,
        seeds_z,
        0.0,
        turns,
        dphi,
        grid["R_grid"],
        grid["Z_grid"],
        grid["Phi_grid"],
        grid["BR_flat"],
        grid["BZ_flat"],
        grid["BPhi_flat"],
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        nfp=grid["nfp"],
    )


def _trace_counts_ok(raw, turns: int) -> bool:
    counts = np.asarray(raw[0])
    return bool(counts.shape[0] == 3 and np.all(counts == turns))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3, help="C++ repetitions (best-of)")
    parser.add_argument(
        "--numpy-repeats", type=int, default=1, help="NumPy reference repetitions (best-of)"
    )
    parser.add_argument("--coils", type=int, default=16, help="number of coils")
    parser.add_argument("--points", type=int, default=97, help="polyline points per coil")
    parser.add_argument("--turns", type=int, default=6, help="Poincare turns per seed")
    parser.add_argument("--dphi", type=float, default=0.01, help="RK4 angular step")
    parser.add_argument("--skip-numpy", action="store_true", help="C++ only")
    args = parser.parse_args()

    if not is_available():
        print("ERROR: the cyna C++ extension is not importable.", file=os.sys.stderr)
        return 1
    if args.repeats < 1 or args.numpy_repeats < 1:
        parser.error("repeats must be >= 1")

    segments, currents = _coil_set(args.coils, args.points)
    seeds_r = np.array([1.65, 1.70, 1.75])
    seeds_z = np.array([0.0, 0.05, -0.05])

    print(
        f"platform={platform.platform()} python={platform.python_version()} "
        f"numpy={np.__version__} cpus={os.cpu_count()}"
    )
    print(
        f"coils={args.coils}x{args.points - 1} segments  "
        f"trace={args.turns} turns x {len(seeds_r)} seeds  DPhi={args.dphi}"
    )

    rows: list[BenchRow] = []
    for shape in GRID_CASES:
        axes = _axes(shape)
        cpp_grid = _build(segments, currents, axes, "cyna")
        cpp_grid_s = _best_of(
            lambda: _build(segments, currents, axes, "cyna"), args.repeats
        )

        if args.skip_numpy:
            numpy_grid_s = float("nan")
            numpy_trace_s = float("nan")
            rel_error = 0.0
            numpy_grid = None
        else:
            numpy_grid = _build(segments, currents, axes, "numpy")
            numpy_grid_s = _best_of(
                lambda: _build(segments, currents, axes, "numpy"),
                args.numpy_repeats,
            )
            scale = max(
                1.0e-30,
                float(np.max(np.abs(numpy_grid["BR"]))),
                float(np.max(np.abs(numpy_grid["BZ"]))),
            )
            rel_error = max(
                float(np.max(np.abs(cpp_grid[key] - numpy_grid[key]))) / scale
                for key in ("BR", "BZ", "BPhi")
            )

        cpp_raw = _trace(cpp_grid, seeds_r, seeds_z, args.turns, args.dphi)
        cpp_trace_s = _best_of(
            lambda: _trace(cpp_grid, seeds_r, seeds_z, args.turns, args.dphi),
            args.repeats,
        )
        if numpy_grid is None:
            numpy_trace_s = float("nan")
            counts_ok = _trace_counts_ok(cpp_raw, args.turns)
        else:
            numpy_raw = _trace(numpy_grid, seeds_r, seeds_z, args.turns, args.dphi)
            numpy_trace_s = _best_of(
                lambda: _trace(numpy_grid, seeds_r, seeds_z, args.turns, args.dphi),
                args.repeats,
            )
            counts_ok = _trace_counts_ok(cpp_raw, args.turns) and _trace_counts_ok(
                numpy_raw, args.turns
            )

        rows.append(
            BenchRow(
                shape=shape,
                cpp_grid_s=cpp_grid_s,
                numpy_grid_s=numpy_grid_s,
                cpp_trace_s=cpp_trace_s,
                numpy_trace_s=numpy_trace_s,
                rel_field_error=rel_error,
                counts_ok=counts_ok,
            )
        )

    print()
    print("| grid (nR x nZ x nPhi) | cyna grid (s) | NumPy grid (s) | grid speedup | "
          "cyna trace (s) | NumPy-grid trace (s) | cyna e2e (s) | NumPy-grid e2e (s) | e2e speedup | "
          "relative field error |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        if np.isnan(row.numpy_grid_s):
            fields = [
                str(row.shape),
                f"{row.cpp_grid_s:.3f}",
                "skipped",
                "-",
                f"{row.cpp_trace_s:.4f}",
                "-",
                f"{row.cpp_e2e_s:.3f}",
                "-",
                "-",
                "-",
            ]
        else:
            fields = [
                str(row.shape),
                f"{row.cpp_grid_s:.3f}",
                f"{row.numpy_grid_s:.3f}",
                f"{row.numpy_grid_s / row.cpp_grid_s:.1f}x",
                f"{row.cpp_trace_s:.4f}",
                f"{row.numpy_trace_s:.4f}",
                f"{row.cpp_e2e_s:.3f}",
                f"{row.numpy_e2e_s:.3f}",
                f"{row.numpy_e2e_s / row.cpp_e2e_s:.1f}x",
                f"{row.rel_field_error:.1e}",
            ]
        print("| " + " | ".join(fields) + " |")

    print()
    print("Notes:")
    print("- e2e = coil-field grid build + 6-turn x 3-seed Poincare trace.")
    print("- The trace kernel is C++ in both columns; only the grid backend differs.")
    print("- NumPy reference emits RuntimeWarning in normal use; suppressed here.")
    print("- relative field error = max |cyna - numpy| / max(|numpy|) over BR, BZ, BPhi.")
    if not all(row.counts_ok for row in rows):
        print("ERROR: at least one trace did not complete all requested turns.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
