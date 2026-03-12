"""Benchmark: CPU serial vs CPU parallel vs GPU CUDA field-line tracing.

Traces 100 field lines of an analytic Solov'ev equilibrium.

Usage
-----
    py -3.13 scripts/benchmark_flt.py
"""
from __future__ import annotations

import sys
import time
import pathlib
import numpy as np

# Allow running from repo root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pyna.flt import FieldLineTracer, get_backend

# ---------------------------------------------------------------------------
# Solov'ev equilibrium â€?analytic field function (for CPU tracer)
# ---------------------------------------------------------------------------

R0, a, B0, q0 = 1.0, 0.3, 1.0, 2.0

def solovev_field(rzphi: np.ndarray) -> np.ndarray:
    """Unit tangent vector of Solov'ev field at rzphi = [R, Z, phi]."""
    R, Z, phi = rzphi
    lam = B0 * a / (q0 * R0)
    dpsi_dR = (R * R - R0 * R0) * R / (R0 * R0 * a * a)
    dpsi_dZ = 2.0 * Z / (a * a)
    BR   = -lam / R * dpsi_dZ
    BZ   =  lam / R * dpsi_dR
    Bphi = B0 * R0 / R
    Bmag = np.sqrt(BR**2 + BZ**2 + Bphi**2) + 1e-30
    return np.array([BR / Bmag, BZ / Bmag, Bphi / (R * Bmag)])


# ---------------------------------------------------------------------------
# Starting points (100 lines on a circle around magnetic axis)
# ---------------------------------------------------------------------------

N_LINES = 100
T_MAX   = 20.0      # arc-length limit
DT      = 0.04
RZLIMIT = (R0 - 1.5*a, R0 + 1.5*a, -1.5*a, 1.5*a)

thetas = np.linspace(0, 2 * np.pi, N_LINES, endpoint=False)
starts = np.column_stack([
    R0 + 0.1 * a * np.cos(thetas),
    0.1 * a * np.sin(thetas),
    np.zeros(N_LINES),
])

results: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# 1. CPU serial (n_workers=1)
# ---------------------------------------------------------------------------
print("Running CPU serial (n_workers=1)â€?)
tracer_serial = FieldLineTracer(solovev_field, dt=DT, RZlimit=RZLIMIT, n_workers=1)
t0 = time.perf_counter()
trajs_serial = tracer_serial.trace_many(starts, T_MAX, n_workers=1)
t_serial = time.perf_counter() - t0
results['cpu_serial'] = {'time': t_serial, 'speedup': 1.0}
print(f"  Serial:   {t_serial:.3f}s  (speedup 1.00Ã—)")

# ---------------------------------------------------------------------------
# 2. CPU parallel ThreadPool (n_workers=8)
# ---------------------------------------------------------------------------
print("Running CPU parallel ThreadPool (n_workers=8)â€?)
tracer_parallel = FieldLineTracer(solovev_field, dt=DT, RZlimit=RZLIMIT, n_workers=8)
t0 = time.perf_counter()
trajs_parallel = tracer_parallel.trace_many(starts, T_MAX, n_workers=8)
t_parallel = time.perf_counter() - t0
speedup_parallel = t_serial / t_parallel
results['cpu_parallel_8'] = {'time': t_parallel, 'speedup': speedup_parallel}
print(f"  Parallel: {t_parallel:.3f}s  (speedup {speedup_parallel:.2f}Ã—)")

# ---------------------------------------------------------------------------
# 3. GPU CUDA
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    print("Running GPU CUDAâ€?)
    cuda_tracer = get_backend(
        'cuda', R0=R0, a=a, B0=B0, q0=q0,
        dt=DT, RZlimit=RZLIMIT,
    )
    # Warm-up
    _ = cuda_tracer.trace_many(starts[:2], T_MAX)
    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()
    trajs_cuda = cuda_tracer.trace_many(starts, T_MAX)
    t_cuda = time.perf_counter() - t0
    speedup_cuda = t_serial / t_cuda
    results['cuda'] = {'time': t_cuda, 'speedup': speedup_cuda}
    print(f"  CUDA:     {t_cuda:.3f}s  (speedup {speedup_cuda:.2f}Ã—)")
except Exception as exc:
    results['cuda'] = {'time': None, 'speedup': None, 'error': str(exc)}
    print(f"  CUDA:     SKIPPED ({exc})")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
lines = [
    "=" * 55,
    "Field-line tracer benchmark",
    f"  N_lines={N_LINES}  t_max={T_MAX}  dt={DT}",
    "=" * 55,
]
for key, v in results.items():
    if v['time'] is None:
        lines.append(f"  {key:<22}  SKIPPED  ({v.get('error', '')})")
    else:
        lines.append(f"  {key:<22}  {v['time']:.4f}s   speedup {v['speedup']:.2f}Ã—")
lines.append("=" * 55)

summary = "\n".join(lines)
print("\n" + summary)

out_path = pathlib.Path(__file__).parent / "benchmark_results.txt"
out_path.write_text(summary + "\n")
print(f"\nResults saved to {out_path}")
