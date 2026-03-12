"""Benchmark pyna cache speedup.

Tests:
  1. A_matrix: first call vs cached call
  2. DPm: first call vs cached
  3. cycle_shift for 8 coils: with vs without cache
  4. flux_surface: contour extraction timing

Usage:  py -3.13 scripts/cache_benchmark.py
"""

import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyna.mag.solovev import SolovevEquilibrium
from pyna.control._cached_fpt import CachedFPTAnalyzer
from pyna.control.fpt import A_matrix as raw_A_matrix, DPm_axisymmetric

# ─── Setup ────────────────────────────────────────────────────────────────────

def make_field_func(eq):
    def field(rzphi):
        R, Z, phi = float(rzphi[0]), float(rzphi[1]), float(rzphi[2])
        BR, BZ = eq.BR_BZ(R, Z)
        Bphi = eq.Bphi(R)
        Bmag = np.sqrt(BR**2 + BZ**2 + Bphi**2) + 1e-30
        return np.array([BR / Bmag, BZ / Bmag, Bphi / (R * Bmag)])
    return field

def make_coil_field(scale=1e-4):
    def coil(rzphi):
        R = float(rzphi[0])
        return np.array([scale / R, 0.0, 0.0])
    return coil

eq = SolovevEquilibrium(R0=6.2, a=2.0, B0=5.3, kappa=1.7, delta=0.33, q0=1.5)
field_func = make_field_func(eq)

R_test = 6.2
Z_test = 0.0

lines = []

def report(label, t_first, t_cached, n_cached=100):
    speedup = t_first / t_cached if t_cached > 0 else float('inf')
    line = (
        f"{label}\n"
        f"  First call : {t_first*1000:.3f} ms\n"
        f"  Cached     : {t_cached*1000:.4f} ms  (avg of {n_cached})\n"
        f"  Speedup    : {speedup:.1f}x\n"
    )
    print(line, end='')
    lines.append(line)

# ─── Benchmark 1: A_matrix ────────────────────────────────────────────────────
print("=" * 60)
print("pyna cache benchmark")
print("=" * 60)
lines.append("=" * 60 + "\npyna cache benchmark\n" + "=" * 60 + "\n")

analyzer = CachedFPTAnalyzer(field_func, eq_hash_str='bench_eq')

t0 = time.perf_counter()
A = analyzer.A_matrix(R_test, Z_test)
t_first_A = time.perf_counter() - t0

N = 500
t0 = time.perf_counter()
for _ in range(N):
    A2 = analyzer.A_matrix(R_test, Z_test)
t_cached_A = (time.perf_counter() - t0) / N

report("1. A_matrix", t_first_A, t_cached_A, N)

# ─── Benchmark 2: DPm ────────────────────────────────────────────────────────
analyzer2 = CachedFPTAnalyzer(field_func, eq_hash_str='bench_eq2')

t0 = time.perf_counter()
DPm = analyzer2.DPm(R_test, Z_test)
t_first_DPm = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(N):
    DPm2 = analyzer2.DPm(R_test, Z_test)
t_cached_DPm = (time.perf_counter() - t0) / N

report("2. DPm (axisymmetric)", t_first_DPm, t_cached_DPm, N)

# ─── Benchmark 3: cycle_shift for 8 coils ────────────────────────────────────
n_coils = 8
coil_funcs = [make_coil_field(scale=(i + 1) * 1e-4) for i in range(n_coils)]

analyzer3 = CachedFPTAnalyzer(field_func, eq_hash_str='bench_eq3')

t0 = time.perf_counter()
for cf in coil_funcs:
    analyzer3.cycle_shift(R_test, Z_test, cf)
t_first_cs = time.perf_counter() - t0

n_rep = 200
t0 = time.perf_counter()
for _ in range(n_rep):
    for cf in coil_funcs:
        analyzer3.cycle_shift(R_test, Z_test, cf)
t_cached_cs = (time.perf_counter() - t0) / (n_rep * n_coils)

line = (
    f"3. cycle_shift (8 coils)\n"
    f"  First call (all 8 coils) : {t_first_cs*1000:.3f} ms\n"
    f"  Cached per coil          : {t_cached_cs*1000:.4f} ms  (avg of {n_rep*n_coils})\n"
    f"  Speedup per coil         : {(t_first_cs/n_coils)/t_cached_cs:.1f}x\n"
)
print(line, end='')
lines.append(line)

# ─── Benchmark 4: flux_surface ────────────────────────────────────────────────
t0 = time.perf_counter()
R_fs, Z_fs = eq.flux_surface(0.5)
t_fs = time.perf_counter() - t0

line = (
    f"4. flux_surface(psi_norm=0.5)\n"
    f"  Time      : {t_fs*1000:.1f} ms\n"
    f"  N contour : {len(R_fs)} pts\n"
)
print(line, end='')
lines.append(line)

# ─── cache_stats ────────────────────────────────────────────────────────────
stats = analyzer3.cache_stats()
line = f"CachedFPTAnalyzer stats: {stats}\n"
print(line)
lines.append(line)

# ─── Save results ────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'cache_benchmark_results.txt')
with open(out_path, 'w') as f:
    f.writelines(lines)

print(f"\nResults saved to: {out_path}")
