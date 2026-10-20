# CPU tracing benchmark: cyna C++ vs NumPy reference

This benchmark answers the CPU-only question: how much does the C++ coil-field
grid builder save for a normal field-line workflow when no NVIDIA GPU is
available?  The Poincare tracer itself is the cyna C++ kernel in both cases;
the only difference is whether the coil field grid was built by
`cyna::coil_field_grid_cpu` or by the test-only NumPy reference.

## Reproduce

```bash
cd /path/to/pyna
CYNA_WITH_CUDA=0 pip install -e .
python scripts/benchmark_cpu_tracing.py --repeats 3 --numpy-repeats 1
```

## Workload

| item | value |
|---|---|
| coil set | 16 closed poloidal coils x 96 straight segments |
| currents | 1.0e5 A-turns per coil |
| grid cases | 32 x 32 x 32, 48 x 48 x 32, 64 x 64 x 48 |
| grid layout | (nR, nZ, nPhi), phi fastest, one field period (nfp=1) |
| trace | 6 turns x 3 seeds, DPhi = 0.01, C++ RK4 Poincare kernel |
| seeds | R = 1.65, 1.70, 1.75 m; Z = 0.00, 0.05, -0.05 m |
| timing | best of 3 C++ runs; best of 1 NumPy run |
| machine | WSL2 Linux 6.18.33.2-microsoft-standard-WSL2, 32 logical CPUs |
| Python | 3.12.3, NumPy 2.4.6, cyna CPU build (`--with-cuda=n`) |

The C++ grid kernel parallelises over (R, Z) rows with OpenMP; the NumPy
reference is a single-process vectorised implementation.

## Results

| grid (nR x nZ x nPhi) | cyna grid (s) | NumPy grid (s) | grid speedup | cyna trace (s) | NumPy-grid trace (s) | cyna e2e (s) | NumPy-grid e2e (s) | e2e speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 x 32 x 32 | 0.076 | 4.817 | 63.4x | 0.0050 | 0.0050 | 0.081 | 4.822 | 59.6x |
| 48 x 48 x 32 | 0.165 | 9.970 | 60.6x | 0.0040 | 0.0040 | 0.169 | 9.974 | 59.2x |
| 64 x 64 x 48 | 0.420 | 26.847 | 64.0x | 0.0029 | 0.0053 | 0.423 | 26.853 | 63.6x |

`e2e` is the end-to-end tracing time: grid build plus the 6-turn x 3-seed
Poincare trace.  For this workload the trace is only a few milliseconds, so
the grid build dominates.

## Interpretation

- The CPU C++ path produces the production 64 x 64 x 48 grid in about 0.4 s;
  the NumPy reference needs about 27 s on the same 32-CPU machine.
- End-to-end, the C++ path is 60-70x faster for this trace size.  The gap
  grows with coil count and grid size because the NumPy reference materialises
  temporary arrays of shape `(chunk, n_coils, n_segments, 3)`.
- The produced fields agree to `max|cyna - numpy| / max|numpy| < 5e-7` over
  `BR`, `BZ`, `BPhi` in these runs.  Unit tests additionally check the
  analytic circular-loop field and a literal pivot2t formula at tighter
  tolerances.
- These numbers are hardware- and thread-count-dependent.  Re-run the script
  on the target machine rather than treating the table as a universal
  performance contract.

## Why the NumPy path remains

It is a readable independent implementation used by
`tests/test_coil_field_grid.py` to catch C++ regressions.  It is not exposed
as a supported production backend: `build_coil_field_grid` emits a
`RuntimeWarning` whenever it is selected.  `backend="cyna"` or `"cpu"`
raises a build/install hint instead of silently falling back.
