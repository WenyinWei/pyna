# Field-Line Tracing Policy

`pyna`/`cyna` is the owner of production magnetic-field-line tracing.

## CPU-only path is the default

Production tracing does not require an NVIDIA GPU or a CUDA toolkit.  The
mandatory `_cyna_ext` extension always contains the C++ CPU kernels:

- `cyna::coil_field_grid_cpu` (OpenMP Biot-Savart grid fill), exposed as
  `pyna.toroidal.flt.build_coil_field_grid`;
- the C++ RK4 Poincare kernels exposed as `pyna._cyna.trace_poincare_batch`
  and `pyna._cyna.trace_poincare_multi`.

The default `backend="auto"` selects the C++ CPU coil-grid kernel whenever
the extension is importable.  Published PyPI wheels are CPU-only and set
`CYNA_WITH_CUDA=0`; no CUDA compiler is needed to install or run them.

Build the CPU extension from source:

```bash
cd /path/to/pyna
CYNA_WITH_CUDA=0 pip install .          # standard CPU install
CYNA_WITH_CUDA=0 pip install -e .       # editable development install
CYNA_WITH_CUDA=0 python -m build        # CPU-only wheel + sdist
```

For wheel CI, `cibuildwheel` reads the CPU-only configuration from
`pyproject.toml` (`.github/workflows/wheels.yml` does the same):

```bash
CIBW_BUILD="cp312-*" cibuildwheel --output-dir dist
```

CUDA is opt-in only: `CYNA_WITH_CUDA=1` additionally builds the optional
`_cyna_cuda_backend` when `nvcc` is available.  A machine that merely has
`nvcc` on `PATH` does not change a standard CPU install.

## NumPy reference is test-only

The pure-NumPy Biot-Savart implementation in
`pyna.toroidal.flt.coil_field_grid._numpy_reference` is kept for unit tests
and correctness cross-checks.  It is not a supported production tracing
backend.  Every use emits a `RuntimeWarning`; `backend="cyna"` (or `"cpu"`)
raises with a build/install hint when the compiled kernel is missing.

See [`CPU_TRACING_BENCHMARK.md`](CPU_TRACING_BENCHMARK.md) for measured
NumPy-vs-C++ CPU grid and 6-turn x 3-seed Poincare timings.

Use `pyna._cyna.trace_poincare_multi`, `trace_poincare_batch`, or the canonical
wrappers in `pyna.toroidal.flt` for batch Poincare and field-line workloads.
Domain repositories such as topoquest should pass prepared field arrays and
seeds into these APIs instead of reimplementing RK4/section-crossing hot loops
in Python.

The expected split is:

- `cyna`: performance-critical tracing, section crossing, wall-aware tracing,
  and future bulk mesh inverse kernels.
- `pyna`: stable Python-facing tracing APIs, coordinate schemas, and reusable
  magnetic-coordinate builders.
- downstream projects: PDE preprocessing, case-specific state files,
  diagnostics, plots, and acceptance metrics.

Any new PEST/Boozer or magnetic-surface-coordinate workflow that needs field
lines should leave the integration in `pyna`/`cyna`; Python code may fit,
resample, cache, validate, and visualize the returned point clouds.

For nested-surface scaffolds based on Poincare points, prefer direct stitching
of points from the same field line. Curve fitting or smoothing is a separate
post-processing choice and should not be hidden inside the tracing backend.
