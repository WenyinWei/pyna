# Field-Line Tracing Policy

`pyna`/`cyna` is the owner of production magnetic-field-line tracing.

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
