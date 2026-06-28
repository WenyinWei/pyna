# `pyna.toroidal.flt` — cyna-backed toroidal field-line tools

This package is the Python-facing wrapper layer for cyna field-line tracing,
Poincare maps, fixed-point searches, and variational/FPT shift helpers.

## Interface convention

New high-level utilities should accept a `VectorFieldCylind`-compatible field
object as the first argument.  Low-level wrappers that take
`R_grid, Z_grid, Phi_grid, BR, BZ, BPhi` are retained for cyna ABI boundaries and
legacy hot-path callers, but should not be the preferred user-facing API.

Use `field_period` for a toroidal angular span.  Use `map_power` for an integer
Poincare-map iterate count and `cycle_length` for the number of section points
in one closed cycle.  New public APIs should avoid an unqualified `period`
because it conflates the angular span with the integer map power.

## FPT verb convention

- `progress_*` fixes `phi_s` and advances only `phi_e`.  Use these functions
  for endpoint objects such as `DX_pol(phi_s, phi_e)` and
  `delta_X_pol(phi_s, phi_e)`.
- `evolve_*_cycle_*` follows a cycle-attached object as the closed-cycle phase
  moves.  For cycle shift, `phi_s` and
  `phi_e = phi_s + map_power * field_period` move together.  Use these
  functions for quantities such as `DPm(phi)` and
  `delta_X_cyc(phi)`.  Prefer `evolve_delta_X_cycle_along_cycle`; the older
  `evolve_delta_X_cycle_along_orbit` spelling is a compatibility alias.

The underlying nonhomogeneous FPT ODE is shared:

```text
d(delta_X)/dphi = d(R B_pol / B_phi)/d(R,Z) * delta_X
                + delta(R B_pol / B_phi)
```

The verb encodes the boundary condition and interpretation, not a different
formula.  `delta_X_pol` progress uses an open-trajectory initial condition at a
chosen source point; `delta_X_cyc` evolution uses the closed-cycle initial
displacement returned by the cycle closure solve.

## Cycle shifts

- `cycle_shift_from_fields` computes the first-order `delta_X_cyc` response of a
  closed field-line cycle for `B0 -> B0 + delta_B`.
- `axis_cycle_shift_from_fields` samples that same shift on an O-cycle axis
  profile.
- `cycle_points_shift_from_fields` applies it to a set of O/X closed-cycle seed
  points and returns their shifted section points.

These functions are pure geometry/FPT utilities over cylindrical vector fields
or compatible field caches.  They do not implement beta-ramp acceptance,
topology gates, amplitude trust regions, or fallback axis models; those policies
belong to the caller.
