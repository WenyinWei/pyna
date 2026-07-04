# `pyna.toroidal.perturbation` — toroidal perturbative-theory landing zone

This package is a lightweight architecture marker for future toroidal
functional perturbation theory work.

## Why it exists

Historically, toroidal perturbative code has been split between:

- `pyna.toroidal.equilibrium` — perturbed equilibrium and force-balance solvers
- `pyna.toroidal.plasma_response` — plasma-response closures / coupled GS solvers
- `pyna.control` — generic FPT for topology response and control

That split is still practical for current code, but it leaves no single honest
namespace for *toroidal perturbative theory as a whole*.  This package provides
that landing zone without overbuilding or moving stable implementations.

## Ownership rule

- Keep **generic dynamical-systems FPT** in `pyna.control`.
- Put **toroidal / MHD-specific perturbation APIs** under
  `pyna.toroidal.perturbation`.
- Keep solver-heavy modules in their existing homes until they naturally need
  consolidation.

## FPT verb convention

Use `progress_*` only when the source phase `phi_s` is fixed and an endpoint
`phi_e` is advanced.  Examples are `DX_pol(phi_s, phi_e)` and
`delta_X_pol(phi_s, phi_e)`.

Use `evolve_*_cycle_*` when the object is attached to a periodic orbit and is
followed as the cycle phase changes.  In that case `phi_s` and
`phi_e = phi_s + 2*pi*m` move together; examples are `DPm(phi)` and
`delta_X_cyc(phi)`.

## Planned buckets

- `equilibrium`
  - finite-β climb / continuation
  - perturbed Grad–Shafranov workflows
  - force-balance correctors and linearisation helpers
- `response`
  - plasma current closures
  - coupled plasma-response solvers
  - vacuum-to-plasma transfer operators
  - response diagnostics and validity checks

## Classical magnetic-spectrum island chains

Use `pyna.toroidal.perturbation_spectrum` when the question is about island-chain
phase, width, or Chirikov overlap under a radial magnetic-perturbation Fourier
spectrum.  This is deliberately separate from FPT cycle-shift APIs: FPT gives
the linear shift of an already identified X/O cycle under one global
perturbation, while the complex resonant coefficient `tilde_b^1_{m,-n}` gives
the natural island-chain phase response:

```python
from pyna import toroidal

tilde_b1 = toroidal.nardon_radial_perturbation(
    R_surf, Z_surf, phi_vals, theta_vals,
    delta_BR, delta_BZ, delta_Bphi, radial_labels,
    denominator_B_phi=B0_phi,
)
spec = toroidal.radial_perturbation_Fourier_spectrum(
    tilde_b1, theta_vals, phi_vals, radial_labels=radial_labels,
)
chains = toroidal.analyze_resonant_island_chains(spec, q_profile, n=2)
theta_O = chains[0].with_phase_shift(phase_shift).fixed_points(phi=0.0)["theta_O"]
```

## Beta-ramp workflow layer

Use `pyna.toroidal.perturbation.beta_ramp` for public, reusable beta-ramp scan
plumbing.  The module does not call private continuation solvers and does not
assume a particular stellarator data layout.  Private projects should adapt
their outputs into a `BetaRampState` containing:

- a cylindrical field grid: `R_grid, Z_grid, Phi_grid, BR, BZ, BPhi`
- magnetic-coordinate surfaces: `R_surf, Z_surf, phi_vals, theta_vals`
- radial labels and either `q_profile` or `iota_profile`
- numeric metadata for solver/tracing health, not paths or internal names

`BetaRampState.from_field_mapping(...)` accepts both pyna field-cache keys
(`R_grid/Z_grid/Phi_grid`, `BR/BZ/BPhi`) and topoquest-style cylindrical payload
keys (`R/Z/Phi`, `B0_R/B0_Z/B0_Phi`).  Set `field_periods=Nfp` when the field
grid stores one native field period; surface coordinates may still span the full
torus and the sampler folds queries modulo `2*pi/Nfp`.

Then diagnose one beta point against a reference:

```python
from pyna.toroidal.perturbation.beta_ramp import (
    BetaRampState,
    beta_scan_summary_rows,
    diagnose_beta_ramp_state,
)

# Project-local adapter code should populate BetaRampState array fields.
base = build_state_from_payload("baseline")
state = build_state_from_payload("beta step")

diag = diagnose_beta_ramp_state(
    state,
    reference=base,
    n_values=[1, 2, 3],
    m_max=24,
    n_max=12,
    small_divisor_tol=3.0e-2,
)
rows = beta_scan_summary_rows([diag])
```

The diagnostic result carries the Nardon `tilde_b^1` spectrum, resonant island
chains, Chirikov overlaps, per-surface `abs(m*iota+n)` reports, nRMP/RMP mode
norms, and a trust label (`ok`, `watch`, or `low-confidence`).  Use the trust
label as a numerical-health gate before interpreting near-limit figures.  The
island-chain finder chooses the resonant Fourier branch from the sign of the
provided q-profile: positive q uses `tilde_b^1_{m,-n}`, while negative q uses
`tilde_b^1_{m,+n}` and records that branch as `chain.coefficient_n`.

For an NCSX beta-ramp artifact layout, point `PYNA_NCSX_ROOT` at the data
directory and run:

```bash
export PYNA_NCSX_ROOT=/path/to/NCSX
python3 scripts/ncsx_magnetic_spectrum_case.py
```

By default, the script writes all generated data under
`$PYNA_NCSX_ROOT/ncsx_magnetic_spectrum_case_20260627_v2/`, not inside the code
repository.

The case script also writes PNG diagnostics by default:

- spectrum heatmap with resonant `(m,-n)` cells highlighted
- radial profiles of `2|tilde_b^1_{m,-n}|`
- PEST-section island-width bars drawn at O-points, with X-points marked
- the same island bars across four toroidal sections
- O-point phase-control scan
- Chirikov overlap bar chart
- a wall/VMEC-LCFS/Poincare overview for NCSX (`Nfp=3`) with island bars

The old `pyna.toroidal.coils.RMP` `rfft2` path has been removed; keep new island
design work on `pyna.toroidal.perturbation_spectrum`.

## Anti-patterns

Avoid placing toroidal perturbative-equilibrium code in:

- `pyna.topo` — topology objects/algorithms only
- compatibility facade namespaces
