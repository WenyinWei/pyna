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

## Anti-patterns

Avoid placing toroidal perturbative-equilibrium code in:

- `pyna.topo` — topology objects/algorithms only
- compatibility facade namespaces
