# Manifold-aware Stage-2 coil optimization

This document fixes the ownership boundary between PyNA and downstream coil
optimization packages such as ESSOS.

## Ownership

- **ESSOS owns design:** coil parameterizations, coil currents, Biot--Savart
  evaluation, Stage-1 target data, engineering losses, constraint handling,
  optimizer state, and continuation weights.
- **PyNA owns topology:** magnetic field-line integration, fixed-toroidal-angle
  maps, periodic-orbit continuation, monodromy, stable and unstable manifolds,
  wall intersections, and topology validation.
- The dependency direction is ESSOS to PyNA.  PyNA must not import ESSOS.

ESSOS supplies a Cartesian field callable

```python
field_fn(xyz, params) -> (Bx, By, Bz)
```

where `params` may be any JAX PyTree.  PyNA converts that field to the
fixed-toroidal-angle equations and integrates the map.  This interface also
supports other downstream field providers without introducing package-specific
types into PyNA.

## Differentiable and validation paths

The optional JAX path is used inside a Stage-2 optimization to compute smooth,
discretely differentiable return-map and topology objectives.  The Cyna path
remains authoritative for production batch tracing, wall intersections, and
accepted-step validation.

The initial differentiable API is:

```python
from pyna.toroidal.flt.jax_poincare import poincare_map

rz1 = poincare_map(
    field_fn,
    params,
    rz0,
    phi_start=phi0,
    phi_span=2 * pi / nfp,
    n_steps=256,
)
```

The map duration is deliberately explicit.  A caller must decide whether its
topological return is one field period, multiple field periods, or one full
toroidal turn.

## Initial scope

The first backend assumes that geometric toroidal angle is monotone along the
traced field line, so `B_phi` must not vanish.  It supports forward and backward
maps through signed `phi_span`.  Strongly shaped cases without a monotone
toroidal angle will require a later Cartesian periodic-curve formulation.

No wall-hit event or nearest-manifold-branch operation is placed in the inner
differentiable path at this stage.  Those non-smooth observables are evaluated
by PyNA/Cyna in the outer validation loop until smooth surrogates and persistent
branch tracking are available.

## Periodic points

`pyna.topo.jax_periodic.solve_periodic_point` solves

```text
P^m(x, params) - x = 0
```

with a fixed-count Newton method.  Its custom JVP applies the implicit-function
theorem directly to the converged root equation, so gradients do not pass
through the Newton iterations.  The initial guess is therefore nondifferentiable
outer-loop state.  The accompanying `periodic_point_state` reports the residual,
monodromy, Greene residue, eigenpairs, and hyperbolicity margin.  The outer
Stage-2 loop must reject or refresh a point whose residual is not converged.

Eigenvectors are stop-gradient diagnostic data for outer-loop manifold branch
seeding.  The root, monodromy invariants, and eigenvalues remain differentiable;
this avoids assigning a spurious derivative to eigenvector ordering and sign.

This formulation is intended for isolated elliptic or hyperbolic points.  It
must not be used at a parabolic point where `M-I` is singular; those topologies
require a later augmented continuation system.

## Invariant-manifold generations

`pyna.topo.jax_manifold` is a sibling backend to the production Cyna tracer;
it does not replace or wrap Cyna.  A differentiable inner-loop branch can be
built as follows:

```python
from pyna.topo.jax_manifold import (
    fundamental_segment_distances,
    hyperbolic_directions,
    manifold_seed_segment,
    trace_manifold_generations,
)

directions = hyperbolic_directions(state.monodromy)
distances = fundamental_segment_distances(
    directions.unstable_expansion,
    minimum_distance=1e-6,
    maximum_distance=1e-3,
    n_seeds=24,
)
seeds = manifold_seed_segment(
    state.position,
    directions.unstable_direction,
    distances,
    side=1,
)
generations = trace_manifold_generations(
    field_fn,
    params,
    seeds,
    stability="unstable",
    phi_span=2 * pi / nfp,
    n_generations=12,
)
```

The returned array has axes `(generation, seed, RZ)`, including the initial
seed segment at generation zero.  Stable branches use the negative-time map;
their discrete inverse-map anchor must be refreshed in the outer loop before
the seeds are constructed.

The two backend tracks have distinct jobs:

- JAX supplies fixed-shape, differentiable generations for the Stage-2 inner
  objective.  JAX may itself execute on a CUDA device when a CUDA-enabled JAX
  runtime is installed.
- Cyna, on either its CPU or CUDA build, supplies high-throughput wall-aware
  branches, strike points, and accepted-step validation.
- `compare_cyna_manifold_generations` compares the same fixed seed and
  generation labels across both discretizations.  A failed count or tolerance
  check invalidates the inner-loop candidate and triggers an outer refresh.

These paths are parallel implementations of the same labelled return-map
problem; they need not execute concurrently.  Eigensystem branch selection,
orientation references, seed distances, side labels, and sample
correspondences remain frozen between outer refreshes.  Gradients continue
through the X-line position and all JAX field-line integrations.

## Outer correspondence and acceptance

The production manifold payload preserves both notions of side:

- `*_point_side` is the physical side occupied on a particular generation;
- `*_point_seed_side` is the initial signed seed segment used by the JAX
  branch.

They differ on every odd generation when the relevant map eigenvalue is
negative.  Keeping both prevents an orientation-reversing saddle from silently
switching the ESSOS branch label.

`pyna.topo.manifold_correspondence.manifold_branch_reference_from_trace`
converts one Cyna payload into a dense `(generation, seed, RZ)` reference.  The
record includes X-line identity, section and map span, stable or unstable
anchor, normalized direction, expansion, initial side, seed orders and
distances, and an availability mask for samples terminated by a wall or domain
limit.  A direction from the preceding accepted state can be supplied to lock
the eigenvector sign and reject a branch whose tangent no longer aligns.

When retracing a retained subset of a seed segment,
`trace_fixed_point_manifolds_field` accepts `seed_orders` alongside explicit
`seed_distances`.  The orders are production identities, not dense array
indices: a retained pair labelled 3 and 7 remains 3 and 7 on the candidate
field.  Empty wall- or domain-limited traces still return every label array
with zero length so disappearance is handled by the correspondence gate.

PyNA also owns the two operations that change or validate inner-loop labels:

- `nearest_available_manifold_sample` chooses a target label from a production
  trace and reports the corresponding JAX seed-axis index.
- `compare_jax_manifold_branch` compares all available fixed labels and rejects
  excessive JAX/Cyna deviation, non-finite JAX output, or incomplete production
  traces when completeness is required.
- `refresh_manifold_sample_match` carries the exact selected label to a new
  production trace.  It returns a structured rejection instead of choosing a
  nearest replacement when identity changes, the sample disappears, or its
  displacement exceeds the outer trust limit.

`pyna.topo.manifold_refresh.refresh_manifold_branch_reference_field` performs
the production half of an accepted-step refresh.  Starting from the preceding
branch anchor, it continues the X-point on the candidate cylindrical field,
recomputes its Cyna monodromy, requires it to remain hyperbolic and inside an
anchor trust radius, retraces the same sparse seed orders, and locks the new
eigendirection against the preceding tangent.  Backend absence is an
environment error; loss of convergence, hyperbolicity, domain coverage, or
correspondence is returned as a structured `ManifoldProductionRefreshReport`
rejection for optimizer rollback.

ESSOS may consume the resulting direction, seed distances, side, generation,
and seed index.  It must not independently infer those quantities from nearest
points during a gradient evaluation.  After an accepted coil step, PyNA/Cyna
rebuilds the record, validates the same label, and only then permits an outer
refresh or relabelling decision.

## First-wall strike identity

`pyna.topo.manifold_strike` extends the same fixed-label rule from section
samples to divertor strikes.  `manifold_branch_strike_seed_bundle` converts one
accepted branch reference into the existing ordered Cyna strike-seed contract,
including the sparse production seed orders.  `select_manifold_strike_match`
chooses a resolved first-wall hit by physical Cartesian distance to a target;
its identity is `(branch label, trace direction, seed order)`, not the current
dense-array index.  The match records its physical distance metric explicitly:
`rz` ignores the symmetry-degenerate toroidal phase for an axisymmetric
tokamak, while `xyz` retains the full three-dimensional hit position for QA or
other non-axisymmetric configurations.  Selection and refresh must use the
same metric.

On a candidate field, `refresh_manifold_strike_match` looks up that exact seed
order and requires it to produce a Cyna first-wall hit within a physical 3-D
trust radius.  A missing seed, unresolved wall hit, changed branch identity, or
excessive strike motion is a structured rejection.  It never substitutes a
nearby strike.  This module contains no JAX dependency: it is the production
selection and refresh half of the interface.

`pyna.topo.jax_strike` supplies the parallel differentiable event solve.  Given
a Cartesian field, one frozen launch seed, a smooth wall level set `g`, and the
production hit angle as an initial guess, it solves
`g(R(phi), Z(phi), phi) = 0` in a fixed local angle window.  Its custom JVP uses
the implicit-function theorem, so gradients include both launch-point motion
and the moving intersection angle without differentiating through Newton
iterations.  Residual, window, and transversality diagnostics are returned;
the outer loop must reject a failed or tangent crossing and compare every
accepted JAX hit with the exact Cyna first-wall hit.  The discrete questions of
which seed hits and whether this is the first hit remain production decisions.

For a sampled tokamak or QA wall, `local_wall_plane_from_strike` continuously
projects the Cyna hit onto the toroidal surface and freezes its local tangent
plane.  If the wall stores one field period, the projected point and normal are
rotated back to Cyna's unwrapped physical hit angle.  This gives JAX a smooth
local level set without pretending that discrete wall-cell or closest-segment
selection is differentiable.  The plane is valid only inside the outer strike
trust region and is rebuilt after each accepted production refresh.

`trace_manifold_strike` closes the differentiable chain.  It continues the
periodic X-line on the live field, reconstructs the frozen sparse seed order
from the accepted eigendirection and distance, and sends that moving seed to
the local implicit wall event.  Thus coil derivatives include X-line motion,
manifold launch motion, field-line evolution, and motion of the strike angle.
The function rejects mismatched branch, direction, seed, or wall-plane labels
before tracing; its periodic-point and wall-event convergence diagnostics still
have to pass at the outer boundary.
