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

ESSOS may consume the resulting direction, seed distances, side, generation,
and seed index.  It must not independently infer those quantities from nearest
points during a gradient evaluation.  After an accepted coil step, PyNA/Cyna
rebuilds the record, validates the same label, and only then permits an outer
refresh or relabelling decision.
