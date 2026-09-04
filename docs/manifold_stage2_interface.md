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
