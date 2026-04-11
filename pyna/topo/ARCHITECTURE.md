"""
pyna.topo  — Topology of Dynamical Systems
==========================================

Design philosophy
-----------------
The class hierarchy mirrors the mathematical structure of a dynamical system,
not the implementation details.  Each abstraction corresponds to a precise
mathematical concept, with clean layering from abstract to concrete.

Layer 0:   Phase space and dynamics
Layer 0.5: Base hierarchy (InvariantSet, InvariantManifold, SectionCuttable)
Layer 1:   Invariant objects (orbits, tori, manifolds)
Layer 2:   Structure (how invariants are organised relative to each other)
Layer 3:   Sections (how invariants appear on Poincaré sections)
Layer 4:   Chains and families (discrete-time organisation)
Layer 5:   Plots (visual representation)
Layer 6:   Regularity / chaos diagnostics


═══════════════════════════════════════════════════════════════════════════
LAYER 0 — Phase space and dynamics
═══════════════════════════════════════════════════════════════════════════

  PhaseSpace(dim: int)
    The ambient space where the dynamics lives.
    Examples: 2D (R,Z) for MCF field lines, 4D (R,Z,pR,pZ) for guiding centre,
              6N for N-body celestial mechanics, 12D for 2 charged particles.

  DynamicalSystem(phase_space: PhaseSpace)
    The flow φ^t: PhaseSpace → PhaseSpace.
    Subclasses:
      ContinuousFlow         d/dt x = f(x, t)
        HamiltonianFlow      H(q,p) = const, symplectic
        MagneticFieldLine    B·∇x = 0 (field-line ODE)
      DiscreteMap            x → P(x)
        StandardMap          Chirikov standard map
        PoincareMap(flow, section)  P = return map to Section
        MCFPoincareMap       cyna-accelerated MCF field-line Poincaré map
        GeneralPoincareMap   arbitrary flow + section


═══════════════════════════════════════════════════════════════════════════
LAYER 0.5 — Base hierarchy (pyna/topo/_base.py)
═══════════════════════════════════════════════════════════════════════════

  InvariantSet (ABC, root)
    Any invariant geometric object:  φ^t(S) ⊆ S  or  P(S) = S.
    Pure interface, no state fields.  Has concrete defaults:
      .label           → None
      .ambient_dim     → None
      .diagnostics()   → {'invariant_type': cls.__name__}
      .section_cut()   → raises NotImplementedError
    Alias: InvariantObject = InvariantSet (backward compat)

  InvariantManifold(InvariantSet)
    An invariant set with a well-defined intrinsic dimension.
      .intrinsic_dim   → int | None
      .codim           → ambient_dim − intrinsic_dim (None if unknown)
    Subclasses:
      FixedPoint       intrinsic_dim = 0
      Cycle            intrinsic_dim = 1
      InvariantTorus   intrinsic_dim = len(rotation_vector)
      StableManifold   intrinsic_dim = dim(E^s)
      UnstableManifold intrinsic_dim = dim(E^u)

  SectionCuttable (Protocol, mixin)
    Structural typing for objects that support ``section_cut(section) → list``.
    Not a subclass of InvariantSet — can be mixed in freely.
    Usage:  ``if isinstance(obj, SectionCuttable): ...``


═══════════════════════════════════════════════════════════════════════════
LAYER 1 — Invariant objects
═══════════════════════════════════════════════════════════════════════════

  FixedPoint(InvariantManifold, intrinsic_dim=0)
    A fixed point of a discrete map (period-m Poincaré intersection).
    Dimension-agnostic via ``coords: ndarray(d,)``.
    MCF backward-compat: .R, .Z, .phi (auto-synced with coords).
    Fields:
      coords: ndarray          generic phase-space coordinates
      DPm: ndarray (d, d)      monodromy matrix
      kind: str                'X' (hyperbolic) or 'O' (elliptic)
      coordinate_names: tuple  human-readable axis labels
      section_angle: float     domain-agnostic alias for phi
    Properties:
      monodromy → MonodromyData
      stability → Stability
      greene_residue → float
      intrinsic_dim = 0

  Cycle(InvariantManifold, intrinsic_dim=1)
    A periodic orbit (continuous-time cycle).
    Fields:
      winding: tuple[int, ...]   resonance numbers (m, n)
      sections: Dict[phi → List[FixedPoint]]
      monodromy: MonodromyData | None

  InvariantTorus(InvariantManifold)
    A KAM torus (quasi-periodic orbit) or rational torus.
    intrinsic_dim = len(rotation_vector).
    Fields:
      rotation_vector: tuple[float, ...]

  StableManifold(InvariantManifold)
    Stable manifold of a hyperbolic cycle.
    intrinsic_dim = number of stable eigenvalue directions.

  UnstableManifold(InvariantManifold)
    Unstable manifold of a hyperbolic cycle.
    intrinsic_dim = number of unstable eigenvalue directions.


═══════════════════════════════════════════════════════════════════════════
LAYER 2 — Structure: how invariants are organised
═══════════════════════════════════════════════════════════════════════════

  Tube(InvariantSet)
    A magnetic island = one nested invariant-torus structure.
    Contains a FAMILY of invariant tori (from centre O-cycle to separatrix).
    Skeleton (lazy):
      o_cycle: Cycle              the O-type orbit at the core
      x_cycles: List[Cycle]       the X-type orbit(s) at the boundary
    Methods:
      section_cut(section) → List[Island]    cut with a Section
      manifold_cut(section) → ManifoldSection  cut W^u/W^s with Section

  TubeChain(InvariantSet)
    ALL Tubes of a given resonance m/n in the system.
    = The complete resonance zone (all islands of this resonance).
    Contains:
      tubes: List[Tube]
    Methods:
      section_cut(section) → IslandChain
      wire_skeletons()      set o_cycle/x_cycles on all Tubes


═══════════════════════════════════════════════════════════════════════════
LAYER 3 — Sections: how invariants appear on Poincaré sections
═══════════════════════════════════════════════════════════════════════════

  Section                                [abstract, any codim-1 surface]
    Subclasses:
      ToroidalSection(phi)               φ = const
      HyperplaneSection(a, c, n)         a·x = c
      ParametricSection(f, grad, n)      f(x) = 0


═══════════════════════════════════════════════════════════════════════════
LAYER 4 — Discrete-time organisation
═══════════════════════════════════════════════════════════════════════════

  Island(InvariantSet)
    The cross-section of one Tube at one Section.
    Attributes:
      O_point: FixedPoint               elliptic fixed point of P^m
      X_points: List[FixedPoint]        hyperbolic fixed points nearby
      child_chains: List[IslandChain]   sub-chains nested inside
    Connectivity:
      step(): Island    P^1 image
      step_back(): Island    P^{-1} image

  IslandChain(InvariantSet)
    All Islands of a given resonance at one Section.
    = TubeChain.section_cut(section)


═══════════════════════════════════════════════════════════════════════════
LAYER 5 — Plots
═══════════════════════════════════════════════════════════════════════════

  pyna.plot
    plot_island, plot_island_chain, plot_tube_section, plot_tube_chain_section


═══════════════════════════════════════════════════════════════════════════
LAYER 6 — Regularity / chaos diagnostics (pyna/topo/regularity.py)
═══════════════════════════════════════════════════════════════════════════

  spectral_regularity(DPk_sequence) → float
    Regularity index from [DP^1, …, DP^m]. Near 0 = regular, large = chaotic.

  spectral_regularity_single(eigenvalues) → float
    Quick estimate from one monodromy matrix.

  classify_orbit(eigenvalue_evolution) → str
    'regular' | 'resonant' | 'weakly_chaotic' | 'strongly_chaotic'

  hessian_regularity(D2Pm) → float
    Second-order variational diagnostic.

  MonodromyData.spectral_regularity(DPk_sequence) → float
    Convenience method on MonodromyData.


═══════════════════════════════════════════════════════════════════════════
KEY PRINCIPLES
═══════════════════════════════════════════════════════════════════════════

1. SEPARATION OF EXISTENCE AND OBSERVATION
   InvariantSet = the mathematical object exists.
   SectionCuttable = how you observe it on a section (optional).
   Not every invariant can be meaningfully sectioned.

2. INTRINSIC DIMENSION AS A FIRST-CLASS CONCEPT
   InvariantManifold provides .intrinsic_dim and .codim.
   Enables generic algorithms: "for all manifolds of dimension ≤ 1, compute …"

3. DIMENSION-AGNOSTIC COORDINATES
   FixedPoint.coords: ndarray — works for 2D MCF, 6D celestial mechanics,
   12D charged-particle dynamics.  MCF (R, Z) is a convenience alias.

4. LAZINESS EVERYWHERE
   Skeleton (o_cycle, x_cycles), manifolds, section cuts, connectivity —
   all computed on demand and cached.

5. BACK-REFERENCES (bottom-up)
   Island → Tube → (TubeChain, PeriodicOrbit)

6. SECTION INDEPENDENCE
   The same Tube can be cut at any Section to yield Islands.

7. SPECTRAL REGULARITY AS A DIAGNOSTIC
   DX_t, DP^k eigenvalues approaching 1 ↔ regular (KAM) region.
   Deviation ↔ chaos.  Available on MonodromyData and in regularity module.


═══════════════════════════════════════════════════════════════════════════
CURRENT IMPLEMENTATION STATUS
═══════════════════════════════════════════════════════════════════════════

Implemented:
  ✅ InvariantSet, InvariantManifold, SectionCuttable  pyna/topo/_base.py
  ✅ FixedPoint (generic coords + MCF compat)          pyna/topo/invariants.py
  ✅ Cycle, InvariantTorus                              pyna/topo/invariants.py
  ✅ StableManifold, UnstableManifold                   pyna/topo/invariants.py
  ✅ MonodromyData.spectral_regularity()                pyna/topo/invariants.py
  ✅ spectral_regularity, classify_orbit, etc.          pyna/topo/regularity.py
  ✅ Section (abstract + MCF)                           pyna/topo/section.py
  ✅ Island, IslandChain                                pyna/topo/island.py
  ✅ Tube, TubeChain                                    pyna/topo/tube.py
  ✅ PhaseSpace, DynamicalSystem hierarchy              pyna/topo/dynamics.py
  ✅ ResonanceNumber                                    pyna/topo/resonance.py

Future (Phase 2+):
  ⬜ Absorb diffeq C++ integrators into cyna/include/cyna/integrators/
  ⬜ SDE solvers (Euler-Maruyama, Milstein, SRI)
  ⬜ Bifurcation analysis, parameter continuation
  ⬜ Center manifold reduction
"""
