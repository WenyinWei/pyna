"""
pyna.topo  — Topology of Dynamical Systems
==========================================

Design philosophy
-----------------
The class hierarchy mirrors the mathematical structure of a dynamical system,
not the implementation details.  Each abstraction corresponds to a precise
mathematical concept, with clean layering from abstract to concrete.

Layer 0: Phase space and dynamics
Layer 1: Invariant objects (orbits, tori, manifolds)
Layer 2: Structure (how invariants are organised relative to each other)
Layer 3: Sections (how invariants appear on Poincaré sections)
Layer 4: Chains and families (discrete-time organisation)
Layer 5: Plots (visual representation)


═══════════════════════════════════════════════════════════════════════════
LAYER 0 — Phase space and dynamics
═══════════════════════════════════════════════════════════════════════════

  PhaseSpace(dim: int)
    The ambient space where the dynamics lives.
    Examples: 2D (R,Z) for MCF field lines, 4D (R,Z,pR,pZ) for guiding centre.

  DynamicalSystem(phase_space: PhaseSpace)
    The flow φ^t: PhaseSpace → PhaseSpace.
    Subclasses:
      ContinuousFlow         d/dt x = f(x, t)
        HamiltonianFlow      H(q,p) = const, symplectic
        MagneticFieldLine    B·∇x = 0 (field-line ODE)
      DiscreteMap            x → P(x)
        PoincareMap(flow, section)  P = return map to Section


═══════════════════════════════════════════════════════════════════════════
LAYER 1 — Invariant objects
═══════════════════════════════════════════════════════════════════════════

  InvariantObject(system: DynamicalSystem)
    Something that is invariant under the flow.

  PeriodicOrbit(InvariantObject)           [continuous time]
    A closed trajectory φ^T(x₀) = x₀ for some period T > 0.
    Attributes:
      seed_point: ndarray          starting point in phase space
      period: float                time to close (real period)
      monodromy(section): ndarray  linearised return map DPm
      stability: Stability         (eigenvalues, type)
      resonance: ResonanceNumber   (m, n, ...) winding numbers
    Properties (lazy):
      is_hyperbolic: bool
      is_elliptic: bool
      stable_manifold: StableManifold
      unstable_manifold: UnstableManifold

  InvariantTorus(InvariantObject)          [continuous time]
    A KAM torus (quasi-periodic orbit) or rational torus (resonant).
    In MCF: a magnetic flux surface.
    Attributes:
      winding_numbers: tuple[float, ...]  ω₁/ω₂/... (irrational for KAM)
      or resonance: ResonanceNumber        (rational / resonant case)

  InvariantManifold(InvariantObject)       [continuous time]
    Stable/unstable manifold of a periodic orbit.
    Subclasses:
      StableManifold(orbit: PeriodicOrbit)
      UnstableManifold(orbit: PeriodicOrbit)


═══════════════════════════════════════════════════════════════════════════
LAYER 2 — Structure: how invariants are organised
═══════════════════════════════════════════════════════════════════════════

  Tube(system: DynamicalSystem)
    A magnetic island = one nested invariant-torus structure.
    Contains a FAMILY of invariant tori (from centre O-cycle to separatrix).
    Skeleton (lazy):
      o_cycle: PeriodicOrbit         the O-type orbit at the core
      x_cycles: List[PeriodicOrbit]  the X-type orbit(s) at the boundary
    Methods:
      section_cut(section) → List[Island]    cut with a Section
      manifold_cut(section) → ManifoldSection  cut W^u/W^s with Section

  TubeChain(resonance: ResonanceNumber)
    ALL Tubes of a given resonance m/n in the system.
    = The complete resonance zone (all islands of this resonance).
    Contains:
      tubes: List[Tube]
    Methods:
      section_cut(section) → IslandChain
      wire_skeletons()      set o_cycle/x_cycles on all Tubes
      x_tubes, o_tubes      filter by seed stability (internal use)

  ResonanceZone(resonance: ResonanceNumber)
    High-level: TubeChain + its bounding separatrix structure.
    = TubeChain + the X-cycle manifolds W^u ∪ W^s that define the zone.
    (Future: stochastic layer, island overlap criterion, etc.)


═══════════════════════════════════════════════════════════════════════════
LAYER 3 — Sections: how invariants appear on Poincaré sections
═══════════════════════════════════════════════════════════════════════════

  Section                                [abstract, any codim-1 surface]
    Subclasses:
      ToroidalSection(phi)               φ = const
      HyperplaneSection(a, c, n)         a·x = c
      ParametricSection(f, grad, n)      f(x) = 0

  CutPoint                               one intersection of Orbit ∩ Section
    Attributes:
      position: ndarray                  coordinates in section
      monodromy: ndarray                 DPm at this cut
      orbit: PeriodicOrbit | None        back-ref (if known)

  ManifoldSection                        W^u or W^s ∩ Section
    Attributes:
      points: ndarray                    (N, dim_section) intersection pts
      manifold: InvariantManifold        back-ref


═══════════════════════════════════════════════════════════════════════════
LAYER 4 — Discrete-time organisation
═══════════════════════════════════════════════════════════════════════════

  Island                                 [discrete time, one section cut]
    The cross-section of one Tube at one Section.
    Attributes:
      O_point: ndarray                   elliptic fixed point of P^m
      X_points: List[ndarray]            hyperbolic fixed points nearby
      tube: Tube | None                  back-ref to continuous Tube (lazy)
      section: Section | None            the Section it lives on
      resonance_index: int | None        which Tube within TubeChain
    Connectivity (lazy):
      next(): Island    P^1 image
      last(): Island    P^{-1} image

  IslandChain                            [discrete time, one section]
    All Islands of a given resonance at one Section.
    = TubeChain.section_cut(section)
    Attributes:
      islands: List[Island]
      tube_chain: TubeChain | None       back-ref
      section: Section | None


═══════════════════════════════════════════════════════════════════════════
LAYER 5 — Plots
═══════════════════════════════════════════════════════════════════════════

  pyna.plot
    plot_island(island, section, ax)
    plot_island_chain(chain, section, ax)
    plot_tube_section(tube, section, ax, tube_idx)
    plot_tube_chain_section(chain, section, ax)
    plot_tube_chain_poincare(chain, sections, ...)   ← new, first-class
    draw_xo_points(ax, xpts, opts, style)
    tube_legend_handles(chain)


═══════════════════════════════════════════════════════════════════════════
KEY PRINCIPLES
═══════════════════════════════════════════════════════════════════════════

1. LAZINESS EVERYWHERE
   Skeleton (o_cycle, x_cycles), manifolds, section cuts, connectivity —
   all computed on demand and cached.  Nothing is computed at construction.

2. BACK-REFERENCES (bottom-up)
   Island → Tube → (TubeChain, PeriodicOrbit)
   NOT Tube → Island (too many possible cuts)

3. SECTION INDEPENDENCE
   The same Tube can be cut at any Section to yield Islands.
   The Section is passed as an argument, not baked into the object.

4. ARBITRARY DIMENSION
   PhaseSpace.dim, ResonanceNumber, Section — all dimension-agnostic.
   MCF (d=2) is a special case, not the defining case.

5. DISCRETE ↔ CONTINUOUS BRIDGE
   Island.tube links discrete to continuous (may be None for pure maps).
   Tube.section_cut is the canonical bridge continuous → discrete.

═══════════════════════════════════════════════════════════════════════════
CURRENT IMPLEMENTATION STATUS
═══════════════════════════════════════════════════════════════════════════

Implemented:
  ✅ ResonanceNumber            pyna/topo/resonance.py
  ✅ Section (abstract + MCF)   pyna/topo/section.py
  ✅ Island (with .tube, .next/.last)  pyna/topo/island.py
  ✅ IslandChain                pyna/topo/island_chain.py
  ✅ Tube (skeleton: o_cycle, x_cycles, section_cut)  pyna/topo/tube.py
  ✅ TubeChain (section_cut, wire_skeletons)  pyna/topo/tube.py
  ✅ plot_tube_* (first-class)  pyna/plot/tube.py
  ✅ PeriodicOrbit (IslandChainOrbit)  pyna/topo/island_chain.py
  ✅ InvariantManifold (StableManifold, UnstableManifold)  pyna/topo/manifold_improve.py

Partial / TODO:
  ⬜ PhaseSpace class
  ⬜ DynamicalSystem / ContinuousFlow / DiscreteMap / PoincareMap
  ⬜ CutPoint (currently TubeCutPoint, section_view.SectionViewPoint)
  ⬜ ManifoldSection
  ⬜ ResonanceZone
  ⬜ Section API propagated through Tube/TubeChain (still phi:float in many places)
  ⬜ InvariantTorus (KAM tori; currently only resonant = PeriodicOrbit families)
"""
