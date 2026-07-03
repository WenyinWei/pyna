Karte der öffentlichen API
==========================

Diese Seite ist der kurze Einstieg in die stabilen Schnittstellen von pyna.  Die
generierte AutoAPI bleibt die vollständige Debug-Referenz; die Einträge unten
sind die bevorzugten Einstiege für Notebooks, Forschungsskripte und abhängige
Pakete.

Geometrisches Vokabular
-----------------------

Beginnen Sie hier, wenn das Objekt im Phasenraum wichtiger ist als der Solver,
der es erzeugt hat.

.. list-table::
   :header-rows: 1

   * - Aufgabe
     - Öffentliche Einstiege
   * - Gesampelte kontinuierliche Bewegung
     - :class:`pyna.topo.core.Trajectory`, :class:`pyna.topo.core.Cycle`,
       :class:`pyna.topo.core.Tube`, :class:`pyna.topo.core.TubeChain`
   * - Dynamik diskreter Abbildungen
     - :class:`pyna.topo.core.Orbit`, :class:`pyna.topo.core.PeriodicOrbit`,
       :class:`pyna.topo.core.Island`, :class:`pyna.topo.core.IslandChain`
   * - Toroidale Schnittgeometrie
     - :mod:`pyna.topo.toroidal`, :mod:`pyna.plot.section_geometry`,
       :mod:`pyna.plot.rmp`
   * - Explizite Promotion und Adapter
     - :class:`pyna.topo.workflow.TopologyWorkflow`,
       :mod:`pyna.topo.builders`, :mod:`pyna.topo.bridges`

Allgemeine Dynamik
------------------

Verwenden Sie :mod:`pyna.dynamics` für nicht-toroidale Modelle, die trotzdem
dieselben Geometrieobjekte zurückgeben sollen.

.. list-table::
   :header-rows: 1

   * - Modellfamilie
     - Öffentliche Einstiege
   * - ODE-Flüsse
     - :class:`pyna.dynamics.CallableFlow`,
       :class:`pyna.dynamics.TimeSeriesSolution`
   * - Hamiltonsche Systeme
     - :class:`pyna.dynamics.HamiltonianSystem`,
       :class:`pyna.dynamics.SeparableHamiltonianSystem`
   * - N-Körper-Systeme
     - :class:`pyna.dynamics.NBodySystem`
   * - Diskrete Abbildungen
     - :class:`pyna.dynamics.CallableMap`,
       :func:`pyna.dynamics.fixed_point_eigenspaces`
   * - SDEs
     - :class:`pyna.dynamics.ItoSDE`,
       :class:`pyna.dynamics.BrownianMotion`,
       :class:`pyna.dynamics.GeometricBrownianMotion`

Toroidale und RMP-Workflows
---------------------------

Für magnetische Koordinaten, Feldlinienverfolgung, Spektralanalyse und
Overlays verwenden Sie diese Module.

.. list-table::
   :header-rows: 1

   * - Bedarf
     - Öffentliche Einstiege
   * - Gleichgewichte und Koordinaten
     - :mod:`pyna.toroidal.equilibrium`, :mod:`pyna.toroidal.coords`,
       :mod:`pyna.toroidal.pest_coords`
   * - Feldlinienverfolgung und cache-aware Workflows
     - :mod:`pyna.toroidal.flt`, :mod:`pyna.workflow.tracing`
   * - Kontravariantes radiales Störungsspektrum
     - :func:`pyna.toroidal.perturbation_spectrum.radial_perturbation_Fourier_spectrum`,
       :func:`pyna.toroidal.perturbation_spectrum.analyze_resonant_island_chains_multi_n`
   * - RMP/nRMP-Diagnostik
     - :mod:`pyna.toroidal.visual.RMP_spectrum`,
       :mod:`pyna.toroidal.torus_deformation`
   * - Magnetische Spektrumfiguren
     - :mod:`pyna.toroidal.visual.magnetic_spectrum`
   * - Poincare-, X/O- und Insel-Overlays
     - :func:`pyna.plot.rmp.plot_rmp_resonance_sections`,
       :func:`pyna.toroidal.visual.tokamak_manifold.draw_manifold_segments`

Wann AutoAPI verwenden?
-----------------------

Nutzen Sie :doc:`/en/api/generated/pyna/index`, wenn Sie Konstruktorsignaturen,
geerbte Member, seltene Diagnostik oder Implementierungsdetails benötigen.
Neue Tutorials und nutzerorientierte Beispiele sollten nach Möglichkeit auf den
öffentlichen Einstiegen oben bleiben.
