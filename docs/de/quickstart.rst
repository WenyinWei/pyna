Schnelleinstieg
===============

Diese Seite führt durch die drei Kernfunktionen von **pyna**:
Feldlinienverfolgung, Poincaré-Karten und Insel-Topologie.  Verwendet wird ein
einfaches analytisches Tokamak-Gleichgewicht, das keine externen Datendateien
benötigt.

.. note::

   Alle Beispiele verwenden das **analytische Solov'ev-Gleichgewicht**
   (Cerfon & Freidberg 2010), skaliert auf EAST-ähnliche Parameter
   (R₀ ≈ 1.86 m, B₀ = 5.3 T).  Es ist ein gutes universelles Testbett: exakte
   Grad-Shafranov-Lösung, geschlossene Ausdrücke für die Feldkomponenten und
   einstellbare Form.

----

1. Analytisches Gleichgewicht aufbauen
--------------------------------------

Beginnen Sie mit dem Import des Gleichgewichts und prüfen Sie seine
Grundparameter:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from pyna.toroidal.equilibrium import solovev_iter_like

   eq = solovev_iter_like(scale=0.3)          # EAST-like size
   Rmaxis, Zmaxis = eq.magnetic_axis

   print(f"R0 = {eq.R0:.2f} m   a = {eq.a:.2f} m   B0 = {eq.B0:.1f} T")
   print(f"κ  = {eq.kappa:.2f}  δ = {eq.delta:.2f}  q0 = {eq.q0:.2f}")
   print(f"Magnetic axis: R = {Rmaxis:.3f} m, Z = {Zmaxis:.3f} m")

Das zurückgegebene Objekt ``eq`` stellt ``eq.BR_BZ(R, Z)``, ``eq.Bphi(R)``,
``eq.psi(R, Z)`` (normalisierter Fluss) und ``eq.q_profile(psi)`` bereit.

----

2. Feldlinien verfolgen und Poincaré-Schnitte sammeln
-----------------------------------------------------

Ein Poincaré-Schnitt speichert die (R, Z)-Koordinaten jedes Mal, wenn eine
Feldlinie einen gewählten toroidalen Schnitt kreuzt (hier φ = 0).  Nach vielen
toroidalen Umläufen erscheinen verschachtelte Flussflächen als geschlossene
Kurven; eine magnetische Insel zeigt sich als Kette diskreter Schnittpunkte.

.. code-block:: python

   from pyna.flt import get_backend
   from pyna.topo.poincare import PoincareToroidalSection, poincare_from_fieldlines

   # The accumulator section detects crossings between sampled 3-D points.
   section = PoincareToroidalSection(0.0)

   # --- unit tangent in cylindrical coordinates: dR/dl, dZ/dl, dφ/dl ---
   def field_rhs(rzphi):
       R, Z, _phi = rzphi
       BR, BZ = eq.BR_BZ(R, Z)
       Bphi   = eq.Bphi(R)
       Bnorm  = np.sqrt(BR**2 + BZ**2 + Bphi**2)
       return [BR / Bnorm, BZ / Bnorm, Bphi / (R * Bnorm)]

   # --- seed 8 field lines radially outward from the axis ---
   R_starts = np.linspace(Rmaxis + 0.05, Rmaxis + 0.45, 8)
   Z_starts = np.zeros(8)

   # --- integrate about 80 toroidal turns per line ---
   n_turns = 80
   flt = get_backend('cpu', field_func=field_rhs, dt=0.08)
   pacc = poincare_from_fieldlines(
       field_func=field_rhs,
       start_pts=np.column_stack([R_starts, Z_starts, np.zeros_like(R_starts)]),
       sections=[section],
       t_max=n_turns * 2 * np.pi * Rmaxis,
       backend=flt,
   )
   poincare_pts = pacc.crossing_array(0)[:, :2]

   # --- plot ---
   fig, ax = plt.subplots(figsize=(6, 6))
   ax.scatter(poincare_pts[:, 0], poincare_pts[:, 1], s=0.8, color='steelblue')
   ax.set_xlabel('R (m)')
   ax.set_ylabel('Z (m)')
   ax.set_aspect('equal')
   ax.set_title('Poincaré map — Solov\'ev equilibrium')
   plt.tight_layout()
   plt.show()

.. figure:: /_static/quickstart_poincare.png
   :align: center
   :width: 80%
   :alt: Poincaré-Karte eines analytischen Solov'ev-Gleichgewichts mit verschachtelten Flussflächen

   **Abbildung 1.** Poincaré-Karte des analytischen Solov'ev-Gleichgewichts
   (EAST-ähnliche Parameter, 250 toroidale Durchläufe pro Feldlinie).  Jede
   Farbe entspricht einer Feldlinie; verschachtelte geschlossene Kurven sind
   Flussflächen.  Das rote Kreuz markiert die magnetische Achse; die schwarze
   Kurve ist die letzte geschlossene Flussfläche (LCFS, ψ = 1).

Jeder konzentrische Ring entspricht einer Feldlinie, die um eine Flussfläche
windet.  Die rationale Fläche q = m/n ist der Ort, an dem eine resonante
Störung (z. B. eine RMP-Spule) eine magnetische Insel öffnen kann.

----

3. Rationale Fläche lokalisieren und Insel vermessen
----------------------------------------------------

Nach Hinzufügen einer kleinen resonanten Störung öffnet sich an der Fläche
q = 2/1 eine magnetische Insel.  pyna kann die Fläche lokalisieren und die
Halbbreite der Insel mit einem einzigen Aufruf in Metern bestimmen:

.. code-block:: python

   from pyna.topo.toroidal_island import locate_rational_surface, island_halfwidth

   # Build q(S) from PEST mesh
   from pyna.toroidal.coords import build_PEST_mesh

   nR, nZ = 100, 100
   R_grid = np.linspace(0.3*eq.R0, 1.5*eq.R0, nR)
   Z_grid = np.linspace(-eq.a*eq.kappa*1.3, eq.a*eq.kappa*1.3, nZ)
   Rg, Zg  = np.meshgrid(R_grid, Z_grid, indexing='ij')

   BR, BZ   = eq.BR_BZ(Rg, Zg)
   Bphi     = eq.Bphi(Rg)
   psi_norm = eq.psi(Rg, Zg)

   S, TET, R_mesh, Z_mesh, q_iS = build_PEST_mesh(
       R_grid, Z_grid, BR, BZ, Bphi, psi_norm,
       Rmaxis, Zmaxis, ns=40, ntheta=181
   )
   S_values = S[1:]
   q_values = q_iS[1:]
   print(f"q range: {q_values[0]:.2f} → {q_values[-1]:.2f}")

   # Locate q = 2/1 surface
   res = locate_rational_surface(S_values, q_values, m=2, n=1)
   print(f"q=2/1 surface at S = {res[0]:.4f}  (ψ_norm = {res[0]**2:.4f})")

Der zurückgegebene Wert ``S_res`` (S = √ψ_norm) gibt exakt an, wo die resonante
Schicht liegt.  Übergeben Sie ihn zusammen mit der gestörten Poincaré-Karte an
``island_halfwidth``, um die Inselbreite in Metern zu erhalten.

----

4. Allgemeine endlichdimensionale Dynamik
-----------------------------------------

pyna ist nicht auf toroidale Feldlinien beschränkt.  Dasselbe
Topologie-Objektmodell steht für Hamiltonsche Systeme, N-Körper-Flüsse,
Abbildungen und SDE-Abtastpfade zur Verfügung.

.. code-block:: python

   import numpy as np
   from pyna.dynamics import (
       SeparableHamiltonianSystem,
       CallableMap,
       GeometricBrownianMotion,
   )

   oscillator = SeparableHamiltonianSystem(
       kinetic=lambda p, t: 0.5 * np.dot(p, p),
       potential=lambda q, t: 0.5 * np.dot(q, q),
       grad_kinetic=lambda p, t: p,
       grad_potential=lambda q, t: q,
       dof=1,
   )
   traj = oscillator.trajectory([1.0, 0.0], (0.0, 2*np.pi), dt=0.01)
   print(traj.final)  # TimeSeriesSolution is a pyna.topo.core.Trajectory

   linear_map = CallableMap(lambda x: np.array([2*x[0], 0.5*x[1]]), dim=2)
   orbit = linear_map.orbit_geometry([1.0, 1.0], n_iter=5)
   print(orbit.period_guess)

   gbm = GeometricBrownianMotion(mu=[0.08], sigma=[0.2])
   print(gbm.expected_log_growth())

Verwenden Sie Objekte aus :mod:`pyna.topo.core` wie ``Cycle``,
``PeriodicOrbit``, ``Tube`` und ``IslandChain``, wenn eine Trajektorie oder ein
Abbildungsorbit aus abgetasteten Daten zu einem geometrisch-topologischen
Objekt überführt wurde.

----

5. Workflow-basierte Konstruktion
---------------------------------

Für größere Projekte und Lehrnotebooks empfiehlt sich ``TopologyWorkflow``,
damit die Analyseabfolge explizit bleibt und nicht mit ad-hoc-Konstruktoren
über den Code verteilt wird.

.. code-block:: python

   import numpy as np
   from pyna.topo import TopologyWorkflow
   from pyna.topo.section import HyperplaneSection

   wf = TopologyWorkflow(closure_tol=1e-3)
   flow = wf.system(
       "callable-flow",
       rhs=lambda x, t: np.array([x[1], -x[0]]),
       dim=2,
       coordinate_names=("q", "p"),
   )

   section = HyperplaneSection(np.array([1.0, 0.0]), 0.0, phase_dim=2)
   pmap = wf.poincare_map(flow, section, dt=0.02)

   closed_traj = wf.trajectory(flow, [1.0, 0.0], (0.0, 2*np.pi), dt=0.01)
   cycle = wf.closed_cycle(closed_traj)

Die Low-Level-Adapter, Builder, Bridges und Factories bleiben für
Bibliotheksautoren verfügbar, aber die meisten Notebooks sollten mit der
Workflow-Fassade beginnen.

----

6. Nächste Schritte
-------------------

- **Tutorials** — ausgearbeitete Beispiele mit Diagrammen:
  :doc:`/de/mini-cases`,
  :doc:`/de/tutorials/sde-monte-carlo`,
  :doc:`/notebooks/i18n/de/tutorials/RMP_resonance_analysis`,
  :doc:`/notebooks/i18n/de/tutorials/magnetic_coordinates_comparison`,
  :doc:`/notebooks/i18n/de/tutorials/RMP_island_validation_solovev`

- **API-Referenz** — vollständige Docstrings:
  :doc:`/en/api/index`

- **CUDA-Beschleunigung** — installieren Sie ``cupy-cuda12x`` und übergeben
  Sie ``backend=get_backend('cuda')`` an den Tracer, um Inselbreitenscans auf
  der GPU um bis zu 100× zu beschleunigen.
