Demarrage rapide
================

Cette page presente les trois capacités centrales de **pyna** - traçage de
lignes de champ, cartes de Poincaré et topologie d'îlots - à l'aide d'un
équilibre tokamak analytique simple qui ne nécessite aucun fichier de données
externe.

.. note::

   Tous les exemples utilisent l'**equilibre analytique de Solov'ev** (Cerfon &
   Freidberg 2010), mis à l'échelle de paramètres de type EAST (R₀ ≈ 1.86 m,
   B₀ = 5.3 T). C'est un banc d'essai generaliste : solution exacte de
   Grad-Shafranov, composantes du champ en forme fermee, forme reglable.

----

1. Construire un équilibre analytique
-------------------------------------

Commencez par importer l'équilibre et inspecter ses paramètres de base :

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from pyna.toroidal.equilibrium import solovev_iter_like

   eq = solovev_iter_like(scale=0.3)          # EAST-like size
   Rmaxis, Zmaxis = eq.magnetic_axis

   print(f"R0 = {eq.R0:.2f} m   a = {eq.a:.2f} m   B0 = {eq.B0:.1f} T")
   print(f"κ  = {eq.kappa:.2f}  δ = {eq.delta:.2f}  q0 = {eq.q0:.2f}")
   print(f"Magnetic axis: R = {Rmaxis:.3f} m, Z = {Zmaxis:.3f} m")

L'objet ``eq`` renvoye expose ``eq.BR_BZ(R, Z)``, ``eq.Bphi(R)``,
``eq.psi(R, Z)`` (flux normalise) et ``eq.q_profile(psi)``.

----

2. Tracer les lignes de champ et accumuler les intersections de Poincaré
------------------------------------------------------------------------

Une section de Poincaré enregistre les coordonnées (R, Z) chaque fois qu'une
ligne de champ traverse une section toroidale choisie (ici φ = 0). Après de
nombreux tours toroidaux, les surfaces de flux emboitees apparaissent comme des
courbes fermées ; un îlot magnétique apparait comme une chaine de points de
section discrets.

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
   :alt: Carte de Poincaré d'un équilibre analytique de Solov'ev montrant des surfaces de flux emboitees

   **Figure 1.** Carte de Poincaré de l'équilibre analytique de Solov'ev
   (paramètres de type EAST, 250 transits toroidaux par ligne de champ).
   Chaque couleur correspond à une ligne de champ ; les courbes fermées
   emboitees sont des surfaces de flux. La croix rouge marque l'axe
   magnétique ; la courbe noire est la dernière surface de flux fermee
   (LCFS, ψ = 1).

Chaque anneau concentrique correspond à une ligne de champ qui s'enroule autour
d'une surface de flux. La surface rationnelle q = m/n est l'endroit ou une
perturbation résonante (par exemple une bobine RMP) peut ouvrir un îlot
magnétique.

----

3. Localiser une surface rationnelle et mesurer un îlot
------------------------------------------------------

Après l'ajout d'une petite perturbation résonante, un îlot magnétique s'ouvre
sur la surface q = 2/1. pyna peut localiser la surface et mesurer le demi-largeur
de l'îlot en un seul appel :

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

La valeur ``S_res`` renvoyee (S = √ψ_norm) indique exactement ou se trouve la
couche résonante. Passez-la a ``island_halfwidth`` avec la carte de Poincaré
perturbee pour obtenir la largeur de l'îlot en metres.

----

4. Dynamique générale de dimension finie
----------------------------------------

pyna n'est pas limité aux lignes de champ toroidales. Le même modèle d'objets
topologiques est disponible pour les systèmes hamiltoniens, les flux à N corps,
les cartes et les chemins échantillonnés de SDE.

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

Utilisez les objets de :mod:`pyna.topo.core` comme ``Cycle``,
``PeriodicOrbit``, ``Tube`` et ``IslandChain`` lorsqu'une trajectoire ou une
orbite de carte a été promue depuis des données échantillonnées vers un objet
géométrique/topologique.

----

5. Construction fondee sur les workflows
----------------------------------------

Pour les projets plus grands et les notebooks d'enseignement, utilisez
``TopologyWorkflow`` afin de garder la séquence d'analyse explicite sans
disperser de constructeurs ad hoc dans le code.

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

Les adaptateurs, builders, bridges et factories de plus bas niveau restent
disponibles pour les auteurs de bibliothèques, mais la plupart des notebooks
devraient commencer par la facade de workflow.

----

6. Etapes suivantes
-------------------

- **Tutoriels** - exemples travailles avec figures :
  :doc:`/fr/mini-cases`,
  :doc:`/fr/tutorials/sde-monte-carlo`,
  :doc:`/notebooks/i18n/fr/tutorials/RMP_resonance_analysis`,
  :doc:`/notebooks/i18n/fr/tutorials/magnetic_coordinates_comparison`,
  :doc:`/notebooks/i18n/fr/tutorials/RMP_island_validation_solovev`

- **Référence API** - docstrings completes :
  :doc:`/fr/api/index`

- **Acceleration CUDA** - installez ``cupy-cuda12x`` et passez
  ``backend=get_backend('cuda')`` au traceur pour accelerer jusqu'a 100x les
  balayages de largeur d'îlot.
