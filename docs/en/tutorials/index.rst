Tutorials and Examples
======================

The public notebooks are grouped by workflow.  The documentation build copies
``notebooks/`` into the Sphinx source tree, so paths below mirror the repository
layout.

Recommended Learning Path
-------------------------

Start with :doc:`/en/quickstart`, then work through the generic geometry
workflow, stochastic models, and then toroidal monodromy/RMP examples:

1. :doc:`/en/mini-cases`
2. :doc:`sde-monte-carlo`
3. :doc:`/notebooks/i18n/en/tutorials/RMP_resonance_analysis`
4. :doc:`/notebooks/i18n/en/tutorials/monodromy_xcycle_analytic`
5. :doc:`/notebooks/i18n/en/tutorials/island_jacobian_analysis`
6. :doc:`/notebooks/i18n/en/tutorials/RMP_island_validation_solovev`

General Dynamical Systems
-------------------------

.. toctree::
   :maxdepth: 1

   sde-monte-carlo

The generic geometry workflow and analytic stellarator fixed-point workflow are
now folded into the RMP resonance tutorial instead of being published as
standalone text-only notebooks.  That tutorial shows the same promotion chain:
sampled crossings -> fixed-point geometry -> X/O classification -> manifold
and coordinate-grid overlays.

Stochastic Differential Equations
---------------------------------

The SDE tutorial is pre-executed locally because distribution estimation often
uses tens or hundreds of thousands of Monte Carlo paths.  GitHub Pages renders
the saved outputs instead of spending CI time on the heavy sampling cells.

.. toctree::
   :maxdepth: 1
   :hidden:

   /notebooks/i18n/en/tutorials/sde_monte_carlo_distribution

Magnetic Coordinates and Equilibria
-----------------------------------

.. toctree::
   :maxdepth: 1

   /notebooks/i18n/en/tutorials/magnetic_coordinates_comparison

RMPs, Islands and Poincare Analysis
-----------------------------------

Start with the resonance analysis notebook when studying magnetic topology.  It
now covers divergence-free RMP templates, the important ``m=1`` branch, ``cyna``
fixed-point validation, multi-component contravariant ``B^r`` magnetic-spectrum
atlases, modular ``q``/``m/n`` resonance maps with optional Poincare and island
overlays, mixed RMP/nRMP spectra, total nRMP response from all non-resonant
modes, field-line speed modulation, and perturbation-order checks.

.. toctree::
   :maxdepth: 1

   /notebooks/i18n/en/tutorials/RMP_resonance_analysis
   /notebooks/i18n/en/tutorials/RMP_island_validation_solovev
   /notebooks/i18n/en/tutorials/island_jacobian_analysis

``RMP_resonance_exec.ipynb`` is kept in the repository as an execution/cache
variant of the resonance analysis workflow, but the public documentation links
to the explanatory version above.

Monodromy and Manifolds
-----------------------

.. toctree::
   :maxdepth: 1

   /notebooks/i18n/en/tutorials/monodromy_mobius_saddle
   /notebooks/i18n/en/tutorials/monodromy_xcycle_analytic

Classical and General Dynamical Systems
---------------------------------------

The repository also includes lightweight notebooks under ``notebooks/examples``:
``Lorenz_attractor.ipynb``, ``resonance_1_1_map.ipynb``,
``Mobiusian_saddle_cycle.ipynb``, ``Xcycle_construction.ipynb`` and
``FPT_DX_to_DP_sympy.ipynb``.  They are kept as source examples rather than
executed documentation pages because several are scratch-style notebooks without
section titles.

Static Tutorial Figures
-----------------------

Several longer workflows are represented in the repository as static figures
and generated outputs under ``notebooks/tutorials``.  They cover q-profile
diagnostics, PEST/Boozer/Hamada/equal-arc coordinates, island suppression scans,
phase control, Poincare manifolds and Solov'ev single-null examples.
