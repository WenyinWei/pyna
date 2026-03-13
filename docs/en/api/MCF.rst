MCF Plasma Extensions (``pyna.MCF``)
======================================

The ``pyna.MCF`` package provides tools for magnetic confinement fusion
research: equilibria, magnetic coordinates, coil modelling, topology control,
plasma response, diagnostics, publication-quality visualization, and
non-resonant torus-deformation theory.

.. contents:: Submodules
   :depth: 2
   :local:

----

Equilibria
----------

.. automodule:: pyna.MCF.equilibrium.Solovev
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyna.MCF.equilibrium.axisymmetric
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyna.MCF.equilibrium.GradShafranov
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyna.MCF.equilibrium.stellarator
   :members:
   :undoc-members:
   :show-inheritance:

----

Magnetic Coordinates
--------------------

.. automodule:: pyna.MCF.coords.PEST
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyna.MCF.coords.Boozer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyna.MCF.coords.Hamada
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyna.MCF.coords.EqualArc
   :members:
   :undoc-members:
   :show-inheritance:

----

Coils & Fields
--------------

.. automodule:: pyna.MCF.coils.coil
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyna.MCF.coils.RMP
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyna.MCF.coils.coil_system
   :members:
   :undoc-members:
   :show-inheritance:

----

MCF Topology Control
--------------------

.. automodule:: pyna.MCF.control.gap_response
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyna.MCF.control.qprofile_response
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyna.MCF.control.wall
   :members:
   :undoc-members:
   :show-inheritance:

----

Plasma Response
---------------

.. automodule:: pyna.MCF.plasma_response.PerturbGS
   :members:
   :undoc-members:
   :show-inheritance:

----

Diagnostics
-----------

.. automodule:: pyna.MCF.diagnostics
   :members:
   :undoc-members:
   :show-inheritance:

----

Visualization (``pyna.MCF.visual``)
------------------------------------

Publication-quality plotting for tokamak research figures.
All functions follow a composable, axes-based API suitable for
single-column PRL/Nuclear Fusion figures.

.. automodule:: pyna.MCF.visual.tokamak_manifold
   :members:
   :undoc-members:
   :show-inheritance:

----

Non-resonant Torus Deformation (``pyna.MCF.torus_deformation``)
----------------------------------------------------------------

Implements the analytic spectral theory of Wei (2025) for the deformation
of invariant tori (flux surfaces) under external magnetic perturbations.

All formulas work in flux coordinates :math:`(r, \theta, \varphi)` with
the Fourier convention :math:`f(\theta,\varphi)=\sum_{mn} f_{mn}\,e^{i(m\theta+n\varphi)}`.

**Key results implemented:**

- **Theorem 2** — full :math:`(\delta r, \delta\theta, \delta\varphi)_{mn}` spectra
- **Theorem 3** — universal 1-D Poincaré-section ring deformation
- **Eq. (4.2)** — mean radial displacement for PF coil perturbations
- **Eq. (4.3)** — DC (m=n=0) simplification
- **Eq. (5.1)** — second-order non-axisymmetric formula

.. automodule:: pyna.MCF.torus_deformation
   :members:
   :undoc-members:
   :show-inheritance:

The :class:`~pyna.MCF.torus_deformation.TorusDeformationSpectrum` dataclass
holds the full output spectrum:

.. autoclass:: pyna.MCF.torus_deformation.TorusDeformationSpectrum
   :members:
   :undoc-members:

**Primary computational functions:**

.. autofunction:: pyna.MCF.torus_deformation.non_resonant_deformation_spectrum
.. autofunction:: pyna.MCF.torus_deformation.poincare_section_deformation
.. autofunction:: pyna.MCF.torus_deformation.iota_variation_pf
.. autofunction:: pyna.MCF.torus_deformation.mean_radial_displacement
.. autofunction:: pyna.MCF.torus_deformation.mean_radial_displacement_pf
.. autofunction:: pyna.MCF.torus_deformation.mean_radial_displacement_dc
.. autofunction:: pyna.MCF.torus_deformation.mean_radial_displacement_second_order
.. autofunction:: pyna.MCF.torus_deformation.deformation_peak_valley
.. autofunction:: pyna.MCF.torus_deformation.green_function_spectrum
.. autofunction:: pyna.MCF.torus_deformation.iota_to_q
.. autofunction:: pyna.MCF.torus_deformation.q_to_iota
.. autofunction:: pyna.MCF.torus_deformation.iota_prime_from_q_prime
