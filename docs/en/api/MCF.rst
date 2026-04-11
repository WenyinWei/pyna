Toroidal Plasma Extensions (``pyna.toroidal``)
==============================================

The ``pyna.toroidal`` package provides tools for magnetic confinement fusion
research: equilibria, magnetic coordinates, coil modelling, topology control,
plasma response, diagnostics, publication-quality visualization, and
non-resonant torus-deformation theory.

.. contents:: Submodules
   :depth: 2
   :local:

----

Equilibria
----------

.. automodule:: pyna.toroidal.equilibrium.Solovev
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyna.toroidal.equilibrium.axisymmetric
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyna.toroidal.equilibrium.GradShafranov
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyna.toroidal.equilibrium.stellarator
   :members:
   :undoc-members:
   :show-inheritance:

----

Magnetic Coordinates
--------------------

.. automodule:: pyna.toroidal.coords.PEST
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyna.toroidal.coords.Boozer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyna.toroidal.coords.Hamada
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyna.toroidal.coords.EqualArc
   :members:
   :undoc-members:
   :show-inheritance:

----

Coils & Fields
--------------

.. automodule:: pyna.toroidal.coils.coil
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyna.toroidal.coils.RMP
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyna.toroidal.coils.coil_system
   :members:
   :undoc-members:
   :show-inheritance:

----

MCF Topology Control
--------------------

.. automodule:: pyna.toroidal.control.gap_response
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyna.toroidal.control.qprofile_response
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyna.toroidal.control.wall
   :members:
   :undoc-members:
   :show-inheritance:

----

Plasma Response
---------------

.. automodule:: pyna.toroidal.plasma_response.PerturbGS
   :members:
   :undoc-members:
   :show-inheritance:

----

Diagnostics
-----------

.. automodule:: pyna.toroidal.diagnostics
   :members:
   :undoc-members:
   :show-inheritance:

----

Visualization (``pyna.toroidal.visual``)
------------------------------------

Publication-quality plotting for tokamak research figures.
All functions follow a composable, axes-based API suitable for
single-column PRL/Nuclear Fusion figures.

.. automodule:: pyna.toroidal.visual.tokamak_manifold
   :members:
   :undoc-members:
   :show-inheritance:

----

Non-resonant Torus Deformation (``pyna.toroidal.torus_deformation``)
---------------------------------------------------------------------

Implements the analytic spectral theory of Wei (2025) for the deformation
of invariant tori (flux surfaces) under external magnetic perturbations.

All formulas work in flux coordinates :math:`(r, \theta, \varphi)` with
the Fourier convention :math:`f(\theta,\varphi)=\sum_{mn} f_{mn}\,e^{i(m\theta+n\varphi)}`.

**Key results implemented:**

- **Theorem 2** -- full :math:`(\delta r, \delta\theta, \delta\varphi)_{mn}` spectra
- **Theorem 3** -- universal 1-D Poincaré-section ring deformation
- **Eq. (4.2)** -- mean radial displacement for PF coil perturbations
- **Eq. (4.3)** -- DC (m=n=0) simplification
- **Eq. (5.1)** -- second-order non-axisymmetric formula

.. automodule:: pyna.toroidal.torus_deformation
   :members:
   :undoc-members:
   :show-inheritance:

