Dynamics Workflows and Extension Helpers
========================================

``pyna.topo`` exposes workflow helpers around the core topology objects.  The
main user-facing entry point is ``TopologyWorkflow``; the lower-level Protocol,
Adapter, Builder, Bridge and Factory modules remain available for downstream
libraries that need stable extension points.

Workflow Facade
---------------

``TopologyWorkflow`` is designed for notebooks and day-to-day scripts.  It
combines system construction, integration/iteration, explicit promotion and
section cuts without adding a new mathematical object type.

.. automodule:: pyna.topo.workflow
   :no-index:
   :members:
   :show-inheritance:

Protocols
---------

Structural protocols describe the extension contracts for external systems.
Third-party objects can participate by implementing the required attributes and
methods; subclassing pyna classes is optional.

.. automodule:: pyna.topo.protocols
   :no-index:
   :members:
   :show-inheritance:

Adapters
--------

Adapters normalize arrays, solver outputs and existing pyna objects into core
geometry representations.  They do not silently promote sampled data to
invariant objects.

.. automodule:: pyna.topo.adapters
   :no-index:
   :members:
   :show-inheritance:

Builders
--------

Builders encode explicit promotion rules.  For example, a trajectory can be
promoted to a ``Cycle`` only through a builder or adapter call that can require
closed samples.

.. automodule:: pyna.topo.builders
   :no-index:
   :members:
   :show-inheritance:

Bridges
-------

Bridges connect the continuous-time and discrete-time object families:
``Cycle -> PeriodicOrbit`` and ``Tube/TubeChain -> IslandChain``.

.. automodule:: pyna.topo.bridges
   :no-index:
   :members:
   :show-inheritance:

Factories and Registries
------------------------

Factories provide stable construction entry points for systems, geometry and
Poincare maps.  Registries are explicit and duplicate-safe so tests and
downstream libraries can isolate their own extensions.

.. automodule:: pyna.topo.factories
   :no-index:
   :members:
   :show-inheritance:
