cyna Acceleration Layer
=======================

``cyna`` is the C++ acceleration layer shipped with pyna.  It is used where
Python hot loops are not acceptable: field-line tracing, Poincare batches,
fixed-point scans, connection-length/wall hits, coil fields, and functional
perturbation theory kernels.

Build Contract
--------------

``pyna._cyna`` expects a compiled ``_cyna_ext`` binary in the package.  Source
installs build it through xmake; PyPI wheels include it.  See
:doc:`../installation` for platform setup and CUDA flags.

The canonical cylindrical field-cache order is:

.. code-block:: text

   BR, BZ, BPhi, R_grid, Z_grid, Phi_grid

Use :func:`pyna._cyna.prepare_field_cache` to convert a
``pyna.fields.VectorFieldCylind`` or a legacy dict into C-contiguous arrays.

High-level vs Low-level APIs
----------------------------

Prefer high-level wrappers for application code:

- ``pyna.flt`` and ``pyna.toroidal.flt`` for tracing
- ``pyna.topo`` for Poincare maps, cycles, islands, manifolds and FPT response
- ``pyna.toroidal.coils`` for coil field construction

Use ``pyna._cyna`` directly only at bridge boundaries, for diagnostics, or when
writing a new high-level wrapper.

Python Wrapper Reference
------------------------

.. automodule:: pyna._cyna
   :no-index:
   :members:
   :show-inheritance:

Utility Helpers
---------------

.. automodule:: pyna._cyna.utils
   :no-index:
   :members:
   :show-inheritance:
