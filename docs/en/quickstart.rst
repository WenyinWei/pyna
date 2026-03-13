Quick Start
===========

Field Line Tracing
------------------
.. code-block:: python

   import numpy as np
   from pyna.MCF.equilibrium.Solovev import SolovevEquilibrium
   from pyna.flt import trace_fieldlines

   # Build analytic equilibrium (Cerfon & Freidberg 2010)
   eq = SolovevEquilibrium(R0=1.86, a=0.6, B0=5.3, kappa=1.7, delta=0.33, q0=3.0)

   # Trace field lines
   starts = np.array([[1.86 + 0.3*i, 0.0] for i in range(5)])
   result = trace_fieldlines(eq.field_line_rhs, starts, n_turns=100)

Poincaré Map
------------
.. code-block:: python

   from pyna.topo.poincare import poincare_from_fieldlines
   
   section_R = np.linspace(1.2, 2.4, 200)
   points = poincare_from_fieldlines(eq.field_line_rhs, section_R, n_turns=500)

FPT Topology Control
--------------------
.. code-block:: python

   from pyna.control.fpt import A_matrix, cycle_shift
   from pyna.MCF.coils.coil import Coil
   
   # Compute FPT A-matrix at X-point
   R_xpt, Z_xpt = 1.55, -0.85
   A = A_matrix(eq.field_line_rhs, R_xpt, Z_xpt)
   
   # Predict X-point shift under coil perturbation
   coil = Coil(R=2.5, Z=-1.2, I=1000)
   delta_g = coil.delta_g_at(R_xpt, Z_xpt, eq)
   shift = cycle_shift(A, delta_g)
   print(f"X-point shift: ?R={shift[0]*1000:.2f} mm, ?Z={shift[1]*1000:.2f} mm")
