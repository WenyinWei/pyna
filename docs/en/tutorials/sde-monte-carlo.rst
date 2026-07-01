SDE Monte Carlo Distributions
=============================

This tutorial shows the practical SDE workflow in pyna:

1. define an Ito model with ``BrownianMotion``, ``GeometricBrownianMotion`` or
   ``ItoSDE``;
2. generate a reproducible sample path as a ``Trajectory``;
3. run a vectorized Monte Carlo ensemble for distribution estimates;
4. compare empirical mean, variance and quantiles with analytic formulas when
   available.

Use pyna's SDE classes for the model boundary and for single-path geometry.
Use vectorized NumPy arrays for large ensembles until pyna grows a dedicated
ensemble geometry class.  This keeps the mathematical object model honest:
a single realization is a sampled trajectory, while a cloud of realizations is
a statistical estimator.

.. note::

   The executable notebook below is committed with saved outputs and has
   ``nbsphinx`` execution disabled.  Re-run it locally when changing numerical
   parameters; the docs workflow will render those saved outputs on GitHub
   Pages.

Executable notebook:

- :doc:`/notebooks/tutorials/sde_monte_carlo_distribution`

Copy-Paste Pattern
------------------

.. code-block:: python

   import numpy as np
   from pyna.dynamics import GeometricBrownianMotion

   gbm = GeometricBrownianMotion(mu=[0.08], sigma=[0.20])
   one_path = gbm.euler_maruyama([100.0], (0.0, 1.0), dt=1/252, rng=7)
   print(one_path.final)  # TimeSeriesSolution is a pyna Trajectory

   n_paths = 200_000
   rng = np.random.default_rng(20260701)
   z = rng.normal(size=n_paths)
   log_terminal = (
       np.log(100.0)
       + gbm.expected_log_growth()[0] * 1.0
       + gbm.sigma[0] * np.sqrt(1.0) * z
   )
   terminal = np.exp(log_terminal)
   print(np.mean(terminal), np.quantile(terminal, [0.05, 0.5, 0.95]))

Extension Notes
---------------

- ``ItoSDE.diffusion_matrix`` accepts scalar, vector or matrix diffusion.
- ``ItoSDE.euler_maruyama`` accepts externally supplied ``dW`` increments, so
  common-random-number experiments and regression tests can be deterministic.
- Promote one sample path through topology objects only when the geometry
  claim is meaningful.  Monte Carlo samples estimate distributions; they are
  not automatically invariant sets.
