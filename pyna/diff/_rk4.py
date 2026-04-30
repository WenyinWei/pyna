"""numpy-based fixed-step RK4 integrator.

Drop-in replacement for ``scipy.integrate.solve_ivp`` for field-line and
variational ODEs in pyna.  Uses a fixed step size (no adaptive step control)
which is faster for smooth ODEs with a known scale.

Replaced scipy.integrate.solve_ivp throughout pyna for performance:
the C++ cyna backend is used for field-line tracing where possible;
for variational/augmented ODEs this numpy RK4 avoids scipy overhead.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Optional, Sequence


class _RK4Sol:
    """Mimics the ``OdeResult`` object returned by ``scipy.integrate.solve_ivp``."""

    def __init__(self, success: bool, message: str,
                 t: np.ndarray, y: np.ndarray,
                 t_events=None, y_events=None,
                 sol: Optional[Callable] = None):
        self.success = success
        self.message = message
        self.t = t
        self.y = y
        self.t_events = t_events or []
        self.y_events = y_events or []
        self._sol = sol

    @property
    def sol(self):
        return self._sol


def _make_interp(ts: np.ndarray, ys: np.ndarray):
    """Build a callable dense interpolant (like sol returned by solve_ivp)."""
    from scipy.interpolate import interp1d
    return interp1d(ts, ys, axis=1, kind='cubic',
                    bounds_error=False, fill_value='extrapolate')


def rk4_solve(fun: Callable, t_span, y0,
              max_step: float = None,
              t_eval=None,
              dense_output: bool = False,
              events=None,
              rtol=None, atol=None, method=None, args=None,
              **kwargs) -> _RK4Sol:
    """Fixed-step RK4 integrator with scipy.integrate.solve_ivp-compatible interface.

    Parameters
    ----------
    fun : callable(t, y) -> array_like
    t_span : (t0, tf)
    y0 : array_like, initial state
    max_step : float, optional
        Step size.  If None, uses (tf-t0)/200.
    t_eval : array_like, optional
        Times at which to report the solution.
    dense_output : bool
        If True, attaches a cubic-spline interpolant as ``sol``.
    events : list of callables, optional
        Each callable ``ev(t, y)`` returns a float; if ``ev.terminal=True``
        integration stops when the value crosses zero from positive to negative.
    rtol, atol, method, args : ignored (kept for API compatibility)

    Returns
    -------
    _RK4Sol with .success, .message, .t, .y (shape n×N), .sol
    """
    t0 = float(t_span[0])
    tf = float(t_span[-1])
    y = np.asarray(y0, dtype=float).copy()
    direction = 1.0 if tf >= t0 else -1.0
    span = abs(tf - t0)

    if max_step is None or max_step <= 0 or not np.isfinite(max_step):
        max_step = span / 200.0
    max_step = float(max_step)

    n_steps = max(int(np.ceil(span / max_step)), 1)
    h = direction * span / n_steps

    # Storage
    ts = np.empty(n_steps + 1)
    ys = np.empty((len(y), n_steps + 1))
    ts[0] = t0
    ys[:, 0] = y

    # Event state for sign-change detection
    event_vals_prev = None
    if events:
        event_vals_prev = [ev(t0, y) for ev in events]

    terminated = False
    t = t0
    for i in range(n_steps):
        k1 = np.asarray(fun(t, y), dtype=float)
        k2 = np.asarray(fun(t + h * 0.5, y + h * 0.5 * k1), dtype=float)
        k3 = np.asarray(fun(t + h * 0.5, y + h * 0.5 * k2), dtype=float)
        k4 = np.asarray(fun(t + h, y + h * k3), dtype=float)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t = t0 + (i + 1) * h  # avoid drift
        ts[i + 1] = t
        ys[:, i + 1] = y

        # Check events
        if events:
            for j, ev in enumerate(events):
                val = ev(t, y)
                if getattr(ev, 'terminal', False):
                    prev = event_vals_prev[j]
                    # stop when sign changes from positive to negative
                    if prev > 0 and val <= 0:
                        ts = ts[:i + 2]
                        ys = ys[:, :i + 2]
                        terminated = True
                        break
                event_vals_prev[j] = val
            if terminated:
                break

    # Trim to actual steps used
    ts = ts[:len(ts)]  # already correct length if not terminated

    # Build interpolant
    sol_func = None
    if dense_output:
        sol_func = _make_interp(ts, ys)

    # Evaluate at t_eval
    if t_eval is not None:
        t_eval_arr = np.asarray(t_eval, dtype=float)
        if sol_func is not None:
            y_out = sol_func(t_eval_arr)
        else:
            interp = _make_interp(ts, ys)
            y_out = interp(t_eval_arr)
        return _RK4Sol(True, 'OK', t_eval_arr, y_out, sol=sol_func)

    return _RK4Sol(True, 'OK', ts, ys, sol=sol_func)


# Alias for compatibility with existing imports
rk4_integrate = rk4_solve
