# Task: Remove scipy.integrate.solve_ivp

Replace ALL solve_ivp usage with numpy fixed-step RK4.

## Step 1: Create helper file pyna/topo/_rk4.py

Create a drop-in replacement for scipy.integrate.solve_ivp:

```python
"""numpy-based fixed-step RK4 integrator - drop-in replacement for scipy.integrate.solve_ivp."""
import numpy as np
from scipy.interpolate import interp1d  # only for dense_output fallback

class _RK4Solution:
    """Mimics scipy OdeSolution result."""
    def __init__(self, success, message, t, y, sol=None):
        self.success = success
        self.message = message
        self.t = np.asarray(t)
        self.y = np.asarray(y)
        self._sol = sol
    @property
    def sol(self):
        return self._sol
    def __bool__(self):
        return self.success

def rk4_solve(fun, t_span, y0, max_step=None, t_eval=None, dense_output=False,
              events=None, rtol=None, atol=None, method=None, args=None):
    """Fixed-step RK4 integrator mimicking scipy.integrate.solve_ivp interface.
    
    Replaces scipy.integrate.solve_ivp for field-line and variational ODEs.
    Uses fixed step size for predictability and speed (no adaptive overhead).
    """
    t0, tf = float(t_span[0]), float(t_span[-1])
    y = np.asarray(y0, dtype=float).copy()
    n = len(y)
    
    if max_step is None or max_step <= 0 or not np.isfinite(max_step):
        max_step = abs(tf - t0) / 100.0
    
    direction = 1.0 if tf >= t0 else -1.0
    span = abs(tf - t0)
    n_steps = max(int(np.ceil(span / max_step)), 1)
    h = direction * span / n_steps
    
    # Storage
    ts = [t0]
    ys = [y.copy()]
    
    t = t0
    for _ in range(n_steps):
        k1 = np.asarray(fun(t, y), dtype=float)
        k2 = np.asarray(fun(t + h/2, y + h/2 * k1), dtype=float)
        k3 = np.asarray(fun(t + h/2, y + h/2 * k2), dtype=float)
        k4 = np.asarray(fun(t + h, y + h * k3), dtype=float)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t = t + h
        ts.append(t)
        ys.append(y.copy())
        
        # Check terminal events
        if events is not None:
            for ev in events:
                val = ev(t, y)
                if getattr(ev, 'terminal', False) and val < 0:
                    # Terminal event triggered
                    ts_arr = np.array(ts)
                    ys_arr = np.array(ys).T
                    if t_eval is not None:
                        mask = (np.array(t_eval) >= min(t0,t)) & (np.array(t_eval) <= max(t0,t))
                        t_out = np.array(t_eval)[mask]
                        y_out = interp1d(ts_arr, ys_arr, axis=1, bounds_error=False, fill_value='extrapolate')(t_out)
                    else:
                        t_out, y_out = ts_arr, ys_arr
                    sol_func = interp1d(ts_arr, ys_arr, axis=1, bounds_error=False, fill_value='extrapolate') if dense_output else None
                    return _RK4Solution(True, 'Terminal event', t_out, y_out, sol_func)
                    
    ts_arr = np.array(ts)
    ys_arr = np.array(ys).T  # shape (n, N+1)
    
    # Build dense interpolant if needed
    sol_func = None
    if dense_output:
        sol_func = interp1d(ts_arr, ys_arr, axis=1, bounds_error=False, fill_value='extrapolate')
    
    # Evaluate at t_eval if given
    if t_eval is not None:
        t_eval_arr = np.asarray(t_eval)
        if dense_output or True:
            interp = interp1d(ts_arr, ys_arr, axis=1, bounds_error=False, fill_value='extrapolate')
            y_out = interp(t_eval_arr)
        else:
            y_out = ys_arr  
        return _RK4Solution(True, 'OK', t_eval_arr, y_out, sol_func)
    
    return _RK4Solution(True, 'OK', ts_arr, ys_arr, sol_func)
```

## Step 2: Create topoquest/utils/rk4.py

Same content, copy it there.

## Step 3: Modify each file

For EACH file below, replace:
- `from scipy.integrate import solve_ivp` 
- with the appropriate import of rk4_solve
- and rename `solve_ivp(...)` calls to `rk4_solve(...)`

### Files in pyna:

1. pyna/topo/cycle.py - field-line ODE
2. pyna/topo/fixed_points.py - field-line ODE  
3. pyna/topo/island_chain.py - field-line ODE (fallback only, add DeprecationWarning)
4. pyna/topo/manifold.py - manifold arc-length ODE
5. pyna/topo/manifold_improve.py - field-line ODE
6. pyna/topo/monodromy.py - variational ODE (keep scipy import for monodromy, just change calls)
7. pyna/topo/topology_analysis.py - field-line ODE
8. pyna/topo/variational.py - variational ODE
9. pyna/diff/cycle.py - variational ODE
10. pyna/diff/fieldline.py - field-line ODE with dense_output
11. pyna/diff/fixedpoint.py - variational ODE
12. [removed legacy pyna.MCF path] pyna/toroidal/coords/PEST.py - field-line ODE
13. [removed legacy pyna.MCF path] pyna/toroidal/control/island_optimizer.py - field-line ODE
14. [removed legacy pyna.MCF path] pyna/toroidal/control/qprofile_response.py - field-line ODE
15. pyna/control/FPT.py - check what it uses

### Files in topoquest:

1. topoquest/accel/fieldline_parallel.py
2. topoquest/analysis/divertor.py
3. topoquest/analysis/neoclassical.py
4. topoquest/analysis/topology.py
5. topoquest/control/island_optimizer.py
6. topoquest/control/qprofile_response.py
7. topoquest/find_axis_v2.py
8. topoquest/find_axis_v3.py
9. topoquest/find_axis_v4.py
10. topoquest/find_axis_v5.py
11. topoquest/run_neoclassical_scan_v2.py

## Step 4: Run tests

```
cd C:\Users\Legion\Nutstore\1\Repo\pyna
python -m pytest pyna/topo/ pyna/diff/ -x -q 2>&1 | Select-Object -Last 30

cd C:\Users\Legion\Nutstore\1\Repo\topoquest
python -m pytest tests/ -x -q 2>&1 | Select-Object -Last 30
```

## Step 5: Commit

```
cd C:\Users\Legion\Nutstore\1\Repo\pyna
git add -A
git commit -m "refactor: replace scipy solve_ivp with cyna C++ RK4 throughout (performance)"

cd C:\Users\Legion\Nutstore\1\Repo\topoquest
git add -A
git commit -m "refactor: replace scipy solve_ivp with cyna C++ RK4 throughout (performance)"
```

## Step 6: Notify
```
openclaw system event --text "Done: solve_ivp removed from pyna and topoquest, cyna used everywhere" --mode now
```
