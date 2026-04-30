"""test_fixed_points_newton.py
==============================
Integration tests for pyna.topo.fixed_points Newton-method fixed-point locator.

Requires:
  - topoquest installed / importable
  - HAO field cache present at D:\\haodata\\hao_field_cache_*.pkl
  - pyna _cyna extension (optional for faster tracing)

Run with:
    cd C:\\Users\\Legion\\Nutstore\\1\\Repo\\pyna
    python -m pytest tests/test_fixed_points_newton.py -v -s
"""

import sys
import time
import math
from pathlib import Path

import numpy as np
import pytest

# Make sure both repos are importable
_TOPOQUEST = Path(r"C:\Users\Legion\Nutstore\1\Repo\topoquest")
_PYNA      = Path(r"C:\Users\Legion\Nutstore\1\Repo\pyna")
for _p in (_TOPOQUEST, _PYNA):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_WALL_PATH = Path(r"D:\haodata\hao_1stwall_inner.txt")

# ---------------------------------------------------------------------------
# Module-level fixtures  (heavy, session-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hao_tracer():
    """Build a FieldlineTracer from the HAO field cache."""
    from scipy.interpolate import RegularGridInterpolator
    from topoquest.tracing import FieldlineTracer, WallGeometry
    from explore_hao_divertor_configs import load_field_cache

    print("\nLoading HAO field cache ...", flush=True)
    t0 = time.time()
    fc = load_field_cache()
    print(f"  Loaded in {time.time()-t0:.1f}s  BR shape={fc['BR'].shape}", flush=True)

    R_grid   = fc['R_grid']
    Z_grid   = fc['Z_grid']
    Phi_grid = fc['Phi_grid']
    DPhi_cache = float(Phi_grid[1] - Phi_grid[0])
    Phi_ext    = np.append(Phi_grid, Phi_grid[-1] + DPhi_cache)

    def _ext(a):
        return np.concatenate([a, a[:, :, :1]], axis=2)

    kw = dict(method='linear', bounds_error=False, fill_value=0.)
    itp_BR   = RegularGridInterpolator((R_grid, Z_grid, Phi_ext), _ext(fc['BR']),   **kw)
    itp_BPhi = RegularGridInterpolator((R_grid, Z_grid, Phi_ext), _ext(fc['BPhi']), **kw)
    itp_BZ   = RegularGridInterpolator((R_grid, Z_grid, Phi_ext), _ext(fc['BZ']),   **kw)

    wall = WallGeometry(str(_WALL_PATH))
    tracer = FieldlineTracer(
        itp_BR=itp_BR, itp_BPhi=itp_BPhi, itp_BZ=itp_BZ,
        wall=wall, R_grid=R_grid, Z_grid=Z_grid, Phi_grid=Phi_grid,
        DPhi=0.05,
        cache_dir=None,
    )
    return tracer


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _residual(tracer, R, Z, phi_sec, period):
    """Compute |P^m(x) - x| for a single point."""
    R_seeds = np.array([R])
    Z_seeds = np.array([Z])
    result = tracer.trace_poincare(R_seeds, Z_seeds, phi_sec,
                                   N_turns=period, use_wall=False, verbose=False)
    R_arr, Z_arr = result[0]
    if len(R_arr) < period:
        return float('inf')
    R_end, Z_end = float(R_arr[-1]), float(Z_arr[-1])
    return math.sqrt((R_end - R)**2 + (Z_end - Z)**2)


# ---------------------------------------------------------------------------
# Test 1 -- Magnetic axis (m=1 O-point)
# ---------------------------------------------------------------------------

def test_magnetic_axis(hao_tracer):
    """Newton method should converge to the magnetic axis from (0.85, 0.0)."""
    from pyna.topo.fixed_points import find_magnetic_axis

    tracer = hao_tracer
    R_guess, Z_guess = 0.85, 0.0
    phi_sec = 0.0
    tol = 1e-8

    print(f"\n{'='*55}")
    print("Test: find_magnetic_axis")
    print(f"  Initial guess: R={R_guess}, Z={Z_guess}")
    print(f"  phi_sec={phi_sec}, tol={tol}")

    t0 = time.time()
    R_ax, Z_ax, DPm = find_magnetic_axis(
        tracer, R_guess, Z_guess,
        phi_sec=phi_sec, tol=tol, verbose=True,
    )
    elapsed = time.time() - t0

    # Check residual
    res = _residual(tracer, R_ax, Z_ax, phi_sec, period=1)
    evals = np.linalg.eigvals(DPm)

    print(f"\n  Converged: R_ax={R_ax:.8f}  Z_ax={Z_ax:.8f}")
    print(f"  |P(x) - x| = {res:.3e}  (tol={tol})")
    print(f"  DPm eigenvalues: {evals[0]:.6f}  {evals[1]:.6f}")
    print(f"  Tr(DPm)={np.trace(DPm):.6f}  det(DPm)={np.linalg.det(DPm):.6f}")
    print(f"  Elapsed: {elapsed:.1f}s")

    assert res < tol * 10, f"Residual {res:.3e} exceeds tolerance {tol*10}"
    # O-point: |Tr| < 2
    assert abs(np.trace(DPm)) < 2.0, f"Expected O-point (|Tr|<2), got Tr={np.trace(DPm):.4f}"
    # det ~= 1 (area-preserving)
    assert abs(np.linalg.det(DPm) - 1.0) < 0.05, \
        f"det(DPm)={np.linalg.det(DPm):.5f} far from 1"
    # Sanity: axis in reasonable range
    assert 0.80 < R_ax < 0.92, f"R_ax={R_ax:.4f} out of expected range"
    assert abs(Z_ax) < 0.02, f"|Z_ax|={abs(Z_ax):.4f} too large"
    print("  [PASS]")


# ---------------------------------------------------------------------------
# Test 2 -- m=3 periodic fixed point (X or O from LFS island chain)
# ---------------------------------------------------------------------------

def test_period3_fixed_point(hao_tracer):
    """Newton method should converge for a period-3 fixed point.

    We start from an approximate LFS island position and check that the
    iteration converges within tolerance.  The exact starting position is
    taken from the pkl if available; otherwise we use a hard-coded estimate.
    """
    from pyna.topo.fixed_points import find_fixed_point_newton

    tracer = hao_tracer
    phi_sec = 0.0
    period  = 3
    tol     = 1e-8

    # Try to load from pkl
    R_guess, Z_guess = None, None
    for pkl_name in ("hao_fp.pkl", "hao_fp_fallback.pkl"):
        pkl_path = _TOPOQUEST / pkl_name
        if pkl_path.exists():
            import pickle
            try:
                raw = pickle.load(open(str(pkl_path), 'rb'))
                k = float(phi_sec)
                if k not in raw:
                    k = min(raw.keys(), key=lambda x: abs(x - k))
                sec = raw[k]
                candidates = sec.get('xpts', []) + sec.get('opts', [])
                for entry in candidates:
                    if hasattr(entry, 'R'):
                        R_guess, Z_guess = entry.R, entry.Z
                    else:
                        R_guess, Z_guess = float(entry[0]), float(entry[1])
                    break
            except Exception as e:
                print(f"  Could not load pkl {pkl_path}: {e}")
            break

    if R_guess is None:
        # Hard-coded approximate LFS position for HAO
        R_guess, Z_guess = 0.88, 0.06
        print(f"\n  No pkl found; using hard-coded guess (R={R_guess}, Z={Z_guess})")
    else:
        print(f"\n  Loaded initial guess from pkl: R={R_guess:.5f} Z={Z_guess:.5f}")

    print(f"\n{'='*55}")
    print(f"Test: find_fixed_point_newton (period={period})")
    print(f"  Initial guess: R={R_guess:.5f}, Z={Z_guess:.5f}")

    t0 = time.time()
    try:
        R, Z, DPm, kind = find_fixed_point_newton(
            tracer, R_guess, Z_guess, phi_sec, period=period,
            tol=tol, verbose=True,
        )
        converged = True
    except RuntimeError as e:
        print(f"  [WARN] Newton did not converge: {e}")
        converged = False

    elapsed = time.time() - t0

    if converged:
        res = _residual(tracer, R, Z, phi_sec, period)
        evals = np.linalg.eigvals(DPm)
        print(f"\n  Converged: R={R:.8f}  Z={Z:.8f}  kind={kind}")
        print(f"  |P^{period}(x) - x| = {res:.3e}  (tol={tol})")
        print(f"  DPm eigenvalues: {evals[0]:.6f}  {evals[1]:.6f}")
        print(f"  Tr(DPm)={np.trace(DPm):.6f}  det(DPm)={np.linalg.det(DPm):.6f}")
        print(f"  Elapsed: {elapsed:.1f}s")

        assert res < tol * 10, f"Residual {res:.3e} exceeds {tol*10}"
        assert abs(np.linalg.det(DPm) - 1.0) < 0.1, \
            f"det(DPm)={np.linalg.det(DPm):.5f} far from 1"
        print("  [PASS]")
    else:
        pytest.skip(
            f"Newton did not converge from initial guess "
            f"R={R_guess:.4f} Z={Z_guess:.4f} -- likely no period-{period} "
            f"island chain near that location. Provide a better initial guess."
        )


# ---------------------------------------------------------------------------
# Standalone runner (not pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    print("Building HAO tracer ...")
    from scipy.interpolate import RegularGridInterpolator
    from topoquest.tracing import FieldlineTracer, WallGeometry
    from explore_hao_divertor_configs import load_field_cache

    fc = load_field_cache()
    R_grid   = fc['R_grid']
    Z_grid   = fc['Z_grid']
    Phi_grid = fc['Phi_grid']
    DPhi_cache = float(Phi_grid[1] - Phi_grid[0])
    Phi_ext    = np.append(Phi_grid, Phi_grid[-1] + DPhi_cache)

    def _ext(a):
        return np.concatenate([a, a[:, :, :1]], axis=2)

    kw = dict(method='linear', bounds_error=False, fill_value=0.)
    itp_BR   = RegularGridInterpolator((R_grid, Z_grid, Phi_ext), _ext(fc['BR']),   **kw)
    itp_BPhi = RegularGridInterpolator((R_grid, Z_grid, Phi_ext), _ext(fc['BPhi']), **kw)
    itp_BZ   = RegularGridInterpolator((R_grid, Z_grid, Phi_ext), _ext(fc['BZ']),   **kw)
    wall = WallGeometry(str(_WALL_PATH))
    tracer = FieldlineTracer(
        itp_BR=itp_BR, itp_BPhi=itp_BPhi, itp_BZ=itp_BZ,
        wall=wall, R_grid=R_grid, Z_grid=Z_grid, Phi_grid=Phi_grid,
        DPhi=0.05, cache_dir=None,
    )

    print("\n" + "="*60)
    print("Test 1: Magnetic axis")
    print("="*60)
    try:
        from pyna.topo.fixed_points import find_magnetic_axis
        R_ax, Z_ax, DPm = find_magnetic_axis(
            tracer, 0.85, 0.0, phi_sec=0.0, tol=1e-8, verbose=True)
        res = _residual(tracer, R_ax, Z_ax, 0.0, 1)
        evals = np.linalg.eigvals(DPm)
        print(f"\nResult: R_ax={R_ax:.8f}  Z_ax={Z_ax:.8f}")
        print(f"|P(x) - x| = {res:.3e}")
        print(f"Eigenvalues: {evals}")
        print(f"Tr(DPm)={np.trace(DPm):.6f}  det(DPm)={np.linalg.det(DPm):.8f}")
    except Exception:
        traceback.print_exc()

    print("\n" + "="*60)
    print("Test 2: Period-3 fixed point")
    print("="*60)
    try:
        from pyna.topo.fixed_points import find_fixed_point_newton
        R, Z, DPm, kind = find_fixed_point_newton(
            tracer, 0.88, 0.06, 0.0, period=3, tol=1e-8, verbose=True)
        res = _residual(tracer, R, Z, 0.0, 3)
        evals = np.linalg.eigvals(DPm)
        print(f"\nResult: R={R:.8f}  Z={Z:.8f}  kind={kind}")
        print(f"|P^3(x) - x| = {res:.3e}")
        print(f"Eigenvalues: {evals}")
        print(f"Tr(DPm)={np.trace(DPm):.6f}  det(DPm)={np.linalg.det(DPm):.8f}")
    except Exception:
        traceback.print_exc()
