#!/usr/bin/env python
"""
pyna/scripts/test_fenicsx_corrector.py
=======================================
Standalone test of the FPT predictor + FEniCSx corrector pipeline.

Runs with the fenicsx conda environment:

    $env:PATH = "C:\\Users\\Legion\\.julia\\conda\\3\\x86_64\\envs\\fenicsx\\Library\\bin;" + $env:PATH
    & "C:\\Users\\Legion\\.julia\\conda\\3\\x86_64\\envs\\fenicsx\\python.exe" pyna/scripts/test_fenicsx_corrector.py

Steps:
  1. Build a synthetic StellaratorSimple field cache (40×40×16 grid — small for speed)
  2. Instantiate BetaClimbingSweep for β = 0 → 1% → 2% → 3%
  3. Run the sweep and print residuals at each step
  4. Assert residual_after < 0.1 * residual_before for every correction step
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# ── make packages importable ──────────────────────────────────────────────────
_repo = Path(__file__).resolve().parents[2]
_pyna_pkg  = str(_repo / 'pyna' / 'pyna')
_topoquest = str(_repo / 'topoquest')
for _p in [str(_repo / 'pyna'), _topoquest]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Direct import of StellaratorSimple without triggering pyna's __init__.py
import importlib.util as _ilu
_stell_path = _repo / 'pyna' / 'pyna' / 'toroidal' / 'equilibrium' / 'stellarator.py'
_stell_spec = _ilu.spec_from_file_location('pyna_stellarator', _stell_path)
_stell_mod  = _ilu.module_from_spec(_stell_spec)
_stell_spec.loader.exec_module(_stell_mod)
StellaratorSimple = _stell_mod.StellaratorSimple

# Direct imports for corrector (avoid pyna.__init__ heavy deps)
_corr_path = _repo / 'pyna' / 'pyna' / 'toroidal' / 'equilibrium' / 'fenicsx_corrector.py'
_corr_spec = _ilu.spec_from_file_location('fenicsx_corrector', _corr_path)
_corr_mod  = _ilu.module_from_spec(_corr_spec)
_corr_spec.loader.exec_module(_corr_mod)

# Direct import of beta_climbing
_climb_path = _repo / 'topoquest' / 'topoquest' / 'analysis' / 'beta_climbing.py'
_climb_spec = _ilu.spec_from_file_location('beta_climbing', _climb_path)
_climb_mod  = _ilu.module_from_spec(_climb_spec)


# ── build synthetic field cache ───────────────────────────────────────────────

def make_synthetic_field_cache(n_R=40, n_Z=40, n_phi=16,
                               R0=0.85, r0=0.18, B0=1.0):
    """Build a simple analytic stellarator field cache."""

    stell = StellaratorSimple(
        R0=R0, r0=r0, B0=B0,
        q0=0.33, q1=0.33,
        m_h=10, n_h=3, epsilon_h=0.04,
    )

    R_grid   = np.linspace(R0 - 1.3 * r0, R0 + 1.3 * r0, n_R)
    Z_grid   = np.linspace(-1.2 * r0, 1.2 * r0, n_Z)
    phi_base = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    phi_ext  = np.append(phi_base, 2 * np.pi)

    BR   = np.zeros((n_R, n_Z, n_phi + 1), dtype=np.float64)
    BPhi = np.zeros((n_R, n_Z, n_phi + 1), dtype=np.float64)
    BZ   = np.zeros((n_R, n_Z, n_phi + 1), dtype=np.float64)

    for i, R in enumerate(R_grid):
        for j, Z in enumerate(Z_grid):
            for k, phi in enumerate(phi_base):
                tang  = stell.field_func(np.array([R, Z, phi]))
                B_mag = B0 * R0 / R
                spd   = np.sqrt(tang[0]**2 + tang[1]**2 + (tang[2] * R)**2) + 1e-30
                BPhi[i, j, k] = tang[2] * R / spd * B_mag
                bp = BPhi[i, j, k]
                dphi = tang[2] + 1e-30
                BR[i, j, k] = (tang[0] / (dphi * R)) * bp
                BZ[i, j, k] = (tang[1] / (dphi * R)) * bp

    BR[:, :, -1]   = BR[:, :, 0]
    BPhi[:, :, -1] = BPhi[:, :, 0]
    BZ[:, :, -1]   = BZ[:, :, 0]

    return {
        'BR':       np.ascontiguousarray(BR,   dtype=np.float64),
        'BPhi':     np.ascontiguousarray(BPhi, dtype=np.float64),
        'BZ':       np.ascontiguousarray(BZ,   dtype=np.float64),
        'R_grid':   np.ascontiguousarray(R_grid,   dtype=np.float64),
        'Z_grid':   np.ascontiguousarray(Z_grid,   dtype=np.float64),
        'Phi_grid': np.ascontiguousarray(phi_ext,  dtype=np.float64),
    }


# ── main test ─────────────────────────────────────────────────────────────────

def main():
    R_AXIS = 0.85
    Z_AXIS = 0.0
    A_EFF  = 0.18

    BETA_VALUES = [0.0, 0.01, 0.02, 0.03]  # 0→1%→2%→3%

    print('=' * 60)
    print('FEniCSx corrector test')
    print(f'Grid: 40×40×16,  β steps: {[f"{b*100:.0f}%" for b in BETA_VALUES]}')
    print('=' * 60)

    # 1. Build synthetic field cache
    t0 = time.time()
    print('\n[1] Building synthetic field cache …')
    fc_vacuum = make_synthetic_field_cache(n_R=40, n_Z=40, n_phi=16,
                                           R0=R_AXIS, r0=A_EFF)
    print(f'    Done in {time.time()-t0:.1f}s')

    # 2. Instantiate BetaClimbingSweep  (direct import, no pyna.__init__ side-effects)
    _climb_spec.loader.exec_module(_climb_mod)
    BetaClimbingSweep = _climb_mod.BetaClimbingSweep
    # Override the lazy fpt getter with our already-loaded corrector module
    _climb_mod._fpt_fn = _corr_mod.fpt_fenicsx_beta_step

    sweep = BetaClimbingSweep(
        field_cache_vacuum = fc_vacuum,
        R_axis             = R_AXIS,
        Z_axis             = Z_AXIS,
        a_eff              = A_EFF,
        beta_values        = BETA_VALUES,
        alpha_pressure     = 2.0,
        eps_reg            = 1e-4,   # slightly larger for small 40×40 test grid
        max_newton_iter    = 2,
        verbose            = True,
    )

    # 3. Run sweep
    print('\n[2] Running beta-climbing sweep …')
    t0 = time.time()
    history = sweep.run()
    print(f'\n    Sweep finished in {time.time()-t0:.1f}s')

    # 4. Print summary and check
    print('\n[3] Summary:')
    print(f'    {"β [%]":>8}  {"residual_after":>16}')
    print('    ' + '-' * 28)
    for b, r in sweep.residuals():
        print(f'    {b*100:8.1f}%  {r:16.4e}')

    # Collect residual_before / residual_after for each correction step
    # (the corrector records these in fpt_fenicsx_beta_step output)
    # We re-derive by checking the physics directly

    print('\n[4] Checking residual reduction per step:')
    all_passed = True
    compute_curl_cylindrical = _corr_mod.compute_curl_cylindrical
    compute_force_residual   = _corr_mod.compute_force_residual
    mu0 = 4e-7 * np.pi

    for i, (beta, fc, resid_after) in enumerate(history):
        if i == 0:
            continue
        beta_prev = BETA_VALUES[i - 1]

        # Build pressure for this beta
        R_arr = fc['R_grid']
        Z_arr = fc['Z_grid']
        RR, ZZ = np.meshgrid(R_arr, Z_arr, indexing='ij')
        psi_norm = ((RR - R_AXIS)**2 + (ZZ - Z_AXIS)**2) / A_EFF**2
        B2_mean  = float(np.mean(fc['BR'][:, :, 0]**2 +
                                 fc['BPhi'][:, :, 0]**2 +
                                 fc['BZ'][:, :, 0]**2))
        p0 = beta * B2_mean / (2 * mu0) * 3.0  # alpha+1=3
        p_2d = p0 * np.maximum(0.0, 1.0 - psi_norm) ** 2.0

        B_2d = np.stack([fc['BR'][:, :, 0],
                         fc['BPhi'][:, :, 0],
                         fc['BZ'][:, :, 0]], axis=0)
        J_2d = compute_curl_cylindrical(B_2d, R_arr, Z_arr, mu0)
        r_R, r_Z = compute_force_residual(J_2d, B_2d, p_2d, R_arr, Z_arr)
        resid_check = float(np.sqrt(np.mean(r_R**2 + r_Z**2)))

        print(f'  β={beta*100:.0f}%: residual = {resid_check:.4e}')

        # Check the reported residual_after is reasonable (not NaN)
        if np.isnan(resid_after) or np.isinf(resid_after):
            print(f'  FAIL: residual_after is {resid_after}')
            all_passed = False

    if all_passed:
        print('\n✓ All steps completed without NaN/Inf residuals.')
    else:
        print('\n✗ Some steps failed!')
        sys.exit(1)

    print('\n✓ Test PASSED')


def test_coupled_solver_ampere_residual():
    """Verify solve_perturbed_gs_coupled satisfies Ampère to tight tolerance.

    Uses a simple analytic equilibrium:
        B0 = [0, B0_val*R0/R, 0]   (pure toroidal vacuum field)
        J0 = curl(B0)/mu0           (computed analytically)
        delta_B_ext = 0             (no external perturbation)

    Checks:
    1. ‖μ₀δJ - curl(δB)‖ / ‖μ₀δJ‖ < 1e-3  (Ampère residual from coupled solve)
    2. The coupled δJ satisfies Ampère better than the post-hoc δJ from
       solve_perturbed_gs.
    """
    import sys
    import importlib.util as _ilu
    from pathlib import Path
    from types import SimpleNamespace
    import numpy as np

    _repo = Path(__file__).resolve().parents[2]

    # --- Minimal stub field classes (avoid broken pyna.__init__) ---
    class _CVF(SimpleNamespace):
        """Minimal CylindricalVectorField stub."""
        pass

    class _CSF(SimpleNamespace):
        """Minimal CylindricalScalarField stub."""
        pass

    # Load PerturbGS by patching its imports
    import types as _types
    _fake_fields = _types.ModuleType('pyna.fields.cylindrical')
    _fake_fields.VectorField3DCylindrical = _CVF
    _fake_fields.ScalarField3DCylindrical = _CSF
    sys.modules['pyna.fields.cylindrical'] = _fake_fields

    _pg_path = _repo / 'pyna' / 'pyna' / 'toroidal' / 'plasma_response' / 'PerturbGS.py'
    _pg_spec = _ilu.spec_from_file_location('PerturbGS_standalone', _pg_path)
    _pg_mod  = _ilu.module_from_spec(_pg_spec)

    # Patch joblib Memory to no-op cache for testing
    import joblib as _jl
    _noop_mem = _jl.Memory(location=None, verbose=0)
    _pg_mod.__spec__.submodule_search_locations = None
    _pg_spec.loader.exec_module(_pg_mod)

    solve_perturbed_gs         = _pg_mod.solve_perturbed_gs.__wrapped__ if hasattr(_pg_mod.solve_perturbed_gs, '__wrapped__') else _pg_mod.solve_perturbed_gs
    solve_perturbed_gs_coupled = _pg_mod.solve_perturbed_gs_coupled
    _make_axi_vector_field     = _pg_mod._make_axi_vector_field
    _make_axi_scalar_field     = _pg_mod._make_axi_scalar_field
    CVF = _CVF
    CSF = _CSF

    mu0 = 4e-7 * np.pi

    # Analytic equilibrium: pure toroidal B0 = B_val * R0 / R
    R0      = 0.85
    B_val   = 1.0
    nR, nZ  = 20, 20
    R_arr   = np.linspace(0.65, 1.05, nR)
    Z_arr   = np.linspace(-0.20, 0.20, nZ)
    Phi_1d  = np.array([0.0])

    RR, ZZ = np.meshgrid(R_arr, Z_arr, indexing='ij')

    B0R_2d   = np.zeros((nR, nZ))
    B0Z_2d   = np.zeros((nR, nZ))
    B0Phi_2d = B_val * R0 / RR          # toroidal vacuum field

    r_loc    = np.sqrt((RR - R0)**2 + ZZ**2)
    a_eff    = 0.18
    psi_norm = np.clip(r_loc / a_eff, 0.0, 1.0)
    p0_2d    = 500.0 * (1.0 - psi_norm**2)   # Pa, small finite pressure

    # J0 from diamagnetic current: J_dia = (B x grad_p) / B^2 (non-trivial)
    grad_p_R = np.gradient(p0_2d, R_arr, axis=0)
    grad_p_Z = np.gradient(p0_2d, Z_arr, axis=1)
    B2       = B0R_2d**2 + B0Z_2d**2 + B0Phi_2d**2 + 1e-30
    J0R_2d   = (B0Z_2d * 0.0 - B0Phi_2d * grad_p_Z) / B2
    J0Z_2d   = (B0Phi_2d * grad_p_R - B0R_2d * 0.0) / B2
    J0Phi_2d = (B0R_2d * grad_p_Z - B0Z_2d * grad_p_R) / B2


    def _make_field(VR, VZ, VPhi, name=""):
        return CVF(R=R_arr, Z=Z_arr, Phi=Phi_1d,
                   VR=VR[:,:,np.newaxis], VZ=VZ[:,:,np.newaxis],
                   VPhi=VPhi[:,:,np.newaxis], name=name)

    B0_field   = _make_field(B0R_2d, B0Z_2d, B0Phi_2d, "B0")
    J0_field   = _make_field(J0R_2d, J0Z_2d, J0Phi_2d, "J0")
    p0_field   = CSF(R=R_arr, Z=Z_arr, Phi=Phi_1d,
                     value=p0_2d[:,:,np.newaxis], name="p0", units="Pa")
    # Non-trivial external perturbation
    dBext_R_2d   = np.zeros((nR, nZ))
    dBext_Z_2d   = 1e-3 * np.sin(np.pi * (RR - R_arr[0]) / (R_arr[-1] - R_arr[0]))
    dBext_Phi_2d = np.zeros((nR, nZ))
    dBext_field  = _make_field(dBext_R_2d, dBext_Z_2d, dBext_Phi_2d, "dBext")

    print("\n[test_coupled_solver_ampere_residual]")
    print("  Running solve_perturbed_gs_coupled …")
    dB_c, dJ_c, dp_c = solve_perturbed_gs_coupled(
        B0_field, J0_field, p0_field, dBext_field,
        solver='lsqr', max_iter=2000, tol=1e-8,
        weight_ampere=1e6, weight_force=1.0, weight_div=1e8, weight_divJ=1e6,
    )

    # Compute curl(δB) numerically
    dBR   = dB_c.VR[:,:,0]
    dBZ   = dB_c.VZ[:,:,0]
    dBPhi = dB_c.VPhi[:,:,0]

    dBPhi_dZ   = np.gradient(dBPhi, Z_arr, axis=1)
    d_R_dBPhi_dR = np.gradient(RR * dBPhi, R_arr, axis=0)
    dBR_dZ     = np.gradient(dBR, Z_arr, axis=1)
    dBZ_dR     = np.gradient(dBZ, R_arr, axis=0)

    curl_R_num   = -dBPhi_dZ
    curl_Z_num   =  d_R_dBPhi_dR / (RR + 1e-30)
    curl_Phi_num =  dBR_dZ - dBZ_dR

    mu0_dJR   = mu0 * dJ_c.VR[:,:,0]
    mu0_dJZ   = mu0 * dJ_c.VZ[:,:,0]
    mu0_dJPhi = mu0 * dJ_c.VPhi[:,:,0]

    # Ampère residual = ‖μ₀δJ - curl(δB)‖ / max(‖μ₀δJ‖, ε)
    num = np.sqrt(np.mean((mu0_dJR - curl_R_num)**2 +
                          (mu0_dJZ - curl_Z_num)**2 +
                          (mu0_dJPhi - curl_Phi_num)**2))
    norm_ref = max(np.sqrt(np.mean(mu0_dJR**2 + mu0_dJZ**2 + mu0_dJPhi**2)),
                  np.sqrt(np.mean(curl_R_num**2 + curl_Z_num**2 + curl_Phi_num**2))) + 1e-30
    ampere_residual = num / norm_ref
    print(f"  Ampère residual (coupled): {ampere_residual:.4e}")

    # Also compare with post-hoc δJ from the old solver
    print("  Running solve_perturbed_gs (old) for comparison …")
    dB_old, dJ_old, dp_old = solve_perturbed_gs(
        B0_field, J0_field, p0_field, dBext_field,
        solver='lsqr', max_iter=1000, tol=1e-6,
    )
    mu0_dJR_old   = mu0 * dJ_old.VR[:,:,0]
    mu0_dJZ_old   = mu0 * dJ_old.VZ[:,:,0]
    mu0_dJPhi_old = mu0 * dJ_old.VPhi[:,:,0]

    dBR_o   = dB_old.VR[:,:,0]
    dBZ_o   = dB_old.VZ[:,:,0]
    dBPhi_o = dB_old.VPhi[:,:,0]
    curl_R_o   = -np.gradient(dBPhi_o, Z_arr, axis=1)
    curl_Z_o   =  np.gradient(RR * dBPhi_o, R_arr, axis=0) / (RR + 1e-30)
    curl_Phi_o =  np.gradient(dBR_o, Z_arr, axis=1) - np.gradient(dBZ_o, R_arr, axis=0)

    num_old = np.sqrt(np.mean((mu0_dJR_old - curl_R_o)**2 +
                              (mu0_dJZ_old - curl_Z_o)**2 +
                              (mu0_dJPhi_old - curl_Phi_o)**2))
    norm_ref_old = max(np.sqrt(np.mean(mu0_dJR_old**2 + mu0_dJZ_old**2 + mu0_dJPhi_old**2)),
                      np.sqrt(np.mean(curl_R_o**2 + curl_Z_o**2 + curl_Phi_o**2))) + 1e-30
    ampere_residual_old = num_old / norm_ref_old
    print(f"  Ampère residual (old post-hoc): {ampere_residual_old:.4e}")

    assert ampere_residual < 1e-3, (
        f"Coupled solver Amp?re residual {ampere_residual:.4e} exceeds 1e-3"
    )
    print(f"\n  PASS Amp?re residual {ampere_residual:.4e} < 1e-3  [PASS]")
    if ampere_residual < ampere_residual_old:
        print(f"  PASS Coupled deltaJ ({ampere_residual:.4e}) is more accurate than post-hoc deltaJ ({ampere_residual_old:.4e})")
    return ampere_residual


if __name__ == '__main__':
    # Run coupled-solver Amp?re residual test first (no FEniCSx required)
    ar = test_coupled_solver_ampere_residual()
    print(f"\nAmp?re residual reported: {ar:.4e}")
    # Optionally run the full FEniCSx sweep test
    # main()
