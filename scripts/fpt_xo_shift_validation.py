"""FPT X/O-point shift validation: Solov'ev + synthetic PF coils.

Validates the Functional Perturbation Theory (FPT) prediction of X/O-point
shifts against numerical finite-difference for a Solov'ev equilibrium with
a synthetic 8-coil PF system.

Usage:
    py -3.13 scripts/fpt_xo_shift_validation.py
"""
from __future__ import annotations

import os
import sys
import time
import functools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize
from scipy.special import ellipk, ellipe
import joblib

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyna.mag.Solovev import SolovevEquilibrium, _eval_psi_cf, _eval_grad_cf
from pyna.control.fpt import A_matrix, cycle_shift, delta_g_from_delta_B

# ---------------------------------------------------------------------------
# Joblib cache
# ---------------------------------------------------------------------------
_CACHE_DIR = os.path.join(os.path.dirname(__file__), '.cache', 'fpt_validation')
memory = joblib.Memory(location=_CACHE_DIR, verbose=0)

# ---------------------------------------------------------------------------
# Equilibrium setup
# ---------------------------------------------------------------------------
EQ_PARAMS = dict(R0=1.86, a=0.6, B0=5.3, kappa=1.7, delta=0.33, q0=3.0)

def make_equilibrium():
    return SolovevEquilibrium(**EQ_PARAMS)

# ---------------------------------------------------------------------------
# PF coil definitions: 8 coils, 4 upper + 4 lower, well outside plasma
# Plasma boundary approx R∈[1.26,2.46], Z∈[-1.02,1.02]
# ---------------------------------------------------------------------------
COIL_R = np.array([0.7, 1.2, 2.3, 2.8,   # upper PF1-PF4
                   0.7, 1.2, 2.3, 2.8])   # lower PF5-PF8

COIL_Z = np.array([1.2, 1.6, 1.4, 0.8,   # upper
                   -1.2, -1.6, -1.4, -0.8])  # lower

COIL_NAMES = [f'PF{i+1}' for i in range(8)]
DELTA_I = 100.0  # A  — current perturbation

# ---------------------------------------------------------------------------
# Coil field: elliptic-integral formula
# ---------------------------------------------------------------------------
def circular_coil_field(R_coil: float, Z_coil: float, I_coil: float,
                        R: float, Z: float):
    """Vacuum field (BR, BZ) of a circular current coil at (R, Z)."""
    dZ = Z - Z_coil
    denom_sq = (R_coil + R)**2 + dZ**2
    k2 = 4.0 * R_coil * R / denom_sq
    k2 = np.clip(k2, 0.0, 0.9999)
    K = ellipk(k2)
    E = ellipe(k2)
    mu0 = 4e-7 * np.pi
    alpha2 = R_coil**2 + R**2 + dZ**2 - 2.0 * R_coil * R
    beta2 = R_coil**2 + R**2 + dZ**2 + 2.0 * R_coil * R
    factor = mu0 * I_coil / (2.0 * np.pi)
    sqrt_beta2 = np.sqrt(beta2)
    denom = alpha2 * sqrt_beta2
    if abs(R) < 1e-10:
        BR = 0.0
    else:
        BR = factor * dZ / (R * denom) * (
            -K + (R_coil**2 + R**2 + dZ**2) / alpha2 * E
        )
    BZ = factor / denom * (
        K + (R_coil**2 - R**2 - dZ**2) / alpha2 * E
    )
    return float(BR), float(BZ), 0.0  # Bphi = 0 for coil

# ---------------------------------------------------------------------------
# Field functions for FPT
# ---------------------------------------------------------------------------
def make_field_func(eq: SolovevEquilibrium):
    """Return field_func(rzphi) -> [BR/|B|, BZ/|B|, Bphi/(R|B|)]."""
    def field_func(rzphi):
        R, Z, phi = rzphi
        BR, BZ = eq.BR_BZ(R, Z)
        Bphi = eq.Bphi(R)
        Bmag = np.sqrt(BR**2 + BZ**2 + Bphi**2)
        return [BR / Bmag, BZ / Bmag, Bphi / (R * Bmag)]
    return field_func


def make_field_func_perturbed(eq: SolovevEquilibrium,
                               coil_R: float, coil_Z: float, delta_I: float):
    """Return field_func with PF coil perturbation added."""
    def field_func_pert(rzphi):
        R, Z, phi = rzphi
        BR, BZ = eq.BR_BZ(R, Z)
        Bphi = eq.Bphi(R)
        dBR, dBZ, _ = circular_coil_field(coil_R, coil_Z, delta_I, R, Z)
        BR += dBR
        BZ += dBZ
        Bmag = np.sqrt(BR**2 + BZ**2 + Bphi**2)
        return [BR / Bmag, BZ / Bmag, Bphi / (R * Bmag)]
    return field_func_pert

# ---------------------------------------------------------------------------
# Critical point finder: minimize |B_pol|²
# ---------------------------------------------------------------------------
def find_critical_point(eq: SolovevEquilibrium, R0: float, Z0: float):
    """Find critical point (B_pol = 0) starting from (R0, Z0)."""
    def obj(rz):
        R, Z = rz
        if R <= 0.0:
            return 1e10
        BR, BZ = eq.BR_BZ(R, Z)
        return BR**2 + BZ**2
    result = minimize(obj, [R0, Z0], method='Nelder-Mead',
                      options={'xatol': 1e-8, 'fatol': 1e-18, 'maxiter': 50000})
    return result.x


def find_critical_point_perturbed(eq: SolovevEquilibrium,
                                   coil_R: float, coil_Z: float, delta_I: float,
                                   R0: float, Z0: float):
    """Find critical point of Bpol in perturbed field using fsolve for accuracy."""
    from scipy.optimize import fsolve
    def eqs(rz):
        R, Z = rz
        BR, BZ = eq.BR_BZ(R, Z)
        dBR, dBZ, _ = circular_coil_field(coil_R, coil_Z, delta_I, R, Z)
        return [float(BR) + dBR, float(BZ) + dBZ]
    try:
        sol = fsolve(eqs, [R0, Z0], full_output=True)
        return sol[0]
    except Exception:
        pass
    # Fallback: Nelder-Mead with small simplex
    def obj(rz):
        R, Z = rz
        if R <= 0.0:
            return 1e10
        BR, BZ = eq.BR_BZ(R, Z)
        dBR, dBZ, _ = circular_coil_field(coil_R, coil_Z, delta_I, R, Z)
        return (float(BR) + dBR)**2 + (float(BZ) + dBZ)**2
    simplex = np.array([[R0, Z0], [R0 + 1e-4, Z0], [R0, Z0 + 1e-4]])
    result = minimize(obj, [R0, Z0], method='Nelder-Mead',
                      options={'xatol': 1e-10, 'fatol': 1e-22, 'maxiter': 100000,
                               'initial_simplex': simplex})
    return result.x


# ---------------------------------------------------------------------------
# Cached expensive computations
# ---------------------------------------------------------------------------
@memory.cache
def _cached_find_critical_points(eq_params_tuple):
    """Find X-point and O-point of the Solov'ev equilibrium (cached)."""
    eq = SolovevEquilibrium(**dict(zip(
        ['R0', 'a', 'B0', 'kappa', 'delta', 'q0'], eq_params_tuple
    )))
    # O-point: magnetic axis ≈ (R0, 0)
    R0 = eq.R0
    opt = find_critical_point(eq, R0, 0.0)
    # X-point: saddle of psi_CF — found by scanning for |B_pol|=0 outside axis
    # Scan for the saddle-point hyperbolic fixed point
    # For this Solov'ev with these parameters, X-pt is near (0.87, -1.97)
    xpt = find_critical_point(eq, 0.87, -1.97)
    return opt.tolist(), xpt.tolist()


@memory.cache
def _cached_coil_field_at_point(R_coil, Z_coil, I_coil, R_pt, Z_pt):
    """Cached coil field at a specific point."""
    return circular_coil_field(R_coil, Z_coil, I_coil, R_pt, Z_pt)


@memory.cache
def _cached_A_matrix(eq_params_tuple, R_cyc, Z_cyc):
    """Cached A-matrix at a cycle position."""
    eq = SolovevEquilibrium(**dict(zip(
        ['R0', 'a', 'B0', 'kappa', 'delta', 'q0'], eq_params_tuple
    )))
    field_func = make_field_func(eq)
    return A_matrix(field_func, R_cyc, Z_cyc).tolist()


@memory.cache
def _cached_perturbed_critical_point(eq_params_tuple, coil_R, coil_Z, delta_I,
                                      R0, Z0):
    """Cached numerical finite-difference X/O-point position under perturbation."""
    eq = SolovevEquilibrium(**dict(zip(
        ['R0', 'a', 'B0', 'kappa', 'delta', 'q0'], eq_params_tuple
    )))
    result = find_critical_point_perturbed(eq, coil_R, coil_Z, delta_I, R0, Z0)
    return result.tolist()


# ---------------------------------------------------------------------------
# FPT prediction
# ---------------------------------------------------------------------------
def fpt_shift(eq: SolovevEquilibrium, R_cyc: float, Z_cyc: float,
              coil_R: float, coil_Z: float, delta_I: float,
              A_mat: np.ndarray) -> np.ndarray:
    """Compute FPT-predicted cycle shift for a single coil perturbation."""
    BR, BZ = eq.BR_BZ(R_cyc, Z_cyc)
    Bphi = float(eq.Bphi(R_cyc))
    dBR, dBZ, dBphi = circular_coil_field(coil_R, coil_Z, delta_I, R_cyc, Z_cyc)
    dg = delta_g_from_delta_B(R_cyc, Z_cyc, 0.0,
                               float(BR), float(BZ), Bphi,
                               dBR, dBZ, dBphi)
    return cycle_shift(A_mat, dg)


# ---------------------------------------------------------------------------
# LCFS contour points
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=4)
def _get_lcfs_contour(eq_params_tuple):
    """Return (R, Z) arrays for the LCFS contour (cached via lru_cache)."""
    eq = SolovevEquilibrium(**dict(zip(
        ['R0', 'a', 'B0', 'kappa', 'delta', 'q0'], eq_params_tuple
    )))
    R_arr = np.linspace(0.8, 3.0, 800)
    Z_arr = np.linspace(-1.5, 1.5, 800)
    RR, ZZ = np.meshgrid(R_arr, Z_arr)
    psi_cf = _eval_psi_cf(RR / eq.R0, ZZ / eq.R0, eq._c, eq.A)
    # Extract contour at psi_CF = 0 (LCFS)
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(RR, ZZ, psi_cf, levels=[0.0])
    lcfs_paths = []
    for path in cs.get_paths():
        v = path.vertices
        if len(v) > 10:
            lcfs_paths.append(v)
    plt.close(fig_tmp)
    return lcfs_paths


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------
def run_validation():
    print("Setting up Solov'ev equilibrium...")
    eq = make_equilibrium()
    eq_key = tuple(EQ_PARAMS[k] for k in ['R0', 'a', 'B0', 'kappa', 'delta', 'q0'])

    # Find X/O points
    print("Finding X-point and O-point...")
    opt_list, xpt_list = _cached_find_critical_points(eq_key)
    R_opt, Z_opt = opt_list
    R_xpt, Z_xpt = xpt_list
    print(f"  O-point: R={R_opt:.4f}, Z={Z_opt:.4f}")
    print(f"  X-point: R={R_xpt:.4f}, Z={Z_xpt:.4f}")

    # A-matrices at X and O points (cached)
    print("Computing A-matrices...")
    A_opt = np.array(_cached_A_matrix(eq_key, R_opt, Z_opt))
    A_xpt = np.array(_cached_A_matrix(eq_key, R_xpt, Z_xpt))
    print(f"  A at O-point:\n{A_opt}")
    print(f"  A at X-point:\n{A_xpt}")

    # LCFS contour
    lcfs_paths = _get_lcfs_contour(eq_key)

    # Collect results
    results = []
    n_coils = len(COIL_R)
    for i in range(n_coils):
        cR, cZ, cname = COIL_R[i], COIL_Z[i], COIL_NAMES[i]
        print(f"\n{cname} (R={cR}, Z={cZ:.1f}):")

        # --- FPT prediction ---
        t0 = time.perf_counter()
        fpt_shift_opt = fpt_shift(eq, R_opt, Z_opt, cR, cZ, DELTA_I, A_opt)
        fpt_shift_xpt = fpt_shift(eq, R_xpt, Z_xpt, cR, cZ, DELTA_I, A_xpt)
        t_fpt = time.perf_counter() - t0

        # --- Numerical FD ---
        t1 = time.perf_counter()
        opt_pert = np.array(_cached_perturbed_critical_point(
            eq_key, float(cR), float(cZ), DELTA_I, R_opt, Z_opt))
        xpt_pert = np.array(_cached_perturbed_critical_point(
            eq_key, float(cR), float(cZ), DELTA_I, R_xpt, Z_xpt))
        t_num = time.perf_counter() - t1

        num_shift_opt = opt_pert - np.array([R_opt, Z_opt])
        num_shift_xpt = xpt_pert - np.array([R_xpt, Z_xpt])

        # Agreement ratios (avoid div by zero)
        def agreement(fpt_vec, num_vec):
            n_fpt = np.linalg.norm(fpt_vec)
            n_num = np.linalg.norm(num_vec)
            if n_num < 1e-12 and n_fpt < 1e-12:
                return 1.0
            if n_num < 1e-12:
                return 0.0
            return float(np.dot(fpt_vec, num_vec) / (n_fpt * n_num + 1e-30)
                         * n_fpt / n_num)

        ratio_opt = np.linalg.norm(fpt_shift_opt) / (np.linalg.norm(num_shift_opt) + 1e-15)
        ratio_xpt = np.linalg.norm(fpt_shift_xpt) / (np.linalg.norm(num_shift_xpt) + 1e-15)

        print(f"  FPT O-shift: dR={fpt_shift_opt[0]:.3e}, dZ={fpt_shift_opt[1]:.3e}")
        print(f"  Num O-shift: dR={num_shift_opt[0]:.3e}, dZ={num_shift_opt[1]:.3e}")
        print(f"  FPT X-shift: dR={fpt_shift_xpt[0]:.3e}, dZ={fpt_shift_xpt[1]:.3e}")
        print(f"  Num X-shift: dR={num_shift_xpt[0]:.3e}, dZ={num_shift_xpt[1]:.3e}")
        print(f"  |FPT/Num| O={ratio_opt:.3f}, X={ratio_xpt:.3f}")
        print(f"  Time: FPT={t_fpt*1e3:.2f}ms, Num≈{t_num*1e3:.2f}ms (cache)")

        results.append(dict(
            name=cname, R=cR, Z=cZ,
            fpt_opt=fpt_shift_opt, num_opt=num_shift_opt,
            fpt_xpt=fpt_shift_xpt, num_xpt=num_shift_xpt,
            ratio_opt=ratio_opt, ratio_xpt=ratio_xpt,
            t_fpt=t_fpt,
        ))

    # ---------------------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------------------
    print("\nGenerating figure...")
    arrow_scale = 5e2  # scale arrows for visibility

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        "FPT vs Numerical X/O-point shift — Solov'ev + PF coils\n"
        r"$\delta I=100\,\mathrm{A}$, arrows scaled ×500",
        fontsize=13)

    R_plot = np.linspace(0.8, 3.0, 300)
    Z_plot = np.linspace(-1.5, 1.5, 300)
    RR, ZZ = np.meshgrid(R_plot, Z_plot)
    eq2 = make_equilibrium()
    psi_cf_grid = _eval_psi_cf(RR / eq2.R0, ZZ / eq2.R0, eq2._c, eq2.A)

    for idx, res in enumerate(results):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]

        # LCFS contour
        ax.contour(RR, ZZ, psi_cf_grid, levels=[0.0], colors='k', linewidths=1.2)

        # O-point and X-point markers
        ax.plot(R_opt, Z_opt, 'ko', ms=5, zorder=5)
        ax.plot(R_xpt, Z_xpt, 'kx', ms=8, mew=2, zorder=5)

        # Coil position
        ax.plot(res['R'], res['Z'], '*', color='orange', ms=12,
                markeredgecolor='k', markeredgewidth=0.5, zorder=6,
                label=f"{res['name']} ★")

        # --- O-point arrows ---
        dR_fpt_o, dZ_fpt_o = res['fpt_opt'] * arrow_scale
        dR_num_o, dZ_num_o = res['num_opt'] * arrow_scale
        ax.annotate('', xy=(R_opt + dR_fpt_o, Z_opt + dZ_fpt_o),
                    xytext=(R_opt, Z_opt),
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.8))
        ax.annotate('', xy=(R_opt + dR_num_o, Z_opt + dZ_num_o),
                    xytext=(R_opt, Z_opt),
                    arrowprops=dict(arrowstyle='->', color='cyan',
                                   lw=1.5, linestyle='dashed'))

        # --- X-point arrows ---
        dR_fpt_x, dZ_fpt_x = res['fpt_xpt'] * arrow_scale
        dR_num_x, dZ_num_x = res['num_xpt'] * arrow_scale
        ax.annotate('', xy=(R_xpt + dR_fpt_x, Z_xpt + dZ_fpt_x),
                    xytext=(R_xpt, Z_xpt),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.8))
        ax.annotate('', xy=(R_xpt + dR_num_x, Z_xpt + dZ_num_x),
                    xytext=(R_xpt, Z_xpt),
                    arrowprops=dict(arrowstyle='->', color='blue',
                                   lw=1.5, linestyle='dashed'))

        ax.set_xlim(0.5, 3.2)
        ax.set_ylim(-2.5, 2.0)
        ax.set_xlabel('R (m)', fontsize=8)
        ax.set_ylabel('Z (m)', fontsize=8)
        ax.set_title(
            f"{res['name']} @ R={res['R']},Z={res['Z']:.1f}\n"
            f"O: |FPT/Num|={res['ratio_opt']:.2f}  X: {res['ratio_xpt']:.2f}",
            fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect('equal', adjustable='box')

    # Legend
    legend_elements = [
        mpatches.Patch(color='red', label='FPT X-point'),
        mpatches.Patch(color='blue', label='Num X-point'),
        mpatches.Patch(color='green', label='FPT O-point'),
        mpatches.Patch(color='cyan', label='Num O-point'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out_path = os.path.join(os.path.dirname(__file__), 'fpt_xo_shift_validation.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)

    return results


if __name__ == '__main__':
    results = run_validation()
    print("\nDone.")
