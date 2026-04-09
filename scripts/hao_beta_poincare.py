"""HAO stellarator beta-climb Poincare visualization.

Generates an n-rows x 4-columns figure of Poincare sections showing
how the magnetic topology changes as <beta> (volume-averaged inside LCFS)
climbs.

Usage
-----
::

    # Quick test (30 coils, coarse tracing)
    python scripts/hao_beta_poincare.py --test

    # Full production run (all 332 coils)
    python scripts/hao_beta_poincare.py --full
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PYNA_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PYNA_ROOT))

from pyna.MCF.equilibrium.finite_beta_perturbation import load_hao_coils

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VACUUM_FIELD_DIR = "/mnt/c/Users/28105/Nutstore/1/haodata/coilsys/vacuum_fields"
EXCLUDED_COILS = {38, 122, 206, 290}
MU0 = 4.0e-7 * np.pi


# ---------------------------------------------------------------------------
# Coil field loading and assembly
# ---------------------------------------------------------------------------

def load_all_coil_fields(coil_files: list[str],
                          R_grid: np.ndarray,
                          Z_grid: np.ndarray,
                          Phi_grid: np.ndarray,
                          max_coils: int | None = None,
                          ) -> dict:
    """Load and sum all coil vacuum fields onto a common grid.

    The *_resp arrays in the npz files are already scaled by the actual
    coil current (extracted from the filename).  We simply sum them.

    Returns field_cache dict: BR, BPhi, BZ, R_grid, Z_grid, Phi_grid.
    """
    if max_coils:
        coil_files = coil_files[:max_coils]

    BR_total = None
    BPhi_total = None
    BZ_total = None

    import re

    for idx, fp in enumerate(coil_files):
        d = np.load(fp)
        BR_c = d["BR_resp"].astype(np.float64)
        BPhi_c = d["BPhi_resp"].astype(np.float64)
        BZ_c = d["BZ_resp"].astype(np.float64)

        # These are already current-scaled — direct sum
        if BR_total is None:
            BR_total = BR_c.copy()
            BPhi_total = BPhi_c.copy()
            BZ_total = BZ_c.copy()
        else:
            BR_total += BR_c
            BPhi_total += BPhi_c
            BZ_total += BZ_c

        if (idx + 1) % 50 == 0:
            print(f"  Loaded {idx+1}/{len(coil_files)} coils...")

    # Check field strength
    B_mag = np.sqrt(BR_total**2 + BPhi_total**2 + BZ_total**2)
    print(f"  Total field: <B> = {np.mean(B_mag):.6e} T, max|B| = {np.max(B_mag):.6e} T")

    return {
        "BR": np.ascontiguousarray(BR_total, dtype=np.float64),
        "BPhi": np.ascontiguousarray(BPhi_total, dtype=np.float64),
        "BZ": np.ascontiguousarray(BZ_total, dtype=np.float64),
        "R_grid": np.ascontiguousarray(R_grid, dtype=np.float64),
        "Z_grid": np.ascontiguousarray(Z_grid, dtype=np.float64),
        "Phi_grid": np.ascontiguousarray(Phi_grid, dtype=np.float64),
    }


def add_plasma_response(field_cache: dict, beta_target: float,
                         R_axis: float, Z_axis: float,
                         alpha_pressure: float = 2.0) -> dict:
    """Add a simplified plasma response field to the vacuum field.

    The plasma response is modeled as a diamagnetic field perturbation
    that grows with β.  Pressure is peaked at the magnetic axis and
    falls to zero at the plasma edge.

    Parameters
    ----------
    field_cache : vacuum field dict
    beta_target : target volume-averaged β
    R_axis, Z_axis : magnetic axis position
    alpha_pressure : pressure profile exponent

    Returns
    -------
    field_cache_new : modified field dict
    beta_avg : actual volume-averaged β inside the confinement region
    """
    BR = field_cache["BR"]
    BPhi = field_cache["BPhi"]
    BZ = field_cache["BZ"]
    R_grid = field_cache["R_grid"]
    Z_grid = field_cache["Z_grid"]
    Phi_grid = field_cache["Phi_grid"]

    nR, nZ, nPhi = BR.shape
    RR, ZZ, PP = np.meshgrid(R_grid, Z_grid, Phi_grid, indexing="ij")

    # Compute |B|
    B_sq = BR**2 + BPhi**2 + BZ**2
    B_mag = np.sqrt(np.maximum(B_sq, 1e-20))
    B_vol_avg = float(np.mean(B_mag))

    # Normalised flux label: ψ_n = 0 at magnetic axis, ψ_n = 1 at edge
    # Use distance from axis normalised to the half-width of the vessel
    a_half = (R_grid.max() - R_grid.min()) / 2.0
    r_from_axis = np.sqrt((RR - R_axis)**2 + (ZZ - Z_axis)**2)
    psi_n = np.clip(r_from_axis / a_half, 0.0, 1.0)

    # Pressure profile: p = p₀ · (1 - ψ_n)^α
    # p₀ chosen so that volume-averaged β ≈ beta_target
    # β = 2μ₀ <p> / <B²>
    # For a parabolic profile <(1-ψ_n)^α> over the unit disk ≈ 1/(α+1)
    # So p₀ ≈ beta_target · <B²> / (2μ₀) · (α+1)
    p0 = beta_target * B_vol_avg**2 / (2.0 * MU0) * (alpha_pressure + 1.0)
    p = p0 * np.maximum(0, 1.0 - psi_n)**alpha_pressure

    # Pressure gradient
    dR = R_grid[1] - R_grid[0]
    dZ = Z_grid[1] - Z_grid[0]
    dPhi = Phi_grid[1] - Phi_grid[0]
    dp_dR = np.gradient(p, dR, axis=0)
    dp_dZ = np.gradient(p, dZ, axis=1)
    dp_dPhi = np.gradient(p, dPhi, axis=2)

    # Diamagnetic field perturbation: δB ∝ μ₀ · (∇p × B) / B² · a_eff
    a_eff = a_half

    scale = MU0 * a_eff / np.maximum(B_sq, 1e-20)

    delta_BR = scale * (dp_dZ * BPhi - dp_dPhi / np.maximum(RR, 1e-10) * BZ)
    delta_BPhi = scale * (dp_dR * BZ - dp_dZ * BR)
    delta_BZ = scale * (dp_dPhi / np.maximum(RR, 1e-10) * BR - dp_dR * BPhi)

    # Add Pfirsch-Schlüter toroidal modulation
    R_mean = R_grid.mean()
    epsilon = (R_grid.max() - R_grid.min()) / (2 * R_mean)
    f_PS = 1.0 + epsilon * np.cos(PP)
    delta_BR *= f_PS
    delta_BPhi *= f_PS
    delta_BZ *= f_PS

    # Total field
    BR_new = BR + delta_BR
    BPhi_new = BPhi + delta_BPhi
    BZ_new = BZ + delta_BZ

    # Compute actual volume-averaged β inside the plasma region
    # Use ψ_n < 0.85 as the "inside LCFS" proxy
    inside = psi_n < 0.85
    if np.any(inside):
        dV = RR * dR * dZ * dPhi
        p_avg = np.sum(p[inside] * dV[inside]) / np.sum(dV[inside])
        B_sq_new = BR_new**2 + BPhi_new**2 + BZ_new**2
        B_sq_avg = np.sum(B_sq_new[inside] * dV[inside]) / np.sum(dV[inside])
        beta_avg = 2.0 * MU0 * p_avg / B_sq_avg if B_sq_avg > 0 else 0.0
    else:
        beta_avg = beta_target

    return {
        "BR": np.ascontiguousarray(BR_new, dtype=np.float64),
        "BPhi": np.ascontiguousarray(BPhi_new, dtype=np.float64),
        "BZ": np.ascontiguousarray(BZ_new, dtype=np.float64),
        "R_grid": field_cache["R_grid"],
        "Z_grid": field_cache["Z_grid"],
        "Phi_grid": field_cache["Phi_grid"],
    }, float(beta_avg)


# ---------------------------------------------------------------------------
# Field line tracing and Poincare
# ---------------------------------------------------------------------------

def _make_interpolators(field_cache: dict):
    """Create periodic interpolators for B field components."""
    R_grid = field_cache["R_grid"]
    Z_grid = field_cache["Z_grid"]
    Phi_grid = field_cache["Phi_grid"]

    # Extend Phi for periodicity
    dPhi = Phi_grid[1] - Phi_grid[0]
    Phi_ext = np.append(Phi_grid, Phi_grid[-1] + dPhi)

    def extend_field(field_3d):
        return np.concatenate([field_3d, field_3d[:, :, :1]], axis=2)

    BR_ext = extend_field(field_cache["BR"])
    BPhi_ext = extend_field(field_cache["BPhi"])
    BZ_ext = extend_field(field_cache["BZ"])

    itp_BR = RegularGridInterpolator(
        (R_grid, Z_grid, Phi_ext), BR_ext, method="linear",
        bounds_error=False, fill_value=0.0)
    itp_BPhi = RegularGridInterpolator(
        (R_grid, Z_grid, Phi_ext), BPhi_ext, method="linear",
        bounds_error=False, fill_value=0.0)
    itp_BZ = RegularGridInterpolator(
        (R_grid, Z_grid, Phi_ext), BZ_ext, method="linear",
        bounds_error=False, fill_value=0.0)

    return itp_BR, itp_BPhi, itp_BZ


def _rhs(R, Z, Phi, itp_BR, itp_BPhi, itp_BZ):
    """Field line ODE: dR/dφ, dZ/dφ from B/B_φ."""
    R_a = np.atleast_1d(np.asarray(R, dtype=np.float64))
    Z_a = np.atleast_1d(np.asarray(Z, dtype=np.float64))
    Phi_a = np.atleast_1d(np.asarray(Phi, dtype=np.float64))

    pts = np.stack([R_a, Z_a, Phi_a], axis=-1)
    BR = itp_BR(pts)
    BPhi = itp_BPhi(pts)
    BZ = itp_BZ(pts)

    # Avoid division by zero
    BPhi = np.where(np.abs(BPhi) < 1e-20, np.sign(BPhi) * 1e-20, BPhi)

    dR_dphi = BR / BPhi * R_a
    dZ_dphi = BZ / BPhi * R_a

    scalar_input = np.isscalar(R) or (isinstance(R, np.ndarray) and R.shape == ())
    if scalar_input:
        return float(dR_dphi[0]), float(dZ_dphi[0])
    return dR_dphi, dZ_dphi


def _rk4(R, Z, Phi, dphi, itp_BR, itp_BPhi, itp_BZ):
    """RK4 step for field line tracing in φ."""
    k1R, k1Z = _rhs(R, Z, Phi, itp_BR, itp_BPhi, itp_BZ)
    k1R *= dphi; k1Z *= dphi

    R2 = R + 0.5*k1R; Z2 = Z + 0.5*k1Z; Phi2 = Phi + 0.5*dphi
    k2R, k2Z = _rhs(R2, Z2, Phi2, itp_BR, itp_BPhi, itp_BZ)
    k2R *= dphi; k2Z *= dphi

    R3 = R + 0.5*k2R; Z3 = Z + 0.5*k2Z; Phi3 = Phi + 0.5*dphi
    k3R, k3Z = _rhs(R3, Z3, Phi3, itp_BR, itp_BPhi, itp_BZ)
    k3R *= dphi; k3Z *= dphi

    R4 = R + k3R; Z4 = Z + k3Z; Phi4 = Phi + dphi
    k4R, k4Z = _rhs(R4, Z4, Phi4, itp_BR, itp_BPhi, itp_BZ)
    k4R *= dphi; k4Z *= dphi

    return (R + (k1R + 2*k2R + 2*k3R + k4R)/6.0,
            Z + (k1Z + 2*k2Z + 2*k3Z + k4Z)/6.0,
            Phi + dphi)


def trace_poincare(field_cache: dict,
                    R_seeds: np.ndarray,
                    Z_seeds: np.ndarray,
                    phi_section: float = 0.0,
                    n_turns: int = 200,
                    dphi: float = 0.05,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Trace fieldlines and collect Poincare crossings at phi = phi_section."""
    itp_BR, itp_BPhi, itp_BZ = _make_interpolators(field_cache)
    Phi_grid = field_cache["Phi_grid"]
    phi_period = 2.0 * np.pi

    R_cross = []
    Z_cross = []
    R_min, R_max = field_cache["R_grid"].min(), field_cache["R_grid"].max()
    Z_min, Z_max = field_cache["Z_grid"].min(), field_cache["Z_grid"].max()

    for R0, Z0 in zip(R_seeds, Z_seeds):
        R, Z, Phi = float(R0), float(Z0), phi_section
        n_steps = int(n_turns * phi_period / dphi)

        for _ in range(n_steps):
            R_old, Z_old, Phi_old = R, Z, Phi
            R, Z, Phi = _rk4(R, Z, Phi, dphi, itp_BR, itp_BPhi, itp_BZ)

            # Check crossing of phi_section
            # Normalize both to [0, 2π)
            phi_old_norm = Phi_old % phi_period
            phi_new_norm = Phi % phi_period

            # Detect crossing: phi_section is between old and new
            crossed = False
            if phi_old_norm <= phi_section < phi_new_norm or phi_new_norm <= phi_section < phi_old_norm:
                # Interpolate
                dphi_step = phi_new_norm - phi_old_norm
                if abs(dphi_step) < 1e-12:
                    continue
                frac = (phi_section - phi_old_norm) / dphi_step
                frac = frac % 1.0
                Rc = R_old + frac * (R - R_old)
                Zc = Z_old + frac * (Z - Z_old)

                if R_min <= Rc <= R_max and Z_min <= Zc <= Z_max:
                    R_cross.append(Rc)
                    Z_cross.append(Zc)
                    crossed = True

    return np.array(R_cross), np.array(Z_cross)


def find_magnetic_axis(field_cache: dict) -> tuple[float, float]:
    """Find magnetic axis as the minimum-|B| point on Z≈0 midplane."""
    BR = field_cache["BR"]
    BPhi = field_cache["BPhi"]
    BZ = field_cache["BZ"]
    R_grid = field_cache["R_grid"]
    Z_grid = field_cache["Z_grid"]

    z_idx = np.argmin(np.abs(Z_grid))
    B_sq_mid = BR[:, z_idx, :]**2 + BPhi[:, z_idx, :]**2 + BZ[:, z_idx, :]**2
    B_avg = np.mean(B_sq_mid, axis=1)

    # Also check neighboring Z planes for robustness
    z_lo = max(0, z_idx - 1)
    z_hi = min(len(Z_grid) - 1, z_idx + 1)
    for zi in [z_lo, z_hi]:
        B_sq_z = BR[:, zi, :]**2 + BPhi[:, zi, :]**2 + BZ[:, zi, :]**2
        B_avg_z = np.mean(B_sq_z, axis=1)
        min_idx_z = np.argmin(B_avg_z)
        if B_avg_z[min_idx_z] < B_avg.min():
            B_avg = B_avg_z
            z_idx = zi

    R_axis_idx = np.argmin(B_avg)
    return float(R_grid[R_axis_idx]), float(Z_grid[z_idx])


def estimate_lcfs_by_survival(field_cache: dict, R_axis: float, Z_axis: float,
                                n_seeds: int = 30, n_turns: int = 80,
                                dphi: float = 0.05) -> float:
    """Estimate LCFS radius by fieldline survival on midplane."""
    itp_BR, itp_BPhi, itp_BZ = _make_interpolators(field_cache)
    R_grid = field_cache["R_grid"]
    Z_grid = field_cache["Z_grid"]
    R_min, R_max = R_grid.min(), R_grid.max()
    Z_min, Z_max = Z_grid.min(), Z_grid.max()

    R_test = np.linspace(R_axis + 0.01, R_max - 0.02, n_seeds)
    surviving = []

    for R0 in R_test:
        R, Z, Phi = R0, Z_axis, 0.0
        n_steps = int(n_turns * 2 * np.pi / dphi)
        hit = False
        for _ in range(min(n_steps, 15000)):
            R, Z, Phi = _rk4(R, Z, Phi, dphi, itp_BR, itp_BPhi, itp_BZ)
            if R < R_min or R > R_max or abs(Z) > Z_max:
                hit = True
                break
        if not hit:
            surviving.append(R0)

    if len(surviving) < 3:
        return max(0.05, (R_max - R_axis) * 0.3)

    return float(max(surviving) - R_axis)


def generate_seeds(R_axis, Z_axis, a_lcfs, n_radial=40, n_per_r=12):
    """Generate seed points on concentric circles."""
    R_s, Z_s = [], []
    for r_frac in np.linspace(0.02, 0.92, n_radial):
        r = r_frac * a_lcfs
        if r < 1e-10:
            R_s.append(R_axis)
            Z_s.append(Z_axis)
        else:
            for i in range(n_per_r):
                theta = 2 * np.pi * i / n_per_r + r_frac * 3.7  # offset for variety
                R_s.append(R_axis + r * np.cos(theta))
                Z_s.append(Z_axis + r * np.sin(theta))
    return np.array(R_s), np.array(Z_s)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_poincare_grid(results: list[dict], output_dir: Path, n_cols: int = 4):
    """Plot Poincare sections: n rows × 4 columns."""
    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 5.5, n_rows * 5.0),
                              squeeze=False)

    for row_idx, res in enumerate(results):
        Rc = res["R_cross"]
        Zc = res["Z_cross"]
        beta_avg = res["beta_avg"]
        beta_in = res["beta_input"]
        R_ax = res["R_axis"]
        a_lcfs = res["a_lcfs"]
        fc = res["field_cache"]

        # Col 0: Full Poincare section
        ax = axes[row_idx, 0]
        if len(Rc) > 0:
            ax.scatter(Rc, Zc, s=0.2, c="steelblue", alpha=0.4, rasterized=True)
        ax.set_title(f"<β> = {beta_avg:.5f}  (target β = {beta_in:.3f})", fontsize=11)
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        if a_lcfs > 0:
            th = np.linspace(0, 2*np.pi, 100)
            ax.plot(R_ax + a_lcfs*np.cos(th), a_lcfs*np.sin(th),
                    "r--", lw=1, alpha=0.5, label="LCFS")
            ax.legend(loc="upper right", fontsize=8)

        # Col 1: Edge zoom
        ax = axes[row_idx, 1]
        if len(Rc) > 0:
            r_from_ax = np.sqrt((Rc - R_ax)**2 + Zc**2)
            edge = r_from_ax > a_lcfs * 0.65 if a_lcfs > 0 else np.ones(len(Rc), bool)
            ax.scatter(Rc[edge], Zc[edge], s=0.2, c="coral", alpha=0.4, rasterized=True)
        ax.set_title("Edge Region", fontsize=11)
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        if a_lcfs > 0:
            margin = a_lcfs * 0.4
            ax.set_xlim(R_ax + a_lcfs - margin, R_ax + a_lcfs + margin)
            ax.set_ylim(-margin, margin)

        # Col 2: |B| colormap + Poincare
        ax = axes[row_idx, 2]
        BR = fc["BR"]; BPhi = fc["BPhi"]; BZ = fc["BZ"]
        B_mag = np.sqrt(BR**2 + BPhi**2 + BZ**2)
        B_2d = np.mean(B_mag, axis=2)  # average over phi
        Rg = fc["R_grid"]; Zg = fc["Z_grid"]
        RR, ZZ = np.meshgrid(Rg, Zg, indexing="ij")
        im = ax.pcolormesh(RR, ZZ, B_2d.T, cmap="magma", shading="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if len(Rc) > 0:
            ax.scatter(Rc, Zc, s=0.1, c="cyan", alpha=0.25, rasterized=True)
        ax.set_title("|B| [T]", fontsize=11)
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_aspect("equal")

        # Col 3: B_Phi component + Poincare
        ax = axes[row_idx, 3]
        BPhi_2d = np.mean(BPhi, axis=2)
        im = ax.pcolormesh(RR, ZZ, BPhi_2d.T, cmap="RdBu_r", shading="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if len(Rc) > 0:
            ax.scatter(Rc, Zc, s=0.1, c="yellow", alpha=0.25, rasterized=True)
        ax.set_title("B_φ [T]", fontsize=11)
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_aspect("equal")

    plt.suptitle("HAO Stellarator: Magnetic Topology vs Volume-Averaged β (inside LCFS)",
                 fontsize=14, y=0.995)
    plt.tight_layout()
    outpath = output_dir / "poincare_beta_climb.png"
    plt.savefig(outpath, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nPoincare plot saved to: {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HAO beta-climb Poincare")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--beta", nargs="+", type=float, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--n-turns", type=int, default=200)
    parser.add_argument("--n-seeds-radial", type=int, default=40)
    parser.add_argument("--n-seeds-per-r", type=int, default=12)
    parser.add_argument("--dphi", type=float, default=0.05)
    args = parser.parse_args()

    # Beta values
    if args.beta is not None:
        beta_values = sorted(args.beta)
    elif args.test:
        beta_values = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    else:
        beta_values = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]

    # Output dir
    if args.output is None:
        out_base = SCRIPT_DIR.parent / "results" / "hao_beta_poincare"
    else:
        out_base = Path(args.output)
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = out_base / ts
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {output_dir}")

    # Coils
    print(f"\nLoading coils from {VACUUM_FIELD_DIR}")
    coil_files = load_hao_coils(VACUUM_FIELD_DIR, exclude_indices=EXCLUDED_COILS)
    print(f"  {len(coil_files)} coils available")

    if args.test:
        coil_files = coil_files[:30]
        print(f"  Test mode: {len(coil_files)} coils")

    # Grid (from first coil file)
    d0 = np.load(coil_files[0])
    R_grid = d0["R_grid"].astype(np.float64)
    Z_grid = d0["Z_grid"].astype(np.float64)
    Phi_grid = d0["Phi_grid"].astype(np.float64)
    print(f"  Grid: {len(R_grid)}R × {len(Z_grid)}Z × {len(Phi_grid)}Phi")

    # Load vacuum field
    print("\nAssembling vacuum field...")
    t0 = time.time()
    max_coils = None if args.full and not args.test else (30 if args.test else None)
    fc_vacuum = load_all_coil_fields(coil_files, R_grid, Z_grid, Phi_grid,
                                       max_coils=max_coils)
    print(f"  Vacuum field assembled ({time.time()-t0:.1f}s)")

    # Find magnetic axis
    R_axis, Z_axis = find_magnetic_axis(fc_vacuum)
    print(f"  Magnetic axis: R={R_axis:.4f} m, Z={Z_axis:.4f} m")

    # LCFS estimate (vacuum)
    a_lcfs_vac = estimate_lcfs_by_survival(
        fc_vacuum, R_axis, Z_axis,
        n_seeds=20, n_turns=40, dphi=0.1 if args.test else 0.05)
    print(f"  LCFS minor radius (vacuum): a={a_lcfs_vac:.4f} m")

    # Parameters
    n_turns = 50 if args.test else args.n_turns
    n_seeds_radial = 15 if args.test else args.n_seeds_radial
    n_seeds_per_r = 6 if args.test else args.n_seeds_per_r
    dphi = 0.1 if args.test else args.dphi

    # Seeds
    R_seeds, Z_seeds = generate_seeds(R_axis, Z_axis, a_lcfs_vac,
                                        n_radial=n_seeds_radial,
                                        n_per_r=n_seeds_per_r)
    print(f"  Seeds: {len(R_seeds)} points")

    # Run beta climb
    results = []
    phi_section = 0.0

    for beta_in in beta_values:
        print(f"\n{'='*60}")
        print(f"β = {beta_in:.4f}")
        print(f"{'='*60}")

        if beta_in == 0.0:
            fc = fc_vacuum
            beta_avg = 0.0
        else:
            t_step = time.time()
            fc, beta_avg = add_plasma_response(
                fc_vacuum, beta_in, R_axis, Z_axis, alpha_pressure=2.0)
            print(f"  Plasma response added ({time.time()-t_step:.1f}s), <β> = {beta_avg:.6f}")

        # Poincare
        t_trace = time.time()
        Rc, Zc = trace_poincare(fc, R_seeds, Z_seeds,
                                 phi_section=phi_section,
                                 n_turns=n_turns, dphi=dphi)
        print(f"  Poincare: {len(Rc)} crossings ({time.time()-t_trace:.1f}s)")

        results.append({
            "beta_input": beta_in,
            "beta_avg": beta_avg,
            "R_axis": R_axis,
            "Z_axis": Z_axis,
            "a_lcfs": a_lcfs_vac,
            "R_cross": Rc,
            "Z_cross": Zc,
            "field_cache": fc,
        })

    # Save
    print(f"\nSaving results...")
    with open(output_dir / "summary.json", "w") as f:
        json.dump([{
            "beta_input": r["beta_input"],
            "beta_avg": r["beta_avg"],
            "n_crossings": len(r["R_cross"]),
        } for r in results], f, indent=2)

    for r in results:
        np.savez_compressed(
            output_dir / f"poincare_beta{r['beta_input']:.4f}_avg{r['beta_avg']:.6f}.npz",
            R_cross=r["R_cross"], Z_cross=r["Z_cross"],
            beta_input=r["beta_input"], beta_avg=r["beta_avg"],
            R_axis=r["R_axis"], Z_axis=r["Z_axis"], a_lcfs=r["a_lcfs"],
        )

    with open(output_dir / "summary.csv", "w") as f:
        f.write("beta_input,beta_avg,n_crossings\n")
        for r in results:
            f.write(f"{r['beta_input']:.6f},{r['beta_avg']:.8f},{len(r['R_cross'])}\n")

    # Plot
    print("\nGenerating plot...")
    plot_poincare_grid(results, output_dir, n_cols=4)

    print(f"\n{'='*60}")
    print(f"Done! Results: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
