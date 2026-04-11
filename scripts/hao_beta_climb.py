"""HAO stellarator finite-beta climb workflow.

This script studies how the magnetic topology of the HAO stellarator
changes as β climbs from 0 (vacuum) to finite values.  At each β step:

1. Pressure profile is scaled:  p(ψ_n) = β · B²_avg/(2μ₀) · (1-ψ_n)^α
2. Current components are computed:
   - Diamagnetic current:  J_dia = (∇p × B) / B²
   - Pfirsch-Schlüter current:  J_PS (neoclassical collisional)
   - Bootstrap current:  J_BS (collisionless trapped particle drifts)
   - Parallel current:  J_∥ (matches q-profile)
3. The perturbation system is solved:
   δJ × B + J × δB = ∇δp
   ∇ · δB = 0
   ∇ × δJ = μ₀ δB
4. Topology analysis is performed:
   - Poincaré sections
   - Island chain detection
   - Rotational transform profile
   - LCFS detection

Usage
-----
::

    # Quick test (small grid, few coils)
    python scripts/hao_beta_climb.py --test

    # Full run (all 332 coils, β = 0 → 0.05)
    python scripts/hao_beta_climb.py --full

    # Custom β range
    python scripts/hao_beta_climb.py --beta 0.0 0.01 0.02 0.03 0.04 0.05

    # Specify output directory
    python scripts/hao_beta_climb.py --full --output ./results/hao_beta_scan
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PYNA_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PYNA_ROOT))

from pyna.toroidal.equilibrium.finite_beta_perturbation import (
    FiniteBetaPerturbation,
    load_hao_coils,
    PerturbationState,
)

# Optional: import topology analysis from topoquest
try:
    sys.path.insert(0, str(PYNA_ROOT.parent / "topoquest"))
    from topoquest.analysis.topology import compute_iota_profile
    from topoquest.analysis.divertor import find_lcfs
    HAS_TOPOQUEST = True
except ImportError:
    HAS_TOPOQUEST = False
    print("Warning: topoquest not available; topology analysis will be limited")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VACUUM_FIELD_DIR = "/mnt/c/Users/28105/Nutstore/1/haodata/coilsys/vacuum_fields"
EXCLUDED_COILS = {38, 122, 206, 290}  # excluded dipole coils

DEFAULT_BETA_SCAN = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]

# Pressure profile parameters
ALPHA_PRESSURE = 2.0  # p ∝ (1 - ψ_n)^α


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HAO stellarator finite-beta climb workflow",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Quick test mode: use only 20 coils and coarse grid",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Full production run: all 332 coils",
    )
    parser.add_argument(
        "--beta", nargs="+", type=float, default=None,
        help="Custom β values to scan (default: 0.0 to 0.05 in steps of 0.005)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: ./results/hao_beta_climb_<timestamp>)",
    )
    parser.add_argument(
        "--vacuum-dir", type=str, default=VACUUM_FIELD_DIR,
        help=f"Path to vacuum field data (default: {VACUUM_FIELD_DIR})",
    )
    parser.add_argument(
        "--alpha", type=float, default=ALPHA_PRESSURE,
        help=f"Pressure profile exponent α (default: {ALPHA_PRESSURE})",
    )
    parser.add_argument(
        "--max-iter", type=int, default=20,
        help="Maximum iterations per β step (default: 20)",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-4,
        help="Convergence tolerance (default: 1e-4)",
    )
    parser.add_argument(
        "--skip-topology", action="store_true",
        help="Skip topology analysis (faster, less output)",
    )
    return parser


def make_output_dir(base_dir: Optional[str] = None) -> Path:
    """Create timestamped output directory."""
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / "results" / "hao_beta_climb"
    else:
        base_dir = Path(base_dir)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out = base_dir / timestamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_beta_convergence(
    history: list[PerturbationState],
    output_dir: Path,
):
    """Plot convergence metrics vs β."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    betas = [s.beta for s in history]
    residuals = [s.residual for s in history]
    n_iters = [s.n_iterations for s in history]
    converged = [s.converged for s in history]

    # Residual vs β
    ax = axes[0, 0]
    ax.semilogy(betas, residuals, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("β")
    ax.set_ylabel("Force-balance residual")
    ax.set_title("Convergence: Residual vs β")
    ax.grid(True, alpha=0.3)

    # Iterations vs β
    ax = axes[0, 1]
    ax.plot(betas, n_iters, "s-", linewidth=2, markersize=8)
    ax.set_xlabel("β")
    ax.set_ylabel("Iterations to converge")
    ax.set_title("Iterations vs β")
    ax.grid(True, alpha=0.3)

    # Convergence status
    ax = axes[1, 0]
    colors = ["green" if c else "red" for c in converged]
    ax.scatter(betas, [1.0] * len(betas), c=colors, s=100, zorder=3)
    ax.set_xlabel("β")
    ax.set_ylabel("Converged")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No", "Yes"])
    ax.set_title("Convergence status")
    ax.grid(True, alpha=0.3)

    # dB/B ratio vs β
    ax = axes[1, 1]
    dB_over_B = []
    for s in history:
        B_mag = np.sqrt(np.sum(s.B_total**2, axis=0))
        dB_mag = np.sqrt(np.sum(s.delta_B**2, axis=0))
        dB_over_B.append(float(np.mean(dB_mag / np.maximum(B_mag, 1e-20))))
    ax.plot(betas, dB_over_B, "^-", linewidth=2, markersize=8)
    ax.set_xlabel("β")
    ax.set_ylabel("⟨|δB|/|B|⟩")
    ax.set_title("Field perturbation magnitude")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "beta_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_field_slices(
    state: PerturbationState,
    R_grid: np.ndarray,
    Z_grid: np.ndarray,
    output_dir: Path,
    phi_slice: int = 0,
):
    """Plot 2D field slices at a given φ."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    BR = state.B_total[0, :, :, phi_slice]
    BPhi = state.B_total[1, :, :, phi_slice]
    BZ = state.B_total[2, :, :, phi_slice]
    p = state.p_profile[:, :, phi_slice]
    dB = state.delta_B[0, :, :, phi_slice]  # δB_R as example

    RR, ZZ = np.meshgrid(R_grid, Z_grid, indexing="ij")

    def plot_contour(ax, data, title, cmap="RdBu_r"):
        im = ax.pcolormesh(RR, ZZ, data.T, cmap=cmap, shading="auto")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_title(title)
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plot_contour(axes[0, 0], BR, f"B_R  (φ={phi_slice}, β={state.beta:.3f})")
    plot_contour(axes[0, 1], BPhi, "B_φ")
    plot_contour(axes[0, 2], BZ, "B_Z")
    plot_contour(axes[1, 0], p, "Pressure p [Pa]", cmap="viridis")
    plot_contour(axes[1, 1], dB, "δB_R (perturbation)")

    # |B| magnitude
    B_mag = np.sqrt(BR**2 + BPhi**2 + BZ**2)
    plot_contour(axes[1, 2], B_mag, "|B| [T]", cmap="magma")

    plt.suptitle(f"HAO Stellarator Fields at β = {state.beta:.4f}")
    plt.tight_layout()
    plt.savefig(output_dir / f"field_slice_beta{state.beta:.4f}_phi{phi_slice}.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def save_results(
    history: list[PerturbationState],
    output_dir: Path,
    R_grid: np.ndarray,
    Z_grid: np.ndarray,
    Phi_grid: np.ndarray,
):
    """Save all results to files."""
    # Metadata
    meta = {
        "beta_values": [s.beta for s in history],
        "converged": [s.converged for s in history],
        "n_iterations": [s.n_iterations for s in history],
        "residuals": [s.residual for s in history],
        "R_grid": R_grid.tolist(),
        "Z_grid": Z_grid.tolist(),
        "Phi_grid": Phi_grid.tolist(),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save field data at each β step
    for state in history:
        fname = f"state_beta_{state.beta:.5f}.npz"
        np.savez_compressed(
            output_dir / fname,
            B_total=state.B_total,
            J_total=state.J_total,
            p_profile=state.p_profile,
            delta_B=state.delta_B,
            delta_p=state.delta_p,
            beta=state.beta,
            converged=state.converged,
            n_iterations=state.n_iterations,
            residual=state.residual,
        )

    # Summary CSV
    with open(output_dir / "summary.csv", "w") as f:
        f.write("beta,converged,n_iterations,residual\n")
        for s in history:
            f.write(f"{s.beta:.6f},{s.converged},{s.n_iterations},{s.residual:.6e}\n")


def run_analysis(
    history: list[PerturbationState],
    R_grid: np.ndarray,
    Z_grid: np.ndarray,
    Phi_grid: np.ndarray,
    output_dir: Path,
    skip_topology: bool = False,
):
    """Run post-analysis on the beta climb results."""
    print(f"\n{'='*60}")
    print("Post-analysis")
    print(f"{'='*60}")

    # Plot convergence
    print("  Plotting convergence...")
    plot_beta_convergence(history, output_dir)

    # Plot field slices for key β values
    print("  Plotting field slices...")
    phi_mid = len(Phi_grid) // 4  # φ = π/2 slice
    for state in history:
        plot_field_slices(state, R_grid, Z_grid, output_dir, phi_slice=phi_mid)

    # Topology analysis (if available)
    if not skip_topology and HAS_TOPOQUEST:
        print("  Running topology analysis...")
        # TODO: connect to topoquest topology analysis
        # This would compute iota profiles, island chains, etc.
        print("    (topology analysis placeholder — connect to topoquest)")

    print(f"\nResults saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = setup_parser()
    args = parser.parse_args()

    # Determine β values
    if args.beta is not None:
        beta_values = sorted(args.beta)
    elif args.test:
        beta_values = [0.0, 0.01, 0.02]
    else:
        beta_values = DEFAULT_BETA_SCAN.copy()

    # Output directory
    output_dir = make_output_dir(args.output)
    print(f"\nOutput directory: {output_dir}")

    # Load coil data
    print(f"\nLoading coil vacuum fields from: {args.vacuum_dir}")
    coil_files = load_hao_coils(args.vacuum_dir, exclude_indices=EXCLUDED_COILS)
    print(f"  Loaded {len(coil_files)} coil files")

    if args.test:
        # Quick test: use only first 20 coils
        coil_files = coil_files[:20]
        print(f"  Test mode: using first {len(coil_files)} coils")

    # Define pressure profile function
    def p_profile_func(psi_n: float) -> float:
        """Base pressure profile shape (normalised to 1 at ψ_n=0)."""
        return max(0.0, 1.0 - psi_n) ** args.alpha

    # Create and run solver
    print(f"\nStarting finite-beta continuation...")
    print(f"  β values: {beta_values}")
    print(f"  α (pressure exponent): {args.alpha}")
    print(f"  Max iterations per step: {args.max_iter}")
    print(f"  Tolerance: {args.tol}")

    t_start = time.time()

    solver = FiniteBetaPerturbation(
        coil_files=coil_files,
        p_profile_func=p_profile_func,
        beta_values=beta_values,
        alpha_pressure=args.alpha,
        max_outer_iter=args.max_iter,
        tol=args.tol,
        verbose=True,
    )

    history = solver.run()
    t_elapsed = time.time() - t_start

    print(f"\nComputation time: {t_elapsed:.1f}s")

    # Save results
    print(f"\nSaving results...")
    save_results(history, output_dir, solver.R_grid, solver.Z_grid, solver.Phi_grid)

    # Run analysis
    run_analysis(
        history,
        solver.R_grid,
        solver.Z_grid,
        solver.Phi_grid,
        output_dir,
        skip_topology=args.skip_topology,
    )

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
