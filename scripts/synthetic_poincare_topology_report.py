#!/usr/bin/env python3
"""Generate a synthetic Poincare boundary-topology report.

The script uses only analytic toy data.  It is intended as a public smoke demo
for the Poincare topology payload API, not as a magnetic-field validation case.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib
import numpy as np

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyna.plot import (
    PoincareCurvedIslandBar,
    plot_poincare_topology_payload_report,
    poincare_topology_report_payload,
)


def synthetic_poincare_points(n_theta: int = 180):
    """Return synthetic Poincare section points and radial labels."""

    theta = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False)
    radial_surfaces = np.asarray([0.30, 0.55, 0.78], dtype=float)
    radial = np.repeat(radial_surfaces, theta.size)
    angle = np.tile(theta, radial_surfaces.size)
    R = 3.0 + radial * np.cos(angle)
    Z = 0.72 * radial * np.sin(angle)
    return R, Z, radial, [theta.size] * radial_surfaces.size


def synthetic_dpk_metrics():
    """Return radial DP^k diagnostics with one chaotic shell."""

    return [
        SimpleNamespace(eigenvalue_ftle=0.08, spectral_recurrence_min=0.002, recurrent_surface_indicator=1.0),
        SimpleNamespace(eigenvalue_ftle=0.09, spectral_recurrence_min=0.070, recurrent_surface_indicator=0.0),
        SimpleNamespace(eigenvalue_ftle=0.01, spectral_recurrence_min=0.001, recurrent_surface_indicator=1.0),
    ]


def synthetic_island_bar() -> PoincareCurvedIslandBar:
    """Return one spectrum-style curved island-width bar."""

    s_path = 0.55 + np.linspace(-0.04, 0.04, 41)
    theta = 0.35 + 0.9 * (s_path - 0.55)
    return PoincareCurvedIslandBar(
        R_path=3.0 + s_path * np.cos(theta),
        Z_path=0.72 * s_path * np.sin(theta),
        mode_m=5,
        mode_n=2,
        radial_label=0.55,
        half_width=0.04,
        amplitude=2.0e-4,
        phase=0.35,
        kind="O",
    )


def synthetic_poincare_topology_payload():
    """Build a complete synthetic Poincare topology report payload."""

    R, Z, radial, trace_counts = synthetic_poincare_points()
    return poincare_topology_report_payload(
        R,
        Z,
        radial_label=radial,
        dpk_radial_labels=[0.30, 0.55, 0.78],
        dpk_metrics=synthetic_dpk_metrics(),
        fixed_points=[
            {"R": 3.55, "Z": 0.0, "kind": "O", "mode_m": 5, "mode_n": 2, "residual": 1.0e-10},
            {"R": 3.48, "Z": 0.20, "kind": "X", "mode_m": 5, "mode_n": 2, "residual": 2.0e-10},
        ],
        island_chains=[SimpleNamespace(m=5, n=2, radial_label=0.55, half_width=0.04)],
        island_bars=[synthetic_island_bar()],
        trace_counts=trace_counts,
        metadata={"case": "synthetic_poincare_topology_report"},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("~/MCFdata/pyna_synthetic/poincare_topology_report.png"),
        help="Output PNG path. Defaults to a public synthetic-data directory under ~/MCFdata.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Saved figure DPI.")
    args = parser.parse_args(argv)

    out = args.out.expanduser()
    payload = synthetic_poincare_topology_payload()
    fig, _axes, _classification = plot_poincare_topology_payload_report(
        payload,
        growth_threshold=0.05,
        recurrence_threshold=0.02,
        out_path=out,
        save_dpi=args.dpi,
        title="Synthetic Poincare boundary-topology report",
    )
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
