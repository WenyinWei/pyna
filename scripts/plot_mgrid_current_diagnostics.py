#!/usr/bin/env python3
"""Plot mgrid current-density diagnostics on smooth PEST coordinates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyna.io import compute_current_density_cylindrical, load_vmec_mgrid
from pyna.plot import (
    plot_mgrid_current_cylindrical_components,
    plot_pest_current_components,
    plot_surface_fourier_ripple_summary,
)
from pyna.toroidal.diagnostics import (
    compute_pest_current_components,
    load_smooth_pest_coordinates,
    surface_fourier_spectrum,
)


def _csv_floats(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mgrid", action="append", type=Path, required=True, help="VMEC mgrid file. Repeat for comparisons.")
    parser.add_argument("--pest", action="append", type=Path, required=True, help="Matching smooth PEST .npz file. Repeat in --mgrid order.")
    parser.add_argument("--label", action="append", default=None, help="Plot label. Repeat in --mgrid order.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sections-deg", default="0,45,90,135")
    parser.add_argument("--rhos", default="0.40,0.64,0.86,0.94")
    args = parser.parse_args()

    if len(args.mgrid) != len(args.pest):
        raise SystemExit("--mgrid and --pest must be repeated the same number of times")
    labels = args.label or [path.stem for path in args.mgrid]
    if len(labels) != len(args.mgrid):
        raise SystemExit("--label must be omitted or repeated once per --mgrid")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sections_deg = _csv_floats(args.sections_deg)
    rhos = _csv_floats(args.rhos)
    current_items = []
    pest_diagnostics = []
    spectra = []
    summary: dict[str, object] = {
        "sections_deg": sections_deg,
        "rhos": rhos,
        "fields": [],
    }

    for label, mgrid_path, pest_path in zip(labels, args.mgrid, args.pest):
        field = load_vmec_mgrid(mgrid_path)
        current = compute_current_density_cylindrical(field)
        coords = load_smooth_pest_coordinates(pest_path)
        pest_components = compute_pest_current_components(current, coords, sections_deg, label=label)
        spectrum = surface_fourier_spectrum(coords, rho_values=rhos, sections_deg=sections_deg)
        current_items.append((label, current, coords))
        pest_diagnostics.append(pest_components)
        spectra.append((label, spectrum))
        summary["fields"].append(
            {
                "label": label,
                "mgrid": str(mgrid_path),
                "pest": str(pest_path),
                "component_stats": pest_components.component_stats(),
                "surface_spectrum": spectrum,
            }
        )

    cyl_png = args.out_dir / "mgrid_current_density_cylindrical_components.png"
    pest_png = args.out_dir / "mgrid_current_density_smooth_pest_components.png"
    ripple_png = args.out_dir / "mgrid_smooth_pest_high_m_ripple_summary.png"
    fig, _axes = plot_mgrid_current_cylindrical_components(current_items, sections_deg, out_path=cyl_png)
    plt.close(fig)
    fig, _axes = plot_pest_current_components(pest_diagnostics, out_path=pest_png)
    plt.close(fig)
    fig, _ax = plot_surface_fourier_ripple_summary(spectra, out_path=ripple_png)
    plt.close(fig)

    summary["outputs"] = {
        "cylindrical_components_png": str(cyl_png),
        "smooth_pest_components_png": str(pest_png),
        "high_m_ripple_png": str(ripple_png),
    }
    summary_path = args.out_dir / "mgrid_current_diagnostics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["outputs"], indent=2))
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
