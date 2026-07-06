#!/usr/bin/env python3
"""Run a public W7-X VMEC counterpart benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyna.benchmark import run_vmec_counterpart_benchmark, write_vmec_benchmark_outputs


DEFAULT_WOUT = Path("~/MCFdata/W7X_public/stagextender_beta1/wout_std_scp00_beta1.nc")
DEFAULT_OUT = Path("~/MCFdata/W7X_public/stagextender_beta1/counterpart_benchmark")


def parse_surfaces(text: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one surface is required")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wout", type=Path, default=DEFAULT_WOUT, help="Public VMEC wout file.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Output directory outside the repository.")
    parser.add_argument("--booz-surfaces", type=parse_surfaces, default=(20, 50, 80), help="1-based booz_xform compute surfaces, comma-separated.")
    parser.add_argument("--mboz", type=int, default=24)
    parser.add_argument("--nboz", type=int, default=24)
    parser.add_argument("--skip-booz-xform", action="store_true", help="Skip hiddenSymmetries booz_xform smoke run.")
    parser.add_argument("--run-desc-booz", action="store_true", help="Also run DESC's make_boozmn_output path.")
    parser.add_argument("--desc-gpu", action="store_true", help="Call desc.set_device('gpu') before DESC operations.")
    parser.add_argument("--include-local-path", action="store_true", help="Include local wout path in JSON. Do not use for private cases.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    out_dir = args.out_dir.expanduser()
    desc_booz_path = out_dir / "desc_boozmn_benchmark.nc" if args.run_desc_booz else None
    report = run_vmec_counterpart_benchmark(
        args.wout.expanduser(),
        booz_surfaces=args.booz_surfaces,
        mboz=args.mboz,
        nboz=args.nboz,
        run_booz_xform=not args.skip_booz_xform,
        run_desc_booz=args.run_desc_booz,
        desc_use_gpu=args.desc_gpu,
        include_local_path=args.include_local_path,
        desc_booz_path=desc_booz_path,
    )
    paths = write_vmec_benchmark_outputs(
        report,
        out_dir,
        prefix="w7x_public_vmec_counterpart_benchmark",
        make_plots=not args.no_plots,
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    failed = [name for name, result in report.readers.items() if not result.ok]
    if report.booz_xform is not None and not report.booz_xform.ok:
        failed.append("booz_xform")
    if report.desc_booz is not None and not report.desc_booz.ok:
        failed.append("desc_booz")
    if failed:
        print("failed backends:", ", ".join(failed), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
