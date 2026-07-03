#!/usr/bin/env python3
"""Check generated documentation artifacts before publishing GitHub Pages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


FORBIDDEN_SUFFIXES = (".doctree", ".ipynb", ".map", ".pickle")


def check_artifacts(build_dir: Path) -> list[str]:
    errors: list[str] = []
    if not build_dir.exists():
        return [f"{build_dir}: build directory does not exist"]

    for path in sorted(build_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix in FORBIDDEN_SUFFIXES:
            errors.append(f"{path}: generated Pages artifact should not be published")
            continue
        if path.suffix not in {".css", ".js"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "sourceMappingURL=" in text:
            errors.append(f"{path}: contains a source map reference")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", nargs="?", default="docs/_build/html", type=Path)
    args = parser.parse_args(argv)

    errors = check_artifacts(args.build_dir.resolve())
    if errors:
        print("Generated documentation artifact check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"Checked generated documentation artifacts under {args.build_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
