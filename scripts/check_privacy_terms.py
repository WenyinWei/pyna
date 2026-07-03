#!/usr/bin/env python3
"""Scan public files for locally supplied private terms without printing them."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


DEFAULT_SKIP_DIRS = {
    ".codegraph",
    ".git",
    ".hypothesis",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "docs/_build",
    "docs/notebooks",
    "pyna_output",
    "wheelhouse",
}
DEFAULT_SKIP_SUFFIXES = {
    ".h5",
    ".hdf5",
    ".ipynb_checkpoints",
    ".jpg",
    ".jpeg",
    ".map",
    ".nc",
    ".npy",
    ".npz",
    ".png",
    ".pyc",
    ".so",
    ".svgz",
    ".whl",
}
LOCAL_PATH_PATTERNS = (
    re.compile(r"(?<![\w:/.-])/(?:home|Users)/[A-Za-z0-9_.-]+/"),
    re.compile(r"(?<![\w:/.-])[A-Za-z]:\\\\Users\\\\[A-Za-z0-9_.-]+\\\\"),
)


def _add_term(terms: list[str], value: str) -> None:
    stripped = value.strip()
    if stripped and not stripped.startswith("#") and stripped not in terms:
        terms.append(stripped)


def _load_terms(paths: list[Path], inline_text: str | None = None) -> list[str]:
    terms: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            _add_term(terms, line)
    if inline_text:
        for line in inline_text.splitlines():
            _add_term(terms, line)
    return terms


def _skip_file(
    path: Path,
    roots: tuple[Path, ...],
    max_bytes: int,
    denylist_paths: set[Path],
) -> bool:
    if path.resolve() in denylist_paths:
        return True
    parts = set(path.parts)
    if parts & DEFAULT_SKIP_DIRS:
        return True
    if path.suffix in DEFAULT_SKIP_SUFFIXES:
        return True
    try:
        if path.stat().st_size > max_bytes:
            return True
    except OSError:
        return True
    for root in roots:
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        rel_text = rel.as_posix()
        if rel_text.startswith("docs/_build/") or rel_text.startswith("docs/notebooks/"):
            return True
    return False


def _iter_files(roots: tuple[Path, ...], max_bytes: int, denylist_paths: set[Path]):
    for root in roots:
        if root.is_file():
            if not _skip_file(root, roots, max_bytes, denylist_paths):
                yield root
            continue
        for path in root.rglob("*"):
            if path.is_file() and not _skip_file(path, roots, max_bytes, denylist_paths):
                yield path


def check_privacy_terms(
    roots: tuple[Path, ...],
    terms: list[str],
    *,
    check_local_paths: bool,
    max_bytes: int,
    denylist_paths: set[Path] | None = None,
) -> list[str]:
    errors: list[str] = []
    for path in _iter_files(roots, max_bytes, denylist_paths or set()):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for index, term in enumerate(terms, start=1):
            if term and term in text:
                errors.append(f"{path}: contains private denylist term #{index}")
        if check_local_paths and any(pattern.search(text) for pattern in LOCAL_PATH_PATTERNS):
            errors.append(f"{path}: contains an absolute local filesystem path")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", default=None, type=Path)
    parser.add_argument("--denylist-file", action="append", default=None, type=Path)
    parser.add_argument("--check-local-paths", action="store_true")
    parser.add_argument("--max-bytes", default=2_000_000, type=int)
    args = parser.parse_args(argv)

    roots = tuple((args.root or [Path(".")]))
    roots = tuple(path.resolve() for path in roots)
    denylist_paths = list(args.denylist_file or [])
    env_path = os.environ.get("PYNA_PRIVACY_DENYLIST")
    if env_path:
        denylist_paths.append(Path(env_path))
    terms = _load_terms(
        [path.resolve() for path in denylist_paths],
        inline_text=os.environ.get("PYNA_PRIVACY_TERMS"),
    )

    errors = check_privacy_terms(
        roots,
        terms,
        check_local_paths=args.check_local_paths,
        max_bytes=args.max_bytes,
        denylist_paths={path.resolve() for path in denylist_paths},
    )
    if errors:
        print("Privacy preflight failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    if terms:
        print(f"Checked {len(terms)} private denylist term(s) under {', '.join(map(str, roots))}")
    else:
        print("Checked public files; no private denylist file was supplied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
