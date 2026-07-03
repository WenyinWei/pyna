#!/usr/bin/env python3
"""Preflight checks for notebooks rendered in the public documentation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


LANGUAGES = ("en", "zh", "ja", "ko", "de", "fr", "ru")
SUSPICIOUS_OUTPUT_MARKERS = ("<script", "javascript:")


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return str(source)


def _read_notebook(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid notebook JSON: {exc}") from exc


def _cell_counts(notebook: dict) -> tuple[int, int]:
    markdown = 0
    code = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "markdown":
            markdown += 1
        elif cell.get("cell_type") == "code":
            code += 1
    return markdown, code


def _markdown_sources(notebook: dict) -> list[str]:
    return [
        _cell_source(cell).strip()
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "markdown"
    ]


def _output_texts(notebook: dict) -> Iterable[str]:
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []) or []:
            data = output.get("data", {}) if isinstance(output, dict) else {}
            if output.get("text"):
                text = output["text"]
                yield "".join(text) if isinstance(text, list) else str(text)
            for value in data.values():
                yield "".join(value) if isinstance(value, list) else str(value)


def _has_saved_output(notebook: dict) -> bool:
    return any(
        cell.get("cell_type") == "code" and bool(cell.get("outputs"))
        for cell in notebook.get("cells", [])
    )


def _has_code(notebook: dict) -> bool:
    return any(cell.get("cell_type") == "code" for cell in notebook.get("cells", []))


def _documented_notebooks(root: Path, languages: tuple[str, ...]) -> Iterable[Path]:
    for lang in languages:
        tutorial_dir = root / "i18n" / lang / "tutorials"
        if tutorial_dir.exists():
            yield from sorted(tutorial_dir.glob("*.ipynb"))


def check_saved_outputs(root: Path, languages: tuple[str, ...]) -> list[str]:
    errors: list[str] = []
    for path in _documented_notebooks(root, languages):
        try:
            notebook = _read_notebook(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if _has_code(notebook) and not _has_saved_output(notebook):
            errors.append(f"{path}: documented tutorial notebook has no saved outputs")
        for text in _output_texts(notebook):
            lower = text.lower()
            if any(marker in lower for marker in SUSPICIOUS_OUTPUT_MARKERS):
                errors.append(f"{path}: notebook output contains active HTML or JavaScript")
                break
    return errors


def check_i18n_parity(
    root: Path,
    languages: tuple[str, ...],
    max_identical_markdown_ratio: float,
) -> list[str]:
    errors: list[str] = []
    english_dir = root / "i18n" / "en" / "tutorials"
    if not english_dir.exists():
        return [f"{english_dir}: English tutorial notebook directory is missing"]

    english_notebooks = {path.name: path for path in sorted(english_dir.glob("*.ipynb"))}
    if not english_notebooks:
        return [f"{english_dir}: no English tutorial notebooks found"]

    for name, english_path in english_notebooks.items():
        english = _read_notebook(english_path)
        english_counts = _cell_counts(english)
        english_markdown = _markdown_sources(english)
        for lang in languages:
            if lang == "en":
                continue
            translated_path = root / "i18n" / lang / "tutorials" / name
            if not translated_path.exists():
                errors.append(f"{translated_path}: translated tutorial notebook is missing")
                continue
            translated = _read_notebook(translated_path)
            translated_counts = _cell_counts(translated)
            if translated_counts != english_counts:
                errors.append(
                    f"{translated_path}: cell type counts {translated_counts} do not "
                    f"match English source {english_counts}"
                )
            translated_markdown = _markdown_sources(translated)
            compared = min(len(english_markdown), len(translated_markdown))
            if compared == 0:
                continue
            identical = sum(
                1
                for source, translated_source in zip(english_markdown, translated_markdown)
                if source == translated_source
            )
            ratio = identical / compared
            if compared >= 3 and ratio > max_identical_markdown_ratio:
                errors.append(
                    f"{translated_path}: {identical}/{compared} markdown cells are still "
                    "identical to the English source"
                )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="notebooks", type=Path)
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    parser.add_argument("--skip-i18n-parity", action="store_true")
    parser.add_argument("--skip-output-check", action="store_true")
    parser.add_argument("--max-identical-markdown-ratio", type=float, default=0.8)
    args = parser.parse_args(argv)

    root = args.root.resolve()
    languages = tuple(args.languages)
    errors: list[str] = []

    if not root.exists():
        errors.append(f"{root}: notebook root does not exist")
    else:
        if not args.skip_output_check:
            errors.extend(check_saved_outputs(root, languages))
        if not args.skip_i18n_parity:
            errors.extend(
                check_i18n_parity(root, languages, args.max_identical_markdown_ratio)
            )

    if errors:
        print("Documentation notebook preflight failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"Checked documented notebooks under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
