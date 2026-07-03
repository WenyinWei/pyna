from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_script(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def test_documented_i18n_notebooks_have_saved_outputs_and_parity():
    result = run_script("scripts/check_docs_notebooks.py", "--root", "notebooks")
    assert result.returncode == 0, result.stderr


def test_docs_artifact_check_rejects_raw_notebook_and_source_map(tmp_path):
    build_dir = tmp_path / "html"
    build_dir.mkdir()
    (build_dir / "index.html").write_text("<html></html>", encoding="utf-8")
    assert run_script("scripts/check_docs_artifacts.py", str(build_dir)).returncode == 0

    (build_dir / "raw.ipynb").write_text("{}", encoding="utf-8")
    result = run_script("scripts/check_docs_artifacts.py", str(build_dir))
    assert result.returncode == 1
    assert "raw.ipynb" in result.stderr


def test_privacy_check_uses_redacted_denylist_terms(tmp_path):
    root = tmp_path / "public"
    root.mkdir()
    denylist = tmp_path / "denylist.txt"
    denylist.write_text("sensitive-private-name\n", encoding="utf-8")
    (root / "page.rst").write_text("This mentions sensitive-private-name.", encoding="utf-8")

    result = run_script(
        "scripts/check_privacy_terms.py",
        "--root",
        str(root),
        "--denylist-file",
        str(denylist),
    )

    assert result.returncode == 1
    assert "term #1" in result.stderr
    assert "sensitive-private-name" not in result.stderr
