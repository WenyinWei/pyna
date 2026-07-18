from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).parents[1] / "scripts" / "check_privacy_terms.py"
_SPEC = importlib.util.spec_from_file_location("check_privacy_terms", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _scan(root: Path) -> list[str]:
    return _MODULE.check_privacy_terms(
        (root,),
        [],
        check_local_paths=True,
        max_bytes=2_000_000,
    )


def test_privacy_scan_skips_xmake_generated_cache(tmp_path: Path) -> None:
    cache = tmp_path / ".xmake" / "linux" / "cache"
    cache.mkdir(parents=True)
    local_path = Path("/") / "home" / "local-user" / "build"
    (cache / "option").write_text(f"{local_path}\n", encoding="utf-8")

    assert _scan(tmp_path) == []


def test_privacy_scan_still_rejects_local_path_in_source(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    local_path = Path("/") / "home" / "local-user" / "data"
    source.write_text(f'DATA_ROOT = "{local_path}"\n', encoding="utf-8")

    errors = _scan(tmp_path)

    assert len(errors) == 1
    assert str(source) in errors[0]
