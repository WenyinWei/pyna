from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
QUICKSTART_LANGUAGES = ("en", "zh", "de", "fr", "ja", "ko", "ru")


def _python_blocks(path: Path) -> tuple[str, ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        if lines[index].strip() != ".. code-block:: python":
            index += 1
            continue
        index += 1
        while index < len(lines) and not lines[index].strip():
            index += 1
        block: list[str] = []
        while index < len(lines):
            line = lines[index]
            if line.startswith("   "):
                block.append(line[3:])
                index += 1
            elif not line.strip():
                block.append("")
                index += 1
            else:
                break
        blocks.append("\n".join(block).rstrip())
    return tuple(blocks)


def _quickstart_blocks(language: str) -> tuple[str, ...]:
    return _python_blocks(REPOSITORY / "docs" / language / "quickstart.rst")


def test_localized_quickstarts_share_the_executed_field_line_example():
    reference = _quickstart_blocks("en")[1]
    for language in QUICKSTART_LANGUAGES[1:]:
        assert _quickstart_blocks(language)[1] == reference


def test_quickstart_field_line_example_uses_current_tracer_api():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup, field_line = _quickstart_blocks("en")[:2]
    namespace = {"__name__": "__quickstart_test__"}
    exec(compile(setup, "docs/en/quickstart.rst:block-1", "exec"), namespace)

    # Keep the published example intact while making the regression test fast.
    smoke_source = field_line.replace("n_turns = 80", "n_turns = 2")
    exec(compile(smoke_source, "docs/en/quickstart.rst:block-2", "exec"), namespace)

    points = namespace["poincare_pts"]
    assert points.ndim == 2
    assert points.shape[1] == 2
    assert points.shape[0] > 0
    plt.close("all")
