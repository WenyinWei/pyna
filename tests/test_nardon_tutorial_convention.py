import json
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    REPOSITORY / "notebooks" / "tutorials" / "RMP_resonance_analysis.ipynb",
    *(
        REPOSITORY
        / "notebooks"
        / "i18n"
        / language
        / "tutorials"
        / "RMP_resonance_analysis.ipynb"
        for language in ("en", "zh", "ja", "ko", "de", "fr", "ru")
    ),
)
THEORY_TAG = "nardon-convention-theory"
EXECUTABLE_TAG = "nardon-convention-executable"


def _read_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _tagged_cells(notebook: dict, tag: str) -> list[dict]:
    return [
        cell
        for cell in notebook["cells"]
        if tag in cell.get("metadata", {}).get("tags", [])
    ]


def test_public_rmp_tutorials_carry_the_same_executable_convention_lock():
    reference_code = None
    for path in NOTEBOOKS:
        notebook = _read_notebook(path)
        theory = _tagged_cells(notebook, THEORY_TAG)
        executable = _tagged_cells(notebook, EXECUTABLE_TAG)

        assert len(theory) == 4, path
        assert len(executable) == 3, path
        assert all(cell["cell_type"] == "markdown" for cell in theory)
        assert all(cell["cell_type"] == "code" for cell in executable)
        assert all(cell.get("outputs") == [] for cell in executable)

        code = tuple(_source(cell) for cell in executable)
        if reference_code is None:
            reference_code = code
        else:
            assert code == reference_code, path


def test_nardon_tutorial_states_the_nonnegotiable_sign_and_field_contracts():
    notebook = _read_notebook(NOTEBOOKS[0])
    theory = "\n".join(
        _source(cell) for cell in _tagged_cells(notebook, THEORY_TAG)
    )

    required_fragments = (
        r"e^{-i(m\theta^*+n_N\varphi)}",
        r"\widetilde b^1_{-m,-n_N}",
        r"(m,-n_0),\qquad(-m,+n_0)",
        r"n_N=N_{\rm fp}k",
        r"\widetilde b^1_{\rm res}=2",
        r"\sigma_{\rm Chir}^{m,m+1}",
        r"\mathbf B=\mathbf B_0+\delta\mathbf B",
        "must not be conflated",
        "must be checked against Newton/cyna periodic points",
    )
    for fragment in required_fragments:
        assert fragment in theory


def test_nardon_tutorial_executable_cells_pass_as_a_small_regression_case():
    notebook = _read_notebook(NOTEBOOKS[0])
    namespace = {"__name__": "__nardon_tutorial_test__"}
    for index, cell in enumerate(_tagged_cells(notebook, EXECUTABLE_TAG)):
        source = _source(cell)
        exec(compile(source, f"nardon_tutorial_cell_{index}.py", "exec"), namespace)

    chain = namespace["chain_nardon"]
    assert chain.coefficient_n == -namespace["n0_nardon"]
    assert namespace["period_spectrum_nardon"].nardon_n[
        namespace["period_idx_nardon"]
    ] == -namespace["nfp_nardon"]
