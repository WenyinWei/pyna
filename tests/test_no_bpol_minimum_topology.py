from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_no_bpol_minimum_topology_locators():
    """Topology points must be found by fixed-point/root solves, not Bpol minima."""

    suspicious = []
    needles = ("argmin", "nanargmin", "minimize")
    topology_words = ("axis", "xpoint", "x_point", "opoint", "o_point", "fixed", "cycle")
    bpol_words = ("bpol", "Bpol", "B_pol", "BR ** 2 + BZ ** 2", "BR**2 + BZ**2")
    for base in (ROOT / "pyna", ROOT / "scripts"):
        for path in base.rglob("*.py"):
            if any(part in {"build", "dist", "__pycache__"} for part in path.parts):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            lower_path = str(path.relative_to(ROOT)).lower()
            if not any(word in lower_path for word in topology_words):
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if any(n in line for n in needles) and any(b in line for b in bpol_words):
                    suspicious.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert suspicious == []
