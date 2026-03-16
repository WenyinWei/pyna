import sys, json, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
nb_path = pathlib.Path("D:/Repo/pyna/notebooks/tutorials/rmp_island_validation_solovev.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))
code_cells_idx = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
src = "".join(nb["cells"][code_cells_idx[11]]["source"])
import re
for m in re.finditer(r"\.grow\([^\)]+\)", src):
    print(m.group())
