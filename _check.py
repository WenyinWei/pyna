import sys, json, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

nb_path = pathlib.Path("D:/Repo/pyna/notebooks/tutorials/rmp_island_validation_solovev.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
print(f"Total code cells: {len(code_cells)}")

# Show Cell 12 first 5 lines to confirm index
print("Cell 12 start:", "".join(code_cells[12]["source"])[:200])
