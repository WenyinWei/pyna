import sys, json, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
nb = json.loads(pathlib.Path("D:/Repo/pyna/notebooks/tutorials/rmp_island_validation_solovev.ipynb").read_text(encoding="utf-8"))
errs = 0
for i, cell in enumerate(nb["cells"]):
    for out in cell.get("outputs", []):
        if out.get("output_type") == "error":
            errs += 1
            print(f"Cell {i} ERROR: {out.get('ename')}: {out.get('evalue','')[:100]}")
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
executed = sum(1 for c in code_cells if c.get("execution_count"))
print(f"Executed cells: {executed}/{len(code_cells)}  Errors: {errs}")
