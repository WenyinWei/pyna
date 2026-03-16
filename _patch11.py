import sys, json, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
nb_path = pathlib.Path("D:/Repo/pyna/notebooks/tutorials/rmp_island_validation_solovev.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))
code_cells_idx = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
cell11_idx = code_cells_idx[11]
src = "".join(nb["cells"][cell11_idx]["source"])
# Fix: add tolerances and reduce n_turns for grow calls
old1 = "sm.grow(n_turns=3, init_length=1e-4, n_init_pts=3, both_sides=True)"
new1 = "sm.grow(n_turns=2, init_length=1e-4, n_init_pts=3, both_sides=True, rtol=1e-5, atol=1e-7)"
old2 = "um.grow(n_turns=3, init_length=1e-4, n_init_pts=3, both_sides=True)"
new2 = "um.grow(n_turns=2, init_length=1e-4, n_init_pts=3, both_sides=True, rtol=1e-5, atol=1e-7)"
count = src.count(old1) + src.count(old2)
src2 = src.replace(old1, new1).replace(old2, new2)
if count > 0:
    nb["cells"][cell11_idx]["source"] = [l+"\n" for l in src2.rstrip("\n").split("\n")]
    nb["cells"][cell11_idx]["source"][-1] = nb["cells"][cell11_idx]["source"][-1].rstrip("\n")
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Fixed {count} grow() calls in Cell 11")
else:
    print("Pattern not found in Cell 11 - current content:")
    print(src[:500])
