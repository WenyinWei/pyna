# Contributing to pyna

## Pre-push Checklist

**Never push code that hasn't passed local verification.** This avoids burning CI minutes on trivial import errors.

### 1. Install all dependencies locally

```bash
pip install -e ".[docs,dev]"
pip install deprecated joblib pandas scikit-image pytest-timeout nbmake
```

### 2. Verify all module imports

Run this before every push touching `pyna/` source:

```bash
py -3 -c "
import sys, importlib; sys.path.insert(0, '.')
mods = [
    'pyna', 'pyna.topo.manifold', 'pyna.diff.cycle',
    'pyna.toroidal.visual.tokamak_manifold', 'pyna.toroidal.control.island_control',
    'pyna.topo.poincare', 'pyna.toroidal.equilibrium.stellarator',
    'pyna.toroidal.coils.coil_system', 'pyna.topo.toroidal_cycle',
    'pyna.topo.variational', 'pyna.topo.manifold_improve',
    'pyna.toroidal.control.gap_response', 'pyna.topo.chaos',
]
ok = True
for m in mods:
    try: importlib.import_module(m); print(f'OK  {m}')
    except Exception as e: print(f'FAIL {m}: {e}'); ok = False
print('ALL OK' if ok else 'SOME FAILED — fix before pushing')
"
```

All lines must print `OK` before pushing.

### 3. Run unit tests

```bash
py -3 -m pytest tests/ -x -q
```

### 4. For notebook changes — run the affected notebook locally

```bash
py -3 -m jupyter nbconvert --to notebook --execute \
    notebooks/tutorials/<changed_notebook>.ipynb \
    --output notebooks/tutorials/<changed_notebook>.ipynb \
    --ExecutePreprocessor.timeout=300
```

Verify the output cells look correct before committing.

### 5. For new external dependencies

- Add to `dependencies` in `pyproject.toml` (not just `docs` extras) if used in core `pyna/` source
- Add to the explicit `pip install` lines in both CI workflows:
  - `.github/workflows/docs.yml`
  - `.github/workflows/notebook-tests.yml`
- Run the import check (step 2) on a clean environment if possible

---

## Why This Matters

CI runs on a fresh Ubuntu environment where only explicitly listed packages are installed.
A missing `import foo` in any file reachable from `pyna/__init__.py` silently breaks **all** notebook imports,
causing misleading `NameError` downstream (not the real `ImportError`).

The rule: **local green → push. Not: push → wait for CI → fix → repeat.**
