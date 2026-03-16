import sys, ast, functools, warnings
sys.stdout.reconfigure(encoding='utf-8')

FALLBACK = (
    "try:\n"
    "    from deprecated import deprecated\n"
    "except ImportError:\n"
    "    import warnings as _w, functools as _ft\n"
    "    def deprecated(*args, **kwargs):\n"
    "        def _dec(func):\n"
    "            @_ft.wraps(func)\n"
    "            def wrapper(*a, **kw):\n"
    "                _w.warn(f\"{func.__name__} is deprecated.\", DeprecationWarning, stacklevel=2)\n"
    "                return func(*a, **kw)\n"
    "            return wrapper\n"
    "        return _dec(args[0]) if (len(args) == 1 and callable(args[0])) else _dec\n"
)

files = ['pyna/diff/cycle.py', 'pyna/topo/manifold.py']
for fpath in files:
    with open(fpath, encoding='utf-8') as f:
        lines = f.readlines()
    bare = [i for i, l in enumerate(lines) if l.strip() == 'from deprecated import deprecated']
    print(f'{fpath}: bare import at lines {[x+1 for x in bare]}')
    if not bare:
        print('  nothing to do')
        continue
    lines[bare[0]] = FALLBACK
    for i in bare[1:]:
        lines[i] = ''
    src = ''.join(lines)
    ast.parse(src)
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(src)
    print('  syntax OK, written')

print('Done')
