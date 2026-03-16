import sys, ast
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

with open('pyna/topo/manifold.py', encoding='utf-8') as f:
    lines = f.readlines()

bare = [i for i, l in enumerate(lines) if l.strip() == 'from deprecated import deprecated']
print('bare deprecated imports at lines:', [x+1 for x in bare])

for idx in reversed(bare):
    lines[idx] = FALLBACK

src = ''.join(lines)
ast.parse(src)
print('syntax OK')

with open('pyna/topo/manifold.py', 'w', encoding='utf-8') as f:
    f.write(src)
print('written')
