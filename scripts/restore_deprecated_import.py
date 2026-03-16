import sys, ast
sys.stdout.reconfigure(encoding='utf-8')

files = ['pyna/topo/manifold.py', 'pyna/diff/cycle.py']
for fp in files:
    with open(fp, encoding='utf-8') as f:
        src = f.read()
    # Remove the try/except wrapper, keep only the direct import
    import re
    pattern = (
        r'try:\s*\n'
        r'    from deprecated import deprecated\s*\n'
        r'except ImportError:.*?(?=\ndef |\nclass |\nfrom |\nimport |\n[A-Za-z_])'
    )
    m = re.search(pattern, src, re.DOTALL)
    if m:
        src = src[:m.start()] + 'from deprecated import deprecated\n' + src[m.end():]
        ast.parse(src)
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(src)
        print(f'{fp}: restored to direct import, syntax OK')
    else:
        print(f'{fp}: no try/except wrapper found, checking...')
        lines = src.splitlines()
        for i, l in enumerate(lines):
            if 'deprecated' in l:
                print(f'  line {i+1}: {l}')
