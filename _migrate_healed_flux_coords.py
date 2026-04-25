from pathlib import Path

src = Path(r'D:\Repo\pyna\pyna\topo\healed_scaffold_3d.py')
dst = Path(r'D:\Repo\pyna\pyna\topo\healed_flux_coords.py')

body = src.read_text(encoding='utf-8')
dst.write_text(body.replace('healed_scaffold_3d.py', 'healed_flux_coords.py', 1), encoding='utf-8')

shim = '''"""Compatibility shim for legacy imports.

Use ``pyna.topo.healed_flux_coords`` as the formal public module.
"""

from pyna.topo.healed_flux_coords import *  # noqa: F401,F403
'''
src.write_text(shim, encoding='utf-8')

print('ok')
