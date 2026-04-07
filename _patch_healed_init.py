from pathlib import Path
p = Path(r'C:\Users\Legion\Nutstore\1\Repo\pyna\pyna\topo\__init__.py')
text = p.read_text(encoding='utf-8', errors='replace')
text = text.replace('    build_from_trajectory_npz,\n)', '    build_from_trajectory_npz,\n    fp_by_section_from_orbits,\n    build_from_orbits,\n)')
text = text.replace('    "build_from_trajectory_npz",\n', '    "build_from_trajectory_npz",\n    "fp_by_section_from_orbits",\n    "build_from_orbits",\n')
p.write_text(text, encoding='utf-8')
print('patched topo __init__')
