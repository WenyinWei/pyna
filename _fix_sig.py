import re

with open(r'C:\Users\Legion\Nutstore\1\Repo\pyna\pyna\topo\island_healed_coords.py',
          encoding='utf-8', errors='replace') as f:
    content = f.read()

# Find and update the build_from_trajectory_npz signature/docstring
old_sig = '''def build_from_trajectory_npz(
    coords_npz: str,
    trajectory_npz: str,
    r_inner_fit: float = 0.82,
    r_island: float = 1.0,
    blend_width: float = 0.12,
    n_fourier: int = 8,
    theta_X: float = np.pi,
    theta_O: float = 0.0,
    n_anchors: int = 40,
) -> \'IslandHealedCoordMap\':'''

new_sig = '''def build_from_trajectory_npz(
    coords_npz: str,
    trajectory_npz: str,
    r_inner_fit: float = 0.82,
    r_island: float = 1.0,
    blend_width: float = 0.12,
    n_fourier: int = 8,
    theta_X: float = np.pi,
    theta_O: float = 0.0,
    n_anchors: int = 40,
    fp_by_section: dict | None = None,
    r_geom_min: float = 0.05,
    cycle_weight: float = 200.0,
) -> \'IslandHealedCoordMap\':'''

# Normalize line endings for comparison
content_norm = content.replace('\r\n', '\n').replace('\r', '\n')
old_sig_norm = old_sig.replace('\r\n', '\n')
new_sig_norm = new_sig.replace('\r\n', '\n')

if old_sig_norm in content_norm:
    content_norm = content_norm.replace(old_sig_norm, new_sig_norm, 1)
    with open(r'C:\Users\Legion\Nutstore\1\Repo\pyna\pyna\topo\island_healed_coords.py',
              'w', encoding='utf-8') as f:
        f.write(content_norm)
    print('Signature updated')
else:
    # Try to find the function
    idx = content_norm.find('def build_from_trajectory_npz(')
    print(f'Function at index {idx}')
    print(repr(content_norm[idx:idx+300]))
