import subprocess
r = subprocess.run(['git', 'add', 'notebooks/tutorials/stellarator_island_control.ipynb'], cwd='D:/Repo/pyna', capture_output=True, text=True)
print(r.stdout, r.stderr)
r = subprocess.run(['git', 'commit', '-m', 'feat(tutorial): enhance stellarator_island_control with publication-quality Poincare visualization'], cwd='D:/Repo/pyna', capture_output=True, text=True)
print(r.stdout, r.stderr)
r = subprocess.run(['git', 'push'], cwd='D:/Repo/pyna', capture_output=True, text=True)
print(r.stdout, r.stderr)
