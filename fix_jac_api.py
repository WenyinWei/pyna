import json

with open('notebooks/tutorials/stellarator_island_control.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][7]['source'])
new = src.replace(
    "xpt_Jac = vq.jacobian_matrix(xpt_seed, phi_span,\n                              method='RK45', rtol=1e-7, atol=1e-9)",
    "xpt_Jac = vq.jacobian_matrix(xpt_seed, phi_span,\n                              solve_ivp_kwargs=dict(method='RK45', rtol=1e-7, atol=1e-9))"
)
nb['cells'][7]['source'] = new
with open('notebooks/tutorials/stellarator_island_control.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Fixed jacobian_matrix API call')
