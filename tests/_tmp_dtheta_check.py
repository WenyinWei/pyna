import numpy as np
from scipy.integrate import solve_ivp

iota0 = 0.3

print('Correct (delta_theta)_mn = iota0 * c / (i*alpha):')
MODES_t = [(2,1),(1,0),(1,-1)]
dBth_F_t = [0.3+0.15j, 0.4+0.1j, 0.3+0.15j]

for (m,n), c in zip(MODES_t, dBth_F_t):
    alpha = m*iota0 + n
    A_correct = iota0 * c / (1j * alpha)
    A_paper = c / (1j * m) if m != 0 else float('nan')
    match = np.isclose(A_correct, A_paper) if n==0 else 'n/a'
    print(f'  (m={m},n={n}) alpha={alpha:.4f}: correct={A_correct:.6f}  paper={A_paper:.6f}  match_n0={match}')

# Numerically verify which formula gives O(eps^2) residual
print()
m_t, n_t = 2, 1
c_t = 0.3 + 0.15j
alpha_t = m_t*iota0 + n_t

A_ode  = iota0 * c_t / (1j * alpha_t)
A_pap  = c_t / (1j * m_t)

print('ODE residuals for single (2,1) mode, pure angle perturbation:')
for label, A in [('ODE formula iota/(i*alpha)', A_ode), ('Paper formula 1/(im)', A_pap)]:
    thetas = np.linspace(0, 2*np.pi, 40, endpoint=False)
    eps_t = 0.005
    resids = []
    for th0 in thetas:
        dth0 = 2*(A.real*np.cos(m_t*th0) - A.imag*np.sin(m_t*th0)) * eps_t
        def rhs_ang(phi, y, c=c_t, m=m_t, n=n_t, iota=iota0, e=eps_t):
            h = 2*(c.real*np.cos(m*y[0]+n*phi) - c.imag*np.sin(m*y[0]+n*phi))
            return [iota + e*iota*h]
        sol = solve_ivp(rhs_ang, [0,2*np.pi],[th0+dth0],method='DOP853',rtol=1e-13,atol=1e-15)
        th_fin = sol.y[0,-1]
        th_fin_mod = th_fin % (2*np.pi)
        dth_fin = 2*(A.real*np.cos(m_t*th_fin_mod) - A.imag*np.sin(m_t*th_fin_mod)) * eps_t
        resids.append(abs(th_fin - (th0 + 2*np.pi*iota0 + dth_fin)))
    print(f'  {label}: max residual = {max(resids):.4e}  (eps^2 = {eps_t**2:.2e})')
