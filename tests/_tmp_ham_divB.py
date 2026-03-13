"""
Test Eq.(5.1) with a genuine Hamiltonian field-line system (div B = 0 by construction).

Hamiltonian:
    H(r, theta, phi) = psi(r) + eps * a * cos(m*theta + n*phi)

where psi(r) = iota0*r + (shear/2)*r^2  =>  psi'(r) = iota(r) = iota0 + shear*r

Hamilton's equations (phi = time):
    dtheta/dphi =  dH/dr = iota(r)            [no angular perturbation since a = const]
    dr/dphi     = -dH/dtheta = eps*m*a*sin(m*theta + n*phi)

The Fourier coefficient of delta_Br in the 2Re[c*exp(i(m*theta+n*phi))] convention:
    delta_Br = eps*m*a*sin(m*theta+n*phi)
             = 2*Re[-i*m*a/2 * exp(i(m*theta+n*phi))]
    => c_mn = -i*m*a/2
    => |c_mn|^2 = (m*a)^2/4

Paper Eq.(5.1), one-sided sum * 2 for real field:
    <delta_r> = -2*|c_mn|^2 * eps^2 / (alpha * shear * (2*pi)^2)
              = -(m*a*eps)^2 / (2 * alpha * shear * (2*pi)^2)
"""
import numpy as np
from scipy.integrate import solve_ivp

iota0 = (np.sqrt(5)-1)/2   # golden ratio -- well separated from resonances
shear = -0.5
a     = 1.0                 # vector potential amplitude (real, constant)

print("Hamiltonian field-line system  (div B = 0 by construction)")
print(f"H = psi(r) + eps*a*cos(m*theta+n*phi),  iota0 = {iota0:.6f},  shear = {shear}")
print()

N_rev = 400
N_th0 = 60
theta0_arr = np.linspace(0, 2*np.pi, N_th0, endpoint=False)

for m_t, n_t in [(1, 0), (2, 1), (1, -1)]:
    alpha = m_t*iota0 + n_t
    # Fourier coefficient (one-sided complex convention for real field):
    c_mn = -1j*m_t*a/2
    # Paper formula (two-sided = 2x one-sided):
    # <delta_r> = -2|c_mn|^2 * eps^2 / (alpha * shear * (2pi)^2)
    def formula(eps):
        return -2*abs(c_mn)**2*eps**2 / (alpha*shear*(2*np.pi)**2)

    print(f"Mode (m={m_t},n={n_t}),  alpha = {alpha:.6f}")
    print(f"  c_mn = {c_mn:.4f},  |c_mn|^2 = {abs(c_mn)**2:.4f}")
    print(f"  {'eps':>8}  {'<r>_Poincare':>14}  {'formula':>14}  {'ratio':>8}")

    for eps in [0.1, 0.05, 0.02, 0.01, 0.005]:
        all_r = []
        for th0 in theta0_arr:
            sol = solve_ivp(
                lambda p, y, e=eps, m=m_t, n=n_t, aa=a: [
                    e * m * aa * np.sin(m*y[1] + n*p),   # dr/dphi
                    iota0 + shear*y[0]                   # dtheta/dphi
                ],
                [0, N_rev*2*np.pi], [0.0, th0],
                method='DOP853', rtol=1e-11, atol=1e-13,
                t_eval=2*np.pi*np.arange(1, N_rev+1))
            all_r.extend(sol.y[0].tolist())
        mean_r = np.mean(all_r)
        f_val  = formula(eps)
        ratio  = mean_r/f_val if abs(f_val) > 1e-20 else float('nan')
        print(f"  {eps:>8.4f}  {mean_r:>14.4e}  {f_val:>14.4e}  {ratio:>8.4f}")

    # Convergence slope: should be eps^2
    eps_vals = [0.02, 0.01, 0.005]
    means = []
    for eps in eps_vals:
        all_r = []
        for th0 in theta0_arr:
            sol = solve_ivp(
                lambda p, y, e=eps, m=m_t, n=n_t, aa=a: [
                    e * m * aa * np.sin(m*y[1] + n*p),
                    iota0 + shear*y[0]
                ],
                [0, N_rev*2*np.pi], [0.0, th0],
                method='DOP853', rtol=1e-11, atol=1e-13,
                t_eval=2*np.pi*np.arange(1, N_rev+1))
        all_r.extend(sol.y[0].tolist())
        means.append(np.mean(all_r))
    if all(abs(m) > 1e-15 for m in means):
        slope = np.polyfit(np.log(eps_vals), np.log(np.abs(means)), 1)[0]
        print(f"  <r> scaling slope = {slope:.3f}  (expected 2)")
    print()
