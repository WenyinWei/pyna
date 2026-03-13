"""
Test Eq.(5.1) with a physically correct (div B = 0) perturbation model.

Vector-potential model:
  A_phi = psi(r) + eps * a * exp(i(m*theta + n*phi)) + c.c.
  psi(r) = iota0*r + shear/2*r^2   (gives iota(r) = iota0 + shear*r)

From B = curl A:
  B_r   = (1/r) * partial_theta(A_phi) = (im/r) * eps * a * exp(i(...)) + c.c.
  B_theta = -partial_r(A_phi) = -(psi'(r) + eps*a' * exp(i(...))) + c.c.
          = -iota(r) - eps*a' * exp(i(...)) + c.c.
  B_phi = 1  (constant, normalized)

Field line equations (dX/dphi = B_X / B_phi):
  dr/dphi     = B_r     = eps * (im/r0) * a * exp(i(m*theta+n*phi)) + c.c.
              = eps * 2*Re[(im/r0)*a * exp(i(m*theta+n*phi))]
  dtheta/dphi = -B_theta = iota(r) + eps * a' * exp(i(m*theta+n*phi)) + c.c.
              = iota(r) + eps * 2*Re[a' * exp(i(m*theta+n*phi))]

This IS a Hamiltonian system with H = psi(r) + eps*(a*exp(i(...))+c.c.).
The paper's formula should apply here.

For simplicity, choose a = const (a' = 0), so:
  dr/dphi = eps * 2*Re[(im*a/r0) * exp(i(m*theta+n*phi))]
  dtheta/dphi = iota(r)    [no angular perturbation when a'=0]

Then delta_Br_mn = im*a*eps/r0,  delta_Btheta_mn = 0
Paper formula (one mode, one-sided):
  <delta_r> = -sum |delta_Br_mn|^2 / (Bphi * (m*iota+n)) / (iota' * (2pi)^2)
            = -(im*a*eps/r0)^2 / (1.0 * alpha) / (shear * (2pi)^2)
            = |m*a*eps/r0|^2 / alpha / (-shear * (2pi)^2)
            ... but wait, (im)^2 = -m^2, so (delta_Br)^2 = -(m*a*eps/r0)^2

Hmm, let me use the absolute value: |delta_Br_mn|^2 = (m*a*eps/r0)^2
"""
import numpy as np
from scipy.integrate import solve_ivp

iota0 = (np.sqrt(5)-1)/2
shear = -0.5
r0    = 1.0       # reference radius
Bphi  = 1.0

# Single mode test: (m=1, n=0), a=1.0 (real, constant)
m_t, n_t = 1, 0
a = 1.0           # vector potential amplitude (a' = 0)
alpha = m_t*iota0 + n_t

# delta_Br_mn = im*a/r0 = i*a/r0 = i (complex)
dBr_mn = 1j*m_t*a/r0   # complex amplitude

print(f"Hamiltonian model, mode (m={m_t},n={n_t}), alpha={alpha:.6f}")
print(f"delta_Br_mn = {dBr_mn}  =>  |delta_Br_mn|^2 = {abs(dBr_mn)**2:.4f}")
print()

def rhs_ham(phi, y, eps):
    """Hamiltonian field-line equations (a=const, a'=0)."""
    r, th = y
    # dr/dphi = eps * 2*Re[(im/r0)*a * exp(i(m*th+n*phi))]
    phase = m_t*th + n_t*phi
    f_r  = 2*(  -m_t*a/r0 * np.sin(phase))   # 2*Re[i*m*a/r0 * exp(i*phase)]
    # dtheta/dphi = iota(r)   (no angular perturbation since a'=0)
    f_th = iota0 + shear*r
    return [eps*f_r, f_th]

# Measure mean torus position via theta0-averaged Poincare map
N_rev = 300
N_th0 = 80
theta0_arr = np.linspace(0, 2*np.pi, N_th0, endpoint=False)

print(f"{'eps':>8}  {'<r>_torus':>14}  {'formula':>14}  {'ratio':>8}")
for eps in [0.1, 0.05, 0.02, 0.01, 0.005]:
    all_r = []
    for th0 in theta0_arr:
        sol = solve_ivp(lambda p,y: rhs_ham(p,y,eps), [0, N_rev*2*np.pi],
                        [0.0, th0], method='DOP853', rtol=1e-11, atol=1e-13,
                        t_eval=2*np.pi*np.arange(1, N_rev+1))
        all_r.extend(sol.y[0].tolist())
    mean_r = np.mean(all_r)

    # Paper formula: <delta_r> = -|dBr_mn|^2 / (Bphi * alpha * shear * (2pi)^2)
    # one-sided: factor 2 for real field
    dr_formula = -2*abs(dBr_mn*eps)**2 / (Bphi * alpha * shear * (2*np.pi)**2)
    ratio = mean_r/dr_formula if abs(dr_formula) > 1e-15 else float('nan')
    print(f"{eps:>8.4f}  {mean_r:>14.4e}  {dr_formula:>14.4e}  {ratio:>8.3f}")
