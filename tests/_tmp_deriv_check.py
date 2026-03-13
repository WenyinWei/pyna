"""
Analytic derivation of delta_iota and <delta_r> to O(eps^2).

We compare the analytic result with ODE and with the paper's Eq.(5.1).
"""
import numpy as np
from scipy.integrate import quad, dblquad, solve_ivp

IOTA0 = (np.sqrt(5)-1)/2
SHEAR = -0.5
BPHI  = 5.0
MODES = [(1,0),(2,1),(1,-1)]
dBr_F = [0.5+0.2j, 0.3-0.15j, -0.4+0.1j]

# ── Analytic computation of delta_iota ──────────────────────────────────────
# System: dr/dphi = eps*f(theta,phi),  dtheta/dphi = iota0 + shear*r
#
# Second-order rotation number shift at r_init=0, averaged over theta0:
#   delta_iota = (eps^2 * shear / (2pi)) * 
#                integral_0^{2pi} <(df/dtheta)(theta0+iota0*phi, phi) * R1(phi)>_theta0 dphi
#
# where R1(phi) = integral_0^phi r1(phi') dphi'
#       r1(phi) = integral_0^phi f(theta0+iota0*phi', phi') dphi'
#
# For single mode (m,n) with c_mn:
#   f_mn  = 2 Re[c * exp(i(m*theta0 + alpha*phi))]         alpha = m*iota0 + n
#   r1_mn = 2 Re[c * exp(im*theta0) * (exp(i*alpha*phi)-1)/(i*alpha)]
#   R1_mn = 2 Re[c * exp(im*theta0) * ((exp(i*alpha*phi)-1)/(i*alpha)^2 - phi/(i*alpha))]
#   (df/dtheta)_mn = 2 Re[im*c * exp(i(m*theta0 + alpha*phi))]
#
# After theta0-average (survives only when exponents cancel, i.e. same (m,n)):
#   <(df/dtheta)_mn * R1_mn>_theta0
#   = 2 Re[ im*c * c.conj * exp(i*alpha*phi) * ((exp(-i*alpha*phi)-1)/(-i*alpha)^2 - phi/(-i*alpha)) ]
#   = 2 Re[ im*|c|^2 * exp(i*alpha*phi) * ((exp(-i*alpha*phi)-1)/(-alpha^2) + phi/(i*alpha)) ]
#   = 2 Re[ im*|c|^2 * ((1 - exp(i*alpha*phi))/(-alpha^2) + phi*exp(i*alpha*phi)/(i*alpha)) ]

# Integrate from 0 to 2pi:
#   integral_0^{2pi} <(df/dtheta)*R1>_theta0 dphi
#   = 2 Re[ im*|c|^2 * integral_0^{2pi} ((1 - exp(i*alpha*phi))/(-alpha^2) + phi*exp(i*alpha*phi)/(i*alpha)) dphi ]
#
# Term 1: integral_0^{2pi} (1 - exp(i*alpha*phi))/(-alpha^2) dphi
#         = (1/(-alpha^2)) * [2pi - (exp(2pi*i*alpha)-1)/(i*alpha)]
#
# Term 2: integral_0^{2pi} phi * exp(i*alpha*phi) / (i*alpha) dphi
#         = (1/(i*alpha)) * [phi*exp(i*alpha*phi)/(i*alpha) - exp(i*alpha*phi)/(i*alpha)^2]_0^{2pi}
#         = (1/(i*alpha)) * [2pi*exp(2pi*i*alpha)/(i*alpha) - (exp(2pi*i*alpha)-1)/(i*alpha)^2]
#
# Combining:
#   I = integral = 2Re[ im|c|^2 * ( (2pi - E/(i*alpha))/(-alpha^2)
#                                   + (1/(i*alpha))*(2pi*E/(i*alpha) - (E-1)/(i*alpha)^2) ) ]
#   where E = exp(2pi*i*alpha)

def compute_delta_iota_analytic(modes, coeffs, iota0, shear, eps):
    total = 0.0
    for (m,n), c in zip(modes, coeffs):
        alpha = m*iota0 + n
        if abs(alpha) < 1e-10:
            continue   # resonant, skip
        E = np.exp(2j*np.pi*alpha)
        c2 = abs(c)**2
        term1 = (2*np.pi - (E-1)/(1j*alpha)) / (-alpha**2)
        term2 = (1/(1j*alpha)) * (2*np.pi*E/(1j*alpha) - (E-1)/(1j*alpha)**2)
        I_mn = 2*np.real(1j*m * c2 * (term1 + term2))
        total += I_mn
    return (eps**2 * shear / (2*np.pi)) * total

print("Analytic delta_iota computation:")
eps_t = 0.01
di_ana = compute_delta_iota_analytic(MODES, dBr_F, IOTA0, SHEAR, eps_t)
print(f"  Analytic delta_iota = {di_ana:.6e}")

# Paper formula: delta_iota = sum |dBr_mn|^2 / ((m*iota+n) * Bphi^2) / (2pi)^2
# With 2x one-sided for real field:
total_paper = sum(2*abs(c*BPHI*eps_t)**2/(m*IOTA0+n)
                  for (m,n),c in zip(MODES,dBr_F))
di_paper = total_paper / (2*np.pi)**2
print(f"  Paper formula delta_iota = {di_paper:.6e}")

dr_ana   = -di_ana   / SHEAR
dr_paper = -di_paper / SHEAR
print(f"\n  <delta_r> analytic = {dr_ana:.6e}")
print(f"  <delta_r> paper    = {dr_paper:.6e}")
print(f"  Ratio analytic/paper = {dr_ana/dr_paper:.4f}")
print()

# Also compute numerically via double integral for verification
print("Numerical double integral (cross-check):")
for (m,n), c in zip(MODES, dBr_F):
    alpha = m*IOTA0 + n
    if abs(alpha) < 1e-10: continue
    # integrand: im*|c|^2 * ((1-exp(i*a*phi))/(-a^2) + phi*exp(i*a*phi)/(i*a))
    def integrand_real(phi):
        E = np.exp(1j*alpha*phi)
        bracket = (1-E)/(-alpha**2) + phi*E/(1j*alpha)
        return np.real(1j*m*abs(c)**2*bracket) * 2
    I_num, _ = quad(integrand_real, 0, 2*np.pi)
    E = np.exp(2j*np.pi*alpha)
    term1 = (2*np.pi - (E-1)/(1j*alpha))/(-alpha**2)
    term2 = (1/(1j*alpha))*(2*np.pi*E/(1j*alpha) - (E-1)/(1j*alpha)**2)
    I_ana = 2*np.real(1j*m*abs(c)**2*(term1+term2))
    print(f"  (m={m},n={n}) alpha={alpha:.4f}: I_num={I_num:.6f}  I_ana={I_ana:.6f}")

print()
# Now validate delta_iota against ODE (short orbit, many theta0 values)
# iota_eff = theta_final / (2*pi*N) - iota0
# average over many theta0 to get the torus-mean rotation number
print("ODE delta_iota verification (N_rev=500, N_theta0=80):")
N_rev, N_th0 = 500, 80
theta0_arr = np.linspace(0, 2*np.pi, N_th0, endpoint=False)
for eps_t in [0.02, 0.01, 0.005, 0.002]:
    rn_list = []
    for th0 in theta0_arr:
        sol = solve_ivp(
            lambda p,y: [sum(2*(c.real*np.cos(m*y[1]+n*p)-c.imag*np.sin(m*y[1]+n*p))
                            for (m,n),c in zip(MODES,dBr_F))*eps_t,
                         IOTA0+SHEAR*y[0]],
            [0, N_rev*2*np.pi], [0.0, th0],
            method='DOP853', rtol=1e-11, atol=1e-13)
        rn_list.append(sol.y[1,-1]/(N_rev*2*np.pi))
    di_ode = np.mean(rn_list) - IOTA0
    di_ana = compute_delta_iota_analytic(MODES, dBr_F, IOTA0, SHEAR, eps_t)
    di_pap = sum(2*abs(c*BPHI*eps_t)**2/(m*IOTA0+n)
                 for (m,n),c in zip(MODES,dBr_F)) / (2*np.pi)**2
    print(f"  eps={eps_t:.4f}: di_ODE={di_ode:.4e}  di_analytic={di_ana:.4e}  di_paper={di_pap:.4e}"
          f"  ODE/analytic={di_ode/di_ana:.3f}  ODE/paper={di_ode/di_pap:.3f}")
