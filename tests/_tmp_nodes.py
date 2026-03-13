"""
Step-by-step verification of the Birkhoff Normal Form derivation.
Single mode, Hamiltonian system. No long ODE needed.

Key nodes:
  N1: <{V,S}> analytic vs numeric quad
  N2: delta_iota from BNF vs short-orbit rotation number
  N3: mean_r of r=0 orbit vs r* (iota0-torus position) -- are they the same?
"""
import numpy as np
from scipy.integrate import quad, solve_ivp

iota0 = (np.sqrt(5)-1)/2
shear = -0.5
a0    = 1.0
m_t, n_t = 1, 0
alpha0 = m_t*iota0 + n_t   # = iota0

print(f"H = psi(r) + eps*a0*cos(theta),  iota0={iota0:.6f}, shear={shear}, alpha={alpha0:.6f}")
print()

# ── NODE 1: <{V,S}> ─────────────────────────────────────────────────────────
print("="*60)
print("NODE 1: <{V,S}> at r=0")
print("="*60)
# V = a0*cos(m*theta),  S = a0/(m*iota(r))*sin(m*theta)
# {V,S} = dV/dtheta * dS/dr - dV/dr * dS/dtheta
#       = (-m*a0*sin(theta)) * (-a0*m*shear/iota^2 * sin(theta)) - 0
#       = m^2*a0^2*shear/iota^2 * sin^2(theta)
# <{V,S}>_theta = m^2*a0^2*shear / (2*iota0^2)
VSmean_ana = m_t**2 * a0**2 * shear / (2 * iota0**2)
print(f"  Analytic: <{{V,S}}> = m^2*a0^2*shear/(2*iota0^2) = {VSmean_ana:.8f}")
# Numeric check via quad
def VS_integrand(theta):
    dV_dth = -m_t*a0*np.sin(m_t*theta)
    dS_dr  = -a0*m_t*shear/iota0**2 * np.sin(m_t*theta)   # at r=0
    return dV_dth * dS_dr
VS_num, _ = quad(VS_integrand, 0, 2*np.pi)
VS_num /= (2*np.pi)   # average
print(f"  Numeric:  <{{V,S}}> = {VS_num:.8f}")

# ── NODE 2: delta_iota from BNF ─────────────────────────────────────────────
print()
print("="*60)
print("NODE 2: delta_iota from BNF formula vs ODE rotation number")
print("="*60)
# H_eff2(r) = (eps^2/2) * <{V,S}>_r = (eps^2/2) * m^2*a0^2*shear(r)/(2*iota(r)^2)
# Since shear = const: = (eps^2/2) * m^2*a0^2*shear/(2*iota(r)^2)
# delta_iota = d(H_eff2)/dr|_{r=0} = (eps^2/2)*m^2*a0^2*shear/2 * d(1/iota^2)/dr|_0
# d(1/iota^2)/dr = -2*shear/iota^3  (since diota/dr = shear)
# delta_iota = (eps^2/2)*m^2*a0^2*shear/2 * (-2*shear/iota0^3)
#            = -eps^2*m^2*a0^2*shear^2/(2*iota0^3)
# <delta_r>  = -delta_iota/shear = eps^2*m^2*a0^2*shear/(2*iota0^3)
# (NOTE: this is m^2, not m^3 -- let me recheck my earlier derivation)
delta_iota_bnf_coeff = -m_t**2 * a0**2 * shear**2 / (2 * iota0**3)
dr_bnf_coeff = -delta_iota_bnf_coeff / shear   # = m^2*a0^2*shear/(2*iota0^3)
print(f"  BNF:  delta_iota/eps^2 = {delta_iota_bnf_coeff:.6f}")
print(f"  BNF:  <delta_r>/eps^2  = {dr_bnf_coeff:.6f}")
print()
# Short ODE: rotation number at r=0 using just 50 revolutions per theta0, 20 theta0
N_rev, N_th0 = 50, 20
theta0_arr = np.linspace(0, 2*np.pi, N_th0, endpoint=False)
print(f"  ODE (N_rev={N_rev}, N_th0={N_th0}):")
for eps in [0.1, 0.05, 0.02, 0.01]:
    rns = []
    for th0 in theta0_arr:
        sol = solve_ivp(
            lambda p, y, e=eps: [
                e * m_t * a0 * np.sin(m_t*y[1] + n_t*p),
                iota0 + shear*y[0]],
            [0, N_rev*2*np.pi], [0.0, th0],
            method='DOP853', rtol=1e-12, atol=1e-14)
        rns.append(sol.y[1,-1] / (N_rev*2*np.pi))
    di_ode = np.mean(rns) - iota0
    print(f"    eps={eps:.3f}: di_ODE={di_ode:.4e}  di_BNF={delta_iota_bnf_coeff*eps**2:.4e}  "
          f"ratio={di_ode/(delta_iota_bnf_coeff*eps**2):.4f}")

# ── NODE 3: mean_r of r=0 orbit vs r* ───────────────────────────────────────
print()
print("="*60)
print("NODE 3: mean_r of r=0 orbit  vs  r* (iota0-torus position)")
print("="*60)
print("  Are these the same quantity? Let's check:")
eps = 0.05
# (a) mean_r: average r over Poincare sequence starting from many theta0
all_r_meanr = []
for th0 in np.linspace(0, 2*np.pi, 40, endpoint=False):
    sol = solve_ivp(
        lambda p, y: [eps*m_t*a0*np.sin(m_t*y[1]+n_t*p), iota0+shear*y[0]],
        [0, 100*2*np.pi], [0.0, th0],
        method='DOP853', rtol=1e-11, atol=1e-13,
        t_eval=2*np.pi*np.arange(1,101))
    all_r_meanr.extend(sol.y[0].tolist())
mean_r_orbit = np.mean(all_r_meanr)
# (b) r*: bisect to find iota_eff(r*)=iota0
def rn(r_init):
    sol = solve_ivp(
        lambda p, y: [eps*m_t*a0*np.sin(m_t*y[1]+n_t*p), iota0+shear*y[0]],
        [0, 200*2*np.pi], [r_init, 0.0],
        method='DOP853', rtol=1e-11, atol=1e-13)
    return sol.y[1,-1]/(200*2*np.pi)
r_lo, r_hi = -0.2, 0.2
rn_lo, rn_hi = rn(r_lo), rn(r_hi)
for _ in range(40):
    r_mid = 0.5*(r_lo+r_hi)
    rn_mid = rn(r_mid)
    if (rn_mid-iota0)*(rn_lo-iota0) < 0: r_hi=r_mid; rn_hi=rn_mid
    else: r_lo=r_mid; rn_lo=rn_mid
    if abs(r_hi-r_lo)<1e-7: break
r_star = 0.5*(r_lo+r_hi)
dr_bnf = dr_bnf_coeff * eps**2
print(f"  eps={eps}")
print(f"  (a) mean_r of r=0 orbit = {mean_r_orbit:.6e}")
print(f"  (b) r* (iota0 torus)    = {r_star:.6e}")
print(f"  BNF formula <dr>/eps^2  = {dr_bnf:.6e}")
print(f"  Are (a)~(b)? ratio = {mean_r_orbit/r_star:.4f}")
print(f"  r* / BNF = {r_star/dr_bnf:.4f}")
print(f"  mean_r / BNF = {mean_r_orbit/dr_bnf:.4f}")
