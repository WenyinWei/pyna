"""
Step-by-step analytic vs ODE check for delta_iota derivation.
Focus: single mode, no shear first (verify r1 formula), then add shear.
"""
import numpy as np
from scipy.integrate import quad, solve_ivp

iota0 = (np.sqrt(5)-1)/2
shear = -0.5
m_t, n_t = 1, 0          # single mode, n=0 first (simplest case)
c_t = 0.5 + 0.2j
alpha = m_t*iota0 + n_t  # = iota0 ≈ 0.618

print(f"Single mode (m={m_t},n={n_t}), alpha={alpha:.6f}, c={c_t}")
print()

# ── Step 1: verify r1 formula analytically vs ODE ───────────────────────────
print("STEP 1: r1(phi) - verify one-revolution integral")
# Analytic: integral_0^{2pi} r1(phi) dphi
#   = integral_0^{2pi} integral_0^phi 2Re[c*exp(i*alpha*phi')] dphi' dphi
#   = integral_0^{2pi} 2Re[c*(exp(i*alpha*phi)-1)/(i*alpha)] dphi
#   = 2Re[ c*(E-1)/(i*alpha)^2 - c*2pi/(i*alpha) ]  where E=exp(2pi*i*alpha)
E = np.exp(2j*np.pi*alpha)
int_r1_ana = 2*np.real(c_t * (E-1)/(1j*alpha)**2 - c_t*2*np.pi/(1j*alpha))
int_r1_num, _ = quad(
    lambda phi: 2*np.real(c_t * (np.exp(1j*alpha*phi)-1)/(1j*alpha)),
    0, 2*np.pi)
print(f"  int_0^2pi r1 dphi:  analytic={int_r1_ana:.8f}  numeric={int_r1_num:.8f}")

# ── Step 2: compute I_mn = int <(df/dtheta)*R1>_theta0 dphi ─────────────────
print()
print("STEP 2: I_mn = integral_0^{2pi} <(df/dtheta)*R1>_theta0 dphi")

def integrand(phi):
    Ep = np.exp(1j*alpha*phi)
    bracket = (1-Ep)/(-alpha**2) + phi*Ep/(1j*alpha)
    return 2*np.real(1j*m_t * abs(c_t)**2 * bracket)

I_num, _ = quad(integrand, 0, 2*np.pi)
# Analytic formula:
T1 = (2*np.pi - (E-1)/(1j*alpha)) / (-alpha**2)
T2 = (1/(1j*alpha)) * (2*np.pi*E/(1j*alpha) - (E-1)/(1j*alpha)**2)
I_ana = 2*np.real(1j*m_t * abs(c_t)**2 * (T1 + T2))
print(f"  I_mn: numeric={I_num:.8f}  analytic={I_ana:.8f}")

# ── Step 3: compute delta_iota and <delta_r> ────────────────────────────────
print()
print("STEP 3: delta_iota = (eps^2 * shear / 2pi) * I_mn")
for eps in [0.1, 0.05, 0.02, 0.01, 0.005]:
    di_ana = (eps**2 * shear / (2*np.pi)) * I_ana
    dr_ana = -di_ana / shear   # = -delta_iota / iota'

    # Paper formula (one-sided, for real field use 2x):
    di_pap = 2*abs(c_t)**2 / alpha / (2*np.pi)**2 * eps**2  # factor Bphi^2 if c=dBr/Bphi
    # Wait: in our notation c_t = (dBr)_mn / Bphi, so (dBr)_mn = c_t * Bphi = c_t (Bphi=1 here)
    # Paper: delta_iota = sum |delta_Br_mn|^2 / (alpha * Bphi^2 * (2pi)^2)
    #                   = |c_t|^2 * Bphi^2 / (alpha * Bphi^2 * (2pi)^2)  one-sided
    # With 2x for real field: 2*|c_t|^2 / (alpha * (2pi)^2)
    dr_pap = -di_pap / shear

    print(f"  eps={eps:.4f}: di_ana={di_ana:.4e}  di_pap={di_pap:.4e}  "
          f"dr_ana={dr_ana:.4e}  dr_pap={dr_pap:.4e}  ratio={di_ana/di_pap:.4f}")

print()
print("STEP 4: ODE validation of delta_iota (500 revs, 60 theta0 values)")
N_rev = 500
N_th0 = 60
theta0_arr = np.linspace(0, 2*np.pi, N_th0, endpoint=False)
for eps in [0.05, 0.02, 0.01, 0.005]:
    rns = []
    for th0 in theta0_arr:
        sol = solve_ivp(
            lambda p,y,e=eps,c=c_t,m=m_t,n=n_t,a=alpha: [
                e*2*(c.real*np.cos(m*y[1]+n*p) - c.imag*np.sin(m*y[1]+n*p)),
                iota0 + shear*y[0]],
            [0, N_rev*2*np.pi], [0.0, th0],
            method='DOP853', rtol=1e-12, atol=1e-14)
        rns.append(sol.y[1,-1]/(N_rev*2*np.pi))
    di_ode = np.mean(rns) - iota0
    di_ana_e = (eps**2 * shear / (2*np.pi)) * I_ana
    di_pap_e = 2*abs(c_t)**2/alpha/(2*np.pi)**2 * eps**2
    print(f"  eps={eps:.4f}: di_ODE={di_ode:.4e}  di_ana={di_ana_e:.4e}  di_pap={di_pap_e:.4e}"
          f"  ODE/ana={di_ode/di_ana_e:.4f}  ODE/pap={di_ode/di_pap_e:.4f}")
