"""
Analytic derivation: compare 3 formulas for the iota0-torus shift.

System: Hamiltonian H = psi(r) + eps*a0*cos(m*theta+n*phi)
  psi(r) = iota0*r + shear/2*r^2  =>  iota(r) = iota0 + shear*r

Hamilton equations:
  dtheta/dphi =  dH/dr = iota(r)                        [NO angular perturbation since a0=const]
  dr/dphi     = -dH/dtheta = eps*m*a0*sin(m*theta+n*phi)

This satisfies the area-preserving (symplectic) condition automatically.

Birkhoff normal form result (from d/dr of H2 at r=0):
  H2(r) = -m^2*a0^2*iota'/(4*alpha(r)^2)   [see derivation in comments]
  delta_iota = eps^2 * dH2/dr|_{r=0} = eps^2 * m^3*a0^2*iota'^2 / (2*alpha0^3)
  <delta_r> = -delta_iota/iota' = -eps^2 * m^3*a0^2*iota' / (2*alpha0^3)
"""
import numpy as np
from scipy.integrate import solve_ivp

iota0 = (np.sqrt(5)-1)/2
shear = -0.5
a0    = 1.0
Bphi  = 1.0

print("Hamiltonian: H = psi(r) + eps*a0*cos(m*theta+n*phi),  a0 = const")
print(f"iota0={iota0:.6f},  shear={shear},  a0={a0}")
print()

for m_t, n_t in [(1, 0), (2, 1), (1, -1)]:
    alpha = m_t*iota0 + n_t
    print(f"--- Mode (m={m_t}, n={n_t}),  alpha = {alpha:.6f} ---")

    # Formula 1: Birkhoff normal form (correct for this Hamiltonian)
    # <delta_r>_BNF = -m^3*a0^2*shear / (2*alpha^3)
    # (no eps^2 -- we show the coefficient per eps^2)
    dr_bnf_coeff = -m_t**3 * a0**2 * shear / (2 * alpha**3)
    print(f"  Birkhoff NF:   <dr>/eps^2 = {dr_bnf_coeff:.6f}")

    # Formula 2: Paper Eq.(5.1):
    # <delta_r> = -|dBr_mn|^2 / (Bphi^2 * alpha * shear * (2pi)^2) * 2
    # Here dBr_mn = -i*m*a0/2 (one-sided complex amplitude), |dBr_mn|^2 = (m*a0)^2/4
    # Factor 2 for real field (two-sided)
    dBr_mn = -1j*m_t*a0/2
    dr_paper_coeff = -2*abs(dBr_mn)**2 / (Bphi**2 * alpha * shear * (2*np.pi)**2)
    print(f"  Paper Eq.(5.1): <dr>/eps^2 = {dr_paper_coeff:.6f}")
    print(f"  Ratio BNF/paper = {dr_bnf_coeff/dr_paper_coeff:.4f}  "
          f"[theory: m*(2pi)^2*shear^2/alpha^2 = {m_t*(2*np.pi)**2*shear**2/alpha**2:.4f}]")

    # Formula 3: ODE Poincare mean (theta0-averaged)
    N_rev = 200
    N_th0 = 40
    theta0_arr = np.linspace(0, 2*np.pi, N_th0, endpoint=False)
    eps_vals = [0.02, 0.01, 0.005]
    means = []
    for eps in eps_vals:
        all_r = []
        for th0 in theta0_arr:
            sol = solve_ivp(
                lambda p, y, e=eps, m=m_t, n=n_t: [
                    e * m * a0 * np.sin(m*y[1] + n*p),
                    iota0 + shear*y[0]],
                [0, N_rev*2*np.pi], [0.0, th0],
                method='DOP853', rtol=1e-11, atol=1e-13,
                t_eval=2*np.pi*np.arange(1, N_rev+1))
            all_r.extend(sol.y[0].tolist())
        means.append(np.mean(all_r))
    # Fit slope and extract coefficient
    log_e = np.log(eps_vals)
    log_m = np.log(np.abs(means))
    slope = np.polyfit(log_e, log_m, 1)[0]
    coeff_ode = np.mean([m/e**2 for m,e in zip(means, eps_vals)])
    print(f"  ODE (theta0-avg): <dr>/eps^2 = {coeff_ode:.6f}  (slope={slope:.3f})")
    print(f"  ODE/BNF = {coeff_ode/dr_bnf_coeff:.4f}")
    print()
