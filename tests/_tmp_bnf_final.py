"""
Clean comparison of three formulas for <delta_r> in Hamiltonian system.
No long ODE needed -- only ~50 revolutions to check scaling.
"""
import numpy as np
from scipy.integrate import solve_ivp

iota0 = (np.sqrt(5)-1)/2  # golden ratio
shear = -0.5
a0 = 1.0

print("System: H = psi(r) + eps*a0*cos(m*theta+n*phi)")
print(f"iota0={iota0:.6f}, shear={shear}, a0={a0}")
print()

for m_t, n_t in [(1, 0), (2, 1), (1, -1)]:
    alpha = m_t*iota0 + n_t
    print(f"Mode (m={m_t},n={n_t}), alpha={alpha:.6f}")

    # ---- Birkhoff Normal Form (correct for this Hamiltonian) ----
    # H = psi(r) + eps*V,  V = a0*cos(m*theta+n*phi)
    # Homological eq: {H0, chi} = -V  =>  chi = -a0*sin/(m*iota(r))
    # H_eff2 = (eps^2/2)*{V,chi}
    # {V,chi} = dV/dtheta * dchi/dr = (-m*a0*sin) * (a0*shear*sin/(m*iota^2))
    #         = -a0^2*shear*sin^2/iota^2
    # <{V,chi}> = -a0^2*shear/(2*iota^2)
    # H_eff2 = -(eps^2)*a0^2*shear/(4*iota^2)
    #
    # delta_iota = d(H_eff2)/dr = -(eps^2)*a0^2*shear/4 * d(1/iota^2)/dr
    #            = -(eps^2)*a0^2*shear/4 * (-2*shear/iota^3)
    #            = +(eps^2)*a0^2*shear^2/(2*iota^3)   [POSITIVE since shear^2>0]
    # BUT: iota(r) = iota0 + shear*r, so at non-resonant mode (m,n):
    # alpha(r) = m*iota(r)+n = alpha0 + m*shear*r
    # The resonance denominator alpha also shifts with r.
    # For the general (m,n) mode we use alpha instead of m*iota:
    # <{V,chi}> = -a0^2*shear*m^2/(2*alpha^2)  [chi = -a0*sin/(m*iota) but via alpha]
    # Wait: chi_mn = -a0/(2*i*alpha) and delta_iota involves partial_r alpha = m*shear
    # Full BNF:
    # d(1/alpha^2)/dr = -2*m*shear/alpha^3
    # H_eff2 = (eps^2/2) * m^2*a0^2*(-shear)/(2*alpha^2) ... let me just use the Poisson bracket

    # {V,chi} = dV/dtheta * dchi/dr
    # V = a0*cos(m*theta+n*phi),  chi = -a0*sin(m*theta+n*phi)/(m*alpha(r)/m)
    #   = -a0*sin(m*theta+n*phi)/alpha(r)   [since chi solves: alpha*dchi/dtheta = -V]
    # dchi/dr = a0*sin(m*theta+n*phi)*m*shear/alpha(r)^2
    # dV/dtheta = -m*a0*sin(m*theta+n*phi)
    # {V,chi} = (-m*a0*sin) * (a0*m*shear*sin/alpha^2) = -m^2*a0^2*shear*sin^2/alpha^2
    # <{V,chi}> = -m^2*a0^2*shear/(2*alpha^2)   [using <sin^2>=1/2]
    # H_eff2 = (eps^2/2) * (-m^2*a0^2*shear/(2*alpha^2)) = -eps^2*m^2*a0^2*shear/(4*alpha^2)
    # delta_iota = d(H_eff2)/dr|_0 = -eps^2*m^2*a0^2*shear/4 * d(1/alpha^2)/dr
    #            = -eps^2*m^2*a0^2*shear/4 * (-2*m*shear/alpha^3)
    #            = +eps^2*m^3*a0^2*shear^2/(2*alpha^3)
    # <delta_r> = -delta_iota/shear = -eps^2*m^3*a0^2*shear/(2*alpha^3)  > 0 when shear<0

    dr_bnf = -m_t**3 * a0**2 * shear / (2 * alpha**3)   # per eps^2, positive!
    delta_iota_bnf = m_t**3 * a0**2 * shear**2 / (2 * alpha**3)  # per eps^2, positive

    # ---- Paper Eq.(5.1) ----
    # c_mn = -i*m*a0/2  (from 2Re[c*exp(i(...))] matching the Hamiltonian field dr/dphi)
    c_mn = -1j * m_t * a0 / 2
    # <delta_r> = -2|c_mn|^2/(alpha*shear*(2pi)^2)  (one-sided * 2 for real field, Bphi=1)
    dr_paper = -2 * abs(c_mn)**2 / (alpha * shear * (2*np.pi)**2)

    print(f"  BNF:   <dr>/eps^2 = {dr_bnf:+.6f}")
    print(f"  Paper: <dr>/eps^2 = {dr_paper:+.6f}")
    r = dr_bnf/dr_paper
    print(f"  Ratio BNF/Paper = {r:.4f}")
    expected_ratio = m_t * (2*np.pi)**2 * shear**2 / alpha**2
    print(f"    [= m*(2pi)^2*shear^2/alpha^2 = {expected_ratio:.4f}]")

    # ---- Quick ODE check: convergence of <r>_Poincare with eps ----
    # Use only N_rev=80, N_th0=20 for speed
    N_rev, N_th0 = 80, 20
    th0_arr = np.linspace(0, 2*np.pi, N_th0, endpoint=False)
    eps_list = [0.04, 0.02, 0.01]
    means = []
    for eps in eps_list:
        all_r = []
        for th0 in th0_arr:
            sol = solve_ivp(
                lambda p, y, e=eps, m=m_t, n=n_t: [
                    e * m * a0 * np.sin(m*y[1]+n*p),
                    iota0 + shear*y[0]],
                [0, N_rev*2*np.pi], [0.0, th0],
                method='DOP853', rtol=1e-12, atol=1e-14,
                t_eval=2*np.pi*np.arange(1, N_rev+1))
            all_r.extend(sol.y[0].tolist())
        means.append(np.mean(all_r))

    slope = np.polyfit(np.log(eps_list), np.log(np.abs(means)), 1)[0]
    coeff = np.mean([m/e**2 for m,e in zip(means, eps_list)])
    print(f"  ODE:   <dr>/eps^2 ≈ {coeff:+.4f}  (slope={slope:.2f}, expect 2)")
    print(f"  ODE/BNF = {coeff/dr_bnf:.4f}")
    print()
