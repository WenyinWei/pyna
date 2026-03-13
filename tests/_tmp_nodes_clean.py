"""
Quick node-by-node verification to pinpoint where Eq.(5.1) breaks down.

Model: H(r,theta,phi) = psi(r) + eps*V,  V = a0*cos(m*theta+n*phi)
       psi(r) = iota0*r + shear/2*r^2
       => dr/dphi = -dH/dtheta = eps*m*a0*sin(...)
          dtheta/dphi = dH/dr = iota(r) = iota0 + shear*r

Nodes to verify:
  N1. One revolution map: dr after one period  (analytic vs ODE)
  N2. Mean radial displacement after one rev, averaged over theta0
  N3. Second-order delta_iota (from rotation number)
  N4. Inferred <delta_r> = -delta_iota/iota'  vs  Paper formula
"""
import numpy as np
from scipy.integrate import quad, solve_ivp

iota0 = (np.sqrt(5)-1)/2
shear = -0.5
a0    = 1.0

def run_case(m, n):
    alpha = m*iota0 + n
    eps   = 0.01   # small so higher-order terms are negligible
    N_th0 = 200
    theta0_arr = np.linspace(0, 2*np.pi, N_th0, endpoint=False)

    print(f"\n{'='*60}")
    print(f"Mode (m={m}, n={n}),  alpha = {alpha:.6f}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # N1: mean of dr = r(2pi) - r(0), averaged over theta0
    #     Theory (1st order, theta0-avg): <dr>_theta0 = 0 for non-resonant
    #     if the formula is 2Re[c*exp(im*theta0)] * integral, which averages to 0
    # ------------------------------------------------------------------
    dr_list = []
    for th0 in theta0_arr:
        sol = solve_ivp(
            lambda p, y, _m=m, _n=n: [
                eps * _m * a0 * np.sin(_m*y[1] + _n*p),
                iota0 + shear*y[0]
            ],
            [0, 2*np.pi], [0.0, th0],
            method='DOP853', rtol=1e-13, atol=1e-15
        )
        dr_list.append(sol.y[0,-1] - 0.0)
    mean_dr_1rev = np.mean(dr_list)
    std_dr_1rev  = np.std(dr_list)
    print(f"N1. <dr>_1rev (theta0-avg) = {mean_dr_1rev:.4e}  (std={std_dr_1rev:.4e})")
    print(f"    This should be O(eps^2)={eps**2:.2e}, got {abs(mean_dr_1rev):.2e}")

    # ------------------------------------------------------------------
    # N2: Mean radial position after N_rev, averaged over theta0
    #     = estimate of the torus mean-r
    # ------------------------------------------------------------------
    N_rev = 100
    r_means = []
    for th0 in theta0_arr:
        sol = solve_ivp(
            lambda p, y, _m=m, _n=n: [
                eps * _m * a0 * np.sin(_m*y[1] + _n*p),
                iota0 + shear*y[0]
            ],
            [0, N_rev*2*np.pi], [0.0, th0],
            method='DOP853', rtol=1e-12, atol=1e-14,
            t_eval=2*np.pi*np.arange(1, N_rev+1)
        )
        r_means.append(sol.y[0].mean())
    mean_r_torus = np.mean(r_means)
    print(f"N2. <r>_torus (Poincare, theta0-avg) = {mean_r_torus:.6e}")

    # ------------------------------------------------------------------
    # N3: delta_iota at r=0 from rotation number (theta0-averaged)
    # ------------------------------------------------------------------
    rn_list = []
    for th0 in theta0_arr:
        sol = solve_ivp(
            lambda p, y, _m=m, _n=n: [
                eps * _m * a0 * np.sin(_m*y[1] + _n*p),
                iota0 + shear*y[0]
            ],
            [0, N_rev*2*np.pi], [0.0, th0],
            method='DOP853', rtol=1e-12, atol=1e-14
        )
        rn_list.append(sol.y[1,-1] / (N_rev*2*np.pi))
    delta_iota_ode = np.mean(rn_list) - iota0
    print(f"N3. delta_iota (ODE) = {delta_iota_ode:.6e}  [per eps^0, actual = {delta_iota_ode:.4e}]")
    print(f"    delta_iota/eps^2  = {delta_iota_ode/eps**2:.4f}")

    # delta_iota from BNF:  +m^3*a0^2*shear^2/(2*alpha^3)  per eps^2
    delta_iota_bnf = m**3 * a0**2 * shear**2 / (2 * alpha**3)
    print(f"    BNF prediction:   = {delta_iota_bnf:.4f}")
    print(f"    ratio ODE/BNF     = {(delta_iota_ode/eps**2)/delta_iota_bnf:.4f}")

    # ------------------------------------------------------------------
    # N4: <delta_r> = -delta_iota_ode / shear  vs  Paper formula
    # ------------------------------------------------------------------
    dr_from_di_ode = -delta_iota_ode / shear
    dr_bnf         = -delta_iota_bnf * eps**2 / shear
    # Paper formula (Bphi=1, dBr_mn = -i*m*a0/2, |dBr_mn|^2 = m^2*a0^2/4):
    dr_paper = -2*(m*a0/2)**2 * eps**2 / (alpha * shear * (2*np.pi)**2)
    print(f"N4. <delta_r> = -delta_iota/iota':")
    print(f"    From ODE delta_iota:    {dr_from_di_ode:.6e}")
    print(f"    BNF prediction:         {dr_bnf:.6e}")
    print(f"    Paper formula Eq(5.1):  {dr_paper:.6e}")
    print(f"    N2 Poincare mean:       {mean_r_torus:.6e}")
    print(f"    Ratios: ODE/BNF={dr_from_di_ode/dr_bnf:.3f}  "
          f"ODE/Paper={dr_from_di_ode/dr_paper:.3f}  "
          f"N2/BNF={(mean_r_torus/dr_bnf):.3f}")

for m, n in [(1, 0), (2, 1), (3, -1)]:
    run_case(m, n)
