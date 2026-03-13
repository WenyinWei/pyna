"""
Clean re-derivation and verification of <delta_r> formula.

Hamiltonian:  H = psi(r) + eps*V,   V = a0*cos(m*theta+n*phi)
              psi(r) = iota0*r + shear/2*r^2
Equations:   dr/dphi = m*a0*eps*sin(m*theta+n*phi)
             dtheta/dphi = iota0 + shear*r

Birkhoff Normal Form (analytic derivation):
  Step 1. Homological equation:  L0*chi = -V,  L0 = alpha*d/dtheta + d/dphi
          Solution: chi = -a0*sin(m*theta+n*phi) / alpha(r),  alpha(r)=m*iota(r)+n
  Step 2. Poisson bracket {V, chi}:
          {V,chi} = dV/dtheta * dchi/dr - dV/dr * dchi/dtheta
          Since V does not depend on r:  dV/dr = 0
          dV/dtheta = -m*a0*sin(m*theta+n*phi)
          dchi/dr   = -a0*sin * d/dr(-1/alpha(r)) = a0*sin * m*shear / alpha^2
          {V,chi}   = (-m*a0*sin) * (a0*m*shear*sin/alpha^2)
                    = -m^2*a0^2*shear*sin^2 / alpha^2
  Step 3. Second-order effective Hamiltonian:
          <H_eff2> = (eps^2/2)*<{V,chi}> = -(eps^2/2)*m^2*a0^2*shear/(2*alpha(r)^2)
                   = -eps^2*m^2*a0^2*shear / (4*alpha(r)^2)
  Step 4. Effective rotation number:
          iota_eff(r) = d<H_eff2>/dr + iota0 + shear*r
          d<H_eff2>/dr = -eps^2*m^2*a0^2*shear/4 * d(1/alpha^2)/dr
                       = -eps^2*m^2*a0^2*shear/4 * (-2*m*shear/alpha^3)
                       = +eps^2*m^3*a0^2*shear^2 / (2*alpha^3)
          delta_iota = eps^2*m^3*a0^2*shear^2 / (2*alpha^3)   [always >= 0]
  Step 5. Torus shift:
          <delta_r> = -delta_iota / shear
                    = -eps^2*m^3*a0^2*shear / (2*alpha^3)

In terms of delta_Br Fourier coefficient (2Re-convention):
  dr/dphi = eps * 2*Re[-i*m*a0/2 * exp(i(m*theta+n*phi))]
  ==> (delta_Br)_mn = -i*m*a0/2  (one-sided complex amplitude)
  |( delta_Br)_mn|^2 = m^2*a0^2/4

Substituting: <delta_r> = -eps^2 * m * 4|(delta_Br)_mn|^2 * shear/(2*alpha^3)  [one-sided]
  For real field (both (m,n) and (-m,-n)):
  The (-m,-n) mode contributes:  (-m)*|(dBr)_{-m,-n}|^2/(-alpha)^3
                               = (-m)*|(dBr)_mn|^2/(-alpha^3)
                               = m*|(dBr)_mn|^2/alpha^3  [SAME as (m,n) term]
  => Total two-sided: 2 * one-sided contribution
  => <delta_r> = -4*eps^2 * sum_mn m*|(delta_Br)_mn|^2 * shear / (2*alpha^3)  [two-sided]
               = -2*eps^2 * sum_mn m*|(delta_Br)_mn|^2 * shear / alpha^3      [two-sided]
               = +2*eps^2 * sum_mn m*|(delta_Br)_mn|^2 * |iota'| / alpha^3    [for shear<0]

CORRECT FORMULA:
  <delta_r> = -2*iota' * sum_{mn} m * |(delta_Br)_mn|^2 / (Bphi^2 * (m*iota+n)^3)  [two-sided]
or equivalently (one-sided half-space sum * 2):
  <delta_r> = -4*iota' * sum_{(m,n) in half-space} m * |(delta_Br)_mn|^2 / (Bphi^2 * (m*iota+n)^3)
"""
import numpy as np
from scipy.integrate import solve_ivp

iota0 = (np.sqrt(5)-1)/2
shear = -0.5   # = iota'
a0    = 1.0
Bphi  = 1.0

def correct_formula(m, n, a0_val, eps, iota0_val, shear_val, Bphi_val=1.0):
    """<delta_r> from BNF: two-sided one-mode formula."""
    alpha = m*iota0_val + n
    dBr_mn = -1j * m * a0_val / 2   # one-sided complex amplitude
    # Two-sided: multiply one-sided contribution by 2
    # one-sided: -2*m*|dBr|^2*shear/alpha^3
    return -2 * m * abs(dBr_mn*eps)**2 * shear_val / (Bphi_val**2 * alpha**3)

def paper_formula(m, n, a0_val, eps, iota0_val, shear_val, Bphi_val=1.0):
    """Paper Eq.(5.1): -2|dBr|^2 / (alpha*shear*(2pi)^2)."""
    alpha = m*iota0_val + n
    dBr_mn = -1j * m * a0_val / 2
    return -2 * abs(dBr_mn*eps)**2 / (Bphi_val**2 * alpha * shear_val * (2*np.pi)**2)

print("Verifying corrected BNF formula vs ODE Poincare mean")
print(f"iota0={iota0:.6f}, shear={shear}, a0={a0}")
print()

N_rev = 150
N_th0 = 80

header = f"{'mode':>10}  {'eps':>6}  {'ODE <r>':>12}  {'BNF_new':>12}  {'paper':>12}  {'ODE/BNF':>8}  {'ODE/paper':>10}"
print(header)
print('-'*len(header))

for m_t, n_t in [(1, 0), (2, 1), (1, -1)]:
    alpha = m_t*iota0 + n_t
    for eps in [0.04, 0.02, 0.01]:
        theta0_arr = np.linspace(0, 2*np.pi, N_th0, endpoint=False)
        all_r = []
        for th0 in theta0_arr:
            sol = solve_ivp(
                lambda p, y, _m=m_t, _n=n_t, _e=eps: [
                    _e * _m * a0 * np.sin(_m*y[1] + _n*p),
                    iota0 + shear*y[0]],
                [0, N_rev*2*np.pi], [0.0, th0],
                method='DOP853', rtol=1e-12, atol=1e-14,
                t_eval=2*np.pi*np.arange(1, N_rev+1))
            all_r.extend(sol.y[0].tolist())
        mean_r = np.mean(all_r)
        bnf    = correct_formula(m_t, n_t, a0, eps, iota0, shear)
        pap    = paper_formula(m_t, n_t, a0, eps, iota0, shear)
        r1 = mean_r/bnf if abs(bnf) > 1e-20 else float('nan')
        r2 = mean_r/pap if abs(pap) > 1e-20 else float('nan')
        print(f"({m_t:+d},{n_t:+d})  {eps:>6.3f}  {mean_r:>12.4e}  {bnf:>12.4e}  {pap:>12.4e}  {r1:>8.3f}  {r2:>10.3f}")
    print()
