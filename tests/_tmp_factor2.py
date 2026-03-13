"""
Final verification: why is ODE/BNF = 2?

The BNF gives the r_bar shift (in transformed coordinates).
The physical r = r_bar + eps * delta_r1(theta, phi) where delta_r1 is the
first-order canonical transformation.

The mean of r over the torus = <r>_torus = r_bar* + eps*<delta_r1>_torus

The first-order transformation is:
  r = r_bar - eps * dchi/dtheta   (from generating function chi)
  chi = -a0*sin(m*theta+n*phi)/alpha

  dchi/dtheta = -m*a0*cos(m*theta+n*phi)/alpha

  <r>_torus = r_bar* + eps * <m*a0*cos(m*theta+n*phi)/alpha>_torus

On the unperturbed torus (r_bar=0):
  theta(phi) = theta0 + iota0*phi  =>  cos(m*(theta0+iota0*phi)+n*phi) oscillates
  Mean over theta0: <cos(m*theta0+alpha*phi)>_theta0 = 0

So eps*<delta_r1>_theta0 = 0, and <r>_torus = r_bar*.

But our ODE averages OVER TIME (phi), not over theta0!
The time average of r along the r=0 orbit in physical coords:

r(phi) = r_bar(phi) + eps*dchi/dtheta = 0 + eps*m*a0*cos(m*theta+n*phi)/alpha
       (r_bar=0 for unperturbed, theta~theta0+iota0*phi)

<r>_phi = eps*m*a0/<cos(...)>_phi = 0   (non-resonant, oscillatory)

So neither theta0-avg nor phi-avg of r from r=0 should be nonzero!
The BNF shift r_bar* IS the physical r shift.
ODE gives 2*BNF -- there must be a factor 2 error in the BNF derivation.

Let me recheck: is the Poisson bracket factor 1/2 or 1?
"""
import numpy as np
from scipy.integrate import solve_ivp, quad

iota0 = (np.sqrt(5)-1)/2
shear = -0.5
a0    = 1.0
m_t, n_t = 1, 0
alpha = m_t*iota0 + n_t
eps = 0.01

print("Checking BNF factor: is it eps^2/2 * {V,chi} or eps^2 * {V,chi} ?")
print()

# The second-order effective Hamiltonian from Lie transform theory:
# If T = exp(eps*L_chi) where L_chi = {chi, .}
# H_eff = T*H = H0 + eps*{chi,H0} + eps*V + eps^2/2*{chi,{chi,H0}} + eps^2*{chi,V} + ...
# The homological equation {chi,H0} = -V eliminates first order.
# Remaining: H_eff2 = eps^2/2*{chi,{chi,H0}} + eps^2*{chi,V}
#           = eps^2/2*{chi,-V} + eps^2*{chi,V}
#           = -eps^2/2*{chi,V} + eps^2*{chi,V}
#           = +eps^2/2*{chi,V}
#           = +eps^2/2*{V,chi} * (-1)   (antisymmetry of Poisson bracket)
#           = -eps^2/2*{V,chi}
#
# So H_eff2 = -eps^2/2 * <{V,chi}> (taking the mean)
# <{V,chi}> = -m^2*a0^2*shear/(2*alpha^2)
# H_eff2 = -eps^2/2 * (-m^2*a0^2*shear/(2*alpha^2)) = eps^2*m^2*a0^2*shear/(4*alpha^2)
#
# But wait, I used {chi, H0} = +V in the equation L_chi(H0) = -V.
# The correct Lie transform: L_chi(f) = {chi, f} (using chi as generator).
# Homological: {chi, H0} = -V  =>  {V,chi} = V - (-V) ??  No.
# {chi, H0} = -V means the FLOW of H0 acting on chi gives -V.
# Actually the standard form: L_0*chi = V where L_0 = alpha*d/dtheta
# so that the first-order term eps*{chi, H0} + eps*V = 0 (they cancel).
# {chi, H0} = -chi_theta * H0_r = ... this needs careful index.
# 
# Let me just MEASURE the second-order contribution numerically.

# Second-order rotation number shift from ODE:
theta0_arr = np.linspace(0, 2*np.pi, 200, endpoint=False)
for eps_t in [0.04, 0.02, 0.01]:
    rn_list = []
    for th0 in theta0_arr:
        sol = solve_ivp(
            lambda p,y,e=eps_t: [e*m_t*a0*np.sin(m_t*y[1]+n_t*p), iota0+shear*y[0]],
            [0, 100*2*np.pi], [0.0, th0],
            method='DOP853', rtol=1e-12, atol=1e-14)
        # Correct rotation number: subtract initial theta0
        rn_list.append((sol.y[1,-1] - th0) / (100*2*np.pi))
    di_ode = np.mean(rn_list) - iota0
    # From N2 (Poincare r mean):
    r_list = []
    for th0 in theta0_arr:
        sol = solve_ivp(
            lambda p,y,e=eps_t: [e*m_t*a0*np.sin(m_t*y[1]+n_t*p), iota0+shear*y[0]],
            [0, 50*2*np.pi], [0.0, th0],
            method='DOP853', rtol=1e-12, atol=1e-14,
            t_eval=2*np.pi*np.arange(1,51))
        r_list.extend(sol.y[0].tolist())
    mean_r = np.mean(r_list)
    di_from_r = shear * mean_r  # delta_iota ~ shear*<r>  (linear approx)
    # BNF prediction (with eps^2/2 factor, hence 1 not 2):
    di_bnf_half = m_t**3 * a0**2 * shear**2 / (2 * alpha**3) * eps_t**2
    di_bnf_full = m_t**3 * a0**2 * shear**2 / alpha**3 * eps_t**2   # without /2
    print(f"eps={eps_t:.3f}: di_ODE={di_ode:.3e}  di_from_r={di_from_r:.3e}  "
          f"bnf/2={di_bnf_half:.3e}  bnf_full={di_bnf_full:.3e}  "
          f"ODE/bnf_full={di_ode/di_bnf_full:.3f}")
