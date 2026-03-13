"""
Careful derivation and test of Eq.(5.1): second-order <delta_r>

Starting from field-line ODE:
  dr/dphi  = eps * f(theta, phi)    where f = 2 Re[sum_mn c_mn exp(i(m*theta+n*phi))]
  dtheta/dphi = iota0 + iota' * r

We want the SHIFT of the invariant torus with rotation number iota0.
This is NOT the same as <r(2pi)> from r=0; it is the r-value r* such that
the orbit starting at r* has rotation number iota0.

Second-order perturbation theory for the rotation number:
  iota_eff(r) = iota0 + iota'*r + eps^2 * delta_iota(r)

The invariant torus is at r* where iota_eff(r*) = iota0:
  iota' * r* + eps^2 * delta_iota(r*) ≈ 0  (to leading order in eps^2)
  r* ≈ -delta_iota(0) / iota'    [r*= O(eps^2)]

So we need delta_iota(0) = change in rotation number at r=0 due to perturbation.

From the standard Lie-transform / averaging perturbation theory:
The rotation number correction is:
  delta_iota = <d(delta_theta)/dphi>_avg / (2*pi)
where delta_theta is the first-order angular displacement after one revolution.

Let me compute this explicitly.
"""
import numpy as np
from scipy.integrate import solve_ivp

IOTA0      = (np.sqrt(5)-1)/2
SHEAR      = -0.5   # = iota'
BPHI       = 5.0
MODES      = np.array([(1,0),(2,1),(1,-1)], dtype=int)
dBr_F      = np.array([0.5+0.2j, 0.3-0.15j, -0.4+0.1j])  # (delta_Br)_mn / Bphi

def f_field(theta, phi):
    val = 0.0
    for (m,n), c in zip(MODES, dBr_F):
        ph = m*theta + n*phi
        val += 2*(c.real*np.cos(ph) - c.imag*np.sin(ph))
    return val

# First-order angular displacement after one revolution (r=0 orbit, theta0 fixed):
#   Delta_theta_1(theta0) = eps * integral_0^{2pi} iota0 * f(theta0+iota0*phi, phi) dphi
#   Wait, that's for a pure ANGULAR perturbation. Our system has RADIAL perturbation.
#   The rotation number change comes from:
#   theta_fin = theta0 + integral_0^{2pi} (iota0 + iota'*r(phi)) dphi
#   = theta0 + 2*pi*iota0 + iota' * integral_0^{2pi} r(phi) dphi
#   iota_eff = iota0 + iota' * <r(phi)> [time-average, not Poincare average]
#
# At r_init=0, theta0=0:
#   r(phi) = eps * integral_0^phi f(iota0*phi', phi') dphi' + O(eps^2)
#
# <r(phi)> = eps/(2pi) * integral_0^{2pi} integral_0^phi f(iota0*phi', phi') dphi' dphi
#           + O(eps^2)
#
# Let's compute this analytically for a single mode (m,n):
# f_mn = 2Re[c*exp(i(m*iota0+n)*phi')] = 2Re[c*exp(i*alpha*phi')]  where alpha=m*iota0+n
# 
# r1(phi) = eps * 2Re[c * (exp(i*alpha*phi)-1)/(i*alpha)]
# 
# integral_0^{2pi} r1(phi) dphi = eps * 2Re[c * integral_0^{2pi} (exp(i*alpha*phi)-1)/(i*alpha) dphi]
# = eps * 2Re[c * ((exp(2pi*i*alpha)-1)/(i*alpha)^2 - 2pi/(i*alpha))]
# 
# For non-resonant alpha (alpha not integer):
# exp(2pi*i*alpha)-1 ≠ 0 in general
# But when we average over theta0:
# <r1(phi)>_{theta0} = 0  (the real part of c*exp(im*theta0) averages to zero)
# 
# Wait! We need to average over theta0 first. At theta0:
# r1(phi; theta0) = eps * 2Re[c*exp(im*theta0) * (exp(i*alpha*phi)-1)/(i*alpha)]
# <r1(phi)>_{theta0} = 0  since <exp(im*theta0)>_{theta0} = 0 for m≠0
#
# But r* from bisection was non-zero... so where does the r* = O(eps^2) come from?
#
# The issue: we're starting all orbits at theta0=0, NOT averaging over theta0.
# The ORBIT at (r=0, theta0=0) has a specific rotation number (not averaged over theta0).
# The r* found by bisection is the r-value where an orbit starting at theta0=0 
# has rotation number iota0. This depends on theta0!
#
# The PHYSICAL quantity is the AVERAGE over theta0: the flux surface that 
# on AVERAGE has rotation number iota0. This is what the paper computes.
# 
# Let's verify: the average r* over theta0 should be ~0, and the VARIANCE 
# is what creates the second-order effect.

print("Checking theta0-dependence of r*:")
n_th = 20
theta0_values = np.linspace(0, 2*np.pi, n_th, endpoint=False)
eps_t = 0.02

r_stars = []
for th0_init in theta0_values:
    # Find r such that rotation number = iota0 starting from (r, th0_init)
    def rotation_number(r_init):
        sol = solve_ivp(
            lambda p,y: [eps_t*f_field(y[1],p), IOTA0+SHEAR*y[0]],
            [0, 20*2*np.pi], [r_init, th0_init],
            method='DOP853', rtol=1e-10, atol=1e-12)
        return sol.y[1,-1]/(20*2*np.pi)
    
    # Quick bisect
    r_lo, r_hi = -0.2, 0.2
    rn_lo = rotation_number(r_lo)
    rn_hi = rotation_number(r_hi)
    if (rn_lo - IOTA0)*(rn_hi - IOTA0) > 0:
        r_stars.append(np.nan)
        continue
    for _ in range(30):
        r_mid = 0.5*(r_lo+r_hi)
        rn_mid = rotation_number(r_mid)
        if (rn_mid-IOTA0)*(rn_lo-IOTA0) < 0:
            r_hi = r_mid; rn_hi = rn_mid
        else:
            r_lo = r_mid; rn_lo = rn_mid
        if abs(r_hi-r_lo) < 1e-6:
            break
    r_stars.append(0.5*(r_lo+r_hi))

r_stars_valid = [r for r in r_stars if not np.isnan(r)]
print(f"  eps={eps_t}, n_theta0={n_th}")
print(f"  r* range: [{min(r_stars_valid):.4f}, {max(r_stars_valid):.4f}]")
print(f"  <r*>_theta0 = {np.mean(r_stars_valid):.6f}  (should be ~0 or formula-predicted value?)")
print(f"  std(r*) = {np.std(r_stars_valid):.4f}")
