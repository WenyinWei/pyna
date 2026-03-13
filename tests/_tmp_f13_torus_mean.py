"""
Analytic derivation of second-order rotation number shift delta_iota.

System:
  dr/dphi  = eps * f(theta, phi)
  dtheta/dphi = iota0 + iota' * r

f = sum_mn 2 Re[c_mn exp(i(m*theta+n*phi))]

Standard Lie-transform / averaging (canonical perturbation theory):
The unperturbed system is integrable with action r, angle theta.
The 'Hamiltonian' (generating the evolution) is:
  K(r, theta, phi) = iota0*r + (iota'/2)*r^2  (twist map generator)
The perturbation is:
  V(r, theta, phi) = eps * integral f dphi ... 

Actually let's use the direct computation.
The rotation number at r=0 in the perturbed system:

iota_eff = lim_{N->inf} theta(2*pi*N) / (2*pi*N)

To second order in eps, at r_init=0:
theta(phi) = theta0 + iota0*phi + iota'*integral_0^phi r(phi') dphi'
r(phi) = eps*r1(phi) + eps^2*r2(phi) + ...

r1(phi) = integral_0^phi f(iota0*phi', phi') dphi'    [along unperturbed orbit theta=theta0+iota0*phi]

theta(2*pi*N) = theta0 + 2*pi*N*iota0 + iota'*integral_0^{2*pi*N} r dphi
             = theta0 + 2*pi*N*iota0 + eps*iota'*integral_0^{2*pi*N} r1 dphi
             + eps^2*iota'*integral_0^{2*pi*N} r2 dphi + ...

The second term: eps*iota'*integral r1 = eps*iota'*integral integral f dphi'dphi
This is a double integral. For non-resonant f, r1(phi) is quasi-periodic with
ZERO mean, so the DOUBLE integral grows like ~phi (not phi^2).
Specifically: integral_0^{2*pi*N} r1(phi) dphi / (2*pi*N)
= (1/2*pi) * integral_0^{2*pi} r1(phi) dphi + O(1/N)
= (1/2*pi) * sum_mn 2Re[c_mn exp(im*theta0) * integral_0^{2*pi} (exp(i*alpha_mn*phi)-1)/(i*alpha_mn) dphi]

This integral = (exp(2*pi*i*alpha)-1)/(i*alpha)^2 - 2*pi/(i*alpha)
For general (non-integer) alpha, this is O(1), so the contribution to iota_eff
at second order in eps from THIS term is eps*(not zero!).

BUT -- this term depends on theta0. When we average over the torus (all theta0),
the theta0-dependent factors average to zero (since <exp(im*theta0)>=0 for m≠0).

The THETA0-INDEPENDENT (torus-mean) contribution to delta_iota:
From r2(phi): this contains terms quadratic in c_mn that survive the theta0 average.

Let's compute r2:
r2 satisfies: dr2/dphi = [df/dtheta * delta_theta + df/dr * r1]_2nd-order
But df/dr = 0 (f doesn't depend on r in our model).
And delta_theta comes from the angular perturbation:
dtheta/dphi = iota0 + iota'*r
= iota0 + iota'*(eps*r1 + eps^2*r2)
So theta(phi) = theta0 + iota0*phi + eps*iota'*int_0^phi r1 dphi' + O(eps^2)

r2(phi) = integral_0^phi (df/dtheta)|_{unperturbed} * (theta_pert - theta_unperturbed)/(eps) dphi'
       = integral_0^phi (df/dtheta)(theta0+iota0*phi') * iota' * R1(phi') dphi'

where R1(phi) = integral_0^phi r1(phi') dphi'  (antiderivative of r1)

The mean of r2 over a full period:
<r2>_T = (1/2*pi) * integral_0^{2*pi} r2(phi) dphi 
       = (iota'/2*pi) * integral_0^{2*pi} f_theta(theta0+iota0*phi) * R1(phi) dphi

where f_theta = df/dtheta = sum_mn 2 Re[i*m*c_mn exp(i(m*theta+n*phi))]

The theta0-average of <r2>_T:
<r2>_{T,theta0} = (iota'/2*pi) * integral_0^{2*pi} <f_theta * R1>_{theta0} dphi

<f_theta(theta0+iota0*phi) * R1(phi)>_{theta0}
= < sum_mn 2Re[i*m*c_mn * exp(im*theta0) * exp(i*alpha_mn*phi)] 
  * sum_kl 2Re[c_kl * exp(ik*theta0) * F_kl(phi)] >_{theta0}

where F_kl(phi) = (exp(i*alpha_kl*phi)-1)/(i*alpha_kl)^2 - phi/(i*alpha_kl)

The theta0-average pairs (m,n) with (-k,-l) terms: k=-m, l=-n
= sum_mn 2Re[i*m*c_mn * (-c_mn*).conj * exp(i*alpha_mn*phi) * F_{mn}(phi)^*]

Wait let me be more careful:
<exp(i(m+k)*theta0)>_theta0 = delta_{m,-k}

So the surviving terms have k = -m, l = -n:
F_{-m,-n}(phi) = (exp(-i*alpha_mn*phi)-1)/(-i*alpha_mn)^2 - phi/(-i*alpha_mn)

The result:
<r2>_{T,theta0} = (iota'/2*pi) * sum_mn 2*Re[i*m*c_mn * c_{-m,-n} * integral_0^{2*pi} 
                  exp(i*alpha*phi) * F_{-m,-n}(phi)^* dphi]

Since c_{-m,-n} = c_mn* (real field), this simplifies.

Let me just compute this NUMERICALLY with a Monte Carlo over theta0.
"""
import numpy as np
from scipy.integrate import solve_ivp

IOTA0  = (np.sqrt(5)-1)/2
SHEAR  = -0.5
BPHI   = 5.0
MODES  = np.array([(1,0),(2,1),(1,-1)])
dBr_F  = np.array([0.5+0.2j, 0.3-0.15j, -0.4+0.1j])

def rhs(phi, y, eps):
    f = sum(2*(c.real*np.cos(m*y[1]+n*phi)-c.imag*np.sin(m*y[1]+n*phi))
            for (m,n),c in zip(MODES, dBr_F))
    return [eps*f, IOTA0+SHEAR*y[0]]

# Use short orbit (100 revs) but average over many theta0 values
# to get unbiased estimate of the torus-mean r
N_rev = 100
N_th0 = 60

print("Torus-mean <r> from theta0-averaged Poincare data:")
print(f"(N_rev={N_rev}, N_theta0={N_th0})\n")
print(f"{'eps':>8} {'<r>_torus':>14} {'formula_2x':>14} {'ratio':>8}")

for eps_t in [0.05, 0.02, 0.01, 0.005, 0.002]:
    theta0_vals = np.linspace(0, 2*np.pi, N_th0, endpoint=False)
    all_r = []
    for th0 in theta0_vals:
        sol = solve_ivp(lambda p,y: rhs(p,y,eps_t), [0,N_rev*2*np.pi], [0.0,th0],
                        method='DOP853', rtol=1e-11, atol=1e-13,
                        t_eval=2*np.pi*np.arange(1,N_rev+1))
        all_r.extend(sol.y[0].tolist())
    mean_r = np.mean(all_r)
    # 2x one-sided formula
    total = sum(2*abs(c*BPHI*eps_t)**2/(m*IOTA0+n) for (m,n),c in zip(MODES,dBr_F))
    dr_formula = -total/(SHEAR*(2*np.pi)**2)
    ratio = mean_r/dr_formula if abs(dr_formula) > 0 else float('nan')
    print(f"{eps_t:>8.4f} {mean_r:>14.4e} {dr_formula:>14.4e} {ratio:>8.3f}")
