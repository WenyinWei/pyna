import numpy as np

iota0 = (np.sqrt(5)-1)/2
m_t, n_t = 1, 0
c_t = 0.5+0.2j
alpha = m_t*iota0 + n_t
E = np.exp(2j*np.pi*alpha)
c2 = abs(c_t)**2

# I_mn from two analytic terms
I_from_parts = (4*np.pi*m_t*c2/alpha**2)*np.imag(E) + (4*m_t*c2/alpha**3)*(np.real(E)-1)

T1 = -2*np.pi/alpha**2 + (E-1)/(1j*alpha**3)
T2 = -2*np.pi*E/alpha**2 + (E-1)/(1j*alpha**3)
I_original = 2*np.real(1j*m_t*c2*(T1+T2))

print(f'I (original)  = {I_original:.8f}')
print(f'I (two terms) = {I_from_parts:.8f}')
print(f'Match: {np.isclose(I_original, I_from_parts)}')

# Compare with paper's implied I_mn
shear = -0.5
I_needed_for_paper = 2*c2 / (np.pi * alpha * shear)
print(f'\nI_mn needed for paper formula to hold: {I_needed_for_paper:.6f}')
print(f'I_mn from derivation:                  {I_original:.6f}')
print(f'Ratio (actual/paper-implied):          {I_original/I_needed_for_paper:.4f}')

# Now: what if the paper's Eq.(5.1) is derived from a HAMILTONIAN system
# where r is the canonical momentum and theta is the angle,
# with Hamiltonian H = iota0*r + iota'*r^2/2 + eps*V(theta,phi)?
# In that case, the second-order correction is given by the standard KAM formula:
# delta_H = eps^2 * {V, chi}  where chi is the solution to L0*chi = V
# L0 = iota0 * partial_theta + partial_phi  (the unperturbed Liouville operator)
# chi_mn = V_mn / (i*(m*iota0+n)) = c_mn/(i*alpha)
# 
# The second-order correction to the Hamiltonian:
# <H2> = (eps^2/2) * {V, chi}_mean
# = (eps^2/2) * sum_mn |V_mn|^2 * (partial_r_of_chi_mn corrected)
# 
# Actually for a twist map H = iota0*J + iota'*J^2/2 + eps*V(theta,phi):
# The frequency shift at J=0 is:
# delta_omega = eps^2 * sum_mn |partial_J V_mn|^2 / (m*omega + n)   [WRONG -- this is for omega-dependent V]
# 
# Actually the standard result for a near-integrable system:
# H = H0(J) + eps*V(J,theta,phi)
# with H0 = iota0*J + (iota'/2)*J^2, V = sum c_mn(J)*exp(i(m*theta+n*phi))
# 
# The second-order effective Hamiltonian (Birkhoff normal form):
# H2 = (eps^2/2) * sum_mn |V_mn|^2 * d/dJ[1/(m*H0'(J)+n)]
#    = (eps^2/2) * sum_mn |V_mn|^2 * (-m*H0''(J)) / (m*H0'(J)+n)^2
#    = -(eps^2/2) * sum_mn |V_mn|^2 * m*iota' / (m*iota0+n)^2
# 
# If V doesn't depend on J (our case), then |V_mn(J)|^2 doesn't change, and:
# The effective iota = partial_J H_eff = iota0 + iota'*J + eps^2 * partial_J H2
# At J=0: delta_iota = eps^2 * partial_J H2 evaluated at J=0
# But H2 has no J-dependence (since V is J-independent), so delta_iota = 0??
# That can't be right...
# 
# Actually the Birkhoff normal form modifies the frequency as:
# omega_eff(J) = omega0 + iota'*J + eps^2 * (partial_J^2 H2) * J + ...
# 
# The SHIFT in the torus location (not iota at fixed J) is what we want.
# The torus with iota=iota0 in the perturbed system is at J* ≠ 0.
# J* satisfies iota_eff(J*) = iota0.
# 
# From above: delta_iota(J=0) = 0 if V is J-independent!
# So <delta_r> = -delta_iota(0) / iota' = 0 ???
# That contradicts the ODE result showing <r>_Poincare ≠ 0.
# 
# KEY REALIZATION: In our 1-form ODE model, r is NOT a canonical action!
# The system is NOT Hamiltonian in the standard sense.
# dr/dphi = eps*f,  dtheta/dphi = iota0+iota'*r
# This is Hamiltonian ONLY if f = partial_theta V for some V.
# i.e., f must be the theta-derivative of a potential.
# 
# Our f = sum 2Re[c*exp(i(m*theta+n*phi))] is NOT a pure theta-derivative unless n=0.
# For n≠0, f includes phi-dependence that cannot be written as partial_theta V alone.
# 
# So the paper's Eq.(5.1) may assume a HAMILTONIAN structure (curl B = 0 condition)
# that our simple ODE test doesn't satisfy!

print()
print('Testing Hamiltonian structure: does our system satisfy curl(B)=0?')
print('For that, we need: partial_phi(f) = partial_theta(g) where g generates theta-eq.')
print('Our system: dr/dphi=eps*f, dtheta/dphi=iota0+iota*r  (no coupling f->g)')
print('This is NOT Hamiltonian in general -- f and g are independent.')
print()
print('The paper formula Eq.(5.1) is derived assuming:')
print('  B = grad(chi) for some stream function chi (i.e., div B = 0 + special structure)')
print('  In particular, the radial field dBr is related to dBtheta via the constraint.')
print('  Our test model violates this -- dBr and dBtheta are chosen independently.')
