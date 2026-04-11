import sys, importlib.util as _ilu
from pathlib import Path
from types import SimpleNamespace
import numpy as np

_repo = Path('C:/Users/Legion/Nutstore/1/Repo')

class _CVF(SimpleNamespace): pass
class _CSF(SimpleNamespace): pass

import types as _types
_fake_fields = _types.ModuleType('pyna.fields.cylindrical')
_fake_fields.VectorField3DCylindrical = _CVF
_fake_fields.ScalarField3DCylindrical = _CSF
sys.modules['pyna.fields.cylindrical'] = _fake_fields

_pg_path = _repo / 'pyna' / 'pyna' / 'toroidal' / 'plasma_response' / 'PerturbGS.py'
_pg_spec = _ilu.spec_from_file_location('PerturbGS_standalone', _pg_path)
_pg_mod  = _ilu.module_from_spec(_pg_spec)
_pg_spec.loader.exec_module(_pg_mod)

solve_perturbed_gs_coupled = _pg_mod.solve_perturbed_gs_coupled

mu0 = 4e-7 * np.pi
R0, B_val = 0.85, 1.0
nR, nZ = 20, 20
R_arr = np.linspace(0.65, 1.05, nR)
Z_arr = np.linspace(-0.20, 0.20, nZ)
Phi_1d = np.array([0.0])
RR, ZZ = np.meshgrid(R_arr, Z_arr, indexing='ij')

B0R_2d   = np.zeros((nR, nZ))
B0Z_2d   = np.zeros((nR, nZ))
B0Phi_2d = B_val * R0 / RR

r_loc    = np.sqrt((RR - R0)**2 + ZZ**2)
a_eff    = 0.18
psi_norm = np.clip(r_loc / a_eff, 0.0, 1.0)
p0_2d    = 500.0 * (1.0 - psi_norm**2)

grad_p_R = np.gradient(p0_2d, R_arr, axis=0)
grad_p_Z = np.gradient(p0_2d, Z_arr, axis=1)
B2       = B0Phi_2d**2 + 1e-30
J0R_2d   = -B0Phi_2d * grad_p_Z / B2
J0Z_2d   =  B0Phi_2d * grad_p_R / B2
J0Phi_2d = np.zeros((nR, nZ))

dBext_Z_2d = 1e-3 * np.sin(np.pi * (RR - R_arr[0]) / (R_arr[-1] - R_arr[0]))

def mk_v(VR, VZ, VPhi, nm=""):
    return _CVF(R=R_arr, Z=Z_arr, Phi=Phi_1d,
                VR=VR[:,:,np.newaxis], VZ=VZ[:,:,np.newaxis],
                VPhi=VPhi[:,:,np.newaxis], name=nm)

B0f  = mk_v(B0R_2d, B0Z_2d, B0Phi_2d, "B0")
J0f  = mk_v(J0R_2d, J0Z_2d, J0Phi_2d, "J0")
p0f  = _CSF(R=R_arr, Z=Z_arr, Phi=Phi_1d, value=p0_2d[:,:,np.newaxis], name="p0", units="Pa")
dBf  = mk_v(np.zeros_like(B0R_2d), dBext_Z_2d, np.zeros_like(B0R_2d), "dBext")

print("J0 magnitudes: R max=", np.max(np.abs(J0R_2d)), "Z max=", np.max(np.abs(J0Z_2d)))
print("dBext_Z max =", np.max(np.abs(dBext_Z_2d)))
print("RHS_Phi approx =", np.max(np.abs(J0R_2d * dBext_Z_2d)))

print("Solving coupled system...")
dB_c, dJ_c, dp_c = solve_perturbed_gs_coupled(
    B0f, J0f, p0f, dBf,
    solver='lsqr', max_iter=3000, tol=1e-10,
    weight_ampere=1e6, weight_force=1.0, weight_div=1e8, weight_divJ=1e6,
)

dBR   = dB_c.VR[:,:,0]; dBZ = dB_c.VZ[:,:,0]; dBPhi = dB_c.VPhi[:,:,0]
dJR   = dJ_c.VR[:,:,0]; dJZ = dJ_c.VZ[:,:,0]; dJPhi = dJ_c.VPhi[:,:,0]

print("dB rms:", np.sqrt(np.mean(dBR**2+dBZ**2+dBPhi**2)))
print("dJ rms:", np.sqrt(np.mean(dJR**2+dJZ**2+dJPhi**2)))

dBPhi_dZ   = np.gradient(dBPhi, Z_arr, axis=1)
curl_R_num = -dBPhi_dZ
d_R_dBPhi_dR = np.gradient(RR * dBPhi, R_arr, axis=0)
curl_Z_num  =  d_R_dBPhi_dR / (RR + 1e-30)
curl_Phi_num = np.gradient(dBR, Z_arr, axis=1) - np.gradient(dBZ, R_arr, axis=0)

print("mu0*dJ rms:", np.sqrt(np.mean((mu0*dJR)**2+(mu0*dJZ)**2+(mu0*dJPhi)**2)))
print("curl(dB) rms:", np.sqrt(np.mean(curl_R_num**2+curl_Z_num**2+curl_Phi_num**2)))

diff_rms = np.sqrt(np.mean((mu0*dJR-curl_R_num)**2+(mu0*dJZ-curl_Z_num)**2+(mu0*dJPhi-curl_Phi_num)**2))
ref = np.sqrt(np.mean(curl_R_num**2+curl_Z_num**2+curl_Phi_num**2)) + 1e-30
print("Ampere residual (diff/curl_ref):", diff_rms / ref)
