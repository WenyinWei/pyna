"""Quick test: verify beta_avg calculation."""
import numpy as np
import glob

MU0 = 4e-7 * np.pi

d = np.load("/mnt/c/Users/28105/Nutstore/1/haodata/coilsys/vacuum_fields/dipole_coil_100_current-87600.00A.npz")
Rg = d["R_grid"].astype(np.float64)
Zg = d["Z_grid"].astype(np.float64)
Pg = d["Phi_grid"].astype(np.float64)

BR = np.zeros((len(Rg), len(Zg), len(Pg)))
BPhi = np.zeros_like(BR)
BZ = np.zeros_like(BR)

files = sorted(glob.glob("/mnt/c/Users/28105/Nutstore/1/haodata/coilsys/vacuum_fields/dipole_coil_*.npz"))[:30]
for fp in files:
    dd = np.load(fp)
    I = float(dd.get("coil_current", 0))
    BR += dd["BR_resp"].astype(np.float64) * I
    BPhi += dd["BPhi_resp"].astype(np.float64) * I
    BZ += dd["BZ_resp"].astype(np.float64) * I

B_sq = BR**2 + BPhi**2 + BZ**2
B_mag = np.sqrt(np.maximum(B_sq, 1e-20))
B_avg = float(np.mean(B_mag))
print(f"B_vol_avg = {B_avg:.4f} T")

# psi_n centered on magnetic axis
R_axis, Z_axis = 0.94, 0.0
a_half = (Rg.max() - Rg.min()) / 2.0
RR, ZZ, PP = np.meshgrid(Rg, Zg, Pg, indexing="ij")
r_from_axis = np.sqrt((RR - R_axis)**2 + (ZZ - Z_axis)**2)
psi_n = np.clip(r_from_axis / a_half, 0, 1)

beta_target = 0.02
alpha = 2.0
p0 = beta_target * B_avg**2 / (2*MU0) * (alpha + 1)
p = p0 * np.maximum(0, 1 - psi_n)**alpha

inside = psi_n < 0.85
dR = Rg[1] - Rg[0]
dZ = Zg[1] - Zg[0]
dPhi = Pg[1] - Pg[0]
dV = RR * dR * dZ * dPhi

p_avg = np.sum(p[inside] * dV[inside]) / np.sum(dV[inside])
B_sq_avg = np.sum(B_sq[inside] * dV[inside]) / np.sum(dV[inside])
beta_avg = 2*MU0 * p_avg / B_sq_avg

print(f"psi_n at R_axis: {psi_n[np.argmin(np.abs(Rg - R_axis)), np.argmin(np.abs(Zg - Z_axis)), 0]:.4f}")
print(f"psi_n at R_max:  {psi_n[-1, 0, 0]:.4f}")
print(f"p0 = {p0:.2f} Pa")
print(f"<p>_V = {p_avg:.2f} Pa")
print(f"<B^2>_V = {B_sq_avg:.4f} T^2")
print(f"<beta> = {beta_avg:.6f}  (target = {beta_target})")

# Test multiple beta values
for bt in [0.005, 0.01, 0.02, 0.03, 0.05]:
    p0_t = bt * B_avg**2 / (2*MU0) * (alpha + 1)
    p_t = p0_t * np.maximum(0, 1 - psi_n)**alpha
    p_avg_t = np.sum(p_t[inside] * dV[inside]) / np.sum(dV[inside])
    ba_t = 2*MU0 * p_avg_t / B_sq_avg
    print(f"  beta_target={bt:.3f} -> <beta>={ba_t:.6f}")
