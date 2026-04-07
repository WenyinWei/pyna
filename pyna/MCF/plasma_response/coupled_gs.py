"""coupled_gs.py
================
Clean rewrite: solve for delta_B from force balance with delta_J = curl(delta_B)/mu0.

Physical problem (vacuum stellarator, J0=0)
--------------------------------------------
Given B0, delta_p, find delta_B satisfying:

    Force balance: [curl(delta_B)/mu0] x B0 = grad(delta_p)
    Div-free:       div(delta_B) = 0

delta_J = curl(delta_B)/mu0  is recovered post-solve (Ampere exact by construction).

Variables: x = [dBR(0..n-1), dBZ(n..2n-1), dBPhi(2n..3n-1)]  -- 3n unknowns
delta_p is GIVEN input (not a variable), grad(delta_p) enters the RHS directly.

Equations per interior plasma point (3 force + 1 div = 4 eqs, overdetermined):

Expanding curl(dB)/mu0 x B0 = grad(dp):

curl(dB)_R   = -ddBphi/dZ
curl(dB)_Z   =  dBphi/R + ddBphi/dR
curl(dB)_phi =  ddBR/dZ - ddBZ/dR

[FR]: (curl_Z*B0phi - curl_phi*B0Z)/mu0 = dp_dR
[FZ]: (curl_phi*B0R - curl_R*B0phi)/mu0 = dp_dZ
[FP]: (curl_R*B0Z  - curl_Z*B0R)/mu0   = 0
[D]:  dBR/R + ddBR/dR + ddBZ/dZ = 0
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr as _lsqr
from typing import Optional, Tuple

__all__ = ["solve_coupled_mhd"]

MU0 = 4e-7 * np.pi


def _grad(arr: np.ndarray, coords: np.ndarray, axis: int) -> np.ndarray:
    spacing = float(coords[1] - coords[0])
    return np.gradient(arr, spacing, axis=axis, edge_order=2)


def solve_coupled_mhd(
    BR_0: np.ndarray,
    BPhi_0: np.ndarray,
    BZ_0: np.ndarray,
    R_arr: np.ndarray,
    Z_arr: np.ndarray,
    delta_p: np.ndarray,
    plasma_mask: Optional[np.ndarray] = None,
    weight_bc: float = 1e3,
    max_iter: int = 5000,
    tol: float = 1e-8,
    verbose: bool = False,
) -> Tuple[tuple, tuple, float]:
    """Solve for delta_B from force balance; delta_J = curl(delta_B)/mu0 exact.

    Parameters
    ----------
    BR_0, BPhi_0, BZ_0 : (nR, nZ) -- background vacuum field
    R_arr, Z_arr       : 1D coordinate arrays
    delta_p            : (nR, nZ) -- pressure perturbation [Pa]
    plasma_mask        : bool (nR, nZ), optional -- physics domain
    weight_bc          : float -- penalty weight for vacuum BC rows
    max_iter, tol      : LSQR parameters
    verbose            : print diagnostics

    Returns
    -------
    (dJR, dJZ, dJPhi) : delta_J components [A/m^2]
    (dBR, dBZ, dBPhi) : delta_B components [T]
    ampere_residual   : float -- should be ~machine epsilon
    """
    nR, nZ = len(R_arr), len(Z_arr)
    n = nR * nZ
    mu0 = MU0

    B_ref   = float(np.sqrt(np.mean(BR_0**2 + BPhi_0**2 + BZ_0**2))) + 1e-12
    p_scale = float(np.abs(delta_p).max()) + 1e-30
    L_ref   = float(0.5 * ((R_arr[-1]-R_arr[0]) + (Z_arr[-1]-Z_arr[0])))

    if plasma_mask is None:
        plasma_mask = np.abs(delta_p) > 0.01 * p_scale

    dp_dR = _grad(delta_p, R_arr, axis=0)
    dp_dZ = _grad(delta_p, Z_arr, axis=1)

    # Variables: dBR, dBZ, dBPhi -- 3n total
    O_BR, O_BZ, O_BP = 0, n, 2*n
    N_VAR = 3 * n

    # Column scale: dB ~ mu0 * p / (B * L) * L = mu0*p/B
    dB_scale = max(mu0 * p_scale / B_ref, 1e-30)
    col_scale = np.full(N_VAR, dB_scale)

    # 4 equations per point
    N_EQ = 4
    n_rows = N_EQ * n
    A = lil_matrix((n_rows, N_VAR), dtype=np.float64)
    b = np.zeros(n_rows, dtype=np.float64)

    cs = dB_scale  # all columns same scale

    def k(i, j):
        return i * nZ + j

    for i in range(nR):
        for j in range(nZ):
            kij     = k(i, j)
            eq      = N_EQ * kij
            Ri      = R_arr[i]
            interior = (0 < i < nR-1) and (0 < j < nZ-1)
            plasma   = bool(plasma_mask[i, j]) and interior

            ip = min(i+1, nR-1); im = max(i-1, 0)
            jp = min(j+1, nZ-1); jm = max(j-1, 0)
            i2R = 1.0 / (R_arr[ip] - R_arr[im])
            i2Z = 1.0 / (Z_arr[jp] - Z_arr[jm])

            if plasma:
                b0R  = BR_0[i, j]
                b0Z  = BZ_0[i, j]
                b0Ph = BPhi_0[i, j]

                # curl(dB) components in terms of dB variables:
                # cR  = -ddBPhi/dZ
                # cZ  =  dBPhi/R + ddBPhi/dR
                # cPh =  ddBR/dZ - ddBZ/dR

                # [FR]: (cZ*b0Ph - cPh*b0Z) / mu0 = dp_dR
                # cZ*b0Ph:
                A[eq+0, O_BP+kij]      += cs * b0Ph / (Ri * mu0)
                A[eq+0, O_BP+k(ip,j)] += cs * b0Ph * i2R / mu0
                A[eq+0, O_BP+k(im,j)] -= cs * b0Ph * i2R / mu0
                # -cPh*b0Z:
                A[eq+0, O_BR+k(i,jp)] -= cs * b0Z * i2Z / mu0
                A[eq+0, O_BR+k(i,jm)] += cs * b0Z * i2Z / mu0
                A[eq+0, O_BZ+k(ip,j)] += cs * b0Z * i2R / mu0
                A[eq+0, O_BZ+k(im,j)] -= cs * b0Z * i2R / mu0
                b[eq+0] = dp_dR[i, j]

                # [FZ]: (cPh*b0R - cR*b0Ph) / mu0 = dp_dZ
                # cPh*b0R:
                A[eq+1, O_BR+k(i,jp)] += cs * b0R * i2Z / mu0
                A[eq+1, O_BR+k(i,jm)] -= cs * b0R * i2Z / mu0
                A[eq+1, O_BZ+k(ip,j)] -= cs * b0R * i2R / mu0
                A[eq+1, O_BZ+k(im,j)] += cs * b0R * i2R / mu0
                # -cR*b0Ph: cR = -ddBPhi/dZ, so -cR*b0Ph = ddBPhi/dZ * b0Ph
                A[eq+1, O_BP+k(i,jp)] += cs * b0Ph * i2Z / mu0
                A[eq+1, O_BP+k(i,jm)] -= cs * b0Ph * i2Z / mu0
                b[eq+1] = dp_dZ[i, j]

                # [FP]: (cR*b0Z - cZ*b0R) / mu0 = 0
                # cR*b0Z: cR = -ddBPhi/dZ
                A[eq+2, O_BP+k(i,jp)] -= cs * b0Z * i2Z / mu0
                A[eq+2, O_BP+k(i,jm)] += cs * b0Z * i2Z / mu0
                # -cZ*b0R:
                A[eq+2, O_BP+kij]      -= cs * b0R / (Ri * mu0)
                A[eq+2, O_BP+k(ip,j)] -= cs * b0R * i2R / mu0
                A[eq+2, O_BP+k(im,j)] += cs * b0R * i2R / mu0
                # b stays 0

                # [D]: dBR/R + ddBR/dR + ddBZ/dZ = 0
                A[eq+3, O_BR+kij]      += cs / Ri
                A[eq+3, O_BR+k(ip,j)] += cs * i2R
                A[eq+3, O_BR+k(im,j)] -= cs * i2R
                A[eq+3, O_BZ+k(i,jp)] += cs * i2Z
                A[eq+3, O_BZ+k(i,jm)] -= cs * i2Z

            else:
                # Vacuum BC: dBPhi=0, dBR=0, dBZ=0  (soft penalty)
                wbc = weight_bc * cs
                A[eq+0, O_BP+kij] = wbc
                A[eq+1, O_BR+kij] = wbc
                A[eq+2, O_BZ+kij] = wbc
                A[eq+3, O_BR+kij] = wbc * 0.1  # weaker, div-free BC

    if verbose:
        print(f"  Matrix: {n_rows}x{N_VAR}, nnz={A.nnz}, b_nnz={(b!=0).sum()}")
        print(f"  B_ref={B_ref:.3f}T  dB_scale={dB_scale:.3e}T  "
              f"J_ref={p_scale/(B_ref*L_ref):.3e}A/m2")

    result = _lsqr(A.tocsc(), b, damp=0.0, iter_lim=max_iter, atol=tol, btol=tol)
    xs = result[0]

    if verbose:
        print(f"  LSQR: status={result[1]}, itn={result[2]}, r1norm={result[3]:.4e}")

    # Physical solution (descale)
    x = xs * col_scale
    dBR  = x[O_BR:O_BR+n].reshape(nR, nZ)
    dBZ  = x[O_BZ:O_BZ+n].reshape(nR, nZ)
    dBPh = x[O_BP:O_BP+n].reshape(nR, nZ)

    # Recover delta_J = curl(delta_B)/mu0  -- exact by construction
    dJR  = -_grad(dBPh, Z_arr, axis=1) / mu0
    dJZ  = (dBPh / R_arr[:, None] + _grad(dBPh, R_arr, axis=0)) / mu0
    dJPh = (_grad(dBR, Z_arr, axis=1) - _grad(dBZ, R_arr, axis=0)) / mu0

    # Ampere residual (sanity check -- should be ~1e-15)
    cR  = -_grad(dBPh, Z_arr, axis=1)
    cZ  = dBPh/R_arr[:, None] + _grad(dBPh, R_arr, axis=0)
    cPh = _grad(dBR, Z_arr, axis=1) - _grad(dBZ, R_arr, axis=0)
    num = float(np.sqrt(np.mean((mu0*dJR-cR)**2+(mu0*dJZ-cZ)**2+(mu0*dJPh-cPh)**2)))
    den = float(np.sqrt(np.mean((mu0*dJR)**2+(mu0*dJZ)**2+(mu0*dJPh)**2))) + 1e-30
    amp_res = num / den

    return (dJR, dJZ, dJPh), (dBR, dBZ, dBPh), amp_res


# Self-test
if __name__ == "__main__":
    import pickle
    print("=== coupled_gs self-test ===\n")
    mu0 = MU0

    # Test 1: B0=(0,1,0), linear pressure
    # Analytic: [FR] cZ*b0Ph/mu0 = dp/dR => (dBPhi/R + ddBPhi/dR)/mu0 * 1 = dp/dR
    # => dBPhi ~ mu0 * dp/dR * R * L (rough)
    # dJ_Z = cZ/mu0 = (dBPhi/R + ddBPhi/dR)/mu0
    # From force balance dJ_Z = dp/dR / B0_phi = -p0/(2a) / 1
    nR, nZ = 24, 24
    R0, a = 0.85, 0.15
    R_arr = np.linspace(R0-a, R0+a, nR)
    Z_arr = np.linspace(-a, a, nZ)
    RR, ZZ = np.meshgrid(R_arr, Z_arr, indexing='ij')
    p0 = 5000.0
    delta_p = p0 * (R_arr[-1] - RR) / (2*a)

    (dJR,dJZ,dJPh),(dBR,dBZ,dBPh),amp = solve_coupled_mhd(
        np.zeros((nR,nZ)), np.ones((nR,nZ)), np.zeros((nR,nZ)),
        R_arr, Z_arr, delta_p,
        weight_bc=1e3, max_iter=20000, tol=1e-11, verbose=True)

    mask = (np.abs(delta_p) > 0.02*p0) & (RR > R_arr[3]) & (RR < R_arr[-4])
    expected = -p0 / (2*a) / 1.0
    got = dJZ[mask].mean()
    err = abs(got - expected) / abs(expected)
    print(f"dJ_Z mean={got:.1f}  expected={expected:.1f}  err={err:.4f}  amp={amp:.2e}")
    print("PASS" if err < 0.05 and amp < 1e-10 else f"FAIL (err={err:.3f}, amp={amp:.2e})")
    print()

    # Test 2: HAO field
    cache_path = r"C:\Users\Legion\Nutstore\1\Repo\topoquest\data\bluestar_starting_config_field_cache.pkl"
    try:
        with open(cache_path, 'rb') as f:
            c = pickle.load(f)
        R2=c['R_grid']; Z2=c['Z_grid']
        BR2=c['BR'][:,:,0].astype(float)
        BPh2=c['BPhi'][:,:,0].astype(float)
        BZ2=c['BZ'][:,:,0].astype(float)
        Bavg2 = float(np.mean(BR2**2+BPh2**2+BZ2**2))
        R_ax,Z_ax,a_eff = 0.85235,-0.000073,0.18
        RR2,_ = np.meshgrid(R2,Z2,indexing='ij')
        psi = np.clip(((RR2-R_ax)**2)/a_eff**2, 0, 1)
        dp2 = 0.01*Bavg2/(2*mu0)*(1-psi)**2
        print(f"Test 2: HAO field (100x100), beta=1%, p0={dp2.max():.0f} Pa")
        (dJR2,dJZ2,dJPh2),(dBR2,dBZ2,dBPh2),amp2 = solve_coupled_mhd(
            BR2, BPh2, BZ2, R2, Z2, dp2,
            weight_bc=1e3, max_iter=5000, tol=1e-8, verbose=True)
        dJmag = np.sqrt(dJR2**2+dJZ2**2+dJPh2**2)
        J_exp = dp2.max()/(np.sqrt(Bavg2)*a_eff)
        print(f"|dJ| mean={dJmag.mean():.3e} A/m2  expected~{J_exp:.1e}  amp={amp2:.2e}")
        print("PASS" if dJmag.mean() > 10 and amp2 < 1e-10 else "FAIL")
    except FileNotFoundError:
        print("Test 2 skipped (cache not found)")
