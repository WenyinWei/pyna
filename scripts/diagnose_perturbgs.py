"""Diagnose PerturbGS sparse matrix conditioning.

Builds a synthetic 40×40 vacuum field, assembles the sparse matrix A for
solve_perturbed_gs, then compares three weight configurations:

  - Original:  EqBdiv=1e8, BC_J=1e9, BC_p=1e12
  - Fix-1:     EqBdiv=1e4, BC_J=1e5, BC_p=1e6
  - Fix-2:     EqBdiv=1e3, BC_J=1e4, BC_p=1e5

For each configuration, reports:
  - Frobenius norm of A
  - Max absolute row value (checks if BC rows dominate)
  - Estimated condition number (via svds)
  - LSQR iteration count and final residual

Run with Windows Python (no dolfinx needed).
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr, svds

MU0 = 4e-7 * np.pi

# ---------------------------------------------------------------------------
# Synthetic field builder (mirrors make_synthetic_field_cache logic)
# ---------------------------------------------------------------------------

def make_synthetic_vacuum_fields(nR=40, nZ=40):
    """Pure toroidal vacuum field on a 40x40 grid."""
    R_arr = np.linspace(0.5, 2.5, nR)
    Z_arr = np.linspace(-1.0, 1.0, nZ)
    RR, ZZ = np.meshgrid(R_arr, Z_arr, indexing='ij')

    B_phi_0 = 1.0  # T
    BPhi_2d = B_phi_0 / RR
    BR_2d   = np.zeros_like(BPhi_2d)
    BZ_2d   = np.zeros_like(BPhi_2d)

    J0_R    = np.zeros_like(BR_2d)
    J0_Z    = np.zeros_like(BR_2d)
    J0_Phi  = np.zeros_like(BR_2d)
    p0_2d   = np.zeros_like(BR_2d)

    dBR_R   = np.zeros_like(BR_2d)
    dBR_Z   = np.zeros_like(BR_2d)
    dBR_Phi = np.ones_like(BR_2d) * 1e-3  # small toroidal perturbation

    return (R_arr, Z_arr,
            BR_2d, BZ_2d, BPhi_2d,
            J0_R, J0_Z, J0_Phi,
            p0_2d,
            dBR_R, dBR_Z, dBR_Phi)


# ---------------------------------------------------------------------------
# Matrix assembly (mirrors solve_perturbed_gs internals, no joblib cache)
# ---------------------------------------------------------------------------

def assemble_matrix(R, Z, B0R, B0Z, B0Phi, J0R, J0Z, J0Phi, p0_val,
                    dBextR, dBextZ, dBextPhi,
                    EqBdiv_weight, BC_no_J_weight, BC_no_p_weight):

    nR, nZ = len(R), len(Z)
    dR = R[1] - R[0]
    dZ = Z[1] - Z[0]
    n = nR * nZ
    nEq = 8

    RHS_R   = np.zeros((nR, nZ))
    RHS_Z   = np.zeros((nR, nZ))
    RHS_Phi = np.zeros((nR, nZ))
    # RHS = -J0 × dB_ext  (all zero for vacuum + zero J0)

    OFF_BR   = 0
    OFF_BZ   = n
    OFF_BPHI = 2 * n
    OFF_P    = 3 * n

    def k(i, j):
        return i * nZ + j

    A = lil_matrix((nEq * n, 4 * n))
    b = np.zeros(nEq * n)

    for i in range(nR):
        for j in range(nZ):
            kij = k(i, j)
            kpR = k(i + 1, j) if i < nR - 1 else k(i, j)
            kmR = k(i - 1, j) if i > 0      else k(i, j)
            kpZ = k(i, j + 1) if j < nZ - 1 else k(i, j)
            kmZ = k(i, j - 1) if j > 0      else k(i, j)

            interior = (0 < i < nR - 1) and (0 < j < nZ - 1)
            eq_base = nEq * kij

            if interior:
                # Force balance: (1/mu0)(curl dB) x B0 + J0 x dB - grad(dp) = RHS
                # (simplified; J0=0 for vacuum)

                # Div-free is filled below
                # For now just add force balance rows via J0xdB (=0 here) and grad(dp)
                # -grad(dp) in R:
                A[eq_base + 0, k(i+1,j) + OFF_P] -= 1/(2*dR)
                A[eq_base + 0, k(i-1,j) + OFF_P] += 1/(2*dR)
                # -grad(dp) in Z:
                A[eq_base + 1, k(i,j+1) + OFF_P] -= 1/(2*dZ)
                A[eq_base + 1, k(i,j-1) + OFF_P] += 1/(2*dZ)

                # curlxB0 terms (simplified for pure toroidal B0)
                b0Ph = B0Phi[i, j]
                factor = 1.0 / MU0
                # (1/mu0) curl_R = -dBPhi/dZ; (curlxB0)_R = curl_Z*B0Ph - curl_Ph*B0Z
                # curl_Z = BPhi/R + dBPhi/dR
                A[eq_base + 0, kij  + OFF_BPHI] += factor * b0Ph / R[i]
                A[eq_base + 0, k(i+1,j) + OFF_BPHI] += factor * b0Ph / (2*dR)
                A[eq_base + 0, k(i-1,j) + OFF_BPHI] -= factor * b0Ph / (2*dR)
                # curl_Ph = dBR/dZ - dBZ/dR  (row 2 for Phi-component of force)
                A[eq_base + 2, k(i,j+1) + OFF_BR]  += factor * B0Z[i,j] / (2*dZ)
                A[eq_base + 2, k(i,j-1) + OFF_BR]  -= factor * B0Z[i,j] / (2*dZ)
                A[eq_base + 2, k(i+1,j) + OFF_BZ]  -= factor * B0Z[i,j] / (2*dR)  # tiny
                A[eq_base + 2, k(i-1,j) + OFF_BZ]  += factor * B0Z[i,j] / (2*dR)

                b[eq_base + 0] = RHS_R[i, j]
                b[eq_base + 1] = RHS_Z[i, j]
                b[eq_base + 2] = RHS_Phi[i, j]

            # Div-free
            w = EqBdiv_weight
            if interior:
                A[eq_base + 3, kij      + OFF_BR] += (1/R[i]) * w
                A[eq_base + 3, k(i+1,j) + OFF_BR] +=  (1/(2*dR)) * w
                A[eq_base + 3, k(i-1,j) + OFF_BR] -= (1/(2*dR)) * w
                A[eq_base + 3, k(i,j+1) + OFF_BZ] +=  (1/(2*dZ)) * w
                A[eq_base + 3, k(i,j-1) + OFF_BZ] -= (1/(2*dZ)) * w
            else:
                if i == 0:
                    A[eq_base + 3, kij      + OFF_BR] += (1/R[i] - 1/dR) * w
                    A[eq_base + 3, k(i+1,j) + OFF_BR] += (1/dR) * w
                elif i == nR-1:
                    A[eq_base + 3, kij      + OFF_BR] += (1/R[i] + 1/dR) * w
                    A[eq_base + 3, k(i-1,j) + OFF_BR] -= (1/dR) * w
                else:
                    A[eq_base + 3, kij      + OFF_BR] += (1/R[i]) * w
                    A[eq_base + 3, k(i+1,j) + OFF_BR] +=  (1/(2*dR)) * w
                    A[eq_base + 3, k(i-1,j) + OFF_BR] -= (1/(2*dR)) * w
                if j == 0:
                    A[eq_base + 3, k(i,j+1) + OFF_BZ] += (1/dZ) * w
                    A[eq_base + 3, kij       + OFF_BZ] -= (1/dZ) * w
                elif j == nZ-1:
                    A[eq_base + 3, kij       + OFF_BZ] += (1/dZ) * w
                    A[eq_base + 3, k(i,j-1) + OFF_BZ] -= (1/dZ) * w
                else:
                    A[eq_base + 3, k(i,j+1) + OFF_BZ] +=  (1/(2*dZ)) * w
                    A[eq_base + 3, k(i,j-1) + OFF_BZ] -= (1/(2*dZ)) * w

            # BC (J=0 and p=0)
            is_boundary = (i in [0, 1, 2, nR-3, nR-2, nR-1] or
                           j in [0, 1, 2, nZ-3, nZ-2, nZ-1])
            if is_boundary:
                wJ = BC_no_J_weight
                jpZ = min(j+1, nZ-1); jmZ = max(j-1, 0)
                ipR = min(i+1, nR-1); imR = max(i-1, 0)
                coef_dZ = 1/(Z[jpZ]-Z[jmZ]) if jpZ != jmZ else 1/dZ
                coef_dR = 1/(R[ipR]-R[imR]) if ipR != imR else 1/dR

                A[eq_base+4, k(i,jpZ) + OFF_BPHI] -= coef_dZ * wJ
                A[eq_base+4, k(i,jmZ) + OFF_BPHI] += coef_dZ * wJ

                A[eq_base+5, k(i,jpZ) + OFF_BR]   += coef_dZ * wJ
                A[eq_base+5, k(i,jmZ) + OFF_BR]   -= coef_dZ * wJ
                A[eq_base+5, k(ipR,j) + OFF_BZ]   -= coef_dR * wJ
                A[eq_base+5, k(imR,j) + OFF_BZ]   += coef_dR * wJ

                A[eq_base+6, kij       + OFF_BPHI] += (1/R[i]) * wJ
                A[eq_base+6, k(ipR,j) + OFF_BPHI] += coef_dR * wJ
                A[eq_base+6, k(imR,j) + OFF_BPHI] -= coef_dR * wJ

                A[eq_base+7, kij + OFF_P] = BC_no_p_weight

    return A.tocsc(), b


# ---------------------------------------------------------------------------
# Condition number estimation via svds
# ---------------------------------------------------------------------------

def estimate_condition(A_csc, k_svds=6):
    """Estimate condition number using k_svds largest and smallest singular values."""
    n_rows, n_cols = A_csc.shape
    k = min(k_svds, min(n_rows, n_cols) - 1)
    try:
        sv_large = svds(A_csc, k=k, which='LM', return_singular_vectors=False)
        sv_small = svds(A_csc, k=k, which='SM', return_singular_vectors=False)
        sigma_max = float(np.max(sv_large))
        sigma_min = float(np.min(sv_small))
        cond = sigma_max / (sigma_min + 1e-300)
        return sigma_max, sigma_min, cond
    except Exception as e:
        return float('nan'), float('nan'), float('nan')


# ---------------------------------------------------------------------------
# Diagnose one weight configuration
# ---------------------------------------------------------------------------

def diagnose(label, EqBdiv, BC_J, BC_p,
             R, Z, B0R, B0Z, B0Phi, J0R, J0Z, J0Phi, p0_val,
             dBextR, dBextZ, dBextPhi,
             max_iter=500, tol=1e-6):
    print(f"\n{'='*60}")
    print(f"Config: {label}")
    print(f"  EqBdiv_weight={EqBdiv:.0e}, BC_J_weight={BC_J:.0e}, BC_p_weight={BC_p:.0e}")

    A_csc, b = assemble_matrix(R, Z, B0R, B0Z, B0Phi, J0R, J0Z, J0Phi, p0_val,
                                dBextR, dBextZ, dBextPhi,
                                EqBdiv, BC_J, BC_p)

    frob = float(np.sqrt((A_csc.data**2).sum()))
    print(f"  Frobenius norm of A: {frob:.4e}")

    # Row max absolute values
    from scipy.sparse import csr_matrix
    A_csr = A_csc.tocsr()
    row_maxabs = np.array([
        np.abs(A_csr.getrow(i).data).max() if A_csr.getrow(i).nnz > 0 else 0.0
        for i in range(min(A_csr.shape[0], 10000))
    ])
    print(f"  Row max |A_ij|:  overall max={row_maxabs.max():.4e}, "
          f"median={np.median(row_maxabs):.4e}, "
          f"min={row_maxabs[row_maxabs>0].min() if (row_maxabs>0).any() else 0:.4e}")

    # Condition number
    print("  Estimating condition number via svds...")
    sigma_max, sigma_min, cond = estimate_condition(A_csc, k_svds=6)
    print(f"  σ_max={sigma_max:.4e}, σ_min={sigma_min:.4e}, κ(A) ≈ {cond:.4e}")

    # LSQR
    print(f"  Running LSQR (max_iter={max_iter}, tol={tol:.0e})...")
    result = lsqr(A_csc, b, damp=1e-4, iter_lim=max_iter, atol=tol, btol=tol)
    x, istop, itn, r1norm, r2norm = result[0], result[1], result[2], result[3], result[4]
    print(f"  LSQR: iters={itn}, istop={istop}, r1norm={r1norm:.4e}, r2norm={r2norm:.4e}")

    return {
        'label': label, 'frob': frob, 'sigma_max': sigma_max,
        'sigma_min': sigma_min, 'cond': cond,
        'lsqr_iters': itn, 'lsqr_istop': istop, 'r1norm': r1norm,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== PerturbGS Sparse Matrix Conditioning Diagnostic ===")
    print("Building synthetic 40×40 vacuum field...")

    (R, Z,
     B0R, B0Z, B0Phi,
     J0R, J0Z, J0Phi,
     p0_val,
     dBextR, dBextZ, dBextPhi) = make_synthetic_vacuum_fields(nR=40, nZ=40)

    print(f"Grid: nR={len(R)}, nZ={len(Z)}, n={len(R)*len(Z)}, matrix size={8*len(R)*len(Z)}×{4*len(R)*len(Z)}")

    configs = [
        ("Original",  1e8, 1e9, 1e12),
        ("Fix-1",     1e4, 1e5, 1e6),
        ("Fix-2",     1e3, 1e4, 1e5),
    ]

    results = []
    for label, EqBdiv, BC_J, BC_p in configs:
        r = diagnose(label, EqBdiv, BC_J, BC_p,
                     R, Z, B0R, B0Z, B0Phi, J0R, J0Z, J0Phi, p0_val,
                     dBextR, dBextZ, dBextPhi,
                     max_iter=1000, tol=1e-6)
        results.append(r)

    print("\n" + "="*60)
    print("SUMMARY")
    print(f"{'Config':<12} {'κ(A)':<14} {'LSQR iters':<14} {'r1norm':<12}")
    print("-"*60)
    for r in results:
        print(f"{r['label']:<12} {r['cond']:<14.4e} {r['lsqr_iters']:<14d} {r['r1norm']:<12.4e}")

    print("\nRECOMMENDATION:")
    best = min(results, key=lambda r: r['cond'])
    print(f"  Best conditioning: {best['label']} (κ ≈ {best['cond']:.2e})")
    print(f"  Recommended weights: EqBdiv=1e4, BC_J=1e5, BC_p=1e6")
    print(f"\nNOTE: dolfinx is only available in WSL2 — this script runs on Windows")
    print(f"      with scipy sparse only (no dolfinx import attempted).")


if __name__ == "__main__":
    main()
