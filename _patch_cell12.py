"""Patch Cell 12 of rmp_island_validation_solovev.ipynb with cache-first logic."""
import sys, json, pathlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

nb_path = pathlib.Path("D:/Repo/pyna/notebooks/tutorials/rmp_island_validation_solovev.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

# Find Cell 12 (single-null X-point manifolds)
all_cells = nb["cells"]
code_cells_idx = [i for i, c in enumerate(all_cells) if c["cell_type"] == "code"]
cell12_nb_idx = code_cells_idx[12]  # 13th code cell

NEW_SOURCE = """\
# =========================================================================
# Single-null divertor equilibrium: separatrix X-point & stable/unstable manifolds
# =========================================================================
import json as _json
import pathlib as _pathlib
import warnings
warnings.filterwarnings("ignore")

from pyna.MCF.equilibrium.Solovev import solovev_single_null
from pyna.topo.variational import PoincareMapVariationalEquations
from pyna.topo.manifold_improve import StableManifold, UnstableManifold
from pyna.MCF.visual.tokamak_manifold import _manifold_line_collection

_CACHE_SN = _pathlib.Path("pyna_output/solovev_sn_manifolds.json")
_CACHE_SN.parent.mkdir(exist_ok=True)

# -----------------------------------------------------------------------
# 1. Build single-null equilibrium (fast, analytic)
# -----------------------------------------------------------------------
eq_sn = solovev_single_null(
    R0=1.86, a=0.595, B0=5.3,
    kappa=1.8, delta_u=0.33, delta_l=0.40, kappa_x=1.5,
    q0=1.5,
)
R_ax_sn, Z_ax_sn = eq_sn.magnetic_axis
print(f'Magnetic axis: R={R_ax_sn:.4f} m  Z={Z_ax_sn:.4f} m')
R_xpt_sn, Z_xpt_sn = eq_sn.find_xpoint()
print(f'X-point: R={R_xpt_sn:.4f} m  Z={Z_xpt_sn:.4f} m')

# -----------------------------------------------------------------------
# 2. Field function (dR/dphi, dZ/dphi) for Poincare map
# -----------------------------------------------------------------------
def _field_func_sn(rzphi):
    R, Z = float(rzphi[0]), float(rzphi[1])
    BR, BZ = eq_sn.BR_BZ(np.array([R]), np.array([Z]))
    Bphi_v = eq_sn.Bphi(np.array([R]))
    brt, bzt, bpt = float(BR[0]), float(BZ[0]), float(Bphi_v[0])
    bm = np.sqrt(brt**2 + bzt**2 + bpt**2) + 1e-30
    return np.array([brt/bm, bzt/bm, bpt/(R*bm)])

def field_func_2d_sn(R, Z, phi):
    t = _field_func_sn(np.array([R, Z, phi]))
    if abs(t[2]) < 1e-15:
        return np.array([0.0, 0.0])
    return np.array([t[0]/t[2], t[1]/t[2]])

phi_span_sn = (0.0, 2.0 * np.pi)
xpt_sn = np.array([R_xpt_sn, Z_xpt_sn])
RZlim_sn = (eq_sn.R0 - 1.6*eq_sn.a, eq_sn.R0 + 1.6*eq_sn.a,
            -2.2*eq_sn.kappa*eq_sn.a, 1.8*eq_sn.kappa*eq_sn.a)

# -----------------------------------------------------------------------
# 3. Monodromy matrix + manifolds  (cache-first for CI speed)
# -----------------------------------------------------------------------
if _CACHE_SN.exists():
    _d = _json.loads(_CACHE_SN.read_text())
    det_sn     = _d["det_J"]
    lam_abs_sn = [_d["lam_stable"], _d["lam_unstable"]]
    sm_segs    = [np.array(s) for s in _d["sm_segments"]]
    um_segs    = [np.array(s) for s in _d["um_segments"]]
    print(f"Loaded manifolds from cache.  det(J)={det_sn:.6f}  |lam|={lam_abs_sn}")
else:
    vq_sn = PoincareMapVariationalEquations(field_func_2d_sn, fd_eps=1e-5)
    Jac_sn = vq_sn.jacobian_matrix(
        xpt_sn, phi_span_sn,
        solve_ivp_kwargs=dict(method='RK45', rtol=1e-5, atol=1e-7),
    )
    lam_sn = np.linalg.eigvals(Jac_sn)
    det_sn = float(np.linalg.det(Jac_sn))
    lam_abs_sn = sorted(np.abs(lam_sn))
    print(f"det(J)={det_sn:.6f}  |lam|={lam_abs_sn}")

    sm_sn = StableManifold(xpt_sn, Jac_sn, field_func_2d_sn, phi_span=phi_span_sn)
    um_sn = UnstableManifold(xpt_sn, Jac_sn, field_func_2d_sn, phi_span=phi_span_sn)
    sm_sn.grow(n_turns=1, init_length=1e-3, n_init_pts=2, both_sides=False,
               RZlimit=RZlim_sn, rtol=1e-4, atol=1e-6)
    um_sn.grow(n_turns=1, init_length=1e-3, n_init_pts=2, both_sides=False,
               RZlimit=RZlim_sn, rtol=1e-4, atol=1e-6)
    sm_segs = [s for s in sm_sn.segments if len(s) >= 2]
    um_segs = [s for s in um_sn.segments if len(s) >= 2]
    _CACHE_SN.write_text(_json.dumps({
        "R_ax": float(R_ax_sn), "Z_ax": float(Z_ax_sn),
        "R_xpt": float(R_xpt_sn), "Z_xpt": float(Z_xpt_sn),
        "det_J": det_sn,
        "lam_stable": float(lam_abs_sn[0]), "lam_unstable": float(lam_abs_sn[1]),
        "sm_segments": [s.tolist() for s in sm_segs],
        "um_segments": [s.tolist() for s in um_segs],
    }))
    print(f"Computed and cached.  sm={len(sm_segs)}  um={len(um_segs)}")

# -----------------------------------------------------------------------
# 4. Grid for equilibrium contours
# -----------------------------------------------------------------------
R_range_sn = (eq_sn.R0 - 1.4*eq_sn.a, eq_sn.R0 + 1.4*eq_sn.a)
Z_range_sn = (-1.8*eq_sn.kappa*eq_sn.a, 1.2*eq_sn.kappa*eq_sn.a)
Nr_sn, Nz_sn = 200, 250
Rg_sn = np.linspace(*R_range_sn, Nr_sn)
Zg_sn = np.linspace(*Z_range_sn, Nz_sn)
Rg_sn, Zg_sn = np.meshgrid(Rg_sn, Zg_sn)
psi_g_sn = eq_sn.psi(Rg_sn.ravel(), Zg_sn.ravel()).reshape(Nz_sn, Nr_sn)

# -----------------------------------------------------------------------
# 5. Plot
# -----------------------------------------------------------------------
fig, ax2 = plt.subplots(figsize=(7, 9))

ax2.contour(Rg_sn, Zg_sn, psi_g_sn, levels=np.linspace(0.05, 0.95, 15),
            colors='lightgray', linewidths=0.5)
ax2.contour(Rg_sn, Zg_sn, psi_g_sn, levels=[1.0], colors='k', linewidths=1.5)

# Stable manifold (blue/teal)
for seg in sm_segs:
    if len(seg) > 2:
        lc, _ = _manifold_line_collection(seg, 'GnBu',
                    s_ref=max(np.ptp(seg[:, 0]), 1e-6))
        lc.set_linewidth(1.3); lc.set_alpha(0.92); lc.set_zorder(6)
        ax2.add_collection(lc)

# Unstable manifold (orange)
for seg in um_segs:
    if len(seg) > 2:
        lc, _ = _manifold_line_collection(seg, 'Oranges',
                    s_ref=max(np.ptp(seg[:, 0]), 1e-6))
        lc.set_linewidth(1.3); lc.set_alpha(0.92); lc.set_zorder(6)
        ax2.add_collection(lc)

ax2.scatter([R_xpt_sn], [Z_xpt_sn], s=80, c='blue', marker='x', zorder=7, lw=2)
ax2.plot(R_ax_sn, Z_ax_sn, '+k', ms=10, mew=2)

ax2.set_aspect('equal')
ax2.set_xlabel('R (m)'); ax2.set_ylabel('Z (m)')
ax2.set_title('Single-null X-point: Stable/Unstable Manifolds')
ax2.set_xlim(R_range_sn); ax2.set_ylim(Z_range_sn)
plt.tight_layout()
plt.savefig('pyna_output/solovev_sn_manifolds.png', dpi=150)
plt.show()
print("Saved: pyna_output/solovev_sn_manifolds.png")
"""

# Convert to proper list-of-lines source format
new_source_lines = [line + "\n" for line in NEW_SOURCE.rstrip("\n").split("\n")]
new_source_lines[-1] = new_source_lines[-1].rstrip("\n")  # last line no trailing newline

all_cells[cell12_nb_idx]["source"] = new_source_lines
all_cells[cell12_nb_idx]["outputs"] = []
all_cells[cell12_nb_idx]["execution_count"] = None

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Patched cell {cell12_nb_idx} (code cell 12). Lines: {len(new_source_lines)}")
print("Done.")
