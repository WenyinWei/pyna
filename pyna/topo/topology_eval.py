"""topology_eval.py
==================
Fast single-config magnetic topology evaluator — device-agnostic.

Supports HAO, EAST, W7X and any arbitrary MCF device via DeviceConfig.

All heavy numerics go through the cyna C++ extension:
  – trace_poincare_batch_twall    – single-section Poincare (with twall)
  – trace_poincare_multi          – simultaneous multi-section Poincare
  – trace_connection_length_twall – binary search for LCFS seed
  – trace_orbit_along_phi         – dense phi-parameterised orbit (for iota)
  – compute_A_matrix_batch        – Jacobian A(r,z,phi) for DPm integration

Python / scipy are used only as fallbacks when cyna is unavailable.

Target: < 10 s on HAO baseline config.
"""
from __future__ import annotations

import pickle
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

warnings.filterwarnings("ignore")

# ── sys.path ─────────────────────────────────────────────────────────────────
TOPOQUEST = Path(r"C:\Users\Legion\Nutstore\1\Repo\topoquest")
PYNA = Path(r"C:\Users\Legion\Nutstore\1\Repo\pyna")
for _p in (str(TOPOQUEST), str(PYNA)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── cyna import ───────────────────────────────────────────────────────────────
from pyna._cyna import (
    is_available              as _cyna_available,
    trace_poincare_batch_twall    as _cyna_poincare_twall,
    trace_poincare_multi          as _cyna_poincare_multi,
    trace_connection_length_twall as _cyna_connlen_twall,
    trace_orbit_along_phi         as _cyna_trace_orbit,
    compute_A_matrix_batch        as _cyna_A_matrix,
    find_fixed_points_batch       as _cyna_find_fp,
)
_HAS_CYNA = _cyna_available()


# ===========================================================================
# DeviceConfig — device-agnostic parameter container
# ===========================================================================

@dataclass
class DeviceConfig:
    """Device-agnostic parameter configuration for topology evaluation.

    Parameters
    ----------
    R_ax_guess : float
        Initial guess for magnetic axis R [m] (used for Newton refinement).
    Z_ax_guess : float
        Initial guess for magnetic axis Z [m].
    a_minor : float
        Minor radius [m] (used for r-normalisation).
    phi_sections : list
        Poincaré section list, e.g. [0, π/4, π/2, 3π/4].
    wall_file : str or None
        Path to wall geometry file. None → use field-grid boundary.
    fp_pkl_path : str or None
        Path to pre-computed fixed-point pkl. None → live search.
    n_sym : int
        Toroidal symmetry number (W7X=5, HAO=2, EAST=1).
    island_period : int
        Period of outermost island chain (used for X-point DPm integration).
    R_search_min : float
        Lower bound of fixed-point search domain [m].
    R_search_max : float
        Upper bound of fixed-point search domain [m].
    name : str
        Human-readable device name (auto-set in presets).
    """
    R_ax_guess: float
    Z_ax_guess: float
    a_minor: float
    phi_sections: list
    wall_file: Optional[str]
    fp_pkl_path: Optional[str]
    n_sym: int = 1
    island_period: int = 3
    R_search_min: float = 0.5
    R_search_max: float = 2.0
    name: str = "unknown"


# ── HAO preset ────────────────────────────────────────────────────────────────
HAO_CONFIG = DeviceConfig(
    R_ax_guess=0.85235,
    Z_ax_guess=-0.000073,
    a_minor=0.30,
    phi_sections=[0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    wall_file=r'D:\haodata\hao_1stwall_inner.txt',
    fp_pkl_path=r'D:\haodata\fixed_points_all_sections.pkl',
    n_sym=2,
    island_period=3,
    R_search_min=0.7,
    R_search_max=1.2,
    name="HAO",
)

# ── EAST preset (placeholder — fill in later) ─────────────────────────────────
EAST_CONFIG = DeviceConfig(
    R_ax_guess=1.85,
    Z_ax_guess=0.0,
    a_minor=0.45,
    phi_sections=[0.0],
    wall_file=None,
    fp_pkl_path=None,
    n_sym=1,
    island_period=3,
    R_search_min=1.3,
    R_search_max=2.4,
    name="EAST",
)

# ── W7X preset (placeholder — fill in later) ──────────────────────────────────
W7X_CONFIG = DeviceConfig(
    R_ax_guess=5.5,
    Z_ax_guess=0.0,
    a_minor=0.5,
    phi_sections=[0.0, 2 * np.pi / 5 / 2, 2 * np.pi / 5],
    wall_file=None,
    fp_pkl_path=None,
    n_sym=5,
    island_period=5,
    R_search_min=4.8,
    R_search_max=6.2,
    name="W7X",
)


# ===========================================================================
# Output dataclass
# ===========================================================================

@dataclass
class TopologyEval:
    """Result of evaluate_topology().

    Fields that were not requested (compute_* = False) will be None.
    """
    R_ax: float
    Z_ax: float
    lcfs_R: Optional[np.ndarray]   # crossings at phi=0; None if compute_lcfs_contour=False
    lcfs_Z: Optional[np.ndarray]   # crossings at phi=0; None if compute_lcfs_contour=False
    V_lcfs: float                  # m^3 (always computed)
    config_type: str               # 'divertor' or 'limiter'
    iota_profile: Optional[tuple]  # (r_norm_arr, iota_arr); None if compute_iota=False
    xpt_DPm_list: Optional[list]   # list of dicts per X-point; None if compute_xpt_DPm=False
    elapsed_s: float = 0.0
    device_name: str = "unknown"


# ===========================================================================
# Pre-processed field arrays
# ===========================================================================

class _FC:
    """Flat contiguous arrays extracted from a field_cache dict."""

    def __init__(self, fc: dict):
        self.Rg  = np.ascontiguousarray(fc["R_grid"], dtype=np.float64)
        self.Zg  = np.ascontiguousarray(fc["Z_grid"], dtype=np.float64)
        Pg = np.asarray(fc["Phi_grid"])
        self.dphi = float(Pg[1] - Pg[0])
        self.Pg_ext = np.ascontiguousarray(
            np.append(Pg, Pg[-1] + self.dphi), dtype=np.float64
        )
        def _ext(a):
            return np.ascontiguousarray(
                np.concatenate([a, a[:, :, :1]], axis=2), dtype=np.float64
            )
        self._BR_3d   = _ext(fc["BR"])
        self._BPhi_3d = _ext(fc["BPhi"])
        self._BZ_3d   = _ext(fc["BZ"])
        # flat ravel for cyna (C-order, same as ravel())
        self.BR   = self._BR_3d.ravel()
        self.BPhi = self._BPhi_3d.ravel()
        self.BZ   = self._BZ_3d.ravel()

    # Bounding 2D box wall for trace_poincare_multi (conservative, no kill)
    def _box_wall(self):
        R0, R1 = float(self.Rg[0]),  float(self.Rg[-1])
        Z0, Z1 = float(self.Zg[0]), float(self.Zg[-1])
        box_R = np.array([R0, R1, R1, R0, R0], dtype=np.float64)
        box_Z = np.array([Z0, Z0, Z1, Z1, Z0], dtype=np.float64)
        return box_R, box_Z

    def build_scipy_itps(self):
        """Build scipy interpolators for Python fallback paths."""
        if hasattr(self, '_itps_built'):
            return
        kw = dict(method='linear', bounds_error=False, fill_value=0.0)
        self.itp_BR   = RegularGridInterpolator(
            (self.Rg, self.Zg, self.Pg_ext), self._BR_3d,   **kw)
        self.itp_BPhi = RegularGridInterpolator(
            (self.Rg, self.Zg, self.Pg_ext), self._BPhi_3d, **kw)
        self.itp_BZ   = RegularGridInterpolator(
            (self.Rg, self.Zg, self.Pg_ext), self._BZ_3d,   **kw)
        self._itps_built = True

    def field_func_py(self, rzphi):
        """Python field_func: rzphi -> (dR/dl, dZ/dl, dphi/dl)."""
        pt = np.array([[rzphi[0], rzphi[1], rzphi[2] % (2 * np.pi)]])
        BR = float(self.itp_BR(pt))
        BP = float(self.itp_BPhi(pt))
        BZ = float(self.itp_BZ(pt))
        R  = float(rzphi[0])
        Bmod = np.sqrt(BR**2 + BP**2 + BZ**2)
        if Bmod < 1e-12:
            return np.zeros(3)
        return np.array([BR/Bmod, BZ/Bmod, BP/(R*Bmod)])


# ===========================================================================
# Wall arrays
# ===========================================================================

def _find_wall_file_for_device(device: DeviceConfig):
    """Resolve wall file for device; tries device.wall_file then fallbacks."""
    if device.wall_file is not None:
        p = Path(device.wall_file)
        if p.exists():
            return str(p)
        # HAO fallback path
        fallback = TOPOQUEST / "data" / p.name
        if fallback.exists():
            return str(fallback)
        # Could not find the specified wall file
        return None
    return None


def _find_wall_file():
    """Legacy backward-compat: find HAO wall file using HAO_CONFIG defaults."""
    return _find_wall_file_for_device(HAO_CONFIG)


def _load_wall(wall_file: str):
    from topoquest.tracing import WallGeometry
    w = WallGeometry(wall_file)
    phi_c = np.ascontiguousarray(w._phi_centers, dtype=np.float64)
    wR    = np.ascontiguousarray(w._R,           dtype=np.float64)
    wZ    = np.ascontiguousarray(w._Z,           dtype=np.float64)
    return phi_c, wR, wZ, w


def _make_virtual_wall_from_grid(fc: _FC):
    """Build a bounding-box virtual wall from the field grid extents."""
    R0, R1 = float(fc.Rg[0]),  float(fc.Rg[-1])
    Z0, Z1 = float(fc.Zg[0]), float(fc.Zg[-1])
    # Inset slightly so particles hitting the boundary are flagged as lost
    eps = 0.001
    box_R = np.array([R0+eps, R1-eps, R1-eps, R0+eps, R0+eps], dtype=np.float64)
    box_Z = np.array([Z0+eps, Z0+eps, Z1-eps, Z1-eps, Z0+eps], dtype=np.float64)
    # phi_centers: single-section virtual wall (phi-independent)
    phi_c = np.array([0.0], dtype=np.float64)
    return phi_c, box_R, box_Z


# ===========================================================================
# Fixed-point pkl loader
# ===========================================================================

def _load_fp_pkl(device: DeviceConfig):
    """Load fixed-point pkl for the given device config."""
    fp_pkl_path = device.fp_pkl_path

    if fp_pkl_path is not None:
        p = Path(fp_pkl_path)
        if not p.exists():
            # try fallback in topoquest/data
            fallback = TOPOQUEST / "data" / p.name
            p = fallback if fallback.exists() else p
    else:
        p = None

    if p is None or not p.exists():
        return None, {}

    try:
        raw = pickle.load(open(str(p), "rb"))
    except Exception as e:
        print(f"  [fp_pkl] load failed: {e}")
        return None, {}

    phi_sections = device.phi_sections
    fp_by_sec = {}
    for phi in phi_sections:
        k = float(phi)
        if k not in raw:
            k = min(raw.keys(), key=lambda x: abs(x - k))
        sec = raw[k]
        fp_by_sec[float(phi)] = {
            "xpts": [
                {
                    "R": float(t[0]), "Z": float(t[1]),
                    "DPm": (np.array(t[2], dtype=float)
                            if len(t) >= 3 and t[2] is not None else None),
                }
                for t in sec.get("xpts", [])
            ],
            "opts": [{"R": float(t[0]), "Z": float(t[1])}
                     for t in sec.get("opts", [])],
        }
    return str(p), fp_by_sec


# ===========================================================================
# Poincare helpers
# ===========================================================================

def _poincare_single_section(
    R_seeds, Z_seeds, phi_sec, n_turns, fc: _FC,
    phi_c, wall_R, wall_Z, DPhi=0.05
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Single-section Poincare using cyna trace_poincare_batch_twall."""
    R_s = np.ascontiguousarray(R_seeds, dtype=np.float64)
    Z_s = np.ascontiguousarray(Z_seeds, dtype=np.float64)
    counts, pR_flat, pZ_flat = _cyna_poincare_twall(
        R_s, Z_s, float(phi_sec), int(n_turns), float(DPhi),
        fc.BR, fc.BPhi, fc.BZ,
        fc.Rg, fc.Zg, fc.Pg_ext,
        phi_c, wall_R, wall_Z,
    )
    result = []
    for i, cnt in enumerate(counts):
        base = i * int(n_turns)
        cnt  = int(cnt)
        result.append((pR_flat[base:base+cnt], pZ_flat[base:base+cnt]))
    return result


def _poincare_multi_section(
    R_seeds, Z_seeds, n_turns, fc: _FC, phi_sections, DPhi=0.05
) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
    """Multi-section Poincare using cyna trace_poincare_multi.

    Returns result[s_idx][i] = (R_arr, Z_arr).
    """
    Rs = np.ascontiguousarray(R_seeds, dtype=np.float64)
    Zs = np.ascontiguousarray(Z_seeds, dtype=np.float64)
    phi_arr = np.ascontiguousarray(phi_sections, dtype=np.float64)
    n_sec    = len(phi_sections)
    n_seeds  = len(Rs)
    box_R, box_Z = fc._box_wall()

    counts_arr, R_flat, Z_flat = _cyna_poincare_multi(
        Rs, Zs, phi_arr, int(n_turns), float(DPhi),
        fc.BR, fc.BPhi, fc.BZ,
        fc.Rg, fc.Zg, fc.Pg_ext,
        box_R, box_Z,
        -1,   # n_threads
    )
    counts = np.asarray(counts_arr)   # (n_seeds, n_sec)
    R_f    = np.asarray(R_flat)
    Z_f    = np.asarray(Z_flat)

    result = []
    for s in range(n_sec):
        sec_pts = []
        for i in range(n_seeds):
            base = i * n_sec * int(n_turns) + s * int(n_turns)
            cnt  = int(counts[i, s])
            sec_pts.append((R_f[base:base+cnt].copy(),
                            Z_f[base:base+cnt].copy()))
        result.append(sec_pts)
    return result


# ===========================================================================
# Connection-length binary search for LCFS seed
# ===========================================================================

def _is_confined_cyna(R, Z, phi_start, max_turns, fc: _FC,
                       phi_c, wall_R, wall_Z, DPhi=0.05):
    L_fwd, _ = _cyna_connlen_twall(
        np.array([R], dtype=np.float64),
        np.array([Z], dtype=np.float64),
        float(phi_start), int(max_turns), float(DPhi),
        fc.BR, fc.BPhi, fc.BZ,
        fc.Rg, fc.Zg, fc.Pg_ext,
        phi_c, wall_R, wall_Z,
    )
    return bool(L_fwd[0] >= 1e29)


def _find_lcfs_seed(R_ax, Z_ax, R_hi, fc: _FC,
                    phi_c, wall_R, wall_Z,
                    phi_start=0.0, max_turns=200, n_iter=22, DPhi=0.05):
    lo = R_ax + 0.005
    def is_conf(R):
        return _is_confined_cyna(R, Z_ax, phi_start, max_turns, fc,
                                  phi_c, wall_R, wall_Z, DPhi)
    if not is_conf(lo):
        lo = R_ax + 0.001
    if is_conf(R_hi):
        return R_hi
    for _ in range(n_iter):
        mid = 0.5 * (lo + R_hi)
        if is_conf(mid):
            lo = mid
        else:
            R_hi = mid
        if (R_hi - lo) < 1e-4:
            break
    return lo


# ===========================================================================
# Iota profile — from core Poincare crossings at phi=0
# ===========================================================================

def _iota_from_poincare_crossings(
    Rp_arr, Zp_arr, R_ax, Z_ax
) -> float:
    """
    Estimate iota from the Poincare crossings (R, Z) at phi=0 of ONE seed.
    Uses linear-regression method for robustness.
    """
    Rp = np.asarray(Rp_arr)
    Zp = np.asarray(Zp_arr)
    if len(Rp) < 5:
        return float("nan")

    angles = np.arctan2(Zp - Z_ax, Rp - R_ax)
    theta_cum = np.unwrap(angles)
    turns = np.arange(len(theta_cum), dtype=float)
    A = np.column_stack([turns * 2 * np.pi, np.ones(len(turns))])
    result, _, _, _ = np.linalg.lstsq(A, theta_cum, rcond=None)
    return float(result[0])


def _compute_iota_profile_from_poincare(
    core_results_phi0, r_norm_core, R_ax, Z_ax
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute iota profile from already-traced phi=0 Poincare crossings."""
    n = len(r_norm_core)
    r_norm_arr = r_norm_core.copy()
    iota_arr   = np.full(n, np.nan)

    for i, (Rp, Zp) in enumerate(core_results_phi0):
        iota = _iota_from_poincare_crossings(Rp, Zp, R_ax, Z_ax)
        if np.isfinite(iota) and 0.0 < abs(iota) < 10.0:
            iota_arr[i] = iota

    mask = np.isfinite(iota_arr)
    return r_norm_arr[mask], iota_arr[mask]


# ===========================================================================
# DPm computation via cyna trace_orbit_along_phi + compute_A_matrix_batch
# ===========================================================================

def _DPm_from_orbit(R_t, Z_t, phi_t, fc: _FC) -> np.ndarray:
    """
    Integrate dDX/dphi = A(r,z,phi) * DX with DX(phi0)=I.
    Returns 2x2 DPm.
    """
    R_arr   = np.ascontiguousarray(R_t,   dtype=np.float64)
    Z_arr   = np.ascontiguousarray(Z_t,   dtype=np.float64)
    phi_arr = np.ascontiguousarray(phi_t, dtype=np.float64)

    A_batch = _cyna_A_matrix(
        R_arr, Z_arr, phi_arr,
        fc.BR, fc.BPhi, fc.BZ,
        fc.Rg, fc.Zg, fc.Pg_ext,
        1e-4,
    )   # shape (N, 2, 2)

    DX = np.eye(2)
    N = len(phi_arr)
    for k in range(N - 1):
        dphi  = phi_arr[k+1] - phi_arr[k]
        A_mid = 0.5 * (A_batch[k] + A_batch[k+1])
        DX    = DX + dphi * (A_mid @ DX)   # trapezoidal
    return DX


def _compute_DPm_cyna(R_xpt, Z_xpt, fc: _FC,
                      phi0=0.0, island_period=3, DPhi=0.05):
    """Trace orbit for island_period turns, then compute DPm via A-batch."""
    phi_span = float(island_period) * 2.0 * np.pi
    dphi_out = DPhi

    R_t, Z_t, phi_t, _, alive_t = _cyna_trace_orbit(
        float(R_xpt), float(Z_xpt), float(phi0),
        phi_span, dphi_out, 0,
        DPhi, 1e-4,
        fc.BR, fc.BPhi, fc.BZ,
        fc.Rg, fc.Zg, fc.Pg_ext,
    )
    alive = np.asarray(alive_t, dtype=bool)
    R_t   = np.asarray(R_t)[alive]
    Z_t   = np.asarray(Z_t)[alive]
    phi_t = np.asarray(phi_t)[alive]
    if len(R_t) < 5:
        return None
    return _DPm_from_orbit(R_t, Z_t, phi_t, fc)


def _compute_DPm_py(R_xpt, Z_xpt, fc: _FC, island_period=3, DPhi=0.05):
    """Python fallback for DPm using pyna.topo.monodromy."""
    from pyna.topo.monodromy import evolve_DPm_along_cycle
    from pyna.topo.cycle import PeriodicOrbit
    fc.build_scipy_itps()

    phi_span = island_period * 2.0 * np.pi
    h = DPhi
    pts = [[R_xpt, Z_xpt, 0.0]]
    R, Z, phi = R_xpt, Z_xpt, 0.0
    n_steps = int(phi_span / h)

    def g(R_, Z_, phi_):
        pt = np.array([[R_, Z_, phi_ % (2 * np.pi)]])
        BP = float(fc.itp_BPhi(pt))
        if abs(BP) < 1e-20:
            return 0.0, 0.0
        return R_ * float(fc.itp_BR(pt)) / BP, R_ * float(fc.itp_BZ(pt)) / BP

    for _ in range(n_steps):
        dR1, dZ1 = g(R, Z, phi)
        dR2, dZ2 = g(R + 0.5*h*dR1, Z + 0.5*h*dZ1, phi + 0.5*h)
        dR3, dZ3 = g(R + 0.5*h*dR2, Z + 0.5*h*dZ2, phi + 0.5*h)
        dR4, dZ4 = g(R + h*dR3, Z + h*dZ3, phi + h)
        R += h*(dR1 + 2*dR2 + 2*dR3 + dR4)/6
        Z += h*(dZ1 + 2*dZ2 + 2*dZ3 + dZ4)/6
        phi += h
        pts.append([R, Z, phi])

    traj = np.array(pts)
    rzphi0 = np.array([R_xpt, Z_xpt, 0.0])
    orbit = PeriodicOrbit(rzphi0=rzphi0, period_m=island_period,
                          trajectory=traj, DPm=np.eye(2))
    cvd = evolve_DPm_along_cycle(fc.field_func_py, orbit, n_turns=island_period)
    return cvd.DPm


# ===========================================================================
# Shoelace area
# ===========================================================================

def _shoelace_area(R, Z):
    R, Z = np.asarray(R), np.asarray(Z)
    if len(R) < 3:
        return 0.0
    cx, cz = np.mean(R), np.mean(Z)
    ang = np.arctan2(Z - cz, R - cx)
    idx = np.argsort(ang)
    R, Z = R[idx], Z[idx]
    return 0.5 * abs(np.dot(R, np.roll(Z, -1)) - np.dot(Z, np.roll(R, -1)))


# ===========================================================================
# Main API
# ===========================================================================

def evaluate_topology(
    field_cache: dict,
    device: DeviceConfig = HAO_CONFIG,
    # Fine-grained quantity control
    compute_iota: bool = True,          # ι profile
    compute_xpt_DPm: bool = True,       # boundary X-point DPm (incl. λ_u)
    compute_lcfs_contour: bool = True,  # LCFS scatter points; False -> only V
    compute_manifolds: bool = False,    # manifolds (slow, off by default)
    # Accuracy control
    n_core: int = 20,
    n_core_turns: int = 150,
    n_lcfs_turns: int = 200,
    n_iota: int = 15,
    n_iota_turns: int = 100,
    # Legacy / advanced parameters (kept for backward compat)
    fp_pkl_path: Optional[str] = None,
    island_period: Optional[int] = None,
    DPhi: float = 0.05,
) -> TopologyEval:
    """Evaluate magnetic topology for one field configuration.

    Parameters
    ----------
    field_cache : dict
        Field cache dict with keys BR, BPhi, BZ, R_grid, Z_grid, Phi_grid.
    device : DeviceConfig
        Device configuration. Defaults to HAO_CONFIG for backward compat.
    compute_iota : bool
        Whether to compute the ι profile. Default True.
    compute_xpt_DPm : bool
        Whether to compute X-point monodromy matrix DPm (incl. λ_u). Default True.
    compute_lcfs_contour : bool
        Whether to return LCFS scatter points (lcfs_R, lcfs_Z). Default True.
        V_lcfs is always computed regardless of this flag.
    compute_manifolds : bool
        Whether to compute manifolds (slow). Default False.
    n_core : int
        Number of core Poincare seeds (radial sweep).
    n_core_turns : int
        Poincare turns for core seeds. Also used for iota profile.
    n_lcfs_turns : int
        Poincare turns for LCFS seed tracing.
    n_iota : int
        Number of radial iota points (subset of n_core grid).
    n_iota_turns : int
        Ignored (iota derived from already-traced core Poincare crossings).
    fp_pkl_path : str or None
        Legacy override: path to fixed_points_all_sections.pkl.
        If set, overrides device.fp_pkl_path.
    island_period : int or None
        Override device.island_period if set.
    DPhi : float
        RK4 toroidal step [rad].

    Returns
    -------
    TopologyEval
        Fields not requested will be None:
        - iota_profile  = None if compute_iota=False
        - xpt_DPm_list  = None if compute_xpt_DPm=False
        - lcfs_R/lcfs_Z = None if compute_lcfs_contour=False

    Examples
    --------
    # Optuna fast mode (equivalent to old fast_mode=True)
    evaluate_topology(fc, compute_iota=False, compute_xpt_DPm=False, compute_lcfs_contour=False)

    # Full analysis (all defaults True)
    evaluate_topology(fc)

    # Iota + V only, no DPm
    evaluate_topology(fc, compute_xpt_DPm=False)
    """
    t0 = time.time()
    print(f"[topo_eval] device={device.name}  cyna={_HAS_CYNA}  "
          f"iota={compute_iota}  xpt_DPm={compute_xpt_DPm}  lcfs_contour={compute_lcfs_contour}")

    # Resolve island_period: explicit arg > device config
    _island_period = island_period if island_period is not None else device.island_period

    # Legacy fp_pkl_path override
    if fp_pkl_path is not None:
        import copy
        device = copy.replace(device, fp_pkl_path=fp_pkl_path)

    # ── 0. Pre-process field arrays ─────────────────────────────────────────
    fc = _FC(field_cache)

    # ── 1. Load wall ─────────────────────────────────────────────────────────
    wall_file_path = _find_wall_file_for_device(device)
    if wall_file_path is not None:
        phi_c, wall_R, wall_Z, _wall = _load_wall(wall_file_path)
        print(f"[topo_eval] Wall: {wall_file_path}")
    else:
        phi_c, wall_R, wall_Z = _make_virtual_wall_from_grid(fc)
        print(f"[topo_eval] Wall: virtual (grid boundary)")

    # ── 2. Load fixed-point pkl ──────────────────────────────────────────────
    pkl_path, fp_by_sec = _load_fp_pkl(device)
    xpts_phi0 = fp_by_sec.get(0.0, {}).get("xpts", [])
    print(f"[topo_eval] FP pkl: {pkl_path}  (phi=0: {len(xpts_phi0)} X-pts)")

    # ── 3. Core seeds — multi-section Poincare ───────────────────────────────
    phi_sections = device.phi_sections
    R_ax_g = device.R_ax_guess
    Z_ax_g = device.Z_ax_guess
    a_minor = device.a_minor

    print(f"[topo_eval] Core multi-section Poincare: "
          f"{n_core} seeds x {n_core_turns} turns...")
    R_core = np.linspace(R_ax_g + 0.005, R_ax_g + 0.87 * a_minor, n_core)
    Z_core = np.full(n_core, Z_ax_g)
    r_norm_core = (R_core - R_ax_g) / a_minor

    if _HAS_CYNA and _cyna_poincare_multi is not None:
        core_by_sec = _poincare_multi_section(
            R_core, Z_core, n_core_turns, fc, phi_sections, DPhi
        )
    else:
        from topoquest.tracing import MultiSectionPoincare
        msp = MultiSectionPoincare(field_cache, wall_file_path, phi_sections,
                                   wall_inset=0.005)
        core_by_sec = msp.run(R_core.tolist(), Z_core.tolist(), n_core_turns)

    core_phi0 = core_by_sec[0]  # list of (R_arr, Z_arr) per seed at phi=0

    # ── 4. Magnetic axis (centroid of innermost traces at phi=0) ─────────────
    iR, iZ = [], []
    for i, (Rp, Zp) in enumerate(core_phi0):
        if r_norm_core[i] < 0.1 and len(Rp) > 2:
            iR.extend(Rp); iZ.extend(Zp)
    R_ax = float(np.mean(iR)) if iR else R_ax_g
    Z_ax = float(np.mean(iZ)) if iZ else Z_ax_g
    print(f"[topo_eval] Axis: R={R_ax:.5f}, Z={Z_ax:.6f}  (t={time.time()-t0:.1f}s)")

    # ── 5. Iota profile ──────────────────────────────────────────────────────
    if compute_iota:
        iota_step = max(1, n_core // n_iota)
        iota_idx  = np.arange(0, n_core, iota_step)[:n_iota]
        r_norm_iota = r_norm_core[iota_idx]
        iota_arr_all = np.full(len(iota_idx), np.nan)
        for k, i in enumerate(iota_idx):
            Rp, Zp = core_phi0[i]
            iota_arr_all[k] = _iota_from_poincare_crossings(Rp, Zp, R_ax, Z_ax)
        mask = np.isfinite(iota_arr_all)
        r_norm_arr = r_norm_iota[mask]
        iota_arr   = iota_arr_all[mask]
        print(f"[topo_eval] Iota profile: {len(iota_arr)} pts  "
              f"range=[{(np.min(iota_arr) if len(iota_arr) else float('nan')):.3f}, "
              f"{(np.max(iota_arr) if len(iota_arr) else float('nan')):.3f}]")
        iota_profile = (r_norm_arr, iota_arr)
    else:
        iota_profile = None

    # ── 6. LCFS binary search ────────────────────────────────────────────────
    print("[topo_eval] LCFS binary search...")
    if xpts_phi0:
        R_xpt_max = max(fp["R"] for fp in xpts_phi0)
        R_hi = R_xpt_max - 0.008
        print(f"  Outermost X-pt R={R_xpt_max:.5f} -> R_hi={R_hi:.5f}")
    else:
        R_hi = R_ax + a_minor

    R0_seed = _find_lcfs_seed(
        R_ax, Z_ax, R_hi, fc, phi_c, wall_R, wall_Z,
        max_turns=200, DPhi=DPhi
    )
    print(f"  LCFS seed: R={R0_seed:.5f}  (t={time.time()-t0:.1f}s)")

    # ── 7. LCFS multi-section Poincare ───────────────────────────────────────
    print(f"[topo_eval] LCFS multi-section Poincare: {n_lcfs_turns} turns...")
    if _HAS_CYNA and _cyna_poincare_multi is not None:
        lcfs_by_sec_raw = _poincare_multi_section(
            np.array([R0_seed]), np.array([Z_ax]),
            n_lcfs_turns, fc, phi_sections, DPhi
        )
        lcfs_by_sec = [lcfs_by_sec_raw[s][0] for s in range(len(phi_sections))]
    else:
        lcfs_by_sec = []
        for phi_sec in phi_sections:
            pts = _poincare_single_section(
                np.array([R0_seed]), np.array([Z_ax]),
                phi_sec, n_lcfs_turns, fc, phi_c, wall_R, wall_Z, DPhi
            )
            lcfs_by_sec.append(pts[0])

    lcfs_R0, lcfs_Z0 = lcfs_by_sec[0]
    if compute_lcfs_contour:
        lcfs_R = np.asarray(lcfs_R0) if len(lcfs_R0) else np.array([R0_seed])
        lcfs_Z = np.asarray(lcfs_Z0) if len(lcfs_Z0) else np.array([Z_ax])
    else:
        lcfs_R = None
        lcfs_Z = None
    n_cross = len(lcfs_R0) if compute_lcfs_contour else 0
    print(f"  phi=0: {n_cross} crossings  (t={time.time()-t0:.1f}s)")

    # ── 8. Volume (multi-section shoelace + cylindrical mean) ────────────────
    areas   = [_shoelace_area(R, Z) for R, Z in lcfs_by_sec]
    R_cents = [float(np.mean(R)) if len(R) > 0 else R_ax for R, Z in lcfs_by_sec]
    V_lcfs  = 2 * np.pi * float(np.mean([A * Rc for A, Rc in zip(areas, R_cents)]))
    areas_cm2 = [f"{a*1e4:.1f}" for a in areas]
    print(f"[topo_eval] V_lcfs = {V_lcfs*1e3:.3f} L  (areas: {areas_cm2} cm2)")

    # ── 9. Config type ───────────────────────────────────────────────────────
    has_xpt = len(xpts_phi0) > 0
    config_type = "divertor" if (has_xpt and V_lcfs > 0.05) else "limiter"
    print(f"[topo_eval] Config: {config_type}  "
          f"(X-pts={len(xpts_phi0)}, V={V_lcfs:.4f} m3)")

    # ── 10. Boundary X-point DPm ─────────────────────────────────────────────
    xpt_DPm_list = None
    if compute_xpt_DPm:
        print("[topo_eval] X-point DPm...")
        outer_xpts = [fp for fp in xpts_phi0 if fp["R"] > R_ax + 0.15]
        print(f"  Outer X-pts (R>{R_ax+0.15:.3f}): {len(outer_xpts)}")
        xpt_DPm_list = []
        for fp in outer_xpts:
            R_x, Z_x = fp["R"], fp["Z"]
            DPm_pkl   = fp.get("DPm")

            if DPm_pkl is not None and not np.allclose(DPm_pkl, np.eye(2)):
                DPm = DPm_pkl;  src = "pkl"
            elif _HAS_CYNA and _cyna_trace_orbit is not None and _cyna_A_matrix is not None:
                DPm = _compute_DPm_cyna(R_x, Z_x, fc, phi0=0.0,
                                         island_period=_island_period, DPhi=DPhi)
                src = "cyna"
            else:
                try:
                    DPm = _compute_DPm_py(R_x, Z_x, fc,
                                           island_period=_island_period, DPhi=DPhi)
                    src = "py"
                except Exception as e:
                    print(f"    DPm failed for ({R_x:.4f},{Z_x:.4f}): {e}")
                    continue

            if DPm is None:
                print(f"    DPm=None for ({R_x:.4f},{Z_x:.4f}), skipping")
                continue

            eigvals       = np.linalg.eigvals(DPm)
            idx_u         = int(np.argmax(np.abs(eigvals)))
            lambda_u      = float(np.real(eigvals[idx_u]))
            tr            = float(np.trace(DPm))
            stability_idx = tr / 2.0
            greene_res    = (2.0 - tr) / 4.0

            xpt_DPm_list.append({
                "R": R_x, "Z": Z_x,
                "DPm": DPm,
                "lambda_u": lambda_u,
                "stability_index": stability_idx,
                "greene_residue": greene_res,
            })
            print(f"  ({R_x:.4f},{Z_x:.4f}) [{src}]  "
                  f"lu={lambda_u:.4f}  SI={stability_idx:.4f}  GR={greene_res:.4f}")

    elapsed = time.time() - t0
    print(f"[topo_eval] Done in {elapsed:.1f}s")

    return TopologyEval(
        R_ax=R_ax, Z_ax=Z_ax,
        lcfs_R=lcfs_R, lcfs_Z=lcfs_Z,
        V_lcfs=V_lcfs,
        config_type=config_type,
        iota_profile=iota_profile,
        xpt_DPm_list=xpt_DPm_list,
        elapsed_s=elapsed,
        device_name=device.name,
    )
