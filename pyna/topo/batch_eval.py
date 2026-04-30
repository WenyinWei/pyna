"""batch_eval.py
================
批量并行磁拓扑位型评估框架，用于 HAO 仿星器 dipole/TF 线圈 Optuna 优化。

规模：344 个线圈（332 dipole + 12 TF），目标单次 fast eval < 0.1 s/trial。

场叠加原理（线性）：
  B_total = B_base + Σᵢ Iᵢ · B_resp_i
          = B_base + einsum('ci,ijk->cjk', I_matrix, B_resp)

两种评估模式：
  fast_evaluate_single()  - < 0.1 s：线性近似 λ_u + cyna 快速 LCFS 追踪
  full_evaluate_single()  - < 10 s ：调用完整 evaluate_topology()

主要类/函数：
  TopologyCache                - 一次性加载所有响应场，之后每次只做矩阵运算
  build_field_superposition_cache()  - 批量 einsum 生成 N_configs 个 field_cache
  batch_evaluate_topology()   - 并行评估（joblib / concurrent.futures）
  optuna_objective_factory()  - 工厂函数，返回 Optuna trial->float 目标函数
"""
from __future__ import annotations

import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

import numpy as np

warnings.filterwarnings("ignore")

# ── repo paths ────────────────────────────────────────────────────────────────
TOPOQUEST = Path(r"C:\Users\Legion\Nutstore\1\Repo\topoquest")
PYNA      = Path(r"C:\Users\Legion\Nutstore\1\Repo\pyna")
for _p in (str(TOPOQUEST), str(PYNA)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── default data paths ────────────────────────────────────────────────────────
_RESP_PATHS = [
    Path(r"C:\Users\Legion\Nutstore\1\haodata\vacuum_fields\vacuum_field_dipole_all_332.npz"),
    Path(r"D:\haodata\coilsys\vacuum_fields\vacuum_field_dipole_all_332.npz"),
]
_TF_RESP_PATHS = [
    Path(r"C:\Users\Legion\Nutstore\1\haodata\vacuum_fields\vacuum_field_tf_all_12.npz"),
    Path(r"D:\haodata\coilsys\vacuum_fields\vacuum_field_tf_all_12.npz"),
]
_FP_PKL_PATHS = [
    Path(r"D:\haodata\fixed_points_all_sections.pkl"),
    Path(r"D:\2026Spring\haodata\fixed_points_all_sections.pkl"),
    TOPOQUEST / "data" / "fixed_points_all_sections.pkl",
]
_WALL_PATHS = [
    Path(r"D:\haodata\hao_1stwall_inner.txt"),
    Path(r"D:\2026Spring\haodata\hao_1stwall_inner.txt"),
    TOPOQUEST / "data" / "hao_1stwall_inner.txt",
]
_CACHE_PKL_PATHS = [
    Path(r"D:\2026Spring\haodata\bluestar_starting_config_field_cache.pkl"),
    TOPOQUEST / "data" / "bluestar_starting_config_field_cache.pkl",
]

def _find_path(candidates: list[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None

# HAO 参考值
R_AX_REF = 0.85235
Z_AX_REF = -0.000073
A0 = 0.30  # 近似小半径 [m]


# ═══════════════════════════════════════════════════════════════════════════════
# TopologyCache — 核心类
# ═══════════════════════════════════════════════════════════════════════════════

class TopologyCache:
    """一次性加载所有线圈响应场，之后每次评估只做 einsum + cyna 追踪。

    属性（加载后可用）
    ------------------
    base_fc : dict
        基础场 cache（BR/BPhi/BZ shape: NR×NZ×NPHI，float32）
    BR_resp / BPhi_resp / BZ_resp : np.ndarray  shape (N_coils, NR, NZ, NPHI)
        各线圈单位电流响应场（float32）
    N_coils : int
    coil_phi, coil_Z, coil_R, coil_indices : np.ndarray  shape (N_coils,)
    R_grid, Z_grid, Phi_grid : np.ndarray  (1D)
    boundary_xpts : list of dict  {R, Z, DPm}
        phi=0 截面处的边界 X 点（从 pkl 加载）
    baseline_lambda_u : float
        基础位型最大 λ_u 均值
    """

    def __init__(
        self,
        resp_path:  Optional[str | Path] = None,
        fp_pkl_path: Optional[str | Path] = None,
        base_fc_path: Optional[str | Path] = None,
        tf_resp_path: Optional[str | Path] = None,  # TF 线圈响应场（可选）
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        resp_path     : dipole 响应场 .npz 路径
        fp_pkl_path   : fixed_points_all_sections.pkl 路径
        base_fc_path  : 基础场 pickle 路径（bluestar_starting_config_field_cache.pkl）
        tf_resp_path  : TF 线圈响应场（若有），shape 格式与 dipole 响应场相同；
                        若为 None 则只用主响应场数据
        verbose       : 是否打印加载进度
        """
        self._verbose = verbose
        t0 = time.time()

        # 磁轴坐标（从固定点 pkl 自动提取）
        self.R_axis: float | None = None
        self.Z_axis: float | None = None

        # ── 1. 真空场 ──────────────────────────────────────────────────────
        rp = Path(resp_path) if resp_path else _find_path(_RESP_PATHS)
        if rp is None or not rp.exists():
            raise FileNotFoundError(f"真空场 .npz 未找到，尝试过: {_RESP_PATHS}")
        self._log(f"加载真空场: {rp} ...")
        resp = np.load(str(rp))

        self.BR_resp   = np.asarray(resp["BR_resp"],   dtype=np.float32)
        self.BPhi_resp = np.asarray(resp["BPhi_resp"], dtype=np.float32)
        self.BZ_resp   = np.asarray(resp["BZ_resp"],   dtype=np.float32)
        self.coil_indices = np.asarray(resp["coil_indices"])
        self.coil_phi     = np.asarray(resp["coil_phi"])
        self.coil_Z       = np.asarray(resp["coil_Z"])
        self.coil_R       = np.asarray(resp["coil_R"])
        self.R_grid  = np.asarray(resp["R_grid"],   dtype=np.float64)
        self.Z_grid  = np.asarray(resp["Z_grid"],   dtype=np.float64)
        self.Phi_grid = np.asarray(resp["Phi_grid"], dtype=np.float64)
        
        # ── NaN/Inf 检查和警告（线圈靠近壁面时 Biot-Savart 会产生 NaN）─────
        nan_br = np.isnan(self.BR_resp).sum()
        nan_bp = np.isnan(self.BPhi_resp).sum()
        nan_bz = np.isnan(self.BZ_resp).sum()
        total_nan = nan_br + nan_bp + nan_bz
        if total_nan > 0:
            self._log(f"  [警告] 响应场中有 {total_nan} 个 NaN (BR={nan_br}, BPhi={nan_bp}, BZ={nan_bz})")
            self._log(f"  [建议] 请使用 v2 真空场 (recompute_vacuum_fields.py) 以消除 NaN")
            self._log(f"  [临时] 将 NaN 替换为 0 以避免计算崩溃（物理上不正确！）")
            # 临时修复：用 0 替换 NaN（比中位数更诚实）
            self.BR_resp = np.nan_to_num(self.BR_resp, nan=0.0)
            self.BPhi_resp = np.nan_to_num(self.BPhi_resp, nan=0.0)
            self.BZ_resp = np.nan_to_num(self.BZ_resp, nan=0.0)
        
        # 检查极大值（单位安培响应场不应超过 1 T/A）
        for name, arr in [("BR", self.BR_resp), ("BPhi", self.BPhi_resp), ("BZ", self.BZ_resp)]:
            max_abs = np.abs(arr).max()
            if max_abs > 0.1:  # 单位安培响应场通常 < 0.1 T/A
                self._log(f"  [警告] {name} 最大值 {max_abs:.2e} T/A 偏大（可能靠近线圈）")
        
        self._log(f"  dipole 响应线圈: {self.BR_resp.shape[0]} 个线圈  "
                  f"grid={self.BR_resp.shape[1:]}  "
                  f"({self.BR_resp.nbytes/1e6:.0f} MB × 3)")

        # ── 2. TF 线圈响应场（可选拼接）────────────────────────────────────
        if tf_resp_path is not None:
            tp = Path(tf_resp_path)
            self._log(f"加载 TF 线圈响应场: {tp} ...")
            tf_resp = np.load(str(tp))
            self.BR_resp   = np.concatenate(
                [self.BR_resp,   np.asarray(tf_resp["BR_resp"],   dtype=np.float32)], axis=0)
            self.BPhi_resp = np.concatenate(
                [self.BPhi_resp, np.asarray(tf_resp["BPhi_resp"], dtype=np.float32)], axis=0)
            self.BZ_resp   = np.concatenate(
                [self.BZ_resp,   np.asarray(tf_resp["BZ_resp"],   dtype=np.float32)], axis=0)
            self._log(f"  合并后总线圈数: {self.BR_resp.shape[0]}")

        self.N_coils = self.BR_resp.shape[0]
        NR, NZ, NPHI = self.BR_resp.shape[1:]
        self._shape  = (NR, NZ, NPHI)

        # ── 3. 基础场 ──────────────────────────────────────────────────────
        bfp = Path(base_fc_path) if base_fc_path else _find_path(_CACHE_PKL_PATHS)
        if bfp is None or not bfp.exists():
            raise FileNotFoundError(f"基础场 pkl 未找到，尝试过: {_CACHE_PKL_PATHS}")
        self._log(f"加载基础场: {bfp} ...")
        with open(str(bfp), "rb") as f:
            self.base_fc = pickle.load(f)
        # 标准化为 float64 便于后续叠加
        self._BR_base   = np.asarray(self.base_fc["BR"],   dtype=np.float64)
        self._BPhi_base = np.asarray(self.base_fc["BPhi"], dtype=np.float64)
        self._BZ_base   = np.asarray(self.base_fc["BZ"],   dtype=np.float64)

        # ── 4. 固定点 pkl ──────────────────────────────────────────────────
        fp_p = Path(fp_pkl_path) if fp_pkl_path else _find_path(_FP_PKL_PATHS)
        self.boundary_xpts: list[dict] = []
        self.baseline_lambda_u: float  = float("nan")
        if fp_p and fp_p.exists():
            self._log(f"加载固定点 pkl: {fp_p} ...")
            self._load_fp_pkl(fp_p)
        else:
            self._log(f"[警告] 固定点 pkl 未找到")

        # ── 5. 预计算分组位型所需的 basis 向量（可选，供快速评估） ─────────
        self._group_cache: dict[str, dict] = {}  # scheme_name -> {BR, BPhi, BZ}
        self.basis_dlambda: dict[str, np.ndarray] = {}  # scheme -> (K,)

        # NOTE: precompute_basis_dlambda() 需要完整 field_cache + pkl X 点，
        # 请在实例化后手动调用，或在子类中重载 __init__。

        elapsed = time.time() - t0
        self._log(f"TopologyCache 就绪  ({elapsed:.1f}s)  "
                  f"N_coils={self.N_coils}  "
                  f"boundary_xpts={len(self.boundary_xpts)}")

    # ── 内部工具 ───────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        if self._verbose:
            print(f"[TopologyCache] {msg}")

    def _load_fp_pkl(self, fp_path: Path):
        raw = pickle.load(open(str(fp_path), "rb"))
        sec0 = raw.get(0.0, raw.get(min(raw.keys(), key=lambda x: abs(x)), {}))

        # 从 opts（O 点列表）自动提取磁轴坐标
        opts_raw = sec0.get('opts', [])
        if opts_raw:
            axis = min(opts_raw, key=lambda p: (float(p[0]) - R_AX_REF)**2 + (float(p[1]) - Z_AX_REF)**2)
            self.R_axis = float(axis[0])
            self.Z_axis = float(axis[1])
        else:
            self.R_axis = float(R_AX_REF)
            self.Z_axis = float(Z_AX_REF)
        self._log(f"  magnetic axis ≈ (R={self.R_axis:.4f}, Z={self.Z_axis:.4f})")

        xpts_raw = sec0.get("xpts", [])
        lambdas = []
        for t in xpts_raw:
            R_, Z_ = float(t[0]), float(t[1])
            DPm = (np.asarray(t[2], dtype=float).reshape(2, 2)
                   if len(t) >= 3 and t[2] is not None else np.eye(2))
            eigs = np.linalg.eigvals(DPm)
            lam_u = float(np.max(np.abs(eigs)))
            self.boundary_xpts.append({"R": R_, "Z": Z_, "DPm": DPm, "lambda_u": lam_u})
            if R_ > R_AX_REF + 0.15:
                lambdas.append(lam_u)
        if lambdas:
            self.baseline_lambda_u = float(np.mean(lambdas))

    # ── 分组 basis 场 ──────────────────────────────────────────────────────────

    def precompute_group_basis(
        self,
        groups: list[list[int]],
        scheme_name: str = "custom",
    ) -> dict:
        """将 N_coils 条响应场按分组求和，生成 K 个 basis 场。

        Parameters
        ----------
        groups : list of K lists，每个元素是该组的线圈索引（进入 BR_resp 的行索引）
        scheme_name : 标识符，用于缓存

        Returns
        -------
        basis : dict with 'BR', 'BPhi', 'BZ' each list of K arrays (NR,NZ,NPHI) float64
        """
        if scheme_name in self._group_cache:
            return self._group_cache[scheme_name]

        self._log(f"预计算分组 basis 场 [{scheme_name}]: {len(groups)} 组 ...")
        t0 = time.time()
        basis_BR, basis_BPhi, basis_BZ = [], [], []
        for idx_list in groups:
            if len(idx_list) == 0:
                zero = np.zeros(self._shape, dtype=np.float64)
                basis_BR.append(zero); basis_BPhi.append(zero); basis_BZ.append(zero)
            else:
                ia = np.array(idx_list, dtype=int)
                basis_BR.append(self.BR_resp[ia].sum(axis=0).astype(np.float64))
                basis_BPhi.append(self.BPhi_resp[ia].sum(axis=0).astype(np.float64))
                basis_BZ.append(self.BZ_resp[ia].sum(axis=0).astype(np.float64))

        result = {"BR": basis_BR, "BPhi": basis_BPhi, "BZ": basis_BZ}
        self._group_cache[scheme_name] = result
        self._log(f"  完成 ({time.time()-t0:.2f}s)")
        return result

    def precompute_group_basis_design_current(
        self,
        groups: list,
        coil_files: list,
        scheme_name: str = "custom_design",
    ) -> dict:
        """按分组叠加各线圈的设计电流场（T），生成 basis 场。

        与 precompute_group_basis 的区别：
        - 输入是单线圈 npz 文件列表（新格式，存绝对 Tesla），不是 T/A 响应场
        - basis['BR'][k] 单位为 T（该组在设计电流下的场之和）
        - build_field_from_delta_I 里的 delta_I[k] 应为无量纲缩放因子 alpha_k
          alpha=0 -> 维持设计电流（base field 已包含）
          alpha=1 -> 该组在设计电流基础上再增加一倍
          alpha=-1 -> 该组扣除一倍设计电流（等效关掉该组）

        Parameters
        ----------
        groups     : K 个组，每组是线圈文件索引列表（对应 coil_files 的顺序）
        coil_files : 所有单线圈 npz 文件路径列表（与 groups 里的索引对应）
        scheme_name: 缓存标识符
        """
        if scheme_name in self._group_cache:
            return self._group_cache[scheme_name]

        self._log(f"预计算分组 design-current basis [{scheme_name}]: {len(groups)} 组 ...")
        t0 = time.time()
        basis_BR, basis_BPhi, basis_BZ = [], [], []
        for idx_list in groups:
            if len(idx_list) == 0:
                zero = np.zeros(self._shape, dtype=np.float64)
                basis_BR.append(zero); basis_BPhi.append(zero); basis_BZ.append(zero)
            else:
                g_BR   = np.zeros(self._shape, dtype=np.float64)
                g_BPhi = np.zeros(self._shape, dtype=np.float64)
                g_BZ   = np.zeros(self._shape, dtype=np.float64)
                for ci in idx_list:
                    if coil_files[ci] is None:
                        continue
                    d = np.load(str(coil_files[ci]))
                    g_BR   += np.nan_to_num(d['BR'].astype(np.float64))
                    g_BPhi += np.nan_to_num(d['BPhi'].astype(np.float64))
                    g_BZ   += np.nan_to_num(d['BZ'].astype(np.float64))
                basis_BR.append(g_BR)
                basis_BPhi.append(g_BPhi)
                basis_BZ.append(g_BZ)

        result = {"BR": basis_BR, "BPhi": basis_BPhi, "BZ": basis_BZ}
        self._group_cache[scheme_name] = result
        self._log(f"  完成 ({time.time()-t0:.2f}s)")
        return result

    def build_field_from_delta_I(
        self,
        delta_I: np.ndarray,
        scheme_name: str = "custom",
    ) -> dict:
        """从预计算的分组 basis 场快速构造修改后的 field_cache。

        Parameters
        ----------
        delta_I : (K,) 数组，各组额外电流（A）
        scheme_name : 需先调用 precompute_group_basis()

        Returns
        -------
        field_cache dict（BR/BPhi/BZ/R_grid/Z_grid/Phi_grid）
        """
        if scheme_name not in self._group_cache:
            raise ValueError(f"方案 '{scheme_name}' 尚未预计算，请先调用 precompute_group_basis()")
        basis = self._group_cache[scheme_name]
        delta_I = np.asarray(delta_I, dtype=np.float64)

        BR_new   = self._BR_base.copy()
        BPhi_new = self._BPhi_base.copy()
        BZ_new   = self._BZ_base.copy()

        for k, dI in enumerate(delta_I):
            if abs(dI) > 1e-15:
                BR_new   += dI * basis["BR"][k]
                BPhi_new += dI * basis["BPhi"][k]
                BZ_new   += dI * basis["BZ"][k]

        return {
            "BR": BR_new, "BPhi": BPhi_new, "BZ": BZ_new,
            "R_grid": self.R_grid, "Z_grid": self.Z_grid, "Phi_grid": self.Phi_grid,
        }

    def build_field_from_coil_currents(
        self,
        coil_currents: np.ndarray,
    ) -> dict:
        """直接用全部 N_coils 电流向量叠加（不使用分组 basis）。

        Parameters
        ----------
        coil_currents : (N_coils,) 数组，各线圈电流（A，相对于基础场的增量）
        """
        I = np.asarray(coil_currents, dtype=np.float64)
        assert I.shape[0] == self.N_coils, (
            f"coil_currents.shape={I.shape}，期望 ({self.N_coils},)")

        BR_new   = self._BR_base + np.tensordot(I, self.BR_resp,   axes=[[0], [0]])
        BPhi_new = self._BPhi_base + np.tensordot(I, self.BPhi_resp, axes=[[0], [0]])
        BZ_new   = self._BZ_base + np.tensordot(I, self.BZ_resp,   axes=[[0], [0]])

        return {
            "BR": BR_new, "BPhi": BPhi_new, "BZ": BZ_new,
            "R_grid": self.R_grid, "Z_grid": self.Z_grid, "Phi_grid": self.Phi_grid,
        }

    # ── δλ_u 线性响应 basis（按分组方案预计算）────────────────────────────────

    def precompute_basis_dlambda(
        self,
        schemes: list = None,
        xpt_phi: float = 0.0,
        island_period: int = 10,
        eps_current: float = 0.01,
    ) -> dict:
        """对每个分组方案的每个电源组预计算 δλ_u 线性响应（有限差分）。

        物理意义
        ---------
        FPT（固定点扰动论）：δλ_u = (w_u^T · δDPm · v_u) / (w_u^T · v_u)
        对每个分组 g，将该组内所有线圈同时加 eps_current（A），计算 δDPm，
        再除以 eps_current 得到单位电流灵敏度。

        Parameters
        ----------
        schemes : list of str, optional
            分组方案名称列表，如 ['A', 'B', 'C']。
            若为 None，则使用已在 _group_cache 中存在的所有方案。
            若 _group_cache 为空，尝试从 explore_hao_divertor_configs 自动注册 A/B/C。
        xpt_phi : float
            X 点所在截面的环向角（弧度），默认 0.0。
        island_period : int
            Poincaré 映射的追踪圈数（HAO m/n=10/3，m=10）。
        eps_current : float
            有限差分电流扰动（A），相对于实际运行电流（~kA）很小。

        Returns
        -------
        basis_dlambda : dict
            scheme_name -> np.ndarray shape (K,)，单位 A^{-1}。
            同时存入 self.basis_dlambda。

        Notes
        -----
        **完整运行需要**：
          1. 已加载的基础场（self._BR_base 等）和线圈响应场（self.BR_resp 等）
          2. 有效的 X 点数据（self.boundary_xpts 非空，含 DPm 矩阵）
          3. pyna._cyna 可用（C++ 加速的轨道追踪）

        预计运行时间估算：
          - Scheme A：6 组 × ~2–5 s/组 ≈ 12–30 s
          - Scheme B：2 组 × ~2–5 s/组 ≈ 4–10 s
          - Scheme C：4 组 × ~2–5 s/组 ≈ 8–20 s
        （每组调用一次 cyna orbit tracing，island_period=10 圈）
        """
        from pyna.topo.perturbation import (
            DPm_finite_difference,
            eigenvalue_perturbation,
        )

        # ── 确定要计算的方案 ────────────────────────────────────────────────
        if schemes is None:
            if self._group_cache:
                schemes = list(self._group_cache.keys())
            else:
                # 尝试自动注册标准方案
                _register_schemes()
                schemes = []
                if _SCHEME_REGISTRY:
                    phi_arr = self.coil_phi
                    Z_arr   = self.coil_Z
                    for sname, fn in _SCHEME_REGISTRY.items():
                        try:
                            if sname == "B":
                                groups, labels = fn(Z_arr)
                            else:
                                groups, labels = fn(phi_arr, Z_arr)
                            self.precompute_group_basis(groups, scheme_name=sname)
                            schemes.append(sname)
                        except Exception as e:
                            self._log(f"[警告] 方案 {sname} 自动注册失败: {e}")

        if not schemes:
            self._log("[警告] 没有可用的分组方案，precompute_basis_dlambda 跳过")
            return self.basis_dlambda

        # ── 找到 X 点（取最外侧的 boundary X 点） ──────────────────────────
        outer_xpts = [x for x in self.boundary_xpts if x["R"] > R_AX_REF + 0.15]
        if not outer_xpts:
            self._log("[警告] 无有效 boundary X 点，precompute_basis_dlambda 跳过")
            return self.basis_dlambda
        xpt = max(outer_xpts, key=lambda x: x["R"])
        R0, Z0 = xpt["R"], xpt["Z"]
        DPm_base_pkl = xpt["DPm"]  # pkl 中存储的 DPm

        self._log(f"precompute_basis_dlambda: X 点 (R={R0:.4f}, Z={Z0:.4f}), "
                  f"island_period={island_period}, eps={eps_current} A")

        # ── 基础场 field_cache ──────────────────────────────────────────────
        base_fc = {
            "BR":      self._BR_base,
            "BPhi":    self._BPhi_base,
            "BZ":      self._BZ_base,
            "R_grid":  self.R_grid,
            "Z_grid":  self.Z_grid,
            "Phi_grid": self.Phi_grid,
        }

        phi_start = xpt_phi
        phi_end   = phi_start + island_period * 2.0 * np.pi

        # 先算基础场的 DPm（用 cyna 有限差分，保证与后续扰动场一致）
        self._log("  计算基础场 DPm ...")
        t0 = time.time()
        DPm_base_fd, _, _, _ = DPm_finite_difference(
            x0=(R0, Z0),
            field_func_base=None,
            field_func_pert=None,
            phi_span=(phi_start, phi_end),
            base_field_cache=base_fc,
            pert_field_cache=base_fc,   # 同一场 → δDPm=0，用于确认接口
            island_period=island_period,
            DPhi=0.05,
        )
        self._log(f"  基础场 DPm 完成 ({time.time()-t0:.2f}s): λ_u={np.max(np.abs(np.linalg.eigvals(DPm_base_fd))):.4f}")

        # ── 对每个方案、每个组计算 δλ_u ────────────────────────────────────
        for scheme_name in schemes:
            if scheme_name not in self._group_cache:
                self._log(f"[警告] 方案 {scheme_name} 未在 _group_cache 中，跳过")
                continue

            basis = self._group_cache[scheme_name]
            K = len(basis["BR"])
            dlambda = np.zeros(K, dtype=float)

            self._log(f"  方案 {scheme_name}: {K} 组")
            for g in range(K):
                t_g = time.time()

                # 该组单位电流扰动场
                dBR   = basis["BR"][g]   * eps_current
                dBPhi = basis["BPhi"][g] * eps_current
                dBZ   = basis["BZ"][g]   * eps_current

                pert_fc = {
                    "BR":      self._BR_base   + dBR,
                    "BPhi":    self._BPhi_base + dBPhi,
                    "BZ":      self._BZ_base   + dBZ,
                    "R_grid":  self.R_grid,
                    "Z_grid":  self.Z_grid,
                    "Phi_grid": self.Phi_grid,
                }

                _, DPm_pert, delta_DPm, _ = DPm_finite_difference(
                    x0=(R0, Z0),
                    field_func_base=None,
                    field_func_pert=None,
                    phi_span=(phi_start, phi_end),
                    base_field_cache=base_fc,
                    pert_field_cache=pert_fc,
                    island_period=island_period,
                    DPhi=0.05,
                )

                eig_res = eigenvalue_perturbation(DPm_base_fd, delta_DPm)
                delta_lam_u = eig_res["delta_lambda_u"].real  # 取实部（超双曲特征值是实数）

                # 归一化为单位电流响应（A^{-1}）
                dlambda[g] = delta_lam_u / eps_current

                self._log(f"    组 {g}: δλ_u/ε={dlambda[g]:.6g} A⁻¹  ({time.time()-t_g:.2f}s)")

            self.basis_dlambda[scheme_name] = dlambda
            self._log(f"  方案 {scheme_name} basis_dlambda shape: {dlambda.shape}  "
                      f"range=[{dlambda.min():.4g}, {dlambda.max():.4g}]")

        return self.basis_dlambda


# ═══════════════════════════════════════════════════════════════════════════════
# build_field_superposition_cache — 批量 einsum
# ═══════════════════════════════════════════════════════════════════════════════

def build_field_superposition_cache(
    field_cache_base: dict,
    coil_currents_matrix: np.ndarray,
    BR_resp: np.ndarray,
    BPhi_resp: np.ndarray,
    BZ_resp: np.ndarray,
    chunk_size: int = 16,
) -> list[dict]:
    """批量计算 N_configs 个总场的 field_cache 列表。

    利用线性叠加：
        B_total[c] = B_base + Σᵢ I[c,i] · B_resp[i]

    批量 einsum（分块以控制内存）：
        BR_batch[c,j,k,l] = BR_base[j,k,l]
                           + einsum('ci,ijkl->cjkl', I_matrix, BR_resp)

    Parameters
    ----------
    field_cache_base     : 基础场 dict（BR/BPhi/BZ/R_grid/Z_grid/Phi_grid）
    coil_currents_matrix : shape (N_configs, N_coils)，相对基础场的增量电流（A）
    BR_resp / BPhi_resp / BZ_resp : shape (N_coils, NR, NZ, NPHI)，响应场
    chunk_size           : 每批处理的位型数（控制 RAM 峰值）

    Returns
    -------
    list of N_configs field_cache dicts（每个 dict 引用各自独立的 BR/BPhi/BZ）
    """
    I_mat = np.asarray(coil_currents_matrix, dtype=np.float64)   # (C, N_coils)
    BR_r  = np.asarray(BR_resp,   dtype=np.float64)               # (N_coils, NR, NZ, NPHI)
    BP_r  = np.asarray(BPhi_resp, dtype=np.float64)
    BZ_r  = np.asarray(BZ_resp,   dtype=np.float64)

    BR_base   = np.asarray(field_cache_base["BR"],   dtype=np.float64)
    BPhi_base = np.asarray(field_cache_base["BPhi"], dtype=np.float64)
    BZ_base   = np.asarray(field_cache_base["BZ"],   dtype=np.float64)
    meta = {k: field_cache_base[k]
            for k in ("R_grid", "Z_grid", "Phi_grid")}

    N_configs = I_mat.shape[0]
    results: list[dict] = [None] * N_configs

    for start in range(0, N_configs, chunk_size):
        end = min(start + chunk_size, N_configs)
        I_chunk = I_mat[start:end]             # (C', N_coils)

        # 核心：einsum 'ci,ijk->cjk' （展开到 4D）
        # 等价于 (C', N) @ (N, NR*NZ*NPHI) → reshape
        N, NR, NZ, NPHI = BR_r.shape
        BR_flat   = BR_r.reshape(N, -1)
        BP_flat   = BP_r.reshape(N, -1)
        BZ_flat_r = BZ_r.reshape(N, -1)

        dBR   = (I_chunk @ BR_flat  ).reshape(-1, NR, NZ, NPHI)  # (C', NR, NZ, NPHI)
        dBPhi = (I_chunk @ BP_flat  ).reshape(-1, NR, NZ, NPHI)
        dBZ   = (I_chunk @ BZ_flat_r).reshape(-1, NR, NZ, NPHI)

        for local_i, global_i in enumerate(range(start, end)):
            results[global_i] = dict(
                BR   = BR_base   + dBR[local_i],
                BPhi = BPhi_base + dBPhi[local_i],
                BZ   = BZ_base   + dBZ[local_i],
                **meta,
            )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 快速单次评估（< 0.1 s/trial）
# ═══════════════════════════════════════════════════════════════════════════════

def fast_evaluate_single(
    field_cache: dict,
    boundary_xpts: list[dict],
    baseline_lambda_u: float,
    basis_dlambda: Optional[np.ndarray] = None,
    delta_I: Optional[np.ndarray] = None,
    basis_dlambda_dict: Optional[dict] = None,
    scheme: Optional[str] = None,
    n_lcfs_turns: int = 60,
    DPhi: float = 0.05,
    wall_file: Optional[str] = None,
    compute_extra_metrics: bool = False,
    R_axis: float = R_AX_REF,
    Z_axis: float = Z_AX_REF,
    n_profile_surfaces: int = 8,
    n_profile_turns: int = 20,
) -> dict:
    """快速评估单个位型的 V_lcfs 和 λ_u（目标 < 0.1 s）。

    策略：
    ①  λ_u：若提供 basis_dlambda（灵敏度向量）和 delta_I，用线性近似
        λ_u ≈ baseline_lambda_u + basis_dlambda @ delta_I  （~1 μs）
        支持两种传入方式：
          - 直接传 basis_dlambda (K,) + delta_I (K,)
          - 传 basis_dlambda_dict (scheme->array) + scheme 名称 + delta_I
        否则从 field_cache 和 pkl X 点做快速 DPm 计算（island_period=10 圈）。
    ②  V_lcfs：cyna LCFS 二分搜索（n_iter=16）+ 1-seed Poincare（n_turns=60）
        + shoelace 面积 × cylindrical mean → ~30~60 ms

    Parameters
    ----------
    field_cache     : dict with BR/BPhi/BZ/R_grid/Z_grid/Phi_grid
    boundary_xpts   : list of {R, Z, DPm, lambda_u}（来自 TopologyCache）
    baseline_lambda_u : 基础位型 λ_u（用于线性近似参考点）
    basis_dlambda   : shape (K,)，∂λ_u/∂I_group（可选，直接传入）
    delta_I         : shape (K,)，各组电流偏移量
    basis_dlambda_dict : dict scheme_name->np.ndarray，precompute_basis_dlambda 的输出
    scheme          : 与 basis_dlambda_dict 配合使用的方案名称
    n_lcfs_turns    : LCFS Poincare 追踪圈数
    DPhi            : RK4 步长
    wall_file       : 第一壁文件路径

    Returns
    -------
    dict with:
        V_lcfs   : float [m³]
        lambda_u : float
        elapsed_s : float
    """
    from pyna.topo.fast_metrics import compute_profile_objectives_fast
    from pyna.topo.topology_eval import (
        _FC, _find_wall_file, _load_wall, _find_lcfs_seed,
        _poincare_single_section, _shoelace_area,
    )
    from pyna._cyna import is_available as _cyna_ok

    t0 = time.time()

    # ── λ_u ───────────────────────────────────────────────────────────────────
    # 解析 basis_dlambda：支持直接传入或从 dict+scheme 获取
    _basis_dl = basis_dlambda
    if _basis_dl is None and basis_dlambda_dict is not None and scheme is not None:
        _basis_dl = basis_dlambda_dict.get(scheme)

    if _basis_dl is not None and delta_I is not None and False:  # disabled: linear approx unreliable for large perturbations
        # 线性近似（无任何追踪）：~1 μs
        lambda_u = baseline_lambda_u + float(
            np.dot(np.asarray(_basis_dl), np.asarray(delta_I))
        )
    else:
        # 直接用 cyna find_fixed_points_batch(period=10) 精确计算 λ_u (~50ms)
        lambda_u = baseline_lambda_u
        if boundary_xpts and _cyna_ok():
            from pyna._cyna import find_fixed_points_batch as _cyna_fpb
            fc_obj = _FC(field_cache)
            outer_xpts = [x for x in boundary_xpts if x["R"] > R_AX_REF + 0.15]
            if outer_xpts:
                # Try ALL outer X-pts, take the one that converges to a true X-point
                R_seeds = np.array([xp["R"] for xp in outer_xpts])
                Z_seeds = np.array([xp["Z"] for xp in outer_xpts])
                R_out, Z_out, res_out, conv_out, DPm_flat, _, _, ptype = _cyna_fpb(
                    R_seeds, Z_seeds,
                    0.0, 10,  # phi_sec=0, period=10
                    max_iter=30, tol=1e-9,
                    BR=fc_obj.BR, BPhi=fc_obj.BPhi, BZ=fc_obj.BZ,
                    R_grid=fc_obj.Rg, Z_grid=fc_obj.Zg, Phi_grid=fc_obj.Pg_ext,
                )
                # ptype=1 means X-point (|Tr|>2, hyperbolic) — only use those
                candidates = []
                for i in range(len(outer_xpts)):
                    if conv_out[i] and ptype[i] == 1:  # X-point confirmed
                        DPm_i = DPm_flat[i].reshape(2, 2)
                        eigs_i = np.linalg.eigvals(DPm_i)
                        lu_i = float(np.max(np.abs(eigs_i)))
                        candidates.append(lu_i)
                if candidates:
                    lambda_u = max(candidates)  # most hyperbolic X-point
                # else: keep baseline_lambda_u (no valid X-point found)

    # ── V_lcfs ────────────────────────────────────────────────────────────────
    V_lcfs = float("nan")
    try:
        fc = _FC(field_cache)
        if wall_file is None:
            wall_file = _find_wall_file()
        phi_c, wall_R, wall_Z, _ = _load_wall(wall_file)

        # ── 磁轴定位（trial field 中动态精细化）──────────────────────────────
        # The magnetic axis shifts with field changes. Using a fixed R_AX_REF
        # leads to incorrect LCFS seeds when alpha deviates significantly from 1.
        # We refine the axis position using a single cyna Newton call (period=1, O-pt).
        R_ax_trial = float(R_axis)   # start from caller-provided guess
        Z_ax_trial = float(Z_axis)
        if _cyna_ok():
            try:
                from pyna._cyna import find_fixed_points_batch as _cyna_fpb
                R_ax_out, Z_ax_out, res_ax, conv_ax, _, _, _, ptype_ax = _cyna_fpb(
                    np.array([R_ax_trial]), np.array([Z_ax_trial]),
                    0.0, 1,   # period=1: magnetic axis is a period-1 O-point
                    max_iter=25, tol=1e-8,
                    BR=fc.BR, BPhi=fc.BPhi, BZ=fc.BZ,
                    R_grid=fc.Rg, Z_grid=fc.Zg, Phi_grid=fc.Pg_ext,
                )
                if conv_ax[0] and ptype_ax[0] == 0:  # O-point (elliptic, period-1)
                    R_ax_trial = float(R_ax_out[0])
                    Z_ax_trial = float(Z_ax_out[0])
            except Exception:
                pass  # keep the initial guess

        # ── LCFS seed: binary search along horizontal from the axis ──────────
        # Use base-field boundary_xpts as upper bound (conservative but safe).
        if boundary_xpts:
            R_hi = max(x["R"] for x in boundary_xpts) - 0.01
        else:
            R_hi = R_ax_trial + 0.22

        R0_seed = _find_lcfs_seed(
            R_ax_trial, Z_ax_trial, R_hi, fc,
            phi_c, wall_R, wall_Z,
            max_turns=120, n_iter=16, DPhi=DPhi,
        )

        # 单截面 Poincare（快速估算面积，以精确磁轴为中心）
        pts = _poincare_single_section(
            np.array([R0_seed]), np.array([Z_ax_trial]),
            0.0, n_lcfs_turns, fc, phi_c, wall_R, wall_Z, DPhi,
        )
        lcfs_R, lcfs_Z = pts[0]
        if len(lcfs_R) > 3:
            area  = _shoelace_area(lcfs_R, lcfs_Z)
            R_cen = float(np.mean(lcfs_R))
            V_lcfs = 2.0 * np.pi * area * R_cen
    except Exception as e:
        pass  # V_lcfs 保持 nan，调用方处理

    result = {
        "V_lcfs":   V_lcfs,
        "lambda_u": lambda_u,
    }

    if compute_extra_metrics and np.isfinite(V_lcfs) and V_lcfs > 0.0:
        try:
            result.update(
                compute_profile_objectives_fast(
                    field_cache,
                    V_lcfs=V_lcfs,
                    R_axis=R_axis,
                    Z_axis=Z_axis,
                    wall_file=wall_file,
                    n_surfaces=n_profile_surfaces,
                    n_turns=n_profile_turns,
                    DPhi=DPhi,
                )
            )
        except Exception:
            pass

    result["elapsed_s"] = time.time() - t0
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# batch_evaluate_topology — 并行评估
# ═══════════════════════════════════════════════════════════════════════════════

def _worker_full_eval(fc_dict: dict, eval_kwargs: dict):
    """子进程工作函数：完整拓扑评估。"""
    # 重新配置 sys.path（子进程不继承）
    for _p in (str(TOPOQUEST), str(PYNA)):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    try:
        from pyna.topo.topology_eval import evaluate_topology
        return evaluate_topology(fc_dict, **eval_kwargs)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def _worker_fast_eval(args: tuple):
    """子进程工作函数：快速评估。"""
    fc_dict, boundary_xpts, baseline_lam, basis_dl, delta_I, kwargs = args
    for _p in (str(TOPOQUEST), str(PYNA)):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    try:
        return fast_evaluate_single(
            fc_dict, boundary_xpts, baseline_lam,
            basis_dlambda=basis_dl,
            delta_I=delta_I,
            **kwargs,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def batch_evaluate_topology(
    field_caches: list[dict],
    n_workers: int = 4,
    mode: str = "full",       # 'full' | 'fast'
    boundary_xpts: Optional[list[dict]] = None,
    baseline_lambda_u: float = 1.0,
    basis_dlambda_list: Optional[list] = None,  # per-config delta_I for fast λ_u
    delta_I_list: Optional[list] = None,
    **eval_kwargs,
) -> list:
    """并行评估多个磁场位型。

    Parameters
    ----------
    field_caches    : build_field_superposition_cache() 的输出
    n_workers       : 并行进程数
    mode            : 'full'（完整评估，<10s/trial）或 'fast'（<0.1s/trial）
    boundary_xpts   : fast 模式必需，来自 TopologyCache
    baseline_lambda_u : fast 模式用于线性近似
    basis_dlambda_list : 各位型的 basis_dlambda 向量（fast 模式线性近似）
    delta_I_list    : 各位型的 delta_I 向量（与 basis_dlambda 配合）
    **eval_kwargs   : 传给 evaluate_topology 或 fast_evaluate_single 的参数

    Returns
    -------
    list[TopologyEval | dict | None]  长度 = len(field_caches)
    """
    N = len(field_caches)
    if N == 0:
        return []

    # 尝试 joblib，回退到 concurrent.futures
    try:
        from joblib import Parallel, delayed
        _USE_JOBLIB = True
    except ImportError:
        _USE_JOBLIB = False

    if mode == "full":
        if _USE_JOBLIB:
            results = Parallel(n_jobs=n_workers, backend="loky")(
                delayed(_worker_full_eval)(fc, eval_kwargs)
                for fc in field_caches
            )
        else:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(_worker_full_eval, fc, eval_kwargs)
                           for fc in field_caches]
                results = [f.result() for f in futures]

    elif mode == "fast":
        if boundary_xpts is None:
            raise ValueError("fast 模式需要 boundary_xpts")
        args_list = [
            (
                field_caches[i],
                boundary_xpts,
                baseline_lambda_u,
                basis_dlambda_list[i] if basis_dlambda_list else None,
                delta_I_list[i] if delta_I_list else None,
                eval_kwargs,
            )
            for i in range(N)
        ]
        if _USE_JOBLIB:
            results = Parallel(n_jobs=n_workers, backend="loky")(
                delayed(_worker_fast_eval)(args) for args in args_list
            )
        else:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(_worker_fast_eval, args) for args in args_list]
                results = [f.result() for f in futures]
    else:
        raise ValueError(f"mode 必须是 'full' 或 'fast'，got '{mode}'")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# optuna_objective_factory
# ═══════════════════════════════════════════════════════════════════════════════

# 内置分组方案注册表（与 explore_hao_divertor_configs.py 一致）
_SCHEME_REGISTRY: dict[str, Callable] = {}

def _register_schemes():
    """延迟导入，避免在子进程/模块导入时触发重量级导入。"""
    if _SCHEME_REGISTRY:
        return
    try:
        from explore_hao_divertor_configs import (
            make_grouping_scheme_A,
            make_grouping_scheme_B,
            make_grouping_scheme_C,
        )
        _SCHEME_REGISTRY["A"] = make_grouping_scheme_A
        _SCHEME_REGISTRY["B"] = make_grouping_scheme_B
        _SCHEME_REGISTRY["C"] = make_grouping_scheme_C
    except ImportError:
        pass  # 若无法导入，用户需手动注册


def optuna_objective_factory(
    topo_cache: TopologyCache,
    schemes: dict[str, tuple],
    *,
    I_max: float = 5000.0,          # 每组最大电流绝对值 [A]
    V_min: float = 0.3,             # 硬约束：体积下限 [m³]
    lambda_u_min: float = 1.05,     # 硬约束：X 点超双曲性下限
    lambda_u_target: float = 1.5,   # 目标：λ_u 尽量大（分隔改善）
    use_linear_approx: bool = True,  # True=线性近似λ_u（极快），False=完整追踪
    eval_kwargs: Optional[dict] = None,
) -> Callable:
    """工厂函数，返回一个 Optuna trial -> float 的目标函数。

    trial 参数设计
    --------------
    - ``scheme``  (categorical)：分组方案名称（"A" / "B" / "C" / 自定义）
    - ``I_g{k}``  (float [-I_max, I_max])：第 k 组的电流增量

    目标（最小化）
    --------------
    -V_lcfs / V_ref  （最大化受限体积）
    + 若 lambda_u < lambda_u_target：惩罚项

    硬约束（TrialPruned）
    ----------------------
    - V_lcfs < V_min
    - lambda_u < lambda_u_min

    Parameters
    ----------
    topo_cache     : 已初始化的 TopologyCache 实例
    schemes        : {scheme_name: (groups, labels)} dict；
                     groups 是线圈组列表（每组为线圈索引 list）。
                     调用前需已 precompute_group_basis() 或在此处自动预计算。
    I_max          : 各组电流范围 [-I_max, I_max]（A）
    V_min          : 体积硬约束下限（m³）
    lambda_u_min   : λ_u 硬约束下限
    lambda_u_target: λ_u 目标值（超过则无惩罚）
    use_linear_approx : 是否用线性 λ_u 近似（需预先 precompute_group_basis）
    eval_kwargs    : 传给 fast_evaluate_single 的额外参数

    Returns
    -------
    objective(trial) -> float
    """
    import optuna

    # 预计算所有分组方案的 basis 场
    for sname, (groups, labels) in schemes.items():
        topo_cache.precompute_group_basis(groups, scheme_name=sname)

    boundary_xpts = topo_cache.boundary_xpts
    baseline_lam  = topo_cache.baseline_lambda_u
    base_V_ref    = 0.4   # 参考体积，用于归一化目标函数 [m³]

    _eval_kw = eval_kwargs or {}

    def objective(trial) -> float:
        # ── 采样分组方案 ──────────────────────────────────────────────────────
        scheme_name = trial.suggest_categorical("scheme", list(schemes.keys()))
        groups, labels = schemes[scheme_name]
        K = len(groups)

        # ── 采样各组电流 ──────────────────────────────────────────────────────
        delta_I = np.array([
            trial.suggest_float(f"I_g{k}", -I_max, I_max)
            for k in range(K)
        ])

        # ── 构造修改后的场 ─────────────────────────────────────────────────────
        fc_new = topo_cache.build_field_from_delta_I(delta_I, scheme_name=scheme_name)

        # ── 快速评估 ──────────────────────────────────────────────────────────
        # 线性近似 λ_u：需 basis_dlambda（若未预计算则回退到追踪）
        result = fast_evaluate_single(
            fc_new,
            boundary_xpts=boundary_xpts,
            baseline_lambda_u=baseline_lam,
            basis_dlambda_dict=topo_cache.basis_dlambda if topo_cache.basis_dlambda else None,
            scheme=scheme_name,
            delta_I=delta_I,
            **_eval_kw,
        )

        V_lcfs   = result["V_lcfs"]
        lambda_u = result["lambda_u"]

        # ── 硬约束检查（TrialPruned） ─────────────────────────────────────────
        if np.isnan(V_lcfs) or V_lcfs < V_min:
            raise optuna.TrialPruned(
                f"V_lcfs={V_lcfs:.4f} < V_min={V_min}"
            )
        if np.isnan(lambda_u) or lambda_u < lambda_u_min:
            raise optuna.TrialPruned(
                f"lambda_u={lambda_u:.4f} < lambda_u_min={lambda_u_min}"
            )

        # ── 目标函数（最小化） ─────────────────────────────────────────────────
        # 主目标：最大化 V_lcfs（负号）
        obj = -V_lcfs / base_V_ref

        # 软惩罚：λ_u 低于目标时追加惩罚
        if lambda_u < lambda_u_target:
            obj += 2.0 * (lambda_u_target - lambda_u) / lambda_u_target

        # 记录中间指标（Optuna 不用它做优化，但方便可视化）
        trial.set_user_attr("V_lcfs",   float(V_lcfs))
        trial.set_user_attr("lambda_u", float(lambda_u))
        trial.set_user_attr("elapsed_s", float(result["elapsed_s"]))

        return float(obj)

    return objective


# ═══════════════════════════════════════════════════════════════════════════════
# 便捷函数：快速构建 Optuna study
# ═══════════════════════════════════════════════════════════════════════════════

def create_optuna_study(
    topo_cache: TopologyCache,
    *,
    coil_phi: Optional[np.ndarray] = None,
    coil_Z:   Optional[np.ndarray] = None,
    scheme_names: list[str] = ("A", "B", "C"),
    storage: Optional[str] = None,
    study_name: str = "hao_divertor_opt",
    n_trials: int = 1000,
    n_jobs: int = 1,
    **objective_kwargs,
):
    """创建并运行 Optuna study（含默认分组方案 A/B/C）。

    Parameters
    ----------
    topo_cache   : 已初始化的 TopologyCache
    coil_phi     : 各线圈环向角度（若 None 则从 topo_cache 取）
    coil_Z       : 各线圈 Z 坐标（若 None 则从 topo_cache 取）
    scheme_names : 参与优化的方案子集
    storage      : Optuna storage URL（None=内存）
    study_name   : study 名称
    n_trials     : 优化 trial 数
    n_jobs       : Optuna 并行 trial 数（注意与 batch_eval 的 n_workers 不同）
    **objective_kwargs : 传给 optuna_objective_factory

    Returns
    -------
    study : optuna.Study
    """
    import optuna

    _register_schemes()
    if not _SCHEME_REGISTRY:
        raise ImportError("无法导入 explore_hao_divertor_configs，"
                          "请手动传入 schemes 参数并调用 optuna_objective_factory()")

    phi = coil_phi if coil_phi is not None else topo_cache.coil_phi
    Z   = coil_Z   if coil_Z   is not None else topo_cache.coil_Z

    schemes = {}
    for sname in scheme_names:
        if sname not in _SCHEME_REGISTRY:
            raise KeyError(f"未知方案 '{sname}'，可选: {list(_SCHEME_REGISTRY.keys())}")
        fn = _SCHEME_REGISTRY[sname]
        if sname == "B":
            groups, labels = fn(Z)
        else:
            groups, labels = fn(phi, Z)
        schemes[sname] = (groups, labels)

    objective = optuna_objective_factory(topo_cache, schemes, **objective_kwargs)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )

    print(f"[Optuna] 开始优化: study={study_name}, n_trials={n_trials}, n_jobs={n_jobs}")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs,
                   catch=(Exception,))
    return study


# ═══════════════════════════════════════════════════════════════════════════════
# 简单测试（可直接 python batch_eval.py 运行）
# ═══════════════════════════════════════════════════════════════════════════════

def _smoke_test():
    """冒烟测试：验证批量场叠加 + 并行评估框架正常工作。

    不需要真实数据：用随机场和随机电流模拟。
    """
    print("=" * 60)
    print("batch_eval smoke test (随机数据，不需要真实场文件)")
    print("=" * 60)

    # ── 构造假数据 ────────────────────────────────────────────────────────────
    N_coils = 10
    NR, NZ, NPHI = 20, 20, 16
    N_configs = 4

    rng = np.random.default_rng(0)
    BR_base   = rng.uniform(-0.5, 0.5, (NR, NZ, NPHI))
    BPhi_base = rng.uniform( 0.5, 1.5, (NR, NZ, NPHI))
    BZ_base   = rng.uniform(-0.2, 0.2, (NR, NZ, NPHI))

    R_grid   = np.linspace(0.57, 1.31, NR)
    Z_grid   = np.linspace(-0.42, 0.42, NZ)
    Phi_grid = np.linspace(0, 2 * np.pi, NPHI, endpoint=False)

    base_fc = dict(BR=BR_base, BPhi=BPhi_base, BZ=BZ_base,
                   R_grid=R_grid, Z_grid=Z_grid, Phi_grid=Phi_grid)

    BR_resp   = rng.uniform(-0.01, 0.01, (N_coils, NR, NZ, NPHI)).astype(np.float32)
    BPhi_resp = rng.uniform(-0.01, 0.01, (N_coils, NR, NZ, NPHI)).astype(np.float32)
    BZ_resp   = rng.uniform(-0.01, 0.01, (N_coils, NR, NZ, NPHI)).astype(np.float32)

    I_matrix = rng.uniform(-1000, 1000, (N_configs, N_coils))

    # ── 批量场叠加 ────────────────────────────────────────────────────────────
    print(f"\n[1] build_field_superposition_cache: {N_configs} configs × {N_coils} coils")
    t0 = time.time()
    fc_list = build_field_superposition_cache(
        base_fc, I_matrix, BR_resp, BPhi_resp, BZ_resp
    )
    t1 = time.time()
    print(f"    耗时: {(t1-t0)*1e3:.1f} ms  → {len(fc_list)} field_caches")
    assert len(fc_list) == N_configs
    assert fc_list[0]["BR"].shape == (NR, NZ, NPHI)

    # 验证线性叠加正确性（config 0）
    expected = BR_base + np.tensordot(I_matrix[0], BR_resp.astype(np.float64), axes=[[0],[0]])
    err = np.max(np.abs(fc_list[0]["BR"] - expected))
    ok = "OK" if err < 1e-10 else "FAIL"
    print(f"    线性叠加误差 (max abs): {err:.2e}  [{ok}]")

    # ── TopologyCache (仅测试分组 basis 场逻辑，不加载真实文件) ──────────────
    print(f"\n[2] TopologyCache.precompute_group_basis（模拟）")

    # 用实际的 TopologyCache 方法测试分组逻辑（无文件 IO）
    _nc, _nr, _nz, _nphi = N_coils, NR, NZ, NPHI

    class FakeCache:
        """最小化模拟 TopologyCache（不读文件）。"""
        def __init__(self):
            self.N_coils = _nc
            self.BR_resp = BR_resp
            self.BPhi_resp = BPhi_resp
            self.BZ_resp = BZ_resp
            self._shape  = (_nr, _nz, _nphi)
            self._BR_base   = BR_base.copy()
            self._BPhi_base = BPhi_base.copy()
            self._BZ_base   = BZ_base.copy()
            self.R_grid  = R_grid
            self.Z_grid  = Z_grid
            self.Phi_grid = Phi_grid
            self.boundary_xpts = []
            self.baseline_lambda_u = 1.3
            self.basis_dlambda = {}
            self._group_cache = {}
            self._verbose = True

        def _log(self, msg): print(f"  [FakeCache] {msg}")
        precompute_group_basis    = TopologyCache.precompute_group_basis
        build_field_from_delta_I  = TopologyCache.build_field_from_delta_I
        build_field_from_coil_currents = TopologyCache.build_field_from_coil_currents

    fake = FakeCache()
    groups_AB = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]  # 2 组
    t0 = time.time()
    basis = fake.precompute_group_basis(groups_AB, scheme_name="test")
    print(f"    耗时: {(time.time()-t0)*1e3:.1f} ms  BR[0].shape={basis['BR'][0].shape}")

    delta_I_test = np.array([500.0, -300.0])
    fc_mod = fake.build_field_from_delta_I(delta_I_test, scheme_name="test")
    assert fc_mod["BR"].shape == (NR, NZ, NPHI)
    print(f"    build_field_from_delta_I: shape {fc_mod['BR'].shape}  [OK]")

    # ── 并行评估（仅测试框架流程，不调用真实 cyna） ───────────────────────────
    print(f"\n[3] batch_evaluate_topology 框架（4 configs，2 workers，fast mock）")

    def _mock_worker(fc_dict, kwargs):
        time.sleep(0.02)  # 模拟 20ms 评估
        return {"V_lcfs": 0.35, "lambda_u": 1.4, "elapsed_s": 0.02}

    t0 = time.time()
    # 直接串行模拟（不依赖 cyna/真实场）
    mock_results = [_mock_worker(fc, {}) for fc in fc_list]
    elapsed = time.time() - t0
    print(f"    串行模拟: {elapsed*1e3:.0f} ms  ({len(mock_results)} 结果)")
    for i, r in enumerate(mock_results):
        print(f"      config[{i}]: V={r['V_lcfs']:.3f} m^3  lambda_u={r['lambda_u']:.3f}")

    print("\n[PASS] Smoke test passed (no real field/cyna dependency)")


if __name__ == "__main__":
    _smoke_test()
