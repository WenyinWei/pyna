"""
pyna.mcf.flt.numba_poincare — Poincaré 追踪器（cyna C++ 底层）

原 numba JIT 实现已移除。底层现在统一使用 cyna C++ 扩展
（pyna._cyna），通过 pybind11 绑定，支持多线程并行、
toroidal 3D wall、固定点搜索等功能。

公开 API 与原 numba 版本保持完全一致，调用方无需修改。

底层函数：
  trace_poincare_batch      → cyna.trace_poincare_batch
  trace_poincare_multi_batch → cyna.trace_poincare_multi（自动 reshape counts）
  precompile_tracer          → no-op（cyna 无需预编译）
  field_arrays_from_interpolators → 保留，提取 numpy 数组供 cyna 使用
"""
from __future__ import annotations

import numpy as np
from pyna._cyna import (
    is_available as _cyna_available,
    trace_poincare_batch as _cyna_trace_poincare_batch,
    trace_poincare_multi as _cyna_trace_poincare_multi,
    trace_poincare_batch_twall as _cyna_trace_poincare_batch_twall,
    find_fixed_points_batch as _cyna_find_fixed_points_batch,
    trace_orbit_along_phi as _cyna_trace_orbit_along_phi,
)


# ---------------------------------------------------------------------------
# 工具函数：从 scipy 插值器中提取 numpy 数组（供调用方用，与 cyna 无关）
# ---------------------------------------------------------------------------

def field_arrays_from_interpolators(itp_BR, itp_BPhi, itp_BZ):
    """
    从 scipy RegularGridInterpolator 中提取 numpy 数组供 cyna 使用。

    假设插值器定义在 (R_grid, Z_grid, Phi_grid) 轴上（按此顺序传入
    RegularGridInterpolator）。

    Returns
    -------
    R_grid, Z_grid, Phi_grid : 1-D float64 arrays
    BR_flat, BPhi_flat, BZ_flat : C-contiguous float64 arrays (NR*NZ*NPhi)
    nx, ny, nz : ints
    """
    R_grid   = np.ascontiguousarray(itp_BR.grid[0],   dtype=np.float64)
    Z_grid   = np.ascontiguousarray(itp_BR.grid[1],   dtype=np.float64)
    Phi_grid = np.ascontiguousarray(itp_BR.grid[2],   dtype=np.float64)
    nx, ny, nz = len(R_grid), len(Z_grid), len(Phi_grid)

    BR_flat   = np.ascontiguousarray(itp_BR.values.ravel(),   dtype=np.float64)
    BPhi_flat = np.ascontiguousarray(itp_BPhi.values.ravel(), dtype=np.float64)
    BZ_flat   = np.ascontiguousarray(itp_BZ.values.ravel(),   dtype=np.float64)
    return R_grid, Z_grid, Phi_grid, BR_flat, BPhi_flat, BZ_flat, nx, ny, nz


# ---------------------------------------------------------------------------
# precompile_tracer — cyna 不需要预编译，保留为 no-op 以保持接口兼容
# ---------------------------------------------------------------------------

def precompile_tracer(R_grid, Z_grid, Phi_grid, BR_flat, BPhi_flat, BZ_flat):
    """
    预编译追踪器（兼容接口，cyna 底层无需此步骤）。

    原 numba 版本需要在第一次计算前触发 JIT 编译。cyna C++ 扩展
    在导入时已编译完毕，因此本函数为空操作（no-op）。保留接口
    以避免修改调用方代码。
    """
    pass  # cyna 无需预编译


# ---------------------------------------------------------------------------
# trace_poincare_batch — 单截面批量追踪
# ---------------------------------------------------------------------------

def trace_poincare_batch(R_seeds, Z_seeds, phi_section, N_turns, DPhi,
                         R_grid, Z_grid, Phi_grid,
                         BR_flat, BPhi_flat, BZ_flat,
                         wall_R, wall_Z):
    """
    批量追踪场线并记录 Poincaré 截面交叉点（单截面）。

    底层调用 cyna C++ 扩展 trace_poincare_batch。

    Parameters
    ----------
    R_seeds, Z_seeds : 1-D float64 arrays, shape (N_seeds,)
    phi_section : float，Poincaré 截面的环向角 [rad]
    N_turns : int，追踪的环向圈数
    DPhi : float，RK4 步长 [rad]
    R_grid, Z_grid, Phi_grid : 1-D float64 arrays（网格轴）
    BR_flat, BPhi_flat, BZ_flat : flat float64 arrays (NR*NZ*NPhi)
    wall_R, wall_Z : 1-D float64 arrays，壁面多边形顶点

    Returns
    -------
    poi_counts : int array (N_seeds,)，每个种子的交叉点数
    poi_R_flat, poi_Z_flat : flat arrays
        种子 s 的交叉点在 [s*N_turns : s*N_turns + poi_counts[s]]
    """
    if not _cyna_available():
        raise ImportError(
            "pyna._cyna C++ 扩展不可用。请先编译 cyna（运行 build_cyna.bat）。"
        )
    return _cyna_trace_poincare_batch(
        np.ascontiguousarray(R_seeds,   dtype=np.float64),
        np.ascontiguousarray(Z_seeds,   dtype=np.float64),
        float(phi_section), int(N_turns), float(DPhi),
        np.ascontiguousarray(BR_flat,   dtype=np.float64),
        np.ascontiguousarray(BPhi_flat, dtype=np.float64),
        np.ascontiguousarray(BZ_flat,   dtype=np.float64),
        np.ascontiguousarray(R_grid,    dtype=np.float64),
        np.ascontiguousarray(Z_grid,    dtype=np.float64),
        np.ascontiguousarray(Phi_grid,  dtype=np.float64),
        np.ascontiguousarray(wall_R,    dtype=np.float64),
        np.ascontiguousarray(wall_Z,    dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# trace_poincare_multi_batch — 多截面批量追踪
# ---------------------------------------------------------------------------

def trace_poincare_multi_batch(R_seeds, Z_seeds, phi_sections_arr, N_turns, DPhi,
                                R_grid, Z_grid, Phi_grid,
                                BR_flat, BPhi_flat, BZ_flat,
                                wall_R, wall_Z):
    """
    批量追踪场线并记录多个 Poincaré 截面的交叉点。

    底层调用 cyna C++ 扩展 trace_poincare_multi。
    返回的 poi_counts 已 reshape 为 (N_seeds, N_sections)，
    与原 numba 版本保持一致。

    Parameters
    ----------
    phi_sections_arr : 1-D float64 array，截面角度列表 [rad]

    Returns
    -------
    poi_counts : int array (N_seeds, N_sections)
    poi_R_flat, poi_Z_flat : flat arrays
        布局：[seed0_sec0 pts..., seed0_sec1 pts..., seed1_sec0 pts..., ...]
        种子 s、截面 sec 的数据在：
            offset = (s * N_sections + sec) * N_turns
            slice  = offset : offset + poi_counts[s, sec]
    """
    if not _cyna_available():
        raise ImportError(
            "pyna._cyna C++ 扩展不可用。请先编译 cyna（运行 build_cyna.bat）。"
        )
    counts, pR, pZ = _cyna_trace_poincare_multi(
        np.ascontiguousarray(R_seeds,          dtype=np.float64),
        np.ascontiguousarray(Z_seeds,          dtype=np.float64),
        np.ascontiguousarray(phi_sections_arr, dtype=np.float64),
        int(N_turns), float(DPhi),
        np.ascontiguousarray(BR_flat,          dtype=np.float64),
        np.ascontiguousarray(BPhi_flat,        dtype=np.float64),
        np.ascontiguousarray(BZ_flat,          dtype=np.float64),
        np.ascontiguousarray(R_grid,           dtype=np.float64),
        np.ascontiguousarray(Z_grid,           dtype=np.float64),
        np.ascontiguousarray(Phi_grid,         dtype=np.float64),
        np.ascontiguousarray(wall_R,           dtype=np.float64),
        np.ascontiguousarray(wall_Z,           dtype=np.float64),
    )
    # cyna 返回 poi_counts 为一维，需 reshape 为 (N_seeds, N_sections)
    N_seeds    = len(R_seeds)
    N_sections = len(phi_sections_arr)
    return counts.reshape(N_seeds, N_sections), pR, pZ


__all__ = [
    "field_arrays_from_interpolators",
    "precompile_tracer",
    "trace_poincare_batch",
    "trace_poincare_multi_batch",
    "trace_poincare_batch_twall",
    "find_fixed_points_batch",
    "trace_orbit_along_phi",
]


# ---------------------------------------------------------------------------
# trace_poincare_batch_twall — 含 toroidal 3-D wall 的批量追踪
# ---------------------------------------------------------------------------

def trace_poincare_batch_twall(R_seeds, Z_seeds, phi_section, N_turns, DPhi,
                                R_grid, Z_grid, Phi_grid,
                                BR_flat, BPhi_flat, BZ_flat,
                                wall_phi, wall_R_all, wall_Z_all):
    """
    批量追踪场线，使用 toroidal 3-D wall（分段面 wall），记录 Poincaré 截面交叉点。

    底层调用 cyna C++ 扩展 trace_poincare_batch_twall。

    Parameters
    ----------
    wall_phi    : 1-D float64 array，各 wall 截面的环向角
    wall_R_all  : 2-D float64 array (N_phi, N_poly)，各截面 R 坐标
    wall_Z_all  : 2-D float64 array (N_phi, N_poly)，各截面 Z 坐标
    """
    if not _cyna_available() or _cyna_trace_poincare_batch_twall is None:
        raise ImportError(
            "pyna._cyna.trace_poincare_batch_twall 不可用。请先编译 cyna（运行 build_cyna.bat）。"
        )
    return _cyna_trace_poincare_batch_twall(
        np.ascontiguousarray(R_seeds,   dtype=np.float64),
        np.ascontiguousarray(Z_seeds,   dtype=np.float64),
        float(phi_section), int(N_turns), float(DPhi),
        np.ascontiguousarray(BR_flat,   dtype=np.float64),
        np.ascontiguousarray(BPhi_flat, dtype=np.float64),
        np.ascontiguousarray(BZ_flat,   dtype=np.float64),
        np.ascontiguousarray(R_grid,    dtype=np.float64),
        np.ascontiguousarray(Z_grid,    dtype=np.float64),
        np.ascontiguousarray(Phi_grid,  dtype=np.float64),
        np.ascontiguousarray(wall_phi,  dtype=np.float64),
        np.ascontiguousarray(wall_R_all, dtype=np.float64),
        np.ascontiguousarray(wall_Z_all, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# find_fixed_points_batch — 批量寻找不动点
# ---------------------------------------------------------------------------

def find_fixed_points_batch(R_seeds, Z_seeds, phi_start, period, N_periods, DPhi,
                             R_grid, Z_grid, Phi_grid,
                             BR_flat, BPhi_flat, BZ_flat,
                             **kwargs):
    """
    从给定种子点出发批量搜索庞加莱映射的不动点（fixed points）。

    底层调用 cyna C++ 扩展 find_fixed_points_batch。
    额外关键字参数透传给底层（tol、max_iter 等）。
    """
    if not _cyna_available() or _cyna_find_fixed_points_batch is None:
        raise ImportError(
            "pyna._cyna.find_fixed_points_batch 不可用。请先编译 cyna（运行 build_cyna.bat）。"
        )
    return _cyna_find_fixed_points_batch(
        np.ascontiguousarray(R_seeds,   dtype=np.float64),
        np.ascontiguousarray(Z_seeds,   dtype=np.float64),
        float(phi_start), int(period), int(N_periods), float(DPhi),
        np.ascontiguousarray(BR_flat,   dtype=np.float64),
        np.ascontiguousarray(BPhi_flat, dtype=np.float64),
        np.ascontiguousarray(BZ_flat,   dtype=np.float64),
        np.ascontiguousarray(R_grid,    dtype=np.float64),
        np.ascontiguousarray(Z_grid,    dtype=np.float64),
        np.ascontiguousarray(Phi_grid,  dtype=np.float64),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# trace_orbit_along_phi — 沿环向角追踪单条轨道
# ---------------------------------------------------------------------------

def trace_orbit_along_phi(R0, Z0, phi_start, phi_end, DPhi,
                           R_grid, Z_grid, Phi_grid,
                           BR_flat, BPhi_flat, BZ_flat):
    """
    从 (R0, Z0, phi_start) 出发，沿环向角追踪单条场线轨道直至 phi_end。

    底层调用 cyna C++ 扩展 trace_orbit_along_phi。

    Returns
    -------
    R_out, Z_out, Phi_out : 1-D float64 arrays，轨道点序列
    """
    if not _cyna_available() or _cyna_trace_orbit_along_phi is None:
        raise ImportError(
            "pyna._cyna.trace_orbit_along_phi 不可用。请先编译 cyna（运行 build_cyna.bat）。"
        )
    return _cyna_trace_orbit_along_phi(
        float(R0), float(Z0), float(phi_start), float(phi_end), float(DPhi),
        np.ascontiguousarray(BR_flat,   dtype=np.float64),
        np.ascontiguousarray(BPhi_flat, dtype=np.float64),
        np.ascontiguousarray(BZ_flat,   dtype=np.float64),
        np.ascontiguousarray(R_grid,    dtype=np.float64),
        np.ascontiguousarray(Z_grid,    dtype=np.float64),
        np.ascontiguousarray(Phi_grid,  dtype=np.float64),
    )
