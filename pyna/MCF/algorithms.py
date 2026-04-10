"""pyna/mcf/algorithms.py
========================
MCF 热路径：cyna C++ 直接调用，返回新不变集类层次对象。

所有数值计算在 C++ 完成，Python 只做结果封装。
禁止在此模块使用 scipy.solve_ivp 或任何 Python ODE solver。
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from pyna.topo.invariants import (
    Cycle, Tube, TubeChain, FixedPoint, MonodromyData, Stability,
    Island, IslandChain,
)


def build_tube_chain_from_cyna(
    field_cache: dict,
    period: int,
    known_n: int,
    phi0: float = 0.0,
    section_phis: list | None = None,
    Np: int = 1,
    R_axis: float | None = None,
    Z_axis: float | None = None,
    seeds_R: np.ndarray | None = None,
    seeds_Z: np.ndarray | None = None,
    DPhi: float = 0.05,
    fd_eps: float = 1e-4,
    max_iter: int = 40,
    tol: float = 1e-9,
    n_threads: int = -1,
) -> tuple[TubeChain, Tube | None]:
    """
    用 cyna batch Newton 搜索固定点，构建完整 TubeChain。

    同时构建磁轴 Tube（period=1 O-cycle），挂载为 TubeChain 的根。

    Parameters
    ----------
    field_cache : dict
        包含 BR/BPhi/BZ/R_grid/Z_grid/Phi_grid 的场数据 dict
    period : int
        m（轨道追踪圈数）
    known_n : int
        n（极向绕数），用于 Cycle.winding=(period, known_n)
    phi0 : float
        主截面 phi 值（rad）
    section_phis : list[float] | None
        需要计算截面点的 phi 列表（默认仅 phi0）
    Np : int
        设备对称数
    R_axis, Z_axis : float | None
        磁轴坐标（用于根 Tube 和排序）
    seeds_R, seeds_Z : ndarray | None
        Newton 搜索初始种子；必须提供
    n_threads : int
        cyna 线程数，-1 = 自动

    Returns
    -------
    tube_chain : TubeChain
        完整 3D 共振结构，X/O Cycle 已填充截面数据
    axis_tube : Tube | None
        磁轴根 Tube（period=1 O-cycle）；R_axis/Z_axis 未提供时为 None
    """
    from pyna._cyna import find_fixed_points_batch as _ffpb

    if seeds_R is None or seeds_Z is None:
        raise ValueError("seeds_R and seeds_Z must be provided")

    if _ffpb is None:
        raise RuntimeError("cyna extension not available; cannot run build_tube_chain_from_cyna")

    # ── 准备 cyna 场数据 ───────────────────────────────────────────────────────
    from pyna._cyna.utils import ensure_c_double, prepare_field_cache

    fc_cyna = prepare_field_cache(field_cache)

    # ── cyna batch Newton（phi0 截面）─────────────────────────────────────────
    R_o, Z_o, _, conv, DPm_flat, _, _, ptype = _ffpb(
        ensure_c_double(np.asarray(seeds_R).ravel()),
        ensure_c_double(np.asarray(seeds_Z).ravel()),
        phi0, period,
        DPhi=DPhi, fd_eps=fd_eps, max_iter=max_iter, tol=tol,
        n_threads=n_threads, **fc_cyna,
    )

    # ── 分拣 X/O seeds ─────────────────────────────────────────────────────────
    # DPm_flat shape: (N, 4) or (N, 2, 2) — normalise to (N, 2, 2)
    if DPm_flat.ndim == 3:
        DPm_arr = DPm_flat          # (N, 2, 2)
    elif DPm_flat.ndim == 2 and DPm_flat.shape[1] == 4:
        DPm_arr = DPm_flat.reshape(-1, 2, 2)
    elif DPm_flat.ndim == 1:
        # single seed → (4,) or unpacked
        DPm_arr = DPm_flat.reshape(-1, 2, 2)
    else:
        # fallback: treat last two dims as 2×2
        DPm_arr = DPm_flat.reshape(len(seeds_R), 2, 2)

    x_seeds, o_seeds = [], []
    x_DPms,  o_DPms  = [], []
    for i in range(len(seeds_R)):
        if not conv[i]:
            continue
        DPm = DPm_arr[i]
        fp_kind = int(ptype[i])
        if fp_kind == 1:
            x_seeds.append((float(R_o[i]), float(Z_o[i])))
            x_DPms.append(DPm)
        else:
            o_seeds.append((float(R_o[i]), float(Z_o[i])))
            o_DPms.append(DPm)

    # ── 对每个 seed 追踪多截面（section_phis）─────────────────────────────────
    section_phis = section_phis or [phi0]
    winding = (period, known_n)

    def _build_cycle(seeds_rz, DPms, is_x: bool) -> list:
        """为每个 seed 在所有截面追踪，构建 Cycle 列表。"""
        cycles = []
        for (R0, Z0), DPm0 in zip(seeds_rz, DPms):
            sections: dict = {}
            for phi in section_phis:
                if abs(phi - phi0) < 1e-9:
                    # phi0 截面直接用 Newton 结果
                    fp = FixedPoint(phi=phi, R=R0, Z=Z0, DPm=DPm0)
                    sections[phi] = [fp]
                else:
                    # 其他截面：用 cyna 再做一次 Newton（phi=phi 截面）
                    R_phi, Z_phi, _, conv_phi, DPm_phi_flat, _, _, ptype_phi = _ffpb(
                        np.array([R0], dtype=np.float64),
                        np.array([Z0], dtype=np.float64),
                        phi, period,
                        DPhi=DPhi, fd_eps=fd_eps, max_iter=max_iter, tol=tol,
                        n_threads=1, **fc_cyna,
                    )
                    if conv_phi[0]:
                        if DPm_phi_flat.ndim == 3:
                            DPm_phi = DPm_phi_flat[0]
                        elif DPm_phi_flat.ndim == 2 and DPm_phi_flat.shape[1] == 4:
                            DPm_phi = DPm_phi_flat[0].reshape(2, 2)
                        else:
                            DPm_phi = DPm_phi_flat.reshape(2, 2)
                        fp_phi = FixedPoint(phi=phi, R=float(R_phi[0]), Z=float(Z_phi[0]), DPm=DPm_phi)
                        sections[phi] = [fp_phi]

            if not sections:
                continue

            mono = MonodromyData(DPm=DPm0, eigenvalues=np.linalg.eigvals(DPm0))
            cycles.append(Cycle(winding=winding, sections=sections, monodromy=mono))
        return cycles

    x_cycles = _build_cycle(x_seeds, x_DPms, is_x=True)
    o_cycles = _build_cycle(o_seeds, o_DPms, is_x=False)

    # ── 构建磁轴根 Tube ────────────────────────────────────────────────────────
    axis_tube = None
    if R_axis is not None and Z_axis is not None:
        axis_sections: dict = {}
        for phi in section_phis:
            fp_ax = FixedPoint(phi=phi, R=R_axis, Z=Z_axis, DPm=np.eye(2))
            axis_sections[phi] = [fp_ax]
        axis_cycle = Cycle(winding=(1, 0), sections=axis_sections)
        axis_tube = Tube(O_cycle=axis_cycle, X_cycles=[])

    # ── 构建 Tube 列表（每个 O-cycle 对应一个 Tube）───────────────────────────
    tubes = [Tube(O_cycle=oc, X_cycles=x_cycles) for oc in o_cycles]

    # ── 构建 TubeChain ─────────────────────────────────────────────────────────
    tc = TubeChain(O_cycles=o_cycles, X_cycles=x_cycles, tubes=tubes)

    # 挂载到磁轴根
    if axis_tube is not None:
        axis_tube.add_child_chain(tc)
        for tube in tubes:
            tube.parent_chain = tc

    return tc, axis_tube


# ---------------------------------------------------------------------------
# High-level factory: grid-scan → Newton → TubeChain
# ---------------------------------------------------------------------------

class _CynaTracerAdapter:
    """Minimal tracer adapter wrapping a field_cache for fixed-point search.

    Provides the interface expected by ``_extract_field_cache`` in
    ``pyna.topo.fixed_points`` so that ``find_island_chain_fixed_points``
    can operate on a raw field-cache dict without a full FieldlineTracer.
    """

    def __init__(self, field_cache: dict) -> None:
        self._fc = field_cache
        self.R_grid = np.ascontiguousarray(field_cache['R_grid'], dtype=np.float64)
        self.Z_grid = np.ascontiguousarray(field_cache['Z_grid'], dtype=np.float64)
        self.Phi_grid = np.ascontiguousarray(field_cache['Phi_grid'], dtype=np.float64)
        # Dummy interpolator objects — _grid_values just returns the stored array.
        self.itp_BR = object()
        self.itp_BPhi = object()
        self.itp_BZ = object()
        self._grid_BR = np.asarray(field_cache['BR'], dtype=np.float64)
        self._grid_BPhi = np.asarray(field_cache['BPhi'], dtype=np.float64)
        self._grid_BZ = np.asarray(field_cache['BZ'], dtype=np.float64)

    def _grid_values(self, itp) -> np.ndarray:
        """Return the grid array for the given interpolator placeholder."""
        if itp is self.itp_BR:
            return self._grid_BR
        if itp is self.itp_BPhi:
            return self._grid_BPhi
        if itp is self.itp_BZ:
            return self._grid_BZ
        raise ValueError(f"Unknown interpolator: {itp!r}")


def find_and_build_tube_chain(
    field_cache: dict,
    period: int,
    known_n: int,
    R_axis: float,
    Z_axis: float,
    phi0: float = 0.0,
    section_phis: list | None = None,
    Np: int = 1,
    *,
    r_min: float = 0.02,
    r_max: float = 0.25,
    n_r: int = 8,
    n_ang: int = 48,
    DPhi: float = 0.05,
    fd_eps: float = 1e-4,
    max_iter: int = 40,
    tol: float = 1e-9,
    n_threads: int = -1,
) -> tuple:
    """Find island-chain fixed points and build a TubeChain.

    High-level factory that combines:
    1. Grid-scan + Newton to find all period-m fixed points
    2. cyna batch Newton to refine and build Cycle/TubeChain objects

    Parameters
    ----------
    field_cache : dict
        Field cache with keys 'BR', 'BPhi', 'BZ', 'R_grid', 'Z_grid', 'Phi_grid'.
    period : int
        Island chain period m (number of toroidal turns).
    known_n : int
        Poloidal winding number n.
    R_axis, Z_axis : float
        Approximate magnetic axis coordinates [m].
    phi0 : float
        Seed section angle [rad]. Default 0.
    section_phis : list[float] | None
        Sections to compute fixed points at. Defaults to [phi0].
    Np : int
        Field toroidal periodicity.
    r_min, r_max : float
        Radial search range relative to magnetic axis [m].
    n_r : int
        Number of radial grid points for coarse scan.
    n_ang : int
        Number of angular grid points for coarse scan.
    DPhi, fd_eps, max_iter, tol, n_threads :
        Passed through to cyna Newton solver.

    Returns
    -------
    tube_chain : TubeChain
    axis_tube : Tube | None
    """
    from pyna.topo.fixed_points import find_island_chain_fixed_points

    # Build a minimal tracer adapter for the fixed-point search
    tracer = _CynaTracerAdapter(field_cache)

    # Grid-scan + Newton to find all period-m X/O fixed points
    fp_list = find_island_chain_fixed_points(
        tracer,
        R_axis,
        Z_axis,
        period,
        phi0,
        r_min=r_min,
        r_max=r_max,
        n_r=n_r,
        n_ang=n_ang,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
    )

    if not fp_list:
        # No fixed points found — return empty chain
        tc = TubeChain(O_cycles=[], X_cycles=[], tubes=[])
        axis_tube = None
        if R_axis is not None and Z_axis is not None:
            axis_sections = {}
            for phi in (section_phis or [phi0]):
                fp_ax = FixedPoint(phi=phi, R=R_axis, Z=Z_axis, DPm=np.eye(2))
                axis_sections[phi] = [fp_ax]
            axis_cycle = Cycle(winding=(1, 0), sections=axis_sections)
            axis_tube = Tube(O_cycle=axis_cycle, X_cycles=[])
        return tc, axis_tube

    # Extract converged seeds by kind
    x_seeds_rz, o_seeds_rz = [], []
    x_DPms, o_DPms = [], []
    for fp in fp_list:
        R = float(fp['R'])
        Z = float(fp['Z'])
        DPm = np.asarray(fp['DPm'], dtype=float)
        kind = fp['kind']
        if kind == 'X':
            x_seeds_rz.append((R, Z))
            x_DPms.append(DPm)
        else:
            o_seeds_rz.append((R, Z))
            o_DPms.append(DPm)

    # Use O-seeds as initial guesses for the full TubeChain build
    # (X-seeds are included via the Tube/O-cycle → X-cycle wiring)
    all_seeds = o_seeds_rz + x_seeds_rz
    if not all_seeds:
        tc = TubeChain(O_cycles=[], X_cycles=[], tubes=[])
        axis_tube = None
        return tc, axis_tube

    seeds_R = np.array([s[0] for s in all_seeds], dtype=np.float64)
    seeds_Z = np.array([s[1] for s in all_seeds], dtype=np.float64)

    return build_tube_chain_from_cyna(
        field_cache=field_cache,
        period=period,
        known_n=known_n,
        phi0=phi0,
        section_phis=section_phis,
        Np=Np,
        R_axis=R_axis,
        Z_axis=Z_axis,
        seeds_R=seeds_R,
        seeds_Z=seeds_Z,
        DPhi=DPhi,
        fd_eps=fd_eps,
        max_iter=max_iter,
        tol=tol,
        n_threads=n_threads,
    )
