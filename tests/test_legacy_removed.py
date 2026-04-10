"""
tests/test_legacy_removed.py
——验证已移除的 legacy API 确实不再存在，防止意外回退。

批次 1: flt.py legacy API、numba_poincare.py numba 依赖
批次 2: pyna.flowsol、pyna.poincare（顶层旧封装）、pyna.MCF 目录大写修正
         + pyna.topo.* 不再直接 from pyna._cyna import，改为 from pyna.MCF.flt import
"""
import pytest


# ---------------------------------------------------------------------------
# Batch 1: flt.py legacy functions removed
# ---------------------------------------------------------------------------

def test_bundle_tracing_module_level_gone():
    """module-level bundle_tracing_with_t_as_DeltaPhi 已删除。"""
    import pyna.flt as flt_mod
    assert not hasattr(flt_mod, "bundle_tracing_with_t_as_DeltaPhi"), (
        "bundle_tracing_with_t_as_DeltaPhi should have been removed from pyna.flt"
    )


def test_bundle_tracing_method_gone():
    """FieldLineTracer.bundle_tracing_with_t_as_DeltaPhi shim 已删除。"""
    from pyna.flt import FieldLineTracer
    assert not hasattr(FieldLineTracer, "bundle_tracing_with_t_as_DeltaPhi"), (
        "FieldLineTracer.bundle_tracing_with_t_as_DeltaPhi shim should be removed"
    )


def test_save_load_poincare_orbits_gone():
    """save/load_Poincare_orbits 已删除。"""
    import pyna.flt as flt_mod
    assert not hasattr(flt_mod, "save_Poincare_orbits"), (
        "save_Poincare_orbits should have been removed"
    )
    assert not hasattr(flt_mod, "load_Poincare_orbits"), (
        "load_Poincare_orbits should have been removed"
    )


def test_odesolution_monkey_patch_gone():
    """OdeSolution.mat_interp monkey-patch 已移除，不应注入外部类。"""
    from scipy.integrate import OdeSolution
    assert not hasattr(OdeSolution, "mat_interp"), (
        "OdeSolution.mat_interp monkey-patch should have been removed"
    )


def test_scipy_solve_ivp_not_imported_in_flt():
    """flt.py 不再直接 import solve_ivp（已移除 legacy 依赖）。"""
    import pyna.flt as flt_mod
    assert not hasattr(flt_mod, "solve_ivp"), (
        "solve_ivp should not be a public name in pyna.flt"
    )
    assert not hasattr(flt_mod, "_scipy_solve_ivp"), (
        "_scipy_solve_ivp should not remain in pyna.flt after legacy removal"
    )


# ---------------------------------------------------------------------------
# Batch 1: numba_poincare — numba 不再引入
# ---------------------------------------------------------------------------

def test_numba_not_loaded_after_pyna_mcf_flt():
    """导入 pyna.mcf.flt.numba_poincare 后 numba 不应出现在 sys.modules。"""
    import sys

    if "numba" in sys.modules:
        pytest.skip("numba already loaded by another test/import; skip isolation check")

    # pyna.mcf.__init__ 有已知 pyna.MCF 大写 bug，直接从文件加载绕过
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "_test_numba_poincare",
        str(pathlib.Path(__file__).parent.parent / "pyna" / "mcf" / "flt" / "numba_poincare.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert "numba" not in sys.modules, (
        "numba should NOT be imported after loading pyna.mcf.flt.numba_poincare"
    )


def test_precompile_tracer_is_noop():
    """precompile_tracer 应是 no-op，不应抛异常也不应有副作用。"""
    import sys, importlib.util, pathlib
    import numpy as np

    # 直接从文件加载，绕过 pyna.mcf.__init__ 大写 bug
    spec = importlib.util.spec_from_file_location(
        "_test_numba_poincare2",
        str(pathlib.Path(__file__).parent.parent / "pyna" / "mcf" / "flt" / "numba_poincare.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    precompile_tracer = mod.precompile_tracer
    field_arrays_from_interpolators = mod.field_arrays_from_interpolators

    from scipy.interpolate import RegularGridInterpolator

    R = np.linspace(0.5, 1.5, 4)
    Z = np.linspace(-0.5, 0.5, 4)
    P = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    ones = np.ones((4, 4, 4))
    itp = RegularGridInterpolator((R, Z, P), ones)

    R_g, Z_g, P_g, BR_f, BP_f, BZ_f, nx, ny, nz = field_arrays_from_interpolators(itp, itp, itp)
    result = precompile_tracer(R_g, Z_g, P_g, BR_f, BP_f, BZ_f)
    assert result is None, "precompile_tracer should return None (no-op)"


# ---------------------------------------------------------------------------
# Batch 2: pyna.flowsol 和 pyna.poincare（顶层旧封装）已删除
# ---------------------------------------------------------------------------

def test_flowsol_module_gone():
    """pyna.flowsol 已删除，不应能导入。"""
    import importlib.util
    spec = importlib.util.find_spec("pyna.flowsol")
    assert spec is None, "pyna.flowsol should have been removed"


def test_top_level_poincare_module_gone():
    """pyna.poincare（顶层旧 solve_ivp 封装）已删除，不应能导入。
    注：pyna.topo.poincare 是保留的，这里只测顶层 pyna.poincare。"""
    import importlib.util
    spec = importlib.util.find_spec("pyna.poincare")
    # 若 spec 不为 None，应检查它不是旧的 solve_ivp 版本
    if spec is not None:
        # 允许 spec 存在但不含旧函数
        import importlib
        m = importlib.import_module("pyna.poincare")
        assert not hasattr(m, "poincare_FlowCallable_2_MapCallable"), (
            "pyna.poincare.poincare_FlowCallable_2_MapCallable (legacy solve_ivp wrapper) should be removed"
        )


# ---------------------------------------------------------------------------
# Batch 2: pyna.MCF 目录大写 — 所有子包可正常导入
# ---------------------------------------------------------------------------

def test_pyna_MCF_importable():
    """pyna.MCF（大写）应可正常导入。"""
    import pyna.MCF
    assert pyna.MCF is not None


def test_pyna_MCF_equilibrium_importable():
    """pyna.MCF.equilibrium 子包可正常导入。"""
    from pyna.MCF.equilibrium.Solovev import EquilibriumSolovev
    assert EquilibriumSolovev is not None


def test_pyna_MCF_flt_importable():
    """pyna.MCF.flt（cyna 包装层）可正常导入，包含所有公开符号。"""
    from pyna.MCF.flt import (
        trace_poincare_batch,
        trace_poincare_multi_batch,
        trace_poincare_batch_twall,
        find_fixed_points_batch,
        trace_orbit_along_phi,
        precompile_tracer,
        field_arrays_from_interpolators,
    )
    assert callable(trace_poincare_batch)
    assert callable(precompile_tracer)


# ---------------------------------------------------------------------------
# Batch 2+: pyna.topo.* 不再直接调用 pyna._cyna，改通过 pyna.MCF.flt
# ---------------------------------------------------------------------------

def _grep_direct_cyna(filename):
    """返回文件中直接 'from pyna._cyna import' 的行列表。"""
    import ast, pathlib
    src = pathlib.Path(filename).read_text(encoding="utf-8")
    lines = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("pyna._cyna"):
                lines.append(node.lineno)
    return lines


def test_topo_island_chain_no_direct_cyna():
    """topo/island_chain.py has been removed (along with IslandChainOrbit/ChainFixedPoint).
    Verify the file no longer exists."""
    import pathlib
    f = pathlib.Path(__file__).parent.parent / "pyna" / "topo" / "island_chain.py"
    assert not f.exists(), (
        "island_chain.py still exists but should have been removed. "
        "IslandChainOrbit and ChainFixedPoint are no longer supported."
    )


def test_topo_manifold_improve_no_direct_cyna():
    """topo/manifold_improve.py 不再直接 from pyna._cyna import。"""
    import pathlib
    f = pathlib.Path(__file__).parent.parent / "pyna" / "topo" / "manifold_improve.py"
    assert _grep_direct_cyna(f) == [], (
        f"manifold_improve.py still has direct 'from pyna._cyna import' at lines: "
        f"{_grep_direct_cyna(f)}"
    )


def test_topo_monodromy_no_direct_cyna():
    """topo/monodromy.py 不再直接 from pyna._cyna import。"""
    import pathlib
    f = pathlib.Path(__file__).parent.parent / "pyna" / "topo" / "monodromy.py"
    assert _grep_direct_cyna(f) == [], (
        f"monodromy.py still has direct 'from pyna._cyna import' at lines: "
        f"{_grep_direct_cyna(f)}"
    )
