import numpy as np
import pytest


def _nfp2_cache():
    nfp = 2
    period = 2.0 * np.pi / nfp
    R = np.array([0.8, 1.0, 1.2])
    Z = np.array([-0.1, 0.0, 0.1])
    Phi = np.linspace(0.0, period, 16, endpoint=False)
    RR, ZZ, PP = np.meshgrid(R, Z, Phi, indexing="ij")
    BR = np.cos(2.0 * PP)
    BZ = np.zeros_like(RR)
    BPhi = np.ones_like(RR)
    return {
        "R_grid": R,
        "Z_grid": Z,
        "Phi_grid": Phi,
        "BR": BR,
        "BZ": BZ,
        "BPhi": BPhi,
    }


def test_prepare_field_cache_dict_closes_inferred_field_period():
    from pyna._cyna.utils import prepare_field_cache

    fc = _nfp2_cache()
    prepared = prepare_field_cache(fc, extend_phi=True)

    assert prepared["Phi_grid"][-1] == pytest.approx(np.pi)
    assert prepared["BR"].shape[2] == fc["BR"].shape[2] + 1
    np.testing.assert_allclose(prepared["BR"][:, :, -1], prepared["BR"][:, :, 0])


def test_close_periodic_phi_grid_does_not_double_extend_closed_field_period():
    from pyna.fields.cylindrical import close_periodic_phi_grid

    period = np.pi
    R = np.array([0.8, 1.0, 1.2])
    Z = np.array([-0.1, 0.0, 0.1])
    Phi = np.linspace(0.0, period, 17, endpoint=True)
    _, _, PP = np.meshgrid(R, Z, Phi, indexing="ij")
    BR = np.cos(2.0 * PP)

    Phi_ext, BR_ext = close_periodic_phi_grid(Phi, BR)

    assert Phi_ext.shape == Phi.shape
    assert BR_ext.shape == BR.shape
    assert Phi_ext[-1] == pytest.approx(period)


def test_prepare_field_cache_dict_extends_single_phi_axisymmetric_cache():
    from pyna._cyna.utils import prepare_field_cache

    R = np.array([0.8, 1.0, 1.2])
    Z = np.array([-0.1, 0.0, 0.1])
    Phi = np.array([0.0])
    shape = (R.size, Z.size, Phi.size)
    fc = {
        "R_grid": R,
        "Z_grid": Z,
        "Phi_grid": Phi,
        "BR": np.zeros(shape),
        "BZ": np.zeros(shape),
        "BPhi": np.ones(shape),
    }

    prepared = prepare_field_cache(fc, extend_phi=True)

    np.testing.assert_allclose(prepared["Phi_grid"], [0.0, 2.0 * np.pi])
    assert prepared["BR"].shape[2] == 2


def test_topology_eval_python_field_func_wraps_by_cache_period():
    pytest.importorskip("pyna._cyna")
    from pyna.topo.topology_eval import _FC

    fc = _FC(_nfp2_cache())
    fc.build_scipy_itps()

    vec = fc.field_func_py([1.0, 0.0, 1.5 * np.pi])

    expected_br = -1.0 / np.sqrt(2.0)
    np.testing.assert_allclose(vec, [expected_br, 0.0, 1.0 / np.sqrt(2.0)], atol=1.0e-10)


def test_monodromy_matrix_uses_current_wrapper_abi_with_field_period_cache():
    cyna = pytest.importorskip("pyna._cyna")
    if not cyna.is_available():
        pytest.skip("cyna extension is unavailable")

    from pyna.topo.monodromy import monodromy_matrix

    R0 = 1.0
    Z0 = 0.0
    omega = 0.25
    nfp = 2
    period = 2.0 * np.pi / nfp
    R = np.linspace(0.8, 1.2, 17)
    Z = np.linspace(-0.2, 0.2, 17)
    Phi = np.linspace(0.0, period, 8, endpoint=False)
    RR, ZZ, PP = np.meshgrid(R, Z, Phi, indexing="ij")
    cache = {
        "R_grid": R,
        "Z_grid": Z,
        "Phi_grid": Phi,
        "BR": -omega * (ZZ - Z0) / RR,
        "BZ": omega * (RR - R0) / RR,
        "BPhi": np.ones_like(RR),
    }

    class BoxWall:
        def get_section(self, _phi):
            return (
                np.array([0.75, 1.25, 1.25, 0.75, 0.75], dtype=float),
                np.array([-0.25, -0.25, 0.25, 0.25, -0.25], dtype=float),
            )

    DPm = monodromy_matrix(R0, Z0, 0.0, 1, cache, BoxWall(), fd_eps=1.0e-4, DPhi=0.01)
    angle = omega * 2.0 * np.pi
    expected = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )

    assert np.all(np.isfinite(DPm))
    np.testing.assert_allclose(DPm, expected, atol=4.0e-3)
