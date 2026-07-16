from __future__ import annotations

import numpy as np
import pytest

from pyna.toroidal.coords import build_vmec_pest_coordinates_from_wout


def _write_synthetic_wout(path, *, lasym: bool = False) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    ns = 4
    with netcdf4.Dataset(path, "w") as dataset:
        dataset.createDimension("radius", ns)
        dataset.createDimension("mn_mode", 2)
        dataset.createDimension("mn_mode_nyq", 1)
        dataset.createVariable("nfp", "i4")[...] = 5
        dataset.createVariable("ns", "i4")[...] = ns
        dataset.createVariable("lasym__logical__", "i4")[...] = int(lasym)
        dataset.createVariable("xm", "f8", ("mn_mode",))[:] = [0.0, 1.0]
        dataset.createVariable("xn", "f8", ("mn_mode",))[:] = [0.0, 0.0]
        dataset.createVariable("xm_nyq", "f8", ("mn_mode_nyq",))[:] = [0.0]
        dataset.createVariable("xn_nyq", "f8", ("mn_mode_nyq",))[:] = [0.0]
        s = np.linspace(0.0, 1.0, ns)
        radius = 0.15 + 0.45 * s
        dataset.createVariable("rmnc", "f8", ("radius", "mn_mode"))[:] = (
            np.column_stack([np.full(ns, 5.5), radius])
        )
        dataset.createVariable("zmns", "f8", ("radius", "mn_mode"))[:] = (
            np.column_stack([np.zeros(ns), 0.8 * radius])
        )
        dataset.createVariable("lmns", "f8", ("radius", "mn_mode"))[:] = 0.0
        dataset.createVariable(
            "bsupvmnc", "f8", ("radius", "mn_mode_nyq")
        )[:] = 1.2
        dataset.createVariable("iotaf", "f8", ("radius",))[:] = 0.9 + 0.1 * s


def test_raw_wout_builder_returns_one_period_smooth_pest_coordinates(tmp_path):
    source = tmp_path / "wout_synthetic.nc"
    _write_synthetic_wout(source)

    result = build_vmec_pest_coordinates_from_wout(
        source,
        normalized_toroidal_flux=(0.4, 0.7, 1.0),
        n_phi_per_period=6,
        n_theta=20,
    )

    coordinates = result.coordinates
    assert coordinates.nfp == 5
    assert coordinates.stores_one_field_period is True
    assert coordinates.R_surf.shape == (6, 3, 20)
    assert coordinates.Z_surf.shape == (6, 3, 20)
    np.testing.assert_allclose(coordinates.rho_vals**2, [0.4, 0.7, 1.0])
    assert result.contravariant_toroidal_field.shape == (6, 3, 20)
    np.testing.assert_allclose(result.contravariant_toroidal_field, 1.2)
    assert np.max(coordinates.phi_vals) < 2.0 * np.pi / 5.0
    assert result.metadata["native_field_period_sampling"] is True
    assert result.metadata["full_torus_intermediate_constructed"] is False
    assert result.metadata["healing_used"] is False
    assert "source_path" not in result.metadata


def test_raw_wout_builder_rejects_asymmetric_equilibrium(tmp_path):
    source = tmp_path / "wout_asymmetric.nc"
    _write_synthetic_wout(source, lasym=True)

    with pytest.raises(NotImplementedError, match="asymmetric"):
        build_vmec_pest_coordinates_from_wout(source)
