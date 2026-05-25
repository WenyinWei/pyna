import numpy as np

from pyna.plot.poloidal import robust_field_max, stream_field_from_components


def test_stream_field_accepts_rz_layout():
    R = np.linspace(1.0, 2.0, 5)
    Z = np.linspace(-0.5, 0.5, 7)
    BR = np.ones((R.size, Z.size))
    BZ = np.full((R.size, Z.size), 2.0)

    field = stream_field_from_components(R, Z, BR, BZ)

    assert field.u.shape == (Z.size, R.size)
    assert np.all(field.u == 1.0)
    assert np.all(field.v == 2.0)


def test_stream_field_accepts_streamplot_layout():
    R = np.linspace(1.0, 2.0, 5)
    Z = np.linspace(-0.5, 0.5, 7)
    BR = np.ones((Z.size, R.size))
    BZ = np.full((Z.size, R.size), 2.0)

    field = stream_field_from_components(R, Z, BR, BZ)

    assert field.u.shape == (Z.size, R.size)
    assert np.all(field.magnitude == np.sqrt(5.0))


def test_robust_field_max_ignores_nan():
    values = np.array([0.0, 1.0, 2.0, np.nan])

    scale = robust_field_max(values, percentile=100.0)

    assert scale == 2.0
