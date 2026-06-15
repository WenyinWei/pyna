import numpy as np

from pyna.toroidal.minor_radius import GeometricMinorRadiusProvider, minor_radius_label


def test_minor_radius_label_uses_phi_dependent_axis_without_angles():
    R = np.array([1.0, 1.5, 2.0])
    Z = np.array([-0.5, 0.0, 0.5])
    Phi = np.array([0.0, 0.1])
    axis_R = np.array([1.0, 1.5])
    axis_Z = np.array([0.0, 0.5])

    rho = minor_radius_label(R, Z, Phi, axis_R, axis_Z, a_eff=0.5, clip=False)

    assert rho.shape == (3, 3, 2)
    assert rho[0, 1, 0] == 0.0
    assert rho[1, 2, 1] == 0.0
    np.testing.assert_allclose(rho[2, 1, 0], 2.0)


def test_minor_radius_label_clips_by_default_for_legacy_pressure_masks():
    rho = minor_radius_label(
        np.array([0.0, 2.0]),
        np.array([0.0]),
        np.array([0.0]),
        axis_R=0.0,
        axis_Z=0.0,
        a_eff=1.0,
    )

    np.testing.assert_allclose(rho[:, 0, 0], [0.0, 1.0])


def test_geometric_minor_radius_provider_uses_config_a_eff():
    class Config:
        a_eff = 2.0

    provider = GeometricMinorRadiusProvider(clip=False)
    eqd = {
        "R": np.array([1.0, 3.0]),
        "Z": np.array([0.0]),
        "Phi": np.array([0.0]),
    }

    rho = provider(eqd, axis_R=1.0, axis_Z=0.0, config=Config())

    np.testing.assert_allclose(rho[:, 0, 0], [0.0, 1.0])
