import matplotlib
import numpy as np

matplotlib.use("Agg")


def _write_linear_bphi_mgrid(path, *, nphi=8, nz=7, nr=9, a=0.3):
    from scipy.io import netcdf_file

    R = np.linspace(1.0, 1.8, nr)
    Z = np.linspace(-0.4, 0.4, nz)
    BPhi = a * R[None, None, :] * np.ones((nphi, nz, nr))
    with netcdf_file(str(path), "w") as ds:
        ds.createDimension("phi", nphi)
        ds.createDimension("zee", nz)
        ds.createDimension("rad", nr)
        ds.createDimension("dim_00001", 1)
        ds.createDimension("scalar", 1)
        ds.createVariable("br_001", "f8", ("phi", "zee", "rad"))[:] = 0.0
        ds.createVariable("bp_001", "f8", ("phi", "zee", "rad"))[:] = BPhi
        ds.createVariable("bz_001", "f8", ("phi", "zee", "rad"))[:] = 0.0
        for name, typecode, value in [
            ("ir", "i", nr),
            ("jz", "i", nz),
            ("kp", "i", nphi),
            ("nfp", "i", 1),
            ("nextcur", "i", 1),
            ("rmin", "f8", R[0]),
            ("rmax", "f8", R[-1]),
            ("zmin", "f8", Z[0]),
            ("zmax", "f8", Z[-1]),
        ]:
            ds.createVariable(name, typecode, ("scalar",))[:] = [value]
        mode = ds.createVariable("mgrid_mode", "c", ("dim_00001",))
        mode[:] = np.asarray([b"S"], dtype="S1")
    return R, Z, a


def _circular_pest(R0=1.4, a=0.3, nphi=8, nrho=5, ntheta=32):
    from pyna.toroidal.diagnostics import SmoothPestCoordinates

    phi = np.arange(nphi) * 2.0 * np.pi / nphi
    rho = np.linspace(0.0, 1.0, nrho)
    theta = np.arange(ntheta) * 2.0 * np.pi / ntheta
    R = R0 + a * rho[None, :, None] * np.cos(theta[None, None, :])
    Z = a * rho[None, :, None] * np.sin(theta[None, None, :])
    R = np.repeat(R, nphi, axis=0)
    Z = np.repeat(Z, nphi, axis=0)
    return SmoothPestCoordinates(
        R_surf=R,
        Z_surf=Z,
        rho_vals=rho,
        theta_vals=theta,
        phi_vals=phi,
        axis_R=np.full(nphi, R0),
        axis_Z=np.zeros(nphi),
    )


def test_vmec_mgrid_loader_and_cylindrical_curl(tmp_path):
    from pyna.io import MU0, compute_current_density_cylindrical, load_vmec_mgrid

    _R, _Z, a = _write_linear_bphi_mgrid(tmp_path / "linear_bphi.nc")
    field = load_vmec_mgrid(tmp_path / "linear_bphi.nc")
    current = compute_current_density_cylindrical(field)

    assert field.BPhi.shape == (8, 7, 9)
    np.testing.assert_allclose(current.JR, 0.0, atol=1.0e-8)
    np.testing.assert_allclose(current.JPhi, 0.0, atol=1.0e-8)
    np.testing.assert_allclose(current.JZ, 2.0 * a / MU0, rtol=1.0e-12)


def test_pest_current_components_for_constant_toroidal_current():
    from pyna.io import MGridCurrent
    from pyna.toroidal.diagnostics import compute_pest_current_components

    R = np.linspace(0.9, 1.9, 16)
    Z = np.linspace(-0.5, 0.5, 14)
    phi = np.arange(8) * 2.0 * np.pi / 8
    shape = (phi.size, Z.size, R.size)
    J0 = 2.5
    current = MGridCurrent(
        R=R,
        Z=Z,
        phi=phi,
        JR=np.zeros(shape),
        JPhi=np.full(shape, J0),
        JZ=np.zeros(shape),
        nfp=1,
        period=2.0 * np.pi,
    )
    coords = _circular_pest()
    diag = compute_pest_current_components(current, coords, [0.0, 90.0], label="constant")

    for section in diag.sections:
        finite = np.isfinite(section.Jphi[1:])
        np.testing.assert_allclose(section.Jrho[1:][finite], 0.0, atol=1.0e-10)
        np.testing.assert_allclose(section.Jtheta[1:][finite], 0.0, atol=1.0e-10)
        np.testing.assert_allclose((section.Jphi[1:] * section.R[1:])[finite], J0, rtol=1.0e-12)


def test_plot_pest_current_components_smoke():
    import matplotlib.pyplot as plt

    from pyna.io import MGridCurrent
    from pyna.plot import plot_pest_current_components
    from pyna.toroidal.diagnostics import compute_pest_current_components

    R = np.linspace(0.9, 1.9, 12)
    Z = np.linspace(-0.5, 0.5, 10)
    phi = np.arange(8) * 2.0 * np.pi / 8
    shape = (phi.size, Z.size, R.size)
    current = MGridCurrent(
        R=R,
        Z=Z,
        phi=phi,
        JR=np.zeros(shape),
        JPhi=np.ones(shape),
        JZ=np.zeros(shape),
        nfp=1,
        period=2.0 * np.pi,
    )
    diag = compute_pest_current_components(current, _circular_pest(nrho=4, ntheta=16), [0.0], label="test")
    fig, axes = plot_pest_current_components([diag])
    assert axes.shape == (4, 1)
    assert len(fig.axes) >= 4
    plt.close(fig)
