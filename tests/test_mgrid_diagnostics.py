import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")


def _write_linear_bphi_mgrid(path, *, nphi=8, nz=7, nr=9, a=0.3, nfp=1):
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
            ("nfp", "i", nfp),
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


def test_smooth_pest_native_field_period_uses_physical_phi_derivative_scale():
    from pyna.toroidal.diagnostics import SmoothPestCoordinates, smooth_pest_derivatives

    nfp = 5
    nphi = 80
    nrho = 3
    ntheta = 12
    period = 2.0 * np.pi / nfp
    phi = np.linspace(0.0, period, nphi, endpoint=False)
    rho = np.linspace(0.2, 0.8, nrho)
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    ripple = 0.03 * np.cos(nfp * phi)[:, None, None]
    R = 5.5 + ripple + rho[None, :, None] * np.cos(theta)[None, None, :]
    Z = np.broadcast_to(rho[None, :, None] * np.sin(theta)[None, None, :], R.shape).copy()
    coords = SmoothPestCoordinates(
        R_surf=R,
        Z_surf=Z,
        rho_vals=rho,
        theta_vals=theta,
        phi_vals=phi,
        nfp=nfp,
        toroidal_period=period,
    )

    dR_dphi = smooth_pest_derivatives(coords)[4]
    expected = np.broadcast_to((-0.03 * nfp * np.sin(nfp * phi))[:, None, None], R.shape)
    np.testing.assert_allclose(dR_dphi, expected, atol=1.7e-4, rtol=2.0e-3)


def test_smooth_pest_npz_preserves_nfp_and_native_domain_period(tmp_path):
    from pyna.toroidal.diagnostics import load_smooth_pest_coordinates

    nfp = 5
    period = 2.0 * np.pi / nfp
    phi = np.linspace(0.0, period, 6, endpoint=False)
    rho = np.linspace(0.2, 0.8, 3)
    theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    shape = (phi.size, rho.size, theta.size)
    path = tmp_path / "native_pest.npz"
    np.savez(
        path,
        R_surf=np.ones(shape),
        Z_surf=np.zeros(shape),
        rho_vals=rho,
        theta_vals=theta,
        phi_vals=phi,
        nfp=np.array(nfp),
        toroidal_period=np.array(period),
    )

    loaded = load_smooth_pest_coordinates(path)

    assert loaded.nfp == nfp
    assert loaded.period == pytest.approx(period)
    assert loaded.stores_one_field_period is True


def _rippled_pest(R0=1.4, a=0.3, nphi=4, nrho=3, ntheta=64):
    from pyna.toroidal.diagnostics import SmoothPestCoordinates

    phi = np.arange(nphi) * 2.0 * np.pi / nphi
    rho = np.linspace(0.0, 1.0, nrho)
    theta = np.arange(ntheta) * 2.0 * np.pi / ntheta
    shape = (
        np.exp(1j * theta)[None, None, :]
        + 0.06 * np.exp(7j * theta)[None, None, :]
        + 0.04 * np.exp(-4j * theta)[None, None, :]
    )
    w = a * rho[None, :, None] * shape
    R = R0 + np.repeat(w.real, nphi, axis=0)
    Z = np.repeat(w.imag, nphi, axis=0)
    return SmoothPestCoordinates(
        R_surf=R,
        Z_surf=Z,
        rho_vals=rho,
        theta_vals=theta,
        phi_vals=phi,
        axis_R=np.full(nphi, R0),
        axis_Z=np.zeros(nphi),
    )


def _add_surface_modes(coords, terms):
    from pyna.toroidal.diagnostics import SmoothPestCoordinates

    theta = coords.theta_vals
    rho = coords.rho_vals
    extra = np.zeros((1, rho.size, theta.size), dtype=np.complex128)
    for mode, amplitude in terms:
        extra += float(amplitude) * rho[None, :, None] * np.exp(1j * int(mode) * theta)[None, None, :]
    extra = np.repeat(extra, coords.R_surf.shape[0], axis=0)
    return SmoothPestCoordinates(
        R_surf=coords.R_surf + extra.real,
        Z_surf=coords.Z_surf + extra.imag,
        rho_vals=coords.rho_vals,
        theta_vals=coords.theta_vals,
        phi_vals=coords.phi_vals,
        axis_R=coords.axis_R,
        axis_Z=coords.axis_Z,
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


def test_mgrid_to_vector_field_uses_canonical_r_z_phi_order(tmp_path):
    from pyna.fields import VectorFieldCylind
    from pyna.io import load_vmec_mgrid, mgrid_to_vector_field

    _write_linear_bphi_mgrid(tmp_path / "linear_bphi.nc", nphi=6, nz=5, nr=8)
    field = load_vmec_mgrid(tmp_path / "linear_bphi.nc")
    vector = mgrid_to_vector_field(field, label="synthetic")

    assert isinstance(vector, VectorFieldCylind)
    assert vector.shape == (8, 5, 6)
    assert vector.nfp == 1
    assert vector.field_period == pytest.approx(2.0 * np.pi)
    assert vector.label == "synthetic"
    np.testing.assert_allclose(vector.BPhi[3, 2, 4], field.BPhi[4, 2, 3])
    np.testing.assert_allclose(vector.BR[3, 2, 4], field.BR[4, 2, 3])
    np.testing.assert_allclose(vector.BZ[3, 2, 4], field.BZ[4, 2, 3])


def test_full_torus_mgrid_keeps_physical_nfp_and_cyna_uses_native_period(tmp_path):
    from pyna.io import load_vmec_mgrid, mgrid_to_vector_field

    nfp = 5
    native_nphi = 6
    path = tmp_path / "full_torus.nc"
    _write_linear_bphi_mgrid(path, nphi=native_nphi, nz=5, nr=8, nfp=nfp)
    loaded = load_vmec_mgrid(path, full_torus=True)
    vector = mgrid_to_vector_field(loaded)
    arrays = vector.cyna_arrays(extend_phi=True)

    assert vector.nfp == nfp
    assert vector.field_period == pytest.approx(2.0 * np.pi / nfp)
    assert vector.periodicity.domain_period == pytest.approx(2.0 * np.pi)
    assert vector.periodicity.domain_period_count == nfp
    assert arrays.nfp == nfp
    assert arrays.Phi_grid.size == native_nphi + 1
    assert arrays.Phi_grid[-1] == pytest.approx(2.0 * np.pi / nfp)


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


def test_pest_diagnostics_wrap_coordinate_arrays_over_native_field_period():
    from pyna.io import MGridCurrent
    from pyna.toroidal.diagnostics import (
        SmoothPestCoordinates,
        compute_pest_current_components,
        periodic_phi_slice,
        surface_fourier_spectrum,
    )

    nfp = 5
    period = 2.0 * np.pi / nfp
    nphi = 8
    phi = np.arange(nphi, dtype=np.float64) * period / nphi
    rho = np.linspace(0.0, 1.0, 4)
    theta = np.arange(32, dtype=np.float64) * 2.0 * np.pi / 32
    phase = 2.0 * np.pi * phi / period
    minor_radius = 0.25 * rho[None, :, None] * (1.0 + 0.25 * np.cos(phase)[:, None, None])
    R_surf = 1.4 + minor_radius * np.cos(theta)[None, None, :]
    Z_surf = minor_radius * np.sin(theta)[None, None, :]
    coords = SmoothPestCoordinates(
        R_surf=R_surf,
        Z_surf=Z_surf,
        rho_vals=rho,
        theta_vals=theta,
        phi_vals=phi,
        axis_R=np.full(nphi, 1.4),
        axis_Z=np.zeros(nphi),
        nfp=nfp,
        toroidal_period=period,
    )
    R = np.linspace(1.0, 1.8, 20)
    Z = np.linspace(-0.4, 0.4, 18)
    shape = (nphi, Z.size, R.size)
    current = MGridCurrent(
        R=R,
        Z=Z,
        phi=phi,
        JR=np.zeros(shape),
        JPhi=np.ones(shape),
        JZ=np.zeros(shape),
        nfp=nfp,
        period=period,
    )

    section_deg = 81.0
    section_phi = np.deg2rad(section_deg)
    diag = compute_pest_current_components(current, coords, [section_deg])
    expected_R = periodic_phi_slice(coords.R_surf, section_phi, period=period)
    np.testing.assert_allclose(diag.sections[0].R, expected_R, rtol=0.0, atol=1.0e-14)

    spectrum = surface_fourier_spectrum(
        coords,
        rho_values=[1.0],
        sections_deg=[section_deg],
        mode_max=4,
        high_modes=(2, 3, 4),
    )[0]
    expected_scale = 1.0 + 0.25 * np.cos(2.0 * np.pi / nphi)
    assert spectrum["amplitudes"][1] == pytest.approx(0.25 * expected_scale)


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


def test_surface_shape_harmonic_spectrum_keeps_signed_poloidal_modes():
    from pyna.toroidal.diagnostics import surface_shape_harmonic_spectrum

    coords = _rippled_pest()
    sections = surface_shape_harmonic_spectrum(
        coords.R_surf,
        coords.Z_surf,
        coords.rho_vals,
        coords.theta_vals,
        coords.phi_vals,
        radial_values=[1.0],
        sections_phi=[0.0],
        axis_R=coords.axis_R,
        axis_Z=coords.axis_Z,
        mode_max=8,
        high_modes=[7],
    )
    assert len(sections) == 1
    amps = sections[0].abs_mode_amplitudes(8)
    np.testing.assert_allclose(amps[1], 0.3, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(amps[4], 0.012, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(amps[7], 0.018, rtol=1.0e-12, atol=1.0e-12)
    assert sections[0].high_mode_fraction > 0.0


def test_low_pass_surface_shape_harmonics_removes_high_m_without_changing_m1():
    from pyna.toroidal.diagnostics import (
        low_pass_surface_shape_harmonics,
        surface_shape_harmonic_spectrum,
    )

    coords = _rippled_pest()
    smooth_R, smooth_Z = low_pass_surface_shape_harmonics(
        coords.R_surf,
        coords.Z_surf,
        mode_cutoff=3,
        axis_R=coords.axis_R,
        axis_Z=coords.axis_Z,
    )
    sections = surface_shape_harmonic_spectrum(
        smooth_R,
        smooth_Z,
        coords.rho_vals,
        coords.theta_vals,
        coords.phi_vals,
        radial_values=[1.0],
        sections_phi=[0.0],
        axis_R=coords.axis_R,
        axis_Z=coords.axis_Z,
        mode_max=8,
        high_modes=[7],
    )
    amps = sections[0].abs_mode_amplitudes(8)
    np.testing.assert_allclose(amps[1], 0.3, rtol=1.0e-12, atol=1.0e-12)
    assert amps[4] < 1.0e-14
    assert amps[7] < 1.0e-14


def test_surface_shape_harmonic_leakage_flags_modes_outside_allowed_set():
    from pyna.toroidal.diagnostics import (
        surface_shape_harmonic_leakage,
        surface_shape_harmonic_spectrum,
    )

    base = _circular_pest(nrho=3, ntheta=64)
    allowed_only = _add_surface_modes(base, [(7, 0.02)])
    leaky = _add_surface_modes(base, [(7, 0.02), (5, 0.01)])

    def spectrum(coords):
        return surface_shape_harmonic_spectrum(
            coords.R_surf,
            coords.Z_surf,
            coords.rho_vals,
            coords.theta_vals,
            coords.phi_vals,
            radial_values=[1.0],
            sections_phi=[0.0],
            axis_R=coords.axis_R,
            axis_Z=coords.axis_Z,
            mode_max=8,
            high_modes=[5, 7],
        )

    clean_rows = surface_shape_harmonic_leakage(
        spectrum(base),
        spectrum(allowed_only),
        allowed_modes=[7],
        amplitude_floor=1.0e-12,
    )
    assert clean_rows[0].leakage_fraction < 1.0e-12
    assert clean_rows[0].leaking_modes == ()

    leaky_rows = surface_shape_harmonic_leakage(
        spectrum(base),
        spectrum(leaky),
        allowed_modes=[7],
        amplitude_floor=1.0e-12,
    )
    assert leaky_rows[0].leakage_fraction > 0.4
    assert 5 in leaky_rows[0].leaking_modes
