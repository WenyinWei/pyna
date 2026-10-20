"""Tests for the cyna coil-field grid builder.

Coverage:
* analytic circular loop vs the exact Smythe/Griffiths expression
* 64 x 64 x 48 production-size timing on the CPU path
* closed/open phi periodicity and the nfp span contract
* cyna vs NumPy reference agreement
* the pivot2t adapter call pattern (K, S, 2, 3 segment endpoints) and a
  downstream Poincare trace with the produced grid
* cyna as the default backend and a loud warning on the test-only NumPy path.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from scipy.special import ellipe, ellipk

import pyna.toroidal.flt.coil_field_grid as coil_grid_module
from pyna.toroidal.flt.coil_field_grid import build_coil_field_grid


MU0 = 4.0 * np.pi * 1.0e-7


def _cyna_kernel_available() -> bool:
    try:
        from pyna._cyna import _cyna_ext
    except Exception:
        return False
    return hasattr(_cyna_ext, "coil_field_grid")


CYNA_AVAILABLE = _cyna_kernel_available()
requires_cyna = pytest.mark.skipif(
    not CYNA_AVAILABLE, reason="cyna coil_field_grid binding is not built"
)


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def _circle_polyline(radius: float, n: int = 20001, z0: float = 0.0) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    pts = np.stack(
        [
            radius * np.cos(theta),
            radius * np.sin(theta),
            np.full_like(theta, z0),
        ],
        axis=-1,
    )
    pts[-1] = pts[0]
    return pts


def _exact_loop_field(radius: float, r: float, z: float, current: float):
    """Exact field of a circular loop (Smythe / Griffiths), in (Br, Bz)."""
    q = (radius + r) ** 2 + z * z
    k2 = 4.0 * radius * r / q
    kk = ellipk(k2)
    ee = ellipe(k2)
    d = (radius - r) ** 2 + z * z
    bz = MU0 * current / (2.0 * np.pi * np.sqrt(q)) * (
        kk + (radius * radius - r * r - z * z) / d * ee
    )
    br = MU0 * current / (2.0 * np.pi) * z / (r * np.sqrt(q)) * (
        -kk + (radius * radius + r * r + z * z) / d * ee
    )
    return float(br), float(bz)


def _toroidal_coil_set(n_coils: int, n_points: int = 97):
    """Closed poloidal loops distributed toroidally (benchmark geometry)."""
    coils = []
    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    for k in range(n_coils):
        phi = 2.0 * np.pi * k / n_coils
        major = 1.55 + 0.35 * np.cos(2.0 * phi)
        minor = 0.13 + 0.04 * np.sin(3.0 * phi)
        r = major + minor * np.cos(theta)
        z = minor * np.sin(theta)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        coils.append(np.stack([x, y, z], axis=-1))
    return np.stack(coils, axis=0)


# ---------------------------------------------------------------------------
# analytic correctness
# ---------------------------------------------------------------------------


def test_analytic_circular_loop_matches_exact_expression():
    radius = 1.5
    current = 1234.5
    segments = _circle_polyline(radius, n=20001)[None, ...]
    R_grid = np.array([1.7])
    Z_grid = np.array([0.3])
    Phi_grid = np.array([0.0, 2.0 * np.pi])

    result = build_coil_field_grid(
        segments, [current], R_grid, Z_grid, Phi_grid, nfp=1, backend="cyna"
    )
    br_exact, bz_exact = _exact_loop_field(radius, 1.7, 0.3, current)
    br = float(result["BR"][0, 0, 0])
    bz = float(result["BZ"][0, 0, 0])
    bphi = float(result["BPhi"][0, 0, 0])

    assert abs(br - br_exact) / abs(br_exact) < 1.0e-6
    assert abs(bz - bz_exact) / abs(bz_exact) < 1.0e-6
    assert abs(bphi) < 1.0e-12 * max(abs(br_exact), abs(bz_exact))


def test_numpy_fallback_matches_analytic_loop():
    radius = 1.5
    current = 1234.5
    segments = _circle_polyline(radius, n=20001)[None, ...]
    with pytest.warns(RuntimeWarning, match="NumPy reference backend"):
        result = build_coil_field_grid(
            segments,
            [current],
            [1.7],
            [0.3],
            [0.0, 2.0 * np.pi],
            nfp=1,
            backend="numpy",
        )
    br_exact, bz_exact = _exact_loop_field(radius, 1.7, 0.3, current)
    assert result["backend"] == "numpy-reference"
    assert abs(result["BR"][0, 0, 0] - br_exact) / abs(br_exact) < 1.0e-6
    assert abs(result["BZ"][0, 0, 0] - bz_exact) / abs(bz_exact) < 1.0e-6


def test_auto_backend_warns_and_falls_back_when_kernel_missing(monkeypatch):
    """The reference path must never be silent, even when auto-selected."""

    segments = _circle_polyline(1.4, n=41)[None, ...]
    monkeypatch.setattr(coil_grid_module, "_cyna_kernel", lambda: None)
    with pytest.warns(RuntimeWarning, match="NumPy reference backend"):
        result = build_coil_field_grid(
            segments,
            [1.0],
            [1.7],
            [0.0],
            [0.0],
            nfp=1,
            backend="auto",
        )
    assert result["backend"] == "numpy-reference"


@requires_cyna
def test_auto_backend_uses_cyna_cpu():
    segments = _circle_polyline(1.4, n=41)[None, ...]
    result = build_coil_field_grid(segments, [1.0], [1.7], [0.0], [0.0])
    assert result["backend"] == "cyna-cpu"


# ---------------------------------------------------------------------------
# phi periodicity / nfp contract
# ---------------------------------------------------------------------------


def test_phi_periodicity_closed_and_open_grids():
    segments = _circle_polyline(1.4, n=4001)[None, ...]
    current = [2.5e5]
    # Stay strictly inside the coil radius: a node on the filament would be
    # a physical singularity, not a periodicity statement.
    R_grid = np.linspace(1.0, 1.2, 4)
    Z_grid = np.linspace(-0.3, 0.3, 3)
    nfp = 2
    period = 2.0 * np.pi / nfp

    closed_phi = np.linspace(0.0, period, 7)
    closed = build_coil_field_grid(
        segments, current, R_grid, Z_grid, closed_phi, nfp=nfp, phi_closed=True
    )
    open_phi = np.linspace(0.0, period, 6, endpoint=False)
    opened = build_coil_field_grid(
        segments, current, R_grid, Z_grid, open_phi, nfp=nfp, phi_closed=False
    )

    # Axisymmetric coil: components are independent of phi.  BPhi is a
    # numerical zero here, so the tolerance is anchored to the field scale.
    field_scale = max(
        1.0e-12,
        float(np.max(np.abs(closed["BR"]))),
        float(np.max(np.abs(closed["BZ"]))),
    )
    for key in ("BR", "BZ", "BPhi"):
        values = closed[key]
        assert np.max(np.abs(values - values[:, :, :1])) < 1.0e-10 * field_scale
        # Closed grid endpoint is an exact copy of the first plane.
        assert np.array_equal(values[:, :, -1], values[:, :, 0])

    # The open grid must agree with the overlapping closed planes.
    for key in ("BR", "BZ", "BPhi"):
        assert np.max(
            np.abs(opened[key] - closed[key][:, :, :6])
        ) < 1.0e-12 * field_scale


def test_closed_phi_span_contract():
    segments = _circle_polyline(1.4, n=101)[None, ...]
    with pytest.raises(ValueError):
        build_coil_field_grid(
            segments,
            [1.0],
            [1.0, 2.0],
            [-0.1, 0.1],
            np.linspace(0.0, 1.5 * np.pi, 5),
            nfp=2,
            phi_closed=True,
            backend="numpy",
        )


# ---------------------------------------------------------------------------
# backend agreement
# ---------------------------------------------------------------------------


@requires_cyna
def test_cyna_and_numpy_backends_agree():
    segments = _toroidal_coil_set(5, n_points=25)
    currents = np.linspace(-2.0e5, 3.0e5, 5)
    R_grid = np.linspace(1.1, 2.3, 4)
    Z_grid = np.linspace(-0.35, 0.35, 3)
    Phi_grid = np.linspace(0.0, 2.0 * np.pi, 5)

    native = build_coil_field_grid(
        segments, currents, R_grid, Z_grid, Phi_grid, backend="cyna"
    )
    with pytest.warns(RuntimeWarning, match="NumPy reference backend"):
        reference = build_coil_field_grid(
            segments, currents, R_grid, Z_grid, Phi_grid, backend="numpy"
        )
    assert native["backend"] == "cyna-cpu"
    for key in ("BR", "BZ", "BPhi"):
        scale = max(1.0e-30, float(np.max(np.abs(reference[key]))))
        assert np.max(np.abs(native[key] - reference[key])) <= 1.0e-10 * scale


# ---------------------------------------------------------------------------
# production-size timing
# ---------------------------------------------------------------------------


@requires_cyna
def test_64x64x48_grid_timing_under_one_second():
    segments = _toroidal_coil_set(16, n_points=97)  # 16 coils x 96 segments
    currents = np.full(16, 1.0e5)
    R_grid = np.linspace(1.15, 2.45, 64)
    Z_grid = np.linspace(-0.55, 0.55, 64)
    Phi_grid = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)

    start = time.perf_counter()
    result = build_coil_field_grid(
        segments, currents, R_grid, Z_grid, Phi_grid, nfp=1, backend="cyna"
    )
    elapsed = time.perf_counter() - start

    assert result["BR"].shape == (64, 64, 48)
    assert np.all(np.isfinite(result["BR"]))
    assert np.all(np.isfinite(result["BZ"]))
    assert np.all(np.isfinite(result["BPhi"]))
    assert elapsed < 1.0, f"64x64x48 cyna grid fill took {elapsed:.3f} s"


# ---------------------------------------------------------------------------
# pivot2t adapter compatibility
# ---------------------------------------------------------------------------


def _coil_segments_pivot2t(coils, nseg: int) -> np.ndarray:
    """Exact layout of pivot2t.jaxfield.coil_segments: (K, S, 2, 3)."""
    out = np.zeros((len(coils), nseg, 2, 3))
    for k, pts in enumerate(coils):
        v = np.asarray(pts, dtype=np.float64)
        out[k, :, 0] = v
        out[k, :, 1] = np.roll(v, -1, axis=0)
    return out


def _field_batch_pivot2t(points, segments, currents) -> np.ndarray:
    """Independent pivot2t.jaxfield.field_batch formula for comparison."""
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    p1 = segments[:, :, 0, :]
    p2 = segments[:, :, 1, :]
    length = p2 - p1
    a = pts[None, :, None, :] - p1[:, None, :, :]
    b = a - length[:, None, :, :]
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    cross = np.cross(length[:, None, :, :], a)
    den = np.sum(cross * cross, axis=-1)
    good = den > 1.0e-30
    den = np.where(good, den, 1.0)
    term = (
        np.sum(a * length[:, None, :, :], axis=-1) / np.maximum(na, 1.0e-300)
        - np.sum(b * length[:, None, :, :], axis=-1) / np.maximum(nb, 1.0e-300)
    )
    scale = (
        MU0
        * np.asarray(currents, dtype=np.float64)[:, None, None]
        / (4.0 * np.pi)
        * np.where(good, term / den, 0.0)
    )
    return np.sum(cross * scale[..., None], axis=(0, 2))


@requires_cyna
def test_pivot2t_adapter_call_pattern_and_trace():
    coils = [
        _circle_polyline(1.2, n=64, z0=0.15),
        _circle_polyline(1.9, n=64, z0=-0.25),
        _circle_polyline(2.4, n=64, z0=0.35),
    ]
    currents = np.array([1.0e5, -2.0e5, 0.5e5])
    segments = _coil_segments_pivot2t(coils, nseg=64)  # (3, 64, 2, 3)

    R_grid = np.linspace(1.15, 2.45, 8)
    Z_grid = np.linspace(-0.55, 0.55, 6)
    Phi_grid = np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False)  # pivot2t layout

    grid = build_coil_field_grid(
        segments, currents, R_grid, Z_grid, Phi_grid, nfp=1
    )
    assert grid["backend"] == "cyna-cpu"
    assert grid["BR_flat"].shape == (R_grid.size * Z_grid.size * Phi_grid.size,)
    for key in ("R_grid", "Z_grid", "Phi_grid", "nfp", "BR_flat", "BZ_flat", "BPhi_flat"):
        assert key in grid

    # Compare cyna against the literal pivot2t field_batch formula at every
    # grid node, in Cartesian components converted to cylindrical ones.
    RR, ZZ, PP = np.meshgrid(R_grid, Z_grid, Phi_grid, indexing="ij")
    pts = np.stack(
        [
            RR.ravel() * np.cos(PP.ravel()),
            RR.ravel() * np.sin(PP.ravel()),
            ZZ.ravel(),
        ],
        axis=-1,
    )
    b_cart = _field_batch_pivot2t(pts, segments, currents)
    cp = np.cos(PP.ravel())
    sp = np.sin(PP.ravel())
    br_ref = b_cart[:, 0] * cp + b_cart[:, 1] * sp
    bp_ref = -b_cart[:, 0] * sp + b_cart[:, 1] * cp
    bz_ref = b_cart[:, 2]

    scale = max(1.0e-30, float(np.max(np.abs(br_ref))), float(np.max(np.abs(bz_ref))))
    assert np.max(np.abs(grid["BR_flat"] - br_ref)) <= 1.0e-10 * scale
    assert np.max(np.abs(grid["BZ_flat"] - bz_ref)) <= 1.0e-10 * scale
    assert np.max(np.abs(grid["BPhi_flat"] - bp_ref)) <= 1.0e-10 * scale

    # Downstream Poincare trace consumes the produced grid unchanged.
    from pyna.toroidal.flt import trace_poincare_batch

    wall_r = 1.75 + 0.7 * np.cos(np.linspace(0.0, 2.0 * np.pi, 128))
    wall_z = 0.9 * np.sin(np.linspace(0.0, 2.0 * np.pi, 128))
    counts, pR, pZ = trace_poincare_batch(
        np.array([1.75]),
        np.array([0.0]),
        0.0,
        1,
        2.0 * np.pi / 60.0,
        grid["R_grid"],
        grid["Z_grid"],
        grid["Phi_grid"],
        grid["BR_flat"],
        grid["BZ_flat"],
        grid["BPhi_flat"],
        wall_r,
        wall_z,
        nfp=grid["nfp"],
    )
    assert counts.shape[0] == 1
    assert pR.shape[0] == 1 and pZ.shape[0] == 1
    assert np.all(np.isfinite(pR)) and np.all(np.isfinite(pZ))
