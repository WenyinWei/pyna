"""Tests for pyna.connection_length — connection-length in cylindrical coords.

All tests use simple analytic geometries where the expected result can be
computed by hand or with a simple formula.

Wall geometry
-------------
Circular wall of radius *a* centred at (R0, 0):
    R_wall = R0 + a cos(θ),  Z_wall = a sin(θ)

Field functions
---------------
1. Vertical drift:   dR/dφ = 0,  dZ/dφ = v_Z
   Starting from (R0, z0) the field line drifts straight up or down.
   Forward arc length to Z = ±a:
       L+ = sqrt(v_Z² + R0²) * |a - z0| / |v_Z|
       L- = sqrt(v_Z² + R0²) * |a + z0| / |v_Z|

2. Pure toroidal (no drift):  dR/dφ = 0,  dZ/dφ = 0
   Field line never reaches the wall → L = inf.
"""
import numpy as np
import pytest
from pyna.connection_length import (
    _point_inside_polygon,
    _segment_wall_crossing,
    connection_length,
    connection_length_map,
)

# ---------------------------------------------------------------------------
# Shared geometry: circular wall
# ---------------------------------------------------------------------------

_R0 = 2.0  # major radius (m)
_a  = 0.5  # wall minor radius (m)
_N_WALL = 128

_theta_wall = np.linspace(0, 2 * np.pi, _N_WALL, endpoint=False)
_R_WALL = _R0 + _a * np.cos(_theta_wall)
_Z_WALL = _a  * np.sin(_theta_wall)
_WALL = (_R_WALL, _Z_WALL)


def _vert_field(vZ: float):
    """Factory: purely vertical-drift 2D field (dR/dφ=0, dZ/dφ=vZ)."""
    def ff(R, Z, phi):
        return np.array([0.0, vZ])
    return ff


def _analytic_Lplus(z0: float, vZ: float) -> float:
    """Arc length from (R0, z0) to top Z = +a (forward, vZ > 0).

    ds = sqrt((dR/dφ)² + R0² + (dZ/dφ)²) dφ
       = sqrt(0 + R0² + vZ²) dφ
    Δφ to reach Z = a:  Δφ = (a - z0) / vZ
    L  = sqrt(R0² + vZ²) * Δφ
    """
    if vZ == 0:
        return float("inf")
    delta_phi = (_a - z0) / vZ
    return float(np.sqrt(_R0**2 + vZ**2) * delta_phi)


def _analytic_Lminus(z0: float, vZ: float) -> float:
    """Arc length from (R0, z0) to bottom Z = -a (backward, vZ > 0)."""
    if vZ == 0:
        return float("inf")
    delta_phi = (z0 + _a) / vZ
    return float(np.sqrt(_R0**2 + vZ**2) * delta_phi)


# ---------------------------------------------------------------------------
# Tests: geometry helpers
# ---------------------------------------------------------------------------

def test_point_inside_polygon_inside():
    assert _point_inside_polygon(_R0, 0.0, _R_WALL, _Z_WALL) is True


def test_point_inside_polygon_outside():
    assert _point_inside_polygon(_R0, _a + 0.1, _R_WALL, _Z_WALL) is False


def test_point_inside_polygon_on_axis():
    # Centre should be inside
    assert _point_inside_polygon(_R0, 0.0, _R_WALL, _Z_WALL) is True


def test_segment_wall_crossing_exits():
    # Step from (R0, 0) to (R0, 2) should cross the wall at Z = a
    cross = _segment_wall_crossing(_R0, 0.0, _R0, 2.0, _R_WALL, _Z_WALL)
    assert cross is not None, "Should find a crossing when step exits the wall"
    t, R_cross, Z_cross = cross
    assert 0.0 < t <= 1.0
    # Crossing should be near Z = a
    assert abs(Z_cross - _a) < 0.02, f"Expected Z_cross ≈ {_a}, got {Z_cross:.4f}"


def test_segment_wall_crossing_no_exit():
    # Short step fully inside the wall
    cross = _segment_wall_crossing(_R0, 0.0, _R0, 0.1, _R_WALL, _Z_WALL)
    assert cross is None, "Short step inside wall should not find a crossing"


# ---------------------------------------------------------------------------
# Tests: connection_length — analytic comparison
# ---------------------------------------------------------------------------

_vZ = 0.15  # drift speed (m/rad)
_z0 = 0.0   # starting Z


def test_connection_length_forward_analytic():
    """Forward L+ should match analytic formula to within 2%."""
    ff = _vert_field(_vZ)
    result = connection_length(ff, [[_R0, _z0]], _WALL,
                               direction="+", max_turns=100, dphi=0.01)
    L_analytic = _analytic_Lplus(_z0, _vZ)
    L_numeric  = float(result["L_plus"][0])
    rel_err = abs(L_numeric - L_analytic) / L_analytic
    assert rel_err < 0.02, (
        f"L+ = {L_numeric:.4f} m, analytic = {L_analytic:.4f} m, "
        f"rel error = {rel_err:.4%}"
    )


def test_connection_length_backward_analytic():
    """Backward L- should match analytic formula to within 2%."""
    ff = _vert_field(_vZ)
    result = connection_length(ff, [[_R0, _z0]], _WALL,
                               direction="-", max_turns=100, dphi=0.01)
    L_analytic = _analytic_Lminus(_z0, _vZ)
    L_numeric  = float(result["L_minus"][0])
    rel_err = abs(L_numeric - L_analytic) / L_analytic
    assert rel_err < 0.02, (
        f"L- = {L_numeric:.4f} m, analytic = {L_analytic:.4f} m, "
        f"rel error = {rel_err:.4%}"
    )


def test_connection_length_sum_symmetric():
    """From the midplane, L+ = L- so L_sum = 2*L+."""
    ff = _vert_field(_vZ)
    result = connection_length(ff, [[_R0, 0.0]], _WALL,
                               direction="both", max_turns=100, dphi=0.01)
    L_plus  = float(result["L_plus"][0])
    L_minus = float(result["L_minus"][0])
    rel_diff = abs(L_plus - L_minus) / max(L_plus, L_minus)
    assert rel_diff < 0.02, (
        f"L+ = {L_plus:.4f} m, L- = {L_minus:.4f} m, rel diff = {rel_diff:.4%}"
    )
    assert abs(result["L_sum"][0] - L_plus - L_minus) < 1e-10


def test_connection_length_asymmetric():
    """Off-midplane start gives L+ ≠ L-."""
    z0 = 0.2  # above midplane
    ff = _vert_field(_vZ)
    result = connection_length(ff, [[_R0, z0]], _WALL,
                               direction="both", max_turns=100, dphi=0.01)
    L_plus  = float(result["L_plus"][0])
    L_minus = float(result["L_minus"][0])
    assert L_plus < L_minus, (
        f"Expected L+ < L- for z0 = {z0} m (above midplane); "
        f"L+ = {L_plus:.4f}, L- = {L_minus:.4f}"
    )


def test_connection_length_max_min():
    """L_max >= L_min, and L_max + L_min = ... not fixed, but ordering holds."""
    z0 = 0.15
    ff = _vert_field(_vZ)
    result = connection_length(ff, [[_R0, z0]], _WALL,
                               direction="both", max_turns=100, dphi=0.01)
    assert np.all(result["L_max"] >= result["L_min"])


def test_connection_length_outside_wall_zero():
    """Starting outside the wall should give zero connection length."""
    ff = _vert_field(_vZ)
    R_out = _R0 + _a + 0.1  # outside the circular wall
    result = connection_length(ff, [[R_out, 0.0]], _WALL,
                               direction="+", max_turns=50, dphi=0.05)
    assert float(result["L_plus"][0]) == 0.0


def test_connection_length_no_wall_hit():
    """Pure toroidal field never reaches the wall → L = inf."""
    def pure_toroidal(R, Z, phi):
        return np.array([0.0, 0.0])   # no drift

    result = connection_length(pure_toroidal, [[_R0, 0.0]], _WALL,
                               direction="+", max_turns=10, dphi=0.1)
    assert float(result["L_plus"][0]) == float("inf"), (
        "Expected inf when field line never reaches the wall"
    )


def test_connection_length_batch():
    """Batch computation on N > 1 starting points returns correct shapes."""
    ff = _vert_field(_vZ)
    starts = np.array([[_R0, -0.2], [_R0, 0.0], [_R0, 0.2]])
    result = connection_length(ff, starts, _WALL,
                               direction="both", max_turns=100, dphi=0.02)
    assert result["L_plus"].shape  == (3,)
    assert result["L_minus"].shape == (3,)
    assert result["L_sum"].shape   == (3,)
    assert result["hit_plus"].shape == (3, 3)
    # All forward hit-points should be on or near the wall (within one step of _a)
    dphi_test = 0.02
    step_bound = np.sqrt(_vZ**2 + _R0**2) * dphi_test * 1.5
    for i in range(3):
        Rh, Zh = result["hit_plus"][i, 0], result["hit_plus"][i, 1]
        dist_to_wall_centre = np.sqrt((Rh - _R0)**2 + Zh**2)
        assert abs(dist_to_wall_centre - _a) < step_bound, (
            f"Hit point [{Rh:.4f}, {Zh:.4f}] is {dist_to_wall_centre:.4f} m "
            f"from wall centre, expected ~{_a:.4f} m"
        )


def test_connection_length_wallgeometry_ducktype():
    """WallGeometry-like object (duck-typed via .R_wall / .Z_wall) is accepted."""
    class _W:
        R_wall = _R_WALL
        Z_wall = _Z_WALL

    ff = _vert_field(_vZ)
    result = connection_length(ff, [[_R0, 0.0]], _W(),
                               direction="+", max_turns=50, dphi=0.05)
    assert np.isfinite(result["L_plus"][0])


def test_connection_length_invalid_direction():
    """An invalid direction string should raise ValueError."""
    ff = _vert_field(_vZ)
    with pytest.raises(ValueError, match="direction must be"):
        connection_length(ff, [[_R0, 0.0]], _WALL, direction="forward")


# ---------------------------------------------------------------------------
# Tests: connection_length_map
# ---------------------------------------------------------------------------

def test_connection_length_map_shape():
    """connection_length_map returns array with same shape as input grid."""
    R_g = np.array([[_R0 - 0.1, _R0, _R0 + 0.1]])
    Z_g = np.array([[0.0, 0.0, 0.0]])
    ff = _vert_field(_vZ)
    cmap = connection_length_map(ff, R_g, Z_g, _WALL,
                                 direction="both", aggregate="sum",
                                 max_turns=50, dphi=0.05)
    assert cmap.shape == (1, 3)
    assert np.all(np.isfinite(cmap))


def test_connection_length_map_aggregate_plus():
    """aggregate='+' returns only L+ values."""
    R_g = np.array([[_R0, _R0]])
    Z_g = np.array([[-0.1, 0.1]])
    ff = _vert_field(_vZ)
    cmap_plus  = connection_length_map(ff, R_g, Z_g, _WALL,
                                       aggregate="+", max_turns=50, dphi=0.05)
    cmap_minus = connection_length_map(ff, R_g, Z_g, _WALL,
                                       aggregate="-", max_turns=50, dphi=0.05)
    # Starting at z0=+0.1 (above midplane): L+ < L-
    assert float(cmap_plus[0, 1]) < float(cmap_minus[0, 1]), (
        "Above midplane: L+ should be shorter than L-"
    )
