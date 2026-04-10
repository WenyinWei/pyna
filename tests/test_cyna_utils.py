"""Tests for pyna._cyna.utils — cyna utility helpers."""
import numpy as np
import pytest

from pyna._cyna.utils import ensure_c_double, prepare_field_cache, build_fixed_points_from_batch


class TestEnsureCDouble:
    def test_already_c_contiguous_float64(self):
        arr = np.zeros((3, 4), dtype=np.float64, order='C')
        result = ensure_c_double(arr)
        assert result is arr  # zero-copy

    def test_fortran_order_copies(self):
        arr = np.zeros((3, 4), dtype=np.float64, order='F')
        result = ensure_c_double(arr)
        assert result is not arr
        assert result.flags['C_CONTIGUOUS']

    def test_float32_converts(self):
        arr = np.zeros((3, 4), dtype=np.float32)
        result = ensure_c_double(arr)
        assert result.dtype == np.float64

    def test_int_converts(self):
        arr = np.array([1, 2, 3])
        result = ensure_c_double(arr)
        assert result.dtype == np.float64


class TestPrepareFieldCache:
    @pytest.fixture
    def dummy_fc(self):
        NR, NZ, NPhi = 5, 6, 8
        return {
            'R_grid':   np.linspace(0.5, 2.0, NR),
            'Z_grid':   np.linspace(-1.0, 1.0, NZ),
            'Phi_grid': np.linspace(0, 2 * np.pi, NPhi, endpoint=False),
            'BR':       np.random.randn(NR, NZ, NPhi),
            'BPhi':     np.ones((NR, NZ, NPhi)),
            'BZ':       np.random.randn(NR, NZ, NPhi),
        }

    def test_output_keys(self, dummy_fc):
        result = prepare_field_cache(dummy_fc)
        for key in ['BR', 'BPhi', 'BZ', 'R_grid', 'Z_grid', 'Phi_grid']:
            assert key in result

    def test_phi_extended(self, dummy_fc):
        NPhi = len(dummy_fc['Phi_grid'])
        result = prepare_field_cache(dummy_fc, extend_phi=True)
        assert result['Phi_grid'][-1] == pytest.approx(2 * np.pi)
        assert result['BR'].shape[2] == NPhi + 1

    def test_no_extend(self, dummy_fc):
        NPhi = len(dummy_fc['Phi_grid'])
        result = prepare_field_cache(dummy_fc, extend_phi=False)
        assert result['BR'].shape[2] == NPhi

    def test_all_c_contiguous(self, dummy_fc):
        result = prepare_field_cache(dummy_fc)
        for key, arr in result.items():
            assert arr.flags['C_CONTIGUOUS'], f"{key} not C-contiguous"
            assert arr.dtype == np.float64, f"{key} not float64"

    def test_float32_input(self, dummy_fc):
        dummy_fc['BR'] = dummy_fc['BR'].astype(np.float32)
        result = prepare_field_cache(dummy_fc)
        assert result['BR'].dtype == np.float64


class TestBuildFixedPointsFromBatch:
    def test_basic(self):
        N = 5
        R_out = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
        Z_out = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        converged = np.array([1, 1, 0, 1, 0])  # 3 converged
        DPm_flat = np.tile(np.eye(2).ravel(), (N, 1))  # (N, 4)
        # Make index 1 X-point (trace > 2)
        DPm_flat[1] = np.array([3.0, 0.0, 0.0, 1.0 / 3.0])
        ptype = np.array([0, 1, -1, 0, -1])

        fps = build_fixed_points_from_batch(
            R_out, Z_out, converged, DPm_flat, ptype, phi=0.5,
        )
        assert len(fps) == 3
        assert fps[0].kind == 'O'
        assert fps[1].kind == 'X'
        assert fps[2].kind == 'O'
        assert fps[0].phi == pytest.approx(0.5)

    def test_empty(self):
        fps = build_fixed_points_from_batch(
            np.array([]), np.array([]),
            np.array([], dtype=int),
            np.zeros((0, 4)),
            np.array([], dtype=int),
            phi=0.0,
        )
        assert fps == []

    def test_3d_DPm(self):
        N = 2
        R_out = np.array([1.0, 1.1])
        Z_out = np.array([0.0, 0.1])
        converged = np.array([1, 1])
        DPm_3d = np.stack([np.eye(2)] * N)  # (N, 2, 2)
        ptype = np.array([0, 0])

        fps = build_fixed_points_from_batch(
            R_out, Z_out, converged, DPm_3d, ptype, phi=0.0,
        )
        assert len(fps) == 2
        assert fps[0].DPm.shape == (2, 2)
