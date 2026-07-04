from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pytest

from pyna.plot import (
    beta_physics_rows,
    beta_ramp_scan_rows,
    plot_beta_physics_dashboard,
    plot_beta_ramp_scan_summary,
    read_beta_physics_csv,
)


def _rows():
    return [
        {
            "scan_index": 1,
            "beta": 0.01,
            "status": "ok",
            "max_chirikov": 0.08,
            "max_island_half_width": 1.0e-4,
            "max_b_res": 2.0e-5,
            "min_abs_miota_plus_n": 0.12,
            "dominant_modes": [(4, 1)],
        },
        {
            "scan_index": 2,
            "beta": 0.02,
            "status": "watch",
            "max_chirikov": 0.74,
            "max_island_half_width": 3.0e-4,
            "max_b_res": 6.0e-5,
            "min_abs_miota_plus_n": 2.0e-2,
            "dominant_modes": [(5, 2), (4, 1)],
        },
        {
            "scan_index": 3,
            "beta": 0.03,
            "status": "low-confidence",
            "max_chirikov": 1.15,
            "max_island_half_width": 8.0e-4,
            "max_b_res": 1.8e-4,
            "min_abs_miota_plus_n": 1.0e-4,
            "dominant_modes": [(5, 2)],
        },
    ]


class _ScanLike:
    def summary_rows(self):
        return _rows()


class _ResultLike:
    def __init__(self, row):
        self._row = row

    def summary(self):
        return self._row


def test_beta_ramp_scan_rows_accepts_scan_results_and_mappings():
    rows = beta_ramp_scan_rows(_ScanLike())
    assert rows[1]["status"] == "watch"

    result_rows = beta_ramp_scan_rows([_ResultLike(row) for row in _rows()])
    assert result_rows[2]["max_chirikov"] == pytest.approx(1.15)

    mapping_rows = beta_ramp_scan_rows(_rows())
    assert mapping_rows[0]["dominant_modes"] == [(4, 1)]


def test_plot_beta_ramp_scan_summary_smoke(tmp_path):
    out = tmp_path / "scan_summary.png"

    fig = plot_beta_ramp_scan_summary(_rows(), out_path=out, title="synthetic beta-ramp")
    try:
        assert out.exists()
        assert len(fig.axes) == 4
        assert fig.axes[0].get_ylabel() == "trust"
        assert fig.axes[1].get_ylabel() == "max Chirikov"
        assert fig.axes[3].get_ylabel() == "min |m*iota+n|"
        assert fig._suptitle.get_text() == "synthetic beta-ramp"
    finally:
        plt.close(fig)


def test_plot_beta_ramp_scan_summary_requires_rows():
    with pytest.raises(ValueError, match="at least one row"):
        plot_beta_ramp_scan_summary([])


def test_plot_beta_ramp_scan_summary_keeps_full_xlim_for_sparse_metrics():
    rows = _rows()
    rows[0]["min_abs_miota_plus_n"] = 0.0
    rows[1]["min_abs_miota_plus_n"] = float("nan")

    fig = plot_beta_ramp_scan_summary(rows)
    try:
        xmin, xmax = fig.axes[3].get_xlim()
        assert xmin < 0.01
        assert xmax > 0.03
    finally:
        plt.close(fig)


def _physics_rows():
    return [
        {
            "beta": 1.0e-4,
            "accepted": 1,
            "protected_fRMS": 0.6,
            "raw_fRMS": 0.8,
            "jfree_force_matrix_fRMS": 0.7,
            "plasma_volume_beta_after": 1.02e-4,
            "actual_beta_step": 1.0e-4,
            "beta_tracking_ratio": 1.02,
            "jfree_matrix_relres": 2.0e-2,
            "cupy_lsqr_relres": 1.0e-3,
            "delta_B_over_B0_max": 2.0e-2,
            "delta_B_over_B0_rms": 3.0e-4,
            "support_delta_B_over_B0_max": 1.0e-2,
            "delta_J_A_per_m2_rms": 2.0e2,
            "support_delta_J_A_per_m2_max": 1.0e3,
            "bootstrap_J_parallel_rms": 4.0e1,
            "diamagnetic_J_rms": 1.5e2,
            "pfirsch_schluter_J_parallel_rms": 5.0e1,
            "current_curl_residual_over_reference": 0.5,
            "current_curl_plasma_residual_over_reference": 0.2,
            "total_divB_plasma_rms_over_B_per_grid_length": 1.0e-4,
            "pressure_target_rel_l2": 1.0e-3,
            "pressure_support_outside_abs_fraction": 0.0,
            "pressure_centroid_axis_displacement_m": 2.0e-3,
            "topology_fitted_fraction": 0.9,
            "topology_mean_self_intersections": 0.0,
        },
        {
            "beta": 2.0e-4,
            "accepted": 0,
            "protected_fRMS": 1.3,
            "raw_fRMS": 1.8,
            "jfree_force_matrix_fRMS": 1.4,
            "plasma_volume_beta_after": 1.95e-4,
            "actual_beta_step": 0.9e-4,
            "beta_tracking_ratio": 0.95,
            "jfree_matrix_relres": 6.0e-2,
            "cupy_lsqr_relres": 4.0e-3,
            "delta_B_over_B0_max": 1.2e-1,
            "delta_B_over_B0_rms": 9.0e-4,
            "support_delta_B_over_B0_max": 8.0e-2,
            "delta_J_A_per_m2_rms": 5.0e2,
            "support_delta_J_A_per_m2_max": 2.0e3,
            "bootstrap_J_parallel_rms": 7.0e1,
            "diamagnetic_J_rms": 2.2e2,
            "pfirsch_schluter_J_parallel_rms": 9.0e1,
            "current_curl_residual_over_reference": 0.8,
            "current_curl_plasma_residual_over_reference": 0.4,
            "total_divB_plasma_rms_over_B_per_grid_length": 3.0e-4,
            "pressure_target_rel_l2": 3.0e-3,
            "pressure_support_outside_abs_fraction": 0.02,
            "pressure_centroid_axis_displacement_m": 3.0e-3,
            "topology_fitted_fraction": 0.55,
            "topology_mean_self_intersections": 0.15,
            "amplitude_gate_reasons": "delta_B_over_B0_max",
        },
    ]


def test_beta_physics_rows_reads_legacy_workflow_csv(tmp_path):
    path = tmp_path / "beta_physics_steps.csv"
    rows = _physics_rows()
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    read_rows = read_beta_physics_csv(path)

    assert read_rows[0]["beta"] == "0.0001"
    assert beta_physics_rows(path)[1]["accepted"] == "0"


def test_plot_beta_physics_dashboard_smoke(tmp_path):
    out = tmp_path / "physics_dashboard.png"

    fig = plot_beta_physics_dashboard(_physics_rows(), out_path=out, title="synthetic PDE dashboard")
    try:
        assert out.exists()
        assert len(fig.axes) == 9
        assert fig.axes[0].get_title() == "step acceptance"
        assert fig.axes[1].get_title() == "PDE force residual"
        assert fig.axes[8].get_title() == "topology retrace"
    finally:
        plt.close(fig)


def test_plot_beta_physics_dashboard_single_beta_uses_local_xlim():
    row = dict(_physics_rows()[0])
    row["beta"] = 2.0e-4

    fig = plot_beta_physics_dashboard([row])
    try:
        xmin, xmax = fig.axes[0].get_xlim()
        assert 1.8e-4 < xmin < 2.0e-4
        assert 2.0e-4 < xmax < 2.2e-4
    finally:
        plt.close(fig)


def test_plot_beta_physics_dashboard_marks_high_residual_low_confidence():
    row = dict(_physics_rows()[0])
    row["accepted"] = 1
    row["protected_fRMS"] = 80.0

    fig = plot_beta_physics_dashboard([row])
    try:
        facecolors = fig.axes[0].collections[0].get_facecolors()
        assert facecolors.shape[0] == 1
        assert facecolors[0][0] > facecolors[0][1]
    finally:
        plt.close(fig)
