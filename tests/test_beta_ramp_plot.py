from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pytest

from pyna.plot import beta_ramp_scan_rows, plot_beta_ramp_scan_summary


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
