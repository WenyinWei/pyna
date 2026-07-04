"""Summary plots for beta-ramp diagnostics.

The plotting helpers consume the public rows returned by
``BetaRampScanDiagnostics.summary_rows()`` or ``beta_scan_summary_rows``.  They
do not know where the equilibrium data came from, which keeps private
stellarator adapters and public examples on the same plotting path.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


STATUS_COLORS = {
    "ok": "#28784a",
    "watch": "#c98200",
    "low-confidence": "#b43a48",
    "unknown": "0.45",
}

STATUS_LEVELS = {
    "ok": 0,
    "watch": 1,
    "low-confidence": 2,
    "unknown": -1,
}

_PHYSICS_COLORS = (
    "#376795",
    "#b43a48",
    "#28784a",
    "#8c5fbf",
    "#c98200",
    "#3b7a78",
)


def beta_ramp_scan_rows(scan_or_rows: Any) -> list[dict[str, Any]]:
    """Normalize beta-ramp diagnostics into plotting rows.

    Accepted inputs are:

    - a ``BetaRampScanDiagnostics``-like object with ``summary_rows()``;
    - a sequence of ``BetaRampSpectrumDiagnostics``-like objects with
      ``summary()``;
    - a sequence of mapping rows.
    """

    if hasattr(scan_or_rows, "summary_rows"):
        return [dict(row) for row in scan_or_rows.summary_rows()]

    rows: list[dict[str, Any]] = []
    for item in scan_or_rows:
        if isinstance(item, Mapping):
            rows.append(dict(item))
        elif hasattr(item, "summary"):
            rows.append(dict(item.summary()))
        else:
            raise TypeError("scan_or_rows must contain mappings or objects with summary()")
    return rows


def read_beta_physics_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read a W7-X/NCSX style ``*_beta_physics_steps.csv`` file."""

    with Path(path).expanduser().open("r", newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def beta_physics_rows(rows_or_csv: Any) -> list[dict[str, Any]]:
    """Normalize old beta-physics workflow rows for dashboard plotting."""

    if isinstance(rows_or_csv, (str, Path)):
        return read_beta_physics_csv(rows_or_csv)
    return beta_ramp_scan_rows(rows_or_csv)


def _float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _metric(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray([_float_or_nan(row.get(key)) for row in rows], dtype=float)


def _first_metric(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> tuple[np.ndarray, str] | None:
    for key in keys:
        values = _metric(rows, key)
        if np.any(np.isfinite(values)):
            return values, key
    return None


def _x_axis(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, str]:
    beta = _metric(rows, "beta")
    if beta.size and np.all(np.isfinite(beta)):
        return beta, "beta"

    scan_index = _metric(rows, "scan_index")
    if scan_index.size and np.all(np.isfinite(scan_index)):
        return scan_index, "scan index"

    return np.arange(len(rows), dtype=float), "row index"


def _status_values(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, list[str], list[str]]:
    labels: list[str] = []
    colors: list[str] = []
    values: list[int] = []
    for row in rows:
        status = str(row.get("status", "unknown") or "unknown")
        if status not in STATUS_LEVELS:
            status = "unknown"
        labels.append(status)
        values.append(STATUS_LEVELS[status])
        colors.append(STATUS_COLORS[status])
    return np.asarray(values, dtype=float), labels, colors


def _positive_for_log(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    arr[~np.isfinite(arr) | (arr <= 0.0)] = np.nan
    return arr


def _padded_xlim(x: np.ndarray) -> tuple[float, float] | None:
    finite = np.asarray(x, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    xmin = float(np.min(finite))
    xmax = float(np.max(finite))
    if xmin == xmax:
        pad = max(abs(xmin) * 0.05, 1.0e-12)
    else:
        pad = 0.05 * (xmax - xmin)
    return xmin - pad, xmax + pad


def _set_empty_axis(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="0.35")
    ax.set_axis_off()


def _format_dominant_mode_label(row: Mapping[str, Any], max_modes: int = 3) -> str:
    modes = row.get("dominant_modes") or ()
    formatted: list[str] = []
    for mode in list(modes)[:max_modes]:
        try:
            m, n = mode
        except (TypeError, ValueError):
            continue
        formatted.append(f"({int(m)},{int(n)})")
    return " ".join(formatted)


def _has_gate_reason(row: Mapping[str, Any]) -> bool:
    for key in ("amplitude_gate_reasons", "topology_gate_reasons", "angular_artifact_gate_reasons"):
        value = str(row.get(key, "") or "").strip()
        if value and value.lower() not in {"nan", "none"}:
            return True
    return False


def _physics_gate_color(row: Mapping[str, Any], accepted: float) -> str:
    if np.isfinite(accepted) and accepted <= 0.0:
        return STATUS_COLORS["low-confidence"]
    residuals = [
        _float_or_nan(row.get("protected_fRMS")),
        _float_or_nan(row.get("raw_fRMS")),
        _float_or_nan(row.get("jfree_force_matrix_fRMS")),
        _float_or_nan(row.get("drive_normalized_fRMS")),
    ]
    max_residual = max((value for value in residuals if np.isfinite(value)), default=float("nan"))
    if np.isfinite(max_residual) and max_residual >= 50.0:
        return STATUS_COLORS["low-confidence"]
    if _has_gate_reason(row) or (np.isfinite(max_residual) and max_residual >= 10.0):
        return STATUS_COLORS["watch"]
    return STATUS_COLORS["ok"]


def _plot_metric_group(
    ax,
    rows: Sequence[Mapping[str, Any]],
    x: np.ndarray,
    specs: Sequence[tuple[Sequence[str], str]],
    *,
    title: str,
    ylabel: str,
    yscale: str = "linear",
    xlim: tuple[float, float] | None = None,
) -> None:
    plotted = 0
    for idx, (keys, label) in enumerate(specs):
        found = _first_metric(rows, keys)
        if found is None:
            continue
        values, _key = found
        if yscale == "log":
            values = _positive_for_log(values)
        if not np.any(np.isfinite(values)):
            continue
        color = _PHYSICS_COLORS[idx % len(_PHYSICS_COLORS)]
        ax.plot(x, values, marker="o", ms=3.6, lw=1.15, color=color, label=label)
        plotted += 1
    if plotted == 0:
        _set_empty_axis(ax, f"no {title} metrics")
        return
    ax.set_title(title)
    ax.set_xlabel("beta")
    ax.set_ylabel(ylabel)
    if yscale == "log":
        ax.set_yscale("log")
    ax.grid(True, which="both", lw=0.3, color="0.86")
    ax.legend(loc="best", fontsize=7.5, frameon=False)
    if xlim is not None:
        ax.set_xlim(*xlim)


def plot_beta_ramp_scan_summary(
    scan_or_rows: Any,
    *,
    out_path: str | Path | None = None,
    title: str = "beta-ramp scan summary",
    figsize: tuple[float, float] = (10.5, 7.4),
    dpi: int = 170,
    chirikov_watch: float = 0.7,
    chirikov_low: float = 1.0,
    small_divisor_watch: float = 3.0e-2,
    annotate_modes: bool = True,
):
    """Plot trust, overlap, island, and small-divisor beta-ramp metrics.

    The 2x2 dashboard is intended as the first figure for a continuation run:
    it shows where confidence degrades before detailed Poincare/fixed-point
    plots are interpreted.
    """

    rows = beta_ramp_scan_rows(scan_or_rows)
    if not rows:
        raise ValueError("scan_or_rows must contain at least one row")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    x, x_label = _x_axis(rows)
    xlim = _padded_xlim(x)
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    ax_status, ax_chirikov, ax_island, ax_small = axes.ravel()

    status_y, _status_labels, status_colors = _status_values(rows)
    ax_status.scatter(x, status_y, s=54, c=status_colors, edgecolors="black", linewidths=0.45, zorder=3)
    ax_status.plot(x, status_y, color="0.72", lw=0.8, zorder=1)
    ax_status.set_yticks([0, 1, 2], ["ok", "watch", "low-confidence"])
    ax_status.set_ylim(-0.6, 2.6)
    ax_status.set_xlabel(x_label)
    ax_status.set_ylabel("trust")
    ax_status.grid(True, axis="x", lw=0.3, color="0.85")
    ax_status.set_title("confidence gates")
    if xlim is not None:
        ax_status.set_xlim(*xlim)
    if annotate_modes and len(rows) <= 12:
        for xi, yi, row in zip(x, status_y, rows):
            label = _format_dominant_mode_label(row)
            if label:
                ax_status.text(xi, yi + 0.12, label, fontsize=7, ha="center", va="bottom", color="0.20")

    max_chirikov = _metric(rows, "max_chirikov")
    if np.any(np.isfinite(max_chirikov)):
        ax_chirikov.plot(x, max_chirikov, marker="o", ms=4.5, lw=1.35, color="#376795")
        ax_chirikov.axhline(float(chirikov_watch), color="#c98200", lw=0.95, ls="--", label="watch")
        ax_chirikov.axhline(float(chirikov_low), color="#b43a48", lw=0.95, ls=":", label="low")
        ax_chirikov.legend(loc="best", fontsize=8, frameon=False)
        ax_chirikov.set_xlabel(x_label)
        ax_chirikov.set_ylabel("max Chirikov")
        ax_chirikov.set_title("island overlap")
        ax_chirikov.grid(True, lw=0.3, color="0.85")
        if xlim is not None:
            ax_chirikov.set_xlim(*xlim)
    else:
        _set_empty_axis(ax_chirikov, "no Chirikov metric")

    half_width = _positive_for_log(_metric(rows, "max_island_half_width"))
    max_b_res = _positive_for_log(_metric(rows, "max_b_res"))
    if np.any(np.isfinite(half_width)) or np.any(np.isfinite(max_b_res)):
        if np.any(np.isfinite(half_width)):
            ax_island.semilogy(x, half_width, marker="o", ms=4.5, lw=1.35, color="#4b8f3a", label="half-width")
        if np.any(np.isfinite(max_b_res)):
            ax_island.semilogy(x, max_b_res, marker="s", ms=4.2, lw=1.2, color="#8c5fbf", label="max b_res")
        ax_island.set_xlabel(x_label)
        ax_island.set_ylabel("positive metric")
        ax_island.set_title("RMP amplitude and width")
        ax_island.grid(True, which="both", lw=0.3, color="0.85")
        ax_island.legend(loc="best", fontsize=8, frameon=False)
        if xlim is not None:
            ax_island.set_xlim(*xlim)
    else:
        _set_empty_axis(ax_island, "no island metric")

    min_small = _positive_for_log(_metric(rows, "min_abs_miota_plus_n"))
    if np.any(np.isfinite(min_small)):
        ax_small.semilogy(x, min_small, marker="o", ms=4.5, lw=1.35, color="#3b7a78")
        ax_small.axhline(float(small_divisor_watch), color="#c98200", lw=0.95, ls="--", label="watch")
        ax_small.set_xlabel(x_label)
        ax_small.set_ylabel("min |m*iota+n|")
        ax_small.set_title("small divisor distance")
        ax_small.grid(True, which="both", lw=0.3, color="0.85")
        ax_small.legend(loc="best", fontsize=8, frameon=False)
        if xlim is not None:
            ax_small.set_xlim(*xlim)
    else:
        _set_empty_axis(ax_small, "no small-divisor metric")

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    if out_path is not None:
        out = Path(out_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=int(dpi), bbox_inches="tight")
    return fig


def plot_beta_physics_dashboard(
    rows_or_csv: Any,
    *,
    out_path: str | Path | None = None,
    title: str = "beta-ramp FPT/PDE physics dashboard",
    figsize: tuple[float, float] = (14.0, 10.2),
    dpi: int = 170,
):
    """Plot the richer W7-X/NCSX beta-physics workflow diagnostics.

    This dashboard is for rows written by the older topoquest beta workflows
    such as ``w7x_beta_physics_steps.csv`` and ``ncsx_beta_physics_steps.csv``.
    It deliberately plots the PDE/FPT health metrics that decide whether a
    beta step is interpretable before downstream Poincare or spectrum figures
    are trusted.
    """

    rows = beta_physics_rows(rows_or_csv)
    if not rows:
        raise ValueError("rows_or_csv must contain at least one row")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    x, _x_label = _x_axis(rows)
    xlim = _padded_xlim(x)
    fig, axes = plt.subplots(3, 3, figsize=figsize, constrained_layout=True)
    axs = axes.ravel()

    accepted = _metric(rows, "accepted")
    if np.any(np.isfinite(accepted)):
        y = np.where(accepted > 0.0, 1.0, 0.0)
    else:
        y = np.ones(len(rows), dtype=float)
    colors = [_physics_gate_color(row, yi) for row, yi in zip(rows, y)]
    axs[0].scatter(x, y, c=colors, s=52, edgecolors="black", linewidths=0.45, zorder=3)
    axs[0].plot(x, y, color="0.72", lw=0.8, zorder=1)
    axs[0].set_yticks([0, 1], ["rejected", "accepted"])
    axs[0].set_ylim(-0.35, 1.35)
    axs[0].set_title("step acceptance")
    axs[0].set_xlabel("beta")
    axs[0].set_ylabel("gate")
    axs[0].grid(True, axis="x", lw=0.3, color="0.86")
    if xlim is not None:
        axs[0].set_xlim(*xlim)

    _plot_metric_group(
        axs[1],
        rows,
        x,
        (
            (("protected_fRMS",), "protected fRMS"),
            (("raw_fRMS",), "raw fRMS"),
            (("jfree_force_matrix_fRMS", "drive_normalized_fRMS"), "force fRMS"),
        ),
        title="PDE force residual",
        ylabel="percent",
        xlim=xlim,
    )
    _plot_metric_group(
        axs[2],
        rows,
        x,
        (
            (("plasma_volume_beta_after",), "plasma beta"),
            (("actual_beta_step",), "actual step"),
            (("beta_tracking_ratio",), "tracking ratio"),
        ),
        title="beta tracking",
        ylabel="value",
        xlim=xlim,
    )
    _plot_metric_group(
        axs[3],
        rows,
        x,
        (
            (("jfree_matrix_relres",), "J-free relres"),
            (("cupy_lsqr_relres",), "CuPy LSQR relres"),
            (("cupy_lsmr_relres",), "CuPy LSMR relres"),
            (("residual_norm",), "linear residual"),
        ),
        title="linear solve residual",
        ylabel="relative residual",
        yscale="log",
        xlim=xlim,
    )
    _plot_metric_group(
        axs[4],
        rows,
        x,
        (
            (("delta_B_over_B0_max",), "max dB/B0"),
            (("delta_B_over_B0_rms",), "rms dB/B0"),
            (("support_delta_B_over_B0_max",), "support max dB/B0"),
        ),
        title="amplitude gate",
        ylabel="fraction",
        yscale="log",
        xlim=xlim,
    )
    _plot_metric_group(
        axs[5],
        rows,
        x,
        (
            (("delta_J_A_per_m2_rms",), "delta J rms"),
            (("support_delta_J_A_per_m2_max",), "support delta J max"),
            (("bootstrap_J_parallel_rms",), "bootstrap J parallel"),
            (("diamagnetic_J_rms",), "diamagnetic J"),
            (("pfirsch_schluter_J_parallel_rms",), "PS J parallel"),
        ),
        title="current response",
        ylabel="A/m^2",
        yscale="log",
        xlim=xlim,
    )
    _plot_metric_group(
        axs[6],
        rows,
        x,
        (
            (("current_curl_residual_over_reference",), "curl total"),
            (("current_curl_plasma_residual_over_reference",), "curl plasma"),
            (("current_curl_plasma_interior_residual_over_reference",), "curl interior"),
            (("total_divB_plasma_rms_over_B_per_grid_length",), "divB plasma"),
            (("total_divB_interior_rms_over_B_per_grid_length",), "divB interior"),
        ),
        title="curl and divB audit",
        ylabel="normalized residual",
        yscale="log",
        xlim=xlim,
    )
    _plot_metric_group(
        axs[7],
        rows,
        x,
        (
            (("pressure_target_rel_l2",), "pressure target L2"),
            (("pressure_support_outside_abs_fraction",), "outside pressure"),
            (("pressure_support_negative_abs_fraction",), "negative pressure"),
            (("pressure_centroid_axis_displacement_m",), "centroid-axis m"),
        ),
        title="pressure support",
        ylabel="value",
        yscale="log",
        xlim=xlim,
    )
    _plot_metric_group(
        axs[8],
        rows,
        x,
        (
            (("topology_fitted_fraction",), "fitted fraction"),
            (("topology_fitted_fraction_min_section",), "min section fit"),
            (("topology_mean_self_intersections",), "self intersections"),
            (("topology_mean_raw_distance_over_a_eff",), "raw dist/a_eff"),
            (("topology_live_trace_fraction",), "live trace fraction"),
        ),
        title="topology retrace",
        ylabel="value",
        xlim=xlim,
    )

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    if out_path is not None:
        out = Path(out_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=int(dpi), bbox_inches="tight")
    return fig


__all__ = [
    "STATUS_COLORS",
    "STATUS_LEVELS",
    "beta_physics_rows",
    "beta_ramp_scan_rows",
    "plot_beta_physics_dashboard",
    "plot_beta_ramp_scan_summary",
    "read_beta_physics_csv",
]
