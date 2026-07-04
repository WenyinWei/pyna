"""Summary plots for beta-ramp diagnostics.

The plotting helpers consume the public rows returned by
``BetaRampScanDiagnostics.summary_rows()`` or ``beta_scan_summary_rows``.  They
do not know where the equilibrium data came from, which keeps private
stellarator adapters and public examples on the same plotting path.
"""

from __future__ import annotations

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


def _float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _metric(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray([_float_or_nan(row.get(key)) for row in rows], dtype=float)


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
        pad = max(abs(xmin), 1.0) * 0.05
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


__all__ = [
    "STATUS_COLORS",
    "STATUS_LEVELS",
    "beta_ramp_scan_rows",
    "plot_beta_ramp_scan_summary",
]
