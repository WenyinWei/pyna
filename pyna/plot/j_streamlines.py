"""PEST-seeded current streamline plotting helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pyna.fields import VectorFieldCylind
from pyna.toroidal.diagnostics.mgrid import SmoothPestCoordinates


TWOPI = 2.0 * np.pi


def _cylindrical_to_cartesian(R: np.ndarray, Z: np.ndarray, phi: np.ndarray) -> np.ndarray:
    R_arr, Z_arr, phi_arr = np.broadcast_arrays(
        np.asarray(R, dtype=np.float64),
        np.asarray(Z, dtype=np.float64),
        np.asarray(phi, dtype=np.float64),
    )
    return np.stack([R_arr * np.cos(phi_arr), R_arr * np.sin(phi_arr), Z_arr], axis=-1)


@dataclass(frozen=True)
class PestSeededStreamlines:
    """Current streamlines seeded from a PEST surface mesh."""

    R: np.ndarray
    Z: np.ndarray
    phi: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    seed_R: np.ndarray
    seed_Z: np.ndarray
    seed_phi: np.ndarray
    seed_rho: np.ndarray
    seed_theta: np.ndarray
    seed_surface_index: np.ndarray
    seed_phi_index: np.ndarray
    metadata: dict[str, object]

    @property
    def xyz(self) -> np.ndarray:
        return np.stack([self.x, self.y, self.z], axis=-1)

    @property
    def n_lines(self) -> int:
        return int(self.R.shape[0])

    @property
    def n_points(self) -> int:
        return int(self.R.shape[1]) if self.R.ndim == 2 else 0


@dataclass(frozen=True)
class _PeriodicVectorFieldEvaluator:
    R: np.ndarray
    Z: np.ndarray
    Phi: np.ndarray
    nfp: int
    field_period_rad: float
    interp_R: RegularGridInterpolator
    interp_Z: RegularGridInterpolator
    interp_Phi: RegularGridInterpolator

    @classmethod
    def from_field(cls, field: VectorFieldCylind) -> "_PeriodicVectorFieldEvaluator":
        arrays = field.cyna_arrays(extend_phi=True)
        R = np.asarray(arrays.R_grid, dtype=np.float64)
        Z = np.asarray(arrays.Z_grid, dtype=np.float64)
        Phi = np.asarray(arrays.Phi_grid, dtype=np.float64)
        nfp = max(int(getattr(field, "nfp", arrays.field_periods)), 1)
        field_period_rad = float(TWOPI / nfp)
        kw = dict(method="linear", bounds_error=False, fill_value=np.nan)
        axes = (R, Z, Phi)
        return cls(
            R=R,
            Z=Z,
            Phi=Phi,
            nfp=nfp,
            field_period_rad=field_period_rad,
            interp_R=RegularGridInterpolator(axes, np.asarray(arrays.BR, dtype=np.float64), **kw),
            interp_Z=RegularGridInterpolator(axes, np.asarray(arrays.BZ, dtype=np.float64), **kw),
            interp_Phi=RegularGridInterpolator(axes, np.asarray(arrays.BPhi, dtype=np.float64), **kw),
        )

    def __call__(self, R: np.ndarray, Z: np.ndarray, phi: np.ndarray) -> np.ndarray:
        phi0 = float(self.Phi[0]) if self.Phi.size else 0.0
        phi_eval = phi0 + np.mod(np.asarray(phi, dtype=np.float64) - phi0, self.field_period_rad)
        pts = np.stack([R, Z, phi_eval], axis=-1)
        return np.stack(
            [
                self.interp_R(pts),
                self.interp_Z(pts),
                self.interp_Phi(pts),
            ],
            axis=-1,
        )


def _pest_from_mapping(pest: Mapping[str, object], *, source: str | None = None) -> SmoothPestCoordinates:
    radial = pest.get("rho_vals", pest.get("radial_labels", pest.get("r_vals")))
    if radial is None:
        raise KeyError("PEST coordinates require rho_vals, radial_labels, or r_vals")
    axis_R = pest.get("axis_R", pest.get("R_AX"))
    axis_Z = pest.get("axis_Z", pest.get("Z_AX"))
    src = source if source is not None else str(pest.get("source", "")) or None
    return SmoothPestCoordinates(
        R_surf=np.asarray(pest["R_surf"], dtype=np.float64),
        Z_surf=np.asarray(pest["Z_surf"], dtype=np.float64),
        rho_vals=np.asarray(radial, dtype=np.float64),
        theta_vals=np.asarray(pest["theta_vals"], dtype=np.float64),
        phi_vals=np.asarray(pest["phi_vals"], dtype=np.float64),
        axis_R=np.asarray(axis_R, dtype=np.float64) if axis_R is not None else None,
        axis_Z=np.asarray(axis_Z, dtype=np.float64) if axis_Z is not None else None,
        source=src,
    )


def _as_pest_coordinates(pest: SmoothPestCoordinates | Mapping[str, object] | str | Path) -> SmoothPestCoordinates:
    if isinstance(pest, SmoothPestCoordinates):
        return pest
    if isinstance(pest, (str, Path)):
        path = Path(pest).expanduser()
        with np.load(path, allow_pickle=False) as data:
            payload = {name: data[name] for name in data.files}
        return _pest_from_mapping(payload, source=str(path))
    if isinstance(pest, Mapping):
        return _pest_from_mapping(pest)
    raise TypeError("pest must be SmoothPestCoordinates, mapping, or path")


def _validate_pest(pest: SmoothPestCoordinates) -> None:
    R = np.asarray(pest.R_surf, dtype=np.float64)
    Z = np.asarray(pest.Z_surf, dtype=np.float64)
    if R.shape != Z.shape or R.ndim != 3:
        raise ValueError("PEST R_surf and Z_surf must have shape (n_phi, n_rho, n_theta)")
    if np.asarray(pest.phi_vals).ndim != 1 or np.asarray(pest.phi_vals).size != R.shape[0]:
        raise ValueError("PEST phi_vals must be one-dimensional and match R_surf axis 0")
    if np.asarray(pest.rho_vals).ndim != 1 or np.asarray(pest.rho_vals).size != R.shape[1]:
        raise ValueError("PEST rho_vals must be one-dimensional and match R_surf axis 1")
    if np.asarray(pest.theta_vals).ndim != 1 or np.asarray(pest.theta_vals).size != R.shape[2]:
        raise ValueError("PEST theta_vals must be one-dimensional and match R_surf axis 2")


def _normalize_indices(index: int | Sequence[int] | None, size: int, *, default: Sequence[int]) -> np.ndarray:
    raw = np.asarray(default if index is None else ([index] if np.isscalar(index) else list(index)), dtype=np.int64)
    if raw.size == 0:
        raise ValueError("index selection must not be empty")
    return np.asarray([int(i) % int(size) for i in raw], dtype=np.int64)


def _wrap_angle_near_reference(angle: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return reference + np.mod(angle - reference + np.pi, TWOPI) - np.pi


def _cartesian_rhs(
    field_eval: _PeriodicVectorFieldEvaluator,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    min_field_norm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = np.sqrt(x * x + y * y)
    phi = np.arctan2(y, x)
    values = np.asarray(field_eval(R, z, phi), dtype=np.float64)
    vR = values[..., 0]
    vZ = values[..., 1]
    vPhi = values[..., 2]
    norm = np.sqrt(vR * vR + vZ * vZ + vPhi * vPhi)
    safe = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
        & np.isfinite(vR)
        & np.isfinite(vZ)
        & np.isfinite(vPhi)
        & (norm > float(min_field_norm))
        & (R > 1.0e-12)
    )
    dx = np.full_like(x, np.nan, dtype=np.float64)
    dy = np.full_like(y, np.nan, dtype=np.float64)
    dz = np.full_like(z, np.nan, dtype=np.float64)
    if np.any(safe):
        cp = np.cos(phi[safe])
        sp = np.sin(phi[safe])
        vx = vR[safe] * cp - vPhi[safe] * sp
        vy = vR[safe] * sp + vPhi[safe] * cp
        dx[safe] = vx / norm[safe]
        dy[safe] = vy / norm[safe]
        dz[safe] = vZ[safe] / norm[safe]
    return dx, dy, dz


def _rk4_step_cartesian(
    field_eval: _PeriodicVectorFieldEvaluator,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    phi_ref: np.ndarray,
    *,
    h: float,
    min_field_norm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k1x, k1y, k1z = _cartesian_rhs(field_eval, x, y, z, min_field_norm=min_field_norm)
    k2x, k2y, k2z = _cartesian_rhs(
        field_eval,
        x + 0.5 * h * k1x,
        y + 0.5 * h * k1y,
        z + 0.5 * h * k1z,
        min_field_norm=min_field_norm,
    )
    k3x, k3y, k3z = _cartesian_rhs(
        field_eval,
        x + 0.5 * h * k2x,
        y + 0.5 * h * k2y,
        z + 0.5 * h * k2z,
        min_field_norm=min_field_norm,
    )
    k4x, k4y, k4z = _cartesian_rhs(
        field_eval,
        x + h * k3x,
        y + h * k3y,
        z + h * k3z,
        min_field_norm=min_field_norm,
    )
    out_x = x + h * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
    out_y = y + h * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
    out_z = z + h * (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0
    out_phi = _wrap_angle_near_reference(np.arctan2(out_y, out_x), phi_ref)
    return out_x, out_y, out_z, out_phi


def _seed_points(
    pest: SmoothPestCoordinates,
    *,
    surface_indices: np.ndarray,
    phi_indices: np.ndarray,
    seed_count: int,
    theta_offset: float,
) -> dict[str, np.ndarray]:
    n_theta = pest.R_surf.shape[2]
    if int(seed_count) <= 0:
        raise ValueError("seed_count must be positive")
    theta_base = np.linspace(0, n_theta, int(seed_count), endpoint=False, dtype=np.float64)
    theta_indices = np.mod(np.rint(theta_base + float(theta_offset) * n_theta / TWOPI).astype(np.int64), n_theta)
    rows = []
    for iphi in phi_indices:
        for irho in surface_indices:
            for itheta in theta_indices:
                R0 = float(pest.R_surf[int(iphi), int(irho), int(itheta)])
                Z0 = float(pest.Z_surf[int(iphi), int(irho), int(itheta)])
                if np.isfinite(R0) and np.isfinite(Z0):
                    rows.append(
                        (
                            R0,
                            Z0,
                            float(pest.phi_vals[int(iphi)]),
                            float(pest.rho_vals[int(irho)]),
                            float(pest.theta_vals[int(itheta)]),
                            int(irho),
                            int(iphi),
                        )
                    )
    if not rows:
        raise ValueError("no finite PEST seed points were found")
    arr = np.asarray(rows, dtype=np.float64)
    return {
        "R": arr[:, 0],
        "Z": arr[:, 1],
        "phi": arr[:, 2],
        "rho": arr[:, 3],
        "theta": arr[:, 4],
        "surface_index": arr[:, 5].astype(np.int64),
        "phi_index": arr[:, 6].astype(np.int64),
    }


def trace_j_streamlines_on_pest(
    field: VectorFieldCylind,
    pest: SmoothPestCoordinates | Mapping[str, object] | str | Path,
    *,
    surface_index: int | Sequence[int] | None = -1,
    phi_indices: Sequence[int] | None = None,
    seed_count: int = 12,
    n_turns: float = 0.2,
    steps_per_turn: int = 512,
    bidirectional: bool = True,
    theta_offset: float = 0.0,
    min_field_norm: float = 1.0e-14,
) -> PestSeededStreamlines:
    """Trace normalized Cartesian current streamlines from PEST-surface seeds.

    ``field`` is a :class:`pyna.fields.VectorFieldCylind`; it is treated as the
    full current-density vector in physical cylindrical components
    ``(J_R, J_Z, J_phi)``.  The PEST mesh is used only for seed placement and
    plotting context, so callers keep control over how the trusted PEST
    coordinates are constructed.
    """

    if not isinstance(field, VectorFieldCylind):
        raise TypeError("field must be a VectorFieldCylind")
    coords = _as_pest_coordinates(pest)
    _validate_pest(coords)
    n_phi, n_rho, _n_theta = coords.R_surf.shape
    default_phi = [0] if n_phi < 4 else [0, n_phi // 4, n_phi // 2, (3 * n_phi) // 4]
    phi_idx = _normalize_indices(phi_indices, n_phi, default=default_phi)
    surf_idx = _normalize_indices(surface_index, n_rho, default=[n_rho - 1])
    seeds = _seed_points(
        coords,
        surface_indices=surf_idx,
        phi_indices=phi_idx,
        seed_count=int(seed_count),
        theta_offset=float(theta_offset),
    )

    seed_R = seeds["R"]
    seed_Z = seeds["Z"]
    seed_phi = seeds["phi"]
    n_seed = seed_R.size
    field_eval = _PeriodicVectorFieldEvaluator.from_field(field)
    n_steps = max(int(round(float(n_turns) * max(int(steps_per_turn), 1))), 1)
    n_points = 2 * n_steps + 1 if bidirectional else n_steps + 1
    h_base = TWOPI * max(float(np.nanmedian(seed_R)), 1.0e-12) / max(int(steps_per_turn), 1)

    def integrate(direction: float, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = seed_R * np.cos(seed_phi)
        y = seed_R * np.sin(seed_phi)
        z = seed_Z.copy()
        phi = seed_phi.copy()
        xs = [x.copy()]
        ys = [y.copy()]
        zs = [z.copy()]
        phis = [phi.copy()]
        for _ in range(count):
            x, y, z, phi = _rk4_step_cartesian(
                field_eval,
                x,
                y,
                z,
                phi,
                h=float(direction) * h_base,
                min_field_norm=float(min_field_norm),
            )
            xs.append(x.copy())
            ys.append(y.copy())
            zs.append(z.copy())
            phis.append(phi.copy())
        return np.stack(xs, axis=1), np.stack(ys, axis=1), np.stack(zs, axis=1), np.stack(phis, axis=1)

    if bidirectional:
        bx, by, bz, bphi = integrate(-1.0, n_steps)
        fx, fy, fz, fphi = integrate(1.0, n_steps)
        x = np.concatenate([bx[:, :0:-1], fx], axis=1)
        y = np.concatenate([by[:, :0:-1], fy], axis=1)
        z = np.concatenate([bz[:, :0:-1], fz], axis=1)
        phi = np.concatenate([bphi[:, :0:-1], fphi], axis=1)
    else:
        x, y, z, phi = integrate(1.0, n_steps)
    R = np.sqrt(x * x + y * y)

    metadata: dict[str, object] = {
        "schema": "pyna_pest_seeded_j_streamlines_v1",
        "trace_backend": "pyna.plot.j_streamlines.python_rk4_cartesian_arclength",
        "trace_parameter": "normalized_cartesian_arclength",
        "field_type": type(field).__name__,
        "nfp": int(field_eval.nfp),
        "field_period_rad": float(field_eval.field_period_rad),
        "pest_source": coords.source,
        "n_turns": float(n_turns),
        "steps_per_turn": int(steps_per_turn),
        "seed_count": int(seed_count),
        "n_seed_lines": int(n_seed),
        "n_points": int(n_points),
        "bidirectional": bool(bidirectional),
        "surface_indices": [int(i) for i in surf_idx],
        "phi_indices": [int(i) for i in phi_idx],
        "finite_fraction": float(np.count_nonzero(np.isfinite(R) & np.isfinite(z)) / max(R.size, 1)),
    }
    return PestSeededStreamlines(
        R=R,
        Z=z,
        phi=phi,
        x=x,
        y=y,
        z=z,
        seed_R=seed_R,
        seed_Z=seed_Z,
        seed_phi=seed_phi,
        seed_rho=seeds["rho"],
        seed_theta=seeds["theta"],
        seed_surface_index=seeds["surface_index"],
        seed_phi_index=seeds["phi_index"],
        metadata=metadata,
    )


def _surface_xyz(
    pest: SmoothPestCoordinates,
    surface_index: int,
    *,
    downsample: int = 1,
) -> np.ndarray:
    step = max(1, int(downsample))
    ir = int(surface_index) % pest.R_surf.shape[1]
    R = np.asarray(pest.R_surf[:, ir, :], dtype=np.float64)[::step, ::step]
    Z = np.asarray(pest.Z_surf[:, ir, :], dtype=np.float64)[::step, ::step]
    phi = np.asarray(pest.phi_vals, dtype=np.float64)[::step, np.newaxis]
    return _cylindrical_to_cartesian(R, Z, phi)


def plot_j_streamlines_on_pest_surface_plotly(
    streamlines: PestSeededStreamlines,
    pest: SmoothPestCoordinates | Mapping[str, object] | str | Path | None = None,
    *,
    surface_index: int | None = None,
    html_path: str | Path | None = None,
    include_plotlyjs: str | bool = "cdn",
    show_surface: bool = True,
    surface_downsample: int = 1,
    surface_opacity: float = 0.24,
    line_width: float = 4.0,
    title: str | None = None,
    width: int = 1100,
    height: int = 850,
):
    """Return a Plotly 3-D view of J streamlines seeded on a PEST surface."""

    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        from plotly.colors import sample_colorscale
    except ImportError as exc:
        raise ImportError("plotly is required for plot_j_streamlines_on_pest_surface_plotly") from exc

    coords = _as_pest_coordinates(pest) if pest is not None else None
    fig = go.Figure()
    if show_surface and coords is not None:
        ir = (
            int(surface_index) % coords.R_surf.shape[1]
            if surface_index is not None
            else int(streamlines.seed_surface_index[0]) % coords.R_surf.shape[1]
        )
        xyz = _surface_xyz(coords, ir, downsample=surface_downsample)
        fig.add_trace(
            go.Surface(
                x=xyz[:, :, 0],
                y=xyz[:, :, 1],
                z=xyz[:, :, 2],
                showscale=False,
                opacity=float(surface_opacity),
                colorscale=[[0.0, "rgb(185,190,198)"], [1.0, "rgb(185,190,198)"]],
                hoverinfo="skip",
                name="PEST surface",
            )
        )

    finite_rho = streamlines.seed_rho[np.isfinite(streamlines.seed_rho)]
    rho_min = float(np.nanmin(finite_rho)) if finite_rho.size else 0.0
    rho_max = float(np.nanmax(finite_rho)) if finite_rho.size else 1.0
    if rho_max <= rho_min:
        rho_max = rho_min + 1.0
    for line_idx in range(streamlines.n_lines):
        keep = (
            np.isfinite(streamlines.x[line_idx])
            & np.isfinite(streamlines.y[line_idx])
            & np.isfinite(streamlines.z[line_idx])
        )
        if np.count_nonzero(keep) < 2:
            continue
        color_value = (float(streamlines.seed_rho[line_idx]) - rho_min) / (rho_max - rho_min)
        color = sample_colorscale("Viridis", color_value)[0]
        fig.add_trace(
            go.Scatter3d(
                x=streamlines.x[line_idx, keep],
                y=streamlines.y[line_idx, keep],
                z=streamlines.z[line_idx, keep],
                mode="lines",
                line=dict(color=color, width=float(line_width)),
                name=f"J line {line_idx}",
                hovertemplate=(
                    "seed rho=%{customdata[0]:.3f}<br>"
                    "seed theta=%{customdata[1]:.3f}<br>"
                    "X=%{x:.3f}<br>Y=%{y:.3f}<br>Z=%{z:.3f}<extra></extra>"
                ),
                customdata=np.column_stack(
                    [
                        np.full(np.count_nonzero(keep), streamlines.seed_rho[line_idx]),
                        np.full(np.count_nonzero(keep), streamlines.seed_theta[line_idx]),
                    ]
                ),
                showlegend=False,
            )
        )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            zaxis_title="Z [m]",
            aspectmode="data",
            camera=dict(eye=dict(x=1.8, y=-1.4, z=0.9), up=dict(x=0.0, y=0.0, z=1.0)),
        ),
        width=int(width),
        height=int(height),
        margin=dict(l=0, r=0, b=0, t=55),
    )
    if html_path is not None:
        pio.write_html(fig, str(Path(html_path)), include_plotlyjs=include_plotlyjs, auto_open=False)
    return fig


def plot_j_streamline_seed_sections(
    streamlines: PestSeededStreamlines,
    pest: SmoothPestCoordinates | Mapping[str, object] | str | Path,
    *,
    section_indices: Sequence[int] | None = None,
    title: str | None = None,
    line_width: float = 1.1,
    alpha: float = 0.86,
):
    """Plot R/Z projections of PEST-seeded J streamlines on seed sections."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from pyna.plot.mgrid import draw_smooth_pest_grid

    coords = _as_pest_coordinates(pest)
    _validate_pest(coords)
    if section_indices is None:
        section_indices = sorted({int(i) for i in streamlines.seed_phi_index})
    sec = np.asarray(list(section_indices), dtype=np.int64)
    if sec.size == 0:
        raise ValueError("section_indices must not be empty")
    ncols = min(int(sec.size), 4)
    nrows = int(np.ceil(sec.size / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 4.0 * nrows), squeeze=False, constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    rho_min = float(np.nanmin(streamlines.seed_rho))
    rho_max = float(np.nanmax(streamlines.seed_rho))
    if rho_max <= rho_min:
        rho_max = rho_min + 1.0
    for ax in axes.ravel()[sec.size:]:
        ax.set_visible(False)
    for out_idx, iphi in enumerate(sec):
        ax = axes.ravel()[out_idx]
        ip = int(iphi) % coords.R_surf.shape[0]
        draw_smooth_pest_grid(ax, coords.R_surf[ip], coords.Z_surf[ip])
        selected = np.flatnonzero(streamlines.seed_phi_index == ip)
        for line_idx in selected:
            keep = np.isfinite(streamlines.R[line_idx]) & np.isfinite(streamlines.Z[line_idx])
            if np.count_nonzero(keep) < 2:
                continue
            cval = (float(streamlines.seed_rho[line_idx]) - rho_min) / (rho_max - rho_min)
            ax.plot(streamlines.R[line_idx, keep], streamlines.Z[line_idx, keep], color=cmap(cval), lw=line_width, alpha=alpha)
        ax.set_title(f"phi={float(coords.phi_vals[ip]):.3f}")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
    if title is not None:
        fig.suptitle(title)
    return fig, axes


__all__ = [
    "PestSeededStreamlines",
    "plot_j_streamline_seed_sections",
    "plot_j_streamlines_on_pest_surface_plotly",
    "trace_j_streamlines_on_pest",
]
