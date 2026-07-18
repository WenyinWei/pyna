"""Wall heat-footprint plotting helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class CameraProjection:
    """Projected camera-plane coordinates and visibility metadata."""

    u: np.ndarray
    v: np.ndarray
    depth: np.ndarray
    visible: np.ndarray
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray


@dataclass(frozen=True)
class WallHeatFootprint:
    """Binned wall-strike heat footprint in toroidal and wall-arclength coordinates."""

    heat: np.ndarray
    phi_edges: np.ndarray
    s_edges: np.ndarray
    hit_R: np.ndarray
    hit_Z: np.ndarray
    hit_phi: np.ndarray
    hit_s: np.ndarray
    hit_weight: np.ndarray
    hit_section_index: np.ndarray
    hit_wall_distance: np.ndarray

    @property
    def phi_centers(self) -> np.ndarray:
        return 0.5 * (self.phi_edges[:-1] + self.phi_edges[1:])

    @property
    def s_centers(self) -> np.ndarray:
        return 0.5 * (self.s_edges[:-1] + self.s_edges[1:])


@dataclass(frozen=True)
class WallSurfaceHeat:
    """Wall-surface heat proxy sampled on a toroidal wall grid."""

    heat: np.ndarray
    heat_flux: np.ndarray
    area: np.ndarray
    wall_phi: np.ndarray
    wall_R: np.ndarray
    wall_Z: np.ndarray
    wall_xyz: np.ndarray
    section_s: np.ndarray


def _section_arclength(R: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, float]:
    R = np.asarray(R, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float).ravel()
    d = np.hypot(np.roll(R, -1) - R, np.roll(Z, -1) - Z)
    edges = np.concatenate([[0.0], np.cumsum(d)])
    total = float(edges[-1])
    centers = edges[:-1] + 0.5 * d
    if total <= 0.0 or not np.isfinite(total):
        return np.zeros_like(R), 1.0
    return centers / total, total


def _phi_section_indices(wall_phi: np.ndarray, hit_phi: np.ndarray, *, field_period: float) -> tuple[np.ndarray, np.ndarray]:
    phi = np.asarray(wall_phi, dtype=float).ravel()
    hits = np.asarray(hit_phi, dtype=float).ravel()
    if phi.size == 0:
        raise ValueError("wall_phi must not be empty")
    phi0 = float(phi[0])
    phi_mod = phi0 + np.mod(phi - phi0, float(field_period))
    hit_mod = phi0 + np.mod(hits - phi0, float(field_period))
    diff = np.angle(np.exp(1j * (hit_mod[:, None] - phi_mod[None, :]) * (TWOPI / float(field_period))))
    idx = np.argmin(np.abs(diff), axis=1)
    return idx.astype(int), hit_mod


def wall_heat_footprint_from_hits(
    hit_R: Sequence[float],
    hit_Z: Sequence[float],
    hit_phi: Sequence[float],
    wall_phi: Sequence[float],
    wall_R: np.ndarray,
    wall_Z: np.ndarray,
    *,
    weights: Sequence[float] | None = None,
    n_phi_bins: int | None = None,
    n_s_bins: int = 160,
    field_period: float | None = None,
) -> WallHeatFootprint:
    """Bin wall strike points by toroidal angle and normalized wall arclength.

    ``hit_phi`` is folded into one field period.  The poloidal coordinate is
    obtained by snapping each hit to the nearest wall vertex on the nearest
    toroidal wall section, then using that section's normalized closed-curve
    arclength.
    """

    hit_R = np.asarray(hit_R, dtype=float).ravel()
    hit_Z = np.asarray(hit_Z, dtype=float).ravel()
    hit_phi = np.asarray(hit_phi, dtype=float).ravel()
    if hit_R.shape != hit_Z.shape or hit_R.shape != hit_phi.shape:
        raise ValueError("hit_R, hit_Z, and hit_phi must have the same shape")
    wall_phi = np.asarray(wall_phi, dtype=float).ravel()
    wall_R = np.asarray(wall_R, dtype=float)
    wall_Z = np.asarray(wall_Z, dtype=float)
    if wall_R.shape != wall_Z.shape or wall_R.ndim != 2:
        raise ValueError("wall_R and wall_Z must have matching shape (n_phi, n_poloidal)")
    if wall_R.shape[0] != wall_phi.size:
        raise ValueError("wall_phi length must match wall_R/Z n_phi")
    if weights is None:
        hit_weight = np.ones(hit_R.shape, dtype=float)
    else:
        hit_weight = np.asarray(weights, dtype=float).ravel()
        if hit_weight.shape != hit_R.shape:
            raise ValueError("weights must have the same shape as hits")
    finite = np.isfinite(hit_R) & np.isfinite(hit_Z) & np.isfinite(hit_phi) & np.isfinite(hit_weight)
    hit_R = hit_R[finite]
    hit_Z = hit_Z[finite]
    hit_phi = hit_phi[finite]
    hit_weight = hit_weight[finite]
    if field_period is None:
        if wall_phi.size > 1:
            period = float(np.nanmax(wall_phi) - np.nanmin(wall_phi) + np.nanmedian(np.diff(np.sort(wall_phi))))
        else:
            period = TWOPI
    else:
        period = float(field_period)
    section_idx, hit_phi_mod = _phi_section_indices(wall_phi, hit_phi, field_period=period)
    section_s = []
    for iphi in range(wall_phi.size):
        s, _length = _section_arclength(wall_R[iphi], wall_Z[iphi])
        section_s.append(s)
    hit_s = np.empty(hit_R.shape, dtype=float)
    hit_dist = np.empty(hit_R.shape, dtype=float)
    for i, iphi in enumerate(section_idx):
        d2 = (wall_R[iphi] - hit_R[i]) ** 2 + (wall_Z[iphi] - hit_Z[i]) ** 2
        j = int(np.nanargmin(d2))
        hit_s[i] = float(section_s[iphi][j])
        hit_dist[i] = float(np.sqrt(d2[j]))
    if n_phi_bins is None:
        n_phi_bins = max(1, wall_phi.size)
    phi0 = float(wall_phi[0])
    phi_edges = np.linspace(phi0, phi0 + period, int(n_phi_bins) + 1)
    s_edges = np.linspace(0.0, 1.0, int(n_s_bins) + 1)
    heat, _phi_edges, _s_edges = np.histogram2d(
        hit_phi_mod,
        hit_s,
        bins=(phi_edges, s_edges),
        weights=hit_weight,
    )
    return WallHeatFootprint(
        heat=heat,
        phi_edges=phi_edges,
        s_edges=s_edges,
        hit_R=hit_R,
        hit_Z=hit_Z,
        hit_phi=hit_phi_mod,
        hit_s=hit_s,
        hit_weight=hit_weight,
        hit_section_index=section_idx,
        hit_wall_distance=hit_dist,
    )


def plot_wall_heat_footprint(
    footprint: WallHeatFootprint,
    *,
    ax=None,
    cmap: str = "inferno",
    log_scale: bool = True,
    colorbar: bool = True,
    title: str | None = None,
):
    """Plot a wall heat footprint as ``phi`` versus normalized wall arclength."""

    if ax is None:
        _fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    values = np.asarray(footprint.heat, dtype=float)
    if log_scale:
        logged = np.full_like(values, np.nan, dtype=float)
        positive = values > 0.0
        logged[positive] = np.log10(values[positive])
        values = logged
        label = "log10(weighted hits)"
    else:
        label = "weighted hits"
    mesh = ax.pcolormesh(footprint.phi_edges, footprint.s_edges, values.T, shading="auto", cmap=cmap)
    ax.set_xlabel("phi [rad]")
    ax.set_ylabel("wall arclength fraction")
    if title is not None:
        ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, label=label)
    return mesh


def cylindrical_to_cartesian(R: Sequence[float], Z: Sequence[float], phi: Sequence[float]) -> np.ndarray:
    """Return Cartesian ``(x, y, z)`` points from cylindrical coordinates."""

    R_arr, Z_arr, phi_arr = np.broadcast_arrays(
        np.asarray(R, dtype=float),
        np.asarray(Z, dtype=float),
        np.asarray(phi, dtype=float),
    )
    return np.stack(
        [
            R_arr * np.cos(phi_arr),
            R_arr * np.sin(phi_arr),
            Z_arr,
        ],
        axis=-1,
    )


def camera_xyz_from_cylindrical(R: float, Z: float, phi: float) -> np.ndarray:
    """Return one Cartesian camera point from cylindrical ``(R, Z, phi)``."""

    return cylindrical_to_cartesian(float(R), float(Z), float(phi)).reshape(3)


def camera_frame(
    camera_position: Sequence[float],
    camera_target: Sequence[float],
    *,
    camera_up: Sequence[float] = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return right, up, and forward unit vectors for a camera."""

    eye = np.asarray(camera_position, dtype=float).reshape(3)
    target = np.asarray(camera_target, dtype=float).reshape(3)
    up_hint = np.asarray(camera_up, dtype=float).reshape(3)
    forward = target - eye
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 0.0:
        raise ValueError("camera_position and camera_target must be distinct")
    forward = forward / forward_norm

    right = np.cross(up_hint, forward)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1.0e-12:
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(fallback, forward))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        right = np.cross(fallback, forward)
        right_norm = float(np.linalg.norm(right))
    right = right / right_norm
    up = np.cross(forward, right)
    up = up / float(np.linalg.norm(up))
    return right, up, forward


def project_camera_points(
    points_xyz: Sequence[Sequence[float]],
    *,
    camera_position: Sequence[float],
    camera_target: Sequence[float],
    camera_up: Sequence[float] = (0.0, 0.0, 1.0),
    projection: str = "orthographic",
    focal_length: float = 1.0,
    near: float = 1.0e-6,
) -> CameraProjection:
    """Project Cartesian points into a camera plane.

    Orthographic projection is useful for wall-strike diagnostics because it
    keeps vessel geometry comparable across depth.  Perspective projection is
    available for camera-like inspection views.
    """

    points = np.asarray(points_xyz, dtype=float)
    if points.shape[-1] != 3:
        raise ValueError("points_xyz must have last dimension 3")
    flat = points.reshape(-1, 3)
    eye = np.asarray(camera_position, dtype=float).reshape(3)
    right, up, forward = camera_frame(
        eye,
        camera_target,
        camera_up=camera_up,
    )
    rel = flat - eye[None, :]
    u = rel @ right
    v = rel @ up
    depth = rel @ forward
    visible = np.isfinite(u) & np.isfinite(v) & np.isfinite(depth) & (depth > float(near))

    kind = projection.lower()
    if kind in {"orthographic", "ortho"}:
        pass
    elif kind in {"perspective", "persp"}:
        scale = np.full_like(depth, np.nan, dtype=float)
        scale[visible] = float(focal_length) / depth[visible]
        u = u * scale
        v = v * scale
    else:
        raise ValueError("projection must be 'orthographic' or 'perspective'")

    out_shape = points.shape[:-1]
    return CameraProjection(
        u=u.reshape(out_shape),
        v=v.reshape(out_shape),
        depth=depth.reshape(out_shape),
        visible=visible.reshape(out_shape),
        right=right,
        up=up,
        forward=forward,
    )


def camera_project_cylindrical(
    R: Sequence[float],
    Z: Sequence[float],
    phi: Sequence[float],
    *,
    camera_position: Sequence[float],
    camera_target: Sequence[float] = (0.0, 0.0, 0.0),
    camera_up: Sequence[float] = (0.0, 0.0, 1.0),
    focal_length: float = 1.0,
    projection: str = "perspective",
    near: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project cylindrical points into a camera view."""

    points = cylindrical_to_cartesian(R, Z, phi)
    projected = project_camera_points(
        points,
        camera_position=camera_position,
        camera_target=camera_target,
        camera_up=camera_up,
        projection=projection,
        focal_length=focal_length,
        near=near,
    )
    return np.asarray(projected.u), np.asarray(projected.v), np.asarray(projected.depth)


def _extract_hit_arrays(
    strike_points=None,
    *,
    R=None,
    Z=None,
    phi=None,
    weights=None,
    color_values=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if strike_points is not None:
        if isinstance(strike_points, WallHeatFootprint):
            R = strike_points.hit_R if R is None else R
            Z = strike_points.hit_Z if Z is None else Z
            phi = strike_points.hit_phi if phi is None else phi
            weights = strike_points.hit_weight if weights is None else weights
        elif isinstance(strike_points, Mapping):
            R = strike_points.get("R", R)
            Z = strike_points.get("Z", Z)
            phi = strike_points.get("phi", strike_points.get("Phi", phi))
            weights = strike_points.get("weight", strike_points.get("weights", weights))
            color_values = strike_points.get("connection_length", strike_points.get("color_values", color_values))
        else:
            arr = np.asarray(strike_points, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 3:
                raise ValueError("strike_points array must have shape (N, >=3)")
            R = arr[:, 0] if R is None else R
            Z = arr[:, 1] if Z is None else Z
            phi = arr[:, 2] if phi is None else phi
            if arr.shape[1] >= 4 and weights is None and color_values is None:
                weights = arr[:, 3]

    if R is None or Z is None or phi is None:
        raise ValueError("strike points require R, Z, and phi arrays")
    R_arr, Z_arr, phi_arr = np.broadcast_arrays(
        np.asarray(R, dtype=float),
        np.asarray(Z, dtype=float),
        np.asarray(phi, dtype=float),
    )
    shape = R_arr.shape
    weight_arr = None if weights is None else np.broadcast_to(np.asarray(weights, dtype=float), shape)
    color_arr = None if color_values is None else np.broadcast_to(np.asarray(color_values, dtype=float), shape)
    return (
        np.ravel(R_arr),
        np.ravel(Z_arr),
        np.ravel(phi_arr),
        None if weight_arr is None else np.ravel(weight_arr),
        None if color_arr is None else np.ravel(color_arr),
    )


def project_strike_points_camera(
    strike_points=None,
    *,
    R=None,
    Z=None,
    phi=None,
    camera_position: Sequence[float],
    camera_target: Sequence[float],
    camera_up: Sequence[float] = (0.0, 0.0, 1.0),
    projection: str = "orthographic",
    focal_length: float = 1.0,
    near: float = 1.0e-6,
) -> dict[str, np.ndarray]:
    """Project wall strike points into a camera view."""

    R_arr, Z_arr, phi_arr, _weights, _colors = _extract_hit_arrays(
        strike_points,
        R=R,
        Z=Z,
        phi=phi,
    )
    projected = project_camera_points(
        cylindrical_to_cartesian(R_arr, Z_arr, phi_arr),
        camera_position=camera_position,
        camera_target=camera_target,
        camera_up=camera_up,
        projection=projection,
        focal_length=focal_length,
        near=near,
    )
    return {
        "u": np.asarray(projected.u, dtype=float),
        "v": np.asarray(projected.v, dtype=float),
        "depth": np.asarray(projected.depth, dtype=float),
        "visible": np.asarray(projected.visible, dtype=bool),
        "R": R_arr,
        "Z": Z_arr,
        "phi": phi_arr,
    }


def _wall_xyz_from_inputs(
    *,
    wall_xyz=None,
    wall_R=None,
    wall_Z=None,
    wall_phi=None,
) -> np.ndarray | None:
    if wall_xyz is not None:
        arr = np.asarray(wall_xyz, dtype=float)
        if arr.shape[-1] != 3:
            raise ValueError("wall_xyz must have last dimension 3")
        return arr
    if wall_R is None or wall_Z is None or wall_phi is None:
        return None
    R_arr = np.asarray(wall_R, dtype=float)
    Z_arr = np.asarray(wall_Z, dtype=float)
    phi_arr = np.asarray(wall_phi, dtype=float)
    if R_arr.shape != Z_arr.shape:
        raise ValueError("wall_R and wall_Z must have matching shape")
    if phi_arr.ndim == 1 and R_arr.ndim == 2:
        if phi_arr.size != R_arr.shape[0]:
            raise ValueError("wall_phi length must match wall_R/Z n_phi")
        phi_arr = phi_arr[:, None]
    return cylindrical_to_cartesian(R_arr, Z_arr, phi_arr)


def _draw_projected_wall(
    ax,
    wall_projection: CameraProjection,
    *,
    phi_stride: int = 4,
    theta_stride: int = 16,
    color: str = "0.72",
    alpha: float = 0.24,
    linewidth: float = 0.35,
) -> list[object]:
    artists: list[object] = []
    u = np.asarray(wall_projection.u, dtype=float)
    v = np.asarray(wall_projection.v, dtype=float)
    visible = np.asarray(wall_projection.visible, dtype=bool)
    if u.ndim < 2:
        mask = visible
        artists.append(ax.scatter(u[mask], v[mask], s=0.1, c=color, alpha=alpha, rasterized=True, zorder=1))
        return artists

    n_phi, n_theta = u.shape[0], u.shape[1]
    for idx in range(0, n_phi, max(1, int(phi_stride))):
        mask = visible[idx]
        if np.count_nonzero(mask) > 1:
            (line,) = ax.plot(u[idx, mask], v[idx, mask], color=color, alpha=alpha, lw=linewidth, zorder=1)
            artists.append(line)
    for idx in range(0, n_theta, max(1, int(theta_stride))):
        mask = visible[:, idx]
        if np.count_nonzero(mask) > 1:
            (line,) = ax.plot(u[mask, idx], v[mask, idx], color=color, alpha=alpha, lw=linewidth, zorder=1)
            artists.append(line)
    return artists


def _visible_limits(u: np.ndarray, v: np.ndarray, mask: np.ndarray, pad_fraction: float = 0.04):
    if not np.any(mask):
        return None
    uu = np.asarray(u, dtype=float)[mask]
    vv = np.asarray(v, dtype=float)[mask]
    finite = np.isfinite(uu) & np.isfinite(vv)
    if not np.any(finite):
        return None
    u_min, u_max = float(np.min(uu[finite])), float(np.max(uu[finite]))
    v_min, v_max = float(np.min(vv[finite])), float(np.max(vv[finite]))
    du = max(u_max - u_min, 1.0e-6)
    dv = max(v_max - v_min, 1.0e-6)
    return (
        u_min - pad_fraction * du,
        u_max + pad_fraction * du,
        v_min - pad_fraction * dv,
        v_max + pad_fraction * dv,
    )


def _clip_to_near_depth(
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    *,
    bins: int | tuple[int, int],
    depth_margin: float,
) -> np.ndarray:
    bins_tuple = bins if isinstance(bins, tuple) else (int(bins), int(0.7 * int(bins)))
    limits = _visible_limits(u, v, mask)
    if limits is None:
        return mask
    u_edges = np.linspace(limits[0], limits[1], int(bins_tuple[0]) + 1)
    v_edges = np.linspace(limits[2], limits[3], int(bins_tuple[1]) + 1)
    ui = np.clip(np.searchsorted(u_edges, u, side="right") - 1, 0, len(u_edges) - 2)
    vi = np.clip(np.searchsorted(v_edges, v, side="right") - 1, 0, len(v_edges) - 2)
    nearest = np.full((len(u_edges) - 1, len(v_edges) - 1), np.inf)
    for idx in np.nonzero(mask)[0]:
        if depth[idx] < nearest[ui[idx], vi[idx]]:
            nearest[ui[idx], vi[idx]] = depth[idx]
    return mask & (depth <= nearest[ui, vi] + float(depth_margin))


def _maybe_smooth_heat(heat: np.ndarray, smooth_sigma: float) -> np.ndarray:
    if smooth_sigma <= 0.0:
        return heat
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        return heat
    return gaussian_filter(heat, sigma=float(smooth_sigma))


def _wall_area_elements(wall_xyz: np.ndarray) -> np.ndarray:
    points = np.asarray(wall_xyz, dtype=float)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("wall_xyz must have shape (n_phi, n_poloidal, 3)")
    d_phi = 0.5 * (np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0))
    d_theta = 0.5 * (np.roll(points, -1, axis=1) - np.roll(points, 1, axis=1))
    area = np.linalg.norm(np.cross(d_phi, d_theta), axis=-1)
    finite = np.isfinite(area) & (area > 0.0)
    if not np.any(finite):
        return np.ones(points.shape[:2], dtype=float)
    floor = float(np.nanpercentile(area[finite], 5.0))
    return np.where(finite, np.maximum(area, floor * 1.0e-3), floor)


def _wall_section_s_grid(wall_R: np.ndarray, wall_Z: np.ndarray) -> np.ndarray:
    wall_R = np.asarray(wall_R, dtype=float)
    wall_Z = np.asarray(wall_Z, dtype=float)
    section_s = np.empty_like(wall_R, dtype=float)
    for iphi in range(wall_R.shape[0]):
        section_s[iphi], _length = _section_arclength(wall_R[iphi], wall_Z[iphi])
    return section_s


def wall_surface_heat_from_footprint(
    footprint: WallHeatFootprint,
    wall_phi: Sequence[float],
    wall_R: np.ndarray,
    wall_Z: np.ndarray,
    *,
    field_period: float | None = None,
    sigma_phi: float | None = None,
    sigma_s: float = 0.018,
    phi_window_sigma: float = 3.0,
    s_window_sigma: float = 3.0,
    normalize_kernel: bool = True,
) -> WallSurfaceHeat:
    """Spread wall strikes from a footprint onto the wall surface grid.

    The returned ``heat`` has the same shape as ``wall_R``/``wall_Z`` and carries
    the summed strike weights after a local Gaussian spread in toroidal angle and
    normalized wall arclength.  ``heat_flux`` divides that heat proxy by an
    approximate wall-surface area element, which is useful for visualization.
    """

    wall_phi = np.asarray(wall_phi, dtype=float).ravel()
    wall_R = np.asarray(wall_R, dtype=float)
    wall_Z = np.asarray(wall_Z, dtype=float)
    if wall_R.shape != wall_Z.shape or wall_R.ndim != 2:
        raise ValueError("wall_R and wall_Z must have matching shape (n_phi, n_poloidal)")
    if wall_R.shape[0] != wall_phi.size:
        raise ValueError("wall_phi length must match wall_R/Z n_phi")
    if field_period is None:
        if wall_phi.size > 1:
            diffs = np.diff(np.sort(wall_phi))
            period = float(np.nanmax(wall_phi) - np.nanmin(wall_phi) + np.nanmedian(diffs))
        else:
            period = TWOPI
    else:
        period = float(field_period)
    if sigma_phi is None:
        if wall_phi.size > 1:
            spacing = float(np.nanmedian(np.diff(np.sort(wall_phi))))
            sigma_phi = max(spacing * 1.5, period / max(8.0 * wall_phi.size, 1.0))
        else:
            sigma_phi = period
    sigma_phi = max(float(sigma_phi), 1.0e-12)
    sigma_s = max(float(sigma_s), 1.0e-12)

    wall_xyz = cylindrical_to_cartesian(wall_R, wall_Z, wall_phi[:, None])
    section_s = _wall_section_s_grid(wall_R, wall_Z)
    heat = np.zeros(wall_R.shape, dtype=float)
    phi0 = float(wall_phi[0])
    wall_phi_mod = phi0 + np.mod(wall_phi - phi0, period)

    hit_phi = np.asarray(footprint.hit_phi, dtype=float).ravel()
    hit_s = np.asarray(footprint.hit_s, dtype=float).ravel()
    hit_weight = np.asarray(footprint.hit_weight, dtype=float).ravel()
    finite_hits = np.isfinite(hit_phi) & np.isfinite(hit_s) & np.isfinite(hit_weight) & (hit_weight != 0.0)
    for phi_hit, s_hit, weight in zip(hit_phi[finite_hits], hit_s[finite_hits], hit_weight[finite_hits]):
        phi_hit_mod = phi0 + np.mod(float(phi_hit) - phi0, period)
        dphi = np.angle(np.exp(1j * (wall_phi_mod - phi_hit_mod) * (TWOPI / period))) * (period / TWOPI)
        phi_keep = np.abs(dphi) <= float(phi_window_sigma) * sigma_phi
        if not np.any(phi_keep):
            phi_keep[int(np.nanargmin(np.abs(dphi)))] = True
        total_kernel = 0.0
        local_rows: list[tuple[int, np.ndarray, np.ndarray]] = []
        for iphi in np.nonzero(phi_keep)[0]:
            ds = np.abs(section_s[iphi] - float(s_hit))
            ds = np.minimum(ds, 1.0 - ds)
            s_keep = ds <= float(s_window_sigma) * sigma_s
            if not np.any(s_keep):
                s_keep[int(np.nanargmin(ds))] = True
            kernel = np.exp(-0.5 * (dphi[iphi] / sigma_phi) ** 2 - 0.5 * (ds[s_keep] / sigma_s) ** 2)
            total_kernel += float(np.sum(kernel))
            local_rows.append((int(iphi), np.nonzero(s_keep)[0], kernel))
        if normalize_kernel and total_kernel > 0.0:
            scale = float(weight) / total_kernel
        else:
            scale = float(weight)
        for iphi, js, kernel in local_rows:
            heat[iphi, js] += scale * kernel

    area = _wall_area_elements(wall_xyz)
    heat_flux = heat / np.maximum(area, 1.0e-300)
    return WallSurfaceHeat(
        heat=heat,
        heat_flux=heat_flux,
        area=area,
        wall_phi=wall_phi,
        wall_R=wall_R,
        wall_Z=wall_Z,
        wall_xyz=wall_xyz,
        section_s=section_s,
    )


def _surface_quad_payload(
    surface: WallSurfaceHeat,
    projected: CameraProjection,
    values: np.ndarray,
    *,
    near: float,
    wrap_phi: bool,
    cell_value_mode: str = "mean",
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    u = np.asarray(projected.u, dtype=float)
    v = np.asarray(projected.v, dtype=float)
    depth = np.asarray(projected.depth, dtype=float)
    visible = np.asarray(projected.visible, dtype=bool)
    values = np.asarray(values, dtype=float)
    xyz = np.asarray(surface.wall_xyz, dtype=float)
    n_phi, n_theta = values.shape
    phi_range = range(n_phi if wrap_phi else max(n_phi - 1, 0))
    polygons: list[np.ndarray] = []
    cell_values: list[float] = []
    cell_depths: list[float] = []
    cell_intensity: list[float] = []
    light = np.asarray(projected.forward, dtype=float) + 0.35 * np.asarray(projected.up, dtype=float)
    light_norm = float(np.linalg.norm(light))
    if light_norm <= 0.0:
        light = np.array([0.0, 0.0, 1.0])
    else:
        light = light / light_norm
    if cell_value_mode not in {"mean", "nearest"}:
        raise ValueError("cell_value_mode must be 'mean' or 'nearest'")
    for i in phi_range:
        i1 = (i + 1) % n_phi
        for j in range(n_theta):
            j1 = (j + 1) % n_theta
            idx = ((i, j), (i1, j), (i1, j1), (i, j1))
            if not all(bool(visible[ii, jj]) for ii, jj in idx):
                continue
            d = np.array([depth[ii, jj] for ii, jj in idx], dtype=float)
            if not np.all(np.isfinite(d)) or float(np.nanmean(d)) <= near:
                continue
            poly = np.array([[u[ii, jj], v[ii, jj]] for ii, jj in idx], dtype=float)
            if not np.all(np.isfinite(poly)):
                continue
            p0, p1, p2, _p3 = (xyz[ii, jj] for ii, jj in idx)
            normal = np.cross(p1 - p0, p2 - p0)
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm > 0.0:
                normal = normal / normal_norm
                intensity = 0.38 + 0.62 * abs(float(np.dot(normal, light)))
            else:
                intensity = 0.55
            polygons.append(poly)
            if cell_value_mode == "mean":
                cell_values.append(float(np.nanmean([values[ii, jj] for ii, jj in idx])))
            else:
                cell_values.append(float(values[i, j]))
            cell_depths.append(float(np.nanmean(d)))
            cell_intensity.append(float(np.clip(intensity, 0.25, 1.0)))
    return (
        polygons,
        np.asarray(cell_values, dtype=float),
        np.asarray(cell_depths, dtype=float),
        np.asarray(cell_intensity, dtype=float),
    )


def _rgba_with_intensity(cmap, norm, values: np.ndarray, intensity: np.ndarray, *, alpha: np.ndarray | float) -> np.ndarray:
    rgba = cmap(norm(values))
    rgba[:, :3] *= intensity[:, None]
    rgba[:, :3] = np.clip(rgba[:, :3], 0.0, 1.0)
    rgba[:, 3] = alpha if np.isscalar(alpha) else np.asarray(alpha, dtype=float)
    return rgba


def plot_wall_heat_camera(
    footprint: WallHeatFootprint,
    *,
    ax=None,
    camera_position: Sequence[float],
    camera_target: Sequence[float] = (0.0, 0.0, 0.0),
    camera_up: Sequence[float] = (0.0, 0.0, 1.0),
    projection: str = "orthographic",
    focal_length: float = 1.0,
    near: float = 1.0e-6,
    wall_xyz=None,
    wall_R=None,
    wall_Z=None,
    wall_phi=None,
    wall_color: str = "0.72",
    wall_alpha: float = 0.22,
    wall_linewidth: float = 0.35,
    wall_phi_stride: int = 4,
    wall_theta_stride: int = 16,
    background: str | None = None,
    cmap: str = "inferno",
    point_size: float = 3.0,
    heatmap_alpha: float = 0.92,
    heatmap_interpolation: str = "gaussian",
    smooth_sigma: float = 0.0,
    log_scale: bool = True,
    colorbar: bool = True,
    bins: int | tuple[int, int] | None = None,
    depth_sort: bool = True,
    depth_shade: bool = False,
    depth_margin: float | None = None,
    limit_pad_fraction: float = 0.04,
    vmin_percentile: float | None = None,
    vmax_percentile: float | None = None,
    title: str | None = None,
):
    """Plot wall-hit heat weights in a camera-projected view.

    Pass ``bins`` to aggregate strikes in camera space before plotting.  This is
    usually the clearer representation for heat-footprint diagnostics.
    """

    if ax is None:
        _fig, ax = plt.subplots(figsize=(6.0, 6.0), constrained_layout=True)
    if background is not None:
        ax.figure.set_facecolor(background)
        ax.set_facecolor(background)

    projected = project_camera_points(
        cylindrical_to_cartesian(footprint.hit_R, footprint.hit_Z, footprint.hit_phi),
        camera_position=camera_position,
        camera_target=camera_target,
        camera_up=camera_up,
        projection=projection,
        focal_length=focal_length,
        near=near,
    )
    u = np.asarray(projected.u, dtype=float).ravel()
    v = np.asarray(projected.v, dtype=float).ravel()
    depth = np.asarray(projected.depth, dtype=float).ravel()
    finite = np.asarray(projected.visible, dtype=bool).ravel() & np.isfinite(u) & np.isfinite(v)
    if depth_margin is not None and np.any(finite):
        finite = _clip_to_near_depth(
            u,
            v,
            depth,
            finite,
            bins=bins if bins is not None else (240, 170),
            depth_margin=float(depth_margin),
        )

    wall_proj = None
    wall_xyz_arr = _wall_xyz_from_inputs(wall_xyz=wall_xyz, wall_R=wall_R, wall_Z=wall_Z, wall_phi=wall_phi)
    if wall_xyz_arr is not None:
        wall_proj = project_camera_points(
            wall_xyz_arr,
            camera_position=camera_position,
            camera_target=camera_target,
            camera_up=camera_up,
            projection=projection,
            focal_length=focal_length,
            near=near,
        )
        _draw_projected_wall(
            ax,
            wall_proj,
            phi_stride=wall_phi_stride,
            theta_stride=wall_theta_stride,
            color=wall_color,
            alpha=wall_alpha,
            linewidth=wall_linewidth,
        )

    image = None
    if bins is not None:
        limit_u = u
        limit_v = v
        limit_mask = finite.copy()
        if wall_proj is not None:
            limit_u = np.concatenate([limit_u, np.ravel(wall_proj.u)])
            limit_v = np.concatenate([limit_v, np.ravel(wall_proj.v)])
            limit_mask = np.concatenate([limit_mask, np.ravel(wall_proj.visible)])
        limits = _visible_limits(np.ravel(limit_u), np.ravel(limit_v), np.ravel(limit_mask), pad_fraction=limit_pad_fraction)
        if limits is None:
            bins_tuple = bins if isinstance(bins, tuple) else (int(bins), int(0.7 * int(bins)))
            values = np.full((int(bins_tuple[0]), int(bins_tuple[1])), np.nan)
            u_edges = np.linspace(0.0, 1.0, values.shape[0] + 1)
            v_edges = np.linspace(0.0, 1.0, values.shape[1] + 1)
        else:
            bins_tuple = bins if isinstance(bins, tuple) else (int(bins), int(0.7 * int(bins)))
            heat, u_edges, v_edges = np.histogram2d(
                u[finite],
                v[finite],
                bins=bins_tuple,
                range=[[limits[0], limits[1]], [limits[2], limits[3]]],
                weights=np.asarray(footprint.hit_weight, dtype=float)[finite],
            )
            heat = _maybe_smooth_heat(heat, float(smooth_sigma))
            values = np.full_like(heat, np.nan, dtype=float)
            positive = heat > 0.0
            values[positive] = np.log10(heat[positive]) if log_scale else heat[positive]
        finite_values = values[np.isfinite(values)]
        vmin = vmax = None
        if finite_values.size:
            if vmin_percentile is not None:
                vmin = float(np.nanpercentile(finite_values, float(vmin_percentile)))
            if vmax_percentile is not None:
                vmax = float(np.nanpercentile(finite_values, float(vmax_percentile)))
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad((0.0, 0.0, 0.0, 0.0))
        image = ax.imshow(
            np.ma.masked_invalid(values.T),
            origin="lower",
            extent=[u_edges[0], u_edges[-1], v_edges[0], v_edges[-1]],
            cmap=cmap_obj,
            alpha=heatmap_alpha,
            aspect="auto",
            interpolation=heatmap_interpolation,
            vmin=vmin,
            vmax=vmax,
            zorder=3,
        )
        cbar_label = "log10(weighted hits)" if log_scale else "weighted hits"
    else:
        order = np.arange(u.size)
        if depth_sort and order.size:
            order = order[np.argsort(depth[order])[::-1]]
        order = order[finite[order]]
        if depth_shade:
            colors = depth[order]
            cmap_arg = cmap
            cbar_label = "camera depth"
        else:
            values = np.full_like(footprint.hit_weight, np.nan, dtype=float)
            weights = np.asarray(footprint.hit_weight, dtype=float)
            positive = weights > 0.0
            values[positive] = np.log10(weights[positive]) if log_scale else weights[positive]
            colors = values[order]
            cmap_arg = cmap
            cbar_label = "log10(weight)" if log_scale else "weight"
        image = ax.scatter(
            u[order],
            v[order],
            c=colors,
            s=float(point_size),
            cmap=cmap_arg,
            linewidths=0,
            alpha=float(heatmap_alpha),
            rasterized=True,
            zorder=4,
        )
    if colorbar and image is not None:
        ax.figure.colorbar(image, ax=ax, label=cbar_label, fraction=0.032, pad=0.012)

    all_u = u
    all_v = v
    all_mask = finite.copy()
    if wall_proj is not None:
        all_u = np.concatenate([np.ravel(all_u), np.ravel(wall_proj.u)])
        all_v = np.concatenate([np.ravel(all_v), np.ravel(wall_proj.v)])
        all_mask = np.concatenate([np.ravel(all_mask), np.ravel(wall_proj.visible)])
    limits = _visible_limits(np.ravel(all_u), np.ravel(all_v), np.ravel(all_mask), pad_fraction=limit_pad_fraction)
    if limits is not None:
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("camera u [m]")
    ax.set_ylabel("camera v [m]")
    if background is not None:
        ax.tick_params(colors="0.86")
        ax.xaxis.label.set_color("0.88")
        ax.yaxis.label.set_color("0.88")
        for spine in ax.spines.values():
            spine.set_edgecolor("0.35")
    if title is not None:
        ax.set_title(title)
        if background is not None:
            ax.title.set_color("0.92")
    return image


def plot_wall_heat_camera_surface(
    surface: WallSurfaceHeat,
    *,
    ax=None,
    camera_position: Sequence[float],
    camera_target: Sequence[float] = (0.0, 0.0, 0.0),
    camera_up: Sequence[float] = (0.0, 0.0, 1.0),
    projection: str = "perspective",
    focal_length: float = 1.0,
    near: float = 1.0e-6,
    value: str = "heat_flux",
    wrap_phi: bool = False,
    cell_value_mode: str = "mean",
    background: str | None = "#08090c",
    cmap: str = "inferno",
    log_scale: bool = True,
    gamma: float | None = 0.55,
    vmin_percentile: float = 5.0,
    vmax_percentile: float = 99.5,
    norm=None,
    surface_color: str = "#18202b",
    empty_alpha: float = 0.22,
    heat_alpha: float = 0.78,
    mesh: bool = True,
    mesh_color: str = "#9aa3b2",
    mesh_alpha: float = 0.12,
    mesh_linewidth: float = 0.18,
    rasterized: bool = True,
    colorbar: bool = True,
    title: str | None = None,
):
    """Render heat on projected wall-surface cells in a camera view.

    Unlike camera-plane histograms, this draws the heat proxy on the actual wall
    mesh.  Surface cells are depth sorted from far to near, which gives a static
    hfcam-style view without requiring a 3-D renderer.
    """

    try:
        from matplotlib import colors as mcolors
        from matplotlib.collections import PolyCollection
    except ImportError as exc:
        raise ImportError("matplotlib is required for plot_wall_heat_camera_surface") from exc

    if ax is None:
        _fig, ax = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    if background is not None:
        ax.figure.set_facecolor(background)
        ax.set_facecolor(background)

    projected = project_camera_points(
        surface.wall_xyz,
        camera_position=camera_position,
        camera_target=camera_target,
        camera_up=camera_up,
        projection=projection,
        focal_length=focal_length,
        near=near,
    )
    if value == "heat_flux":
        raw_values = np.asarray(surface.heat_flux, dtype=float)
        cbar_label = "heat proxy / area"
    elif value == "heat":
        raw_values = np.asarray(surface.heat, dtype=float)
        cbar_label = "weighted hits"
    else:
        raise ValueError("value must be 'heat_flux' or 'heat'")

    polygons, cell_values, cell_depths, intensity = _surface_quad_payload(
        surface,
        projected,
        raw_values,
        near=near,
        wrap_phi=wrap_phi,
        cell_value_mode=cell_value_mode,
    )
    if not polygons:
        ax.set_aspect("equal", adjustable="box")
        return None

    finite_positive = np.isfinite(cell_values) & (cell_values > 0.0)
    plot_values = cell_values
    finite_plot = np.isfinite(plot_values) & finite_positive
    if norm is not None:
        shared_norm = norm
    else:
        if np.any(finite_plot):
            vmin = float(np.nanpercentile(plot_values[finite_plot], float(vmin_percentile)))
            vmax = float(np.nanpercentile(plot_values[finite_plot], float(vmax_percentile)))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin = float(np.nanmin(plot_values[finite_plot]))
                vmax = float(np.nanmax(plot_values[finite_plot]))
            if vmax <= vmin:
                vmax = vmin + 1.0
        else:
            vmin, vmax = 0.0, 1.0
        if log_scale and vmin > 0.0:
            shared_norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        elif gamma is not None:
            shared_norm = mcolors.PowerNorm(gamma=float(gamma), vmin=vmin, vmax=vmax)
        else:
            shared_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    base_rgba = np.asarray(mcolors.to_rgba(surface_color), dtype=float)
    facecolors = np.empty((len(polygons), 4), dtype=float)
    facecolors[:] = base_rgba
    facecolors[:, :3] *= intensity[:, None]
    facecolors[:, 3] = float(empty_alpha)
    if np.any(finite_plot):
        heat_rgba = _rgba_with_intensity(
            cmap_obj,
            shared_norm,
            plot_values[finite_plot],
            intensity[finite_plot],
            alpha=float(heat_alpha),
        )
        facecolors[finite_plot] = heat_rgba

    order = np.argsort(cell_depths)[::-1]
    edgecolors = mcolors.to_rgba(mesh_color, alpha=float(mesh_alpha)) if mesh else "none"
    collection = PolyCollection(
        [polygons[int(idx)] for idx in order],
        facecolors=facecolors[order],
        edgecolors=edgecolors,
        linewidths=float(mesh_linewidth) if mesh else 0.0,
        antialiaseds=True,
        rasterized=bool(rasterized),
        zorder=2,
    )
    collection.set_clip_on(True)
    collection.set_clip_path(ax.patch)
    ax.add_collection(collection)

    u = np.asarray(projected.u, dtype=float)
    v = np.asarray(projected.v, dtype=float)
    visible = np.asarray(projected.visible, dtype=bool)
    limits = _visible_limits(np.ravel(u), np.ravel(v), np.ravel(visible))
    if limits is not None:
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("camera u [m]")
    ax.set_ylabel("camera v [m]")
    if background is not None:
        ax.tick_params(colors="0.86")
        ax.xaxis.label.set_color("0.88")
        ax.yaxis.label.set_color("0.88")
        for spine in ax.spines.values():
            spine.set_edgecolor("0.35")
    if title is not None:
        ax.set_title(title)
        if background is not None:
            ax.title.set_color("0.92")

    if colorbar and np.any(finite_plot):
        mappable = plt.cm.ScalarMappable(cmap=cmap_obj, norm=shared_norm)
        mappable.set_array(plot_values[finite_plot])
        cbar = ax.figure.colorbar(mappable, ax=ax, label=cbar_label, fraction=0.032, pad=0.012)
        if background is not None:
            cbar.ax.yaxis.set_tick_params(color="0.86")
            cbar.ax.yaxis.label.set_color("0.88")
            plt.setp(cbar.ax.get_yticklabels(), color="0.86")
    return collection


def _default_wall_heat_cameras(surface: WallSurfaceHeat) -> tuple[np.ndarray, list[dict[str, Any]]]:
    points = np.asarray(surface.wall_xyz, dtype=float).reshape(-1, 3)
    finite = np.all(np.isfinite(points), axis=1)
    if not np.any(finite):
        raise ValueError("surface.wall_xyz must contain finite wall points")
    target = np.mean(points[finite], axis=0)
    radius = 2.8 * float(np.max(np.linalg.norm(points[finite] - target, axis=1)))
    radius = max(radius, 1.0)
    cameras: list[dict[str, Any]] = []
    for index, angle in enumerate(np.linspace(0.0, TWOPI, 4, endpoint=False)):
        cameras.append(
            {
                "camera_position": target + radius * np.array([np.cos(angle), np.sin(angle), 0.28]),
                "title": f"camera {index + 1}",
            }
        )
    return target, cameras


def _camera_view_spec(camera: Mapping[str, Any] | Sequence[float], default_target: np.ndarray, index: int) -> dict[str, Any]:
    if isinstance(camera, Mapping):
        position = camera.get("camera_position", camera.get("position"))
        if position is None:
            raise ValueError("camera mappings require 'camera_position' or 'position'")
        target = camera.get("camera_target", camera.get("target", default_target))
        up = camera.get("camera_up", camera.get("up", (0.0, 0.0, 1.0)))
        title = camera.get("title", f"camera {index + 1}")
    else:
        position = camera
        target = default_target
        up = (0.0, 0.0, 1.0)
        title = f"camera {index + 1}"
    return {
        "camera_position": np.asarray(position, dtype=float).reshape(3),
        "camera_target": np.asarray(target, dtype=float).reshape(3),
        "camera_up": np.asarray(up, dtype=float).reshape(3),
        "title": str(title),
    }


def _draw_camera_collision_hits(
    ax,
    collision_hits,
    *,
    camera: Mapping[str, Any],
    projection: str,
    focal_length: float,
    near: float,
    color: str,
    alpha: float,
    point_size: float,
):
    R, Z, phi, _weights, _colors = _extract_hit_arrays(collision_hits)
    projected = project_camera_points(
        cylindrical_to_cartesian(R, Z, phi),
        camera_position=camera["camera_position"],
        camera_target=camera["camera_target"],
        camera_up=camera["camera_up"],
        projection=projection,
        focal_length=focal_length,
        near=near,
    )
    visible = np.asarray(projected.visible, dtype=bool).ravel()
    if not np.any(visible):
        return None
    depth = np.asarray(projected.depth, dtype=float).ravel()
    order = np.flatnonzero(visible)
    order = order[np.argsort(depth[order])[::-1]]
    return ax.scatter(
        np.asarray(projected.u, dtype=float).ravel()[order],
        np.asarray(projected.v, dtype=float).ravel()[order],
        s=float(point_size),
        c=color,
        alpha=float(alpha),
        linewidths=0,
        rasterized=True,
        zorder=4,
    )


def plot_wall_heat_camera_views(
    surface: WallSurfaceHeat,
    *,
    cameras: Sequence[Mapping[str, Any] | Sequence[float]] | None = None,
    collision_hits=None,
    value: str = "heat_flux",
    log_scale: bool = True,
    norm=None,
    cmap: str = "inferno",
    ncols: int = 2,
    projection: str = "perspective",
    focal_length: float = 1.0,
    near: float = 1.0e-6,
    background: str | None = "#08090c",
    empty_alpha: float = 0.22,
    heat_alpha: float = 0.78,
    collision_color: str = "#d9e2ef",
    collision_alpha: float = 0.42,
    collision_point_size: float = 2.0,
    cell_value_mode: str = "mean",
    rasterized: bool = True,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Render shared-normalized, depth-sorted wall heat views from several cameras.

    Every panel receives the same normalization calculated from the complete
    wall surface, so viewpoint visibility cannot change the physical color
    scale.  ``collision_hits`` is an optional overlay of unmodified strike
    coordinates; it is not used to construct the colored surface.
    """

    from matplotlib import colors as mcolors

    if value == "heat_flux":
        values = np.asarray(surface.heat_flux, dtype=float)
        default_label = "heat flux"
    elif value == "heat":
        values = np.asarray(surface.heat, dtype=float)
        default_label = "deposited heat"
    else:
        raise ValueError("value must be 'heat_flux' or 'heat'")
    if int(ncols) < 1:
        raise ValueError("ncols must be at least one")

    default_target, default_cameras = _default_wall_heat_cameras(surface)
    camera_specs = [_camera_view_spec(camera, default_target, index) for index, camera in enumerate(cameras or default_cameras)]
    if not camera_specs:
        raise ValueError("at least one camera is required")

    positive = values[np.isfinite(values) & (values > 0.0)]
    if norm is None:
        if positive.size:
            vmin, vmax = float(np.min(positive)), float(np.max(positive))
            if vmax <= vmin:
                vmax = vmin * 1.01
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax) if log_scale else mcolors.Normalize(vmin=0.0, vmax=vmax)
        else:
            norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    ncols = int(ncols)
    nrows = int(np.ceil(len(camera_specs) / ncols))
    if figsize is None:
        figsize = (6.6 * ncols, 5.8 * nrows)
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, constrained_layout=True)
    axes = np.asarray(axes_grid).ravel()
    collections = []
    for index, (ax, camera) in enumerate(zip(axes, camera_specs)):
        collection = plot_wall_heat_camera_surface(
            surface,
            ax=ax,
            camera_position=camera["camera_position"],
            camera_target=camera["camera_target"],
            camera_up=camera["camera_up"],
            projection=projection,
            focal_length=focal_length,
            near=near,
            value=value,
            cell_value_mode=cell_value_mode,
            background=background,
            cmap=cmap,
            log_scale=log_scale,
            norm=norm,
            empty_alpha=empty_alpha,
            heat_alpha=heat_alpha,
            rasterized=rasterized,
            colorbar=False,
            title=camera["title"],
        )
        if collision_hits is not None:
            _draw_camera_collision_hits(
                ax,
                collision_hits,
                camera=camera,
                projection=projection,
                focal_length=focal_length,
                near=near,
                color=collision_color,
                alpha=collision_alpha,
                point_size=collision_point_size,
            )
        collections.append(collection)
    for ax in axes[len(camera_specs):]:
        ax.set_visible(False)

    if colorbar and positive.size:
        mappable = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=norm)
        mappable.set_array(positive)
        cbar = fig.colorbar(mappable, ax=list(axes[:len(camera_specs)]), fraction=0.025, pad=0.018)
        cbar.set_label(colorbar_label or default_label)
        if background is not None:
            cbar.ax.yaxis.set_tick_params(color="0.86")
            cbar.ax.yaxis.label.set_color("0.88")
            plt.setp(cbar.ax.get_yticklabels(), color="0.86")
    return fig, axes[:len(camera_specs)], collections


def write_wall_heat_surface_plotly_html(
    surface: WallSurfaceHeat,
    path,
    *,
    value: str = "heat_flux",
    log_scale: bool = True,
    vmin_percentile: float = 5.0,
    vmax_percentile: float = 99.5,
    colorscale: str = "Inferno",
    title: str | None = None,
    camera_eye: Mapping[str, float] | None = None,
    width: int = 1100,
    height: int = 850,
    include_plotlyjs: str | bool = "cdn",
    downsample: int = 1,
    opacity: float = 0.72,
):
    """Write an interactive Plotly wall-surface heat view to HTML.

    Plotly is an optional dependency.  This helper keeps the same
    ``WallSurfaceHeat`` data contract as the static matplotlib render while
    providing the interactive surface view used for inspection workflows.
    """

    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError as exc:
        raise ImportError("plotly is required for write_wall_heat_surface_plotly_html") from exc

    step = max(1, int(downsample))
    xyz = np.asarray(surface.wall_xyz, dtype=float)[::step, ::step]
    if value == "heat_flux":
        raw_values = np.asarray(surface.heat_flux, dtype=float)[::step, ::step]
        colorbar_title = "log10(heat proxy / area)" if log_scale else "heat proxy / area"
    elif value == "heat":
        raw_values = np.asarray(surface.heat, dtype=float)[::step, ::step]
        colorbar_title = "log10(weighted hits)" if log_scale else "weighted hits"
    else:
        raise ValueError("value must be 'heat_flux' or 'heat'")

    positive = np.isfinite(raw_values) & (raw_values > 0.0)
    if log_scale:
        surfacecolor = np.full_like(raw_values, np.nan, dtype=float)
        surfacecolor[positive] = np.log10(raw_values[positive])
    else:
        surfacecolor = raw_values.astype(float, copy=True)
    finite = np.isfinite(surfacecolor)
    if np.any(finite):
        cmin = float(np.nanpercentile(surfacecolor[finite], float(vmin_percentile)))
        cmax = float(np.nanpercentile(surfacecolor[finite], float(vmax_percentile)))
        if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax <= cmin:
            cmin = float(np.nanmin(surfacecolor[finite]))
            cmax = float(np.nanmax(surfacecolor[finite]))
        if cmax <= cmin:
            cmax = cmin + 1.0
        surfacecolor = np.where(finite, surfacecolor, cmin)
    else:
        cmin, cmax = 0.0, 1.0
        surfacecolor = np.zeros_like(raw_values, dtype=float)

    x = xyz[:, :, 0]
    y = xyz[:, :, 1]
    z = xyz[:, :, 2]
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=surfacecolor,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            opacity=float(opacity),
            colorbar=dict(title=colorbar_title, thickness=18),
            lighting=dict(ambient=0.55, diffuse=0.85, specular=0.25, roughness=0.45),
            lightposition=dict(x=2.0, y=2.0, z=1.0),
            hovertemplate="X=%{x:.3f}<br>Y=%{y:.3f}<br>Z=%{z:.3f}<br>value=%{surfacecolor:.3f}<extra></extra>",
        )
    )
    if camera_eye is None:
        camera_eye = {"x": 1.8, "y": -1.2, "z": 0.8}
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            zaxis_title="Z [m]",
            aspectmode="data",
            camera=dict(eye=dict(camera_eye), center=dict(x=0.0, y=0.0, z=0.0), up=dict(x=0.0, y=0.0, z=1.0)),
        ),
        width=int(width),
        height=int(height),
        margin=dict(l=0, r=80, b=0, t=60),
    )
    pio.write_html(fig, str(path), include_plotlyjs=include_plotlyjs, auto_open=False)
    return fig


__all__ = [
    "CameraProjection",
    "WallHeatFootprint",
    "WallSurfaceHeat",
    "camera_frame",
    "camera_project_cylindrical",
    "camera_xyz_from_cylindrical",
    "cylindrical_to_cartesian",
    "plot_wall_heat_camera",
    "plot_wall_heat_camera_surface",
    "plot_wall_heat_camera_views",
    "plot_wall_heat_footprint",
    "project_camera_points",
    "project_strike_points_camera",
    "wall_surface_heat_from_footprint",
    "wall_heat_footprint_from_hits",
    "write_wall_heat_surface_plotly_html",
]
