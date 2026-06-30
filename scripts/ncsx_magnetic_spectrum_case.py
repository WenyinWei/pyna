"""NCSX beta-ramp magnetic-spectrum island-chain example.

This script uses rebuilt PEST-like NCSX surfaces as the straight-field-line
coordinate grid, samples a beta-ramp ``delta_B`` state on those surfaces, and
computes Nardon-style resonant spectra:

    tilde_b^1_mn,  b_res = 2 |tilde_b^1_{m,-n}|,
    island half-widths, O/X phases, and Chirikov overlap.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyna.toroidal.perturbation_spectrum import (
    analyze_resonant_island_chains,
    chirikov_overlaps,
    nardon_radial_perturbation,
    radial_perturbation_Fourier_spectrum,
    sample_cylindrical_vector_grid_on_surfaces,
)


DEFAULT_NCSX_ROOT = Path(os.environ.get("PYNA_NCSX_ROOT", "data/NCSX"))
DEFAULT_COORDS = (
    DEFAULT_NCSX_ROOT
    / "ncsx_beta_jfree_retrace_diag_nosmooth_beta003_fullres_20260614_v1"
    / "ncsx_qa_beta_pest_topology_coords.npz"
)
DEFAULT_OUT_DIR = DEFAULT_NCSX_ROOT / "ncsx_magnetic_spectrum_case_20260627_v2"
DEFAULT_MGRID = DEFAULT_NCSX_ROOT / "mgrid_c09r00.nc"
DEFAULT_WOUT = DEFAULT_NCSX_ROOT / "wout_ncsx_c09r00_free.nc"
DEFAULT_VESSEL = DEFAULT_NCSX_ROOT / "ncsx_vessel_3D.dat"
DEFAULT_PLOT_DIR = DEFAULT_OUT_DIR / "figures"
DEFAULT_JSON = DEFAULT_OUT_DIR / "ncsx_magnetic_spectrum_case.json"


def parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def _read_wout_extcur(path: Path | None) -> tuple[float, ...] | None:
    if path is None or not path.exists():
        return None
    from netCDF4 import Dataset

    with Dataset(str(path), "r") as ds:
        if "extcur" not in ds.variables:
            return None
        return tuple(float(x) for x in np.asarray(ds.variables["extcur"][:], dtype=float).ravel())


def _coil_scale(mode: str, raw: float, ext: float | None) -> float:
    if mode.upper() == "S":
        return float(ext) if ext is not None else (float(raw) if abs(raw) > 0.0 else 1.0)
    if ext is not None and abs(raw) > 0.0:
        return float(ext) / float(raw)
    return 1.0


def load_vmec_mgrid_vacuum(mgrid_path: Path, wout_path: Path | None) -> dict[str, np.ndarray]:
    """Load a VMEC mgrid vacuum field as full-torus ``(R,Z,Phi)`` arrays."""

    from netCDF4 import Dataset

    ext_current = _read_wout_extcur(wout_path)
    with Dataset(str(mgrid_path), "r") as ds:
        ir = int(ds.variables["ir"][()])
        jz = int(ds.variables["jz"][()])
        kp = int(ds.variables["kp"][()])
        nfp = int(ds.variables["nfp"][()])
        nextcur = int(ds.variables["nextcur"][()])
        rmin = float(ds.variables["rmin"][()])
        rmax = float(ds.variables["rmax"][()])
        zmin = float(ds.variables["zmin"][()])
        zmax = float(ds.variables["zmax"][()])
        raw = tuple(float(x) for x in np.asarray(ds.variables["raw_coil_cur"][:], dtype=float).ravel())
        mode = b"".join(ds.variables["mgrid_mode"][:]).decode(errors="ignore").strip()
        br_1p = np.zeros((kp, jz, ir), dtype=np.float64)
        bp_1p = np.zeros_like(br_1p)
        bz_1p = np.zeros_like(br_1p)
        for idx in range(nextcur):
            tag = f"{idx + 1:03d}"
            ext = ext_current[idx] if ext_current is not None and idx < len(ext_current) else None
            scale = _coil_scale(mode, raw[idx], ext)
            br_1p += np.asarray(ds.variables[f"br_{tag}"][:], dtype=np.float64) * scale
            bp_1p += np.asarray(ds.variables[f"bp_{tag}"][:], dtype=np.float64) * scale
            bz_1p += np.asarray(ds.variables[f"bz_{tag}"][:], dtype=np.float64) * scale
    br = np.concatenate([br_1p] * max(nfp, 1), axis=0)
    bp = np.concatenate([bp_1p] * max(nfp, 1), axis=0)
    bz = np.concatenate([bz_1p] * max(nfp, 1), axis=0)
    return {
        "R": np.linspace(rmin, rmax, ir, dtype=np.float64),
        "Z": np.linspace(zmin, zmax, jz, dtype=np.float64),
        "Phi": np.linspace(0.0, 2.0 * np.pi, br.shape[0], endpoint=False, dtype=np.float64),
        "BR": np.ascontiguousarray(np.transpose(br, (2, 1, 0))),
        "BPhi": np.ascontiguousarray(np.transpose(bp, (2, 1, 0))),
        "BZ": np.ascontiguousarray(np.transpose(bz, (2, 1, 0))),
        "nfp": int(nfp),
    }


def vmec_axis_from_wout(path: Path, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    from netCDF4 import Dataset

    with Dataset(str(path), "r") as ds:
        nfp = int(ds.variables["nfp"][()])
        rcc = np.asarray(ds.variables["raxis_cc"][:], dtype=float)
        zcs = np.asarray(ds.variables["zaxis_cs"][:], dtype=float)
    modes = np.arange(len(rcc), dtype=float)
    angles = modes[None, :] * nfp * np.asarray(phi, dtype=float)[:, None]
    axis_R = np.sum(rcc[None, :] * np.cos(angles), axis=1)
    axis_Z = -np.sum(zcs[None, :] * np.sin(angles), axis=1)
    return axis_R, axis_Z, nfp


def vmec_lcfs_from_wout(path: Path, phi: float, *, ntheta: int = 720) -> tuple[np.ndarray, np.ndarray]:
    from netCDF4 import Dataset

    with Dataset(str(path), "r") as ds:
        rmnc = np.asarray(ds.variables["rmnc"][-1, :], dtype=float)
        zmns = np.asarray(ds.variables["zmns"][-1, :], dtype=float)
        xm = np.asarray(ds.variables["xm"][:], dtype=float)
        xn = np.asarray(ds.variables["xn"][:], dtype=float)
    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta), endpoint=True, dtype=np.float64)
    phase = xm[None, :] * theta[:, None] - xn[None, :] * float(phi)
    return np.sum(rmnc[None, :] * np.cos(phase), axis=1), np.sum(zmns[None, :] * np.sin(phase), axis=1)


def load_vessel_sections(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline().split()
    if len(header) < 3:
        raise ValueError(f"unexpected vessel header in {path}")
    nfp, ntheta, nphi = (int(header[0]), int(header[1]), int(header[2]))
    data = np.loadtxt(path, skiprows=1, dtype=np.float64)
    if data.shape[0] != ntheta * nphi:
        raise ValueError(f"unexpected vessel data shape {data.shape}")
    sections = data[:, :2].reshape(nphi, ntheta, 2)
    phi = np.linspace(0.0, 2.0 * np.pi / max(nfp, 1), nphi, endpoint=False, dtype=np.float64)
    return phi, sections, nfp


def nearest_vessel_section(
    vessel_phi: np.ndarray,
    vessel: np.ndarray,
    nfp: int,
    phi: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    period = 2.0 * np.pi / max(int(nfp), 1)
    target = np.mod(float(phi), period)
    delta = np.mod(vessel_phi - target + 0.5 * period, period) - 0.5 * period
    idx = int(np.argmin(np.abs(delta)))
    section = np.asarray(vessel[idx], dtype=np.float64)
    return section[:, 0], section[:, 1], float(vessel_phi[idx])


def periodic_curve_interp(angle_src: np.ndarray, values: np.ndarray, angle_dst: np.ndarray) -> np.ndarray:
    src = np.mod(np.asarray(angle_src, dtype=np.float64), 2.0 * np.pi)
    vals = np.asarray(values, dtype=np.float64)
    dst = np.mod(np.asarray(angle_dst, dtype=np.float64), 2.0 * np.pi)
    order = np.argsort(src)
    src = src[order]
    vals = vals[order]
    _, unique_idx = np.unique(np.round(src, 12), return_index=True)
    src = src[unique_idx]
    vals = vals[unique_idx]
    src_ext = np.concatenate([src[-1:] - 2.0 * np.pi, src, src[:1] + 2.0 * np.pi])
    vals_ext = np.concatenate([vals[-1:], vals, vals[:1]])
    return np.interp(dst, src_ext, vals_ext)


def _coords_iota_profile(data, radial_labels: np.ndarray) -> np.ndarray | None:
    for key in ("iota", "iota_profile"):
        if key not in data.files:
            continue
        iota = np.asarray(data[key], dtype=np.float64)
        if iota.shape == radial_labels.shape and np.all(np.isfinite(iota)):
            return iota.copy()
    return None


def vmec_iota_profile_from_wout(path: Path, radial_labels: np.ndarray) -> np.ndarray:
    """Read VMEC iota and interpolate it to ``s = sqrt(psi_norm)`` labels."""

    from netCDF4 import Dataset

    labels = np.asarray(radial_labels, dtype=np.float64)
    if np.nanmin(labels) < -1.0e-12 or np.nanmax(labels) > 1.0 + 1.0e-12:
        raise ValueError("VMEC iota interpolation expects normalized radial labels in [0, 1]")
    with Dataset(str(path), "r") as ds:
        if "iotaf" in ds.variables:
            iota = np.asarray(ds.variables["iotaf"][:], dtype=np.float64)
        elif "iotas" in ds.variables:
            iota = np.asarray(ds.variables["iotas"][:], dtype=np.float64)
        else:
            raise ValueError(f"{path} does not contain iotaf or iotas")
        if "phi" in ds.variables:
            flux = np.asarray(ds.variables["phi"][:], dtype=np.float64)
            scale = float(np.nanmax(np.abs(flux)))
            flux = np.abs(flux) / scale if scale > 0.0 else np.linspace(0.0, 1.0, iota.size)
        else:
            flux = np.linspace(0.0, 1.0, iota.size, dtype=np.float64)
    order = np.argsort(flux)
    flux = flux[order]
    iota = iota[order]
    return np.interp(np.clip(labels, 0.0, 1.0) ** 2, flux, iota)


def load_surface_coordinates(
    path: Path,
    *,
    wout_path: Path | None,
    iota_source: str,
    radial_min: float,
    radial_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    with np.load(path, allow_pickle=False) as coords:
        R_surf = np.asarray(coords["R_surf"], dtype=np.float64).copy()
        Z_surf = np.asarray(coords["Z_surf"], dtype=np.float64).copy()
        phi_vals = np.asarray(coords["phi_vals"], dtype=np.float64).copy()
        theta_vals = np.asarray(coords["theta_vals"], dtype=np.float64).copy()
        if "radial_labels" in coords.files:
            radial_labels = np.asarray(coords["radial_labels"], dtype=np.float64).copy()
        elif "r_vals" in coords.files:
            radial_labels = np.asarray(coords["r_vals"], dtype=np.float64).copy()
        else:
            raise ValueError(f"{path} contains neither radial_labels nor r_vals")

        source = iota_source
        iota = None
        if iota_source == "auto":
            iota = _coords_iota_profile(coords, radial_labels)
            if iota is not None:
                source = "coords"
            elif wout_path is not None and radial_labels[0] >= -1.0e-12 and radial_labels[-1] <= 1.0 + 1.0e-12:
                iota = vmec_iota_profile_from_wout(wout_path, radial_labels)
                source = "wout"
        elif iota_source == "coords":
            iota = _coords_iota_profile(coords, radial_labels)
        elif iota_source == "wout":
            if wout_path is None:
                raise ValueError("--iota-source=wout requires --wout")
            iota = vmec_iota_profile_from_wout(wout_path, radial_labels)
        else:
            raise ValueError(f"unsupported iota_source={iota_source!r}")

    if iota is None:
        raise ValueError(f"could not determine iota profile for {path}; pass --iota-source wout/coords explicitly")
    mask = (
        np.isfinite(radial_labels)
        & np.isfinite(iota)
        & (radial_labels >= float(radial_min))
        & (radial_labels <= float(radial_max))
    )
    if np.count_nonzero(mask) < 3:
        raise ValueError("radial filter leaves fewer than three surfaces")
    return (
        R_surf[:, mask, :],
        Z_surf[:, mask, :],
        phi_vals,
        theta_vals,
        radial_labels[mask],
        np.asarray(iota, dtype=np.float64)[mask],
        source,
    )


def poincare_seed_grid_from_vmec_lcfs(
    wout_path: Path,
    *,
    fractions: list[float],
    n_angles: int,
    phi0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis_R, axis_Z, _ = vmec_axis_from_wout(wout_path, np.array([phi0], dtype=np.float64))
    lcfs_R, lcfs_Z = vmec_lcfs_from_wout(wout_path, phi0, ntheta=1080)
    geom = np.mod(np.arctan2(lcfs_Z - axis_Z[0], lcfs_R - axis_R[0]), 2.0 * np.pi)
    angles = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)
    edge_R = periodic_curve_interp(geom, lcfs_R, angles)
    edge_Z = periodic_curve_interp(geom, lcfs_Z, angles)
    seeds_R = []
    seeds_Z = []
    seed_fraction = []
    for frac in fractions:
        for r_edge, z_edge in zip(edge_R, edge_Z):
            seeds_R.append(axis_R[0] + float(frac) * (r_edge - axis_R[0]))
            seeds_Z.append(axis_Z[0] + float(frac) * (z_edge - axis_Z[0]))
            seed_fraction.append(float(frac))
    return np.asarray(seeds_R), np.asarray(seeds_Z), np.asarray(seed_fraction)


def poincare_seed_grid_from_coordinates(
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    radial_labels: np.ndarray,
    *,
    labels: list[float],
    n_angles: int,
    phi0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = np.asarray(phi_vals, dtype=np.float64)
    theta = np.asarray(theta_vals, dtype=np.float64)
    radial = np.asarray(radial_labels, dtype=np.float64)
    iphi = int(np.argmin(np.abs(np.angle(np.exp(1j * (phi - float(phi0)))))))
    angles = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False, dtype=np.float64)
    seeds_R = []
    seeds_Z = []
    seed_label = []
    for label in labels:
        target = float(label)
        if target < radial[0] or target > radial[-1]:
            continue
        for angle in angles:
            R_theta = np.array(
                [periodic_curve_interp(theta, R_surf[iphi, ir], np.asarray([angle]))[0] for ir in range(radial.size)]
            )
            Z_theta = np.array(
                [periodic_curve_interp(theta, Z_surf[iphi, ir], np.asarray([angle]))[0] for ir in range(radial.size)]
            )
            seeds_R.append(float(np.interp(target, radial, R_theta)))
            seeds_Z.append(float(np.interp(target, radial, Z_theta)))
            seed_label.append(target)
    return np.asarray(seeds_R, dtype=np.float64), np.asarray(seeds_Z, dtype=np.float64), np.asarray(seed_label)


def cyna_field_cache(R: np.ndarray, Z: np.ndarray, Phi: np.ndarray, BR: np.ndarray, BPhi: np.ndarray, BZ: np.ndarray) -> dict:
    phi = np.asarray(Phi, dtype=np.float64)
    br = np.asarray(BR, dtype=np.float64)
    bp = np.asarray(BPhi, dtype=np.float64)
    bz = np.asarray(BZ, dtype=np.float64)
    if phi.size > 1:
        dphi = float(phi[1] - phi[0])
        endpoint = float(phi[-1] + dphi)
        if abs(endpoint - 2.0 * np.pi) < max(1.0e-8, 10.0 * abs(dphi) * np.finfo(float).eps):
            phi = np.append(phi, 2.0 * np.pi)
            br = np.concatenate([br, br[:, :, :1]], axis=2)
            bp = np.concatenate([bp, bp[:, :, :1]], axis=2)
            bz = np.concatenate([bz, bz[:, :, :1]], axis=2)
    return {
        "R": np.ascontiguousarray(R, dtype=np.float64),
        "Z": np.ascontiguousarray(Z, dtype=np.float64),
        "Phi": np.ascontiguousarray(phi, dtype=np.float64),
        "BR": np.ascontiguousarray(br, dtype=np.float64).ravel(),
        "BPhi": np.ascontiguousarray(bp, dtype=np.float64).ravel(),
        "BZ": np.ascontiguousarray(bz, dtype=np.float64).ravel(),
    }


def trace_poincare_multi_sections(
    field: dict,
    seeds_R: np.ndarray,
    seeds_Z: np.ndarray,
    phi_sections: np.ndarray,
    *,
    n_turns: int,
    dphi: float,
    n_threads: int = -1,
) -> dict:
    import pyna._cyna as cyna

    box_R = np.array([field["R"][0], field["R"][-1], field["R"][-1], field["R"][0], field["R"][0]], dtype=np.float64)
    box_Z = np.array([field["Z"][0], field["Z"][0], field["Z"][-1], field["Z"][-1], field["Z"][0]], dtype=np.float64)
    counts, R_flat, Z_flat = cyna.trace_poincare_multi(
        np.ascontiguousarray(seeds_R, dtype=np.float64),
        np.ascontiguousarray(seeds_Z, dtype=np.float64),
        np.ascontiguousarray(phi_sections, dtype=np.float64),
        int(n_turns),
        float(dphi),
        field["BR"],
        field["BZ"],
        field["BPhi"],
        field["R"],
        field["Z"],
        field["Phi"],
        box_R,
        box_Z,
        int(n_threads),
    )
    return {
        "counts": np.asarray(counts, dtype=np.int64),
        "R_flat": np.asarray(R_flat, dtype=np.float64),
        "Z_flat": np.asarray(Z_flat, dtype=np.float64),
        "phi_sections": np.asarray(phi_sections, dtype=np.float64),
        "n_turns": int(n_turns),
    }


def poincare_section_points(poincare: dict, seed_fraction: np.ndarray, section_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.asarray(poincare["counts"], dtype=np.int64)
    R_flat = np.asarray(poincare["R_flat"], dtype=np.float64)
    Z_flat = np.asarray(poincare["Z_flat"], dtype=np.float64)
    n_turns = int(poincare["n_turns"])
    n_sec = counts.shape[1]
    out_R = []
    out_Z = []
    out_frac = []
    for i_seed in range(counts.shape[0]):
        cnt = int(counts[i_seed, section_index])
        base = i_seed * n_sec * n_turns + section_index * n_turns
        if cnt <= 0:
            continue
        out_R.append(R_flat[base : base + cnt])
        out_Z.append(Z_flat[base : base + cnt])
        out_frac.append(np.full(cnt, seed_fraction[i_seed], dtype=np.float64))
    if not out_R:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty
    return np.concatenate(out_R), np.concatenate(out_Z), np.concatenate(out_frac)


def latest_beta_state(root: Path) -> Path:
    states = sorted(root.rglob("ncsx_beta_final_state.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not states:
        raise FileNotFoundError(f"no ncsx_beta_final_state.npz found below {root}")
    return states[0]


def scalar_from_npz(data, key: str, default=None):
    if key not in data:
        return default
    value = data[key]
    if np.ndim(value) == 0:
        return value.item()
    return value


def chain_to_dict(chain) -> dict:
    pts = chain.fixed_points(0.0)
    return {
        "m": chain.m,
        "n": chain.n,
        "radial_label": chain.radial_label,
        "q": chain.q,
        "q_prime": chain.q_prime,
        "coefficient_real": float(np.real(chain.coefficient)),
        "coefficient_imag": float(np.imag(chain.coefficient)),
        "b_res": chain.b_res,
        "phase_rad": chain.phase,
        "phase_deg": float(np.degrees(chain.phase)),
        "half_width": chain.half_width,
        "theta_O_phi0_deg": [float(x) for x in np.degrees(pts["theta_O"][0])],
        "theta_X_phi0_deg": [float(x) for x in np.degrees(pts["theta_X"][0])],
    }


def overlap_to_dict(overlap) -> dict:
    return {
        "left_mode": [overlap.left.m, overlap.left.n],
        "right_mode": [overlap.right.m, overlap.right.n],
        "left_radial_label": overlap.left.radial_label,
        "right_radial_label": overlap.right.radial_label,
        "separation": overlap.separation,
        "sigma": overlap.sigma,
    }


def save_plots(
    plot_dir: Path,
    *,
    spectrum,
    chains,
    overlaps,
    R_surf,
    Z_surf,
    phi_vals,
    theta_vals,
    radial_labels,
    phase_shifts_deg,
    poincare=None,
    poincare_seed_fraction=None,
    poincare_seed_label="seed radial label",
    vessel_phi=None,
    vessel_sections=None,
    vessel_nfp=None,
    wout_path=None,
    poincare_label="",
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pyna.toroidal.visual.magnetic_spectrum import (
        plot_chirikov_overlaps,
        plot_island_chains_on_section,
        plot_island_phase_scan,
        plot_resonant_radial_profiles,
        plot_spectrum_heatmap,
    )

    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    ordered = sorted(chains, key=lambda c: c.b_res, reverse=True)
    strongest = ordered[0] if ordered else None
    radial_index = len(radial_labels) // 2
    if strongest is not None:
        radial_index = int(np.argmin(np.abs(np.asarray(radial_labels) - strongest.radial_label)))

    fig, ax = plot_spectrum_heatmap(
        spectrum,
        radial_index=radial_index,
        m_max=min(24, max(1, int(np.max(np.abs(spectrum.m))))),
        n_max=min(12, max(1, int(np.max(np.abs(spectrum.n))))),
        chains=ordered,
        title="NCSX beta-ramp magnetic spectrum",
    )
    path = plot_dir / "01_spectrum_heatmap.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    fig, ax = plot_resonant_radial_profiles(spectrum, ordered, max_modes=10)
    path = plot_dir / "02_resonant_radial_profiles.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    fig, ax, _ = plot_island_chains_on_section(
        R_surf,
        Z_surf,
        phi_vals,
        theta_vals,
        radial_labels,
        ordered,
        phi_section=0.0,
        max_chains=4,
        title="NCSX PEST section with O-point island-width bars",
    )
    path = plot_dir / "03_island_width_bars_phi0.png"
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    section_phis = np.radians([0.0, 30.0, 60.0, 90.0])
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10.0), constrained_layout=True)
    for ax, phi_section in zip(axes.ravel(), section_phis):
        plot_island_chains_on_section(
            R_surf,
            Z_surf,
            phi_vals,
            theta_vals,
            radial_labels,
            ordered,
            phi_section=float(phi_section),
            max_chains=3,
            show_legend=False,
            ax=ax,
            title=f"phi={np.degrees(phi_section):.0f} deg",
        )
    fig.suptitle("NCSX island-width bars across toroidal sections", fontsize=13)
    path = plot_dir / "04_island_width_bars_multisection.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    if poincare is not None and vessel_phi is not None and wout_path is not None:
        phi_sections = np.asarray(poincare["phi_sections"], dtype=np.float64)
        n_panels = len(phi_sections)
        ncols = 2 if n_panels == 4 else min(3, max(1, n_panels))
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5.2 * ncols, 5.4 * nrows),
            squeeze=False,
            constrained_layout=True,
        )
        for ax in axes.ravel()[n_panels:]:
            ax.axis("off")
        for i_sec, (ax, phi_section) in enumerate(zip(axes.ravel(), phi_sections)):
            if vessel_sections is not None and vessel_nfp is not None:
                wall_R, wall_Z, _ = nearest_vessel_section(vessel_phi, vessel_sections, int(vessel_nfp), float(phi_section))
                ax.plot(
                    np.r_[wall_R, wall_R[0]],
                    np.r_[wall_Z, wall_Z[0]],
                    color="0.25",
                    lw=1.6,
                    alpha=0.85,
                    label="NCSX vessel",
                    zorder=0,
                )
            lcfs_R, lcfs_Z = vmec_lcfs_from_wout(Path(wout_path), float(phi_section), ntheta=720)
            ax.plot(lcfs_R, lcfs_Z, color="black", lw=1.0, alpha=0.8, label="VMEC LCFS", zorder=2)
            pR, pZ, pfrac = poincare_section_points(poincare, np.asarray(poincare_seed_fraction), i_sec)
            if pR.size:
                sc = ax.scatter(
                    pR,
                    pZ,
                    c=pfrac,
                    s=2.2,
                    cmap="turbo",
                    vmin=float(np.nanmin(poincare_seed_fraction)),
                    vmax=float(np.nanmax(poincare_seed_fraction)),
                    alpha=0.46,
                    linewidths=0.0,
                    rasterized=True,
                    zorder=3,
                )
            plot_island_chains_on_section(
                R_surf,
                Z_surf,
                phi_vals,
                theta_vals,
                radial_labels,
                ordered,
                phi_section=float(phi_section),
                max_chains=3,
                show_legend=False,
                ax=ax,
                title=f"phi={np.degrees(phi_section):.0f} deg",
            )
            ax.set_xlim(0.45, 2.55)
            ax.set_ylim(-1.08, 1.08)
            ax.set_aspect("equal", adjustable="box")
            if i_sec == 0:
                ax.legend(loc="upper right", fontsize=7)
        if "sc" in locals():
            cbar = fig.colorbar(sc, ax=axes.ravel()[:n_panels], shrink=0.78, pad=0.01)
            cbar.set_label(poincare_seed_label)
        fig.suptitle(f"NCSX wall, Poincare traces, and magnetic-spectrum island bars ({poincare_label})", fontsize=13)
        path = plot_dir / "07_wall_poincare_island_design_overview.png"
        fig.savefig(path, dpi=210, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    if strongest is not None:
        phase_shifts = np.radians(np.asarray(phase_shifts_deg, dtype=float))
        fig, ax = plot_island_phase_scan(strongest, phase_shifts=phase_shifts, phi_section=0.0)
        path = plot_dir / "05_phase_scan_strongest_chain.png"
        fig.savefig(path, dpi=190, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    if overlaps:
        fig, ax = plot_chirikov_overlaps(overlaps)
        path = plot_dir / "06_chirikov_overlaps.png"
        fig.savefig(path, dpi=190, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncsx-root", type=Path, default=DEFAULT_NCSX_ROOT)
    parser.add_argument("--coords", type=Path, default=DEFAULT_COORDS)
    parser.add_argument("--mgrid", type=Path, default=DEFAULT_MGRID)
    parser.add_argument("--wout", type=Path, default=DEFAULT_WOUT)
    parser.add_argument("--vessel", type=Path, default=DEFAULT_VESSEL)
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--iota-source", choices=("auto", "coords", "wout"), default="auto")
    parser.add_argument("--radial-min", type=float, default=0.04)
    parser.add_argument("--radial-max", type=float, default=1.0)
    parser.add_argument("--n-values", type=parse_ints, default=parse_ints("2,3,4"))
    parser.add_argument("--m-max", type=int, default=24)
    parser.add_argument("--spectrum-n-max", type=int, default=12)
    parser.add_argument("--min-amplitude", type=float, default=0.0)
    parser.add_argument("--min-b-res", type=float, default=0.0)
    parser.add_argument("--phase-shifts-deg", type=parse_floats, default=parse_floats("0,15,30,45,60,75,90,105,120,135,150,165,180"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-poincare", action="store_true")
    parser.add_argument("--poincare-field", choices=("total", "vacuum-state", "mgrid"), default="total")
    parser.add_argument("--poincare-seed-source", choices=("coords", "vmec-lcfs"), default="coords")
    parser.add_argument("--poincare-sections-deg", type=parse_floats, default=parse_floats("0,30,60,90"))
    parser.add_argument(
        "--seed-fractions",
        type=parse_floats,
        default=parse_floats("0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95"),
    )
    parser.add_argument("--seed-angles", type=int, default=8)
    parser.add_argument("--poincare-turns", type=int, default=90)
    parser.add_argument("--poincare-dphi", type=float, default=0.035)
    parser.add_argument("--poincare-threads", type=int, default=-1)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.out == DEFAULT_JSON and args.out_dir != DEFAULT_OUT_DIR:
        args.out = args.out_dir / DEFAULT_JSON.name
    if args.plot_dir == DEFAULT_PLOT_DIR and args.out_dir != DEFAULT_OUT_DIR:
        args.plot_dir = args.out_dir / "figures"
    state_path = args.state if args.state is not None else latest_beta_state(args.ncsx_root)

    R_surf, Z_surf, phi_vals, theta_vals, radial_labels, iota, iota_source = load_surface_coordinates(
        args.coords,
        wout_path=args.wout,
        iota_source=args.iota_source,
        radial_min=args.radial_min,
        radial_max=args.radial_max,
    )

    poincare_grid = None
    with np.load(state_path, allow_pickle=False) as state:
        grid_R = state["R"]
        grid_Z = state["Z"]
        grid_phi = state["Phi"]
        beta = scalar_from_npz(state, "beta")
        dBR, dBphi, dBZ = sample_cylindrical_vector_grid_on_surfaces(
            grid_R,
            grid_Z,
            grid_phi,
            state["delta_B_R"],
            state["delta_B_Phi"],
            state["delta_B_Z"],
            R_surf,
            Z_surf,
            phi_vals,
            theta_vals,
        )
        _, B0phi, _ = sample_cylindrical_vector_grid_on_surfaces(
            grid_R,
            grid_Z,
            grid_phi,
            state["B0_R"],
            state["B0_Phi"],
            state["B0_Z"],
            R_surf,
            Z_surf,
            phi_vals,
            theta_vals,
        )
        if args.poincare_field == "total":
            poincare_grid = {
                "R": grid_R,
                "Z": grid_Z,
                "Phi": grid_phi,
                "BR": state["B0_R"] + state["delta_B_R"],
                "BPhi": state["B0_Phi"] + state["delta_B_Phi"],
                "BZ": state["B0_Z"] + state["delta_B_Z"],
                "label": "B0 + delta_B beta-ramp field",
            }
        elif args.poincare_field == "vacuum-state":
            poincare_grid = {
                "R": grid_R,
                "Z": grid_Z,
                "Phi": grid_phi,
                "BR": state["B0_R"],
                "BPhi": state["B0_Phi"],
                "BZ": state["B0_Z"],
                "label": "vacuum B0 from beta artifact",
            }

    if args.poincare_field == "mgrid":
        mgrid = load_vmec_mgrid_vacuum(args.mgrid, args.wout)
        poincare_grid = {
            "R": mgrid["R"],
            "Z": mgrid["Z"],
            "Phi": mgrid["Phi"],
            "BR": mgrid["BR"],
            "BPhi": mgrid["BPhi"],
            "BZ": mgrid["BZ"],
            "label": f"vacuum mgrid Nfp={mgrid['nfp']}",
        }

    tilde_b1 = nardon_radial_perturbation(
        R_surf,
        Z_surf,
        phi_vals,
        theta_vals,
        dBR,
        dBZ,
        dBphi,
        radial_labels,
        denominator_B_phi=B0phi,
    )
    finite_fraction = float(np.mean(np.isfinite(tilde_b1)))
    tilde_b1 = np.where(np.isfinite(tilde_b1), tilde_b1, 0.0)

    spectrum = radial_perturbation_Fourier_spectrum(
        tilde_b1,
        theta_vals,
        phi_vals,
        radial_labels=radial_labels,
        m_max=args.m_max,
        n_max=args.spectrum_n_max,
        min_amplitude=args.min_amplitude,
    )

    q_profile = 1.0 / np.asarray(iota, dtype=np.float64)
    chains = []
    for n_val in args.n_values:
        chains.extend(
            analyze_resonant_island_chains(
                spectrum,
                q_profile,
                n=n_val,
                m_values=range(1, args.m_max + 1),
                min_b_res=args.min_b_res,
            )
        )
    overlaps = chirikov_overlaps(chains)

    vessel_phi = vessel_sections = vessel_nfp = None
    poincare = None
    seed_fraction = None
    if not args.no_poincare:
        vessel_phi, vessel_sections, vessel_nfp = load_vessel_sections(args.vessel)
        print(f"vessel: {args.vessel}  Nfp={vessel_nfp} sections={len(vessel_phi)}")
        if int(vessel_nfp) != 3:
            print(f"warning: expected NCSX Nfp=3, vessel file reports {vessel_nfp}")
        if args.poincare_seed_source == "coords":
            seeds_R, seeds_Z, seed_fraction = poincare_seed_grid_from_coordinates(
                R_surf,
                Z_surf,
                phi_vals,
                theta_vals,
                radial_labels,
                labels=args.seed_fractions,
                n_angles=args.seed_angles,
                phi0=0.0,
            )
            seed_label_name = "seed radial label s"
        else:
            seeds_R, seeds_Z, seed_fraction = poincare_seed_grid_from_vmec_lcfs(
                args.wout,
                fractions=args.seed_fractions,
                n_angles=args.seed_angles,
                phi0=0.0,
            )
            seed_label_name = "seed fraction to VMEC LCFS"
        if seeds_R.size == 0:
            raise RuntimeError("no valid Poincare seeds after radial filtering")
        print(
            f"tracing Poincare: field={poincare_grid['label']} seed_source={args.poincare_seed_source} seeds={seeds_R.size} "
            f"turns={args.poincare_turns} sections={args.poincare_sections_deg}"
        )
        field = cyna_field_cache(
            poincare_grid["R"],
            poincare_grid["Z"],
            poincare_grid["Phi"],
            poincare_grid["BR"],
            poincare_grid["BPhi"],
            poincare_grid["BZ"],
        )
        phi_sections = np.radians(np.asarray(args.poincare_sections_deg, dtype=np.float64))
        poincare = trace_poincare_multi_sections(
            field,
            seeds_R,
            seeds_Z,
            phi_sections,
            n_turns=args.poincare_turns,
            dphi=args.poincare_dphi,
            n_threads=args.poincare_threads,
        )
        poincare_path = args.out_dir / "ncsx_poincare_traces.npz"
        np.savez_compressed(
            poincare_path,
            seed_R=seeds_R,
            seed_Z=seeds_Z,
            seed_fraction=seed_fraction,
            counts=poincare["counts"],
            R_flat=poincare["R_flat"],
            Z_flat=poincare["Z_flat"],
            phi_sections=poincare["phi_sections"],
            n_turns=np.array(poincare["n_turns"], dtype=np.int64),
            field_label=np.array(poincare_grid["label"]),
        )
        print(f"wrote Poincare traces: {poincare_path}")

    print(f"coords: {args.coords}")
    print(f"state:  {state_path}")
    print(f"beta:   {beta}")
    print(f"iota:   {iota_source}")
    print(
        f"surfaces: n_phi={phi_vals.size} n_s={radial_labels.size} n_theta={theta_vals.size} "
        f"s=[{radial_labels[0]:.4g}, {radial_labels[-1]:.4g}] "
        f"q=[{np.nanmin(q_profile):.4g}, {np.nanmax(q_profile):.4g}] "
        f"finite_tilde_b1={finite_fraction:.3f}"
    )
    print("\nResonant island chains at phi=0:")
    print("  m/n     s_res      q        q_prime      b_res        phase_deg    width_s    thetaO0_deg thetaX0_deg")
    for chain in chains:
        pts = chain.fixed_points(0.0)
        theta_o0 = float(np.degrees(pts["theta_O"][0, 0]))
        theta_x0 = float(np.degrees(pts["theta_X"][0, 0]))
        print(
            f"  {chain.m:2d}/{chain.n:<2d}  {chain.radial_label:8.5f}  {chain.q:7.4f} "
            f"{chain.q_prime:11.4e}  {chain.b_res:11.4e} {np.degrees(chain.phase):11.3f} "
            f"{chain.half_width:9.4e} {theta_o0:11.3f} {theta_x0:11.3f}"
        )

    if chains:
        strongest = max(chains, key=lambda c: c.b_res)
        print(f"\nPhase scan for strongest chain ({strongest.m},{strongest.n}) at phi=0:")
        base_o = strongest.fixed_points(0.0)["theta_O"][0, 0]
        for shift_deg in args.phase_shifts_deg:
            shifted = strongest.with_phase_shift(np.radians(shift_deg))
            candidates = shifted.fixed_points(0.0)["theta_O"][0]
            expected_o = base_o - np.radians(shift_deg) / float(strongest.m)
            candidate_idx = int(np.argmin(np.abs(np.angle(np.exp(1j * (candidates - expected_o))))))
            theta_o = candidates[candidate_idx]
            dtheta = np.degrees(np.angle(np.exp(1j * (theta_o - base_o))))
            print(
                f"  dphase={shift_deg:8.3f} deg -> theta_O0={np.degrees(theta_o):9.3f} deg "
                f"(delta={dtheta:9.3f} deg, expected={-shift_deg / strongest.m:9.3f} deg)"
            )

    if overlaps:
        print("\nChirikov overlaps:")
        for overlap in overlaps:
            print(
                f"  ({overlap.left.m},{overlap.left.n})-({overlap.right.m},{overlap.right.n}) "
                f"sep={overlap.separation:.4e} sigma={overlap.sigma:.4g}"
            )

    if args.out is not None:
        payload = {
            "coords": str(args.coords),
            "state": str(state_path),
            "mgrid": str(args.mgrid),
            "wout": str(args.wout),
            "vessel": str(args.vessel),
            "beta": beta,
            "iota_source": iota_source,
            "radial_min": float(args.radial_min),
            "radial_max": float(args.radial_max),
            "finite_tilde_b1_fraction": finite_fraction,
            "radial_labels": radial_labels.tolist(),
            "iota": np.asarray(iota).tolist(),
            "q_profile": q_profile.tolist(),
            "chains": [chain_to_dict(chain) for chain in chains],
            "overlaps": [overlap_to_dict(overlap) for overlap in overlaps],
            "poincare": None
            if poincare is None
            else {
                "field": poincare_grid["label"],
                "sections_deg": [float(x) for x in args.poincare_sections_deg],
                "n_turns": int(args.poincare_turns),
                "n_seeds": int(len(seed_fraction)),
                "seed_fractions": [float(x) for x in args.seed_fractions],
                "seed_source": args.poincare_seed_source,
            },
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")

    if not args.no_plots:
        paths = save_plots(
            args.plot_dir,
            spectrum=spectrum,
            chains=chains,
            overlaps=overlaps,
            R_surf=R_surf,
            Z_surf=Z_surf,
            phi_vals=phi_vals,
            theta_vals=theta_vals,
            radial_labels=radial_labels,
            phase_shifts_deg=args.phase_shifts_deg,
            poincare=poincare,
            poincare_seed_fraction=seed_fraction,
            poincare_seed_label=seed_label_name if seed_fraction is not None else "seed radial label",
            vessel_phi=vessel_phi,
            vessel_sections=vessel_sections,
            vessel_nfp=vessel_nfp,
            wout_path=args.wout,
            poincare_label="" if poincare_grid is None else poincare_grid["label"],
        )
        print("\nwrote plots:")
        for path in paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
