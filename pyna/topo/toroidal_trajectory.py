from __future__ import annotations

"""Toroidal sampled trajectories.

This module is explicitly coordinate-system-specific: it stores trajectories in
(R, Z, phi) and provides toroidal section-intersection helpers.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

from pyna.topo._base import Trajectory


@dataclass
class ToroidalTrajectory(Trajectory):
    """Sampled trajectory in a toroidal vector field, stored as (R, Z, phi)."""

    R: np.ndarray
    Z: np.ndarray
    phi: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def states(self) -> np.ndarray:
        return np.column_stack([self.R, self.Z])

    @property
    def times(self) -> np.ndarray:
        return self.phi

    @property
    def time_name(self) -> str:
        return "phi"

    @property
    def coordinate_names(self) -> Tuple[str, str]:
        return ("R", "Z")

    @property
    def ambient_dim(self) -> int:
        return 2

    @property
    def n_samples(self) -> int:
        return int(len(self.phi))

    def interpolate_at(self, phi_value: float) -> np.ndarray:
        Rv, Zv = self.intersect(float(phi_value), return_first=True)
        return np.array([Rv, Zv], dtype=float)

    def intersect(
        self,
        phi_target: float,
        *,
        return_first: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray] | Tuple[float, float]:
        """Return (R, Z) points where the trajectory crosses phi_target."""
        Np = self.metadata.get('Np', 1)
        phi_period = 2.0 * np.pi / Np
        phi_t = phi_target % phi_period

        shifted = self.phi - phi_t
        floor_i = np.floor(shifted[:-1] / phi_period)
        floor_i1 = np.floor(shifted[1:] / phi_period)
        cross_idx = np.where(floor_i1 > floor_i)[0]

        R_cross, Z_cross = [], []
        for idx in cross_idx:
            phi_cross = phi_t + (floor_i[idx] + 1.0) * phi_period
            h = self.phi[idx + 1] - self.phi[idx]
            if h < 1e-30:
                continue
            t = (phi_cross - self.phi[idx]) / h
            if not (0.0 <= t <= 1.0):
                continue
            R_cross.append(self.R[idx] + t * (self.R[idx + 1] - self.R[idx]))
            Z_cross.append(self.Z[idx] + t * (self.Z[idx + 1] - self.Z[idx]))

        tol_end = phi_period * 1e-4
        first_shifted = abs(self.phi[0] % phi_period - phi_t)
        last_shifted = abs(self.phi[-1] % phi_period - phi_t)
        first_shifted = min(first_shifted, phi_period - first_shifted)
        last_shifted = min(last_shifted, phi_period - last_shifted)
        if first_shifted < tol_end:
            R_cross.insert(0, float(self.R[0]))
            Z_cross.insert(0, float(self.Z[0]))
        if last_shifted < tol_end and len(self.phi) > 1:
            if not (first_shifted < tol_end and len(R_cross) == 1):
                R_cross.append(float(self.R[-1]))
                Z_cross.append(float(self.Z[-1]))

        if return_first:
            if not R_cross:
                raise ValueError(f"no crossing found at phi={phi_target}")
            return float(R_cross[0]), float(Z_cross[0])
        return np.array(R_cross), np.array(Z_cross)

    def cross_section_area(self, phi_target: float, R_ax: float, Z_ax: float) -> Tuple[float, float]:
        R_c, Z_c = self.intersect(phi_target)
        if len(R_c) < 3:
            return 0.0, R_ax
        theta = np.arctan2(Z_c - Z_ax, R_c - R_ax)
        order = np.argsort(theta)
        R_s, Z_s = R_c[order], Z_c[order]
        area = 0.5 * abs(np.sum(R_s * np.roll(Z_s, -1) - np.roll(R_s, -1) * Z_s))
        return float(area), float(np.mean(R_s))

    def volume(self, R_ax: float, Z_ax: float, Np: int = 2, n_sections: int = 8) -> float:
        phi_period = 2.0 * np.pi / Np
        phis = np.linspace(0.0, phi_period, n_sections, endpoint=False)
        V_sum, n_valid = 0.0, 0
        for phi in phis:
            area, R_cent = self.cross_section_area(phi, R_ax, Z_ax)
            if area > 0.0:
                V_sum += area * R_cent
                n_valid += 1
        if n_valid == 0:
            return 0.0
        return phi_period * V_sum / n_valid

    def r_eff(self, R_ax: float, Z_ax: float, Np: int = 2) -> float:
        V = self.volume(R_ax, Z_ax, Np)
        return float(np.sqrt(max(V, 0.0) / (2.0 * np.pi ** 2 * R_ax)))

    def save(self, path: str) -> None:
        import h5py

        with h5py.File(path, 'w') as f:
            f.create_dataset('R', data=self.R, compression='gzip')
            f.create_dataset('Z', data=self.Z, compression='gzip')
            f.create_dataset('phi', data=self.phi, compression='gzip')
            grp = f.create_group('metadata')
            for k, v in self.metadata.items():
                try:
                    grp.attrs[k] = v
                except Exception:
                    grp.attrs[k] = str(v)

    @classmethod
    def load(cls, path: str) -> 'ToroidalTrajectory':
        import h5py

        with h5py.File(path, 'r') as f:
            R = f['R'][:]
            Z = f['Z'][:]
            phi = f['phi'][:]
            meta = dict(f['metadata'].attrs)
        return cls(R=R, Z=Z, phi=phi, metadata=meta)

    def plot3d(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        X = self.R * np.cos(self.phi)
        Y = self.R * np.sin(self.phi)
        ax.plot(X, Y, self.Z, **kwargs)
        return ax


def trace_toroidal_trajectory(
    R_seed: float,
    Z_seed: float,
    phi_seed: float,
    field_func: Callable[[float, float, float], Tuple[float, float]],
    n_turns: int = 300,
    DPhi: float = 0.05,
    metadata: Optional[dict] = None,
) -> ToroidalTrajectory:
    """Trace a sampled toroidal trajectory using RK4 in phi."""
    phi_end = phi_seed + n_turns * 2.0 * np.pi
    n_steps = int(round((phi_end - phi_seed) / DPhi))

    phis = np.empty(n_steps + 1)
    Rs = np.empty(n_steps + 1)
    Zs = np.empty(n_steps + 1)
    phis[0] = phi_seed
    Rs[0] = R_seed
    Zs[0] = Z_seed

    phi_cur = phi_seed
    R_cur = R_seed
    Z_cur = Z_seed

    for i in range(n_steps):
        h = min(DPhi, phi_end - phi_cur)
        if h <= 0.0:
            phis = phis[:i + 1]
            Rs = Rs[:i + 1]
            Zs = Zs[:i + 1]
            break

        k1R, k1Z = field_func(R_cur, Z_cur, phi_cur)
        k2R, k2Z = field_func(R_cur + h/2*k1R, Z_cur + h/2*k1Z, phi_cur + h/2)
        k3R, k3Z = field_func(R_cur + h/2*k2R, Z_cur + h/2*k2Z, phi_cur + h/2)
        k4R, k4Z = field_func(R_cur + h*k3R, Z_cur + h*k3Z, phi_cur + h)

        R_cur += h / 6.0 * (k1R + 2*k2R + 2*k3R + k4R)
        Z_cur += h / 6.0 * (k1Z + 2*k2Z + 2*k3Z + k4Z)
        phi_cur += h

        phis[i + 1] = phi_cur
        Rs[i + 1] = R_cur
        Zs[i + 1] = Z_cur

    meta = dict(metadata) if metadata else {}
    meta.update({'R_seed': R_seed, 'Z_seed': Z_seed, 'phi_seed': phi_seed, 'n_turns': n_turns, 'DPhi': DPhi})
    return ToroidalTrajectory(R=Rs, Z=Zs, phi=phis, metadata=meta)


__all__ = ["ToroidalTrajectory", "trace_toroidal_trajectory"]
