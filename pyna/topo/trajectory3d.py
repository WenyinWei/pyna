"""pyna.topo.trajectory3d - sampled toroidal trajectory representation.

This module stores finite sampled geometry in a toroidal coordinate system.
These objects are *not* assumed to be invariant sets just because they were
numerically traced.  Exact periodicity / invariance belongs to higher-level
objects such as :class:`pyna.topo.invariants.Cycle`.

Design principle
----------------
Represent a toroidal curve once in 3D, then derive 2D section cuts by
intersection.  Do not recompute unrelated copies at each phi section.

Naming
------
``Trajectory3DToroidal`` remains the concrete toroidal sampled-trajectory
container.  ``Trajectory3D`` is now only a compatibility alias.
"""

import numpy as np
import h5py
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

from pyna.topo._base import Trajectory


# ---------------------------------------------------------------------------
# Concrete toroidal sampled trajectory
# ---------------------------------------------------------------------------

@dataclass
class Trajectory3DToroidal(Trajectory):
    """A 3D trajectory in a toroidal vector field, stored as (R, Z, phi) arrays.

    Generic for any toroidal dynamical system (magnetic field lines,
    guiding-centre drift orbits, etc.).  All
    cross-sections are derived by intersection of this single 3D object —
    never recomputed independently per section.

    Attributes
    ----------
    R   : ndarray (N,) – major radius [arbitrary units, typically m]
    Z   : ndarray (N,) – vertical coordinate [same units]
    phi : ndarray (N,) – toroidal angle [rad], monotonically increasing
    metadata : dict – provenance info (seed, n_turns, periodicity, ...)
    """
    R: np.ndarray
    Z: np.ndarray
    phi: np.ndarray
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Cross-section / intersection
    # ------------------------------------------------------------------

    def intersect(self, phi_target: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (R, Z) points where the trajectory crosses phi_target.

        The toroidal period ``2*pi / Np`` is taken from ``metadata['Np']``
        (default Np=1, period = 2*pi).  ``phi_target`` is automatically
        reduced modulo the period before searching.

        Uses linear interpolation for sub-step accuracy.

        Parameters
        ----------
        phi_target : float
            Target toroidal angle [rad].

        Returns
        -------
        R_cross, Z_cross : ndarray
            Crossing coordinates (may be empty if no crossings found).
        """
        Np = self.metadata.get('Np', 1)
        phi_period = 2.0 * np.pi / Np
        phi_t = phi_target % phi_period

        # Robust crossing detection: phi is monotonically increasing.
        # A crossing of phi_target (mod phi_period) occurs whenever
        #   floor((phi[i]   - phi_t) / phi_period)
        #   < floor((phi[i+1] - phi_t) / phi_period)
        # This is exact and has no edge cases at phi_t = 0 or phi_period.
        shifted = self.phi - phi_t
        floor_i  = np.floor(shifted[:-1] / phi_period)
        floor_i1 = np.floor(shifted[1:]  / phi_period)
        cross_idx = np.where(floor_i1 > floor_i)[0]

        R_cross, Z_cross = [], []
        for idx in cross_idx:
            # Interpolation fraction within the step
            phi_cross = phi_t + (floor_i[idx] + 1.0) * phi_period
            h = self.phi[idx + 1] - self.phi[idx]
            if h < 1e-30:
                continue
            t = (phi_cross - self.phi[idx]) / h
            if not (0.0 <= t <= 1.0):
                continue
            R_cross.append(self.R[idx] + t * (self.R[idx + 1] - self.R[idx]))
            Z_cross.append(self.Z[idx] + t * (self.Z[idx + 1] - self.Z[idx]))

        # ── Endpoint fix ──────────────────────────────────────────────────────
        # When the trajectory starts very close to phi_target (e.g. a periodic
        # orbit stored from phi=0 to phi=N*2π), the first and last points lie
        # right on the section plane but the step-crossing detector above finds
        # no crossing because phi[-1] forms no full upward step.  In this case
        # we include phi[0] / phi[-1] directly (using the raw endpoint value).
        tol_end = phi_period * 1e-4   # ~0.06 mrad tolerance
        first_shifted = abs(self.phi[0]  % phi_period - phi_t)
        last_shifted  = abs(self.phi[-1] % phi_period - phi_t)
        # Normalise to be within [0, phi_period/2]
        first_shifted = min(first_shifted, phi_period - first_shifted)
        last_shifted  = min(last_shifted,  phi_period - last_shifted)
        if first_shifted < tol_end:
            R_cross.insert(0, float(self.R[0]))
            Z_cross.insert(0, float(self.Z[0]))
        if last_shifted < tol_end and len(self.phi) > 1:
            # Only add if distinct from the first point
            if not (first_shifted < tol_end and len(R_cross) == 1):
                R_cross.append(float(self.R[-1]))
                Z_cross.append(float(self.Z[-1]))

        return np.array(R_cross), np.array(Z_cross)

    # ------------------------------------------------------------------
    # Geometric derived quantities
    # ------------------------------------------------------------------

    def cross_section_area(
        self, phi_target: float, R_ax: float, Z_ax: float
    ) -> Tuple[float, float]:
        """Shoelace area and centroid-R of the cross-section at ``phi_target``.

        Parameters
        ----------
        phi_target : float – toroidal angle [rad]
        R_ax, Z_ax : float – magnetic / geometric axis position (used to sort
            crossing points by poloidal angle)

        Returns
        -------
        area_m2 : float
        R_centroid : float
        """
        R_c, Z_c = self.intersect(phi_target)
        if len(R_c) < 3:
            return 0.0, R_ax
        theta = np.arctan2(Z_c - Z_ax, R_c - R_ax)
        order = np.argsort(theta)
        R_s, Z_s = R_c[order], Z_c[order]
        area = 0.5 * abs(
            np.sum(R_s * np.roll(Z_s, -1) - np.roll(R_s, -1) * Z_s)
        )
        return float(area), float(np.mean(R_s))

    def volume(
        self,
        R_ax: float,
        Z_ax: float,
        Np: int = 2,
        n_sections: int = 8,
    ) -> float:
        """Enclosed toroidal volume via multi-section shoelace integration.

        ``V ≈ (2π/Np) · mean_k[ A(φ_k) · R_centroid(φ_k) ]``

        Parameters
        ----------
        R_ax, Z_ax : float – axis position
        Np : int – field periodicity (default 2)
        n_sections : int – number of phi sections to average over

        Returns
        -------
        V : float [units³]
        """
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
        """Effective minor radius: ``r_eff = sqrt(V / (2π² R_ax))``."""
        V = self.volume(R_ax, Z_ax, Np)
        return float(np.sqrt(max(V, 0.0) / (2.0 * np.pi ** 2 * R_ax)))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save trajectory to an HDF5 file."""
        with h5py.File(path, 'w') as f:
            f.create_dataset('R',   data=self.R,   compression='gzip')
            f.create_dataset('Z',   data=self.Z,   compression='gzip')
            f.create_dataset('phi', data=self.phi, compression='gzip')
            grp = f.create_group('metadata')
            for k, v in self.metadata.items():
                try:
                    grp.attrs[k] = v
                except Exception:
                    grp.attrs[k] = str(v)

    @classmethod
    def load(cls, path: str) -> 'Trajectory3DToroidal':
        """Load trajectory from an HDF5 file."""
        with h5py.File(path, 'r') as f:
            R   = f['R'][:]
            Z   = f['Z'][:]
            phi = f['phi'][:]
            meta = dict(f['metadata'].attrs)
        return cls(R=R, Z=Z, phi=phi, metadata=meta)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot3d(self, ax=None, **kwargs):
        """Plot the trajectory in 3-D Cartesian space.

        Parameters
        ----------
        ax : mpl_toolkits.mplot3d.Axes3D, optional
        **kwargs : forwarded to ``ax.plot``

        Returns
        -------
        ax : Axes3D
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        X = self.R * np.cos(self.phi)
        Y = self.R * np.sin(self.phi)
        ax.plot(X, Y, self.Z, **kwargs)
        return ax


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def trace_toroidal_trajectory(
    R_seed: float,
    Z_seed: float,
    phi_seed: float,
    field_func: Callable[[float, float, float], Tuple[float, float]],
    n_turns: int = 300,
    DPhi: float = 0.05,
    metadata: Optional[dict] = None,
) -> Trajectory3DToroidal:
    """Trace a 3-D trajectory in a toroidal vector field using RK4.

    This is the generic factory for :class:`Trajectory3DToroidal`.  Pass a
    ``field_func(R, Z, phi) -> (dR/dphi, dZ/dphi)`` that encodes the
    right-hand side of the field-line (or any toroidal flow) ODE.

    For magnetic field lines from a ``FieldlineTracer``, use the
    higher-level helper ``trace_toroidal_trajectory_from_tracer`` which
    constructs ``field_func`` automatically.

    Parameters
    ----------
    R_seed, Z_seed, phi_seed : float
        Initial position [same units as field_func output].
    field_func : callable
        ``(R, Z, phi) -> (dR/dphi, dZ/dphi)``
    n_turns : int
        Number of toroidal turns to integrate.
    DPhi : float
        RK4 step size [rad].
    metadata : dict, optional
        Additional provenance info stored in the trajectory.

    Returns
    -------
    Trajectory3DToroidal
    """
    phi_end = phi_seed + n_turns * 2.0 * np.pi
    n_steps = int(round((phi_end - phi_seed) / DPhi))

    phis = np.empty(n_steps + 1)
    Rs   = np.empty(n_steps + 1)
    Zs   = np.empty(n_steps + 1)

    phis[0] = phi_seed
    Rs[0]   = R_seed
    Zs[0]   = Z_seed

    phi_cur = phi_seed
    R_cur   = R_seed
    Z_cur   = Z_seed

    for i in range(n_steps):
        h = min(DPhi, phi_end - phi_cur)
        if h <= 0.0:
            phis = phis[:i + 1]
            Rs   = Rs[:i + 1]
            Zs   = Zs[:i + 1]
            break

        # RK4
        k1R, k1Z = field_func(R_cur,             Z_cur,             phi_cur)
        k2R, k2Z = field_func(R_cur + h/2*k1R,   Z_cur + h/2*k1Z,  phi_cur + h/2)
        k3R, k3Z = field_func(R_cur + h/2*k2R,   Z_cur + h/2*k2Z,  phi_cur + h/2)
        k4R, k4Z = field_func(R_cur + h*k3R,      Z_cur + h*k3Z,    phi_cur + h)

        R_cur += h / 6.0 * (k1R + 2*k2R + 2*k3R + k4R)
        Z_cur += h / 6.0 * (k1Z + 2*k2Z + 2*k3Z + k4Z)
        phi_cur += h

        phis[i + 1] = phi_cur
        Rs[i + 1]   = R_cur
        Zs[i + 1]   = Z_cur

    meta = dict(metadata) if metadata else {}
    meta.update({
        'R_seed': R_seed, 'Z_seed': Z_seed, 'phi_seed': phi_seed,
        'n_turns': n_turns, 'DPhi': DPhi,
    })
    return Trajectory3DToroidal(R=Rs, Z=Zs, phi=phis, metadata=meta)


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------
#: ``Trajectory3D`` is retained as a compatibility alias.  The intermediate
#: generic 3-D subclass layer has been removed; use ``Trajectory3DToroidal``
#: directly for toroidal sampled curves.
Trajectory3D = Trajectory3DToroidal
