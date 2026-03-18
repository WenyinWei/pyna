"""ITER IMAS (Integrated Modelling & Analysis Suite) data format compatibility.

Provides converters between pyna internal data structures and IMAS IDS
(Interface Data Structure) conventions.

Supported IDS:
- equilibrium IDS (psi, q, flux surfaces, B field)
- mhd_linear IDS (perturbation spectra, mode amplitudes)
- coils_non_axisymmetric IDS (external coil geometry and currents)
- poincare_mapping IDS (Poincaré map data)

References: IMAS data dictionary https://imas.iter.org/
"""
from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ndarray_to_list(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


# ---------------------------------------------------------------------------
# equilibrium IDS
# ---------------------------------------------------------------------------

@dataclass
class IMASEquilibriumIDS:
    """Simplified equilibrium IDS compatible with IMAS conventions.

    Field names follow IMAS equilibrium IDS naming:
    https://imas.iter.org/mds/IMAS/latest/html/equilibrium.html
    """
    # time_slice.global_quantities
    ip: float = 0.0       # plasma current (A)
    b0: float = 0.0       # vacuum toroidal field at R0 (T)
    r0: float = 0.0       # reference major radius (m)

    # time_slice.profiles_1d (on psi grid)
    psi: np.ndarray = field(default_factory=lambda: np.array([]))
    q: np.ndarray = field(default_factory=lambda: np.array([]))

    # time_slice.boundary
    r_boundary: np.ndarray = field(default_factory=lambda: np.array([]))
    z_boundary: np.ndarray = field(default_factory=lambda: np.array([]))

    # time_slice.profiles_2d
    r_2d: Optional[np.ndarray] = None    # (nR, nZ) R grid
    z_2d: Optional[np.ndarray] = None    # (nR, nZ) Z grid
    psi_2d: Optional[np.ndarray] = None  # (nR, nZ) psi map

    def to_dict(self):
        """Convert to JSON-serializable dict."""
        return {
            'ids_type': 'equilibrium',
            'ip': float(self.ip),
            'b0': float(self.b0),
            'r0': float(self.r0),
            'psi': _ndarray_to_list(self.psi),
            'q': _ndarray_to_list(self.q),
            'r_boundary': _ndarray_to_list(self.r_boundary),
            'z_boundary': _ndarray_to_list(self.z_boundary),
            'r_2d': _ndarray_to_list(self.r_2d),
            'z_2d': _ndarray_to_list(self.z_2d),
            'psi_2d': _ndarray_to_list(self.psi_2d),
        }

    def to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d):
        """Reconstruct from a dict (as returned by to_dict)."""
        def _arr(v):
            return np.array(v) if v is not None else None

        return cls(
            ip=d.get('ip', 0.0),
            b0=d.get('b0', 0.0),
            r0=d.get('r0', 0.0),
            psi=np.array(d.get('psi', [])),
            q=np.array(d.get('q', [])),
            r_boundary=np.array(d.get('r_boundary', [])),
            z_boundary=np.array(d.get('z_boundary', [])),
            r_2d=_arr(d.get('r_2d')),
            z_2d=_arr(d.get('z_2d')),
            psi_2d=_arr(d.get('psi_2d')),
        )

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)

    @classmethod
    def from_stellarator(cls, stellarator, n_psi: int = 64, n_theta: int = 128):
        """Build from a StellaratorSimple equilibrium.

        Parameters
        ----------
        stellarator : StellaratorSimple
        n_psi : int
            Number of flux surfaces for 1-D profiles.
        n_theta : int
            Number of theta points for boundary.
        """
        psi_arr = np.linspace(0.01, 1.0, n_psi)
        q_arr = stellarator.q_of_psi(psi_arr)

        # LCFS boundary at psi=1
        theta_arr = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        r_bnd = stellarator.R0 + stellarator.r0 * np.cos(theta_arr)
        z_bnd = stellarator.r0 * np.sin(theta_arr)

        return cls(
            ip=0.0,
            b0=stellarator.B0,
            r0=stellarator.R0,
            psi=psi_arr,
            q=q_arr,
            r_boundary=r_bnd,
            z_boundary=z_bnd,
        )


# ---------------------------------------------------------------------------
# coils_non_axisymmetric IDS
# ---------------------------------------------------------------------------

@dataclass
class IMASCoilsNonAxisymmetric:
    """Simplified coils_non_axisymmetric IDS.

    Each coil is a list of 3D conductor segments.
    """
    coil_names: List[str] = field(default_factory=list)
    coil_current: List[float] = field(default_factory=list)   # (A)
    coil_conductor: List[np.ndarray] = field(default_factory=list)

    def to_dict(self):
        return {
            'ids_type': 'coils_non_axisymmetric',
            'coil_names': self.coil_names,
            'coil_current': [float(I) for I in self.coil_current],
            'coil_conductor': [
                pts.tolist() if isinstance(pts, np.ndarray) else pts
                for pts in self.coil_conductor
            ],
        }

    def to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d):
        return cls(
            coil_names=d.get('coil_names', []),
            coil_current=d.get('coil_current', []),
            coil_conductor=[np.array(pts) for pts in d.get('coil_conductor', [])],
        )

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)

    @classmethod
    def from_coil_set(cls, coil_set):
        """Build from a pyna CoilSet."""
        names = [f'coil_{k:03d}' for k in range(len(coil_set))]
        currents = [float(I) for _, I in coil_set.coils]
        conductors = [pts.copy() for pts, _ in coil_set.coils]
        return cls(coil_names=names, coil_current=currents, coil_conductor=conductors)


# ---------------------------------------------------------------------------
# poincare_mapping IDS
# ---------------------------------------------------------------------------

@dataclass
class IMASPoincareMapping:
    """Simplified poincare_mapping IDS."""
    phi_sections: List[float] = field(default_factory=list)
    r_crossings: List[List[float]] = field(default_factory=list)
    z_crossings: List[List[float]] = field(default_factory=list)

    def to_dict(self):
        return {
            'ids_type': 'poincare_mapping',
            'phi_sections': list(self.phi_sections),
            'r_crossings': [
                r.tolist() if isinstance(r, np.ndarray) else list(r)
                for r in self.r_crossings
            ],
            'z_crossings': [
                z.tolist() if isinstance(z, np.ndarray) else list(z)
                for z in self.z_crossings
            ],
        }

    def to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d):
        return cls(
            phi_sections=d.get('phi_sections', []),
            r_crossings=[list(r) for r in d.get('r_crossings', [])],
            z_crossings=[list(z) for z in d.get('z_crossings', [])],
        )

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)

    @classmethod
    def from_poincare_data(cls, R_arr, Z_arr, phi_section=0.0):
        """Build from arrays of (R, Z) crossings at a single phi section.

        Parameters
        ----------
        R_arr, Z_arr : array-like, shape (N,) or (n_lines, n_crossings)
            Poincaré crossing coordinates.
        phi_section : float
            Toroidal angle of the Poincaré section (rad).
        """
        R_arr = np.asarray(R_arr)
        Z_arr = np.asarray(Z_arr)
        return cls(
            phi_sections=[float(phi_section)],
            r_crossings=[R_arr.ravel().tolist()],
            z_crossings=[Z_arr.ravel().tolist()],
        )


# ---------------------------------------------------------------------------
# mhd_linear IDS (perturbation spectra)
# ---------------------------------------------------------------------------

@dataclass
class IMASMHDLinearIDS:
    """Simplified mhd_linear IDS for perturbation spectra."""
    # Mode spectrum
    m_list: List[int] = field(default_factory=list)
    n_list: List[int] = field(default_factory=list)
    # Complex amplitudes at resonant surfaces
    b_mn_amplitude: List[float] = field(default_factory=list)
    b_mn_phase: List[float] = field(default_factory=list)
    # Resonant surface locations
    psi_resonant: List[float] = field(default_factory=list)

    def to_dict(self):
        return {
            'ids_type': 'mhd_linear',
            'm_list': self.m_list,
            'n_list': self.n_list,
            'b_mn_amplitude': self.b_mn_amplitude,
            'b_mn_phase': self.b_mn_phase,
            'psi_resonant': self.psi_resonant,
        }

    def to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k != 'ids_type'})

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)
