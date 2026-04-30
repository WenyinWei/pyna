"""pyna.topo.field — Toroidal vector field data structures.

Provides clean dataclasses for magnetic fields and perturbations
on (R, Z) grids at fixed toroidal angle, plus convenience methods
for slicing, arithmetic, and Ampere's law.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


MU0 = 4e-7 * np.pi


@dataclass
class ToroidalField:
    """Equilibrium magnetic field on an (R, Z) grid at fixed toroidal angle.

    Carries the field components (BR, BPhi, BZ), grid coordinates,
    and optionally the equilibrium current density (J0_R, J0_Phi, J0_Z).

    Parameters
    ----------
    R_arr : (nR,) array
        Major radius coordinates [m].
    Z_arr : (nZ,) array
        Vertical coordinates [m].
    BR : (nR, nZ) array
        Radial magnetic field component [T].
    BPhi : (nR, nZ) array
        Toroidal magnetic field component [T].
    BZ : (nR, nZ) array
        Vertical magnetic field component [T].
    phi : float
        Toroidal angle of this cross-section [rad].
    J0_R, J0_Phi, J0_Z : (nR, nZ) array or None
        Equilibrium current density components [A/m^2].
    label : str or None
        Human-readable identifier.
    """

    R_arr: np.ndarray
    Z_arr: np.ndarray
    BR: np.ndarray
    BPhi: np.ndarray
    BZ: np.ndarray
    phi: float = 0.0
    J0_R: Optional[np.ndarray] = None
    J0_Phi: Optional[np.ndarray] = None
    J0_Z: Optional[np.ndarray] = None
    label: Optional[str] = None

    @property
    def shape(self) -> Tuple[int, int]:
        """Grid shape (nR, nZ)."""
        return self.BR.shape

    @property
    def nR(self) -> int:
        return self.BR.shape[0]

    @property
    def nZ(self) -> int:
        return self.BR.shape[1]

    @property
    def B_ref(self) -> float:
        """RMS field magnitude [T]."""
        return float(np.sqrt(np.mean(self.BR**2 + self.BPhi**2 + self.BZ**2)))

    @property
    def B_pol(self) -> np.ndarray:
        """Poloidal field magnitude (nR, nZ) [T]."""
        return np.sqrt(self.BR**2 + self.BZ**2)

    @property
    def has_current(self) -> bool:
        """True if any J0 component is provided."""
        return any(x is not None for x in (self.J0_R, self.J0_Phi, self.J0_Z))

    def _zero(self) -> np.ndarray:
        return np.zeros_like(self.BR)

    def get_J0_R(self) -> np.ndarray:
        return self._zero() if self.J0_R is None else self.J0_R

    def get_J0_Phi(self) -> np.ndarray:
        return self._zero() if self.J0_Phi is None else self.J0_Phi

    def get_J0_Z(self) -> np.ndarray:
        return self._zero() if self.J0_Z is None else self.J0_Z

    def downsample(self, skip: int) -> "ToroidalField":
        """Return a new field with grid downsampled by factor `skip`."""
        return ToroidalField(
            R_arr=self.R_arr[::skip],
            Z_arr=self.Z_arr[::skip],
            BR=self.BR[::skip, ::skip],
            BPhi=self.BPhi[::skip, ::skip],
            BZ=self.BZ[::skip, ::skip],
            phi=self.phi,
            J0_R=self.J0_R[::skip, ::skip] if self.has_current else None,
            J0_Phi=self.J0_Phi[::skip, ::skip] if self.J0_Phi is not None else None,
            J0_Z=self.J0_Z[::skip, ::skip] if self.J0_Z is not None else None,
            label=self.label,
        )

    def compute_J0(self) -> "ToroidalField":
        """Compute and attach J0 = curl(B) / mu0 via finite differences.

        Returns a new ToroidalField with J0_* populated.
        """
        dR = self.R_arr[1] - self.R_arr[0]
        dZ = self.Z_arr[1] - self.Z_arr[0]
        J0_R = -np.gradient(self.BPhi, dZ, axis=1, edge_order=2) / MU0
        J0_Phi = (np.gradient(self.BR, dZ, axis=1, edge_order=2)
                  - np.gradient(self.BZ, dR, axis=0, edge_order=2)) / MU0
        J0_Z = (self.BPhi / self.R_arr[:, None]
                + np.gradient(self.BPhi, dR, axis=0, edge_order=2)) / MU0
        return ToroidalField(
            R_arr=self.R_arr, Z_arr=self.Z_arr,
            BR=self.BR, BPhi=self.BPhi, BZ=self.BZ,
            phi=self.phi,
            J0_R=J0_R, J0_Phi=J0_Phi, J0_Z=J0_Z,
            label=self.label,
        )

    def __repr__(self) -> str:
        lbl = f"'{self.label}'" if self.label else ""
        cur = " +J0" if self.has_current else ""
        return (f"ToroidalField({self.nR}x{self.nZ}, "
                f"B_ref={self.B_ref:.3f}T{cur}{lbl})")


@dataclass
class PerturbedField:
    """Perturbation to a toroidal field on the same (R, Z) grid.

    Represents delta_B_plasma or delta_B_ext (external coil response).

    Parameters
    ----------
    R_arr, Z_arr : 1D arrays (must match the parent ToroidalField).
    dB_R, dB_Phi, dB_Z : (nR, nZ) arrays — perturbation field [T].
    label : str or None
    """

    R_arr: np.ndarray
    Z_arr: np.ndarray
    dB_R: np.ndarray
    dB_Phi: np.ndarray
    dB_Z: np.ndarray
    label: Optional[str] = None

    @property
    def shape(self) -> Tuple[int, int]:
        return self.dB_R.shape

    def _zero(self) -> np.ndarray:
        return np.zeros_like(self.dB_R)

    @classmethod
    def zero_like(cls, field: ToroidalField, label: str = "") -> "PerturbedField":
        """Create a zero perturbation field matching a ToroidalField's grid."""
        z = np.zeros_like(field.BR)
        return cls(field.R_arr, field.Z_arr, z, z, z, label=label)

    def __repr__(self) -> str:
        lbl = f"'{self.label}'" if self.label else ""
        return f"PerturbedField({self.shape[0]}x{self.shape[1]}{lbl})"
