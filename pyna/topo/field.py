"""pyna.topo.field — Toroidal vector field data structures.

Provides a clean dataclass for toroidal magnetic fields and perturbations
on (R, Z) grids at fixed toroidal angle, plus convenience methods
for slicing, arithmetic, Ampere's law, and current computation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


MU0 = 4e-7 * np.pi


@dataclass
class ToroidalField:
    """Toroidal vector field on an (R, Z) grid at fixed toroidal angle.

    Represents equilibrium fields, perturbations, or their sum.
    J0 is optional — perturbations naturally have J0=None.

    Parameters
    ----------
    R_arr : (nR,) array — Major radius [m].
    Z_arr : (nZ,) array — Vertical coordinate [m].
    BR : (nR, nZ) array — Radial field [T].
    BPhi : (nR, nZ) array — Toroidal field [T].
    BZ : (nR, nZ) array — Vertical field [T].
    phi : float — Toroidal angle [rad].
    J0_R, J0_Phi, J0_Z : (nR, nZ) array or None — Equilibrium current [A/m²].
    label : str or None.
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

    # -- grid helpers --
    @property
    def shape(self) -> Tuple[int, int]:
        return self.BR.shape

    @property
    def nR(self) -> int:
        return self.BR.shape[0]

    @property
    def nZ(self) -> int:
        return self.BR.shape[1]

    # -- field magnitude --
    @property
    def B_ref(self) -> float:
        return float(np.sqrt(np.mean(self.BR**2 + self.BPhi**2 + self.BZ**2)))

    @property
    def B_pol(self) -> np.ndarray:
        return np.sqrt(self.BR**2 + self.BZ**2)

    # -- current helpers --
    @property
    def has_current(self) -> bool:
        return any(x is not None for x in (self.J0_R, self.J0_Phi, self.J0_Z))

    def _zero(self) -> np.ndarray:
        return np.zeros_like(self.BR)

    def get_J0_R(self) -> np.ndarray:
        return self._zero() if self.J0_R is None else self.J0_R

    def get_J0_Phi(self) -> np.ndarray:
        return self._zero() if self.J0_Phi is None else self.J0_Phi

    def get_J0_Z(self) -> np.ndarray:
        return self._zero() if self.J0_Z is None else self.J0_Z

    # -- construction --
    @classmethod
    def zero_like(cls, other: "ToroidalField", label: str = "") -> "ToroidalField":
        """Create a zero field on the same grid."""
        z = np.zeros_like(other.BR)
        return cls(other.R_arr, other.Z_arr, z, z, z, phi=other.phi, label=label)

    def downsample(self, skip: int) -> "ToroidalField":
        """Return field on a coarser grid (every `skip`-th point)."""
        return ToroidalField(
            R_arr=self.R_arr[::skip], Z_arr=self.Z_arr[::skip],
            BR=self.BR[::skip, ::skip], BPhi=self.BPhi[::skip, ::skip],
            BZ=self.BZ[::skip, ::skip], phi=self.phi,
            J0_R=self.J0_R[::skip, ::skip] if self.J0_R is not None else None,
            J0_Phi=self.J0_Phi[::skip, ::skip] if self.J0_Phi is not None else None,
            J0_Z=self.J0_Z[::skip, ::skip] if self.J0_Z is not None else None,
            label=self.label,
        )

    def compute_J0(self) -> "ToroidalField":
        """Compute J0 = curl(B)/mu0 and return new field with J0 attached."""
        dR = self.R_arr[1] - self.R_arr[0]
        dZ = self.Z_arr[1] - self.Z_arr[0]
        return ToroidalField(
            R_arr=self.R_arr, Z_arr=self.Z_arr,
            BR=self.BR, BPhi=self.BPhi, BZ=self.BZ, phi=self.phi,
            J0_R=-np.gradient(self.BPhi, dZ, axis=1, edge_order=2) / MU0,
            J0_Phi=(np.gradient(self.BR, dZ, axis=1, edge_order=2)
                    - np.gradient(self.BZ, dR, axis=0, edge_order=2)) / MU0,
            J0_Z=(self.BPhi / self.R_arr[:, None]
                  + np.gradient(self.BPhi, dR, axis=0, edge_order=2)) / MU0,
            label=self.label,
        )

    @classmethod
    def from_cache(cls, cache: dict, phi_idx: int = 0, *, label: str = "") -> "ToroidalField":
        """Create from a 3D pyna field-cache dict at a given toroidal slice.

        The cache dict must have keys: BR, BPhi, BZ, R_grid, Z_grid, Phi_grid.
        BR/BPhi/BZ have shape (nR, nZ, nPhi).
        """
        return cls(
            R_arr=cache["R_grid"], Z_arr=cache["Z_grid"],
            BR=cache["BR"][:, :, phi_idx],
            BPhi=cache["BPhi"][:, :, phi_idx],
            BZ=cache["BZ"][:, :, phi_idx],
            phi=float(cache["Phi_grid"][phi_idx]),
            label=label or f"cache_phi{phi_idx}",
        )

    # -- arithmetic --
    def __add__(self, other: "ToroidalField") -> "ToroidalField":
        """Add two fields (grids must match)."""
        return ToroidalField(
            R_arr=self.R_arr, Z_arr=self.Z_arr,
            BR=self.BR + other.BR, BPhi=self.BPhi + other.BPhi,
            BZ=self.BZ + other.BZ, phi=self.phi,
            J0_R=(self.J0_R + other.J0_R) if (self.J0_R is not None and other.J0_R is not None)
            else (self.J0_R or other.J0_R),
            J0_Phi=(self.J0_Phi + other.J0_Phi) if (self.J0_Phi is not None and other.J0_Phi is not None)
            else (self.J0_Phi or other.J0_Phi),
            J0_Z=(self.J0_Z + other.J0_Z) if (self.J0_Z is not None and other.J0_Z is not None)
            else (self.J0_Z or other.J0_Z),
        )

    def __repr__(self) -> str:
        lbl = f"'{self.label}'" if self.label else ""
        cur = " +J0" if self.has_current else ""
        return (f"ToroidalField({self.nR}x{self.nZ}, "
                f"B_ref={self.B_ref:.3f}T{cur}{lbl})")
