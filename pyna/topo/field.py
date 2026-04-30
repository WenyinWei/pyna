"""pyna.topo.field — Toroidal vector field data structures.

ToroidalField: pure vector field (BR, BPhi, BZ) on (R, Z) grid.
Equilibrium: B field + J field pair — two separate ToroidalFields.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


MU0 = 4e-7 * np.pi


@dataclass
class ToroidalField:
    """Toroidal vector field on an (R, Z) grid at fixed toroidal angle.

    Pure field representation — holds only (BR, BPhi, BZ) components.
    For equilibrium current, see Equilibrium (B + J pair) or compute_J0().

    Parameters
    ----------
    R_arr : (nR,) — Major radius [m].
    Z_arr : (nZ,) — Vertical coordinate [m].
    BR : (nR, nZ) — Radial component [units].
    BPhi : (nR, nZ) — Toroidal component [units].
    BZ : (nR, nZ) — Vertical component [units].
    phi : float — Toroidal angle [rad].
    label : str or None.
    """

    R_arr: np.ndarray
    Z_arr: np.ndarray
    BR: np.ndarray
    BPhi: np.ndarray
    BZ: np.ndarray
    phi: float = 0.0
    label: Optional[str] = None

    # -- grid --
    @property
    def shape(self) -> Tuple[int, int]:
        return self.BR.shape

    @property
    def nR(self) -> int:
        return self.BR.shape[0]

    @property
    def nZ(self) -> int:
        return self.BR.shape[1]

    # -- magnitude --
    @property
    def B_ref(self) -> float:
        return float(np.sqrt(np.mean(self.BR**2 + self.BPhi**2 + self.BZ**2)))

    @property
    def B_pol(self) -> np.ndarray:
        return np.sqrt(self.BR**2 + self.BZ**2)

    # -- construction --
    @classmethod
    def zero_like(cls, other: "ToroidalField", label: str = "") -> "ToroidalField":
        z = np.zeros_like(other.BR)
        return cls(other.R_arr, other.Z_arr, z, z, z, phi=other.phi, label=label)

    @classmethod
    def from_cache(cls, cache: dict, phi_idx: int = 0, *, label: str = "") -> "ToroidalField":
        """Create from a 3D pyna field-cache dict at given toroidal slice.

        Cache keys: BR, BPhi, BZ, R_grid, Z_grid, Phi_grid (shape nR×nZ×nPhi).
        """
        return cls(
            R_arr=cache["R_grid"], Z_arr=cache["Z_grid"],
            BR=cache["BR"][:, :, phi_idx],
            BPhi=cache["BPhi"][:, :, phi_idx],
            BZ=cache["BZ"][:, :, phi_idx],
            phi=float(cache["Phi_grid"][phi_idx]),
            label=label or f"cache_phi{phi_idx}",
        )

    def downsample(self, skip: int) -> "ToroidalField":
        return ToroidalField(
            R_arr=self.R_arr[::skip], Z_arr=self.Z_arr[::skip],
            BR=self.BR[::skip, ::skip], BPhi=self.BPhi[::skip, ::skip],
            BZ=self.BZ[::skip, ::skip], phi=self.phi, label=self.label,
        )

    def compute_J(self) -> "ToroidalField":
        """Compute J = curl(B)/mu0 via finite differences. Returns new ToroidalField."""
        dR = self.R_arr[1] - self.R_arr[0]
        dZ = self.Z_arr[1] - self.Z_arr[0]
        return ToroidalField(
            R_arr=self.R_arr, Z_arr=self.Z_arr,
            BR=-np.gradient(self.BPhi, dZ, axis=1, edge_order=2) / MU0,
            BPhi=(np.gradient(self.BR, dZ, axis=1, edge_order=2)
                  - np.gradient(self.BZ, dR, axis=0, edge_order=2)) / MU0,
            BZ=(self.BPhi / self.R_arr[:, None]
                + np.gradient(self.BPhi, dR, axis=0, edge_order=2)) / MU0,
            phi=self.phi,
            label=f"J({self.label})" if self.label else "J",
        )

    # -- arithmetic --
    def __add__(self, other: "ToroidalField") -> "ToroidalField":
        return ToroidalField(
            R_arr=self.R_arr, Z_arr=self.Z_arr,
            BR=self.BR + other.BR, BPhi=self.BPhi + other.BPhi,
            BZ=self.BZ + other.BZ, phi=self.phi,
        )

    def __repr__(self) -> str:
        lbl = f"'{self.label}'" if self.label else ""
        return f"ToroidalField({self.nR}x{self.nZ}, B_ref={self.B_ref:.3f}{lbl})"


@dataclass
class Equilibrium:
    """MHD equilibrium: B field + J field on the same (R, Z) grid.

    J0 can be None (vacuum).  Use compute_J() on B0 to populate it.
    """

    B0: ToroidalField
    J0: Optional[ToroidalField] = None

    @property
    def R_arr(self): return self.B0.R_arr
    @property
    def Z_arr(self): return self.B0.Z_arr
    @property
    def phi(self):   return self.B0.phi
    @property
    def nR(self):    return self.B0.nR
    @property
    def nZ(self):    return self.B0.nZ

    def _zero_J(self):
        z = np.zeros_like(self.B0.BR)
        return ToroidalField(self.R_arr, self.Z_arr, z, z, z)

    def get_J0(self) -> ToroidalField:
        """Return J0, or zero field if not set."""
        return self.J0 if self.J0 is not None else self._zero_J()

    @classmethod
    def from_cache(cls, cache: dict, phi_idx: int = 0, *,
                   label: str = "", compute_J0: bool = False) -> "Equilibrium":
        """Create from 3D field cache.  Optionally compute J0 via curl(B)/mu0."""
        B0 = ToroidalField.from_cache(cache, phi_idx, label=label)
        J0 = B0.compute_J() if compute_J0 else None
        return cls(B0=B0, J0=J0)

    def __repr__(self) -> str:
        lbl = f"'{self.B0.label}'" if self.B0.label else ""
        j = " +J0" if self.J0 is not None else " (vacuum)"
        return f"Equilibrium({self.nR}x{self.nZ}{j}{lbl})"
