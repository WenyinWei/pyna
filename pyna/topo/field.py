"""pyna.topo.field — Toroidal vector field and equilibrium data structures.

Design principles
-----------------
ToroidalField      — pure vector field (geometry + 3 components).
                    Not magnetic-specific; stores any (BR, BPhi, BZ).
Equilibrium        — slot-based MHD equilibrium.  B0 is mandatory;
                    everything else (J, p, u, ...) is optional and
                    compositionally attached.
EquilibriumLike    — Protocol: minimal interface for interop with
                    EFIT, LIUQE, TGYRO, SD1D, … via adapters.

Physics functions (standalone, not methods on ToroidalField):
    compute_J_by_curl(B)   → J = curl(B)/mu0
    compute_B_pol(B)       → |(BR, BZ)|

Interop adapters live in pyna.io.* (future):
    pyna.io.efit      — read/write GEQDSK
    pyna.io.liuqe     — LIUQE output
    pyna.io.tgyro     — TGYRO input profiles
    ...

Example usage
-------------
    from pyna.topo.field import ToroidalField, Equilibrium, compute_J_by_curl

    B0 = ToroidalField(R, Z, BR, BPhi, BZ, label="EFIT_B0")
    J0 = compute_J_by_curl(B0)

    eq = Equilibrium(B0=B0, J_total=J0, p_total=p_arr)
    eq = Equilibrium(B0=B0, J_bootstrap=J_bs, J_ecrh=J_ec, ...)

    # Protocol interop
    def analyze(equilibrium: EquilibriumLike): ...
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Optional, Protocol, Tuple, runtime_checkable

import numpy as np


MU0 = 4e-7 * np.pi

# ===========================================================================
# ToroidalField — pure vector field
# ===========================================================================

@dataclass
class ToroidalField:
    """Generic toroidal vector field on an (R, Z) grid at fixed phi.

    Not magnetic-specific.  Use for B, J, u, dB, … — any 3-component
    vector on a poloidal cross-section.

    Parameters
    ----------
    R_arr : (nR,) — Major radius [m].
    Z_arr : (nZ,) — Vertical coordinate [m].
    BR : (nR, nZ) — "R" component.
    BPhi : (nR, nZ) — Toroidal component.
    BZ : (nR, nZ) — "Z" component.
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
    def abs(self) -> np.ndarray:
        """Point-wise magnitude sqrt(BR² + BPhi² + BZ²)."""
        return np.sqrt(self.BR**2 + self.BPhi**2 + self.BZ**2)

    @property
    def rms(self) -> float:
        """RMS magnitude over the grid."""
        return float(np.sqrt(np.mean(self.BR**2 + self.BPhi**2 + self.BZ**2)))

    @property
    def poloidal_abs(self) -> np.ndarray:
        """Poloidal magnitude sqrt(BR² + BZ²)."""
        return np.sqrt(self.BR**2 + self.BZ**2)

    # -- construction --
    @classmethod
    def zero_like(cls, other: "ToroidalField", label: str = "") -> "ToroidalField":
        z = np.zeros_like(other.BR)
        return cls(other.R_arr, other.Z_arr, z, z, z, phi=other.phi, label=label)

    @classmethod
    def from_cache(cls, cache: dict, phi_idx: int = 0, *, label: str = "") -> "ToroidalField":
        """Slice a 3D field-cache dict at given toroidal index.

        Cache keys: BR, BPhi, BZ, R_grid, Z_grid, Phi_grid (nR×nZ×nPhi).
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

    def __add__(self, other: "ToroidalField") -> "ToroidalField":
        return ToroidalField(
            R_arr=self.R_arr, Z_arr=self.Z_arr,
            BR=self.BR + other.BR, BPhi=self.BPhi + other.BPhi,
            BZ=self.BZ + other.BZ, phi=self.phi,
        )

    def __repr__(self) -> str:
        lbl = f"'{self.label}'" if self.label else ""
        return f"ToroidalField({self.nR}x{self.nZ}, rms={self.rms:.3g}{lbl})"


# ===========================================================================
# Standalone physics functions
# ===========================================================================

def compute_J_by_curl(B: ToroidalField) -> ToroidalField:
    """Compute current density J = curl(B) / mu0 via finite differences."""
    dR = B.R_arr[1] - B.R_arr[0]
    dZ = B.Z_arr[1] - B.Z_arr[0]
    J_R = -np.gradient(B.BPhi, dZ, axis=1, edge_order=2) / MU0
    J_Phi = (np.gradient(B.BR, dZ, axis=1, edge_order=2)
             - np.gradient(B.BZ, dR, axis=0, edge_order=2)) / MU0
    J_Z = (B.BPhi / B.R_arr[:, None]
           + np.gradient(B.BPhi, dR, axis=0, edge_order=2)) / MU0
    return ToroidalField(B.R_arr, B.Z_arr, J_R, J_Phi, J_Z, phi=B.phi,
                         label=f"curl({B.label})" if B.label else "J")


def poloidal_B(B: ToroidalField) -> np.ndarray:
    """Poloidal field magnitude |(BR, BZ)|."""
    return B.poloidal_abs


# ===========================================================================
# Equilibrium — slot-based container
# ===========================================================================

@runtime_checkable
class EquilibriumLike(Protocol):
    """Minimal protocol for any equilibrium representation.

    Any object with .B0 (ToroidalField) and .R_arr/.Z_arr satisfies this.
    Adapters for EFIT, LIUQE, TGYRO, etc. implement this protocol.
    """

    @property
    def B0(self) -> ToroidalField: ...

    @property
    def R_arr(self) -> np.ndarray:
        return self.B0.R_arr

    @property
    def Z_arr(self) -> np.ndarray:
        return self.B0.Z_arr


@dataclass
class Equilibrium:
    """MHD equilibrium on a toroidal cross-section.

    Design: slot-based container.  B0 is mandatory; everything else
    is an optional typed slot.  Use composition — only populate what
    you have.  For interop with specific codes, see adapters in
    pyna.io.* (future).

    Slots
    -----
    B0 : ToroidalField                  — magnetic field [T] (required)
    J_total, J_ohmic, J_bootstrap,      — current density [A/m²]
    J_ecrh, J_icrh, J_nbi, J_fast,
    J_diamagnetic : Optional[ToroidalField]
    p_total, p_ion, p_electron,         — pressure [Pa] (nR,nZ arrays)
    p_fast, p_rad : Optional[np.ndarray]
    u_ion, u_electron, u_fast,          — flow velocity [m/s]
    u_impurity : Optional[ToroidalField]
    psi : Optional[np.ndarray]           — poloidal flux [Wb/rad]
    q_profile : Optional[callable]      — q(psi)
    p_profile_1d : Optional[callable]   — p(psi)
    """

    B0: ToroidalField

    # -- Current components --
    J_total: Optional[ToroidalField] = None
    J_ohmic: Optional[ToroidalField] = None
    J_bootstrap: Optional[ToroidalField] = None
    J_ecrh: Optional[ToroidalField] = None
    J_icrh: Optional[ToroidalField] = None
    J_nbi: Optional[ToroidalField] = None
    J_fast: Optional[ToroidalField] = None
    J_diamagnetic: Optional[ToroidalField] = None

    # -- Pressure --
    p_total: Optional[np.ndarray] = None
    p_ion: Optional[np.ndarray] = None
    p_electron: Optional[np.ndarray] = None
    p_fast: Optional[np.ndarray] = None
    p_rad: Optional[np.ndarray] = None

    # -- Rotation / flow --
    u_ion: Optional[ToroidalField] = None
    u_electron: Optional[ToroidalField] = None
    u_fast: Optional[ToroidalField] = None
    u_impurity: Optional[ToroidalField] = None

    # -- Flux functions (1D) --
    psi: Optional[np.ndarray] = None          # poloidal flux on grid
    q_profile: Optional[object] = None        # q(psi): callable or 1D array
    p_profile_1d: Optional[object] = None     # p(psi): callable or 1D array

    # -- Metadata --
    label: Optional[str] = None
    meta: dict = dc_field(default_factory=dict)  # free-form metadata

    # -- derived --
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

    # -- helpers --
    def _zero_J(self):
        z = np.zeros_like(self.B0.BR)
        return ToroidalField(self.R_arr, self.Z_arr, z, z, z)

    def get_J_total(self) -> ToroidalField:
        """Return J_total, or zero field if not set."""
        return self.J_total if self.J_total is not None else self._zero_J()

    def _nonzero_slots(self) -> list[str]:
        slots = []
        for name in ["J_total", "J_ohmic", "J_bootstrap", "J_ecrh", "J_icrh",
                     "J_nbi", "J_fast", "J_diamagnetic",
                     "p_total", "p_ion", "p_electron", "p_fast", "p_rad",
                     "u_ion", "u_electron", "u_fast", "u_impurity",
                     "psi", "q_profile", "p_profile_1d"]:
            val = getattr(self, name)
            if val is not None and not (isinstance(val, np.ndarray) and not val.any()):
                continue
            if val is not None:
                slots.append(name)
        return slots

    def __repr__(self) -> str:
        lbl = f"'{self.label}'" if self.label else ""
        slots = self._nonzero_slots()
        slot_str = ", ".join(slots[:6])
        if len(slots) > 6:
            slot_str += f", +{len(slots)-6}"
        return f"Equilibrium({self.nR}x{self.nZ}, {slot_str}{lbl})"


# Backward compat: re-export MU0
__all__ = [
    "ToroidalField", "Equilibrium", "EquilibriumLike",
    "compute_J_by_curl", "poloidal_B",
    "MU0",
]
