"""Toroidal field compatibility helpers built on the canonical field hierarchy.

This module keeps the lightweight fixed-section magnetic-field containers that
historically lived in :mod:`pyna.topo.field`, but makes them subclasses/aliases
of :mod:`pyna.fields` rather than a parallel hierarchy.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from pyna.fields.base import ScalarField, TensorField
from pyna.fields.cylindrical import VectorField3DAxiSymmetric


MU0 = 4e-7 * np.pi


class VectorFieldCylind(VectorField3DAxiSymmetric):
    """Vector field on an ``(R, Z)`` grid at a fixed toroidal angle.

    Component order is ``(R, Z, phi)``.  This compatibility container presents
    2-D ``BR/BZ/BPhi`` arrays like the historical topology API while inheriting
    from the canonical :class:`pyna.fields.VectorField` tree.
    """

    def __init__(
        self,
        R_arr: np.ndarray,
        Z_arr: np.ndarray,
        BR: np.ndarray,
        BZ: np.ndarray,
        BPhi: np.ndarray,
        phi: float = 0.0,
        label: Optional[str] = None,
    ) -> None:
        self.phi = float(phi)
        self.label = label
        super().__init__(
            R_arr,
            Z_arr,
            BR,
            BZ,
            BPhi,
            name=label or "",
            units="",
        )

    @property
    def R_arr(self) -> np.ndarray:
        return self.R

    @property
    def Z_arr(self) -> np.ndarray:
        return self.Z

    @property
    def coordinate_names(self) -> Tuple[str, str, str]:
        return ("R", "Z", "phi")

    @property
    def BR(self) -> np.ndarray:
        return self._VR[:, :, 0]

    @property
    def BZ(self) -> np.ndarray:
        return self._VZ[:, :, 0]

    @property
    def BPhi(self) -> np.ndarray:
        return self._VPhi[:, :, 0]

    @property
    def VR(self) -> np.ndarray:
        return self.BR

    @property
    def VZ(self) -> np.ndarray:
        return self.BZ

    @property
    def VPhi(self) -> np.ndarray:
        return self.BPhi

    @property
    def components(self) -> np.ndarray:
        return np.stack([self.BR, self.BZ, self.BPhi], axis=0)

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(np.sum(self.components**2, axis=0))))

    @property
    def shape(self) -> Tuple[int, int]:
        return self.BR.shape

    @property
    def nR(self) -> int:
        return self.BR.shape[0]

    @property
    def nZ(self) -> int:
        return self.BR.shape[1]

    @property
    def abs(self) -> np.ndarray:
        return np.sqrt(self.BR**2 + self.BZ**2 + self.BPhi**2)

    @property
    def poloidal_abs(self) -> np.ndarray:
        return np.sqrt(self.BR**2 + self.BZ**2)

    @classmethod
    def zero_like(cls, other: "VectorFieldCylind", label: str = "") -> "VectorFieldCylind":
        z = np.zeros_like(other.BR)
        return cls(other.R_arr, other.Z_arr, z, z, z, phi=other.phi, label=label)

    @classmethod
    def from_cache(cls, cache: dict, phi_idx: int = 0, *, label: str = "") -> "VectorFieldCylind":
        return cls(
            R_arr=cache["R_grid"],
            Z_arr=cache["Z_grid"],
            BR=cache["BR"][:, :, phi_idx],
            BZ=cache["BZ"][:, :, phi_idx],
            BPhi=cache["BPhi"][:, :, phi_idx],
            phi=float(cache["Phi_grid"][phi_idx]),
            label=label or f"cache_phi{phi_idx}",
        )

    def downsample(self, skip: int) -> "VectorFieldCylind":
        return VectorFieldCylind(
            R_arr=self.R_arr[::skip],
            Z_arr=self.Z_arr[::skip],
            BR=self.BR[::skip, ::skip],
            BZ=self.BZ[::skip, ::skip],
            BPhi=self.BPhi[::skip, ::skip],
            phi=self.phi,
            label=self.label,
        )

    def __add__(self, other: "VectorFieldCylind") -> "VectorFieldCylind":
        return VectorFieldCylind(
            R_arr=self.R_arr,
            Z_arr=self.Z_arr,
            BR=self.BR + other.BR,
            BZ=self.BZ + other.BZ,
            BPhi=self.BPhi + other.BPhi,
            phi=self.phi,
        )

    def __repr__(self) -> str:
        lbl = f"'{self.label}'" if self.label else ""
        return f"VectorFieldCylind({self.nR}x{self.nZ}, rms={self.rms:.3g}{lbl})"


class VectorFieldCylindAxisym(VectorFieldCylind):
    """Axisymmetric cylindrical vector field: ``∂/∂phi = 0``."""


ToroidalField = VectorFieldCylind
AxisymmetricField = VectorFieldCylindAxisym


def compute_J_by_curl(B: VectorFieldCylind) -> VectorFieldCylind:
    """Compute current density ``J = curl(B) / mu0`` via finite differences."""
    dR = B.R_arr[1] - B.R_arr[0]
    dZ = B.Z_arr[1] - B.Z_arr[0]
    J_R = -np.gradient(B.BPhi, dZ, axis=1, edge_order=2) / MU0
    J_Z = (B.BPhi / B.R_arr[:, None] + np.gradient(B.BPhi, dR, axis=0, edge_order=2)) / MU0
    J_Phi = (np.gradient(B.BR, dZ, axis=1, edge_order=2) - np.gradient(B.BZ, dR, axis=0, edge_order=2)) / MU0
    return VectorFieldCylind(
        B.R_arr,
        B.Z_arr,
        J_R,
        J_Z,
        J_Phi,
        phi=B.phi,
        label=f"curl({B.label})" if B.label else "J",
    )


@runtime_checkable
class EquilibriumLike(Protocol):
    @property
    def B(self) -> VectorFieldCylind: ...

    @property
    def R_arr(self) -> np.ndarray:
        return self.B.R_arr

    @property
    def Z_arr(self) -> np.ndarray:
        return self.B.Z_arr


@dataclass
class Equilibrium:
    """MHD equilibrium on a toroidal cross-section."""

    B: VectorFieldCylind
    J_total: Optional[VectorFieldCylind] = None
    J_ohmic: Optional[VectorFieldCylind] = None
    J_bootstrap: Optional[VectorFieldCylind] = None
    J_ecrh: Optional[VectorFieldCylind] = None
    J_icrh: Optional[VectorFieldCylind] = None
    J_nbi: Optional[VectorFieldCylind] = None
    J_fast: Optional[VectorFieldCylind] = None
    J_diamagnetic: Optional[VectorFieldCylind] = None
    p_total: Optional[np.ndarray] = None
    p_ion: Optional[np.ndarray] = None
    p_electron: Optional[np.ndarray] = None
    p_fast: Optional[np.ndarray] = None
    p_rad: Optional[np.ndarray] = None
    u_ion: Optional[VectorFieldCylind] = None
    u_electron: Optional[VectorFieldCylind] = None
    u_fast: Optional[VectorFieldCylind] = None
    u_impurity: Optional[VectorFieldCylind] = None
    psi: Optional[np.ndarray] = None
    q_profile: Optional[object] = None
    p_profile_1d: Optional[object] = None
    label: Optional[str] = None
    meta: dict = dc_field(default_factory=dict)

    @property
    def R_arr(self):
        return self.B.R_arr

    @property
    def Z_arr(self):
        return self.B.Z_arr

    @property
    def phi(self):
        return self.B.phi

    @property
    def nR(self):
        return self.B.nR

    @property
    def nZ(self):
        return self.B.nZ

    def _zero_J(self):
        z = np.zeros_like(self.B.BR)
        return VectorFieldCylind(self.R_arr, self.Z_arr, z, z, z)

    def get_J_total(self) -> VectorFieldCylind:
        return self.J_total if self.J_total is not None else self._zero_J()

    @classmethod
    def from_cache(cls, cache: dict, phi_idx: int = 0, *, label: str = "", compute_J0: bool = False) -> "Equilibrium":
        B = VectorFieldCylind.from_cache(cache, phi_idx, label=label)
        J_total = compute_J_by_curl(B) if compute_J0 else None
        return cls(B=B, J_total=J_total)

    def __repr__(self) -> str:
        lbl = f"'{self.label}'" if self.label else ""
        j = " +J" if self.J_total is not None else ""
        return f"Equilibrium({self.nR}x{self.nZ}{j}{lbl})"


__all__ = [
    "ScalarField",
    "VectorFieldCylind",
    "VectorFieldCylindAxisym",
    "TensorField",
    "ToroidalField",
    "AxisymmetricField",
    "Equilibrium",
    "EquilibriumLike",
    "compute_J_by_curl",
    "MU0",
]
