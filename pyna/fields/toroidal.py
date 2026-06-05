"""Toroidal field compatibility helpers built on cylindrical field objects.

The canonical cylindrical vector-field classes live in
:mod:`pyna.fields.cylindrical`:

* ``VectorFieldCylind`` for full 3-D cylindrical grids
* ``VectorFieldCylindAxisym`` for axisymmetric fields

This module keeps section-equilibrium helpers and historical imports from
``pyna.topo.field`` working without defining a second vector-field class.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from pyna.fields.base import ScalarField, TensorField
from pyna.fields.cylindrical import (
    CylindricalFieldArrays,
    VectorFieldCylind,
    VectorFieldCylindAxisym,
    as_vector_field_cylind,
    as_vector_field_cylindrical,
)


MU0 = 4e-7 * np.pi

ToroidalField = VectorFieldCylind
AxisymmetricField = VectorFieldCylindAxisym


def _section_components(B: VectorFieldCylind) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 2-D section components from a cylindrical field."""

    B = as_vector_field_cylindrical(B)
    if B.is_section:
        return np.asarray(B.BR), np.asarray(B.BZ), np.asarray(B.BPhi)
    if B.nPhi != 1:
        raise NotImplementedError("section helper requires a fixed-section or nPhi=1 field")
    return B.components_3d[0][:, :, 0], B.components_3d[1][:, :, 0], B.components_3d[2][:, :, 0]


def compute_J_by_curl(B: VectorFieldCylind) -> VectorFieldCylind:
    """Compute current density ``J = curl(B) / mu0`` on a fixed section."""

    B = as_vector_field_cylindrical(B)
    BR, BZ, BPhi = _section_components(B)
    dR = B.R_arr[1] - B.R_arr[0]
    dZ = B.Z_arr[1] - B.Z_arr[0]
    J_R = -np.gradient(BPhi, dZ, axis=1, edge_order=2) / MU0
    dBPhi_dR = np.gradient(BPhi, dR, axis=0, edge_order=2)
    J_Z = (BPhi / B.R_arr[:, None] + dBPhi_dR) / MU0
    dBR_dZ = np.gradient(BR, dZ, axis=1, edge_order=2)
    dBZ_dR = np.gradient(BZ, dR, axis=0, edge_order=2)
    J_Phi = (dBR_dZ - dBZ_dR) / MU0
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

    def __post_init__(self):
        self.B = as_vector_field_cylindrical(self.B)

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
        z = np.zeros_like(_section_components(self.B)[0])
        return VectorFieldCylind(self.R_arr, self.Z_arr, z, z, z, phi=self.phi)

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
    "CylindricalFieldArrays",
    "TensorField",
    "ToroidalField",
    "AxisymmetricField",
    "Equilibrium",
    "EquilibriumLike",
    "as_vector_field_cylindrical",
    "as_vector_field_cylind",
    "compute_J_by_curl",
    "MU0",
]
