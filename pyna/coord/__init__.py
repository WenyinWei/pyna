"""
pyna.coord — coordinate system utilities.

Includes the old coord.py utilities and the new PEST coordinate system.
"""

# Legacy flat-module functions (previously in pyna/coord.py)
import numpy as np


def RZPhi_range_2_XYZ_mesh(R, Z, Phi):
    """Convert (R, Z, Phi) range arrays to a 3-D Cartesian mesh."""
    Rv, Zv, Phiv = np.meshgrid(R, Z, Phi, indexing='ij')
    return RZPhi_mesh_2_XYZ_mesh(Rv, Zv, Phiv)


def RZPhi_mesh_2_XYZ_mesh(Rv, Zv, Phiv):
    """Convert cylindrical mesh arrays to Cartesian (X, Y, Z)."""
    Xv = Rv * np.cos(Phiv)
    Yv = Rv * np.sin(Phiv)
    return Xv, Yv, Zv


# PEST coordinate system
from pyna.coord.PEST import (
    build_PEST_mesh,
    RZmesh_isoSTET,
    g_i_g__i_from_STET_mesh,
    counter_comp_of_a_field,
    co_comp_of_a_field,
)

__all__ = [
    "RZPhi_range_2_XYZ_mesh",
    "RZPhi_mesh_2_XYZ_mesh",
    "build_PEST_mesh",
    "RZmesh_isoSTET",
    "g_i_g__i_from_STET_mesh",
    "counter_comp_of_a_field",
    "co_comp_of_a_field",
]
