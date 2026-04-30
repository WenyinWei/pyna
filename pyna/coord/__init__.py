"""pyna.coord — backward-compatibility shim.
Files have moved to pyna.toroidal.coords; this module re-exports everything.
"""
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


from pyna.toroidal.coords.PEST import (
    build_PEST_mesh,
    RZmesh_isoSTET,
    g_i_g__i_from_STET_mesh,
    counter_comp_of_a_field,
    co_comp_of_a_field,
)
from pyna.toroidal.coords.EqualArc import build_equal_arc_mesh
from pyna.toroidal.coords.Hamada import build_Hamada_mesh
from pyna.toroidal.coords.Boozer import build_Boozer_mesh

__all__ = [
    "RZPhi_range_2_XYZ_mesh",
    "RZPhi_mesh_2_XYZ_mesh",
    "build_PEST_mesh",
    "RZmesh_isoSTET",
    "g_i_g__i_from_STET_mesh",
    "counter_comp_of_a_field",
    "co_comp_of_a_field",
    "build_equal_arc_mesh",
    "build_Hamada_mesh",
    "build_Boozer_mesh",
]
