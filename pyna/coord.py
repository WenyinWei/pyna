import numpy as np

def RZPhi_range_2_XYZ_mesh(R, Z, Phi):
    Rv, Zv, Phiv = np.meshgrid(R, Z, Phi, indexing='ij')
    return RZPhi_mesh_2_XYZ_mesh(Rv, Zv, Phiv)

def RZPhi_mesh_2_XYZ_mesh(Rv, Zv, Phiv):
    Xv = Rv * np.cos(Phiv)
    Yv = Rv * np.sin(Phiv)
    Zv = Zv
    return Xv, Yv, Zv
