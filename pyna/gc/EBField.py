from abc import ABC, abstractmethod
import numpy as np

def space_dirderivative(f, xyz, t, direction, h=1e-8):
    direction = np.array(direction)
    norm = np.linalg.norm(direction)
    if norm == 0:
        return np.zeros_like(f(*xyz, t))
    direction = direction / norm  # Normalize the direction vector
    
    x, y, z = xyz
    dx, dy, dz = direction * h
    return (f(x + dx, y + dy, z + dz, t) - f(x - dx, y - dy, z - dz, t)) / (2 * h)

class EBField(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        """示范性构造函数，不允许直接实例化"""
        pass   

    @abstractmethod
    def E_at(self, x, y, z, t):
        """Calculate the electric field at a given point and time."""
        pass
    @abstractmethod
    def B_at(self, x, y, z, t):
        """Calculate the magnetic field at a given point and time."""
        pass

    def B_abs_at(self, x, y, z, t):
        return np.linalg.norm(self.B_at(x, y, z, t))
    def E_and_B_at(self, x, y, z, t):
        return self.E_at(x, y, z, t), self.B_at(x, y, z, t)
    def hatb_at(self, x, y, z, t):
        return self.B_at(x, y, z, t) / np.linalg.norm(self.B_at(x, y, z, t))
    def vE_at(self, x, y, z, t):
        return np.cross(self.E_at(x, y, z, t), self.hatb_at(x, y, z, t)) / self.B_abs_at(x, y, z, t)
    def gradB_abs_at(self, x, y, z, t):
        xyz = np.array([x, y, z])
        return np.array([
            space_dirderivative(self.B_abs_at, xyz, t, [1,0,0]),
            space_dirderivative(self.B_abs_at, xyz, t, [0,1,0]),
            space_dirderivative(self.B_abs_at, xyz, t, [0,0,1]),
        ])
    def kappa_at(self, x, y, z, t):
        return space_dirderivative(self.hatb_at, np.array([x, y, z]), t, self.hatb_at(x, y, z, t))