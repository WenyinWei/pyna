
from .coord_transform import CoordTransform
class Cylindrical2Cartesian(CoordTransform):
    def __init__(self, syms=None):
        from sympy import symbols, cos, sin
        if syms is None:
            syms = (symbols('R', positive=True), *symbols('Z, phi', real=True))
        R, Z, phi = syms[0], syms[1], syms[2]
        CoordTransform.__init__(self, syms, [R*cos(phi), R*sin(phi), Z])

class Cartesian2Cylindrical(CoordTransform):
    def __init__(self, syms=None):
        from sympy import symbols, sqrt, atan2
        if syms is None:
            syms = symbols('x, y, z', real=True)
        x, y, z = syms[0], syms[1], syms[2]
        CoordTransform.__init__(self, syms, [sqrt(x**2+y**2), z, atan2(y, x)])


