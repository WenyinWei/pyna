"""Algebra utilities involved in the deduction procedure of fusion plasma physics formula.

Author: Wenyin Wei wenyin.wei@ipp.ac.cn

"""

from sympy import Array as _Array

def r_transform(a, from_coord, to_coord):
    """
    Args:
    from_coord='cartesian',to_coord='spherical'
    """
    from sympy import sin, cos, atan, sqrt, atan2
    if from_coord=='spherical' and to_coord=='cartesian':
        r, theta, phi = a[0], a[1], a[2]
        return _Array[(
            r*sin(theta)*cos(phi), 
            r*sin(theta)*sin(phi), 
            r*cos(theta))]
    if from_coord=='cartesian' and to_coord=='spherical':
        x, y, z = a[0], a[1], a[2]
        return _Array[(
            sqrt(x**2 + y**2 + z**2), 
            atan(sqrt(x**2+y**2)/z), 
            atan2(y, x))]
    elif from_coord=='cylindrical' and to_coord=='cartesian':
        rho, phi, z = a[0], a[1], a[2]
        return _Array[(
            rho*cos(phi), 
            rho*sin(phi), 
            z)]
    if from_coord == 'cartesian' and to_coord=='cylindrical':
        x, y, z = a[0], a[1], a[2] 
        return _Array[(
            sqrt(x**2+y**2), 
            atan2(y, x), 
            z)]
    else:
        raise NotImplementedError("The coordinate system transform, from {from_coord} to {to_coord}, for position vector r has not yet been prepared.")

def dot(a, b): 
    if a.rank() !=1 or b.rank() !=1:
        raise ValueError("a, b should be vector, i.e. rank==1.")
    if a.shape != b.shape:
        raise ValueError("Unmatched Array shape. a, b shape should identical for dot operation.")
        
    res = 0
    for i in range(a.shape[0]):
        res += a[i]*b[i]
    return res
        
def dot4r(a, b, coord='cartesian'):
    if coord!='cartesian':
        a = r_transform(a, from_coord=coord, to_coord='cartesian')
        b = r_transform(b, from_coord=coord, to_coord='cartesian')
    
    ans = dot(a, b, metric=None)
    
    if coord!='cartesian':
        return r_transform(ans, from_coord='cartesian', to_coord=coord)
    else:
        return ans
        
def cross(a, b):
    return _Array([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]])
        

def cross4r(a, b, coord='cartesian'):
    if coord!='cartesian':
        a = r_transform(a, from_coord=coord, to_coord='cartesian')
        b = r_transform(b, from_coord=coord, to_coord='cartesian')
    
    ans = cross(a, b, metric=None)
    
    if coord!='cartesian':
        return r_transform(ans, from_coord='cartesian', to_coord=coord)
    else:
        return ans

def triple_prod(a, b, c):
    if a.rank() !=1 or b.rank() != 1 or c.rank() != 1:
        raise ValueError("a, b, c should be vector, i.e. rank==1.")
    if a.shape[0] != (3) or b.shape[0] != (3) or c.shape[0] != (3):
        raise ValueError("Unmatched Array shape. a, b, c should be vector in 3D.")
        
    return dot(cross(a, b), c)

def norm(arr):
    from operator import add
    from functools import reduce
    from sympy import sqrt
    return sqrt(reduce(add, arr.applyfunc(lambda x: x**2))).simplify().refine()


def diff4r(r, x, n, coord='cartesian'):
    if coord=='cartesian':
        return r.diff(x, n)
    elif coord=='spherical':
        raise NotImplementedError()
    elif coord=='cylindrical':
        raise NotImplementedError()
    return 

if __name__ == '__main__':
    from sympy import symbols
    r1, theta1, phi1, r2, theta2, phi2 = symbols('r_1, theta_1, phi_1, r_2, theta_2, phi_2', real=True, negative=False)
    vr1 = (r1, theta1, phi1)
    vr2 = (r2, theta2, phi2)
    cross4r(vr1, vr2, coord='spherical')[0].simplify()

    cross4r(vr1, vr2, coord='spherical')[1].simplify()

    cross4r(vr1, vr2, coord='spherical')[2].simplify()