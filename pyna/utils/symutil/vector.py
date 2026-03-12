"""Symbolic vector/tensor algebra utilities.

General-purpose helpers for sympy Array operations:
coordinate transforms, dot/cross products, norms.

Author: Wenyin Wei wenyin.wei@ipp.ac.cn
Migrated from: silkpy/silkpy/sympy_utility.py
"""

from sympy import Array as _Array


def r_transform(a, from_coord, to_coord):
    """Transform a position vector between coordinate systems.

    Args:
        a: sympy Array or sequence [coord1, coord2, coord3]
        from_coord: 'cartesian' | 'spherical' | 'cylindrical'
        to_coord:   'cartesian' | 'spherical' | 'cylindrical'
    """
    from sympy import sin, cos, atan, sqrt, atan2
    if from_coord == 'spherical' and to_coord == 'cartesian':
        r, theta, phi = a[0], a[1], a[2]
        return _Array([r*sin(theta)*cos(phi),
                       r*sin(theta)*sin(phi),
                       r*cos(theta)])
    if from_coord == 'cartesian' and to_coord == 'spherical':
        x, y, z = a[0], a[1], a[2]
        return _Array([sqrt(x**2 + y**2 + z**2),
                       atan(sqrt(x**2 + y**2) / z),
                       atan2(y, x)])
    if from_coord == 'cylindrical' and to_coord == 'cartesian':
        rho, phi, z = a[0], a[1], a[2]
        return _Array([rho*cos(phi), rho*sin(phi), z])
    if from_coord == 'cartesian' and to_coord == 'cylindrical':
        x, y, z = a[0], a[1], a[2]
        return _Array([sqrt(x**2 + y**2), atan2(y, x), z])
    raise NotImplementedError(
        f"r_transform from '{from_coord}' to '{to_coord}' not implemented.")


def dot(a, b):
    """Dot product of two rank-1 sympy Arrays."""
    if a.rank() != 1 or b.rank() != 1:
        raise ValueError("a, b must be rank-1 (vector) Arrays.")
    if a.shape != b.shape:
        raise ValueError("Shape mismatch for dot product.")
    return sum(a[i]*b[i] for i in range(a.shape[0]))


def cross(a, b):
    """Cross product of two 3D rank-1 sympy Arrays."""
    return _Array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ])


def norm(arr):
    """Euclidean norm of a sympy Array."""
    from operator import add
    from functools import reduce
    from sympy import sqrt
    return sqrt(reduce(add, arr.applyfunc(lambda x: x**2))).simplify().refine()


def triple_prod(a, b, c):
    """Scalar triple product a · (b × c) for 3D vectors."""
    if any(v.rank() != 1 for v in (a, b, c)):
        raise ValueError("a, b, c must be rank-1 (vector) Arrays.")
    if any(v.shape[0] != 3 for v in (a, b, c)):
        raise ValueError("a, b, c must be 3D vectors.")
    return dot(cross(a, b), c)
