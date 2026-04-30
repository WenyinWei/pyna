"""symutil — domain-agnostic sympy utility helpers for pyna.

Submodules
----------
basics      : array equation utilities (from wagglepy)
op          : time-averaging of trig expressions (from wagglepy)
characteristics : trig period detection (from wagglepy)
vector      : symbolic vector/tensor algebra, coord transforms (from silkpy)
"""

from .basics import divide_Array_Eq
from .vector import r_transform, dot, cross, norm, triple_prod
