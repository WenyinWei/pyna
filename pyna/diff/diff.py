from ..map import MapSympy, MapCallable
import sympy
from sympy import derive_by_array
import numpy as np

def jac_col_2_mat(jac):
    return np.array( [
        [jac[2], jac[0]], 
        [jac[3], jac[1]]])

def Jacobian_4_MapSympy(pmap: MapSympy):
    from ..withparam import ImmutableDenseNDimArrayWithParam
    return ImmutableDenseNDimArrayWithParam(
        derive_by_array( pmap.next_xi_exprs, pmap.xi_syms ), 
        pmap.param_dict
    ) 

def Jacobian_4_MapCallable(pmap):
    pass


def Jacobian_4_field_line_Poincare(pmap):
    pass

