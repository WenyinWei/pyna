from pyna.polynomial import Poly2d
import numpy as np

class PolyMap2d:
    def __init__(self, cor1poly, cor2poly) -> None:
        self._cor1poly = cor1poly # TODO: deep copy needed, but I have not yet come up with the copy constructor
        self._cor2poly = cor2poly

    def __matmul__(self, other): # composition, # z(y1(x1, x2), y2(x1, x2)) = z(y), where y=y(x)=(y1(x1,x2), y2(x1,x2))
        return PolyMap2d( 
            self._cor1poly @ ( other._cor1poly, other._cor2poly, ),
            self._cor2poly @ ( other._cor1poly, other._cor2poly, ) )
        
    def __pow__(self, pw):
        if pw==0:
            return PolyMap2d(
                Poly2d( np.array([[0,0], [1,0],]), ),
                Poly2d( np.array([[0,1], [0,0],]), ), 
            )            
        elif pw==1:
            return self
        return self**(pw-1) @ self

from numpy import linalg as LA
from pyna.polynomial import anti_diagnoal, set_Poly2d_anti_diagnoal
from scipy.optimize import minimize

def inv_PolyMap2d(orimap, tofind_poly_ord:int = None):
    """_summary_

    Args:
        orimap (_type_): _description_
        tofind_poly_size (int, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_

    """
    if orimap._cor1poly._arr[0,0]!=0.0 or orimap._cor2poly._arr[0,0]!=0.0 :
        raise ValueError("To inverse a 2D map composed by 2-variate polynomials, it is required that the origin is mapped to origin. Otherwise, this function would be too complicated.")
    if tofind_poly_ord is None:
        tofind_poly_ord = max( orimap._cor1poly._arr.shape ) - 1
        tofind_poly_ord = max( tofind_poly_ord, 4 ) # we require at least accuracy on the 4th order

    tofind_cor1poly, tofind_cor2poly = [ Poly2d( np.zeros([
        tofind_poly_ord+1, 
        tofind_poly_ord+1]) ) for _ in range(2) ]
    def _loss(x, tofind_poly, polyord:int, corind:int=None):
        ori_map_cor1poly = orimap._cor1poly
        ori_map_cor2poly = orimap._cor2poly
        set_Poly2d_anti_diagnoal( tofind_poly, polyord, x )
        approx_id_poly = tofind_poly @ (ori_map_cor1poly, ori_map_cor2poly) # TODO: abundant computation can be lessoned by memorizing the known low order composition results
        if polyord > 1:
            return LA.norm( anti_diagnoal(approx_id_poly._arr, offset=polyord) )
        elif polyord == 1:
            if corind == 1:
                return LA.norm( anti_diagnoal(approx_id_poly._arr, offset=polyord) - [0,1] )
            elif corind == 2:
                return LA.norm( anti_diagnoal(approx_id_poly._arr, offset=polyord) - [1,0] )
    for polyord in range(1, tofind_poly_ord+1):
        minimize(lambda x: _loss(x, tofind_cor1poly, polyord, corind=1), x0=np.zeros([polyord+1]), method='nelder-mead',
            options={'xatol': 1e-10, 'disp': False})
        minimize(lambda x: _loss(x, tofind_cor2poly, polyord, corind=2), x0=np.zeros([polyord+1]), method='nelder-mead',
            options={'xatol': 1e-10, 'disp': False})
    
    return PolyMap2d(
        tofind_cor1poly,
        tofind_cor2poly,
    )