import numpy as np
from functools import lru_cache

class RegualrCylindricalGridField:
    def __init__(self, R, Z, Phi, BR, BZ, BPhi) -> None:
        self._R = R
        self._Z = Z
        self._Phi = Phi
        self._BR = BR
        self._BZ = BZ
        self._BPhi = BPhi

    @property
    def R(self):
        return self._R
    @property
    def Z(self):
        return self._Z
    @property
    def Phi(self):
        return self._Phi
    @property
    def BR(self):
        return self._BR
    @property
    def BZ(self):
        return self._BZ
    @property
    def BPhi(self):
        return self._BPhi
    
# @lru_cache
# def diff_RZ(self, nR:int, nZ:int):
#     if nR == 0 and nZ == 0:
#         return self._field.copy()
#     elif nZ > 0:
#         return np.gradient(self.diff_RZ(nR, nZ-1), self._Z, axis=-2, edge_order=2)
#     elif nR > 0:
#         return np.gradient(self.diff_RZ(nR-1, nZ), self._R, axis=-3, edge_order=2)
#     else:
#         raise ValueError("nR, nZ to differentiate in the R,Z axis shall be >= 0.")

# @lru_cache
# def diff_RZ_interpolator(self, nR:int, nZ:int):
#     from scipy.interpolate import RegularGridInterpolator
#     if nR > 0 and nZ > 0:
#         return RegularGridInterpolator( 
#             (self._R[nR:-nR], self._Z[nZ:-nZ], self._Phi), self.diff_RZ(nR, nZ)[...,nR:-nR, nZ:-nZ,:],
#             method="linear", bounds_error=True )
#     elif nR > 0 and nZ == 0:
#         return RegularGridInterpolator( 
#             (self._R[nR:-nR], self._Z, self._Phi), self.diff_RZ(nR, nZ)[...,nR:-nR, :,:],
#             method="linear", bounds_error=True )
#     elif nR == 0 and nZ > 0:
#         return RegularGridInterpolator( 
#             (self._R, self._Z[nZ:-nZ], self._Phi), self.diff_RZ(nR, nZ)[...,:, nZ:-nZ,:],
#             method="linear", bounds_error=True )
#     elif nR == 0 and nZ == 0:
#         return RegularGridInterpolator( 
#             (self._R, self._Z, self._Phi), self.diff_RZ(nR, nZ)[...,:,:,:],
#             method="linear", bounds_error=True )
#     else:
#         raise ValueError("nR, nZ to differentiate in the R,Z axis shall be >= 0.")

from scipy.interpolate import RegularGridInterpolator
def RBRBZoBPhi_diffRZ_interpolator_list(afield:RegualrCylindricalGridField, upto_order:int=5):
    R, Z, Phi = afield.R, afield.Z, afield.Phi
    BR, BZ, BPhi = afield.BR, afield.BZ, afield.BPhi
    
    _RBRoBPhi_diffRZ_lastord_list = [ R[:,None,None] * BR / BPhi ]
    _RBZoBPhi_diffRZ_lastord_list = [ R[:,None,None] * BZ / BPhi ]
    _RBRoBPhi_diffRZ_thisord_list = [ ]
    _RBZoBPhi_diffRZ_thisord_list = [ ]
    
    _RBRoBPhi_diffRZ_interpolator_list = []
    _RBZoBPhi_diffRZ_interpolator_list = []
    
    _RBRoBPhi_diffRZ_interpolator_list.append( 
        RegularGridInterpolator( 
            (R, Z, Phi), _RBRoBPhi_diffRZ_lastord_list[0],
            method="linear", bounds_error=True )
    )
    _RBZoBPhi_diffRZ_interpolator_list.append(
        RegularGridInterpolator( 
            (R, Z, Phi), _RBZoBPhi_diffRZ_lastord_list[0],
            method="linear", bounds_error=True )
    )
    
    for ord in range(1, upto_order+1):
        for i in range(ord):
            _RBRoBPhi_diffRZ_thisord_list.append(
                np.gradient(_RBRoBPhi_diffRZ_lastord_list[i], 
                    R, axis=0, edge_order=2)
            )
            _RBZoBPhi_diffRZ_thisord_list.append(
                np.gradient(_RBZoBPhi_diffRZ_lastord_list[i], 
                    R, axis=0, edge_order=2)
            )
        _RBRoBPhi_diffRZ_thisord_list.append(
            np.gradient(_RBRoBPhi_diffRZ_lastord_list[-1], 
                Z, axis=1, edge_order=2)
        )
        _RBZoBPhi_diffRZ_thisord_list.append(
            np.gradient(_RBZoBPhi_diffRZ_lastord_list[-1], 
                Z, axis=1, edge_order=2)
        )
        
        _RBRoBPhi_diffRZ_interpolator_list.append( 
            RegularGridInterpolator( 
                (R, Z, Phi), np.stack(_RBRoBPhi_diffRZ_thisord_list, axis=-1),
                method="linear", bounds_error=True )
        )
        _RBZoBPhi_diffRZ_interpolator_list.append(
            RegularGridInterpolator( 
                (R, Z, Phi), np.stack(_RBZoBPhi_diffRZ_thisord_list, axis=-1),
                method="linear", bounds_error=True )
        )
        
        _RBRoBPhi_diffRZ_lastord_list = _RBRoBPhi_diffRZ_thisord_list
        _RBZoBPhi_diffRZ_lastord_list = _RBZoBPhi_diffRZ_thisord_list
        _RBRoBPhi_diffRZ_thisord_list = [ ]
        _RBZoBPhi_diffRZ_thisord_list = [ ]
        
    return _RBRoBPhi_diffRZ_interpolator_list, _RBZoBPhi_diffRZ_interpolator_list

class Ak_interpolator:
    def __init__(self, afield:RegularGridInterpolator, upto_order=5):
        self._upto_order = upto_order
        self._RBRoBPhi_diffRZ_interpolator_list, self._RBZoBPhi_diffRZ_interpolator_list = RBRBZoBPhi_diffRZ_interpolator_list(afield, upto_order)
    def __call__(self, k:int, xi, method="linear"):
        if k > self._upto_order:
            raise ValueError("k > upto_order (defaults to be 5), please initialize a bigger upto_order.")
        RBRoBPhi_kdiffRZ = self._RBRoBPhi_diffRZ_interpolator_list[k](xi)[0,:]
        RBZoBPhi_kdiffRZ = self._RBZoBPhi_diffRZ_interpolator_list[k](xi)[0,:]
        Ak = np.empty([2,]*(k+1) )
        for i, _ in np.ndenumerate( Ak ):
            if i[0] == 0:
                Ak[i] = RBRoBPhi_kdiffRZ[sum( i[1:] )]
            elif i[0] == 1:
                Ak[i] = RBZoBPhi_kdiffRZ[sum( i[1:] )]
        return Ak
    