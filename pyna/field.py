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
    

class CylindricalGridVectorField:
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
    
class CylindricalGridAxiVectorField(CylindricalGridVectorField):
    def __init__(self, R, Z, BR, BZ, BPhi) -> None:
        self._R = R
        self._Z = Z
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
    def BR(self):
        return self._BR
    @property
    def BZ(self):
        return self._BZ
    @property
    def BPhi(self):
        return self._BPhi
    
    def __add__(self, other):
        """计算两个矢量场的和"""
        return CylindricalGridAxiVectorField(
            self.R, self.Z,
            BR=self.BR + other.BR,
            BZ=self.BZ + other.BZ,
            BPhi=self.BPhi + other.BPhi
        )
    def __sub__(self, other):
        """计算两个矢量场的差"""
        return CylindricalGridAxiVectorField(
            self.R, self.Z,
            BR=self.BR - other.BR,
            BZ=self.BZ - other.BZ,
            BPhi=self.BPhi - other.BPhi
        )
    def __neg__(self):
        """计算矢量场的负"""
        return CylindricalGridAxiVectorField(
            self.R, self.Z,
            BR  = -self.BR,
            BZ  = -self.BZ,
            BPhi= -self.BPhi
        )
    def __mul__(self, other):
        """计算矢量场乘以一个标量"""
        if isinstance(other, CylindricalGridAxiVectorField):
            return self.dot(other)
        if isinstance(other, CylindricalGridAxiScalarField):
            return CylindricalGridAxiVectorField(
            self.R, self.Z,
            BR=self.BR * other.B,
            BZ=self.BZ * other.B,
            BPhi=self.BPhi * other.B
            )
        if isinstance(other, (int, float)):
            return CylindricalGridAxiVectorField(
            self.R, self.Z,
            BR=self.BR * other,
            BZ=self.BZ * other,
            BPhi=self.BPhi * other
            )
        else:
            raise TypeError(f"CylindricalGridAxiVectorField cannot multiply with a {type(other)}.")
        __rmul__ = __mul__

    def dot(self, other):
        """计算两个矢量场的点乘"""
        dot_product = self.BR * other.BR + self.BZ * other.BZ + self.BPhi * other.BPhi
        return CylindricalGridAxiScalarField(
            self.R, self.Z,
            B=dot_product
        )
    
    def cross(self, other):
        """计算两个矢量场的叉乘"""
        cross_BR = self.BPhi * other.BZ - self.BZ * other.BPhi
        cross_BPhi = self.BZ * other.BR - self.BR * other.BZ
        cross_BZ = self.BR * other.BPhi - self.BPhi * other.BR
        return CylindricalGridAxiVectorField(
            self.R, self.Z,
            BR  = cross_BR,
            BZ  = cross_BZ,
            BPhi= cross_BPhi
        )
    
    def curl(self):
        """计算矢量场的旋度"""
        curl_BR = - np.gradient(self.BZ, self.Phi, axis=2)
        curl_BPhi = (np.gradient(self.BR, self.Z, axis=1) - np.gradient(self.BZ, self.R, axis=0) )
        curl_BZ = self.BPhi / self.R[:,None] + np.gradient( self.BPhi, self.R, axis=0) 
        return CylindricalGridAxiVectorField(
            self.R, self.Z,
            BR  = curl_BR,
            BZ  = curl_BZ,
            BPhi= curl_BPhi
        )
    

class CylindricalGridScalarField:
    def __init__(self, R, Z, Phi, B) -> None:
        self._R = R
        self._Z = Z
        self._Phi = Phi
        self._B = B

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
        def B(self):
            return self._B
        
class CylindricalGridAxiScalarField(CylindricalGridScalarField):
    def __init__(self, R, Z, B) -> None:
        self._R = R
        self._Z = Z
        self._B = B

    @property
    def R(self):
        return self._R
    @property
    def Z(self):
        return self._Z
    @property
    def B(self):
        return self._B

    def __add__(self, other):
        """计算两个标量场的和"""
        return CylindricalGridAxiScalarField(
            self.R, self.Z,
            B=self.B + other.B
        )
    def __sub__(self, other):
        """计算两个标量场的差"""
        return CylindricalGridAxiScalarField(
            self.R, self.Z,
            B=self.B - other.B
        )
    def __mul__(self, other):
        """计算标量场乘以一个标量"""
        return CylindricalGridAxiScalarField(
            self.R, self.Z,
            B=self.B * other
        )
    __rmul__ = __mul__

    def grad(self):
        """计算标量场的梯度"""
        grad_R = np.gradient(self.B, self.R, axis=0)
        grad_Z = np.gradient(self.B, self.Z, axis=1)
        return CylindricalGridAxiVectorField(
            self.R, self.Z,
            BR=grad_R,
            BZ=grad_Z,
            BPhi=np.zeros_like(grad_R)
        )

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
    