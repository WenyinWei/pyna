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

from numpy import ndarray
from multiprocessing import shared_memory
class RegualrCylindricalGridFieldNumpySharedMemory(RegualrCylindricalGridField):
    """
    According to the document of multiprocessing.shared_memory:
    
    As a resource for sharing data across processes, shared memory blocks may outlive the original process that created them. When one process no longer needs access to a shared memory block that might still be needed by other processes, the `close()` method should be called. When a shared memory block is no longer needed by any process, the `unlink()` method should be called to ensure proper cleanup.
    
    原则上，一个进程不再使用这个共享资源的时候调用`close()`方法，当所有进程都不再需要它的时候调用`unlink()`，但实际上在科学计算中我们并不是很关心内存泄漏，只要在用户的电脑有足够的内存，完全可以等程序把你电脑炸了重新启动就好了，let it crash。

    Args:
        RegualrCylindricalGridField: super class
    """
    def __init__(self, 
                 shm_names:list, 
                 np_shapes:list,
                 np_dtypes:list, ) -> None:
        self._R_shm_name = shm_names[0]
        self._Z_shm_name = shm_names[1]
        self._Phi_shm_name = shm_names[2]
        self._BR_shm_name = shm_names[3]
        self._BZ_shm_name = shm_names[4]
        self._BPhi_shm_name = shm_names[5]
        
        self._R_np_shape = np_shapes[0]
        self._Z_np_shape = np_shapes[1]
        self._Phi_np_shape = np_shapes[2]
        self._BR_np_shape = np_shapes[3]
        self._BZ_np_shape = np_shapes[4]
        self._BPhi_np_shape = np_shapes[5]
        
        self._R_np_dtype = np_dtypes[0]
        self._Z_np_dtype = np_dtypes[1]
        self._Phi_np_dtype = np_dtypes[2]
        self._BR_np_dtype = np_dtypes[3]
        self._BZ_np_dtype = np_dtypes[4]
        self._BPhi_np_dtype = np_dtypes[5]
    
        self._have_read_or_not = False
        
    def _read(self):
        if self._have_read_or_not:
            pass
        else:
            self._have_read_or_not = True
            self._R_shm = shared_memory.SharedMemory(name=self._R_shm_name, create=False)
            self._Z_shm = shared_memory.SharedMemory(name=self._Z_shm_name, create=False)
            self._Phi_shm = shared_memory.SharedMemory(name=self._Phi_shm_name, create=False)
            self._BR_shm = shared_memory.SharedMemory(name=self._BR_shm_name, create=False)
            self._BZ_shm = shared_memory.SharedMemory(name=self._BZ_shm_name, create=False)
            self._BPhi_shm = shared_memory.SharedMemory(name=self._BPhi_shm_name, create=False)
        
    @property
    def R(self):
        self._read()
        return ndarray(self._R_np_shape, dtype=self._R_np_dtype, buffer=self._R_shm.buf)
    @property
    def Z(self):
        self._read()
        return ndarray(self._Z_np_shape, dtype=self._Z_np_dtype, buffer=self._Z_shm.buf)
    @property
    def Phi(self):
        self._read()
        return ndarray(self._Phi_np_shape, dtype=self._Phi_np_dtype, buffer=self._Phi_shm.buf)
    @property
    def BR(self):
        self._read()
        return ndarray(self._BR_np_shape, dtype=self._BR_np_dtype, buffer=self._BR_shm.buf)
    @property
    def BZ(self):
        self._read()
        return ndarray(self._BZ_np_shape, dtype=self._BZ_np_dtype, buffer=self._BZ_shm.buf)
    @property
    def BPhi(self):
        self._read()
        return ndarray(self._BPhi_np_shape, dtype=self._BPhi_np_dtype, buffer=self._BPhi_shm.buf)
    
    
    def close(self):
        if self._have_read_or_not:
            self._R_shm.close()
            self._Z_shm.close()
            self._Phi_shm.close()
            self._BR_shm.close()
            self._BZ_shm.close()
            self._BPhi_shm.close()
        
    def unlink(self):
        if self._have_read_or_not:
            self._R_shm.unlink()
            self._Z_shm.unlink()
            self._Phi_shm.unlink()
            self._BR_shm.unlink()
            self._BZ_shm.unlink()
            self._BPhi_shm.unlink()

def CreateRegualrCylindricalGridFieldNumpySharedMemory(
    R:ndarray, Z:ndarray, Phi:ndarray,
    BR:ndarray, BZ:ndarray, BPhi:ndarray):
    
    R_shm = shared_memory.SharedMemory(create=True, size=R.nbytes)
    Z_shm = shared_memory.SharedMemory(create=True, size=Z.nbytes)
    Phi_shm = shared_memory.SharedMemory(create=True, size=Phi.nbytes)
    BR_shm = shared_memory.SharedMemory(create=True, size=BR.nbytes)
    BZ_shm = shared_memory.SharedMemory(create=True, size=BZ.nbytes)
    BPhi_shm = shared_memory.SharedMemory(create=True, size=BPhi.nbytes)
    
    shm_names = [R_shm.name, Z_shm.name, Phi_shm.name, BR_shm.name, BZ_shm.name, BPhi_shm.name]
    
    R_ = ndarray(R.shape, dtype=R.dtype, buffer=R_shm.buf)
    Z_ = ndarray(Z.shape, dtype=Z.dtype, buffer=Z_shm.buf)
    Phi_ = ndarray(Phi.shape, dtype=Phi.dtype, buffer=Phi_shm.buf)
    BR_ = ndarray(BR.shape, dtype=BR.dtype, buffer=BR_shm.buf)
    BZ_ = ndarray(BZ.shape, dtype=BZ.dtype, buffer=BZ_shm.buf)
    BPhi_ = ndarray(BPhi.shape, dtype=BPhi.dtype, buffer=BPhi_shm.buf)
    
    R_[:] = R
    Z_[:] = Z
    Phi_[:] = Phi
    BR_[:] = BR
    BZ_[:] = BZ
    BPhi_[:] = BPhi
    
    R_shm.close()
    Z_shm.close()
    Phi_shm.close()
    BR_shm.close()
    BZ_shm.close()
    BPhi_shm.close()
    
    return RegualrCylindricalGridFieldNumpySharedMemory(
        shm_names, 
        np_shapes=[arr.shape for arr in [R,Z,Phi,BR,BZ,BPhi]],
        np_dtypes=[arr.dtype for arr in [R,Z,Phi,BR,BZ,BPhi]]
    )