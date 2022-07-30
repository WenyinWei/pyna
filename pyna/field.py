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
    
from numpy import ndarray
from multiprocessing import shared_memory
class RegualrCylindricalGridFieldNumpySharedMemory(RegualrCylindricalGridField):
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
    print("shm_names:", shm_names)
    
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