from pyna.field import RegualrCylindricalGridField


from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import OdeSolution
def _mat_interp(self, t):
    return self.__call__(t).reshape( (self.pts_num, 3), order='F')
OdeSolution.mat_interp = _mat_interp
from scipy.integrate import solve_ivp

import numpy as np


def bundle_tracing_with_t_as_DeltaPhi(afield:RegualrCylindricalGridField, total_deltaPhi, initpts_RZPhi, phi_increasing:bool, *arg, **kwarg):
    R, Z, Phi, BR, BZ, BPhi = afield.R, afield.Z, afield.Phi, afield.BR, afield.BZ, afield.BPhi
    RBRdBPhi = R[:,None,None]*BR/BPhi
    RBZdBPhi = R[:,None,None]*BZ/BPhi
    RBRdBPhi_interp = RegularGridInterpolator( 
        (R, Z, Phi), RBRdBPhi[...,:,:,:],
        method="linear", bounds_error=True )
    RBZdBPhi_interp = RegularGridInterpolator( 
        (R, Z, Phi), RBZdBPhi[...,:,:,:],
        method="linear", bounds_error=True )
    
    pts_num = initpts_RZPhi.shape[0] 
    initps_RZPhi_flattened = np.reshape(initpts_RZPhi, (-1), order='F') # (initpoints_num, 3) -> (initpoints_num*3) where the first initpoints_num are R coordinates, the second initpoints_num are Z coordinates, the third initpoints_num are Phi coordinates.

    dPhidPhi = np.ones((pts_num))
    if phi_increasing:
        def dXRXZdPhi(t, y):
            R_ = y[:pts_num]
            Z_ = y[pts_num:2*pts_num]
            Phi_ = y[2*pts_num:3*pts_num] % (2*np.pi)
            pts_RZPhi = np.stack( (R_, Z_, Phi_) , axis=1)  # reshaped to (pts_num, 3)
            dXRdPhi = RBRdBPhi_interp(pts_RZPhi) 
            dXZdPhi = RBZdBPhi_interp(pts_RZPhi) 
            return np.concatenate((dXRdPhi, dXZdPhi, dPhidPhi))
    else:
        def dXRXZdPhi(t, y):
            R_ = y[:pts_num]
            Z_ = y[pts_num:2*pts_num]
            Phi_ = y[2*pts_num:3*pts_num] % (2*np.pi)
            pts_RZPhi = np.stack( (R_, Z_, Phi_) , axis=1)  # reshaped to (pts_num, 3)
            dXRdPhi =-RBRdBPhi_interp(pts_RZPhi) 
            dXZdPhi =-RBZdBPhi_interp(pts_RZPhi) 
            return np.concatenate((dXRdPhi, dXZdPhi,-dPhidPhi))
        
    def out_of_grid(t, y):
        R_, Z_ = y[:pts_num], y[pts_num:2*pts_num]
        R_max, R_min = max(R_), min(R_)
        Z_max, Z_min = max(Z_), min(Z_)
        return min( 
            R_min - R[1], R[-2] - R_max, 
            Z_min - Z[1], Z[-2] - Z_max, )
    out_of_grid.terminal = True
    
    fltres = solve_ivp(
        dXRXZdPhi, 
        [0.0, total_deltaPhi], 
        initps_RZPhi_flattened, events=out_of_grid, dense_output=True, *arg, **kwarg)
    
    fltres.sol.pts_num = pts_num
    # def mat_interp(self, t):
    #     return self.__call__(t).reshape( (self.pts_num, 3), order='F')
    # fltres.sol.mat_interp = mat_interp # lambda t: fltres.sol.__call__(t).reshape( (pts_num, 3), order='F')
    fltres.phi_increasing = phi_increasing
    return fltres

def save_Poincare_orbits(filename:str, list_of_arrRZPhi):
    np.savez(filename, *list_of_arrRZPhi)
def load_Poincare_orbits(filename:str):
    Poincare_orbits_list = [ ]
    Poincare_orbits_npz = np.load(filename)
    for var in Poincare_orbits_npz.files:
        Poincare_orbits_list.append( Poincare_orbits_npz[var] )
    return Poincare_orbits_list
# def read_Poincare_orbits(filename:str): # The old reading function, which reads the data into a list of lists of numpy array shape [3]
#     Poincare_orbits_list = [ ]
#     Poincare_orbits_npz = np.load(filename)
#     for var in Poincare_orbits_npz.files:
#         Poincare_orbits_list.append([])
#         for i in range(Poincare_orbits_npz[var].shape[0]):
#             Poincare_orbits_list[-1].append(Poincare_orbits_npz[var][i,:])
#     return Poincare_orbits_list