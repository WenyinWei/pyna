from pyna.field import RegualrCylindricalGridField

from scipy.interpolate import RegularGridInterpolator
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
    fltres = solve_ivp(dXRXZdPhi, [0.0, total_deltaPhi], initps_RZPhi_flattened, dense_output=True, *arg, **kwarg)
    fltres.sol.mat_interp = lambda t: fltres.sol.__call__(t).reshape( (pts_num, 3), order='F')
    fltres.phi_increasing = phi_increasing
    return fltres