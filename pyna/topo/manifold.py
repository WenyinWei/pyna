from mimetypes import init
from pyna.field import RegualrCylindricalGridField
from pyna.flt import bundle_tracing_with_t_as_DeltaPhi
from pyna.diff.fieldline import _FieldDifferenatiableRZ
from pyna.diff.cycle import eigvec_interpolator_along_Xcycle

from multiprocessing.sharedctypes import Value
from scipy.integrate import solve_ivp
import numpy as np
from numpy import ndarray


def grow_manifold_from_Xcycle_naive_init_segment(
    afield:RegualrCylindricalGridField, 
    Xcycle_RZdiff, 
    Jac_evosol_along_Xcycle, 
    eigind:int, Phi_span, total_deltaPhi:float, ptsnum_initseg:int=300, initseg_len=0.8e-4, *arg_sovle_ivp, **kwarg_solve_ivp):    
    Phi_start, Phi_end = Phi_span[0], Phi_span[1]

    # Use eigind : 1, 2, 3, 4. 1 & 3 are on the contrary, 2 & 4 are on the contrary.
    if eigind in [1, 2]:
        eigind_equiv = eigind 
    elif eigind in [3, 4]: 
        eigind_equiv = eigind - 2
    else:
        raise ValueError("eigind should be either 1, 2, 3, or 4.")
    eigval_interp, eigvec_interp = eigvec_interpolator_along_Xcycle(Jac_evosol_along_Xcycle)

    Xcycle_eigvec = eigvec_interp(Phi_start)[:,eigind_equiv-1]
    Xcycle_RZ = Xcycle_RZdiff[0].sol(Phi_start) # in shape [2,]
    if eigind in [1, 2]:
        eigvec_rev_or_not = 1.0
    elif eigind in [3, 4]: 
        eigvec_rev_or_not =-1.0
    
    initpts_RZPhi = np.empty( (ptsnum_initseg, 3) )
    initpts_RZPhi[:,2] = Phi_start
    eigval = eigval_interp(Phi_start)[eigind_equiv-1]
    if eigval > 1.0: # Unstable manifold of Phi-increasing Poincare map
        initpts_RZPhi[:,:2] = Xcycle_RZ + eigvec_rev_or_not * initseg_len * np.linspace(1.0,     eigval, num=ptsnum_initseg, endpoint=False)[:,None] * Xcycle_eigvec[None,:]
        fltres = bundle_tracing_with_t_as_DeltaPhi(afield, total_deltaPhi, initpts_RZPhi, phi_increasing=True, *arg_sovle_ivp, **kwarg_solve_ivp)
    elif 0.0 < eigval < 1.0: # Stable manifold of Phi-increasing Poincare map
        initpts_RZPhi[:,:2] = Xcycle_RZ + eigvec_rev_or_not * initseg_len * np.linspace(1.0, 1.0/eigval, num=ptsnum_initseg, endpoint=False)[:,None] * Xcycle_eigvec[None,:]
        fltres = bundle_tracing_with_t_as_DeltaPhi(afield, total_deltaPhi, initpts_RZPhi, phi_increasing=False, *arg_sovle_ivp, **kwarg_solve_ivp)
    else:
        raise NotImplementedError("We have not yet implemented the grow_manifold_from_Xcycle function for Mobiusian cycles.")

    return fltres

def grow_manifold_from_Xcycle_naive_carousel(afield:RegualrCylindricalGridField, Xcycle_RZdiff, Jac_evosol_along_Xcycle, eigind:int, Phi_span, W_nPhi:int, Ind_num:int, first_step=5e-5, max_step=1e-4):    
    Phi_start, Phi_end = Phi_span[0], Phi_span[1]
    W_Phi = np.linspace(Phi_start, Phi_end, num=W_nPhi, endpoint=True)
    W_dPhi = W_Phi[1]-W_Phi[0]
#     fltres_list = []
#     while True:
#         try:
#             fltres_list.append( 
#                 bundle_tracing_with_t_as_DeltaPhi(R,Z,Phi,BR,BZ,BPhi, 5.0, testarr) 
#             ) # fltres.sol.mat_interp(2.0)
#         except:
#             break

    # Use eigind : 1, 2, 3, 4. 1 & 3 are on the contrary, 2 & 4 are on the contrary.
    if eigind in [1, 2]:
        eigind_equiv = eigind 
    elif eigind in [3, 4]: 
        eigind_equiv = eigind - 2
    else:
        raise ValueError("eigind should be either 1, 2, 3, or 4.")
    eigval_interp, eigvec_interp = eigvec_interpolator_along_Xcycle(Jac_evosol_along_Xcycle)

    Xcycle_eigvec = eigvec_interp(W_Phi)[:,:,eigind_equiv-1]
    W_PhiInd_RZ = np.empty( (W_nPhi, Ind_num, 2) ) 
    Xcycle_RZ_arr = Xcycle_RZdiff[0].sol(W_Phi).T # in shape [W_nPhi, 2]
    if eigind in [1, 2]:
        Xcycle_bitshift_along_eigvec_RZ = Xcycle_RZ_arr + first_step * Xcycle_eigvec
    elif eigind in [3, 4]: 
        Xcycle_bitshift_along_eigvec_RZ = Xcycle_RZ_arr - first_step * Xcycle_eigvec
    W_PhiInd_RZ[:,0,:] = Xcycle_RZ_arr
    W_PhiInd_RZ[:,1,:] = Xcycle_bitshift_along_eigvec_RZ
    # FIXME: carousel does not guarantee the manifold points we acquired are ordered.
    # W_PhiInd_s = np.empty( (W_nPhi, Ind_num,) )
    # W_PhiInd_s[:,0], W_PhiInd_s[:,1] = 0.0, first_step # initial cycle s = 0, the bitshift cycle in the direction of eigenvector s = first_step
    
    total_DeltaPhi = Ind_num * W_dPhi
    initpts_RZPhi = np.stack( (W_PhiInd_RZ[:,1,0], W_PhiInd_RZ[:,1,1], W_Phi) , axis=1)[:-1,:] # in shape of [W_nPhi-1, 3]

    if eigval_interp(Phi_start)[eigind_equiv-1] > 1.0: # Unstable manifold of Phi-increasing Poincare map
        fltres = bundle_tracing_with_t_as_DeltaPhi(afield, total_DeltaPhi, initpts_RZPhi, phi_increasing=True, max_step=W_dPhi/3)
        for i in range(2, Ind_num):
            W_PhiInd_RZ[:-1,i,:] = np.roll( fltres.sol.mat_interp( (i-1)*W_dPhi )[:,:-1], i-1, axis=0 ) # in shape of [W_nPhi-1, 2]
    elif 0.0 < eigval_interp(Phi_start)[eigind_equiv-1] < 1.0: # Stable manifold of Phi-increasing Poincare map
        fltres = bundle_tracing_with_t_as_DeltaPhi(afield, total_DeltaPhi, initpts_RZPhi, phi_increasing=False, max_step=W_dPhi/3)
        for i in range(2, Ind_num):
            W_PhiInd_RZ[:-1,i,:] = np.roll( fltres.sol.mat_interp( (i-1)*W_dPhi )[:,:-1], -(i-1), axis=0 ) # in shape of [W_nPhi-1, 2]
    else:
        raise NotImplementedError("We have not yet implemented the grow_manifold_from_Xcycle function for Mobiusian cycles.")
    
    W_PhiInd_RZ[-1,:,:] = W_PhiInd_RZ[0,:,:] # seam the head and tail of manifold
    
    # for i in range(2, Ind_num):
    #     W_PhiInd_s[:,i] = W_PhiInd_s[:,i-1] + np.sqrt(
    #         (W_PhiInd_RZ[:,i,0] - W_PhiInd_RZ[:,i-1,0])**2 
    #       + (W_PhiInd_RZ[:,i,1] - W_PhiInd_RZ[:,i-1,1])**2)
    
    return W_Phi, W_PhiInd_RZ

def _central_finite_difference_first_derivative(arr:np.ndarray, dPhi:float, accuracy_order=4):
    """1st derivative of a periodic function with spacing dPhi

    Args:
        arr (np.ndarray): the function values sampled on a uniform mesh
        dPhi (float): the spacing of the mesh
        accuracy_order (int, optional): Defaults to 4.

    Returns:
        np.ndarray: 1st derivative of the periodic function on the uniform mesh

    Note: 
        The coefficients are from [Finite difference coefficient](https://en.wikipedia.org/wiki/Finite_difference_coefficient)
    """
    if accuracy_order == 2:
        return (np.roll(arr, 1) - np.roll(arr, -1) )  / (2*dPhi)
    elif accuracy_order == 4:
        return ( (np.roll(arr, 1) - np.roll(arr, -1) ) * ( 2 / 3 ) \
            + (np.roll(arr, 2) - np.roll(arr, -2) )  * ( -1 / 12 ) ) / dPhi
    elif accuracy_order == 6:
        return ( (np.roll(arr, 1) - np.roll(arr, -1) ) * ( 3 / 4 ) \
            + (np.roll(arr, 2) - np.roll(arr, -2) )  * ( -3 / 20 ) \
            + (np.roll(arr, 3) - np.roll(arr, -3) )  * ( 1 / 60 )  ) / dPhi
    elif accuracy_order == 8:
        return ( (np.roll(arr, 1) - np.roll(arr, -1) ) * ( 4 / 5 ) \
            + (np.roll(arr, 2) - np.roll(arr, -2) )  * ( -1 / 5 ) \
            + (np.roll(arr, 3) - np.roll(arr, -3) )  * ( 4 / 105 ) \
            + (np.roll(arr, 4) - np.roll(arr, -4) )  * ( -1 / 280 )  ) / dPhi

from deprecated import deprecated
def grow_manifold_from_Xcycle_eig_interp(afield:RegualrCylindricalGridField, Xcycle_RZdiff, Jac_evosol_along_Xcycle, eigind, S_span, S_num:int, Phi_span, Phi_num:int, rev_eigvec=False, first_step=5e-5, max_step=1e-4):
    R, Z, Phi, BR, BZ, BPhi = afield.R, afield.Z, afield.Phi, afield.BR, afield.BZ, afield.BPhi

    Phi_start, Phi_end = Phi_span[0], Phi_span[1]
    dPhi = Phi[1]-Phi[0]
    W_Phi = np.linspace(Phi_start, Phi_end, num=Phi_num, endpoint=True)
    W_nPhi = len(W_Phi)
    W_dPhi = W_Phi[1]-W_Phi[0]

    RZ_Xcycle_arr = Xcycle_RZdiff[0].sol(W_Phi).T
    eigval_interp, eigvec_interp = eigvec_interpolator_along_Xcycle(Jac_evosol_along_Xcycle)
    eigvec_Xcycle = eigvec_interp(W_Phi)[:,:,eigind]
    if eigval_interp(Phi_start)[eigind] > 1.0: # Unstable manifold of Poincare map
        StaOrNot = False
    else: # Stable manifold of Poincare map
        StaOrNot = True

    RBRdBPhi = R[:,None,None]*BR/BPhi
    RBZdBPhi = R[:,None,None]*BZ/BPhi
    RBRdBPhi_field = _FieldDifferenatiableRZ(RBRdBPhi, R, Z, Phi)
    RBZdBPhi_field = _FieldDifferenatiableRZ(RBZdBPhi, R, Z, Phi)

    def manifold_growth_ODE(t,y):
        
        XR_each_phi, XZ_each_phi = y[:W_nPhi-1], y[W_nPhi-1:]
        pXRpPhi = _central_finite_difference_first_derivative(XR_each_phi, W_dPhi)
        pXZpPhi = _central_finite_difference_first_derivative(XZ_each_phi, W_dPhi)
        
        try:
            RBRdBPhi_oncycle = RBRdBPhi_field.diff_RZ_interpolator(0,0)( np.vstack( (XR_each_phi, XZ_each_phi, W_Phi[:-1]%(2*np.pi) ) ).T )
            RBZdBPhi_oncycle = RBZdBPhi_field.diff_RZ_interpolator(0,0)( np.vstack( (XR_each_phi, XZ_each_phi, W_Phi[:-1]%(2*np.pi) ) ).T )
        except Exception as e:
            raise RuntimeError(f"Error when growing the manifold from X cycle at {t}. The error is {e}")
        dsdPhi = np.sqrt(  (RBRdBPhi_oncycle-pXRpPhi)**2 + (RBZdBPhi_oncycle-pXZpPhi)**2 ) # as denominator of pXRps and pXZps expressions
        pXRps = (RBRdBPhi_oncycle-pXRpPhi) / dsdPhi
        pXZps = (RBZdBPhi_oncycle-pXZpPhi) / dsdPhi
        return np.concatenate( (pXRps, pXZps), axis=0)

    if not rev_eigvec:
        RZ_Xcycle_bitshift_along_eigvec = RZ_Xcycle_arr + S_span[0] * eigvec_Xcycle
    else:
        RZ_Xcycle_bitshift_along_eigvec = RZ_Xcycle_arr - S_span[0] * eigvec_Xcycle
    RZ_Xcycle_bitshift_along_eigvec = RZ_Xcycle_bitshift_along_eigvec[:-1,:] # Remove the last repeated element

    if StaOrNot:
        W_sol = solve_ivp(-manifold_growth_ODE, S_span, 
                        RZ_Xcycle_bitshift_along_eigvec.reshape( (2*(W_nPhi-1),), order='F' ), 
                        dense_output=True, first_step=first_step, max_step=max_step)
    else:
        W_sol = solve_ivp(manifold_growth_ODE, S_span, 
                        RZ_Xcycle_bitshift_along_eigvec.reshape( (2*(W_nPhi-1),), order='F' ), 
                        dense_output=True, first_step=first_step, max_step=max_step)

    S_arr = np.empty( (S_num) )
    S_arr[0], S_arr[1:] = 0.0, np.linspace(S_span[0], S_span[1], num=S_num-1)
    # (2, s_len, nPhi_manifold), 2 for (XR, XZ), s_len for len(s_arr), nPhi_manifold for 
    W_RZ_SPhi = np.empty( (2, S_num, W_nPhi) ) 
    # initial cycle whose s = 0
    W_RZ_SPhi[:,0,:] = RZ_Xcycle_arr.T
    W_RZ_SPhi[:,1:S_num,:W_nPhi-1] = W_sol.sol(S_arr[1:]).reshape( (W_nPhi-1,2,S_num-1), order='F' ).transpose(1,2,0)
    # seam the head and tail of manifold
    W_RZ_SPhi[:,:,-1] = W_RZ_SPhi[:,:,0]
    return S_arr, W_Phi, W_RZ_SPhi


def smooth1D(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal. This function is borrowed 
    from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.") 


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def _transect_initPhi0_Wivp_at_a_phi(Wivp_bundle, phi:float, mturn:int):
    """assert Phi_span of bundle_tracing starts from 0.0
    """
    import numpy as np
    phi %= 2*mturn*np.pi
    
    ordered_seglist = []
    if Wivp_bundle.phi_increasing:
        t = phi
        while Wivp_bundle.t.min() <= t <= Wivp_bundle.t.max():
            ordered_seglist.append(Wivp_bundle.sol.mat_interp( t ))
            t += 2*mturn*np.pi
    else:
        t = 2*mturn*np.pi - phi
        while Wivp_bundle.t.min() <= t <= Wivp_bundle.t.max():
            ordered_seglist.append(Wivp_bundle.sol.mat_interp( t ))
            t += 2*mturn*np.pi
    W1d_RZPhi = np.concatenate(ordered_seglist , axis=0)
    return W1d_RZPhi



def accumulate_s_from_RZ_arr(W1d_RZ:ndarray):
    W1d_s = np.empty( (W1d_RZ.shape[0]) )
    W1d_s[0] = 0.0
    W1d_s[1:] = np.add.accumulate(
        np.sqrt(
            (W1d_RZ[1:,0] - W1d_RZ[:-1,0])**2 + (W1d_RZ[1:,1] - W1d_RZ[:-1,1])**2
        ))
    return W1d_s


def create_W1d_interpolator_s_to_RZdRZds(
    afield:RegualrCylindricalGridField, 
    Wivp_bundle, phi:float, phi_epsilon:float, mturn=int):
    R, Z, Phi, BR, BZ, BPhi = afield.R, afield.Z, afield.Phi, afield.BR, afield.BZ, afield.BPhi
    
    W1d_phi0_RZPhi = _transect_initPhi0_Wivp_at_a_phi(Wivp_bundle, phi+                        0.0, mturn)
    W1d_phip_RZPhi = _transect_initPhi0_Wivp_at_a_phi(Wivp_bundle, phi+                phi_epsilon, mturn)
    W1d_phim_RZPhi = _transect_initPhi0_Wivp_at_a_phi(Wivp_bundle, phi+2*(mturn)*np.pi-phi_epsilon, mturn)

    W1d_phi0_s = accumulate_s_from_RZ_arr(W1d_phi0_RZPhi[:,:-1])
    W1d_phip_s = accumulate_s_from_RZ_arr(W1d_phip_RZPhi[:,:-1])
    W1d_phim_s = accumulate_s_from_RZ_arr(W1d_phim_RZPhi[:,:-1])

    from scipy.interpolate import interp1d
    W1d_phi0_s_interp_R, W1d_phi0_s_interp_Z = interp1d(W1d_phi0_s, W1d_phi0_RZPhi[:,0], ), interp1d(W1d_phi0_s, W1d_phi0_RZPhi[:,1], )
    W1d_phip_s_interp_R, W1d_phip_s_interp_Z = interp1d(W1d_phip_s, W1d_phip_RZPhi[:,0], ), interp1d(W1d_phip_s, W1d_phip_RZPhi[:,1], )
    W1d_phim_s_interp_R, W1d_phim_s_interp_Z = interp1d(W1d_phim_s, W1d_phim_RZPhi[:,0], ), interp1d(W1d_phim_s, W1d_phim_RZPhi[:,1], )

    RBRoBPhi = R[:,None,None]*BR/BPhi
    RBZoBPhi = R[:,None,None]*BZ/BPhi
    RBRoBPhi_field = _FieldDifferenatiableRZ(RBRoBPhi, R, Z, Phi)
    RBZoBPhi_field = _FieldDifferenatiableRZ(RBZoBPhi, R, Z, Phi)
    
    def _interpolator(s:ndarray):
        x, y = W1d_phi0_s_interp_R(s), W1d_phi0_s_interp_Z(s)
        dx = RBRoBPhi_field.diff_RZ_interpolator(0,0)( np.stack([x, y, phi*np.ones_like(x) % (2*np.pi) ], axis=-1) ) - (W1d_phip_s_interp_R(s)-W1d_phim_s_interp_R(s))/(2*phi_epsilon)
        dy = RBZoBPhi_field.diff_RZ_interpolator(0,0)( np.stack([x, y, phi*np.ones_like(x) % (2*np.pi) ], axis=-1) ) - (W1d_phip_s_interp_Z(s)-W1d_phim_s_interp_Z(s))/(2*phi_epsilon)
        dl = (dx**2+dy**2)**(1/2)
        dx/= dl
        dy/= dl
        return np.stack( [x,y,dx,dy], axis=-1 )
    return _interpolator