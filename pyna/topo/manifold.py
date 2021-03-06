from pyna.diff.fieldline import _FieldDifferenatiableRZ
from pyna.diff.fixedpoint import Jac_evolution_along_Xcycle, eigvec_interpolator_along_Xcycle
from scipy.integrate import solve_ivp
import numpy as np



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
        
def grow_manifold_from_Xcycle_eig_interp(R, Z, Phi, BR, BZ, BPhi, Xcycle_RZdiff, Jac_evosol_along_Xcycle, eigind, S_span, S_num:int, Phi_span, Phi_num:int, rev_eigvec=False, first_step=5e-5, max_step=1e-4):
    
    Phi_start, Phi_end = Phi_span[0], Phi_span[1]
    dPhi = Phi[1]-Phi[0]
    Phi_manifold = np.linspace(Phi_start, Phi_end, num=Phi_num, endpoint=True)
    nPhi_manifold = len(Phi_manifold)
    dPhi_manifold = Phi_manifold[1]-Phi_manifold[0]

    RZ_Xcycle_arr = Xcycle_RZdiff[0].sol(Phi_manifold).T
    eigval_interp, eigvec_interp = eigvec_interpolator_along_Xcycle(Jac_evosol_along_Xcycle)
    eigvec_Xcycle = eigvec_interp(Phi_manifold)[:,:,eigind]
    if eigval_interp(Xcycle_RZdiff[0].t[0])[eigind] > 1.0: # Unstable manifold of Poincare map
        StaOrNot = False
    else: # Stable manifold of Poincare map
        StaOrNot = True

    RBRdBPhi = R[:,None,None]*BR/BPhi
    RBZdBPhi = R[:,None,None]*BZ/BPhi
    RBRdBPhi_field = _FieldDifferenatiableRZ(RBRdBPhi, R, Z, Phi)
    RBZdBPhi_field = _FieldDifferenatiableRZ(RBZdBPhi, R, Z, Phi)

    def Usta_manifold_growth_ODE(t,y):
        
        XR_each_phi, XZ_each_phi = y[:nPhi_manifold-1], y[nPhi_manifold-1:]
        dXRdPhi = _central_finite_difference_first_derivative(XR_each_phi, dPhi_manifold)
        dXZdPhi = _central_finite_difference_first_derivative(XZ_each_phi, dPhi_manifold)
        
        try:
            sampled_RBRdBPhi = RBRdBPhi_field.diff_RZ_interpolator(0,0)( np.vstack( (XR_each_phi, XZ_each_phi, Phi_manifold[:-1]%(2*np.pi) ) ).T )
            sampled_RBZdBPhi = RBZdBPhi_field.diff_RZ_interpolator(0,0)( np.vstack( (XR_each_phi, XZ_each_phi, Phi_manifold[:-1]%(2*np.pi) ) ).T )
        except Exception as e:
            raise RuntimeError(f"Error when growing the manifold from X cycle at {t}. The error is {e}")
        dsdPhi = np.sqrt(  (sampled_RBRdBPhi-dXRdPhi)**2 + (sampled_RBZdBPhi-dXZdPhi)**2 ) # as denominator of dXRds and dXZds expressions
        dXRds = (sampled_RBRdBPhi-dXRdPhi) / dsdPhi
        dXZds = (sampled_RBZdBPhi-dXZdPhi) / dsdPhi
        return np.concatenate( (dXRds, dXZds), axis=0)
    def Sta_manifold_growth_ODE(t,y):
        
        XR_each_phi, XZ_each_phi = y[:nPhi_manifold-1], y[nPhi_manifold-1:]
        dXRdPhi = _central_finite_difference_first_derivative(XR_each_phi, dPhi_manifold)
        dXZdPhi = _central_finite_difference_first_derivative(XZ_each_phi, dPhi_manifold)
        
        try:
            sampled_RBRdBPhi = RBRdBPhi_field.diff_RZ_interpolator(0,0)( np.vstack( (XR_each_phi, XZ_each_phi, Phi_manifold[:-1]%(2*np.pi) ) ).T )
            sampled_RBZdBPhi = RBZdBPhi_field.diff_RZ_interpolator(0,0)( np.vstack( (XR_each_phi, XZ_each_phi, Phi_manifold[:-1]%(2*np.pi) ) ).T )
        except Exception as e:
            raise RuntimeError(f"Error when growing the manifold from X cycle at {t}. The error is {e}")
        dsdPhi = np.sqrt(  (sampled_RBRdBPhi-dXRdPhi)**2 + (sampled_RBZdBPhi-dXZdPhi)**2 ) # as denominator of dXRds and dXZds expressions
        dXRds = (sampled_RBRdBPhi-dXRdPhi) / dsdPhi
        dXZds = (sampled_RBZdBPhi-dXZdPhi) / dsdPhi
        return -np.concatenate( (dXRds, dXZds), axis=0)

    if not rev_eigvec:
        RZ_Xcycle_bitshift_along_eigvec = RZ_Xcycle_arr + S_span[0] * eigvec_Xcycle
    else:
        RZ_Xcycle_bitshift_along_eigvec = RZ_Xcycle_arr - S_span[0] * eigvec_Xcycle
    RZ_Xcycle_bitshift_along_eigvec = RZ_Xcycle_bitshift_along_eigvec[:-1,:] # Remove the last repeated element

    if StaOrNot:
        manifold_sol = solve_ivp(Sta_manifold_growth_ODE, S_span, 
                        RZ_Xcycle_bitshift_along_eigvec.reshape( (2*(nPhi_manifold-1),), order='F' ), 
                        dense_output=True, first_step=first_step, max_step=max_step)
    else:
        manifold_sol = solve_ivp(Usta_manifold_growth_ODE, S_span, 
                        RZ_Xcycle_bitshift_along_eigvec.reshape( (2*(nPhi_manifold-1),), order='F' ), 
                        dense_output=True, first_step=first_step, max_step=max_step)

    S_arr = np.empty( (S_num) )
    S_arr[0], S_arr[1:] = 0.0, np.linspace(S_span[0], S_span[1], num=S_num-1)
    # (2, s_len, nPhi_manifold), 2 for (XR, XZ), s_len for len(s_arr), nPhi_manifold for 
    manifold_RZ_SPhi = np.empty( (2, S_num, nPhi_manifold) ) 
    # initial cycle whose s = 0
    manifold_RZ_SPhi[:,0,:] = RZ_Xcycle_arr.T
    manifold_RZ_SPhi[:,1:S_num,:nPhi_manifold-1] = manifold_sol.sol(S_arr[1:]).reshape( (nPhi_manifold-1,2,S_num-1), order='F' ).transpose(1,2,0)
    # seam the head and tail of manifold
    manifold_RZ_SPhi[:,:,-1] = manifold_RZ_SPhi[:,:,0]
    return S_arr, Phi_manifold, manifold_RZ_SPhi


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

def grow_manifold_from_Xcycle(R, Z, Phi, BR, BZ, BPhi, Xcycle_RZdiff, Xcycle_eig_theta, S_span, S_num:int, Phi_span, Phi_num:int, rev_eigvec=False, first_step=5e-5, max_step=1e-4):

    Phi_start, Phi_end = Phi_span[0], Phi_span[1]
    dPhi = Phi[1]-Phi[0]
    Phi_manifold = np.linspace(Phi_start, Phi_end, num=Phi_num, endpoint=True)
    nPhi_manifold = len(Phi_manifold)
    dPhi_manifold = Phi_manifold[1]-Phi_manifold[0]

    RZ_Xcycle_arr = Xcycle_RZdiff[0].sol(Phi_manifold).T
    eigentheta_Xcycle = Xcycle_eig_theta.sol(Phi_manifold)[0,:]
    if Xcycle_eig_theta.sol(Phi_manifold)[1,0] > 1.0: # Unstable manifold of Poincare map
        StaOrNot = False
    else: # Stable manifold of Poincare map
        StaOrNot = True


    RBRdBPhi = R[:,None,None]*BR/BPhi
    RBZdBPhi = R[:,None,None]*BZ/BPhi
    RBRdBPhi_field = _FieldDifferenatiableRZ(RBRdBPhi, R, Z, Phi)
    RBZdBPhi_field = _FieldDifferenatiableRZ(RBZdBPhi, R, Z, Phi)

    def Usta_manifold_growth_ODE(t,y):
        
        XR_each_phi, XZ_each_phi = y[:nPhi_manifold-1], y[nPhi_manifold-1:]
        dXRdPhi = _central_finite_difference_first_derivative(XR_each_phi, dPhi_manifold)
        dXZdPhi = _central_finite_difference_first_derivative(XZ_each_phi, dPhi_manifold)
        
        try:
            sampled_RBRdBPhi = RBRdBPhi_field.diff_RZ_interpolator(0,0)( np.vstack( (XR_each_phi, XZ_each_phi, Phi_manifold[:-1]) ).T )
            sampled_RBZdBPhi = RBZdBPhi_field.diff_RZ_interpolator(0,0)( np.vstack( (XR_each_phi, XZ_each_phi, Phi_manifold[:-1]) ).T )
        except Exception as e:
            raise RuntimeError(f"Error when growing the manifold from X cycle at {t}. The error is {e}")
        dsdPhi = np.sqrt(  (sampled_RBRdBPhi-dXRdPhi)**2 + (sampled_RBZdBPhi-dXZdPhi)**2 ) # as denominator of dXRds and dXZds expressions
        dXRds = (sampled_RBRdBPhi-dXRdPhi) / dsdPhi
        dXZds = (sampled_RBZdBPhi-dXZdPhi) / dsdPhi
        return np.concatenate( (dXRds, dXZds), axis=0)
    def Sta_manifold_growth_ODE(t,y):
        
        XR_each_phi, XZ_each_phi = y[:nPhi_manifold-1], y[nPhi_manifold-1:]
        dXRdPhi = _central_finite_difference_first_derivative(XR_each_phi, dPhi_manifold)
        dXZdPhi = _central_finite_difference_first_derivative(XZ_each_phi, dPhi_manifold)
        
        try:
            sampled_RBRdBPhi = RBRdBPhi_field.diff_RZ_interpolator(0,0)( np.vstack( (XR_each_phi, XZ_each_phi, Phi_manifold[:-1]) ).T )
            sampled_RBZdBPhi = RBZdBPhi_field.diff_RZ_interpolator(0,0)( np.vstack( (XR_each_phi, XZ_each_phi, Phi_manifold[:-1]) ).T )
        except Exception as e:
            raise RuntimeError(f"Error when growing the manifold from X cycle at {t}. The error is {e}")
        dsdPhi = np.sqrt(  (sampled_RBRdBPhi-dXRdPhi)**2 + (sampled_RBZdBPhi-dXZdPhi)**2 ) # as denominator of dXRds and dXZds expressions
        dXRds = (sampled_RBRdBPhi-dXRdPhi) / dsdPhi
        dXZds = (sampled_RBZdBPhi-dXZdPhi) / dsdPhi
        return -np.concatenate( (dXRds, dXZds), axis=0)

    if not rev_eigvec:
        RZ_Xcycle_bitshift_along_eigvec = RZ_Xcycle_arr + S_span[0] * np.array([np.cos( eigentheta_Xcycle ) , np.sin( eigentheta_Xcycle )] ).T
    else:
        RZ_Xcycle_bitshift_along_eigvec = RZ_Xcycle_arr + S_span[0] * np.array([np.cos( eigentheta_Xcycle+np.pi ) , np.sin( eigentheta_Xcycle+np.pi )] ).T
    RZ_Xcycle_bitshift_along_eigvec = RZ_Xcycle_bitshift_along_eigvec[:-1,:] # Remove the last repeated element

    if StaOrNot:
        manifold_sol = solve_ivp(Sta_manifold_growth_ODE, S_span, 
                        RZ_Xcycle_bitshift_along_eigvec.reshape( (2*(nPhi_manifold-1),), order='F' ), 
                        dense_output=True, first_step=first_step, max_step=max_step)
    else:
        manifold_sol = solve_ivp(Usta_manifold_growth_ODE, S_span, 
                        RZ_Xcycle_bitshift_along_eigvec.reshape( (2*(nPhi_manifold-1),), order='F' ), 
                        dense_output=True, first_step=first_step, max_step=max_step)

    S_arr = np.empty( (S_num) )
    S_arr[0], S_arr[1:] = 0.0, np.linspace(S_span[0], S_span[1], num=S_num-1)
    # (2, s_len, nPhi_manifold), 2 for (XR, XZ), s_len for len(s_arr), nPhi_manifold for 
    manifold_RZ_SPhi = np.empty( (2, S_num, nPhi_manifold) ) 
    # initial cycle whose s = 0
    manifold_RZ_SPhi[:,0,:] = RZ_Xcycle_arr.T
    manifold_RZ_SPhi[:,1:S_num,:nPhi_manifold-1] = manifold_sol.sol(S_arr[1:]).reshape( (nPhi_manifold-1,2,S_num-1), order='F' ).transpose(1,2,0)
    # seam the head and tail of manifold
    manifold_RZ_SPhi[:,:,-1] = manifold_RZ_SPhi[:,:,0]
    return S_arr, Phi_manifold, manifold_RZ_SPhi