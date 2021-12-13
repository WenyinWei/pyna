from pyna.diff.fieldline import _FieldDifferenatiableRZ
from pyna.diff.fixedpoint import Jac_evolution_along_Xcycle
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
        

def grow_manifold_from_Xcycle(R, Z, Phi, BR, BZ, BPhi, Xcycle_RZdiff, S_span, S_num:int, Phi_span, Phi_num:int, S_init = 2e-2):

    DP_Xcycle = Jac_evolution_along_Xcycle(R, Z, Phi, BR, BZ, BPhi, Xcycle_RZdiff, Phi_span)

    Phi_start, Phi_end = Phi_span[0], Phi_span[1]
    dPhi = Phi[1]-Phi[0]
    Phi_manifold = np.linspace(Phi_start, Phi_end, num=Phi_num, endpoint=True)
    nPhi_manifold = len(Phi_manifold)
    dPhi_manifold = Phi_manifold[1]-Phi_manifold[0]

    RZ_Xcycle_arr = Xcycle_RZdiff[0].sol(Phi_manifold).T
    DP_Xcycle_arr = DP_Xcycle.sol(Phi_manifold).T.reshape( (len(Phi_manifold),2,2) )
    import numpy.linalg
    eigen_val_Xcycle, eigen_vec_Xcycle = numpy.linalg.eig(DP_Xcycle_arr)

    RBRdBPhi = R[:,None,None]*BR/BPhi
    RBZdBPhi = R[:,None,None]*BZ/BPhi
    RBRdBPhi_field = _FieldDifferenatiableRZ(RBRdBPhi, R, Z, Phi)
    RBZdBPhi_field = _FieldDifferenatiableRZ(RBZdBPhi, R, Z, Phi)

    def manifold_growth_ODE(t,y):
        
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


    RZ_Xcycle_bitshift_along_eigvec = RZ_Xcycle_arr + S_init * eigen_vec_Xcycle[:,:,0] # the first eigen vec direction
    RZ_Xcycle_bitshift_along_eigvec = RZ_Xcycle_bitshift_along_eigvec[:-1,:] # Remove the last repeated element

    manifold_sol = solve_ivp(manifold_growth_ODE, S_span, 
                    RZ_Xcycle_bitshift_along_eigvec.reshape( (2*(nPhi_manifold-1),), order='F' ), dense_output=True, 
                    first_step=0.5e-4, max_step=1e-4
                            )

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