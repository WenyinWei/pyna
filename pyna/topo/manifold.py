
from pyna.diff.fieldline import _FieldDifferenatiableRZ
from scipy.integrate import solve_ivp
import numpy as np



def central_finite_difference_first_derivative(arr:np.ndarray, dPhi:float, accuracy_order=4):
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
        

def solve_manifold_growth(R, Z, Phi, BR, BZ, BPhi):
    RBRdBPhi = R[:,None,None]*BR/BPhi
    RBZdBPhi = R[:,None,None]*BZ/BPhi
    RBRdBPhi_field = _FieldDifferenatiableRZ(RBRdBPhi, R, Z, Phi)
    RBZdBPhi_field = _FieldDifferenatiableRZ(RBZdBPhi, R, Z, Phi)

    def manifold_growth_ODE(t,y):
        
        XR_each_phi, XZ_each_phi = y[:nPhi_manifold-1], y[nPhi_manifold-1:]
    #     dXRdPhi = (np.roll(XR_each_phi, 1) - np.roll(XR_each_phi, -1) )  / (2*dPhi_manifold)
    #     dXZdPhi = (np.roll(XZ_each_phi, 1) - np.roll(XZ_each_phi, -1) )  / (2*dPhi_manifold)
        dXRdPhi = (np.roll(XR_each_phi, 1) - np.roll(XR_each_phi, -1) )  / (dPhi_manifold) * 2 / 3 \
            - (np.roll(XR_each_phi, 2) - np.roll(XR_each_phi, -2) )  / (dPhi_manifold) * 1 / 12 
        dXZdPhi = (np.roll(XZ_each_phi, 1) - np.roll(XZ_each_phi, -1) )  / (dPhi_manifold) * 2 / 3 \
            - (np.roll(XZ_each_phi, 2) - np.roll(XZ_each_phi, -2) )  / (dPhi_manifold) * 1 / 12 
        
        try:
            sampled_RBRdBPhi = RBRdBPhi_field.diff_RZ_interpolator(0,0)( np.vstack( (XR_each_phi, XZ_each_phi, Phi_manifold[:-1]) ).T )
            sampled_RBZdBPhi = RBZdBPhi_field.diff_RZ_interpolator(0,0)( np.vstack( (XR_each_phi, XZ_each_phi, Phi_manifold[:-1]) ).T )
        except:
            print("error at ", t,)
            print(XR_each_phi)
            print(XZ_each_phi)
            print(Phi)
        dsdPhi = np.sqrt(  (sampled_RBRdBPhi-dXRdPhi)**2 + (sampled_RBZdBPhi-dXZdPhi)**2 ) # as denominator of dXRds and dXZds expressions
        dXRds = (sampled_RBRdBPhi-dXRdPhi) / dsdPhi
        dXZds = (sampled_RBZdBPhi-dXZdPhi) / dsdPhi
        return np.concatenate( (dXRds, dXZds), axis=0)


    s_span = [0.02, 2.5]
    RZ_Xcycle_bitshift_along_eigvec = RZ_Xcycle_arr + s_span[0] * eigen_vec_Xcycle[:,:,0] # the first eigen vec direction
    RZ_Xcycle_bitshift_along_eigvec = RZ_Xcycle_bitshift_along_eigvec[:-1,:] # Remove the last repeated element

    manifold_sol = solve_ivp(manifold_growth_ODE, s_span, 
                    RZ_Xcycle_bitshift_along_eigvec.reshape( (2*(nPhi_manifold-1),), order='F' ), dense_output=True, 
                    first_step=0.5e-4, max_step=1e-4
                            )