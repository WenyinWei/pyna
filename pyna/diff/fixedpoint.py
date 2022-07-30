from random import sample
from pyna.diff.fieldline import _FieldDifferenatiableRZ
from pyna.diff.fieldline import RZ_partial_derivative_of_map_4_Flow_Phi_as_t
from pyna.field import RegualrCylindricalGridField

import numpy as np
from numpy import linalg as LA 
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

def Newton_discrete(afield:RegualrCylindricalGridField, 
                    x0_RZPhi, tor_turn:int=1, zone_of_interest_callback=None, h=1.0, epsilon=1e-5, ret_trace=False):
    R, Z, Phi, BR, BZ, BPhi = afield.R, afield.Z, afield.Phi, afield.BR, afield.BZ, afield.BPhi
    
    xRZ_trace = [np.asarray(x0_RZPhi[:-1])]
    while len(xRZ_trace)==1 or np.linalg.norm( xRZ_trace[-2] - xRZ_trace[-1] ) > epsilon:
        RZdiff_sols = RZ_partial_derivative_of_map_4_Flow_Phi_as_t(afield, [x0_RZPhi[-1], x0_RZPhi[-1] + 2*tor_turn*np.pi], xRZ_trace[-1], highest_order=1)
        x_mapped = RZdiff_sols[0].sol(x0_RZPhi[-1] + 2*tor_turn*np.pi) 
        if zone_of_interest_callback is not None:
            if not zone_of_interest_callback(*x_mapped):
                if ret_trace:
                    return xRZ_trace.append(None)
                else:
                    return None
        Jac_comp = RZdiff_sols[1].sol(x0_RZPhi[-1] + 2*tor_turn*np.pi)
        Jac = np.array( [
            [Jac_comp[2]-1, Jac_comp[0]], 
            [Jac_comp[3], Jac_comp[1]-1]])
        xRZ_trace.append( np.ravel(
            xRZ_trace[-1] - h * np.matmul( np.linalg.inv(Jac), np.array([[x_mapped[0]- xRZ_trace[-1][0] ], [x_mapped[1]- xRZ_trace[-1][1] ]])).T) )
    if not ret_trace:
        return xRZ_trace[-1]
    else:
        return xRZ_trace


def draw_eig_direction(x0, eigtheta, eigval, fig, ax, unit_data_len:float=0.05, circ_wanted=False):
    ax.arrow(x=x0[0], y=x0[1], 
        dx=eigval*np.cos(eigtheta)*unit_data_len, 
        dy=eigval*np.sin(eigtheta)*unit_data_len,
        length_includes_head=True, head_width=16e-3)
    if circ_wanted:
        circ_theta = np.linspace(0, 2*np.pi)
        circ_theta = np.linspace(0, 2*np.pi)
        circ_x = x0[0] + unit_data_len * np.cos( circ_theta ) 
        circ_y = x0[1] + unit_data_len * np.sin( circ_theta ) 
        ax.plot(circ_x, circ_y)

def draw_Jac_direction(x0, jac, fig, ax, unit_data_len:float=0.05):
    from numpy import linalg as LA
    eigvals, eigvecs = LA.eig(jac)
    for i in range(2): # draw two eigen vectors
        ax.arrow(x=x0[0], y=x0[1], 
            dx=eigvals[i]*eigvecs[0,i]*unit_data_len, 
            dy=eigvals[i]*eigvecs[1,i]*unit_data_len)
    circ_theta = np.linspace(0, 2*np.pi)
    circ_theta = np.linspace(0, 2*np.pi)
    circ_x = x0[0] + unit_data_len * np.cos( circ_theta ) 
    circ_y = x0[1] + unit_data_len * np.sin( circ_theta ) 
    ax.plot(circ_x, circ_y)
    
def Jac_evolution_along_Xcycle(afield:RegualrCylindricalGridField, Xcycle_RZdiff, Phi_span):
    R, Z, Phi, BR, BZ, BPhi = afield.R, afield.Z, afield.Phi, afield.BR, afield.BZ, afield.BPhi
    
    Phi_start, Phi_end = Phi_span[0], Phi_span[-1]
    nturn = round( (Phi_end - Phi_start) / (2*np.pi) )
    if abs( nturn - (Phi_end - Phi_start) / (2*np.pi) ) > 0.10:
        raise ValueError("The specified Xcycle_RZdiff shall have seamless head and tail. Now the Phi_tail is not Phi_head + 2pi*n. ")
    if not np.allclose( Xcycle_RZdiff[0].sol(Phi_start), Xcycle_RZdiff[0].sol(Phi_end) ):
        raise ValueError("The specified Xcycle_RZdiff shall have seamless head and tail, i.e., its beginning point and end point should be identical.")

    RBRdBPhi = R[:,None,None]*BR/BPhi
    RBZdBPhi = R[:,None,None]*BZ/BPhi
    RBRdBPhi_field = _FieldDifferenatiableRZ(RBRdBPhi, R, Z, Phi)
    RBZdBPhi_field = _FieldDifferenatiableRZ(RBZdBPhi, R, Z, Phi)
    
    # Jac evolution ode equation expressions
    M_A_interp = RBRdBPhi_field.diff_RZ_interpolator(1,0)
    M_B_interp = RBRdBPhi_field.diff_RZ_interpolator(0,1)
    M_C_interp = RBZdBPhi_field.diff_RZ_interpolator(1,0)
    M_D_interp = RBZdBPhi_field.diff_RZ_interpolator(0,1)
    M_lambda = lambda R_, Z_, Phi_: np.array([
        [M_A_interp([R_, Z_, Phi_])[0], M_B_interp([R_, Z_, Phi_])[0] ], 
        [M_C_interp([R_, Z_, Phi_])[0], M_D_interp([R_, Z_, Phi_])[0] ] ])
    def cycle_dDPdPhi(t, y):
        M = M_lambda( *Xcycle_RZdiff[0].sol(t), t)
        DP = y.reshape( (2,2) )
        dDPdPhi = M @ DP - DP @ M
        return dDPdPhi.reshape( (4,) )
    
    # Jac evolution init condition
    DP_init = np.array([
        Xcycle_RZdiff[1].y[2], 
        Xcycle_RZdiff[1].y[0], 
        Xcycle_RZdiff[1].y[3], 
        Xcycle_RZdiff[1].y[1] ]).flatten()

    return solve_ivp(cycle_dDPdPhi, [Phi_start, Phi_end], DP_init, dense_output=True, 
            first_step = Phi[1]-Phi[0], max_step = Phi[1]-Phi[0] ) 
    
    