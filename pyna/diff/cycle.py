from pyna.diff.fieldline import _FieldDifferenatiableRZ
from pyna.field import RegualrCylindricalGridField
from scipy.integrate import solve_ivp
import numpy as np

from deprecated import deprecated


def Jac_evolution_along_cycle(afield:RegualrCylindricalGridField, Xcycle_RZdiff, Phi_span):
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
        M = M_lambda( *Xcycle_RZdiff[0].sol(t), t%(2*np.pi))
        DP = y.reshape( (2,2) )
        dDPdPhi = M @ DP - DP @ M
        return dDPdPhi.reshape( (4,) )
    
    # Jac evolution init condition
    DP_init = np.array([
        Xcycle_RZdiff[1].sol(Phi_end)[2], 
        Xcycle_RZdiff[1].sol(Phi_end)[0], 
        Xcycle_RZdiff[1].sol(Phi_end)[3], 
        Xcycle_RZdiff[1].sol(Phi_end)[1] ]).flatten()

    return solve_ivp(cycle_dDPdPhi, [Phi_start, Phi_end], DP_init, dense_output=True, 
            first_step = Phi[1]-Phi[0], max_step = Phi[1]-Phi[0] ) 


from deprecated import deprecated
@deprecated(reason="This function is not reliable to point out the right eigenvector directions. Test it later. Enhancement needed.")
def Jac_theta_val_evolution_along_Xcycle(afield:RegualrCylindricalGridField, Xcycle_RZdiff, Phi_span):
    R, Z, Phi, BR, BZ, BPhi = afield.R, afield.Z, afield.Phi, afield.BR, afield.BZ, afield.BPhi

    jac_sol = Jac_evolution_along_cycle(afield, Xcycle_RZdiff, Phi_span)
    Phi_start, Phi_end = Phi_span[0], Phi_span[-1]

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
    def cycle_dtheta_lambda_dPhi(t, y):
        M = M_lambda( *Xcycle_RZdiff[0].sol(t) , t%(2*np.pi))
        eigtheta = y[0]
        eigvalue = y[1]
        DP = jac_sol.sol(t).reshape( (2,2) )
        dDPdPhi = M @ DP - DP @ M
        DP_A, DP_B = DP[0,0], DP[0,1]
        DP_C, DP_D = DP[1,0], DP[1,1]
        first_mat = np.array([
            [DP_B*np.cos(eigtheta) - (DP_A-eigvalue)*np.sin(eigtheta), -np.cos(eigtheta)], 
            [(DP_D-eigvalue)*np.cos(eigtheta) - DP_C*np.sin(eigtheta), -np.sin(eigtheta)]])
        return ( LA.inv(first_mat) @ dDPdPhi @ np.array([[np.cos(eigtheta)], [np.sin(eigtheta)]]) ).flatten()
    # def cycle_dthetas_dPhi(t, y, eval1, eval2, nturn):
    #     M = M_lambda( *Xcycle_RZdiff[0].sol(t%(2*nturn*np.pi)) , t%(2*np.pi))
    #     etet1, etet2 = y[0], y[1]
    #     costet1, sintet1 = np.cos(etet1), np.sin(etet1)
    #     costet2, sintet2 = np.cos(etet2), np.sin(etet2)
        
    #     DP = jac_sol.sol(t).reshape( (2,2) )
    #     dDPdPhi = M @ DP - DP @ M
    # #     DP_evals, DP_X = LA.eig( DP )
    # #     DP_Y__T = LA.inv(DP_X)
    #     DP_X = np.array([[costet1, costet2], [sintet1, sintet2]]) 
    #     DP_Y__T = np.array([[sintet2,-costet2], [-sintet1, costet1]]) / (-sintet1*costet2+sintet2*costet1)
    #     assert( np.allclose( DP_Y__T@DP_X, np.identity(2)  )   )
        
    #     # check dλdPhi = 0
    #     deval1_dPhi = DP_Y__T[0,:] @ dDPdPhi @ DP_X[:,0]
    #     deval2_dPhi = DP_Y__T[1,:] @ dDPdPhi @ DP_X[:,1]
        
    #     c21 = ( DP_Y__T[1,:] @ dDPdPhi @ DP_X[:,0] ) / (eval1 - eval2)
    #     c12 = ( DP_Y__T[0,:] @ dDPdPhi @ DP_X[:,1] ) / (eval2 - eval1)
    #     c11 = - ( costet1*costet2 + sintet1*sintet2 )*c21
    #     c22 = - ( costet1*costet2 + sintet1*sintet2 )*c12
        
    #     C = np.array([[c11, c12], [c21, c22]])
        
    #     Lambda_diag_mat = DP_Y__T @ np.array([[0,1],[-1,0]]) @ DP_X @ C
        
    #     detet1_dPhi = Lambda_diag_mat[0,0]
    #     detet2_dPhi = Lambda_diag_mat[1,1]
        
    # ## Simplified Calculation of Eigenvector Derivatives, Richard B. Nelson
    # #     F_1 = - dDPdPhi @ np.array([costet1, sintet1])
    # #     F_2 = - dDPdPhi @ np.array([costet2, sintet2])
        
    # #     Y2F1_over_eval2_eval1 = (DP_Y__T[1,:] @ F_1) / (eval2 - eval1)
    # #     Y1F2_over_eval1_eval2 = (DP_Y__T[0,:] @ F_2) / (eval1 - eval2)
        
    # #     evecs_dot = costet1*costet2 + sintet1*sintet2

    # #     if np.abs(sintet1) > 0.717:
    # #         detet1_dPhi = costet2*Y2F1_over_eval2_eval1 - evecs_dot*Y2F1_over_eval2_eval1*costet1
    # #         detet1_dPhi/= -sintet1
    # #     else:
    # #         detet1_dPhi = sintet2*Y2F1_over_eval2_eval1 - evecs_dot*Y2F1_over_eval2_eval1*sintet1
    # #         detet1_dPhi/=  costet1
            
    # #     if np.abs(sintet2) > 0.717:
    # #         detet2_dPhi = costet1*Y1F2_over_eval1_eval2 - evecs_dot*Y1F2_over_eval1_eval2*costet2
    # #         detet2_dPhi/= -sintet2
    # #     else:
    # #         detet2_dPhi = sintet1*Y1F2_over_eval1_eval2 - evecs_dot*Y1F2_over_eval1_eval2*sintet2
    # #         detet2_dPhi/=  costet2
            
            
    # #     DP_A, DP_B = DP[0,0], DP[0,1]
    # #     DP_C, DP_D = DP[1,0], DP[1,1]
    # #     first_mat = np.array([
    # #         [costet1, 0       ,-eval1*sintet1  -DP_A*sintet1+DP_B*costet1, 0], 
    # #         [0       , costet2, 0                   ,-eval2*sintet2  -DP_A*sintet2+DP_B*costet2],
    # #         [sintet1, 0       , eval1*costet1  -DP_C*sintet1+DP_D*costet1, 0                 ],
    # #         [0       , sintet2, 0                   , eval2*costet2  -DP_C*sintet2+DP_D*costet2] ])
    # #     print("Now runs to: ", t, " evals: ", [ deval1_dPhi, deval2_dPhi, detet1_dPhi, detet2_dPhi])
    #     return [ detet1_dPhi, detet2_dPhi]

    # Jac evolution init condition
    w, v = LA.eig( jac_sol.sol(Phi_start).reshape( (2,2) ) )
    return [
        solve_ivp(cycle_dtheta_lambda_dPhi, [Phi_start, Phi_end], np.array([np.arctan2(v[1,0], v[0,0]), w[0] ]), dense_output=True, 
            first_step = Phi[1]-Phi[0], max_step = Phi[1]-Phi[0] ), 
        solve_ivp(cycle_dtheta_lambda_dPhi, [Phi_start, Phi_end], np.array([np.arctan2(v[1,1], v[0,1]), w[1] ]), dense_output=True, 
            first_step = Phi[1]-Phi[0], max_step = Phi[1]-Phi[0] )]

    # nturn=3
    # w, v = LA.eig( jac_sol.sol(0.0).reshape( (2,2) ) )
    # Jac_vals_tets = solve_ivp(
    #     cycle_dlambdas_thetas_dPhi, [0.0, 6*np.pi], 
    #     np.array([ np.arctan2(v[1,0], v[0,0]), np.arctan2(v[1,1], v[0,1]), ]), dense_output=True, args=(w[0], w[1], nturn),
    #     first_step = (Phi[1]-Phi[0])/50, max_step = (Phi[1]-Phi[0])/50, method="LSODA")

import numpy.linalg as LA
from scipy.interpolate import interp1d
def eigvec_interpolator_along_Xcycle(Jac_evosol_along_Xcycle):
    t1, t2 = Jac_evosol_along_Xcycle.t[0], Jac_evosol_along_Xcycle.t[1]
    t = Jac_evosol_along_Xcycle.t
    
    DPs = Jac_evosol_along_Xcycle.sol(t).T.reshape((len(t),2,2))
    eigvals, eigvecs = LA.eig(DPs)
    DP_init = Jac_evosol_along_Xcycle.sol(t1).reshape( (2,2) )
    eigval_init, eigvec_init = LA.eig(DP_init)
    
    for i in range(len(t)):
        if not np.allclose( eigvals[i,0], eigval_init[0], rtol=1e-3, atol=1e-4):
            eigvals[i,[0,1]] = eigvals[i,[1,0]]
            eigvecs[i,:,[0,1]] = eigvecs[i,:,[1,0]]
    
    if np.dot( eigvecs[0,:,0], eigvec_init[:,0] ) < 0:
        eigvecs[0,:,0] *= -1
    if np.dot( eigvecs[0,:,1], eigvec_init[:,1] ) < 0:
        eigvecs[0,:,1] *= -1
    
    for i in range(1, len(t)):
        if np.dot( eigvecs[i,:,0], eigvecs[i-1,:,0] ) < 0:
            eigvecs[i,:,0] *= -1
        if np.dot( eigvecs[i,:,1], eigvecs[i-1,:,1] ) < 0:
            eigvecs[i,:,1] *= -1
    
    return interp1d(t, eigvals, axis=0), interp1d(t, eigvecs, axis=0)  
    
    
@deprecated(version='0.1.0', reason="Numerical blow up, unstoppable computation. Never reach the end.")
def Jac_theta_evolution(afield:RegualrCylindricalGridField, Xcycle_RZdiff, Phi_span ):
    """_summary_

    Args:
        R (_type_): _description_
        Z (_type_): _description_
        Phi (_type_): _description_
        BR (_type_): _description_
        BZ (_type_): _description_
        BPhi (_type_): _description_
        Xcycle_RZdiff (_type_): _description_
        Phi_span (_type_): _description_

    Returns:
        _type_: _description_

    Example:
        from pyna.diff.Xcycle import Jac_theta_evolution
        jac_theta_sols = Jac_theta_evolution(R,Z,Phi, BR_tot, BZ_tot, BPhi_tot, UPX_RZdiff, Phi_span=[0.0, 2*np.pi])

    """
    R, Z, Phi, BR, BZ, BPhi = afield.R, afield.Z, afield.Phi, afield.BR, afield.BZ, afield.BPhi
    
    jac_sol = Jac_evolution_along_cycle(afield, Xcycle_RZdiff, Phi_span)
    eigvals, eigvecs = np.linalg.eig( jac_sol.sol(0.0).reshape( (2,2) ) )
    tet1_init = np.arctan2(eigvecs[1,0], eigvecs[0,0])
    tet2_init = np.arctan2(eigvecs[1,1], eigvecs[0,1])
    
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
    def cycle_dtheta_dPhi(t, y):
        M = M_lambda( *Xcycle_RZdiff[0].sol(t) , t%(2*np.pi)) # TODO: consider how to mod Xcycle_RZdiff[0].sol(t) by $2k\pi$
        tet1, tet2 = y[0], y[1]
        DP = jac_sol.sol(t).reshape( (2,2) ) # TODO: consider how to mod jac_sol.sol(t) by $2k\pi$
        dDPdPhi = M @ DP - DP @ M
        Rot = np.array([
            [0.0, -1.0], 
            [1.0,  0.0]])
        Lam=np.array([
            [eigvals[0],  0.0], 
            [0.0,  eigvals[1]]])
        V = np.array([
            [np.cos(tet1), np.cos(tet2)], 
            [np.sin(tet1), np.sin(tet2)]])
        TET__prime = np.linalg.inv( Rot@V@Lam - DP@Rot@V  )@dDPdPhi@V
        return [TET__prime[0,0], TET__prime[1,1] ]

    Jac_theta_sol = solve_ivp(cycle_dtheta_dPhi, Phi_span, 
                         np.array([tet1_init, tet2_init,]), dense_output=True, method="LSODA", # NOTE: by our numerical experiments, when the eigenvector varies just a little bit along the cycle, only "LSODA" method does not cause numerical issue. In fact, other methods would let the time step become too big since TET' is small.
                         init_step=(Phi[1]-Phi[0]), max_step = (Phi[1]-Phi[0]) )
    Jac_theta_sol.eigvals = list(eigvals)
    return Jac_theta_sol