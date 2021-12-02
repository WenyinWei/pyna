import numpy as np
from pyna.diff.fieldline import RZ_partial_derivative_of_map_4_Flow_Phi_as_t

def Newton_discrete(BR, BZ, BPhi, R, Z, Phi, x0, h=0.1, epsilon=5e-4, output_trace=True):
    x_trace = [np.asarray(x0)]
    while len(x_trace)==1 or np.linalg.norm( x_trace[-2] - x_trace[-1] ) > epsilon:
        try:
            RZdiff_sols = RZ_partial_derivative_of_map_4_Flow_Phi_as_t(BR, BZ, BPhi, R, Z, Phi, [0.0, 2*np.pi], x_trace[-1], highest_order=1)
        except ValueError as e:
            print(e)
            return x_trace
        x_mapped = RZdiff_sols[0].sol(2*np.pi) 
        Jac_comp = RZdiff_sols[1].sol(2*np.pi)
        Jac = np.array( [
            [Jac_comp[2]-1, Jac_comp[0]], 
            [Jac_comp[3], Jac_comp[1]-1]])
        x_trace.append( np.ravel(
            x_trace[-1] - h * np.matmul( np.linalg.inv(Jac), np.array([[x_mapped[0]- x_trace[-1][0] ], [x_mapped[1]- x_trace[-1][1] ]])).T) )
    return x_trace

