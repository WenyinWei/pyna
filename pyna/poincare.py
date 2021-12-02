from scipy.integrate import solve_ivp
from .flow import FlowCallable
from .map import MapUniCallable
import numpy as np

def poincare_FlowCallable_2_MapCallable(pflow: FlowCallable, section_phi=2*np.pi):
    _diff_xi_lambdas = pflow.diff_xi_lambdas

    def fun(t, y):
        return [lam(*y) for lam in _diff_xi_lambdas]
    
    return MapUniCallable(
        lambda xi_init: [next_xi[0] for next_xi in solve_ivp(fun, [0, section_phi], xi_init, t_eval=[section_phi], dense_output=False).y])
