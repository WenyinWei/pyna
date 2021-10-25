from scipy.integrate import solve_ivp, OdeSolution

from .flow import FlowCallable

def solve_Flow(pflow: FlowCallable, t_span, y0, *arg, **kwarg):
    _diff_xi_lambdas = pflow.diff_xi_lambdas()
    def fun(t, y):
        return [lam(*y) for lam in _diff_xi_lambdas]
    return solve_ivp(fun, t_span, y0, *arg, **kwarg)

