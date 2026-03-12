from .curve import ParametricCurve 
from sympy import Symbol 

def curve_normalize(
    old_curve:ParametricCurve,
    new_var=Symbol('s', real=True)):
    from sympy import S, solveset, Eq
    from sympy import integrate
    from silkpy.sympy_utility import norm
    
    drdt = norm(old_curve.exprs.diff(old_curve.sym(0)))
    new_var_in_old = integrate(drdt, old_curve.sym(0)).simplify()
    solset = solveset(Eq(new_var, new_var_in_old), old_curve.sym(0), domain=S.Reals).simplify()
    try:
        if len(solset) != 1:
            raise RuntimeError(f"We have not yet succedded in inverse s(t) into t(s).\
            It found these solutions: {solset}.\
            Users need to choose from them or deduce manually, and then set it by obj.param_norm(s_symbol, t_expressed_by_s")
    except:
        raise RuntimeError(f"We have not yet succedded in inverse s(t) into t(s). Try the curve_param_transform function instead and set the transform relation manually.")
    else:
        old_var_in_new = next(iter(solset))
    return ParametricCurve(
        (new_var, 
         new_var_in_old.subs(old_curve.sym(0), old_curve.sym_limit(0)[0]),
         new_var_in_old.subs(old_curve.sym(0), old_curve.sym_limit(0)[1])),
        old_curve.exprs.subs(old_curve.sym(0), old_var_in_new)
        )

# TODO: Check the following function  
def curve_param_transform(old_curve, newt, t_expr=None):
    from sympy import S, solveset, Eq
    return ParametricCurve(
        old_curve._r.applyfunc(lambda x: x.subs(old_curve._t, t_expr)), 
        (newt, newt_expr.subs(old_curve._t, old_curve._t_limit[0]), newt_expr.subs(old_curve._t, old_curve._t_limit[1])), old_curve._sys)