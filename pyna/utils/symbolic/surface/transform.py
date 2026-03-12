def param_transform(other_surface, newu=None, newv=None, u_expr=None, v_expr=None):
    from sympy import S, solveset, Eq
    from silkpy.sympy_utility import norm

    if newu is None or newv is None: 
        from sympy import Symbol
        newu = Symbol('u', real=True)
        newv = Symbol('v', real=True)
        
        from sympy import integrate
        drdt = norm(other_surface.expr().diff(other_surface._t))
        s_expr = integrate(drdt, other_surface._t).simplify()
        solset = solveset(Eq(s, s_expr), t, domain=S.Reals)
        if len(solset) != 1:
            raise RuntimeError(f"Sympy is not smart enough to inverse s(t) into t(s).\
            It found these solutions: {solset}.\
            Users need to choose from them or deduce manually, and then set it by obj.param_norm(s_symbol, t_expressed_by_s")
        t_expr = next(iter(solset))
    return Surface(
        other_surface.expr().applyfunc(lambda x: x.subs(other_surface._u, u_expr)).applyfunc(lambda x: x.subs(other_surface._v, v_expr)), 
        (newu, newu_expr.subs(other_surface._u, other_surface._u_limit[0]), newu_expr.subs(other_surface._u, other_surface._u_limit[1])),
        (newv, newv_expr.subs(other_surface._v, other_surface._v_limit[0]), newv_expr.subs(other_surface._v, other_surface._v_limit[1])), other_surface._sys)