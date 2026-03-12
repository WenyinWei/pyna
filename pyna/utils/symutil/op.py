def time_average_of_TrigFunc(expr, t, method="Integrate_Then_Average", T=None):
    from .characteristics import min_period
    from sympy import integrate
    if method=="Integrate_Then_Average":
        from sympy import integrate
        
        if len(T)==1:
            return integrate(expr, (t, 0, T)) / T
        elif T is None:
            T = min_period(expr, t)
            return integrate(expr, (t, 0, T)) / T
        elif len(T)==2:
            return integrate(expr, (t, T[0], T[1])) / (T[1]-T[0])

        
#     elif method=="Expand_Multiple_Angle_Then_Subs":
#         raise NotImplementedError(f"{__name__} not yet implemented for the method '{method}'.")
    else:
        raise ValueError(f"The `method` parameter can only be 'Integrate_Then_Average' right now.")
