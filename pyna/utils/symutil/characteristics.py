def find_all_trig_period(expr, var):
#     from collections.abc import Iterable
#     if isinstance(var, Iterable):
#         return {find_trig_omega(expr, a_var) for a_var in var}
    from sympy.functions.elementary.trigonometric import TrigonometricFunction
    if isinstance(expr, TrigonometricFunction):
        try:
            return {expr.period(symbol=var)}
        except:
            return set()
    else:
        periods = [find_all_trig_period(sub_expr, var) for sub_expr in expr.args]
        return set().union(*periods)

def min_period(expr, var):
    Ts = find_all_trig_period(expr, var)
    if len(Ts) == 0:
        raise ValueError("The expression does not contain Trigonometric functions, please decide the T parameter manually.")
    try:
        return min(Ts)
    except TypeError: # If we can not decide which period is the min one, a random one is chosen.
        return next(iter(Ts))
    