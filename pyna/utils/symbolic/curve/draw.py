from .curve import ParametricCurve as _ParametricCurve

def draw_curve_plotly(curve:_ParametricCurve, num=50, domain=(-1, 1), exist_range=[(None, None), (None, None), (None, None)], color_func=(lambda curve: curve.curvature().simplify()), fig=None, line=dict(width=4, showscale=True), mode="lines", *arg, **kwarg):
    from numpy import linspace
    from sympy import lambdify, Interval
    from sympy.sets.sets import Union
    from sympy.solvers.inequalities import solve_univariate_inequality
    import plotly.graph_objects as go

    domain = Interval(*domain).intersect(
        Interval(*curve.sym_limit(0)))
    for i, lim in enumerate(exist_range):
        if lim[0] is not None:
            domain = solve_univariate_inequality(lim[0] < curve.expr(i), curve.sym(0), relational=False, domain=domain)
        if lim[1] is not None:
            domain = solve_univariate_inequality(lim[1] > curve.expr(i), curve.sym(0), relational=False, domain=domain)
    if isinstance(domain, Union):
        domain = domain.args[0]

    domain_ = linspace(float(domain.start), float(domain.end), num=num)
    values_ = lambdify(curve.sym(0), curve.expr(), 'numpy')(domain_)
    if color_func is not None:
        colors_ = lambdify(curve.sym(0), color_func(curve), 'numpy')(domain_)

    kwarg['x'] = values_[0]
    kwarg['y'] = values_[1]
    kwarg['z'] = values_[2]
    kwarg['line'] = line
    if color_func is not None: kwarg['line']['color'] = colors_
    kwarg['mode'] = mode

    if fig is None:
        fig = go.Figure(data=go.Scatter3d(*arg, **kwarg))
    else:
        fig.add_trace(go.Scatter3d(*arg, **kwarg))
    fig.update_layout(scene=dict(
        aspectratio = dict(x=1, y=1, z=1),
        aspectmode = 'data'))

    return fig