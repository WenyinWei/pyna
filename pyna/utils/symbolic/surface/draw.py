from .surface import ParametricSurface

# (lambda surface: surface.curvature().simplify())
def draw_surface_plotly(
    surface:ParametricSurface, 
    domain=[(-1, 1), (-1, 1)], num=[50, 50], exist_range=[(None, None), (None, None), (None, None)], 
    color_func=None,
    fig=None, *arg, **kwarg):
    from numpy import mgrid
    from sympy import lambdify, Interval
    from sympy.sets.sets import Union
    # from sympy.solvers.inequalities import solve_univariate_inequality
    import plotly.graph_objects as go

    for i in range(2):
        domain[i] = Interval(*(domain[i])).intersect(
            Interval(*surface.sym_limit(i)))
        # We has not yet prepared for exist_range arg, so it is invalid now.  
        # for i, lim in enumerate(exist_range):
        #     if lim[0] is not None:
        #         domain = solve_univariate_inequality(lim[0] < surface.expr(i), surface.sym(0), relational=False, domain=domain)
        #     if lim[1] is not None:
        #         domain = solve_univariate_inequality(lim[1] > surface.expr(i), surface.sym(0), relational=False, domain=domain)
        if isinstance(domain[i], Union):
            domain[i] = domain[i].args[0]

    u_domain_, v_domain_ = mgrid[
        float(domain[0].start):float(domain[0].end):int(num[0])*1j,
        float(domain[1].start):float(domain[1].end):int(num[1])*1j,
    ]

    values_ = lambdify(surface.syms, surface.exprs, 'numpy')(u_domain_, v_domain_)
    if color_func is not None:
        colors_ = lambdify(surface.syms, color_func(surface), 'numpy')(u_domain_, v_domain_)

    kwarg['x'] = values_[0]
    kwarg['y'] = values_[1]
    kwarg['z'] = values_[2]
    if color_func is not None: kwarg['surfacecolor'] = colors_

    if fig is None:
        fig = go.Figure(data=go.Surface(*arg, **kwarg))
    else:
        fig.add_trace(go.Surface(*arg, **kwarg))
    fig.update_layout(scene=dict(
        aspectratio = dict(x=1, y=1, z=1),
        aspectmode = 'data'))

    return fig