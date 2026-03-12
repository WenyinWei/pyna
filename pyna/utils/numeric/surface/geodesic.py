from silkpy.symbolic.curve.curve import ParametricCurve 
from silkpy.symbolic.surface.surface import ParametricSurface 


def geodesic_ncurve(surface: ParametricSurface, ic_uv, ic_uv_t, t1=5, dt=0.05):
    from sympy import lambdify
    from scipy.integrate import ode as sciode
    import numpy as np
    from sympy import symbols, Function, Array, tensorproduct, tensorcontraction
    t = symbols('t', real=True)
    u = Function(surface.sym(0), real=True)(t)
    v = Function(surface.sym(1), real=True)(t)

    second_term_tensor = tensorproduct(
        surface.christoffel_symbol.tensor().subs(
            {surface.sym(0):u, surface.sym(1):v}),
        Array([u, v]).diff(t), 
        Array([u, v]).diff(t))
    second_term_tensor = tensorcontraction(second_term_tensor, (1, 3), (2, 4))

    u_t = Function(str(u)+'^{\prime}', real=True)(t)
    v_t = Function(str(v)+'^{\prime}', real=True)(t)
    lambdify_sympy = lambdify((u, u_t, v, v_t), [
        u_t, 
        -second_term_tensor[0].subs({u.diff(t):u_t, v.diff(t):v_t}), 
        v_t,
        -second_term_tensor[1].subs({u.diff(t):u_t, v.diff(t):v_t})])
    
    x0, t0 = [ic_uv[0], ic_uv_t[0], ic_uv[1], ic_uv_t[1]], 0.0
    scioder = sciode(lambda t,X: lambdify_sympy(*X)).set_integrator('vode', method='bdf')
    scioder.set_initial_value(x0, t0)
    num_of_t = int(t1 / dt); # num_of_t
    u_arr = np.empty((num_of_t, 4)); u_arr[0] = x0
    t_arr = np.arange(num_of_t) * dt
    i = 0
    while scioder.successful() and i < num_of_t-1:
        i += 1
        u_arr[i] = scioder.integrate(scioder.t+dt)
    return t_arr, (u_arr[:, 0], u_arr[:, 2])

def geodesic_polar_ncoordinate(surface: ParametricSurface, origin_uv: tuple, rho1=1.2, nrho=12, ntheta=48):
    import numpy as np
    from sympy import pi, sin, cos
    from scipy.interpolate import interp1d

    surface_func = surface.lambdified()
    rho_arr = np.linspace(0.0, rho1, num=nrho, endpoint=True)
    theta_arr=np.linspace(0.0, 2*float(pi), num=ntheta, endpoint=False)
    u_grid = np.empty((len(rho_arr), len(theta_arr)))
    v_grid = np.empty((len(rho_arr), len(theta_arr)))
    for theta_i, theta in enumerate(theta_arr):
        t_arr, (u_arr, v_arr) = geodesic_ncurve(
            surface, origin_uv, [cos(theta), sin(theta)])
        x_arr, y_arr, z_arr = surface_func(u_arr, v_arr)
        dx_arr, dy_arr, dz_arr = np.diff(x_arr), np.diff(y_arr), np.diff(z_arr)
        len_arr = np.sqrt( dx_arr**2 + dy_arr**2 + dz_arr**2 )
        cumlen_arr = np.insert( np.cumsum(len_arr), 0, 0.0 )
        index_arr = np.arange( len(cumlen_arr) )
        t_arr_selected = interp1d(cumlen_arr, t_arr, kind='cubic')(rho_arr)
        u_grid[:, theta_i] = interp1d(t_arr, u_arr, kind='cubic')(t_arr_selected)
        v_grid[:, theta_i] = interp1d(t_arr, v_arr, kind='cubic')(t_arr_selected)
    return rho_arr, theta_arr, u_grid, v_grid
