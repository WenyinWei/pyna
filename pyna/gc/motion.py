import numpy as np


def time_derivative(f, xyz, t, h=1e-5):
    return (f(*xyz, t+h) - f(*xyz, t-h)) / (2 * h)

def space_dirderivative(f, xyz, t, direction, h=1e-8):
    direction = np.array(direction)
    norm = np.linalg.norm(direction)
    if norm == 0:
        return np.zeros_like(f(*xyz, t))
    direction = direction / norm  # Normalize the direction vector
    
    x, y, z = xyz
    dx, dy, dz = direction * h
    return (f(x + dx, y + dy, z + dz, t) - f(x - dx, y - dy, z - dz, t)) / (2 * h)


def particle_motion_forward(t, state, p):
    x, y, z, vx, vy, vz = state
    q, m0, EBfield = p
    E, B = EBfield.E_and_B_at(x, y, z, t)
    v = np.array([vx, vy, vz])
    F = q * (E + np.cross(v, B))
    ax, ay, az = F / m0
    return [vx, vy, vz, ax, ay, az]
def particle_motion_backward(t, state, p):
    x, y, z, vx, vy, vz = state
    q, m0, EBfield = p
    E, B = EBfield.E_and_B_at(x, y, z, -t)
    v = np.array([vx, vy, vz])
    F = q * (E + np.cross(v, B))
    ax, ay, az = F / m0
    return [-vx, -vy, -vz, -ax, -ay, -az]

def particle_motion_withoutALforce_forward(t, state, p):
    x, y, z, vx, vy, vz = state
    xyz = np.array([x, y, z])
    q, m0, EBfield = p
    E = EBfield.E_at(x, y, z, t)
    B = EBfield.B_at(x, y, z, t)
    v = np.array([vx, vy, vz])
    F_ext = q * (E + np.cross(v, B))
    c = 299792458  # speed of light in m/s
    mu_0 = 4 * np.pi * 1e-7  # vacuum permeability
    gamma = 1 / np.sqrt(1 - np.dot(v, v) / c**2)
    
    # Calculate initial acceleration without AL force
    I = np.eye(3)
    vvT = np.outer(v, v)
    a = (I - vvT / c**2) @ F_ext / (m0 * gamma)
    
    return [vx, vy, vz, *a]

def particle_motion_withALforce_forward(t, state, p):
    x, y, z, vx, vy, vz = state
    xyz = np.array([x, y, z])
    q, m0, EBfield = p
    E = EBfield.E_at(x, y, z, t)
    B = EBfield.B_at(x, y, z, t)
    v = np.array([vx, vy, vz])
    F_ext = q * (E + np.cross(v, B))
    c = 299792458  # speed of light in m/s
    mu_0 = 4 * np.pi * 1e-7  # vacuum permeability
    gamma = 1 / np.sqrt(1 - np.dot(v, v) / c**2)
    
    # Calculate initial acceleration without AL force
    I = np.eye(3)
    vvT = np.outer(v, v)
    a = (I - vvT / c**2) @ F_ext / (m0 * gamma)
    
    # Calculate AL force
    v_dot_a = np.dot(v, a)
    v_cross_B = np.cross(v, B)
    v_norm = np.linalg.norm(v)
    term1 = -gamma * v_dot_a / c**2 * (I - vvT / c**2) @ (E + v_cross_B)
    term2 = - 1/gamma * (np.outer(a, v) + np.outer(v, a)) / c**2 @ (E + v_cross_B)
    term3 = 1/gamma * (I - vvT / c**2) @ ( 
        time_derivative(EBfield.E_at, xyz, t) + v_norm*space_dirderivative(EBfield.E_at, xyz, t, v) 
        + np.cross(a, B) 
        + np.cross(v, time_derivative(EBfield.B_at, xyz, t) + v_norm*space_dirderivative(EBfield.B_at, xyz, t, v) )
    )
    
    a_dot = (q / m0) * (term1 + term2 + term3)
    a_ALforce = mu_0 * q**2 / (6 * np.pi * c) / (m0 * gamma) * (
        gamma**2 * a_dot 
        + 3 * gamma**2 * v_dot_a * a / c**2
        + 3 * gamma**4 * v_dot_a**2 * v / c**4
    )

    a_total = a + a_ALforce
    # print("a: ", a)
    # print("a_ALforce: ", a_ALforce)
    # print("Percentage of AL force: ", np.linalg.norm(a_ALforce) / np.linalg.norm(a) * 100, "%")
    return [vx, vy, vz, *a_total]

def gc_motion_forward(t, state, p):
    x, y, z, v_parallel = state
    xyz = np.array([x, y, z])
    q, m0, mu, EBfield = p
    E, B = EBfield.E_at(x, y, z, t), EBfield.B_at(x, y, z, t)
    hatb = EBfield.hatb_at(x, y, z, t)
    g = np.array([0, 0, 0])  
    # g = np.array([0, 0, -9.81])  # Assuming gravitational acceleration
    
    ppt_hatb = time_derivative(EBfield.hatb_at, xyz, t)
    pps_hatb = space_dirderivative(EBfield.hatb_at, xyz, t, hatb)
    vE = EBfield.vE_at(x, y, z, t)

    term1 = -E + (mu / q) * EBfield.gradB_abs_at(x, y, z, t)
    term2 = (m0 / q) * (
        -g 
        + v_parallel * ppt_hatb 
        + v_parallel**2 * pps_hatb 
        + v_parallel * np.linalg.norm(vE) * space_dirderivative(EBfield.hatb_at, xyz, t, vE) 
    )
    term3 = (m0 / q) * (
        time_derivative(EBfield.vE_at, xyz, t)
        + v_parallel * space_dirderivative(EBfield.vE_at, xyz, t, hatb)
        + np.linalg.norm(vE) * space_dirderivative(EBfield.vE_at, xyz, t, vE)
    )

    R_perp_dot = np.cross(hatb, term1 + term2 + term3) / np.linalg.norm(B)

    g_parallel = np.dot(g, hatb)
    E_parallel = np.dot(E, hatb)
    v_parallel_dot = (
        (m0 / q) * g_parallel 
        + E_parallel 
        - (mu / q) * space_dirderivative(EBfield.B_abs_at, xyz, t, hatb) 
        + (m0 / q) * np.dot(vE, 
                            ppt_hatb 
                            + v_parallel * pps_hatb 
                            + np.linalg.norm(vE) * space_dirderivative(EBfield.hatb_at, xyz, t, vE)
                            )
    )
    v_parallel_dot /= m0 / q

    return [*(v_parallel*hatb + R_perp_dot), v_parallel_dot]

def particle_motion_relativistic_forward(t, state, p):
    x, y, z, vx, vy, vz, ax, ay, az = state
    q, m0, EBfield = p
    E, B = EBfield.E_and_B_at(x, y, z, t)
    v = np.array([vx, vy, vz])
    a = np.array([ax, ay, az])
    F_ext = q * (E + np.cross(v, B))
    c = 299792458  # speed of light in m/s
    mu_0 = 4 * np.pi * 1e-7  # vacuum permeability
    gamma = 1 / np.sqrt(1 - np.dot(v, v) / c**2)
    # if np.isnan(gamma):
    #     print(v)
    #     print(np.dot(v, v) / c**2)
    #     print(gamma)
    I = np.eye(3)
    term1 = (6 * np.pi * c / (mu_0 * q**2 * gamma**2)) * (m0 * gamma * a - (I - np.outer(v, v) / c**2) @ F_ext)
    term2 = -3 * np.dot(v, a) * a / c**2
    term3 = -3 * gamma**2 * (np.dot(v, a)**2) * v / c**4
    # print("Terms: ", term1, term2, term3)
    a_dot = term1 + term2 + term3
    return [vx, vy, vz, ax, ay, az, *a_dot]