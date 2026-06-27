import numpy as np
from pyna.topo.healed_flux_coords import build_xo_sequence, build_cxo_spline


def _star_section(n=7, seed=0):
    rng = np.random.default_rng(seed)
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    O = np.column_stack([
        6.0 + (1.15 + 0.22 * np.cos(2 * ang)) * np.cos(ang),
        0.2 + (0.72 + 0.16 * np.sin(3 * ang)) * np.sin(ang),
    ])
    X = np.column_stack([
        6.0 + (1.02 + 0.05 * np.cos(2 * (ang + np.pi / n))) * np.cos(ang + np.pi / n),
        0.2 + (0.55 + 0.04 * np.sin(3 * (ang + np.pi / n))) * np.sin(ang + np.pi / n),
    ])
    O = O[[0, 3, 1, 5, 2, 6, 4]] + rng.normal(scale=0.01, size=O.shape)
    X = X[[4, 1, 6, 0, 3, 5, 2]] + rng.normal(scale=0.008, size=X.shape)
    X = np.vstack([X, X[2] + np.array([1e-7, -1e-7])])
    return O, X


def run_demo():
    axis = (6.0, 0.2)
    O, X = _star_section()
    xo = build_xo_sequence(O, X, axis=axis, dedup_tol=1e-5, rho_min=0.1)
    assert xo is not None, 'build_xo_sequence failed'
    assert xo.diagnostics['self_intersections'] == 0
    assert xo.diagnostics['winding_monotone']
    assert xo.diagnostics['sequence_cleanup_removed'] >= 0
    assert xo.diagnostics['n_slots_filled'] >= 5

    sR, sZ, xo2 = build_cxo_spline(O, X, axis=axis, dedup_tol=1e-5, rho_min=0.1, validate_winding=True, sample_count=512)
    assert sR is not None and sZ is not None and xo2 is not None, 'build_cxo_spline failed'
    ss = np.linspace(xo2.s_closed[0], xo2.s_closed[-1], 512, endpoint=False)
    rr = np.hypot(sR(ss) - axis[0], sZ(ss) - axis[1])
    ang = np.unwrap(np.arctan2(sZ(ss) - axis[1], sR(ss) - axis[0]))
    assert np.min(rr) > 0.0
    assert np.all(np.diff(ang) > -1e-6)

    print('healed_scaffold_numpy_demo: OK')
    print('slots', xo2.diagnostics['n_slots_filled'], '/', xo2.diagnostics['n_O'])
    print('cleanup_removed', xo2.diagnostics['sequence_cleanup_removed'])
    print('signed_area', f"{xo2.diagnostics['signed_area']:.6f}")
    print('total_length', f"{xo2.diagnostics['total_length']:.6f}")


if __name__ == '__main__':
    run_demo()
