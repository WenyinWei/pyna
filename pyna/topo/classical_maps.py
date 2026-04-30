"""Classical discrete area-preserving maps for chaos research.

These are the 2D analogues of the Poincaré map for field lines.
Used for testing island-finding algorithms and monodromy analysis
without needing a full 3D field line integrator.

Maps implemented:
- Hénon map: x' = -y + 2(K - x²), y' = x
- Standard (Chirikov) map: p' = p + K sin(x), x' = x + p' (mod 2π)

Reference:
  Hénon, M. (1969). Numerical study of quadratic area-preserving mappings.
    Quarterly of Applied Mathematics, 27(3), 291-312.
  Chirikov, B.V. (1979). A universal instability of many-dimensional oscillator systems.
    Physics Reports, 52(5), 263-379.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class HenonMap:
    """Hénon area-preserving quadratic map.

    x' = -y + 2(K - x²)
    y' =  x

    Fixed points: x* = ±sqrt(K) for K > 0.
    The Jacobian has det = 1 (area-preserving).

    Stability: |tr(J)| < 2 at a fixed point implies elliptic (O-point),
    |tr(J)| > 2 implies hyperbolic (X-point).
    For the Hénon map at x* = +sqrt(K): tr = -4*sqrt(K),
    so elliptic when sqrt(K) < 0.5, i.e. K < 0.25.
    """
    K: float = 0.5

    def step(self, x, y):
        """One iteration of the map."""
        x_new = -y + 2 * (self.K - x**2)
        y_new = x
        return x_new, y_new

    def iterate(self, x0, y0, n_steps):
        """Iterate n_steps times, return full trajectory arrays."""
        xs, ys = [x0], [y0]
        x, y = x0, y0
        for _ in range(n_steps):
            x, y = self.step(x, y)
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def fixed_points(self):
        """Return (O-point, X-point) fixed point coordinates.

        Fixed points satisfy x' = x, y' = y with y* = x*:
          -x* + 2(K - x*²) = x*  →  x*² + x* - K = 0
          x* = (-1 ± √(1 + 4K)) / 2

        Returns
        -------
        o_point : ndarray [x, y] or None
            Elliptic (O-point) fixed point — larger x root.
        x_point : ndarray [x, y] or None
            Hyperbolic (X-point) fixed point — smaller x root.
        """
        disc = 1 + 4 * self.K
        if disc < 0:
            return None, None
        sqrt_disc = np.sqrt(disc)
        xo = (-1 + sqrt_disc) / 2   # larger root → elliptic
        xx = (-1 - sqrt_disc) / 2   # smaller root → hyperbolic
        return np.array([xo, xo]), np.array([xx, xx])

    def jacobian_at(self, x, y):
        """2×2 Jacobian of the map at (x, y).

        J = [[-4x, -1],
             [ 1,   0]]

        det(J) = 1 always (area-preserving).
        """
        return np.array([[-4 * x, -1.0], [1.0, 0.0]])

    def monodromy_at_fixed_point(self, point):
        """Monodromy matrix (= Jacobian) at a fixed point."""
        x, y = point
        return self.jacobian_at(x, y)


@dataclass
class StandardMap:
    """Chirikov standard (twist) map.

    p' = p + K sin(x)    (mod 2π)
    x' = x + p'          (mod 2π)

    K = 0:     integrable, concentric circles
    K ≈ 0.972: KAM breakup, last invariant torus destroyed
    K > 1:     mostly chaotic

    The Jacobian is:
      J = [[1, K*cos(x)],
           [1, 1 + K*cos(x)]]
    with det(J) = 1 (area-preserving).
    """
    K: float = 0.5

    def step(self, p, x):
        """One iteration of the map."""
        p_new = (p + self.K * np.sin(x)) % (2 * np.pi)
        x_new = (x + p_new) % (2 * np.pi)
        return p_new, x_new

    def iterate(self, p0, x0, n_steps):
        """Iterate n_steps times, return full trajectory arrays."""
        ps, xs = [p0], [x0]
        p, x = p0, x0
        for _ in range(n_steps):
            p, x = self.step(p, x)
            ps.append(p)
            xs.append(x)
        return np.array(ps), np.array(xs)

    def jacobian_at(self, p, x):
        """2×2 Jacobian at (p, x)."""
        kcosx = self.K * np.cos(x)
        return np.array([[1.0, kcosx], [1.0, 1.0 + kcosx]])


def poincare_map_henon(K_values, n_steps=5000, n_seeds=30):
    """Generate Poincaré-like scatter plot data for Hénon map across K values.

    Parameters
    ----------
    K_values : iterable of float
        Parameter values to scan.
    n_steps : int
        Number of iterations per seed.
    n_seeds : int
        Number of initial conditions per K value.

    Returns
    -------
    dict : K -> (xs, ys) trajectory arrays (bounded orbits only, |x|,|y| < 5)
    """
    results = {}
    for K in K_values:
        hmap = HenonMap(K=K)
        all_xs, all_ys = [], []
        x_seeds = np.linspace(-np.sqrt(abs(K)) * 1.2, np.sqrt(abs(K)) * 1.2, n_seeds)
        for x0 in x_seeds:
            xs, ys = hmap.iterate(x0, 0.0, n_steps)
            mask = (np.abs(xs) < 5) & (np.abs(ys) < 5)
            all_xs.extend(xs[mask].tolist())
            all_ys.extend(ys[mask].tolist())
        results[K] = (np.array(all_xs), np.array(all_ys))
    return results
