"""
coords.py — Coordinate systems for pyna.fields.
Depends only on numpy; no intra-package imports.
"""
from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod


class CoordinateSystem(ABC):
    """Abstract coordinate system base."""

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @property
    @abstractmethod
    def coord_names(self) -> list[str]: ...

    @abstractmethod
    def metric_tensor(self, coords: np.ndarray) -> np.ndarray:
        """g_ij at coords, shape (..., dim, dim)."""
        ...

    def christoffel_symbols(self, coords: np.ndarray) -> np.ndarray:
        """Γ^k_ij at coords, shape (..., dim, dim, dim).
        Default: numerical central differences on metric_tensor."""
        coords = np.asarray(coords, dtype=float)
        shape = coords.shape[:-1]
        d = self.dim
        h = 1e-5

        # g and its partial derivatives  ∂_l g_{ij}
        g = self.metric_tensor(coords)           # (..., d, d)
        dg = np.zeros(shape + (d, d, d))         # dg[..., l, i, j] = ∂_l g_ij

        for l in range(d):
            ep = np.zeros(d)
            ep[l] = h
            gp = self.metric_tensor(coords + ep)
            gm = self.metric_tensor(coords - ep)
            dg[..., l, :, :] = (gp - gm) / (2 * h)

        # g^{kl}: inverse of g
        g_inv = np.linalg.inv(g)                 # (..., d, d)

        # Γ^k_ij = (1/2) g^{kl} (∂_i g_{lj} + ∂_j g_{li} - ∂_l g_{ij})
        # dg[..., i, l, j] = ∂_i g_{lj}
        term = (dg[..., np.newaxis, :, :].swapaxes(-3, -1)   # wrong shape; build explicitly
                )
        # Build (d_i g_lj + d_j g_li - d_l g_ij) with shape (..., i, j, l)
        # dg shape: (..., l, i, j)
        # ∂_i g_{lj} → dg[..., l, i, j] indexed as dg[..., l][..., i, j]
        # Rearrange to (..., i, j, l):
        dg_ilj = np.moveaxis(dg, -3, -1)         # (..., i, j, l)  <- was (..., l, i, j)
        # wait: dg[..., l, i, j] means axis order is (..., l, i, j)
        # moveaxis(-3, -1): move axis -3 (l) to -1 → (..., i, j, l)  ✓
        # ∂_i g_{lj}: need index (l, j) with partial w.r.t. i
        #   = dg[..., i, l, j]  in original (..., l, i, j) ordering? No.
        # dg[..., l, i, j] = ∂_l g_{ij}.  So:
        #   ∂_i g_{lj} = dg[..., l, j] with diff index i
        #              = dg permuted so first free index is i: swap axes
        # Let's be explicit with einsum / direct indexing.

        # Recompute cleanly:
        # S[..., i, j, l] = ∂_i g_{lj} + ∂_j g_{li} - ∂_l g_{ij}
        # dg has shape (..., l, i, j) meaning dg[..., a, b, c] = ∂_a g_{bc}
        # ∂_i g_{lj} = dg[..., l, j] w.r.t i → need dg with diff order
        # Actually dg[..., a, b, c] = ∂_a g_{bc}, so:
        #   ∂_i g_{lj} : a=i, b=l, c=j  → dg[..., i, l, j]
        #   ∂_j g_{li} : a=j, b=l, c=i  → dg[..., j, l, i]
        #   ∂_l g_{ij} : a=l, b=i, c=j  → dg[..., l, i, j]
        # dg shape (..., d, d, d)
        # Build S[..., i, j, l]:
        # np.einsum-free approach: use transpose
        # dg[..., i, l, j] → transpose last 3 axes from (l,i,j) to (i,l,j)
        dg_ilj2 = dg.transpose(list(range(len(shape))) + [-2, -3, -1])  # (..., i, l, j)
        # dg[..., j, l, i] → transpose to (j, l, i) but we want (i, j, l):
        dg_jli = dg.transpose(list(range(len(shape))) + [-1, -3, -2])   # (..., j, l, i) = (..., i, l, j) with i<->j and transposing last?
        # This is getting confusing. Use explicit loops or np.einsum.

        # Use np.einsum with named indices
        # S_{ijl} = ∂_i g_{lj} + ∂_j g_{li} - ∂_l g_{ij}
        # dg[..., a, b, c] = ∂_a g_{bc}
        # ∂_i g_{lj} = dg[..., i, l, j]  → we want this as S component
        # Build arrays with proper axis ordering:
        n_batch = int(np.prod(shape)) if shape else 1
        dg_flat = dg.reshape(n_batch, d, d, d)   # (B, l, i, j) = ∂_l g_{ij}

        # S_flat[B, i, j, l] = dg_flat[B, i, l, j] + dg_flat[B, j, l, i] - dg_flat[B, l, i, j]
        S = (dg_flat[:, :, :, :].transpose(0, 2, 3, 1)   # (B, i, j, l) ← (B, l, i, j) with (l→last)
             )
        # Hmm, dg_flat has axes (B, l, i, j). transpose(0,2,3,1) → (B, i, j, l). That gives ∂_l g_{ij} reordered as S[B,i,j,l] = dg[B,l,i,j]. That's ∂_l g_{ij} ✓ (the subtracted term).
        # For ∂_i g_{lj}: we need dg[B, i, l, j]. With axes (B,l,i,j) we want (B,i,l,j) → transpose(0,2,1,3).
        # Then reorder to (B,i,j,l): from (B,i,l,j) → transpose(0,1,3,2).
        dg_ilj_flat = dg_flat.transpose(0, 2, 1, 3)  # (B, i, l, j) = ∂_i g_{lj}
        # Now make (B, i, j, l):  from (B,i,l,j) swap last two: transpose(0,1,3,2)
        term1 = dg_ilj_flat.transpose(0, 1, 3, 2)     # (B, i, j, l) = ∂_i g_{lj}

        # For ∂_j g_{li}: dg[B, j, l, i] → axes (B, l, i, j): want (B, j, l, i) → transpose(0,3,1,2)
        # Then reorder to (B, i, j, l): from (B,j,l,i) → transpose(0,3,1,2)? 
        # Actually we want S[B,i,j,l], so the j-index is axis 2 and i-index is axis 1.
        # ∂_j g_{li}: original dg[B, a, b, c]=∂_a g_{bc}, so ∂_j g_{li} = dg[B, j, l, i].
        # Axes of dg are (B,l,i,j)=(B,0,1,2). dg[B,j,l,i] in terms of axis values:
        #   first free index = j → that's axis 3 of dg_flat
        #   second free index = l → axis 1 of dg_flat
        #   third free index = i → axis 2 of dg_flat
        # So ∂_j g_{li}[B, :, :, :] = dg_flat[B, l_axis, i_axis, j_axis] = dg_flat[:, :, :, :] with (B, j, l, i)
        # To get array (B, j, l, i) from (B, l, i, j): transpose(0, 3, 1, 2)
        dg_jli_flat = dg_flat.transpose(0, 3, 1, 2)   # (B, j, l, i)
        # Reorder to (B, i, j, l): from (B, j, l, i) → swap i and j, move l last
        # (B, j, l, i) → we want (B, i, j, l): swap axis 1 and 3, then axis 2 and 3? 
        # (B, j, l, i): axes 0,1,2,3. Want (B, i, j, l) = (B, axis3, axis1, axis2):
        term2 = dg_jli_flat.transpose(0, 3, 1, 2)     # (B, i, j, l) = ∂_j g_{li}

        # term3: S[B,i,j,l] = dg[B,l,i,j], already computed as S from before
        term3 = S                                       # (B, i, j, l) = ∂_l g_{ij}

        S_final = term1 + term2 - term3                # (B, i, j, l)

        # Γ^k_ij = (1/2) g^{kl} S_{ijl}
        g_inv_flat = np.linalg.inv(g.reshape(n_batch, d, d))  # (B, d, d)
        # gamma[B, k, i, j] = 0.5 * sum_l g_inv[B, k, l] * S_final[B, i, j, l]
        gamma_flat = 0.5 * np.einsum('bkl,bijl->bkij', g_inv_flat, S_final)  # (B, k, i, j)

        return gamma_flat.reshape(shape + (d, d, d))

    @abstractmethod
    def to_cartesian(self, coords: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def from_cartesian(self, coords: np.ndarray) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# CartesianCoords
# ---------------------------------------------------------------------------

class CartesianCoords(CoordinateSystem):
    """General n-dimensional Cartesian coordinate system."""

    _DEFAULT_NAMES = ['x', 'y', 'z', 'w']

    def __init__(self, dim: int = 3, names: list[str] | None = None):
        self._dim = dim
        if names is not None:
            self._names = list(names)
        else:
            if dim <= len(self._DEFAULT_NAMES):
                self._names = self._DEFAULT_NAMES[:dim]
            else:
                self._names = [f'x{i}' for i in range(dim)]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def coord_names(self) -> list[str]:
        return self._names

    def metric_tensor(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        shape = coords.shape[:-1]
        g = np.broadcast_to(np.eye(self._dim), shape + (self._dim, self._dim)).copy()
        return g

    def christoffel_symbols(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        shape = coords.shape[:-1]
        return np.zeros(shape + (self._dim, self._dim, self._dim))

    def to_cartesian(self, coords: np.ndarray) -> np.ndarray:
        return np.asarray(coords, dtype=float).copy()

    def from_cartesian(self, coords: np.ndarray) -> np.ndarray:
        return np.asarray(coords, dtype=float).copy()


# ---------------------------------------------------------------------------
# CylindricalCoords3D
# ---------------------------------------------------------------------------

class CylindricalCoords3D(CoordinateSystem):
    """Cylindrical coordinates (R, Z, phi)."""

    @property
    def dim(self) -> int:
        return 3

    @property
    def coord_names(self) -> list[str]:
        return ['R', 'Z', 'phi']

    def metric_tensor(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        shape = coords.shape[:-1]
        R = coords[..., 0]
        g = np.zeros(shape + (3, 3))
        g[..., 0, 0] = 1.0
        g[..., 1, 1] = 1.0
        g[..., 2, 2] = R ** 2
        return g

    def christoffel_symbols(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        shape = coords.shape[:-1]
        R = coords[..., 0]
        gamma = np.zeros(shape + (3, 3, 3))
        # Γ^R_φφ = -R  →  gamma[..., 0, 2, 2]
        gamma[..., 0, 2, 2] = -R
        # Γ^φ_Rφ = Γ^φ_φR = 1/R  →  gamma[..., 2, 0, 2] and gamma[..., 2, 2, 0]
        gamma[..., 2, 0, 2] = 1.0 / R
        gamma[..., 2, 2, 0] = 1.0 / R
        return gamma

    def to_cartesian(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        R, Z, phi = coords[..., 0], coords[..., 1], coords[..., 2]
        out = np.stack([R * np.cos(phi), R * np.sin(phi), Z], axis=-1)
        return out

    def from_cartesian(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
        R = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return np.stack([R, z, phi], axis=-1)


# ---------------------------------------------------------------------------
# SphericalCoords3D
# ---------------------------------------------------------------------------

class SphericalCoords3D(CoordinateSystem):
    """Spherical coordinates (r, theta, phi)."""

    @property
    def dim(self) -> int:
        return 3

    @property
    def coord_names(self) -> list[str]:
        return ['r', 'theta', 'phi']

    def metric_tensor(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        shape = coords.shape[:-1]
        r = coords[..., 0]
        theta = coords[..., 1]
        g = np.zeros(shape + (3, 3))
        g[..., 0, 0] = 1.0
        g[..., 1, 1] = r**2
        g[..., 2, 2] = (r * np.sin(theta))**2
        return g

    def christoffel_symbols(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        shape = coords.shape[:-1]
        r = coords[..., 0]
        theta = coords[..., 1]
        gamma = np.zeros(shape + (3, 3, 3))
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        # Γ^r_θθ = -r
        gamma[..., 0, 1, 1] = -r
        # Γ^r_φφ = -r sin²θ
        gamma[..., 0, 2, 2] = -r * sin_t**2
        # Γ^θ_rθ = Γ^θ_θr = 1/r
        gamma[..., 1, 0, 1] = 1.0 / r
        gamma[..., 1, 1, 0] = 1.0 / r
        # Γ^θ_φφ = -sinθ cosθ
        gamma[..., 1, 2, 2] = -sin_t * cos_t
        # Γ^φ_rφ = Γ^φ_φr = 1/r
        gamma[..., 2, 0, 2] = 1.0 / r
        gamma[..., 2, 2, 0] = 1.0 / r
        # Γ^φ_θφ = Γ^φ_φθ = cosθ/sinθ
        gamma[..., 2, 1, 2] = cos_t / sin_t
        gamma[..., 2, 2, 1] = cos_t / sin_t
        return gamma

    def to_cartesian(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        r, theta, phi = coords[..., 0], coords[..., 1], coords[..., 2]
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.stack([x, y, z], axis=-1)

    def from_cartesian(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return np.stack([r, theta, phi], axis=-1)


# ---------------------------------------------------------------------------
# TorCoords3D (Tokamak toroidal)
# ---------------------------------------------------------------------------

class TorCoords3D(CoordinateSystem):
    """Tokamak toroidal coordinates (r, theta_pol, phi_tor)."""

    def __init__(self, R0: float = 3.0):
        """R0: major radius [m]."""
        self.R0 = R0

    @property
    def dim(self) -> int:
        return 3

    @property
    def coord_names(self) -> list[str]:
        return ['r', 'theta_pol', 'phi_tor']

    def metric_tensor(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        shape = coords.shape[:-1]
        r = coords[..., 0]
        theta = coords[..., 1]
        R_eff = self.R0 + r * np.cos(theta)
        g = np.zeros(shape + (3, 3))
        g[..., 0, 0] = 1.0
        g[..., 1, 1] = r**2
        g[..., 2, 2] = R_eff**2
        return g

    # christoffel_symbols: use numerical default from base class

    def to_cartesian(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        r, theta, phi = coords[..., 0], coords[..., 1], coords[..., 2]
        R_eff = self.R0 + r * np.cos(theta)
        x = R_eff * np.cos(phi)
        y = R_eff * np.sin(phi)
        z = r * np.sin(theta)
        return np.stack([x, y, z], axis=-1)

    def from_cartesian(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
        R_cyl = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        r = np.sqrt((R_cyl - self.R0)**2 + z**2)
        theta = np.arctan2(z, R_cyl - self.R0)
        return np.stack([r, theta, phi], axis=-1)


# ---------------------------------------------------------------------------
# MinkowskiCoords4D
# ---------------------------------------------------------------------------

class MinkowskiCoords4D(CoordinateSystem):
    """Minkowski spacetime (t, x, y, z)."""

    def __init__(self, c: float = 1.0, signature: str = '-+++'):
        self.c = c
        self.signature = signature

    @property
    def dim(self) -> int:
        return 4

    @property
    def coord_names(self) -> list[str]:
        return ['t', 'x', 'y', 'z']

    def metric_tensor(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        shape = coords.shape[:-1]
        g = np.zeros(shape + (4, 4))
        if self.signature == '-+++':
            g[..., 0, 0] = -(self.c**2)
            g[..., 1, 1] = 1.0
            g[..., 2, 2] = 1.0
            g[..., 3, 3] = 1.0
        else:  # '+---'
            g[..., 0, 0] = self.c**2
            g[..., 1, 1] = -1.0
            g[..., 2, 2] = -1.0
            g[..., 3, 3] = -1.0
        return g

    def christoffel_symbols(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        shape = coords.shape[:-1]
        return np.zeros(shape + (4, 4, 4))

    def to_cartesian(self, coords: np.ndarray) -> np.ndarray:
        return np.asarray(coords, dtype=float).copy()

    def from_cartesian(self, coords: np.ndarray) -> np.ndarray:
        return np.asarray(coords, dtype=float).copy()


# ---------------------------------------------------------------------------
# SchwarzschildCoords4D
# ---------------------------------------------------------------------------

class SchwarzschildCoords4D(CoordinateSystem):
    """Schwarzschild spacetime (t, r, theta, phi)."""

    def __init__(self, M: float, G: float = 1.0, c: float = 1.0):
        self.M = M
        self.G = G
        self.c = c

    @property
    def dim(self) -> int:
        return 4

    @property
    def coord_names(self) -> list[str]:
        return ['t', 'r', 'theta', 'phi']

    @property
    def schwarzschild_radius(self) -> float:
        return 2.0 * self.G * self.M / (self.c**2)

    def metric_tensor(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        shape = coords.shape[:-1]
        r = coords[..., 1]
        theta = coords[..., 2]
        rs = self.schwarzschild_radius
        f = 1.0 - rs / r
        g = np.zeros(shape + (4, 4))
        g[..., 0, 0] = -(self.c**2) * f
        g[..., 1, 1] = 1.0 / f
        g[..., 2, 2] = r**2
        g[..., 3, 3] = (r * np.sin(theta))**2
        return g

    # christoffel_symbols: numerical (base class)

    def to_cartesian(self, coords: np.ndarray) -> np.ndarray:
        """Spherical (r,theta,phi) part → Cartesian; t passed through."""
        coords = np.asarray(coords, dtype=float)
        t = coords[..., 0]
        r, theta, phi = coords[..., 1], coords[..., 2], coords[..., 3]
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.stack([t, x, y, z], axis=-1)

    def from_cartesian(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        t = coords[..., 0]
        x, y, z = coords[..., 1], coords[..., 2], coords[..., 3]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return np.stack([t, r, theta, phi], axis=-1)


# ---------------------------------------------------------------------------
# KerrCoords4D (Boyer-Lindquist)
# ---------------------------------------------------------------------------

class KerrCoords4D(CoordinateSystem):
    """Kerr spacetime in Boyer-Lindquist coordinates (t, r, theta, phi)."""

    def __init__(self, M: float, a: float, G: float = 1.0, c: float = 1.0):
        self.M = M
        self.a = a
        self.G = G
        self.c = c

    @property
    def dim(self) -> int:
        return 4

    @property
    def coord_names(self) -> list[str]:
        return ['t', 'r', 'theta', 'phi']

    @property
    def event_horizon_radius(self) -> float:
        """r_+ = M + sqrt(M² - a²)  (geometric units G=c=1 scaled)."""
        # In geometric units with G,c: r_+ = GM/c² + sqrt((GM/c²)² - a²)
        # But a is given as the spin parameter. Standard BL uses GM/c² as mass scale.
        # Here we follow: r_+ = M + sqrt(M² - a²) treating M as GM/c² already,
        # but with explicit G,c: M_geom = GM/c², so
        # r_+ = GM/c² + sqrt((GM/c²)² - a²)
        M_g = self.G * self.M / self.c**2
        return M_g + np.sqrt(M_g**2 - self.a**2)

    def ergosphere_radius(self, theta: float) -> float:
        """r_ergo = GM/c² + sqrt((GM/c²)² - a²cos²θ)."""
        M_g = self.G * self.M / self.c**2
        return M_g + np.sqrt(M_g**2 - self.a**2 * np.cos(theta)**2)

    def metric_tensor(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        shape = coords.shape[:-1]
        r = coords[..., 1]
        theta = coords[..., 2]
        M_g = self.G * self.M / (self.c**2)  # GM/c²
        a = self.a
        c = self.c
        Sigma = r**2 + a**2 * np.cos(theta)**2
        Delta = r**2 - 2 * M_g * r + a**2
        sin2 = np.sin(theta)**2

        g = np.zeros(shape + (4, 4))
        g[..., 0, 0] = -(1.0 - 2.0 * M_g * r / Sigma) * c**2
        g[..., 1, 1] = Sigma / Delta
        g[..., 2, 2] = Sigma
        g[..., 3, 3] = (r**2 + a**2 + 2.0 * M_g * a**2 * r * sin2 / Sigma) * sin2
        g_tphi = -2.0 * M_g * a * r * sin2 / Sigma * c
        g[..., 0, 3] = g_tphi
        g[..., 3, 0] = g_tphi
        return g

    # christoffel_symbols: numerical (base class)

    def to_cartesian(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        t = coords[..., 0]
        r, theta, phi = coords[..., 1], coords[..., 2], coords[..., 3]
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.stack([t, x, y, z], axis=-1)

    def from_cartesian(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        t = coords[..., 0]
        x, y, z = coords[..., 1], coords[..., 2], coords[..., 3]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return np.stack([t, r, theta, phi], axis=-1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Schwarzschild r_s
    cs = SchwarzschildCoords4D(M=1.0)
    assert abs(cs.schwarzschild_radius - 2.0) < 1e-12, f"r_s = {cs.schwarzschild_radius}"

    # 2. Cylindrical metric at (R=2, Z=0, phi=0)
    cyl = CylindricalCoords3D()
    g = cyl.metric_tensor(np.array([[2.0, 0.0, 0.0]]))
    assert abs(g[0, 0, 0] - 1.0) < 1e-10, f"g_RR = {g[0,0,0]}"
    assert abs(g[0, 2, 2] - 4.0) < 1e-10, f"g_phiphi = {g[0,2,2]}"

    # 3. Cylindrical to_cartesian
    xyz = cyl.to_cartesian(np.array([[2.0, 1.0, np.pi / 2]]))
    assert abs(xyz[0, 0] - 0.0) < 1e-10, f"x = {xyz[0,0]}"
    assert abs(xyz[0, 1] - 2.0) < 1e-10, f"y = {xyz[0,1]}"

    # 4. Kerr event horizon  (G=c=1, M=1, a=0.5)
    kerr = KerrCoords4D(M=1.0, a=0.5)
    r_plus = 1.0 + np.sqrt(1.0 - 0.25)
    assert abs(kerr.event_horizon_radius - r_plus) < 1e-10, f"r_+ = {kerr.event_horizon_radius}"

    print("All coords tests passed.")
