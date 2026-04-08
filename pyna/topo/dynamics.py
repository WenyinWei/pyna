"""pyna.topo.dynamics -- Base classes for dynamical systems.

This module defines the abstract foundations of the class hierarchy:
  PhaseSpace, DynamicalSystem, ContinuousFlow, DiscreteMap, PoincareMap.

These are intentionally abstract / thin — the mathematics says "there exists
a flow φ^t", not "the flow is computed by this specific ODE solver".
Concrete subclasses live in pyna.MCF (magnetic field line systems),
pyna.Hamiltonian, etc.

Layer 0 in the pyna.topo architecture (see ARCHITECTURE.md).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyna.topo.section import Section


# ────────────────────────────────────────────────────────────────────────────
# Phase space
# ────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PhaseSpace:
    """The ambient space where the dynamics lives.

    Parameters
    ----------
    dim : int
        Dimension of the phase space.
    coordinate_names : tuple of str, optional
        Names for coordinates (e.g. ('R', 'Z') for MCF field lines).
    symplectic : bool
        True when the phase space carries a symplectic structure
        (i.e. dim is even and the flow preserves a symplectic form).
    label : str, optional
        Human-readable label.

    Examples
    --------
    MCF field-line phase space (2D, R and Z):
        MCF2D = PhaseSpace(dim=2, coordinate_names=('R', 'Z'), label='MCF poloidal')

    Guiding-centre phase space (4D):
        GC4D = PhaseSpace(dim=4, coordinate_names=('R', 'Z', 'pR', 'pZ'),
                          symplectic=True, label='Guiding centre')
    """
    dim: int
    coordinate_names: Tuple[str, ...] = field(default=())
    symplectic: bool = False
    label: Optional[str] = None

    def __post_init__(self):
        if self.coordinate_names and len(self.coordinate_names) != self.dim:
            raise ValueError(
                f"coordinate_names length {len(self.coordinate_names)} "
                f"must equal dim={self.dim}"
            )

    def zero(self) -> np.ndarray:
        """Return the zero vector in this phase space."""
        return np.zeros(self.dim, dtype=float)

    def __repr__(self) -> str:
        lbl = self.label or f"dim={self.dim}"
        return f"PhaseSpace({lbl})"


# Standard phase spaces
MCF_2D = PhaseSpace(dim=2, coordinate_names=('R', 'Z'), label='MCF 2D (R,Z)')
GC_4D  = PhaseSpace(dim=4, coordinate_names=('R', 'Z', 'pR', 'pZ'),
                    symplectic=True, label='Guiding centre 4D')


# ────────────────────────────────────────────────────────────────────────────
# Dynamical system (abstract base)
# ────────────────────────────────────────────────────────────────────────────

class DynamicalSystem(ABC):
    """Abstract base for a dynamical system on a PhaseSpace.

    A Dynamical system is the pair (PhaseSpace, evolution rule).
    The evolution rule can be a continuous flow φ^t or a discrete map P.

    Subclasses implement either:
      flow(x, t)  → x' = φ^t(x)      for ContinuousFlow
      step(x)     → x' = P(x)         for DiscreteMap
    """

    @property
    @abstractmethod
    def phase_space(self) -> PhaseSpace:
        """The phase space this system lives in."""

    @property
    def dim(self) -> int:
        return self.phase_space.dim

    def make_poincare_map(self, section: "Section") -> "PoincareMap":
        """Construct the Poincaré return map to a given Section.

        Parameters
        ----------
        section : Section
            The Poincaré section (codim-1 surface).

        Returns
        -------
        PoincareMap
        """
        return PoincareMap(flow=self, section=section)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(phase_space={self.phase_space})"


# ────────────────────────────────────────────────────────────────────────────
# Continuous flow
# ────────────────────────────────────────────────────────────────────────────

class ContinuousFlow(DynamicalSystem, ABC):
    """A dynamical system defined by a continuous-time flow φ^t.

    The flow satisfies:
        d/dt φ^t(x) = f(φ^t(x), t)
    where f is the vector field.
    """

    @abstractmethod
    def vector_field(self, x: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Vector field f(x, t) at phase-space point x and time t.

        Returns an array of shape (dim,).
        """

    def flow(self, x0: np.ndarray, t: float, **kwargs) -> np.ndarray:
        """Integrate the flow from x0 for time t.

        Default: RK4 integration.  Override for faster / specialised solvers.
        """
        from pyna.topo._rk4 import rk4_integrate
        # rk4_integrate signature: f(t, x) → dx/dt, returns x(t)
        def f(t_, x_): return self.vector_field(np.asarray(x_), float(t_))
        return rk4_integrate(f, t_span=(0.0, t), y0=x0, **kwargs)

    @property
    def is_time_reversible(self) -> bool:
        """True when the flow has a time-reversal symmetry."""
        return False


class HamiltonianFlow(ContinuousFlow, ABC):
    """A symplectic (Hamiltonian) continuous flow.

    H(q, p) generates the flow via Hamilton's equations:
        dq/dt = ∂H/∂p,   dp/dt = -∂H/∂q
    """

    @abstractmethod
    def hamiltonian(self, x: np.ndarray, t: float = 0.0) -> float:
        """The Hamiltonian H(q, p, t)."""

    def vector_field(self, x: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Hamilton's equations via finite-difference ∂H/∂x."""
        n = self.dim // 2
        eps = 1e-5
        dH = np.zeros(self.dim)
        for i in range(self.dim):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            dH[i] = (self.hamiltonian(xp, t) - self.hamiltonian(xm, t)) / (2 * eps)
        # q = x[:n], p = x[n:]
        f = np.empty(self.dim)
        f[:n] =  dH[n:]   # dq/dt = +∂H/∂p
        f[n:] = -dH[:n]   # dp/dt = -∂H/∂q
        return f

    @property
    def is_time_reversible(self) -> bool:
        return True


class MagneticFieldLine(ContinuousFlow, ABC):
    """Continuous flow along magnetic field lines.

    The ODE is  dx/dφ = (R·B_pol) / B_φ  projected to the (R,Z) poloidal plane.
    The 'time' parameter is the toroidal angle φ.

    This is the base for MCF field-line topology computations.
    """

    @property
    def phase_space(self) -> PhaseSpace:
        return MCF_2D

    @abstractmethod
    def field(self, R: float, Z: float, phi: float) -> Tuple[float, float, float]:
        """Return (B_R, B_phi, B_Z) at (R, Z, φ)."""

    def vector_field(self, x: np.ndarray, t: float = 0.0) -> np.ndarray:
        """RHS of field-line ODE: dx/dφ = [R·B_R/B_φ, R·B_Z/B_φ]."""
        R, Z = float(x[0]), float(x[1])
        BR, BPhi, BZ = self.field(R, Z, float(t))
        if abs(BPhi) < 1e-30:
            return np.zeros(2)
        return np.array([R * BR / BPhi, R * BZ / BPhi])


# ────────────────────────────────────────────────────────────────────────────
# Discrete map
# ────────────────────────────────────────────────────────────────────────────

class DiscreteMap(DynamicalSystem, ABC):
    """A dynamical system defined by a discrete map P: x → P(x).

    Examples: standard map, Hénon map, Poincaré return map.
    """

    @abstractmethod
    def step(self, x: np.ndarray) -> np.ndarray:
        """Apply the map once: x → P(x)."""

    def iterate(self, x0: np.ndarray, n: int) -> np.ndarray:
        """Return P^n(x0) (n applications of the map)."""
        x = np.asarray(x0, dtype=float).copy()
        for _ in range(n):
            x = self.step(x)
        return x

    def orbit(self, x0: np.ndarray, n_iter: int) -> np.ndarray:
        """Return the orbit [x0, P(x0), P²(x0), ..., P^n(x0)].

        Returns array of shape (n_iter+1, dim).
        """
        x = np.asarray(x0, dtype=float).copy()
        pts = [x.copy()]
        for _ in range(n_iter):
            x = self.step(x)
            pts.append(x.copy())
        return np.array(pts)


class StandardMap(DiscreteMap):
    """The Chirikov standard map (twist map on the 2-torus).

    P: (θ, I) → (θ + I + K·sin(θ)  mod 2π,  I + K·sin(θ))

    Parameters
    ----------
    K : float
        Nonlinearity parameter (K=0: integrable; K≳1: stochastic).
    """

    def __init__(self, K: float):
        self.K = float(K)

    @property
    def phase_space(self) -> PhaseSpace:
        return PhaseSpace(dim=2, coordinate_names=('theta', 'I'),
                          symplectic=True, label=f'Standard map (K={self.K:.3f})')

    def step(self, x: np.ndarray) -> np.ndarray:
        theta, I = float(x[0]), float(x[1])
        I_new = I + self.K * np.sin(theta)
        theta_new = (theta + I_new) % (2 * np.pi)
        return np.array([theta_new, I_new])


class PoincareMap(DiscreteMap):
    """The Poincaré return map of a ContinuousFlow to a Section.

    P(x₀) = φ^{T(x₀)}(x₀)  where T(x₀) is the first return time.

    This bridges the continuous and discrete layers: given a ContinuousFlow
    and a Section, the PoincareMap is the canonical DiscreteMap.

    Parameters
    ----------
    flow : ContinuousFlow
        The underlying continuous-time system.
    section : Section
        The Poincaré section.
    """

    def __init__(self, flow: ContinuousFlow, section: "Section"):
        self._flow = flow
        self._section = section

    @property
    def flow(self) -> ContinuousFlow:
        return self._flow

    @property
    def section(self) -> "Section":
        return self._section

    @property
    def phase_space(self) -> PhaseSpace:
        # The map acts on the section (dim - 1)
        ps = self._flow.phase_space
        return PhaseSpace(
            dim=ps.dim - 1,
            coordinate_names=ps.coordinate_names[:-1] if ps.coordinate_names else (),
            symplectic=ps.symplectic,
            label=f"Poincaré section of {ps.label}",
        )

    def step(self, x: np.ndarray) -> np.ndarray:
        """Not generically implemented — requires flow integration to section.

        Concrete MCF implementations override this via cyna field-line tracing.
        """
        raise NotImplementedError(
            "PoincareMap.step requires a concrete integration method. "
            "Use a subclass (e.g. MagneticPoincareMap) that implements "
            "field-line tracing to the section."
        )
