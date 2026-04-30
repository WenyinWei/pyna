"""pyna.topo.dynamics -- Base classes for dynamical systems.

This module defines the abstract foundations of the class hierarchy:
  PhaseSpace, DynamicalSystem, ContinuousFlow, DiscreteMap, PoincareMap.

These are intentionally abstract / thin — the mathematics says "there exists
a flow φ^t", not "the flow is computed by this specific ODE solver".
Concrete subclasses live in pyna.toroidal and other domain packages
(magnetic field line systems, Hamiltonian flows, etc.),
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
            "Use a subclass (e.g. MCFPoincareMap) that implements "
            "field-line tracing to the section."
        )


# ────────────────────────────────────────────────────────────────────────────
# MCF concrete Poincaré map  (cyna-accelerated)
# ────────────────────────────────────────────────────────────────────────────

class MCFPoincareMap(DiscreteMap):
    """Poincaré return map for toroidal field-line tracing, backed by cyna C++.

    This is the concrete toroidal implementation of :class:`PoincareMap`.
    Instead of ODE integration via scipy.solve_ivp, it uses the cyna C++
    extension (``pyna._cyna``) for fast parallel field-line tracing.

    The map P: (R, Z) → (R', Z') traces a field line from section
    ``phi_section`` for exactly ``n_turns`` toroidal turns and returns the
    next intersection with the same section.

    Parameters
    ----------
    field_cache : dict
        Field cache dict with keys:
        ``'BR', 'BPhi', 'BZ'``  — 3-D arrays (NR, NZ, NPhi)
        ``'R_grid', 'Z_grid', 'Phi_grid'``  — 1-D coordinate arrays
    Np : int
        Field toroidal periodicity (stellarator period).
    phi_section : float
        Toroidal angle of the Poincaré section [rad].
    n_turns : int
        Number of toroidal turns per map application (default 1).
    DPhi : float
        cyna RK4 step size [rad] (default 0.05).
    n_threads : int
        Number of parallel cyna threads (0 = auto).

    Examples
    --------
    >>> pm = MCFPoincareMap(field_cache, Np=2, phi_section=0.0)
    >>> R1, Z1 = pm.step(np.array([1.5, 0.0]))  # one-turn map
    >>> cloud = pm.poincare_trace(R_arr, Z_arr, n_turns=200)
    """

    def __init__(
        self,
        field_cache: dict,
        *,
        Np: int = 1,
        phi_section: float = 0.0,
        n_turns: int = 1,
        DPhi: float = 0.05,
        n_threads: int = 0,
    ):
        self._fc = field_cache
        self._Np = int(Np)
        self._phi_section = float(phi_section)
        self._n_turns = int(n_turns)
        self._DPhi = float(DPhi)
        self._n_threads = int(n_threads)

        # Pre-process field arrays (extend phi periodicity)
        self._BR_c, self._BPhi_c, self._BZ_c, self._Rg, self._Zg, self._Pg = \
            self._prepare_arrays(field_cache)

    # ── Array preparation ─────────────────────────────────────────────────────

    @staticmethod
    def _prepare_arrays(fc: dict):
        """Extend phi periodicity and ensure C-contiguous float64 arrays.

        The cyna interp3d function requires nPhi = N_phi_original + 1,
        with the last phi slice = 2*pi (copy of phi=0 data). If the input
        already has this extension, we skip the extra copy to avoid a
        stride mismatch between B arrays and Phi_grid.
        """
        Phi_grid = np.asarray(fc['Phi_grid'], dtype=np.float64)
        needs_ext = abs(Phi_grid[-1] - 2 * np.pi) > 1e-6

        if needs_ext:
            Phi_ext = np.append(Phi_grid, 2 * np.pi)
        else:
            Phi_ext = Phi_grid.copy()

        def _ext(a):
            a = np.asarray(a, dtype=np.float64)
            return np.ascontiguousarray(np.concatenate([a, a[:, :, :1]], axis=2))

        if needs_ext:
            BR_c   = _ext(fc['BR'])
            BPhi_c = _ext(fc['BPhi'])
            BZ_c   = _ext(fc['BZ'])
        else:
            # Phi already includes 2*pi copy → B arrays already have the right size
            BR_c   = np.ascontiguousarray(fc['BR'], dtype=np.float64)
            BPhi_c = np.ascontiguousarray(fc['BPhi'], dtype=np.float64)
            BZ_c   = np.ascontiguousarray(fc['BZ'], dtype=np.float64)
        Rg = np.ascontiguousarray(fc['R_grid'],  dtype=np.float64)
        Zg = np.ascontiguousarray(fc['Z_grid'],  dtype=np.float64)
        Pg = np.ascontiguousarray(Phi_ext)

        return BR_c, BPhi_c, BZ_c, Rg, Zg, Pg

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def field_cache(self) -> dict:
        """The field cache dict (BR, BPhi, BZ, grids)."""
        return self._fc

    @property
    def Np(self) -> int:
        """Field toroidal periodicity."""
        return self._Np

    @property
    def phi_section(self) -> float:
        """Toroidal angle of the Poincaré section [rad]."""
        return self._phi_section

    @property
    def n_turns(self) -> int:
        """Number of toroidal turns per map application."""
        return self._n_turns

    @property
    def phase_space(self) -> "PhaseSpace":
        """MCF 2D phase space (R, Z)."""
        return MCF_2D

    # ── DiscreteMap interface ─────────────────────────────────────────────────

    def step(self, x: np.ndarray) -> np.ndarray:
        """Apply the Poincaré map once: (R, Z) → P(R, Z).

        Uses cyna ``trace_poincare_batch`` to trace one field-line turn
        from ``phi_section`` and return the next crossing.

        Parameters
        ----------
        x : ndarray of shape (2,)
            Initial point (R, Z) on the section.

        Returns
        -------
        ndarray of shape (2,)
            Next crossing (R', Z'), or (nan, nan) if lost.
        """
        R_out, Z_out = self._batch_step(np.array([x[0]]), np.array([x[1]]))
        return np.array([float(R_out[0]), float(Z_out[0])])

    def step_n(self, x: np.ndarray, n: int) -> np.ndarray:
        """Apply the map n times: x → P^n(x).

        Parameters
        ----------
        x : ndarray of shape (2,)
        n : int

        Returns
        -------
        ndarray of shape (2,)
        """
        pt = np.asarray(x, dtype=float).copy()
        for _ in range(n):
            pt = self.step(pt)
            if np.any(np.isnan(pt)):
                break
        return pt

    def poincare_trace(
        self,
        R_arr: np.ndarray,
        Z_arr: np.ndarray,
        n_turns: int,
    ) -> tuple:
        """Batch Poincaré trace for multiple starting points.

        Parameters
        ----------
        R_arr, Z_arr : ndarray of shape (N,)
            Starting (R, Z) positions on the section.
        n_turns : int
            Number of Poincaré iterations to collect.

        Returns
        -------
        (R_out, Z_out) : tuple of ndarray, each shape (N, n_turns)
            Poincaré crossing arrays for each seed.
        """
        try:
            import pyna._cyna as _cyna
            result = _cyna.trace_poincare_batch(
                np.ascontiguousarray(R_arr, dtype=np.float64),
                np.ascontiguousarray(Z_arr, dtype=np.float64),
                float(self._phi_section),
                int(n_turns),
                float(self._DPhi),
                self._BR_c, self._BPhi_c, self._BZ_c,
                self._Rg, self._Zg, self._Pg,
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                int(self._n_threads),
            )
            n_seeds = len(R_arr)
            R_out = result[1].reshape(n_seeds, n_turns)
            Z_out = result[2].reshape(n_seeds, n_turns)
            return R_out, Z_out
        except Exception as exc:
            raise RuntimeError(
                f"MCFPoincareMap.poincare_trace: cyna trace failed: {exc}"
            ) from exc

    def _batch_step(
        self,
        R_arr: np.ndarray,
        Z_arr: np.ndarray,
    ) -> tuple:
        """Internal: one-turn batch step for N points."""
        try:
            import pyna._cyna as _cyna
            result = _cyna.trace_poincare_batch(
                np.ascontiguousarray(R_arr, dtype=np.float64),
                np.ascontiguousarray(Z_arr, dtype=np.float64),
                float(self._phi_section),
                int(self._n_turns),
                float(self._DPhi),
                self._BR_c, self._BPhi_c, self._BZ_c,
                self._Rg, self._Zg, self._Pg,
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                int(self._n_threads),
            )
            n_seeds = len(R_arr)
            R_out_full = result[1].reshape(n_seeds, self._n_turns)
            Z_out_full = result[2].reshape(n_seeds, self._n_turns)
            if R_out_full.ndim == 2:
                return R_out_full[:, 0], Z_out_full[:, 0]
            return R_out_full, Z_out_full
        except Exception as exc:
            raise RuntimeError(
                f"MCFPoincareMap.step: cyna trace failed: {exc}"
            ) from exc

    def __repr__(self) -> str:
        return (f"MCFPoincareMap(Np={self._Np}, phi_section={self._phi_section:.4f}, "
                f"n_turns={self._n_turns})")


# ────────────────────────────────────────────────────────────────────────────
# GeneralPoincareMap  (arbitrary Section, trajectory-scanning)
# ────────────────────────────────────────────────────────────────────────────

class GeneralPoincareMap(DiscreteMap):
    """Poincaré return map for an arbitrary ContinuousFlow and Section.

    This is the generic (dimension-independent) implementation of
    :class:`PoincareMap`.  Unlike :class:`MCFPoincareMap` (which uses cyna
    and is restricted to ToroidalSection), this class works with any
    :class:`~pyna.topo.section.Section` (hyperplane, parametric, etc.) by
    integrating the continuous flow and scanning for crossings via
    :meth:`Section.detect_crossing`.

    For MCF with ToroidalSection: prefer :class:`MCFPoincareMap` (much faster).
    Use :class:`GeneralPoincareMap` when:
    - The section is not a toroidal phi=const plane.
    - The system is not MCF (double pendulum, guiding-centre, etc.).
    - You need a portable, dependency-free Poincaré map.

    Parameters
    ----------
    flow : ContinuousFlow
        The underlying continuous-time system.  Must implement
        :meth:`ContinuousFlow.vector_field`.
    section : Section
        The Poincaré section (arbitrary codim-1 surface).
    dt : float
        Maximum ODE integration step (arc-length or time parameter).
    t_max : float
        Maximum integration length per step (to prevent infinite loops).
    direction : int or None
        Crossing direction filter: +1 (positive normal), -1 (negative),
        or None (both directions accepted).  Default None.

    Examples
    --------
    >>> from pyna.topo.dynamics import StandardMap, GeneralPoincareMap
    >>> from pyna.topo.section import HyperplaneSection
    >>> import numpy as np
    >>> # Poincaré map of a standard map on the theta=0 plane
    >>> sm = StandardMap(K=0.9)
    >>> sec = HyperplaneSection(np.array([1., 0.]), 0.0, phase_dim=2)
    >>> pm = GeneralPoincareMap(sm, sec)
    >>> x0 = np.array([0.1, 0.5])
    >>> x1 = pm.step(x0)
    """

    def __init__(
        self,
        flow: "ContinuousFlow",
        section: "Section",
        *,
        dt: float = 0.05,
        t_max: float = 200.0,
        direction: Optional[int] = None,
    ):
        self._flow = flow
        self._section = section
        self._dt = float(dt)
        self._t_max = float(t_max)
        self._direction = direction

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def flow(self) -> "ContinuousFlow":
        return self._flow

    @property
    def section(self) -> "Section":
        return self._section

    @property
    def phase_space(self) -> "PhaseSpace":
        ps = self._flow.phase_space
        return PhaseSpace(
            dim=ps.dim - 1,
            coordinate_names=ps.coordinate_names[:-1] if ps.coordinate_names else (),
            symplectic=ps.symplectic,
            label=f"Section of {ps.label}",
        )

    # ── Core step ─────────────────────────────────────────────────────────────

    def step(self, x: np.ndarray) -> np.ndarray:
        """Integrate the flow from x and return the next section crossing.

        The flow is integrated in steps of ``dt`` up to ``t_max``.  At each
        step, :meth:`Section.detect_crossing` is called on consecutive
        trajectory points.  The first valid crossing (matching ``direction``
        if set) is returned.

        Parameters
        ----------
        x : ndarray of shape (dim_phase,)
            Starting point on or near the section.
        
        Returns
        -------
        ndarray of shape (dim_phase,)
            The crossing point (in phase-space coordinates), or ``x`` with
            NaN values if no crossing was found within ``t_max``.
        """
        from pyna.topo._rk4 import rk4_integrate

        dim = self._flow.phase_space.dim
        x0 = np.asarray(x, dtype=float).copy()

        # Use the flow's vector_field as ODE rhs
        def rhs(t, y):
            return self._flow.vector_field(np.asarray(y), float(t))

        t = 0.0
        y_prev = x0.copy()
        pt_prev = np.append(y_prev, [t])  # extended with parameter t

        while t < self._t_max:
            t_next = min(t + self._dt, self._t_max)
            # Single RK4 step
            sol = rk4_integrate(rhs, (t, t_next), y_prev, max_step=self._dt)
            if not sol.success or sol.y.shape[1] == 0:
                break
            y_curr = sol.y[:, -1].copy()
            t = float(sol.t[-1])

            # Build 3-component points for detect_crossing: (coord..., param)
            # For 2D flows: (x0, x1, t)
            # Section.detect_crossing expects (3,) for MCF or (dim+1,) in general
            # We use the project() method to map to section coords
            pt_curr_full = np.append(y_curr, [t])

            # detect_crossing for the section
            crossing = self._detect(y_prev, y_curr, t - self._dt, t)
            if crossing is not None:
                return crossing

            y_prev = y_curr

        # No crossing found within t_max
        result = np.full(dim, float('nan'))
        return result

    def _detect(
        self,
        y_prev: np.ndarray,
        y_curr: np.ndarray,
        t_prev: float,
        t_curr: float,
    ) -> Optional[np.ndarray]:
        """Detect crossing between y_prev and y_curr; return crossing or None.

        Handles:
        1. Generic Section via Section.f(x) sign change.
        2. ToroidalSection (legacy) via detect_crossing with (R, Z, phi) tuples.
        """
        from pyna.topo.section import Section as _Section

        sec = self._section

        # Path 1: Section with a defining function f(x)=0 (generic)
        if hasattr(sec, 'f'):
            try:
                f_prev = sec.f(y_prev)
                f_curr = sec.f(y_curr)
                if f_prev * f_curr >= 0:
                    return None  # no sign change
                # Direction filter
                if self._direction is not None:
                    # Estimate normal direction at midpoint
                    mid = 0.5 * (y_prev + y_curr)
                    n = sec.normal(mid)
                    v = y_curr - y_prev
                    dot = float(np.dot(n, v))
                    if self._direction > 0 and dot <= 0:
                        return None
                    if self._direction < 0 and dot >= 0:
                        return None
                # Linear interpolation to find crossing
                t_frac = abs(f_prev) / (abs(f_prev) + abs(f_curr) + 1e-30)
                crossing = y_prev + t_frac * (y_curr - y_prev)
                return crossing
            except NotImplementedError:
                pass  # Section.f not implemented (e.g. ToroidalSection)

        # Path 2: Sections with detect_crossing (3-component format for MCF)
        if hasattr(sec, 'detect_crossing'):
            try:
                # Reconstruct (R, Z, phi) tuples if dim=2 MCF flow
                # Append a fake phi parameter estimated from t
                phi_prev = t_prev % (2 * np.pi)
                phi_curr = t_curr % (2 * np.pi)
                pt_prev_3 = np.append(y_prev[:2], [phi_prev])
                pt_curr_3 = np.append(y_curr[:2], [phi_curr])
                hit = sec.detect_crossing(pt_prev_3, pt_curr_3)
                if hit is not None:
                    return hit[:2]  # return (R, Z) only
            except Exception:
                pass

        return None

    def trajectory(
        self,
        x0: np.ndarray,
        n_crossings: int,
    ) -> np.ndarray:
        """Collect n_crossings successive Poincaré crossings starting from x0.

        Parameters
        ----------
        x0 : ndarray of shape (dim_phase,)
            Starting point.
        n_crossings : int
            Number of crossings to collect.

        Returns
        -------
        ndarray of shape (n_crossings, dim_phase)
            The Poincaré crossing points.
        """
        pts = []
        x = np.asarray(x0, dtype=float).copy()
        for _ in range(n_crossings):
            x = self.step(x)
            if np.any(np.isnan(x)):
                break
            pts.append(x.copy())
        if not pts:
            return np.empty((0, self._flow.phase_space.dim), dtype=float)
        return np.array(pts)

    def __repr__(self) -> str:
        return (f"GeneralPoincareMap(flow={self._flow.__class__.__name__}, "
                f"section={self._section!r}, dt={self._dt})")
