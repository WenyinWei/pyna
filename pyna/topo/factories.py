"""Registries and factories for dynamics and topology objects."""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping, Optional


def _key(name: str) -> str:
    return str(name).strip().lower().replace("_", "-")


class Registry:
    """Small named factory registry with explicit duplicate handling."""

    def __init__(self, initial: Optional[Mapping[str, Callable[..., Any]]] = None) -> None:
        self._items: Dict[str, Callable[..., Any]] = {}
        for name, factory in dict(initial or {}).items():
            self.register(name, factory)

    def register(
        self,
        name: str,
        factory: Optional[Callable[..., Any]] = None,
        *,
        overwrite: bool = False,
    ):
        """Register a factory or use as a decorator."""

        norm = _key(name)

        def _install(func: Callable[..., Any]) -> Callable[..., Any]:
            if not overwrite and norm in self._items:
                raise KeyError(f"registry key {norm!r} is already registered.")
            self._items[norm] = func
            return func

        if factory is None:
            return _install
        return _install(factory)

    def unregister(self, name: str) -> None:
        norm = _key(name)
        if norm not in self._items:
            raise KeyError(f"registry key {norm!r} is not registered.")
        del self._items[norm]

    def get(self, name: str) -> Callable[..., Any]:
        norm = _key(name)
        try:
            return self._items[norm]
        except KeyError as exc:
            known = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(f"unknown registry key {norm!r}; known keys: {known}") from exc

    def create(self, name: str, **kwargs: Any) -> Any:
        return self.get(name)(**kwargs)

    def keys(self) -> Iterable[str]:
        return tuple(sorted(self._items))

    def copy(self) -> "Registry":
        return Registry(self._items)


class DynamicalSystemFactory:
    """Factory for ready-to-use finite-dimensional dynamical systems."""

    _registry: Optional[Registry] = None

    @classmethod
    def registry(cls) -> Registry:
        if cls._registry is None:
            cls._registry = Registry()
            cls._register_defaults(cls._registry)
        return cls._registry

    @staticmethod
    def _register_defaults(registry: Registry) -> None:
        registry.register("callable-flow", lambda **kw: __import__("pyna.dynamics", fromlist=["CallableFlow"]).CallableFlow(**kw))
        registry.register("flow", lambda **kw: __import__("pyna.dynamics", fromlist=["CallableFlow"]).CallableFlow(**kw))
        registry.register("callable-map", lambda **kw: __import__("pyna.dynamics", fromlist=["CallableMap"]).CallableMap(**kw))
        registry.register("map", lambda **kw: __import__("pyna.dynamics", fromlist=["CallableMap"]).CallableMap(**kw))
        registry.register("hamiltonian", lambda **kw: __import__("pyna.dynamics", fromlist=["HamiltonianSystem"]).HamiltonianSystem(**kw))
        registry.register(
            "separable-hamiltonian",
            lambda **kw: __import__("pyna.dynamics", fromlist=["SeparableHamiltonianSystem"]).SeparableHamiltonianSystem(**kw),
        )
        registry.register("nbody", lambda **kw: __import__("pyna.dynamics", fromlist=["NBodySystem"]).NBodySystem(**kw))
        registry.register("ito-sde", lambda **kw: __import__("pyna.dynamics", fromlist=["ItoSDE"]).ItoSDE(**kw))
        registry.register("brownian-motion", lambda **kw: __import__("pyna.dynamics", fromlist=["BrownianMotion"]).BrownianMotion(**kw))
        registry.register(
            "geometric-brownian-motion",
            lambda **kw: __import__("pyna.dynamics", fromlist=["GeometricBrownianMotion"]).GeometricBrownianMotion(**kw),
        )

    @classmethod
    def register(cls, name: str, factory: Optional[Callable[..., Any]] = None, *, overwrite: bool = False):
        return cls.registry().register(name, factory, overwrite=overwrite)

    @classmethod
    def create(cls, kind: str, **kwargs: Any) -> Any:
        return cls.registry().create(kind, **kwargs)

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> Any:
        data = dict(spec)
        kind = data.pop("kind")
        params = dict(data.pop("params", {}))
        params.update(data)
        return cls.create(kind, **params)


class GeometryFactory:
    """Factory for topology geometry using the explicit builder layer."""

    @staticmethod
    def create(kind: str, **kwargs: Any) -> Any:
        from pyna.topo.builders import GeometryBuilder, IslandChainBuilder, TubeChainBuilder
        from pyna.topo.core import IslandChain, TubeChain

        norm = _key(kind)
        builder = GeometryBuilder(closure_tol=float(kwargs.pop("closure_tol", 1e-8)))
        if norm == "trajectory":
            obj = kwargs.pop("obj", kwargs.pop("states", None))
            return builder.trajectory(obj, **kwargs)
        if norm == "orbit":
            obj = kwargs.pop("obj", kwargs.pop("states", None))
            return builder.orbit(obj, **kwargs)
        if norm == "cycle":
            obj = kwargs.pop("obj")
            return builder.cycle(obj, **kwargs)
        if norm in {"periodic-orbit", "periodic-orbit-core"}:
            obj = kwargs.pop("obj", kwargs.pop("points", None))
            return builder.periodic_orbit(obj, **kwargs)
        if norm == "island-chain":
            if "islands" in kwargs:
                return IslandChain(islands=list(kwargs.pop("islands")), **kwargs)
            return IslandChainBuilder.from_periodic_orbits(**kwargs)
        if norm == "tube-chain":
            if "tubes" in kwargs:
                return TubeChain(tubes=list(kwargs.pop("tubes")), **kwargs)
            return TubeChainBuilder.from_cycles(**kwargs)
        known = "trajectory, orbit, cycle, periodic-orbit, island-chain, tube-chain"
        raise KeyError(f"unknown geometry kind {kind!r}; known kinds: {known}")

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> Any:
        data = dict(spec)
        kind = data.pop("kind")
        params = dict(data.pop("params", {}))
        params.update(data)
        return cls.create(kind, **params)


class PoincareMapFactory:
    """Factory for executable Poincare map implementations."""

    @staticmethod
    def create(flow: Any, section: Any, *, backend: str = "auto", **kwargs: Any) -> Any:
        from pyna.topo.dynamics import GeneralPoincareMap, MCFPoincareMap, PoincareMap
        from pyna.topo.section import ToroidalSection, coerce_section

        norm = _key(backend)
        section_obj = coerce_section(section)
        if norm == "auto":
            has_field_cache = "field_cache" in kwargs or getattr(flow, "field_cache", None) is not None
            norm = "mcf-cyna" if has_field_cache else "general"

        if norm in {"general", "rk4", "portable"}:
            allowed = {"dt", "t_max", "direction"}
            opts = {name: kwargs[name] for name in allowed if name in kwargs}
            return GeneralPoincareMap(flow, section_obj, **opts)

        if norm in {"abstract", "base"}:
            return PoincareMap(flow=flow, section=section_obj)

        if norm in {"mcf-cyna", "cyna", "toroidal-cyna"}:
            field_cache = kwargs.pop("field_cache", getattr(flow, "field_cache", None))
            if field_cache is None:
                raise ValueError("mcf-cyna backend requires field_cache or flow.field_cache.")
            phi_section = kwargs.pop(
                "phi_section",
                float(section_obj.phi) if isinstance(section_obj, ToroidalSection) else 0.0,
            )
            opts = {
                "Np": kwargs.pop("Np", kwargs.pop("nfp", 1)),
                "phi_section": phi_section,
                "n_turns": kwargs.pop("n_turns", 1),
                "DPhi": kwargs.pop("DPhi", 0.05),
                "n_threads": kwargs.pop("n_threads", 0),
            }
            return MCFPoincareMap(field_cache, **opts)

        raise KeyError("backend must be 'auto', 'general', 'abstract', or 'mcf-cyna'.")


__all__ = [
    "Registry",
    "DynamicalSystemFactory",
    "GeometryFactory",
    "PoincareMapFactory",
]
