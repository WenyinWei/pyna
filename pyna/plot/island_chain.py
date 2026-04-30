from __future__ import annotations

import numpy as np

from pyna.plot.island import _as_point_xy, plot_island


def island_chain_section_points(chain, phi: float | None = None) -> dict:
    """Return all section footprints for an IslandChain-like object.

    Semantics:
    - IslandChain = the whole chain, regardless of whether its flux tube is
      connected or disconnected.
    - Island = one object/component of interest.
    - In 3D toroidal flows, a chain may intersect one section in one or more
      islands; this accessor is meant to return them all.
    """
    O_points = []
    X_points = []
    islands = []

    if hasattr(chain, "islands") and chain.islands is not None:
        islands.extend(list(chain.islands))
    if hasattr(chain, "subchains") and chain.subchains:
        for sub in chain.subchains:
            if hasattr(sub, "islands") and sub.islands:
                islands.extend(list(sub.islands))

    for isl in islands:
        O = getattr(isl, "O_point", None)
        if O is not None:
            O_points.append(_as_point_xy(O))
        for x in getattr(isl, "X_points", []):
            X_points.append(_as_point_xy(x))

    if hasattr(chain, "fixed_points") and chain.fixed_points:
        # IslandChainOrbit style: all section representatives already live here
        for fp in chain.fixed_points:
            if phi is not None:
                dphi = np.angle(np.exp(1j * (float(fp.phi) - float(phi))))
                if abs(dphi) > 1e-3:
                    continue
            if getattr(fp, "kind", None) == "O":
                O_points.append((float(fp.R), float(fp.Z)))
            else:
                X_points.append((float(fp.R), float(fp.Z)))
    elif hasattr(chain, "sections") and chain.sections:
        # Cycle style: sections is {phi: List[FixedPoint]}
        for sec_phi, fps in chain.sections.items():
            if phi is not None:
                dphi = np.angle(np.exp(1j * (float(sec_phi) - float(phi))))
                if abs(dphi) > 1e-3:
                    continue
            for fp in (fps if isinstance(fps, list) else [fps]):
                if getattr(fp, "kind", None) == "O":
                    O_points.append((float(fp.R), float(fp.Z)))
                else:
                    X_points.append((float(fp.R), float(fp.Z)))

    return {
        "phi": phi,
        "O_points": O_points,
        "X_points": X_points,
        "islands": islands,
    }


def plot_island_chain(
    chain,
    ax=None,
    phi: float | None = None,
    *,
    show_all_islands: bool = True,
    show_O: bool = True,
    show_X: bool = True,
    alpha_island: float = 0.95,
):
    """Plot the whole IslandChain (all islands / all section intersections)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    sec = island_chain_section_points(chain, phi=phi)

    if show_all_islands:
        for isl in sec["islands"]:
            plot_island(isl, ax=ax, phi=phi, show_O=False, show_X=False, show_label=False)

    if show_O:
        for R, Z in sec["O_points"]:
            ax.plot(R, Z, "o", color="limegreen", ms=6, mec="k", mew=0.5, alpha=alpha_island, zorder=10)
    if show_X:
        for R, Z in sec["X_points"]:
            ax.plot(R, Z, "x", color="crimson", ms=7, mew=1.8, alpha=alpha_island, zorder=10)

    ax.set_aspect("equal")
    return ax
