"""pyna.plasma_response — backward-compatibility shim.
Files have moved to pyna.MCF.plasma_response.
"""
from pyna.MCF.plasma_response.PerturbGS import solve_perturbed_gs, compute_plasma_response

__all__ = ["solve_perturbed_gs", "compute_plasma_response"]
