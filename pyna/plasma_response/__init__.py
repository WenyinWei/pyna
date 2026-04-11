"""pyna.plasma_response — backward-compatibility shim.
Files have moved to pyna.toroidal.plasma_response.
"""
from pyna.toroidal.plasma_response import solve_perturbed_gs, compute_plasma_response

__all__ = ["solve_perturbed_gs", "compute_plasma_response"]
