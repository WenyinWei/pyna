"""Legacy compatibility shim for :mod:`pyna.toroidal.torus_deformation`.

The canonical implementation now lives in :mod:`pyna.toroidal.torus_deformation`.
New code should import from :mod:`pyna.toroidal`.
"""

from pyna.toroidal.torus_deformation import *  # noqa: F401,F403
