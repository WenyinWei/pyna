"""Backward-compatible magnetic-control namespace.

Prefer importing from :mod:`pyna.toroidal.control` in new code.
"""

from pyna.toroidal.control import *  # noqa: F401, F403
from pyna.toroidal.control import __all__  # noqa: F401
