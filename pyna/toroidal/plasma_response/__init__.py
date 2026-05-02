"""pyna.toroidal.plasma_response — DEPRECATED shim.

⚠️  All plasma-response solvers have moved to :mod:`topoquest.analysis.response`.

This module exists only for backward compatibility and will be removed
in a future release.  Please update imports:

    from pyna.toroidal.plasma_response import compute_plasma_response   # ⛔ old
    from topoquest.analysis.response import compute_plasma_response    # ✅ new
"""

import warnings

warnings.warn(
    "pyna.toroidal.plasma_response is deprecated. "
    "Use topoquest.analysis.response instead.",
    DeprecationWarning, stacklevel=2,
)

from topoquest.analysis.response import *  # noqa: F401, F403, E402
