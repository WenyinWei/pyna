"""pyna.toroidal.perturbation.response — DEPRECATED shim.

⚠️  All toroidal plasma-response functionality has moved to
:mod:`topoquest.analysis.response`.

Update imports:
    from pyna.toroidal.perturbation.response import compute_plasma_response   # ⛔
    from topoquest.analysis.response import compute_plasma_response          # ✅
"""

import warnings

warnings.warn(
    "pyna.toroidal.perturbation.response is deprecated. "
    "Use topoquest.analysis.response instead.",
    DeprecationWarning, stacklevel=2,
)

from topoquest.analysis.response import *  # noqa: F401, F403, E402
