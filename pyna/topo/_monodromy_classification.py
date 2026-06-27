"""Shared 2x2 monodromy classification helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MonodromyClassification:
    """Classification and diagnostics for a 2x2 return-map Jacobian."""

    kind: str
    trace: float
    determinant: float
    discriminant: float
    eigenvalues: np.ndarray
    area_preserving: bool
    reason: str


def classify_monodromy_2x2(
    DPm,
    *,
    det_tol: float = 5.0e-2,
    eig_tol: float = 1.0e-8,
) -> MonodromyClassification:
    """Classify a 2-D Poincare-map monodromy matrix.

    ``|Tr| > 2`` is only a safe X/O shortcut for an area-preserving 2x2 map;
    both ``Tr > 2`` and ``Tr < -2`` are hyperbolic X cases.  Field-line return
    maps should therefore pass a determinant sanity check before we use the
    eigen-structure for X/O labels.  Matrices that fail the consistency checks
    are returned as ``kind == "U"`` instead of being folded into O-points.
    """

    mat = np.asarray(DPm, dtype=float)
    nan_eig = np.asarray([np.nan + 0j, np.nan + 0j])
    if mat.shape != (2, 2) or not np.all(np.isfinite(mat)):
        return MonodromyClassification("U", np.nan, np.nan, np.nan, nan_eig, False, "invalid_matrix")

    tr = float(np.trace(mat))
    det = float(np.linalg.det(mat))
    disc = float(tr * tr - 4.0 * det)
    try:
        eig = np.linalg.eigvals(mat)
    except np.linalg.LinAlgError:
        return MonodromyClassification("U", tr, det, disc, nan_eig, False, "eig_failed")

    if not (np.isfinite(tr) and np.isfinite(det) and np.isfinite(disc) and np.all(np.isfinite(eig))):
        return MonodromyClassification("U", tr, det, disc, eig, False, "nonfinite_spectrum")

    area_like = abs(det - 1.0) <= float(det_tol)
    if not area_like:
        return MonodromyClassification("U", tr, det, disc, eig, False, "det_not_area_preserving")

    imag_scale = max(1.0, float(np.max(np.abs(eig.real))))
    real_pair = bool(np.max(np.abs(eig.imag)) <= float(eig_tol) * imag_scale)
    mods = np.abs(eig)

    if real_pair:
        if (
            (mods[0] > 1.0 + eig_tol and mods[1] < 1.0 - eig_tol)
            or (mods[1] > 1.0 + eig_tol and mods[0] < 1.0 - eig_tol)
        ):
            return MonodromyClassification("X", tr, det, disc, eig, True, "real_stable_unstable_pair")
        if np.all(np.abs(mods - 1.0) <= max(10.0 * eig_tol, abs(det - 1.0) + eig_tol)):
            return MonodromyClassification("P", tr, det, disc, eig, True, "parabolic_or_near_identity")
        return MonodromyClassification("U", tr, det, disc, eig, True, "real_pair_not_reciprocal")

    return MonodromyClassification("O", tr, det, disc, eig, True, "complex_elliptic_pair")


def monodromy_kind(DPm, *, det_tol: float = 5.0e-2, eig_tol: float = 1.0e-8) -> str:
    """Return only the compact kind label from :func:`classify_monodromy_2x2`."""

    return classify_monodromy_2x2(DPm, det_tol=det_tol, eig_tol=eig_tol).kind
