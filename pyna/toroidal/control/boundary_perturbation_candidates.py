"""Spectrum screening helpers for boundary perturbation candidates."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from pyna.toroidal.coils.boundary_local import BoundaryLoopCoilSpec, boundary_loop_coil_superposition
from pyna.toroidal.perturbation_spectrum import (
    RadialPerturbationFourierSpectrum,
    nardon_radial_perturbation,
    radial_perturbation_Fourier_spectrum,
)


@dataclass(frozen=True)
class PerturbationCandidateSpectrumResponse:
    """Nardon projection and Fourier spectrum for a perturbation candidate."""

    delta_BR: np.ndarray
    delta_BZ: np.ndarray
    delta_BPhi: np.ndarray
    tilde_b1: np.ndarray
    spectrum: RadialPerturbationFourierSpectrum
    metadata: dict = field(default_factory=dict)

    def mode_coefficient(self, m: int, n: int, radial_index: int | None = None) -> complex:
        """Return one ``tilde_b`` Fourier coefficient."""

        return self.spectrum.mode_coefficient(m, n, radial_index=radial_index)


def _candidate_to_field(candidate):
    if hasattr(candidate, "B_at"):
        return candidate, {"candidate_kind": type(candidate).__name__}
    specs = tuple(candidate)
    if not all(isinstance(spec, BoundaryLoopCoilSpec) for spec in specs):
        raise TypeError("candidate must provide B_at or be a sequence of BoundaryLoopCoilSpec")
    return boundary_loop_coil_superposition(specs), {
        "candidate_kind": "BoundaryLoopCoilSpec",
        "n_loop_specs": len(specs),
    }


def _phi_grid_for_surface(R_surf: np.ndarray, phi_vals: Sequence[float]) -> np.ndarray:
    R = np.asarray(R_surf, dtype=float)
    phi = np.asarray(phi_vals, dtype=float).ravel()
    if R.ndim == 2:
        if R.shape[0] != phi.size:
            raise ValueError("2-D surfaces must have shape (n_phi, n_theta)")
        return np.broadcast_to(phi[:, None], R.shape)
    if R.ndim == 3:
        if R.shape[0] != phi.size:
            raise ValueError("3-D surfaces must have shape (n_phi, n_r, n_theta)")
        return np.broadcast_to(phi[:, None, None], R.shape)
    raise ValueError("surface arrays must have shape (n_phi, n_theta) or (n_phi, n_r, n_theta)")


def sample_perturbation_candidate_on_surfaces(
    candidate,
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    phi_vals: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a perturbation candidate on PEST-like surface coordinates."""

    R = np.asarray(R_surf, dtype=float)
    Z = np.asarray(Z_surf, dtype=float)
    if R.shape != Z.shape:
        raise ValueError("R_surf and Z_surf must have matching shapes")
    phi_grid = _phi_grid_for_surface(R, phi_vals)
    field, _metadata = _candidate_to_field(candidate)
    BR, BZ, BPhi = field.B_at(R, Z, phi_grid)
    return (
        np.asarray(BR, dtype=float).reshape(R.shape),
        np.asarray(BZ, dtype=float).reshape(R.shape),
        np.asarray(BPhi, dtype=float).reshape(R.shape),
    )


def perturbation_candidate_nardon_response(
    candidate,
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    phi_vals: Sequence[float],
    theta_vals: Sequence[float],
    radial_labels: Sequence[float],
    *,
    denominator_B_phi: np.ndarray | None = None,
    denominator_B3: np.ndarray | None = None,
    m_max: int | None = None,
    n_max: int | None = None,
    min_amplitude: float = 0.0,
    metadata: Mapping[str, object] | None = None,
) -> PerturbationCandidateSpectrumResponse:
    """Project a candidate perturbation to ``tilde_b^1`` and Fourier modes.

    This is the screening step for boundary-topology design: candidate fields are
    sampled on the healed/integrable surfaces, normalized by the background
    contravariant toroidal field, and converted to the same ``(m,n)`` spectrum
    used by island-width and phase predictors.
    """

    R = np.asarray(R_surf, dtype=float)
    if R.ndim != 3:
        raise ValueError("Nardon spectrum response requires a radial surface stack")
    field, candidate_metadata = _candidate_to_field(candidate)
    phi_grid = _phi_grid_for_surface(R, phi_vals)
    Z = np.asarray(Z_surf, dtype=float)
    if Z.shape != R.shape:
        raise ValueError("R_surf and Z_surf must have matching shapes")
    delta_BR, delta_BZ, delta_BPhi = field.B_at(R, Z, phi_grid)
    delta_BR = np.asarray(delta_BR, dtype=float).reshape(R.shape)
    delta_BZ = np.asarray(delta_BZ, dtype=float).reshape(R.shape)
    delta_BPhi = np.asarray(delta_BPhi, dtype=float).reshape(R.shape)
    labels = np.asarray(radial_labels, dtype=float)
    tilde = nardon_radial_perturbation(
        R,
        Z,
        np.asarray(phi_vals, dtype=float),
        np.asarray(theta_vals, dtype=float),
        delta_BR,
        delta_BZ,
        delta_BPhi,
        labels,
        denominator_B_phi=denominator_B_phi,
        denominator_B3=denominator_B3,
    )
    response_metadata = dict(candidate_metadata)
    response_metadata.update({} if metadata is None else dict(metadata))
    spectrum = radial_perturbation_Fourier_spectrum(
        tilde,
        np.asarray(theta_vals, dtype=float),
        np.asarray(phi_vals, dtype=float),
        radial_labels=labels,
        m_max=m_max,
        n_max=n_max,
        min_amplitude=min_amplitude,
        metadata=response_metadata,
    )
    return PerturbationCandidateSpectrumResponse(
        delta_BR=delta_BR,
        delta_BZ=delta_BZ,
        delta_BPhi=delta_BPhi,
        tilde_b1=tilde,
        spectrum=spectrum,
        metadata=response_metadata,
    )


__all__ = [
    "PerturbationCandidateSpectrumResponse",
    "perturbation_candidate_nardon_response",
    "sample_perturbation_candidate_on_surfaces",
]
