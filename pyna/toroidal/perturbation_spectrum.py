"""Radial magnetic-perturbation projection and Fourier spectra on flux surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pyna.toroidal._periodic_grid import (
    TWOPI,
    drop_endpoint,
    prepare_surface_arrays,
    periodic_derivative,
    strip_field_grid,
    strip_periodic_endpoint,
)


_SIGNATURE_SCHEMA_NAME = "pyna.toroidal.perturbation_spectrum.signature"
_SIGNATURE_SCHEMA_VERSION = 1


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def _array_summary(value: Any) -> dict[str, Any]:
    arr = np.asarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    if arr.dtype == object:
        h.update(repr(arr.tolist()).encode("utf-8"))
    else:
        h.update(np.ascontiguousarray(arr).tobytes())
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": h.hexdigest(),
    }


def signature_digest(signature: Any) -> str:
    """Return the stable SHA-256 digest for a provenance signature."""

    return hashlib.sha256(_json_dumps(signature).encode("utf-8")).hexdigest()


def _short_digest(signature: Any) -> str:
    return "none" if signature is None else signature_digest(signature)[:16]


def require_matching_signatures(
    expected: Any,
    actual: Any,
    *,
    context: str = "provenance",
    allow_missing: bool = False,
) -> None:
    """Require two provenance signatures to match without exposing their content."""

    if expected is None or actual is None:
        if allow_missing:
            return
        raise ValueError(f"{context} signature is required")
    if expected != actual:
        raise ValueError(
            f"{context} signature mismatch; "
            f"expected sha256={_short_digest(expected)}, actual sha256={_short_digest(actual)}"
        )


def require_matching_field_signature(
    expected: Any,
    actual: Any,
    *,
    context: str = "field",
    allow_missing: bool = False,
) -> None:
    """Require two field signatures to match without printing private metadata."""

    require_matching_signatures(expected, actual, context=context, allow_missing=allow_missing)


def cylindrical_field_grid_signature(
    grid_R: np.ndarray,
    grid_Z: np.ndarray,
    grid_phi: np.ndarray,
    field_R: np.ndarray,
    field_phi: np.ndarray,
    field_Z: np.ndarray,
    *,
    field_periods: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a path-free signature for a cylindrical vector-field grid.

    The signature intentionally contains array shapes, dtypes, hashes, and the
    field-period count, but no source paths or raw filenames.
    """

    axis_R = np.asarray(grid_R, dtype=np.float64)
    axis_Z = np.asarray(grid_Z, dtype=np.float64)
    axis_phi = np.asarray(grid_phi, dtype=np.float64)
    if axis_R.ndim != 1 or axis_Z.ndim != 1 or axis_phi.ndim != 1:
        raise ValueError("grid_R, grid_Z, and grid_phi must be one-dimensional")
    field_periods_i = int(field_periods)
    if field_periods_i < 1:
        raise ValueError("field_periods must be positive")
    shape = (axis_R.size, axis_Z.size, axis_phi.size)
    components = {
        "R": np.asarray(field_R),
        "phi": np.asarray(field_phi),
        "Z": np.asarray(field_Z),
    }
    for name, values in components.items():
        if values.shape != shape:
            raise ValueError(f"field_{name} shape {values.shape} does not match grid {shape}")
    signature: dict[str, Any] = {
        "schema_name": _SIGNATURE_SCHEMA_NAME,
        "schema_version": _SIGNATURE_SCHEMA_VERSION,
        "kind": "cylindrical_field_grid",
        "field_periods": field_periods_i,
        "grid": {
            "R": _array_summary(axis_R),
            "Z": _array_summary(axis_Z),
            "phi": _array_summary(axis_phi),
        },
        "components": {
            "R": _array_summary(components["R"]),
            "phi": _array_summary(components["phi"]),
            "Z": _array_summary(components["Z"]),
        },
    }
    if metadata:
        signature["metadata"] = dict(metadata)
    return signature


def _surface_background_field_signature(surface_signature: Mapping[str, Any] | None) -> Any:
    if not isinstance(surface_signature, Mapping):
        return None
    return surface_signature.get("background_field_signature", surface_signature.get("field_signature"))


def _cylindrical_field_grid_identity_signature(field_signature: Any) -> dict[str, Any] | None:
    if not isinstance(field_signature, Mapping):
        return None
    if field_signature.get("kind") != "cylindrical_field_grid":
        return None
    return {
        "schema_name": field_signature.get("schema_name"),
        "schema_version": field_signature.get("schema_version"),
        "kind": field_signature.get("kind"),
        "field_periods": field_signature.get("field_periods"),
        "grid": field_signature.get("grid"),
    }


def _require_cylindrical_field_grid_identity(field_signature: Any, *, context: str) -> dict[str, Any]:
    identity = _cylindrical_field_grid_identity_signature(field_signature)
    if identity is None:
        raise ValueError(f"{context} must carry a cylindrical field grid signature")
    return identity


def _require_same_cylindrical_field_grid(expected: Any, actual: Any, *, context: str) -> None:
    require_matching_signatures(
        _require_cylindrical_field_grid_identity(expected, context=context),
        _require_cylindrical_field_grid_identity(actual, context=context),
        context=f"{context} grid",
    )


def surface_coordinate_signature(
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    radial_labels: np.ndarray,
    *,
    background_field_signature: Any = None,
    coordinate_system: str = "magnetic",
    radial_coordinate: str = "s",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a provenance signature for a radial stack of magnetic surfaces."""

    R, Z, phi, theta = prepare_surface_arrays(R_surf, Z_surf, phi_vals, theta_vals)
    labels = _validate_radial_labels(radial_labels, R.shape[1])
    signature: dict[str, Any] = {
        "schema_name": _SIGNATURE_SCHEMA_NAME,
        "schema_version": _SIGNATURE_SCHEMA_VERSION,
        "kind": "surface_coordinates",
        "coordinate_system": str(coordinate_system),
        "radial_coordinate": str(radial_coordinate),
        "R_surf": _array_summary(R),
        "Z_surf": _array_summary(Z),
        "phi": _array_summary(phi),
        "theta": _array_summary(theta),
        "radial_labels": _array_summary(labels),
    }
    if background_field_signature is not None:
        signature["background_field_signature"] = background_field_signature
    if metadata:
        signature["metadata"] = dict(metadata)
    return signature


def radial_profile_signature(
    radial_labels: np.ndarray,
    values: np.ndarray,
    *,
    quantity: str,
    surface_signature: Any = None,
    background_field_signature: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a provenance signature for a q/iota-like radial profile."""

    labels = _validate_radial_labels(radial_labels, np.asarray(radial_labels).size)
    vals = np.asarray(values, dtype=np.float64)
    if vals.shape != labels.shape:
        raise ValueError("profile values must match radial_labels")
    surface_background = _surface_background_field_signature(surface_signature)
    if surface_background is not None and background_field_signature is not None:
        require_matching_field_signature(
            surface_background,
            background_field_signature,
            context=f"{quantity} profile background field",
        )
    signature: dict[str, Any] = {
        "schema_name": _SIGNATURE_SCHEMA_NAME,
        "schema_version": _SIGNATURE_SCHEMA_VERSION,
        "kind": "radial_profile",
        "quantity": str(quantity),
        "radial_labels": _array_summary(labels),
        "values": _array_summary(vals),
    }
    if surface_signature is not None:
        signature["surface_signature"] = surface_signature
    if background_field_signature is not None:
        signature["background_field_signature"] = background_field_signature
    if metadata:
        signature["metadata"] = dict(metadata)
    return signature


@dataclass(frozen=True)
class MagneticCoordinateProfile:
    """Radial q/iota profile bound to the surface/background-field identity."""

    quantity: str
    radial_labels: np.ndarray
    values: np.ndarray
    surface_signature: Mapping[str, Any] | None = None
    background_field_signature: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        labels = _validate_radial_labels(self.radial_labels, np.asarray(self.radial_labels).size)
        values = np.asarray(self.values, dtype=np.float64)
        if values.shape != labels.shape:
            raise ValueError("values must match radial_labels")
        object.__setattr__(self, "radial_labels", labels)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        surface_background = _surface_background_field_signature(self.surface_signature)
        if surface_background is not None and self.background_field_signature is not None:
            require_matching_field_signature(
                surface_background,
                self.background_field_signature,
                context=f"{self.quantity} profile background field",
            )

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.values, dtype=dtype)

    @property
    def signature(self) -> dict[str, Any]:
        """Return the profile provenance signature."""

        return radial_profile_signature(
            self.radial_labels,
            self.values,
            quantity=self.quantity,
            surface_signature=self.surface_signature,
            background_field_signature=self.background_field_signature,
            metadata=self.metadata,
        )

    def require_compatible_with(
        self,
        *,
        surface_signature: Any = None,
        background_field_signature: Any = None,
        context: str | None = None,
    ) -> None:
        """Validate this profile against another surface or background field."""

        label = self.quantity if context is None else str(context)
        if surface_signature is not None and self.surface_signature is not None:
            require_matching_signatures(
                self.surface_signature,
                surface_signature,
                context=f"{label} surface",
            )
        if background_field_signature is not None and self.background_field_signature is not None:
            require_matching_field_signature(
                self.background_field_signature,
                background_field_signature,
                context=f"{label} background field",
            )


_DELTA_DEFINITION_VALUES = {
    "explicit_delta",
    "external_delta",
    "healed_total_radial_flux",
}


def _field_decomposition_residual_summary(
    total_B_R: np.ndarray,
    total_B_phi: np.ndarray,
    total_B_Z: np.ndarray,
    background_B_R: np.ndarray,
    background_B_phi: np.ndarray,
    background_B_Z: np.ndarray,
    delta_B_R: np.ndarray,
    delta_B_phi: np.ndarray,
    delta_B_Z: np.ndarray,
) -> dict[str, Any]:
    components = {
        "R": (
            np.asarray(total_B_R),
            np.asarray(background_B_R),
            np.asarray(delta_B_R),
        ),
        "phi": (
            np.asarray(total_B_phi),
            np.asarray(background_B_phi),
            np.asarray(delta_B_phi),
        ),
        "Z": (
            np.asarray(total_B_Z),
            np.asarray(background_B_Z),
            np.asarray(delta_B_Z),
        ),
    }
    component_summary: dict[str, Any] = {}
    max_abs = 0.0
    rms_terms: list[float] = []
    scale = 0.0
    for name, (total, background, delta) in components.items():
        residual = total - background - delta
        residual_abs = np.abs(residual)
        component_max = float(np.nanmax(residual_abs)) if residual_abs.size else 0.0
        component_rms = float(np.sqrt(np.nanmean(residual_abs**2))) if residual_abs.size else 0.0
        component_scale = float(
            max(
                np.nanmax(np.abs(total)) if total.size else 0.0,
                np.nanmax(np.abs(background)) if background.size else 0.0,
                np.nanmax(np.abs(delta)) if delta.size else 0.0,
            )
        )
        component_summary[name] = {
            "max_abs": component_max,
            "rms": component_rms,
            "scale": component_scale,
        }
        max_abs = max(max_abs, component_max)
        scale = max(scale, component_scale)
        rms_terms.append(component_rms**2)
    return {
        "max_abs": max_abs,
        "rms": float(np.sqrt(np.mean(rms_terms))) if rms_terms else 0.0,
        "scale": scale,
        "components": component_summary,
    }


@dataclass(frozen=True)
class IntegrableFieldDecomposition:
    """Path-free provenance contract for ``B = B0 + delta_B`` analysis products."""

    total_field_signature: Mapping[str, Any] | None
    background_field_signature: Mapping[str, Any]
    delta_field_signature: Mapping[str, Any]
    surface_signature: Mapping[str, Any] | None = None
    q_profile: MagneticCoordinateProfile | None = None
    iota_profile: MagneticCoordinateProfile | None = None
    delta_definition: str = "explicit_delta"
    residual_summary: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.background_field_signature is None:
            raise ValueError("background_field_signature is required")
        if self.delta_field_signature is None:
            raise ValueError("delta_field_signature is required")
        delta_definition = str(self.delta_definition)
        if delta_definition not in _DELTA_DEFINITION_VALUES:
            allowed = ", ".join(sorted(_DELTA_DEFINITION_VALUES))
            raise ValueError(f"delta_definition must be one of: {allowed}")
        object.__setattr__(self, "delta_definition", delta_definition)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        object.__setattr__(
            self,
            "residual_summary",
            None if self.residual_summary is None else dict(self.residual_summary),
        )

        _require_cylindrical_field_grid_identity(
            self.background_field_signature,
            context="background field",
        )
        _require_cylindrical_field_grid_identity(
            self.delta_field_signature,
            context="delta field",
        )
        _require_same_cylindrical_field_grid(
            self.background_field_signature,
            self.delta_field_signature,
            context="delta field",
        )
        if self.total_field_signature is not None:
            _require_same_cylindrical_field_grid(
                self.total_field_signature,
                self.background_field_signature,
                context="background field",
            )
            _require_same_cylindrical_field_grid(
                self.total_field_signature,
                self.delta_field_signature,
                context="delta field",
            )
        if self.surface_signature is not None:
            self.require_surface(self.surface_signature)
        if self.q_profile is not None:
            self.require_profile(self.q_profile, context="q_profile")
        if self.iota_profile is not None:
            self.require_profile(self.iota_profile, context="iota_profile")

    @property
    def signature(self) -> dict[str, Any]:
        """Return a hashable decomposition signature without source paths."""

        signature: dict[str, Any] = {
            "schema_name": _SIGNATURE_SCHEMA_NAME,
            "schema_version": _SIGNATURE_SCHEMA_VERSION,
            "kind": "integrable_field_decomposition",
            "delta_definition": self.delta_definition,
            "total_field_signature": self.total_field_signature,
            "background_field_signature": self.background_field_signature,
            "delta_field_signature": self.delta_field_signature,
        }
        if self.surface_signature is not None:
            signature["surface_signature"] = self.surface_signature
        if self.q_profile is not None:
            signature["q_profile_signature"] = self.q_profile.signature
        if self.iota_profile is not None:
            signature["iota_profile_signature"] = self.iota_profile.signature
        if self.residual_summary is not None:
            signature["residual_summary"] = dict(self.residual_summary)
        if self.metadata:
            signature["metadata"] = dict(self.metadata)
        return signature

    @property
    def digest(self) -> str:
        """Stable SHA-256 digest for this decomposition contract."""

        return signature_digest(self.signature)

    def require_surface(self, surface_signature: Any, *, context: str = "surface") -> None:
        """Require a surface signature to be bound to this decomposition's ``B0``."""

        if self.surface_signature is not None:
            require_matching_signatures(
                self.surface_signature,
                surface_signature,
                context=f"{context} signature",
            )
        bound_background = _surface_background_field_signature(surface_signature)
        if bound_background is not None:
            require_matching_field_signature(
                self.background_field_signature,
                bound_background,
                context=f"{context} background field",
            )

    def require_profile(
        self,
        profile: MagneticCoordinateProfile,
        *,
        context: str | None = None,
    ) -> None:
        """Require a q/iota profile to use this decomposition's surface and ``B0``."""

        if not isinstance(profile, MagneticCoordinateProfile):
            raise TypeError("profile must be a MagneticCoordinateProfile")
        profile.require_compatible_with(
            surface_signature=self.surface_signature,
            background_field_signature=self.background_field_signature,
            context=context or profile.quantity,
        )

    def require_projection(
        self,
        projection: "NardonRadialPerturbationProjection",
        *,
        context: str = "projection",
    ) -> None:
        """Require a Nardon projection to use this decomposition's ``B0`` and ``delta_B``."""

        require_matching_field_signature(
            self.background_field_signature,
            projection.background_field_signature,
            context=f"{context} background field",
        )
        require_matching_field_signature(
            self.delta_field_signature,
            projection.delta_field_signature,
            context=f"{context} delta field",
        )
        if projection.surface_signature is not None:
            self.require_surface(projection.surface_signature, context=context)

    def require_spectrum(
        self,
        spectrum: "RadialPerturbationFourierSpectrum",
        *,
        context: str = "spectrum",
    ) -> None:
        """Require a Fourier spectrum to use this decomposition's surface and fields."""

        require_matching_field_signature(
            self.background_field_signature,
            spectrum.background_field_signature,
            context=f"{context} background field",
        )
        require_matching_field_signature(
            self.delta_field_signature,
            spectrum.delta_field_signature,
            context=f"{context} delta field",
        )
        if spectrum.surface_signature is not None:
            self.require_surface(spectrum.surface_signature, context=context)


def integrable_field_decomposition_from_grids(
    grid_R: np.ndarray,
    grid_Z: np.ndarray,
    grid_phi: np.ndarray,
    total_B_R: np.ndarray,
    total_B_phi: np.ndarray,
    total_B_Z: np.ndarray,
    background_B_R: np.ndarray,
    background_B_phi: np.ndarray,
    background_B_Z: np.ndarray,
    delta_B_R: np.ndarray,
    delta_B_phi: np.ndarray,
    delta_B_Z: np.ndarray,
    *,
    field_periods: int = 1,
    surface_signature: Mapping[str, Any] | None = None,
    q_profile: MagneticCoordinateProfile | None = None,
    iota_profile: MagneticCoordinateProfile | None = None,
    delta_definition: str = "explicit_delta",
    validate_explicit_delta: bool = True,
    residual_atol: float = 1.0e-12,
    residual_rtol: float = 1.0e-10,
    total_metadata: Mapping[str, Any] | None = None,
    background_metadata: Mapping[str, Any] | None = None,
    delta_metadata: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> IntegrableFieldDecomposition:
    """Build a decomposition contract from matching cylindrical field grids.

    The returned object records only signatures and numerical residual
    summaries.  Signature mismatch errors report digests rather than source
    metadata, so private paths or filenames are not exposed by validation.
    """

    total_sig = cylindrical_field_grid_signature(
        grid_R,
        grid_Z,
        grid_phi,
        total_B_R,
        total_B_phi,
        total_B_Z,
        field_periods=field_periods,
        metadata=total_metadata,
    )
    background_sig = cylindrical_field_grid_signature(
        grid_R,
        grid_Z,
        grid_phi,
        background_B_R,
        background_B_phi,
        background_B_Z,
        field_periods=field_periods,
        metadata=background_metadata,
    )
    delta_sig = cylindrical_field_grid_signature(
        grid_R,
        grid_Z,
        grid_phi,
        delta_B_R,
        delta_B_phi,
        delta_B_Z,
        field_periods=field_periods,
        metadata=delta_metadata,
    )
    residual_summary = _field_decomposition_residual_summary(
        total_B_R,
        total_B_phi,
        total_B_Z,
        background_B_R,
        background_B_phi,
        background_B_Z,
        delta_B_R,
        delta_B_phi,
        delta_B_Z,
    )
    if delta_definition == "explicit_delta" and validate_explicit_delta:
        allowed = float(residual_atol) + float(residual_rtol) * max(1.0, float(residual_summary["scale"]))
        if float(residual_summary["max_abs"]) > allowed:
            raise ValueError(
                "explicit delta field is not consistent with total-background; "
                f"max_abs={float(residual_summary['max_abs']):.6e}, allowed={allowed:.6e}"
            )
    return IntegrableFieldDecomposition(
        total_field_signature=total_sig,
        background_field_signature=background_sig,
        delta_field_signature=delta_sig,
        surface_signature=surface_signature,
        q_profile=q_profile,
        iota_profile=iota_profile,
        delta_definition=delta_definition,
        residual_summary=residual_summary,
        metadata={} if metadata is None else metadata,
    )


@dataclass(frozen=True)
class RadialPerturbationFourierSpectrum:
    """Fourier spectrum of the magnetic perturbation normal to a flux surface.

    Coefficients follow the Nardon convention
    ``tilde_b^1_mn = integral tilde_b^1 exp(-i * (m * theta + n * phi))``
    with inverse basis ``exp(i * (m * theta + n * phi))``.  For real fields,
    ``tilde_b^1_-m,-n = conjugate(tilde_b^1_mn)``.

    :attr:`nardon_n` is the signed full-torus Nardon index.  The stored ``n``
    array is retained only as the legacy local FFT harmonic over one field
    period.  :attr:`physical_n` and ``physical_mode_*`` preserve their prior
    sign-reversed compatibility behavior and must not be used for Nardon-mode
    selection.
    """

    m: np.ndarray
    n: np.ndarray
    dBr: np.ndarray
    dBr_grid: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    radial_labels: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    field_periods: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        if isinstance(self.field_periods, (bool, np.bool_)):
            raise ValueError("field_periods must be a positive integer")
        try:
            field_periods = int(self.field_periods)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("field_periods must be a positive integer") from exc
        if field_periods < 1 or float(self.field_periods) != float(field_periods):
            raise ValueError("field_periods must be a positive integer")
        object.__setattr__(self, "field_periods", field_periods)

    @property
    def field_period_harmonic(self) -> np.ndarray:
        """Signed FFT harmonic over the stored toroidal field period.

        This is the compatibility name for :attr:`n`.  It multiplies the local
        field-period angle with a plus sign.
        """

        return self.n

    @property
    def nardon_n(self) -> np.ndarray:
        """Signed Nardon toroidal index in ``exp(i * (m*theta + n*phi))``.

        A one-field-period FFT has local harmonic ``k``; its full-torus
        Nardon index is ``n = field_periods * k``.
        """

        return self.field_periods * np.asarray(self.n, dtype=int)

    @property
    def physical_n(self) -> np.ndarray:
        """Compatibility index equal to ``-nardon_n``.

        This preserves the historical ``exp(i*(m*theta - physical_n*phi))``
        API.  New Nardon-convention callers must use :attr:`nardon_n`.
        """

        return -self.nardon_n

    @property
    def resonance_family_n0(self) -> np.ndarray:
        """Positive family labels ``n0 = abs(nardon_n)``, not mode indices."""

        return np.abs(self.nardon_n)

    @property
    def amplitude(self) -> np.ndarray:
        """Complex-mode amplitudes ``abs(dBr_mn)``."""

        return np.abs(self.dBr)

    @property
    def phase(self) -> np.ndarray:
        """Complex-mode phases ``arg(dBr_mn)`` in radians."""

        return np.angle(self.dBr)

    def split(self, iota: float, resonance_tol: float = 1.0e-9, radial_index: int | None = None):
        """Split this radial spectrum into resonant and non-resonant modes."""

        from pyna.toroidal.torus_deformation import split_radial_perturbation_spectrum

        dBr = self.dBr
        if dBr.ndim != 1:
            if radial_index is None:
                raise ValueError("radial_index is required when splitting a radial stack spectrum")
            dBr = dBr[int(radial_index)]
        return split_radial_perturbation_spectrum(
            self.m,
            self.nardon_n,
            dBr,
            iota=iota,
            resonance_tol=resonance_tol,
        )

    def mode_index(self, m: int, n: int) -> int | None:
        """Return the index for poloidal ``m`` and legacy field-period harmonic ``n``."""

        idx = np.where((self.m == int(m)) & (self.n == int(n)))[0]
        return None if idx.size == 0 else int(idx[0])

    def physical_mode_index(self, m: int, n: int) -> int | None:
        """Compatibility lookup for historical sign-reversed ``physical_n``."""

        idx = np.where((self.m == int(m)) & (self.physical_n == int(n)))[0]
        return None if idx.size == 0 else int(idx[0])

    def nardon_mode_index(self, m: int, n: int) -> int | None:
        """Return the packed-mode index for Nardon ``(m, nardon_n)``."""

        idx = np.where((self.m == int(m)) & (self.nardon_n == int(n)))[0]
        return None if idx.size == 0 else int(idx[0])

    def mode_coefficient(self, m: int, n: int, radial_index: int | None = None) -> complex:
        """Return a coefficient addressed by the legacy field-period harmonic."""

        idx = self.mode_index(m, n)
        if idx is None:
            return 0.0 + 0.0j
        if self.dBr.ndim == 1:
            return complex(self.dBr[idx])
        if radial_index is None:
            raise ValueError("radial_index is required for a radial stack spectrum")
        return complex(self.dBr[int(radial_index), idx])

    def physical_mode_coefficient(
        self,
        m: int,
        n: int,
        radial_index: int | None = None,
    ) -> complex:
        """Return a coefficient through the historical ``physical_n`` API."""

        idx = self.physical_mode_index(m, n)
        if idx is None:
            return 0.0 + 0.0j
        if self.dBr.ndim == 1:
            return complex(self.dBr[idx])
        if radial_index is None:
            raise ValueError("radial_index is required for a radial stack spectrum")
        return complex(self.dBr[int(radial_index), idx])

    def nardon_mode_coefficient(
        self,
        m: int,
        n: int,
        radial_index: int | None = None,
    ) -> complex:
        """Return a coefficient addressed by Nardon ``(m, nardon_n)``."""

        idx = self.nardon_mode_index(m, n)
        if idx is None:
            return 0.0 + 0.0j
        if self.dBr.ndim == 1:
            return complex(self.dBr[idx])
        if radial_index is None:
            raise ValueError("radial_index is required for a radial stack spectrum")
        return complex(self.dBr[int(radial_index), idx])

    @property
    def surface_signature(self) -> Any:
        """Surface-coordinate signature carried by the spectrum, if present."""

        return dict(self.metadata).get("surface_signature")

    @property
    def background_field_signature(self) -> Any:
        """Background-field signature carried by the spectrum, if present."""

        return dict(self.metadata).get("background_field_signature")

    @property
    def delta_field_signature(self) -> Any:
        """Perturbation-field signature carried by the spectrum, if present."""

        return dict(self.metadata).get("delta_field_signature")


@dataclass(frozen=True)
class NardonRadialPerturbationProjection:
    """Nardon ``tilde_b^1`` projection sampled from explicit ``B0`` and ``delta_B``."""

    tilde_b1: np.ndarray
    delta_B1: np.ndarray
    background_B3: np.ndarray
    delta_BR: np.ndarray
    delta_BZ: np.ndarray
    delta_BPhi: np.ndarray
    background_BR: np.ndarray
    background_BZ: np.ndarray
    background_BPhi: np.ndarray
    R_surf: np.ndarray
    Z_surf: np.ndarray
    phi_vals: np.ndarray
    theta_vals: np.ndarray
    radial_labels: np.ndarray
    surface_signature: Mapping[str, Any] | None = None
    background_field_signature: Mapping[str, Any] | None = None
    delta_field_signature: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def fourier_spectrum(
        self,
        *,
        field_periods: int | None = None,
        m_max: int | None = None,
        n_max: int | None = None,
        min_amplitude: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> RadialPerturbationFourierSpectrum:
        """Compute a Fourier spectrum while preserving projection provenance."""

        md = {
            "surface_signature": self.surface_signature,
            "background_field_signature": self.background_field_signature,
            "delta_field_signature": self.delta_field_signature,
        }
        md.update(dict(self.metadata or {}))
        if metadata:
            md.update(dict(metadata))
        return radial_perturbation_Fourier_spectrum(
            self.tilde_b1,
            self.theta_vals,
            self.phi_vals,
            radial_labels=self.radial_labels,
            layout="phi-radial-theta",
            field_periods=field_periods,
            m_max=m_max,
            n_max=n_max,
            min_amplitude=min_amplitude,
            metadata=md,
        )


@dataclass(frozen=True)
class ResonantIslandChain:
    """Nardon-style resonant island-chain estimate from a helical Fourier branch.

    ``n`` is retained as the historical positive resonance-family label
    ``n0``.  The signed Fourier mode is :attr:`nardon_n` (``coefficient_n``).
    """

    m: int
    n: int
    radial_label: float
    q: float
    q_prime: float
    coefficient: complex
    b_res: float
    half_width: float
    coefficient_n: int | None = None

    @property
    def nardon_n(self) -> int:
        """Signed Nardon index of the selected Fourier coefficient."""

        return -int(self.n) if self.coefficient_n is None else int(self.coefficient_n)

    @property
    def resonance_family_n0(self) -> int:
        """Derived positive resonance-family label ``abs(nardon_n)``."""

        return abs(self.nardon_n)

    @property
    def phase(self) -> float:
        """Phase of the selected resonant Fourier coefficient in radians."""

        return float(np.angle(self.coefficient))

    def fixed_points(self, phi: float | np.ndarray, *, q_prime_sign: int | None = None) -> dict[str, np.ndarray]:
        """Return O/X poloidal angles for one or more toroidal sections.

        The convention is the Nardon expansion
        ``tilde_b^1 = sum b_mn exp(i(m theta* + n phi))``.  The selected
        resonant branch satisfies
        ``m theta* + coefficient_n phi + arg(b) = +/- pi/2``.
        """

        sign = int(np.sign(self.q_prime)) if q_prime_sign is None else int(np.sign(q_prime_sign))
        if sign == 0:
            sign = 1
        return island_chain_fixed_points(
            self.m,
            self.n,
            self.coefficient,
            phi,
            q_prime_sign=sign,
            coefficient_n=self.coefficient_n,
        )

    def with_phase_shift(self, phase_shift: float) -> "ResonantIslandChain":
        """Return a copy with ``arg(coefficient)`` advanced by ``phase_shift``."""

        return ResonantIslandChain(
            m=self.m,
            n=self.n,
            radial_label=self.radial_label,
            q=self.q,
            q_prime=self.q_prime,
            coefficient=self.coefficient * np.exp(1j * float(phase_shift)),
            b_res=self.b_res,
            half_width=self.half_width,
            coefficient_n=self.coefficient_n,
        )


@dataclass(frozen=True)
class ChirikovOverlap:
    """Chirikov overlap between two adjacent resonant island chains."""

    left: ResonantIslandChain
    right: ResonantIslandChain
    separation: float
    sigma: float

    @property
    def modes(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return ``((m_left, n_left), (m_right, n_right))``."""

        return (self.left.m, self.left.n), (self.right.m, self.right.n)


@dataclass(frozen=True)
class ResonantSurfaceGroup:
    """One radial resonant surface with possibly multiple Fourier contributors."""

    radial_label: float
    q: float
    q_prime: float
    half_width: float
    b_res: float
    chains: tuple[ResonantIslandChain, ...]

    @property
    def modes(self) -> tuple[tuple[int, int], ...]:
        """Return all physical ``(m, n)`` modes in this surface group."""

        return tuple((chain.m, chain.n) for chain in self.chains)

    @property
    def mode_count(self) -> int:
        """Return the number of resonant Fourier contributors."""

        return len(self.chains)

    @property
    def dominant_chain(self) -> ResonantIslandChain:
        """Return the contributor with the largest half-width estimate."""

        return max(self.chains, key=lambda chain: float(chain.half_width))


@dataclass(frozen=True)
class ChirikovOverlapBand:
    """Radial band covered by two adjacent island chains and their overlap."""

    left: ResonantIslandChain
    right: ResonantIslandChain
    separation: float
    sigma: float
    inner: float
    outer: float
    overlap_inner: float
    overlap_outer: float
    overlap_width: float

    @property
    def modes(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return ``((m_left, n_left), (m_right, n_right))``."""

        return (self.left.m, self.left.n), (self.right.m, self.right.n)

    @property
    def same_toroidal_family(self) -> bool:
        """Return whether both chains have the same physical toroidal mode."""

        return bool(self.left.n == self.right.n)

    @property
    def is_overlapping(self) -> bool:
        """Return whether the separatrix half-widths overlap."""

        return bool(self.sigma >= 1.0 or self.overlap_width > 0.0)


@dataclass(frozen=True)
class ChaoticLayerInterval:
    """Merged radial interval where adjacent island-chain bands overlap."""

    inner: float
    outer: float
    max_sigma: float
    bands: tuple[ChirikovOverlapBand, ...]

    @property
    def width(self) -> float:
        """Return interval width in the radial label coordinate."""

        return float(self.outer - self.inner)

    @property
    def modes(self) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
        """Return mode pairs contributing to this interval."""

        return tuple(band.modes for band in self.bands)


@dataclass(frozen=True)
class PeriodicOrbitGeometryDistance:
    """Cyclic geometric mismatch between two sampled periodic orbits."""

    n_points: int
    shift: int
    reversed: bool
    mean_distance: float
    rms_distance: float
    max_distance: float


@dataclass(frozen=True)
class PeriodicOrbitSurfaceAlignment:
    """Alignment of a periodic orbit with one healed magnetic surface/field line."""

    n_points: int
    target_radial_label: float
    radial_label_mean: float
    radial_label_rms_spread: float
    radial_error_rms: float
    radial_error_max: float
    surface_distance_rms: float
    surface_distance_max: float
    fieldline_phase_offset: float | None = None
    fieldline_phase_rms: float | None = None
    fieldline_phase_max: float | None = None


def periodic_orbit_geometry_distance(
    reference_R: np.ndarray,
    reference_Z: np.ndarray,
    candidate_R: np.ndarray,
    candidate_Z: np.ndarray,
    *,
    reference_phi: np.ndarray | None = None,
    candidate_phi: np.ndarray | None = None,
    allow_reverse: bool = True,
) -> PeriodicOrbitGeometryDistance:
    """Return the best cyclic distance between two X/O periodic-orbit rings.

    The samples must represent the same number of points on corresponding
    rings.  Cyclic shifts, and optionally reversed ordering, are tried so the
    result is independent of the chosen starting section.
    """

    ref_R = np.asarray(reference_R, dtype=float).reshape(-1)
    ref_Z = np.asarray(reference_Z, dtype=float).reshape(-1)
    cand_R = np.asarray(candidate_R, dtype=float).reshape(-1)
    cand_Z = np.asarray(candidate_Z, dtype=float).reshape(-1)
    if ref_R.shape != ref_Z.shape or cand_R.shape != cand_Z.shape or ref_R.shape != cand_R.shape:
        raise ValueError("reference and candidate orbit arrays must have the same one-dimensional length")
    n_points = int(ref_R.size)
    if n_points == 0:
        raise ValueError("periodic orbit arrays must not be empty")

    if (reference_phi is None) != (candidate_phi is None):
        raise ValueError("reference_phi and candidate_phi must be supplied together")
    if reference_phi is None:
        reference = np.column_stack([ref_R, ref_Z])
        candidate = np.column_stack([cand_R, cand_Z])
    else:
        ref_phi = np.asarray(reference_phi, dtype=float).reshape(-1)
        cand_phi = np.asarray(candidate_phi, dtype=float).reshape(-1)
        if ref_phi.shape != ref_R.shape or cand_phi.shape != cand_R.shape:
            raise ValueError("phi arrays must match the orbit array length")
        reference = np.column_stack([ref_R * np.cos(ref_phi), ref_R * np.sin(ref_phi), ref_Z])
        candidate = np.column_stack([cand_R * np.cos(cand_phi), cand_R * np.sin(cand_phi), cand_Z])

    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("periodic orbit arrays must be finite")

    best: PeriodicOrbitGeometryDistance | None = None
    orders = [(False, candidate)]
    if allow_reverse and n_points > 1:
        orders.append((True, candidate[::-1]))
    for reversed_order, ordered in orders:
        for shift in range(n_points):
            shifted = np.roll(ordered, -shift, axis=0)
            distances = np.linalg.norm(reference - shifted, axis=1)
            result = PeriodicOrbitGeometryDistance(
                n_points=n_points,
                shift=int(shift),
                reversed=bool(reversed_order),
                mean_distance=float(np.mean(distances)),
                rms_distance=float(np.sqrt(np.mean(distances * distances))),
                max_distance=float(np.max(distances)),
            )
            if best is None or (result.rms_distance, result.max_distance) < (best.rms_distance, best.max_distance):
                best = result
    assert best is not None
    return best


def _interp_surface_section_at_phi(values: np.ndarray, phi_vals: np.ndarray, phi: float) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    phi_axis = np.asarray(phi_vals, dtype=float).reshape(-1)
    if vals.shape[0] != phi_axis.size:
        raise ValueError("surface values leading axis must match phi_vals")
    src = np.mod(phi_axis, TWOPI)
    order = np.argsort(src)
    src = src[order]
    vals = vals[order]
    src_ext = np.concatenate([src[-1:] - TWOPI, src, src[:1] + TWOPI])
    vals_ext = np.concatenate([vals[-1:], vals, vals[:1]], axis=0)
    query = float(np.mod(float(phi), TWOPI))
    idx = int(np.searchsorted(src_ext, query, side="right") - 1)
    idx = max(0, min(idx, src_ext.size - 2))
    span = float(src_ext[idx + 1] - src_ext[idx])
    frac = 0.0 if span <= 0.0 else float((query - src_ext[idx]) / span)
    return (1.0 - frac) * vals_ext[idx] + frac * vals_ext[idx + 1]


def periodic_orbit_surface_alignment(
    orbit_R: np.ndarray,
    orbit_Z: np.ndarray,
    orbit_phi: np.ndarray,
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    radial_labels: np.ndarray,
    *,
    target_radial_label: float | None = None,
    iota: float | None = None,
    helicity_m: int | None = None,
    coefficient_n: int | None = None,
) -> PeriodicOrbitSurfaceAlignment:
    """Measure whether a fixed-point ring lies on a healed surface field line.

    This is the geometry constraint used when a non-integrable island chain has
    been healed into the integrable reference: the original X/O ring should
    project to one radial surface and one magnetic field-line phase, not to an
    X/O ring of the integrable field.
    """

    R_points = np.asarray(orbit_R, dtype=float).reshape(-1)
    Z_points = np.asarray(orbit_Z, dtype=float).reshape(-1)
    phi_points = np.asarray(orbit_phi, dtype=float).reshape(-1)
    if R_points.shape != Z_points.shape or R_points.shape != phi_points.shape:
        raise ValueError("orbit_R, orbit_Z, and orbit_phi must have the same one-dimensional length")
    if R_points.size == 0:
        raise ValueError("orbit arrays must not be empty")
    if not (np.all(np.isfinite(R_points)) and np.all(np.isfinite(Z_points)) and np.all(np.isfinite(phi_points))):
        raise ValueError("orbit arrays must be finite")
    R, Z, phi, theta = prepare_surface_arrays(R_surf, Z_surf, phi_vals, theta_vals)
    labels = _validate_radial_labels(radial_labels, R.shape[1])

    projected_s = np.empty(R_points.size, dtype=float)
    projected_theta = np.empty(R_points.size, dtype=float)
    distances = np.empty(R_points.size, dtype=float)
    for i, (rp, zp, pp) in enumerate(zip(R_points, Z_points, phi_points)):
        R_section = _interp_surface_section_at_phi(R, phi, float(pp))
        Z_section = _interp_surface_section_at_phi(Z, phi, float(pp))
        d2 = (R_section - float(rp)) ** 2 + (Z_section - float(zp)) ** 2
        idx = int(np.nanargmin(d2))
        ir, it = np.unravel_index(idx, R_section.shape)
        projected_s[i] = labels[int(ir)]
        projected_theta[i] = theta[int(it)]
        distances[i] = float(np.sqrt(d2[int(ir), int(it)]))

    if target_radial_label is None:
        target = float(np.mean(projected_s))
    else:
        target = float(target_radial_label)
    radial_error = projected_s - target
    radial_spread = projected_s - float(np.mean(projected_s))

    phase_offset = None
    phase_rms = None
    phase_max = None
    if helicity_m is not None:
        m_int = int(helicity_m)
        if m_int <= 0:
            raise ValueError("helicity_m must be positive")
        n_coeff = 0 if coefficient_n is None else int(coefficient_n)
        phase = m_int * projected_theta + n_coeff * phi_points
        divisor = float(m_int)
    elif iota is not None:
        phase = projected_theta - float(iota) * phi_points
        divisor = 1.0
    else:
        phase = None
        divisor = 1.0
    if phase is not None:
        z = np.exp(1j * phase)
        mean = np.mean(z)
        phase_offset = float(np.angle(mean))
        residual = np.angle(np.exp(1j * (phase - phase_offset))) / divisor
        phase_rms = float(np.sqrt(np.mean(residual * residual)))
        phase_max = float(np.max(np.abs(residual)))

    return PeriodicOrbitSurfaceAlignment(
        n_points=int(R_points.size),
        target_radial_label=target,
        radial_label_mean=float(np.mean(projected_s)),
        radial_label_rms_spread=float(np.sqrt(np.mean(radial_spread * radial_spread))),
        radial_error_rms=float(np.sqrt(np.mean(radial_error * radial_error))),
        radial_error_max=float(np.max(np.abs(radial_error))),
        surface_distance_rms=float(np.sqrt(np.mean(distances * distances))),
        surface_distance_max=float(np.max(distances)),
        fieldline_phase_offset=phase_offset,
        fieldline_phase_rms=phase_rms,
        fieldline_phase_max=phase_max,
    )


def surface_unit_normal_cylindrical(
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    *,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute outward surface normals in cylindrical components."""

    R_arr = np.asarray(R_surf, dtype=np.float64)
    Z_arr = np.asarray(Z_surf, dtype=np.float64)
    squeeze_r = False
    if R_arr.ndim == 2:
        R_arr = R_arr[:, np.newaxis, :]
        Z_arr = Z_arr[:, np.newaxis, :]
        squeeze_r = True
    R, Z, _, _ = prepare_surface_arrays(R_arr, Z_arr, phi_vals, theta_vals)
    dR_dtheta = periodic_derivative(R, TWOPI, axis=2)
    dZ_dtheta = periodic_derivative(Z, TWOPI, axis=2)
    dR_dphi = periodic_derivative(R, TWOPI, axis=0)
    dZ_dphi = periodic_derivative(Z, TWOPI, axis=0)

    n_R = R * dZ_dtheta
    n_phi = dZ_dphi * dR_dtheta - dR_dphi * dZ_dtheta
    n_Z = -R * dR_dtheta
    if normalize:
        norm = np.sqrt(n_R * n_R + n_Z * n_Z + n_phi * n_phi)
        norm = np.maximum(norm, 1.0e-300)
        n_R = n_R / norm
        n_Z = n_Z / norm
        n_phi = n_phi / norm
    if squeeze_r:
        return n_R[:, 0], n_Z[:, 0], n_phi[:, 0]
    return n_R, n_Z, n_phi


def radial_perturbation_component(
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    delta_B_R: np.ndarray,
    delta_B_Z: np.ndarray,
    delta_B_phi: np.ndarray | None = None,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Project an external magnetic perturbation onto the surface-normal direction."""

    n_R, n_Z, n_phi = surface_unit_normal_cylindrical(
        R_surf,
        Z_surf,
        phi_vals,
        theta_vals,
        normalize=normalize,
    )
    dBR = strip_field_grid(np.asarray(delta_B_R, dtype=complex), theta_vals, phi_vals)
    dBZ = strip_field_grid(np.asarray(delta_B_Z, dtype=complex), theta_vals, phi_vals)
    if delta_B_phi is None:
        dBphi = np.zeros_like(dBR, dtype=complex)
    else:
        dBphi = strip_field_grid(np.asarray(delta_B_phi, dtype=complex), theta_vals, phi_vals)
    if dBR.shape != n_R.shape or dBZ.shape != n_R.shape or dBphi.shape != n_R.shape:
        raise ValueError("delta_B arrays must match the surface shape after removing endpoints")
    return dBR * n_R + dBZ * n_Z + dBphi * n_phi


def _validate_radial_labels(radial_labels: np.ndarray, n_r: int) -> np.ndarray:
    labels = np.asarray(radial_labels, dtype=np.float64)
    if labels.ndim != 1 or labels.size != int(n_r):
        raise ValueError("radial_labels must be one-dimensional and match the radial surface count")
    if not np.all(np.isfinite(labels)) or np.any(np.diff(labels) <= 0.0):
        raise ValueError("radial_labels must be finite and strictly increasing")
    return labels


def contravariant_radial_component(
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    B_R: np.ndarray,
    B_Z: np.ndarray,
    B_phi: np.ndarray | None,
    radial_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``B^1 = B dot grad(s)`` and ``B^3 = B dot grad(phi)``.

    ``R_surf`` and ``Z_surf`` must be a radial stack with shape
    ``(n_phi, n_r, n_theta)``.  The returned arrays have the same stripped
    shape, after any duplicated periodic endpoints have been removed.
    Cylindrical field components use the physical orthonormal basis
    ``(e_R, e_phi, e_Z)``.
    """

    R, Z, _, _ = prepare_surface_arrays(R_surf, Z_surf, phi_vals, theta_vals)
    labels = _validate_radial_labels(radial_labels, R.shape[1])
    BR = strip_field_grid(np.asarray(B_R, dtype=complex), theta_vals, phi_vals)
    BZ = strip_field_grid(np.asarray(B_Z, dtype=complex), theta_vals, phi_vals)
    if B_phi is None:
        Bphi = np.zeros_like(BR, dtype=complex)
    else:
        Bphi = strip_field_grid(np.asarray(B_phi, dtype=complex), theta_vals, phi_vals)
    if BR.shape != R.shape or BZ.shape != R.shape or Bphi.shape != R.shape:
        raise ValueError("field arrays must match the surface shape after removing endpoints")

    edge_order = 2 if labels.size >= 3 else 1
    dR_ds = np.gradient(R, labels, axis=1, edge_order=edge_order)
    dZ_ds = np.gradient(Z, labels, axis=1, edge_order=edge_order)
    dR_dtheta = periodic_derivative(R, TWOPI, axis=2)
    dZ_dtheta = periodic_derivative(Z, TWOPI, axis=2)
    dR_dphi = periodic_derivative(R, TWOPI, axis=0)
    dZ_dphi = periodic_derivative(Z, TWOPI, axis=0)

    # Reciprocal basis: grad(s) = (e_theta x e_phi) / J.
    # Components are in the local right-handed cylindrical basis
    # (e_R, e_phi, e_Z).
    cross_R = -R * dZ_dtheta
    cross_phi = dZ_dtheta * dR_dphi - dR_dtheta * dZ_dphi
    cross_Z = R * dR_dtheta
    jac = dR_ds * cross_R + dZ_ds * cross_Z
    jac = np.where(np.abs(jac) < 1.0e-300, np.nan, jac)
    grad_s_R = cross_R / jac
    grad_s_phi = cross_phi / jac
    grad_s_Z = cross_Z / jac

    B1 = BR * grad_s_R + Bphi * grad_s_phi + BZ * grad_s_Z
    B3 = Bphi / np.maximum(R, 1.0e-300)
    return B1, B3


@dataclass(frozen=True)
class SurfaceFieldAlignmentDiagnostics:
    """Alignment diagnostics between a field and candidate flux coordinates."""

    radial_labels: np.ndarray
    radial_ratio_rms: np.ndarray
    radial_ratio_p95: np.ndarray
    radial_ratio_max: np.ndarray
    global_radial_ratio_rms: float
    edge_radial_ratio_rms: float
    iota_from_field: np.ndarray | None = None
    iota_profile_error_rms: float | None = None
    iota_profile_sign_flipped_error_rms: float | None = None

    @property
    def is_field_aligned(self) -> bool:
        """Return whether the field is approximately tangent to the surfaces."""

        return bool(np.isfinite(self.global_radial_ratio_rms) and self.global_radial_ratio_rms < 1.0e-3)


def _contravariant_theta_component(
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    B_R: np.ndarray,
    B_Z: np.ndarray,
    B_phi: np.ndarray,
    radial_labels: np.ndarray,
) -> np.ndarray:
    R, Z, _, _ = prepare_surface_arrays(R_surf, Z_surf, phi_vals, theta_vals)
    labels = _validate_radial_labels(radial_labels, R.shape[1])
    BR = strip_field_grid(np.asarray(B_R, dtype=complex), theta_vals, phi_vals)
    BZ = strip_field_grid(np.asarray(B_Z, dtype=complex), theta_vals, phi_vals)
    Bphi = strip_field_grid(np.asarray(B_phi, dtype=complex), theta_vals, phi_vals)
    if BR.shape != R.shape or BZ.shape != R.shape or Bphi.shape != R.shape:
        raise ValueError("field arrays must match the surface shape after removing endpoints")

    edge_order = 2 if labels.size >= 3 else 1
    dR_ds = np.gradient(R, labels, axis=1, edge_order=edge_order)
    dZ_ds = np.gradient(Z, labels, axis=1, edge_order=edge_order)
    dR_dtheta = periodic_derivative(R, TWOPI, axis=2)
    dZ_dtheta = periodic_derivative(Z, TWOPI, axis=2)
    dR_dphi = periodic_derivative(R, TWOPI, axis=0)
    dZ_dphi = periodic_derivative(Z, TWOPI, axis=0)

    cross_s_R = -R * dZ_dtheta
    cross_s_Z = R * dR_dtheta
    jac = dR_ds * cross_s_R + dZ_ds * cross_s_Z
    jac = np.where(np.abs(jac) < 1.0e-300, np.nan, jac)
    grad_theta_R = R * dZ_ds / jac
    grad_theta_phi = (dZ_dphi * dR_ds - dR_dphi * dZ_ds) / jac
    grad_theta_Z = -R * dR_ds / jac
    return BR * grad_theta_R + Bphi * grad_theta_phi + BZ * grad_theta_Z


def surface_field_alignment_diagnostics(
    grid_R: np.ndarray,
    grid_Z: np.ndarray,
    grid_phi: np.ndarray,
    B_R: np.ndarray,
    B_phi: np.ndarray,
    B_Z: np.ndarray,
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    radial_labels: np.ndarray,
    *,
    field_periods: int = 1,
    iota_profile: np.ndarray | None = None,
    bounds_error: bool = False,
    fill_value: float | None = np.nan,
    eps: float = 1.0e-300,
) -> SurfaceFieldAlignmentDiagnostics:
    """Measure whether ``B`` is tangent to candidate flux surfaces.

    The main metric is ``abs(B^s / B^phi)``.  A valid integrable background
    field should make this small on the surfaces used for a magnetic spectrum.
    When ``iota_profile`` is supplied, it is compared with the surface average
    of ``B^theta / B^phi`` in the same coordinates.
    """

    sampled_BR, sampled_BPhi, sampled_BZ = sample_cylindrical_vector_grid_on_surfaces(
        grid_R,
        grid_Z,
        grid_phi,
        B_R,
        B_phi,
        B_Z,
        R_surf,
        Z_surf,
        phi_vals,
        theta_vals,
        bounds_error=bounds_error,
        fill_value=fill_value,
        field_periods=field_periods,
    )
    B1, B3 = contravariant_radial_component(
        R_surf,
        Z_surf,
        phi_vals,
        theta_vals,
        sampled_BR,
        sampled_BZ,
        sampled_BPhi,
        radial_labels,
    )
    safe_B3 = np.where(np.abs(B3) < float(eps), np.nan + 0.0j, B3)
    ratio = np.abs(B1 / safe_B3)
    radial_ratio_rms = np.sqrt(np.nanmean(ratio * ratio, axis=(0, 2)))
    radial_ratio_p95 = np.nanpercentile(ratio, 95.0, axis=(0, 2))
    radial_ratio_max = np.nanmax(ratio, axis=(0, 2))
    global_rms = float(np.sqrt(np.nanmean(ratio * ratio)))
    edge = ratio[:, -min(2, ratio.shape[1]) :, :]
    edge_rms = float(np.sqrt(np.nanmean(edge * edge)))

    iota_from_field = None
    iota_error = None
    iota_flipped_error = None
    if iota_profile is not None:
        Btheta = _contravariant_theta_component(
            R_surf,
            Z_surf,
            phi_vals,
            theta_vals,
            sampled_BR,
            sampled_BZ,
            sampled_BPhi,
            radial_labels,
        )
        iota_from_field = np.real(np.nanmean(Btheta / safe_B3, axis=(0, 2)))
        iota_arr = np.asarray(iota_profile, dtype=float)
        if iota_arr.shape != iota_from_field.shape:
            raise ValueError("iota_profile must match radial_labels")
        iota_error = float(np.sqrt(np.nanmean((iota_arr - iota_from_field) ** 2)))
        iota_flipped_error = float(np.sqrt(np.nanmean((iota_arr + iota_from_field) ** 2)))

    return SurfaceFieldAlignmentDiagnostics(
        radial_labels=_validate_radial_labels(radial_labels, ratio.shape[1]),
        radial_ratio_rms=np.asarray(radial_ratio_rms, dtype=float),
        radial_ratio_p95=np.asarray(radial_ratio_p95, dtype=float),
        radial_ratio_max=np.asarray(radial_ratio_max, dtype=float),
        global_radial_ratio_rms=global_rms,
        edge_radial_ratio_rms=edge_rms,
        iota_from_field=None if iota_from_field is None else np.asarray(iota_from_field, dtype=float),
        iota_profile_error_rms=iota_error,
        iota_profile_sign_flipped_error_rms=iota_flipped_error,
    )


def nardon_radial_perturbation(
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    delta_B_R: np.ndarray,
    delta_B_Z: np.ndarray,
    delta_B_phi: np.ndarray | None,
    radial_labels: np.ndarray,
    *,
    denominator_B_phi: np.ndarray | None = None,
    denominator_B3: np.ndarray | None = None,
    eps: float = 1.0e-300,
) -> np.ndarray:
    """Compute Nardon's ``tilde_b^1 = delta B^1 / B_0^3`` on surfaces.

    Pass ``denominator_B_phi`` when the denominator should be the background
    toroidal contravariant field ``B_0 dot grad(phi)``.  Pass
    ``denominator_B3`` directly if it is already available on the same
    ``(phi, radial, theta)`` surface grid.
    """

    delta_B1, delta_B3 = contravariant_radial_component(
        R_surf,
        Z_surf,
        phi_vals,
        theta_vals,
        delta_B_R,
        delta_B_Z,
        delta_B_phi,
        radial_labels,
    )
    if denominator_B3 is not None:
        denom = strip_field_grid(np.asarray(denominator_B3, dtype=complex), theta_vals, phi_vals)
    elif denominator_B_phi is not None:
        R, _, _, _ = prepare_surface_arrays(R_surf, Z_surf, phi_vals, theta_vals)
        denom = strip_field_grid(np.asarray(denominator_B_phi, dtype=complex), theta_vals, phi_vals)
        denom = denom / np.maximum(R, 1.0e-300)
    else:
        denom = delta_B3
    if denom.shape != delta_B1.shape:
        raise ValueError("denominator field must match the surface shape after removing endpoints")
    denom = np.where(np.abs(denom) < float(eps), np.nan + 0.0j, denom)
    return delta_B1 / denom


def nardon_radial_perturbation_from_decomposition(
    grid_R: np.ndarray,
    grid_Z: np.ndarray,
    grid_phi: np.ndarray,
    B0_R: np.ndarray,
    B0_phi: np.ndarray,
    B0_Z: np.ndarray,
    delta_B_R: np.ndarray,
    delta_B_phi: np.ndarray,
    delta_B_Z: np.ndarray,
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    radial_labels: np.ndarray,
    *,
    field_periods: int = 1,
    bounds_error: bool = False,
    fill_value: float | None = np.nan,
    background_field_signature: Mapping[str, Any] | None = None,
    delta_field_signature: Mapping[str, Any] | None = None,
    surface_signature: Mapping[str, Any] | None = None,
    coordinate_system: str = "magnetic",
    radial_coordinate: str = "s",
    metadata: Mapping[str, Any] | None = None,
    eps: float = 1.0e-300,
) -> NardonRadialPerturbationProjection:
    """Sample ``delta_B`` on ``B0`` flux surfaces and compute Nardon's ``tilde_b^1``.

    This high-level entry point makes the decomposition explicit:
    ``B0`` supplies the nested-surface denominator ``B0^3`` and ``delta_B``
    supplies the radial perturbation ``delta_B^1``.  If a supplied surface
    signature is already bound to a background field, that background must match
    the ``B0`` signature.
    """

    background_sig = (
        dict(background_field_signature)
        if background_field_signature is not None
        else cylindrical_field_grid_signature(
            grid_R,
            grid_Z,
            grid_phi,
            B0_R,
            B0_phi,
            B0_Z,
            field_periods=field_periods,
        )
    )
    delta_sig = (
        dict(delta_field_signature)
        if delta_field_signature is not None
        else cylindrical_field_grid_signature(
            grid_R,
            grid_Z,
            grid_phi,
            delta_B_R,
            delta_B_phi,
            delta_B_Z,
            field_periods=field_periods,
        )
    )
    if surface_signature is None:
        surface_sig = surface_coordinate_signature(
            R_surf,
            Z_surf,
            phi_vals,
            theta_vals,
            radial_labels,
            background_field_signature=background_sig,
            coordinate_system=coordinate_system,
            radial_coordinate=radial_coordinate,
        )
    else:
        surface_sig = dict(surface_signature)
        bound_background = _surface_background_field_signature(surface_sig)
        if bound_background is not None:
            require_matching_field_signature(
                bound_background,
                background_sig,
                context="surface background field",
            )

    delta_BR, delta_BPhi, delta_BZ = sample_cylindrical_vector_grid_on_surfaces(
        grid_R,
        grid_Z,
        grid_phi,
        delta_B_R,
        delta_B_phi,
        delta_B_Z,
        R_surf,
        Z_surf,
        phi_vals,
        theta_vals,
        bounds_error=bounds_error,
        fill_value=fill_value,
        field_periods=field_periods,
    )
    background_BR, background_BPhi, background_BZ = sample_cylindrical_vector_grid_on_surfaces(
        grid_R,
        grid_Z,
        grid_phi,
        B0_R,
        B0_phi,
        B0_Z,
        R_surf,
        Z_surf,
        phi_vals,
        theta_vals,
        bounds_error=bounds_error,
        fill_value=fill_value,
        field_periods=field_periods,
    )
    delta_B1, _delta_B3 = contravariant_radial_component(
        R_surf,
        Z_surf,
        phi_vals,
        theta_vals,
        delta_BR,
        delta_BZ,
        delta_BPhi,
        radial_labels,
    )
    _background_B1, background_B3 = contravariant_radial_component(
        R_surf,
        Z_surf,
        phi_vals,
        theta_vals,
        background_BR,
        background_BZ,
        background_BPhi,
        radial_labels,
    )
    denom = np.where(np.abs(background_B3) < float(eps), np.nan + 0.0j, background_B3)
    tilde_b1 = delta_B1 / denom
    R_prepared, Z_prepared, phi_prepared, theta_prepared = prepare_surface_arrays(
        R_surf,
        Z_surf,
        phi_vals,
        theta_vals,
    )
    labels = _validate_radial_labels(radial_labels, R_prepared.shape[1])
    return NardonRadialPerturbationProjection(
        tilde_b1=tilde_b1,
        delta_B1=delta_B1,
        background_B3=background_B3,
        delta_BR=delta_BR,
        delta_BZ=delta_BZ,
        delta_BPhi=delta_BPhi,
        background_BR=background_BR,
        background_BZ=background_BZ,
        background_BPhi=background_BPhi,
        R_surf=R_prepared,
        Z_surf=Z_prepared,
        phi_vals=phi_prepared,
        theta_vals=theta_prepared,
        radial_labels=labels,
        surface_signature=surface_sig,
        background_field_signature=background_sig,
        delta_field_signature=delta_sig,
        metadata={} if metadata is None else metadata,
    )


def nardon_radial_perturbation_from_healed_surfaces(
    grid_R: np.ndarray,
    grid_Z: np.ndarray,
    grid_phi: np.ndarray,
    total_B_R: np.ndarray,
    total_B_phi: np.ndarray,
    total_B_Z: np.ndarray,
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    radial_labels: np.ndarray,
    *,
    denominator_B_R: np.ndarray | None = None,
    denominator_B_phi: np.ndarray | None = None,
    denominator_B_Z: np.ndarray | None = None,
    field_periods: int = 1,
    bounds_error: bool = False,
    fill_value: float | None = np.nan,
    surface_signature: Mapping[str, Any] | None = None,
    background_field_signature: Mapping[str, Any] | None = None,
    total_field_signature: Mapping[str, Any] | None = None,
    coordinate_system: str = "magnetic",
    radial_coordinate: str = "s",
    metadata: Mapping[str, Any] | None = None,
    eps: float = 1.0e-300,
) -> NardonRadialPerturbationProjection:
    """Project radial perturbation using healed surfaces as the B0 definition.

    This convention treats the supplied surface stack as the current integrable
    reference: by definition ``B0^1 = B0 dot grad(s) = 0`` on those surfaces.
    Therefore the perturbing radial component is ``B_total^1`` rather than
    ``B_total^1 - B_file^1``.  An optional denominator field may still supply
    ``B0^3``; if omitted, the total field's ``B^3`` is used.
    """

    total_sig = (
        dict(total_field_signature)
        if total_field_signature is not None
        else cylindrical_field_grid_signature(
            grid_R,
            grid_Z,
            grid_phi,
            total_B_R,
            total_B_phi,
            total_B_Z,
            field_periods=field_periods,
        )
    )
    background_sig = None
    if denominator_B_R is not None or denominator_B_phi is not None or denominator_B_Z is not None:
        if denominator_B_R is None or denominator_B_phi is None or denominator_B_Z is None:
            raise ValueError("denominator field components must be supplied together")
        background_sig = (
            dict(background_field_signature)
            if background_field_signature is not None
            else cylindrical_field_grid_signature(
                grid_R,
                grid_Z,
                grid_phi,
                denominator_B_R,
                denominator_B_phi,
                denominator_B_Z,
                field_periods=field_periods,
            )
        )
    elif background_field_signature is not None:
        background_sig = dict(background_field_signature)

    if surface_signature is None:
        surface_sig = surface_coordinate_signature(
            R_surf,
            Z_surf,
            phi_vals,
            theta_vals,
            radial_labels,
            background_field_signature=background_sig,
            coordinate_system=coordinate_system,
            radial_coordinate=radial_coordinate,
        )
    else:
        surface_sig = dict(surface_signature)
        bound_background = _surface_background_field_signature(surface_sig)
        if bound_background is not None and background_sig is not None:
            require_matching_field_signature(
                bound_background,
                background_sig,
                context="surface background field",
            )

    total_BR, total_BPhi, total_BZ = sample_cylindrical_vector_grid_on_surfaces(
        grid_R,
        grid_Z,
        grid_phi,
        total_B_R,
        total_B_phi,
        total_B_Z,
        R_surf,
        Z_surf,
        phi_vals,
        theta_vals,
        bounds_error=bounds_error,
        fill_value=fill_value,
        field_periods=field_periods,
    )
    total_B1, total_B3 = contravariant_radial_component(
        R_surf,
        Z_surf,
        phi_vals,
        theta_vals,
        total_BR,
        total_BZ,
        total_BPhi,
        radial_labels,
    )

    if denominator_B_R is None:
        background_BR = total_BR
        background_BPhi = total_BPhi
        background_BZ = total_BZ
        background_B3 = total_B3
    else:
        background_BR, background_BPhi, background_BZ = sample_cylindrical_vector_grid_on_surfaces(
            grid_R,
            grid_Z,
            grid_phi,
            denominator_B_R,
            denominator_B_phi,
            denominator_B_Z,
            R_surf,
            Z_surf,
            phi_vals,
            theta_vals,
            bounds_error=bounds_error,
            fill_value=fill_value,
            field_periods=field_periods,
        )
        _background_B1, background_B3 = contravariant_radial_component(
            R_surf,
            Z_surf,
            phi_vals,
            theta_vals,
            background_BR,
            background_BZ,
            background_BPhi,
            radial_labels,
        )

    denom = np.where(np.abs(background_B3) < float(eps), np.nan + 0.0j, background_B3)
    R_prepared, Z_prepared, phi_prepared, theta_prepared = prepare_surface_arrays(
        R_surf,
        Z_surf,
        phi_vals,
        theta_vals,
    )
    labels = _validate_radial_labels(radial_labels, R_prepared.shape[1])
    md = dict(metadata or {})
    md.setdefault("reference", "healed_surfaces")
    md.setdefault("delta_B1_convention", "total_B1")
    return NardonRadialPerturbationProjection(
        tilde_b1=total_B1 / denom,
        delta_B1=total_B1,
        background_B3=background_B3,
        delta_BR=total_BR,
        delta_BZ=total_BZ,
        delta_BPhi=total_BPhi,
        background_BR=background_BR,
        background_BZ=background_BZ,
        background_BPhi=background_BPhi,
        R_surf=R_prepared,
        Z_surf=Z_prepared,
        phi_vals=phi_prepared,
        theta_vals=theta_prepared,
        radial_labels=labels,
        surface_signature=surface_sig,
        background_field_signature=background_sig,
        delta_field_signature=total_sig,
        metadata=md,
    )


def _fourier_layout(layout: str | Iterable[str] | None, ndim: int) -> str | None:
    if layout is None:
        return None
    if isinstance(layout, str):
        raw_tokens = layout.lower().replace("_", "-").replace(",", "-").replace(" ", "-").split("-")
    else:
        try:
            raw_tokens = [str(value).lower() for value in layout]
        except TypeError as exc:
            raise ValueError("layout must name the dBr_grid axes") from exc
    aliases = {"r": "radial", "radius": "radial", "p": "phi", "t": "theta"}
    tokens = tuple(aliases.get(token, token) for token in raw_tokens if token)
    canonical = "-".join(tokens)
    allowed = {2: {"phi-theta"}, 3: {"radial-phi-theta", "phi-radial-theta"}}
    if canonical not in allowed[ndim]:
        choices = " or ".join(f"'{value}'" for value in sorted(allowed[ndim]))
        raise ValueError(f"layout for a {ndim}-D dBr_grid must be {choices}")
    return canonical


def _positive_field_periods(value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("field_periods must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("field_periods must be a positive integer") from exc
    if result < 1 or float(value) != float(result):
        raise ValueError("field_periods must be a positive integer")
    return result


def _infer_field_periods(phi_vals: np.ndarray) -> int:
    phi = np.asarray(phi_vals, dtype=np.float64)
    if phi.ndim != 1 or phi.size < 3:
        raise ValueError("phi_vals must be one-dimensional with at least three points")
    if not np.all(np.isfinite(phi)):
        raise ValueError("phi_vals must contain only finite values")
    steps = np.diff(phi)
    positive_steps = steps[steps > 1.0e-12]
    if positive_steps.size == 0 or not np.allclose(
        positive_steps,
        positive_steps[0],
        rtol=1.0e-9,
        atol=1.0e-12,
    ):
        raise ValueError(
            "field_periods is required when the toroidal domain cannot be inferred "
            "from a uniform phi_vals axis"
        )

    candidates = set()
    for intervals in (phi.size, phi.size - 1):
        domain = float(positive_steps[0]) * float(intervals)
        candidate = int(round(TWOPI / domain))
        if candidate >= 1 and np.isclose(
            domain,
            TWOPI / float(candidate),
            rtol=1.0e-9,
            atol=1.0e-12,
        ):
            try:
                _prepare_fourier_axis(phi, TWOPI / float(candidate), "phi_vals")
            except ValueError:
                continue
            candidates.add(candidate)
    if len(candidates) != 1:
        raise ValueError(
            "field_periods is required because the toroidal domain is not an "
            "unambiguous integer field period"
        )
    return candidates.pop()


def _prepare_fourier_axis(
    axis_values: np.ndarray,
    period: float,
    name: str,
) -> tuple[np.ndarray, bool, int]:
    raw = np.asarray(axis_values, dtype=np.float64)
    if raw.ndim != 1 or raw.size < 3:
        raise ValueError(f"{name} must be one-dimensional with at least three points")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must contain only finite values")

    unwrapped = np.unwrap(raw * (TWOPI / period)) * (period / TWOPI)
    has_endpoint = bool(
        np.isclose(unwrapped[-1] - unwrapped[0], period, rtol=1.0e-10, atol=1.0e-12)
    )
    axis = unwrapped[:-1] if has_endpoint else unwrapped
    expected_step = period / float(axis.size)
    if not np.allclose(
        np.diff(axis),
        expected_step,
        rtol=1.0e-9,
        atol=1.0e-12 * max(1.0, abs(period)),
    ):
        raise ValueError(
            f"{name} must uniformly sample exactly one period ({period:.16g}); "
            "set field_periods to match the toroidal domain"
        )
    return axis, has_endpoint, raw.size


def _grid_axis_matches(size: int, input_size: int, stripped_size: int) -> bool:
    return size == input_size or size == stripped_size


def _strip_grid_axis_endpoint(
    grid: np.ndarray,
    *,
    axis: int,
    input_size: int,
    stripped_size: int,
    has_endpoint: bool,
    coordinate_name: str,
) -> np.ndarray:
    size = grid.shape[axis]
    if not _grid_axis_matches(size, input_size, stripped_size):
        raise ValueError(
            f"dBr_grid axis {axis} has length {size}, which does not match "
            f"{coordinate_name} length {input_size}"
        )
    return drop_endpoint(grid, axis=axis, has_endpoint=has_endpoint and size == input_size)


def _fourier_mode_limit(value: int | None, default: int, name: str) -> int:
    if value is None:
        return int(default)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if result < 0 or float(value) != float(result):
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def radial_perturbation_Fourier_spectrum(
    dBr_grid: np.ndarray,
    theta_vals: np.ndarray,
    phi_vals: np.ndarray,
    *,
    radial_labels: np.ndarray | None = None,
    layout: str | Iterable[str] | None = None,
    field_periods: int | None = None,
    m_max: int | None = None,
    n_max: int | None = None,
    min_amplitude: float = 0.0,
    metadata: Mapping[str, Any] | None = None,
) -> RadialPerturbationFourierSpectrum:
    """Compute an absolute-coordinate radial perturbation Fourier spectrum.

    The Nardon convention is
    ``f(theta, phi) = sum tilde_b1_mn exp(i * (m * theta + nardon_n * phi))``.
    A stored one-field-period FFT harmonic ``k`` maps explicitly to
    ``nardon_n = field_periods * k``.  The spectrum's ``n`` and
    :attr:`RadialPerturbationFourierSpectrum.physical_n` are retained
    compatibility APIs; ``physical_n = -nardon_n``.

    ``dBr_grid`` may be ``(phi, theta)``, ``(radial, phi, theta)``, or
    ``(phi, radial, theta)``.  Set ``layout`` to ``"radial-phi-theta"`` or
    ``"phi-radial-theta"`` for a 3-D stack.  When omitted, layout is inferred
    only if exactly one leading axis matches ``phi_vals``; equal-length
    candidates are rejected as ambiguous.  ``phi_vals`` must sample exactly
    one domain of length ``2*pi/field_periods``.  When ``field_periods`` is
    omitted, it is inferred only when the uniform angular span identifies one
    integer field period unambiguously.  ``n_max`` limits the magnitude of the
    Nardon toroidal index, not local field-period harmonics.
    """

    grid = np.asarray(dBr_grid, dtype=complex)
    if grid.ndim not in (2, 3):
        raise ValueError(
            "dBr_grid must have shape (n_phi, n_theta), (n_r, n_phi, n_theta), "
            "or (n_phi, n_r, n_theta)"
    )
    resolved_layout = _fourier_layout(layout, grid.ndim)
    nfp = _infer_field_periods(phi_vals) if field_periods is None else _positive_field_periods(field_periods)
    theta, theta_has_endpoint, theta_input_size = _prepare_fourier_axis(
        theta_vals,
        TWOPI,
        "theta_vals",
    )
    phi, phi_has_endpoint, phi_input_size = _prepare_fourier_axis(
        phi_vals,
        TWOPI / float(nfp),
        "phi_vals",
    )
    grid = _strip_grid_axis_endpoint(
        grid,
        axis=-1,
        input_size=theta_input_size,
        stripped_size=theta.size,
        has_endpoint=theta_has_endpoint,
        coordinate_name="theta_vals",
    )

    single_surface = grid.ndim == 2
    if single_surface:
        grid = _strip_grid_axis_endpoint(
            grid,
            axis=0,
            input_size=phi_input_size,
            stripped_size=phi.size,
            has_endpoint=phi_has_endpoint,
            coordinate_name="phi_vals",
        )
    else:
        radial_first_possible = _grid_axis_matches(grid.shape[1], phi_input_size, phi.size)
        phi_first_possible = _grid_axis_matches(grid.shape[0], phi_input_size, phi.size)
        if resolved_layout is None:
            if radial_first_possible and phi_first_possible:
                raise ValueError(
                    "ambiguous 3-D dBr_grid layout: both axis 0 and axis 1 match phi_vals; "
                    "pass layout='radial-phi-theta' or layout='phi-radial-theta'"
                )
            if radial_first_possible:
                resolved_layout = "radial-phi-theta"
            elif phi_first_possible:
                resolved_layout = "phi-radial-theta"
            else:
                raise ValueError(
                    "3-D dBr_grid must be radial-first (n_r, n_phi, n_theta) or "
                    "phi-first (n_phi, n_r, n_theta)"
                )
        phi_axis = 1 if resolved_layout == "radial-phi-theta" else 0
        grid = _strip_grid_axis_endpoint(
            grid,
            axis=phi_axis,
            input_size=phi_input_size,
            stripped_size=phi.size,
            has_endpoint=phi_has_endpoint,
            coordinate_name="phi_vals",
        )
        if resolved_layout == "phi-radial-theta":
            grid = np.moveaxis(grid, 1, 0)

    if grid.shape[-2:] != (phi.size, theta.size):
        raise ValueError("dBr_grid shape must match phi_vals and theta_vals")
    labels = None
    if radial_labels is not None:
        if single_surface:
            raise ValueError("radial_labels are only valid for radial stack spectra")
        labels = _validate_radial_labels(radial_labels, grid.shape[0])

    fft = np.fft.fft2(grid, axes=(-2, -1)) / float(theta.size * phi.size)
    m_freq = np.fft.fftfreq(theta.size, 1.0 / theta.size).astype(int)
    field_period_freq = np.fft.fftfreq(phi.size, 1.0 / phi.size).astype(int)
    m_limit = _fourier_mode_limit(m_max, int(np.max(np.abs(m_freq))), "m_max")
    nardon_n_available = nfp * field_period_freq
    n_limit = _fourier_mode_limit(
        n_max,
        int(np.max(np.abs(nardon_n_available))),
        "n_max",
    )

    modes_m = []
    modes_n = []
    coeffs = []
    retained_field_period_harmonics = sorted(
        int(value)
        for value in field_period_freq
        if abs(nfp * int(value)) <= n_limit
    )
    for m_val in range(-m_limit, m_limit + 1):
        m_idx = np.where(m_freq == m_val)[0]
        if m_idx.size == 0:
            continue
        for field_period_harmonic in retained_field_period_harmonics:
            n_idx = np.where(field_period_freq == field_period_harmonic)[0]
            nardon_n = nfp * field_period_harmonic
            coeff = fft[..., int(n_idx[0]), int(m_idx[0])]
            coeff = coeff * np.exp(-1j * (m_val * theta[0] + nardon_n * phi[0]))
            if np.max(np.abs(coeff)) < float(min_amplitude):
                continue
            modes_m.append(m_val)
            modes_n.append(field_period_harmonic)
            coeffs.append(coeff)

    if coeffs:
        dBr = np.stack(coeffs, axis=-1)
    else:
        dBr = np.empty(grid.shape[:-2] + (0,), dtype=complex)
    if single_surface:
        dBr = np.asarray(dBr, dtype=complex).reshape((-1,))
    return RadialPerturbationFourierSpectrum(
        m=np.asarray(modes_m, dtype=int),
        n=np.asarray(modes_n, dtype=int),
        dBr=dBr,
        dBr_grid=grid,
        theta=theta,
        phi=phi,
        radial_labels=labels,
        metadata={} if metadata is None else metadata,
        field_periods=nfp,
    )


def island_chain_fixed_points(
    m: int,
    n: int,
    coefficient: complex,
    phi: float | np.ndarray,
    *,
    q_prime_sign: int = 1,
    coefficient_n: int | None = None,
) -> dict[str, np.ndarray]:
    """Return O/X poloidal angles implied by a resonant Fourier coefficient.

    The returned ``theta_O`` and ``theta_X`` arrays have shape ``(n_phi, m)``.
    A phase change ``coefficient *= exp(1j * alpha)`` rotates every branch by
    ``-alpha / m`` at fixed toroidal section.
    """

    m_int = int(m)
    n_int = int(n)
    if m_int <= 0 or n_int <= 0:
        raise ValueError("m and n must be positive resonant mode numbers")
    coefficient_n_int = -n_int if coefficient_n is None else int(coefficient_n)
    if abs(coefficient_n_int) != n_int:
        raise ValueError("coefficient_n must have magnitude n")
    sign = 1 if int(np.sign(q_prime_sign)) >= 0 else -1
    phi_arr = np.atleast_1d(np.asarray(phi, dtype=np.float64))
    phase = float(np.angle(coefficient))
    if sign >= 0:
        base_O = -coefficient_n_int * phi_arr - 0.5 * np.pi - phase
        base_X = -coefficient_n_int * phi_arr + 0.5 * np.pi - phase
    else:
        base_O = -coefficient_n_int * phi_arr + 0.5 * np.pi - phase
        base_X = -coefficient_n_int * phi_arr - 0.5 * np.pi - phase
    branches = np.arange(m_int, dtype=np.float64)
    theta_O = (base_O[:, None] + TWOPI * branches[None, :]) / float(m_int)
    theta_X = (base_X[:, None] + TWOPI * branches[None, :]) / float(m_int)
    return {
        "phi": np.mod(phi_arr, TWOPI),
        "theta_O": np.mod(theta_O, TWOPI),
        "theta_X": np.mod(theta_X, TWOPI),
    }


def nardon_resonant_amplitude(coefficient: complex) -> float:
    """Return ``tilde_b_res^1 = 2 |tilde_b^1_{m,-n}|``."""

    return float(2.0 * abs(coefficient))


def nardon_island_half_width(q: float, q_prime: float, m: int, b_res: float) -> float:
    """Return Nardon's magnetic-island half-width in the radial coordinate.

    The thesis formula is ``sqrt(4 q^2 b_res / (q' m))``.  This implementation
    returns a positive geometric width and therefore uses ``abs(q' m)`` in the
    denominator.
    """

    m_int = int(m)
    if m_int <= 0:
        raise ValueError("m must be positive")
    denom = abs(float(q_prime) * float(m_int))
    if denom <= 0.0:
        return float("nan")
    value = 4.0 * float(q) * float(q) * max(float(b_res), 0.0) / denom
    return float(np.sqrt(value))


def _as_mode_values(m_values: Iterable[int] | None, q_profile: np.ndarray, n: int) -> list[int]:
    if m_values is not None:
        out = sorted({int(m) for m in m_values if int(m) > 0})
        return out
    q_abs = np.abs(np.asarray(q_profile, dtype=np.float64))
    q_min = float(np.nanmin(q_abs))
    q_max = float(np.nanmax(q_abs))
    lo = int(np.floor(min(q_min, q_max) * int(n))) - 1
    hi = int(np.ceil(max(q_min, q_max) * int(n))) + 1
    return [m for m in range(max(1, lo), max(1, hi) + 1)]


def _q_helicity_sign(q_profile: np.ndarray) -> int:
    q_arr = np.asarray(q_profile, dtype=np.float64)
    finite = q_arr[np.isfinite(q_arr) & (np.abs(q_arr) > 0.0)]
    if finite.size == 0:
        return 1
    if np.all(finite < 0.0):
        return -1
    return 1


def _find_crossings(radial: np.ndarray, values: np.ndarray, target: float) -> list[float]:
    roots: list[float] = []
    diff = np.asarray(values, dtype=np.float64) - float(target)
    for i in range(radial.size - 1):
        f0 = diff[i]
        f1 = diff[i + 1]
        if not np.isfinite(f0) or not np.isfinite(f1):
            continue
        if f0 == 0.0:
            roots.append(float(radial[i]))
        if f0 * f1 < 0.0:
            t = -f0 / (f1 - f0)
            roots.append(float(radial[i] + t * (radial[i + 1] - radial[i])))
    if diff[-1] == 0.0:
        roots.append(float(radial[-1]))
    return roots


def _interp_complex(x: np.ndarray, y: np.ndarray, x0: float) -> complex:
    return complex(
        np.interp(float(x0), x, np.real(y)),
        np.interp(float(x0), x, np.imag(y)),
    )


def analyze_resonant_island_chains(
    spectrum: RadialPerturbationFourierSpectrum,
    q_profile: np.ndarray | MagneticCoordinateProfile,
    *,
    n: int,
    radial_labels: np.ndarray | None = None,
    m_values: Iterable[int] | None = None,
    min_b_res: float = 0.0,
) -> list[ResonantIslandChain]:
    """Analyze resonant island chains from a radial Fourier spectrum.

    ``n`` is the positive resonance-family label ``n0``, not a Fourier index.
    For each requested ``m`` this finds roots of ``q(s) = q_sign*m/n0`` and
    explicitly selects Nardon's signed coefficient
    ``tilde_b^1_(m, -q_sign*n0)``.  It then evaluates Nardon's island
    half-width formula in the same radial coordinate ``s``.
    """

    if spectrum.dBr.ndim != 2:
        raise ValueError("analyze_resonant_island_chains requires a radial stack spectrum")
    n_int = int(n)
    if n_int <= 0:
        raise ValueError("n must be positive")
    if isinstance(q_profile, MagneticCoordinateProfile):
        if str(q_profile.quantity).lower() not in {"q", "safety_factor"}:
            raise ValueError("q_profile must be a q/safety-factor MagneticCoordinateProfile")
        q_profile.require_compatible_with(
            surface_signature=spectrum.surface_signature,
            background_field_signature=spectrum.background_field_signature,
            context="q_profile",
        )
        if radial_labels is None:
            radial_labels = q_profile.radial_labels
        q_profile_values = q_profile.values
    else:
        q_profile_values = q_profile
    radial = spectrum.radial_labels if radial_labels is None else radial_labels
    if radial is None:
        raise ValueError("radial_labels are required")
    radial = _validate_radial_labels(radial, spectrum.dBr.shape[0])
    q_arr = np.asarray(q_profile_values, dtype=np.float64)
    if q_arr.shape != radial.shape:
        raise ValueError("q_profile must have the same shape as radial_labels")
    q_prime_profile = np.gradient(q_arr, radial, edge_order=2 if radial.size >= 3 else 1)
    q_sign = _q_helicity_sign(q_arr)
    coefficient_n = -q_sign * n_int

    chains: list[ResonantIslandChain] = []
    for m_int in _as_mode_values(m_values, q_arr, n_int):
        idx = spectrum.nardon_mode_index(m_int, coefficient_n)
        if idx is None:
            continue
        roots = _find_crossings(radial, q_arr, q_sign * float(m_int) / float(n_int))
        coeff_profile = spectrum.dBr[:, idx]
        for s_res in roots:
            q_res = float(np.interp(s_res, radial, q_arr))
            q_prime = float(np.interp(s_res, radial, q_prime_profile))
            coeff = _interp_complex(radial, coeff_profile, s_res)
            b_res = nardon_resonant_amplitude(coeff)
            if b_res < float(min_b_res):
                continue
            chains.append(
                ResonantIslandChain(
                    m=m_int,
                    n=n_int,
                    radial_label=float(s_res),
                    q=q_res,
                    q_prime=q_prime,
                    coefficient=coeff,
                    b_res=b_res,
                    half_width=nardon_island_half_width(q_res, q_prime, m_int, b_res),
                    coefficient_n=coefficient_n,
                )
            )
    chains.sort(key=lambda chain: (chain.radial_label, chain.m, chain.n))
    return chains


def analyze_resonant_island_chains_multi_n(
    spectrum: RadialPerturbationFourierSpectrum,
    q_profile: np.ndarray,
    *,
    n_values: Iterable[int] | None = None,
    radial_labels: np.ndarray | None = None,
    m_values: Iterable[int] | Mapping[int, Iterable[int]] | None = None,
    min_b_res: float = 0.0,
) -> list[ResonantIslandChain]:
    """Analyze all requested resonant ``(m, n)`` island chains together.

    This is the multi-component counterpart to
    :func:`analyze_resonant_island_chains`.  For each positive toroidal mode
    number ``n`` it finds every requested ``q(s)=m/n`` crossing, interpolates
    the resonant coefficient ``tilde_b^1_{m,-n}``, and returns one combined
    list sorted by radial position and mode number.

    Parameters
    ----------
    spectrum:
        Radial stack of ``tilde_b^1_{mn}`` Fourier coefficients.
    q_profile:
        Safety-factor profile sampled on ``radial_labels``.
    n_values:
        Positive resonance-family labels ``n0`` to scan.  If omitted, they are
        derived as ``abs(spectrum.nardon_n)``.
    radial_labels:
        Optional radial labels overriding ``spectrum.radial_labels``.
    m_values:
        Optional positive poloidal mode numbers.  Pass a mapping ``{n: m_list}``
        when different toroidal families need different candidate ``m`` values.
    min_b_res:
        Drop chains with ``2*abs(tilde_b^1_{m,-n})`` below this threshold.
    """

    if n_values is None:
        n_scan = sorted({int(n0) for n0 in np.asarray(spectrum.resonance_family_n0).ravel() if int(n0) != 0})
    else:
        n_scan = sorted({int(n_val) for n_val in n_values if int(n_val) > 0})
    chains: list[ResonantIslandChain] = []
    for n_int in n_scan:
        if isinstance(m_values, Mapping):
            m_for_n = m_values.get(n_int)
        else:
            m_for_n = m_values
        chains.extend(
            analyze_resonant_island_chains(
                spectrum,
                q_profile,
                n=n_int,
                radial_labels=radial_labels,
                m_values=m_for_n,
                min_b_res=min_b_res,
            )
        )
    chains.sort(key=lambda chain: (chain.radial_label, chain.n, chain.m))
    return chains


def chirikov_overlaps(chains: Iterable[ResonantIslandChain]) -> list[ChirikovOverlap]:
    """Compute Chirikov overlap for adjacent chains with the same toroidal ``n``."""

    grouped: dict[int, list[ResonantIslandChain]] = {}
    for chain in chains:
        grouped.setdefault(chain.n, []).append(chain)
    overlaps: list[ChirikovOverlap] = []
    for same_n in grouped.values():
        ordered = sorted(same_n, key=lambda chain: chain.radial_label)
        for left, right in zip(ordered[:-1], ordered[1:]):
            separation = abs(right.radial_label - left.radial_label)
            if separation <= 0.0:
                sigma = float("inf")
            else:
                sigma = float((left.half_width + right.half_width) / separation)
            overlaps.append(ChirikovOverlap(left=left, right=right, separation=separation, sigma=sigma))
    return overlaps


def _finite_width_chain(chain: ResonantIslandChain) -> bool:
    return bool(
        np.isfinite(chain.radial_label)
        and np.isfinite(chain.half_width)
        and float(chain.half_width) >= 0.0
    )


def _combined_norm(values: Iterable[float], rule: str) -> float:
    vals = np.asarray([float(value) for value in values], dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals >= 0.0)]
    if vals.size == 0:
        return 0.0
    rule_key = str(rule).lower()
    if rule_key in {"rss", "quadrature", "root_sum_square"}:
        return float(np.sqrt(np.sum(vals * vals)))
    if rule_key == "sum":
        return float(np.sum(vals))
    if rule_key == "max":
        return float(np.max(vals))
    raise ValueError("width_rule must be 'rss', 'sum', or 'max'")


def group_resonant_island_chains(
    chains: Iterable[ResonantIslandChain],
    *,
    radial_tol: float = 1.0e-10,
    q_tol: float | None = None,
    width_rule: str = "rss",
) -> list[ResonantSurfaceGroup]:
    """Group co-radial resonant Fourier contributors on the same surface.

    Multi-harmonic contributors such as ``(9, 3)``, ``(12, 4)``, and ``(15, 5)``
    can land on the same rational surface.  Treating those as adjacent Chirikov
    pairs would create a zero-separation false positive.  This helper groups
    such co-radial contributors first and reports a combined width proxy.
    """

    ordered = sorted(
        (chain for chain in chains if _finite_width_chain(chain)),
        key=lambda chain: (float(chain.radial_label), int(chain.n), int(chain.m)),
    )
    if not ordered:
        return []
    radial_eps = max(0.0, float(radial_tol))
    q_eps = None if q_tol is None else max(0.0, float(q_tol))
    grouped: list[list[ResonantIslandChain]] = []
    for chain in ordered:
        if not grouped:
            grouped.append([chain])
            continue
        ref = grouped[-1][-1]
        same_radial = abs(float(chain.radial_label) - float(ref.radial_label)) <= radial_eps
        same_q = True if q_eps is None else abs(float(chain.q) - float(ref.q)) <= q_eps
        if same_radial and same_q:
            grouped[-1].append(chain)
        else:
            grouped.append([chain])

    groups: list[ResonantSurfaceGroup] = []
    for members in grouped:
        radial = np.asarray([float(chain.radial_label) for chain in members], dtype=np.float64)
        q_vals = np.asarray([float(chain.q) for chain in members], dtype=np.float64)
        qp_vals = np.asarray([float(chain.q_prime) for chain in members], dtype=np.float64)
        groups.append(
            ResonantSurfaceGroup(
                radial_label=float(np.mean(radial)),
                q=float(np.mean(q_vals)),
                q_prime=float(np.mean(qp_vals)),
                half_width=_combined_norm((chain.half_width for chain in members), width_rule),
                b_res=_combined_norm((chain.b_res for chain in members), width_rule),
                chains=tuple(members),
            )
        )
    return groups


def coalesce_resonant_island_chains(
    chains: Iterable[ResonantIslandChain],
    *,
    radial_tol: float = 1.0e-10,
    q_tol: float | None = None,
    width_rule: str = "rss",
) -> list[ResonantIslandChain]:
    """Return effective chains after combining same-surface contributors."""

    effective: list[ResonantIslandChain] = []
    for group in group_resonant_island_chains(
        chains,
        radial_tol=radial_tol,
        q_tol=q_tol,
        width_rule=width_rule,
    ):
        rep = group.dominant_chain
        effective.append(
            ResonantIslandChain(
                m=rep.m,
                n=rep.n,
                radial_label=group.radial_label,
                q=group.q,
                q_prime=group.q_prime,
                coefficient=rep.coefficient,
                b_res=group.b_res,
                half_width=group.half_width,
                coefficient_n=rep.coefficient_n,
            )
        )
    effective.sort(key=lambda chain: (chain.radial_label, chain.n, chain.m))
    return effective


def _overlap_band_from_pair(
    left: ResonantIslandChain,
    right: ResonantIslandChain,
    *,
    radial_min: float | None = None,
    radial_max: float | None = None,
) -> ChirikovOverlapBand:
    if float(right.radial_label) < float(left.radial_label):
        left, right = right, left
    separation = abs(float(right.radial_label) - float(left.radial_label))
    sum_width = float(left.half_width) + float(right.half_width)
    sigma = float("inf") if separation <= 0.0 else float(sum_width / separation)
    inner = float(min(left.radial_label - left.half_width, right.radial_label - right.half_width))
    outer = float(max(left.radial_label + left.half_width, right.radial_label + right.half_width))
    overlap_inner = float(right.radial_label - right.half_width)
    overlap_outer = float(left.radial_label + left.half_width)
    overlap_width = max(0.0, overlap_outer - overlap_inner)
    if radial_min is not None:
        lo = float(radial_min)
        inner = max(inner, lo)
        overlap_inner = max(overlap_inner, lo)
    if radial_max is not None:
        hi = float(radial_max)
        outer = min(outer, hi)
        overlap_outer = min(overlap_outer, hi)
    overlap_width = max(0.0, overlap_outer - overlap_inner)
    return ChirikovOverlapBand(
        left=left,
        right=right,
        separation=separation,
        sigma=sigma,
        inner=inner,
        outer=outer,
        overlap_inner=overlap_inner,
        overlap_outer=overlap_outer,
        overlap_width=overlap_width,
    )


def chirikov_overlap_bands(
    chains: Iterable[ResonantIslandChain],
    *,
    include_cross_n: bool = True,
    radial_min: float | None = None,
    radial_max: float | None = None,
    min_sigma: float = 0.0,
    co_radial_tol: float = 0.0,
) -> list[ChirikovOverlapBand]:
    """Compute radial Chirikov bands for adjacent island chains.

    By default this compares all adjacent chains in radial order, including
    different toroidal families.  Set ``include_cross_n=False`` to recover the
    older same-``n`` family comparison.  Co-radial pairs are skipped because
    they are same-surface multi-harmonic contributors, not adjacent surfaces.
    """

    finite = [chain for chain in chains if _finite_width_chain(chain)]
    bands: list[ChirikovOverlapBand] = []
    if include_cross_n:
        groups = [sorted(finite, key=lambda chain: (chain.radial_label, chain.n, chain.m))]
    else:
        grouped: dict[int, list[ResonantIslandChain]] = {}
        for chain in finite:
            grouped.setdefault(chain.n, []).append(chain)
        groups = [sorted(group, key=lambda chain: chain.radial_label) for group in grouped.values()]
    for group in groups:
        for left, right in zip(group[:-1], group[1:]):
            if abs(float(right.radial_label) - float(left.radial_label)) <= float(co_radial_tol):
                continue
            band = _overlap_band_from_pair(left, right, radial_min=radial_min, radial_max=radial_max)
            if band.sigma >= float(min_sigma):
                bands.append(band)
    bands.sort(key=lambda band: (band.inner, band.outer, band.left.n, band.left.m, band.right.n, band.right.m))
    return bands


def chaotic_layer_intervals(
    bands: Iterable[ChirikovOverlapBand],
    *,
    sigma_threshold: float = 1.0,
) -> list[ChaoticLayerInterval]:
    """Merge overlap bands above ``sigma_threshold`` into radial chaotic layers."""

    selected = [
        band
        for band in bands
        if np.isfinite(band.inner)
        and np.isfinite(band.outer)
        and band.outer >= band.inner
        and float(band.sigma) >= float(sigma_threshold)
    ]
    selected.sort(key=lambda band: (band.inner, band.outer))
    intervals: list[ChaoticLayerInterval] = []
    for band in selected:
        if not intervals or band.inner > intervals[-1].outer:
            intervals.append(
                ChaoticLayerInterval(
                    inner=float(band.inner),
                    outer=float(band.outer),
                    max_sigma=float(band.sigma),
                    bands=(band,),
                )
            )
            continue
        prev = intervals[-1]
        intervals[-1] = ChaoticLayerInterval(
            inner=prev.inner,
            outer=max(prev.outer, float(band.outer)),
            max_sigma=max(prev.max_sigma, float(band.sigma)),
            bands=prev.bands + (band,),
        )
    return intervals


def sample_cylindrical_vector_grid_on_surfaces(
    grid_R: np.ndarray,
    grid_Z: np.ndarray,
    grid_phi: np.ndarray,
    field_R: np.ndarray,
    field_phi: np.ndarray,
    field_Z: np.ndarray,
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    phi_vals: np.ndarray,
    theta_vals: np.ndarray,
    *,
    bounds_error: bool = False,
    fill_value: float | None = np.nan,
    field_periods: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a rectilinear cylindrical vector grid on ``(phi, radial, theta)`` surfaces.

    ``field_periods`` lets the field grid describe one native field period while
    the surface coordinates span the full torus.
    """

    from scipy.interpolate import RegularGridInterpolator

    R, Z, phi, _ = prepare_surface_arrays(R_surf, Z_surf, phi_vals, theta_vals)
    axis_R = np.asarray(grid_R, dtype=np.float64)
    axis_Z = np.asarray(grid_Z, dtype=np.float64)
    axis_phi = np.asarray(grid_phi, dtype=np.float64)
    if axis_phi.ndim != 1 or axis_phi.size < 2:
        raise ValueError("grid_phi must be one-dimensional with at least two points")
    field_periods_i = int(field_periods)
    if field_periods_i < 1:
        raise ValueError("field_periods must be positive")
    field_period = TWOPI / float(field_periods_i)
    phi0 = float(axis_phi[0])
    phi_stripped, phi_has_endpoint = strip_periodic_endpoint(axis_phi, field_period, "grid_phi")
    phi_span = float(phi_stripped[-1] - phi_stripped[0])
    tol = max(1.0e-10, 1.0e-10 * abs(field_period))
    if phi_span > field_period + tol:
        raise ValueError("grid_phi span exceeds one field period for field_periods")

    def extend(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vals = np.asarray(values)
        vals = drop_endpoint(vals, axis=2, has_endpoint=phi_has_endpoint)
        if vals.shape != (axis_R.size, axis_Z.size, phi_stripped.size):
            raise ValueError("field arrays must have shape (n_R, n_Z, n_phi)")
        vals_ext = np.concatenate([vals, vals[:, :, :1]], axis=2)
        phi_ext = np.concatenate([phi_stripped, [phi0 + field_period]])
        return phi_ext, vals_ext

    phi_ext, vals_R = extend(field_R)
    _, vals_phi = extend(field_phi)
    _, vals_Z = extend(field_Z)
    pts = np.column_stack(
        [
            R.ravel(),
            Z.ravel(),
            (
                np.mod(
                    np.repeat(phi[:, None], R.shape[1] * R.shape[2], axis=1).ravel() - phi0,
                    field_period,
                )
                + phi0
            ),
        ]
    )
    kwargs = {"bounds_error": bounds_error, "fill_value": fill_value}
    interp_R = RegularGridInterpolator((axis_R, axis_Z, phi_ext), vals_R, **kwargs)
    interp_phi = RegularGridInterpolator((axis_R, axis_Z, phi_ext), vals_phi, **kwargs)
    interp_Z = RegularGridInterpolator((axis_R, axis_Z, phi_ext), vals_Z, **kwargs)
    out_shape = R.shape
    return (
        interp_R(pts).reshape(out_shape),
        interp_phi(pts).reshape(out_shape),
        interp_Z(pts).reshape(out_shape),
    )


__all__ = [
    "ChaoticLayerInterval",
    "ChirikovOverlap",
    "ChirikovOverlapBand",
    "IntegrableFieldDecomposition",
    "MagneticCoordinateProfile",
    "NardonRadialPerturbationProjection",
    "PeriodicOrbitGeometryDistance",
    "PeriodicOrbitSurfaceAlignment",
    "RadialPerturbationFourierSpectrum",
    "ResonantIslandChain",
    "ResonantSurfaceGroup",
    "SurfaceFieldAlignmentDiagnostics",
    "analyze_resonant_island_chains",
    "analyze_resonant_island_chains_multi_n",
    "chaotic_layer_intervals",
    "chirikov_overlap_bands",
    "chirikov_overlaps",
    "coalesce_resonant_island_chains",
    "contravariant_radial_component",
    "cylindrical_field_grid_signature",
    "group_resonant_island_chains",
    "island_chain_fixed_points",
    "integrable_field_decomposition_from_grids",
    "nardon_island_half_width",
    "nardon_radial_perturbation",
    "nardon_radial_perturbation_from_decomposition",
    "nardon_radial_perturbation_from_healed_surfaces",
    "nardon_resonant_amplitude",
    "periodic_orbit_geometry_distance",
    "periodic_orbit_surface_alignment",
    "radial_profile_signature",
    "radial_perturbation_Fourier_spectrum",
    "radial_perturbation_component",
    "require_matching_field_signature",
    "require_matching_signatures",
    "sample_cylindrical_vector_grid_on_surfaces",
    "signature_digest",
    "surface_coordinate_signature",
    "surface_field_alignment_diagnostics",
    "surface_unit_normal_cylindrical",
]
