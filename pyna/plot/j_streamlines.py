"""PEST-seeded current streamline plotting helpers."""
from __future__ import annotations

from dataclasses import dataclass, replace
import multiprocessing as mp
import operator
from pathlib import Path
from threading import Lock
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pyna.fields import VectorFieldCartesian, VectorFieldCylind
from pyna.toroidal.diagnostics.mgrid import SmoothPestCoordinates, smooth_pest_derivatives


TWOPI = 2.0 * np.pi


_PEST_TRACE_FORK_LOCK = Lock()
_PEST_TRACE_FORK_FIELD: object | None = None
_PEST_TRACE_FORK_PEST: object | None = None
_PEST_TRACE_FORK_KWARGS: dict[str, object] | None = None


def _cylindrical_to_cartesian(R: np.ndarray, Z: np.ndarray, phi: np.ndarray) -> np.ndarray:
    R_arr, Z_arr, phi_arr = np.broadcast_arrays(
        np.asarray(R, dtype=np.float64),
        np.asarray(Z, dtype=np.float64),
        np.asarray(phi, dtype=np.float64),
    )
    return np.stack([R_arr * np.cos(phi_arr), R_arr * np.sin(phi_arr), Z_arr], axis=-1)


@dataclass(frozen=True)
class PestSeededStreamlines:
    """Current streamlines seeded from a PEST surface mesh."""

    R: np.ndarray
    Z: np.ndarray
    phi: np.ndarray
    theta: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    seed_R: np.ndarray
    seed_Z: np.ndarray
    seed_phi: np.ndarray
    seed_rho: np.ndarray
    seed_theta: np.ndarray
    seed_surface_index: np.ndarray
    seed_phi_index: np.ndarray
    metadata: dict[str, object]

    @property
    def xyz(self) -> np.ndarray:
        return np.stack([self.x, self.y, self.z], axis=-1)

    @property
    def n_lines(self) -> int:
        return int(self.R.shape[0])

    @property
    def n_points(self) -> int:
        return int(self.R.shape[1]) if self.R.ndim == 2 else 0


@dataclass(frozen=True)
class PlotlyStreamlineStyle:
    """Visual controls for 3-D Plotly J/B streamline overlays."""

    surface_opacity: float = 0.22
    line_width: float = 3.4
    companion_line_width: float = 1.7
    j_color: str | None = "rgba(136, 28, 25, 0.92)"
    companion_color: str = "rgba(14, 116, 235, 0.50)"
    line_opacity: float = 0.94
    companion_line_opacity: float = 0.54
    show_arrows: bool = True
    arrow_count_per_line: int = 1
    companion_arrow_count_per_line: int = 1
    arrow_line_stride: int = 1
    companion_arrow_line_stride: int = 1
    arrow_size: float = 0.16
    companion_arrow_size: float | None = 0.18
    j_arrow_color: str = "#f97316"
    companion_arrow_color: str = "#0284c7"

    def to_plotly_kwargs(self) -> dict[str, object]:
        """Return keyword arguments accepted by ``plot_j_streamlines_on_pest_surface_plotly``."""

        return {
            "surface_opacity": float(self.surface_opacity),
            "line_width": float(self.line_width),
            "companion_line_width": float(self.companion_line_width),
            "j_color": self.j_color,
            "companion_color": self.companion_color,
            "line_opacity": float(self.line_opacity),
            "companion_line_opacity": float(self.companion_line_opacity),
            "show_arrows": bool(self.show_arrows),
            "arrow_count_per_line": int(self.arrow_count_per_line),
            "companion_arrow_count_per_line": int(self.companion_arrow_count_per_line),
            "arrow_line_stride": int(self.arrow_line_stride),
            "companion_arrow_line_stride": int(self.companion_arrow_line_stride),
            "arrow_size": float(self.arrow_size),
            "companion_arrow_size": (
                None if self.companion_arrow_size is None else float(self.companion_arrow_size)
            ),
            "j_arrow_color": self.j_arrow_color,
            "companion_arrow_color": self.companion_arrow_color,
        }


def plotly_streamline_style(name: str = "stellarator_j_b") -> PlotlyStreamlineStyle:
    """Return a named visual preset for J/B streamline Plotly figures."""

    key = str(name).strip().lower().replace("_", "-")
    if key in {"stellarator-j-b", "stellarator-j-b-balanced", "balanced"}:
        return PlotlyStreamlineStyle()
    if key in {"stellarator-j-b-dense", "one-period-dense", "dense"}:
        return PlotlyStreamlineStyle(
            companion_line_width=1.35,
            companion_line_opacity=0.38,
            companion_arrow_line_stride=3,
            companion_arrow_size=0.14,
        )
    raise ValueError("unknown streamline Plotly style preset")


def field_period_phi_range(nfp: int, *, period_index: int = 0, start: float = 0.0) -> tuple[float, float]:
    """Return the toroidal angle range for one field period."""

    nfp_int = max(int(nfp), 1)
    width = TWOPI / float(nfp_int)
    phi0 = float(start) + int(period_index) * width
    return float(phi0), float(phi0 + width)


@dataclass(frozen=True)
class GriddedPestVectorField:
    """Vector field sampled on the same ``(phi, rho, theta)`` mesh as PEST coords."""

    JR: np.ndarray
    JZ: np.ndarray
    JPhi: np.ndarray
    phi_vals: np.ndarray
    theta_vals: np.ndarray
    Jx: np.ndarray | None = None
    Jy: np.ndarray | None = None
    Jz: np.ndarray | None = None
    Jtheta: np.ndarray | None = None
    Jphi: np.ndarray | None = None
    nfp: int = 1
    phi_period: float = TWOPI
    source: str | None = None

    @property
    def field_period_rad(self) -> float:
        return TWOPI / max(int(self.nfp), 1)

    @classmethod
    def from_pest_coordinates(
        cls,
        pest: SmoothPestCoordinates | Mapping[str, object] | str | Path,
        *,
        JR: np.ndarray,
        JZ: np.ndarray,
        JPhi: np.ndarray,
        Jx: np.ndarray | None = None,
        Jy: np.ndarray | None = None,
        Jz: np.ndarray | None = None,
        Jtheta: np.ndarray | None = None,
        Jphi: np.ndarray | None = None,
        nfp: int | None = None,
        source: str | None = None,
    ) -> "GriddedPestVectorField":
        coords = _as_pest_coordinates(pest)
        return cls(
            JR=np.asarray(JR, dtype=np.float64),
            JZ=np.asarray(JZ, dtype=np.float64),
            JPhi=np.asarray(JPhi, dtype=np.float64),
            Jx=None if Jx is None else np.asarray(Jx, dtype=np.float64),
            Jy=None if Jy is None else np.asarray(Jy, dtype=np.float64),
            Jz=None if Jz is None else np.asarray(Jz, dtype=np.float64),
            Jtheta=None if Jtheta is None else np.asarray(Jtheta, dtype=np.float64),
            Jphi=None if Jphi is None else np.asarray(Jphi, dtype=np.float64),
            phi_vals=np.asarray(coords.phi_vals, dtype=np.float64),
            theta_vals=np.asarray(coords.theta_vals, dtype=np.float64),
            nfp=int(getattr(coords, "nfp", 1) if nfp is None else nfp),
            phi_period=float(getattr(coords, "period", TWOPI) or TWOPI),
            source=source,
        )

    def _surface_component(self, values: np.ndarray, surface_index: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 3:
            raise ValueError("gridded PEST vector components must have shape (n_phi, n_rho, n_theta)")
        if arr.shape[0] != self.phi_vals.size or arr.shape[2] != self.theta_vals.size:
            raise ValueError("gridded PEST vector component shapes do not match phi/theta grids")
        ir = int(surface_index) % arr.shape[1]
        phi0 = float(self.phi_vals[0]) if self.phi_vals.size else 0.0
        theta0 = float(self.theta_vals[0]) if self.theta_vals.size else 0.0
        return _periodic_surface_bilinear(
            arr[:, ir, :],
            phi,
            theta,
            phi0=phi0,
            theta0=theta0,
            phi_period=float(self.phi_period),
        )

    def evaluate_pest_surface(
        self,
        surface_index: int,
        theta: np.ndarray,
        phi: np.ndarray,
        *,
        R: np.ndarray | None = None,
        Z: np.ndarray | None = None,
    ) -> np.ndarray:
        return np.stack(
            [
                self._surface_component(self.JR, surface_index, theta, phi),
                self._surface_component(self.JZ, surface_index, theta, phi),
                self._surface_component(self.JPhi, surface_index, theta, phi),
            ],
            axis=-1,
        )

    def evaluate_pest_surface_cartesian(
        self,
        surface_index: int,
        theta: np.ndarray,
        phi: np.ndarray,
        *,
        R: np.ndarray | None = None,
        Z: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.Jx is not None and self.Jy is not None and self.Jz is not None:
            return np.stack(
                [
                    self._surface_component(self.Jx, surface_index, theta, phi),
                    self._surface_component(self.Jy, surface_index, theta, phi),
                    self._surface_component(self.Jz, surface_index, theta, phi),
                ],
                axis=-1,
            )
        cyl = self.evaluate_pest_surface(surface_index, theta, phi, R=R, Z=Z)
        cp = np.cos(phi)
        sp = np.sin(phi)
        return np.stack(
            [
                cyl[..., 0] * cp - cyl[..., 2] * sp,
                cyl[..., 0] * sp + cyl[..., 2] * cp,
                cyl[..., 1],
            ],
            axis=-1,
        )

    def evaluate_pest_surface_tangent_components(
        self,
        surface_index: int,
        theta: np.ndarray,
        phi: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if self.Jtheta is None or self.Jphi is None:
            return None
        return (
            self._surface_component(self.Jtheta, surface_index, theta, phi),
            self._surface_component(self.Jphi, surface_index, theta, phi),
        )


@dataclass(frozen=True)
class VmecCurrentFourier:
    """VMEC full-mesh current harmonics for streamline diagnostics.

    VMEC stores ``currumnc/currvmnc`` on the full radial mesh. In VMEC and
    VMEC++ workflows these arrays are used as the Jacobian-weighted
    contravariant current components in VMEC ``(u, v)`` angles; the common
    Jacobian factor does not affect streamline direction.
    """

    s: np.ndarray
    xm: np.ndarray
    xn: np.ndarray
    sqrtgJ_u_cos: np.ndarray
    sqrtgJ_v_cos: np.ndarray
    sqrtgJ_u_sin: np.ndarray | None = None
    sqrtgJ_v_sin: np.ndarray | None = None
    nfp: int = 1
    source: str | None = None

    @classmethod
    def from_wout(cls, path: str | Path) -> "VmecCurrentFourier":
        """Load VMEC ``currumn*``/``currvmn*`` current harmonics from a wout."""

        from netCDF4 import Dataset

        wout = Path(path).expanduser()
        with Dataset(str(wout), "r") as ds:
            variables = ds.variables
            missing = [name for name in ("currumnc", "currvmnc") if name not in variables]
            if missing:
                raise KeyError(f"VMEC wout lacks current harmonics: {', '.join(missing)}")
            ns = int(np.asarray(variables["ns"][...]).item()) if "ns" in variables else int(variables["currumnc"].shape[0])
            nfp = int(np.asarray(variables["nfp"][...]).item()) if "nfp" in variables else 1
            xm_key = "xm_nyq" if "xm_nyq" in variables else "xm"
            xn_key = "xn_nyq" if "xn_nyq" in variables else "xn"
            xm = _netcdf_array(variables[xm_key])
            xn = _netcdf_array(variables[xn_key])
            s = np.linspace(0.0, 1.0, ns, dtype=np.float64)
            cos_u = _vmec_mode_array(_netcdf_array(variables["currumnc"]), s_size=s.size, mode_size=xm.size)
            cos_v = _vmec_mode_array(_netcdf_array(variables["currvmnc"]), s_size=s.size, mode_size=xm.size)
            sin_u = (
                _vmec_mode_array(_netcdf_array(variables["currumns"]), s_size=s.size, mode_size=xm.size)
                if "currumns" in variables
                else None
            )
            sin_v = (
                _vmec_mode_array(_netcdf_array(variables["currvmns"]), s_size=s.size, mode_size=xm.size)
                if "currvmns" in variables
                else None
            )
        return cls(
            s=s,
            xm=xm,
            xn=xn,
            sqrtgJ_u_cos=cos_u,
            sqrtgJ_v_cos=cos_v,
            sqrtgJ_u_sin=sin_u,
            sqrtgJ_v_sin=sin_v,
            nfp=nfp,
            source=str(wout),
        )

    def evaluate(self, rho: np.ndarray, theta_vmec: np.ndarray, zeta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate VMEC current harmonics at ``(rho, theta_vmec, zeta)``."""

        rho_arr, theta_arr, zeta_arr = np.broadcast_arrays(
            np.asarray(rho, dtype=np.float64),
            np.asarray(theta_vmec, dtype=np.float64),
            np.asarray(zeta, dtype=np.float64),
        )
        shape = rho_arr.shape
        s_eval = np.clip(np.ravel(rho_arr) ** 2, float(self.s[0]), float(self.s[-1]))
        theta_flat = np.ravel(theta_arr)
        zeta_flat = np.ravel(zeta_arr)
        ju = _evaluate_vmec_current_modes(
            self.s,
            self.xm,
            self.xn,
            self.sqrtgJ_u_cos,
            self.sqrtgJ_u_sin,
            s_eval,
            theta_flat,
            zeta_flat,
        )
        jv = _evaluate_vmec_current_modes(
            self.s,
            self.xm,
            self.xn,
            self.sqrtgJ_v_cos,
            self.sqrtgJ_v_sin,
            s_eval,
            theta_flat,
            zeta_flat,
        )
        return ju.reshape(shape), jv.reshape(shape)


def pest_tangent_components_to_cylindrical(
    pest: SmoothPestCoordinates | Mapping[str, object] | str | Path,
    *,
    Jtheta: np.ndarray,
    Jphi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert PEST-surface tangent components to physical cylindrical components."""

    coords = _as_pest_coordinates(pest)
    deriv = smooth_pest_derivatives(coords)
    R = np.asarray(coords.R_surf, dtype=np.float64)
    dR_dtheta = np.asarray(deriv[2], dtype=np.float64)
    dZ_dtheta = np.asarray(deriv[3], dtype=np.float64)
    dR_dphi = np.asarray(deriv[4], dtype=np.float64)
    dZ_dphi = np.asarray(deriv[5], dtype=np.float64)
    jt = np.asarray(Jtheta, dtype=np.float64)
    jp = np.asarray(Jphi, dtype=np.float64)
    JR = jt * dR_dtheta + jp * dR_dphi
    JPhi = jp * R
    JZ = jt * dZ_dtheta + jp * dZ_dphi
    return JR, JPhi, JZ


def vmec_current_fourier_to_pest_field(
    pest: SmoothPestCoordinates | Mapping[str, object] | str | Path,
    current: VmecCurrentFourier,
    *,
    theta_vmec: np.ndarray,
    zeta: np.ndarray,
    theta_pest_t: np.ndarray | None = None,
    theta_pest_z: np.ndarray | None = None,
    vmec_to_desc_theta_sign: float = 1.0,
    source: str | None = None,
) -> GriddedPestVectorField:
    """Sample VMEC current harmonics as a :class:`GriddedPestVectorField`.

    ``theta_pest_t`` and ``theta_pest_z`` are derivatives of PEST theta with
    respect to the coordinate used by the PEST geometry backend. For DESC
    ``VMECIO.load`` geometry, pass ``vmec_to_desc_theta_sign=-1`` because DESC
    flips the VMEC poloidal angle orientation.
    """

    coords = _as_pest_coordinates(pest)
    shape = np.asarray(coords.R_surf).shape
    rho = np.broadcast_to(np.asarray(coords.rho_vals, dtype=np.float64)[None, :, None], shape)
    theta_arr = np.asarray(theta_vmec, dtype=np.float64)
    zeta_arr = np.asarray(zeta, dtype=np.float64)
    if theta_arr.shape != shape or zeta_arr.shape != shape:
        raise ValueError("theta_vmec and zeta must match the PEST surface shape")
    ju_vmec, jv = current.evaluate(rho, theta_arr, zeta_arr)
    theta_t = np.ones(shape, dtype=np.float64) if theta_pest_t is None else np.asarray(theta_pest_t, dtype=np.float64)
    theta_z = np.zeros(shape, dtype=np.float64) if theta_pest_z is None else np.asarray(theta_pest_z, dtype=np.float64)
    if theta_t.shape != shape or theta_z.shape != shape:
        raise ValueError("theta_pest_t and theta_pest_z must match the PEST surface shape")
    ju_backend = float(vmec_to_desc_theta_sign) * ju_vmec
    Jtheta = theta_t * ju_backend + theta_z * jv
    Jphi = jv
    JR, JPhi, JZ = pest_tangent_components_to_cylindrical(coords, Jtheta=Jtheta, Jphi=Jphi)
    phi = np.asarray(coords.phi_vals, dtype=np.float64)[:, None, None]
    cp = np.cos(phi)
    sp = np.sin(phi)
    Jx = JR * cp - JPhi * sp
    Jy = JR * sp + JPhi * cp
    return GriddedPestVectorField.from_pest_coordinates(
        coords,
        JR=np.ascontiguousarray(JR),
        JZ=np.ascontiguousarray(JZ),
        JPhi=np.ascontiguousarray(JPhi),
        Jx=np.ascontiguousarray(Jx),
        Jy=np.ascontiguousarray(Jy),
        Jz=np.ascontiguousarray(JZ),
        Jtheta=np.ascontiguousarray(Jtheta),
        Jphi=np.ascontiguousarray(Jphi),
        nfp=current.nfp,
        source=source or f"VMEC current harmonics from {current.source or 'wout'}",
    )


def _netcdf_array(var) -> np.ndarray:
    return np.asarray(np.ma.filled(var[...], 0.0), dtype=np.float64)


def _vmec_mode_array(values: np.ndarray, *, s_size: int, mode_size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape == (s_size, mode_size):
        return np.ascontiguousarray(arr)
    if arr.shape == (mode_size, s_size):
        return np.ascontiguousarray(arr.T)
    raise ValueError(f"VMEC mode array shape {arr.shape} is not compatible with {(s_size, mode_size)}")


def _evaluate_vmec_current_modes(
    s_grid: np.ndarray,
    xm: np.ndarray,
    xn: np.ndarray,
    cos_coeffs: np.ndarray,
    sin_coeffs: np.ndarray | None,
    s_eval: np.ndarray,
    theta: np.ndarray,
    zeta: np.ndarray,
) -> np.ndarray:
    out = np.zeros_like(s_eval, dtype=np.float64)
    for mode_idx, (m, n) in enumerate(zip(np.asarray(xm, dtype=np.float64), np.asarray(xn, dtype=np.float64))):
        angle = float(m) * theta - float(n) * zeta
        out += np.interp(s_eval, s_grid, cos_coeffs[:, mode_idx]) * np.cos(angle)
        if sin_coeffs is not None:
            out += np.interp(s_eval, s_grid, sin_coeffs[:, mode_idx]) * np.sin(angle)
    return out


@dataclass(frozen=True)
class _PeriodicVectorFieldEvaluator:
    R: np.ndarray
    Z: np.ndarray
    Phi: np.ndarray
    nfp: int
    field_period_rad: float
    interp_R: RegularGridInterpolator
    interp_Z: RegularGridInterpolator
    interp_Phi: RegularGridInterpolator

    @classmethod
    def from_field(cls, field: VectorFieldCylind) -> "_PeriodicVectorFieldEvaluator":
        arrays = field.cyna_arrays(extend_phi=True)
        R = np.asarray(arrays.R_grid, dtype=np.float64)
        Z = np.asarray(arrays.Z_grid, dtype=np.float64)
        Phi = np.asarray(arrays.Phi_grid, dtype=np.float64)
        nfp = max(int(getattr(field, "nfp", arrays.nfp)), 1)
        field_period_rad = float(TWOPI / nfp)
        kw = dict(method="linear", bounds_error=False, fill_value=np.nan)
        axes = (R, Z, Phi)
        return cls(
            R=R,
            Z=Z,
            Phi=Phi,
            nfp=nfp,
            field_period_rad=field_period_rad,
            interp_R=RegularGridInterpolator(axes, np.asarray(arrays.BR, dtype=np.float64), **kw),
            interp_Z=RegularGridInterpolator(axes, np.asarray(arrays.BZ, dtype=np.float64), **kw),
            interp_Phi=RegularGridInterpolator(axes, np.asarray(arrays.BPhi, dtype=np.float64), **kw),
        )

    def __call__(self, R: np.ndarray, Z: np.ndarray, phi: np.ndarray) -> np.ndarray:
        phi0 = float(self.Phi[0]) if self.Phi.size else 0.0
        phi_eval = phi0 + np.mod(np.asarray(phi, dtype=np.float64) - phi0, self.field_period_rad)
        pts = np.stack([R, Z, phi_eval], axis=-1)
        return np.stack(
            [
                self.interp_R(pts),
                self.interp_Z(pts),
                self.interp_Phi(pts),
            ],
            axis=-1,
        )


@dataclass(frozen=True)
class _CartesianVectorFieldEvaluator:
    field: VectorFieldCartesian
    nfp: int = 1
    field_period_rad: float = TWOPI

    def evaluate_cartesian(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        pts = np.stack([x, y, z], axis=-1)
        return np.asarray(self.field(pts), dtype=np.float64)

    def __call__(self, R: np.ndarray, Z: np.ndarray, phi: np.ndarray) -> np.ndarray:
        cp = np.cos(phi)
        sp = np.sin(phi)
        values = self.evaluate_cartesian(R * cp, R * sp, Z)
        vx = values[..., 0]
        vy = values[..., 1]
        vz = values[..., 2]
        return np.stack([vx * cp + vy * sp, vz, -vx * sp + vy * cp], axis=-1)


class _PestSurfaceFieldProtocol(Protocol):
    nfp: int
    field_period_rad: float

    def evaluate_pest_surface(
        self,
        surface_index: int,
        theta: np.ndarray,
        phi: np.ndarray,
        *,
        R: np.ndarray | None = None,
        Z: np.ndarray | None = None,
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class _CallableVectorFieldEvaluator:
    fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    nfp: int = 1
    field_period_rad: float = TWOPI

    def __call__(self, R: np.ndarray, Z: np.ndarray, phi: np.ndarray) -> np.ndarray:
        values = np.asarray(self.fn(R, Z, phi), dtype=np.float64)
        if values.shape[-1:] != (3,):
            raise ValueError("callable vector-field evaluator must return an array with final dimension 3")
        return values


@dataclass(frozen=True)
class _PestSurfaceFieldAdapter:
    field: _PestSurfaceFieldProtocol
    nfp: int
    field_period_rad: float

    @classmethod
    def from_field(cls, field: _PestSurfaceFieldProtocol) -> "_PestSurfaceFieldAdapter":
        nfp = max(int(getattr(field, "nfp", 1)), 1)
        return cls(
            field=field,
            nfp=nfp,
            field_period_rad=float(getattr(field, "field_period_rad", TWOPI / nfp)),
        )

    def __call__(self, R: np.ndarray, Z: np.ndarray, phi: np.ndarray) -> np.ndarray:
        if not callable(self.field):
            raise TypeError("surface-native field evaluator cannot be sampled away from a PEST surface")
        values = np.asarray(self.field(R, Z, phi), dtype=np.float64)
        if values.shape[-1:] != (3,):
            raise ValueError("surface-native callable fallback must return an array with final dimension 3")
        return values

    def evaluate_pest_surface(
        self,
        surface_index: int,
        theta: np.ndarray,
        phi: np.ndarray,
        *,
        R: np.ndarray | None = None,
        Z: np.ndarray | None = None,
    ) -> np.ndarray:
        values = np.asarray(
            self.field.evaluate_pest_surface(int(surface_index), theta, phi, R=R, Z=Z),
            dtype=np.float64,
        )
        if values.shape[-1:] != (3,):
            raise ValueError("evaluate_pest_surface must return an array with final dimension 3")
        return values

    def evaluate_pest_surface_cartesian(
        self,
        surface_index: int,
        theta: np.ndarray,
        phi: np.ndarray,
        *,
        R: np.ndarray | None = None,
        Z: np.ndarray | None = None,
    ) -> np.ndarray:
        if hasattr(self.field, "evaluate_pest_surface_cartesian"):
            values = np.asarray(
                self.field.evaluate_pest_surface_cartesian(int(surface_index), theta, phi, R=R, Z=Z),
                dtype=np.float64,
            )
        else:
            cyl = self.evaluate_pest_surface(int(surface_index), theta, phi, R=R, Z=Z)
            cp = np.cos(phi)
            sp = np.sin(phi)
            values = np.stack(
                [
                    cyl[..., 0] * cp - cyl[..., 2] * sp,
                    cyl[..., 0] * sp + cyl[..., 2] * cp,
                    cyl[..., 1],
                ],
                axis=-1,
            )
        if values.shape[-1:] != (3,):
            raise ValueError("evaluate_pest_surface_cartesian must return an array with final dimension 3")
        return values

    def evaluate_pest_surface_tangent_components(
        self,
        surface_index: int,
        theta: np.ndarray,
        phi: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if not hasattr(self.field, "evaluate_pest_surface_tangent_components"):
            return None
        return self.field.evaluate_pest_surface_tangent_components(int(surface_index), theta, phi)


def _as_vector_field_evaluator(field: object):
    if isinstance(field, VectorFieldCylind):
        return _PeriodicVectorFieldEvaluator.from_field(field)
    if isinstance(field, VectorFieldCartesian):
        return _CartesianVectorFieldEvaluator(field)
    if hasattr(field, "evaluate_pest_surface"):
        return _PestSurfaceFieldAdapter.from_field(field)  # type: ignore[arg-type]
    if callable(field):
        nfp = max(int(getattr(field, "nfp", 1)), 1)
        return _CallableVectorFieldEvaluator(
            field,  # type: ignore[arg-type]
            nfp=nfp,
            field_period_rad=float(getattr(field, "field_period_rad", TWOPI / nfp)),
        )
    raise TypeError("field must be a VectorFieldCylind or a compatible vector-field evaluator")


def _pest_from_mapping(pest: Mapping[str, object], *, source: str | None = None) -> SmoothPestCoordinates:
    radial = pest.get("rho_vals", pest.get("radial_labels", pest.get("r_vals")))
    if radial is None:
        raise KeyError("PEST coordinates require rho_vals, radial_labels, or r_vals")
    axis_R = pest.get("axis_R", pest.get("R_AX"))
    axis_Z = pest.get("axis_Z", pest.get("Z_AX"))
    src = source if source is not None else str(pest.get("source", "")) or None
    nfp_value = pest.get("nfp", pest.get("field_periods", 1))
    period_value = pest.get("toroidal_period", pest.get("phi_period", pest.get("period")))
    return SmoothPestCoordinates(
        R_surf=np.asarray(pest["R_surf"], dtype=np.float64),
        Z_surf=np.asarray(pest["Z_surf"], dtype=np.float64),
        rho_vals=np.asarray(radial, dtype=np.float64),
        theta_vals=np.asarray(pest["theta_vals"], dtype=np.float64),
        phi_vals=np.asarray(pest["phi_vals"], dtype=np.float64),
        axis_R=np.asarray(axis_R, dtype=np.float64) if axis_R is not None else None,
        axis_Z=np.asarray(axis_Z, dtype=np.float64) if axis_Z is not None else None,
        source=src,
        nfp=int(np.asarray(nfp_value).item()),
        toroidal_period=None if period_value is None else float(np.asarray(period_value).item()),
    )


def _as_pest_coordinates(pest: SmoothPestCoordinates | Mapping[str, object] | str | Path) -> SmoothPestCoordinates:
    if isinstance(pest, SmoothPestCoordinates):
        return pest
    if isinstance(pest, (str, Path)):
        path = Path(pest).expanduser()
        with np.load(path, allow_pickle=False) as data:
            payload = {name: data[name] for name in data.files}
        return _pest_from_mapping(payload, source=str(path))
    if isinstance(pest, Mapping):
        return _pest_from_mapping(pest)
    raise TypeError("pest must be SmoothPestCoordinates, mapping, or path")


def _validate_pest(pest: SmoothPestCoordinates) -> None:
    R = np.asarray(pest.R_surf, dtype=np.float64)
    Z = np.asarray(pest.Z_surf, dtype=np.float64)
    if R.shape != Z.shape or R.ndim != 3:
        raise ValueError("PEST R_surf and Z_surf must have shape (n_phi, n_rho, n_theta)")
    if np.asarray(pest.phi_vals).ndim != 1 or np.asarray(pest.phi_vals).size != R.shape[0]:
        raise ValueError("PEST phi_vals must be one-dimensional and match R_surf axis 0")
    if np.asarray(pest.rho_vals).ndim != 1 or np.asarray(pest.rho_vals).size != R.shape[1]:
        raise ValueError("PEST rho_vals must be one-dimensional and match R_surf axis 1")
    if np.asarray(pest.theta_vals).ndim != 1 or np.asarray(pest.theta_vals).size != R.shape[2]:
        raise ValueError("PEST theta_vals must be one-dimensional and match R_surf axis 2")


def _normalize_indices(index: int | Sequence[int] | None, size: int, *, default: Sequence[int]) -> np.ndarray:
    raw = np.asarray(default if index is None else ([index] if np.isscalar(index) else list(index)), dtype=np.int64)
    if raw.size == 0:
        raise ValueError("index selection must not be empty")
    return np.asarray([int(i) % int(size) for i in raw], dtype=np.int64)


def _normalize_phi_range(phi_range: Sequence[float] | None, *, period: float) -> tuple[float, float, float] | None:
    if phi_range is None:
        return None
    values = tuple(float(x) for x in phi_range)
    if len(values) != 2:
        raise ValueError("phi_range must contain exactly two angles")
    start, end = values
    if not (np.isfinite(start) and np.isfinite(end)):
        raise ValueError("phi_range angles must be finite")
    phi_period = float(period)
    if not np.isfinite(phi_period) or phi_period <= 0.0:
        phi_period = TWOPI
    raw_span = end - start
    span = float(np.mod(raw_span, phi_period))
    if span <= 1.0e-14 and abs(raw_span) > 1.0e-14:
        span = phi_period
    if span <= 1.0e-14:
        raise ValueError("phi_range must have positive angular span")
    return start, start + span, span


def _phi_in_range(phi: np.ndarray, normalized_phi_range: tuple[float, float, float] | None, *, period: float) -> np.ndarray:
    phi_arr = np.asarray(phi, dtype=np.float64)
    if normalized_phi_range is None:
        return np.ones(phi_arr.shape, dtype=bool)
    start, _end, span = normalized_phi_range
    phi_period = float(period)
    relative = np.mod(phi_arr - float(start), phi_period)
    return np.isfinite(phi_arr) & (relative <= float(span) + 1.0e-12)


def _nearest_phi_indices(phi_vals: np.ndarray, phi_values: np.ndarray, *, period: float) -> np.ndarray:
    grid = np.asarray(phi_vals, dtype=np.float64)
    values = np.asarray(phi_values, dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0:
        return np.full(values.shape, -1, dtype=np.int64)
    phi_period = float(period)
    diff = np.abs(
        np.mod(grid[None, :] - values[:, None] + 0.5 * phi_period, phi_period)
        - 0.5 * phi_period
    )
    return np.asarray(np.nanargmin(diff, axis=1), dtype=np.int64)


def _wrap_angle_near_reference(angle: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return reference + np.mod(angle - reference + np.pi, TWOPI) - np.pi


def _nan_stat(values: list[np.ndarray], *, percentile: float | None = None) -> float:
    if not values:
        return float("nan")
    arr = np.concatenate([np.ravel(np.asarray(v, dtype=np.float64)) for v in values])
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    if percentile is None:
        return float(np.nanmedian(finite))
    return float(np.nanpercentile(finite, float(percentile)))


def _periodic_surface_bilinear(
    values: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    *,
    phi0: float,
    theta0: float,
    phi_period: float,
) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim != 2:
        raise ValueError("surface values must have shape (n_phi, n_theta)")
    n_phi, n_theta = vals.shape
    if n_phi < 2 or n_theta < 2:
        raise ValueError("surface interpolation requires at least two phi and theta points")
    phi_arr, theta_arr = np.broadcast_arrays(
        np.asarray(phi, dtype=np.float64),
        np.asarray(theta, dtype=np.float64),
    )
    u_phi = np.mod(phi_arr - float(phi0), float(phi_period)) * (float(n_phi) / float(phi_period))
    f_phi = np.floor(u_phi)
    i0 = np.asarray(f_phi, dtype=np.int64) % n_phi
    i1 = (i0 + 1) % n_phi
    a = u_phi - f_phi

    u_theta = np.mod(theta_arr - float(theta0), TWOPI) * (float(n_theta) / TWOPI)
    f_theta = np.floor(u_theta)
    j0 = np.asarray(f_theta, dtype=np.int64) % n_theta
    j1 = (j0 + 1) % n_theta
    b = u_theta - f_theta

    return (
        (1.0 - a) * (1.0 - b) * vals[i0, j0]
        + a * (1.0 - b) * vals[i1, j0]
        + (1.0 - a) * b * vals[i0, j1]
        + a * b * vals[i1, j1]
    )


def _surface_ring_at_phi(
    values: np.ndarray,
    phi: float,
    *,
    phi0: float,
    phi_period: float,
) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim != 2:
        raise ValueError("surface ring values must have shape (n_phi, n_theta)")
    n_phi = vals.shape[0]
    if n_phi < 2:
        return vals[0].copy()
    u_phi = np.mod(float(phi) - float(phi0), float(phi_period)) * (float(n_phi) / float(phi_period))
    f_phi = np.floor(u_phi)
    i0 = int(f_phi) % n_phi
    i1 = (i0 + 1) % n_phi
    a = float(u_phi - f_phi)
    return (1.0 - a) * vals[i0] + a * vals[i1]


def _nearest_periodic_ring_theta(
    R_ring: np.ndarray,
    Z_ring: np.ndarray,
    theta_vals: np.ndarray,
    R_point: float,
    Z_point: float,
) -> tuple[float, float]:
    R = np.asarray(R_ring, dtype=np.float64)
    Z = np.asarray(Z_ring, dtype=np.float64)
    theta = np.asarray(theta_vals, dtype=np.float64)
    if R.ndim != 1 or Z.ndim != 1 or theta.ndim != 1 or R.size != Z.size or R.size != theta.size:
        raise ValueError("ring arrays must be one-dimensional with matching lengths")
    if R.size < 3 or not (np.isfinite(R_point) and np.isfinite(Z_point)):
        return float("nan"), float("nan")
    R1 = np.roll(R, -1)
    Z1 = np.roll(Z, -1)
    dR = R1 - R
    dZ = Z1 - Z
    seg2 = dR * dR + dZ * dZ
    valid = np.isfinite(R) & np.isfinite(Z) & np.isfinite(R1) & np.isfinite(Z1) & (seg2 > 0.0)
    if not np.any(valid):
        return float("nan"), float("nan")
    t = np.zeros_like(R, dtype=np.float64)
    t[valid] = np.clip(((float(R_point) - R[valid]) * dR[valid] + (float(Z_point) - Z[valid]) * dZ[valid]) / seg2[valid], 0.0, 1.0)
    closest_R = R + t * dR
    closest_Z = Z + t * dZ
    dist2 = (closest_R - float(R_point)) ** 2 + (closest_Z - float(Z_point)) ** 2
    dist2[~valid] = np.inf
    idx = int(np.nanargmin(dist2))
    if not np.isfinite(dist2[idx]):
        return float("nan"), float("nan")
    n_theta = theta.size
    j1 = (idx + 1) % n_theta
    dtheta = float(np.mod(theta[j1] - theta[idx], TWOPI))
    if dtheta <= 0.0:
        dtheta = TWOPI / float(n_theta)
    theta_point = float(theta[idx] + t[idx] * dtheta)
    return theta_point, float(np.sqrt(dist2[idx]))


def _default_surface_projection_distance(coords: SmoothPestCoordinates, surface_index: int) -> float:
    ir = int(surface_index) % coords.R_surf.shape[1]
    R = np.asarray(coords.R_surf[:, ir, :], dtype=np.float64)
    Z = np.asarray(coords.Z_surf[:, ir, :], dtype=np.float64)
    dR = np.diff(np.concatenate([R, R[:, :1]], axis=1), axis=1)
    dZ = np.diff(np.concatenate([Z, Z[:, :1]], axis=1), axis=1)
    step = np.sqrt(dR * dR + dZ * dZ)
    finite = step[np.isfinite(step) & (step > 0.0)]
    if finite.size:
        return float(max(2.5 * np.nanmedian(finite), 1.0e-9))
    span = max(float(np.nanmax(R) - np.nanmin(R)), float(np.nanmax(Z) - np.nanmin(Z)), 1.0e-12)
    return float(0.03 * span)


def _cartesian_line_theta_on_seed_surface(
    coords: SmoothPestCoordinates,
    *,
    surface_index: int,
    R_line: np.ndarray,
    Z_line: np.ndarray,
    phi_line: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ir = int(surface_index) % coords.R_surf.shape[1]
    R_arr = np.asarray(R_line, dtype=np.float64)
    Z_arr = np.asarray(Z_line, dtype=np.float64)
    phi_arr = np.asarray(phi_line, dtype=np.float64)
    theta_out = np.full(R_arr.shape, np.nan, dtype=np.float64)
    dist_out = np.full(R_arr.shape, np.nan, dtype=np.float64)
    phi0 = float(coords.phi_vals[0]) if np.asarray(coords.phi_vals).size else 0.0
    phi_period = float(getattr(coords, "period", TWOPI) or TWOPI)
    theta_vals = np.asarray(coords.theta_vals, dtype=np.float64)
    R_surface = np.asarray(coords.R_surf[:, ir, :], dtype=np.float64)
    Z_surface = np.asarray(coords.Z_surf[:, ir, :], dtype=np.float64)
    for idx in range(R_arr.size):
        if not (np.isfinite(R_arr[idx]) and np.isfinite(Z_arr[idx]) and np.isfinite(phi_arr[idx])):
            continue
        R_ring = _surface_ring_at_phi(R_surface, float(phi_arr[idx]), phi0=phi0, phi_period=phi_period)
        Z_ring = _surface_ring_at_phi(Z_surface, float(phi_arr[idx]), phi0=phi0, phi_period=phi_period)
        theta, distance = _nearest_periodic_ring_theta(R_ring, Z_ring, theta_vals, float(R_arr[idx]), float(Z_arr[idx]))
        theta_out[idx] = theta
        dist_out[idx] = distance
    return theta_out, dist_out


def _surface_points_at_theta_phi(
    coords: SmoothPestCoordinates,
    *,
    surface_index: int,
    theta: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ir = int(surface_index) % coords.R_surf.shape[1]
    phi0 = float(coords.phi_vals[0]) if np.asarray(coords.phi_vals).size else 0.0
    theta0 = float(coords.theta_vals[0]) if np.asarray(coords.theta_vals).size else 0.0
    phi_period = float(getattr(coords, "period", TWOPI) or TWOPI)
    R = _periodic_surface_bilinear(
        coords.R_surf[:, ir, :],
        phi,
        theta,
        phi0=phi0,
        theta0=theta0,
        phi_period=phi_period,
    )
    Z = _periodic_surface_bilinear(
        coords.Z_surf[:, ir, :],
        phi,
        theta,
        phi0=phi0,
        theta0=theta0,
        phi_period=phi_period,
    )
    return R, Z


def _project_cartesian_points_to_seed_surface(
    coords: SmoothPestCoordinates,
    surface_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    max_surface_distance: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    R = np.sqrt(x * x + y * y)
    phi = np.arctan2(y, x)
    out_x = np.full_like(x, np.nan, dtype=np.float64)
    out_y = np.full_like(y, np.nan, dtype=np.float64)
    out_z = np.full_like(z, np.nan, dtype=np.float64)
    out_phi = np.full_like(phi, np.nan, dtype=np.float64)
    for ir in np.unique(surface_indices):
        selected = surface_indices == int(ir)
        if not np.any(selected):
            continue
        theta, distance = _cartesian_line_theta_on_seed_surface(
            coords,
            surface_index=int(ir),
            R_line=R[selected],
            Z_line=z[selected],
            phi_line=phi[selected],
        )
        distance_limit = (
            _default_surface_projection_distance(coords, int(ir))
            if max_surface_distance is None
            else float(max_surface_distance)
        )
        good = np.isfinite(theta) & np.isfinite(distance) & (distance <= distance_limit)
        if not np.any(good):
            continue
        idx = np.flatnonzero(selected)[good]
        phi_good = phi[selected][good]
        R_proj, Z_proj = _surface_points_at_theta_phi(
            coords,
            surface_index=int(ir),
            theta=theta[good],
            phi=phi_good,
        )
        good2 = np.isfinite(R_proj) & np.isfinite(Z_proj)
        if not np.any(good2):
            continue
        idx2 = idx[good2]
        phi2 = phi_good[good2]
        out_x[idx2] = R_proj[good2] * np.cos(phi2)
        out_y[idx2] = R_proj[good2] * np.sin(phi2)
        out_z[idx2] = Z_proj[good2]
        out_phi[idx2] = phi2
    return out_x, out_y, out_z, out_phi


@dataclass(frozen=True)
class _PestSurfaceEvaluator:
    R: np.ndarray
    Z: np.ndarray
    dR_dtheta: np.ndarray
    dZ_dtheta: np.ndarray
    dR_dphi: np.ndarray
    dZ_dphi: np.ndarray
    phi0: float
    theta0: float
    phi_period: float

    @classmethod
    def from_pest(
        cls,
        pest: SmoothPestCoordinates,
        surface_index: int,
        *,
        derivatives: tuple[np.ndarray, ...] | None = None,
    ) -> "_PestSurfaceEvaluator":
        deriv = smooth_pest_derivatives(pest) if derivatives is None else derivatives
        if len(deriv) != 6:
            raise ValueError("PEST derivative bundle must contain six arrays")
        ir = int(surface_index) % pest.R_surf.shape[1]
        phi_vals = np.asarray(pest.phi_vals, dtype=np.float64)
        theta_vals = np.asarray(pest.theta_vals, dtype=np.float64)
        return cls(
            R=np.asarray(pest.R_surf[:, ir, :], dtype=np.float64),
            Z=np.asarray(pest.Z_surf[:, ir, :], dtype=np.float64),
            dR_dtheta=np.asarray(deriv[2][:, ir, :], dtype=np.float64),
            dZ_dtheta=np.asarray(deriv[3][:, ir, :], dtype=np.float64),
            dR_dphi=np.asarray(deriv[4][:, ir, :], dtype=np.float64),
            dZ_dphi=np.asarray(deriv[5][:, ir, :], dtype=np.float64),
            phi0=float(phi_vals[0]) if phi_vals.size else 0.0,
            theta0=float(theta_vals[0]) if theta_vals.size else 0.0,
            phi_period=float(getattr(pest, "period", TWOPI) or TWOPI),
        )

    def evaluate(self, phi: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, ...]:
        kwargs = dict(
            phi0=self.phi0,
            theta0=self.theta0,
            phi_period=self.phi_period,
        )
        return (
            _periodic_surface_bilinear(self.R, phi, theta, **kwargs),
            _periodic_surface_bilinear(self.Z, phi, theta, **kwargs),
            _periodic_surface_bilinear(self.dR_dtheta, phi, theta, **kwargs),
            _periodic_surface_bilinear(self.dZ_dtheta, phi, theta, **kwargs),
            _periodic_surface_bilinear(self.dR_dphi, phi, theta, **kwargs),
            _periodic_surface_bilinear(self.dZ_dphi, phi, theta, **kwargs),
        )


def _cartesian_rhs(
    field_eval: _PeriodicVectorFieldEvaluator,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    min_field_norm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = np.sqrt(x * x + y * y)
    phi = np.arctan2(y, x)
    values = np.asarray(field_eval(R, z, phi), dtype=np.float64)
    vR = values[..., 0]
    vZ = values[..., 1]
    vPhi = values[..., 2]
    norm = np.sqrt(vR * vR + vZ * vZ + vPhi * vPhi)
    safe = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
        & np.isfinite(vR)
        & np.isfinite(vZ)
        & np.isfinite(vPhi)
        & (norm > float(min_field_norm))
        & (R > 1.0e-12)
    )
    dx = np.full_like(x, np.nan, dtype=np.float64)
    dy = np.full_like(y, np.nan, dtype=np.float64)
    dz = np.full_like(z, np.nan, dtype=np.float64)
    if np.any(safe):
        cp = np.cos(phi[safe])
        sp = np.sin(phi[safe])
        vx = vR[safe] * cp - vPhi[safe] * sp
        vy = vR[safe] * sp + vPhi[safe] * cp
        dx[safe] = vx / norm[safe]
        dy[safe] = vy / norm[safe]
        dz[safe] = vZ[safe] / norm[safe]
    return dx, dy, dz


def _rk4_step_cartesian(
    field_eval: _PeriodicVectorFieldEvaluator,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    phi_ref: np.ndarray,
    *,
    h: float,
    min_field_norm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k1x, k1y, k1z = _cartesian_rhs(field_eval, x, y, z, min_field_norm=min_field_norm)
    k2x, k2y, k2z = _cartesian_rhs(
        field_eval,
        x + 0.5 * h * k1x,
        y + 0.5 * h * k1y,
        z + 0.5 * h * k1z,
        min_field_norm=min_field_norm,
    )
    k3x, k3y, k3z = _cartesian_rhs(
        field_eval,
        x + 0.5 * h * k2x,
        y + 0.5 * h * k2y,
        z + 0.5 * h * k2z,
        min_field_norm=min_field_norm,
    )
    k4x, k4y, k4z = _cartesian_rhs(
        field_eval,
        x + h * k3x,
        y + h * k3y,
        z + h * k3z,
        min_field_norm=min_field_norm,
    )
    out_x = x + h * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
    out_y = y + h * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
    out_z = z + h * (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0
    out_phi = _wrap_angle_near_reference(np.arctan2(out_y, out_x), phi_ref)
    return out_x, out_y, out_z, out_phi


def _cartesian_surface_projected_rhs(
    coords: SmoothPestCoordinates,
    surface_indices: np.ndarray,
    field_eval,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    min_field_norm: float,
    max_surface_distance: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = np.sqrt(x * x + y * y)
    phi = np.arctan2(y, x)
    dx = np.full_like(x, np.nan, dtype=np.float64)
    dy = np.full_like(y, np.nan, dtype=np.float64)
    dz = np.full_like(z, np.nan, dtype=np.float64)
    for ir in np.unique(surface_indices):
        selected = surface_indices == int(ir)
        if not np.any(selected):
            continue
        theta, distance = _cartesian_line_theta_on_seed_surface(
            coords,
            surface_index=int(ir),
            R_line=R[selected],
            Z_line=z[selected],
            phi_line=phi[selected],
        )
        distance_limit = (
            _default_surface_projection_distance(coords, int(ir))
            if max_surface_distance is None
            else float(max_surface_distance)
        )
        local_good = (
            np.isfinite(theta)
            & np.isfinite(distance)
            & (distance <= distance_limit)
            & np.isfinite(R[selected])
            & np.isfinite(z[selected])
            & np.isfinite(phi[selected])
        )
        if not np.any(local_good):
            continue
        global_idx = np.flatnonzero(selected)[local_good]
        phi_good = phi[selected][local_good]
        if hasattr(field_eval, "evaluate_pest_surface_cartesian"):
            values = np.asarray(
                field_eval.evaluate_pest_surface_cartesian(
                    int(ir),
                    theta[local_good],
                    phi_good,
                    R=R[selected][local_good],
                    Z=z[selected][local_good],
                ),
                dtype=np.float64,
            )
            vx = values[..., 0]
            vy = values[..., 1]
            vz = values[..., 2]
        else:
            values = np.asarray(
                field_eval.evaluate_pest_surface(
                    int(ir),
                    theta[local_good],
                    phi_good,
                    R=R[selected][local_good],
                    Z=z[selected][local_good],
                ),
                dtype=np.float64,
            )
            vR = values[..., 0]
            vZ = values[..., 1]
            vPhi = values[..., 2]
            cp_local = np.cos(phi_good)
            sp_local = np.sin(phi_good)
            vx = vR * cp_local - vPhi * sp_local
            vy = vR * sp_local + vPhi * cp_local
            vz = vZ
        norm = np.sqrt(vx * vx + vy * vy + vz * vz)
        good = np.isfinite(vx) & np.isfinite(vy) & np.isfinite(vz) & (norm > float(min_field_norm))
        if not np.any(good):
            continue
        idx = global_idx[good]
        dx[idx] = vx[good] / norm[good]
        dy[idx] = vy[good] / norm[good]
        dz[idx] = vz[good] / norm[good]
    return dx, dy, dz


def _rk4_step_cartesian_surface_projected(
    coords: SmoothPestCoordinates,
    surface_indices: np.ndarray,
    field_eval,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    phi_ref: np.ndarray,
    *,
    h: float,
    min_field_norm: float,
    max_surface_distance: float | None,
    snap_to_surface: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kw = dict(
        coords=coords,
        surface_indices=surface_indices,
        field_eval=field_eval,
        min_field_norm=min_field_norm,
        max_surface_distance=max_surface_distance,
    )
    k1x, k1y, k1z = _cartesian_surface_projected_rhs(x=x, y=y, z=z, **kw)
    k2x, k2y, k2z = _cartesian_surface_projected_rhs(
        x=x + 0.5 * h * k1x,
        y=y + 0.5 * h * k1y,
        z=z + 0.5 * h * k1z,
        **kw,
    )
    k3x, k3y, k3z = _cartesian_surface_projected_rhs(
        x=x + 0.5 * h * k2x,
        y=y + 0.5 * h * k2y,
        z=z + 0.5 * h * k2z,
        **kw,
    )
    k4x, k4y, k4z = _cartesian_surface_projected_rhs(
        x=x + h * k3x,
        y=y + h * k3y,
        z=z + h * k3z,
        **kw,
    )
    out_x = x + h * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
    out_y = y + h * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
    out_z = z + h * (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0
    out_phi = _wrap_angle_near_reference(np.arctan2(out_y, out_x), phi_ref)
    if bool(snap_to_surface):
        out_x, out_y, out_z, snapped_phi = _project_cartesian_points_to_seed_surface(
            coords,
            surface_indices,
            out_x,
            out_y,
            out_z,
            max_surface_distance=max_surface_distance,
        )
        out_phi = _wrap_angle_near_reference(snapped_phi, phi_ref)
    return out_x, out_y, out_z, out_phi


def _pest_surface_rhs_single(
    surface_eval: _PestSurfaceEvaluator,
    field_eval,
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    surface_index: int | None = None,
    min_field_norm: float,
    min_tangent_norm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    R, Z, dR_dtheta, dZ_dtheta, dR_dphi, dZ_dphi = surface_eval.evaluate(phi, theta)
    cp = np.cos(phi)
    sp = np.sin(phi)

    e_theta = np.stack([dR_dtheta * cp, dR_dtheta * sp, dZ_dtheta], axis=-1)
    e_phi = np.stack([dR_dphi * cp - R * sp, dR_dphi * sp + R * cp, dZ_dphi], axis=-1)
    tangent_components = None
    if hasattr(field_eval, "evaluate_pest_surface_tangent_components") and surface_index is not None:
        tangent_components = field_eval.evaluate_pest_surface_tangent_components(int(surface_index), theta, phi)
    if tangent_components is not None:
        a0, b0 = tangent_components
        j_cart = a0[..., None] * e_theta + b0[..., None] * e_phi
    else:
        if hasattr(field_eval, "evaluate_pest_surface") and surface_index is not None:
            values = np.asarray(
                field_eval.evaluate_pest_surface(int(surface_index), theta, phi, R=R, Z=Z),
                dtype=np.float64,
            )
        else:
            values = np.asarray(field_eval(R, Z, phi), dtype=np.float64)
        jR = values[..., 0]
        jZ = values[..., 1]
        jPhi = values[..., 2]
        jx = jR * cp - jPhi * sp
        jy = jR * sp + jPhi * cp
        jz = jZ
        j_cart = np.stack([jx, jy, jz], axis=-1)

    gtt = np.sum(e_theta * e_theta, axis=-1)
    gtp = np.sum(e_theta * e_phi, axis=-1)
    gpp = np.sum(e_phi * e_phi, axis=-1)
    rhs_t = np.sum(j_cart * e_theta, axis=-1)
    rhs_p = np.sum(j_cart * e_phi, axis=-1)
    det = gtt * gpp - gtp * gtp

    j_norm = np.sqrt(np.sum(j_cart * j_cart, axis=-1))
    normal = np.cross(e_theta, e_phi)
    normal_norm = np.sqrt(np.sum(normal * normal, axis=-1))
    normal_unit = normal / np.where(normal_norm > 0.0, normal_norm, np.nan)[..., None]
    normal_leakage = np.abs(np.sum(j_cart * normal_unit, axis=-1)) / j_norm

    dtheta = np.full_like(theta, np.nan, dtype=np.float64)
    dphi = np.full_like(phi, np.nan, dtype=np.float64)
    tangent_fraction = np.full_like(theta, np.nan, dtype=np.float64)
    safe = (
        np.isfinite(theta)
        & np.isfinite(phi)
        & np.isfinite(j_norm)
        & np.isfinite(det)
        & (j_norm > float(min_field_norm))
        & (np.abs(det) > 1.0e-28)
    )
    if np.any(safe):
        a = np.full_like(theta, np.nan, dtype=np.float64)
        b = np.full_like(phi, np.nan, dtype=np.float64)
        a[safe] = (rhs_t[safe] * gpp[safe] - rhs_p[safe] * gtp[safe]) / det[safe]
        b[safe] = (gtt[safe] * rhs_p[safe] - gtp[safe] * rhs_t[safe]) / det[safe]
        tangent = a[..., None] * e_theta + b[..., None] * e_phi
        tangent_norm = np.sqrt(np.sum(tangent * tangent, axis=-1))
        good = safe & np.isfinite(tangent_norm) & (tangent_norm > float(min_tangent_norm))
        dtheta[good] = a[good] / tangent_norm[good]
        dphi[good] = b[good] / tangent_norm[good]
        tangent_fraction[good] = tangent_norm[good] / j_norm[good]
    normal_leakage[~np.isfinite(normal_leakage)] = np.nan
    return dtheta, dphi, normal_leakage, tangent_fraction


def _pest_surface_rhs(
    surface_evals: Mapping[int, _PestSurfaceEvaluator],
    surface_indices: np.ndarray,
    field_eval,
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    min_field_norm: float,
    min_tangent_norm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dtheta = np.full_like(theta, np.nan, dtype=np.float64)
    dphi = np.full_like(phi, np.nan, dtype=np.float64)
    leakage = np.full_like(theta, np.nan, dtype=np.float64)
    tangent_fraction = np.full_like(theta, np.nan, dtype=np.float64)
    for ir, surface_eval in surface_evals.items():
        selected = surface_indices == int(ir)
        if not np.any(selected):
            continue
        dth, dph, leak, tfrac = _pest_surface_rhs_single(
            surface_eval,
            field_eval,
            theta[selected],
            phi[selected],
            surface_index=int(ir),
            min_field_norm=min_field_norm,
            min_tangent_norm=min_tangent_norm,
        )
        dtheta[selected] = dth
        dphi[selected] = dph
        leakage[selected] = leak
        tangent_fraction[selected] = tfrac
    return dtheta, dphi, leakage, tangent_fraction


def _rk4_step_pest_surface(
    surface_evals: Mapping[int, _PestSurfaceEvaluator],
    surface_indices: np.ndarray,
    field_eval,
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    h: float,
    min_field_norm: float,
    min_tangent_norm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k1t, k1p, leakage, tangent_fraction = _pest_surface_rhs(
        surface_evals,
        surface_indices,
        field_eval,
        theta,
        phi,
        min_field_norm=min_field_norm,
        min_tangent_norm=min_tangent_norm,
    )
    k2t, k2p, _, _ = _pest_surface_rhs(
        surface_evals,
        surface_indices,
        field_eval,
        theta + 0.5 * h * k1t,
        phi + 0.5 * h * k1p,
        min_field_norm=min_field_norm,
        min_tangent_norm=min_tangent_norm,
    )
    k3t, k3p, _, _ = _pest_surface_rhs(
        surface_evals,
        surface_indices,
        field_eval,
        theta + 0.5 * h * k2t,
        phi + 0.5 * h * k2p,
        min_field_norm=min_field_norm,
        min_tangent_norm=min_tangent_norm,
    )
    k4t, k4p, _, _ = _pest_surface_rhs(
        surface_evals,
        surface_indices,
        field_eval,
        theta + h * k3t,
        phi + h * k3p,
        min_field_norm=min_field_norm,
        min_tangent_norm=min_tangent_norm,
    )
    out_theta = theta + h * (k1t + 2.0 * k2t + 2.0 * k3t + k4t) / 6.0
    out_phi = phi + h * (k1p + 2.0 * k2p + 2.0 * k3p + k4p) / 6.0
    return out_theta, out_phi, leakage, tangent_fraction


def _surface_points_from_theta_phi(
    surface_evals: Mapping[int, _PestSurfaceEvaluator],
    surface_indices: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    R = np.full_like(theta, np.nan, dtype=np.float64)
    Z = np.full_like(theta, np.nan, dtype=np.float64)
    for ir, surface_eval in surface_evals.items():
        selected = surface_indices == int(ir)
        if not np.any(selected):
            continue
        R[selected], Z[selected], *_ = surface_eval.evaluate(phi[selected], theta[selected])
    return R, Z


def _seed_theta_values(
    pest: SmoothPestCoordinates,
    *,
    surface_index: int,
    phi_value: float,
    seed_count: int,
    theta_offset: float,
    seed_spacing: str,
) -> np.ndarray:
    n_theta = pest.R_surf.shape[2]
    theta_vals = np.asarray(pest.theta_vals, dtype=np.float64)
    if int(seed_count) <= 0:
        raise ValueError("seed_count must be positive")
    spacing = str(seed_spacing).lower()
    if spacing in {"theta", "index", "grid"}:
        theta_base = np.linspace(0, n_theta, int(seed_count), endpoint=False, dtype=np.float64)
        theta_indices = np.mod(
            np.rint(theta_base + float(theta_offset) * n_theta / TWOPI).astype(np.int64),
            n_theta,
        )
        return theta_vals[theta_indices]
    if spacing not in {"arclength", "arc", "perimeter"}:
        raise ValueError("seed_spacing must be 'arclength' or 'theta'")

    ir = int(surface_index) % pest.R_surf.shape[1]
    phi0 = float(pest.phi_vals[0]) if np.asarray(pest.phi_vals).size else 0.0
    phi_period = float(getattr(pest, "period", TWOPI) or TWOPI)
    R_ring = _surface_ring_at_phi(
        np.asarray(pest.R_surf[:, ir, :], dtype=np.float64),
        float(phi_value),
        phi0=phi0,
        phi_period=phi_period,
    )
    Z_ring = _surface_ring_at_phi(
        np.asarray(pest.Z_surf[:, ir, :], dtype=np.float64),
        float(phi_value),
        phi0=phi0,
        phi_period=phi_period,
    )
    finite = np.isfinite(R_ring) & np.isfinite(Z_ring) & np.isfinite(theta_vals)
    if np.count_nonzero(finite) < 4:
        return _seed_theta_values(
            pest,
            surface_index=surface_index,
            phi_value=phi_value,
            seed_count=seed_count,
            theta_offset=theta_offset,
            seed_spacing="theta",
        )
    if not np.all(finite):
        return _seed_theta_values(
            pest,
            surface_index=surface_index,
            phi_value=phi_value,
            seed_count=seed_count,
            theta_offset=theta_offset,
            seed_spacing="theta",
        )
    theta_ext = np.concatenate([theta_vals, [theta_vals[0] + TWOPI]])
    R_ext = np.concatenate([R_ring, [R_ring[0]]])
    Z_ext = np.concatenate([Z_ring, [Z_ring[0]]])
    step = np.sqrt(np.diff(R_ext) ** 2 + np.diff(Z_ext) ** 2)
    if not np.all(np.isfinite(step)):
        return _seed_theta_values(
            pest,
            surface_index=surface_index,
            phi_value=phi_value,
            seed_count=seed_count,
            theta_offset=theta_offset,
            seed_spacing="theta",
        )
    perimeter = float(np.nansum(step))
    if not np.isfinite(perimeter) or perimeter <= 0.0:
        return _seed_theta_values(
            pest,
            surface_index=surface_index,
            phi_value=phi_value,
            seed_count=seed_count,
            theta_offset=theta_offset,
            seed_spacing="theta",
        )
    cumulative = np.concatenate([[0.0], np.cumsum(step)])
    phase = float(np.mod(float(theta_offset), TWOPI) / TWOPI)
    targets = np.mod((np.arange(int(seed_count), dtype=np.float64) / float(seed_count)) + phase, 1.0) * perimeter
    targets.sort()
    return np.interp(targets, cumulative, theta_ext)


def _seed_points(
    pest: SmoothPestCoordinates,
    *,
    surface_indices: np.ndarray,
    phi_indices: np.ndarray,
    phi_values: np.ndarray,
    seed_count: int,
    theta_offset: float,
    seed_spacing: str,
) -> dict[str, np.ndarray]:
    if int(seed_count) <= 0:
        raise ValueError("seed_count must be positive")
    if np.asarray(phi_indices).size != np.asarray(phi_values).size:
        raise ValueError("phi_indices and phi_values must have the same size")
    rows = []
    for iphi, phi_value in zip(phi_indices, phi_values):
        for irho in surface_indices:
            theta_values = _seed_theta_values(
                pest,
                surface_index=int(irho),
                phi_value=float(phi_value),
                seed_count=int(seed_count),
                theta_offset=float(theta_offset),
                seed_spacing=seed_spacing,
            )
            phi_arr = np.full(theta_values.shape, float(phi_value), dtype=np.float64)
            R_seed, Z_seed = _surface_points_at_theta_phi(
                pest,
                surface_index=int(irho),
                theta=theta_values,
                phi=phi_arr,
            )
            for R0, Z0, theta0 in zip(R_seed, Z_seed, theta_values):
                R0 = float(R0)
                Z0 = float(Z0)
                if np.isfinite(R0) and np.isfinite(Z0):
                    rows.append(
                        (
                            R0,
                            Z0,
                            float(phi_value),
                            float(pest.rho_vals[int(irho)]),
                            float(theta0),
                            int(irho),
                            int(iphi),
                        )
                    )
    if not rows:
        raise ValueError("no finite PEST seed points were found")
    arr = np.asarray(rows, dtype=np.float64)
    return {
        "R": arr[:, 0],
        "Z": arr[:, 1],
        "phi": arr[:, 2],
        "rho": arr[:, 3],
        "theta": arr[:, 4],
        "surface_index": arr[:, 5].astype(np.int64),
        "phi_index": arr[:, 6].astype(np.int64),
    }


@dataclass(frozen=True)
class _PestSeedPlan:
    phi_period: float
    phi_range: tuple[float, float, float] | None
    phi_indices: np.ndarray
    phi_values: np.ndarray
    surface_indices: np.ndarray
    full_seeds: dict[str, np.ndarray]
    selected_seed_indices: np.ndarray
    seeds: dict[str, np.ndarray]

    @property
    def full_seed_count(self) -> int:
        return int(self.full_seeds["R"].size)


@dataclass(frozen=True)
class _PestTraceRawDiagnostics:
    seed_leakage: np.ndarray
    seed_tangent_fraction: np.ndarray
    trajectory_leakage: np.ndarray
    trajectory_tangent_fraction: np.ndarray


def _prepare_pest_seed_plan(
    coords: SmoothPestCoordinates,
    *,
    surface_index: int | Sequence[int] | None,
    phi_indices: Sequence[int] | None,
    phi_values: Sequence[float] | None,
    phi_range: Sequence[float] | None,
    phi_seed_count: int | None,
    clip_phi_range: bool,
    seed_count: int,
    seed_spacing: str,
    seed_line_indices: Sequence[int] | None,
    theta_offset: float,
) -> _PestSeedPlan:
    """Build the complete deterministic seed grid and an optional row subset."""

    n_phi, n_rho, _n_theta = coords.R_surf.shape
    phi_period = float(getattr(coords, "period", TWOPI) or TWOPI)
    phi_range_norm = _normalize_phi_range(phi_range, period=phi_period)
    default_phi = [0] if n_phi < 4 else [0, n_phi // 4, n_phi // 2, (3 * n_phi) // 4]
    if phi_indices is not None and phi_values is not None:
        raise ValueError("provide either phi_indices or phi_values, not both")
    if phi_values is not None:
        seed_phi_values = np.asarray(list(phi_values), dtype=np.float64)
        if seed_phi_values.size == 0:
            raise ValueError("phi_values must not be empty")
        if not np.all(np.isfinite(seed_phi_values)):
            raise ValueError("phi_values must be finite")
        if phi_range_norm is not None and bool(clip_phi_range):
            keep = _phi_in_range(seed_phi_values, phi_range_norm, period=phi_period)
            seed_phi_values = seed_phi_values[keep]
            if seed_phi_values.size == 0:
                raise ValueError("no phi_values lie inside phi_range")
        phi_idx = _nearest_phi_indices(
            np.asarray(coords.phi_vals, dtype=np.float64),
            seed_phi_values,
            period=phi_period,
        )
    elif phi_range_norm is not None and phi_indices is None:
        n_phi_seed = max(
            int(phi_seed_count) if phi_seed_count is not None else min(4, n_phi),
            1,
        )
        start, _end, span = phi_range_norm
        seed_phi_values = float(start) + (
            np.arange(n_phi_seed, dtype=np.float64) + 0.5
        ) * float(span) / float(n_phi_seed)
        phi_idx = _nearest_phi_indices(
            np.asarray(coords.phi_vals, dtype=np.float64),
            seed_phi_values,
            period=phi_period,
        )
    else:
        phi_idx = _normalize_indices(phi_indices, n_phi, default=default_phi)
        seed_phi_values = np.asarray(coords.phi_vals, dtype=np.float64)[phi_idx]
        if phi_range_norm is not None and bool(clip_phi_range):
            keep = _phi_in_range(seed_phi_values, phi_range_norm, period=phi_period)
            phi_idx = phi_idx[keep]
            seed_phi_values = seed_phi_values[keep]
            if seed_phi_values.size == 0:
                raise ValueError("no selected phi_indices lie inside phi_range")
    surf_idx = _normalize_indices(surface_index, n_rho, default=[n_rho - 1])
    full_seeds = _seed_points(
        coords,
        surface_indices=surf_idx,
        phi_indices=phi_idx,
        phi_values=seed_phi_values,
        seed_count=int(seed_count),
        theta_offset=float(theta_offset),
        seed_spacing=str(seed_spacing),
    )
    full_seed_count = int(full_seeds["R"].size)
    if seed_line_indices is None:
        selected_seed_indices = np.arange(full_seed_count, dtype=np.int64)
    else:
        raw_seed_indices = np.asarray(list(seed_line_indices))
        if raw_seed_indices.ndim != 1 or raw_seed_indices.size == 0:
            raise ValueError("seed_line_indices must be a non-empty 1-D sequence")
        if raw_seed_indices.dtype.kind not in {"i", "u"}:
            raise ValueError("seed_line_indices must contain only integers")
        selected_seed_indices = raw_seed_indices.astype(np.int64, copy=False)
        if np.any(selected_seed_indices < 0) or np.any(
            selected_seed_indices >= full_seed_count
        ):
            raise ValueError("seed_line_indices must lie inside the complete seed grid")
        if np.unique(selected_seed_indices).size != selected_seed_indices.size:
            raise ValueError("seed_line_indices must be unique")
    seeds = {
        key: np.asarray(values)[selected_seed_indices]
        for key, values in full_seeds.items()
    }
    return _PestSeedPlan(
        phi_period=phi_period,
        phi_range=phi_range_norm,
        phi_indices=phi_idx,
        phi_values=seed_phi_values,
        surface_indices=surf_idx,
        full_seeds=full_seeds,
        selected_seed_indices=selected_seed_indices,
        seeds=seeds,
    )


def _seed_surface_perimeters(
    coords: SmoothPestCoordinates,
    seeds: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Return the physical seed-section perimeter for every seed row.

    The integration step intentionally uses one robust median length, but a
    seed-section return must be normalized by the perimeter of that seed's own
    surface.  Keep the row-resolved values so downstream closure audits do not
    silently substitute the global integration scale for the local geometry.
    """

    seed_phi = np.asarray(seeds["phi"], dtype=np.float64)
    seed_surface = np.asarray(seeds["surface_index"], dtype=np.int64)
    seed_R = np.asarray(seeds["R"], dtype=np.float64)
    if not (
        seed_phi.ndim == seed_surface.ndim == seed_R.ndim == 1
        and seed_phi.size == seed_surface.size == seed_R.size
    ):
        raise ValueError("seed arrays must be aligned one-dimensional rows")

    result = np.full(seed_phi.shape, np.nan, dtype=np.float64)
    theta = np.asarray(coords.theta_vals, dtype=np.float64)
    phi0 = float(coords.phi_vals[0]) if np.asarray(coords.phi_vals).size else 0.0
    phi_period = float(getattr(coords, "period", TWOPI) or TWOPI)
    cache: dict[tuple[int, float], float] = {}
    for row, (phi_value, surface_index) in enumerate(
        zip(seed_phi, seed_surface, strict=True)
    ):
        key = (int(surface_index), float(phi_value))
        perimeter = cache.get(key)
        if perimeter is None:
            irho = int(surface_index) % coords.R_surf.shape[1]
            R_ring = _surface_ring_at_phi(
                np.asarray(coords.R_surf[:, irho, :], dtype=np.float64),
                float(phi_value),
                phi0=phi0,
                phi_period=phi_period,
            )
            Z_ring = _surface_ring_at_phi(
                np.asarray(coords.Z_surf[:, irho, :], dtype=np.float64),
                float(phi_value),
                phi0=phi0,
                phi_period=phi_period,
            )
            finite = np.isfinite(R_ring) & np.isfinite(Z_ring) & np.isfinite(theta)
            if np.count_nonzero(finite) >= 4 and np.all(finite):
                step = np.hypot(
                    np.diff(np.r_[R_ring, R_ring[0]]),
                    np.diff(np.r_[Z_ring, Z_ring[0]]),
                )
                perimeter = float(np.sum(step))
            else:
                perimeter = np.nan
            if not np.isfinite(perimeter) or perimeter <= 0.0:
                raise ValueError(
                    "cannot compute a finite positive perimeter for seed "
                    f"row {row} on surface {int(surface_index)}"
                )
            cache[key] = perimeter
        result[row] = perimeter
    return result


def _trace_j_streamlines_on_pest_serial(
    field: VectorFieldCylind | object,
    pest: SmoothPestCoordinates | Mapping[str, object] | str | Path,
    *,
    surface_index: int | Sequence[int] | None = -1,
    phi_indices: Sequence[int] | None = None,
    phi_values: Sequence[float] | None = None,
    phi_range: Sequence[float] | None = None,
    phi_seed_count: int | None = None,
    clip_phi_range: bool = True,
    seed_count: int = 12,
    seed_spacing: str = "arclength",
    seed_line_indices: Sequence[int] | None = None,
    n_turns: float = 0.2,
    steps_per_turn: int = 512,
    bidirectional: bool = True,
    theta_offset: float = 0.0,
    min_field_norm: float = 1.0e-14,
    min_tangent_norm: float | None = None,
    constrain_to_surface: bool = True,
    max_surface_distance: float | None = None,
    snap_cartesian_to_surface: bool = True,
    _return_diagnostics: bool = False,
) -> PestSeededStreamlines | tuple[PestSeededStreamlines, _PestTraceRawDiagnostics]:
    """Trace current streamlines from PEST-surface seeds.

    ``field`` is usually a :class:`pyna.fields.VectorFieldCylind`; it is treated
    as the full current-density vector in physical cylindrical components
    ``(J_R, J_Z, J_phi)``.  For finite-beta equilibrium backends that can
    evaluate directly on a PEST surface, ``field`` may instead expose
    ``evaluate_pest_surface(surface_index, theta, phi, R=None, Z=None)`` and
    ``nfp`` attributes.  By default, streamlines are constrained to the selected
    PEST surfaces by projecting ``J`` onto the surface tangent plane before
    stepping in ``(theta, phi)``.  Set ``constrain_to_surface=False`` to integrate
    in physical Cartesian space; the PEST mesh is then used only for seeding and
    section projection.

    ``surface_index`` may select one or more magnetic surfaces.  ``seed_count``
    is the number of poloidal seeds per selected surface and toroidal seed
    value.  With the default ``seed_spacing="arclength"``, those poloidal seeds
    are distributed by cross-section arclength instead of raw theta index.
    ``phi_range`` limits the toroidal sector used for automatically generated
    seed values and, by default, stops each line once it leaves that sector.
    ``seed_line_indices`` optionally selects rows from that complete,
    deterministic seed grid after construction.  It is intended for adaptive
    continuation of previously identified seed lines: indices are unique,
    preserve the requested order, and retain the integration step computed
    from the complete seed grid.  ``n_turns`` keeps the historical pyna.plot meaning: it is a multiplier of
    ``steps_per_turn`` in normalized seed-surface arclength, not a literal
    toroidal turn count.
    """

    coords = _as_pest_coordinates(pest)
    _validate_pest(coords)
    field_eval = _as_vector_field_evaluator(field)
    coords_nfp = int(coords.nfp)
    field_nfp = int(field_eval.nfp)
    if coords_nfp != field_nfp:
        raise ValueError(
            "PEST/current field-period mismatch: "
            f"coords.nfp={coords_nfp}, field.nfp={field_nfp}"
        )
    expected_period = TWOPI / float(coords_nfp)
    if not coords.stores_one_field_period or not np.isclose(
        coords.period,
        expected_period,
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise ValueError(
            "PEST coordinates must explicitly store one field period: "
            f"coords.period={coords.period}, expected 2*pi/nfp={expected_period}"
        )
    if not np.isclose(
        field_eval.field_period_rad,
        expected_period,
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise ValueError(
            "current field has inconsistent nfp and field_period_rad: "
            f"nfp={field_nfp}, field_period_rad={field_eval.field_period_rad}, "
            f"expected={expected_period}"
        )
    seed_plan = _prepare_pest_seed_plan(
        coords,
        surface_index=surface_index,
        phi_indices=phi_indices,
        phi_values=phi_values,
        phi_range=phi_range,
        phi_seed_count=phi_seed_count,
        clip_phi_range=bool(clip_phi_range),
        seed_count=int(seed_count),
        seed_spacing=str(seed_spacing),
        seed_line_indices=seed_line_indices,
        theta_offset=float(theta_offset),
    )
    phi_period = seed_plan.phi_period
    phi_range_norm = seed_plan.phi_range
    phi_idx = seed_plan.phi_indices
    seed_phi_values = seed_plan.phi_values
    surf_idx = seed_plan.surface_indices
    full_seeds = seed_plan.full_seeds
    full_seed_count = seed_plan.full_seed_count
    selected_seed_indices = seed_plan.selected_seed_indices
    seeds = seed_plan.seeds
    full_seed_surface_perimeter_m = _seed_surface_perimeters(coords, full_seeds)
    surface_arclength_per_turn = float(
        np.median(full_seed_surface_perimeter_m)
    )

    seed_R = seeds["R"]
    seed_Z = seeds["Z"]
    seed_phi = seeds["phi"]
    n_seed = seed_R.size
    n_steps = max(int(round(float(n_turns) * max(int(steps_per_turn), 1))), 1)
    n_points = 2 * n_steps + 1 if bidirectional else n_steps + 1
    h_base = float(surface_arclength_per_turn) / max(int(steps_per_turn), 1)
    tangent_floor = float(min_field_norm if min_tangent_norm is None else min_tangent_norm)
    trajectory_leakage_samples: list[np.ndarray] = []
    trajectory_tangent_fraction_samples: list[np.ndarray] = []
    # Keep evaluators for the complete requested surface grid, even when
    # ``seed_line_indices`` selects unresolved rows from only a subset of those
    # surfaces.  The integrator uses the selected rows, while the section-level
    # tangent/leakage diagnostics below intentionally retain the complete-grid
    # contract and therefore iterate over ``surf_idx``.
    surface_derivatives = smooth_pest_derivatives(coords)
    surface_evals = {
        int(ir): _PestSurfaceEvaluator.from_pest(
            coords,
            int(ir),
            derivatives=surface_derivatives,
        )
        for ir in surf_idx
    }
    _, _, seed_leakage, seed_tangent_fraction = _pest_surface_rhs(
        surface_evals,
        seeds["surface_index"],
        field_eval,
        np.asarray(seeds["theta"], dtype=np.float64),
        seed_phi,
        min_field_norm=float(min_field_norm),
        min_tangent_norm=tangent_floor,
    )
    section_leakage_samples: list[np.ndarray] = []
    section_tangent_fraction_samples: list[np.ndarray] = []
    theta_grid = np.asarray(coords.theta_vals, dtype=np.float64)
    for ir in surf_idx:
        surface_eval = surface_evals[int(ir)]
        for phi_value in seed_phi_values:
            phi_grid = np.full_like(theta_grid, float(phi_value), dtype=np.float64)
            _, _, leakage, tangent_fraction = _pest_surface_rhs_single(
                surface_eval,
                field_eval,
                theta_grid,
                phi_grid,
                surface_index=int(ir),
                min_field_norm=float(min_field_norm),
                min_tangent_norm=tangent_floor,
            )
            section_leakage_samples.append(leakage)
            section_tangent_fraction_samples.append(tangent_fraction)

    if constrain_to_surface:
        def integrate_surface(direction: float, count: int) -> tuple[np.ndarray, np.ndarray]:
            theta = np.asarray(seeds["theta"], dtype=np.float64).copy()
            phi = seed_phi.copy()
            thetas = [theta.copy()]
            phis = [phi.copy()]
            for _ in range(count):
                theta, phi, leakage, tangent_fraction = _rk4_step_pest_surface(
                    surface_evals,
                    seeds["surface_index"],
                    field_eval,
                    theta,
                    phi,
                    h=float(direction) * h_base,
                    min_field_norm=float(min_field_norm),
                    min_tangent_norm=tangent_floor,
                )
                if phi_range_norm is not None and bool(clip_phi_range):
                    inside = _phi_in_range(phi, phi_range_norm, period=phi_period)
                    theta = np.where(inside, theta, np.nan)
                    phi = np.where(inside, phi, np.nan)
                trajectory_leakage_samples.append(leakage)
                trajectory_tangent_fraction_samples.append(tangent_fraction)
                thetas.append(theta.copy())
                phis.append(phi.copy())
            return np.stack(thetas, axis=1), np.stack(phis, axis=1)

        if bidirectional:
            btheta, bphi = integrate_surface(-1.0, n_steps)
            ftheta, fphi = integrate_surface(1.0, n_steps)
            theta = np.concatenate([btheta[:, :0:-1], ftheta], axis=1)
            phi = np.concatenate([bphi[:, :0:-1], fphi], axis=1)
        else:
            theta, phi = integrate_surface(1.0, n_steps)
        R, z = _surface_points_from_theta_phi(
            surface_evals,
            seeds["surface_index"],
            theta,
            phi,
        )
        xyz = _cylindrical_to_cartesian(R, z, phi)
        x = xyz[..., 0]
        y = xyz[..., 1]
        trace_backend = "pyna.plot.j_streamlines.python_rk4_pest_surface_arclength"
        trace_parameter = "normalized_pest_surface_arclength"
        trace_mode = "pest_surface_constrained"
    else:
        theta = np.full((n_seed, n_points), np.nan, dtype=np.float64)

        use_surface_projected_cartesian = hasattr(field_eval, "evaluate_pest_surface")

        def integrate_cartesian(direction: float, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            x0 = seed_R * np.cos(seed_phi)
            y0 = seed_R * np.sin(seed_phi)
            z0 = seed_Z.copy()
            phi0 = seed_phi.copy()
            xs = [x0.copy()]
            ys = [y0.copy()]
            zs = [z0.copy()]
            phis = [phi0.copy()]
            x_curr, y_curr, z_curr, phi_curr = x0, y0, z0, phi0
            for _ in range(count):
                if use_surface_projected_cartesian:
                    x_curr, y_curr, z_curr, phi_curr = _rk4_step_cartesian_surface_projected(
                        coords,
                        seeds["surface_index"],
                        field_eval,
                        x_curr,
                        y_curr,
                        z_curr,
                        phi_curr,
                        h=float(direction) * h_base,
                        min_field_norm=float(min_field_norm),
                        max_surface_distance=max_surface_distance,
                        snap_to_surface=bool(snap_cartesian_to_surface),
                    )
                else:
                    x_curr, y_curr, z_curr, phi_curr = _rk4_step_cartesian(
                        field_eval,
                        x_curr,
                        y_curr,
                        z_curr,
                        phi_curr,
                        h=float(direction) * h_base,
                        min_field_norm=float(min_field_norm),
                    )
                if phi_range_norm is not None and bool(clip_phi_range):
                    inside = _phi_in_range(phi_curr, phi_range_norm, period=phi_period)
                    x_curr = np.where(inside, x_curr, np.nan)
                    y_curr = np.where(inside, y_curr, np.nan)
                    z_curr = np.where(inside, z_curr, np.nan)
                    phi_curr = np.where(inside, phi_curr, np.nan)
                xs.append(x_curr.copy())
                ys.append(y_curr.copy())
                zs.append(z_curr.copy())
                phis.append(phi_curr.copy())
            return np.stack(xs, axis=1), np.stack(ys, axis=1), np.stack(zs, axis=1), np.stack(phis, axis=1)

        if bidirectional:
            bx, by, bz, bphi = integrate_cartesian(-1.0, n_steps)
            fx, fy, fz, fphi = integrate_cartesian(1.0, n_steps)
            x = np.concatenate([bx[:, :0:-1], fx], axis=1)
            y = np.concatenate([by[:, :0:-1], fy], axis=1)
            z = np.concatenate([bz[:, :0:-1], fz], axis=1)
            phi = np.concatenate([bphi[:, :0:-1], fphi], axis=1)
        else:
            x, y, z, phi = integrate_cartesian(1.0, n_steps)
        R = np.sqrt(x * x + y * y)
        trace_backend = (
            "pyna.plot.j_streamlines.python_rk4_cartesian_surface_projected_arclength"
            if use_surface_projected_cartesian
            else "pyna.plot.j_streamlines.python_rk4_cartesian_arclength"
        )
        trace_parameter = "normalized_cartesian_arclength"
        trace_mode = "cartesian_surface_projected" if use_surface_projected_cartesian else "raw_cartesian_unconstrained"

    metadata: dict[str, object] = {
        "schema": "pyna_pest_seeded_j_streamlines_v1",
        "trace_backend": trace_backend,
        "trace_mode": trace_mode,
        "trace_parameter": trace_parameter,
        "surface_constraint": bool(constrain_to_surface),
        "field_type": type(field).__name__,
        "nfp": int(field_eval.nfp),
        "field_period_rad": float(field_eval.field_period_rad),
        "pest_source": coords.source,
        "n_turns": float(n_turns),
        "steps_per_turn": int(steps_per_turn),
        "seed_count": int(seed_count),
        "seed_spacing": str(seed_spacing),
        "phi_seed_count": int(seed_phi_values.size),
        "phi_values": [float(x) for x in seed_phi_values],
        "phi_range": None if phi_range_norm is None else [float(phi_range_norm[0]), float(phi_range_norm[1])],
        "clip_phi_range": bool(clip_phi_range),
        "n_seed_lines": int(n_seed),
        "full_seed_line_count": int(full_seed_count),
        "seed_line_indices": [int(value) for value in selected_seed_indices],
        "n_points": int(n_points),
        "surface_arclength_per_turn": float(surface_arclength_per_turn),
        "surface_arclength_per_turn_role": "global_integration_step_scale_only",
        "surface_perimeter_m_by_full_seed_line": [
            float(value) for value in full_seed_surface_perimeter_m
        ],
        "surface_perimeter_normalization": (
            "physical_RZ_perimeter_of_each_seed_surface_at_its_seed_phi"
        ),
        "integration_step_arclength": float(abs(h_base)),
        "max_surface_distance": None if max_surface_distance is None else float(max_surface_distance),
        "snap_cartesian_to_surface": bool(snap_cartesian_to_surface) if not constrain_to_surface else False,
        "bidirectional": bool(bidirectional),
        "surface_indices": [int(i) for i in surf_idx],
        "phi_indices": [int(i) for i in phi_idx],
        "finite_fraction": float(np.count_nonzero(np.isfinite(R) & np.isfinite(z)) / max(R.size, 1)),
        "normal_leakage_abs_over_norm_median": _nan_stat(section_leakage_samples),
        "normal_leakage_abs_over_norm_p95": _nan_stat(section_leakage_samples, percentile=95.0),
        "surface_tangent_fraction_median": _nan_stat(section_tangent_fraction_samples),
        "surface_tangent_fraction_p05": _nan_stat(section_tangent_fraction_samples, percentile=5.0),
        "seed_normal_leakage_abs_over_norm_median": _nan_stat([seed_leakage]),
        "seed_normal_leakage_abs_over_norm_p95": _nan_stat([seed_leakage], percentile=95.0),
        "seed_surface_tangent_fraction_median": _nan_stat([seed_tangent_fraction]),
        "seed_surface_tangent_fraction_p05": _nan_stat([seed_tangent_fraction], percentile=5.0),
        "trajectory_normal_leakage_abs_over_norm_median": _nan_stat(trajectory_leakage_samples),
        "trajectory_normal_leakage_abs_over_norm_p95": _nan_stat(trajectory_leakage_samples, percentile=95.0),
        "trajectory_surface_tangent_fraction_median": _nan_stat(trajectory_tangent_fraction_samples),
        "trajectory_surface_tangent_fraction_p05": _nan_stat(trajectory_tangent_fraction_samples, percentile=5.0),
    }
    streamlines = PestSeededStreamlines(
        R=R,
        Z=z,
        phi=phi,
        theta=theta,
        x=x,
        y=y,
        z=z,
        seed_R=seed_R,
        seed_Z=seed_Z,
        seed_phi=seed_phi,
        seed_rho=seeds["rho"],
        seed_theta=seeds["theta"],
        seed_surface_index=seeds["surface_index"],
        seed_phi_index=seeds["phi_index"],
        metadata=metadata,
    )
    if not bool(_return_diagnostics):
        return streamlines
    trajectory_leakage = (
        np.stack(trajectory_leakage_samples, axis=0)
        if trajectory_leakage_samples
        else np.empty((0, n_seed), dtype=np.float64)
    )
    trajectory_tangent_fraction = (
        np.stack(trajectory_tangent_fraction_samples, axis=0)
        if trajectory_tangent_fraction_samples
        else np.empty((0, n_seed), dtype=np.float64)
    )
    return streamlines, _PestTraceRawDiagnostics(
        seed_leakage=np.asarray(seed_leakage, dtype=np.float64),
        seed_tangent_fraction=np.asarray(seed_tangent_fraction, dtype=np.float64),
        trajectory_leakage=trajectory_leakage,
        trajectory_tangent_fraction=trajectory_tangent_fraction,
    )


def _exact_positive_worker_count(workers: int) -> int:
    if isinstance(workers, (bool, np.bool_)):
        raise TypeError("workers must be an exact positive integer")
    try:
        value = operator.index(workers)
    except TypeError as exc:
        raise TypeError("workers must be an exact positive integer") from exc
    if value <= 0:
        raise ValueError("workers must be positive")
    return int(value)


def _pest_trace_surface_chunks(
    plan: _PestSeedPlan,
    workers: int,
) -> list[list[int]]:
    selected = np.asarray(plan.selected_seed_indices, dtype=np.int64)
    selected_surfaces = np.asarray(plan.seeds["surface_index"], dtype=np.int64)
    active_surfaces = [
        int(surface)
        for surface in plan.surface_indices
        if np.any(selected_surfaces == int(surface))
    ]
    chunk_count = min(int(workers), len(active_surfaces))
    if chunk_count <= 1:
        return [[int(value) for value in selected]]
    chunks: list[list[int]] = []
    for surface_chunk in np.array_split(
        np.asarray(active_surfaces, dtype=np.int64),
        chunk_count,
    ):
        mask = np.isin(selected_surfaces, surface_chunk)
        chunks.append([int(value) for value in selected[mask]])
    return [chunk for chunk in chunks if chunk]


def _trace_pest_surface_chunk_fork(
    seed_line_indices: Sequence[int],
) -> tuple[PestSeededStreamlines, _PestTraceRawDiagnostics]:
    if (
        _PEST_TRACE_FORK_FIELD is None
        or _PEST_TRACE_FORK_PEST is None
        or _PEST_TRACE_FORK_KWARGS is None
    ):
        raise RuntimeError("parallel PEST trace fork context is unavailable")
    result = _trace_j_streamlines_on_pest_serial(
        _PEST_TRACE_FORK_FIELD,
        _PEST_TRACE_FORK_PEST,
        seed_line_indices=seed_line_indices,
        _return_diagnostics=True,
        **_PEST_TRACE_FORK_KWARGS,
    )
    if not isinstance(result, tuple):
        raise RuntimeError("parallel PEST trace worker lost diagnostics")
    return result


def _pest_parallel_trace_metadata(
    plan: _PestSeedPlan,
    chunks: Sequence[Sequence[int]],
    *,
    requested_workers: int,
    process_start_method: str | None,
) -> dict[str, object]:
    return {
        "schema": "pyna.pest_surface_parallel_trace.v1",
        "requested_workers": int(requested_workers),
        "used_workers": int(len(chunks)),
        "execution_mode": (
            "fork_process_pool" if process_start_method is not None else "serial_single_surface"
        ),
        "process_start_method": process_start_method,
        "partition_axis": "seed_surface_index",
        "surface_indices_by_chunk": [
            sorted(
                {
                    int(plan.full_seeds["surface_index"][int(seed_index)])
                    for seed_index in chunk
                }
            )
            for chunk in chunks
        ],
        "seed_line_count_by_chunk": [int(len(chunk)) for chunk in chunks],
        "seed_line_indices_by_chunk": [
            [int(seed_index) for seed_index in chunk] for chunk in chunks
        ],
        "merge_order": "requested_complete_seed_grid_row_order",
    }


def _merge_parallel_pest_streamlines(
    plan: _PestSeedPlan,
    chunks: Sequence[Sequence[int]],
    results: Sequence[tuple[PestSeededStreamlines, _PestTraceRawDiagnostics]],
    *,
    requested_workers: int,
) -> PestSeededStreamlines:
    if len(chunks) != len(results) or not results:
        raise RuntimeError("parallel PEST trace lost surface chunks")
    concatenated_indices = np.asarray(
        [int(value) for chunk in chunks for value in chunk],
        dtype=np.int64,
    )
    selected_indices = np.asarray(plan.selected_seed_indices, dtype=np.int64)
    if (
        concatenated_indices.size != selected_indices.size
        or np.unique(concatenated_indices).size != selected_indices.size
        or set(concatenated_indices.tolist()) != set(selected_indices.tolist())
    ):
        raise RuntimeError("parallel PEST trace changed complete seed identities")
    concatenated_position = {
        int(seed_index): int(position)
        for position, seed_index in enumerate(concatenated_indices)
    }
    order = np.asarray(
        [concatenated_position[int(seed_index)] for seed_index in selected_indices],
        dtype=np.int64,
    )

    lines = [result[0] for result in results]
    diagnostics = [result[1] for result in results]
    for chunk, item in zip(chunks, lines):
        metadata = item.metadata
        if metadata.get("seed_line_indices") != [int(value) for value in chunk]:
            raise RuntimeError("parallel PEST trace worker changed seed identities")
        if int(metadata.get("full_seed_line_count", -1)) != plan.full_seed_count:
            raise RuntimeError("parallel PEST trace worker changed the complete seed grid")
        if item.n_lines != len(chunk):
            raise RuntimeError("parallel PEST trace worker returned the wrong row count")

    trajectory_names = ("R", "Z", "phi", "theta", "x", "y", "z")
    seed_names = (
        "seed_R",
        "seed_Z",
        "seed_phi",
        "seed_rho",
        "seed_theta",
        "seed_surface_index",
        "seed_phi_index",
    )
    merged_arrays = {
        name: np.concatenate(
            [np.asarray(getattr(item, name)) for item in lines], axis=0
        )[order]
        for name in (*trajectory_names, *seed_names)
    }
    seed_leakage = np.concatenate(
        [np.asarray(item.seed_leakage) for item in diagnostics], axis=0
    )[order]
    seed_tangent_fraction = np.concatenate(
        [np.asarray(item.seed_tangent_fraction) for item in diagnostics], axis=0
    )[order]
    trajectory_sample_counts = {
        int(item.trajectory_leakage.shape[0]) for item in diagnostics
    }
    if len(trajectory_sample_counts) != 1 or any(
        item.trajectory_leakage.shape != item.trajectory_tangent_fraction.shape
        for item in diagnostics
    ):
        raise RuntimeError("parallel PEST trace worker diagnostics are inconsistent")
    trajectory_leakage = np.concatenate(
        [item.trajectory_leakage for item in diagnostics], axis=1
    )[:, order]
    trajectory_tangent_fraction = np.concatenate(
        [item.trajectory_tangent_fraction for item in diagnostics], axis=1
    )[:, order]

    metadata = dict(lines[0].metadata)
    R = np.asarray(merged_arrays["R"], dtype=np.float64)
    Z = np.asarray(merged_arrays["Z"], dtype=np.float64)
    metadata.update(
        {
            "n_seed_lines": int(selected_indices.size),
            "full_seed_line_count": int(plan.full_seed_count),
            "seed_line_indices": [int(value) for value in selected_indices],
            "finite_fraction": float(
                np.count_nonzero(np.isfinite(R) & np.isfinite(Z))
                / max(R.size, 1)
            ),
            "seed_normal_leakage_abs_over_norm_median": _nan_stat(
                [seed_leakage]
            ),
            "seed_normal_leakage_abs_over_norm_p95": _nan_stat(
                [seed_leakage], percentile=95.0
            ),
            "seed_surface_tangent_fraction_median": _nan_stat(
                [seed_tangent_fraction]
            ),
            "seed_surface_tangent_fraction_p05": _nan_stat(
                [seed_tangent_fraction], percentile=5.0
            ),
            "trajectory_normal_leakage_abs_over_norm_median": _nan_stat(
                [trajectory_leakage]
            ),
            "trajectory_normal_leakage_abs_over_norm_p95": _nan_stat(
                [trajectory_leakage], percentile=95.0
            ),
            "trajectory_surface_tangent_fraction_median": _nan_stat(
                [trajectory_tangent_fraction]
            ),
            "trajectory_surface_tangent_fraction_p05": _nan_stat(
                [trajectory_tangent_fraction], percentile=5.0
            ),
            "parallel_trace": _pest_parallel_trace_metadata(
                plan,
                chunks,
                requested_workers=requested_workers,
                process_start_method="fork",
            ),
        }
    )
    return replace(lines[0], **merged_arrays, metadata=metadata)


def trace_j_streamlines_on_pest(
    field: VectorFieldCylind | object,
    pest: SmoothPestCoordinates | Mapping[str, object] | str | Path,
    *,
    surface_index: int | Sequence[int] | None = -1,
    phi_indices: Sequence[int] | None = None,
    phi_values: Sequence[float] | None = None,
    phi_range: Sequence[float] | None = None,
    phi_seed_count: int | None = None,
    clip_phi_range: bool = True,
    seed_count: int = 12,
    seed_spacing: str = "arclength",
    seed_line_indices: Sequence[int] | None = None,
    n_turns: float = 0.2,
    steps_per_turn: int = 512,
    bidirectional: bool = True,
    theta_offset: float = 0.0,
    min_field_norm: float = 1.0e-14,
    min_tangent_norm: float | None = None,
    constrain_to_surface: bool = True,
    max_surface_distance: float | None = None,
    snap_cartesian_to_surface: bool = True,
    workers: int = 1,
) -> PestSeededStreamlines:
    """Trace current streamlines from a deterministic PEST-surface seed grid.

    ``workers=1`` follows the historical serial path.  On POSIX systems,
    ``workers>1`` partitions selected seed rows by magnetic surface and uses a
    fork process pool.  Every worker calls the same serial tracer with the
    complete surface request, so the normalized-arclength step and explicit
    ``nfp``/one-field-period contract are unchanged.  Results are restored to
    the original complete-seed-grid order, including adaptive subsets selected
    with ``seed_line_indices``.
    """

    worker_count = _exact_positive_worker_count(workers)
    kwargs: dict[str, object] = {
        "surface_index": surface_index,
        "phi_indices": phi_indices,
        "phi_values": phi_values,
        "phi_range": phi_range,
        "phi_seed_count": phi_seed_count,
        "clip_phi_range": bool(clip_phi_range),
        "seed_count": int(seed_count),
        "seed_spacing": str(seed_spacing),
        "n_turns": float(n_turns),
        "steps_per_turn": int(steps_per_turn),
        "bidirectional": bool(bidirectional),
        "theta_offset": float(theta_offset),
        "min_field_norm": float(min_field_norm),
        "min_tangent_norm": min_tangent_norm,
        "constrain_to_surface": bool(constrain_to_surface),
        "max_surface_distance": max_surface_distance,
        "snap_cartesian_to_surface": bool(snap_cartesian_to_surface),
    }
    if worker_count == 1:
        result = _trace_j_streamlines_on_pest_serial(
            field,
            pest,
            seed_line_indices=seed_line_indices,
            **kwargs,
        )
        if isinstance(result, tuple):
            raise RuntimeError("serial PEST trace unexpectedly returned diagnostics")
        return result

    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError("parallel PEST tracing requires POSIX fork support")
    if mp.current_process().daemon:
        raise RuntimeError("parallel PEST tracing cannot spawn from a daemon process")
    coords = _as_pest_coordinates(pest)
    _validate_pest(coords)
    plan = _prepare_pest_seed_plan(
        coords,
        surface_index=surface_index,
        phi_indices=phi_indices,
        phi_values=phi_values,
        phi_range=phi_range,
        phi_seed_count=phi_seed_count,
        clip_phi_range=bool(clip_phi_range),
        seed_count=int(seed_count),
        seed_spacing=str(seed_spacing),
        seed_line_indices=seed_line_indices,
        theta_offset=float(theta_offset),
    )
    chunks = _pest_trace_surface_chunks(plan, worker_count)
    if len(chunks) == 1:
        result = _trace_j_streamlines_on_pest_serial(
            field,
            pest,
            seed_line_indices=seed_line_indices,
            **kwargs,
        )
        if isinstance(result, tuple):
            raise RuntimeError("serial PEST trace unexpectedly returned diagnostics")
        metadata = dict(result.metadata)
        metadata["parallel_trace"] = _pest_parallel_trace_metadata(
            plan,
            chunks,
            requested_workers=worker_count,
            process_start_method=None,
        )
        return replace(result, metadata=metadata)

    global _PEST_TRACE_FORK_FIELD, _PEST_TRACE_FORK_PEST, _PEST_TRACE_FORK_KWARGS
    with _PEST_TRACE_FORK_LOCK:
        _PEST_TRACE_FORK_FIELD = field
        _PEST_TRACE_FORK_PEST = pest
        _PEST_TRACE_FORK_KWARGS = kwargs
        try:
            context = mp.get_context("fork")
            with context.Pool(processes=len(chunks)) as pool:
                results = pool.map(_trace_pest_surface_chunk_fork, chunks)
        finally:
            _PEST_TRACE_FORK_FIELD = None
            _PEST_TRACE_FORK_PEST = None
            _PEST_TRACE_FORK_KWARGS = None
    return _merge_parallel_pest_streamlines(
        plan,
        chunks,
        results,
        requested_workers=worker_count,
    )


def _surface_xyz(
    pest: SmoothPestCoordinates,
    surface_index: int,
    *,
    downsample: int = 1,
    phi_range: Sequence[float] | None = None,
    phi_samples: int | None = None,
) -> np.ndarray:
    step = max(1, int(downsample))
    ir = int(surface_index) % pest.R_surf.shape[1]
    phi_vals = np.asarray(pest.phi_vals, dtype=np.float64)
    theta_vals = np.asarray(pest.theta_vals, dtype=np.float64)
    phi_period = float(getattr(pest, "period", TWOPI) or TWOPI)
    normalized = _normalize_phi_range(phi_range, period=phi_period)
    if normalized is None:
        phi_plot = phi_vals[::step]
        if phi_plot.size and abs((phi_plot[-1] - phi_plot[0]) - phi_period) > 1.0e-12:
            phi_plot = np.concatenate([phi_plot, [phi_plot[0] + phi_period]])
    else:
        start, _end, span = normalized
        if phi_samples is None:
            grid_equiv = int(np.ceil(max(phi_vals.size, 2) * float(span) / phi_period / step)) + 1
            n_phi_plot = max(12, grid_equiv)
        else:
            n_phi_plot = max(int(phi_samples), 2)
        phi_plot = float(start) + np.linspace(0.0, float(span), n_phi_plot, dtype=np.float64)
    if phi_plot.size == 0:
        phi_plot = phi_vals[::step]
    theta_plot = theta_vals[::step]
    if theta_plot.size and abs((theta_plot[-1] - theta_plot[0]) - TWOPI) > 1.0e-12:
        theta_plot = np.concatenate([theta_plot, [theta_plot[0] + TWOPI]])
    phi_grid, theta_grid = np.meshgrid(phi_plot, theta_plot, indexing="ij")
    R, Z = _surface_points_at_theta_phi(
        pest,
        surface_index=ir,
        theta=theta_grid,
        phi=phi_grid,
    )
    return _cylindrical_to_cartesian(R, Z, phi_grid)


def _plot_segmented_section_line(
    ax,
    R_line: np.ndarray,
    Z_line: np.ndarray,
    *,
    max_step: float,
    color,
    lw: float,
    alpha: float,
) -> None:
    R = np.asarray(R_line, dtype=np.float64)
    Z = np.asarray(Z_line, dtype=np.float64)
    finite = np.isfinite(R) & np.isfinite(Z)
    if np.count_nonzero(finite) < 2:
        return
    split = np.ones(R.size, dtype=bool)
    split[1:] = (
        ~finite[1:]
        | ~finite[:-1]
        | (np.sqrt(np.diff(R) ** 2 + np.diff(Z) ** 2) > float(max_step))
    )
    starts = np.flatnonzero(finite & split)
    for start in starts:
        end = start + 1
        while end < R.size and finite[end] and not split[end]:
            end += 1
        if end - start >= 2:
            ax.plot(R[start:end], Z[start:end], color=color, lw=lw, alpha=alpha)


def _finite_segment_slices(
    mask: np.ndarray,
    xyz: np.ndarray | None = None,
    *,
    max_step: float | None = None,
    max_angle_deg: float | None = None,
) -> list[slice]:
    finite = np.asarray(mask, dtype=bool)
    split = np.ones(finite.size, dtype=bool)
    split[1:] = ~finite[1:] | ~finite[:-1]
    if xyz is not None:
        pts = np.asarray(xyz, dtype=np.float64)
        if pts.shape == (finite.size, 3):
            seg = np.diff(pts, axis=0)
            seg_len = np.sqrt(np.sum(seg * seg, axis=1))
            if max_step is not None and np.isfinite(float(max_step)) and float(max_step) > 0.0:
                split[1:] |= seg_len > float(max_step)
            if max_angle_deg is not None and np.isfinite(float(max_angle_deg)) and 0.0 < float(max_angle_deg) < 180.0:
                good_seg = np.isfinite(seg).all(axis=1) & (seg_len > 0.0)
                unit = np.full_like(seg, np.nan, dtype=np.float64)
                unit[good_seg] = seg[good_seg] / seg_len[good_seg, None]
                if unit.shape[0] >= 2:
                    dots = np.sum(unit[:-1] * unit[1:], axis=1)
                    angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
                    sharp = np.isfinite(angles) & (angles > float(max_angle_deg))
                    split[1:-1] |= sharp
    segments: list[slice] = []
    start: int | None = None
    for idx, value in enumerate(finite):
        if value and start is not None and split[idx]:
            if idx - start >= 2:
                segments.append(slice(start, idx))
            start = idx
        elif value and start is None:
            start = int(idx)
        elif not value and start is not None:
            if idx - start >= 2:
                segments.append(slice(start, idx))
            start = None
    if start is not None and finite.size - start >= 2:
        segments.append(slice(start, finite.size))
    return segments


def _streamline_arrow_samples(
    xyz: np.ndarray,
    keep: np.ndarray,
    *,
    arrow_count: int,
    max_step: float | None = None,
    max_angle_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if int(arrow_count) <= 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    points = np.asarray(xyz, dtype=np.float64)
    segments = _finite_segment_slices(
        keep,
        points,
        max_step=max_step,
        max_angle_deg=max_angle_deg,
    )
    segment_data: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    total_length = 0.0
    for segment in segments:
        pts = points[segment]
        if pts.shape[0] < 2:
            continue
        delta = np.diff(pts, axis=0)
        seg_len = np.sqrt(np.sum(delta * delta, axis=1))
        good = np.isfinite(seg_len) & (seg_len > 0.0) & np.isfinite(delta).all(axis=1)
        if not np.any(good):
            continue
        cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])
        length = float(cumulative[-1])
        if not np.isfinite(length) or length <= 0.0:
            continue
        segment_data.append((pts, delta, cumulative, length))
        total_length += length
    if not segment_data or total_length <= 0.0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    positions: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    targets = (np.arange(int(arrow_count), dtype=np.float64) + 0.5) * total_length / float(arrow_count)
    for target in targets:
        cursor = float(target)
        for pts, delta, cumulative, length in segment_data:
            if cursor > length:
                cursor -= length
                continue
            idx = int(np.searchsorted(cumulative, cursor, side="right") - 1)
            idx = max(0, min(idx, delta.shape[0] - 1))
            step_length = float(cumulative[idx + 1] - cumulative[idx])
            if step_length <= 0.0:
                break
            frac = float((cursor - cumulative[idx]) / step_length)
            direction = delta[idx] / step_length
            positions.append(pts[idx] + frac * delta[idx])
            directions.append(direction)
            break
    if not positions:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    return np.asarray(positions, dtype=np.float64), np.asarray(directions, dtype=np.float64)


def plot_j_streamlines_on_pest_surface_plotly(
    streamlines: PestSeededStreamlines,
    pest: SmoothPestCoordinates | Mapping[str, object] | str | Path | None = None,
    *,
    surface_index: int | Sequence[int] | None = None,
    phi_range: Sequence[float] | None = None,
    companion_streamlines: PestSeededStreamlines | Sequence[PestSeededStreamlines] | None = None,
    companion_name: str = "B",
    style: str | PlotlyStreamlineStyle | Mapping[str, object] | None = None,
    html_path: str | Path | None = None,
    include_plotlyjs: str | bool = "cdn",
    show_surface: bool = True,
    surface_downsample: int = 1,
    surface_phi_samples: int | None = None,
    surface_opacity: float = 0.24,
    line_width: float = 4.0,
    companion_line_width: float = 2.4,
    j_color: str | None = None,
    j_colorscale: str = "Turbo",
    companion_color: str = "rgba(37, 99, 235, 0.72)",
    line_opacity: float = 0.96,
    companion_line_opacity: float = 0.72,
    show_arrows: bool = False,
    arrow_count_per_line: int = 1,
    companion_arrow_count_per_line: int = 1,
    arrow_line_stride: int = 1,
    companion_arrow_line_stride: int = 2,
    arrow_size: float = 0.055,
    companion_arrow_size: float | None = None,
    j_arrow_color: str = "#b91c1c",
    companion_arrow_color: str = "#2563eb",
    max_segment_step: float | None = None,
    max_segment_angle_deg: float | None = 45.0,
    title: str | None = None,
    width: int = 1100,
    height: int = 850,
):
    """Return a Plotly 3-D view of PEST-seeded J streamlines.

    The plot can show several selected PEST surfaces, clip both surfaces and
    lines to a toroidal ``phi_range``, and overlay a companion set of
    streamlines, for example magnetic-field lines drawn with a thinner style.
    Direction arrows are optional Plotly cone traces sampled along the visible
    streamline segments.  ``style`` may be a named preset, a
    :class:`PlotlyStreamlineStyle`, or a mapping of Plotly keyword overrides.
    """

    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        from plotly.colors import sample_colorscale
    except ImportError as exc:
        raise ImportError("plotly is required for plot_j_streamlines_on_pest_surface_plotly") from exc

    coords = _as_pest_coordinates(pest) if pest is not None else None
    phi_period = float(getattr(coords, "period", TWOPI) or TWOPI) if coords is not None else TWOPI
    normalized_phi_range = _normalize_phi_range(phi_range, period=phi_period)
    if style is not None:
        if isinstance(style, str):
            style_kwargs = plotly_streamline_style(style).to_plotly_kwargs()
        elif isinstance(style, PlotlyStreamlineStyle):
            style_kwargs = style.to_plotly_kwargs()
        elif isinstance(style, Mapping):
            style_kwargs = dict(style)
        else:
            raise TypeError("style must be a preset name, PlotlyStreamlineStyle, mapping, or None")
        surface_opacity = float(style_kwargs.get("surface_opacity", surface_opacity))
        line_width = float(style_kwargs.get("line_width", line_width))
        companion_line_width = float(style_kwargs.get("companion_line_width", companion_line_width))
        j_color = style_kwargs.get("j_color", j_color)  # type: ignore[assignment]
        j_colorscale = str(style_kwargs.get("j_colorscale", j_colorscale))
        companion_color = str(style_kwargs.get("companion_color", companion_color))
        line_opacity = float(style_kwargs.get("line_opacity", line_opacity))
        companion_line_opacity = float(style_kwargs.get("companion_line_opacity", companion_line_opacity))
        show_arrows = bool(style_kwargs.get("show_arrows", show_arrows))
        arrow_count_per_line = int(style_kwargs.get("arrow_count_per_line", arrow_count_per_line))
        companion_arrow_count_per_line = int(
            style_kwargs.get("companion_arrow_count_per_line", companion_arrow_count_per_line)
        )
        arrow_line_stride = int(style_kwargs.get("arrow_line_stride", arrow_line_stride))
        companion_arrow_line_stride = int(
            style_kwargs.get("companion_arrow_line_stride", companion_arrow_line_stride)
        )
        arrow_size = float(style_kwargs.get("arrow_size", arrow_size))
        companion_arrow_size = style_kwargs.get("companion_arrow_size", companion_arrow_size)  # type: ignore[assignment]
        j_arrow_color = str(style_kwargs.get("j_arrow_color", j_arrow_color))
        companion_arrow_color = str(style_kwargs.get("companion_arrow_color", companion_arrow_color))
    fig = go.Figure()
    companion_list: list[PestSeededStreamlines]
    if companion_streamlines is None:
        companion_list = []
    elif isinstance(companion_streamlines, PestSeededStreamlines):
        companion_list = [companion_streamlines]
    else:
        companion_list = list(companion_streamlines)

    default_surfaces = sorted({int(i) for i in streamlines.seed_surface_index if np.isfinite(i)})
    for companion in companion_list:
        default_surfaces.extend(int(i) for i in companion.seed_surface_index if np.isfinite(i))
    default_surfaces = sorted(set(default_surfaces)) or [0]
    allowed_surface_set: set[int] | None = None
    if show_surface and coords is not None:
        surface_indices = _normalize_indices(surface_index, coords.R_surf.shape[1], default=default_surfaces)
        allowed_surface_set = {int(i) for i in surface_indices}
        for surface_pos, ir in enumerate(surface_indices):
            xyz = _surface_xyz(
                coords,
                int(ir),
                downsample=surface_downsample,
                phi_range=phi_range,
                phi_samples=surface_phi_samples,
            )
            rho_label = (
                f"rho={float(coords.rho_vals[int(ir) % coords.R_surf.shape[1]]):.3f}"
                if np.asarray(coords.rho_vals).size
                else f"surface {int(ir)}"
            )
            fig.add_trace(
                go.Surface(
                    x=xyz[:, :, 0],
                    y=xyz[:, :, 1],
                    z=xyz[:, :, 2],
                    showscale=False,
                    opacity=float(surface_opacity),
                    colorscale=[[0.0, "rgb(184,190,199)"], [1.0, "rgb(184,190,199)"]],
                    lighting=dict(ambient=0.64, diffuse=0.72, specular=0.12, roughness=0.8),
                    lightposition=dict(x=2.0, y=2.0, z=1.0),
                    hoverinfo="skip",
                    name=f"PEST surface {rho_label}",
                    showlegend=surface_pos == 0,
                    legendgroup="PEST surface",
                )
            )

    if allowed_surface_set is None and surface_index is not None and coords is not None:
        allowed_surface_set = {
            int(i) for i in _normalize_indices(surface_index, coords.R_surf.shape[1], default=default_surfaces)
        }

    def add_streamline_traces(
        lines: PestSeededStreamlines,
        *,
        label: str,
        width_value: float,
        colorscale: str | None,
        fixed_color: str | None,
        opacity: float,
        arrow_color: str,
        arrow_trace_size: float,
        arrows_per_line: int,
        arrow_stride: int,
    ) -> None:
        finite_rho = lines.seed_rho[np.isfinite(lines.seed_rho)]
        rho_min = float(np.nanmin(finite_rho)) if finite_rho.size else 0.0
        rho_max = float(np.nanmax(finite_rho)) if finite_rho.size else 1.0
        if rho_max <= rho_min:
            rho_max = rho_min + 1.0
        legend_added = False
        arrow_positions: list[np.ndarray] = []
        arrow_directions: list[np.ndarray] = []
        visible_line_count = 0
        for line_idx in range(lines.n_lines):
            if allowed_surface_set is not None and int(lines.seed_surface_index[line_idx]) not in allowed_surface_set:
                continue
            keep = (
                np.isfinite(lines.x[line_idx])
                & np.isfinite(lines.y[line_idx])
                & np.isfinite(lines.z[line_idx])
            )
            if normalized_phi_range is not None:
                keep &= _phi_in_range(lines.phi[line_idx], normalized_phi_range, period=phi_period)
            if np.count_nonzero(keep) < 2:
                continue
            use_for_arrows = (
                bool(show_arrows)
                and int(arrows_per_line) > 0
                and int(arrow_stride) > 0
                and visible_line_count % int(arrow_stride) == 0
            )
            visible_line_count += 1
            if fixed_color is None:
                color_value = (float(lines.seed_rho[line_idx]) - rho_min) / (rho_max - rho_min)
                color = sample_colorscale(colorscale or "Turbo", color_value)[0]
            else:
                color = fixed_color
            xyz_line = lines.xyz[line_idx]
            for segment in _finite_segment_slices(
                keep,
                xyz_line,
                max_step=max_segment_step,
                max_angle_deg=max_segment_angle_deg,
            ):
                point_count = int(segment.stop - segment.start)
                fig.add_trace(
                    go.Scatter3d(
                        x=lines.x[line_idx, segment],
                        y=lines.y[line_idx, segment],
                        z=lines.z[line_idx, segment],
                        mode="lines",
                        line=dict(color=color, width=float(width_value)),
                        opacity=float(opacity),
                        name=label,
                        legendgroup=label,
                        hovertemplate=(
                            f"{label}<br>"
                            "seed rho=%{customdata[0]:.3f}<br>"
                            "seed theta=%{customdata[1]:.3f}<br>"
                            "X=%{x:.3f}<br>Y=%{y:.3f}<br>Z=%{z:.3f}<extra></extra>"
                        ),
                        customdata=np.column_stack(
                            [
                                np.full(point_count, lines.seed_rho[line_idx]),
                                np.full(point_count, lines.seed_theta[line_idx]),
                            ]
                        ),
                        showlegend=not legend_added,
                    )
                )
                legend_added = True
            if use_for_arrows:
                pos, direction = _streamline_arrow_samples(
                    xyz_line,
                    keep,
                    arrow_count=int(arrows_per_line),
                    max_step=max_segment_step,
                    max_angle_deg=max_segment_angle_deg,
                )
                if pos.size:
                    arrow_positions.append(pos)
                    arrow_directions.append(direction)
        if bool(show_arrows) and arrow_positions:
            pos_arr = np.concatenate(arrow_positions, axis=0)
            dir_arr = np.concatenate(arrow_directions, axis=0)
            fig.add_trace(
                go.Cone(
                    x=pos_arr[:, 0],
                    y=pos_arr[:, 1],
                    z=pos_arr[:, 2],
                    u=dir_arr[:, 0],
                    v=dir_arr[:, 1],
                    w=dir_arr[:, 2],
                    sizemode="absolute",
                    sizeref=float(arrow_trace_size),
                    anchor="tail",
                    colorscale=[[0.0, arrow_color], [1.0, arrow_color]],
                    showscale=False,
                    opacity=min(max(float(opacity), 0.0), 1.0),
                    name=f"{label} direction",
                    legendgroup=label,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    add_streamline_traces(
        streamlines,
        label="J streamlines",
        width_value=float(line_width),
        colorscale=j_colorscale,
        fixed_color=j_color,
        opacity=float(line_opacity),
        arrow_color=j_arrow_color,
        arrow_trace_size=float(arrow_size),
        arrows_per_line=int(arrow_count_per_line),
        arrow_stride=int(arrow_line_stride),
    )
    for companion in companion_list:
        add_streamline_traces(
            companion,
            label=f"{companion_name} streamlines",
            width_value=float(companion_line_width),
            colorscale=None,
            fixed_color=companion_color,
            opacity=float(companion_line_opacity),
            arrow_color=companion_arrow_color,
            arrow_trace_size=float(companion_arrow_size if companion_arrow_size is not None else 0.8 * float(arrow_size)),
            arrows_per_line=int(companion_arrow_count_per_line),
            arrow_stride=int(companion_arrow_line_stride),
        )
    fig.update_layout(
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        scene=dict(
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            zaxis_title="Z [m]",
            aspectmode="data",
            camera=dict(eye=dict(x=1.8, y=-1.2, z=0.85), center=dict(x=0.0, y=0.0, z=0.0), up=dict(x=0.0, y=0.0, z=1.0)),
        ),
        width=int(width),
        height=int(height),
        margin=dict(l=0, r=20, b=0, t=60),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.72)"),
    )
    if html_path is not None:
        pio.write_html(fig, str(Path(html_path)), include_plotlyjs=include_plotlyjs, auto_open=False)
    return fig


def plot_j_streamline_seed_sections(
    streamlines: PestSeededStreamlines,
    pest: SmoothPestCoordinates | Mapping[str, object] | str | Path,
    *,
    section_indices: Sequence[int] | None = None,
    title: str | None = None,
    line_width: float = 1.1,
    alpha: float = 0.86,
    max_projection_step: float | None = None,
    project_cartesian_to_pest: bool = True,
    max_surface_distance: float | None = None,
):
    """Plot PEST-seeded J streamlines projected back to their seed sections."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from pyna.plot.mgrid import draw_smooth_pest_grid

    coords = _as_pest_coordinates(pest)
    _validate_pest(coords)
    if section_indices is None:
        section_indices = sorted({int(i) for i in streamlines.seed_phi_index})
    sec = np.asarray(list(section_indices), dtype=np.int64)
    if sec.size == 0:
        raise ValueError("section_indices must not be empty")
    ncols = min(int(sec.size), 4)
    nrows = int(np.ceil(sec.size / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 4.0 * nrows), squeeze=False, constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    rho_min = float(np.nanmin(streamlines.seed_rho))
    rho_max = float(np.nanmax(streamlines.seed_rho))
    if rho_max <= rho_min:
        rho_max = rho_min + 1.0
    use_pest_projection = (
        streamlines.metadata.get("trace_mode") == "pest_surface_constrained"
        and np.shape(streamlines.theta) == np.shape(streamlines.R)
    )
    use_cartesian_projection = bool(project_cartesian_to_pest) and not use_pest_projection
    phi0 = float(coords.phi_vals[0]) if np.asarray(coords.phi_vals).size else 0.0
    theta0 = float(coords.theta_vals[0]) if np.asarray(coords.theta_vals).size else 0.0
    phi_period = float(getattr(coords, "period", TWOPI) or TWOPI)
    for ax in axes.ravel()[sec.size:]:
        ax.set_visible(False)
    for out_idx, iphi in enumerate(sec):
        ax = axes.ravel()[out_idx]
        ip = int(iphi) % coords.R_surf.shape[0]
        draw_smooth_pest_grid(ax, coords.R_surf[ip], coords.Z_surf[ip])
        sec_R = np.asarray(coords.R_surf[ip], dtype=np.float64)
        sec_Z = np.asarray(coords.Z_surf[ip], dtype=np.float64)
        if max_projection_step is None:
            span = max(float(np.nanmax(sec_R) - np.nanmin(sec_R)), float(np.nanmax(sec_Z) - np.nanmin(sec_Z)), 1.0e-12)
            step_limit = 0.08 * span
        else:
            step_limit = float(max_projection_step)
        selected = np.flatnonzero(streamlines.seed_phi_index == ip)
        for line_idx in selected:
            keep = np.isfinite(streamlines.R[line_idx]) & np.isfinite(streamlines.Z[line_idx])
            if use_pest_projection:
                keep &= np.isfinite(streamlines.theta[line_idx])
            if use_cartesian_projection:
                keep &= np.isfinite(streamlines.phi[line_idx])
            if np.count_nonzero(keep) < 2:
                continue
            if use_pest_projection:
                ir = int(streamlines.seed_surface_index[line_idx]) % coords.R_surf.shape[1]
                theta_line = streamlines.theta[line_idx, keep]
                phi_line = np.full_like(theta_line, float(coords.phi_vals[ip]), dtype=np.float64)
                R_line = _periodic_surface_bilinear(
                    coords.R_surf[:, ir, :],
                    phi_line,
                    theta_line,
                    phi0=phi0,
                    theta0=theta0,
                    phi_period=phi_period,
                )
                Z_line = _periodic_surface_bilinear(
                    coords.Z_surf[:, ir, :],
                    phi_line,
                    theta_line,
                    phi0=phi0,
                    theta0=theta0,
                    phi_period=phi_period,
                )
            elif use_cartesian_projection:
                ir = int(streamlines.seed_surface_index[line_idx]) % coords.R_surf.shape[1]
                theta_line_all, distance_all = _cartesian_line_theta_on_seed_surface(
                    coords,
                    surface_index=ir,
                    R_line=streamlines.R[line_idx],
                    Z_line=streamlines.Z[line_idx],
                    phi_line=streamlines.phi[line_idx],
                )
                distance_limit = (
                    _default_surface_projection_distance(coords, ir)
                    if max_surface_distance is None
                    else float(max_surface_distance)
                )
                keep = (
                    keep
                    & np.isfinite(theta_line_all)
                    & np.isfinite(distance_all)
                    & (distance_all <= distance_limit)
                )
                if np.count_nonzero(keep) < 2:
                    continue
                theta_line = theta_line_all[keep]
                phi_line = np.full_like(theta_line, float(coords.phi_vals[ip]), dtype=np.float64)
                R_line = _periodic_surface_bilinear(
                    coords.R_surf[:, ir, :],
                    phi_line,
                    theta_line,
                    phi0=phi0,
                    theta0=theta0,
                    phi_period=phi_period,
                )
                Z_line = _periodic_surface_bilinear(
                    coords.Z_surf[:, ir, :],
                    phi_line,
                    theta_line,
                    phi0=phi0,
                    theta0=theta0,
                    phi_period=phi_period,
                )
            else:
                R_line = streamlines.R[line_idx, keep]
                Z_line = streamlines.Z[line_idx, keep]
            cval = (float(streamlines.seed_rho[line_idx]) - rho_min) / (rho_max - rho_min)
            _plot_segmented_section_line(
                ax,
                R_line,
                Z_line,
                max_step=step_limit,
                color=cmap(cval),
                lw=line_width,
                alpha=alpha,
            )
        ax.set_title(f"phi={float(coords.phi_vals[ip]):.3f}")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
    if title is not None:
        fig.suptitle(title)
    return fig, axes


__all__ = [
    "GriddedPestVectorField",
    "PestSeededStreamlines",
    "PlotlyStreamlineStyle",
    "VmecCurrentFourier",
    "field_period_phi_range",
    "pest_tangent_components_to_cylindrical",
    "plotly_streamline_style",
    "plot_j_streamline_seed_sections",
    "plot_j_streamlines_on_pest_surface_plotly",
    "trace_j_streamlines_on_pest",
    "vmec_current_fourier_to_pest_field",
]
