"""
pyna._cyna — thin Python wrapper around the _cyna_ext C++ extension.
"""
import importlib.util
import pathlib

_cyna_ext = None
_available = False

# Try to load _cyna_ext.pyd from this package directory, then from the cyna/ build dir
_p = pathlib.Path(__file__).parent
_build_dir = _p.parent.parent / "cyna"  # <repo>/cyna/ — xmake build output
_candidates = (
    list(_p.glob("_cyna_ext*.pyd")) + list(_p.glob("_cyna_ext*.so"))
    + list(_build_dir.glob("_cyna_ext*.pyd")) + list(_build_dir.glob("_cyna_ext*.so"))
    + ([_build_dir / "_cyna_ext.pyd"] if (_build_dir / "_cyna_ext.pyd").exists() else [])
)
# Deduplicate while preserving order
_seen = set()
_candidates_dedup = []
for _c in _candidates:
    if _c not in _seen:
        _seen.add(_c)
        _candidates_dedup.append(_c)
_candidates = _candidates_dedup
for _f in _candidates:
    try:
        _spec = importlib.util.spec_from_file_location("_cyna_ext", _f)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _cyna_ext = _mod
        _available = True
        break
    except Exception:
        continue

if not _available:
    # Last resort: try normal import (e.g., installed as package)
    try:
        import importlib
        _cyna_ext = importlib.import_module("pyna._cyna._cyna_ext")
        _available = True
    except Exception as _exc:
        raise ImportError(
            "pyna._cyna requires the compiled cyna extension, but _cyna_ext "
            "could not be loaded. Install a platform wheel from PyPI or build "
            "from source with xmake and a C++17 compiler, for example "
            "`pip install -e .` from the pyna repository."
        ) from _exc


def is_available() -> bool:
    """Return True if the C++ extension was successfully loaded."""
    return _available


def get_version() -> str:
    """Return the extension version string."""
    return _cyna_ext.__version__ if _available else "not available"


coil_circular_field           = getattr(_cyna_ext, "coil_circular_field", None)
coil_biot_savart              = getattr(_cyna_ext, "coil_biot_savart",     None)
cuda_backend_available        = getattr(_cyna_ext, "cuda_backend_available", lambda: False)
VectorFieldCylind             = getattr(_cyna_ext, "VectorFieldCylind",             None)
trace_poincare_batch = _cyna_ext.trace_poincare_batch
trace_poincare_multi = _cyna_ext.trace_poincare_multi
trace_poincare_batch_twall    = getattr(_cyna_ext, "trace_poincare_batch_twall",    None)
trace_map_batch_span          = getattr(_cyna_ext, "trace_map_batch_span",          None)
trace_connection_length_twall = getattr(_cyna_ext, "trace_connection_length_twall", None)
trace_wall_hits_twall         = getattr(_cyna_ext, "trace_wall_hits_twall",         None)
find_fixed_points_batch       = getattr(_cyna_ext, "find_fixed_points_batch",       None)
find_fixed_points_batch_span  = getattr(_cyna_ext, "find_fixed_points_batch_span",  None)
trace_orbit_along_phi         = getattr(_cyna_ext, "trace_orbit_along_phi",         None)
trace_poincare_dpk_growth     = getattr(_cyna_ext, "trace_poincare_dpk_growth",     None)
trace_poincare_dpk_growth_twall = getattr(_cyna_ext, "trace_poincare_dpk_growth_twall", None)
compute_cycle_perturbation_shift = getattr(_cyna_ext, "compute_cycle_perturbation_shift", None)
compute_A_matrix_batch        = getattr(_cyna_ext, "compute_A_matrix_batch",        None)
progress_DX_pol_along_orbit   = getattr(_cyna_ext, "progress_DX_pol_along_orbit",   None)
progress_delta_X_along_orbit  = getattr(_cyna_ext, "progress_delta_X_along_orbit",  None)
evolve_delta_X_cycle_along_cycle = getattr(
    _cyna_ext,
    "evolve_delta_X_cycle_along_cycle",
    getattr(_cyna_ext, "evolve_delta_X_cycle_along_orbit", None),
)
evolve_delta_X_cycle_along_orbit = getattr(
    _cyna_ext, "evolve_delta_X_cycle_along_orbit", evolve_delta_X_cycle_along_cycle
)
evolve_DPm_along_cycle        = getattr(_cyna_ext, "evolve_DPm_along_cycle",        None)
trace_surface_metrics_batch_twall = getattr(_cyna_ext, "trace_surface_metrics_batch_twall", None)
summarize_profile_objectives  = getattr(_cyna_ext, "summarize_profile_objectives",  None)
trace_poincare_beta_sweep     = getattr(_cyna_ext, "trace_poincare_beta_sweep",     None)

__all__ = [
    "is_available",
    "get_version",
    "coil_circular_field",
    "coil_biot_savart",
    "cuda_backend_available",
    "VectorFieldCylind",
    "trace_poincare_batch",
    "trace_poincare_multi",
    "trace_poincare_batch_twall",
    "trace_map_batch_span",
    "trace_connection_length_twall",
    "trace_wall_hits_twall",
    "find_fixed_points_batch",
    "find_fixed_points_batch_span",
    "trace_orbit_along_phi",
    "trace_poincare_dpk_growth",
    "trace_poincare_dpk_growth_twall",
    "compute_cycle_perturbation_shift",
    "compute_A_matrix_batch",
    "progress_DX_pol_along_orbit",
    "progress_delta_X_along_orbit",
    "evolve_delta_X_cycle_along_cycle",
    "evolve_delta_X_cycle_along_orbit",
    "evolve_DPm_along_cycle",
    "trace_surface_metrics_batch_twall",
    "summarize_profile_objectives",
    "trace_poincare_beta_sweep",
    # Utility helpers (pure Python, always available)
    "ensure_c_double",
    "prepare_field_cache",
    "build_fixed_points_from_batch",
    "build_cycles_from_batch",
]

# Pure-Python utilities are always available (no C++ extension required)
from pyna._cyna.utils import (
    ensure_c_double,
    prepare_field_cache,
    build_fixed_points_from_batch,
    build_cycles_from_batch,
)
