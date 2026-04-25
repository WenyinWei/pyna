"""healed_flux_coords.py
=================================
Formal public entry points for healed flux-coordinate construction.

This module is the stable-facing home for healed 3D boundary transport,
section-wise boundary correction, and healed flux-coordinate scaffold/bundle
assembly.  The current implementation re-exports the mature infrastructure
from the legacy internal module ``healed_scaffold_3d`` while we gradually
finish the naming migration.
"""

from pyna.topo.healed_scaffold_3d import (
    XOArcPoint,
    XOSequence,
    build_xo_sequence,
    xo_sequence_boundary_arcs,
    build_cxo_spline,
    SectionFit,
    BoundarySection,
    BoundaryConstraintSet,
    BoundaryFamily3D,
    SectionScaffoldBundle,
    TransportedSection,
    SectionCorrespondence,
    FieldLineScaffold3D,
    trace_grid_to_phi,
    trace_section_curve_to_phi,
    trace_surface_family_to_sections,
    fit_ring_fourier,
    build_pchip_family,
    eval_fourier_family,
    build_section_scaffold_bundle,
    correct_boundary_with_constraints,
)

__all__ = [
    "XOArcPoint",
    "XOSequence",
    "build_xo_sequence",
    "xo_sequence_boundary_arcs",
    "build_cxo_spline",
    "SectionFit",
    "BoundarySection",
    "BoundaryConstraintSet",
    "BoundaryFamily3D",
    "SectionScaffoldBundle",
    "TransportedSection",
    "SectionCorrespondence",
    "FieldLineScaffold3D",
    "trace_grid_to_phi",
    "trace_section_curve_to_phi",
    "trace_surface_family_to_sections",
    "fit_ring_fourier",
    "build_pchip_family",
    "eval_fourier_family",
    "build_section_scaffold_bundle",
    "correct_boundary_with_constraints",
]
