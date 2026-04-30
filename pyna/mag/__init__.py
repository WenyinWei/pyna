# pyna.mag �?backward-compatibility shim
# Files have moved to pyna.toroidal.*; this module re-exports everything.

from pyna.toroidal.coords.coordinate import (
    rzphi_to_xyz,
    xyz_to_rzphi,
    coord_system_change,
    coord_mirror,
    Jac_rz2stheta,
    RZ2STET,
    STET2RZ,
    calc_dRZdSTET_mesh,
)

from pyna.toroidal.equilibrium.GradShafranov import (
    recover_pressure_simplest,
    solve_GS_perturbed,
)

from pyna.toroidal.equilibrium.Solovev import EquilibriumSolovev
from pyna.toroidal.equilibrium.axisymmetric import EquilibriumAxisym, EquilibriumTokamakCircularSynthetic
from pyna.toroidal.equilibrium.stellarator import StellaratorSimple, simple_stellarator
from pyna.toroidal.coils.coil import BRBZ_induced_by_current_loop, BRBZ_induced_by_thick_finitelen_solenoid
from pyna.toroidal.coils.coil_system import CoilSet, Biot_Savart_field
from pyna.toroidal.coils.field import VectorField3DCylindrical, VectorField3DAxiSymmetric
from pyna.toroidal.coils.RMP import normalize_b, RMP_spectrum_2d, island_width_at_rational_surfaces
from pyna.toroidal.diagnostics import (
    field_line_length,
    field_line_endpoints,
    field_line_min_psi,
)
from pyna.toroidal.control import (
    compute_resonant_amplitude,
    island_suppression_current,
    phase_control_current,
    multi_mode_control,
)

