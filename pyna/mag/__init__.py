# pyna.mag — backward-compatibility shim
# Files have moved to pyna.MCF.*; this module re-exports everything.

from pyna.MCF.coords.coordinate import (
    rzphi_to_xyz,
    xyz_to_rzphi,
    coord_system_change,
    coord_mirror,
    Jac_rz2stheta,
    RZ2STET,
    STET2RZ,
    calc_dRZdSTET_mesh,
)

from pyna.MCF.equilibrium.GradShafranov import (
    recover_pressure_simplest,
    solve_GS_perturbed,
)

from pyna.MCF.equilibrium.Solovev import SolovevEquilibrium
from pyna.MCF.equilibrium.axisymmetric import AxisymEquilibrium, SyntheticCircularTokamakEquilibrium
from pyna.MCF.equilibrium.stellarator import SimpleStellarartor, simple_stellarator
from pyna.MCF.coils.coil import BRBZ_induced_by_current_loop, BRBZ_induced_by_thick_finitelen_solenoid
from pyna.MCF.coils.coil_system import CoilSet, Biot_Savart_field, biot_savart_field
from pyna.MCF.coils.field import CylindricalGridVectorField3D, CylindricalGridAxiVectorField3D
from pyna.MCF.coils.RMP import normalize_b, rmp_spectrum_2d, island_width_at_rational_surfaces
from pyna.MCF.diagnostics.measure import field_line_length, field_line_endpoints, field_line_min_psi
from pyna.MCF.control.island_control import (
    compute_resonant_amplitude, island_suppression_current,
    phase_control_current, multi_mode_control,
)
