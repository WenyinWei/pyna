"""pyna.MCF.coils — Coil geometry and field computation."""
from pyna.MCF.coils.base import CoilFieldVacuum, CoilFieldSuperposition, CoilFieldScaled
from pyna.MCF.coils.coil import BRBZ_induced_by_current_loop, BRBZ_induced_by_thick_finitelen_solenoid, CoilFieldAnalyticCircular
from pyna.MCF.coils.coil_system import CoilSet, Biot_Savart_field, CoilFieldBiotSavart
from pyna.MCF.coils.field import VectorField3DCylindrical, VectorField3DAxiSymmetric
from pyna.MCF.coils.RMP import normalize_b, RMP_spectrum_2d, island_width_at_rational_surfaces
from pyna.MCF.coils.vector_potential import CoilFieldVectorPotential
from pyna.MCF.coils.accel import (
    analytic_coil_field_batched_gpu,
    biot_savart_all_coils_gpu,
    CircularCoilTemplate,
    get_template,
)
