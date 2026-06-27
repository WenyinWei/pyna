"""pyna.toroidal.coils — toroidal coil geometry and vacuum fields."""
from pyna.toroidal.coils.base import CoilFieldVacuum, CoilFieldSuperposition, CoilFieldScaled
from pyna.toroidal.coils.coil import (
    BRBZ_induced_by_current_loop,
    BRBZ_induced_by_thick_finitelen_solenoid,
    CoilFieldAnalyticCircular,
    CoilFieldAnalyticRectangularSection,
)
from pyna.toroidal.coils.coil_system import (
    CoilSet,
    StellaratorControlCoils,
    Biot_Savart_field,
    CoilFieldBiotSavart,
)
from pyna.toroidal.coils.field import (
    VectorFieldCylind,
    VectorFieldCylindAxisym,
    VectorFieldCylind,
    VectorFieldCylindAxisym,
)
from pyna.toroidal.coils.vector_potential import CoilFieldVectorPotential
from pyna.toroidal.coils.accel import (
    analytic_coil_field_batched_gpu,
    biot_savart_all_coils_gpu,
    CircularCoilTemplate,
    get_template,
)
