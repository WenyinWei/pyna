"""pyna.toroidal.coils — toroidal coil geometry and vacuum fields."""
from importlib import import_module

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
from pyna.fields.cylindrical import (
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


def __getattr__(name):
    """Load boundary-local coil construction only when it is requested."""

    module = import_module("pyna.toroidal.coils.boundary_local")
    if name == "boundary_local":
        globals()[name] = module
        return module
    if name in module.__all__:
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
