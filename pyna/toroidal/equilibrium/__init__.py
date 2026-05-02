"""pyna.toroidal.equilibrium — toroidal MHD equilibrium models.

FEniCSx solvers → :mod:`topoquest.analysis.fem`
Plasma response   → :mod:`topoquest.analysis.response`
Equilibrium ABCs  → stays here in pyna (General-Dynamical-Systems layer)
Analytic models   → stays here (Solov'ev, stellarator)
"""
import warnings

from pyna.toroidal.equilibrium.axisymmetric import EquilibriumAxisym, EquilibriumTokamakCircularSynthetic
from pyna.toroidal.equilibrium.Solovev import EquilibriumSolovev
from pyna.toroidal.equilibrium.GradShafranov import recover_pressure_simplest, solve_GS_perturbed
from pyna.toroidal.equilibrium.stellarator import StellaratorSimple, simple_stellarator

# ── Re-export from topoquest (canonical home) ─────────────────────────
from topoquest.analysis.response.feedback_boozer import (
    BoozerSurface,
    BoozerPerturbation,
    compute_boozer_response,
)
from topoquest.analysis.response.feedback_cylindrical import (
    CylindricalGrid,
    PerturbationField,
    PlasmaResponse,
    compute_plasma_response,
    feedback_correction_field,
    iterative_equilibrium_correction,
)
from topoquest.analysis.response.feedback_cylindrical_utils import (
    greens_function_cylinder,
    lundquist_number,
    toroidal_fft,
    convergence_monitor,
)
