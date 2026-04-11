import importlib


def test_toroidal_control_exports_island_stack():
    control = importlib.import_module("pyna.toroidal.control")
    assert hasattr(control, "island_suppression_current")
    assert hasattr(control, "phase_control_current")
    assert hasattr(control, "multi_mode_control")
    assert hasattr(control, "IslandOptimizer")
    assert hasattr(control, "OptimisationResult")


def test_legacy_mcf_island_modules_forward_to_toroidal():
    legacy_control = importlib.import_module("pyna.MCF.control.island_control")
    toroidal_control = importlib.import_module("pyna.toroidal.control.island_control")
    legacy_optimizer = importlib.import_module("pyna.MCF.control.island_optimizer")
    toroidal_optimizer = importlib.import_module("pyna.toroidal.control.island_optimizer")

    assert legacy_control.compute_resonant_amplitude is toroidal_control.compute_resonant_amplitude
    assert legacy_control.island_suppression_current is toroidal_control.island_suppression_current
    assert legacy_optimizer.IslandOptimizer is toroidal_optimizer.IslandOptimizer
    assert legacy_optimizer.OptimisationResult is toroidal_optimizer.OptimisationResult


def test_mag_namespace_uses_toroidal_owned_island_exports():
    mag = importlib.import_module("pyna.mag")
    toroidal_control = importlib.import_module("pyna.toroidal.control")

    assert mag.compute_resonant_amplitude is toroidal_control.compute_resonant_amplitude
    assert mag.island_suppression_current is toroidal_control.island_suppression_current


def test_root_package_exposes_preferred_toroidal_namespace():
    pyna_root = importlib.import_module("pyna")

    assert "toroidal" in pyna_root.__all__
    assert callable(pyna_root.__getattr__)


def test_legacy_mcf_root_is_only_a_facade_over_toroidal():
    legacy_root = importlib.import_module("pyna.MCF")
    toroidal = importlib.import_module("pyna.toroidal")

    assert legacy_root.EquilibriumSolovev is toroidal.EquilibriumSolovev
    assert legacy_root.mean_radial_displacement is toroidal.mean_radial_displacement
