import importlib


def test_toroidal_control_exports_island_stack():
    control = importlib.import_module("pyna.toroidal.control")
    assert hasattr(control, "island_suppression_current")
    assert hasattr(control, "phase_control_current")
    assert hasattr(control, "multi_mode_control")
    assert hasattr(control, "IslandOptimizer")
    assert hasattr(control, "OptimisationResult")


def test_removed_mcf_namespace_no_longer_imports():
    try:
        importlib.import_module("pyna.MCF")
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError("pyna.MCF should not remain importable")


def test_mag_namespace_uses_toroidal_owned_island_exports():
    mag = importlib.import_module("pyna.mag")
    toroidal_control = importlib.import_module("pyna.toroidal.control")

    assert mag.compute_resonant_amplitude is toroidal_control.compute_resonant_amplitude
    assert mag.island_suppression_current is toroidal_control.island_suppression_current


def test_root_package_exposes_preferred_toroidal_namespace():
    pyna_root = importlib.import_module("pyna")

    assert "toroidal" in pyna_root.__all__
    assert callable(pyna_root.__getattr__)
