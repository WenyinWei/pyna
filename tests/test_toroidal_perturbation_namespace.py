import importlib


def test_toroidal_perturbation_namespace_is_available():
    perturbation = importlib.import_module("pyna.toroidal.perturbation")
    assert hasattr(perturbation, "beta_ramp")
    assert hasattr(perturbation, "equilibrium")
    assert hasattr(perturbation, "response")
    assert hasattr(perturbation, "BetaRampState")
    assert hasattr(perturbation, "BetaRampScanDiagnostics")
    assert hasattr(perturbation, "diagnose_beta_ramp_scan")


def test_toroidal_perturbation_equilibrium_namespace_declares_bucket_exports():
    ns = importlib.import_module("pyna.toroidal.perturbation.equilibrium")

    assert "FiniteBetaPerturbation" in ns.__all__
    assert "solve_GS_perturbed" in ns.__all__
    assert "solve_force_balance_correction" in ns.__all__


def test_toroidal_beta_ramp_symbols_are_lazy_exported():
    toroidal = importlib.import_module("pyna.toroidal")

    assert hasattr(toroidal, "BetaRampState")
    assert hasattr(toroidal, "diagnose_beta_ramp_state")
    assert hasattr(toroidal, "diagnose_beta_ramp_scan")
    assert hasattr(toroidal, "beta_ramp_states_from_fields")
