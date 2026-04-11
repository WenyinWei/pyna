import importlib


def test_toroidal_perturbation_namespace_is_available():
    perturbation = importlib.import_module("pyna.toroidal.perturbation")
    assert hasattr(perturbation, "equilibrium")
    assert hasattr(perturbation, "response")


def test_toroidal_perturbation_equilibrium_namespace_declares_bucket_exports():
    ns = importlib.import_module("pyna.toroidal.perturbation.equilibrium")

    assert "FiniteBetaPerturbation" in ns.__all__
    assert "solve_GS_perturbed" in ns.__all__
    assert "solve_force_balance_correction" in ns.__all__

