import numpy as np
import pytest

from pyna._cyna.utils import prepare_field_cache
from pyna.topo.fpt import (
    InvariantTorusPerturbationResponse,
    StableManifoldPerturbationResponse,
    compute_cycle_response,
    compute_cycle_response_from_cache,
)


def _constant_toroidal_cache():
    R = np.linspace(0.8, 1.2, 5)
    Z = np.linspace(-0.2, 0.2, 5)
    Phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    shape = (len(R), len(Z), len(Phi))
    cache = {
        "R_grid": R,
        "Z_grid": Z,
        "Phi_grid": Phi,
        "BR": np.zeros(shape),
        "BZ": np.zeros(shape),
        "BPhi": np.ones(shape),
    }
    return prepare_field_cache(cache, extend_phi=True)


def test_cycle_response_zero_perturbation_if_cyna_available():
    import pyna._cyna as cyna

    if not cyna.is_available() or cyna.compute_cycle_perturbation_response is None:
        pytest.skip("cyna cycle perturbation response is unavailable")

    base = _constant_toroidal_cache()
    response = compute_cycle_response_from_cache(
        1.0,
        0.0,
        0.0,
        0.5,
        base,
        base,
        dphi_out=0.1,
        DPhi=0.025,
        fd_eps=1e-4,
    )

    assert response.R.shape == response.Z.shape == response.phi.shape
    assert response.DP.shape[1:] == (2, 2)
    np.testing.assert_allclose(response.delta_X_pol, 0.0, atol=1e-13)
    np.testing.assert_allclose(response.delta_X_cyc, 0.0, atol=1e-13)
    np.testing.assert_allclose(response.periodic_residual, 0.0, atol=1e-13)


def test_cycle_response_accepts_vector_field_if_cyna_available():
    import pyna._cyna as cyna
    from pyna.fields import VectorFieldCylind

    if not cyna.is_available() or cyna.compute_cycle_perturbation_response is None:
        pytest.skip("cyna cycle perturbation response is unavailable")

    R = np.linspace(0.8, 1.2, 5)
    Z = np.linspace(-0.2, 0.2, 5)
    Phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    shape = (len(R), len(Z), len(Phi))
    field = VectorFieldCylind(
        R,
        Z,
        Phi,
        BR=np.zeros(shape),
        BZ=np.zeros(shape),
        BPhi=np.ones(shape),
    )

    response = compute_cycle_response(
        1.0,
        0.0,
        0.0,
        0.5,
        field,
        field,
        dphi_out=0.1,
        DPhi=0.025,
        fd_eps=1e-4,
    )

    np.testing.assert_allclose(response.delta_X_pol, 0.0, atol=1e-13)
    np.testing.assert_allclose(response.delta_X_cyc, 0.0, atol=1e-13)


def test_fpt_placeholders_are_explicit():
    with pytest.raises(NotImplementedError):
        InvariantTorusPerturbationResponse()
    with pytest.raises(NotImplementedError):
        StableManifoldPerturbationResponse()
