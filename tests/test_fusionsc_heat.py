import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pyna.fields import VectorFieldCylind
from pyna.toroidal.control.fusionsc_heat import (
    FusionSCBackendUnavailableError,
    FusionSCComputedField,
    FusionSCEnsembleHeatModel,
    FusionSCFieldLineDiffusionHeatModel,
    FusionSCSeedSpec,
    FusionSCTopologyGuidedHeatModel,
    FusionSCTraceError,
    FusionSCTransportSpec,
    FusionSCWallSurfaceSpec,
    fusionsc_computed_field_from_cylindrical,
    load_fusionsc_rz_section_wall,
)
from pyna.toroidal.control.strike_heat import StrikeSeedBundle


def _wall_surface():
    phi = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    PP, TT = np.meshgrid(phi, theta, indexing="ij")
    R = 2.0 + 0.32 * (1.0 + 0.08 * np.cos(PP)) * np.cos(TT)
    Z = 0.28 * (1.0 + 0.06 * np.sin(PP)) * np.sin(TT) + 0.025 * np.sin(PP)
    return FusionSCWallSurfaceSpec(phi_values=phi, R=R, Z=Z)


def _computed_field():
    shape = (2, 3, 4)
    B_phi = np.arange(np.prod(shape), dtype=float).reshape(shape) + 1.0
    B_z = B_phi + 100.0
    B_R = -B_phi - 200.0
    grid = {
        "rMin": 1.0,
        "rMax": 3.0,
        "zMin": -1.0,
        "zMax": 1.0,
        "nSym": 1,
        "nPhi": shape[0],
        "nZ": shape[1],
        "nR": shape[2],
    }
    return FusionSCComputedField(grid=grid, B_phi=B_phi, B_z=B_z, B_R=B_R)


def test_cylindrical_grid_conversion_enforces_current_fusionsc_axes_and_components():
    R = np.linspace(1.0, 2.0, 4)
    Z = np.linspace(-0.4, 0.4, 3)
    phi = np.linspace(0.0, np.pi, 5, endpoint=False)
    RR, ZZ, PP = np.meshgrid(R, Z, phi, indexing="ij")
    field = VectorFieldCylind(
        R=R,
        Z=Z,
        Phi=phi,
        BR=RR + 2.0 * PP,
        BZ=ZZ - PP,
        BPhi=3.0 + RR,
        nfp=2,
    )

    converted = fusionsc_computed_field_from_cylindrical(field)

    assert converted.tensor.shape == (5, 3, 4, 3)
    np.testing.assert_allclose(converted.tensor[..., 0], np.transpose(field.BPhi, (2, 1, 0)))
    np.testing.assert_allclose(converted.tensor[..., 1], np.transpose(field.BZ, (2, 1, 0)))
    np.testing.assert_allclose(converted.tensor[..., 2], np.transpose(field.BR, (2, 1, 0)))
    assert converted.grid["nSym"] == 2


def test_headered_rz_wall_loader_tiles_a_field_period_to_full_torus(tmp_path):
    nfp, n_poloidal, n_phi = 2, 5, 4
    theta = np.linspace(0.0, 2.0 * np.pi, n_poloidal, endpoint=False)
    rows = []
    for index in range(n_phi):
        rows.extend(np.column_stack([2.0 + 0.3 * np.cos(theta), 0.2 * np.sin(theta)]))
    path = tmp_path / "wall.txt"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{nfp} {n_poloidal} {n_phi}\n")
        np.savetxt(handle, np.asarray(rows))

    wall = load_fusionsc_rz_section_wall(path)

    assert wall.R.shape == (nfp * n_phi, n_poloidal)
    assert wall.phi_values[-1] < 2.0 * np.pi
    assert wall.toroidal_period == pytest.approx(2.0 * np.pi)
    np.testing.assert_allclose(wall.R[:n_phi], wall.R[n_phi:])


def _transport():
    return FusionSCTransportSpec(
        isotropic_diffusion_coefficient=0.04,
        parallel_convection_velocity=2.5,
        mean_free_path=0.3,
        distance_limit=25.0,
        step_size=0.01,
    )


def _install_fake_fusionsc(monkeypatch, captured):
    config_token = object()
    geometry_token = object()

    class MagneticConfig:
        @staticmethod
        def fromComputed(grid, field):
            captured["computed_grid"] = grid
            captured["computed_tensor"] = np.asarray(field)
            return config_token

    class Geometry:
        @staticmethod
        def quadMesh(vertices, wrapU=False, wrapV=False):
            captured["mesh_vertices"] = np.asarray(vertices)
            captured["mesh_wrap_u"] = wrapU
            captured["mesh_wrap_v"] = wrapV
            return geometry_token

    def unused_trace(*args, **kwargs):
        raise AssertionError("the injected trace function was not used")

    fake = SimpleNamespace(
        magnetics=SimpleNamespace(MagneticConfig=MagneticConfig),
        geometry=SimpleNamespace(Geometry=Geometry),
        flt=SimpleNamespace(trace=unused_trace),
    )
    monkeypatch.setitem(sys.modules, "fusionsc", fake)
    return config_token, geometry_token


def _collision_endpoints(wall, indices):
    phi = np.array([wall.phi_values[i_phi] for i_phi, _i_pol in indices])
    R = np.array([wall.R[i_phi, i_pol] for i_phi, i_pol in indices])
    Z = np.array([wall.Z[i_phi, i_pol] for i_phi, i_pol in indices])
    return np.vstack((R * np.cos(phi), R * np.sin(phi), Z, np.ones(phi.size)))


def _evaluate(model):
    return model.evaluate(object(), object(), object(), (), ())


def test_current_fusionsc_component_point_and_full_wall_contracts(monkeypatch):
    captured = {}
    config_token, geometry_token = _install_fake_fusionsc(monkeypatch, captured)
    wall = _wall_surface()
    field = _computed_field()
    seeds = FusionSCSeedSpec(
        R=np.array([1.85, 2.1]),
        Z=np.array([0.03, -0.07]),
        phi=np.array([0.5 * np.pi, 1.5 * np.pi]),
        weights=np.array([1.0, 3.0]),
    )
    endpoints = _collision_endpoints(wall, ((1, 0), (3, 4)))

    def trace(points, config, geometry=None, **kwargs):
        captured["trace_points"] = np.asarray(points)
        captured["trace_config"] = config
        captured["trace_geometry"] = geometry
        captured["trace_kwargs"] = kwargs
        return {
            "endPoints": endpoints,
            "stopReasons": np.array(["collisionLimit", "collisionLimit"], dtype=object),
        }

    model = FusionSCFieldLineDiffusionHeatModel(
        field_builder=lambda case, request: field,
        wall=wall,
        seeds=seeds,
        transport=_transport(),
        total_power=8.0,
        n_phi_bins=4,
        n_s_bins=8,
        trace_function=trace,
    )

    state = _evaluate(model)

    tensor = captured["computed_tensor"]
    assert tensor.shape == field.B_phi.shape + (3,)
    np.testing.assert_allclose(tensor[..., 0], field.B_phi)
    np.testing.assert_allclose(tensor[..., 1], field.B_z)
    np.testing.assert_allclose(tensor[..., 2], field.B_R)
    assert captured["computed_grid"] is field.grid

    np.testing.assert_allclose(captured["trace_points"], seeds.cartesian_points)
    assert captured["trace_config"] is config_token
    assert captured["trace_geometry"] is geometry_token
    np.testing.assert_allclose(captured["mesh_vertices"], wall.cartesian_vertices)
    assert captured["mesh_vertices"].shape == (3, wall.R.shape[0], wall.R.shape[1])
    assert captured["mesh_wrap_u"] is True
    assert captured["mesh_wrap_v"] is True

    kwargs = captured["trace_kwargs"]
    assert kwargs["isotropicDiffusionCoefficient"] == pytest.approx(0.04)
    assert kwargs["parallelConvectionVelocity"] == pytest.approx(2.5)
    assert "rzDiffusionCoefficient" not in kwargs
    assert "parallelDiffusionCoefficient" not in kwargs
    assert kwargs["collisionLimit"] == 1
    vertices = wall.cartesian_vertices.reshape(3, -1)
    geometry_grid = kwargs["geometryGrid"]
    assert geometry_grid["xMin"] < float(np.min(vertices[0]))
    assert geometry_grid["xMax"] > float(np.max(vertices[0]))

    row_power = np.sum(state.heat * state.cell_areas, axis=1)
    np.testing.assert_allclose(row_power, [0.0, 2.0, 0.0, 6.0], atol=1.0e-12)
    assert state.metadata["field_component_order"] == ("Bphi", "Bz", "Br")
    assert state.metadata["trace_coordinates"] == "cartesian_xyz"
    assert state.metadata["endpoint_coordinates"] == "cartesian_xyz_converted_to_cylindrical"
    assert state.metadata["wall_geometry"] == "full_3d_quad_mesh"


def test_heat_flux_is_area_normalized_to_requested_total_power(monkeypatch):
    captured = {}
    _install_fake_fusionsc(monkeypatch, captured)
    wall = _wall_surface()
    seeds = FusionSCSeedSpec(
        R=[1.8, 1.9, 2.0],
        Z=[0.0, 0.02, -0.03],
        phi=[0.0, np.pi, 1.5 * np.pi],
        weights=[2.0, 1.0, 4.0],
    )

    def trace(points, config, geometry=None, **kwargs):
        del points, config, geometry, kwargs
        return {
            "endPoints": _collision_endpoints(wall, ((0, 1), (2, 3), (3, 6))),
            "stopReasons": np.full(3, "collisionLimit", dtype=object),
        }

    model = FusionSCFieldLineDiffusionHeatModel(
        field_builder=lambda case, request: _computed_field(),
        wall=wall,
        seeds=seeds,
        transport=FusionSCTransportSpec(
            rz_diffusion_coefficient=0.015,
            parallel_diffusion_coefficient=4.0,
        ),
        total_power=13.5,
        n_phi_bins=4,
        n_s_bins=12,
        trace_function=trace,
    )

    state = _evaluate(model)

    assert state.cell_areas.shape == state.heat.shape == (4, 12)
    assert np.all(np.isfinite(state.heat))
    assert np.all(state.cell_areas > 0.0)
    assert float(np.sum(state.heat * state.cell_areas)) == pytest.approx(13.5)
    assert state.metadata["actual_diffusive_trace"] is True
    assert state.metadata["wall_collision_verified"] is True
    assert state.metadata["ballistic_fallback"] is False
    assert state.metadata["perpendicular_diffusion_model"] == "rz"
    assert state.metadata["parallel_transport_model"] == "diffusion"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"parallel_convection_velocity": 1.0}, "exactly one of isotropic"),
        (
            {
                "isotropic_diffusion_coefficient": 0.1,
                "rz_diffusion_coefficient": 0.2,
                "parallel_convection_velocity": 1.0,
            },
            "exactly one of isotropic",
        ),
        ({"isotropic_diffusion_coefficient": 0.1}, "exactly one of parallel"),
        (
            {
                "isotropic_diffusion_coefficient": 0.1,
                "parallel_convection_velocity": 1.0,
                "parallel_diffusion_coefficient": 2.0,
            },
            "exactly one of parallel",
        ),
        (
            {
                "isotropic_diffusion_coefficient": 0.1,
                "parallel_convection_velocity": 1.0,
                "collision_limit": 2,
            },
            "first wall hit",
        ),
    ],
)
def test_transport_spec_cannot_represent_ballistic_or_ambiguous_trace(kwargs, message):
    with pytest.raises(ValueError, match=message):
        FusionSCTransportSpec(**kwargs)


def test_non_collision_trace_result_is_rejected(monkeypatch):
    captured = {}
    _install_fake_fusionsc(monkeypatch, captured)
    wall = _wall_surface()
    seeds = FusionSCSeedSpec(R=[1.9], Z=[0.0], phi=[0.0])

    def trace(points, config, geometry=None, **kwargs):
        del points, config, geometry, kwargs
        return {
            "endPoints": _collision_endpoints(wall, ((0, 0),)),
            "stopReasons": np.array(["distanceLimit"], dtype=object),
        }

    model = FusionSCFieldLineDiffusionHeatModel(
        field_builder=lambda case, request: _computed_field(),
        wall=wall,
        seeds=seeds,
        transport=_transport(),
        trace_function=trace,
    )

    with pytest.raises(FusionSCTraceError, match="no diffusive trace reached the wall"):
        _evaluate(model)


def test_partial_collisions_preserve_unresolved_power_accounting(monkeypatch):
    captured = {}
    _install_fake_fusionsc(monkeypatch, captured)
    wall = _wall_surface()
    seeds = FusionSCSeedSpec(
        R=[1.8, 1.9, 2.0, 2.1],
        Z=[0.0, 0.02, -0.03, 0.01],
        phi=[0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi],
        weights=[1.0, 1.0, 1.0, 1.0],
    )

    def trace(points, config, geometry=None, **kwargs):
        del points, config, geometry, kwargs
        return {
            "endPoints": _collision_endpoints(wall, ((0, 1), (1, 2), (2, 3), (3, 4))),
            "stopReasons": np.array(
                ["collisionLimit", "distanceLimit", "collisionLimit", "distanceLimit"],
                dtype=object,
            ),
        }

    model = FusionSCFieldLineDiffusionHeatModel(
        field_builder=lambda case, request: _computed_field(),
        wall=wall,
        seeds=seeds,
        transport=_transport(),
        total_power=12.0,
        minimum_collision_fraction=0.5,
        trace_function=trace,
    )

    state = _evaluate(model)

    assert float(np.sum(state.heat * state.cell_areas)) == pytest.approx(6.0)
    assert state.metadata["wall_collision_count"] == 2
    assert state.metadata["wall_collision_power_fraction"] == pytest.approx(0.5)
    assert state.metadata["deposited_power"] == pytest.approx(6.0)
    assert state.metadata["unresolved_power"] == pytest.approx(6.0)
    assert state.metadata["renormalize_deposited_power"] is False


def test_ensemble_averages_diffusive_heat_and_reports_uncertainty(monkeypatch):
    captured = {}
    _install_fake_fusionsc(monkeypatch, captured)
    wall = _wall_surface()
    seeds = FusionSCSeedSpec(
        R=[1.8, 1.9, 2.0, 2.1],
        Z=[0.0, 0.02, -0.03, 0.01],
        phi=[0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi],
    )
    collision_counts = iter((2, 3, 4))

    def trace(points, config, geometry=None, **kwargs):
        del points, config, geometry, kwargs
        count = next(collision_counts)
        reasons = np.full(4, "distanceLimit", dtype=object)
        reasons[:count] = "collisionLimit"
        return {
            "endPoints": _collision_endpoints(wall, ((0, 1), (1, 2), (2, 3), (3, 4))),
            "stopReasons": reasons,
        }

    member = FusionSCFieldLineDiffusionHeatModel(
        field_builder=lambda case, request: _computed_field(),
        wall=wall,
        seeds=seeds,
        transport=_transport(),
        total_power=8.0,
        n_phi_bins=4,
        n_s_bins=8,
        minimum_collision_fraction=0.25,
        trace_function=trace,
    )
    ensemble = FusionSCEnsembleHeatModel(member, repeats=3)

    state = _evaluate(ensemble)

    assert float(np.sum(state.heat * state.cell_areas)) == pytest.approx(6.0)
    assert state.metadata["model"] == "fusionsc_field_line_diffusion_ensemble"
    assert state.metadata["ensemble_repeats_successful"] == 3
    assert state.metadata["ensemble_repeats_failed"] == 0
    assert state.metadata["deposited_power"] == pytest.approx(6.0)
    assert state.metadata["deposited_power_standard_deviation"] == pytest.approx(2.0)
    assert state.metadata["deposited_power_standard_error"] == pytest.approx(2.0 / np.sqrt(3.0))
    assert state.metadata["wall_collision_power_fraction"] == pytest.approx(0.75)
    assert state.metadata["wall_collision_power_fraction_standard_deviation"] == pytest.approx(0.25)
    assert state.metadata["heat_flux_standard_error"].shape == state.heat.shape
    assert np.any(state.metadata["heat_flux_standard_error"] > 0.0)


def test_ensemble_can_skip_trace_failures_but_enforces_minimum_successes(monkeypatch):
    captured = {}
    _install_fake_fusionsc(monkeypatch, captured)
    wall = _wall_surface()
    seeds = FusionSCSeedSpec(R=[1.8, 1.9], Z=[0.0, 0.02], phi=[0.0, np.pi])
    outcomes = iter((True, False, True))

    def trace(points, config, geometry=None, **kwargs):
        del points, config, geometry, kwargs
        collided = next(outcomes)
        reasons = (
            np.full(2, "collisionLimit", dtype=object)
            if collided
            else np.full(2, "distanceLimit", dtype=object)
        )
        return {
            "endPoints": _collision_endpoints(wall, ((0, 1), (2, 3))),
            "stopReasons": reasons,
        }

    member = FusionSCFieldLineDiffusionHeatModel(
        field_builder=lambda case, request: _computed_field(),
        wall=wall,
        seeds=seeds,
        transport=_transport(),
        trace_function=trace,
    )
    state = _evaluate(
        FusionSCEnsembleHeatModel(
            member,
            repeats=3,
            minimum_successful_repeats=2,
        )
    )

    assert state.metadata["ensemble_repeats_successful"] == 2
    assert state.metadata["ensemble_repeats_failed"] == 1
    assert state.metadata["ensemble_failure_counts"] == {"FusionSCTraceError": 1}


def test_ensemble_requires_repeats_for_uncertainty_estimate():
    member = FusionSCFieldLineDiffusionHeatModel(
        field_builder=lambda case, request: _computed_field(),
        wall=_wall_surface(),
        seeds=FusionSCSeedSpec(R=[1.9], Z=[0.0], phi=[0.0]),
        transport=_transport(),
    )
    with pytest.raises(ValueError, match="at least two"):
        FusionSCEnsembleHeatModel(member, repeats=1)


def test_trace_failure_is_not_retried_without_diffusion(monkeypatch):
    captured = {}
    _install_fake_fusionsc(monkeypatch, captured)
    calls = []

    def trace(points, config, geometry=None, **kwargs):
        del points, config, geometry
        calls.append(dict(kwargs))
        raise RuntimeError("backend failed")

    model = FusionSCFieldLineDiffusionHeatModel(
        field_builder=lambda case, request: _computed_field(),
        wall=_wall_surface(),
        seeds=FusionSCSeedSpec(R=[1.9], Z=[0.0], phi=[0.0]),
        transport=_transport(),
        trace_function=trace,
    )

    with pytest.raises(FusionSCTraceError, match="ballistic fallback is disabled") as error:
        _evaluate(model)

    assert isinstance(error.value.__cause__, RuntimeError)
    assert len(calls) == 1
    assert calls[0]["isotropicDiffusionCoefficient"] == pytest.approx(0.04)
    assert calls[0]["parallelConvectionVelocity"] == pytest.approx(2.5)


def test_missing_optional_backend_raises_clear_error(monkeypatch):
    import pyna.toroidal.control.fusionsc_heat as module

    def missing(name):
        assert name == "fusionsc"
        raise ModuleNotFoundError("not installed")

    monkeypatch.setattr(module.importlib, "import_module", missing)
    model = FusionSCFieldLineDiffusionHeatModel(
        field_builder=lambda case, request: _computed_field(),
        wall=_wall_surface(),
        seeds=FusionSCSeedSpec(R=[1.9], Z=[0.0], phi=[0.0]),
        transport=_transport(),
    )

    with pytest.raises(FusionSCBackendUnavailableError, match="no ballistic fallback"):
        _evaluate(model)


def _strike_bundle(
    label,
    direction,
    weights,
    *,
    weight_kind="relative",
    quantitative=True,
):
    weights = np.asarray(weights, dtype=float)
    count = weights.size
    return StrikeSeedBundle(
        label=label,
        mode="chaotic_manifold",
        R=np.linspace(1.85, 1.95, count),
        Z=np.linspace(-0.02, 0.02, count),
        phi=np.linspace(0.0, 0.2, count),
        direction=direction,
        weights=weights,
        weight_kind=weight_kind,
        source_coordinate=np.arange(count, dtype=float),
        metadata={
            "quantitative": quantitative,
            "topology_provenance": f"{label}.fixed_point_manifold",
            "weight_provenance": f"{label}.flux_tube_power",
        },
    )


def _directed_transport(direction):
    return FusionSCTransportSpec(
        isotropic_diffusion_coefficient=0.04,
        parallel_convection_velocity=2.5,
        mean_free_path=0.3,
        distance_limit=25.0,
        step_size=0.01,
        direction=direction,
    )


def test_topology_guided_model_dispatches_explicit_phi_directions(monkeypatch):
    captured = {}
    _install_fake_fusionsc(monkeypatch, captured)
    wall = _wall_surface()
    bundles = (
        _strike_bundle("unstable.plus", "+", [1.0, 3.0]),
        _strike_bundle("stable.minus", "-", [2.0]),
    )
    context = tuple(object() for _ in range(5))
    resolver_calls = []
    field_builder_calls = []
    trace_calls = []

    def bundle_resolver(case, request, spectrum, chains, intervals):
        resolver_calls.append((case, request, spectrum, chains, intervals))
        return bundles

    def trace(points, config, geometry=None, **kwargs):
        del config, geometry
        count = points.shape[1]
        trace_calls.append((kwargs["direction"], count))
        section = 1 if kwargs["direction"] == "backward" else 3
        return {
            "endPoints": _collision_endpoints(wall, [(section, index) for index in range(count)]),
            "stopReasons": np.full(count, "collisionLimit", dtype=object),
        }

    def field_builder(case, request):
        field_builder_calls.append((case, request))
        return _computed_field()

    model = FusionSCTopologyGuidedHeatModel(
        field_builder=field_builder,
        wall=wall,
        seed_bundles=bundle_resolver,
        transport_by_direction={
            "+": _directed_transport("backward"),
            "-": _directed_transport("cw"),
        },
        transport_provenance_by_direction={
            "+": {"provenance": "validated plus mapping", "quantitative": True},
            "-": {"provenance": "validated minus mapping", "quantitative": True},
        },
        total_power=12.0,
        n_phi_bins=4,
        n_s_bins=8,
        trace_function=trace,
    )

    state = model.evaluate(*context)

    assert resolver_calls == [context]
    assert field_builder_calls == [context[:2]]
    assert trace_calls == [("backward", 2), ("cw", 1)]
    assert state.metadata["phi_direction_dispatch"]["+"]["allocated_power"] == pytest.approx(8.0)
    assert state.metadata["phi_direction_dispatch"]["-"]["allocated_power"] == pytest.approx(4.0)
    assert state.metadata["phi_direction_dispatch"]["+"]["fusionsc_trace_direction"] == "backward"
    assert state.metadata["phi_direction_dispatch"]["-"]["fusionsc_trace_direction"] == "cw"
    assert float(np.sum(state.heat * state.cell_areas)) == pytest.approx(12.0)
    assert state.metadata["unresolved_power"] == pytest.approx(0.0)
    assert state.metadata["quantitative"] is True
    assert state.metadata["proxy"] is False
    provenance = state.metadata["topology_seed_provenance"]
    assert [item["label"] for item in provenance] == ["unstable.plus", "stable.minus"]
    assert provenance[0]["topology_provenance"] == "unstable.plus.fixed_point_manifold"

    ensemble = FusionSCEnsembleHeatModel(model, repeats=2)
    ensemble_state = ensemble.evaluate(*context)
    assert ensemble_state.metadata["model"] == "fusionsc_topology_guided_heat_ensemble"
    assert ensemble_state.metadata["ensemble_repeats_successful"] == 2


def test_topology_guided_absolute_power_preserves_unresolved_power(monkeypatch):
    captured = {}
    _install_fake_fusionsc(monkeypatch, captured)
    wall = _wall_surface()
    bundles = (
        _strike_bundle("plus", "+", [2.0, 3.0], weight_kind="power"),
        _strike_bundle("minus", "-", [5.0], weight_kind="power"),
    )

    def trace(points, config, geometry=None, **kwargs):
        del config, geometry
        count = points.shape[1]
        reasons = np.full(count, "collisionLimit", dtype=object)
        if kwargs["direction"] == "backward":
            reasons[-1] = "distanceLimit"
        return {
            "endPoints": _collision_endpoints(wall, [(0, index) for index in range(count)]),
            "stopReasons": reasons,
        }

    model = FusionSCTopologyGuidedHeatModel(
        field_builder=lambda case, request: _computed_field(),
        wall=wall,
        seed_bundles=bundles,
        transport_by_direction={
            "+": _directed_transport("backward"),
            "-": _directed_transport("forward"),
        },
        minimum_collision_fraction=0.3,
        n_phi_bins=4,
        n_s_bins=8,
        trace_function=trace,
    )

    state = _evaluate(model)

    deposited = float(np.sum(state.heat * state.cell_areas))
    assert deposited == pytest.approx(7.0)
    assert state.metadata["launched_power"] == pytest.approx(10.0)
    assert state.metadata["deposited_power"] == pytest.approx(7.0)
    assert state.metadata["unresolved_power"] == pytest.approx(3.0)
    assert deposited + state.metadata["unresolved_power"] == pytest.approx(10.0)
    assert state.metadata["topology_weight_kind"] == "power"
    assert state.metadata["quantitative"] is False
    assert state.metadata["transport_quantitative"] is False


def test_topology_guided_quantitative_flag_requires_topology_weights_and_transport(monkeypatch):
    captured = {}
    _install_fake_fusionsc(monkeypatch, captured)
    wall = _wall_surface()

    def trace(points, config, geometry=None, **kwargs):
        del config, geometry, kwargs
        return {
            "endPoints": _collision_endpoints(wall, [(2, 2)] * points.shape[1]),
            "stopReasons": np.full(points.shape[1], "collisionLimit", dtype=object),
        }

    model = FusionSCTopologyGuidedHeatModel(
        field_builder=lambda case, request: _computed_field(),
        wall=wall,
        seed_bundles=(_strike_bundle("proxy-weights", "+", [1.0], quantitative=False),),
        transport_by_direction={"+": _directed_transport("ccw")},
        transport_provenance_by_direction={
            "+": {"provenance": "validated transport", "quantitative": True}
        },
        total_power=3.0,
        trace_function=trace,
    )

    state = _evaluate(model)

    assert state.metadata["topology_geometry_quantitative"] is False
    assert state.metadata["topology_weights_quantitative"] is False
    assert state.metadata["transport_quantitative"] is True
    assert state.metadata["quantitative"] is False
    assert state.metadata["proxy"] is True


def test_topology_guided_model_rejects_mixed_missing_and_mismatched_power(monkeypatch):
    relative = _strike_bundle("relative", "+", [1.0])
    absolute = _strike_bundle("absolute", "-", [2.0], weight_kind="power")
    common = {
        "field_builder": lambda case, request: _computed_field(),
        "wall": _wall_surface(),
        "transport_by_direction": {
            "+": _directed_transport("forward"),
            "-": _directed_transport("backward"),
        },
    }

    with pytest.raises(ValueError, match="cannot be mixed"):
        FusionSCTopologyGuidedHeatModel(
            seed_bundles=(relative, absolute), total_power=3.0, **common
        )

    with pytest.raises(ValueError, match="cannot renormalize absolute"):
        FusionSCTopologyGuidedHeatModel(
            seed_bundles=(absolute,), total_power=3.0, **common
        )

    with pytest.raises(ValueError, match="explicit '\\+' or '-'"):
        FusionSCTopologyGuidedHeatModel(
            field_builder=common["field_builder"],
            wall=common["wall"],
            seed_bundles=(relative,),
            transport_by_direction={"forward": _directed_transport("forward")},
            total_power=1.0,
        )

    captured = {}
    _install_fake_fusionsc(monkeypatch, captured)
    with pytest.raises(ValueError, match="missing FusionSC transport mapping"):
        FusionSCTopologyGuidedHeatModel(
            field_builder=common["field_builder"],
            wall=common["wall"],
            seed_bundles=(relative, _strike_bundle("relative-minus", "-", [1.0])),
            transport_by_direction={"+": _directed_transport("forward")},
            total_power=2.0,
        )
