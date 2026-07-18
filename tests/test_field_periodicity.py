import numpy as np
import pytest

from pyna.fields import ToroidalPeriodicity


def test_field_period_is_angular_width_not_nfp():
    periodicity = ToroidalPeriodicity(nfp=5)

    assert periodicity.nfp == 5
    assert periodicity.field_period == pytest.approx(2.0 * np.pi / 5.0)
    assert "field_periods" not in periodicity.as_dict()


def test_wrap_maps_exact_field_period_multiples_to_origin():
    periodicity = ToroidalPeriodicity(nfp=5, origin=0.125)
    multiples = periodicity.origin + periodicity.field_period * np.arange(-3, 7)

    np.testing.assert_allclose(
        periodicity.wrap(multiples),
        periodicity.origin,
        atol=2.0e-15,
    )


def test_domain_period_is_distinct_from_physical_field_period():
    periodicity = ToroidalPeriodicity(nfp=5, domain_period=2.0 * np.pi)

    assert periodicity.field_period == pytest.approx(2.0 * np.pi / 5.0)
    assert periodicity.domain_period == pytest.approx(2.0 * np.pi)
    assert periodicity.domain_period_count == 5
    assert periodicity.native_sample_count(40) == 8
    assert periodicity.native_sample_count(41, endpoint_included=True) == 9


def test_domain_period_must_be_integer_field_period_multiple():
    with pytest.raises(ValueError, match="integer number"):
        ToroidalPeriodicity(nfp=5, domain_period=0.3)


def test_periodicity_can_be_recovered_from_field_period():
    periodicity = ToroidalPeriodicity.from_field_period(2.0 * np.pi / 5.0)

    assert periodicity.nfp == 5
    assert periodicity.field_period == pytest.approx(2.0 * np.pi / 5.0)

    with pytest.raises(ValueError, match=r"2\*pi/nfp"):
        ToroidalPeriodicity.from_field_period(0.3)


def test_legacy_field_periods_is_read_only_compatibility_input():
    periodicity = ToroidalPeriodicity.from_object({"field_periods": 5})

    assert periodicity.nfp == 5
    assert periodicity.field_period == pytest.approx(2.0 * np.pi / 5.0)
    assert "field_periods" not in periodicity.as_dict()
