"""Tests for the waiting-based fuel / CO2 proxy (units, linearity, validation, provenance)."""

import json
import math

import pytest

from qtraffic.environment import (
    DEFAULT_IDLE_FUEL_RATE_LPH, DIESEL_KG_CO2_PER_LITER, EPA_DIESEL_G_CO2_PER_US_GALLON,
    EPA_GASOLINE_G_CO2_PER_US_GALLON, GASOLINE_KG_CO2_PER_LITER, LITERS_PER_US_GALLON, PROXY_LABEL,
    SECONDS_PER_HOUR, SENSITIVITY_IDLE_FUEL_RATES_LPH, EnvironmentalConfig, co2_kg, estimate, fuel_liters,
    percent_change, sensitivity_configs, waiting_hours,
)


# -- configuration and provenance ------------------------------------------------------------------
def test_defaults_and_the_epa_derived_emission_factor():
    cfg = EnvironmentalConfig()
    assert cfg.idle_fuel_rate_lph == DEFAULT_IDLE_FUEL_RATE_LPH == 0.6
    assert cfg.emission_factor_kg_per_liter == GASOLINE_KG_CO2_PER_LITER and cfg.fuel_type == "gasoline"
    # EPA: 8,887 g CO2 per US gallon of gasoline, 10,180 g per gallon of diesel; 1 gal = 3.785411784 L
    assert EPA_GASOLINE_G_CO2_PER_US_GALLON == 8887.0 and EPA_DIESEL_G_CO2_PER_US_GALLON == 10180.0
    assert GASOLINE_KG_CO2_PER_LITER == pytest.approx(8.887 / 3.785411784)
    assert DIESEL_KG_CO2_PER_LITER == pytest.approx(10.180 / 3.785411784)
    assert 2.34 < GASOLINE_KG_CO2_PER_LITER < 2.36 and 2.68 < DIESEL_KG_CO2_PER_LITER < 2.70
    assert LITERS_PER_US_GALLON == 3.785411784


def test_provenance_separates_source_derived_from_assumption():
    p = EnvironmentalConfig().provenance()
    assert "PROXY" in p["label"] and p["label"] == PROXY_LABEL
    assert p["idle_fuel_rate_status"].startswith("CONFIGURABLE SIMULATION PARAMETER")
    assert "Argonne National Laboratory" in p["idle_fuel_rate_basis"] and "0.606 L/h" in p["idle_fuel_rate_basis"]
    assert "not a calibrated fleet-average" in p["idle_fuel_rate_basis"].lower()
    assert p["emission_factor_basis"].startswith("SOURCE-DERIVED") and "EPA" in p["emission_factor_basis"]
    json.dumps(p)  # serialisable


def test_coefficients_can_be_overridden():
    cfg = EnvironmentalConfig(idle_fuel_rate_lph=1.2, emission_factor_kg_per_liter=2.0, fuel_type="custom")
    assert fuel_liters(3600.0, cfg) == 1.2 and co2_kg(1.5, cfg) == 3.0


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), True, "0.6", None])
def test_negative_or_invalid_coefficients_are_rejected(bad):
    with pytest.raises(ValueError):
        EnvironmentalConfig(idle_fuel_rate_lph=bad)
    with pytest.raises(ValueError):
        EnvironmentalConfig(emission_factor_kg_per_liter=bad)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_negative_or_invalid_inputs_are_rejected(bad):
    with pytest.raises(ValueError):
        waiting_hours(bad)
    with pytest.raises(ValueError):
        fuel_liters(bad)
    with pytest.raises(ValueError):
        co2_kg(bad)


# -- unit conversions ---------------------------------------------------------------------------------------
def test_unit_conversions():
    assert SECONDS_PER_HOUR == 3600.0
    assert waiting_hours(3600.0) == 1.0 and waiting_hours(1800.0) == 0.5
    cfg = EnvironmentalConfig(idle_fuel_rate_lph=0.6)
    assert fuel_liters(3600.0, cfg) == pytest.approx(0.6)  # one vehicle-hour at 0.6 L/h
    assert fuel_liters(1800.0, cfg) == pytest.approx(0.3)
    assert fuel_liters(36000.0, cfg) == pytest.approx(6.0)  # ten vehicle-hours
    assert co2_kg(1.0, EnvironmentalConfig()) == pytest.approx(GASOLINE_KG_CO2_PER_LITER)
    diesel = EnvironmentalConfig(emission_factor_kg_per_liter=DIESEL_KG_CO2_PER_LITER)
    assert co2_kg(1.0, diesel) == pytest.approx(2.6893, abs=1e-4)


def test_fuel_and_co2_of_a_known_amount_of_waiting():
    e = estimate(7200.0, EnvironmentalConfig(idle_fuel_rate_lph=0.6))  # two vehicle-hours
    assert e.fuel_liters == pytest.approx(1.2)
    assert e.co2_kg == pytest.approx(1.2 * GASOLINE_KG_CO2_PER_LITER)
    assert e.waiting_vehicle_seconds == 7200.0 and "PROXY" in e.label


# -- linearity and zero ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("k", [0.5, 2.0, 7.3])
def test_fuel_scales_linearly_with_waiting_and_with_the_idle_rate(k):
    cfg = EnvironmentalConfig()
    assert fuel_liters(k * 5000.0, cfg) == pytest.approx(k * fuel_liters(5000.0, cfg))
    scaled = EnvironmentalConfig(idle_fuel_rate_lph=k * cfg.idle_fuel_rate_lph)
    assert fuel_liters(5000.0, scaled) == pytest.approx(k * fuel_liters(5000.0, cfg))


@pytest.mark.parametrize("k", [0.5, 3.0])
def test_co2_scales_linearly_with_fuel_and_with_the_emission_factor(k):
    cfg = EnvironmentalConfig()
    assert co2_kg(k * 12.0, cfg) == pytest.approx(k * co2_kg(12.0, cfg))
    scaled = EnvironmentalConfig(emission_factor_kg_per_liter=k * cfg.emission_factor_kg_per_liter)
    assert co2_kg(12.0, scaled) == pytest.approx(k * co2_kg(12.0, cfg))


def test_zero_waiting_gives_zero_proxy_and_outputs_are_never_negative():
    e = estimate(0.0)
    assert e.fuel_liters == 0.0 and e.co2_kg == 0.0 and e.per_vehicle(10) == (0.0, 0.0)
    for w in (0.0, 1.0, 123456.7):
        est = estimate(w)
        assert est.fuel_liters >= 0 and est.co2_kg >= 0
    assert estimate(5000.0, EnvironmentalConfig(idle_fuel_rate_lph=0.0)).fuel_liters == 0.0


def test_relative_differences_do_not_depend_on_the_coefficients():
    """The proxy is linear in waiting, so any two runs differ by the same % under every coefficient set."""
    a, b = 40000.0, 46000.0
    changes = set()
    for cfg in sensitivity_configs():
        fa, fb = estimate(a, cfg), estimate(b, cfg)
        changes.add(round(percent_change(fa.fuel_liters, fb.fuel_liters), 9))
        assert percent_change(fa.co2_kg, fb.co2_kg) == pytest.approx(percent_change(a, b))
    assert changes == {round(percent_change(a, b), 9)}


def test_per_vehicle_normalisation():
    e = estimate(3600.0 * 10, EnvironmentalConfig(idle_fuel_rate_lph=0.6))
    fuel_pv, co2_pv = e.per_vehicle(100.0)
    assert fuel_pv == pytest.approx(6.0 / 100.0) and co2_pv == pytest.approx(e.co2_kg / 100.0)
    with pytest.raises(ValueError):
        e.per_vehicle(-1.0)


def test_percent_change():
    assert percent_change(100.0, 110.0) == pytest.approx(10.0)
    assert percent_change(100.0, 75.0) == pytest.approx(-25.0)
    assert percent_change(0.0, 5.0) is None and percent_change(50.0, 50.0) == 0.0


def test_sensitivity_grid_covers_the_documented_range():
    cfgs = sensitivity_configs()
    assert len(cfgs) == len(SENSITIVITY_IDLE_FUEL_RATES_LPH) * 2
    assert {c.idle_fuel_rate_lph for c in cfgs} == set(SENSITIVITY_IDLE_FUEL_RATES_LPH)
    assert {c.fuel_type for c in cfgs} == {"gasoline", "diesel"}
    assert min(SENSITIVITY_IDLE_FUEL_RATES_LPH) == DEFAULT_IDLE_FUEL_RATE_LPH  # default is the low end
    assert all(math.isfinite(c.idle_fuel_rate_lph) for c in cfgs)


def test_module_documents_limitations_and_sources():
    import qtraffic.environment as env

    doc = env.__doc__
    for phrase in ("SIMULATION PROXY", "NOT A MEASUREMENT", "acceleration", "NOx", "epa.gov",
                   "CONFIGURABLE SIMULATION PARAMETER", "Idling Reduction Savings Calculator", "Argonne National Laboratory",
                   "not a calibrated fleet-average value", "Ford Focus", "Crown Victoria", "0.606 L/h"):
        assert phrase in doc


def test_idle_rate_default_matches_the_published_example_conversion():
    from qtraffic.environment import (
        ARGONNE_IDLE_EXAMPLES_GAL_PER_HOUR, DOE_GENERAL_IDLE_RANGE_GAL_PER_HOUR,
        gallons_per_hour_to_liters_per_hour,
    )

    assert ARGONNE_IDLE_EXAMPLES_GAL_PER_HOUR == {
        "Ford Focus (gasoline)": 0.16, "Volkswagen Jetta (diesel)": 0.17, "Ford Crown Victoria (gasoline)": 0.39}
    assert gallons_per_hour_to_liters_per_hour(0.16) == pytest.approx(0.606, abs=1e-3)  # 0.16 gal/h x 3.78541 L/gal
    assert round(gallons_per_hour_to_liters_per_hour(0.16), 1) == DEFAULT_IDLE_FUEL_RATE_LPH  # default is the rounded Focus figure
    assert DEFAULT_IDLE_FUEL_RATE_LPH <= min(gallons_per_hour_to_liters_per_hour(v)
                                             for v in ARGONNE_IDLE_EXAMPLES_GAL_PER_HOUR.values()) + 0.01
    lo, hi = (gallons_per_hour_to_liters_per_hour(v) for v in DOE_GENERAL_IDLE_RANGE_GAL_PER_HOUR)
    assert lo == pytest.approx(0.946, abs=1e-3) and hi == pytest.approx(1.893, abs=1e-3)
    # every sensitivity value lies within the published examples' overall span (0.16 .. 0.5 gal/h)
    lo_all, hi_all = 0.16 * LITERS_PER_US_GALLON, 0.5 * LITERS_PER_US_GALLON
    assert all(lo_all - 0.01 <= v <= hi_all + 0.01 for v in SENSITIVITY_IDLE_FUEL_RATES_LPH)
