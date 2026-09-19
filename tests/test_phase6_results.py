"""Guards the committed Phase 6 artifacts (results/phase6): serialization, formulas, honesty labels."""

import csv
import json
from pathlib import Path

import pytest

from qtraffic.environment import GASOLINE_KG_CO2_PER_LITER, PROXY_LABEL

DIR = Path(__file__).resolve().parents[1] / "results" / "phase6"


@pytest.fixture(scope="module")
def summary():
    if not (DIR / "summary.json").exists():
        pytest.skip("results/phase6 not generated")
    return json.loads((DIR / "summary.json").read_text())


def test_verification_block_passes(summary):
    v = summary["verification"]
    assert v["all_pass"] is True and v["conservation_max_abs_error"] < 1e-6
    assert v["controller_runs_checked"] == 105
    assert v["waiting_time_matches_recorded_phase3_and_phase4"] and v["emergency_travel_time_matches_recorded_phase5"]
    assert v["deterministic_replay_controller_metrics"] and v["deterministic_replay_emergency_metrics"]
    assert summary["LABEL"] == PROXY_LABEL


def test_configuration_labels_proxy_and_separates_source_from_assumption():
    cfg = json.loads((DIR / "configuration.json").read_text())
    assert "PROXY" in cfg["LABEL"]
    src = cfg["environmental_proxy"]["sources"]
    assert src["emission_factor"]["status"].startswith("SOURCE-DERIVED")
    assert src["emission_factor"]["quoted"]["gasoline_g_co2_per_us_gallon"] == 8887.0
    idle = src["idle_fuel_rate"]
    assert idle["status"].startswith("CONFIGURABLE SIMULATION PARAMETER")
    assert idle["source"]["publisher"] == "Argonne National Laboratory"
    assert idle["source"]["title"] == "Idling Reduction Savings Calculator"
    assert idle["source"]["published_examples_gal_per_hour_no_load"]["Ford Focus (gasoline)"] == 0.16
    assert "0.606" in idle["default_derivation"] and idle["default_lph"] == 0.6
    assert idle["sensitivity_values_lph"] == [0.6, 0.9, 1.2, 1.9]  # unchanged scenario parameters
    assert "not independently re-read" in idle["source"]["figures_provenance"]
    statement = cfg["environmental_proxy"]["statement"]
    assert "not measurements" in statement and "SIMULATION PROXY - NOT A MEASUREMENT" in statement
    assert "not a calibrated fleet-average value" in statement
    assert "NOx/PM emissions" in cfg["environmental_proxy"]["not_modelled"]


def test_every_row_satisfies_the_documented_formulas():
    rows = list(csv.DictReader(open(DIR / "controller_comparison.csv")))
    assert len(rows) == 105
    for r in rows:
        wait, rate, ef = (float(r["waiting_vehicle_seconds"]), float(r["idle_fuel_rate_lph_assumed"]),
                          float(r["emission_factor_kg_per_liter_assumed"]))
        assert float(r["fuel_liters_proxy"]) == pytest.approx(wait / 3600.0 * rate, rel=1e-9)
        assert float(r["co2_kg_proxy"]) == pytest.approx(float(r["fuel_liters_proxy"]) * ef, rel=1e-9)
        assert ef == pytest.approx(GASOLINE_KG_CO2_PER_LITER)
        assert float(r["vehicles_demanded"]) == pytest.approx(
            float(r["vehicles_admitted"]) + float(r["vehicles_rejected"]), abs=1e-6)
        if r["controller"] == "fixed":
            assert float(r["waiting_seconds_per_admitted_pct_vs_fixed"]) == 0.0


def test_environmental_file_columns_are_labelled_as_proxy():
    rows = list(csv.DictReader(open(DIR / "environmental_metrics.csv")))
    assert rows and all("PROXY" in r["PROXY_LABEL"] for r in rows)
    first = rows[0]
    assert "fuel_liters_proxy" in first and "co2_kg_proxy" in first and "waiting_vehicle_seconds_measured" in first


def test_sensitivity_shows_comparisons_are_invariant_to_the_coefficients(summary):
    rows = list(csv.DictReader(open(DIR / "sensitivity.csv")))
    assert len({r["parameter_set"] for r in rows}) == 8
    assert max(v["spread_pct_points"] for v in summary["sensitivity_stability_fuel_pct_vs_fixed"].values()) < 1e-9
    assert summary["verification"]["sensitivity_controller_ranking_identical_for_every_coefficient_set"]
    emergency = [r for r in rows if r["kind"] == "emergency_corridor_B_vs_A"]
    assert emergency and all("PROXY" in r["PROXY_LABEL"] for r in emergency)


def test_emergency_files_are_serialised_for_both_modes_and_three_windows():
    rows = list(csv.DictReader(open(DIR / "emergency_metrics.csv")))
    assert {r["mode"] for r in rows} == {"A_no_override", "B_corridor"}
    assert {r["window"] for r in rows} == {"during", "post", "total"}
    cmp_rows = list(csv.DictReader(open(DIR / "emergency_comparison.csv")))
    assert len(cmp_rows) == 30 and all(float(r["A_travel_time_s"]) > 0 for r in cmp_rows)
