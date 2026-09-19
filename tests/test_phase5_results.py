"""Guards the committed Phase 5 experiment artifacts (results/phase5)."""

import csv
import json
from pathlib import Path

import pytest

DIR = Path(__file__).resolve().parents[1] / "results" / "phase5"


@pytest.fixture(scope="module")
def repro():
    if not (DIR / "reproducibility.json").exists():
        pytest.skip("results/phase5 not generated")
    return json.loads((DIR / "reproducibility.json").read_text())


def test_experiment_is_reproducible_and_inputs_were_identical(repro):
    assert repro["primary_all_identical"] is True
    assert repro["demand_identical_between_A_and_B_in_every_pair"] is True
    assert repro["pairs_checked_for_identical_demand"] == 45
    assert repro["pre_emergency_window_identical_A_vs_B"] is True
    cfg = repro["configuration"]
    assert cfg["route"] == ["I1", "I2", "I3", "I6"] and cfg["start_cycle"] == 40 and cfg["link_seconds"] == 30.0


def test_every_pair_completed_and_travel_time_is_free_flow_plus_measured_waiting(repro):
    rows = list(csv.DictReader(open(DIR / "emergency_runs.csv")))
    assert len(rows) == 90 and all(r["completed"] == "True" for r in rows)  # 45 pairs x (A, B)
    for r in rows:
        waits = sum(float(r[k]) for k in r if k.startswith("wait_") and r[k] != "")
        assert float(r["emergency_travel_time_s"]) == pytest.approx(float(r["free_flow_time_s"]) + waits, abs=1e-6)
        assert float(r["emergency_delay_s"]) == pytest.approx(waits, abs=1e-6)


def test_option_a_never_overrides_and_option_b_only_after_the_event_starts():
    rows = list(csv.DictReader(open(DIR / "primary_signal_log.csv")))
    assert rows
    for r in rows:
        if r["mode"] == "A_no_override":
            assert r["priority"] == "False" and r["final_plan"] == r["base_plan"]
        elif int(r["cycle"]) < 40:
            assert r["priority"] == "False" and r["final_plan"] == r["base_plan"]


def test_ablation_is_labelled_as_such():
    rows = list(csv.DictReader(open(DIR / "ablation.csv")))
    assert rows and all("not the primary result" in r["ABLATION"] for r in rows)
