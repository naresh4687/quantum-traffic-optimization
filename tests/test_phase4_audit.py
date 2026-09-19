"""Guards the committed Phase 4 audit (scripts/audit_qaoa.py -> results/phase4/audit.json)."""

import json
from pathlib import Path

import pytest

AUDIT = Path(__file__).resolve().parents[1] / "results" / "phase4" / "audit.json"


@pytest.fixture(scope="module")
def audit():
    if not AUDIT.exists():
        pytest.skip("results/phase4/audit.json not generated")
    return json.loads(AUDIT.read_text())


def test_audit_passed_with_no_failures(audit):
    assert audit["all_checks_pass"] is True and audit["failures"] == []


def test_primary_p2_and_the_warm_start_ablation_are_labelled_separately(audit):
    roles = audit["variant_roles"]
    assert roles["qaoa_p2"].startswith("primary") and "ramp start" in roles["qaoa_p2"]
    assert roles["qaoa_p2_warm"].startswith("ABLATION ONLY")
    for case in audit["cases"].values():
        primary = case["variants"]["qaoa_p2"]["optimizer"]["initial_angles"]
        warm = case["variants"]["qaoa_p2_warm"]["optimizer"]["initial_angles"]
        assert primary == [0.1875, 0.5625, 0.5625, 0.1875]  # deterministic ramp, same as p=1's rule
        assert warm[1] == 0.0 and warm[3] == 0.0  # zero second layer: the p=1 optimum


def test_no_exact_information_reaches_the_optimiser(audit):
    iso = audit["optimiser_isolation"]
    assert iso["run_qaoa_parameters"] == ["qubo", "config", "initial_angles"]
    assert iso["exact_related_names_in_run_qaoa_or_engine_source"] == []
    assert iso["qaoa_runs_with_exact_solver_and_enumeration_disabled"] is True
    assert iso["angles_identical_to_saved_experiment"] is True


def test_tie_analysis_covers_all_five_cases(audit):
    assert set(audit["cases"]) == {"balanced_medium", "ns_heavy", "ew_heavy", "time_varying", "downstream_congested"}
    counts = {k: v["exact"]["n_optimal_configurations"] for k, v in audit["cases"].items()}
    assert counts == {"balanced_medium": 2, "ns_heavy": 1, "ew_heavy": 1, "time_varying": 1, "downstream_congested": 4}
    assert audit["cases"]["balanced_medium"]["equal_energy_but_different_simulator_performance"]
    assert audit["cases"]["downstream_congested"]["equal_energy_but_different_simulator_performance"]
