"""Tests for the Phase 3 validation harness (kept short: few cycles)."""

import numpy as np
import pytest

from qtraffic import FixedTimeController, Simulator, grid_network
from qtraffic.optimization import QuboController, StaticPlanController
from qtraffic.optimization.validation import (
    CASES, CONTROLLER_KEYS, copy_state, evaluate_case, initial_state_for, run_from_state,
    state_digest,
)

BY_NAME = {c.name: c for c in CASES}
NET = grid_network(2, 3)


def test_required_cases_exist():
    assert {"balanced_medium", "ns_heavy", "ew_heavy", "time_varying",
            "downstream_congested"} <= set(BY_NAME)


def test_every_controller_sees_identical_initial_state_and_demand():
    r = evaluate_case(BY_NAME["ns_heavy"], 1, cycles=6, correlation=False)
    rows = r["controllers"]
    assert set(rows) == set(CONTROLLER_KEYS)
    assert len({rows[k]["demand_generated"] for k in rows}) == 1  # same demand stream
    assert r["qubo"]["feasible"] and r["qubo"]["n_evaluated_feasible"] == 729
    assert r["qubo"]["full_space"]["n_evaluated"] == 2**18 and r["qubo"]["full_space"]["min_is_feasible"]
    assert set(r["qubo"]["plans"]) == set(NET.nodes)


def test_evaluate_case_is_reproducible():
    a = evaluate_case(BY_NAME["time_varying"], 2, cycles=5, correlation=False)
    b = evaluate_case(BY_NAME["time_varying"], 2, cycles=5, correlation=False)
    assert a == b


def test_run_from_state_starts_from_the_given_state_and_leaves_it_untouched():
    case = BY_NAME["downstream_congested"]
    sim = Simulator(NET, case.scenario.build_demand(NET, 0))
    start = initial_state_for(case, sim)
    digest = state_digest(start)
    res = run_from_state(sim, start, FixedTimeController(), 3)
    assert state_digest(start) == digest  # not mutated
    assert res.history[0].queue_after_arrivals[next(a for a in NET.approaches if str(a) == "I3:E")] > 30
    assert copy_state(start) is not start


def test_warm_start_state_is_the_same_for_every_call():
    case = BY_NAME["balanced_medium"]
    s1 = initial_state_for(case, Simulator(NET, case.scenario.build_demand(NET, 0)))
    s2 = initial_state_for(case, Simulator(NET, case.scenario.build_demand(NET, 0)))
    assert state_digest(s1) == state_digest(s2) and s1.cycle == case.warmup


def test_qubo_static_row_is_the_decoded_solution_run_through_the_simulator():
    r = evaluate_case(BY_NAME["ew_heavy"], 0, cycles=6, correlation=False)
    case = BY_NAME["ew_heavy"]
    sim = Simulator(NET, case.scenario.build_demand(NET, 0))
    start = initial_state_for(case, sim)
    from qtraffic.optimization import PLANS
    labels = {p.label: p for p in PLANS}
    plans = {n: labels[l] for n, l in r["qubo"]["plans"].items()}
    direct = run_from_state(sim, start, StaticPlanController(plans), 6).metrics
    assert r["controllers"]["qubo_static"]["total_waiting_time"] == direct.total_waiting_time
    assert r["controllers"]["qubo_static"]["vehicles_exited"] == direct.vehicles_exited
    assert r["controllers"]["qubo_static"]["qubo_energy"] == r["qubo"]["energy"]


def test_correlation_analysis_covers_all_729_assignments():
    r = evaluate_case(BY_NAME["balanced_medium"], 0, cycles=4)
    c = r["correlation"]
    assert c["n_assignments"] == 729
    assert 1 <= c["qubo_choice_rank_by_waiting"] <= 729
    assert -1.0 <= c["pearson_energy_vs_waiting"] <= 1.0 and -1.0 <= c["spearman_energy_vs_waiting"] <= 1.0
    assert c["best_static_waiting"] <= c["qubo_choice_waiting"] <= c["worst_static_waiting"]
    # the surrogate is designed to be exact over its own 2-cycle horizon when nothing blocks
    assert c["short_horizon_max_abs_rel_error"] < 1e-9
    assert c["short_horizon_pearson"] == pytest.approx(1.0)


def test_receding_controller_is_a_plain_controller():
    ctrl = QuboController(record=True)
    sim = Simulator(NET, BY_NAME["ns_heavy"].scenario.build_demand(NET, 0))
    start = initial_state_for(BY_NAME["ns_heavy"], sim)
    res = run_from_state(sim, start, ctrl, 3)
    assert len(ctrl.energies) == 3 and np.isfinite(ctrl.energies).all()
    assert res.metrics.cycles == 3
