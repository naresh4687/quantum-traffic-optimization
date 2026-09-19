"""Tests for the Fixed-vs-Adaptive experiment harness."""

import pytest

from qtraffic import (
    AdaptiveController, DemandConfig, DemandModel, FixedTimeController, Simulator, grid_network,
)
from qtraffic.experiments import (
    CANDIDATES, METRICS, SCENARIOS, TimeVaryingDemand, aggregate, demand_digest, pct_diff,
    plan_stats, run_pair, verdict,
)

NET = grid_network(2, 3)
BY_NAME = {s.name: s for s in SCENARIOS}


def test_required_scenarios_are_defined():
    assert {"low", "medium", "high", "ns_heavy", "ew_heavy", "time_varying"} <= set(BY_NAME)


def test_run_pair_matches_independent_simulator_runs():
    """Nothing is hard-coded: run_pair reproduces a direct Simulator run exactly."""
    sc, seed, cycles = BY_NAME["medium"], 2, 30
    pair = run_pair(sc, seed, cycles)
    demand = sc.build_demand(NET, seed)
    sim = Simulator(NET, demand)
    fixed = sim.run(FixedTimeController(), cycles).metrics
    adaptive = sim.run(AdaptiveController(), cycles).metrics
    for m in METRICS:
        assert pair["fixed"][m] == getattr(fixed, m)
        assert pair["adaptive"][m] == getattr(adaptive, m)


@pytest.mark.parametrize("name", sorted(BY_NAME))
def test_fixed_and_adaptive_see_identical_demand(name):
    sc, seed, cycles = BY_NAME[name], 1, 40
    demand = sc.build_demand(NET, seed)
    sim = Simulator(NET, demand)
    f = sim.run(FixedTimeController(), cycles)
    a = sim.run(AdaptiveController(), cycles)
    assert [r.demand for r in f.history] == [r.demand for r in a.history]
    assert demand_digest([r.demand for r in f.history]) == demand_digest([r.demand for r in a.history])
    assert run_pair(sc, seed, cycles)["demand_digest"] == demand_digest([r.demand for r in f.history])


def test_demand_digest_detects_differences():
    d0 = DemandModel(NET, DemandConfig(seed=0))
    d1 = DemandModel(NET, DemandConfig(seed=1))
    rows0 = [d0.arrivals(c) for c in range(5)]
    assert demand_digest(rows0) == demand_digest([d0.arrivals(c) for c in range(5)])
    assert demand_digest(rows0) != demand_digest([d1.arrivals(c) for c in range(5)])


def test_run_pair_is_reproducible():
    a = run_pair(BY_NAME["time_varying"], 3, 30)
    b = run_pair(BY_NAME["time_varying"], 3, 30)
    assert a == b


def test_time_varying_demand_switches_direction_on_schedule():
    tv = BY_NAME["time_varying"].build_demand(NET, 0)
    assert isinstance(tv, TimeVaryingDemand)

    def ns_share(cycle):
        row = tv.arrivals(cycle)
        ns = sum(v for a, v in row.items() if a.heading.value in "NS")
        return ns / sum(row.values())

    assert ns_share(10) > 0.6   # NS-heavy block
    assert ns_share(70) < 0.4   # EW-heavy block
    assert 0.4 < ns_share(40) < 0.7 and 0.4 < ns_share(100) < 0.7  # balanced blocks


def test_time_varying_demand_validates_segments():
    src = DemandModel(NET)
    with pytest.raises(ValueError):
        TimeVaryingDemand([])
    with pytest.raises(ValueError):
        TimeVaryingDemand([(5, src)])
    with pytest.raises(ValueError):
        TimeVaryingDemand([(0, src), (10, src), (10, src)])


def test_pct_diff_and_verdict():
    assert pct_diff(100.0, 110.0) == pytest.approx(10.0)
    assert pct_diff(100.0, 75.0) == pytest.approx(-25.0)
    assert pct_diff(0.0, 5.0) is None
    assert verdict("total_waiting_time", 100.0, 90.0) == "better"
    assert verdict("total_waiting_time", 100.0, 110.0) == "worse"
    assert verdict("throughput_per_hour", 100.0, 110.0) == "better"
    assert verdict("throughput_per_hour", 100.0, 100.0) == "same"


def test_plan_stats_counts_switches_and_shares():
    pair = run_pair(BY_NAME["ns_heavy"], 0, 20)
    fixed, adaptive = pair["fixed"], pair["adaptive"]
    assert fixed["plan_switches"] == 0 and fixed["share_NS30/EW30"] == 1.0
    assert sum(adaptive[k] for k in ("share_NS20/EW40", "share_NS30/EW30", "share_NS40/EW20")) == pytest.approx(1.0)
    assert adaptive["share_NS40/EW20"] > 0
    assert 0.0 <= adaptive["plan_switch_rate"] <= 1.0


def test_aggregate_is_computed_from_the_pairs():
    pairs = [run_pair(BY_NAME["ns_heavy"], s, 20) for s in (0, 1)]
    row = aggregate(pairs)[0]
    expected = sum(p["fixed"]["vehicles_exited"] for p in pairs) / 2
    assert row["vehicles_exited_fixed"] == pytest.approx(expected)
    assert row["seeds"] == 2 and row["candidate"] == "adaptive"
    assert set(CANDIDATES) == {"adaptive", "adaptive_dwell2"}
