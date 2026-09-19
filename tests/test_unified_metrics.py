"""Tests for the unified metrics layer (traffic + signal + emergency + environmental proxy)."""

import pytest

from qtraffic import (
    AdaptiveController, DemandConfig, DemandModel, FixedTimeController, Observation, Simulator,
    grid_network,
)
from qtraffic.emergency import EmergencyEvent, run_emergency
from qtraffic.environment import EnvironmentalConfig, estimate, percent_change
from qtraffic.metrics import compute_metrics
from qtraffic.optimization import QuboController, StaticPlanController, TrafficState, build_qubo, solve_exact
from qtraffic.simulator import TrafficState as SimState
from qtraffic.unified_metrics import relative_changes, unified_metrics
from tests.helpers import CyclingController

NET = grid_network(2, 3)
ROUTE = ("I1", "I2", "I3", "I6")


def sim(seed=0, level="high"):
    return Simulator(NET, DemandModel(NET, DemandConfig.from_level(level, seed=seed)))


def warm(s, n=10):
    st, fx = s.initial_state(), FixedTimeController()
    for _ in range(n):
        s.step(st, fx)
    return st


def run_from(s, ctrl, cycles=30, start=None):
    st = s.initial_state() if start is None else SimState(start.cycle, dict(start.queues), dict(start.in_transit))
    ctrl.reset()
    return tuple(s.step(st, ctrl) for _ in range(cycles)), st


# -- reuse, not redefinition -------------------------------------------------------------------------------
def test_traffic_view_is_exactly_the_existing_metrics():
    s = sim()
    hist, _ = run_from(s, FixedTimeController())
    m, u = compute_metrics(hist), unified_metrics(hist).traffic
    assert u.vehicles_demanded == m.demand_generated and u.vehicles_admitted == m.vehicles_entered
    assert u.vehicles_rejected == m.vehicles_rejected and u.vehicles_served_stop_line == m.vehicles_served
    assert u.vehicles_exited == m.vehicles_exited and u.throughput_vehicles_per_hour == m.throughput_per_hour
    assert u.blocked_vehicle_cycles == m.total_blocked and u.waiting_vehicle_seconds == m.total_waiting_time
    assert u.average_queue_vehicles == m.average_queue_length and u.max_queue_vehicles == m.max_queue_length
    assert u.final_queue_vehicles == m.final_queue_total and u.in_transit_final_vehicles == m.in_transit_final


# -- consistency with the simulator's semantics ---------------------------------------------------------
@pytest.mark.parametrize("warmup", [0, 10])
def test_demand_admitted_rejected_and_the_exact_conservation_identity(warmup):
    s = sim()
    start = warm(s, warmup) if warmup else None
    hist, st = run_from(s, FixedTimeController(), 40, start)
    u = unified_metrics(hist).traffic
    assert u.vehicles_demanded == pytest.approx(u.vehicles_admitted + u.vehicles_rejected)
    q0 = (sum(start.queues.values()) + sum(start.in_transit.values())) if start else 0.0
    res_queue = sum(st.queues.values()) + sum(st.in_transit.values())
    assert q0 + u.vehicles_admitted == pytest.approx(u.vehicles_exited + res_queue)
    assert u.final_queue_vehicles + u.in_transit_final_vehicles == pytest.approx(res_queue)


def test_served_is_not_exited_and_throughput_uses_exited():
    hist, _ = run_from(sim(), FixedTimeController(), 40)
    u = unified_metrics(hist).traffic
    assert u.vehicles_served_stop_line > u.vehicles_exited  # vehicles cross several stop lines
    assert u.throughput_vehicles_per_hour == pytest.approx(u.vehicles_exited / u.simulated_hours)
    assert u.simulated_hours == pytest.approx(40 * 60 / 3600)


def test_non_negativity_of_every_measured_and_proxy_quantity():
    for ctrl in (FixedTimeController(), AdaptiveController(), CyclingController()):
        hist, _ = run_from(sim(), ctrl, 30)
        u = unified_metrics(hist)
        for k, v in u.flat().items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and "pct" not in k:
                assert v >= 0, k
        assert all(q >= 0 for r in hist for q in r.queue_end.values())
        assert u.traffic.waiting_vehicle_seconds >= 0
        assert u.environmental.fuel_liters >= 0 and u.environmental.co2_kg >= 0


def test_per_admitted_normalisations():
    hist, _ = run_from(sim(), FixedTimeController())
    t = unified_metrics(hist).traffic
    assert t.waiting_seconds_per_admitted == pytest.approx(t.waiting_vehicle_seconds / t.vehicles_admitted)
    assert t.waiting_minutes_per_admitted == pytest.approx(t.waiting_seconds_per_admitted / 60.0)
    idle, _ = run_from(Simulator(NET, DemandModel(NET, DemandConfig(base_rate=0.0))), FixedTimeController(), 5)
    z = unified_metrics(idle)
    assert z.traffic.waiting_seconds_per_admitted == 0.0 and z.traffic.vehicles_admitted == 0.0
    assert z.environmental.fuel_liters == 0.0 and z.environmental.co2_kg == 0.0  # zero waiting -> zero proxy


def test_empty_history_is_rejected():
    with pytest.raises(ValueError):
        unified_metrics([])


# -- environmental proxy wiring --------------------------------------------------------------------------
def test_proxy_is_computed_from_the_measured_waiting_and_scales_linearly():
    hist, _ = run_from(sim(), FixedTimeController())
    base = unified_metrics(hist, EnvironmentalConfig(idle_fuel_rate_lph=0.6))
    dbl = unified_metrics(hist, EnvironmentalConfig(idle_fuel_rate_lph=1.2))
    e = estimate(base.traffic.waiting_vehicle_seconds, EnvironmentalConfig(idle_fuel_rate_lph=0.6))
    assert base.environmental.fuel_liters == pytest.approx(e.fuel_liters) and base.environmental.co2_kg == pytest.approx(e.co2_kg)
    assert dbl.environmental.fuel_liters == pytest.approx(2 * base.environmental.fuel_liters)
    assert base.environmental.fuel_liters_per_admitted == pytest.approx(
        base.environmental.fuel_liters / base.traffic.vehicles_admitted)
    flat = base.flat()
    assert "fuel_liters_proxy" in flat and "co2_kg_proxy" in flat and "fuel_liters" not in flat  # proxy is named as such
    assert flat["idle_fuel_rate_lph_assumed"] == 0.6


# -- signal metrics ---------------------------------------------------------------------------------------------
def test_signal_metrics_for_a_fixed_plan():
    hist, _ = run_from(sim(), FixedTimeController(), 20)
    sg = unified_metrics(hist).signal
    assert sg.plan_changes == 0 and sg.plan_change_rate == 0.0
    assert sg.node_cycles == 6 * 20
    assert sg.green_seconds_ns == sg.green_seconds_ew == 6 * 20 * 30 and sg.ns_green_share == 0.5
    assert 0 <= sg.phase_utilization_ns <= 1 and 0 <= sg.phase_utilization_ew <= 1
    assert sg.emergency_priority_cycles == 0 and sg.emergency_priority_node_cycles == 0


def test_signal_metrics_count_plan_changes_and_conserve_green_time():
    hist, _ = run_from(sim(), CyclingController(), 12)
    sg = unified_metrics(hist).signal
    assert sg.plan_changes == 6 * 11 and sg.plan_change_rate == pytest.approx(1.0)  # every node changes every cycle
    assert sg.green_seconds_ns + sg.green_seconds_ew == 6 * 12 * 60  # 60 s of green per node per cycle
    assert 0 < sg.ns_green_share < 1


# -- works for every controller family ------------------------------------------------------------------------
def controllers_for(start):
    obs = Observation(start.cycle, dict(start.queues), dict(start.in_transit), NET)
    plans = solve_exact(build_qubo(TrafficState.from_observation(obs))).plans
    return {"fixed": FixedTimeController(), "adaptive": AdaptiveController(), "qubo_receding": QuboController(),
            "qubo_static": StaticPlanController(plans, "QuboExactStatic"),
            "qaoa_derived_static": StaticPlanController(plans, "QaoaDerivedStatic")}


@pytest.mark.parametrize("name", ["fixed", "adaptive", "qubo_receding", "qubo_static", "qaoa_derived_static"])
def test_unified_metrics_work_with_every_controller(name):
    s = sim()
    start = warm(s)
    hist, _ = run_from(s, controllers_for(start)[name], 15, start)
    u = unified_metrics(hist)
    assert u.emergency is None and u.traffic.cycles == 15
    assert u.traffic.waiting_vehicle_seconds == compute_metrics(hist).total_waiting_time


def test_controller_comparison_uses_identical_demand_and_is_reproducible():
    s = sim(2)
    start = warm(s)
    a1 = unified_metrics(run_from(s, AdaptiveController(), 30, start)[0])
    a2 = unified_metrics(run_from(s, AdaptiveController(), 30, start)[0])
    assert a1 == a2  # deterministic replay: identical environmental metrics too
    f_hist, _ = run_from(s, FixedTimeController(), 30, start)
    a_hist, _ = run_from(s, AdaptiveController(), 30, start)
    assert [r.demand for r in f_hist] == [r.demand for r in a_hist]  # same demand stream
    f, a = unified_metrics(f_hist), unified_metrics(a_hist)
    assert unified_metrics(f_hist) == f  # same inputs -> same baseline
    keys = ["waiting_vehicle_seconds", "fuel_liters_proxy", "co2_kg_proxy"]
    ch = relative_changes(f.flat(), a.flat(), keys)
    assert ch["fuel_liters_proxy"] == pytest.approx(ch["waiting_vehicle_seconds"])  # proxy % change == waiting % change
    assert ch["co2_kg_proxy"] == pytest.approx(ch["waiting_vehicle_seconds"])
    assert relative_changes({"x": 0.0}, {"x": 5.0}, ["x"]) == {"x": None}


# -- emergency integration --------------------------------------------------------------------------------------
def test_emergency_metrics_are_carried_through_the_unified_view():
    s = sim()
    start = warm(s, 10)
    ev = EmergencyEvent("EV1", ROUTE, 12)
    off = run_emergency(s, FixedTimeController(), ev, 30, False, start)
    on = run_emergency(s, FixedTimeController(), ev, 30, True, start)
    uo, un = unified_metrics(off.history, None, None, off), unified_metrics(on.history, None, None, on)
    for run, u in ((off, uo), (on, un)):
        v = run.vehicle
        e = u.emergency
        assert e.completed and e.travel_time_s == v.travel_time and e.free_flow_time_s == 90.0
        assert e.delay_s == pytest.approx(e.travel_time_s - e.free_flow_time_s)
        assert e.stops == v.stops and e.completion_cycle == v.completion_cycle
        assert e.prioritized_intersections == v.priority_intersections
    assert uo.signal.emergency_priority_cycles == 0 and uo.signal.emergency_override_plan_changes == 0
    assert un.emergency.prioritized_intersections == 4
    assert un.signal.emergency_priority_cycles >= 3 and un.signal.emergency_priority_node_cycles >= 4
    assert un.signal.emergency_override_plan_changes >= 4
    assert un.signal.plan_changes > uo.signal.plan_changes  # the override is visible in the applied plans
    assert un.emergency.travel_time_s < uo.emergency.travel_time_s  # measured for this fixed configuration
    # environmental proxy difference == waiting difference, in %
    assert percent_change(uo.environmental.co2_kg, un.environmental.co2_kg) == pytest.approx(
        percent_change(uo.traffic.waiting_vehicle_seconds, un.traffic.waiting_vehicle_seconds))


def test_windowed_emergency_metrics_only_count_priority_inside_the_window():
    s = sim()
    start = warm(s, 10)
    on = run_emergency(s, FixedTimeController(), EmergencyEvent("EV1", ROUTE, 12), 30, True, start)
    first_part = [r for r in on.history if r.cycle < 12]
    assert unified_metrics(first_part, None, None, on).signal.emergency_priority_cycles == 0
    whole = unified_metrics(on.history, None, None, on).signal.emergency_priority_cycles
    assert whole > 0
