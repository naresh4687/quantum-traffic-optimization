"""Tests for the emergency vehicle, the priority override layer and the corridor experiment."""

import pytest

from qtraffic import (
    PLAN_NS20_EW40, PLAN_NS30_EW30, PLAN_NS40_EW20, AdaptiveController, Approach, Controller,
    CycleRecord, DemandConfig, DemandModel, FixedTimeController, Heading, Observation, Phase,
    Simulator, grid_network,
)
from qtraffic.emergency import (
    CYCLE_LENGTH, EmergencyEvent, EmergencyOverrideController, EmergencyVehicle, RouteError,
    Status, phase_window, priority_plan, route_approaches, run_emergency, validate_route,
)
from qtraffic.optimization import QuboController
from qtraffic.simulator import TrafficState

NET = grid_network(2, 3)
ROUTE = ("I1", "I2", "I3", "I6")
S = 0.5  # saturation flow


def A(node, heading):
    return Approach(node, Heading[heading])


def simulator(seed=0, level="medium"):
    return Simulator(NET, DemandModel(NET, DemandConfig.from_level(level, seed=seed)))


def warm_state(sim, cycles=10):
    st, fx = sim.initial_state(), FixedTimeController()
    for _ in range(cycles):
        sim.step(st, fx)
    return st


def event(**kw):
    kw.setdefault("start_cycle", 12)
    return EmergencyEvent("EV1", ROUTE, **kw)


def observation(cycle=0):
    zeros = {a: 0.0 for a in NET.approaches}
    return Observation(cycle, dict(zeros), dict(zeros), NET)


def record(cycle, plans, q0=None, served=None, blocked=None):
    """Hand-built CycleRecord: only the fields the emergency vehicle reads matter."""
    z = {a: 0.0 for a in NET.approaches}
    fill = lambda d: {**z, **(d or {})}
    return CycleRecord(cycle=cycle, plans=plans, demand={}, admitted={}, transit_arrivals=dict(z),
                       queue_after_arrivals=fill(q0), served=fill(served), blocked=fill(blocked),
                       queue_end=dict(z), in_transit_end=dict(z), exited=0.0)


ALL30 = {n: PLAN_NS30_EW30 for n in NET.nodes}


# -- 1. event creation ------------------------------------------------------------------------------
def test_event_creation_and_defaults():
    ev = EmergencyEvent("EV1", list(ROUTE), 10)
    assert ev.route == ROUTE and isinstance(ev.route, tuple)
    assert (ev.origin, ev.destination, ev.start_cycle) == ("I1", "I6", 10)
    assert ev.link_seconds == 30.0 and ev.lead_seconds == 0.0
    assert ev.free_flow_seconds == 90.0  # 3 links x 30 s
    with pytest.raises(Exception):
        ev.start_cycle = 11  # frozen


@pytest.mark.parametrize("kw", [{"vehicle_id": ""}, {"start_cycle": -1}, {"start_cycle": 1.5},
                                {"start_cycle": True}, {"link_seconds": 0}, {"lead_seconds": -1}])
def test_invalid_event_is_rejected(kw):
    args = {"vehicle_id": "EV1", "route": ROUTE, "start_cycle": 3, **kw}
    with pytest.raises(ValueError):
        EmergencyEvent(**args)


# -- 9. route validation ----------------------------------------------------------------------------------
def test_valid_route_and_its_approaches():
    validate_route(NET, ROUTE)
    approaches = route_approaches(NET, ROUTE)
    assert [str(a) for a in approaches] == ["I1:E", "I2:E", "I3:E", "I6:S"]  # arrival headings; turn at I3
    assert [a.phase for a in approaches] == [Phase.EW, Phase.EW, Phase.EW, Phase.NS]
    validate_route(NET, ("I6", "I5", "I4", "I1"))  # reverse-direction roads exist too


@pytest.mark.parametrize("route, why", [
    (("I1",), "one node"), ((), "empty"),
    (("I1", "I9"), "unknown intersection"), (("I1", "I3"), "not adjacent"),
    (("I1", "I5"), "diagonal, no road"), (("I1", "I2", "I1"), "revisits"),
    (("I1", "I2", "I6"), "I2->I6 does not exist"),
])
def test_invalid_routes_are_rejected(route, why):
    with pytest.raises(RouteError):
        validate_route(NET, route)
    with pytest.raises(RouteError):
        EmergencyVehicle(EmergencyEvent("EV", route, 1) if len(route) else EmergencyEvent("EV", (), 1), NET)


def test_emergency_cannot_run_over_a_nonexistent_edge():
    sim = simulator()
    bad = EmergencyEvent("EV", ("I1", "I2", "I6"), 3)
    with pytest.raises(RouteError):
        run_emergency(sim, FixedTimeController(), bad, 10)


# -- 3. activation ---------------------------------------------------------------------------------------------
def test_vehicle_activates_at_its_start_cycle_and_not_before():
    v = EmergencyVehicle(event(start_cycle=5), NET)
    assert v.status is Status.PENDING and v.position() == {"status": "pending"}
    for c in range(5):
        v.begin_cycle(c)
        assert v.status is Status.PENDING
    v.begin_cycle(5)
    assert v.active and v.entry_cycle == 5 and v.entry_time == 5 * CYCLE_LENGTH
    assert v.position() == {"status": "active", "route_index": 0, "at_stop_line_of": "I1"}
    assert v.events[0]["event"] == "entered"


def test_start_cycle_outside_the_simulated_window_is_rejected():
    sim = simulator()
    with pytest.raises(ValueError):
        run_emergency(sim, FixedTimeController(), event(start_cycle=50), 10)
    st = warm_state(sim, 10)
    with pytest.raises(ValueError):
        run_emergency(sim, FixedTimeController(), event(start_cycle=3), 10, start_state=st)


# -- unit tests of the timeline on hand-built records (4, 5) ----------------------------------------------
def vehicle_at_origin(**kw):
    v = EmergencyVehicle(EmergencyEvent("EV", ("I1", "I2"), 0, **kw), NET, S)
    v.begin_cycle(0)
    return v


def test_empty_queue_and_open_green_crosses_at_arrival_without_stopping():
    v = vehicle_at_origin()
    v.arrival_time = 40.0  # arrives 10 s into the EW green window [30, 60)
    v.process_cycle(record(0, ALL30), {})
    c = v.crossings[0]
    assert c.crossing_time == 40.0 and c.wait == 0.0 and not c.stopped
    assert v.arrival_time == 70.0 and v.index == 1  # 30 s link to I2


def test_arriving_on_red_waits_for_green_plus_queue_discharge_headway():
    v = vehicle_at_origin()  # arrives at t=0; EW green (NS first) is [30, 60)
    v.process_cycle(record(0, ALL30, q0={A("I1", "E"): 4.0}), {})
    c = v.crossings[0]
    assert c.crossing_time == pytest.approx(30 + (4 + 1) / S)  # 5th vehicle out at 2 s per vehicle
    assert c.wait == pytest.approx(c.crossing_time) and c.stopped and c.cycle == 0


def test_priority_order_puts_the_emergency_phase_first():
    v = vehicle_at_origin()
    v.process_cycle(record(0, ALL30), {"I1": Phase.EW})  # EW first: green [0, 30), empty queue
    assert v.crossings[0].crossing_time == 0.0 and not v.crossings[0].stopped


def test_queue_longer_than_the_cycles_service_carries_over_to_later_cycles():
    v = vehicle_at_origin()
    a = A("I1", "E")
    v.process_cycle(record(0, ALL30, q0={a: 40.0}, served={a: 15.0}), {})  # 15 discharge per 30 s green
    assert v.active and v.carried_ahead == 25.0 and not v.crossings
    v.process_cycle(record(1, ALL30, q0={a: 35.0}, served={a: 15.0}), {})  # 25 ahead: still not its turn
    assert v.active and v.carried_ahead == 10.0 and not v.crossings
    v.process_cycle(record(2, ALL30, q0={a: 30.0}, served={a: 15.0}), {})  # 10 ahead: the 11th vehicle
    c = v.crossings[0]
    assert c.cycle == 2 and c.crossing_time == pytest.approx(120 + 30 + 11 / S)  # EW green starts at +30 s
    assert c.wait == pytest.approx(c.crossing_time) and c.stopped


def test_spillback_blocking_holds_the_emergency_vehicle_too():
    v = vehicle_at_origin()
    a = A("I1", "E")
    # queue of 3 but the road ahead is full: only 2 vehicles were served (blocked > 0)
    v.process_cycle(record(0, ALL30, q0={a: 3.0}, served={a: 2.0}, blocked={a: 5.0}), {"I1": Phase.EW})
    assert v.active and not v.crossings  # would need 4th slot; only 2 discharged
    assert v.carried_ahead == 1.0


def test_capacity_limits_the_vehicle_behind_a_full_green():
    v = vehicle_at_origin()
    a = A("I1", "E")
    # 15 ahead uses the whole 30 s green (15 vehicles); the emergency vehicle would be the 16th
    v.process_cycle(record(0, PLANS_30 := ALL30, q0={a: 15.0}, served={a: 15.0}), {"I1": Phase.EW})
    assert v.active and not v.crossings and v.carried_ahead == 0.0
    v.process_cycle(record(1, ALL30), {"I1": Phase.EW})  # next cycle: first out
    assert v.crossings[0].crossing_time == pytest.approx(60 + 1 / S)


def test_phase_window_helper():
    assert phase_window(PLAN_NS20_EW40, Phase.NS, Phase.NS, 2) == (120.0, 140.0)
    assert phase_window(PLAN_NS20_EW40, Phase.EW, Phase.NS, 2) == (140.0, 180.0)
    assert phase_window(PLAN_NS20_EW40, Phase.EW, Phase.EW, 2) == (120.0, 160.0)
    assert priority_plan(Phase.NS) == PLAN_NS40_EW20 and priority_plan(Phase.EW) == PLAN_NS20_EW40


# -- 4. signal priority ----------------------------------------------------------------------------------------------
class ScriptedBase(Controller):
    def __init__(self, plan):
        self.plan = plan

    def select_plans(self, observation):
        return {n: self.plan for n in observation.network.nodes}


def active_vehicle(**kw):
    v = EmergencyVehicle(event(start_cycle=0, **kw), NET, S)
    v.begin_cycle(0)
    return v


def test_priority_gives_the_emergency_phase_its_maximum_green_and_cuts_the_conflict():
    v = active_vehicle()
    ctrl = EmergencyOverrideController(FixedTimeController(), v)
    plans = ctrl.select_plans(observation(0))
    # I1 and I2 are reached within the first 60 s; I3 (ETA 60 s) and I6 (90 s) are not yet
    assert ctrl.log[0].priority_nodes == ("I1", "I2")
    for node in ("I1", "I2"):
        assert plans[node] == PLAN_NS20_EW40 and plans[node].green_time(Phase.EW) == 40
    assert plans["I3"] == plans["I6"] == plans["I4"] == plans["I5"] == PLAN_NS30_EW30  # untouched
    assert ctrl.log[0].first_phase == {"I1": Phase.EW, "I2": Phase.EW}
    assert ctrl.log[0].changed_nodes == ("I1", "I2")


def test_no_change_when_the_base_plan_already_gives_maximum_green():
    v = active_vehicle()
    ctrl = EmergencyOverrideController(ScriptedBase(PLAN_NS20_EW40), v)
    plans = ctrl.select_plans(observation(0))
    assert all(p == PLAN_NS20_EW40 for p in plans.values())
    assert ctrl.log[0].changed_nodes == () and ctrl.log[0].priority_nodes == ("I1", "I2")  # order still set


def test_north_south_emergency_gets_the_ns_plan():
    ev = EmergencyEvent("EV", ("I1", "I4"), 0)  # I1 -> I4 heads south
    v = EmergencyVehicle(ev, NET, S)
    v.begin_cycle(0)
    plans = EmergencyOverrideController(FixedTimeController(), v).select_plans(observation(0))
    assert plans["I1"] == PLAN_NS40_EW20 and plans["I4"] == PLAN_NS40_EW20


def test_override_is_inactive_before_the_event_and_when_disabled():
    pending = EmergencyVehicle(event(start_cycle=5), NET, S)
    ctrl = EmergencyOverrideController(FixedTimeController(), pending)
    assert ctrl.select_plans(observation(0)) == {n: PLAN_NS30_EW30 for n in NET.nodes}
    assert ctrl.log[0].active is False and ctrl.priority_nodes(0) == ()
    v = active_vehicle()
    off = EmergencyOverrideController(FixedTimeController(), v, enabled=False)
    assert off.select_plans(observation(0)) == {n: PLAN_NS30_EW30 for n in NET.nodes}
    assert off.priority_nodes(0) == () and off.name == "FixedTimeController"


def test_the_base_controller_is_always_consulted_and_layering_is_visible():
    calls = []

    class Spy(Controller):
        def select_plans(self, observation):
            calls.append(observation.cycle)
            return {n: PLAN_NS30_EW30 for n in observation.network.nodes}

    v = active_vehicle()
    ctrl = EmergencyOverrideController(Spy(), v)
    ctrl.select_plans(observation(0))
    ctrl.select_plans(observation(1))
    assert calls == [0, 1]
    assert ctrl.log[0].base_plans["I1"] == PLAN_NS30_EW30 and ctrl.log[0].final_plans["I1"] == PLAN_NS20_EW40
    assert ctrl.name == "Spy+EmergencyCorridor"


# -- 5/6/7. movement, progression, completion in the real simulator ------------------------------------------------
def primary_pair(base=FixedTimeController, seed=0, cycles=40):
    sim = simulator(seed)
    st = warm_state(sim, 10)
    ev = event(start_cycle=12)
    off = run_emergency(sim, base(), ev, cycles, corridor=False, start_state=st)
    on = run_emergency(sim, base(), ev, cycles, corridor=True, start_state=st)
    return sim, st, ev, off, on


def test_emergency_completes_its_route_in_order_with_consistent_timing():
    _, _, ev, off, on = primary_pair()
    for run in (off, on):
        v = run.vehicle
        assert v.completed and v.completion_cycle is not None and v.status is Status.COMPLETED
        assert [c.node for c in v.crossings] == list(ROUTE)
        assert v.entry_time == 12 * CYCLE_LENGTH
        for prev, nxt in zip(v.crossings, v.crossings[1:]):  # every hop takes the free-flow link time
            assert nxt.arrival_time == pytest.approx(prev.crossing_time + ev.link_seconds)
        assert v.travel_time == pytest.approx(v.completion_time - v.entry_time)
        assert v.travel_time == pytest.approx(v.free_flow_time + sum(c.wait for c in v.crossings))
        assert v.delay >= -1e-9 and v.stops == sum(c.wait > 1e-9 for c in v.crossings)
        assert v.completion_cycle == v.crossings[-1].cycle
        assert v.position() == {"status": "completed", "at": "I6"}


def test_emergency_only_crosses_during_a_green_of_its_own_phase():
    _, _, _, off, on = primary_pair()
    for run in (off, on):
        by_cycle = {rec.cycle: (rec, log) for rec, log in zip(run.history, run.log)}
        for c in run.vehicle.crossings:
            rec, log = by_cycle[c.cycle]
            ws, we = phase_window(rec.plans[c.node], c.approach.phase, log.first_phase.get(c.node, Phase.NS), c.cycle)
            assert ws - 1e-9 <= c.crossing_time <= we + 1e-9  # never through a red
            assert c.crossing_time >= c.arrival_time - 1e-9  # and never before it arrived
            if c.wait > 1e-9:  # if it had to wait it left at a queue-discharge slot, one headway after green start
                assert c.crossing_time >= ws + 1 / S - 1e-9


def test_corridor_activates_hop_by_hop_and_releases_behind_the_vehicle():
    _, _, _, off, on = primary_pair()
    v = on.vehicle
    firsts = [min(v.priority_cycles[n]) for n in ROUTE]
    assert firsts == sorted(firsts) and firsts[0] == 12 and len(set(firsts)) > 1  # propagates, not all at once
    first_log = next(r for r in on.log if r.priority_nodes)
    assert set(first_log.priority_nodes) < set(ROUTE)  # never the whole route in the first cycle
    for node, crossing in zip(ROUTE, v.crossings):
        assert max(v.priority_cycles[node]) <= crossing.cycle  # released after the vehicle crossed it
    assert v.priority_intersections == 4
    assert all(not r.priority_nodes for r in off.log) and off.vehicle.priority_intersections == 0


def test_normal_traffic_without_override_is_exactly_the_plain_simulation():
    """The vehicle is massless, so with the override off nothing about normal traffic changes."""
    sim, st, _, off, _ = primary_pair()
    fresh = TrafficState(st.cycle, dict(st.queues), dict(st.in_transit))
    plain = [sim.step(fresh, FixedTimeController()) for _ in range(40)]
    assert [r.plans for r in off.history] == [r.plans for r in plain]
    assert [r.queue_end for r in off.history] == [r.queue_end for r in plain]


def test_override_changes_normal_traffic_only_after_the_emergency_starts():
    _, _, ev, off, on = primary_pair()
    for a, b in zip(off.history, on.history):
        if a.cycle < ev.start_cycle:
            assert a.plans == b.plans and a.queue_end == b.queue_end  # identical until the event
    assert any(a.plans != b.plans for a, b in zip(off.history, on.history))  # then the plans differ
    assert any(a.queue_end != b.queue_end for a, b in zip(off.history, on.history))
    assert [r.demand for r in off.history] == [r.demand for r in on.history]  # identical demand


def test_vehicle_conservation_holds_with_the_override():
    sim, st, _, _, on = primary_pair()
    m = on.result.metrics
    start_total = sum(st.queues.values()) + sum(st.in_transit.values())
    end_total = sum(on.result.final_queues.values()) + sum(on.result.final_in_transit.values())
    assert start_total + m.vehicles_entered == pytest.approx(m.vehicles_exited + end_total)


def test_vehicle_that_does_not_finish_in_time_is_reported_incomplete():
    sim = simulator()
    run = run_emergency(sim, FixedTimeController(), event(start_cycle=3), 4)  # entry at cycle 3, ends at 3
    assert not run.vehicle.completed and run.vehicle.travel_time is None and run.vehicle.delay is None
    assert run.vehicle.active or run.vehicle.status is Status.ACTIVE


# -- 8. restoration (mandatory) ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("base", [FixedTimeController, AdaptiveController])
def test_after_the_emergency_the_base_controller_is_back_in_charge(base):
    _, _, _, _, on = primary_pair(base, cycles=40)
    done = on.vehicle.completion_cycle
    assert done is not None and done < on.start_cycle + 39  # there are cycles left after completion
    assert any(r.changed_nodes for r in on.log if r.cycle <= done)  # override really acted before
    for rec in on.log:
        if rec.cycle > done:
            assert rec.priority_nodes == () and rec.changed_nodes == () and not rec.active
            assert rec.final_plans == rec.base_plans  # final plan IS the base plan
            assert rec.first_phase == {}
    # override is released at each intersection as soon as the vehicle has passed it
    for rec in on.log:
        for node in rec.priority_nodes:
            assert not any(c.node == node and c.cycle < rec.cycle for c in on.vehicle.crossings)


def test_override_does_not_lock_the_network_into_emergency_priority():
    _, _, _, _, on = primary_pair(FixedTimeController, cycles=40)
    done = on.vehicle.completion_cycle
    after = [rec for rec in on.history if rec.cycle > done]
    assert len(after) >= 20
    assert all(p == PLAN_NS30_EW30 for rec in after for p in rec.plans.values())  # Fixed plan restored everywhere


def test_qubo_controller_resumes_after_the_emergency():
    sim = simulator(0)
    st = warm_state(sim, 10)
    on = run_emergency(sim, QuboController(), event(start_cycle=12), 25, corridor=True, start_state=st)
    done = on.vehicle.completion_cycle
    assert done is not None
    late = [r for r in on.log if r.cycle > done]
    assert late and all(r.final_plans == r.base_plans and not r.priority_nodes for r in late)


def test_a_second_run_resets_the_vehicle_and_override_state():
    sim = simulator(0)
    st = warm_state(sim, 10)
    ctrl_run_1 = run_emergency(sim, FixedTimeController(), event(start_cycle=12), 30, True, st)
    ctrl_run_2 = run_emergency(sim, FixedTimeController(), event(start_cycle=12), 30, True, st)
    assert ctrl_run_1.vehicle.travel_time == ctrl_run_2.vehicle.travel_time
    assert len(ctrl_run_2.vehicle.crossings) == 4  # not accumulated from run 1


# -- 10. determinism -------------------------------------------------------------------------------------------------------------
def test_same_configuration_gives_identical_results():
    a = primary_pair()[3:]
    b = primary_pair()[3:]
    for x, y in zip(a, b):
        assert x.vehicle.travel_time == y.vehicle.travel_time
        assert x.vehicle.completion_cycle == y.vehicle.completion_cycle
        assert x.vehicle.events == y.vehicle.events
        assert [r.final_plans for r in x.log] == [r.final_plans for r in y.log]
        assert [r.plans for r in x.history] == [r.plans for r in y.history]
        assert x.result.metrics == y.result.metrics


# -- 11. baseline vs corridor ---------------------------------------------------------------------------------------------------------
def test_baseline_versus_corridor_uses_identical_inputs_and_reports_measured_differences():
    _, st, ev, off, on = primary_pair()
    assert off.start_cycle == on.start_cycle == st.cycle
    assert off.vehicle.event == on.vehicle.event  # same route and start cycle
    lo, hi = ev.start_cycle, max(off.vehicle.completion_cycle, on.vehicle.completion_cycle)
    w_off, w_on = off.window_metrics(lo, hi), on.window_metrics(lo, hi)
    assert w_off is not None and w_on is not None
    pre_off, pre_on = off.window_metrics(st.cycle, lo - 1), on.window_metrics(st.cycle, lo - 1)
    assert pre_off == pre_on  # before the event the two runs are indistinguishable
    # measured (not assumed) result for this fixed configuration: the corridor shortens the trip
    assert on.vehicle.travel_time < off.vehicle.travel_time


@pytest.mark.parametrize("link", [20.0, 30.0, 45.0])
def test_free_flow_time_follows_the_link_parameter_and_delay_is_never_negative(link):
    sim = simulator(1)
    st = warm_state(sim, 10)
    run = run_emergency(sim, FixedTimeController(), event(start_cycle=12, link_seconds=link), 40, True, st)
    v = run.vehicle
    assert v.free_flow_time == 3 * link and v.delay >= -1e-9 and v.travel_time >= v.free_flow_time - 1e-9
