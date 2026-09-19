import pytest
from helpers import CyclingController, PerNodeController, ScriptedDemand

from qtraffic import (
    PLAN_NS20_EW40, PLAN_NS40_EW20, Approach, DemandConfig, DemandModel, FixedTimeController,
    Heading, SignalPlan, SimulationConfig, Simulator, grid_network,
)

E, W, N, S = Heading.E, Heading.W, Heading.N, Heading.S
EPS = 1e-9


def make_sim(demand_level="high", seed=42, net=None, **cfg):
    net = net or grid_network()
    demand = DemandModel(net, DemandConfig.from_level(demand_level, seed=seed))
    return Simulator(net, demand, SimulationConfig(**cfg))


# -- determinism ------------------------------------------------------------
def test_simulation_is_deterministic():
    r1 = make_sim().run(FixedTimeController(), 20)
    r2 = make_sim().run(FixedTimeController(), 20)
    assert r1.history == r2.history
    assert r1.metrics == r2.metrics
    assert r1.final_queues == r2.final_queues


def test_repeated_runs_on_the_same_simulator_are_identical():
    sim = make_sim()
    assert sim.run(FixedTimeController(), 10).history == sim.run(FixedTimeController(), 10).history


def test_different_seed_changes_the_outcome():
    a = make_sim(seed=1).run(FixedTimeController(), 10).metrics
    b = make_sim(seed=2).run(FixedTimeController(), 10).metrics
    assert a.demand_generated != b.demand_generated


# -- invariants under stress ------------------------------------------------
@pytest.mark.parametrize("controller", [FixedTimeController(), CyclingController()])
@pytest.mark.parametrize("level", ["low", "high"])
def test_queues_never_negative_or_above_capacity(controller, level):
    net = grid_network(road_capacity=20.0)
    result = make_sim(level, net=net).run(controller, 40)
    for rec in result.history:
        for a in net.approaches:
            assert rec.queue_end[a] >= 0.0
            assert rec.queue_after_arrivals[a] >= 0.0
            assert rec.served[a] >= 0.0 and rec.blocked[a] >= 0.0
            assert rec.queue_after_arrivals[a] <= net.capacity(a) + EPS
            assert rec.queue_end[a] + rec.in_transit_end[a] <= net.capacity(a) + EPS


@pytest.mark.parametrize("controller", [FixedTimeController(), CyclingController()])
def test_vehicle_conservation(controller):
    net = grid_network(road_capacity=15.0, entry_capacity=15.0)
    result = make_sim("high", net=net).run(controller, 30)
    m = result.metrics
    assert m.vehicles_rejected > 0  # the scenario really overflows the entries
    # every generated vehicle was either admitted or rejected...
    assert m.demand_generated == pytest.approx(m.vehicles_entered + m.vehicles_rejected, abs=EPS)
    # ...and every admitted vehicle has exited, is queued, or is in transit.
    assert m.vehicles_entered == pytest.approx(
        m.vehicles_exited + m.final_queue_total + m.in_transit_final, abs=1e-6
    )
    # the same balance holds cycle by cycle from the network state
    entered = exited = 0.0
    for rec in result.history:
        entered += sum(rec.admitted.values())
        exited += rec.exited
        held = sum(rec.queue_end.values()) + sum(rec.in_transit_end.values())
        assert entered - exited == pytest.approx(held, abs=1e-6)


def test_state_carries_over_between_cycles():
    result = make_sim("high").run(FixedTimeController(), 15)
    for prev, nxt in zip(result.history, result.history[1:]):
        for a in prev.queue_end:
            carried = prev.queue_end[a] + prev.in_transit_end[a]
            assert nxt.queue_after_arrivals[a] == pytest.approx(
                carried + nxt.admitted.get(a, 0.0), abs=EPS
            )
            assert nxt.transit_arrivals[a] == prev.in_transit_end[a]


# -- propagation ------------------------------------------------------------
def test_upstream_release_arrives_downstream_next_cycle_and_exits():
    net = grid_network()
    demand = ScriptedDemand({0: {Approach("I1", E): 10.0}})
    result = Simulator(net, demand).run(FixedTimeController(), 4)
    h = result.history
    i1, i2, i3 = Approach("I1", E), Approach("I2", E), Approach("I3", E)
    # cycle 0: I1 discharges all 10; nothing has reached I2 yet
    assert h[0].served[i1] == 10.0 and h[0].transit_arrivals[i2] == 0.0
    assert h[0].in_transit_end[i2] == 10.0
    # cycle 1: the 10 vehicles arrive at I2 and are discharged toward I3
    assert h[1].transit_arrivals[i2] == 10.0 and h[1].served[i2] == 10.0
    assert h[1].in_transit_end[i3] == 10.0 and h[1].exited == 0.0
    # cycle 2: they arrive at I3 and leave the network
    assert h[2].transit_arrivals[i3] == 10.0 and h[2].served[i3] == 10.0
    assert h[2].exited == 10.0
    # cycle 3: the network is empty again
    assert sum(h[3].queue_after_arrivals.values()) == 0.0
    # nothing leaked into unrelated approaches
    touched = {i1, i2, i3}
    for rec in h:
        assert all(v == 0.0 for a, v in rec.served.items() if a not in touched)


def test_flow_is_split_by_direction_not_mixed():
    """Southbound traffic follows columns, eastbound follows rows."""
    net = grid_network()
    demand = ScriptedDemand({0: {Approach("I2", S): 8.0, Approach("I4", E): 6.0}})
    h = Simulator(net, demand).run(FixedTimeController(), 3).history
    assert h[1].transit_arrivals[Approach("I5", S)] == 8.0
    assert h[1].transit_arrivals[Approach("I5", E)] == 6.0
    assert h[1].transit_arrivals[Approach("I3", E)] == 0.0


def test_link_conservation_release_equals_next_cycle_arrival():
    net = grid_network()
    h = make_sim("medium").run(FixedTimeController(), 12).history
    for prev, nxt in zip(h, h[1:]):
        for a in net.approaches:
            down = net.downstream_approach(a)
            if down is not None:
                assert nxt.transit_arrivals[down] == pytest.approx(prev.served[a], abs=EPS)


def test_green_time_limits_discharge():
    net = grid_network()
    demand = ScriptedDemand(constant={Approach("I1", E): 40.0, Approach("I1", S): 40.0})
    plan = PLAN_NS20_EW40
    h = Simulator(net, demand).run(FixedTimeController(plan), 1).history[0]
    assert h.served[Approach("I1", E)] == 0.5 * 40  # EW green 40 s
    assert h.served[Approach("I1", S)] == 0.5 * 20  # NS green 20 s


# -- downstream capacity / spillback ----------------------------------------
def _bottleneck_run(cycles=12):
    """Row I1-I2-I3; I2 gives EW only 20 s, so it can pass 2 veh/cycle while I1
    can send 3. The road ahead of I2 holds at most 6 vehicles."""
    net = grid_network(1, 3, road_capacity=6.0, entry_capacity=1000.0)
    demand = ScriptedDemand(constant={Approach("I1", E): 20.0})
    controller = PerNodeController({"I2": PLAN_NS40_EW20})
    sim = Simulator(net, demand, SimulationConfig(saturation_flow=0.1))
    return net, sim.run(controller, cycles)


def test_downstream_capacity_is_never_exceeded():
    net, result = _bottleneck_run()
    i2 = Approach("I2", E)
    assert net.capacity(i2) == 6.0
    for rec in result.history:
        assert rec.queue_after_arrivals[i2] <= 6.0 + EPS
        assert rec.queue_end[i2] + rec.in_transit_end[i2] <= 6.0 + EPS


def test_spillback_holds_vehicles_upstream_and_limits_release_to_downstream_rate():
    _, result = _bottleneck_run()
    i1, i2 = Approach("I1", E), Approach("I2", E)
    early, late = result.history[0], result.history[-1]
    # at first I1 can release its full 3 veh/cycle green capacity, nothing blocked
    assert early.served[i1] == pytest.approx(3.0) and early.blocked[i1] == 0.0
    # once I2's road is full, I1 is throttled to what I2 discharges (2 veh/cycle)
    assert late.served[i1] == pytest.approx(2.0)
    assert late.blocked[i1] == pytest.approx(1.0)
    assert late.served[i2] == pytest.approx(2.0)
    # vehicles the downstream cannot take pile up at I1 instead of vanishing
    assert late.queue_end[i1] > early.queue_end[i1]
    assert result.metrics.total_blocked > 0


def test_without_bottleneck_nothing_is_blocked():
    net = grid_network(1, 3, road_capacity=6.0, entry_capacity=1000.0)
    demand = ScriptedDemand(constant={Approach("I1", E): 20.0})
    result = Simulator(net, demand, SimulationConfig(saturation_flow=0.1)).run(
        FixedTimeController(), 12
    )
    assert result.metrics.total_blocked == 0.0


def test_entry_overflow_is_rejected_not_silently_queued():
    net = grid_network(1, 2, road_capacity=10.0, entry_capacity=10.0)
    demand = ScriptedDemand(constant={Approach("I1", E): 30.0})
    h = Simulator(net, demand).run(FixedTimeController(), 3).history
    assert h[0].admitted[Approach("I1", E)] == 10.0
    assert h[0].rejected == 20.0


# -- multi-cycle behaviour, config and input validation ---------------------
def test_multi_cycle_run_produces_one_record_per_cycle():
    result = make_sim().run(FixedTimeController(), 20)
    assert [r.cycle for r in result.history] == list(range(20))
    assert result.metrics.cycles == 20


def test_overloaded_entry_queue_grows_by_the_analytic_amount():
    """Demand 20/cycle vs 15/cycle capacity => queue grows by exactly 5 per cycle."""
    net = grid_network(entry_capacity=1000.0)  # capacity not binding
    demand = ScriptedDemand(constant={Approach("I1", E): 20.0})
    h = Simulator(net, demand).run(FixedTimeController(), 8).history
    assert [r.queue_end[Approach("I1", E)] for r in h] == pytest.approx([5.0 * (k + 1) for k in range(8)])


def test_overloaded_entry_queue_saturates_at_capacity_and_rejects_excess():
    """With entry capacity 40: q0 is capped at 40, so q1 settles at 40 - 15 = 25."""
    net = grid_network()
    demand = ScriptedDemand(constant={Approach("I1", E): 20.0})
    h = Simulator(net, demand).run(FixedTimeController(), 8).history
    a = Approach("I1", E)
    assert [r.queue_end[a] for r in h[-3:]] == [25.0] * 3
    assert h[-1].admitted[a] == 15.0 and h[-1].rejected == 5.0


def test_saturation_flow_scales_discharge():
    net = grid_network()
    demand = ScriptedDemand(constant={Approach("I1", E): 100.0})
    fast = Simulator(net, demand, SimulationConfig(0.5)).run(FixedTimeController(), 1)
    slow = Simulator(net, demand, SimulationConfig(0.25)).run(FixedTimeController(), 1)
    assert fast.history[0].served[Approach("I1", E)] == 15.0
    assert slow.history[0].served[Approach("I1", E)] == 7.5


def test_simulator_rejects_invalid_signal_plans_from_a_controller():
    from qtraffic import Controller

    forged = object.__new__(SignalPlan)
    object.__setattr__(forged, "ns_green", 45)
    object.__setattr__(forged, "ew_green", 15)

    class Bad(Controller):
        def select_plans(self, obs):
            return {n: forged for n in obs.network.nodes}

    class Incomplete(Controller):
        def select_plans(self, obs):
            return {"I1": PLAN_NS20_EW40}

    sim = make_sim()
    with pytest.raises(ValueError):
        sim.run(Bad(), 1)
    with pytest.raises(ValueError, match="missing"):
        sim.run(Incomplete(), 1)


def test_simulator_rejects_bad_demand_and_cycle_counts():
    net = grid_network()
    with pytest.raises(ValueError, match="non-entry"):
        Simulator(net, ScriptedDemand(constant={Approach("I2", E): 1.0})).run(FixedTimeController(), 1)
    with pytest.raises(ValueError, match="invalid demand"):
        Simulator(net, ScriptedDemand(constant={Approach("I1", E): -1.0})).run(FixedTimeController(), 1)
    with pytest.raises(ValueError):
        make_sim().run(FixedTimeController(), 0)
    with pytest.raises(ValueError):
        SimulationConfig(saturation_flow=0)


def test_simulator_does_not_import_qiskit():
    import subprocess
    import sys

    code = "import sys, qtraffic; assert not any(m.startswith('qiskit') for m in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_works_on_an_eight_intersection_grid():
    net = grid_network(2, 4)
    sim = Simulator(net, DemandModel(net, DemandConfig.from_level("medium", seed=3)))
    result = sim.run(FixedTimeController(), 10)
    assert set(result.metrics.final_queue) == set(net.nodes) and len(net.nodes) == 8
    assert result.metrics.vehicles_entered == pytest.approx(
        result.metrics.vehicles_exited + result.metrics.final_queue_total + result.metrics.in_transit_final
    )
