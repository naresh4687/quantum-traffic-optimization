"""Behavioural tests for the classical adaptive (queue-pressure) controller."""

import numpy as np
import pytest

from qtraffic import (
    PLAN_NS20_EW40, PLAN_NS30_EW30, PLAN_NS40_EW20, VALID_PLANS, AdaptiveConfig,
    AdaptiveController, Approach, Controller, DemandConfig, DemandModel, FixedTimeController,
    Heading, Observation, Simulator, grid_network,
)

NET = grid_network(2, 3)
NS, EW = (Heading.N, Heading.S), (Heading.E, Heading.W)


def obs(queues=None, in_transit=None, cycle=0, net=NET):
    q = {a: 0.0 for a in net.approaches}
    t = {a: 0.0 for a in net.approaches}
    q.update(queues or {})
    t.update(in_transit or {})
    return Observation(cycle, q, t, net)


def A(node, heading):
    return Approach(node, Heading[heading])


def node_queues(node, ns, ew):
    """Queue ``ns`` / ``ew`` on one NS / one EW approach of ``node``; everything else empty.

    With every other queue empty, the approach's downstream queue is 0, so its pressure is
    exactly the queue length and the expected decision is unambiguous. (Loading several
    connected intersections at once is deliberately avoided: a loaded approach lowers the
    pressure of the approach feeding it, by design.)
    """
    mine = [a for a in NET.approaches if a.node == node]
    ns_a = next(a for a in mine if a.heading in NS)
    ew_a = next(a for a in mine if a.heading in EW)
    return {ns_a: ns, ew_a: ew}


def labels(plans):
    return {n: p.label for n, p in plans.items()}


# -- decision rule -------------------------------------------------------------------
def test_empty_network_gives_30_30_everywhere():
    plans = AdaptiveController().select_plans(obs())
    assert set(plans) == set(NET.nodes)
    assert all(p == PLAN_NS30_EW30 for p in plans.values())


@pytest.mark.parametrize("node", NET.nodes)
def test_balanced_traffic_gives_30_30(node):
    assert AdaptiveController().select_plans(obs(node_queues(node, 12.0, 12.0)))[node] == PLAN_NS30_EW30


@pytest.mark.parametrize("node", NET.nodes)
def test_ns_heavy_traffic_gives_40_20(node):
    assert AdaptiveController().select_plans(obs(node_queues(node, 20.0, 0.0)))[node] == PLAN_NS40_EW20


@pytest.mark.parametrize("node", NET.nodes)
def test_ew_heavy_traffic_gives_20_40(node):
    assert AdaptiveController().select_plans(obs(node_queues(node, 0.0, 20.0)))[node] == PLAN_NS20_EW40


def test_mild_imbalance_inside_the_dead_band_keeps_30_30():
    o = obs(node_queues("I1", 14.0, 10.0))  # delta = 4 < 5
    assert AdaptiveController().select_plans(o)["I1"] == PLAN_NS30_EW30


def test_in_transit_vehicles_count_towards_demand():
    only_transit = obs(in_transit={A("I1", "S"): 20.0})
    assert AdaptiveController().select_plans(only_transit)["I1"] == PLAN_NS40_EW20


def test_pressure_formula_matches_documented_definition():
    ctrl = AdaptiveController()
    o = obs(
        {A("I1", "S"): 14.0, A("I4", "S"): 6.0, A("I1", "E"): 9.0, A("I2", "E"): 4.0},
        {A("I1", "S"): 1.0, A("I4", "S"): 2.0},
    )
    # I1:S -> downstream I4:S ; I4:S is an exit; I1:N and I1:W exit; I1:E -> I2:E
    assert ctrl.approach_pressure(A("I1", "S"), o) == (14 + 1) - (6 + 2)
    assert ctrl.approach_pressure(A("I4", "S"), o) == 6 + 2
    assert ctrl.approach_pressure(A("I1", "E"), o) == 9 - 4
    p_ns, p_ew = ctrl.phase_pressures("I1", o)
    assert p_ns == 7.0 and p_ew == 5.0  # I1:N and I1:W contribute 0


def test_threshold_is_a_dead_band_around_balance():
    ctrl = AdaptiveController()
    t = ctrl.config.switch_threshold
    assert ctrl.choose_plan(0.0) == PLAN_NS30_EW30
    assert ctrl.choose_plan(t) == PLAN_NS30_EW30  # exactly at the threshold: no change
    assert ctrl.choose_plan(-t) == PLAN_NS30_EW30
    assert ctrl.choose_plan(t + 0.01) == PLAN_NS40_EW20
    assert ctrl.choose_plan(-t - 0.01) == PLAN_NS20_EW40


# -- downstream congestion -----------------------------------------------------------
def test_downstream_congestion_reduces_pressure_and_can_flip_the_decision():
    ctrl = AdaptiveController()
    up = A("I1", "S")  # discharges into I4:S
    ew = A("I1", "E")

    clear = obs({up: 20.0})
    jammed = obs({up: 20.0, A("I4", "S"): 20.0})
    assert ctrl.approach_pressure(up, jammed) < ctrl.approach_pressure(up, clear)
    assert ctrl.select_plans(clear)["I1"] == PLAN_NS40_EW20
    assert ctrl.select_plans(jammed)["I1"] != PLAN_NS40_EW20  # no green pushed into a full road

    # with a lightly loaded EW approach the congested NS demand actually loses green
    jammed_ew = obs({up: 20.0, A("I4", "S"): 40.0, ew: 8.0})
    assert ctrl.select_plans(jammed_ew)["I1"] == PLAN_NS20_EW40


def test_downstream_vehicles_in_transit_also_count():
    ctrl = AdaptiveController()
    o = obs({A("I1", "S"): 20.0}, {A("I4", "S"): 20.0})
    assert ctrl.approach_pressure(A("I1", "S"), o) == 0.0


def test_downstream_weight_zero_ignores_congestion():
    ignore = AdaptiveController(AdaptiveConfig(downstream_weight=0.0))
    o = obs({A("I1", "S"): 20.0, A("I4", "S"): 40.0})
    assert ignore.select_plans(o)["I1"] == PLAN_NS40_EW20
    assert AdaptiveController().select_plans(o)["I1"] != PLAN_NS40_EW20


def test_exit_approaches_have_no_downstream_penalty():
    ctrl = AdaptiveController()
    exit_approach = A("I4", "S")
    assert NET.downstream_approach(exit_approach) is None
    assert ctrl.approach_pressure(exit_approach, obs({exit_approach: 9.0})) == 9.0


# -- intersections are independent ---------------------------------------------------
def test_six_intersections_choose_their_own_plans():
    """One cycle where the three plans are all in use at the same time."""
    q = {
        A("I1", "S"): 20.0,  # I1: NS-heavy
        A("I2", "E"): 20.0,  # I2: EW-heavy
        A("I3", "S"): 10.0, A("I3", "W"): 10.0,  # I3: balanced
    }
    plans = AdaptiveController().select_plans(obs(q))
    assert plans["I1"] == PLAN_NS40_EW20
    assert plans["I2"] == PLAN_NS20_EW40
    assert plans["I3"] == PLAN_NS30_EW30
    assert plans["I4"] == plans["I5"] == plans["I6"] == PLAN_NS30_EW30
    assert {plans["I1"], plans["I2"], plans["I3"]} == set(VALID_PLANS)


def test_a_plan_depends_only_on_own_and_downstream_approaches():
    """Changing traffic at one intersection never changes an unrelated one's plan."""
    base = {A("I1", "S"): 20.0, A("I5", "E"): 12.0}
    ctrl = AdaptiveController()
    before = labels(ctrl.select_plans(obs(base)))
    for node in NET.nodes:
        # perturb every approach of `node`, then check nobody unrelated moved
        perturbed = dict(base)
        for a in NET.approaches:
            if a.node == node:
                perturbed[a] = perturbed.get(a, 0.0) + 15.0
        after = labels(AdaptiveController().select_plans(obs(perturbed)))
        related = {node} | {
            a.node for a in NET.approaches
            if NET.downstream_approach(a) is not None and NET.downstream_approach(a).node == node
        }
        for other in NET.nodes:
            if other not in related:
                assert after[other] == before[other], f"{other} reacted to traffic at {node}"


# -- determinism, validity ------------------------------------------------------------
def test_decisions_are_deterministic():
    rng = np.random.default_rng(7)
    observations = [
        obs({a: float(v) for a, v in zip(NET.approaches, rng.uniform(0, 40, len(NET.approaches)))},
            {a: float(v) for a, v in zip(NET.approaches, rng.uniform(0, 10, len(NET.approaches)))},
            cycle=c)
        for c in range(25)
    ]
    first, second = AdaptiveController(), AdaptiveController()
    run1 = [dict(first.select_plans(o)) for o in observations]
    run2 = [dict(second.select_plans(o)) for o in observations]
    assert run1 == run2
    first.reset()
    assert [dict(first.select_plans(o)) for o in observations] == run1


def test_same_observation_gives_same_plans_without_hysteresis():
    ctrl = AdaptiveController()
    o = obs(node_queues('I1', 20.0, 0.0))
    assert ctrl.select_plans(o) == ctrl.select_plans(o) == ctrl.select_plans(o)


def test_full_simulation_is_deterministic():
    sim = Simulator(NET, DemandModel(NET, DemandConfig.from_level("high", seed=3)))
    a = sim.run(AdaptiveController(), 40)
    b = sim.run(AdaptiveController(), 40)
    assert [r.plans for r in a.history] == [r.plans for r in b.history]
    assert a.metrics == b.metrics


def test_selected_plans_are_always_valid_and_complete():
    rng = np.random.default_rng(11)
    ctrl = AdaptiveController()
    for c in range(200):
        q = {a: float(rng.uniform(0, 40)) for a in NET.approaches}
        t = {a: float(rng.uniform(0, 15)) for a in NET.approaches}
        plans = ctrl.select_plans(obs(q, t, cycle=c))
        assert set(plans) == set(NET.nodes)
        for p in plans.values():
            assert p in VALID_PLANS and p.cycle_length == 60


def test_only_the_three_existing_plans_are_ever_applied_in_simulation():
    sim = Simulator(NET, DemandModel(NET, DemandConfig.from_level("high", seed=1)))
    hist = sim.run(AdaptiveController(), 60).history  # simulator validates every plan too
    used = {p for r in hist for p in r.plans.values()}
    assert used <= set(VALID_PLANS)


# -- interface / config --------------------------------------------------------------
def test_adaptive_controller_implements_the_controller_interface():
    ctrl = AdaptiveController()
    assert isinstance(ctrl, Controller)
    assert ctrl.name == "AdaptiveController"


def test_config_defaults_and_validation():
    cfg = AdaptiveConfig()
    assert (cfg.downstream_weight, cfg.switch_threshold, cfg.min_hold_cycles) == (1.0, 5.0, 1)
    for bad in ({"downstream_weight": -1}, {"switch_threshold": -0.1},
                {"switch_threshold": float("nan")}, {"min_hold_cycles": 0},
                {"min_hold_cycles": 1.5}, {"min_hold_cycles": True}):
        with pytest.raises(ValueError):
            AdaptiveConfig(**bad)


def test_optional_dwell_holds_a_plan_and_reset_clears_it():
    ctrl = AdaptiveController(AdaptiveConfig(min_hold_cycles=2))
    ns, ew = obs(node_queues('I1', 20.0, 0.0)), obs(node_queues('I1', 0.0, 20.0))
    assert ctrl.select_plans(ns)["I1"] == PLAN_NS40_EW20
    assert ctrl.select_plans(ew)["I1"] == PLAN_NS40_EW20  # held: only 1 cycle in force
    assert ctrl.select_plans(ew)["I1"] == PLAN_NS20_EW40  # 2 cycles served, may switch
    ctrl.reset()
    assert ctrl.select_plans(ew)["I1"] == PLAN_NS20_EW40  # no memory after reset


def test_decision_trace_records_pressures_and_plan():
    ctrl = AdaptiveController(record=True)
    ctrl.select_plans(obs({A("I1", "S"): 20.0}))
    d = next(x for x in ctrl.decisions if x.node == "I1")
    assert (d.pressure_ns, d.pressure_ew, d.delta) == (20.0, 0.0, 20.0)
    assert d.plan == d.desired == PLAN_NS40_EW20
    assert len(ctrl.decisions) == len(NET.nodes)


# -- behaviour in the simulator -------------------------------------------------------
def test_adaptive_serves_ns_heavy_demand_better_than_fixed():
    mult = {a: (1.5 if a.heading in NS else 0.5) for a in NET.entry_approaches}
    demand = DemandModel(NET, DemandConfig.from_level("medium", seed=0, multipliers=mult))
    sim = Simulator(NET, demand)
    fixed = sim.run(FixedTimeController(), 60).metrics
    adaptive = sim.run(AdaptiveController(), 60).metrics
    assert adaptive.vehicles_exited > fixed.vehicles_exited
    assert adaptive.vehicles_rejected < fixed.vehicles_rejected
