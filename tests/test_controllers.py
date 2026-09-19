import pytest

from qtraffic import (
    PLAN_NS20_EW40, PLAN_NS30_EW30, DemandConfig, DemandModel, FixedTimeController, Observation,
    SignalPlan, Simulator, grid_network,
)
from qtraffic.controllers import Controller


def _observation(net):
    zeros = {a: 0.0 for a in net.approaches}
    return Observation(0, zeros, dict(zeros), net)


def test_fixed_controller_returns_30_30_for_every_intersection():
    net = grid_network()
    plans = FixedTimeController().select_plans(_observation(net))
    assert set(plans) == set(net.nodes)
    assert all((p.ns_green, p.ew_green) == (30, 30) for p in plans.values())


def test_fixed_controller_ignores_traffic_state():
    net = grid_network()
    ctrl = FixedTimeController()
    busy = Observation(5, {a: 30.0 for a in net.approaches}, {a: 3.0 for a in net.approaches}, net)
    assert ctrl.select_plans(busy) == ctrl.select_plans(_observation(net))


def test_fixed_controller_accepts_other_valid_plan_but_not_invalid_one():
    assert FixedTimeController(PLAN_NS20_EW40).plan == PLAN_NS20_EW40
    forged = object.__new__(SignalPlan)
    object.__setattr__(forged, "ns_green", 45)
    object.__setattr__(forged, "ew_green", 15)
    with pytest.raises(ValueError):
        FixedTimeController(forged)


def test_controller_interface_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Controller()


def test_custom_controller_is_injected_and_observes_state():
    """Any Controller subclass plugs into run(); it sees the evolving state."""
    net = grid_network()
    seen = []

    class Recorder(Controller):
        def select_plans(self, observation):
            seen.append((observation.cycle, sum(observation.queues.values())))
            return {n: PLAN_NS30_EW30 for n in observation.network.nodes}

    sim = Simulator(net, DemandModel(net, DemandConfig(base_rate=20.0, seed=1)))
    sim.run(Recorder(), 4)
    assert [c for c, _ in seen] == [0, 1, 2, 3]
    assert seen[0][1] == 0.0 and seen[-1][1] > 0.0  # queues built up under overload


def test_run_resets_controller_state():
    class Counting(FixedTimeController):
        resets = 0

        def reset(self):
            type(self).resets += 1

    net = grid_network()
    sim = Simulator(net, DemandModel(net, DemandConfig(seed=1)))
    sim.run(Counting(), 2)
    sim.run(Counting(), 2)
    assert Counting.resets == 2
