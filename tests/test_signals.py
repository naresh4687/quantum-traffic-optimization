import pytest

from qtraffic import (
    CYCLE_LENGTH, PLAN_NS20_EW40, PLAN_NS30_EW30, PLAN_NS40_EW20, VALID_PLANS, Heading, Phase,
    SignalPlan,
)
from qtraffic.signals import validate_plan, validate_plan_assignment


def test_exactly_three_valid_plans_with_60s_cycle():
    assert CYCLE_LENGTH == 60
    assert [(p.ns_green, p.ew_green) for p in VALID_PLANS] == [(20, 40), (30, 30), (40, 20)]
    assert all(p.cycle_length == 60 for p in VALID_PLANS)
    assert PLAN_NS20_EW40.label == "NS20/EW40"


@pytest.mark.parametrize(
    "ns, ew",
    [(25, 35), (30, 40), (0, 60), (60, 0), (10, 50), (31, 29), (-20, 80), (20, 20), (30.0, 30.0)],
)
def test_invalid_plans_are_rejected(ns, ew):
    with pytest.raises(ValueError):
        SignalPlan(ns, ew)


def test_bool_greens_are_rejected():
    with pytest.raises(ValueError):
        SignalPlan(True, False)


def test_from_ns_builds_complement_and_validates():
    assert SignalPlan.from_ns(40) == PLAN_NS40_EW20
    with pytest.raises(ValueError):
        SignalPlan.from_ns(35)


def test_green_time_depends_on_phase_and_heading():
    plan = PLAN_NS20_EW40
    assert plan.green_time(Phase.NS) == 20
    assert plan.green_time(Phase.EW) == 40
    assert Heading.E.phase is Phase.EW and Heading.W.phase is Phase.EW
    assert Heading.N.phase is Phase.NS and Heading.S.phase is Phase.NS


def test_validate_plan_catches_plan_built_by_bypassing_init():
    forged = object.__new__(SignalPlan)
    object.__setattr__(forged, "ns_green", 45)
    object.__setattr__(forged, "ew_green", 15)
    with pytest.raises(ValueError):
        validate_plan(forged)
    with pytest.raises(ValueError):
        validate_plan((30, 30))


def test_plan_assignment_must_cover_exactly_the_nodes():
    nodes = ["A", "B"]
    ok = {"A": PLAN_NS30_EW30, "B": PLAN_NS40_EW20}
    assert validate_plan_assignment(ok, nodes) == ok
    with pytest.raises(ValueError, match="missing"):
        validate_plan_assignment({"A": PLAN_NS30_EW30}, nodes)
    with pytest.raises(ValueError, match="unknown"):
        validate_plan_assignment({**ok, "C": PLAN_NS30_EW30}, nodes)
