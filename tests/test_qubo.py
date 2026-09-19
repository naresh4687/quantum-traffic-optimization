"""Tests for the QUBO formulation, exact solver and their link to the simulator."""

import itertools

import numpy as np
import pytest

from qtraffic import (
    PLAN_NS20_EW40, PLAN_NS30_EW30, PLAN_NS40_EW20, VALID_PLANS, Approach, DemandConfig,
    DemandModel, Heading, Observation, Simulator, grid_network,
)
from qtraffic.optimization import (
    PLAN_LABELS, PLANS, QUBO, CostConfig, InfeasibleAssignment, QuboController,
    StaticPlanController, TrafficState, VariableMap, build_qubo, check_full_space,
    enumerate_feasible, penalty_lower_bound, solve_exact, surrogate_cost, traffic_costs,
)
from qtraffic.optimization.analysis import pearson, rank_average, rank_of, spearman
from qtraffic.simulator import TrafficState as SimState

NET = grid_network(2, 3)
NODES = NET.nodes
A_, B_, C_ = PLAN_NS20_EW40, PLAN_NS30_EW30, PLAN_NS40_EW20


def A(node, heading):
    return Approach(node, Heading[heading])


def state(queues=None, in_transit=None, arrivals=None):
    q = {a: 0.0 for a in NET.approaches}
    t = {a: 0.0 for a in NET.approaches}
    q.update(queues or {})
    t.update(in_transit or {})
    return TrafficState(NET, q, t, dict(arrivals or {}))


def jam_state():
    return state({A("I1", "S"): 25, A("I2", "S"): 25, A("I1", "E"): 30, A("I2", "E"): 34,
                  A("I3", "E"): 38, A("I4", "N"): 22, A("I5", "W"): 12},
                 {A("I4", "S"): 15, A("I5", "S"): 8})


def random_state(seed):
    rng = np.random.default_rng(seed)
    q = {a: float(rng.uniform(0, 40)) for a in NET.approaches}
    t = {a: float(rng.uniform(0, 12)) for a in NET.approaches}
    arr = {a: float(rng.uniform(0, 16)) for a in NET.entry_approaches}
    return state(q, t, arr)


def assignment(choice):
    """{node: plan index} -> 18 bits"""
    bits = [0] * 18
    for i, node in enumerate(NODES):
        bits[3 * i + choice[node]] = 1
    return bits


# -- 1. variable mapping ------------------------------------------------------------------
def test_there_are_exactly_18_uniquely_indexed_variables():
    vm = VariableMap(NODES)
    assert vm.n_variables == 18
    assert [v.index for v in vm.variables] == list(range(18))
    assert len(set(vm.names)) == 18
    assert vm.names[:3] == ("x[I1,A]", "x[I1,B]", "x[I1,C]")
    assert vm.names[-1] == "x[I6,C]"


def test_variable_mapping_round_trips_and_uses_documented_plans():
    vm = VariableMap(NODES)
    for v in vm.variables:
        assert vm.index(v.node, v.plan_index) == v.index == 3 * NODES.index(v.node) + v.plan_index
    assert PLAN_LABELS == ("A", "B", "C")
    assert PLANS == (PLAN_NS20_EW40, PLAN_NS30_EW30, PLAN_NS40_EW20) == VALID_PLANS
    assert list(vm.node_bits("I2")) == [3, 4, 5]
    with pytest.raises(ValueError):
        vm.index("I1", 3)
    with pytest.raises(ValueError):
        VariableMap(["I1", "I1"])


# -- 2 & 6. one-hot constraints and the decoder ------------------------------------------------
def test_every_valid_one_hot_solution_decodes_to_exactly_one_plan_per_intersection():
    q = build_qubo(state())
    space = enumerate_feasible(q)
    assert space.assignments.shape == (729, 18)
    seen = set()
    for bits, idx in zip(space.assignments, space.plan_indices):
        plans = q.decode(bits)
        assert list(plans) == list(NODES) and all(p in VALID_PLANS for p in plans.values())
        assert [PLANS.index(plans[n]) for n in NODES] == list(idx)
        assert q.encode(plans) == tuple(bits)  # decode/encode are inverse
        seen.add(tuple(idx))
    assert len(seen) == 729  # 3^6 distinct complete assignments


def test_invalid_assignments_are_rejected_by_the_decoder():
    q = build_qubo(state())
    good = assignment({n: 1 for n in NODES})
    none_on = list(good); none_on[3:6] = [0, 0, 0]
    two_on = list(good); two_on[0] = 1
    assert q.is_feasible(good) and q.violations(good) == {}
    assert q.violations(none_on) == {"I2": 0}
    assert q.violations(two_on) == {"I1": 2}
    for bad in (none_on, two_on):
        assert not q.is_feasible(bad)
        with pytest.raises(InfeasibleAssignment):
            q.decode(bad)
    for malformed in ([0] * 17, [2] + [0] * 17, [0.5] * 18):
        with pytest.raises(ValueError):
            q.energy(malformed)


# -- 3. penalty behaviour ------------------------------------------------------------------------
def test_penalty_expansion_matches_the_squared_constraint():
    """With no traffic cost the QUBO energy is exactly P * sum_i (x_iA + x_iB + x_iC - 1)^2."""
    q = build_qubo(state(), penalty=7.0)
    assert all(c == 0 for c in q.traffic.values())  # empty network: no traffic cost at all
    rng = np.random.default_rng(0)
    for _ in range(300):
        x = rng.integers(0, 2, 18)
        expected = 7.0 * sum((x[3 * i : 3 * i + 3].sum() - 1) ** 2 for i in range(6))
        assert q.energy(x) == pytest.approx(expected)
    assert q.energy(assignment({n: 0 for n in NODES})) == 0.0


@pytest.mark.parametrize("make", [jam_state, lambda: random_state(1), lambda: random_state(2)])
def test_invalid_assignment_can_never_beat_the_best_valid_one(make):
    """Brute force over all 2^18 strings with the automatically chosen penalty."""
    q = build_qubo(make())
    check = check_full_space(q)
    assert check.n_evaluated == 2**18 and check.n_feasible == 729
    assert check.min_is_feasible
    assert check.n_infeasible_at_or_below_optimum == 0
    assert check.min_energy == pytest.approx(check.min_energy_feasible)
    assert q.penalty > penalty_lower_bound(q.traffic, 18)


def test_an_insufficient_penalty_lets_an_invalid_assignment_win():
    """Shows the penalty matters: with a tiny P the unconstrained minimum is infeasible."""
    q = build_qubo(jam_state(), penalty=1e-3)
    check = check_full_space(q)
    assert not check.min_is_feasible
    assert check.min_energy < check.min_energy_feasible


# -- 4. QUBO energy = the mathematical definition -------------------------------------------------
def test_hand_computed_energy_of_a_tiny_state():
    """Only I1:S loaded, exit approaches only (no downstream terms), so it is checkable by hand.

    I4:S is empty and I1:S has q0 = 20. Under plan C (NS40): cap = 0.5*40 = 20 -> v = 20,
      W1 = 60*(20 - 20/2) = 600, leftover 0.  Under B: cap 15 -> v = 15, W1 = 60*(20-7.5) = 750,
      leftover 5; A: cap 10 -> v = 10, W1 = 900, leftover 10.
    Cycle 2 (same plan held) on I1:S: Q2 = leftover, W2 = 60*(Q2 - min(Q2, cap)/2):
      C: 0;  B: 60*(5 - 2.5) = 150;  A: 60*(10 - 5) = 300.
    I4:S receives v: Q2 = v, W2 = 60*(v - min(v, cap_{I4:S}(p_I4))/2); with I4 on B (cap 15):
      v=20 -> 60*(20-7.5) = 750; v=15 -> 450; v=10 -> 300.
    """
    s = state({A("I1", "S"): 20.0})
    q = build_qubo(s)
    plans = {n: B_ for n in NODES}
    expected = {C_: 600 + 0 + 750, B_: 750 + 150 + 450, A_: 900 + 300 + 300}
    for plan, value in expected.items():
        p = dict(plans, I1=plan)
        assert surrogate_cost(s, p) == pytest.approx(value)
        assert q.energy(q.encode(p)) == pytest.approx(value)  # empty elsewhere: nothing else costs


@pytest.mark.parametrize("make", [jam_state, lambda: random_state(3), lambda: random_state(4)])
def test_qubo_energy_equals_direct_surrogate_cost_for_random_assignments(make):
    """The assembled QUBO reproduces the per-approach definition (unary + pair tables)."""
    s = make()
    q = build_qubo(s)
    rng = np.random.default_rng(5)
    for _ in range(60):
        choice = {n: int(rng.integers(0, 3)) for n in NODES}
        plans = {n: PLANS[k] for n, k in choice.items()}
        assert q.energy(assignment(choice)) == pytest.approx(surrogate_cost(s, plans), rel=1e-9)


def test_energy_matches_an_independent_quadratic_form():
    q = build_qubo(jam_state())
    rng = np.random.default_rng(6)
    for _ in range(40):
        x = rng.integers(0, 2, 18)
        direct = q.offset + sum(c * x[u] * x[v] for (u, v), c in q.coefficients.items())
        assert q.energy(x) == pytest.approx(direct)


def test_ising_form_gives_the_same_energies():
    q = build_qubo(jam_state())
    ising = q.to_ising()
    rng = np.random.default_rng(7)
    for _ in range(50):
        x = rng.integers(0, 2, 18)
        assert ising.energy(1 - 2 * x) == pytest.approx(q.energy(x), rel=1e-9, abs=1e-6)  # z = 1-2x
    with pytest.raises(ValueError):
        ising.energy([0] * 18)


def test_all_traffic_costs_are_nonnegative_vehicle_seconds():
    for seed in range(5):
        q = build_qubo(random_state(seed))
        assert all(c >= 0 for c in q.traffic.values())


# -- coupling ---------------------------------------------------------------------------------------
def _pair_keys(s):
    costs = traffic_costs(s, VariableMap(NODES))
    return set(costs.pair)


def test_pair_terms_exist_only_between_neighbouring_intersections():
    roads = {frozenset((NODES.index(u), NODES.index(v))) for u, v, *_ in NET.roads()}
    keys = _pair_keys(random_state(0))
    assert keys and all(frozenset(k) in roads for k in keys)
    assert keys == {tuple(sorted(r)) for r in roads}  # every road couples its two ends


def test_downstream_congestion_changes_the_cost_of_pushing_traffic_into_it():
    """Same load on I1:S; making its downstream road I4:S long adds cost to every pair of
    plans on that edge (vehicles wait behind the downstream queue, and release is capped by
    the space left)."""
    base = {A("I1", "S"): 25.0}
    clear = state(base)
    jammed = state({**base, A("I4", "S"): 36.0})
    vm = VariableMap(NODES)
    i, j = vm.node_index("I1"), vm.node_index("I4")

    def pair_table(s):
        return traffic_costs(s, vm).pair[(i, j)]

    t_clear, t_jam = pair_table(clear), pair_table(jammed)
    assert (t_jam > t_clear).all()
    # the release itself is capped by the space left ahead: 40 - (36 - 15) = 19 < 20 for NS40
    assert traffic_costs(jammed, vm).pair[(i, j)][2, 1] > traffic_costs(clear, vm).pair[(i, j)][2, 1]


def test_lookahead_off_removes_the_downstream_transfer_cost():
    s = state({A("I1", "S"): 25.0, A("I4", "S"): 36.0})
    vm = VariableMap(NODES)
    on = traffic_costs(s, vm).pair[(0, 3)]
    off = traffic_costs(s, vm, CostConfig(downstream_lookahead=False)).pair[(0, 3)]
    assert not np.allclose(on, off)
    assert (on >= off).all()


def test_downstream_congestion_can_change_the_optimal_plan_of_the_upstream_intersection():
    load = {A("I1", "S"): 30.0, A("I1", "E"): 14.0}
    clear = solve_exact(build_qubo(state(load))).plans["I1"]
    jammed = solve_exact(build_qubo(state({**load, A("I4", "S"): 40.0, A("I2", "E"): 40.0}))).plans["I1"]
    assert clear == C_  # heavy NS backlog, free road ahead: NS40
    assert jammed != C_  # road ahead is full: extra NS green is wasted


def test_pair_terms_are_symmetric_in_the_orientation_they_are_stored():
    """Vertical/horizontal roads in both directions between the same pair are summed once."""
    s = random_state(8)
    vm = VariableMap(NODES)
    pair = traffic_costs(s, vm).pair
    assert all(i < j for i, j in pair) and all(t.shape == (3, 3) for t in pair.values())


# -- 5. determinism ---------------------------------------------------------------------------------
def test_same_state_gives_identical_qubo_and_identical_solution():
    s = random_state(9)
    q1, q2 = build_qubo(s), build_qubo(random_state(9))
    assert q1.coefficients == q2.coefficients and q1.offset == q2.offset and q1.penalty == q2.penalty
    s1, s2 = solve_exact(q1), solve_exact(q2)
    assert s1 == s2


def test_qubo_is_built_without_mutating_the_state():
    s = random_state(10)
    before = (dict(s.queues), dict(s.in_transit), dict(s.expected_arrivals))
    build_qubo(s)
    assert (dict(s.queues), dict(s.in_transit), dict(s.expected_arrivals)) == before


# -- exact solver --------------------------------------------------------------------------------------
def test_exact_solver_finds_the_true_minimum_and_reports_its_work():
    q = build_qubo(jam_state())
    sol = solve_exact(q)
    assert sol.n_evaluated == 729 and sol.feasible and q.is_feasible(sol.assignment)
    assert sol.energy == pytest.approx(q.energy(sol.assignment))
    brute = min(q.energy(assignment(dict(zip(NODES, c)))) for c in itertools.product(range(3), repeat=6))
    assert sol.energy == pytest.approx(brute)
    full = check_full_space(q)
    assert sol.energy == pytest.approx(full.min_energy) and full.n_evaluated == 2**18


def test_exact_solution_matches_the_surrogate_cost_of_its_plans():
    s = jam_state()
    sol = solve_exact(build_qubo(s))
    assert sol.energy == pytest.approx(surrogate_cost(s, sol.plans))


def test_ties_prefer_the_neutral_plan_deterministically():
    sol = solve_exact(build_qubo(state()))  # empty network: every assignment has energy 0
    assert sol.n_optimal == 729
    assert all(p == PLAN_NS30_EW30 for p in sol.plans.values())


def test_directional_load_selects_the_matching_plan():
    ns = solve_exact(build_qubo(state({A("I2", "S"): 30.0}))).plans["I2"]
    ew = solve_exact(build_qubo(state({A("I1", "E"): 30.0}))).plans["I1"]
    assert ns == PLAN_NS40_EW20 and ew == PLAN_NS20_EW40


# -- 7. simulator integration ------------------------------------------------------------------------------
def test_decoded_solution_runs_in_the_simulator_and_conserves_vehicles():
    demand = DemandModel(NET, DemandConfig.from_level("high", seed=2))
    sim = Simulator(NET, demand)
    sol = solve_exact(build_qubo(jam_state()))
    result = sim.run(StaticPlanController(sol.plans), 20)
    assert all(rec.plans == sol.plans for rec in result.history)
    m = result.metrics
    assert m.vehicles_exited > 0
    assert m.vehicles_entered == pytest.approx(
        m.vehicles_exited + sum(result.final_queues.values()) + sum(result.final_in_transit.values()))


def test_qubo_controller_drives_the_simulator_with_valid_plans():
    sim = Simulator(NET, DemandModel(NET, DemandConfig.from_level("medium", seed=1)))
    ctrl = QuboController(record=True)
    result = sim.run(ctrl, 8)
    assert len(ctrl.energies) == 8
    assert all(p in VALID_PLANS for rec in result.history for p in rec.plans.values())
    again = sim.run(QuboController(), 8)  # deterministic: same run, same plans and metrics
    assert [r.plans for r in again.history] == [r.plans for r in result.history]
    assert again.metrics == result.metrics


def test_surrogate_matches_the_simulator_exactly_when_nothing_is_blocked():
    """One cycle under the chosen plans plus one more under the same plans, no new demand:
    the surrogate's two-cycle waiting equals the simulator's, for every assignment tried."""
    from tests.helpers import ScriptedDemand

    sim = Simulator(NET, ScriptedDemand())  # no external demand
    q = {A("I1", "S"): 18.0, A("I2", "S"): 12.0, A("I4", "N"): 9.0, A("I3", "W"): 14.0,
         A("I1", "E"): 10.0, A("I6", "E"): 6.0}
    t = {A("I4", "S"): 7.0, A("I5", "S"): 4.0, A("I2", "E"): 5.0}
    s = state(q, t)
    rng = np.random.default_rng(12)
    for _ in range(25):
        plans = {n: PLANS[int(rng.integers(0, 3))] for n in NODES}
        st = SimState(0, dict(s.queues), dict(s.in_transit))
        ctrl = StaticPlanController(plans)
        real = sim.step(st, ctrl).waiting_time + sim.step(st, ctrl).waiting_time
        assert surrogate_cost(s, plans) == pytest.approx(real, rel=1e-9)


def test_surrogate_with_expected_arrivals_matches_the_simulator_on_constant_demand():
    from tests.helpers import ScriptedDemand

    arrivals = {a: 6.0 for a in NET.entry_approaches}
    sim = Simulator(NET, ScriptedDemand(constant=arrivals))
    s = state({A("I1", "S"): 10.0, A("I5", "N"): 8.0}, {A("I4", "S"): 3.0}, arrivals)
    for choice in ({n: 0 for n in NODES}, {n: 1 for n in NODES}, {n: 2 for n in NODES}):
        plans = {n: PLANS[k] for n, k in choice.items()}
        st = SimState(0, dict(s.queues), dict(s.in_transit))
        ctrl = StaticPlanController(plans)
        real = sim.step(st, ctrl).waiting_time + sim.step(st, ctrl).waiting_time
        assert surrogate_cost(s, plans) == pytest.approx(real, rel=1e-9)


def test_reference_plan_lookahead_is_available_as_the_superseded_variant():
    s = state({A("I1", "S"): 25.0})
    held = build_qubo(s)
    ref = build_qubo(s, CostConfig(horizon_plan="reference"))
    assert held.coefficients != ref.coefficients
    with pytest.raises(ValueError):
        CostConfig(horizon_plan="nonsense")
    with pytest.raises(ValueError):
        CostConfig(saturation_flow=0)


def test_state_from_observation_copies_the_observation():
    obs = Observation(3, {a: 1.0 for a in NET.approaches}, {a: 0.5 for a in NET.approaches}, NET)
    ts = TrafficState.from_observation(obs, {A("I1", "E"): 4.0})
    assert ts.cycle == 3 and ts.expected_arrivals == {A("I1", "E"): 4.0}
    assert ts.queues == dict(obs.queues) and ts.queues is not obs.queues


def test_qubo_matrix_view_agrees_with_dictionary_form():
    q = build_qubo(jam_state())
    m = q.matrix()
    assert m.shape == (18, 18) and np.allclose(np.tril(m, -1), 0)
    assert isinstance(q, QUBO) and q.as_dict()["penalty"] == q.penalty


# -- analysis helpers ---------------------------------------------------------------------------------------
def test_rank_and_correlation_helpers():
    assert list(rank_average(np.array([10, 20, 20, 30]))) == [1.0, 2.5, 2.5, 4.0]
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert pearson(x, 2 * x + 1) == pytest.approx(1.0)
    assert pearson(x, -x) == pytest.approx(-1.0)
    assert spearman(x, x**3) == pytest.approx(1.0)  # monotone but non-linear
    assert pearson(x, x**3) < 1.0
    assert np.isnan(pearson(x, np.ones(5)))
    assert rank_of(np.array([5.0, 1.0, 3.0]), 2) == 2
    with pytest.raises(ValueError):
        pearson(x, x[:3])
