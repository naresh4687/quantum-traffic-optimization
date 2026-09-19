"""Phase 3.5 audit of the one-hot penalty bound P = B + 1.

B_u = |Q_uu| + sum_{v != u} |Q_uv| covers every linear AND quadratic (coupling) coefficient
touching bit u. These tests check that this is really sufficient, without assuming the
coefficients look like the ones the traffic model happens to produce.
"""

import numpy as np
import pytest

from qtraffic import Approach, Heading, grid_network
from qtraffic.optimization import (
    TrafficState, VariableMap, build_qubo, check_full_space, check_repair_lemma,
    penalty_lower_bound, qubo_from_traffic, solve_exact,
)

NET = grid_network(2, 3)
NODES = NET.nodes


def A(node, heading):
    return Approach(node, Heading[heading])


def state(queues=None, in_transit=None):
    q = {a: 0.0 for a in NET.approaches}
    t = {a: 0.0 for a in NET.approaches}
    q.update(queues or {})
    t.update(in_transit or {})
    return TrafficState(NET, q, t, {})


def jam_state():
    return state({A("I1", "S"): 25, A("I2", "S"): 25, A("I1", "E"): 30, A("I2", "E"): 34,
                  A("I3", "E"): 38, A("I4", "N"): 22, A("I5", "W"): 12},
                 {A("I4", "S"): 15, A("I5", "S"): 8})


def random_traffic_state(seed):
    rng = np.random.default_rng(seed)
    return TrafficState(
        NET,
        {a: float(rng.uniform(0, 40)) for a in NET.approaches},
        {a: float(rng.uniform(0, 12)) for a in NET.approaches},
        {a: float(rng.uniform(0, 16)) for a in NET.entry_approaches},
    )


def random_dense_traffic(seed, n=18, scale=1000.0):
    """Signed coefficients on every (u <= v) pair, including pairs inside one intersection."""
    rng = np.random.default_rng(seed)
    return {(u, v): float(rng.uniform(-scale, scale)) for u in range(n) for v in range(u, n)}


# -- the bound uses linear and quadratic coefficients ----------------------------------------------
def test_bound_counts_every_linear_and_quadratic_coefficient_touching_a_bit():
    traffic = {(0, 0): -3.0, (0, 1): 2.0, (0, 5): -4.0, (1, 1): 1.0, (2, 5): 10.0}
    # bit 0: 3 + 2 + 4 = 9 ; bit 1: 1 + 2 = 3 ; bit 5: 4 + 10 = 14 ; bit 2: 10
    assert penalty_lower_bound(traffic, 6) == 14.0
    assert penalty_lower_bound({(3, 3): 5.0}, 6) == 5.0  # linear alone is covered too
    assert penalty_lower_bound({(0, 1): 0.0}, 6) == 0.0


def test_default_penalty_is_the_bound_plus_one_and_deterministic():
    q = build_qubo(jam_state())
    assert q.penalty == penalty_lower_bound(q.traffic, 18) + 1.0
    assert build_qubo(jam_state()).penalty == q.penalty
    vm = VariableMap(NODES)
    rebuilt = qubo_from_traffic(vm, q.traffic)  # same construction from raw coefficients
    assert rebuilt.coefficients == q.coefficients and rebuilt.offset == q.offset


def test_qubo_from_traffic_rejects_lower_triangular_keys():
    with pytest.raises(ValueError):
        qubo_from_traffic(VariableMap(NODES), {(5, 2): 1.0})


# -- exhaustive: every infeasible string has an energy-lowering repair ---------------------------------
@pytest.mark.parametrize("make", [jam_state, lambda: random_traffic_state(1), lambda: random_traffic_state(2)])
def test_every_infeasible_string_can_be_repaired_to_strictly_lower_energy(make):
    q = build_qubo(make())
    assert check_repair_lemma(q) == 0
    check = check_full_space(q)
    assert check.n_evaluated == 2**18 and check.n_feasible == 729
    assert check.min_is_feasible and check.n_infeasible_at_or_below_optimum == 0


def test_the_repair_check_is_not_vacuous():
    """With a far-too-small penalty, many infeasible strings have no improving repair."""
    assert check_repair_lemma(build_qubo(jam_state(), penalty=1e-3)) > 0


# -- generality: signed, dense, intra-intersection coefficients ----------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bound_is_sufficient_for_arbitrary_signed_dense_coefficients(seed):
    vm = VariableMap(NODES)
    q = qubo_from_traffic(vm, random_dense_traffic(seed))
    assert q.penalty == penalty_lower_bound(q.traffic, 18) + 1.0
    assert check_repair_lemma(q) == 0
    check = check_full_space(q)
    assert check.min_is_feasible and check.n_infeasible_at_or_below_optimum == 0
    assert solve_exact(q).energy == pytest.approx(check.min_energy)


def test_a_bound_from_linear_coefficients_alone_is_not_sufficient():
    """Strong attractive coupling between two intersections: max |Q_uu| + 1 = 1, but the
    all-bits-on string beats every valid assignment, so the quadratic terms must be in B."""
    vm = VariableMap(["I1", "I2"])
    traffic = {(p, 3 + r): -100.0 for p in range(3) for r in range(3)}  # linear terms all 0
    linear_only = max((abs(c) for (u, v), c in traffic.items() if u == v), default=0.0) + 1.0
    assert linear_only == 1.0
    weak = qubo_from_traffic(vm, traffic, penalty=linear_only)
    assert not check_full_space(weak).min_is_feasible  # an invalid string wins

    correct = qubo_from_traffic(vm, traffic)
    assert correct.penalty == 301.0  # B_u = 3 couplings * 100 = 300, plus 1
    check = check_full_space(correct)
    assert check.min_is_feasible and check.n_infeasible_at_or_below_optimum == 0
    assert check_repair_lemma(correct) == 0


def test_penalty_bound_is_tight_enough_to_matter_but_not_larger_than_needed_by_construction():
    """P = B + 1 is strictly above B (needed for strictness) and P = B - 1 can fail."""
    vm = VariableMap(["I1", "I2"])
    traffic = {(p, 3 + r): -100.0 for p in range(3) for r in range(3)}
    b = penalty_lower_bound(traffic, 6)
    assert check_full_space(qubo_from_traffic(vm, traffic, penalty=b + 1)).min_is_feasible
    assert not check_full_space(qubo_from_traffic(vm, traffic, penalty=b - 250)).min_is_feasible
