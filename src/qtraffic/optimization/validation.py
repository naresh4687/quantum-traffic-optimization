"""Phase 3 validation: exact-QUBO plans vs Fixed and Adaptive, on identical inputs.

For each (case, seed):

1. Build the starting traffic state: an explicit synthetic state, or the state after
   ``warmup`` cycles of ``FixedTimeController`` from an empty network (same for everyone).
2. Build the QUBO from that state, solve it exactly, decode six plans.
3. From the *same* state, with the *same* demand stream from the state's cycle onward, run
   for ``cycles`` cycles under: Fixed, Adaptive, the static QUBO plans, and a QUBO
   controller that re-solves every cycle. All numbers come from the simulator.
4. Simulate all 729 static assignments the same way and compare their QUBO energy with
   the simulator's waiting time (Pearson / Spearman), plus a short-horizon fidelity check.

The QUBO is built from queues and in-transit vehicles only (no demand forecast), the
same information the Phase 2 controllers use.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..controllers import AdaptiveController, Controller, FixedTimeController, Observation
from ..experiments import SCENARIOS, Scenario, demand_digest, summarise
from ..metrics import compute_metrics
from ..network import Approach, Heading, Network, grid_network
from ..simulator import SimulationResult, Simulator, TrafficState as SimState
from .adapter import QuboController, StaticPlanController
from .analysis import pearson, rank_of, spearman
from .exact import check_full_space, enumerate_feasible, solve_exact
from .qubo import PLANS, CostConfig, TrafficState, build_qubo, penalty_lower_bound

DEFAULT_WARMUP = 30
DEFAULT_CYCLES = 60
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
CONTROLLER_KEYS = ("fixed", "adaptive", "qubo_static", "qubo_receding")


class NoDemand:
    """Demand source with no external arrivals (used for the short-horizon fidelity check)."""

    def arrivals(self, cycle: int) -> dict[Approach, float]:
        return {}


def _congested_state(net: Network) -> SimState:
    """EW corridors jammed towards their downstream end (limit of each road is 40)."""
    queues = {a: 0.0 for a in net.approaches}
    for node, level in {"I1": 30, "I2": 34, "I3": 38, "I4": 30, "I5": 34, "I6": 38}.items():
        queues[Approach(node, Heading.E)] = float(level)
    return SimState(cycle=0, queues=queues, in_transit={a: 0.0 for a in net.approaches})


@dataclass(frozen=True)
class Case:
    name: str
    description: str
    scenario: Scenario
    warmup: int = DEFAULT_WARMUP
    initial_state: Callable[[Network], SimState] | None = None


_SC = {s.name: s for s in SCENARIOS}
CASES: tuple[Case, ...] = (
    Case("balanced_medium", "uniform medium demand, state after 30 fixed-time cycles", _SC["medium"]),
    Case("ns_heavy", "NS-heavy demand, state after 30 fixed-time cycles", _SC["ns_heavy"]),
    Case("ew_heavy", "EW-heavy demand, state after 30 fixed-time cycles", _SC["ew_heavy"]),
    Case("time_varying", "time-varying demand; state at cycle 30, then balanced (30-59) and "
         "EW-heavy (60-89) blocks", _SC["time_varying"]),
    Case("downstream_congested", "EW-heavy demand from a jammed start: eastbound queues 30/34/38 "
         "along each row (I1,I2,I3 and I4,I5,I6)", _SC["ew_heavy"], warmup=0,
         initial_state=_congested_state),
)


# -- helpers -----------------------------------------------------------------------------
def copy_state(state: SimState) -> SimState:
    return SimState(state.cycle, dict(state.queues), dict(state.in_transit))


def initial_state_for(case: Case, sim: Simulator) -> SimState:
    if case.initial_state is not None:
        return case.initial_state(sim.network)
    state, fixed = sim.initial_state(), FixedTimeController()
    for _ in range(case.warmup):
        sim.step(state, fixed)
    return state


def state_digest(state: SimState) -> str:
    h = hashlib.sha256(f"cycle={state.cycle}\n".encode())
    for a in sorted(state.queues, key=str):
        h.update(f"{a}|{state.queues[a]!r}|{state.in_transit[a]!r}\n".encode())
    return h.hexdigest()


def run_from_state(sim: Simulator, state: SimState, controller: Controller, cycles: int) -> SimulationResult:
    """Like ``Simulator.run`` but starting from ``state`` (which is left untouched)."""
    st = copy_state(state)
    controller.reset()
    history = tuple(sim.step(st, controller) for _ in range(cycles))
    return SimulationResult(controller.name, history, compute_metrics(history),
                            dict(st.queues), dict(st.in_transit))


def _metrics(result: SimulationResult, nodes) -> dict:
    row = summarise(result, nodes)
    row["waiting_per_admitted"] = (
        row["total_waiting_time"] / row["vehicles_entered"] if row["vehicles_entered"] else 0.0
    )
    return row


def _plan_row(plans) -> dict[str, str]:
    return {node: plan.label for node, plan in plans.items()}


def _finite(x: float) -> float | None:
    return None if x != x else float(x)


# -- one case ------------------------------------------------------------------------------
def evaluate_case(
    case: Case,
    seed: int,
    cycles: int = DEFAULT_CYCLES,
    correlation: bool = True,
    network: Network | None = None,
    cost_config: CostConfig | None = None,
) -> dict:
    net = network or grid_network(2, 3)
    demand = case.scenario.build_demand(net, seed)
    sim = Simulator(net, demand)
    start = initial_state_for(case, sim)
    nodes = net.nodes

    # QUBO from the starting state, solved exactly
    observation = Observation(start.cycle, dict(start.queues), dict(start.in_transit), net)
    qubo = build_qubo(TrafficState.from_observation(observation), cost_config)
    solution = solve_exact(qubo)
    full = check_full_space(qubo)

    controllers: dict[str, Controller] = {
        "fixed": FixedTimeController(),
        "adaptive": AdaptiveController(),
        "qubo_static": StaticPlanController(solution.plans, "QuboExactStatic"),
        "qubo_receding": QuboController(cost_config=cost_config, record=True),
    }
    results = {k: run_from_state(sim, start, c, cycles) for k, c in controllers.items()}

    src_digest = demand_digest([demand.arrivals(start.cycle + c) for c in range(cycles)])
    for name, res in results.items():
        if demand_digest([r.demand for r in res.history]) != src_digest:
            raise AssertionError(f"demand mismatch for {name} in {case.name} seed {seed}")

    rows = {k: _metrics(r, nodes) for k, r in results.items()}
    rows["qubo_static"]["qubo_energy"] = solution.energy
    receding = controllers["qubo_receding"]
    rows["qubo_receding"]["qubo_energy"] = float(np.mean(receding.energies))

    out = {
        "case": case.name, "seed": seed, "cycles": cycles, "start_cycle": start.cycle,
        "initial_state_digest": state_digest(start), "demand_digest": src_digest,
        "qubo": {
            "n_variables": qubo.variables.n_variables,
            "penalty": qubo.penalty,
            "penalty_bound_B": penalty_lower_bound(qubo.traffic, qubo.variables.n_variables),
            "energy": solution.energy,
            "plans": _plan_row(solution.plans),
            "feasible": solution.feasible,
            "n_evaluated_feasible": solution.n_evaluated,
            "n_optimal_ties": solution.n_optimal,
            "full_space": {
                "n_evaluated": full.n_evaluated, "min_energy": full.min_energy,
                "min_energy_feasible": full.min_energy_feasible,
                "min_is_feasible": full.min_is_feasible,
                "infeasible_at_or_below_optimum": full.n_infeasible_at_or_below_optimum,
            },
        },
        "controllers": rows,
    }
    if correlation:
        out["correlation"] = correlate(sim, start, qubo, solution.assignment, cycles, cost_config)
    return out


# -- does minimising the QUBO minimise real traffic cost? ----------------------------------
def correlate(sim: Simulator, start: SimState, qubo, chosen_assignment, cycles: int,
              cost_config: CostConfig | None = None) -> dict:
    """Simulate all 729 static assignments and compare their QUBO energy with real waiting."""
    net = sim.network
    space = enumerate_feasible(qubo)
    chosen = int(np.flatnonzero((space.assignments == np.array(chosen_assignment)).all(axis=1))[0])
    short_sim = Simulator(net, NoDemand())
    cfg = cost_config or CostConfig()

    waiting = np.zeros(len(space.energies))
    avg_queue = np.zeros_like(waiting)
    exited = np.zeros_like(waiting)
    short_wait = np.zeros_like(waiting)
    for k, row in enumerate(space.plan_indices):
        plans = {n: PLANS[int(p)] for n, p in zip(net.nodes, row)}
        m = run_from_state(sim, start, StaticPlanController(plans), cycles).metrics
        waiting[k], avg_queue[k], exited[k] = m.total_waiting_time, m.average_queue_length, m.vehicles_exited
        # surrogate fidelity: the QUBO's own two-cycle horizon (no new demand); cycle 2 runs
        # the plan the cost model assumes there (the same plan, or the reference plan in v1)
        second = StaticPlanController(plans) if cfg.horizon_plan == "held" else FixedTimeController(cfg.reference_plan)
        st = copy_state(start)
        short_wait[k] = (short_sim.step(st, StaticPlanController(plans)).waiting_time
                         + short_sim.step(st, second).waiting_time)

    e = space.energies
    best = int(np.argmin(waiting))
    return {
        "n_assignments": len(e),
        "pearson_energy_vs_waiting": _finite(pearson(e, waiting)),
        "spearman_energy_vs_waiting": _finite(spearman(e, waiting)),
        "pearson_energy_vs_avg_queue": _finite(pearson(e, avg_queue)),
        "spearman_energy_vs_avg_queue": _finite(spearman(e, avg_queue)),
        "spearman_energy_vs_neg_exited": _finite(spearman(e, -exited)),
        "qubo_choice_rank_by_waiting": rank_of(waiting, chosen),  # 1 = best of 729
        "qubo_choice_waiting": float(waiting[chosen]),
        "best_static_waiting": float(waiting[best]),
        "worst_static_waiting": float(waiting.max()),
        "best_static_plans": _plan_row({n: PLANS[int(p)] for n, p in zip(net.nodes, space.plan_indices[best])}),
        "qubo_gap_to_best_static_pct": float((waiting[chosen] - waiting[best]) / waiting[best] * 100.0),
        "energy_rank_of_best_static": rank_of(e, best),
        "short_horizon_pearson": _finite(pearson(e, short_wait)),
        "short_horizon_spearman": _finite(spearman(e, short_wait)),
        "short_horizon_max_abs_rel_error": float(np.max(np.abs(e - short_wait) / np.maximum(short_wait, 1.0))),
    }
