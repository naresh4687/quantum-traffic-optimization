"""Glue between the optimisation layer and the simulator's ``Controller`` interface.

    Observation -> TrafficState -> build_qubo -> solver -> plans -> Simulator

The solver is injected, so a later QAOA solver can replace ``solve_exact`` without
touching anything else. These classes live in the optimisation package on purpose: the
``controllers`` package is unchanged.
"""

from __future__ import annotations

from typing import Callable, Mapping

from ..controllers.base import Controller, Observation
from ..signals import SignalPlan, validate_plan_assignment
from .exact import ExactSolution, solve_exact
from .qubo import QUBO, CostConfig, TrafficState, build_qubo

Solver = Callable[[QUBO], ExactSolution]


class StaticPlanController(Controller):
    """Applies one fixed plan per intersection every cycle (e.g. a decoded QUBO solution)."""

    def __init__(self, plans: Mapping[str, SignalPlan], label: str = "StaticPlan"):
        self.plans = dict(plans)
        self.label = label

    @property
    def name(self) -> str:
        return self.label

    def select_plans(self, observation: Observation) -> Mapping[str, SignalPlan]:
        return validate_plan_assignment(self.plans, observation.network.nodes)


class QuboController(Controller):
    """Re-builds and re-solves the QUBO from the observed state every cycle.

    Uses only what the Phase 2 controllers see (queues and in-transit vehicles); no
    external-arrival forecast is supplied, so ``expected_arrivals`` is empty.
    """

    def __init__(self, solver: Solver = solve_exact, cost_config: CostConfig | None = None,
                 record: bool = False):
        self.solver = solver
        self.cost_config = cost_config
        self.record = record
        self.energies: list[float] = []

    @property
    def name(self) -> str:
        return "QuboController"

    def reset(self) -> None:
        self.energies = []

    def select_plans(self, observation: Observation) -> Mapping[str, SignalPlan]:
        qubo = build_qubo(TrafficState.from_observation(observation), self.cost_config)
        solution = self.solver(qubo)
        if self.record:
            self.energies.append(solution.energy)
        return solution.plans
