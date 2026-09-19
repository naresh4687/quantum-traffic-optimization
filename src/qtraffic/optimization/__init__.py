"""QUBO formulation and exact classical solver for signal-plan selection."""

from .adapter import QuboController, StaticPlanController
from .exact import (
    ExactSolution, FeasibleSpace, FullSpaceCheck, check_full_space, check_repair_lemma, enumerate_feasible,
    solve_exact,
)
from .qaoa import (
    ExactComparison, IsingHamiltonian, QAOAConfig, QAOAEngine, QAOAResult, SampleSummary,
    build_qaoa_circuit, compare_with_exact, evaluate_samples, key_to_bits, qubo_to_ising, run_qaoa,
    tqa_initial_angles,
)
from .qubo import (
    PLAN_LABELS, PLANS, QUBO, CostConfig, InfeasibleAssignment, IsingModel, TrafficCosts,
    TrafficState, Variable, VariableMap, build_qubo, penalty_lower_bound,
    qubo_from_traffic, surrogate_cost, traffic_costs,
)

__all__ = [
    "ExactComparison", "IsingHamiltonian", "QAOAConfig", "QAOAEngine", "QAOAResult",
    "SampleSummary", "build_qaoa_circuit", "compare_with_exact", "evaluate_samples",
    "key_to_bits", "qubo_to_ising", "run_qaoa", "tqa_initial_angles",
    "CostConfig", "ExactSolution", "FeasibleSpace", "FullSpaceCheck", "InfeasibleAssignment",
    "IsingModel", "PLANS", "PLAN_LABELS", "QUBO", "QuboController", "StaticPlanController",
    "TrafficCosts", "TrafficState", "Variable", "VariableMap", "build_qubo",
    "check_full_space", "check_repair_lemma", "enumerate_feasible", "penalty_lower_bound",
    "qubo_from_traffic", "solve_exact", "surrogate_cost", "traffic_costs",
]
