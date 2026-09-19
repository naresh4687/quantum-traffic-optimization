"""Phase 3.5 audit: re-verify the QUBO for every Phase 3 case/seed. No experiments are rerun.

For each of the 25 (case, seed) start states it rebuilds the QUBO and checks:
  * 18 variables with unique indices, 729 valid one-hot assignments
  * P = B + 1 with B built from all linear and quadratic traffic coefficients
  * full 2^18 scan: the global minimum is feasible, no infeasible string is at or below the
    valid optimum
  * repair lemma: every infeasible string has a one-intersection repair that lowers the energy
  * construction and exact solution are deterministic (rebuilt twice, bit-identical)
  * the rebuilt QUBO matches the plans / energy / penalty recorded in results/phase3

Run:  python scripts/audit_qubo.py [--out results/phase3/audit.json]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from qtraffic import Observation, Simulator, grid_network
from qtraffic.optimization import (
    TrafficState, build_qubo, check_full_space, check_repair_lemma, enumerate_feasible,
    penalty_lower_bound, solve_exact,
)
from qtraffic.optimization.validation import CASES, DEFAULT_SEEDS, initial_state_for


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase3/audit.json")
    ap.add_argument("--recorded", default="results/phase3/qubo_solutions.csv")
    args = ap.parse_args()

    recorded = {(r["case"], int(r["seed"])): r for r in csv.DictReader(open(args.recorded))}
    net = grid_network(2, 3)
    rows, failures = [], []
    for case in CASES:
        for seed in DEFAULT_SEEDS:
            sim = Simulator(net, case.scenario.build_demand(net, seed))
            start = initial_state_for(case, sim)
            obs = Observation(start.cycle, dict(start.queues), dict(start.in_transit), net)
            state = TrafficState.from_observation(obs)
            q1, q2 = build_qubo(state), build_qubo(state)
            s1, s2 = solve_exact(q1), solve_exact(q2)
            full = check_full_space(q1)
            rec = recorded[(case.name, seed)]
            B = penalty_lower_bound(q1.traffic, q1.variables.n_variables)
            row = {
                "case": case.name, "seed": seed,
                "n_variables": q1.variables.n_variables,
                "n_unique_variable_indices": len({v.index for v in q1.variables.variables}),
                "n_feasible_assignments": len(enumerate_feasible(q1).energies),
                "penalty": q1.penalty, "B": B, "penalty_is_B_plus_1": q1.penalty == B + 1.0,
                "full_space_evaluated": full.n_evaluated,
                "full_space_min_is_feasible": full.min_is_feasible,
                "infeasible_at_or_below_optimum": full.n_infeasible_at_or_below_optimum,
                "infeasible_without_improving_repair": check_repair_lemma(q1),
                "exact_energy": s1.energy, "exact_is_min_of_full_scan": abs(s1.energy - full.min_energy) < 1e-6,
                "qubo_deterministic": q1.coefficients == q2.coefficients and q1.offset == q2.offset,
                "solution_deterministic": s1 == s2,
                "matches_recorded_plans": [p.label for p in s1.plans.values()]
                    == [rec[f"plan_{n}"] for n in net.nodes],
                "matches_recorded_energy": abs(s1.energy - float(rec["qubo_energy"])) < 1e-6,
                "matches_recorded_penalty": abs(q1.penalty - float(rec["penalty"])) < 1e-6,
            }
            ok = (row["n_variables"] == 18 and row["n_unique_variable_indices"] == 18
                  and row["n_feasible_assignments"] == 729 and row["penalty_is_B_plus_1"]
                  and row["full_space_evaluated"] == 2**18 and row["full_space_min_is_feasible"]
                  and row["infeasible_at_or_below_optimum"] == 0
                  and row["infeasible_without_improving_repair"] == 0
                  and row["exact_is_min_of_full_scan"] and row["qubo_deterministic"]
                  and row["solution_deterministic"] and row["matches_recorded_plans"]
                  and row["matches_recorded_energy"] and row["matches_recorded_penalty"])
            row["all_checks_pass"] = ok
            rows.append(row)
            if not ok:
                failures.append((case.name, seed))
            print(f"  {case.name} seed {seed}: {'PASS' if ok else 'FAIL'}", flush=True)

    Path(args.out).write_text(json.dumps(
        {"n_checked": len(rows), "all_pass": not failures, "failures": failures, "rows": rows}, indent=1))
    print(f"{len(rows)} QUBOs audited; failures: {failures or 'none'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
