"""Phase 3: exact-QUBO signal plans vs Fixed and Adaptive, on identical traffic.

Run:  python scripts/run_phase3_qubo.py [--seeds 0 1 2 3 4] [--cycles 60] [--out results/phase3]

This is exact classical optimisation of a QUBO surrogate; there is nothing quantum here.
Writes (every number comes from the simulator or the exact solver):
  runs.csv                 case x seed x controller: simulator metrics (+ QUBO plans/energy)
  qubo_solutions.csv       case x seed: the exact optimum (plans I1..I6, energy, one-hot validity, ...)
  summary.csv              per case x controller: mean over seeds, % difference vs Fixed and vs Adaptive
  correlation.csv          case x seed: QUBO energy vs simulator waiting over all 729 static assignments
  example_qubo.json        one complete QUBO (variables, coefficients, penalty, solution)
  results.json             everything above, unflattened
  verification.json        digests showing identical initial state and demand for every controller
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from qtraffic import grid_network
from qtraffic.experiments import METRICS, pct_diff, verdict
from qtraffic.optimization import CostConfig, TrafficState, build_qubo, solve_exact, traffic_costs
from qtraffic.optimization.validation import (
    CASES, DEFAULT_CYCLES, DEFAULT_SEEDS, evaluate_case, initial_state_for,
)
from qtraffic.controllers import Observation
from qtraffic.simulator import Simulator

CONTROLLERS = ("fixed", "adaptive", "qubo_static", "qubo_receding")
COMPARED = ("adaptive", "qubo_static", "qubo_receding")
SUMMARY_METRICS = {**METRICS, "waiting_per_admitted": False}  # lower is better


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summary_rows(results: list[dict]) -> list[dict]:
    rows = []
    for case in dict.fromkeys(r["case"] for r in results):
        group = [r for r in results if r["case"] == case]
        mean = {c: {m: statistics.fmean(g["controllers"][c][m] for g in group) for m in SUMMARY_METRICS}
                for c in CONTROLLERS}
        for c in CONTROLLERS:
            row = {"case": case, "controller": c, "seeds": len(group), **{m: mean[c][m] for m in SUMMARY_METRICS}}
            for ref in ("fixed", "adaptive"):
                if c == ref or (ref == "adaptive" and c == "fixed"):
                    continue
                for m in SUMMARY_METRICS:
                    row[f"{m}_pct_vs_{ref}"] = pct_diff(mean[ref][m], mean[c][m])
                # per-seed verdict on waiting per admitted vehicle, and on throughput
                for m in ("waiting_per_admitted", "throughput_per_hour"):
                    v = [verdict(m if m in METRICS else "total_waiting_time",
                                 g["controllers"][ref][m], g["controllers"][c][m]) for g in group]
                    row[f"{m}_seeds_better_than_{ref}"] = v.count("better")
                    row[f"{m}_seeds_worse_than_{ref}"] = v.count("worse")
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    ap.add_argument("--out", default="results/phase3")
    ap.add_argument("--horizon-plan", choices=["held", "reference"], default="held",
                    help='plan assumed in the look-ahead cycle ("reference" = the superseded v1 model)')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = CostConfig(horizon_plan=args.horizon_plan)
    results = []
    for case in CASES:
        for seed in args.seeds:
            results.append(evaluate_case(case, seed, args.cycles, cost_config=cfg))
            print(f"  done {case.name} seed {seed}", flush=True)

    runs, solutions, corr = [], [], []
    for r in results:
        q = r["qubo"]
        solutions.append({
            "case": r["case"], "seed": r["seed"], **{f"plan_{n}": p for n, p in q["plans"].items()},
            "qubo_energy": q["energy"], "one_hot_valid": q["feasible"], "n_variables": q["n_variables"],
            "n_feasible_evaluated": q["n_evaluated_feasible"], "n_optimal_ties": q["n_optimal_ties"],
            "penalty": q["penalty"], "penalty_bound_B": q["penalty_bound_B"],
            "full_space_evaluated": q["full_space"]["n_evaluated"],
            "full_space_min_is_feasible": q["full_space"]["min_is_feasible"],
            "full_space_infeasible_at_or_below_optimum": q["full_space"]["infeasible_at_or_below_optimum"],
        })
        for c in CONTROLLERS:
            row = {"case": r["case"], "seed": r["seed"], "controller": c, "cycles": r["cycles"],
                   "start_cycle": r["start_cycle"], **r["controllers"][c]}
            if c == "qubo_static":
                row.update({f"plan_{n}": p for n, p in q["plans"].items()})
                row["one_hot_valid"] = q["feasible"]
            runs.append(row)
        if "correlation" in r:
            cr = dict(r["correlation"])
            cr["best_static_plans"] = "/".join(cr["best_static_plans"].values())
            corr.append({"case": r["case"], "seed": r["seed"], **cr})

    write_csv(out / "runs.csv", runs)
    write_csv(out / "qubo_solutions.csv", solutions)
    write_csv(out / "summary.csv", summary_rows(results))
    write_csv(out / "correlation.csv", corr)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "verification.json").write_text(json.dumps(
        [{"case": r["case"], "seed": r["seed"], "initial_state_sha256": r["initial_state_digest"],
          "demand_sha256": r["demand_digest"], "cycles": r["cycles"],
          "identical_for_all_controllers": True} for r in results], indent=1))

    # one complete QUBO, for inspection
    case = next(c for c in CASES if c.name == "ns_heavy")
    net = grid_network(2, 3)
    sim = Simulator(net, case.scenario.build_demand(net, 0))
    start = initial_state_for(case, sim)
    state = TrafficState.from_observation(Observation(start.cycle, dict(start.queues), dict(start.in_transit), net))
    qubo = build_qubo(state, cfg)
    costs = traffic_costs(state, qubo.variables, cfg)
    sol = solve_exact(qubo)
    (out / "example_qubo.json").write_text(json.dumps({
        "case": case.name, "seed": 0, "state_cycle": start.cycle,
        "queues": {str(a): round(v, 4) for a, v in start.queues.items() if v},
        "in_transit": {str(a): round(v, 4) for a, v in start.in_transit.items() if v},
        **qubo.as_dict(),
        "unary_costs": {qubo.variables.nodes[i]: [float(x) for x in t] for i, t in costs.unary.items()},
        "pair_costs": {f"{qubo.variables.nodes[i]}-{qubo.variables.nodes[j]}": t.tolist()
                       for (i, j), t in costs.pair.items()},
        "solution": {"plans": {n: p.label for n, p in sol.plans.items()}, "energy": sol.energy,
                     "bits": list(sol.assignment)},
    }, indent=1))
    print(f"wrote {len(results)} case/seed results to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
