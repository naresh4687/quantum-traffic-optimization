"""Phase 4: QAOA (Qiskit Aer) on the Phase 3 QUBOs, compared with the exact solver and validated
in the traffic simulator.

Run:  python scripts/run_phase4_qaoa.py [--seed 0] [--cycles 60] [--out results/phase4]

Per case: same traffic state and QUBO as Phase 3 -> exact solution (ground truth, used only for
comparison) -> QAOA p=1 and p=2 with ONE fixed configuration -> sampled bitstrings scored by the
existing QUBO evaluator -> best sampled feasible plans run in the real simulator next to Fixed,
Adaptive and the exact-QUBO plans. Nothing is tuned per case.

Variants (all use the identical QAOAConfig apart from p):
  qaoa_p1        p=1, deterministic TQA start angles
  qaoa_p2        p=2, deterministic TQA start angles
  qaoa_p2_warm   p=2 started from the p=1 optimum with a zero second layer (diagnostic: separates
                 "p=2 cannot help" from "the optimiser got stuck"; still uses nothing but <H>)

Writes to results/phase4/ (Phase 3 results are only read):
  qaoa_runs.csv          simulator metrics per case x controller
  qaoa_solutions.csv     QAOA vs exact per case x variant (energies, ratios, probabilities, angles, cost)
  qaoa_counts.json       every sampled bitstring and its count (variable order x_0..x_17)
  qaoa_parameters.json   configuration, initial and optimised angles, objective trace
  qaoa_verification.json checks (energy equivalence, bit order, determinism, Phase 3 agreement, ...)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import qiskit
import qiskit_aer
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qtraffic import AdaptiveController, FixedTimeController, Observation, Simulator, grid_network
from qtraffic.optimization import (
    QAOAConfig, StaticPlanController, TrafficState, build_qubo, compare_with_exact,
    enumerate_feasible, key_to_bits, qubo_to_ising, run_qaoa, solve_exact,
)
from qtraffic.optimization.qaoa import variable_order_string
from qtraffic.optimization.validation import CASES, _metrics, initial_state_for, run_from_state

RANDOM_BASELINE_SEED = 12345


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plans_str(plans) -> str:
    return "/".join(p.label for p in plans.values()) if plans else ""


def random_baseline(qubo, shots: int, n: int) -> dict:
    """Uniform random bitstrings with a fixed seed: what unstructured sampling would give."""
    rng = np.random.default_rng(RANDOM_BASELINE_SEED)
    best, n_feasible = None, 0
    for bits in rng.integers(0, 2, size=(shots, n)):
        if qubo.is_feasible(bits):
            n_feasible += 1
            e = qubo.energy(bits)
            best = e if best is None or e < best else best
    return {"random_feasible_samples": n_feasible, "random_best_feasible_energy": best}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cycles", type=int, default=60)
    ap.add_argument("--out", default="results/phase4")
    ap.add_argument("--phase3", default="results/phase3")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = QAOAConfig()  # the single configuration used everywhere
    net = grid_network(2, 3)

    phase3_runs = {(r["case"], r["controller"]): r for r in csv.DictReader(open(f"{args.phase3}/runs.csv"))
                   if int(r["seed"]) == args.seed}
    phase3_sol = {r["case"]: r for r in csv.DictReader(open(f"{args.phase3}/qubo_solutions.csv"))
                  if int(r["seed"]) == args.seed}

    runs, solutions, counts_out, params_out, verification_cases = [], [], {}, {}, []
    for case in CASES:
        sim = Simulator(net, case.scenario.build_demand(net, args.seed))
        start = initial_state_for(case, sim)
        state = TrafficState.from_observation(
            Observation(start.cycle, dict(start.queues), dict(start.in_transit), net))
        qubo = build_qubo(state)
        exact = solve_exact(qubo)
        ham = qubo_to_ising(qubo)
        space = enumerate_feasible(qubo)  # evaluation only
        baseline = random_baseline(qubo, base.shots, 18)

        results = {}
        results["qaoa_p1"] = run_qaoa(qubo, base.with_p(1))
        results["qaoa_p2"] = run_qaoa(qubo, base.with_p(2))
        p1 = results["qaoa_p1"]
        results["qaoa_p2_warm"] = run_qaoa(
            qubo, base.with_p(2), initial_angles=[p1.gammas[0], 0.0, p1.betas[0], 0.0])
        print(f"  {case.name}: " + ", ".join(
            f"{k} {r.best_feasible_energy if r.best_feasible_energy is None else round(r.best_feasible_energy)}"
            for k, r in results.items()) + f" | exact {round(exact.energy)}", flush=True)

        # simulator: identical start state and demand for every controller
        controllers = {"fixed": FixedTimeController(), "adaptive": AdaptiveController(),
                       "qubo_exact": StaticPlanController(exact.plans, "QuboExactStatic")}
        for k, r in results.items():
            if r.best_feasible_plans is not None:
                controllers[k] = StaticPlanController(r.best_feasible_plans, k)
        sim_metrics = {k: _metrics(run_from_state(sim, start, c, args.cycles), net.nodes)
                       for k, c in controllers.items()}

        for k, m in sim_metrics.items():
            row = {"case": case.name, "seed": args.seed, "controller": k, "cycles": args.cycles, **m}
            if k == "qubo_exact":
                row.update(qubo_energy=exact.energy, plans=plans_str(exact.plans),
                           **{f"plan_{n}": p.label for n, p in exact.plans.items()})
            elif k in results:
                r = results[k]
                row.update(qubo_energy=r.best_feasible_energy, plans=plans_str(r.best_feasible_plans),
                           **{f"plan_{n}": p.label for n, p in r.best_feasible_plans.items()})
            runs.append(row)
        for k in results:
            if results[k].best_feasible_plans is None:
                runs.append({"case": case.name, "seed": args.seed, "controller": k,
                             "note": "no feasible sample: not simulated (no repair applied)"})

        feas_e = space.energies
        for k, r in results.items():
            cmp = compare_with_exact(r, qubo, exact)
            feas_counts = [(r.sampled_energy[nm], c) for nm, c in r.counts.items()
                           if qubo.is_feasible([int(ch) for ch in nm])]
            mean_feas_sample = (sum(e * c for e, c in feas_counts) / sum(c for _, c in feas_counts)
                                if feas_counts else None)
            solutions.append({
                "case": case.name, "seed": args.seed, "variant": k, "p": r.config.p,
                "n_qubits": r.n_qubits, "shots": r.shots, "optimizer": r.config.optimizer,
                "maxiter": r.config.maxiter, "optimizer_iterations": r.n_iterations,
                "function_evaluations": r.n_function_evals, "optimizer_converged": r.optimizer_success,
                "exact_energy": cmp.exact_energy, "exact_plans": plans_str(cmp.exact_plans),
                "exact_n_optimal_assignments": cmp.exact_n_optimal,
                "qaoa_best_feasible_energy": cmp.qaoa_energy, "qaoa_plans": plans_str(r.best_feasible_plans),
                "qaoa_best_sampled_energy_any": r.best_sampled_energy,
                "energy_gap": cmp.energy_gap, "approximation_ratio": cmp.approximation_ratio,
                "optimum_over_qaoa": cmp.optimum_over_qaoa, "found_exact_optimum": cmp.found_optimum,
                "optimum_probability_sampled": cmp.optimum_probability_sampled,
                "optimum_probability_statevector": cmp.optimum_probability_exact,
                "uniform_optimum_probability": cmp.uniform_optimum_probability,
                "feasible_rate_sampled": r.feasible_rate, "feasible_probability_statevector": r.feasible_probability,
                "uniform_feasible_probability": cmp.uniform_feasible_probability,
                "unique_bitstrings_sampled": r.n_unique_sampled, "unique_feasible_sampled": r.n_unique_feasible,
                "expected_energy_initial": r.initial_expectation, "expected_energy_optimised": r.expectation,
                "mean_energy_of_feasible_samples": mean_feas_sample,
                "mean_energy_of_all_729_feasible": float(feas_e.mean()),
                "feasible_energy_spread_over_scale": float((feas_e.max() - feas_e.min()) / r.scale),
                "hamiltonian_scale": r.scale,
                "gammas": " ".join(f"{v:.6f}" for v in r.gammas), "betas": " ".join(f"{v:.6f}" for v in r.betas),
                "logical_depth": r.logical_depth, "logical_gates": sum(r.logical_ops.values()),
                "basis_depth": r.basis_depth, "basis_gates": sum(r.basis_ops.values()),
                "basis_cx": r.basis_ops.get("cx", 0),
                "optimize_seconds": r.optimize_seconds, "sample_seconds": r.sample_seconds,
                "total_seconds": r.total_seconds, **baseline,
            })
            counts_out.setdefault(case.name, {})[k] = {
                "bit_order": "x_0 x_1 ... x_17 left to right (variable order; NOT Qiskit little-endian)",
                "shots": r.shots, "counts": dict(sorted(r.counts.items(), key=lambda kv: -kv[1])),
            }
            params_out.setdefault(case.name, {})[k] = {
                "config": asdict(r.config), "hamiltonian_scale": r.scale, "hamiltonian_constant": r.constant,
                "initial_angles": list(r.initial_angles), "gammas": list(r.gammas), "betas": list(r.betas),
                "optimizer_success": r.optimizer_success, "optimizer_message": r.optimizer_message,
                "objective_trace_normalised_expectation": list(r.objective_trace),
            }

        # -- per-case verification ------------------------------------------------------
        rng = np.random.default_rng(0)
        err_random = max(abs(ham.energy(x) - qubo.energy(x)) for x in rng.integers(0, 2, size=(1000, 18)))
        err_valid = max(abs(ham.energy(row) - qubo.energy(row)) for row in space.assignments[::7])
        p3 = phase3_sol[case.name]
        p3_match = {
            "exact_plans_match_phase3": [p.label for p in exact.plans.values()]
            == [p3[f"plan_I{i}"] for i in range(1, 7)],
            "exact_energy_matches_phase3": abs(exact.energy - float(p3["qubo_energy"])) < 1e-6,
            **{f"{c}_waiting_matches_phase3": abs(sim_metrics[c]["total_waiting_time"]
                - float(phase3_runs[(case.name, {"qubo_exact": "qubo_static"}.get(c, c))]["total_waiting_time"])) < 1e-6
               for c in ("fixed", "adaptive", "qubo_exact")},
        }
        verification_cases.append({
            "case": case.name, "n_qubits": ham.n,
            "max_abs_energy_error_ising_vs_qubo_1000_random_strings": float(err_random),
            "max_abs_energy_error_ising_vs_qubo_valid_strings": float(err_valid),
            "statevector_probabilities_sum": {k: float(r.probabilities.sum()) for k, r in results.items()},
            "sampled_counts_sum_to_shots": all(sum(r.counts.values()) == r.shots for r in results.values()),
            "every_sample_scored_by_qubo_evaluator": all(
                abs(r.sampled_energy[nm] - qubo.energy([int(ch) for ch in nm])) < 1e-9
                for r in results.values() for nm in r.counts),
            **p3_match,
        })

    # -- global verification -------------------------------------------------------------
    backend = AerSimulator(method="statevector")
    known = [0] * 18
    for i, plan_idx in enumerate((2, 0, 1, 2, 0, 0)):
        known[3 * i + plan_idx] = 1
    qc = QuantumCircuit(18)
    for u, b in enumerate(known):
        if b:
            qc.x(u)
    qc.measure_all()
    (key,) = backend.run(transpile(qc, backend), shots=8, seed_simulator=1).result().get_counts()
    bit_order_ok = key_to_bits(key, 18) == tuple(known) and key[::-1] == variable_order_string(known)

    ns = next(c for c in CASES if c.name == "ns_heavy")  # repeat one run: determinism of the whole pipeline
    sim = Simulator(net, ns.scenario.build_demand(net, args.seed))
    st = initial_state_for(ns, sim)
    q_ns = build_qubo(TrafficState.from_observation(Observation(st.cycle, dict(st.queues), dict(st.in_transit), net)))
    first, second = run_qaoa(q_ns, base.with_p(1)), run_qaoa(q_ns, base.with_p(1))
    deterministic = (first.gammas == second.gammas and first.betas == second.betas
                     and first.counts == second.counts)

    configs = [{k: v for k, v in params_out[c][v]["config"].items() if k != "p"}
               for c in params_out for v in params_out[c]]
    same_config = all(cfg == configs[0] for cfg in configs)  # only p may differ

    verification = {
        "qiskit_version": qiskit.__version__, "qiskit_aer_version": qiskit_aer.__version__,
        "backend": "AerSimulator(method='statevector')", "n_qubits": 18, "seed": args.seed,
        "qaoa_config": asdict(base),
        "same_configuration_for_every_case_and_variant_except_p": same_config,
        "qiskit_bit_order_check_passed": bool(bit_order_ok),
        "repeat_run_identical_angles_and_counts": bool(deterministic),
        "exact_optimum_passed_to_optimiser": False,
        "exact_optimum_used_only_in_compare_with_exact": True,
        "cases": verification_cases,
        "all_checks_pass": bool(
            same_config and bit_order_ok and deterministic and all(
                c["max_abs_energy_error_ising_vs_qubo_1000_random_strings"] < 1e-6
                and c["max_abs_energy_error_ising_vs_qubo_valid_strings"] < 1e-6
                and c["sampled_counts_sum_to_shots"] and c["every_sample_scored_by_qubo_evaluator"]
                and all(abs(v - 1.0) < 1e-9 for v in c["statevector_probabilities_sum"].values())
                and c["exact_plans_match_phase3"] and c["exact_energy_matches_phase3"]
                and c["fixed_waiting_matches_phase3"] and c["adaptive_waiting_matches_phase3"]
                and c["qubo_exact_waiting_matches_phase3"] for c in verification_cases)),
    }
    write_csv(out / "qaoa_runs.csv", runs)
    write_csv(out / "qaoa_solutions.csv", solutions)
    (out / "qaoa_counts.json").write_text(json.dumps(counts_out))
    (out / "qaoa_parameters.json").write_text(json.dumps(params_out, indent=1))
    (out / "qaoa_verification.json").write_text(json.dumps(verification, indent=1))
    print(f"wrote results to {out}/ ; verification all_checks_pass = {verification['all_checks_pass']}")
    return 0 if verification["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
