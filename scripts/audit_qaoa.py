"""Phase 4.5 audit of the Phase 4 QAOA experiment. Nothing is re-optimised or re-tuned.

Every number is recomputed from the saved artifacts in results/phase4 (counts, optimised angles) and
from the Phase 3 QUBOs rebuilt from their start states:
  * Ising == QUBO energies (random, all-valid and the full 2^18 spectrum), Qiskit bit ordering,
    p=1 / p=2 18-qubit circuit structure
  * the exact QUBO energy and one-hot feasibility of every sampled bitstring, recomputed from counts
  * that no exact-optimum information reaches the optimiser (signature, source, and a run with the
    exact solver disabled)
  * feasibility / optimum probabilities of QAOA vs uniform sampling: from the saved counts and from
    a statevector re-simulation at the saved optimised angles
  * tie analysis: every exact-optimal configuration, whether QAOA sampled it, and its simulator result
  * primary p=2 (deterministic ramp start) reported separately from the p=2 warm-start ablation

Run:  python scripts/audit_qaoa.py [--out results/phase4/audit.json]
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

import qtraffic.optimization.exact as exact_mod
import qtraffic.optimization.qaoa as qaoa_mod
from qtraffic import Observation, Simulator, grid_network
from qtraffic.optimization import (
    PLANS, QAOAConfig, QAOAEngine, StaticPlanController, TrafficState, build_qaoa_circuit, build_qubo,
    enumerate_feasible, key_to_bits, qubo_to_ising, run_qaoa, solve_exact,
)
from qtraffic.optimization.exact import TIE_TOL
from qtraffic.optimization.qaoa import bits_to_index, bits_to_key, variable_order_string
from qtraffic.optimization.validation import CASES, _metrics, initial_state_for, run_from_state

TOL = 1e-6
ROLE = {"qaoa_p1": "primary (p=1)", "qaoa_p2": "primary (p=2, same deterministic ramp start as p=1)",
        "qaoa_p2_warm": "ABLATION ONLY (p=2 started from the p=1 optimum, second layer zero)"}


def bits_of(name: str) -> tuple[int, ...]:
    return tuple(int(c) for c in name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase4/audit.json")
    ap.add_argument("--dir", default="results/phase4")
    args = ap.parse_args()
    d = Path(args.dir)
    counts = json.load(open(d / "qaoa_counts.json"))
    params = json.load(open(d / "qaoa_parameters.json"))
    rows = {(r["case"], r["variant"]): r for r in csv.DictReader(open(d / "qaoa_solutions.csv"))}
    runs = {(r["case"], r["controller"]): r for r in csv.DictReader(open(d / "qaoa_runs.csv"))}
    net = grid_network(2, 3)
    backend = AerSimulator(method="statevector")
    report: dict = {"variant_roles": ROLE, "cases": {}, "failures": []}

    def check(name: str, ok: bool) -> bool:
        if not ok:
            report["failures"].append(name)
        return bool(ok)

    # ---- global checks: bit order, circuits, optimiser isolation -----------------------------
    known = [0] * 18
    for i, k in enumerate((2, 0, 1, 2, 0, 0)):
        known[3 * i + k] = 1
    qc = QuantumCircuit(18)
    for u, b in enumerate(known):
        if b:
            qc.x(u)
    qc.measure_all()
    (key,) = backend.run(transpile(qc, backend), shots=16, seed_simulator=1).result().get_counts()
    report["bit_order"] = {
        "prepared_variable_order": variable_order_string(known), "qiskit_key": key,
        "key_is_reverse_of_variable_order": key == variable_order_string(known)[::-1],
        "decoded_back_correctly": check("bit_order", key_to_bits(key, 18) == tuple(known)
                                        and bits_to_key(known) == key),
        "statevector_index_of_state": bits_to_index(known),
    }

    signature = list(inspect.signature(run_qaoa).parameters)
    isolation_src = inspect.getsource(run_qaoa) + inspect.getsource(QAOAEngine)
    forbidden = [w for w in ("solve_exact", "ExactSolution", "enumerate_feasible", "compare_with_exact")
                 if w in isolation_src]

    # ---- per case -------------------------------------------------------------------------------
    for case in CASES:
        sim = Simulator(net, case.scenario.build_demand(net, 0))
        start = initial_state_for(case, sim)
        qubo = build_qubo(TrafficState.from_observation(
            Observation(start.cycle, dict(start.queues), dict(start.in_transit), net)))
        exact = solve_exact(qubo)
        ham = qubo_to_ising(qubo)
        space = enumerate_feasible(qubo)
        n_all = 2**18
        entry: dict = {}

        # Ising == QUBO
        rng = np.random.default_rng(0)
        rand_err = max(abs(ham.energy(x) - qubo.energy(x)) for x in rng.integers(0, 2, size=(1000, 18)))
        diag = ham.diagonal()
        idx = np.arange(n_all)
        bits = ((idx[:, None] >> np.arange(18)[None, :]) & 1).astype(float)
        qubo_all = qubo.offset + ((bits @ qubo.matrix()) * bits).sum(axis=1)
        entry["energy_equivalence"] = {
            "max_abs_error_1000_random_strings": float(rand_err),
            "max_abs_error_all_2^18_strings": float(np.abs(diag - qubo_all).max()),
            "min_of_ising_diag_equals_min_of_qubo": bool(abs(diag.min() - qubo_all.min()) < TOL),
        }
        check(f"{case.name}:ising_equiv", rand_err < TOL and entry["energy_equivalence"]["max_abs_error_all_2^18_strings"] < TOL)

        # optimal configurations (ties)
        e_opt = exact.energy
        tol = TIE_TOL * max(1.0, abs(e_opt))
        opt_rows = np.flatnonzero(space.energies <= e_opt + tol)
        opt_configs = {tuple(int(v) for v in space.plan_indices[r]): r for r in opt_rows}
        n_opt = len(opt_configs)
        entry["exact"] = {
            "energy": e_opt, "n_optimal_configurations": n_opt,
            "exact_solver_choice": "/".join(p.label for p in exact.plans.values()),
            "optimal_configurations": ["/".join(PLANS[k].label for k in cfg) for cfg in opt_configs],
        }
        check(f"{case.name}:n_opt_matches_solver", n_opt == exact.n_optimal)

        # simulator result of EVERY exact-optimal configuration (tie analysis)
        tie_sim = {}
        for cfg in opt_configs:
            plans = {n: PLANS[k] for n, k in zip(net.nodes, cfg)}
            m = _metrics(run_from_state(sim, start, StaticPlanController(plans), 60), net.nodes)
            tie_sim["/".join(PLANS[k].label for k in cfg)] = {
                "qubo_energy": float(space.energies[opt_configs[cfg]]),
                "waiting_per_admitted": m["waiting_per_admitted"], "throughput_per_hour": m["throughput_per_hour"],
                "vehicles_rejected": m["vehicles_rejected"], "total_blocked": m["total_blocked"],
                "final_queue_total": m["final_queue_total"]}
        w = [v["waiting_per_admitted"] for v in tie_sim.values()]
        entry["tie_simulator_results"] = tie_sim
        entry["equal_energy_but_different_simulator_performance"] = bool(n_opt > 1 and max(w) - min(w) > 1e-6)
        entry["simulator_waiting_spread_among_ties_pct"] = float((max(w) - min(w)) / min(w) * 100) if n_opt > 1 else 0.0

        entry["variants"] = {}
        for variant in ("qaoa_p1", "qaoa_p2", "qaoa_p2_warm"):
            c = counts[case.name][variant]
            cs = c["counts"]
            shots = c["shots"]
            v: dict = {"role": ROLE[variant]}

            # every sampled bitstring: exact QUBO energy + one-hot check, recomputed here
            en = {nm: qubo.energy(bits_of(nm)) for nm in cs}
            feas = {nm: qubo.is_feasible(bits_of(nm)) for nm in cs}
            best_feas = min((e for nm, e in en.items() if feas[nm]), default=None)
            n_feas_shots = sum(n for nm, n in cs.items() if feas[nm])
            opt_shots = sum(n for nm, n in cs.items()
                            if feas[nm] and en[nm] <= e_opt + tol)
            sampled_optimal_configs = sorted({
                "/".join(PLANS[int(np.argmax(bits_of(nm)[3 * i:3 * i + 3]))].label for i in range(6))
                for nm in cs if feas[nm] and en[nm] <= e_opt + tol})
            csv_row = rows[(case.name, variant)]
            v["sampled"] = {
                "shots": shots, "shots_sum_check": sum(cs.values()) == shots,
                "unique_bitstrings": len(cs), "unique_feasible": sum(feas.values()),
                "feasible_shots": n_feas_shots, "feasible_rate": n_feas_shots / shots,
                "best_feasible_energy": best_feas, "best_any_energy": min(en.values()),
                "shots_on_an_exact_optimal_configuration": opt_shots,
                "optimum_probability_sampled": opt_shots / shots,
                "optimal_configurations_sampled": sampled_optimal_configs,
                "sampled_an_exact_optimal_configuration": opt_shots > 0,
            }
            check(f"{case.name}/{variant}:counts_sum", v["sampled"]["shots_sum_check"])
            check(f"{case.name}/{variant}:csv_feasible_rate", abs(v["sampled"]["feasible_rate"] - float(csv_row["feasible_rate_sampled"])) < 1e-12)
            check(f"{case.name}/{variant}:csv_best_energy", best_feas is not None and abs(best_feas - float(csv_row["qaoa_best_feasible_energy"])) < TOL)
            check(f"{case.name}/{variant}:csv_unique_feasible", v["sampled"]["unique_feasible"] == int(csv_row["unique_feasible_sampled"]))
            check(f"{case.name}/{variant}:csv_optimum_prob", abs(v["sampled"]["optimum_probability_sampled"] - float(csv_row["optimum_probability_sampled"])) < 1e-12)

            # statevector re-simulation at the SAVED optimised angles
            saved = params[case.name][variant]
            p = len(saved["gammas"])
            engine = QAOAEngine(qubo, p, backend)
            angles = np.array(saved["gammas"] + saved["betas"])
            probs = engine.probabilities(angles)
            feas_mask = (space_bits := ((idx[:, None] >> np.arange(18)[None, :]) & 1)).reshape(n_all, 6, 3).sum(axis=2)
            feas_mask = (feas_mask == 1).all(axis=1)
            opt_idx = [bits_to_index(space.assignments[r]) for r in opt_rows]
            p_feas, p_opt = float(probs[feas_mask].sum()), float(probs[opt_idx].sum())
            expectation = engine.to_energy(engine.expectation_normalized(angles))
            v["statevector"] = {
                "feasible_probability": p_feas, "optimum_probability": p_opt,
                "expected_energy": expectation,
                "optimum_probability_given_feasible": p_opt / p_feas,
                "enrichment_of_optimum_among_feasible": (p_opt / p_feas) / (n_opt / 729),
                "feasible_probability_vs_uniform": p_feas / (729 / n_all),
                "optimum_probability_vs_uniform": p_opt / (n_opt / n_all),
            }
            check(f"{case.name}/{variant}:statevector_feasible", abs(p_feas - float(csv_row["feasible_probability_statevector"])) < 1e-9)
            check(f"{case.name}/{variant}:statevector_opt", abs(p_opt - float(csv_row["optimum_probability_statevector"])) < 1e-9)
            check(f"{case.name}/{variant}:expectation", abs(expectation - float(csv_row["expected_energy_optimised"])) < 1e-4)

            # sampled feasible energy vs the average feasible assignment
            fe = [(en[nm], n) for nm, n in cs.items() if feas[nm]]
            v["mean_energy_of_feasible_samples"] = sum(e * n for e, n in fe) / sum(n for _, n in fe) if fe else None
            v["mean_energy_of_all_729_feasible"] = float(space.energies.mean())

            # simulator result of the reported best feasible plan
            r = runs.get((case.name, variant))
            v["simulator"] = {k: float(r[k]) for k in ("waiting_per_admitted", "throughput_per_hour", "vehicles_rejected",
                                                       "total_blocked", "final_queue_total", "average_queue_length", "max_queue_length")
                              } if r and not r.get("note") else None
            v["optimizer"] = {"initial_angles": saved["initial_angles"], "final_gammas": saved["gammas"],
                              "final_betas": saved["betas"], "function_evaluations": int(csv_row["function_evaluations"]),
                              "converged_before_maxiter": csv_row["optimizer_converged"] == "True",
                              "message": saved["optimizer_message"], "config": saved["config"]}
            entry["variants"][variant] = v

        # tie consequence: QAOA's best plan vs the exact solver's choice
        entry["qaoa_plans_vs_exact_choice"] = {
            variant: {"qaoa_plans": rows[(case.name, variant)]["qaoa_plans"],
                      "same_plans_as_exact_solver_choice": rows[(case.name, variant)]["qaoa_plans"] == entry["exact"]["exact_solver_choice"],
                      "same_qubo_energy_as_optimum": abs(float(rows[(case.name, variant)]["qaoa_best_feasible_energy"]) - e_opt) < TOL}
            for variant in ("qaoa_p1", "qaoa_p2", "qaoa_p2_warm")}
        report["cases"][case.name] = entry
        print(f"  audited {case.name}", flush=True)

    # ---- circuits at p=1 and p=2 on the real 18-qubit Hamiltonian -----------------------------------
    ham0 = qubo_to_ising(build_qubo(TrafficState.from_observation(Observation(
        0, {a: 5.0 for a in net.approaches}, {a: 1.0 for a in net.approaches}, net))))
    nz_j, nz_h = sum(1 for c in ham0.J.values() if c != 0), int(np.count_nonzero(ham0.h))
    circuits = {}
    for p in (1, 2):
        ops = dict(build_qaoa_circuit(ham0, p).circuit.count_ops())
        circuits[f"p={p}"] = {"ops": ops, "qubits": 18,
                              "ok": ops["h"] == 18 and ops["rx"] == 18 * p and ops["rzz"] == p * nz_j and ops["rz"] == p * nz_h}
        check(f"circuit_p{p}", circuits[f"p={p}"]["ok"])
    report["circuits"] = circuits

    # ---- no exact-optimum information reaches the optimiser -----------------------------------------
    real_solve, real_enum = exact_mod.solve_exact, exact_mod.enumerate_feasible

    def boom(*_a, **_k):
        raise RuntimeError("exact solver / feasible enumeration touched during QAOA optimisation")

    qaoa_case = CASES[1]
    sim = Simulator(net, qaoa_case.scenario.build_demand(net, 0))
    st = initial_state_for(qaoa_case, sim)
    qubo_ns = build_qubo(TrafficState.from_observation(Observation(st.cycle, dict(st.queues), dict(st.in_transit), net)))
    exact_mod.solve_exact = exact_mod.enumerate_feasible = boom
    qaoa_mod.enumerate_feasible = boom
    try:
        with_exact_disabled = run_qaoa(qubo_ns, QAOAConfig(p=1))
        ran_ok = True
    except RuntimeError:
        with_exact_disabled, ran_ok = None, False
    finally:
        exact_mod.solve_exact, exact_mod.enumerate_feasible = real_solve, real_enum
        qaoa_mod.enumerate_feasible = real_enum
    saved_ns = params["ns_heavy"]["qaoa_p1"]
    report["optimiser_isolation"] = {
        "run_qaoa_parameters": signature,
        "exact_related_names_in_run_qaoa_or_engine_source": forbidden,
        "qaoa_runs_with_exact_solver_and_enumeration_disabled": ran_ok,
        "angles_identical_to_saved_experiment": bool(ran_ok and list(with_exact_disabled.gammas) == saved_ns["gammas"]
                                                      and list(with_exact_disabled.betas) == saved_ns["betas"]),
    }
    check("optimiser_isolation", signature == ["qubo", "config", "initial_angles"] and not forbidden
          and ran_ok and report["optimiser_isolation"]["angles_identical_to_saved_experiment"])

    report["all_checks_pass"] = not report["failures"]
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"audit written to {args.out}; failures: {report['failures'] or 'none'}")
    return 0 if report["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
