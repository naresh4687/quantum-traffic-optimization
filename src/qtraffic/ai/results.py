"""Build analysis contexts from the project's SAVED results (Phases 3-6).

Nothing is simulated or recomputed here beyond simple means / percentages of saved numbers. File paths
are used only to read; they never enter a context. Missing files or rows raise ``FileNotFoundError`` /
``LookupError`` instead of producing invented data.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from .context import TrafficAnalysisContext

PROXY_LABEL = "SIMULATION PROXY (waiting/idling only) - estimate, not a measurement"
QAOA_BACKEND = "Qiskit Aer statevector simulator (classical; not quantum hardware)"
VARIANT_NOTES = {
    "qaoa_p1": "primary result, p=1",
    "qaoa_p2": "primary result, p=2 (same deterministic ramp start as p=1)",
    "qaoa_p2_warm": "ABLATION ONLY: p=2 started from the p=1 optimum",
}


def _r(x: float | None, digits: int = 4):
    return None if x is None else round(float(x), digits)


def _rows(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _pct(baseline: float, value: float) -> float | None:
    return None if baseline == 0 else _r((value - baseline) / abs(baseline) * 100.0, 3)


def _mean(rows: list[dict], key: str) -> float:
    return statistics.fmean(float(r[key]) for r in rows)


def _phase3_plans(results_dir: Path, scenario: str, seed: int) -> dict | None:
    for r in _rows(results_dir / "phase3" / "qubo_solutions.csv"):
        if r["case"] == scenario and int(r["seed"]) == seed:
            return {f"I{i}": r[f"plan_I{i}"] for i in range(1, 7)}
    return None


def _phase3_energy(results_dir: Path, scenario: str, seed: int) -> float | None:
    for r in _rows(results_dir / "phase3" / "qubo_solutions.csv"):
        if r["case"] == scenario and int(r["seed"]) == seed:
            return float(r["qubo_energy"])
    return None


def _phase4_plans(results_dir: Path, scenario: str, seed: int, controller: str = "qaoa_p1") -> dict | None:
    for r in _rows(results_dir / "phase4" / "qaoa_runs.csv"):
        if r["case"] == scenario and int(r["seed"]) == seed and r["controller"] == controller and r.get("plan_I1"):
            return {f"I{i}": r[f"plan_I{i}"] for i in range(1, 7)}
    return None


def load_controller_context(
    results_dir: str | Path, scenario: str, controller: str, baseline: str = "fixed", seed: int | None = None,
) -> TrafficAnalysisContext:
    """Phase 6 controller comparison for one scenario (mean over seeds, or a single ``seed``)."""
    d = Path(results_dir)
    rows = _rows(d / "phase6" / "controller_comparison.csv")

    def pick(name: str) -> list[dict]:
        sel = [r for r in rows if r["scenario"] == scenario and r["controller"] == name
               and (seed is None or int(r["seed"]) == seed)]
        if not sel:
            raise LookupError(f"no Phase 6 rows for scenario={scenario!r} controller={name!r} seed={seed!r}")
        return sel

    mine = pick(controller)
    seeds = tuple(sorted(int(r["seed"]) for r in mine))
    base = [r for r in pick(baseline) if int(r["seed"]) in seeds]  # identical seeds, identical demand
    m = lambda k: _mean(mine, k)
    b = lambda k: _mean(base, k)

    traffic = {k: _r(m(k)) for k in ("vehicles_admitted", "vehicles_rejected", "vehicles_exited",
                                     "throughput_vehicles_per_hour", "waiting_vehicle_seconds",
                                     "waiting_seconds_per_admitted", "average_queue_vehicles",
                                     "max_queue_vehicles", "final_queue_vehicles", "blocked_vehicle_cycles")}
    signal = {"plan_changes": _r(m("plan_changes")), "plan_change_rate": _r(m("plan_change_rate")),
              "ns_green_share": _r(m("ns_green_share"))}
    if len(seeds) == 1:
        plans = (_phase3_plans(d, scenario, seeds[0]) if controller == "qubo_static"
                 else _phase4_plans(d, scenario, seeds[0]) if controller == "qaoa_p1_saved" else None)
        if plans:
            signal["signal_plans"] = plans
    optimization = {}
    if len(seeds) == 1 and controller == "qubo_static":
        energy = _phase3_energy(d, scenario, seeds[0])
        if energy is not None:  # the exact solver's optimum is the energy of the chosen assignment
            optimization = {"qubo_energy": _r(energy, 3), "exact_energy": _r(energy, 3), "method": "exact enumeration of the QUBO"}
    comparison, environmental = {}, {
        "label": PROXY_LABEL, "fuel_liters_proxy": _r(m("fuel_liters_proxy")), "co2_kg_proxy": _r(m("co2_kg_proxy")),
        "fuel_liters_proxy_per_admitted": _r(m("fuel_liters_proxy_per_admitted"), 6),
        "co2_kg_proxy_per_admitted": _r(m("co2_kg_proxy_per_admitted"), 6),
        "idle_fuel_rate_lph_assumed": _r(m("idle_fuel_rate_lph_assumed"), 3),
        "emission_factor_kg_per_liter_assumed": _r(m("emission_factor_kg_per_liter_assumed"), 4)}
    if controller != baseline:
        per_seed = {int(r["seed"]): float(r["waiting_seconds_per_admitted"]) for r in mine}
        base_seed = {int(r["seed"]): float(r["waiting_seconds_per_admitted"]) for r in base}
        diffs = [per_seed[s] - base_seed[s] for s in seeds]
        comparison = {
            "waiting_seconds_per_admitted_pct": _pct(b("waiting_seconds_per_admitted"), m("waiting_seconds_per_admitted")),
            "waiting_vehicle_seconds_pct": _pct(b("waiting_vehicle_seconds"), m("waiting_vehicle_seconds")),
            "throughput_pct": _pct(b("throughput_vehicles_per_hour"), m("throughput_vehicles_per_hour")),
            "average_queue_pct": _pct(b("average_queue_vehicles"), m("average_queue_vehicles")),
            "max_queue_pct": _pct(b("max_queue_vehicles"), m("max_queue_vehicles")),
            "vehicles_rejected_pct": _pct(b("vehicles_rejected"), m("vehicles_rejected")),
            "final_queue_pct": _pct(b("final_queue_vehicles"), m("final_queue_vehicles")),
            "seeds_compared": len(seeds), "seeds_better": sum(x < -1e-9 for x in diffs),
            "seeds_worse": sum(x > 1e-9 for x in diffs)}
        environmental["co2_proxy_pct_vs_baseline"] = _pct(b("co2_kg_proxy"), m("co2_kg_proxy"))
        environmental["waiting_pct_vs_baseline"] = comparison["waiting_vehicle_seconds_pct"]
    return TrafficAnalysisContext(
        scenario=scenario, controller=controller, baseline_controller=baseline if controller != baseline else None,
        seeds=seeds, horizon_cycles=int(float(mine[0]["cycles"])), traffic=traffic, signal=signal,
        baseline_comparison=comparison, environmental=environmental, optimization=optimization)


def load_qaoa_context(results_dir: str | Path, scenario: str, variant: str = "qaoa_p1", seed: int = 0) -> TrafficAnalysisContext:
    """Phase 4 QAOA-vs-exact result for one case, with its simulator validation vs Fixed."""
    d = Path(results_dir)
    row = next((r for r in _rows(d / "phase4" / "qaoa_solutions.csv")
                if r["case"] == scenario and r["variant"] == variant and int(r["seed"]) == seed), None)
    if row is None:
        raise LookupError(f"no Phase 4 row for scenario={scenario!r} variant={variant!r} seed={seed}")
    f = lambda k: None if row[k] in ("", "None") else float(row[k])
    n_opt = int(row["exact_n_optimal_assignments"])
    p_opt, p_feas = f("optimum_probability_statevector"), f("feasible_probability_statevector")
    qaoa = {"backend": QAOA_BACKEND, "p": int(row["p"]), "n_qubits": int(row["n_qubits"]), "shots": int(row["shots"]),
            "logical_depth": int(row["logical_depth"]), "logical_gates": int(row["logical_gates"]),
            "basis_cx_gates": int(row["basis_cx"]), "best_sampled_feasible_energy": _r(f("qaoa_best_feasible_energy"), 3),
            "exact_energy": _r(f("exact_energy"), 3), "energy_gap": _r(f("energy_gap"), 3),
            "approximation_ratio": _r(f("approximation_ratio"), 6), "found_exact_optimum": row["found_exact_optimum"] == "True",
            "exact_n_optimal_assignments": n_opt, "feasible_rate_sampled": _r(f("feasible_rate_sampled"), 5),
            "feasible_probability_statevector": _r(p_feas, 5), "uniform_feasible_probability": _r(f("uniform_feasible_probability"), 6),
            "optimum_probability_sampled": _r(f("optimum_probability_sampled"), 6),
            "optimum_probability_statevector": _r(p_opt, 6), "uniform_optimum_probability": _r(f("uniform_optimum_probability"), 8),
            "optimum_enrichment_among_feasible": _r((p_opt / p_feas) / (n_opt / 729), 3) if p_opt and p_feas else None,
            "optimizer_converged": row["optimizer_converged"] == "True", "function_evaluations": int(row["function_evaluations"]),
            "variant_note": VARIANT_NOTES.get(variant, variant)}
    sim_rows = {r["controller"]: r for r in _rows(d / "phase4" / "qaoa_runs.csv")
                if r["case"] == scenario and int(r["seed"]) == seed and r.get("total_waiting_time")}
    traffic, comparison = {}, {}
    if variant in sim_rows and "fixed" in sim_rows:
        q, fx = sim_rows[variant], sim_rows["fixed"]
        traffic = {"waiting_seconds_per_admitted": _r(q["waiting_per_admitted"]), "throughput_vehicles_per_hour": _r(q["throughput_per_hour"]),
                   "max_queue_vehicles": _r(q["max_queue_length"]), "final_queue_vehicles": _r(q["final_queue_total"]),
                   "vehicles_rejected": _r(q["vehicles_rejected"]), "blocked_vehicle_cycles": _r(q["total_blocked"])}
        comparison = {"waiting_seconds_per_admitted_pct": _pct(float(fx["waiting_per_admitted"]), float(q["waiting_per_admitted"])),
                      "throughput_pct": _pct(float(fx["throughput_per_hour"]), float(q["throughput_per_hour"])), "seeds_compared": 1}
    plans = _phase4_plans(d, scenario, seed, variant)
    return TrafficAnalysisContext(
        scenario=scenario, controller=variant, baseline_controller="fixed" if comparison else None, seeds=(seed,),
        horizon_cycles=60, traffic=traffic, baseline_comparison=comparison, qaoa=qaoa,
        optimization={"qubo_energy": _r(f("qaoa_best_feasible_energy"), 3), "exact_energy": _r(f("exact_energy"), 3),
                      "method": "QAOA best sampled feasible solution vs exact enumeration (ground truth)"},
        signal={"signal_plans": plans} if plans else {})


def load_emergency_context(results_dir: str | Path, scenario: str, base: str = "fixed") -> TrafficAnalysisContext:
    """Phase 6 emergency-corridor A/B (mean over seeds) plus Phase 5 restoration evidence when available."""
    d = Path(results_dir)
    summary = json.loads((d / "phase6" / "summary.json").read_text())
    row = next((e for e in summary["emergency_summary"] if e["scenario"] == scenario and e["base_controller"] == base), None)
    if row is None:
        raise LookupError(f"no emergency summary for scenario={scenario!r} base={base!r}")
    repro = json.loads((d / "phase5" / "reproducibility.json").read_text())["configuration"]
    runs = [r for r in _rows(d / "phase5" / "emergency_runs.csv")
            if r["scenario"] == scenario and r["base_controller"] == base and r["mode"] == "B_corridor"]
    emergency = {
        "route": list(repro["route"]), "seeds": row["seeds"], "free_flow_time_s": _r(3 * repro["link_seconds"], 1),
        "travel_time_s_no_override": _r(row["emergency_travel_time_s_A"], 1), "travel_time_s_corridor": _r(row["emergency_travel_time_s_B"], 1),
        "delay_s_no_override": _r(row["emergency_delay_s_A"], 1), "delay_s_corridor": _r(row["emergency_delay_s_B"], 1),
        "stops_no_override": _r(row["signal_stops_A"], 2), "stops_corridor": _r(row["signal_stops_B"], 2),
        "prioritized_intersections": _r(_mean(runs, "priority_intersections"), 2) if runs else None,
        "normal_waiting_pct_corridor_vs_no_override": _r(row["waiting_pct_B_vs_A_of_means"], 3),
        "seeds_normal_waiting_worse": row["seeds_normal_waiting_worse_with_corridor"],
        "seeds_normal_waiting_better": row["seeds_normal_waiting_better_with_corridor"],
        "max_queue_no_override": _r(row["max_queue_A"], 2), "max_queue_corridor": _r(row["max_queue_B"], 2),
        "final_queue_no_override": _r(row["final_queue_A"], 2), "final_queue_corridor": _r(row["final_queue_B"], 2),
        "co2_proxy_pct_corridor_vs_no_override": _r(row["co2_proxy_pct_B_vs_A_of_means"], 3)}
    if (scenario, base) == ("balanced_medium", "fixed"):  # the Phase 5 primary case has the per-cycle signal log
        timeline = json.loads((d / "phase5" / "primary_emergency_timeline.json").read_text())["B_corridor"]
        done = timeline["completion_cycle"]
        after = [r for r in _rows(d / "phase5" / "primary_signal_log.csv") if r["mode"] == "B_corridor" and int(r["cycle"]) > done]
        if after:
            emergency["cycles_after_completion_checked"] = len({int(r["cycle"]) for r in after})
            emergency["cycles_after_completion_with_override_active"] = len({int(r["cycle"]) for r in after if r["priority"] == "True"})
            emergency["plans_restored_to_base_controller"] = all(r["final_plan"] == r["base_plan"] for r in after)
    return TrafficAnalysisContext(scenario=scenario, controller=f"{base}+emergency_corridor", baseline_controller=f"{base} (no override)",
                                  seeds=tuple(range(int(row["seeds"]))), horizon_cycles=60, emergency=emergency)
