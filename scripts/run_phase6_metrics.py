"""Phase 6: unified metrics and the waiting-based fuel/CO2 PROXY, on the existing simulator.

Run:  python scripts/run_phase6_metrics.py [--out results/phase6]

Every traffic number is a simulator output. Every fuel / CO2 number is a SIMULATION PROXY computed from
the simulator's waiting vehicle-seconds and two configurable coefficients (see qtraffic.environment);
it is not a measurement and not a calibrated fuel model.

1. Controller comparison. Five Phase 3/4 scenarios, seeds 0-4, identical network / start state / demand
   stream / 60-cycle horizon for every controller: Fixed, Adaptive, QUBO static (exact), QUBO receding
   (exact, re-solved each cycle) and, for seed 0 only, the QAOA p=1 plan SAVED by Phase 4 (not rerun).
2. Emergency corridor. The Phase 5 A/B experiment (same route, start cycle, horizon; Fixed and Adaptive
   bases; balanced_medium / ns_heavy / ew_heavy; seeds 0-4) re-expressed with unified metrics.
3. Sensitivity of the proxy to its coefficients. It is linear in waiting time, so relative comparisons
   cannot change; the analysis shows that explicitly instead of assuming it.
4. Consistency checks against the recorded Phase 3 / 4 / 5 results.

Writes results/phase6/: controller_comparison.csv, controller_summary.csv, environmental_metrics.csv,
emergency_metrics.csv, emergency_comparison.csv, sensitivity.csv, configuration.json, summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import qiskit
import qiskit_aer

from qtraffic import (
    AdaptiveController, FixedTimeController, Observation, Simulator, SignalPlan, grid_network,
)
from qtraffic.emergency import EmergencyEvent, run_emergency
from qtraffic.environment import (
    ARGONNE_IDLE_EXAMPLES_GAL_PER_HOUR, DEFAULT_IDLE_FUEL_RATE_LPH, DOE_GENERAL_IDLE_RANGE_GAL_PER_HOUR,
    gallons_per_hour_to_liters_per_hour, DIESEL_KG_CO2_PER_LITER, EPA_DIESEL_G_CO2_PER_US_GALLON,
    EPA_GASOLINE_G_CO2_PER_US_GALLON, GASOLINE_KG_CO2_PER_LITER, LITERS_PER_US_GALLON, PROXY_LABEL,
    SENSITIVITY_IDLE_FUEL_RATES_LPH, EnvironmentalConfig, percent_change, sensitivity_configs,
)
from qtraffic.experiments import demand_digest
from qtraffic.optimization import QuboController, StaticPlanController, TrafficState, build_qubo, solve_exact
from qtraffic.optimization.validation import CASES, initial_state_for, run_from_state
from qtraffic.unified_metrics import UnifiedMetrics, unified_metrics

SEEDS = (0, 1, 2, 3, 4)
CYCLES = 60
CONTROLLERS = ("fixed", "adaptive", "qubo_static", "qubo_receding", "qaoa_p1_saved")
EMERGENCY_ROUTE = ("I1", "I2", "I3", "I6")
EMERGENCY_START_CYCLE = 40
EMERGENCY_BASES = ("fixed", "adaptive")
EMERGENCY_SCENARIOS = ("balanced_medium", "ns_heavy", "ew_heavy")
COMPARE_KEYS = ("waiting_seconds_per_admitted", "waiting_vehicle_seconds", "average_queue_vehicles",
                "max_queue_vehicles", "throughput_vehicles_per_hour", "vehicles_rejected",
                "blocked_vehicle_cycles", "final_queue_vehicles", "fuel_liters_proxy", "co2_kg_proxy",
                "fuel_liters_proxy_per_admitted", "co2_kg_proxy_per_admitted")
LOWER_BETTER = {k for k in COMPARE_KEYS if k != "throughput_vehicles_per_hour"}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def parse_plan(label: str) -> SignalPlan:
    """'NS20/EW40' -> SignalPlan."""
    return SignalPlan.from_ns(int(label.split("/")[0].removeprefix("NS")))


def saved_qaoa_plans(phase4_dir: Path) -> dict[tuple[str, int], dict[str, SignalPlan]]:
    """Phase 4's saved qaoa_p1 best-sampled feasible plans, keyed by (case, seed)."""
    out = {}
    for r in csv.DictReader(open(phase4_dir / "qaoa_runs.csv")):
        if r["controller"] == "qaoa_p1" and r.get("plan_I1"):
            out[(r["case"], int(r["seed"]))] = {f"I{i}": parse_plan(r[f"plan_I{i}"]) for i in range(1, 7)}
    return out


def make_controller(name, sim, start, case, seed, qubo, saved):
    if name == "fixed":
        return FixedTimeController()
    if name == "adaptive":
        return AdaptiveController()
    if name == "qubo_static":
        return StaticPlanController(solve_exact(qubo).plans, "QuboExactStatic")
    if name == "qubo_receding":
        return QuboController()
    if name == "qaoa_p1_saved":
        plans = saved.get((case.name, seed))
        return None if plans is None else StaticPlanController(plans, "QaoaP1Saved")
    raise ValueError(name)


def conservation_error(start, result) -> float:
    """|queued0 + transit0 + admitted - (exited + final queue + final transit)|: exact identity, so ~0."""
    m = result.metrics
    lhs = sum(start.queues.values()) + sum(start.in_transit.values()) + m.vehicles_entered
    rhs = m.vehicles_exited + sum(result.final_queues.values()) + sum(result.final_in_transit.values())
    return abs(lhs - rhs)


def pct_columns(row: dict, base: dict) -> dict:
    return {f"{k}_pct_vs_fixed": percent_change(base[k], row[k]) for k in COMPARE_KEYS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase6")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    out, res_dir = Path(args.out), Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    env = EnvironmentalConfig()  # the documented defaults
    net = grid_network(2, 3)
    saved = saved_qaoa_plans(res_dir / "phase4")
    p3 = {(r["case"], int(r["seed"]), r["controller"]): float(r["total_waiting_time"])
          for r in csv.DictReader(open(res_dir / "phase3" / "runs.csv"))}
    p4 = {(r["case"], int(r["seed"]), r["controller"]): float(r["total_waiting_time"])
          for r in csv.DictReader(open(res_dir / "phase4" / "qaoa_runs.csv")) if r.get("total_waiting_time")}
    p5 = {(r["scenario"], r["base_controller"], int(r["seed"]), r["mode"]): float(r["emergency_travel_time_s"])
          for r in csv.DictReader(open(res_dir / "phase5" / "emergency_runs.csv"))}

    # ------------------------------------------------------------------ 1. controller comparison
    comp, unified_by_key, checks = [], {}, {"conservation_max_abs_error": 0.0, "phase3_phase4_mismatches": [],
                                            "phase5_mismatches": [], "demand_mismatches": []}
    for case in CASES:
        for seed in SEEDS:
            sim = Simulator(net, case.scenario.build_demand(net, seed))
            start = initial_state_for(case, sim)
            qubo = build_qubo(TrafficState.from_observation(
                Observation(start.cycle, dict(start.queues), dict(start.in_transit), net)))
            src = demand_digest([sim.demand.arrivals(start.cycle + c) for c in range(CYCLES)])
            rows = {}
            for name in CONTROLLERS:
                ctrl = make_controller(name, sim, start, case, seed, qubo, saved)
                if ctrl is None:
                    continue  # QAOA plan only exists for the seed Phase 4 ran
                res = run_from_state(sim, start, ctrl, CYCLES)
                if demand_digest([r.demand for r in res.history]) != src:
                    checks["demand_mismatches"].append((case.name, seed, name))
                checks["conservation_max_abs_error"] = max(checks["conservation_max_abs_error"], conservation_error(start, res))
                um = unified_metrics(res.history, env)
                unified_by_key[(case.name, seed, name)] = um
                rows[name] = {"scenario": case.name, "seed": seed, "controller": name, "cycles": CYCLES,
                              **um.flat()}
                ref = {"fixed": p3.get((case.name, seed, "fixed")), "adaptive": p3.get((case.name, seed, "adaptive")),
                       "qubo_static": p3.get((case.name, seed, "qubo_static")),
                       "qubo_receding": p3.get((case.name, seed, "qubo_receding")),
                       "qaoa_p1_saved": p4.get((case.name, seed, "qaoa_p1"))}[name]
                if ref is not None and abs(ref - um.traffic.waiting_vehicle_seconds) > 1e-6:
                    checks["phase3_phase4_mismatches"].append((case.name, seed, name, ref, um.traffic.waiting_vehicle_seconds))
            for name, row in rows.items():
                row.update(pct_columns(row, rows["fixed"]))
                comp.append(row)

    write_csv(out / "controller_comparison.csv", comp)
    env_rows = [{"scenario": r["scenario"], "seed": r["seed"], "controller": r["controller"],
                 "PROXY_LABEL": PROXY_LABEL, "waiting_vehicle_seconds_measured": r["waiting_vehicle_seconds"],
                 "vehicles_admitted": r["vehicles_admitted"], "fuel_liters_proxy": r["fuel_liters_proxy"],
                 "co2_kg_proxy": r["co2_kg_proxy"], "fuel_liters_proxy_per_admitted": r["fuel_liters_proxy_per_admitted"],
                 "co2_kg_proxy_per_admitted": r["co2_kg_proxy_per_admitted"],
                 "fuel_proxy_pct_vs_fixed": r["fuel_liters_proxy_pct_vs_fixed"],
                 "co2_proxy_pct_vs_fixed": r["co2_kg_proxy_pct_vs_fixed"],
                 "idle_fuel_rate_lph_assumed": r["idle_fuel_rate_lph_assumed"],
                 "emission_factor_kg_per_liter_assumed": r["emission_factor_kg_per_liter_assumed"]} for r in comp]
    write_csv(out / "environmental_metrics.csv", env_rows)

    summary_rows = []
    for scenario in dict.fromkeys(r["scenario"] for r in comp):
        for ctrl in CONTROLLERS:
            g = [r for r in comp if r["scenario"] == scenario and r["controller"] == ctrl]
            if not g:
                continue
            fx = [r for r in comp if r["scenario"] == scenario and r["controller"] == "fixed" and r["seed"] in {x["seed"] for x in g}]
            row = {"scenario": scenario, "controller": ctrl, "seeds": len(g), "seed_list": " ".join(str(x["seed"]) for x in g)}
            for k in COMPARE_KEYS + ("plan_change_rate", "ns_green_share", "phase_utilization_ns", "phase_utilization_ew"):
                vals = [x[k] for x in g if x[k] is not None]
                row[f"{k}_mean"] = statistics.fmean(vals) if vals else None
            for k in COMPARE_KEYS:
                base = statistics.fmean(x[k] for x in fx)
                row[f"{k}_pct_vs_fixed_of_means"] = percent_change(base, row[f"{k}_mean"])
            diffs = [x["waiting_seconds_per_admitted_pct_vs_fixed"] for x in g if x["waiting_seconds_per_admitted_pct_vs_fixed"] is not None]
            row["seeds_waiting_per_admitted_better_than_fixed"] = sum(d < -1e-9 for d in diffs)
            row["seeds_waiting_per_admitted_worse_than_fixed"] = sum(d > 1e-9 for d in diffs)
            summary_rows.append(row)
    write_csv(out / "controller_summary.csv", summary_rows)

    # ------------------------------------------------------------------ 2. emergency corridor
    event = EmergencyEvent("EV1", EMERGENCY_ROUTE, EMERGENCY_START_CYCLE)
    em_rows, em_cmp, em_unified = [], [], {}
    cases = {c.name: c for c in CASES}
    for scenario in EMERGENCY_SCENARIOS:
        for base in EMERGENCY_BASES:
            for seed in SEEDS:
                sim = Simulator(net, cases[scenario].scenario.build_demand(net, seed))
                start = initial_state_for(cases[scenario], sim)
                runs = {}
                for mode, corridor in (("A_no_override", False), ("B_corridor", True)):
                    ctrl = FixedTimeController() if base == "fixed" else AdaptiveController()
                    runs[mode] = run_emergency(sim, ctrl, event, CYCLES, corridor, start)
                    if (scenario, base, seed, mode) in p5 and abs(p5[(scenario, base, seed, mode)] - runs[mode].vehicle.travel_time) > 1e-6:
                        checks["phase5_mismatches"].append((scenario, base, seed, mode))
                first, last = start.cycle, start.cycle + CYCLES - 1
                end_during = max(r.vehicle.completion_cycle for r in runs.values())
                windows = {"during": (event.start_cycle, end_during), "post": (end_during + 1, last), "total": (first, last)}
                um = {}
                for mode, run in runs.items():
                    for wname, (lo, hi) in windows.items():
                        hist = [r for r in run.history if lo <= r.cycle <= hi]
                        um[(mode, wname)] = unified_metrics(hist, env, sim.config.saturation_flow, run)
                        em_rows.append({"scenario": scenario, "base_controller": base, "seed": seed, "mode": mode,
                                        "window": wname, "first_cycle": lo, "last_cycle": hi,
                                        **um[(mode, wname)].flat()})
                a_e, b_e = runs["A_no_override"].vehicle, runs["B_corridor"].vehicle
                row = {"scenario": scenario, "base_controller": base, "seed": seed,
                       "A_travel_time_s": a_e.travel_time, "B_travel_time_s": b_e.travel_time,
                       "A_delay_s": a_e.delay, "B_delay_s": b_e.delay, "A_stops": a_e.stops, "B_stops": b_e.stops,
                       "A_completion_cycle": a_e.completion_cycle, "B_completion_cycle": b_e.completion_cycle,
                       "travel_time_pct": percent_change(a_e.travel_time, b_e.travel_time),
                       "priority_intersections_B": b_e.priority_intersections}
                for wname in windows:
                    a, b = um[("A_no_override", wname)].flat(), um[("B_corridor", wname)].flat()
                    for k in ("waiting_vehicle_seconds", "average_queue_vehicles", "max_queue_vehicles",
                              "throughput_vehicles_per_hour", "final_queue_vehicles", "vehicles_rejected",
                              "blocked_vehicle_cycles", "fuel_liters_proxy", "co2_kg_proxy"):
                        row[f"{wname}_{k}_A"], row[f"{wname}_{k}_B"] = a[k], b[k]
                        row[f"{wname}_{k}_diff"] = b[k] - a[k]
                        row[f"{wname}_{k}_pct"] = percent_change(a[k], b[k])
                em_cmp.append(row)
                em_unified[(scenario, base, seed)] = um
    write_csv(out / "emergency_metrics.csv", em_rows)
    write_csv(out / "emergency_comparison.csv", em_cmp)

    # ------------------------------------------------------------------ 3. sensitivity
    sens = []
    configs = sensitivity_configs()
    for cfg in configs:
        pid = f"idle{cfg.idle_fuel_rate_lph:g}Lph_{cfg.fuel_type}"
        for scenario in dict.fromkeys(r["scenario"] for r in comp):
            means = {}
            for ctrl in CONTROLLERS:
                g = [(sd, unified_by_key[(scenario, sd, ctrl)]) for sd in SEEDS if (scenario, sd, ctrl) in unified_by_key]
                if not g:
                    continue
                from qtraffic.environment import estimate
                ests = [estimate(u.traffic.waiting_vehicle_seconds, cfg) for _, u in g]
                means[ctrl] = (statistics.fmean(e.fuel_liters for e in ests), statistics.fmean(e.co2_kg for e in ests),
                               statistics.fmean(u.traffic.waiting_vehicle_seconds for _, u in g), len(g))
            fx_seeds = None
            for ctrl, (fuel, co2, wait, n) in means.items():
                # baseline: Fixed over the same seeds (QAOA covers seed 0 only)
                base_seeds = [sd for sd in SEEDS if (scenario, sd, ctrl) in unified_by_key]
                base_fuel = statistics.fmean(estimate(unified_by_key[(scenario, sd, "fixed")].traffic.waiting_vehicle_seconds, cfg).fuel_liters for sd in base_seeds)
                base_co2 = statistics.fmean(estimate(unified_by_key[(scenario, sd, "fixed")].traffic.waiting_vehicle_seconds, cfg).co2_kg for sd in base_seeds)
                sens.append({"kind": "controller_comparison", "parameter_set": pid, "idle_fuel_rate_lph": cfg.idle_fuel_rate_lph,
                             "emission_factor_kg_per_liter": cfg.emission_factor_kg_per_liter, "fuel_type": cfg.fuel_type,
                             "scenario": scenario, "subject": ctrl, "seeds": n, "waiting_vehicle_seconds_mean_measured": wait,
                             "fuel_liters_proxy_mean": fuel, "co2_kg_proxy_mean": co2,
                             "fuel_proxy_pct_vs_fixed": percent_change(base_fuel, fuel),
                             "co2_proxy_pct_vs_fixed": percent_change(base_co2, co2), "PROXY_LABEL": PROXY_LABEL})
        for scenario in EMERGENCY_SCENARIOS:
            for base in EMERGENCY_BASES:
                from qtraffic.environment import estimate
                fa = statistics.fmean(estimate(em_unified[(scenario, base, sd)][("A_no_override", "total")].traffic.waiting_vehicle_seconds, cfg).fuel_liters for sd in SEEDS)
                fb = statistics.fmean(estimate(em_unified[(scenario, base, sd)][("B_corridor", "total")].traffic.waiting_vehicle_seconds, cfg).fuel_liters for sd in SEEDS)
                ca = statistics.fmean(estimate(em_unified[(scenario, base, sd)][("A_no_override", "total")].traffic.waiting_vehicle_seconds, cfg).co2_kg for sd in SEEDS)
                cb = statistics.fmean(estimate(em_unified[(scenario, base, sd)][("B_corridor", "total")].traffic.waiting_vehicle_seconds, cfg).co2_kg for sd in SEEDS)
                sens.append({"kind": "emergency_corridor_B_vs_A", "parameter_set": pid, "idle_fuel_rate_lph": cfg.idle_fuel_rate_lph,
                             "emission_factor_kg_per_liter": cfg.emission_factor_kg_per_liter, "fuel_type": cfg.fuel_type,
                             "scenario": scenario, "subject": f"{base}: corridor vs no override", "seeds": len(SEEDS),
                             "fuel_liters_proxy_mean_A": fa, "fuel_liters_proxy_mean_B": fb, "co2_kg_proxy_mean_A": ca,
                             "co2_kg_proxy_mean_B": cb, "fuel_proxy_pct_B_vs_A": percent_change(fa, fb),
                             "co2_proxy_pct_B_vs_A": percent_change(ca, cb), "PROXY_LABEL": PROXY_LABEL})
    write_csv(out / "sensitivity.csv", sens)

    # stability of controller comparisons across the parameter grid
    stability = {}
    for scenario in dict.fromkeys(r["scenario"] for r in comp):
        for ctrl in CONTROLLERS:
            vals = [r["fuel_proxy_pct_vs_fixed"] for r in sens if r["kind"] == "controller_comparison"
                    and r["scenario"] == scenario and r["subject"] == ctrl and r["fuel_proxy_pct_vs_fixed"] is not None]
            if vals:
                stability[f"{scenario}/{ctrl}"] = {"min_pct": min(vals), "max_pct": max(vals), "spread_pct_points": max(vals) - min(vals)}
    rank_stable = True
    for scenario in dict.fromkeys(r["scenario"] for r in comp):
        orders = set()
        for pid in {r["parameter_set"] for r in sens}:
            rows = [r for r in sens if r["kind"] == "controller_comparison" and r["scenario"] == scenario
                    and r["parameter_set"] == pid and r["seeds"] == len(SEEDS)]
            orders.add(tuple(r["subject"] for r in sorted(rows, key=lambda r: (round(r["fuel_liters_proxy_mean"], 9), r["subject"]))))
        rank_stable &= len(orders) == 1

    # ------------------------------------------------------------------ 4. determinism
    case0 = cases["balanced_medium"]
    sim0 = Simulator(net, case0.scenario.build_demand(net, 0))
    st0 = initial_state_for(case0, sim0)
    a = unified_metrics(run_from_state(sim0, st0, AdaptiveController(), CYCLES).history, env)
    b = unified_metrics(run_from_state(sim0, st0, AdaptiveController(), CYCLES).history, env)
    deterministic = a == b
    e1 = run_emergency(sim0, FixedTimeController(), event, CYCLES, True, st0)
    e2 = run_emergency(sim0, FixedTimeController(), event, CYCLES, True, st0)
    deterministic_em = unified_metrics(e1.history, env, None, e1) == unified_metrics(e2.history, env, None, e2)

    verification = {
        "conservation_max_abs_error": checks["conservation_max_abs_error"],
        "controller_runs_checked": len(unified_by_key),
        "demand_identical_across_controllers": not checks["demand_mismatches"],
        "waiting_time_matches_recorded_phase3_and_phase4": not checks["phase3_phase4_mismatches"],
        "emergency_travel_time_matches_recorded_phase5": not checks["phase5_mismatches"],
        "deterministic_replay_controller_metrics": deterministic,
        "deterministic_replay_emergency_metrics": deterministic_em,
        "sensitivity_controller_ranking_identical_for_every_coefficient_set": rank_stable,
    }
    verification["all_pass"] = all(v for k, v in verification.items() if k not in ("conservation_max_abs_error", "controller_runs_checked")) \
        and verification["conservation_max_abs_error"] < 1e-6

    # ------------------------------------------------------------------ configuration + summary
    configuration = {
        "LABEL": PROXY_LABEL,
        "environmental_proxy": {
            "formulas": {"fuel_liters": "waiting_vehicle_seconds / 3600 * idle_fuel_rate_lph",
                         "co2_kg": "fuel_liters * emission_factor_kg_per_liter"},
            "defaults": env.provenance(),
            "sources": {
                "emission_factor": {
                    "status": "SOURCE-DERIVED, page read for this project",
                    "source": "U.S. EPA, Greenhouse Gas Emissions from a Typical Passenger Vehicle",
                    "url": "https://www.epa.gov/greenvehicles/greenhouse-gas-emissions-typical-passenger-vehicle",
                    "quoted": {"gasoline_g_co2_per_us_gallon": EPA_GASOLINE_G_CO2_PER_US_GALLON,
                               "diesel_g_co2_per_us_gallon": EPA_DIESEL_G_CO2_PER_US_GALLON},
                    "page_last_updated": "June 3, 2026", "liters_per_us_gallon": LITERS_PER_US_GALLON,
                    "derived_kg_per_liter": {"gasoline": GASOLINE_KG_CO2_PER_LITER, "diesel": DIESEL_KG_CO2_PER_LITER},
                    "scope": "tailpipe combustion CO2 only (not life-cycle)"},
                "idle_fuel_rate": {
                    "status": "CONFIGURABLE SIMULATION PARAMETER (not a universal vehicle constant; not a calibrated fleet average)",
                    "default_lph": DEFAULT_IDLE_FUEL_RATE_LPH,
                    "default_derivation": "0.16 gal/h x 3.78541 L/gal = 0.606 L/h, rounded to 0.6",
                    "default_choice": "the lowest of the published examples, so the proxy does not overstate fuel use",
                    "source": {
                        "publisher": "Argonne National Laboratory",
                        "title": "Idling Reduction Savings Calculator",
                        "url": "https://www.anl.gov/sites/www/files/2018-02/idling_worksheet.pdf",
                        "published_examples_gal_per_hour_no_load": ARGONNE_IDLE_EXAMPLES_GAL_PER_HOUR,
                        "published_examples_liters_per_hour": {k: gallons_per_hour_to_liters_per_hour(v)
                                                               for k, v in ARGONNE_IDLE_EXAMPLES_GAL_PER_HOUR.items()},
                        "figures_provenance": "document title, publisher and URL confirmed; figures supplied to the project by "
                                              "its owner from that worksheet; the PDF text could not be retrieved by the build "
                                              "tooling (HTTP 403), so the figures were not independently re-read there"},
                    "doe_general_statement_gal_per_hour": list(DOE_GENERAL_IDLE_RANGE_GAL_PER_HOUR),
                    "doe_general_statement_provenance": "general statement supplied to the project; no specific page cited",
                    "sensitivity_values_lph": list(SENSITIVITY_IDLE_FUEL_RATES_LPH),
                    "sensitivity_values_meaning": "scenario parameters spanning the examples and the DOE range, not vehicle characteristics"}},
            "not_modelled": ["acceleration/deceleration", "vehicle speed", "vehicle type", "engine efficiency",
                             "road gradient", "temperature", "congestion-dependent fuel burn", "actual vehicle trajectories",
                             "exhaust composition", "NOx/PM emissions", "fuel used while moving or in transit"],
            "statement": "SIMULATION PROXY - NOT A MEASUREMENT. The environmental model uses a configurable waiting/idling "
                         "fuel-rate proxy. The default is based on published passenger-vehicle idling examples and is not a "
                         "calibrated fleet-average value. Fuel and CO2 are linear in the simulator's waiting vehicle-seconds; "
                         "they are not measurements and not calibrated real-world estimates."},
        "experiment": {"scenarios": [c.name for c in CASES], "seeds": list(SEEDS), "cycles": CYCLES,
                       "start_state": "state after 30 fixed-time cycles (Phase 3/4), or the synthetic jam for downstream_congested",
                       "controllers": list(CONTROLLERS), "qaoa": "plans SAVED by Phase 4 (seed 0 only); QAOA not rerun",
                       "emergency": {"route": list(EMERGENCY_ROUTE), "start_cycle": EMERGENCY_START_CYCLE,
                                     "link_seconds": event.link_seconds, "lead_seconds": event.lead_seconds,
                                     "scenarios": list(EMERGENCY_SCENARIOS), "bases": list(EMERGENCY_BASES)}},
        "versions": {"qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__},
        "metric_definitions": "see qtraffic.unified_metrics docstring (demanded/admitted/served/exited/throughput/in transit/waiting/queue)"}
    (out / "configuration.json").write_text(json.dumps(configuration, indent=1))

    def sc(scenario, ctrl, k):
        r = next((x for x in summary_rows if x["scenario"] == scenario and x["controller"] == ctrl), None)
        return None if r is None else r[k]

    emergency_summary = []
    for scenario in EMERGENCY_SCENARIOS:
        for base in EMERGENCY_BASES:
            g = [r for r in em_cmp if r["scenario"] == scenario and r["base_controller"] == base]
            m = lambda k: statistics.fmean(r[k] for r in g)
            emergency_summary.append({
                "scenario": scenario, "base_controller": base, "seeds": len(g),
                "emergency_travel_time_s_A": m("A_travel_time_s"), "emergency_travel_time_s_B": m("B_travel_time_s"),
                "emergency_delay_s_A": m("A_delay_s"), "emergency_delay_s_B": m("B_delay_s"),
                "signal_stops_A": m("A_stops"), "signal_stops_B": m("B_stops"),
                "normal_total_window_waiting_vehicle_seconds_A": m("total_waiting_vehicle_seconds_A"),
                "normal_total_window_waiting_vehicle_seconds_B": m("total_waiting_vehicle_seconds_B"),
                "waiting_pct_B_vs_A_of_means": percent_change(m("total_waiting_vehicle_seconds_A"), m("total_waiting_vehicle_seconds_B")),
                "fuel_liters_proxy_A": m("total_fuel_liters_proxy_A"), "fuel_liters_proxy_B": m("total_fuel_liters_proxy_B"),
                "co2_kg_proxy_A": m("total_co2_kg_proxy_A"), "co2_kg_proxy_B": m("total_co2_kg_proxy_B"),
                "co2_proxy_pct_B_vs_A_of_means": percent_change(m("total_co2_kg_proxy_A"), m("total_co2_kg_proxy_B")),
                "max_queue_A": m("total_max_queue_vehicles_A"), "max_queue_B": m("total_max_queue_vehicles_B"),
                "final_queue_A": m("total_final_queue_vehicles_A"), "final_queue_B": m("total_final_queue_vehicles_B"),
                "seeds_normal_waiting_worse_with_corridor": sum(r["total_waiting_vehicle_seconds_diff"] > 1e-9 for r in g),
                "seeds_normal_waiting_better_with_corridor": sum(r["total_waiting_vehicle_seconds_diff"] < -1e-9 for r in g)})
    summary = {"LABEL": PROXY_LABEL, "verification": verification, "sensitivity_stability_fuel_pct_vs_fixed": stability,
               "controller_summary_rows": summary_rows, "emergency_summary": emergency_summary}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote results to {out}/ ; verification all_pass = {verification['all_pass']}")
    print(json.dumps(verification, indent=1))
    return 0 if verification["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
