"""Phase 5: Emergency Green Corridor, A/B on identical traffic.

Run:  python scripts/run_phase5_emergency.py [--out results/phase5]

For every (scenario, base controller, seed) the SAME network, start state, demand stream, emergency
route/start cycle and horizon are simulated twice; only the emergency override differs:
  A  no override   (the emergency vehicle still travels the network under the base controller)
  B  corridor      (base controller -> emergency override -> final plan)
The emergency vehicle is massless (see qtraffic.emergency), so A's normal traffic equals a simulation
with no emergency at all; every difference between A and B is caused by the override.

Configuration fixed BEFORE looking at results (nothing here is tuned):
  route I1 -> I2 -> I3 -> I6, start cycle 40, link 30 s, lead 0 s; start state = the Phase 3/4 state
  after 30 fixed-time cycles; horizon 60 cycles (cycles 30..89); seeds 0..4.

Windows for normal-traffic metrics: pre = before the emergency starts (identical in A and B by
construction), during = start cycle .. the later of the two completion cycles, post = the rest.
The ABLATION section varies the modelling assumptions (link time, lead time, route) and is labelled as such.

Writes results/phase5/: emergency_runs.csv, emergency_comparison.csv, emergency_summary.csv,
primary_signal_log.csv, primary_emergency_timeline.json, ablation.csv, reproducibility.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import asdict
from pathlib import Path

from qtraffic import (
    AdaptiveController, FixedTimeController, Observation, Simulator, grid_network,
)
from qtraffic.emergency import EmergencyEvent, run_emergency
from qtraffic.experiments import demand_digest
from qtraffic.optimization import (
    QAOAConfig, QuboController, StaticPlanController, TrafficState, build_qubo, run_qaoa, solve_exact,
)
from qtraffic.optimization.validation import CASES, initial_state_for, state_digest

ROUTE = ("I1", "I2", "I3", "I6")
START_CYCLE = 40
CYCLES = 60
SEEDS = (0, 1, 2, 3, 4)
PRIMARY = ("balanced_medium", "fixed", 0)
SCENARIO_BASES = {
    "balanced_medium": ("fixed", "adaptive", "qubo_receding", "qubo_exact_static", "qaoa_p1_static"),
    "ns_heavy": ("fixed", "adaptive"),
    "ew_heavy": ("fixed", "adaptive"),
}
NORMAL_KEYS = ("total_waiting_time", "waiting_per_admitted", "average_queue_length", "max_queue_length",
               "throughput_per_hour", "vehicles_entered", "vehicles_rejected", "vehicles_exited",
               "total_blocked", "final_queue_total")
# metrics where a LOWER value is better (for the better/worse verdicts)
LOWER_BETTER = {"total_waiting_time", "waiting_per_admitted", "average_queue_length", "max_queue_length",
                "vehicles_rejected", "total_blocked", "final_queue_total"}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def normal_row(m) -> dict:
    return {"total_waiting_time": m.total_waiting_time,
            "waiting_per_admitted": m.total_waiting_time / m.vehicles_entered if m.vehicles_entered else 0.0,
            "average_queue_length": m.average_queue_length, "max_queue_length": m.max_queue_length,
            "throughput_per_hour": m.throughput_per_hour, "vehicles_entered": m.vehicles_entered,
            "vehicles_rejected": m.vehicles_rejected, "vehicles_exited": m.vehicles_exited,
            "total_blocked": m.total_blocked, "final_queue_total": m.final_queue_total}


def emergency_row(run) -> dict:
    v = run.vehicle
    return {"completed": v.completed, "entry_cycle": v.entry_cycle, "completion_cycle": v.completion_cycle,
            "emergency_travel_time_s": v.travel_time, "free_flow_time_s": v.free_flow_time,
            "emergency_delay_s": v.delay, "signal_stops": v.stops, "priority_intersections": v.priority_intersections,
            **{f"wait_{c.node}_s": c.wait for c in v.crossings}}


def make_base(name: str, sim: Simulator, start, qubo_cache: dict):
    """Instantiate the base controller. Static plans come from the exact QUBO / QAOA p=1 on the start state."""
    if name == "fixed":
        return FixedTimeController()
    if name == "adaptive":
        return AdaptiveController()
    if name == "qubo_receding":
        return QuboController()
    if "qubo" not in qubo_cache:
        obs = Observation(start.cycle, dict(start.queues), dict(start.in_transit), sim.network)
        qubo_cache["qubo"] = build_qubo(TrafficState.from_observation(obs))
    qubo = qubo_cache["qubo"]
    if name == "qubo_exact_static":
        return StaticPlanController(solve_exact(qubo).plans, "QuboExactStatic")
    if name == "qaoa_p1_static":
        plans = run_qaoa(qubo, QAOAConfig(p=1)).best_feasible_plans
        return StaticPlanController(plans, "QaoaP1Static")
    raise ValueError(name)


def run_pair(case, base_name: str, seed: int, event: EmergencyEvent, net) -> dict:
    sim = Simulator(net, case.scenario.build_demand(net, seed))
    start = initial_state_for(case, sim)
    cache: dict = {}
    off = run_emergency(sim, make_base(base_name, sim, start, cache), event, CYCLES, False, start)
    on = run_emergency(sim, make_base(base_name, sim, start, cache), event, CYCLES, True, start)
    first, last = start.cycle, start.cycle + CYCLES - 1
    end_during = max(c for c in (off.vehicle.completion_cycle, on.vehicle.completion_cycle) if c is not None) \
        if (off.vehicle.completed or on.vehicle.completed) else last
    windows = {"pre": (first, event.start_cycle - 1), "during": (event.start_cycle, end_during),
               "post": (end_during + 1, last), "total": (first, last)}
    src = demand_digest([sim.demand.arrivals(first + c) for c in range(CYCLES)])
    same_demand = all(demand_digest([r.demand for r in run.history]) == src for run in (off, on))
    return {"sim": sim, "start": start, "off": off, "on": on, "windows": windows,
            "demand_digest": src, "same_demand": same_demand, "state_digest": state_digest(start)}


def flatten(pair: dict, scenario: str, base: str, seed: int) -> tuple[list[dict], dict]:
    runs, wm = [], {}
    for mode, run in (("A_no_override", pair["off"]), ("B_corridor", pair["on"])):
        row = {"scenario": scenario, "base_controller": base, "seed": seed, "mode": mode,
               **emergency_row(run)}
        wm[mode] = {}
        for wname, (lo, hi) in pair["windows"].items():
            m = run.window_metrics(lo, hi) if lo <= hi else None
            if m is None:
                continue
            wm[mode][wname] = normal_row(m)
            for k in NORMAL_KEYS:
                row[f"{wname}_{k}"] = wm[mode][wname][k]
        runs.append(row)
    return runs, wm


def compare(scenario, base, seed, runs, wm) -> dict:
    a, b = runs[0], runs[1]
    row = {"scenario": scenario, "base_controller": base, "seed": seed,
           "A_completion_cycle": a["completion_cycle"], "B_completion_cycle": b["completion_cycle"]}
    for k in ("emergency_travel_time_s", "emergency_delay_s", "signal_stops"):
        row[f"A_{k}"], row[f"B_{k}"] = a[k], b[k]
        row[f"diff_{k}"] = b[k] - a[k] if a[k] is not None and b[k] is not None else None
    for w in ("during", "post", "total"):
        for k in NORMAL_KEYS:
            if w in wm["A_no_override"] and w in wm["B_corridor"]:
                va, vb = wm["A_no_override"][w][k], wm["B_corridor"][w][k]
                row[f"{w}_{k}_A"], row[f"{w}_{k}_B"], row[f"{w}_{k}_diff"] = va, vb, vb - va
                row[f"{w}_{k}_pct"] = (vb - va) / abs(va) * 100.0 if va else None
    return row


def verdict(metric: str, diff: float, tol: float = 1e-9) -> str:
    if abs(diff) <= tol:
        return "same"
    return "better" if (diff < 0) == (metric in LOWER_BETTER) else "worse"


def summarise(comparisons: list[dict]) -> list[dict]:
    out = []
    keys = sorted({(c["scenario"], c["base_controller"]) for c in comparisons})
    for scenario, base in keys:
        g = [c for c in comparisons if (c["scenario"], c["base_controller"]) == (scenario, base)]
        row = {"scenario": scenario, "base_controller": base, "seeds": len(g)}
        for k in ("emergency_travel_time_s", "emergency_delay_s", "signal_stops"):
            row[f"A_{k}_mean"] = statistics.fmean(c[f"A_{k}"] for c in g)
            row[f"B_{k}_mean"] = statistics.fmean(c[f"B_{k}"] for c in g)
            diffs = [c[f"diff_{k}"] for c in g]
            row[f"diff_{k}_mean"] = statistics.fmean(diffs)
            row[f"{k}_seeds_better"] = sum(d < -1e-9 for d in diffs)
            row[f"{k}_seeds_worse"] = sum(d > 1e-9 for d in diffs)
        for w in ("during", "post", "total"):
            for k in ("total_waiting_time", "waiting_per_admitted", "average_queue_length", "max_queue_length",
                      "throughput_per_hour", "vehicles_rejected", "total_blocked", "final_queue_total"):
                diffs = [c[f"{w}_{k}_diff"] for c in g]
                row[f"{w}_{k}_A_mean"] = statistics.fmean(c[f"{w}_{k}_A"] for c in g)
                row[f"{w}_{k}_B_mean"] = statistics.fmean(c[f"{w}_{k}_B"] for c in g)
                row[f"{w}_{k}_diff_mean"] = statistics.fmean(diffs)
                row[f"{w}_{k}_seeds_worse_for_normal_traffic"] = sum(verdict(k, d) == "worse" for d in diffs)
                row[f"{w}_{k}_seeds_better_for_normal_traffic"] = sum(verdict(k, d) == "better" for d in diffs)
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase5")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    net = grid_network(2, 3)
    event = EmergencyEvent("EV1", ROUTE, START_CYCLE)
    cases = {c.name: c for c in CASES}

    all_runs, comparisons, primary, demand_checks = [], [], None, []
    for scenario, bases in SCENARIO_BASES.items():
        for base in bases:
            for seed in SEEDS:
                pair = run_pair(cases[scenario], base, seed, event, net)
                demand_checks.append(pair["same_demand"])  # A and B consumed exactly the source demand
                runs, wm = flatten(pair, scenario, base, seed)
                all_runs += runs
                comparisons.append(compare(scenario, base, seed, runs, wm))
                if (scenario, base, seed) == PRIMARY:
                    primary = (pair, runs, wm)
            print(f"  done {scenario} / {base}", flush=True)

    pair, _, _ = primary
    # primary case detail: per-cycle signal decisions and the emergency timeline
    log_rows = []
    for mode, run in (("A_no_override", pair["off"]), ("B_corridor", pair["on"])):
        for rec, rec_hist in zip(run.log, run.history):
            if START_CYCLE - 2 <= rec.cycle <= START_CYCLE + 12:
                for node in net.nodes:
                    log_rows.append({"mode": mode, "cycle": rec.cycle, "node": node,
                                     "base_plan": rec.base_plans[node].label, "final_plan": rec.final_plans[node].label,
                                     "priority": node in rec.priority_nodes, "override_changed_plan": node in rec.changed_nodes,
                                     "first_phase": rec.first_phase[node].value if node in rec.first_phase else "",
                                     "emergency_active": rec.active})
    write_csv(out / "primary_signal_log.csv", log_rows)
    timeline = {}
    for mode, run in (("A_no_override", pair["off"]), ("B_corridor", pair["on"])):
        v = run.vehicle
        timeline[mode] = {
            "vehicle_id": v.vehicle_id, "route": list(v.route), "status": v.status.value,
            "entry_cycle": v.entry_cycle, "entry_time_s": v.entry_time, "completion_cycle": v.completion_cycle,
            "completion_time_s": v.completion_time, "travel_time_s": v.travel_time, "free_flow_time_s": v.free_flow_time,
            "delay_s": v.delay, "stops": v.stops,
            "priority_cycles_by_intersection": v.priority_cycles,
            "crossings": [{"node": c.node, "approach": str(c.approach), "arrival_time_s": c.arrival_time,
                           "crossing_time_s": c.crossing_time, "wait_s": c.wait, "stopped": c.stopped,
                           "cycle": c.cycle, "priority": c.priority} for c in v.crossings],
            "events": v.events}
    (out / "primary_emergency_timeline.json").write_text(json.dumps(timeline, indent=1))

    # reproducibility: rerun the primary configuration and compare everything
    again = run_pair(cases[PRIMARY[0]], PRIMARY[1], PRIMARY[2], event, net)
    ident = {}
    for mode in ("off", "on"):
        x, y = pair[mode], again[mode]
        ident[mode] = {
            "emergency_travel_time_identical": x.vehicle.travel_time == y.vehicle.travel_time,
            "completion_cycle_identical": x.vehicle.completion_cycle == y.vehicle.completion_cycle,
            "emergency_events_identical": x.vehicle.events == y.vehicle.events,
            "signal_decisions_identical": [r.final_plans for r in x.log] == [r.final_plans for r in y.log],
            "normal_metrics_identical": x.result.metrics == y.result.metrics}
    pre_a = pair["off"].window_metrics(*pair["windows"]["pre"])
    pre_b = pair["on"].window_metrics(*pair["windows"]["pre"])
    repro = {
        "configuration": {"route": list(ROUTE), "start_cycle": START_CYCLE, "link_seconds": event.link_seconds,
                          "lead_seconds": event.lead_seconds, "cycles": CYCLES, "seeds": list(SEEDS),
                          "scenario_bases": SCENARIO_BASES, "primary": list(PRIMARY),
                          "saturation_flow": pair["sim"].config.saturation_flow, "event": asdict(event)},
        "primary_run_twice_identical": ident, "primary_all_identical": all(all(v.values()) for v in ident.values()),
        "pairs_checked_for_identical_demand": len(demand_checks),
        "demand_identical_between_A_and_B_in_every_pair": all(demand_checks),
        "pre_emergency_window_identical_A_vs_B": pre_a == pre_b,
        "primary_start_state_sha256": pair["state_digest"], "primary_demand_sha256": pair["demand_digest"],
        "emergency_vehicle_is_massless_so_A_equals_no_emergency": True,
    }
    (out / "reproducibility.json").write_text(json.dumps(repro, indent=1))
    write_csv(out / "emergency_runs.csv", all_runs)
    write_csv(out / "emergency_comparison.csv", comparisons)
    write_csv(out / "emergency_summary.csv", summarise(comparisons))

    # ---- ABLATION (modelling assumptions; NOT the primary result) --------------------------------------
    abl = []
    routes = {"primary I1-I2-I3-I6": ROUTE, "row I1-I2-I3": ("I1", "I2", "I3"),
              "west-bound I6-I5-I4-I1": ("I6", "I5", "I4", "I1"), "north-south I1-I4": ("I1", "I4")}
    settings = [(rname, r, link, lead) for rname, r in routes.items() for link in (30.0,) for lead in (0.0,)]
    settings += [("primary I1-I2-I3-I6", ROUTE, link, lead) for link in (20.0, 45.0, 60.0) for lead in (0.0,)]
    settings += [("primary I1-I2-I3-I6", ROUTE, 30.0, 30.0), ("primary I1-I2-I3-I6", ROUTE, 30.0, 60.0)]
    for rname, route, link, lead in settings:
        ev = EmergencyEvent("EV1", route, START_CYCLE, link_seconds=link, lead_seconds=lead)
        rows = []
        for seed in SEEDS:
            p = run_pair(cases["balanced_medium"], "fixed", seed, ev, net)
            runs, wm = flatten(p, "balanced_medium", "fixed", seed)
            rows.append(compare("balanced_medium", "fixed", seed, runs, wm))
        abl.append({"ABLATION": "assumption sensitivity (not the primary result)", "route": rname, "link_seconds": link,
                    "lead_seconds": lead, "seeds": len(rows),
                    "A_travel_time_mean_s": statistics.fmean(r["A_emergency_travel_time_s"] for r in rows),
                    "B_travel_time_mean_s": statistics.fmean(r["B_emergency_travel_time_s"] for r in rows),
                    "diff_travel_time_mean_s": statistics.fmean(r["diff_emergency_travel_time_s"] for r in rows),
                    "A_delay_mean_s": statistics.fmean(r["A_emergency_delay_s"] for r in rows),
                    "B_delay_mean_s": statistics.fmean(r["B_emergency_delay_s"] for r in rows),
                    "seeds_corridor_faster": sum(r["diff_emergency_travel_time_s"] < -1e-9 for r in rows),
                    "seeds_corridor_slower": sum(r["diff_emergency_travel_time_s"] > 1e-9 for r in rows),
                    "total_waiting_time_diff_mean": statistics.fmean(r["total_total_waiting_time_diff"] for r in rows),
                    "total_waiting_time_pct_mean": statistics.fmean(r["total_total_waiting_time_pct"] for r in rows)})
    write_csv(out / "ablation.csv", abl)
    print(f"wrote results to {out}/ ; primary identical on rerun: {repro['primary_all_identical']}")
    return 0 if repro["primary_all_identical"] and repro["pre_emergency_window_identical_A_vs_B"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
