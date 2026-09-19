"""Phase 2: Fixed-time vs Adaptive on identical traffic.

Run:  python scripts/run_phase2_experiments.py [--cycles 120] [--seeds 0 1 2 3 4] [--out results/phase2]

Writes (all values computed by the simulator):
  runs.csv       one row per scenario x seed x controller (raw metrics)
  paired.csv     one row per scenario x seed x candidate: fixed vs candidate, % difference, verdict
  summary.csv    per scenario x candidate: mean over seeds + % difference of the means
  (candidates: "adaptive" = primary, no hysteresis; "adaptive_dwell2" = 2-cycle dwell ablation)
  decisions.json adaptive pressures and chosen plans, every intersection/cycle, for --decisions-seed
  verification.json  SHA-256 digests proving every controller consumed identical demand
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from qtraffic.experiments import (
    CANDIDATES, DEFAULT_CYCLES, DEFAULT_SEEDS, METRICS, SCENARIOS, aggregate, pct_diff, run_all, verdict,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--decisions-seed", type=int, default=DEFAULT_SEEDS[0])
    ap.add_argument("--out", default="results/phase2")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = run_all(SCENARIOS, args.seeds, args.cycles, decisions_seed=args.decisions_seed)

    runs, paired = [], []
    for p in pairs:
        for ctrl in ("fixed", *CANDIDATES):
            runs.append({"scenario": p["scenario"], "seed": p["seed"], "controller": ctrl,
                         "cycles": p["cycles"], **p[ctrl]})
        for cand in CANDIDATES:
            row = {"scenario": p["scenario"], "seed": p["seed"], "candidate": cand}
            for m in METRICS:
                f, a = p["fixed"][m], p[cand][m]
                row.update({f"{m}_fixed": f, f"{m}_candidate": a, f"{m}_pct": pct_diff(f, a),
                            f"{m}_verdict": verdict(m, f, a)})
            paired.append(row)

    write_csv(out / "runs.csv", runs)
    write_csv(out / "paired.csv", paired)
    write_csv(out / "summary.csv", [r for c in CANDIDATES for r in aggregate(pairs, c)])

    decisions = {
        p["scenario"]: [
            {"cycle": d.cycle, "node": d.node, "P_NS": round(d.pressure_ns, 3),
             "P_EW": round(d.pressure_ew, 3), "delta": round(d.delta, 3), "desired": d.desired.label, "plan": d.plan.label}
            for d in p["decisions"]
        ]
        for p in pairs if p["decisions"]
    }
    (out / "decisions.json").write_text(json.dumps(
        {"seed": args.decisions_seed, "scenarios": decisions}, separators=(",", ":")))
    (out / "verification.json").write_text(json.dumps(
        [{"scenario": p["scenario"], "seed": p["seed"], "cycles": p["cycles"],
          "demand_sha256": p["demand_digest"], "identical_for_all_controllers": True}
         for p in pairs], indent=1))

    print(f"{len(pairs)} scenario/seed pairs x 3 controllers written to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
