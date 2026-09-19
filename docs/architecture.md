# Architecture

The system is a layered pipeline. Every layer only reads from the layers above it, and the simulator stays the source of truth for
traffic numbers.

```
Traffic Demand                     demand.py            seeded, reproducible arrivals at the boundary of the network
      |
Traffic Network                    network.py           2 x 3 grid of intersections I1..I6, 24 approaches, 10 entry approaches
      |
Multi-Intersection Simulator       simulator.py         cycle-by-cycle queue-based fluid model, 60 s cycles
      |
Traffic State / Observation        controllers/base.py  queues + in-transit vehicles at the start of a cycle
      |
Controller                         chooses NS20/EW40, NS30/EW30 or NS40/EW20 for every intersection, every cycle
      |-- Fixed                    controllers/fixed.py      always NS30/EW30
      |-- Adaptive                 controllers/adaptive.py   queue-pressure rule (classical)
      |-- QUBO Static              optimization/adapter.py   exact solve of the QUBO once (initial state), plan held
      |-- QUBO Receding            optimization/adapter.py   exact solve of the QUBO re-built every cycle
      `-- QAOA p=1                 optimization/adapter.py   replays the plan saved by the QAOA experiment (optimization/qaoa.py), held
      |
Signal Decisions                   signals.py           validated one plan per intersection
      |
Metrics                            metrics.py, unified_metrics.py, environment.py
      |
Dashboard                          app.py, dashboard/   Streamlit control room


Emergency Vehicle                  emergency.py
      |
Emergency Green Corridor           priority override that WRAPS any base controller
      |
Priority Override                  base plan -> override -> final plan (the simulator is unchanged)


Saved Results / live run           results/phase2 .. phase6 (CSV / JSON written by scripts/) and the dashboard's live simulator run
      |
Structured AI Context              ai/context.py        whitelisted, JSON, numbers only
      |
Featherless (optional) / local fallback   ai/featherless.py, ai/explain.py
      |
Explanation                        text shown in the dashboard's AI Traffic Analyst panel
```

## The AI layer explains; it does not control

The AI layer sits at the end of the pipeline: no simulation, optimization or control code depends on it. It reads structured summaries of results that the simulator and
the optimizers already produced and returns text. It does not select traffic signals, does not build or solve the QUBO, does not set
QAOA parameters, does not operate the emergency corridor and does not touch the simulator. `qtraffic.ai` imports nothing from the
simulator, controllers, optimization or emergency code, and none of that code imports `qtraffic.ai` (this is tested). Without a
Featherless key the layer produces a deterministic local explanation from the same numbers. See `docs/featherless_integration.md`.

## Module map

| Area | Files |
|---|---|
| Simulation | `src/qtraffic/{signals,network,demand,simulator,metrics,experiments}.py` |
| Controllers | `src/qtraffic/controllers/{base,fixed,adaptive}.py` |
| QUBO, exact solver, QAOA | `src/qtraffic/optimization/{qubo,exact,adapter,qaoa,analysis,validation}.py` |
| Emergency corridor | `src/qtraffic/emergency.py` |
| Metrics and environmental proxy | `src/qtraffic/{unified_metrics,environment}.py` |
| AI explanation layer | `src/qtraffic/ai/{config,context,results,featherless,explain}.py` |
| Dashboard | `app.py` (thin entry point), `src/qtraffic/dashboard/{state,saved,components,network_view,charts,theme}.py` |
| Experiments and audits | `scripts/` (see `docs/reproducibility.md`) |
| Saved results | `results/` |

`qtraffic.optimization` is the only part that needs Qiskit and SciPy; the simulator, controllers, emergency corridor and metrics need only
NumPy and NetworkX.

## Dashboard data flow

The dashboard is a presentation layer. `dashboard/state.py` runs the configured scenario LIVE with the existing simulator (plus a Fixed
baseline on the same demand, and for an emergency run the corridor and no-override pair) and derives the views from the simulator's
per-cycle records. The controller-comparison, QAOA and sensitivity panels read the saved Phase 4 to 6 results through
`dashboard/saved.py`. Nothing is hard-coded; a missing result file becomes a visible warning, not a fabricated value. The QAOA p=1
controller replays the plan saved by the Phase 4 experiment (seed 0 only); the dashboard never re-runs QAOA.
