# Quantum-Enhanced Adaptive Urban Traffic Optimization

A simulation study of signal control on a small urban road network. Classical controllers, an exact QUBO optimizer and a QAOA
circuit (run on a classical simulator) choose signal plans for six intersections; the results are shown in a Streamlit control-room
dashboard with an Emergency Green Corridor and an AI analyst that only explains results.

> **What this project is and is not.** All numbers come from a deterministic queue-based simulation. QAOA was evaluated on a
> classical quantum-circuit simulator (Qiskit Aer): no quantum hardware was used, and the project makes **no claim of quantum
> advantage or speedup**. Fuel and CO2 values are waiting-based **simulation proxy** values, not real-world measurements.

## Overview

- A 2 x 3 grid of intersections (I1 to I6) simulated cycle by cycle (60 s per cycle) with a queue-based fluid model.
- Five controllers on identical demand: Fixed, Adaptive, QUBO Static, QUBO Receding and QAOA p=1.
- The signal-plan choice is written as a QUBO with 18 binary variables and solved exactly (ground truth) and with QAOA.
- An Emergency Green Corridor that gives an emergency vehicle priority along I1 → I2 → I3 → I6 and then restores normal control.
- A waiting-based fuel/CO2 simulation proxy, a Streamlit dashboard, and an optional Featherless AI analyst with a deterministic
  local fallback.

## Problem

Urban signal control has to decide, for every intersection and every cycle, how to split a cycle between the two conflicting
directions. Neighbouring intersections interact: green given to a direction sends vehicles onto a road that may already be full.
The project asks how fixed, rule-based and optimization-based controllers compare in a transparent simulator, and how an emergency
vehicle can be served without leaving the normal controller broken afterwards.

## Solution

Each cycle, every intersection chooses one of three plans: NS 20 s / EW 40 s, NS 30 s / EW 30 s or NS 40 s / EW 20 s. The
controllers differ only in how they choose. The simulator applies the plans and reports waiting time, throughput and queues, so
every controller is judged by the same simulator, not by its own objective.

## Key features

- Deterministic, seeded simulator with conservation checks; 452 automated tests.
- Exact QUBO solver used as ground truth; 18-qubit QAOA (p=1, p=2) on Qiskit Aer, with the exact optimum used only for evaluation.
- Emergency Green Corridor implemented as an override that wraps any controller.
- Environmental proxy with the coefficients and their sources documented.
- Dashboard with a live network schematic, playback, intersection inspector, timelines, optimization and comparison panels.
- Optional AI explanation layer that cannot control anything and works without an API key.
- Saved experiment results for Phases 2 to 6 that regenerate exactly from the scripts.

## System architecture

```
Traffic Demand -> Traffic Network -> Multi-Intersection Simulator -> Traffic State / Observation
   -> Controller: Fixed | Adaptive | QUBO Static | QUBO Receding | QAOA p=1
   -> Signal Decisions -> Metrics -> Dashboard

Emergency Vehicle -> Emergency Green Corridor -> Priority Override (wraps the controller above)

Saved Results -> Structured AI Context -> Featherless / local fallback -> Explanation
```

The AI layer explains validated simulation results. It does not select traffic signals, optimize the QUBO, operate the emergency
corridor or control the simulator. See [docs/architecture.md](docs/architecture.md).

## Traffic simulation

- **Network:** 6 intersections in a 2 x 3 grid, 24 approaches (one queue each), 10 of them entry approaches; each road holds at most
  40 vehicles. Vehicles go straight through (no turning movements).
- **Cycle:** 60 s. Each phase discharges at a saturation flow of 0.5 vehicles per second of green, limited by its queue and by free
  space on the road ahead (spillback is modelled).
- **Demand:** seeded, per entry approach and cycle; medium demand is 12 vehicles per cycle per entry with a +/-25% fluctuation.
- **Scenarios:** `balanced_medium`, `ns_heavy` (north-south x1.5, east-west x0.5), `ew_heavy`, `time_varying` (30-cycle blocks) and
  `downstream_congested` (east-west demand from a jammed eastbound start).
- **Runs:** 60 cycles (one simulated hour) from the state after 30 fixed-time warm-up cycles (a synthetic jam for
  `downstream_congested`); all controllers see the same start state and demand stream.
- **Metrics:** waiting vehicle-seconds and waiting per admitted vehicle, throughput, average / maximum / final queue, blocked and
  rejected vehicles, with an exact vehicle-conservation check. Definitions: [docs/metrics_and_environmental_model.md](docs/metrics_and_environmental_model.md).

## Controllers

All controllers see only the queues and in-transit vehicles at the start of a cycle.

### Fixed

Always NS 30 s / EW 30 s at every intersection. The baseline.

### Adaptive

A classical queue-pressure (max-pressure style) rule per intersection. The pressure of an approach is its queue plus in-transit vehicles
minus the queue and in-transit vehicles of the road it discharges into. If the north-south pressure exceeds the east-west pressure by more than 5 vehicles it picks NS 40 / EW 20;
if it is lower by more than 5 it picks NS 20 / EW 40; otherwise NS 30 / EW 30. The parameters were fixed by reasoning before any
experiment and are not tuned. The rule flip-flops under balanced demand; this is a documented limitation.

### QUBO Static

Builds the QUBO from the initial traffic state, solves it exactly, and holds the selected plan for the whole run.

### QUBO Receding

Rebuilds and exactly re-solves the QUBO from the observed state at the start of every cycle.

### QAOA p=1

Replays the plan selected by the saved QAOA p=1 run of the Phase 4 experiment (Qiskit Aer, seed 0) and holds it. It is available only
where a saved plan exists.

## QUBO formulation

Six intersections x three plans = **18 binary variables** with a one-hot constraint per intersection: **729** valid configurations out
of **262,144** (2^18) binary strings. The objective is a two-cycle look-ahead surrogate of waiting time in vehicle-seconds that couples
neighbouring intersections through downstream road space, plus a one-hot penalty with `P = B + 1` (a proven, untuned bound, checked
exhaustively over all 2^18 strings for all 25 scenario-seed start states). The QUBO energy is a surrogate, not a waiting time. Because it looks only two cycles ahead, a plan
held for a whole run can be poor when demand changes; the receding variant mitigates this. Details and limitations:
[docs/optimization.md](docs/optimization.md).

## QAOA implementation

QUBO -> Ising Hamiltonian -> 18-qubit QAOA circuit (p=1 and p=2; `p` is the circuit depth, not the number of qubits) -> Qiskit Aer
statevector simulation -> 8192 shots with a fixed seed -> best feasible sample. Parameters are optimized classically with SciPy COBYLA
from deterministic start angles; the optimizer never receives the exact optimum, which is used only afterwards for comparison.

**QAOA was evaluated using a classical quantum-circuit simulator (Qiskit Aer).** The experiment measures solution quality and sampling
behaviour. It does not use quantum hardware and makes no claim of speedup or advantage.

Known limitations, all kept in the results: the cold-start p=2 optimization hits its 200-evaluation limit in every scenario and misses
the exact optimum in two of five scenarios; p=1 misses it in `ew_heavy`; the warm-start p=2 run is a separate diagnostic experiment,
not a replacement for the primary benchmark; tied optimal energies can decode to plans with different simulator outcomes; QAOA was run
for one seed only. See [docs/optimization.md](docs/optimization.md).

## Emergency Green Corridor

An emergency vehicle (EV1) travels **I1 → I2 → I3 → I6** (30 s per road, 90 s at free flow). An override wraps the running controller
and, for the intersections the vehicle is about to reach, applies the plan that gives its phase the most green (40 s) and runs that phase
first. Each intersection is released the cycle after the vehicle crosses it. After the vehicle completes the route the applied plans are
again exactly the base controller's own decisions.

In the saved experiment (start cycle 40, seeds 0 to 4) the corridor shortens the emergency vehicle's travel time in every scenario and
controller combination reported in [docs/emergency_corridor.md](docs/emergency_corridor.md); for example, with the Fixed controller in
`ns_heavy` the mean falls from 242.0 s to 142.0 s. **It costs normal traffic**: total waiting rises by up to about 8% in the tested cases
(for example +7.33% for Fixed in `ns_heavy`), and the increase appears in all five seeds in several of them.

## Environmental impact proxy

Fuel and CO2 are a **simulation proxy** (a simulation estimate, not a measurement) derived only from simulated waiting vehicle-seconds:

```
fuel_liters = waiting_vehicle_seconds / 3600 x idle_fuel_rate_lph          (default 0.6 L per vehicle-hour)
co2_kg      = fuel_liters x emission_factor_kg_per_liter                    (default 2.3477 kg CO2 per litre, gasoline)
```

The idle rate is a configurable scenario parameter based on published passenger-car idling examples, not a calibrated fleet average;
the emission factor is derived from a U.S. EPA figure (sources and provenance in the docs). Both formulas are linear, so a percentage
change in waiting time is the same percentage change in the estimate. These values are **not real-world measurements**, and the
project does not claim any real-world emission reduction. See [docs/metrics_and_environmental_model.md](docs/metrics_and_environmental_model.md).

## Featherless AI analyst

Optional. The dashboard's AI Traffic Analyst explains measured results in words. It is configured only through environment variables
(`FEATHERLESS_API_KEY`, `FEATHERLESS_MODEL`, and optionally `FEATHERLESS_BASE_URL`, default `https://api.featherless.ai/v1`); no default
model is assumed. Without a key or model, or on any error, it shows a deterministic **LOCAL FALLBACK** explanation built from the same
numbers, so the demo never needs a key. Only a small whitelisted context of numbers is sent (no paths, keys or logs), and model text is
validated before it is shown. The AI layer explains validated simulation results; it does not select traffic signals, optimize the QUBO,
operate the emergency corridor or control the simulator. See [docs/featherless_integration.md](docs/featherless_integration.md).

## Dashboard

`streamlit run app.py` starts a dark control-room dashboard:

- Control Center: scenario, controller, seed, cycles, emergency on/off, route, start cycle, Run / Run emergency / Reset.
- Header status (SYSTEM READY / SIMULATION RUNNING / EMERGENCY ACTIVE) and KPI strip (waiting, throughput, queues, CO2 proxy vs Fixed).
- Live network schematic (queue intensity, signal phases, emergency priority, EV position), playback, intersection inspector and
  signal state, queue and signal-plan timelines.
- Emergency panel, quantum-optimization panel (saved Phase 4 QAOA metrics), environmental panel, controller comparison from the saved
  results, AI analyst, and a system-architecture view.

The live run is computed by the simulator; the comparison, QAOA and sensitivity panels read the saved results. Nothing is hard-coded.

## Experimental results

Source: `results/phase6/controller_summary.csv` (means over seeds 0 to 4, 60 cycles, identical demand and start state per scenario). The
QAOA p=1 column is the single saved seed-0 run, so it is **not directly comparable** with the five-seed means. Percentages are relative to
Fixed; for the QAOA p=1 column they are relative to Fixed on the same seed 0 (Fixed itself is shown as the five-seed mean). Each metric is a separate table; **no overall ranking or composite score is given, and none is implied.** The outcomes are mixed:
for example Adaptive is worse than Fixed in `balanced_medium`, QUBO Static is worse than Fixed in `time_varying`, and the saved QAOA p=1
plan is worse than Fixed in `balanced_medium` and `time_varying`.

**Waiting time, seconds per admitted vehicle (lower is better; % vs Fixed)**

| scenario | Fixed | Adaptive | QUBO Static | QUBO Receding | QAOA p=1 (seed 0 only) |
|---|---|---|---|---|---|
| balanced_medium | 72.0 | 76.5 (+6.3%) | 72.0 (+0.0%) | 72.0 (+0.0%) | 106.9 (+48.7%) |
| ns_heavy | 144.8 | 97.5 (-32.6%) | 73.6 (-49.2%) | 83.2 (-42.5%) | 73.8 (-49.0%) |
| ew_heavy | 140.8 | 131.4 (-6.7%) | 86.2 (-38.8%) | 87.2 (-38.1%) | 135.8 (-3.5%) |
| time_varying | 103.4 | 99.5 (-3.8%) | 140.0 (+35.4%) | 83.1 (-19.6%) | 140.6 (+34.0%) |
| downstream_congested | 189.0 | 132.7 (-29.8%) | 180.4 (-4.6%) | 93.2 (-50.7%) | 96.4 (-48.9%) |

**Throughput, vehicles per hour (higher is better; % vs Fixed)**

| scenario | Fixed | Adaptive | QUBO Static | QUBO Receding | QAOA p=1 (seed 0 only) |
|---|---|---|---|---|---|
| balanced_medium | 7,195 | 7,187 (-0.1%) | 7,195 (+0.0%) | 7,195 (+0.0%) | 6,987 (-2.9%) |
| ns_heavy | 6,835 | 7,967 (+16.6%) | 8,035 (+17.6%) | 8,008 (+17.2%) | 8,023 (+17.3%) |
| ew_heavy | 5,763 | 6,409 (+11.2%) | 6,529 (+13.3%) | 6,528 (+13.3%) | 6,137 (+6.5%) |
| time_varying | 6,614 | 6,844 (+3.5%) | 5,816 (-12.1%) | 6,935 (+4.8%) | 5,818 (-12.2%) |
| downstream_congested | 5,668 | 6,325 (+11.6%) | 5,752 (+1.5%) | 6,467 (+14.1%) | 6,501 (+14.7%) |

**Average queue, vehicles per approach (lower is better; % vs Fixed)**

| scenario | Fixed | Adaptive | QUBO Static | QUBO Receding | QAOA p=1 (seed 0 only) |
|---|---|---|---|---|---|
| balanced_medium | 5.99 | 6.37 (+6.3%) | 5.99 (+0.0%) | 5.99 (+0.0%) | 8.77 (+46.2%) |
| ns_heavy | 11.45 | 8.92 (-22.2%) | 6.73 (-41.3%) | 7.61 (-33.6%) | 6.75 (-41.1%) |
| ew_heavy | 9.39 | 9.78 (+4.2%) | 6.44 (-31.4%) | 6.52 (-30.6%) | 9.68 (+3.1%) |
| time_varying | 7.88 | 7.86 (-0.2%) | 9.34 (+18.5%) | 6.57 (-16.6%) | 9.38 (+17.1%) |
| downstream_congested | 12.70 | 9.82 (-22.6%) | 12.33 (-2.9%) | 6.95 (-45.2%) | 7.23 (-43.0%) |

**Maximum queue, vehicles (road capacity is 40)**

| scenario | Fixed | Adaptive | QUBO Static | QUBO Receding | QAOA p=1 (seed 0 only) |
|---|---|---|---|---|---|
| balanced_medium | 15.0 | 19.7 | 15.0 | 15.0 | 40.0 |
| ns_heavy | 40.0 | 40.0 | 40.0 | 40.0 | 40.0 |
| ew_heavy | 40.0 | 40.0 | 40.0 | 40.0 | 40.0 |
| time_varying | 40.0 | 40.0 | 40.0 | 39.5 | 40.0 |
| downstream_congested | 40.0 | 40.0 | 40.0 | 40.0 | 40.0 |

**CO2 simulation proxy, kg per 60-cycle run (not a measurement; % vs Fixed)**

| scenario | Fixed | Adaptive | QUBO Static | QUBO Receding | QAOA p=1 (seed 0 only) |
|---|---|---|---|---|---|
| balanced_medium | 202.6 | 215.4 (+6.3%) | 202.6 (+0.0%) | 202.6 (+0.0%) | 296.4 (+46.2%) |
| ns_heavy | 387.2 | 301.4 (-22.2%) | 227.4 (-41.3%) | 257.3 (-33.6%) | 228.1 (-41.1%) |
| ew_heavy | 317.5 | 330.7 (+4.2%) | 217.8 (-31.4%) | 220.3 (-30.6%) | 327.2 (+3.1%) |
| time_varying | 266.3 | 265.7 (-0.2%) | 315.7 (+18.5%) | 222.2 (-16.6%) | 317.2 (+17.1%) |
| downstream_congested | 429.3 | 332.1 (-22.6%) | 416.8 (-2.9%) | 235.1 (-45.2%) | 244.3 (-43.0%) |

The maximum queue is 40 (the road capacity) in almost every case, so it does not separate the controllers. The CO2 simulation proxy is linear in
waiting vehicle-seconds; it is not a measurement. QAOA p=1 quality versus the exact optimum is tabulated in
[docs/optimization.md](docs/optimization.md); emergency results in [docs/emergency_corridor.md](docs/emergency_corridor.md).

## Reproducibility

Setup (Python 3.12; commands validated on Windows 11 with PowerShell, see [docs/reproducibility.md](docs/reproducibility.md) for what was and was not validated):

```powershell
git clone https://github.com/naresh4687/quantum-traffic-optimization.git
cd quantum-traffic-optimization
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
streamlit run app.py
```

`python -m pytest -q` reports 452 passed. `pip install -e .` is required in addition to `requirements.txt`. All saved Phase 2 to 6 results
regenerate exactly from the scripts in `scripts/` (timing fields aside), and no experiment needs a network, a Featherless key or special
hardware. The full command list, timings and notes are in [docs/reproducibility.md](docs/reproducibility.md).

## Demo

Live demo preset: NS-heavy, Adaptive, seed 0, 60 cycles, emergency ON, route I1 → I2 → I3 → I6, **start cycle 30**. Open
`http://localhost:8501/?scenario=ns_heavy&controller=adaptive&seed=0&cycles=60&emergency=1&route=I1,I2,I3,I6&start=30`
after `streamlit run app.py`. This is a live simulator run, **not** the saved experiment: the saved emergency results use start cycle
40 and average five seeds, and the dashboard's default (and Reset) uses start cycle 40 so it can be checked against them. The preset is
a demonstration, not a claim of optimality. See [docs/demo_preset.md](docs/demo_preset.md).

## Project structure

```
app.py                     Streamlit entry point (thin)
src/qtraffic/              simulator, controllers, emergency corridor, metrics, environment model
  controllers/             Fixed and Adaptive controllers, controller interface
  optimization/            QUBO, exact solver, QAOA, adapters, validation
  ai/                      Featherless integration and deterministic fallback
  dashboard/               dashboard logic, charts, theme
scripts/                   experiment, audit and demo scripts
tests/                     452 tests
results/                   saved experiment results (phase2 - phase6) and the dashboard test count
docs/                      architecture, optimization, emergency, metrics, Featherless, reproducibility, demo preset
.streamlit/config.toml     dashboard theme
.env.example               Featherless variables (empty key)
requirements.txt, pyproject.toml
```

## Limitations

- A simulation, not a real road: queue-based fluid model, no turning movements, no yellow or all-red time, no pedestrians or
  vehicle types; results are not field measurements.
- 6 intersections and 5 synthetic scenarios; 5 seeds for the comparisons and a single seed for QAOA.
- The QUBO is a two-cycle surrogate; static QUBO plans can be poor when demand changes (`time_varying`), and tied optima can decode to
  plans with different simulator results (`balanced_medium`).
- QAOA ran only on a classical simulator with 18 qubits; the cold-start p=2 optimizer stops at its iteration limit, and p=1 misses the
  optimum in `ew_heavy`. Exact enumeration of 729 assignments is trivial at this size, so no advantage is possible or claimed.
- Adaptive control oscillates under balanced demand and can be worse than Fixed there.
- The emergency corridor uses modelling assumptions (massless vehicle, fixed 30 s links) and increases normal-traffic waiting.
- Fuel and CO2 are waiting-only simulation estimates with configurable, documented coefficients.
- The AI analyst is optional and only explains; its text is not a fact-check of the numbers, which are shown separately.
- The saved QAOA plans exist for seed 0 only, so the QAOA p=1 controller cannot run for other seeds.

## Technology stack

Python 3.12; NumPy, NetworkX, SciPy; Qiskit and Qiskit Aer; Streamlit and Plotly (pandas as a Streamlit dependency); the OpenAI-compatible
`openai` SDK for the optional Featherless integration; pytest.

## Testing

`python -m pytest -q`: **452 tests passed**, none skipped, 6 warnings (Qiskit / SciPy deprecation and efficiency notices). The suite
covers the simulator, controllers, QUBO (including exhaustive penalty checks), QAOA, the emergency corridor, metrics and the proxy,
the Featherless layer (with a fake client and the network blocked) and the dashboard logic.

## Team

Team: QUAN115.

Contributor and member details can be added here later if needed.
