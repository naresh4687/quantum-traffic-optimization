# QUBO formulation and QAOA implementation

This page summarizes the implementation in `src/qtraffic/optimization/`; the module docstrings (`qubo.py`, `qaoa.py`) are the detailed
source of truth. Nothing here claims a quantum advantage or a quantum speedup, and no quantum hardware was used.

## The decision problem

Every 60 s cycle, each of the 6 intersections chooses one of three signal plans:

| plan | NS green | EW green |
|---|---|---|
| A | 20 s | 40 s |
| B | 30 s | 30 s |
| C | 40 s | 20 s |

## QUBO formulation

- **Variables.** `x[i,p] = 1` means intersection `i` runs plan `p`. 6 intersections x 3 plans = **18 binary variables** (index `3*i + p`).
- **Search space.** There are 2^18 = **262,144** bit strings, but only 3^6 = **729** are valid: exactly one plan per intersection (one-hot).
- **Objective.** `E(x) = E_traffic(x) + E_penalty(x)`.
- **One-hot penalty.** `P * (x[i,A] + x[i,B] + x[i,C] - 1)^2` per intersection: exactly 0 for a valid choice. The penalty is
  `P = B + 1`, where `B` is the largest total absolute traffic coefficient on any single bit. A proof in the module docstring shows that
  with `P > B` no infeasible string can be a global minimum or tie a feasible optimum; it is checked exhaustively over all 2^18 strings
  (`check_repair_lemma`, `check_full_space`, `scripts/audit_qubo.py`). It is a deterministic bound and is deliberately not tuned.
  A large `P` stretches the energy range, which is unfavourable for heuristic solvers such as QAOA.
- **Traffic objective.** A **two-cycle look-ahead surrogate** built from the simulator's own rules, in vehicle-seconds. Cycle 1 charges
  each approach its waiting time while it is served; cycle 2 charges the waiting of the vehicles that were released downstream or left
  in the queue. The plan chosen for cycle 1 is held for cycle 2. There are no fitted weights.
- **Neighbour / downstream coupling.** How many vehicles an approach can release depends on the free space on the road ahead, which
  depends on the plan of the *next* intersection. Every term therefore involves at most two intersections joined by a road, so the
  traffic objective is a sum of unary tables per intersection and pairwise tables per neighbouring pair.
- **Inputs.** The QUBO is built from what the classical controllers also see: queues and in-transit vehicles (no demand forecast).
- **The QUBO energy is a surrogate, not a waiting time.** The simulator remains the source of truth for traffic performance.

### Exact solver (ground truth)

`solve_exact` enumerates all 729 valid assignments and returns the minimum. Ties are broken towards the fewest departures from the
neutral plan B. Exact enumeration is the ground truth used to judge every QAOA run. Nothing here is quantum: it is classical
optimization of a QUBO.

### Controllers built on the QUBO

- **QUBO Static** solves the exact QUBO once from the initial traffic state and holds the selected plan for the whole run.
- **QUBO Receding** rebuilds and re-solves the exact QUBO from the observed state at the start of every cycle.
- **QAOA p=1** replays the plan chosen by the saved Phase 4 QAOA run (see below) and holds it. It is only available where a saved
  plan exists (seed 0 of each scenario).

### Known limitation: short horizon

The surrogate looks two cycles ahead, ignores blocking in the second cycle and knows no future demand. A static plan that is optimal
for the initial state can be a poor plan later when demand changes. This is visible in the saved results (for example, QUBO Static is
worse than Fixed in the time-varying scenario), and the receding variant, which re-solves every cycle, is the mitigation. The
formulation was **not** changed to improve any single scenario. An earlier variant that assumed the neutral plan in cycle 2 is kept as an
ablation in `results/phase3_ablation_reference_plan/`.

## QAOA implementation

Pipeline: QUBO -> Ising Hamiltonian -> parameterised QAOA circuit -> Qiskit Aer statevector simulation -> sampled bit strings ->
QUBO energy and one-hot check -> comparison with the exact optimum.

- **QUBO to Ising.** Substituting `x = (1 - z)/2` gives `E = c + sum_u h_u z_u + sum_{u<v} J_uv z_u z_v`. The identity holds on every
  bit string. The constant `c` is dropped from the circuit and added back when energies are reported; the Hamiltonian is scaled by
  its largest coefficient, which does not change the energy ordering. QUBO and Ising energies agree to about 1e-10 (tested).
- **Qubits.** 18 qubits, one per QUBO variable. `p` is the QAOA **depth** (number of cost/mixer layers), not the number of qubits.
- **Circuit.** `|+>` on every qubit, then for each layer an `RZ`/`RZZ` cost layer and an `RX` mixer layer.
  Logical depth 30 for p=1 and 44 for p=2.
- **Simulator.** `AerSimulator(method='statevector')` from Qiskit Aer, a **classical** simulator of quantum circuits.
- **Parameters.** Optimized classically with SciPy COBYLA (max 200 evaluations, initial step 0.2, tolerance 1e-4) on the expectation
  value of the Hamiltonian. The start angles are deterministic (a Trotterized-annealing ramp). The optimizer never sees the exact optimum.
- **Sampling.** 8192 shots with a fixed seed (0), so runs are deterministic. Re-running the Phase 4 experiment reproduces the saved
  counts exactly.
- **Result.** The best feasible (one-hot) sampled string is decoded into six plans. Every sample is scored by the existing QUBO
  evaluator. Only in evaluation (`compare_with_exact`) is the QAOA result compared with the exact optimum.
- **Configuration.** One configuration is used for every scenario; only `p` differs. Nothing is tuned per scenario.

### Saved results (Phase 4, seed 0, `results/phase4/`)

| scenario | variant | best feasible sample = exact optimum | energy gap | feasible-sample rate | COBYLA |
|---|---|---|---|---|---|
| balanced_medium | p=1 | yes | 0.0 | 13.4% | reported success |
| balanced_medium | p=2 (cold start) | yes | 0.0 | 11.1% | stopped at 200 evaluations |
| balanced_medium | p=2 warm start (diagnostic) | yes | 0.0 | 29.3% | stopped at 200 evaluations |
| ns_heavy | p=1 | yes | 0.0 | 12.2% | reported success |
| ns_heavy | p=2 (cold start) | no | 300.0 | 6.6% | stopped at 200 evaluations |
| ns_heavy | p=2 warm start (diagnostic) | yes | 0.0 | 12.1% | reported success |
| ew_heavy | p=1 | no | 150.0 | 13.4% | reported success |
| ew_heavy | p=2 (cold start) | yes | 0.0 | 15.1% | stopped at 200 evaluations |
| ew_heavy | p=2 warm start (diagnostic) | yes | 0.0 | 36.3% | stopped at 200 evaluations |
| time_varying | p=1 | yes | 0.0 | 12.2% | reported success |
| time_varying | p=2 (cold start) | no | 300.0 | 6.6% | stopped at 200 evaluations |
| time_varying | p=2 warm start (diagnostic) | yes | 0.0 | 12.1% | reported success |
| downstream_congested | p=1 | yes | 0.0 | 13.3% | reported success |
| downstream_congested | p=2 (cold start) | yes | 0.0 | 12.3% | stopped at 200 evaluations |
| downstream_congested | p=2 warm start (diagnostic) | yes | 0.0 | 34.6% | stopped at 200 evaluations |

For scale, uniform random sampling of 18 bits gives a valid (one-hot) string with probability 729 / 262,144 = 0.28%.

### Limitations that stay documented

- **p=2 cold start.** The cold-start p=2 optimization stops at its 200-evaluation limit in every scenario, and in `ns_heavy` and
  `time_varying` its best feasible sample is not the exact optimum (energy gap 300.0). Deeper QAOA is not better here: this is an
  optimizer convergence limitation, not evidence about QAOA in general.
- **p=1 in EW-heavy.** In `ew_heavy`, p=1 does not find the exact optimum (gap 150.0).
- **Warm start.** The warm-start p=2 run (started from the p=1 optimum with a zero second layer) is a separate **diagnostic** experiment
  that separates "p=2 cannot help" from "the optimizer got stuck". It is reported next to, never instead of, the primary p=2 result, and
  it is not shown in the dashboard.
- **Tied optima.** A QUBO can have several assignments with exactly the same energy (`balanced_medium` has 2, `downstream_congested` has 4).
  The exact solver breaks ties towards the neutral plan; a QAOA sample can be a different tied optimum. Equal QUBO energy does not mean
  equal simulator results over a longer horizon: in `balanced_medium` the saved QAOA p=1 plan is an optimal-energy assignment but
  gives 106.9 s waiting per admitted vehicle on seed 0, against 71.9 s for Fixed, QUBO Static and QUBO Receding on the same seed
  (`results/phase6/controller_comparison.csv`).
- **One seed.** QAOA was run once per scenario (seed 0), so the QAOA p=1 controller has a single saved plan per scenario.
- **Classical simulation.** QAOA was evaluated on a classical quantum-circuit simulator (Qiskit Aer). The experiment measures solution
  quality and sampling behaviour. It does not measure, and the project does not claim, any quantum hardware execution, speedup or
  advantage. Exact classical enumeration of 729 assignments is trivial at this size.

Related audits: `scripts/audit_qubo.py` (QUBO/penalty audit for all 25 scenario-seed states) and `scripts/audit_qaoa.py` (Ising
equivalence, bit order, "no exact optimum reaches the optimizer", probabilities and ties).
