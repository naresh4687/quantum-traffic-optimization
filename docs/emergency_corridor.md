# Emergency Green Corridor

The corridor is implemented in `src/qtraffic/emergency.py` beside the simulator, which is unchanged. It has two parts: a discrete
emergency vehicle (EV) and a signal-priority override that wraps whichever controller is running.

## Route and timing

The documented route is **I1 → I2 → I3 → I6** (east through I1, I2, I3, then south to I6). The EV enters at its start cycle and
completes when it clears the stop line of the last intersection.

| assumption | value |
|---|---|
| clock | cycle `c` spans `[60c, 60(c+1))` seconds, the simulator's own clock |
| link time (free flow) | 30 s per road, so the free-flow route time is 3 x 30 = **90 s** |
| phase order inside a cycle | NS first, then EW; under priority the EV's phase runs first at that intersection |
| queueing | the EV joins the tail of its approach queue and leaves when the vehicles ahead have been served |
| mass | the EV is a single vehicle and is not added to the fluid queues: it takes no capacity and delays no one directly |
| turns | the simulator has no turning movements; the EV uses the phase of the approach it arrives on and may turn onto the next road |

These are modelling conventions needed to give a discrete vehicle a timeline in a per-cycle fluid model. The EV's travel time is the result
of simulating it against the plans that were actually applied, never a constant.

## How the override works

`EmergencyOverrideController` wraps any base controller (Fixed, Adaptive, QUBO, a static QAOA plan):

```
base controller plan  ->  emergency override  ->  final plan applied by the simulator
```

1. The base controller always runs first and chooses its plans.
2. For every route intersection the EV has not yet passed whose free-flow arrival time falls before the end of the cycle, the override
   grants **priority**: the valid plan that gives the EV's phase the most green (40 s) is applied and that phase runs first.
   The conflicting phase is cut to 20 s only if the base plan gave the EV's phase less than 40 s.
3. An intersection is released the cycle after the EV crosses it, so the corridor moves along the route hop by hop.
4. Before the EV enters and after it completes, the final plan **is** the base plan.

## Normal controller preservation and restoration

The base controller runs unmodified every cycle, and the override only changes intersections on the route that the EV is about to reach.
After completion no priority remains and the applied plans equal the base controller's own decisions on the live state (checked by the
tests and by an independent audit for Adaptive and Fixed). The dashboard shows "CORRIDOR COMPLETE" and "Normal controller restored" only when the
simulated state actually satisfies this.

## Results and the cost to normal traffic

The primary saved experiment (`results/phase5/`, `results/phase6/`) runs each configuration twice on identical demand, with and
without the override: route I1 → I2 → I3 → I6, **start cycle 40**, 60 cycles (simulator cycles 30 to 89), seeds 0 to 4.

Mean over seeds (`results/phase5/emergency_summary.csv`):

| scenario | base controller | EV travel time, no override -> corridor | signal stops | normal traffic: total waiting, corridor vs no override |
|---|---|---|---|---|
| balanced_medium | Fixed | 207.3 s -> 141.5 s | 3.4 -> 3.0 | +0.35% |
| balanced_medium | Adaptive | 248.6 s -> 153.5 s | 3.8 -> 3.6 | +0.23% |
| ns_heavy | Fixed | 242.0 s -> 142.0 s | 4.0 -> 3.0 | +7.33% |
| ns_heavy | Adaptive | 250.0 s -> 146.0 s | 4.0 -> 3.0 | +3.58% |
| ew_heavy | Fixed | 433.2 s -> 302.0 s | 4.0 -> 4.0 | +8.28% |
| ew_heavy | Adaptive | 366.3 s -> 266.0 s | 4.0 -> 3.6 | -0.34% |

The corridor shortens the EV's travel time in every row. It is **not free for normal traffic**: total waiting rises in most rows, by up to
about 8%, and the increase is present in every seed for several of them. This normal-traffic impact is reported, not hidden.
For the single dashboard demo run (ns_heavy, Adaptive, seed 0, start cycle 40) the EV needs 142.0 s with the corridor and 242.0 s without.
The corridor concentrates green on one route at a time; it makes no claim about optimal emergency routing.

## Limits of the model

Fluid queues, no turning movements or conflicts, no yellow or all-red time, a massless EV, fixed 30 s links, and a single emergency
vehicle at a time. Results are simulation outcomes, not field measurements.
