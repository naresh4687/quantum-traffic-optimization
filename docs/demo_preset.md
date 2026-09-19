# Demo preset (dashboard)

One reproducible configuration for demonstrations. It is a demo preset only: it is **not** claimed to be optimal, and the numbers
below belong to this exact configuration (deterministic: the same inputs always give the same values).

## Live demo preset

| setting | value |
|---|---|
| scenario | NS-heavy (`ns_heavy`) |
| controller | Adaptive |
| seed | 0 |
| simulation | 60 cycles (simulator cycles 30-89) |
| emergency | ON, EV1, route I1 → I2 → I3 → I6, **start cycle 30** |

Launch (Python 3.12 environment from `docs/reproducibility.md`; no API key needed, the AI analyst uses its LOCAL FALLBACK):

```
streamlit run app.py
```

then open
`http://localhost:8501/?scenario=ns_heavy&controller=adaptive&seed=0&cycles=60&emergency=1&route=I1,I2,I3,I6&start=30`
(use the port Streamlit prints if it is not 8501), or set the same values in the Control Center and press **Run simulation**.

Measured values for this preset (a **live simulator run**; percentages compare with the Fixed controller on the same demand and the same
emergency):

- waiting 101.0 s per admitted vehicle (-36.0% vs Fixed), throughput 7,914 veh/h (+16.4%), average queue 9.2 vehicles (-26.4%),
  maximum queue 40.0 (equal to Fixed);
- CO2 proxy 310.4 kg (-26.4% vs Fixed): a waiting-based **simulation estimate**, not a measurement;
- EV1 travel time 142.0 s with the corridor vs 242.0 s without it (free flow 90 s), 3 signal stops vs 4, completed in cycle 32,
  4 intersections granted priority;
- cost to normal traffic: total waiting +4.50% and maximum queue unchanged (40.0) relative to the same run without the override.

Cycle 30 is the first simulated cycle, so the emergency vehicle enters as the run starts; the dashboard parks the playhead at cycle 31
(corridor active). Scrubbing or playing to a later cycle shows CORRIDOR COMPLETE and "Normal controller restored".

## Saved experiment results are a different experiment

The saved Phase 5 and Phase 6 emergency results (`results/phase5/`, `results/phase6/`) use **start cycle 40**, not 30, and average
seeds 0 to 4.

| | live demo preset | saved experiment results |
|---|---|---|
| emergency start cycle | 30 | 40 |
| seeds | 0 | 0-4 (mean over five seeds) |
| origin | computed live when the dashboard runs | files written by `scripts/run_phase5_emergency.py` and `scripts/run_phase6_metrics.py` |
| dashboard label | "no saved counterpart for this configuration" | "reproduces the saved Phase 5/6 emergency result" |

The dashboard's default configuration (and **Reset**) is the same scenario, controller, seed and route with the emergency starting at
**cycle 40**, so the live run can be checked against the saved seed-0 result: it gives EV1 142.0 s with the corridor and 242.0 s without,
matching the saved file. The two configurations must not be presented as the same experiment. The controller-comparison, QAOA and
sensitivity panels always show the saved experiment results (means over seeds 0-4, QAOA seed 0 only), never the live preset run.
