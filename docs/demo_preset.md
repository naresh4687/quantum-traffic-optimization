# Demo preset (dashboard)

One reproducible configuration for demonstrations. It is a demo preset only: it is **not** claimed to be optimal, and the numbers
below belong to this exact configuration (deterministic: the same inputs always give the same values).

| setting | value |
|---|---|
| scenario | NS-heavy (`ns_heavy`) |
| controller | Adaptive |
| seed | 0 |
| simulation | 60 cycles (simulator cycles 30-89) |
| emergency | ON, EV1, route I1 > I2 > I3 > I6, start cycle 30 |

Launch (no API key needed; the AI analyst uses its LOCAL FALLBACK):

```
streamlit run app.py
```

then open `http://localhost:8501/?scenario=ns_heavy&controller=adaptive&seed=0&cycles=60&emergency=1&route=I1,I2,I3,I6&start=30`,
or set the same values in the Control Center and press **Run simulation**. **Reset** returns to the default demo, which is the same
configuration with the emergency starting at cycle 40 (the start cycle of the saved Phase 5/6 primary emergency result).

Measured values for this preset (live simulator run; percentages compare with the Fixed controller on the same demand and the same emergency):

- waiting 101.0 s per admitted vehicle (-36.0% vs Fixed), throughput 7,914 veh/h (+16.4%), average queue 9.2 vehicles (-26.4%),
  maximum queue 40.0 (equal to Fixed);
- CO2 proxy 310.4 kg (-26.4% vs Fixed): a waiting-based **simulation proxy**, not a measurement;
- EV1 travel time 142.0 s with the corridor vs 242.0 s without it (free flow 90 s), 3 signal stops vs 4, completed in cycle 32,
  4 intersections granted priority;
- cost to normal traffic: total waiting +4.50% and maximum queue unchanged (40.0) relative to the same run without the override.

Cycle 30 is the first simulated cycle, so the emergency vehicle enters as the run starts; the dashboard parks the playhead at cycle 31
(corridor active). Scrubbing or playing to a later cycle shows CORRIDOR COMPLETE and "Normal controller restored".
The dashboard states "no saved counterpart" for this preset because only the start-cycle-40 emergency configuration is stored in the
saved Phase 5/6 results.
