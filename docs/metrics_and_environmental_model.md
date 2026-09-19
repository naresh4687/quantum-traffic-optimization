# Metrics and the environmental proxy (Phase 6)

Everything below is computed from the existing simulator (`metrics.py`, `simulator.py`). Nothing is
re-defined: `qtraffic.unified_metrics` regroups `compute_metrics` and adds signal, emergency and
per-vehicle views. The environmental figures are a **simulation proxy**, not a measurement.

## Traffic metric definitions (units: vehicles, seconds)

| Term | Meaning |
|---|---|
| demanded | external vehicles that tried to enter at boundary approaches |
| admitted (`vehicles_entered`) | the part of demand that fit on the road; `demanded = admitted + rejected` |
| rejected | demand turned away because the entry road was full (never entered the network) |
| served (stop line) | discharges through stop lines; a vehicle crossing 3 intersections is served 3 times, so `served >= exited` |
| exited | distinct vehicles that left the network past a boundary stop line |
| throughput | `exited / (cycles * 60 / 3600)` vehicles per simulated hour |
| in transit | released by an upstream stop line, travelling to the next queue: not queued, not waiting, not exited |
| waiting (vehicle-seconds) | per approach and cycle `(q0 + q1)/2 * 60`; transit is not waiting |
| average / max / final queue | time-averaged queue per approach / largest `q0` of any approach in any cycle / vehicles queued after the last cycle |
| blocked (vehicle-cycles) | vehicles whose discharge was withheld by a full road ahead, summed over cycles |

Exact conservation law (any start state): `queued_at_start + in_transit_at_start + admitted = exited + final_queue + in_transit_final`.
`served` is not in it, and `rejected` is outside it.

`waiting_seconds_per_admitted` = window waiting / window admitted. For a run that starts from a non-empty
state this is a per-admitted normalisation, not an average over distinct vehicles.

## Environmental proxy (SIMULATION PROXY, waiting/idling only)

```
fuel_liters = waiting_vehicle_seconds / 3600 * idle_fuel_rate_lph
co2_kg      = fuel_liters * emission_factor_kg_per_liter
```

Both formulas are linear, so a change of x% in waiting time is a change of x% in fuel and CO2 under any
coefficients. The proxy adds no evidence beyond the waiting-time result; the coefficients set the absolute
scale only.

| Coefficient | Default | Status |
|---|---|---|
| `emission_factor_kg_per_liter` | 2.3477 (gasoline, exactly 2.347697); diesel 2.6893 | **Source-derived.** U.S. EPA, *Greenhouse Gas Emissions from a Typical Passenger Vehicle*, https://www.epa.gov/greenvehicles/greenhouse-gas-emissions-typical-passenger-vehicle (page read for this project; shows "last updated June 3, 2026"): 8,887 g CO2 per US gallon of gasoline, 10,180 g per gallon of diesel; converted at 3.785411784 L per gallon. Tailpipe combustion CO2 only. |
| `idle_fuel_rate_lph` | 0.6 L per vehicle-hour (0.16 US gal/h x 3.78541 L/gal = 0.606 L/h, rounded) | **Configurable simulation parameter, not a universal vehicle constant.** Based on published passenger-vehicle idling examples (below). Not a calibrated fleet-average value. |

The environmental model uses a configurable waiting/idling fuel-rate proxy. The default is based on published
passenger-vehicle idling examples and is not a calibrated fleet-average value.

### Idle fuel rate: published examples

Argonne National Laboratory, *Idling Reduction Savings Calculator*,
https://www.anl.gov/sites/www/files/2018-02/idling_worksheet.pdf. Passenger cars, no load:

| Vehicle | US gal/h | L/h |
|---|---|---|
| Ford Focus (gasoline) | 0.16 | 0.61 |
| Volkswagen Jetta (diesel) | 0.17 | 0.64 |
| Ford Crown Victoria (gasoline) | 0.39 | 1.48 |

The U.S. Department of Energy gives the broader general statement that idling can use roughly 0.25 to 0.5 gal/h
(0.95 to 1.9 L/h). The default (0.6 L/h) is the lowest example, so the proxy does not overstate fuel use.
The sensitivity values 0.6, 0.9, 1.2 and 1.9 L/h (about 0.16 to 0.5 gal/h) are scenario parameters spanning
these examples and the DOE range, not vehicle characteristics.

**Provenance of the idle-rate figures.** The worksheet's title, publisher and URL were confirmed, and the figures
were supplied to this project by its owner from that worksheet. The tooling used to build the project could not
retrieve the PDF text (HTTP 403), so the figures were not independently re-read there. The DOE 0.25 to 0.5 gal/h
statement is likewise a general statement supplied to the project, with no specific page cited.

## Not modelled

Acceleration/deceleration, vehicle speed, vehicle type, engine efficiency, road gradient, temperature,
congestion-dependent fuel burn, actual vehicle trajectories, exhaust composition, NOx/PM emissions, and fuel
used while moving or in transit. The proxy estimates only a waiting/idling-related quantity.

## How to state results

Say: "Under the configured waiting-based simulation proxy, estimated CO2 changed by X%."
Do not say that the system reduced real-world CO2. Report both total and per-admitted values: a controller
that admits more vehicles can have more total waiting yet less waiting per admitted vehicle.
