r"""Unified metrics: traffic + signal + emergency + environmental proxy, from one simulator history.

Nothing is redefined here. Traffic numbers come from ``metrics.compute_metrics`` (the single
authoritative definition); this module only groups them, adds the signal and emergency views, and
attaches the waiting-based environmental PROXY from ``environment.py``.

The distinctions that must stay clear (all in vehicles unless stated)
--------------------------------------------------------------------
demanded    external vehicles that TRIED to enter at boundary approaches.
admitted    (``vehicles_entered``) the part of demand that fit on the road; the rest is
            ``vehicles_rejected``. demanded = admitted + rejected.
served      stop-line discharges. A vehicle that crosses three intersections is served three times,
            so served >= exited and served is not a count of distinct vehicles.
exited      vehicles that left the network past a boundary stop line; distinct vehicles.
throughput  exited vehicles per simulated hour (``exited / (cycles * 60 / 3600)``).
in transit  released by an upstream stop line and travelling to the next queue; not queued, not
            waiting, not yet exited.
waiting     vehicle-seconds spent in queues: per approach and cycle ``(q0 + q1) / 2 * 60`` where ``q0``
            is the queue after arrivals and ``q1`` after service. Transit is not waiting.
queue       ``average_queue`` = time-averaged queue per approach (occupancy, in vehicles);
            ``max_queue`` = largest ``q0`` of any approach in any cycle; ``final_queue`` = vehicles
            queued after the last cycle.
blocked     ``blocked_vehicle_cycles``: vehicles whose discharge was withheld by a full road ahead,
            summed over cycles (a vehicle blocked for 3 cycles counts 3 times).

The one conservation law that holds exactly (checked in tests), for any start state::

    queued_at_start + in_transit_at_start + admitted = exited + final_queue + in_transit_final

``served`` does not appear in it (it counts one vehicle once per intersection crossed), and
``rejected`` is outside it (rejected vehicles never entered).

``waiting_seconds_per_admitted`` divides the window's waiting by the window's admitted vehicles. When a
run starts from a non-empty state, vehicles that were already queued also wait, so it is a per-admitted
normalisation, not an average over distinct vehicles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from .emergency import EmergencyRun, EmergencyVehicle
from .environment import EnvironmentalConfig, EnvironmentalEstimate, estimate, percent_change
from .metrics import CycleRecord, Metrics, compute_metrics
from .signals import CYCLE_LENGTH, Phase
from .simulator import SimulationConfig

SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class TrafficMetrics:
    cycles: int
    simulated_hours: float
    vehicles_demanded: float
    vehicles_admitted: float
    vehicles_rejected: float
    vehicles_served_stop_line: float
    vehicles_exited: float
    throughput_vehicles_per_hour: float
    blocked_vehicle_cycles: float
    waiting_vehicle_seconds: float
    waiting_seconds_per_admitted: float
    waiting_minutes_per_admitted: float
    average_queue_vehicles: float
    max_queue_vehicles: float
    final_queue_vehicles: float
    in_transit_final_vehicles: float


@dataclass(frozen=True)
class SignalMetrics:
    node_cycles: int
    plan_changes: int  # node-cycles whose plan differs from the same node's previous cycle
    plan_change_rate: float  # plan_changes / (nodes * (cycles - 1))
    green_seconds_ns: float  # summed over intersections and cycles
    green_seconds_ew: float
    ns_green_share: float
    phase_utilization_ns: float | None  # served / (saturation_flow * green) over NS approaches
    phase_utilization_ew: float | None
    emergency_priority_cycles: int  # cycles in which the override held any intersection
    emergency_priority_node_cycles: int
    emergency_override_plan_changes: int  # node-cycles where the override altered the base plan


@dataclass(frozen=True)
class EmergencyMetrics:
    completed: bool
    entry_cycle: int | None
    completion_cycle: int | None
    travel_time_s: float | None
    free_flow_time_s: float
    delay_s: float | None
    stops: int
    prioritized_intersections: int


@dataclass(frozen=True)
class EnvironmentalMetrics:
    """PROXY values from waiting vehicle-seconds only (see ``environment.py``)."""

    estimate: EnvironmentalEstimate
    fuel_liters_per_admitted: float
    co2_kg_per_admitted: float

    @property
    def fuel_liters(self) -> float:
        return self.estimate.fuel_liters

    @property
    def co2_kg(self) -> float:
        return self.estimate.co2_kg


@dataclass(frozen=True)
class UnifiedMetrics:
    traffic: TrafficMetrics
    signal: SignalMetrics
    emergency: EmergencyMetrics | None
    environmental: EnvironmentalMetrics

    def flat(self, prefix: str = "") -> dict:
        """Flat column dict with explicit units in the names. Proxy columns carry ``_proxy``."""
        row = {**asdict(self.traffic), **asdict(self.signal)}
        if self.emergency is not None:
            row.update({f"emergency_{k}": v for k, v in asdict(self.emergency).items()})
        env = self.environmental
        row.update({"fuel_liters_proxy": env.fuel_liters, "co2_kg_proxy": env.co2_kg,
                    "fuel_liters_proxy_per_admitted": env.fuel_liters_per_admitted,
                    "co2_kg_proxy_per_admitted": env.co2_kg_per_admitted,
                    "idle_fuel_rate_lph_assumed": env.estimate.config.idle_fuel_rate_lph,
                    "emission_factor_kg_per_liter_assumed": env.estimate.config.emission_factor_kg_per_liter})
        return {f"{prefix}{k}": v for k, v in row.items()}


# -- pieces (small, independent) ------------------------------------------------------------------
def traffic_metrics(history: Sequence[CycleRecord]) -> TrafficMetrics:
    """Regroup ``compute_metrics`` and add the per-admitted normalisations."""
    m: Metrics = compute_metrics(history)
    admitted = m.vehicles_entered
    waiting = m.total_waiting_time
    per_admitted = waiting / admitted if admitted else 0.0
    return TrafficMetrics(
        cycles=m.cycles, simulated_hours=m.cycles * CYCLE_LENGTH / SECONDS_PER_HOUR,
        vehicles_demanded=m.demand_generated, vehicles_admitted=admitted,
        vehicles_rejected=m.vehicles_rejected, vehicles_served_stop_line=m.vehicles_served,
        vehicles_exited=m.vehicles_exited, throughput_vehicles_per_hour=m.throughput_per_hour,
        blocked_vehicle_cycles=m.total_blocked, waiting_vehicle_seconds=waiting,
        waiting_seconds_per_admitted=per_admitted, waiting_minutes_per_admitted=per_admitted / 60.0,
        average_queue_vehicles=m.average_queue_length, max_queue_vehicles=m.max_queue_length,
        final_queue_vehicles=m.final_queue_total, in_transit_final_vehicles=m.in_transit_final)


def signal_metrics(
    history: Sequence[CycleRecord],
    saturation_flow: float | None = None,
    emergency_run: EmergencyRun | None = None,
) -> SignalMetrics:
    """Plan changes, green allocation and phase utilisation from the plans that were actually applied."""
    s = SimulationConfig().saturation_flow if saturation_flow is None else saturation_flow
    nodes = list(history[0].plans)
    changes = sum(1 for prev, cur in zip(history, history[1:]) for n in nodes if cur.plans[n] != prev.plans[n])
    denom = len(nodes) * max(len(history) - 1, 0)
    green = {Phase.NS: 0.0, Phase.EW: 0.0}
    for rec in history:
        for n in nodes:
            green[Phase.NS] += rec.plans[n].ns_green
            green[Phase.EW] += rec.plans[n].ew_green
    served = {Phase.NS: 0.0, Phase.EW: 0.0}
    capacity = {Phase.NS: 0.0, Phase.EW: 0.0}
    for rec in history:
        for a, v in rec.served.items():
            served[a.phase] += v
            capacity[a.phase] += s * rec.plans[a.node].green_time(a.phase)
    cycles = {r.cycle for r in history}
    prio_cycles = prio_node_cycles = overrides = 0
    if emergency_run is not None:
        for entry in emergency_run.log:
            if entry.cycle in cycles:
                prio_cycles += bool(entry.priority_nodes)
                prio_node_cycles += len(entry.priority_nodes)
                overrides += len(entry.changed_nodes)
    total_green = green[Phase.NS] + green[Phase.EW]
    return SignalMetrics(
        node_cycles=len(nodes) * len(history), plan_changes=changes,
        plan_change_rate=changes / denom if denom else 0.0,
        green_seconds_ns=green[Phase.NS], green_seconds_ew=green[Phase.EW],
        ns_green_share=green[Phase.NS] / total_green if total_green else 0.0,
        phase_utilization_ns=served[Phase.NS] / capacity[Phase.NS] if capacity[Phase.NS] else None,
        phase_utilization_ew=served[Phase.EW] / capacity[Phase.EW] if capacity[Phase.EW] else None,
        emergency_priority_cycles=prio_cycles, emergency_priority_node_cycles=prio_node_cycles,
        emergency_override_plan_changes=overrides)


def emergency_metrics(vehicle: EmergencyVehicle) -> EmergencyMetrics:
    return EmergencyMetrics(
        completed=vehicle.completed, entry_cycle=vehicle.entry_cycle, completion_cycle=vehicle.completion_cycle,
        travel_time_s=vehicle.travel_time, free_flow_time_s=vehicle.free_flow_time, delay_s=vehicle.delay,
        stops=vehicle.stops, prioritized_intersections=vehicle.priority_intersections)


def environmental_metrics(traffic: TrafficMetrics, config: EnvironmentalConfig | None = None) -> EnvironmentalMetrics:
    est = estimate(traffic.waiting_vehicle_seconds, config)
    fuel_pv, co2_pv = est.per_vehicle(traffic.vehicles_admitted)
    return EnvironmentalMetrics(est, fuel_pv, co2_pv)


def unified_metrics(
    history: Sequence[CycleRecord],
    environment: EnvironmentalConfig | None = None,
    saturation_flow: float | None = None,
    emergency_run: EmergencyRun | None = None,
) -> UnifiedMetrics:
    """All four views of one simulator history (any controller; optionally an emergency run)."""
    if not history:
        raise ValueError("cannot compute metrics for an empty history")
    traffic = traffic_metrics(history)
    return UnifiedMetrics(
        traffic=traffic,
        signal=signal_metrics(history, saturation_flow, emergency_run),
        emergency=None if emergency_run is None else emergency_metrics(emergency_run.vehicle),
        environmental=environmental_metrics(traffic, environment))


def relative_changes(baseline: Mapping[str, float], other: Mapping[str, float], keys: Sequence[str]) -> dict:
    """Signed % change of ``other`` vs ``baseline`` per key (None when the baseline is 0)."""
    return {k: percent_change(baseline[k], other[k]) for k in keys}
