"""Per-cycle records and summary metrics.

Definitions (units: vehicles, seconds)
--------------------------------------
* Waiting time: for each approach and cycle, the queue is assumed to fall linearly
  from its size after arrivals (q0) to its size after service (q1), so the
  vehicle-seconds waited are ``(q0 + q1) / 2 * CYCLE_LENGTH``. Vehicles in transit
  between intersections are not waiting. Arrivals are assumed to occur at the start
  of the cycle, which slightly overstates waiting for vehicles arriving later in it.
* Average queue length: time-averaged queue per approach, i.e.
  total waiting time / (CYCLE_LENGTH * cycles * approaches).
* Maximum queue length: largest q0 of any approach in any cycle (the peak).
* Vehicles served: stop-line discharges. A vehicle crossing three intersections
  counts three times.
* Throughput: vehicles that left the network, per hour.
* Final queue: vehicles still queued after the last cycle, per intersection. Vehicles
  in transit at that moment are reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping, Sequence

from .network import Approach
from .signals import CYCLE_LENGTH, SignalPlan


@dataclass(frozen=True)
class CycleRecord:
    """Everything that happened in one simulated cycle (per-approach dicts)."""

    cycle: int
    plans: Mapping[str, SignalPlan]
    demand: Mapping[Approach, float]  # external vehicles trying to enter
    admitted: Mapping[Approach, float]  # external vehicles that fit
    transit_arrivals: Mapping[Approach, float]  # released upstream last cycle
    queue_after_arrivals: Mapping[Approach, float]  # q0
    served: Mapping[Approach, float]  # discharged through the stop line
    blocked: Mapping[Approach, float]  # discharge withheld by full downstream road
    queue_end: Mapping[Approach, float]  # q1
    in_transit_end: Mapping[Approach, float]  # released this cycle, arrive next
    exited: float  # vehicles that left the network this cycle

    @property
    def rejected(self) -> float:
        return sum(self.demand.values()) - sum(self.admitted.values())

    @property
    def waiting_time(self) -> float:
        return sum(
            (self.queue_after_arrivals[a] + self.queue_end[a]) / 2.0 * CYCLE_LENGTH
            for a in self.queue_end
        )


@dataclass(frozen=True)
class Metrics:
    cycles: int
    demand_generated: float
    vehicles_entered: float
    vehicles_rejected: float
    vehicles_served: float
    vehicles_exited: float
    throughput_per_hour: float
    total_waiting_time: float
    average_queue_length: float
    max_queue_length: float
    total_blocked: float
    final_queue: Mapping[str, float]
    final_queue_total: float
    in_transit_final: float

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def compute_metrics(history: Sequence[CycleRecord]) -> Metrics:
    if not history:
        raise ValueError("cannot compute metrics for an empty history")
    n_cycles = len(history)
    n_approaches = len(history[0].queue_end)
    total_wait = sum(r.waiting_time for r in history)
    exited = sum(r.exited for r in history)
    final_queue: dict[str, float] = {}
    for approach, q in history[-1].queue_end.items():
        final_queue[approach.node] = final_queue.get(approach.node, 0.0) + q
    return Metrics(
        cycles=n_cycles,
        demand_generated=sum(sum(r.demand.values()) for r in history),
        vehicles_entered=sum(sum(r.admitted.values()) for r in history),
        vehicles_rejected=sum(r.rejected for r in history),
        vehicles_served=sum(sum(r.served.values()) for r in history),
        vehicles_exited=exited,
        throughput_per_hour=exited / (n_cycles * CYCLE_LENGTH / 3600.0),
        total_waiting_time=total_wait,
        average_queue_length=total_wait / (CYCLE_LENGTH * n_cycles * n_approaches),
        max_queue_length=max(max(r.queue_after_arrivals.values()) for r in history),
        total_blocked=sum(sum(r.blocked.values()) for r in history),
        final_queue=final_queue,
        final_queue_total=sum(final_queue.values()),
        in_transit_final=sum(history[-1].in_transit_end.values()),
    )
