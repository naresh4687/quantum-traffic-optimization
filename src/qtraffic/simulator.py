"""Cycle-by-cycle queue-based fluid simulator (no Qiskit, no optimisation).

One call to ``step`` simulates one 60 s cycle, in this order:

1. Observe: snapshot queues and in-transit flow for the controller.
2. Controller selects a signal plan for every intersection.
3. The plans are validated and applied.
4. Arrivals: vehicles released upstream last cycle join their queues, then external
   demand joins entry queues up to road capacity (the excess is rejected).
5. Service: an approach with green time ``g`` can discharge at most
   ``saturation_flow * g`` vehicles, limited by its queue and by the free space on
   the downstream road.
6. Released vehicles become in-transit and arrive downstream next cycle (or leave the
   network if there is no downstream intersection).
7. Queues are updated and the cycle is recorded.

Downstream space is measured after the downstream queue has been served
(approaches are processed downstream-first), and each approach has at most one
feeder, so queues can never exceed capacity. Queues that cannot discharge because
the road ahead is full stay upstream: that is the spillback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from .controllers.base import Controller, Observation
from .demand import DemandSource
from .metrics import CycleRecord, Metrics, compute_metrics
from .network import Approach, Network
from .signals import validate_plan_assignment


@dataclass(frozen=True)
class SimulationConfig:
    saturation_flow: float = 0.5  # vehicles per second of green, per approach

    def __post_init__(self) -> None:
        if not (math.isfinite(self.saturation_flow) and self.saturation_flow > 0):
            raise ValueError("saturation_flow must be finite and > 0")


@dataclass
class TrafficState:
    cycle: int
    queues: dict[Approach, float]
    in_transit: dict[Approach, float]


@dataclass(frozen=True)
class SimulationResult:
    controller: str
    history: tuple[CycleRecord, ...]
    metrics: Metrics
    final_queues: Mapping[Approach, float] = field(repr=False)
    final_in_transit: Mapping[Approach, float] = field(repr=False)


class Simulator:
    def __init__(
        self,
        network: Network,
        demand: DemandSource,
        config: SimulationConfig | None = None,
    ):
        self.network = network
        self.demand = demand
        self.config = config or SimulationConfig()

    def initial_state(self) -> TrafficState:
        zeros = {a: 0.0 for a in self.network.approaches}
        return TrafficState(cycle=0, queues=dict(zeros), in_transit=dict(zeros))

    def run(self, controller: Controller, cycles: int) -> SimulationResult:
        """Simulate ``cycles`` cycles from an empty network under ``controller``."""
        if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 1:
            raise ValueError("cycles must be a positive integer")
        controller.reset()
        state = self.initial_state()
        history = tuple(self.step(state, controller) for _ in range(cycles))
        return SimulationResult(
            controller=controller.name,
            history=history,
            metrics=compute_metrics(history),
            final_queues=dict(state.queues),
            final_in_transit=dict(state.in_transit),
        )

    def step(self, state: TrafficState, controller: Controller) -> CycleRecord:
        """Advance ``state`` by one cycle in place and return what happened."""
        net = self.network
        cycle = state.cycle

        # 1-3. observe, select, validate/apply plans
        observation = Observation(cycle, dict(state.queues), dict(state.in_transit), net)
        plans = validate_plan_assignment(controller.select_plans(observation), net.nodes)

        # 4. arrivals
        queues = state.queues
        transit_arrivals = dict(state.in_transit)
        for a, v in transit_arrivals.items():
            queues[a] += v
        demand = self._external_demand(cycle)
        admitted: dict[Approach, float] = {}
        for a in net.entry_approaches:
            admitted[a] = min(demand[a], max(0.0, net.capacity(a) - queues[a]))
            queues[a] += admitted[a]
        q0 = dict(queues)

        # 5-6. service, capacity check, propagation
        served = {a: 0.0 for a in net.approaches}
        blocked = {a: 0.0 for a in net.approaches}
        new_transit = {a: 0.0 for a in net.approaches}
        exited = 0.0
        for a in net.service_order:
            green = plans[a.node].green_time(a.phase)
            wanted = min(q0[a], self.config.saturation_flow * green)
            down = net.downstream_approach(a)
            # queues[down] is already post-service: downstream is processed first.
            space = math.inf if down is None else max(0.0, net.capacity(down) - queues[down])
            released = min(wanted, space)
            queues[a] = q0[a] - released
            served[a] = released
            blocked[a] = wanted - released
            if down is None:
                exited += released
            else:
                new_transit[down] += released

        # 7. update state and record
        state.in_transit = new_transit
        state.cycle = cycle + 1
        return CycleRecord(
            cycle=cycle,
            plans=plans,
            demand=demand,
            admitted=admitted,
            transit_arrivals=transit_arrivals,
            queue_after_arrivals=q0,
            served=served,
            blocked=blocked,
            queue_end=dict(queues),
            in_transit_end=dict(new_transit),
            exited=exited,
        )

    def _external_demand(self, cycle: int) -> dict[Approach, float]:
        raw = self.demand.arrivals(cycle)
        entries = set(self.network.entry_approaches)
        unknown = set(raw) - entries
        if unknown:
            raise ValueError(f"demand given for non-entry approaches: {sorted(map(str, unknown))}")
        for a, v in raw.items():
            if not (math.isfinite(v) and v >= 0):
                raise ValueError(f"invalid demand {v!r} at {a}")
        return {a: float(raw.get(a, 0.0)) for a in self.network.entry_approaches}
