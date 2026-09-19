r"""Emergency Green Corridor: a discrete emergency vehicle plus a signal-priority override layer.

The traffic simulator is a per-cycle fluid model (60 s cycles, queues in vehicles, no intra-cycle
clock, no discrete vehicles). Rather than rewrite it, this module adds two things beside it and leaves
``simulator.py`` untouched:

* ``EmergencyOverrideController``: wraps ANY base ``Controller`` (Fixed, Adaptive, QUBO, a static QAOA
  plan, ...). Base plan in -> emergency override -> final plan out. With no emergency, or once the
  emergency has finished, the final plan IS the base plan.
* ``EmergencyVehicle``: a real, stateful entity (id, route, position, status, timestamps, delay, stops)
  advanced in lockstep with the simulator from each ``CycleRecord`` (the plans that were actually
  applied, the queue after arrivals, the vehicles served). Its travel time is the result of that
  simulation, never a constant.

Modelling assumptions (the simulator has no road lengths and no intra-cycle time, so these are needed
to give a discrete vehicle a timeline; all are parameters or documented conventions)
--------------------------------------------------------------------------------------------------
1. Time. Cycle ``c`` spans ``[60 c, 60 (c + 1))`` seconds, exactly the simulator's clock.
2. Phase order inside a cycle. NS runs first, then EW (``signals.py`` lists NS first; the fluid model
   itself ignores order because each phase runs once per cycle). Under emergency priority the
   emergency vehicle's phase is run FIRST at that intersection: the controller may choose the phase
   order, the fluid queue arithmetic does not depend on it. No yellow/all-red (as in the simulator).
3. Discharge. During its green a queue discharges at the saturation flow ``s`` (veh/s) in FIFO order.
   The vehicle in position ``n`` departs at ``green_start + n / s``, provided ``n <= s * green`` (the
   capacity the green offers). A vehicle that reaches an already-open green with nobody ahead of it
   crosses at once; one that arrived earlier and queued departs at its queue slot. If the simulator blocked the approach that cycle (full road ahead,
   ``record.blocked > 0``) the vehicles actually served are the limit, so spillback holds the
   emergency vehicle too.
4. Queueing. An emergency vehicle joins the TAIL of its approach queue (behind everything queued or
   arrived that cycle: the fluid model puts all arrivals at the start of the cycle), and leaves once
   the vehicles ahead have been served. It waits, and may stop, for red or for a queue.
5. Mass. The emergency vehicle is a single vehicle and is not added to the fluid queues: it neither
   consumes capacity nor delays other traffic. Normal traffic is affected only through the signal
   plans the override applies.
6. Links. The simulator has no road lengths, so a link takes ``link_seconds`` (default 30 s) to
   traverse in free flow. Free-flow route time = ``(len(route) - 1) * link_seconds``.
7. Turns. The simulator has no turning movements. An emergency vehicle uses the phase of the approach
   it arrives on and may turn onto the next road of its route (e.g. east-bound at I3 turning south to
   I6); turn conflicts are not modelled. It enters at the origin on the approach whose heading is the
   heading of the first road, and completes when it clears the stop line of the last intersection.

Corridor logic
--------------
At the start of each cycle the override looks at where the vehicle is (route index ``k``, time it
reaches that stop line) and grants priority to every not-yet-passed intersection ``m >= k`` whose
free-flow ETA ``t_now + (m - k) * link_seconds`` is before the end of the cycle
(+ ``lead_seconds``, default 0). Priority means: pick the valid plan (20/40, 30/30, 40/20) that gives
the vehicle's phase the most green (40 s) and put that phase first; the conflicting phase is cut to
20 s only if the base plan gave the vehicle's phase less than 40 s. An intersection is released the
cycle after the vehicle crosses it, so the corridor propagates intersection by intersection and is
never held on the whole route at once (unless the vehicle really is that close to all of it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from .controllers.base import Controller, Observation
from .metrics import CycleRecord, Metrics, compute_metrics
from .network import Approach, Network
from .signals import (
    CYCLE_LENGTH, PLAN_NS20_EW40, PLAN_NS40_EW20, Phase, SignalPlan, validate_plan_assignment,
)
from .simulator import SimulationResult, Simulator, TrafficState

DEFAULT_LINK_SECONDS = 30.0
EPS = 1e-9


class RouteError(ValueError):
    """The emergency route is not a valid path in the network."""


class Status(Enum):
    PENDING = "pending"  # event not yet started
    ACTIVE = "active"  # in the network
    COMPLETED = "completed"  # cleared the last stop line


# -- event and route validation ---------------------------------------------------------
@dataclass(frozen=True)
class EmergencyEvent:
    """Deterministic emergency configuration."""

    vehicle_id: str
    route: tuple[str, ...]
    start_cycle: int
    link_seconds: float = DEFAULT_LINK_SECONDS
    lead_seconds: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "route", tuple(self.route))
        if not self.vehicle_id or not isinstance(self.vehicle_id, str):
            raise ValueError("vehicle_id must be a non-empty string")
        if isinstance(self.start_cycle, bool) or not isinstance(self.start_cycle, int) or self.start_cycle < 0:
            raise ValueError("start_cycle must be a non-negative integer")
        if not self.link_seconds > 0:
            raise ValueError("link_seconds must be > 0")
        if self.lead_seconds < 0:
            raise ValueError("lead_seconds must be >= 0")

    @property
    def origin(self) -> str:
        return self.route[0]

    @property
    def destination(self) -> str:
        return self.route[-1]

    @property
    def free_flow_seconds(self) -> float:
        return (len(self.route) - 1) * self.link_seconds


def validate_route(network: Network, route: Sequence[str]) -> None:
    """Raise ``RouteError`` unless ``route`` is a simple directed path of at least two intersections."""
    route = tuple(route)
    if len(route) < 2:
        raise RouteError("route needs at least two intersections")
    if len(set(route)) != len(route):
        raise RouteError("route must not revisit an intersection")
    unknown = [n for n in route if n not in network.graph]
    if unknown:
        raise RouteError(f"unknown intersections in route: {unknown}")
    for u, v in zip(route, route[1:]):
        if not network.graph.has_edge(u, v):
            raise RouteError(f"no road from {u} to {v}")


def road_heading(network: Network, u: str, v: str):
    return network.graph[u][v]["heading"]


def route_approaches(network: Network, route: Sequence[str]) -> list[Approach]:
    """Approach used at each route intersection (heading of the road it arrives on; the origin uses
    the heading of the first road)."""
    validate_route(network, route)
    out = []
    for k, node in enumerate(route):
        heading = road_heading(network, route[0], route[1]) if k == 0 else road_heading(network, route[k - 1], node)
        approach = Approach(node, heading)
        if approach not in network.approaches:
            raise RouteError(f"{node} has no {heading.value}-bound approach")
        out.append(approach)
    return out


# -- signal timeline helper --------------------------------------------------------------
def phase_window(plan: SignalPlan, phase: Phase, first: Phase, cycle: int) -> tuple[float, float]:
    """Absolute green window ``[start, end)`` in seconds of ``phase`` in ``cycle`` (phase ``first`` runs first)."""
    base = float(CYCLE_LENGTH * cycle)
    first_green = float(plan.green_time(first))
    if phase is first:
        return base, base + first_green
    return base + first_green, base + CYCLE_LENGTH


# -- the emergency vehicle --------------------------------------------------------------------
@dataclass
class Crossing:
    node: str
    approach: Approach
    arrival_time: float
    crossing_time: float
    wait: float
    stopped: bool
    cycle: int  # simulator cycle in which it crossed
    priority: bool  # override active at this intersection in that cycle


class EmergencyVehicle:
    """Stateful emergency vehicle advanced cycle by cycle alongside the simulator."""

    def __init__(self, event: EmergencyEvent, network: Network, saturation_flow: float = 0.5):
        validate_route(network, event.route)
        self.event = event
        self.network = network
        self.saturation_flow = saturation_flow
        self.approaches = route_approaches(network, event.route)
        self.priority_cycles: dict[str, list[int]] = {n: [] for n in event.route}
        self.reset()

    def reset(self) -> None:
        self.status = Status.PENDING
        self.index = 0  # route index of the intersection it is approaching / waiting at
        self.arrival_time = 0.0  # when it reaches that stop line
        self.carried_ahead: float | None = None  # vehicles ahead of it if it is waiting in a queue
        self.entry_cycle: int | None = None
        self.entry_time: float | None = None
        self.completion_cycle: int | None = None
        self.completion_time: float | None = None
        self.crossings: list[Crossing] = []
        self.events: list[dict] = []
        for cycles in self.priority_cycles.values():
            cycles.clear()

    # -- properties -----------------------------------------------------------------------
    @property
    def vehicle_id(self) -> str:
        return self.event.vehicle_id

    @property
    def route(self) -> tuple[str, ...]:
        return self.event.route

    @property
    def active(self) -> bool:
        return self.status is Status.ACTIVE

    @property
    def completed(self) -> bool:
        return self.status is Status.COMPLETED

    @property
    def current_node(self) -> str | None:
        return self.route[self.index] if self.active else None

    @property
    def travel_time(self) -> float | None:
        return None if self.completion_time is None else self.completion_time - self.entry_time

    @property
    def free_flow_time(self) -> float:
        return self.event.free_flow_seconds

    @property
    def delay(self) -> float | None:
        t = self.travel_time
        return None if t is None else t - self.free_flow_time

    @property
    def stops(self) -> int:
        return sum(c.stopped for c in self.crossings)

    @property
    def priority_intersections(self) -> int:
        return sum(1 for cycles in self.priority_cycles.values() if cycles)

    def position(self) -> dict:
        """Where the vehicle is: waiting at a stop line, or on a road between two intersections."""
        if self.status is Status.PENDING:
            return {"status": "pending"}
        if self.status is Status.COMPLETED:
            return {"status": "completed", "at": self.route[-1]}
        k = self.index
        if k > 0 and self.crossings and self.crossings[-1].node == self.route[k - 1]:
            return {"status": "active", "route_index": k, "next_intersection": self.route[k],
                    "on_road": (self.route[k - 1], self.route[k]),
                    "reaches_stop_line_at": self.arrival_time}
        return {"status": "active", "route_index": k, "at_stop_line_of": self.route[k]}

    # -- lockstep with the simulator ------------------------------------------------------------
    def begin_cycle(self, cycle: int) -> None:
        """Called before the simulator steps ``cycle``: the vehicle enters at its start cycle."""
        if self.status is Status.PENDING:
            if cycle > self.event.start_cycle:
                raise ValueError("emergency start_cycle is before the first simulated cycle")
            if cycle == self.event.start_cycle:
                self.status = Status.ACTIVE
                self.index = 0
                self.arrival_time = float(CYCLE_LENGTH * cycle)
                self.entry_cycle, self.entry_time = cycle, self.arrival_time
                self.events.append({"cycle": cycle, "time": self.arrival_time, "event": "entered",
                                    "node": self.route[0]})

    def eta_from_now(self, cycle: int) -> list[float]:
        """Free-flow ETA at every route intersection from ``index`` on, as seen at the start of ``cycle``."""
        t_now = max(self.arrival_time, float(CYCLE_LENGTH * cycle))
        return [t_now + (m - self.index) * self.event.link_seconds for m in range(self.index, len(self.route))]

    def process_cycle(self, record: CycleRecord, first_phase: Mapping[str, Phase],
                      priority_nodes: Sequence[str] = ()) -> None:
        """Advance the vehicle through ``record.cycle`` using what the simulator actually did."""
        if not self.active:
            return
        c = record.cycle
        cycle_start, cycle_end = float(CYCLE_LENGTH * c), float(CYCLE_LENGTH * (c + 1))
        s = self.saturation_flow
        while self.active:
            t_eff = max(self.arrival_time, cycle_start)
            if t_eff >= cycle_end - EPS:
                return  # its next event is in a later cycle
            a = self.approaches[self.index]
            ws, we = phase_window(record.plans[a.node], a.phase, first_phase.get(a.node, Phase.NS), c)
            served = record.served[a]
            # discharge this cycle can offer one more (massless) vehicle: the green's capacity, unless the
            # simulator blocked the approach (full road ahead), where what it served is the binding limit
            limit = served if record.blocked[a] > EPS else s * (we - ws)
            ahead = self.carried_ahead if self.carried_ahead is not None else record.queue_after_arrivals[a]
            slot = ws + (ahead + 1.0) / s
            if ahead <= EPS and self.arrival_time >= ws - EPS:
                t_cross = t_eff  # reached an already-open green with nobody ahead: no start-up wait
            else:
                t_cross = max(t_eff, slot)
            if ahead + 1.0 <= limit + EPS and t_cross <= we + EPS:
                wait = t_cross - self.arrival_time
                self.crossings.append(Crossing(a.node, a, self.arrival_time, t_cross, wait, wait > EPS, c,
                                               a.node in priority_nodes))
                self.events.append({"cycle": c, "time": t_cross, "event": "crossed", "node": a.node,
                                    "wait": wait, "stopped": wait > EPS})
                self.carried_ahead = None
                if self.index == len(self.route) - 1:
                    self.status = Status.COMPLETED
                    self.completion_cycle, self.completion_time = c, t_cross
                    self.events.append({"cycle": c, "time": t_cross, "event": "completed", "node": a.node})
                else:
                    self.index += 1
                    self.arrival_time = t_cross + self.event.link_seconds
                continue
            # not served this cycle (red, or the queue ahead is longer than this cycle's service)
            self.carried_ahead = max(0.0, ahead - served)
            self.events.append({"cycle": c, "time": cycle_end, "event": "waiting", "node": a.node,
                                "vehicles_ahead": self.carried_ahead})
            return


# -- the priority override layer -----------------------------------------------------------------
@dataclass(frozen=True)
class OverrideRecord:
    cycle: int
    active: bool  # emergency vehicle in the network and override enabled
    priority_nodes: tuple[str, ...]
    changed_nodes: tuple[str, ...]  # nodes whose base plan the override actually changed
    base_plans: Mapping[str, SignalPlan]
    final_plans: Mapping[str, SignalPlan]
    first_phase: Mapping[str, Phase]


def priority_plan(phase: Phase) -> SignalPlan:
    """The valid plan giving ``phase`` its maximum green (40 s) and the conflicting phase 20 s."""
    return PLAN_NS40_EW20 if phase is Phase.NS else PLAN_NS20_EW40


class EmergencyOverrideController(Controller):
    """Base controller -> emergency override -> final plan."""

    def __init__(self, base: Controller, vehicle: EmergencyVehicle, enabled: bool = True):
        self.base = base
        self.vehicle = vehicle
        self.enabled = enabled
        self.log: list[OverrideRecord] = []

    @property
    def name(self) -> str:
        return f"{self.base.name}+EmergencyCorridor" if self.enabled else self.base.name

    def reset(self) -> None:
        self.base.reset()
        self.vehicle.reset()
        self.log = []

    def priority_nodes(self, cycle: int) -> tuple[str, ...]:
        v = self.vehicle
        if not (self.enabled and v.active):
            return ()
        horizon = CYCLE_LENGTH * (cycle + 1) + v.event.lead_seconds
        etas = v.eta_from_now(cycle)
        return tuple(v.route[v.index + i] for i, eta in enumerate(etas) if eta < horizon - EPS)

    def select_plans(self, observation: Observation) -> Mapping[str, SignalPlan]:
        base_plans = dict(self.base.select_plans(observation))  # the base controller always runs
        final = dict(base_plans)
        first: dict[str, Phase] = {}
        changed = []
        nodes = self.priority_nodes(observation.cycle)
        for node in nodes:
            phase = self.vehicle.approaches[self.vehicle.route.index(node)].phase
            target = priority_plan(phase)
            if base_plans[node].green_time(phase) < target.green_time(phase):
                final[node] = target  # cut the conflicting phase only when the base gave less than max green
                changed.append(node)
            first[node] = phase
            self.vehicle.priority_cycles[node].append(observation.cycle)
        self.log.append(OverrideRecord(observation.cycle, self.enabled and self.vehicle.active, nodes,
                                       tuple(changed), base_plans, final, first))
        return validate_plan_assignment(final, observation.network.nodes)


# -- running an emergency experiment --------------------------------------------------------------------
@dataclass(frozen=True)
class EmergencyRun:
    corridor: bool
    controller: str
    result: SimulationResult
    vehicle: EmergencyVehicle
    log: tuple[OverrideRecord, ...]
    start_cycle: int  # first simulated cycle

    @property
    def history(self) -> tuple[CycleRecord, ...]:
        return self.result.history

    def window_metrics(self, first_cycle: int, last_cycle: int) -> Metrics | None:
        """Normal-traffic metrics over simulator cycles ``first_cycle..last_cycle`` inclusive."""
        rows = [r for r in self.history if first_cycle <= r.cycle <= last_cycle]
        return compute_metrics(rows) if rows else None


def run_emergency(
    simulator: Simulator,
    base: Controller,
    event: EmergencyEvent,
    cycles: int,
    corridor: bool = True,
    start_state: TrafficState | None = None,
) -> EmergencyRun:
    """Simulate ``cycles`` cycles with the emergency vehicle; ``corridor`` switches only the override."""
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 1:
        raise ValueError("cycles must be a positive integer")
    state = simulator.initial_state() if start_state is None else TrafficState(
        start_state.cycle, dict(start_state.queues), dict(start_state.in_transit))
    first_cycle = state.cycle
    if not first_cycle <= event.start_cycle < first_cycle + cycles:
        raise ValueError("emergency start_cycle must fall inside the simulated cycles")
    vehicle = EmergencyVehicle(event, simulator.network, simulator.config.saturation_flow)
    controller = EmergencyOverrideController(base, vehicle, enabled=corridor)
    controller.reset()
    history = []
    for _ in range(cycles):
        vehicle.begin_cycle(state.cycle)
        record = simulator.step(state, controller)
        entry = controller.log[-1]
        vehicle.process_cycle(record, entry.first_phase, entry.priority_nodes)
        history.append(record)
    history = tuple(history)
    result = SimulationResult(controller.name, history, compute_metrics(history),
                              dict(state.queues), dict(state.in_transit))
    return EmergencyRun(corridor, controller.name, result, vehicle, tuple(controller.log), first_cycle)
