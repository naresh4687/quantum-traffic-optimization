"""Dashboard logic layer: configuration, live simulation runs, view-model transformations, AI contexts.

No Streamlit and no plotting here. Everything is computed from the existing, validated qtraffic APIs
(simulator, controllers, QUBO, emergency corridor, unified metrics) or from saved results; nothing is
hard-coded. The UI modules only render what this module returns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping

from ..ai import Explainer, TrafficAnalysisContext, load_qaoa_context
from ..ai.explain import explain_comparison, explain_emergency, explain_environment, explain_optimization, explain_qaoa
from ..controllers import AdaptiveController, Controller, FixedTimeController, Observation
from ..emergency import EmergencyEvent, EmergencyRun, RouteError, run_emergency, validate_route
from ..environment import PROXY_LABEL, EnvironmentalConfig, percent_change
from ..network import Network, grid_network
from ..optimization import QuboController, StaticPlanController, TrafficState, build_qubo, solve_exact
from ..optimization.validation import CASES, initial_state_for, run_from_state
from ..signals import CYCLE_LENGTH, Phase
from ..simulator import Simulator
from ..unified_metrics import UnifiedMetrics, unified_metrics
from .saved import SavedResults
from .theme import CONGESTION_THRESHOLDS

# ------------------------------------------------------------------------------------------- catalogue
SCENARIO_INFO = {
    "balanced_medium": ("Balanced medium", "uniform medium demand"),
    "ns_heavy": ("NS-heavy", "north-south demand x1.5, east-west x0.5"),
    "ew_heavy": ("EW-heavy", "east-west demand x1.5, north-south x0.5"),
    "time_varying": ("Time-varying", "30-cycle blocks: NS-heavy, balanced, EW-heavy, balanced"),
    "downstream_congested": ("Downstream congested", "EW-heavy demand from a jammed eastbound start"),
}
_CASES = {c.name: c for c in CASES}
SCENARIOS = tuple(SCENARIO_INFO)


@dataclass(frozen=True)
class ControllerSpec:
    key: str
    label: str
    description: str
    summary_name: str  # name used in the saved Phase 6 tables and in the Phase 7 explanation layer


CONTROLLERS = {
    "fixed": ControllerSpec("fixed", "Fixed", "same NS 30 s / EW 30 s plan everywhere", "fixed"),
    "adaptive": ControllerSpec("adaptive", "Adaptive", "queue-pressure rule, one plan per intersection per cycle", "adaptive"),
    "qubo_static": ControllerSpec("qubo_static", "QUBO Static", "exact QUBO solution from the start state, held", "qubo_static"),
    "qubo_receding": ControllerSpec("qubo_receding", "QUBO Receding", "exact QUBO re-solved every cycle", "qubo_receding"),
    "qaoa_p1": ControllerSpec("qaoa_p1", "QAOA p=1", "best sampled plan from the saved Phase 4 QAOA run (Aer simulator)", "qaoa_p1_saved"),
}
EMERGENCY_ROUTES = {
    "I1 > I2 > I3 > I6": ("I1", "I2", "I3", "I6"),
    "I1 > I2 > I3": ("I1", "I2", "I3"),
    "I4 > I5 > I6": ("I4", "I5", "I6"),
    "I6 > I5 > I4 > I1": ("I6", "I5", "I4", "I1"),
    "I1 > I4": ("I1", "I4"),
}
PRIMARY_ROUTE = EMERGENCY_ROUTES["I1 > I2 > I3 > I6"]
MIN_CYCLES, MAX_CYCLES, DEFAULT_CYCLES = 20, 120, 60
MAX_SEED = 99
SATURATION = 0.5


class DashboardConfigError(ValueError):
    """The requested dashboard configuration cannot be run."""


def first_cycle_of(scenario: str) -> int:
    """First simulated cycle of a scenario (its start state: warm-up cycles, or 0 for the synthetic jam)."""
    if scenario not in _CASES:
        raise DashboardConfigError(f"unknown scenario {scenario!r}")
    case = _CASES[scenario]
    return 0 if case.initial_state is not None else case.warmup


def emergency_bounds(scenario: str, cycles: int) -> tuple[int, int, int]:
    """(first allowed start cycle, last allowed start cycle, default start cycle) for a scenario and horizon."""
    first = first_cycle_of(scenario)
    last = first + cycles - 1
    return first, last, min(first + 10, last)


@dataclass(frozen=True)
class RunConfig:
    scenario: str = "ns_heavy"
    controller: str = "adaptive"
    seed: int = 0
    cycles: int = DEFAULT_CYCLES
    emergency_enabled: bool = True
    route: tuple[str, ...] = PRIMARY_ROUTE
    emergency_start: int | None = None  # absolute cycle; None -> the scenario default

    def resolved_start(self) -> int:
        return self.emergency_start if self.emergency_start is not None else emergency_bounds(self.scenario, self.cycles)[2]


DEFAULT_CONFIG = RunConfig()


def normalize_config(cfg: RunConfig) -> RunConfig:
    """Canonical form for comparing configurations: emergency settings are irrelevant when the emergency is off,
    and an unset start cycle means the scenario default."""
    if not cfg.emergency_enabled:
        return RunConfig(cfg.scenario, cfg.controller, cfg.seed, cfg.cycles, False, PRIMARY_ROUTE, None)
    return RunConfig(cfg.scenario, cfg.controller, cfg.seed, cfg.cycles, True, cfg.route, cfg.resolved_start())


def validate_config(cfg: RunConfig, saved: SavedResults | None = None, net: Network | None = None) -> None:
    """Raise ``DashboardConfigError`` unless the configuration can really be simulated."""
    if cfg.scenario not in SCENARIO_INFO:
        raise DashboardConfigError(f"unknown scenario {cfg.scenario!r}; choose from {list(SCENARIO_INFO)}")
    if cfg.controller not in CONTROLLERS:
        raise DashboardConfigError(f"unknown controller {cfg.controller!r}; choose from {list(CONTROLLERS)}")
    if isinstance(cfg.seed, bool) or not isinstance(cfg.seed, int) or not 0 <= cfg.seed <= MAX_SEED:
        raise DashboardConfigError(f"seed must be an integer in 0..{MAX_SEED}")
    if isinstance(cfg.cycles, bool) or not isinstance(cfg.cycles, int) or not MIN_CYCLES <= cfg.cycles <= MAX_CYCLES:
        raise DashboardConfigError(f"cycles must be an integer in {MIN_CYCLES}..{MAX_CYCLES}")
    if cfg.controller == "qaoa_p1" and (saved is None or saved.qaoa_plans(cfg.scenario, cfg.seed) is None):
        available = saved.qaoa_seeds(cfg.scenario) if saved else []
        raise DashboardConfigError(f"no saved QAOA p=1 plan for scenario {cfg.scenario!r} with seed {cfg.seed} "
                                   f"(saved Phase 4 seeds: {available or 'none'})")
    if cfg.emergency_enabled:
        try:
            validate_route(net or grid_network(2, 3), cfg.route)
        except RouteError as exc:
            raise DashboardConfigError(f"invalid emergency route: {exc}") from exc
        lo, hi, _ = emergency_bounds(cfg.scenario, cfg.cycles)
        start = cfg.resolved_start()
        if not lo <= start <= hi:
            raise DashboardConfigError(f"emergency start cycle {start} must be within the simulated cycles {lo}..{hi}")


def available_controllers(scenario: str, seed: int, saved: SavedResults | None) -> list[str]:
    """Controller keys that can actually run for this scenario and seed (QAOA needs a saved Phase 4 plan)."""
    keys = [k for k in CONTROLLERS if k != "qaoa_p1"]
    if saved is not None and saved.qaoa_plans(scenario, seed) is not None:
        keys.append("qaoa_p1")
    return keys


def config_from_query(params: Mapping[str, str], saved: SavedResults | None = None) -> tuple[RunConfig, int | None, list[str]]:
    """Build a config (and optional playhead cycle) from URL query parameters; bad values are ignored with a note."""
    notes: list[str] = []
    values = {}

    def take(name: str, cast, allowed=None):
        if name in params:
            try:
                v = cast(params[name])
                if allowed is not None and v not in allowed:
                    raise ValueError
                values[name] = v
            except (ValueError, TypeError):
                notes.append(f"ignored invalid ?{name}={params[name]!r}")

    take("scenario", str, SCENARIO_INFO)
    take("controller", str, CONTROLLERS)
    take("seed", int)
    take("cycles", int)
    take("start", int)
    if "emergency" in params:
        values["emergency"] = str(params["emergency"]).lower() in ("1", "true", "yes", "on")
    if "route" in params:
        route = tuple(s.strip() for s in str(params["route"]).replace(">", ",").split(",") if s.strip())
        values["route"] = route
    cfg = RunConfig(
        scenario=values.get("scenario", DEFAULT_CONFIG.scenario), controller=values.get("controller", DEFAULT_CONFIG.controller),
        seed=values.get("seed", DEFAULT_CONFIG.seed), cycles=values.get("cycles", DEFAULT_CONFIG.cycles),
        emergency_enabled=values.get("emergency", DEFAULT_CONFIG.emergency_enabled), route=values.get("route", DEFAULT_CONFIG.route),
        emergency_start=values.get("start"))
    try:
        validate_config(cfg, saved)
    except DashboardConfigError as exc:
        notes.append(f"query configuration not runnable ({exc}); using defaults")
        cfg = DEFAULT_CONFIG
    playhead = None
    if "cycle" in params:
        try:
            playhead = int(params["cycle"])
        except ValueError:
            notes.append(f"ignored invalid ?cycle={params['cycle']!r}")
    return cfg, playhead, notes


# ------------------------------------------------------------------------------------------- live runs
@dataclass
class Verification:
    ok: bool | None  # True: reproduces a saved result; None: there is no saved counterpart
    text: str


@dataclass
class DashboardRun:
    config: RunConfig
    net: Network
    first_cycle: int
    history: tuple  # CycleRecord per simulated cycle (the displayed run)
    unified: UnifiedMetrics
    emergency: EmergencyRun | None  # corridor run (displayed) when emergency is enabled
    emergency_off: EmergencyRun | None  # same run without the override (measured comparison)
    unified_off: UnifiedMetrics | None
    baseline_unified: UnifiedMetrics | None  # Fixed on the same demand (and same emergency setting); None if this IS Fixed
    start_queues: dict = field(repr=False, default_factory=dict)
    start_in_transit: dict = field(repr=False, default_factory=dict)
    qubo_energy: float | None = None
    generated_at: str = ""
    source: str = "live simulator run"
    verification: Verification = field(default_factory=lambda: Verification(None, ""))

    @property
    def n_cycles(self) -> int:
        return len(self.history)

    @property
    def spec(self) -> ControllerSpec:
        return CONTROLLERS[self.config.controller]

    def cycle_at(self, idx: int) -> int:
        return self.history[self._clip(idx)].cycle

    def record_at(self, idx: int):
        return self.history[self._clip(idx)]

    def _clip(self, idx: int) -> int:
        return max(0, min(int(idx), self.n_cycles - 1))


def build_controller(key: str, sim: Simulator, start, cfg: RunConfig, saved: SavedResults | None) -> tuple[Controller, float | None]:
    """Instantiate a controller (and the exact QUBO energy for the static QUBO controller)."""
    if key == "fixed":
        return FixedTimeController(), None
    if key == "adaptive":
        return AdaptiveController(), None
    if key == "qubo_receding":
        return QuboController(), None
    if key == "qubo_static":
        obs = Observation(start.cycle, dict(start.queues), dict(start.in_transit), sim.network)
        solution = solve_exact(build_qubo(TrafficState.from_observation(obs)))
        return StaticPlanController(solution.plans, "QuboExactStatic"), solution.energy
    if key == "qaoa_p1":
        plans = saved.qaoa_plans(cfg.scenario, cfg.seed) if saved is not None else None
        if plans is None:
            raise DashboardConfigError(f"no saved QAOA p=1 plan for {cfg.scenario!r} seed {cfg.seed}")
        return StaticPlanController(plans, "QaoaP1Saved"), None
    raise DashboardConfigError(f"unknown controller {key!r}")


def run_dashboard(cfg: RunConfig, saved: SavedResults | None = None, env: EnvironmentalConfig | None = None) -> DashboardRun:
    """Run the configured simulation LIVE with the existing simulator (plus the Fixed baseline and, if enabled, the
    corridor / no-override pair). Deterministic: the same configuration always gives the same run."""
    net = grid_network(2, 3)
    validate_config(cfg, saved, net)
    env = env or EnvironmentalConfig()
    case = _CASES[cfg.scenario]
    sim = Simulator(net, case.scenario.build_demand(net, cfg.seed))
    start = initial_state_for(case, sim)
    event = EmergencyEvent("EV1", cfg.route, cfg.resolved_start()) if cfg.emergency_enabled else None

    def simulate(key: str, corridor: bool):
        controller, energy = build_controller(key, sim, start, cfg, saved)
        if event is None:
            result = run_from_state(sim, start, controller, cfg.cycles)
            return result.history, None, energy
        run = run_emergency(sim, controller, event, cfg.cycles, corridor, start)
        return run.history, run, energy

    history, em_run, energy = simulate(cfg.controller, True)
    unified = unified_metrics(history, env, SATURATION, em_run)
    off_run = unified_off = None
    if event is not None:
        h_off, off_run, _ = simulate(cfg.controller, False)
        unified_off = unified_metrics(h_off, env, SATURATION, off_run)
    baseline = None
    if cfg.controller != "fixed":
        h_base, base_run, _ = simulate("fixed", True)
        baseline = unified_metrics(h_base, env, SATURATION, base_run)
    run = DashboardRun(
        config=cfg, net=net, first_cycle=start.cycle, history=tuple(history), unified=unified, emergency=em_run, emergency_off=off_run,
        unified_off=unified_off, baseline_unified=baseline, start_queues=dict(start.queues), start_in_transit=dict(start.in_transit),
        qubo_energy=energy, generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    run.verification = verify_against_saved(run, saved)
    return run


def verify_against_saved(run: DashboardRun, saved: SavedResults | None) -> Verification:
    """Compare the live run with the saved experiment results when a saved counterpart exists."""
    if saved is None:
        return Verification(None, "")
    cfg = run.config
    name = CONTROLLERS[cfg.controller].summary_name
    if not cfg.emergency_enabled:
        row = next((r for r in saved.controller_rows if r["scenario"] == cfg.scenario and r["seed"] == cfg.seed and r["controller"] == name
                    and r["cycles"] == cfg.cycles), None)
        if row is None:
            return Verification(None, "no saved counterpart for this configuration")
        ok = math.isclose(row["waiting_vehicle_seconds"], run.unified.traffic.waiting_vehicle_seconds, rel_tol=1e-9, abs_tol=1e-6)
        return Verification(ok, "reproduces the saved Phase 6 result" if ok else "differs from the saved Phase 6 result")
    if cfg.controller in ("fixed", "adaptive") and cfg.route == PRIMARY_ROUTE and cfg.resolved_start() == 40 and cfg.cycles == DEFAULT_CYCLES:
        row = next((r for r in saved.emergency_comparison if r["scenario"] == cfg.scenario and r["base_controller"] == cfg.controller
                    and r["seed"] == cfg.seed), None)
        if row is not None and run.emergency is not None and run.emergency.vehicle.travel_time is not None:
            ok = math.isclose(row["B_travel_time_s"], run.emergency.vehicle.travel_time, abs_tol=1e-6)
            return Verification(ok, "reproduces the saved Phase 5/6 emergency result" if ok else "differs from the saved emergency result")
    return Verification(None, "no saved counterpart for this configuration")


# ------------------------------------------------------------------------------------------- KPIs
def kpis(run: DashboardRun) -> dict:
    """Headline numbers of the displayed run (measured simulator outputs; fuel/CO2 are the labelled proxy)."""
    t, e = run.unified.traffic, run.unified.environmental
    out = {"waiting_seconds_per_admitted": t.waiting_seconds_per_admitted, "throughput_vehicles_per_hour": t.throughput_vehicles_per_hour,
           "average_queue_vehicles": t.average_queue_vehicles, "max_queue_vehicles": t.max_queue_vehicles,
           "co2_kg_proxy": e.co2_kg, "fuel_liters_proxy": e.fuel_liters, "waiting_vehicle_seconds": t.waiting_vehicle_seconds,
           "vehicles_admitted": t.vehicles_admitted, "vehicles_rejected": t.vehicles_rejected, "final_queue_vehicles": t.final_queue_vehicles,
           "blocked_vehicle_cycles": t.blocked_vehicle_cycles}
    b = run.baseline_unified
    if b is not None:
        bt, be = b.traffic, b.environmental
        out["delta_vs_fixed_pct"] = {
            "waiting_seconds_per_admitted": percent_change(bt.waiting_seconds_per_admitted, t.waiting_seconds_per_admitted),
            "throughput_vehicles_per_hour": percent_change(bt.throughput_vehicles_per_hour, t.throughput_vehicles_per_hour),
            "average_queue_vehicles": percent_change(bt.average_queue_vehicles, t.average_queue_vehicles),
            "max_queue_vehicles": percent_change(bt.max_queue_vehicles, t.max_queue_vehicles),
            "co2_kg_proxy": percent_change(be.co2_kg, e.co2_kg)}
    return out


# ------------------------------------------------------------------------------------------- network view model
def congestion_level(ratio: float) -> str:
    if ratio >= CONGESTION_THRESHOLDS["severe"]:
        return "severe"
    return "high" if ratio >= CONGESTION_THRESHOLDS["high"] else "normal"


@dataclass(frozen=True)
class NodeView:
    node: str
    row: int
    col: int
    plan: str  # plan actually applied this cycle, e.g. "NS40/EW20"
    base_plan: str  # what the base controller chose
    ns_green: int
    ew_green: int
    first_phase: str  # phase that opens the cycle ("NS" default order; the emergency phase under priority)
    queue_ns: float
    queue_ew: float
    congestion: str
    priority: bool  # emergency priority active at this intersection this cycle
    overridden: bool  # the override changed the base plan
    pressure_ns: float | None
    pressure_ew: float | None

    @property
    def queue_total(self) -> float:
        return self.queue_ns + self.queue_ew


@dataclass(frozen=True)
class RoadView:
    approach: str
    node: str
    heading: str
    upstream: tuple[float, float]  # grid coords (col, row) of the road's upstream end (may be outside the grid)
    downstream: tuple[float, float]
    queue: float
    capacity: float
    ratio: float
    congestion: str
    external_entry: bool


@dataclass(frozen=True)
class NetworkView:
    cycle: int
    index: int
    nodes: tuple
    roads: tuple
    total_queue: float
    exit_stubs: tuple  # (node, heading, (col,row) start, (col,row) end) for roads that leave the network

    def node(self, name: str) -> NodeView:
        return next(n for n in self.nodes if n.node == name)


def _prev_state(run: DashboardRun, idx: int):
    if idx <= 0:
        return run.start_queues, run.start_in_transit
    prev = run.history[idx - 1]
    return prev.queue_end, prev.in_transit_end


def network_view(run: DashboardRun, idx: int) -> NetworkView:
    """The network state for one simulated cycle, taken directly from the simulator's CycleRecord and the override log."""
    idx = max(0, min(idx, run.n_cycles - 1))
    rec, net = run.history[idx], run.net
    log = run.emergency.log[idx] if run.emergency is not None and idx < len(run.emergency.log) else None
    queues, transit = _prev_state(run, idx)
    obs = Observation(rec.cycle, dict(queues), dict(transit), net)
    adaptive = AdaptiveController()
    nodes = []
    for name in net.nodes:
        row, col = net.graph.nodes[name]["pos"]
        plan = rec.plans[name]
        base = log.base_plans[name] if log else plan
        first = log.first_phase.get(name, Phase.NS) if log else Phase.NS
        q = {Phase.NS: 0.0, Phase.EW: 0.0}
        worst = 0.0
        for a in net.approaches:
            if a.node == name:
                q[a.phase] += rec.queue_after_arrivals[a]
                worst = max(worst, rec.queue_after_arrivals[a] / net.capacity(a))
        p_ns, p_ew = adaptive.phase_pressures(name, obs)
        nodes.append(NodeView(
            node=name, row=row, col=col, plan=plan.label, base_plan=base.label, ns_green=plan.ns_green, ew_green=plan.ew_green,
            first_phase=first.value, queue_ns=q[Phase.NS], queue_ew=q[Phase.EW], congestion=congestion_level(worst),
            priority=bool(log and name in log.priority_nodes), overridden=bool(log and name in log.changed_nodes),
            pressure_ns=p_ns, pressure_ew=p_ew))
    roads, stubs = [], []
    for a in net.approaches:
        row, col = net.graph.nodes[a.node]["pos"]
        dr, dc = a.heading.step
        up = (col - dc, row - dr)
        ext = net.is_entry(a)
        cap = net.capacity(a)
        q = rec.queue_after_arrivals[a]
        roads.append(RoadView(str(a), a.node, a.heading.value, up, (float(col), float(row)), q, cap, q / cap, congestion_level(q / cap), ext))
        if net.downstream_approach(a) is None:  # vehicles leave the network past this node
            stubs.append((a.node, a.heading.value, (float(col), float(row)), (col + dc, row + dr)))
    return NetworkView(rec.cycle, idx, tuple(nodes), tuple(roads), sum(rec.queue_after_arrivals.values()), tuple(stubs))


# ------------------------------------------------------------------------------------------- emergency view model
@dataclass(frozen=True)
class EmergencyView:
    phase: str  # "off" | "pending" | "active" | "complete"
    vehicle_id: str | None = None
    route: tuple = ()
    completed: tuple = ()
    current: str | None = None
    upcoming: tuple = ()
    position: tuple | None = None  # (col, row) in grid coordinates
    priority_nodes: tuple = ()
    restored: bool = False  # complete AND the override is inactive AND plans equal the base controller's
    travel_time_s: float | None = None
    free_flow_time_s: float | None = None
    delay_s: float | None = None
    stops: int | None = None
    completion_cycle: int | None = None
    prioritized_intersections: int | None = None
    travel_time_s_no_override: float | None = None
    delay_s_no_override: float | None = None
    stops_no_override: int | None = None

    @property
    def active(self) -> bool:
        return self.phase == "active"


def _step(net: Network, u: str, v: str):
    return net.graph[u][v]["heading"].step


def emergency_view(run: DashboardRun, idx: int) -> EmergencyView:
    """Where the emergency vehicle is at the middle of the selected cycle, read from its simulated timeline."""
    if run.emergency is None:
        return EmergencyView("off")
    idx = max(0, min(idx, run.n_cycles - 1))
    v, net = run.emergency.vehicle, run.net
    rec = run.history[idx]
    t = CYCLE_LENGTH * rec.cycle + CYCLE_LENGTH / 2.0  # mid-cycle reference time
    log = run.emergency.log[idx] if idx < len(run.emergency.log) else None
    route = v.route
    base = dict(vehicle_id=v.vehicle_id, route=route, priority_nodes=tuple(log.priority_nodes) if log else (),
                travel_time_s=v.travel_time, free_flow_time_s=v.free_flow_time, delay_s=v.delay, stops=v.stops,
                completion_cycle=v.completion_cycle, prioritized_intersections=v.priority_intersections)
    off = run.emergency_off.vehicle if run.emergency_off is not None else None
    if off is not None:
        base.update(travel_time_s_no_override=off.travel_time, delay_s_no_override=off.delay, stops_no_override=off.stops)
    if v.entry_time is None or t < v.entry_time:
        return EmergencyView("pending", upcoming=route, **base)
    crossed = [c for c in v.crossings if c.crossing_time <= t]
    done_nodes = tuple(c.node for c in crossed)
    if v.completed and t >= v.completion_time:
        restored = bool(log and not log.priority_nodes and all(log.final_plans[n] == log.base_plans[n] for n in log.final_plans))
        return EmergencyView("complete", completed=route, position=None, restored=restored, **base)
    k = len(crossed)  # index of the intersection the vehicle is at / approaching
    k = min(k, len(route) - 1)
    node = route[k]
    hr, hc = net.graph.nodes[node]["pos"]
    if k == 0:
        dr, dc = _step(net, route[0], route[1])
        pos = (hc - 0.22 * dc, hr - 0.22 * dr)
    else:
        prev = crossed[-1]
        arrival = v.crossings[k].arrival_time if k < len(v.crossings) else v.arrival_time
        if t < arrival:  # travelling on the road between route[k-1] and route[k]
            f = min((t - prev.crossing_time) / max(arrival - prev.crossing_time, 1e-9), 0.84)  # stop short of the intersection ring
            pr, pc = net.graph.nodes[route[k - 1]]["pos"]
            pos = (pc + f * (hc - pc), pr + f * (hr - pr))
        else:  # queued at the stop line
            dr, dc = _step(net, route[k - 1], node)
            pos = (hc - 0.22 * dc, hr - 0.22 * dr)
    return EmergencyView("active", completed=done_nodes, current=node, upcoming=tuple(route[k + 1:]), position=pos, **base)


def status_for(run: DashboardRun, idx: int, playing: bool) -> tuple[str, str]:
    """Header status label and CSS class."""
    if emergency_view(run, idx).active:
        return "EMERGENCY ACTIVE", "emergency"
    return ("SIMULATION RUNNING", "running") if playing else ("SYSTEM READY", "ready")


def default_playhead(run: DashboardRun) -> int:
    """Where to park the playhead so the first view is informative: just after the emergency vehicle enters, else mid-run."""
    if run.emergency is not None and run.emergency.vehicle.entry_cycle is not None:
        return min(run.emergency.vehicle.entry_cycle - run.first_cycle + 1, run.n_cycles - 1)
    return run.n_cycles // 2


def parse_selected_node(event, nodes) -> str | None:
    """Extract the clicked intersection from a Plotly selection event (dict-like), or None."""
    try:
        points = event["selection"]["points"] if isinstance(event, Mapping) else event.selection["points"]
    except (KeyError, TypeError, AttributeError):
        return None
    for p in points or []:
        cd = p.get("customdata")
        cd = cd[0] if isinstance(cd, (list, tuple)) and cd else cd
        if cd in nodes:
            return cd
    return None


# ------------------------------------------------------------------------------------------- AI contexts (existing Phase 7 API)
def _pct(a, b):
    p = percent_change(a, b)
    return None if p is None else round(p, 3)


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def live_context(run: DashboardRun) -> TrafficAnalysisContext:
    """Analysis context for the CURRENT live run (measured numbers only; environmental values are the labelled proxy)."""
    t, s, e = run.unified.traffic, run.unified.signal, run.unified.environmental
    traffic = {"vehicles_admitted": _r(t.vehicles_admitted), "vehicles_rejected": _r(t.vehicles_rejected), "vehicles_exited": _r(t.vehicles_exited),
               "throughput_vehicles_per_hour": _r(t.throughput_vehicles_per_hour), "waiting_vehicle_seconds": _r(t.waiting_vehicle_seconds),
               "waiting_seconds_per_admitted": _r(t.waiting_seconds_per_admitted), "average_queue_vehicles": _r(t.average_queue_vehicles),
               "max_queue_vehicles": _r(t.max_queue_vehicles), "final_queue_vehicles": _r(t.final_queue_vehicles),
               "blocked_vehicle_cycles": _r(t.blocked_vehicle_cycles)}
    signal = {"plan_changes": s.plan_changes, "plan_change_rate": _r(s.plan_change_rate), "ns_green_share": _r(s.ns_green_share)}
    if s.plan_changes == 0:  # a held plan: report it (computed by the controller, not by the AI layer)
        signal["signal_plans"] = {n: p.label for n, p in run.history[0].plans.items()}
    environmental = {"label": PROXY_LABEL, "fuel_liters_proxy": _r(e.fuel_liters), "co2_kg_proxy": _r(e.co2_kg),
                     "fuel_liters_proxy_per_admitted": _r(e.fuel_liters_per_admitted, 6), "co2_kg_proxy_per_admitted": _r(e.co2_kg_per_admitted, 6),
                     "idle_fuel_rate_lph_assumed": _r(e.estimate.config.idle_fuel_rate_lph, 3),
                     "emission_factor_kg_per_liter_assumed": _r(e.estimate.config.emission_factor_kg_per_liter, 4)}
    comparison = {}
    b = run.baseline_unified
    if b is not None:
        bt = b.traffic
        w = _pct(bt.waiting_seconds_per_admitted, t.waiting_seconds_per_admitted)
        comparison = {"waiting_seconds_per_admitted_pct": w, "waiting_vehicle_seconds_pct": _pct(bt.waiting_vehicle_seconds, t.waiting_vehicle_seconds),
                      "throughput_pct": _pct(bt.throughput_vehicles_per_hour, t.throughput_vehicles_per_hour),
                      "average_queue_pct": _pct(bt.average_queue_vehicles, t.average_queue_vehicles),
                      "max_queue_pct": _pct(bt.max_queue_vehicles, t.max_queue_vehicles),
                      "vehicles_rejected_pct": _pct(bt.vehicles_rejected, t.vehicles_rejected),
                      "final_queue_pct": _pct(bt.final_queue_vehicles, t.final_queue_vehicles), "seeds_compared": 1,
                      "seeds_better": int(w is not None and w < -1e-9), "seeds_worse": int(w is not None and w > 1e-9)}
        environmental["co2_proxy_pct_vs_baseline"] = _pct(b.environmental.co2_kg, e.co2_kg)
        environmental["waiting_pct_vs_baseline"] = comparison["waiting_vehicle_seconds_pct"]
    optimization = {}
    if run.qubo_energy is not None:
        optimization = {"qubo_energy": _r(run.qubo_energy, 3), "exact_energy": _r(run.qubo_energy, 3), "method": "exact enumeration of the QUBO"}
    return TrafficAnalysisContext(
        scenario=run.config.scenario, controller=run.spec.summary_name, baseline_controller="fixed" if b is not None else None,
        seeds=(run.config.seed,), horizon_cycles=run.n_cycles, traffic=traffic, signal=signal, baseline_comparison=comparison,
        environmental=environmental, optimization=optimization)


def live_emergency_context(run: DashboardRun) -> TrafficAnalysisContext | None:
    """Emergency A/B context from the live corridor / no-override pair; None when no emergency was simulated."""
    if run.emergency is None or run.emergency_off is None or run.unified_off is None:
        return None
    on, off = run.emergency.vehicle, run.emergency_off.vehicle
    u_on, u_off = run.unified.traffic, run.unified_off.traffic
    worse = int(u_on.waiting_vehicle_seconds > u_off.waiting_vehicle_seconds + 1e-9)
    better = int(u_on.waiting_vehicle_seconds < u_off.waiting_vehicle_seconds - 1e-9)
    em = {"route": list(on.route), "seeds": 1, "free_flow_time_s": _r(on.free_flow_time, 1),
          "travel_time_s_no_override": _r(off.travel_time, 1), "travel_time_s_corridor": _r(on.travel_time, 1),
          "delay_s_no_override": _r(off.delay, 1), "delay_s_corridor": _r(on.delay, 1), "stops_no_override": _r(off.stops, 2),
          "stops_corridor": _r(on.stops, 2), "prioritized_intersections": _r(on.priority_intersections, 2),
          "normal_waiting_pct_corridor_vs_no_override": _pct(u_off.waiting_vehicle_seconds, u_on.waiting_vehicle_seconds),
          "seeds_normal_waiting_worse": worse, "seeds_normal_waiting_better": better,
          "max_queue_no_override": _r(u_off.max_queue_vehicles, 2), "max_queue_corridor": _r(u_on.max_queue_vehicles, 2),
          "final_queue_no_override": _r(u_off.final_queue_vehicles, 2), "final_queue_corridor": _r(u_on.final_queue_vehicles, 2),
          "co2_proxy_pct_corridor_vs_no_override": _pct(run.unified_off.environmental.co2_kg, run.unified.environmental.co2_kg)}
    if on.completion_cycle is not None:
        log = [r for r in run.emergency.log if r.cycle > on.completion_cycle]
        if log:
            em["cycles_after_completion_checked"] = len(log)
            em["cycles_after_completion_with_override_active"] = sum(1 for r in log if r.priority_nodes)
            em["plans_restored_to_base_controller"] = all(r.final_plans == r.base_plans for r in log)
    return TrafficAnalysisContext(scenario=run.config.scenario, controller=f"{run.spec.summary_name}+emergency_corridor",
                                  baseline_controller=f"{run.spec.summary_name} (no override)", seeds=(run.config.seed,),
                                  horizon_cycles=run.n_cycles, emergency=em)


# ------------------------------------------------------------------------------------------- explanations (existing Phase 7 layer)
AI_ACTIONS = ("state", "optimization", "emergency", "environment")
AI_ACTION_LABELS = {"state": "Explain current state", "optimization": "Explain optimization", "emergency": "Explain emergency",
                    "environment": "Explain environmental impact"}


def explain_action(action: str, run: DashboardRun, saved: SavedResults | None, explainer: Explainer, results_dir=None) -> list:
    """Run one of the four analyst actions through the EXISTING Phase 7 explanation functions. Returns
    ``[(title, ExplanationResult)]``. Never raises because Featherless is missing: the layer falls back on its own."""
    if action == "state":
        return [("Current state", explain_comparison(live_context(run), explainer))]
    if action == "optimization":
        out = [("Optimization", explain_optimization(live_context(run), explainer))]
        if results_dir is not None and saved is not None and saved.qaoa_row(run.config.scenario, "qaoa_p1") is not None:
            try:
                out.append(("QAOA experiment (saved Phase 4, seed 0)", explain_qaoa(load_qaoa_context(results_dir, run.config.scenario, "qaoa_p1"), explainer)))
            except (OSError, LookupError):
                pass
        return out
    if action == "emergency":
        ctx = live_emergency_context(run)
        if ctx is None:
            raise DashboardConfigError("enable the emergency and run the simulation first")
        return [("Emergency corridor", explain_emergency(ctx, explainer))]
    if action == "environment":
        return [("Environmental proxy", explain_environment(live_context(run), explainer))]
    raise DashboardConfigError(f"unknown analyst action {action!r}")


def badge_for(result) -> tuple[str, str, str]:
    """(badge text, css class, one-line explanation). A missing Featherless configuration is normal, not an error."""
    if result.source == "featherless":
        return "FEATHERLESS", "ok", f"generated by {result.model} and validated; numbers come from the simulator"
    reason = result.fallback_reason
    detail = {"missing_api_key": "Featherless is not configured", "missing_model": "no Featherless model is configured"}.get(
        reason, f"Featherless unavailable ({reason})")
    return "LOCAL FALLBACK", "local", f"{detail}. Deterministic analysis built from the same measured numbers"
