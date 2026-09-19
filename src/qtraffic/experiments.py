"""Fixed-time vs adaptive experiments on identical traffic.

Every (scenario, seed) pair is run on the same network, from the same empty initial
state, with the same demand source, for the same number of cycles, once per controller:
``FixedTimeController``, ``AdaptiveController`` (no hysteresis) and the same rule with
a 2-cycle minimum dwell (the tried-and-rejected hysteresis, kept as an ablation). All numbers come from the simulator's own records;
nothing is hard-coded.

Demand identity is verified in ``run_pair``: the demand source is queried directly, and
the demand each controller's simulator run actually consumed must match that (and so
each other) via a SHA-256 digest, otherwise the run raises.
"""

from __future__ import annotations

import hashlib
import statistics
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .controllers import AdaptiveConfig, AdaptiveController, FixedTimeController
from .demand import DEMAND_LEVELS, DemandConfig, DemandModel, DemandSource
from .metrics import CycleRecord
from .network import Approach, Heading, Network, grid_network
from .signals import VALID_PLANS
from .simulator import SimulationResult, Simulator

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_CYCLES = 120  # 2 hours of 60 s cycles
# Compared against "fixed". "adaptive" is the primary controller (default config);
# "adaptive_dwell2" adds a 2-cycle minimum-dwell hysteresis (ablation; it measured worse).
CANDIDATES: tuple[str, ...] = ("adaptive", "adaptive_dwell2")
HEAVY, LIGHT = 1.5, 0.5  # directional multipliers on the medium base rate (mean stays 12)

# metric -> True when a larger value is better
METRICS: dict[str, bool] = {
    "vehicles_entered": True,
    "vehicles_rejected": False,
    "vehicles_exited": True,
    "throughput_per_hour": True,
    "total_waiting_time": False,
    "average_queue_length": False,
    "max_queue_length": False,
    "final_queue_total": False,
    "total_blocked": False,
}


# -- demand ---------------------------------------------------------------------
class TimeVaryingDemand:
    """Piecewise demand: ``segments`` is ``[(start_cycle, DemandSource), ...]`` by start."""

    def __init__(self, segments: Sequence[tuple[int, DemandSource]]):
        if not segments or segments[0][0] != 0:
            raise ValueError("first segment must start at cycle 0")
        starts = [s for s, _ in segments]
        if starts != sorted(set(starts)):
            raise ValueError("segment start cycles must be strictly increasing")
        self.segments = list(segments)

    def arrivals(self, cycle: int) -> Mapping[Approach, float]:
        source = next(src for start, src in reversed(self.segments) if start <= cycle)
        return source.arrivals(cycle)


def directional_multipliers(network: Network, ns: float, ew: float) -> dict[Approach, float]:
    return {
        a: (ns if a.heading in (Heading.N, Heading.S) else ew) for a in network.entry_approaches
    }


def _directional(network: Network, seed: int, ns: float, ew: float) -> DemandModel:
    cfg = DemandConfig.from_level("medium", seed=seed, multipliers=directional_multipliers(network, ns, ew))
    return DemandModel(network, cfg)


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    build_demand: Callable[[Network, int], DemandSource]


def _uniform(level: str) -> Callable[[Network, int], DemandSource]:
    return lambda net, seed: DemandModel(net, DemandConfig.from_level(level, seed=seed))


def _time_varying(net: Network, seed: int) -> DemandSource:
    # 30-cycle blocks: NS-heavy, balanced, EW-heavy, balanced (same seed throughout).
    return TimeVaryingDemand([
        (0, _directional(net, seed, HEAVY, LIGHT)),
        (30, _directional(net, seed, 1.0, 1.0)),
        (60, _directional(net, seed, LIGHT, HEAVY)),
        (90, _directional(net, seed, 1.0, 1.0)),
    ])


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("low", f"uniform demand, {DEMAND_LEVELS['low']:g} veh/cycle/entry", _uniform("low")),
    Scenario("medium", f"uniform demand, {DEMAND_LEVELS['medium']:g} veh/cycle/entry", _uniform("medium")),
    Scenario("high", f"uniform demand, {DEMAND_LEVELS['high']:g} veh/cycle/entry", _uniform("high")),
    Scenario(
        "ns_heavy",
        f"medium base; NS entries x{HEAVY}, EW entries x{LIGHT}",
        lambda net, seed: _directional(net, seed, HEAVY, LIGHT),
    ),
    Scenario(
        "ew_heavy",
        f"medium base; EW entries x{HEAVY}, NS entries x{LIGHT}",
        lambda net, seed: _directional(net, seed, LIGHT, HEAVY),
    ),
    Scenario(
        "time_varying",
        "medium base; 30-cycle blocks: NS-heavy, balanced, EW-heavy, balanced",
        _time_varying,
    ),
)


# -- running --------------------------------------------------------------------
def demand_digest(rows: Sequence[Mapping[Approach, float]]) -> str:
    """SHA-256 over the per-cycle, per-entry demand values (exact float reprs)."""
    h = hashlib.sha256()
    for cycle, row in enumerate(rows):
        for a in sorted(row, key=str):
            h.update(f"{cycle}|{a}|{float(row[a])!r}\n".encode())
    return h.hexdigest()


def plan_stats(history: Sequence[CycleRecord], nodes: Sequence[str]) -> dict:
    counts = {p.label: 0 for p in VALID_PLANS}
    switches = 0
    for node in nodes:
        prev = None
        for rec in history:
            plan = rec.plans[node]
            counts[plan.label] += 1
            if prev is not None and plan != prev:
                switches += 1
            prev = plan
    total = sum(counts.values())
    return {
        "plan_share": {k: v / total for k, v in counts.items()},
        "plan_switches": switches,
        "plan_switch_rate": switches / (len(nodes) * (len(history) - 1)) if len(history) > 1 else 0.0,
    }


def summarise(result: SimulationResult, nodes: Sequence[str]) -> dict:
    m = result.metrics.as_dict()
    row = {k: m[k] for k in METRICS}
    row["demand_generated"] = m["demand_generated"]
    row["vehicles_served"] = m["vehicles_served"]
    row["in_transit_final"] = m["in_transit_final"]
    stats = plan_stats(result.history, nodes)
    for label, share in stats["plan_share"].items():
        row[f"share_{label}"] = share
    row["plan_switches"] = stats["plan_switches"]
    row["plan_switch_rate"] = stats["plan_switch_rate"]
    return row


def run_pair(
    scenario: Scenario,
    seed: int,
    cycles: int = DEFAULT_CYCLES,
    network: Network | None = None,
    record_decisions: bool = False,
) -> dict:
    """Run every controller on identical inputs; raise if demand is not identical."""
    net = network or grid_network(2, 3)
    demand = scenario.build_demand(net, seed)
    sim = Simulator(net, demand)

    adaptive_ctrl = AdaptiveController(record=record_decisions)
    controllers = {
        "fixed": FixedTimeController(),
        "adaptive": adaptive_ctrl,
        "adaptive_dwell2": AdaptiveController(AdaptiveConfig(min_hold_cycles=2)),
    }
    results = {name: sim.run(ctrl, cycles) for name, ctrl in controllers.items()}

    source_digest = demand_digest([demand.arrivals(c) for c in range(cycles)])
    for name, res in results.items():
        if demand_digest([r.demand for r in res.history]) != source_digest:
            raise AssertionError(f"demand mismatch for {name} in {scenario.name} seed {seed}")

    return {
        "scenario": scenario.name,
        "seed": seed,
        "cycles": cycles,
        "demand_digest": source_digest,
        **{name: summarise(res, net.nodes) for name, res in results.items()},
        "decisions": adaptive_ctrl.decisions if record_decisions else None,
    }


def pct_diff(fixed: float, adaptive: float) -> float | None:
    """Signed % change of adaptive relative to fixed; None when fixed is 0."""
    if fixed == 0:
        return None
    return (adaptive - fixed) / abs(fixed) * 100.0


def verdict(metric: str, fixed: float, adaptive: float, rel_tol: float = 1e-9) -> str:
    """'better' / 'worse' / 'same' for adaptive versus fixed on one metric."""
    if abs(adaptive - fixed) <= rel_tol * max(1.0, abs(fixed)):
        return "same"
    return "better" if (adaptive > fixed) == METRICS[metric] else "worse"


def run_all(
    scenarios: Sequence[Scenario] = SCENARIOS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    cycles: int = DEFAULT_CYCLES,
    decisions_seed: int | None = None,
) -> list[dict]:
    """Run every scenario x seed; keep adaptive decision traces for ``decisions_seed``."""
    return [
        run_pair(s, seed, cycles, record_decisions=(seed == decisions_seed))
        for s in scenarios
        for seed in seeds
    ]


def aggregate(pairs: Sequence[dict], candidate: str = "adaptive") -> list[dict]:
    """Per scenario: mean over seeds for fixed and ``candidate``, and % difference of the means."""
    out = []
    for name in dict.fromkeys(p["scenario"] for p in pairs):
        group = [p for p in pairs if p["scenario"] == name]
        row: dict = {"scenario": name, "candidate": candidate, "seeds": len(group)}
        for metric, higher_better in METRICS.items():
            f = statistics.fmean(p["fixed"][metric] for p in group)
            a = statistics.fmean(p[candidate][metric] for p in group)
            row[f"{metric}_fixed"] = f
            row[f"{metric}_adaptive"] = a
            row[f"{metric}_pct"] = pct_diff(f, a)
            row[f"{metric}_verdict"] = verdict(metric, f, a)
            wins = sum(verdict(metric, p["fixed"][metric], p[candidate][metric]) == "better" for p in group)
            losses = sum(verdict(metric, p["fixed"][metric], p[candidate][metric]) == "worse" for p in group)
            row[f"{metric}_seeds_better"] = wins
            row[f"{metric}_seeds_worse"] = losses
        for key in ("plan_switch_rate", "share_NS20/EW40", "share_NS30/EW30", "share_NS40/EW20"):
            row[f"candidate_{key}"] = statistics.fmean(p[candidate][key] for p in group)
        out.append(row)
    return out
