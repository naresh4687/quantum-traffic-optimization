"""Classical adaptive controller: queue-pressure (max-pressure style) plan selection.

Decision rule (evaluated independently for every intersection, every cycle)
---------------------------------------------------------------------------
For an approach ``a`` at the intersection, with ``d = downstream_approach(a)``
(``None`` when vehicles leave the network after ``a``)::

    demand(a)   = queue[a] + in_transit[a]        vehicles that will be at the stop line
    downstream  = queue[d] + in_transit[d]  if d exists, else 0
    pressure(a) = demand(a) - beta * downstream

``queue`` and ``in_transit`` come straight from the Observation. Only the approach's
own state and that of its one downstream approach (the road it discharges into) are
used; no other intersection is consulted. The phase pressures and the decision are::

    P_NS  = sum of pressure(a) over the NS approaches (headings N, S)
    P_EW  = sum of pressure(a) over the EW approaches (headings E, W)
    delta = P_NS - P_EW

    delta >  +threshold  ->  NS40 / EW20
    delta <  -threshold  ->  NS20 / EW40
    otherwise            ->  NS30 / EW30

Parameters (see ``AdaptiveConfig``)
-----------------------------------
* ``downstream_weight`` (beta) = 1.0: classical max-pressure weighting. A vehicle
  queued downstream cancels one queued upstream, so green is steered away from
  movements that would only push vehicles into an already-congested road.
* ``switch_threshold`` = 5.0 vehicles: the dead band around balanced pressure. Moving
  10 s of green from one phase to the other shifts ``0.5 veh/s * 10 s = 5`` vehicles
  of discharge capacity per approach (default saturation flow), so a pressure
  difference smaller than that is not worth acting on and 30/30 is kept.
* ``min_hold_cycles`` = 1: no hysteresis. An optional minimum dwell (keep a changed
  plan for at least this many cycles) exists but is OFF by default; see below.

``downstream_weight`` and ``switch_threshold`` were fixed from the reasoning above
before any experiment was run. They are not tuned against results.

Hysteresis: tried, measured, rejected
-------------------------------------
Without hysteresis the rule flip-flops under balanced demand: the plan favouring NS
drains the NS queue and builds the EW queue, so the next cycle's pressure favours EW,
and so on (the pressure difference swings by about +-10 vehicles, larger than the
5-vehicle dead band). The measured plan-switch rate is near 1 per intersection per
cycle at medium/high uniform demand, and it does worse than fixed-time there.

A minimum dwell of 2 cycles (the smallest value that forbids cycle-by-cycle
alternation) was then tried as the hysteresis. It did NOT help: it turns the 1-cycle
flip-flop into a 2-cycle one (switch rate ~0.5) and delays the response to real
imbalance, and it measured worse than no hysteresis on uniform and EW-heavy demand
(results/phase2/summary.csv, candidate ``adaptive_dwell2``). Hysteresis is therefore
not enabled: it is not "necessary" in the sense of curing the oscillation, and it
costs performance. The dwell option is kept only so that ablation stays reproducible.
The oscillation itself is a property of using the start-of-cycle queue snapshot as the
control signal, and is left as an honest limitation of this classical baseline.

Determinism
-----------
Given the same sequence of Observations after ``reset()``, the controller returns the
same plans: there is no randomness, and the only memory is the per-intersection
(current plan, cycles held) used by the optional dwell rule (unused when
``min_hold_cycles == 1``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from ..network import Approach, Network
from ..signals import PLAN_NS20_EW40, PLAN_NS30_EW30, PLAN_NS40_EW20, Phase, SignalPlan
from .base import Controller, Observation


@dataclass(frozen=True)
class AdaptiveConfig:
    downstream_weight: float = 1.0
    switch_threshold: float = 5.0
    min_hold_cycles: int = 1  # 1 = no hysteresis (see module docstring)

    def __post_init__(self) -> None:
        hold = self.min_hold_cycles
        if isinstance(hold, bool) or not isinstance(hold, int) or hold < 1:
            raise ValueError("min_hold_cycles must be an integer >= 1")
        if not (math.isfinite(self.downstream_weight) and self.downstream_weight >= 0):
            raise ValueError("downstream_weight must be finite and >= 0")
        if not (math.isfinite(self.switch_threshold) and self.switch_threshold >= 0):
            raise ValueError("switch_threshold must be finite and >= 0")


@dataclass(frozen=True)
class Decision:
    """One intersection's decision for one cycle, with the numbers behind it."""

    cycle: int
    node: str
    pressure_ns: float
    pressure_ew: float
    delta: float
    plan: SignalPlan  # plan actually applied
    desired: SignalPlan  # plan the pressure rule alone would pick


class AdaptiveController(Controller):
    """Picks NS20/30/40 per intersection from local queue pressure (see module docstring)."""

    def __init__(self, config: AdaptiveConfig | None = None, record: bool = False):
        self.config = config or AdaptiveConfig()
        self.record = record
        self.decisions: list[Decision] = []
        self._held: dict[str, tuple[SignalPlan, int]] = {}  # node -> (plan, cycles in force)

    @property
    def name(self) -> str:
        hold = self.config.min_hold_cycles
        return "AdaptiveController" if hold == 1 else f"AdaptiveController(dwell={hold})"

    def reset(self) -> None:
        self.decisions = []
        self._held = {}

    # -- pressure ------------------------------------------------------------
    def approach_pressure(self, approach: Approach, observation: Observation) -> float:
        net: Network = observation.network
        demand = observation.queues[approach] + observation.in_transit[approach]
        down = net.downstream_approach(approach)
        downstream = 0.0 if down is None else observation.queues[down] + observation.in_transit[down]
        return demand - self.config.downstream_weight * downstream

    def phase_pressures(self, node: str, observation: Observation) -> tuple[float, float]:
        """(P_NS, P_EW) for one intersection."""
        totals = {Phase.NS: 0.0, Phase.EW: 0.0}
        for a in observation.network.approaches:
            if a.node == node:
                totals[a.phase] += self.approach_pressure(a, observation)
        return totals[Phase.NS], totals[Phase.EW]

    def choose_plan(self, delta: float) -> SignalPlan:
        t = self.config.switch_threshold
        if delta > t:
            return PLAN_NS40_EW20
        if delta < -t:
            return PLAN_NS20_EW40
        return PLAN_NS30_EW30

    # -- Controller interface -------------------------------------------------
    def select_plans(self, observation: Observation) -> Mapping[str, SignalPlan]:
        plans: dict[str, SignalPlan] = {}
        for node in observation.network.nodes:
            p_ns, p_ew = self.phase_pressures(node, observation)
            desired = self.choose_plan(p_ns - p_ew)
            plan = desired
            current = self._held.get(node)
            if current is not None:
                held_plan, cycles = current
                # dwell rule: keep the plan until it has been in force min_hold_cycles
                plan = held_plan if cycles < self.config.min_hold_cycles else desired
                self._held[node] = (plan, cycles + 1 if plan == held_plan else 1)
            else:
                self._held[node] = (plan, 1)
            plans[node] = plan
            if self.record:
                self.decisions.append(
                    Decision(observation.cycle, node, p_ns, p_ew, p_ns - p_ew, plan, desired)
                )
        return plans
