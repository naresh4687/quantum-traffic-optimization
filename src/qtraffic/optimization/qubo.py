r"""QUBO formulation of the six-intersection signal-plan selection problem.

Nothing here is quantum: this module only turns a traffic state into a Quadratic
Unconstrained Binary Optimisation problem, and back. The simulator stays the source of
truth for real traffic performance; **QUBO energy is a surrogate, not a waiting time.**

Variables (one-hot, 18 binaries)
--------------------------------
For intersection ``i`` (in ``network.nodes`` order) and plan ``p`` in {A, B, C}::

    A = NS20/EW40      B = NS30/EW30      C = NS40/EW20       (= signals.VALID_PLANS)
    x[i,p] = 1  <=>  intersection i runs plan p          index(i, p) = 3*i + p

There are 2^18 = 262,144 bit strings but only 3^6 = 729 are feasible (exactly one plan
per intersection).

Objective
---------
    E(x) = E_traffic(x) + E_penalty(x)

**Penalty.** ``P * (x[i,A] + x[i,B] + x[i,C] - 1)^2`` per intersection. Using
``x^2 = x`` it expands to::

    P * ( -x[i,A] - x[i,B] - x[i,C]
          + 2 x[i,A]x[i,B] + 2 x[i,A]x[i,C] + 2 x[i,B]x[i,C]  + 1 )

i.e. linear coefficient ``-P``, pair coefficient ``+2P`` and constant ``+P`` per
intersection. It is exactly 0 for a one-hot choice.

*Sufficiency of P (audited in Phase 3.5).* Write the traffic part as
``T(x) = sum_{u<=v} Q_uv x_u x_v`` with **all** its linear (``Q_uu``) and quadratic
(``Q_uv``, ``u != v``, including the neighbour-coupling terms) coefficients, and let::

    B_u = |Q_uu| + sum_{v != u} |Q_uv|        B = max_u B_u

Lemma 1. Flipping a set ``S`` of bits changes ``T`` by at most ``sum_{u in S} B_u``.
Proof: the change is a sum over every monomial touching ``S`` of a coefficient times a
difference of products of bits in {0,1}, so each monomial contributes at most its
absolute coefficient; a monomial ``Q_uv x_u x_v`` with both ends in ``S`` is counted in
``B_u`` *and* ``B_v``, so the bound only over-counts. Nothing assumes the coefficients
are non-negative, sparse, or restricted to different intersections.

Claim. If ``P > B``, every global minimiser of ``E`` is one-hot at every intersection.
Take any bit string and an intersection ``i`` with ``k = x[i,A] + x[i,B] + x[i,C] != 1``.
Only ``i``'s own penalty term depends on ``i``'s bits, so:

* ``k = 0``: switch one bit of ``i`` on. The penalty falls from ``P`` to 0 and, by
  Lemma 1, ``T`` rises by at most ``B``: net change ``<= B - P < 0``;
* ``k >= 2``: switch ``k - 1`` of ``i``'s bits off, keeping one. The penalty falls from
  ``P (k-1)^2 >= P (k-1)`` to 0 and ``T`` changes by at most ``(k-1) B``: net change
  ``<= (k-1)(B - P) < 0``.

Each such repair strictly lowers ``E`` and cannot raise any other intersection's
penalty. Repeating it terminates at a one-hot string with strictly lower energy, so no
infeasible string can be a global minimum, nor can it tie a feasible optimum.
``build_qubo`` uses ``P = B + 1`` (energies are in vehicle-seconds, so 1 is a small
margin). The bound is deterministic and coarse; it is deliberately not tuned. Tests check
the repair step exhaustively (for every infeasible string of the 2^18 an energy-lowering
one-intersection repair exists: ``check_repair_lemma``), and on random signed dense QUBOs.
A bound built from linear coefficients alone (``max_u |Q_uu|``) is NOT sufficient in
general, so the coupling terms are required. A large P has a cost: it stretches the
energy range, which is bad for heuristic solvers (relevant for QAOA later), so it is
reported.

**Traffic cost** (a two-cycle look-ahead built from the simulator's own rules). The
simulator charges each approach ``a`` a waiting time of ``CYCLE * (q0 - released / 2)``
vehicle-seconds per cycle (queue assumed to fall linearly from ``q0`` to
``q0 - released``), where ``q0`` is the queue after arrivals, and

    released = min( q0,  s * green,  space_downstream )

with ``s`` the saturation flow (veh/s of green) and ``space_downstream`` the free room
on the road ahead. ``q0`` does not depend on this cycle's plan, so the plan matters only
through ``green`` and through the downstream space. Notation (a = an approach, ``d(a)``
its downstream approach, ``f(a)`` its feeder, ``g_a(p)`` the green that plan ``p`` gives
``a``'s phase, ``q0`` the queue after this cycle's arrivals)::

    cap_a(p)     = s * g_a(p)                      discharge capacity
    L_a(p)       = q0_a - min(q0_a, cap_a(p))       queue left at the end of cycle 1
    space_a(p)   = max(0, road_capacity_a - L_a(p)) room a leaves for its feeder
    v_f(p_f,p_a) = min(q0_f, cap_f(p_f), space_a(p_a))     vehicles f releases into a
                   (no space limit if f leaves the network)

    cycle 1:  W1_f = CYCLE * (q0_f - v_f / 2)                   waiting while f is served
    cycle 2:  Q2_a = L_a(p_a) + inflow_a      inflow = v_f (fed) or external arrivals
              W2_a = CYCLE * (Q2_a - min(Q2_a, s * g_a(p_a)) / 2)
              (the chosen plan is held for cycle 2, exactly as the plan is applied in the
               simulator; blocking in cycle 2 is ignored)

    E_traffic = sum_f W1_f + sum_a W2_a          [vehicle-seconds]

*Coupling.* ``v_f`` depends on the plan of ``f``'s intersection **and** of the
intersection it discharges into (through ``space``), and ``W2`` of the downstream
approach depends on both. Every term therefore involves at most two intersections
(an edge of the road network), so ``E_traffic`` is exactly a sum of unary tables
``u_i[p]`` and pairwise tables ``T_ij[p, p']`` over neighbouring intersections. In QUBO
form: ``Q[(i,p),(i,p)] += u_i[p]`` and ``Q[(i,p),(j,p')] += T_ij[p,p']``. A plan that
sends vehicles into a nearly full or already long downstream queue pays for it twice:
its release is capped by ``space`` (cycle 1) and the vehicles wait behind ``L_d``
(cycle 2).

*Why the plan is held in cycle 2.* A first version assumed the neutral 30/30 plan in
cycle 2 (``CostConfig(horizon_plan="reference")``, still available). That silently
caps every downstream approach at 15 vehicles per cycle, so any release above 15 into
it was charged as if it could not be served, biasing the optimum towards 30/30 (with
exact ties at every intersection in directional traffic) and adding an arbitrary
constant (the reference plan). Holding the chosen plan removes both. The v1 results
are kept in ``results/phase3_ablation_reference_plan``.

*Approximations (the surrogate is not the simulator).* The downstream leftover ``L_d``
ignores that ``d`` may itself be blocked; cycle 2 ignores blocking; no demand beyond the
``expected_arrivals`` supplied is known; the future is two cycles long. All coefficients are in vehicle-seconds and follow from the
simulator's rules and the state: there are no fitted weights. When nothing is blocked
the surrogate reproduces the simulator's two-cycle waiting exactly (tested).

Ising form: substituting ``x = (1 - z)/2`` with ``z`` in {+1, -1} gives
``E = c + sum_u h_u z_u + sum_{u<v} J_uv z_u z_v`` (see ``QUBO.to_ising``); ``z = +1``
corresponds to bit 0 (the |0> state).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..controllers.base import Observation
from ..network import Approach, Network
from ..signals import (
    CYCLE_LENGTH, PLAN_NS20_EW40, PLAN_NS30_EW30, PLAN_NS40_EW20, VALID_PLANS, SignalPlan,
)
from ..simulator import SimulationConfig

PLANS: tuple[SignalPlan, ...] = VALID_PLANS
PLAN_LABELS: tuple[str, ...] = ("A", "B", "C")
N_PLANS = len(PLANS)
assert PLANS == (PLAN_NS20_EW40, PLAN_NS30_EW30, PLAN_NS40_EW20)  # A, B, C as documented

Bits = Sequence[int]


class InfeasibleAssignment(ValueError):
    """A bit string that is not one-hot at every intersection."""


# -- variables -------------------------------------------------------------------------
@dataclass(frozen=True)
class Variable:
    index: int
    node: str
    plan_index: int

    @property
    def plan(self) -> SignalPlan:
        return PLANS[self.plan_index]

    @property
    def label(self) -> str:
        return PLAN_LABELS[self.plan_index]

    @property
    def name(self) -> str:
        return f"x[{self.node},{self.label}]"


class VariableMap:
    """Bijection between (intersection, plan) and the flat binary index ``3*i + p``."""

    def __init__(self, nodes: Sequence[str]):
        self.nodes: tuple[str, ...] = tuple(nodes)
        if len(set(self.nodes)) != len(self.nodes) or not self.nodes:
            raise ValueError("nodes must be non-empty and unique")
        self.variables: tuple[Variable, ...] = tuple(
            Variable(N_PLANS * i + p, node, p)
            for i, node in enumerate(self.nodes)
            for p in range(N_PLANS)
        )
        self._node_index = {n: i for i, n in enumerate(self.nodes)}

    @property
    def n_variables(self) -> int:
        return len(self.variables)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(v.name for v in self.variables)

    def node_index(self, node: str) -> int:
        return self._node_index[node]

    def index(self, node: str, plan_index: int) -> int:
        if not 0 <= plan_index < N_PLANS:
            raise ValueError(f"plan_index must be in 0..{N_PLANS - 1}")
        return N_PLANS * self._node_index[node] + plan_index

    def node_bits(self, node: str) -> range:
        i = self._node_index[node]
        return range(N_PLANS * i, N_PLANS * (i + 1))


# -- traffic state and cost --------------------------------------------------------------
@dataclass(frozen=True)
class TrafficState:
    """What the QUBO is built from: a snapshot of the simulator's state.

    ``expected_arrivals`` are external vehicles expected per cycle at entry approaches
    (default none). Queues/in-transit are exactly those of ``controllers.Observation``.
    """

    network: Network
    queues: Mapping[Approach, float]
    in_transit: Mapping[Approach, float]
    expected_arrivals: Mapping[Approach, float]
    cycle: int = 0

    @classmethod
    def from_observation(
        cls, observation: Observation, expected_arrivals: Mapping[Approach, float] | None = None
    ) -> TrafficState:
        return cls(
            observation.network,
            dict(observation.queues),
            dict(observation.in_transit),
            dict(expected_arrivals or {}),
            observation.cycle,
        )


@dataclass(frozen=True)
class CostConfig:
    """Parameters of the surrogate cost. Defaults reuse the simulator's / controller's."""

    saturation_flow: float = SimulationConfig().saturation_flow  # veh per second of green
    downstream_lookahead: bool = True  # include cycle-2 waiting (W2)
    horizon_plan: str = "held"  # plan in cycle 2: "held" (the chosen one) or "reference"
    reference_plan: SignalPlan = PLAN_NS30_EW30  # used only when horizon_plan == "reference"

    def __post_init__(self) -> None:
        if not (math.isfinite(self.saturation_flow) and self.saturation_flow > 0):
            raise ValueError("saturation_flow must be finite and > 0")
        if self.reference_plan not in PLANS:
            raise ValueError("reference_plan must be one of the three valid plans")
        if self.horizon_plan not in ("held", "reference"):
            raise ValueError('horizon_plan must be "held" or "reference"')


def _q0(state: TrafficState, a: Approach) -> float:
    """Queue after this cycle's arrivals (simulator step 4)."""
    net = state.network
    q = state.queues[a] + state.in_transit[a]
    if net.is_entry(a):
        q += min(state.expected_arrivals.get(a, 0.0), max(0.0, net.capacity(a) - q))
    return q


def _capacity(a: Approach, plan: SignalPlan, cfg: CostConfig) -> float:
    return cfg.saturation_flow * plan.green_time(a.phase)


def _leftover(state: TrafficState, a: Approach, plan: SignalPlan, cfg: CostConfig) -> float:
    q0 = _q0(state, a)
    return q0 - min(q0, _capacity(a, plan, cfg))


def _space(state: TrafficState, a: Approach, plan: SignalPlan, cfg: CostConfig) -> float:
    return max(0.0, state.network.capacity(a) - _leftover(state, a, plan, cfg))


def _released(
    state: TrafficState, f: Approach, plan_f: SignalPlan, plan_down: SignalPlan | None, cfg: CostConfig
) -> float:
    wanted = min(_q0(state, f), _capacity(f, plan_f, cfg))
    down = state.network.downstream_approach(f)
    if down is None:
        return wanted
    assert plan_down is not None
    return min(wanted, _space(state, down, plan_down, cfg))


def _w1(state: TrafficState, f: Approach, plan_f: SignalPlan, plan_down: SignalPlan | None,
        cfg: CostConfig) -> float:
    return CYCLE_LENGTH * (_q0(state, f) - _released(state, f, plan_f, plan_down, cfg) / 2.0)


def _external_inflow(state: TrafficState, a: Approach, plan: SignalPlan, cfg: CostConfig) -> float:
    net = state.network
    return min(state.expected_arrivals.get(a, 0.0), max(0.0, net.capacity(a) - _leftover(state, a, plan, cfg)))


def _w2(state: TrafficState, a: Approach, plan_a: SignalPlan, inflow: float, cfg: CostConfig) -> float:
    if not cfg.downstream_lookahead:
        return 0.0
    q2 = _leftover(state, a, plan_a, cfg) + inflow
    plan2 = plan_a if cfg.horizon_plan == "held" else cfg.reference_plan
    return CYCLE_LENGTH * (q2 - min(q2, _capacity(a, plan2, cfg)) / 2.0)


def surrogate_cost(
    state: TrafficState, plans: Mapping[str, SignalPlan], config: CostConfig | None = None
) -> float:
    """``E_traffic`` of a complete plan assignment, straight from the definition."""
    cfg = config or CostConfig()
    net = state.network
    total = 0.0
    for f in net.approaches:
        down = net.downstream_approach(f)
        total += _w1(state, f, plans[f.node], None if down is None else plans[down.node], cfg)
    for a in net.approaches:
        feeder = net.upstream_approach(a)
        if feeder is None:
            inflow = _external_inflow(state, a, plans[a.node], cfg)
        else:
            inflow = _released(state, feeder, plans[feeder.node], plans[a.node], cfg)
        total += _w2(state, a, plans[a.node], inflow, cfg)
    return total


@dataclass(frozen=True)
class TrafficCosts:
    """``E_traffic`` decomposed into unary tables ``u_i[p]`` and edge tables ``T_ij[p,p']``.

    ``pair`` keys are ``(i, j)`` node indices with ``i < j``; a table is indexed
    ``[plan of i, plan of j]``.
    """

    unary: Mapping[int, np.ndarray]
    pair: Mapping[tuple[int, int], np.ndarray]


def traffic_costs(
    state: TrafficState, variables: VariableMap, config: CostConfig | None = None
) -> TrafficCosts:
    cfg = config or CostConfig()
    net = state.network
    unary = {i: np.zeros(N_PLANS) for i in range(len(variables.nodes))}
    pair: dict[tuple[int, int], np.ndarray] = {}
    for f in net.approaches:
        i = variables.node_index(f.node)
        down = net.downstream_approach(f)
        if down is None:
            for p, plan_f in enumerate(PLANS):
                unary[i][p] += _w1(state, f, plan_f, None, cfg)
            continue
        j = variables.node_index(down.node)
        table = np.zeros((N_PLANS, N_PLANS))  # [plan of f's node, plan of down's node]
        for p, plan_f in enumerate(PLANS):
            for r, plan_d in enumerate(PLANS):
                v = _released(state, f, plan_f, plan_d, cfg)
                table[p, r] = _w1(state, f, plan_f, plan_d, cfg) + _w2(state, down, plan_d, v, cfg)
        key, table = ((i, j), table) if i < j else ((j, i), table.T)
        pair[key] = pair.get(key, 0.0) + table
    for a in net.entry_approaches:
        i = variables.node_index(a.node)
        for p, plan_a in enumerate(PLANS):
            unary[i][p] += _w2(state, a, plan_a, _external_inflow(state, a, plan_a, cfg), cfg)
    return TrafficCosts(unary, pair)


# -- the QUBO --------------------------------------------------------------------------
@dataclass(frozen=True)
class IsingModel:
    """``E(z) = offset + sum_u h[u] z_u + sum_{u<v} J[(u,v)] z_u z_v``, ``z`` in {+1, -1}."""

    h: np.ndarray
    J: Mapping[tuple[int, int], float]
    offset: float

    def energy(self, z: Sequence[int]) -> float:
        z = np.asarray(z, dtype=float)
        if not np.all(np.abs(z) == 1):
            raise ValueError("spins must be +1 or -1")
        return float(self.offset + self.h @ z + sum(c * z[u] * z[v] for (u, v), c in self.J.items()))


@dataclass(frozen=True)
class QUBO:
    """``E(x) = offset + sum_{u<=v} coefficients[(u,v)] x_u x_v`` over binary ``x``.

    Diagonal entries ``(u,u)`` are the linear terms (``x_u^2 = x_u``). ``traffic`` holds
    the traffic part alone (used for the penalty bound); ``coefficients`` and ``offset``
    include the one-hot penalty.
    """

    variables: VariableMap
    traffic: Mapping[tuple[int, int], float]
    coefficients: Mapping[tuple[int, int], float]
    offset: float
    penalty: float

    # -- evaluation --------------------------------------------------------------
    def matrix(self) -> np.ndarray:
        """Upper-triangular ``Q`` such that ``E(x) = offset + x^T Q x``."""
        n = self.variables.n_variables
        q = np.zeros((n, n))
        for (u, v), c in self.coefficients.items():
            q[u, v] += c
        return q

    def _bits(self, x: Bits) -> np.ndarray:
        arr = np.asarray(x)
        if arr.shape != (self.variables.n_variables,) or not np.all((arr == 0) | (arr == 1)):
            raise ValueError(f"expected {self.variables.n_variables} bits, each 0 or 1")
        return arr.astype(float)

    def energy(self, x: Bits) -> float:
        bits = self._bits(x)
        return float(self.offset + bits @ self.matrix() @ bits)

    # -- feasibility / decoding --------------------------------------------------
    def violations(self, x: Bits) -> dict[str, int]:
        """Intersections whose one-hot constraint fails, mapped to their bit count."""
        bits = self._bits(x)
        return {
            n: int(bits[list(self.variables.node_bits(n))].sum())
            for n in self.variables.nodes
            if bits[list(self.variables.node_bits(n))].sum() != 1
        }

    def is_feasible(self, x: Bits) -> bool:
        return not self.violations(x)

    def decode(self, x: Bits) -> dict[str, SignalPlan]:
        """Plans chosen by a feasible bit string; raises ``InfeasibleAssignment`` otherwise."""
        bad = self.violations(x)
        if bad:
            raise InfeasibleAssignment(f"one-hot violated at {bad}")
        bits = self._bits(x)
        return {
            n: PLANS[int(np.argmax(bits[list(self.variables.node_bits(n))]))]
            for n in self.variables.nodes
        }

    def encode(self, plans: Mapping[str, SignalPlan]) -> tuple[int, ...]:
        if set(plans) != set(self.variables.nodes):
            raise ValueError("plans must cover exactly the QUBO's intersections")
        bits = [0] * self.variables.n_variables
        for node, plan in plans.items():
            bits[self.variables.index(node, PLANS.index(plan))] = 1
        return tuple(bits)

    # -- Ising ---------------------------------------------------------------------
    def to_ising(self) -> IsingModel:
        n = self.variables.n_variables
        h = np.zeros(n)
        J: dict[tuple[int, int], float] = {}
        offset = self.offset
        for (u, v), c in self.coefficients.items():
            if u == v:  # c x_u = c (1 - z_u) / 2
                offset += c / 2.0
                h[u] -= c / 2.0
            else:  # c x_u x_v = c (1 - z_u - z_v + z_u z_v) / 4
                offset += c / 4.0
                h[u] -= c / 4.0
                h[v] -= c / 4.0
                J[(u, v)] = J.get((u, v), 0.0) + c / 4.0
        return IsingModel(h, J, offset)

    # -- reporting -------------------------------------------------------------------
    def as_dict(self) -> dict:
        return {
            "variables": [{"index": v.index, "name": v.name, "plan": v.plan.label}
                          for v in self.variables.variables],
            "penalty": self.penalty,
            "offset": self.offset,
            "coefficients": [{"u": u, "v": v, "value": c}
                             for (u, v), c in sorted(self.coefficients.items())],
        }


def penalty_lower_bound(traffic: Mapping[tuple[int, int], float], n_variables: int) -> float:
    """``B = max_u B_u`` with ``B_u`` the total absolute traffic coefficient on bit ``u``."""
    touch = np.zeros(n_variables)
    for (u, v), c in traffic.items():
        touch[u] += abs(c)
        if u != v:
            touch[v] += abs(c)
    return float(touch.max())


def build_qubo(
    state: TrafficState,
    config: CostConfig | None = None,
    penalty: float | None = None,
) -> QUBO:
    """Assemble the complete QUBO for ``state`` (deterministic; no hidden state).

    ``penalty`` defaults to ``B + 1`` (see module docstring); pass a value only to study
    an insufficient one.
    """
    variables = VariableMap(state.network.nodes)
    costs = traffic_costs(state, variables, config)

    traffic: dict[tuple[int, int], float] = {}
    for i, table in costs.unary.items():
        for p in range(N_PLANS):
            u = N_PLANS * i + p
            traffic[(u, u)] = traffic.get((u, u), 0.0) + float(table[p])
    for (i, j), table in costs.pair.items():
        for p in range(N_PLANS):
            for r in range(N_PLANS):
                key = (N_PLANS * i + p, N_PLANS * j + r)
                traffic[key] = traffic.get(key, 0.0) + float(table[p, r])

    return qubo_from_traffic(variables, traffic, penalty)


def qubo_from_traffic(
    variables: VariableMap,
    traffic: Mapping[tuple[int, int], float],
    penalty: float | None = None,
) -> QUBO:
    """Add the one-hot penalty to arbitrary traffic coefficients ``traffic[(u, v)]``, ``u <= v``.

    ``penalty`` defaults to ``B + 1`` where ``B`` is ``penalty_lower_bound(traffic)``. The
    bound holds for any real linear and quadratic coefficients (signed, dense, or with
    terms inside one intersection), which is what makes it a proof and not a tuning.
    """
    if any(u > v for u, v in traffic):
        raise ValueError("traffic coefficient keys must satisfy u <= v")
    traffic = dict(traffic)
    if penalty is None:
        penalty = penalty_lower_bound(traffic, variables.n_variables) + 1.0
    if not (math.isfinite(penalty) and penalty > 0):
        raise ValueError("penalty must be finite and > 0")

    coefficients = dict(traffic)
    for node in variables.nodes:  # P * (sum_p x_p - 1)^2
        bits = list(variables.node_bits(node))
        for u in bits:
            coefficients[(u, u)] = coefficients.get((u, u), 0.0) - penalty
        for a in range(len(bits)):
            for b in range(a + 1, len(bits)):
                key = (bits[a], bits[b])
                coefficients[key] = coefficients.get(key, 0.0) + 2.0 * penalty
    offset = penalty * len(variables.nodes)
    return QUBO(variables, traffic, coefficients, offset, float(penalty))
