"""Exact classical solver for the signal-plan QUBO: exhaustive enumeration.

This is *not* quantum optimisation. It is the ground-truth reference (the true minimum of
the QUBO) that a later heuristic/quantum solver must be compared against.

``solve_exact`` enumerates the 3^6 = 729 one-hot assignments directly. ``check_full_space``
enumerates all 2^18 = 262,144 bit strings, which independently confirms that the
penalty is large enough (no infeasible string beats the feasible optimum).

Ties. Several assignments can have the same energy (for instance every assignment when
the network is empty). The winner is then chosen deterministically: among assignments
within ``TIE_TOL`` of the minimum, the one with the fewest intersections deviating from
the neutral plan B (NS30/EW30), then the first in enumeration order. ``n_optimal``
reports how many assignments tied.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from ..signals import SignalPlan
from .qubo import N_PLANS, PLANS, QUBO

NEUTRAL_PLAN_INDEX = 1  # plan B = NS30/EW30
TIE_TOL = 1e-9  # relative to max(1, |minimum energy|)


@dataclass(frozen=True)
class ExactSolution:
    assignment: tuple[int, ...]  # the 18 bits
    plans: dict[str, SignalPlan]
    energy: float
    feasible: bool  # one-hot valid at every intersection
    n_evaluated: int  # assignments whose energy was computed
    n_optimal: int  # assignments tied with the minimum (within TIE_TOL)


@dataclass(frozen=True)
class FeasibleSpace:
    """Every feasible assignment and its QUBO energy (rows align)."""

    assignments: np.ndarray  # (3^n, 3n) of 0/1
    plan_indices: np.ndarray  # (3^n, n) plan index per intersection
    energies: np.ndarray  # (3^n,)


@dataclass(frozen=True)
class FullSpaceCheck:
    n_evaluated: int  # 2^18
    min_energy: float  # over all bit strings
    min_energy_feasible: float  # over one-hot strings only
    min_is_feasible: bool  # the global minimum is a valid plan assignment
    n_feasible: int
    n_infeasible_at_or_below_optimum: int  # infeasible strings with energy <= feasible optimum


def _energies(qubo: QUBO, bits: np.ndarray) -> np.ndarray:
    x = bits.astype(float)
    return qubo.offset + ((x @ qubo.matrix()) * x).sum(axis=1)


def enumerate_feasible(qubo: QUBO) -> FeasibleSpace:
    """All 3^n one-hot assignments (in ``itertools.product`` order) with their energies."""
    n_nodes = len(qubo.variables.nodes)
    plan_indices = np.array(list(itertools.product(range(N_PLANS), repeat=n_nodes)), dtype=np.int64)
    bits = np.zeros((len(plan_indices), N_PLANS * n_nodes), dtype=np.int64)
    rows = np.arange(len(plan_indices))[:, None]
    bits[rows, N_PLANS * np.arange(n_nodes)[None, :] + plan_indices] = 1
    return FeasibleSpace(bits, plan_indices, _energies(qubo, bits))


def solve_exact(qubo: QUBO) -> ExactSolution:
    """Minimum-energy valid assignment by exhaustive enumeration of the one-hot space."""
    space = enumerate_feasible(qubo)
    e_min = float(space.energies.min())
    tied = np.flatnonzero(space.energies <= e_min + TIE_TOL * max(1.0, abs(e_min)))
    deviations = (space.plan_indices[tied] != NEUTRAL_PLAN_INDEX).sum(axis=1)
    best = int(tied[np.argmin(deviations)])  # argmin returns the first minimum: stable
    bits = tuple(int(b) for b in space.assignments[best])
    solution = ExactSolution(
        assignment=bits,
        plans=qubo.decode(bits),
        energy=float(space.energies[best]),
        feasible=qubo.is_feasible(bits),
        n_evaluated=len(space.energies),
        n_optimal=len(tied),
    )
    if not solution.feasible:  # cannot happen by construction; fail loudly if it ever does
        raise AssertionError("exact solver produced an infeasible assignment")
    return solution


def check_repair_lemma(qubo: QUBO) -> int:
    """Exhaustively test the penalty proof's repair step over all 2^n bit strings.

    For every infeasible string, take its first violating intersection and try the
    repairs from the proof: ``k = 0`` -> switch one bit on; ``k >= 2`` -> keep one of the
    set bits and clear the others. Returns how many infeasible strings have **no** repair
    that strictly lowers the energy (0 confirms the proof's mechanism for this QUBO).
    """
    n, nodes = qubo.variables.n_variables, qubo.variables.nodes
    idx = np.arange(2**n)
    bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)
    energies = _energies(qubo, bits)
    per_node = bits.reshape(len(bits), len(nodes), N_PLANS).sum(axis=2)
    remaining = ~(per_node == 1).all(axis=1)  # infeasible strings not yet shown repairable
    handled = np.zeros(len(idx), dtype=bool)
    for i, node in enumerate(nodes):
        mask = sum(1 << u for u in qubo.variables.node_bits(node))
        violates = per_node[:, i] != 1
        first = violates & ~handled  # the first violating intersection is i
        handled |= violates
        best = np.full(len(idx), np.inf)
        for p in range(N_PLANS):
            target = 1 << qubo.variables.index(node, p)
            k0 = first & (per_node[:, i] == 0)  # switch bit p on
            best[k0] = np.minimum(best[k0], energies[idx[k0] | target])
            k2 = first & (per_node[:, i] >= 2) & ((idx & target) != 0)  # keep bit p only
            best[k2] = np.minimum(best[k2], energies[(idx[k2] & ~mask) | target])
        remaining[first] &= ~(best[first] < energies[first])
    return int(remaining.sum())


def check_full_space(qubo: QUBO) -> FullSpaceCheck:
    """Evaluate all 2^n bit strings and compare the global minimum with the feasible one."""
    n = qubo.variables.n_variables
    bits = ((np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)
    energies = _energies(qubo, bits)
    per_node = bits.reshape(len(bits), len(qubo.variables.nodes), N_PLANS).sum(axis=2)
    feasible = (per_node == 1).all(axis=1)
    best_feasible = float(energies[feasible].min())
    tol = TIE_TOL * max(1.0, abs(best_feasible))
    return FullSpaceCheck(
        n_evaluated=len(energies),
        min_energy=float(energies.min()),
        min_energy_feasible=best_feasible,
        min_is_feasible=bool(energies.min() >= best_feasible - tol),
        n_feasible=int(feasible.sum()),
        n_infeasible_at_or_below_optimum=int((energies[~feasible] <= best_feasible + tol).sum()),
    )
