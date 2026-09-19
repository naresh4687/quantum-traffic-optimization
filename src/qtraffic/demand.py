"""Reproducible external traffic demand.

Demand enters the network only at entry approaches. For each (cycle, entry) the
arrival volume in vehicles is

    base_rate * multiplier(entry) * (1 + variation * (2u - 1)),   u ~ U(0, 1)

so its mean is ``base_rate * multiplier`` and it stays within
``+/- variation`` of that. The random stream for a cycle depends only on
``(seed, cycle)``, so ``arrivals(c)`` is a pure function: it gives the same answer
regardless of what was queried before it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Protocol

import numpy as np

from .network import Approach, Network

# Vehicles per cycle per entry approach. For scale, a 30 s green at the default
# saturation flow (0.5 veh/s) can discharge 15 vehicles per cycle.
DEMAND_LEVELS: dict[str, float] = {"low": 8.0, "medium": 12.0, "high": 16.0}


class DemandSource(Protocol):
    """Anything that can say how many vehicles arrive at each entry in a cycle."""

    def arrivals(self, cycle: int) -> Mapping[Approach, float]: ...


@dataclass(frozen=True)
class DemandConfig:
    base_rate: float = DEMAND_LEVELS["medium"]  # vehicles / cycle / entry approach
    seed: int = 0
    variation: float = 0.25  # relative half-width of the per-cycle fluctuation
    multipliers: Mapping[Approach, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (math.isfinite(self.base_rate) and self.base_rate >= 0):
            raise ValueError("base_rate must be finite and >= 0")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not 0 <= self.variation <= 1:
            raise ValueError("variation must be within [0, 1]")
        for approach, m in self.multipliers.items():
            if not (math.isfinite(m) and m >= 0):
                raise ValueError(f"multiplier for {approach} must be finite and >= 0")

    @classmethod
    def from_level(cls, level: str, **kwargs) -> DemandConfig:
        if level not in DEMAND_LEVELS:
            raise ValueError(f"unknown demand level {level!r}; choose from {sorted(DEMAND_LEVELS)}")
        return cls(base_rate=DEMAND_LEVELS[level], **kwargs)


class DemandModel:
    """Seeded demand generator for the entry approaches of a network."""

    def __init__(self, network: Network, config: DemandConfig | None = None):
        self.config = config or DemandConfig()
        self._entries = network.entry_approaches
        unknown = set(self.config.multipliers) - set(self._entries)
        if unknown:
            raise ValueError(f"multipliers given for non-entry approaches: {sorted(map(str, unknown))}")

    def arrivals(self, cycle: int) -> dict[Approach, float]:
        if cycle < 0:
            raise ValueError("cycle must be >= 0")
        cfg = self.config
        u = np.random.default_rng([cfg.seed, cycle]).random(len(self._entries))
        return {
            a: cfg.base_rate * cfg.multipliers.get(a, 1.0) * (1.0 + cfg.variation * (2.0 * float(x) - 1.0))
            for a, x in zip(self._entries, u)
        }
