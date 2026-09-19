"""Controller interface and the state snapshot controllers observe."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

from ..network import Approach, Network
from ..signals import SignalPlan


@dataclass(frozen=True)
class Observation:
    """Traffic state at the start of a cycle, before arrivals are processed.

    ``queues`` are vehicles waiting at each approach; ``in_transit`` are vehicles
    already released upstream that will join each approach at the start of this
    cycle. Both are copies: mutating them cannot change the simulation.
    """

    cycle: int
    queues: Mapping[Approach, float]
    in_transit: Mapping[Approach, float]
    network: Network


class Controller(ABC):
    """Chooses one signal plan per intersection for each cycle."""

    @property
    def name(self) -> str:
        return type(self).__name__

    def reset(self) -> None:
        """Clear any internal state; called at the start of every run."""

    @abstractmethod
    def select_plans(self, observation: Observation) -> Mapping[str, SignalPlan]:
        """Return a valid SignalPlan for every intersection in the network."""
