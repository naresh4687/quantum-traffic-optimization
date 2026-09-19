"""Signal phases and the fixed 60-second, three-split signal plan model.

All intersections share one global cycle clock: every cycle starts at the same
instant everywhere and each intersection runs the NS phase and the EW phase once
per cycle. Yellow/all-red time and protected turns are intentionally not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

CYCLE_LENGTH = 60  # seconds
VALID_SPLITS: tuple[tuple[int, int], ...] = ((20, 40), (30, 30), (40, 20))  # (NS, EW)


class Phase(Enum):
    NS = "NS"
    EW = "EW"


@dataclass(frozen=True)
class SignalPlan:
    """Green durations (seconds) for the NS and EW phases of one cycle."""

    ns_green: int
    ew_green: int

    def __post_init__(self) -> None:
        for value in (self.ns_green, self.ew_green):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"green durations must be integers, got {value!r}")
        if (self.ns_green, self.ew_green) not in VALID_SPLITS:
            raise ValueError(
                f"invalid signal plan NS={self.ns_green}, EW={self.ew_green}; "
                f"valid (NS, EW) splits are {list(VALID_SPLITS)}"
            )

    @classmethod
    def from_ns(cls, ns_green: int) -> SignalPlan:
        """Build the plan whose NS green is ``ns_green`` (EW gets the remainder)."""
        return cls(ns_green, CYCLE_LENGTH - ns_green)

    @property
    def cycle_length(self) -> int:
        return self.ns_green + self.ew_green

    def green_time(self, phase: Phase) -> int:
        return self.ns_green if phase is Phase.NS else self.ew_green

    @property
    def label(self) -> str:
        return f"NS{self.ns_green}/EW{self.ew_green}"


PLAN_NS20_EW40 = SignalPlan(20, 40)
PLAN_NS30_EW30 = SignalPlan(30, 30)
PLAN_NS40_EW20 = SignalPlan(40, 20)
VALID_PLANS: tuple[SignalPlan, ...] = (PLAN_NS20_EW40, PLAN_NS30_EW30, PLAN_NS40_EW20)


def validate_plan(plan: object) -> SignalPlan:
    """Return ``plan`` if it is a valid SignalPlan, else raise ValueError.

    Re-checks the values so that a plan built by bypassing ``__init__`` is still
    rejected before it can reach the simulator.
    """
    if not isinstance(plan, SignalPlan):
        raise ValueError(f"expected a SignalPlan, got {type(plan).__name__}")
    SignalPlan.__post_init__(plan)
    return plan


def validate_plan_assignment(
    plans: Mapping[str, object], nodes: Iterable[str]
) -> dict[str, SignalPlan]:
    """Check that exactly the given nodes each have one valid plan."""
    nodes = list(nodes)
    if set(plans) != set(nodes):
        missing = sorted(set(nodes) - set(plans))
        extra = sorted(set(plans) - set(nodes))
        raise ValueError(f"plan assignment mismatch: missing={missing}, unknown={extra}")
    return {node: validate_plan(plans[node]) for node in nodes}
