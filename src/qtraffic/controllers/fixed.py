"""Fixed-time baseline controller."""

from __future__ import annotations

from typing import Mapping

from ..signals import PLAN_NS30_EW30, SignalPlan, validate_plan
from .base import Controller, Observation


class FixedTimeController(Controller):
    """Applies the same signal plan (default NS=30 / EW=30) everywhere, every cycle."""

    def __init__(self, plan: SignalPlan = PLAN_NS30_EW30):
        self.plan = validate_plan(plan)

    def select_plans(self, observation: Observation) -> Mapping[str, SignalPlan]:
        return {node: self.plan for node in observation.network.nodes}
