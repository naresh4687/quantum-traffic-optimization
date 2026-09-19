"""Small test doubles shared by the simulator and metrics tests."""

from qtraffic import PLAN_NS30_EW30, Controller


class ScriptedDemand:
    """Demand defined explicitly: ``script[cycle][approach] -> vehicles``.

    ``constant`` (optional) is used for every cycle not listed in ``script``.
    """

    def __init__(self, script=None, constant=None):
        self.script = script or {}
        self.constant = constant or {}

    def arrivals(self, cycle):
        return dict(self.script.get(cycle, self.constant))


class PerNodeController(Controller):
    """Assigns a chosen plan to specific nodes and a default plan to the rest."""

    def __init__(self, overrides=None, default=PLAN_NS30_EW30):
        self.overrides = overrides or {}
        self.default = default

    def select_plans(self, observation):
        return {n: self.overrides.get(n, self.default) for n in observation.network.nodes}


class CyclingController(Controller):
    """Rotates through all three valid plans, differently per node (stress test)."""

    def select_plans(self, observation):
        from qtraffic import VALID_PLANS

        return {
            n: VALID_PLANS[(observation.cycle + i) % 3]
            for i, n in enumerate(observation.network.nodes)
        }
