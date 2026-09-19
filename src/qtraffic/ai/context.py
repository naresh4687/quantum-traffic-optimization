"""Structured analysis context: the ONLY data that ever reaches the explanation layer.

Every value comes from the simulator, the optimisation results or the environmental PROXY (each is
labelled). The context is small, JSON-serialisable and whitelisted: unknown keys, non-finite numbers and
path-like strings are rejected, so raw logs, file paths, credentials and environment variables cannot be
sent to a model by accident. The numbers in this object remain authoritative; any model text is only an
explanation of them.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

TOP_LEVEL_KEYS = frozenset({
    "scenario", "controller", "baseline_controller", "seeds", "horizon_cycles", "traffic", "signal",
    "baseline_comparison", "environmental", "optimization", "qaoa", "emergency", "limitations",
})
SECTION_KEYS: dict[str, frozenset[str]] = {
    # measured simulator outputs
    "traffic": frozenset({
        "vehicles_admitted", "vehicles_rejected", "vehicles_exited", "throughput_vehicles_per_hour",
        "waiting_vehicle_seconds", "waiting_seconds_per_admitted", "average_queue_vehicles",
        "max_queue_vehicles", "final_queue_vehicles", "blocked_vehicle_cycles"}),
    "signal": frozenset({"plan_changes", "plan_change_rate", "ns_green_share", "signal_plans"}),
    # % change of the controller vs the baseline on identical demand (measured)
    "baseline_comparison": frozenset({
        "waiting_seconds_per_admitted_pct", "waiting_vehicle_seconds_pct", "throughput_pct",
        "average_queue_pct", "max_queue_pct", "vehicles_rejected_pct", "final_queue_pct",
        "seeds_better", "seeds_worse", "seeds_compared"}),
    # SIMULATION PROXY, derived from waiting time; never a measurement
    "environmental": frozenset({
        "label", "fuel_liters_proxy", "co2_kg_proxy", "fuel_liters_proxy_per_admitted",
        "co2_kg_proxy_per_admitted", "co2_proxy_pct_vs_baseline", "idle_fuel_rate_lph_assumed",
        "emission_factor_kg_per_liter_assumed", "waiting_pct_vs_baseline"}),
    "optimization": frozenset({"qubo_energy", "exact_energy", "method"}),
    "qaoa": frozenset({
        "backend", "p", "n_qubits", "shots", "logical_depth", "logical_gates", "basis_cx_gates",
        "best_sampled_feasible_energy", "exact_energy", "energy_gap", "approximation_ratio",
        "found_exact_optimum", "exact_n_optimal_assignments", "feasible_rate_sampled",
        "feasible_probability_statevector", "uniform_feasible_probability",
        "optimum_probability_sampled", "optimum_probability_statevector", "uniform_optimum_probability",
        "optimum_enrichment_among_feasible", "optimizer_converged", "function_evaluations",
        "variant_note"}),
    "emergency": frozenset({
        "route", "seeds", "travel_time_s_no_override", "travel_time_s_corridor", "delay_s_no_override",
        "delay_s_corridor", "free_flow_time_s", "stops_no_override", "stops_corridor",
        "prioritized_intersections", "normal_waiting_pct_corridor_vs_no_override",
        "seeds_normal_waiting_worse", "seeds_normal_waiting_better", "max_queue_no_override",
        "max_queue_corridor", "final_queue_no_override", "final_queue_corridor",
        "co2_proxy_pct_corridor_vs_no_override", "cycles_after_completion_checked",
        "cycles_after_completion_with_override_active", "plans_restored_to_base_controller"}),
}

_PATHLIKE = re.compile(r"([A-Za-z]:[\\/])|(^|\s)/(home|Users|etc|var|tmp|mnt)/|\\\\")
_MAX_STRING = 400


class ContextError(ValueError):
    """The context contains something that must not be sent."""


def _check_value(where: str, value: Any) -> None:
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContextError(f"{where}: non-finite number")
        return
    if isinstance(value, str):
        if len(value) > _MAX_STRING:
            raise ContextError(f"{where}: string too long")
        if _PATHLIKE.search(value):
            raise ContextError(f"{where}: looks like a filesystem path")
        return
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _check_value(f"{where}[{i}]", v)
        return
    if isinstance(value, Mapping):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ContextError(f"{where}: keys must be strings")
            _check_value(f"{where}.{k}", v)
        return
    raise ContextError(f"{where}: unsupported type {type(value).__name__}")


STANDARD_LIMITATIONS = (
    "Traffic numbers are outputs of a deterministic queue-based fluid simulation, not field measurements.",
    "Fuel and CO2 are a waiting-based SIMULATION PROXY (linear in waiting time), not measured emissions.",
    "QAOA was run on Qiskit Aer, a classical quantum-circuit simulator; no quantum hardware was used and "
    "no quantum speedup is measured.",
)


@dataclass(frozen=True)
class TrafficAnalysisContext:
    """Measured results for one scenario/controller (any section may be absent)."""

    scenario: str
    controller: str
    baseline_controller: str | None = None
    seeds: tuple[int, ...] = ()
    horizon_cycles: int | None = None
    traffic: dict = field(default_factory=dict)
    signal: dict = field(default_factory=dict)
    baseline_comparison: dict = field(default_factory=dict)
    environmental: dict = field(default_factory=dict)
    optimization: dict = field(default_factory=dict)
    qaoa: dict = field(default_factory=dict)
    emergency: dict = field(default_factory=dict)
    limitations: tuple[str, ...] = STANDARD_LIMITATIONS

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.scenario or not self.controller:
            raise ContextError("scenario and controller are required")
        for name, allowed in SECTION_KEYS.items():
            section = getattr(self, name)
            unknown = set(section) - allowed
            if unknown:
                raise ContextError(f"{name}: keys not allowed in the context: {sorted(unknown)}")
        payload = self.to_dict()
        for k, v in payload.items():
            _check_value(k, v)
        json.dumps(payload)  # must be serialisable

    def to_dict(self) -> dict:
        """JSON-ready dict; empty sections are omitted."""
        data: dict[str, Any] = {"scenario": self.scenario, "controller": self.controller}
        if self.baseline_controller:
            data["baseline_controller"] = self.baseline_controller
        if self.seeds:
            data["seeds"] = list(self.seeds)
        if self.horizon_cycles is not None:
            data["horizon_cycles"] = self.horizon_cycles
        for name in SECTION_KEYS:
            section = getattr(self, name)
            if section:
                data[name] = dict(section)
        data["limitations"] = list(self.limitations)
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrafficAnalysisContext:
        unknown = set(data) - TOP_LEVEL_KEYS
        if unknown:
            raise ContextError(f"keys not allowed in the context: {sorted(unknown)}")
        kwargs = {k: v for k, v in data.items() if k in TOP_LEVEL_KEYS}
        kwargs["seeds"] = tuple(kwargs.get("seeds", ()))
        kwargs["limitations"] = tuple(kwargs.get("limitations", STANDARD_LIMITATIONS))
        return cls(**kwargs)
