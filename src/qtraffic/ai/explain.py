"""Explanations of already-computed results: deterministic fallback text, prompts, validation, orchestration.

Architecture (one direction only)::

    simulator -> controllers / QUBO / QAOA -> measured results -> TrafficAnalysisContext
                                                                        |
                                            deterministic fallback  <---+--->  Featherless (optional)
                                                                        v
                                                              natural-language explanation

This layer NEVER chooses, changes or generates a signal plan, QUBO coefficient, QAOA parameter or emergency
decision, and it is never a source of simulation truth. ``explain_signal_choice`` only reports the plan that
the optimisation pipeline already computed (present in the context). The numbers in the context are
authoritative; model text is explanation only, and is discarded in favour of the deterministic text if it
fails validation or the request fails.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import FeatherlessConfig
from .context import TrafficAnalysisContext
from .featherless import ChatClient, FeatherlessError, OpenAICompatibleClient, classify_exception

KINDS = ("optimization", "qaoa", "emergency", "environment", "comparison", "signal_choice")
SOURCE_MODEL = "featherless"
SOURCE_FALLBACK = "deterministic_fallback"


# =============================================================================== formatting helpers
def _num(x, nd: int = 1) -> str:
    return "n/a" if x is None else f"{x:,.{nd}f}"


def _pct(x, nd: int = 1) -> str:
    return "n/a" if x is None else f"{x:+.{nd}f}%"


def _direction(pct) -> str:
    if pct is None:
        return "not comparable"
    return "unchanged" if abs(pct) < 0.05 else ("higher" if pct > 0 else "lower")


def _plans_summary(plans: dict) -> str:
    groups: dict[str, list[str]] = {}
    for node, label in plans.items():
        groups.setdefault(label, []).append(node)
    return "; ".join(f"{label} at {', '.join(nodes)}" for label, nodes in sorted(groups.items()))


CONTROLLER_DESCRIPTIONS = {
    "fixed": "Fixed-time control applies the same NS 30 s / EW 30 s plan at every intersection in every cycle.",
    "adaptive": "Adaptive control picks one of the three valid plans per intersection each cycle from a "
                "queue-pressure rule that also looks at the downstream queue.",
    "qubo_static": "The QUBO controller builds the QUBO from the starting traffic state, solves it exactly by "
                   "enumeration, and holds the resulting plans.",
    "qubo_receding": "The receding QUBO controller rebuilds and exactly solves the QUBO every cycle.",
    "qaoa_p1_saved": "The plans are the best feasible bitstring sampled from a QAOA circuit run on a classical "
                     "simulator, held for the whole run.",
}


def _limitations_line(ctx: TrafficAnalysisContext) -> str:
    return " ".join(ctx.limitations[:1]) if ctx.limitations else ""


# =============================================================================== deterministic fallbacks
def _comparison_core(ctx: TrafficAnalysisContext) -> str:
    t, bc = ctx.traffic, ctx.baseline_comparison
    seeds = f"{len(ctx.seeds)} seed{'s' if len(ctx.seeds) != 1 else ''}" if ctx.seeds else "the recorded seeds"
    if not bc:
        return (f"In scenario '{ctx.scenario}' ({seeds}, {ctx.horizon_cycles} cycles), {ctx.controller} produced waiting "
                f"of {_num(t.get('waiting_seconds_per_admitted'))} s per admitted vehicle, throughput of "
                f"{_num(t.get('throughput_vehicles_per_hour'), 0)} vehicles/hour, average queue "
                f"{_num(t.get('average_queue_vehicles'), 2)} and maximum queue {_num(t.get('max_queue_vehicles'))} vehicles. "
                "No baseline comparison is available in the supplied results. " + _limitations_line(ctx))
    parts = [
        f"In scenario '{ctx.scenario}' ({seeds}, {ctx.horizon_cycles} cycles, identical demand for every controller), "
        f"{ctx.controller} compared with {ctx.baseline_controller} gave:",
        f"waiting per admitted vehicle {_num(t.get('waiting_seconds_per_admitted'))} s ({_pct(bc.get('waiting_seconds_per_admitted_pct'))}, "
        f"{_direction(bc.get('waiting_seconds_per_admitted_pct'))}); throughput {_num(t.get('throughput_vehicles_per_hour'), 0)} vehicles/hour "
        f"({_pct(bc.get('throughput_pct'))}); average queue {_num(t.get('average_queue_vehicles'), 2)} vehicles ({_pct(bc.get('average_queue_pct'))}); "
        f"maximum queue {_num(t.get('max_queue_vehicles'))} ({_pct(bc.get('max_queue_pct'))}); rejected vehicles {_num(t.get('vehicles_rejected'))} "
        f"({_pct(bc.get('vehicles_rejected_pct'))}); final queue {_num(t.get('final_queue_vehicles'))} ({_pct(bc.get('final_queue_pct'))}).",
    ]
    if bc.get("seeds_compared", 0) > 1:
        parts.append(f"Waiting per admitted vehicle was lower in {bc.get('seeds_better')} and higher in {bc.get('seeds_worse')} "
                     f"of {bc.get('seeds_compared')} seeds.")
    w, wt = bc.get("waiting_seconds_per_admitted_pct"), bc.get("waiting_vehicle_seconds_pct")
    if w is not None and wt is not None and w < 0 < wt:
        parts.append("Total waiting rose even though waiting per admitted vehicle fell, because this controller admitted "
                     "more vehicles (fewer were rejected), so both views are reported.")
    if w is not None and w > 0:
        parts.append("On waiting per admitted vehicle this controller was worse than the baseline in this scenario.")
    if w is not None and w < 0 and (bc.get("max_queue_pct") or 0) > 0:
        parts.append("The maximum queue was higher even though average waiting per admitted vehicle was lower.")
    parts.append("These are measured simulator outputs for this scenario only and do not establish general superiority. "
                 + _limitations_line(ctx))
    return " ".join(parts)


def fallback_comparison(ctx: TrafficAnalysisContext) -> str:
    """What the two controllers are, then the measured comparison."""
    base = (ctx.baseline_controller or "").split(" ")[0]
    described = [CONTROLLER_DESCRIPTIONS[c] for c in (ctx.controller, base) if c in CONTROLLER_DESCRIPTIONS]
    return " ".join(described + [_comparison_core(ctx)])


def fallback_optimization(ctx: TrafficAnalysisContext) -> str:
    parts = [CONTROLLER_DESCRIPTIONS.get(ctx.controller, f"Controller: {ctx.controller}.")]
    sig, opt = ctx.signal, ctx.optimization
    if sig.get("signal_plans"):
        parts.append(f"Computed signal plans: {_plans_summary(sig['signal_plans'])}.")
    if sig.get("plan_change_rate") is not None:
        parts.append(f"Plans changed in {_num(100 * sig['plan_change_rate'], 1)}% of intersection-cycle transitions, and "
                     f"{_num(100 * sig.get('ns_green_share', 0), 1)}% of green time went to the north-south phase.")
    if opt.get("qubo_energy") is not None:
        parts.append(f"QUBO energy of the chosen assignment: {_num(opt['qubo_energy'], 1)}"
                     + (f" (exact optimum {_num(opt['exact_energy'], 1)})" if opt.get("exact_energy") is not None else "")
                     + ". QUBO energy is a surrogate objective, not a waiting time.")
    parts.append(_comparison_core(ctx))
    return " ".join(parts)


def fallback_qaoa(ctx: TrafficAnalysisContext) -> str:
    q = ctx.qaoa
    if not q:
        return "No QAOA results are present in the supplied context."
    seed = f" (seed {ctx.seeds[0]})" if ctx.seeds else ""
    parts = [
        f"QAOA (p={q['p']}, {q['n_qubits']} qubits, {q['shots']} shots; {q.get('variant_note', 'run')}){seed} was evaluated using a "
        "classical quantum-circuit simulator (Qiskit Aer). The experiment measures solution quality and sampling behavior, "
        "not quantum hardware speedup, and exact enumeration remains the ground truth.",
        f"The circuit has logical depth {q['logical_depth']} and {q['logical_gates']} gates ({q['basis_cx_gates']} CX after decomposition).",
        f"The best sampled feasible energy was {_num(q['best_sampled_feasible_energy'], 1)} against an exact optimum of "
        f"{_num(q['exact_energy'], 1)} (gap {_num(q['energy_gap'], 1)}, approximation ratio {_num(q['approximation_ratio'], 4)}); the exact "
        f"optimum {'was' if q['found_exact_optimum'] else 'was not'} among the best sampled feasible solutions"
        + (f", and {q['exact_n_optimal_assignments']} assignments tie at that energy." if q["exact_n_optimal_assignments"] > 1 else "."),
        f"{_num(100 * q['feasible_rate_sampled'], 2)}% of shots were valid one-hot assignments (statevector "
        f"{_num(100 * q['feasible_probability_statevector'], 2)}%, against {_num(100 * q['uniform_feasible_probability'], 2)}% for uniform sampling). "
        f"The probability of sampling an optimal assignment was {_num(q['optimum_probability_sampled'], 6)} sampled and "
        f"{_num(q['optimum_probability_statevector'], 6)} from the statevector, against {_num(q['uniform_optimum_probability'], 8)} for uniform sampling.",
    ]
    if q.get("optimum_enrichment_among_feasible") is not None:
        parts.append(f"Among feasible assignments the optimum was {_num(q['optimum_enrichment_among_feasible'], 2)} times as likely as under a "
                     "uniform pick (1.0 would mean no preference for it).")
    if not q.get("optimizer_converged", True):
        parts.append(f"The optimizer stopped at its iteration limit after {q['function_evaluations']} evaluations.")
    bc = ctx.baseline_comparison
    if bc.get("waiting_seconds_per_admitted_pct") is not None:
        parts.append(f"Run in the traffic simulator against Fixed, the sampled plan changed waiting per admitted vehicle by "
                     f"{_pct(bc['waiting_seconds_per_admitted_pct'])} and throughput by {_pct(bc.get('throughput_pct'))}.")
    parts.append("QUBO energy is a surrogate, equal-energy plans can behave differently in the simulator, and this is a single seed.")
    return " ".join(parts)


def fallback_emergency(ctx: TrafficAnalysisContext) -> str:
    e = ctx.emergency
    if not e:
        return "No emergency-corridor results are present in the supplied context."
    parts = [
        f"Emergency corridor on route {' -> '.join(e['route'])} in scenario '{ctx.scenario}' ({e['seeds']} seeds, base controller "
        f"{ctx.controller.split('+')[0]}): the emergency travel time was {_num(e['travel_time_s_no_override'])} s without the override and "
        f"{_num(e['travel_time_s_corridor'])} s with the corridor (free-flow {_num(e['free_flow_time_s'])} s), a delay of "
        f"{_num(e['delay_s_no_override'])} s versus {_num(e['delay_s_corridor'])} s, with {_num(e['stops_no_override'], 1)} versus "
        f"{_num(e['stops_corridor'], 1)} signal stops on average"
        + (f" and {_num(e['prioritized_intersections'], 0)} intersections prioritized." if e.get("prioritized_intersections") is not None else "."),
        f"Normal traffic waiting changed by {_pct(e['normal_waiting_pct_corridor_vs_no_override'], 2)} (worse in {e['seeds_normal_waiting_worse']} and "
        f"better in {e['seeds_normal_waiting_better']} seeds); maximum queue {_num(e['max_queue_no_override'])} -> {_num(e['max_queue_corridor'])} "
        f"vehicles and final queue {_num(e['final_queue_no_override'])} -> {_num(e['final_queue_corridor'])} vehicles.",
        f"Under the configured waiting-based simulation proxy (not a measurement), estimated CO2 changed by "
        f"{_pct(e['co2_proxy_pct_corridor_vs_no_override'], 2)}.",
    ]
    if "plans_restored_to_base_controller" in e:
        parts.append(f"Restoration: in the {e['cycles_after_completion_checked']} logged cycles after the vehicle finished, the override was active in "
                     f"{e['cycles_after_completion_with_override_active']} and the applied plans "
                     f"{'equalled' if e['plans_restored_to_base_controller'] else 'did not equal'} the base controller's plans.")
    parts.append("The vehicle timeline rests on modelling assumptions (30 s per link, NS-before-EW phase order, a vehicle with no mass), so "
                 "the result describes this simulation, not real roads.")
    return " ".join(parts)


def fallback_environment(ctx: TrafficAnalysisContext) -> str:
    env, e = ctx.environmental, ctx.emergency
    if not env and e:
        return (f"Environmental figures here are a SIMULATION PROXY derived from waiting time, not measurements. Under the configured waiting-based "
                f"simulation proxy, estimated CO2 changed by {_pct(e['co2_proxy_pct_corridor_vs_no_override'], 2)} with the emergency corridor "
                f"versus no override in scenario '{ctx.scenario}', because normal-traffic waiting changed by "
                f"{_pct(e['normal_waiting_pct_corridor_vs_no_override'], 2)}. Not modelled: acceleration, speed, vehicle type, engine efficiency, "
                "NOx/PM emissions, or fuel used while moving.")
    if not env:
        return "No environmental proxy values are present in the supplied context."
    parts = [
        f"Environmental figures are a SIMULATION PROXY (waiting/idling only), not measurements: fuel = waiting vehicle-seconds / 3600 x an assumed idle "
        f"rate of {_num(env['idle_fuel_rate_lph_assumed'], 2)} L per vehicle-hour, and CO2 = fuel x {_num(env['emission_factor_kg_per_liter_assumed'], 4)} kg/L.",
        f"For {ctx.controller} in scenario '{ctx.scenario}', measured waiting of {_num(ctx.traffic.get('waiting_vehicle_seconds'), 0)} vehicle-seconds gives "
        f"an estimated {_num(env['fuel_liters_proxy'], 1)} L of fuel and {_num(env['co2_kg_proxy'], 1)} kg of CO2 (proxy), or "
        f"{_num(env['fuel_liters_proxy_per_admitted'], 4)} L and {_num(env['co2_kg_proxy_per_admitted'], 4)} kg per admitted vehicle.",
    ]
    if env.get("co2_proxy_pct_vs_baseline") is not None:
        parts.append(f"Relative to {ctx.baseline_controller} on identical demand, total waiting changed by {_pct(env['waiting_pct_vs_baseline'])}, so under the "
                     f"configured waiting-based simulation proxy, estimated CO2 changed by {_pct(env['co2_proxy_pct_vs_baseline'])}; the proxy is linear in "
                     "waiting, so the two percentages are equal by construction. Total waiting also depends on how many vehicles were admitted.")
    parts.append("Not modelled: acceleration and deceleration, speed, vehicle type, engine efficiency, temperature, NOx/PM emissions, or fuel used while "
                 "moving. The idle rate is a configurable parameter, not a calibrated fleet average.")
    return " ".join(parts)


def fallback_signal_choice(ctx: TrafficAnalysisContext, node: str | None = None) -> str:
    plans = ctx.signal.get("signal_plans") or {}
    if node is None or node not in plans:
        return (f"The supplied results contain no computed signal plan for {node or 'the requested intersection'} in scenario '{ctx.scenario}' "
                f"({ctx.controller}). The explanation layer does not choose signal plans; it only reports plans that the optimization pipeline computed.")
    label = plans[node]
    ns, ew = re.match(r"NS(\d+)/EW(\d+)", label).groups()
    return (f"For scenario '{ctx.scenario}' with {ctx.controller}, the optimization pipeline computed plan {label} for {node}: {ns} s of north-south green "
            f"and {ew} s of east-west green per 60 s cycle. This result comes from the deterministic {ctx.controller} pipeline, not from the AI "
            "layer, which only explains computed plans and never chooses or changes a signal plan.")


FALLBACKS = {"optimization": fallback_optimization, "qaoa": fallback_qaoa, "emergency": fallback_emergency,
             "environment": fallback_environment, "comparison": fallback_comparison}


def deterministic_explanation(kind: str, ctx: TrafficAnalysisContext, node: str | None = None) -> str:
    """Pure function of the context: same input, same text, no network, no model."""
    if kind == "signal_choice":
        return fallback_signal_choice(ctx, node)
    if kind not in FALLBACKS:
        raise ValueError(f"unknown explanation kind {kind!r}; choose from {KINDS}")
    return FALLBACKS[kind](ctx)


# =============================================================================== prompt
SYSTEM_PROMPT = (
    "You explain results from a deterministic traffic-signal simulation project. You are an explanation layer only.\n"
    "Rules:\n"
    "1. Use ONLY the supplied structured results. Do not invent numbers, experiments, scenarios or comparisons.\n"
    "2. Never choose, recommend, change or generate a traffic-signal plan, a QUBO coefficient, a QAOA parameter or an emergency-control decision. "
    "If asked which signal an intersection should use, explain the plan already computed in the results.\n"
    "3. QAOA was run on Qiskit Aer, a classical quantum-circuit simulator. Never claim quantum advantage or speedup, or that QAOA is faster than "
    "classical methods. Say the experiment measures solution quality and sampling behavior.\n"
    "4. Fuel and CO2 values are a waiting-based SIMULATION PROXY, not measurements. Never claim real-world emissions were reduced; say "
    "\"under the configured waiting-based simulation proxy, estimated CO2 changed by X%\".\n"
    "5. Distinguish measured simulator outputs from proxies. Do not state causal conclusions the results do not support. Mention relevant limitations.\n"
    "6. Be concise: 100 to 250 words of plain prose, no markdown tables."
)
TASKS = {
    "optimization": "Explain which controller produced these results, the signal plans or behaviour it produced, how it compared with the baseline, and the trade-offs.",
    "qaoa": "Explain the QAOA run: depth, sampled feasibility, sampled energy versus the exact reference, sampling probabilities, and limitations.",
    "emergency": "Explain the emergency-corridor result: emergency travel time and delay, the impact on normal traffic, and the restoration behaviour.",
    "environment": "Explain the environmental proxy values and the underlying waiting-time change, stating clearly that they are a simulation proxy.",
    "comparison": "Compare the controller with the baseline using only the supplied measured values and state the trade-offs.",
    "signal_choice": "Report which signal plan the results say the named intersection uses, without choosing or changing anything.",
}


def build_messages(kind: str, ctx: TrafficAnalysisContext, node: str | None = None) -> list[dict]:
    """System rules + the task + the structured context (JSON). Nothing else is ever sent."""
    task = TASKS[kind] + (f" Intersection of interest: {node}." if node and kind == "signal_choice" else "")
    user = f"Task: {task}\nStructured results (JSON):\n{ctx.to_json()}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


# =============================================================================== response validation
MIN_CHARS, MAX_CHARS = 60, 3000
_SECRET_PATTERNS = [re.compile(p, re.I) for p in (r"bearer\s+\S{8,}", r"\bsk-[A-Za-z0-9_\-]{16,}", r"api[_-]?key\s*[:=]",
                                                  r"FEATHERLESS_(API_KEY|MODEL|BASE_URL)", r"authorization\s*:")]
_PATH = re.compile(r"[A-Za-z]:[\\/]|(^|\s)/(home|Users|etc|var)/")
_NEGATION = re.compile(r"\b(no|not|never|neither|nor|without|cannot|can't|isn't|doesn't|does not|do not|don't|nothing)\b", re.I)
_QUANTUM_CLAIM = re.compile(r"quantum[- ](advantage|speed-?up|supremacy)|faster than classical|outperform\w*\s+(the\s+)?(classical|exact)", re.I)
_EMISSION_CLAIM = re.compile(r"(reduc\w+|sav\w+|cut\w*|lower\w*)[^.]{0,40}(co2|co₂|emission|carbon)", re.I)
_REAL_WORLD_CLAIM = re.compile(r"real[- ]world\s+(co2|co₂|emission|fuel|carbon)", re.I)
_QUALIFIER = re.compile(r"proxy|estimated", re.I)  # a generic word like "simulator" is NOT enough
REQUIRED_WORDS = {"qaoa": ("simulator",), "environment": ("proxy",), "emergency": ("emergency",)}


def validate_llm_text(text: Any, kind: str, config: FeatherlessConfig | None = None) -> str:
    """Lightweight guard, not a fact-check. Returns the cleaned text or raises ``FeatherlessError('validation_failed')``."""
    def bad(reason: str):
        raise FeatherlessError("validation_failed", reason)

    if not isinstance(text, str):
        bad("response is not text")
    text = text.strip()
    if not text:
        bad("empty response")
    if len(text) < MIN_CHARS:
        bad("response too short")
    if len(text) > MAX_CHARS:
        bad("response too long")
    if config is not None and config.api_key and config.api_key in text:
        bad("response contains the API key")
    if any(p.search(text) for p in _SECRET_PATTERNS):
        bad("response contains credential-like text")
    if _PATH.search(text):
        bad("response contains a filesystem path")
    lowered = text.lower()
    for word in REQUIRED_WORDS.get(kind, ()):
        if word not in lowered:
            bad(f"expected content missing: {word!r}")
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
        if _QUANTUM_CLAIM.search(sentence) and not _NEGATION.search(sentence):
            bad("asserts a quantum advantage / speedup")
        if _REAL_WORLD_CLAIM.search(sentence) and not _NEGATION.search(sentence):
            bad("makes a real-world emissions claim")
        if _EMISSION_CLAIM.search(sentence) and not _QUALIFIER.search(sentence) and not _NEGATION.search(sentence):
            bad("states an emissions change without the simulation-proxy qualifier")
    return text


# =============================================================================== orchestration
@dataclass(frozen=True)
class ExplanationResult:
    available: bool  # True only when the text came from Featherless and passed validation
    source: str  # "featherless" | "deterministic_fallback"
    kind: str
    explanation: str
    fallback_reason: str | None = None  # why the model was not used (see featherless.KINDS)
    error_detail: str | None = None  # short, key-redacted
    model: str | None = None  # model name when source == "featherless"; never a key
    deterministic_summary: str = ""  # always present, always computed from the context
    context: dict = field(default_factory=dict)  # the authoritative numbers

    def to_dict(self) -> dict:
        return asdict(self)


class Explainer:
    """Produces explanations; never lets a model outage affect anything else."""

    def __init__(self, config: FeatherlessConfig | None = None, client: ChatClient | None = None):
        self.config = config or FeatherlessConfig.from_env()
        self._client = client
        self._cache: dict[str, ExplanationResult] = {}
        self.requests_made = 0  # API requests attempted (fallback-only paths never increment it)

    def _key(self, kind: str, ctx: TrafficAnalysisContext, node: str | None) -> str:
        raw = "|".join((kind, self.config.model or "", self.config.base_url, node or "", ctx.to_json()))
        return hashlib.sha256(raw.encode()).hexdigest()

    def _fallback(self, kind, ctx, node, reason, detail=None) -> ExplanationResult:
        text = deterministic_explanation(kind, ctx, node)
        return ExplanationResult(False, SOURCE_FALLBACK, kind, text, reason, self.config.redact(detail)[:200] if detail else None,
                                 None, text, ctx.to_dict())

    def explain(self, kind: str, ctx: TrafficAnalysisContext, node: str | None = None) -> ExplanationResult:
        if kind not in KINDS:
            raise ValueError(f"unknown explanation kind {kind!r}; choose from {KINDS}")
        key = self._key(kind, ctx, node)
        if key in self._cache:  # same result asked again: no second request
            return self._cache[key]
        client = self._client
        if client is None:
            reason = self.config.unavailable_reason()
            if reason is not None:
                return self._fallback(kind, ctx, node, reason)  # no client is built, nothing is sent
            try:
                client = OpenAICompatibleClient(self.config)
            except FeatherlessError as exc:
                return self._fallback(kind, ctx, node, exc.kind, exc.detail)
        self.requests_made += 1
        try:
            raw = client.complete(build_messages(kind, ctx, node), self.config.max_tokens, self.config.timeout_seconds)
            text = validate_llm_text(raw, kind, self.config)
        except FeatherlessError as exc:
            return self._fallback(kind, ctx, node, exc.kind, exc.detail)
        except Exception as exc:  # noqa: BLE001 - a broken client must never break the application
            return self._fallback(kind, ctx, node, classify_exception(exc), str(exc))
        result = ExplanationResult(True, SOURCE_MODEL, kind, text, None, None, self.config.model,
                                   deterministic_explanation(kind, ctx, node), ctx.to_dict())
        self._cache[key] = result
        return result


# =============================================================================== public API
def explain_optimization(context: TrafficAnalysisContext, explainer: Explainer | None = None) -> ExplanationResult:
    return (explainer or Explainer()).explain("optimization", context)


def explain_qaoa(context: TrafficAnalysisContext, explainer: Explainer | None = None) -> ExplanationResult:
    return (explainer or Explainer()).explain("qaoa", context)


def explain_emergency(context: TrafficAnalysisContext, explainer: Explainer | None = None) -> ExplanationResult:
    return (explainer or Explainer()).explain("emergency", context)


def explain_environment(context: TrafficAnalysisContext, explainer: Explainer | None = None) -> ExplanationResult:
    return (explainer or Explainer()).explain("environment", context)


def explain_comparison(context: TrafficAnalysisContext, explainer: Explainer | None = None) -> ExplanationResult:
    return (explainer or Explainer()).explain("comparison", context)


def explain_signal_choice(context: TrafficAnalysisContext, node: str, explainer: Explainer | None = None) -> ExplanationResult:
    """Explain the plan ALREADY computed for ``node``. Never selects or changes a plan."""
    return (explainer or Explainer()).explain("signal_choice", context, node=node)
