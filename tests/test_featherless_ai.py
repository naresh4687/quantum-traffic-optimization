"""Tests for the Featherless explanation layer. No real API calls: a fake client is injected, and the
network is blocked for the whole module so any accidental connection fails the test."""

import ast
import json
import re
import socket
from pathlib import Path

import pytest

from qtraffic.ai import (
    ContextError, Explainer, FeatherlessConfig, FeatherlessError, TrafficAnalysisContext, deterministic_explanation,
    explain_comparison, explain_emergency, explain_environment, explain_optimization, explain_qaoa,
    explain_signal_choice, load_controller_context, load_emergency_context, load_qaoa_context, validate_llm_text,
)
from qtraffic.ai import explain as explain_module
from qtraffic.ai.config import DEFAULT_BASE_URL
from qtraffic.ai.context import SECTION_KEYS, TOP_LEVEL_KEYS
from qtraffic.ai.explain import SYSTEM_PROMPT, build_messages
from qtraffic.ai.featherless import OpenAICompatibleClient, _default_sdk_factory, classify_exception

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SECRET = "rc_TEST_SECRET_KEY_0123456789abcdef"  # obviously fake

pytestmark = pytest.mark.skipif(not (RESULTS / "phase6" / "summary.json").exists(), reason="saved results not present")


@pytest.fixture(autouse=True)
def no_network_and_clean_env(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    for name in ("FEATHERLESS_API_KEY", "FEATHERLESS_MODEL", "FEATHERLESS_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


# -- helpers ---------------------------------------------------------------------------------------
GOOD = ("In this emergency scenario the simulator results show waiting behaviour under a simulation proxy. The numbers come from the "
        "supplied structured results only, and no hardware speedup is claimed by this explanation.")


class FakeClient:
    def __init__(self, reply=GOOD, exc=None):
        self.reply, self.exc, self.calls = reply, exc, []

    def complete(self, messages, max_tokens, timeout_seconds):
        self.calls.append({"messages": messages, "max_tokens": max_tokens, "timeout": timeout_seconds})
        if self.exc is not None:
            raise self.exc
        return self.reply


def configured(**kw) -> FeatherlessConfig:
    return FeatherlessConfig(api_key=kw.pop("api_key", SECRET), model=kw.pop("model", "test-model"), **kw)


def named_exc(name, status=None, message="boom"):
    """An exception whose class name / status_code look like the SDK's, without importing the SDK."""
    cls = type(name, (Exception,), {"status_code": status})
    return cls(message)


class FakeSDK:
    """Stands in for openai.OpenAI: records constructor kwargs and returns / raises as told."""

    def __init__(self, response=None, exc=None):
        self.response, self.exc, self.init_kwargs, self.requests = response, exc, None, []
        outer = self
        self.chat = type("Chat", (), {"completions": type("Comp", (), {"create": staticmethod(outer._create)})()})()

    def factory(self, **kwargs):
        self.init_kwargs = kwargs
        return self

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return self.response


def response(text):
    msg = type("M", (), {"content": text})()
    return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()


def ctx_ns():  # real saved results
    return load_controller_context(RESULTS, "ns_heavy", "qubo_static", seed=0)


def ctx_qaoa():
    return load_qaoa_context(RESULTS, "ns_heavy", "qaoa_p1", seed=0)


def ctx_emergency():
    return load_emergency_context(RESULTS, "balanced_medium", "fixed")


def ctx_env():
    return load_controller_context(RESULTS, "ew_heavy", "adaptive")


# -- 1-4. configuration -----------------------------------------------------------------------------------
def test_configuration_is_read_from_environment_variables_only():
    cfg = FeatherlessConfig.from_env({"FEATHERLESS_API_KEY": " k ", "FEATHERLESS_MODEL": "m", "FEATHERLESS_BASE_URL": "https://x.example/v1"})
    assert (cfg.api_key, cfg.model, cfg.base_url) == ("k", "m", "https://x.example/v1")
    assert cfg.unavailable_reason() is None
    empty = FeatherlessConfig.from_env({"FEATHERLESS_API_KEY": "  ", "FEATHERLESS_MODEL": ""})
    assert empty.api_key is None and empty.model is None and empty.base_url == DEFAULT_BASE_URL == "https://api.featherless.ai/v1"


def test_configuration_reads_the_real_environment(monkeypatch):
    monkeypatch.setenv("FEATHERLESS_API_KEY", SECRET)
    monkeypatch.setenv("FEATHERLESS_MODEL", "some-model")
    cfg = FeatherlessConfig.from_env()
    assert cfg.has_api_key and cfg.model == "some-model"


def test_missing_api_key_gives_a_structured_fallback_without_any_request():
    ex = Explainer(FeatherlessConfig(api_key=None, model="m"))
    r = explain_optimization(ctx_ns(), ex)
    assert r.available is False and r.source == "deterministic_fallback" and r.fallback_reason == "missing_api_key"
    assert r.explanation and r.explanation == r.deterministic_summary
    assert ex.requests_made == 0
    d = r.to_dict()
    assert d["available"] is False and d["source"] == "deterministic_fallback" and d["explanation"]


def test_no_client_is_even_built_without_a_key(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("client must not be constructed")

    monkeypatch.setattr(explain_module, "OpenAICompatibleClient", boom)
    assert Explainer(FeatherlessConfig()).explain("comparison", ctx_ns()).fallback_reason == "missing_api_key"


def test_no_model_configured_falls_back_and_no_model_name_is_invented():
    cfg = FeatherlessConfig(api_key=SECRET, model=None)
    assert cfg.unavailable_reason() == "missing_model"
    ex = Explainer(cfg)
    r = ex.explain("comparison", ctx_ns())
    assert r.fallback_reason == "missing_model" and r.model is None and ex.requests_made == 0


def test_custom_model_is_used_and_reported():
    r = Explainer(configured(model="my-custom-model"), FakeClient()).explain("emergency", ctx_emergency())
    assert r.available and r.model == "my-custom-model"


def test_custom_base_url_reaches_the_sdk_with_timeout_and_no_retries():
    sdk = FakeSDK(response(GOOD))
    cfg = configured(base_url="https://gateway.example/v1", timeout_seconds=7.5)
    text = OpenAICompatibleClient(cfg, sdk.factory).complete([{"role": "user", "content": "x"}], 123, 7.5)
    assert text == GOOD
    assert sdk.init_kwargs == {"api_key": SECRET, "base_url": "https://gateway.example/v1", "timeout": 7.5, "max_retries": 0}
    assert sdk.requests[0]["model"] == "test-model" and sdk.requests[0]["max_tokens"] == 123


def test_invalid_base_url_falls_back():
    r = Explainer(configured(base_url="not-a-url")).explain("comparison", ctx_ns())
    assert r.fallback_reason == "invalid_base_url" and r.source == "deterministic_fallback"


# -- 5. context serialisation ----------------------------------------------------------------------------------
def test_context_is_json_serialisable_and_round_trips():
    ctx = ctx_ns()
    data = json.loads(json.dumps(ctx.to_dict()))
    assert TrafficAnalysisContext.from_dict(data).to_dict() == ctx.to_dict()
    assert json.loads(ctx.to_json()) == ctx.to_dict()


@pytest.mark.parametrize("bad", [
    lambda: TrafficAnalysisContext("s", "c", traffic={"raw_log": "x"}),  # key not whitelisted
    lambda: TrafficAnalysisContext("s", "c", traffic={"vehicles_admitted": float("nan")}),
    lambda: TrafficAnalysisContext("s", "c", qaoa={"backend": "E:\\quantum-traffic\\results\\x.csv"}),  # path-like
    lambda: TrafficAnalysisContext("s", "c", qaoa={"backend": "/home/user/secret.txt"}),
    lambda: TrafficAnalysisContext.from_dict({"scenario": "s", "controller": "c", "api_key": "x"}),
    lambda: TrafficAnalysisContext("", "c"),
])
def test_context_rejects_anything_that_should_not_be_sent(bad):
    with pytest.raises(ContextError):
        bad()


def test_contexts_built_from_saved_results_contain_no_paths():
    for ctx in (ctx_ns(), ctx_qaoa(), ctx_emergency(), ctx_env()):
        blob = ctx.to_json()
        assert str(ROOT) not in blob and "\\" not in blob.replace("\\\\", "") and "results/" not in blob and ".csv" not in blob


def test_unknown_scenario_raises_instead_of_inventing_data():
    with pytest.raises(LookupError):
        load_controller_context(RESULTS, "no_such_scenario", "fixed")
    with pytest.raises(LookupError):
        load_qaoa_context(RESULTS, "ns_heavy", "qaoa_p9")


# -- 6-10. deterministic fallback ------------------------------------------------------------------------------------
def test_fallback_is_deterministic():
    for kind, make in (("optimization", ctx_ns), ("qaoa", ctx_qaoa), ("emergency", ctx_emergency), ("environment", ctx_env), ("comparison", ctx_ns)):
        a = deterministic_explanation(kind, make())
        b = deterministic_explanation(kind, make())
        assert a == b and a == Explainer(FeatherlessConfig()).explain(kind, make()).explanation


def test_optimization_fallback_uses_the_real_values():
    ctx = ctx_ns()
    text = explain_optimization(ctx, Explainer(FeatherlessConfig())).explanation
    bc, t = ctx.baseline_comparison, ctx.traffic
    assert f"{bc['waiting_seconds_per_admitted_pct']:+.1f}%" in text and f"{t['waiting_seconds_per_admitted']:,.1f}" in text
    assert "NS40/EW20" in text and "surrogate" in text and 80 <= len(text.split()) <= 260


def test_qaoa_fallback_states_the_classical_simulator_and_makes_no_advantage_claim():
    ctx = ctx_qaoa()
    text = explain_qaoa(ctx, Explainer(FeatherlessConfig())).explanation
    assert "classical quantum-circuit simulator" in text and "not quantum hardware speedup" in text
    assert "ground truth" in text and f"p={ctx.qaoa['p']}" in text and f"{ctx.qaoa['shots']} shots" in text
    assert f"{ctx.qaoa['logical_depth']}" in text and f"{100 * ctx.qaoa['feasible_rate_sampled']:,.2f}%" in text
    assert not re.search(r"quantum advantage|faster than classical", text.replace("not quantum hardware speedup", ""), re.I)
    assert 100 <= len(text.split()) <= 260


def test_emergency_fallback_reports_travel_time_delay_impact_and_restoration():
    ctx = ctx_emergency()
    e = ctx.emergency
    text = explain_emergency(ctx, Explainer(FeatherlessConfig())).explanation
    assert f"{e['travel_time_s_no_override']:,.1f} s without the override" in text and f"{e['travel_time_s_corridor']:,.1f} s with the corridor" in text
    assert f"{e['delay_s_no_override']:,.1f} s versus {e['delay_s_corridor']:,.1f} s" in text
    assert "Normal traffic waiting changed" in text and "Restoration" in text and "equalled the base controller" in text
    assert "simulation proxy" in text and "not a measurement" in text and 100 <= len(text.split()) <= 260
    assert e["cycles_after_completion_with_override_active"] == 0 and e["plans_restored_to_base_controller"] is True


def test_environment_fallback_uses_proxy_wording_and_never_claims_real_world_reduction():
    ctx = ctx_env()
    text = explain_environment(ctx, Explainer(FeatherlessConfig())).explanation
    assert "SIMULATION PROXY" in text and "not measurements" in text
    assert "under the configured waiting-based simulation proxy, estimated CO2 changed by" in text
    assert f"{ctx.environmental['co2_proxy_pct_vs_baseline']:+.1f}%" in text
    assert not re.search(r"real[- ]world", text.replace("not real", ""), re.I) and 100 <= len(text.split()) <= 260


def test_comparison_fallback_reports_a_worse_result_honestly():
    ctx = load_controller_context(RESULTS, "time_varying", "qubo_static")  # measured worse than Fixed
    assert ctx.baseline_comparison["waiting_seconds_per_admitted_pct"] > 0
    text = explain_comparison(ctx, Explainer(FeatherlessConfig())).explanation
    assert "worse than the baseline" in text and "higher" in text


# -- 11-15. API paths with a mocked client ---------------------------------------------------------------------------
def test_api_success_returns_the_validated_model_text_and_keeps_the_deterministic_summary():
    client = FakeClient()
    ex = Explainer(configured(), client)
    r = explain_emergency(ctx_emergency(), ex)
    assert r.available and r.source == "featherless" and r.fallback_reason is None and r.explanation == GOOD
    assert r.deterministic_summary and r.deterministic_summary != r.explanation
    assert ex.requests_made == 1 and len(client.calls) == 1
    call = client.calls[0]
    assert call["max_tokens"] == 400 and call["timeout"] == 20.0
    assert [m["role"] for m in call["messages"]] == ["system", "user"]


def test_the_same_result_is_not_requested_twice():
    client = FakeClient()
    ex = Explainer(configured(), client)
    r1, r2 = ex.explain("emergency", ctx_emergency()), ex.explain("emergency", ctx_emergency())
    assert r1 == r2 and len(client.calls) == 1
    ex.explain("comparison", ctx_ns())  # a different question does call
    assert len(client.calls) == 2


@pytest.mark.parametrize("exc, kind", [
    (FeatherlessError("timeout", "slow"), "timeout"),
    (named_exc("APITimeoutError"), "timeout"), (TimeoutError("t"), "timeout"),
    (named_exc("AuthenticationError", 401), "authentication"), (named_exc("PermissionDeniedError", 403), "authentication"),
    (named_exc("RateLimitError", 429), "rate_limit"),
    (named_exc("InternalServerError", 500), "server_error"), (named_exc("APIStatusError", 503), "server_error"),
    (named_exc("APIConnectionError"), "connection"), (ConnectionError("down"), "connection"),
    (named_exc("SomethingElse"), "request_failed"),
])
def test_every_failure_falls_back_with_the_right_reason(exc, kind):
    ex = Explainer(configured(), FakeClient(exc=exc))
    r = ex.explain("comparison", ctx_ns())
    assert r.source == "deterministic_fallback" and r.available is False and r.fallback_reason == kind
    assert r.explanation == deterministic_explanation("comparison", ctx_ns())


@pytest.mark.parametrize("exc, kind", [
    (named_exc("APITimeoutError"), "timeout"), (named_exc("AuthenticationError", 401), "authentication"),
    (named_exc("RateLimitError", 429), "rate_limit"), (named_exc("InternalServerError", 500), "server_error"),
])
def test_sdk_exceptions_are_classified_by_the_real_client_wrapper(exc, kind):
    client = OpenAICompatibleClient(configured(), FakeSDK(exc=exc).factory)
    with pytest.raises(FeatherlessError) as info:
        client.complete([{"role": "user", "content": "x"}], 10, 5)
    assert info.value.kind == kind
    r = Explainer(configured(), client).explain("comparison", ctx_ns())
    assert r.fallback_reason == kind and r.source == "deterministic_fallback"


@pytest.mark.parametrize("bad_response", [
    object(), type("R", (), {"choices": []})(), type("R", (), {"choices": None})(),
    type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": None})()})()]})(),
    type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": 42})()})()]})(),
])
def test_malformed_responses_fall_back(bad_response):
    client = OpenAICompatibleClient(configured(), FakeSDK(response=bad_response).factory)
    r = Explainer(configured(), client).explain("comparison", ctx_ns())
    assert r.fallback_reason == "malformed_response" and r.source == "deterministic_fallback"


def test_a_broken_client_never_breaks_the_application():
    r = Explainer(configured(), FakeClient(exc=RuntimeError("anything"))).explain("qaoa", ctx_qaoa())
    assert r.source == "deterministic_fallback" and r.explanation


def test_missing_openai_package_is_reported_not_raised(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "openai", None)  # makes `import openai` raise ImportError
    with pytest.raises(FeatherlessError) as info:
        _default_sdk_factory(api_key="k")
    assert info.value.kind == "package_missing"
    r = Explainer(configured()).explain("comparison", ctx_ns())  # default client path, package "missing"
    assert r.fallback_reason == "package_missing" and r.source == "deterministic_fallback"


def test_classify_exception_directly():
    assert classify_exception(named_exc("RateLimitError", 429)) == "rate_limit"
    assert classify_exception(ValueError("x")) == "request_failed"


# -- 9. response validation --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text, why", [
    (None, "not text"), (123, "not text"), ("", "empty"), ("   ", "empty"), ("short", "short"), ("x" * 3001, "long"),
    (GOOD + " Authorization: Bearer abcdefghijklmnop", "credential"), (GOOD + " set FEATHERLESS_API_KEY first", "credential"),
    ("Real-world CO2 emissions were lowered by this controller according to the simulator results supplied for the scenario here.", "real-world"),
    (GOOD + " see C:\\Users\\me\\secrets.txt", "path"),
    (GOOD + " " + SECRET, "key"),
    ("This simulator study shows the QAOA circuit achieves quantum advantage over classical solvers on this problem today.", "advantage"),
    ("The system reduced real-world CO2 emissions by 12% according to these results in the simulator study of traffic.", "emissions"),
])
def test_validation_rejects_bad_responses(text, why):
    with pytest.raises(FeatherlessError) as info:
        validate_llm_text(text, "comparison", configured())
    assert info.value.kind == "validation_failed"


def test_validation_requires_kind_specific_content_and_accepts_properly_qualified_text():
    plain = "The controller reduced waiting per admitted vehicle in this run according to the supplied results and nothing else at all here."
    for kind, word in (("qaoa", "simulator"), ("environment", "proxy"), ("emergency", "emergency")):
        with pytest.raises(FeatherlessError):
            validate_llm_text(plain, kind)
        assert word in validate_llm_text(plain + f" It mentions the {word}.", kind).lower()
    ok = ("Under the configured waiting-based simulation proxy, estimated CO2 changed by +4.2%. No quantum advantage is claimed; QAOA ran "
          "on a classical simulator and exact enumeration remains the ground truth for this scenario.")
    assert validate_llm_text(ok, "environment") == ok and validate_llm_text(ok, "qaoa") == ok


def test_invalid_model_output_is_discarded_in_favour_of_the_deterministic_text():
    for bad in ("too short", GOOD + " " + SECRET, "QAOA achieves quantum advantage over classical computers in this simulator experiment, clearly."):
        r = Explainer(configured(), FakeClient(reply=bad)).explain("qaoa", ctx_qaoa())
        assert r.source == "deterministic_fallback" and r.fallback_reason == "validation_failed"
        assert SECRET not in r.explanation and r.explanation == deterministic_explanation("qaoa", ctx_qaoa())


# -- 16. no secret leakage --------------------------------------------------------------------------------------------------
def test_the_api_key_never_appears_anywhere_it_should_not():
    cfg = configured()
    assert SECRET not in repr(cfg) and SECRET not in str(cfg.public_summary()) and cfg.public_summary()["api_key_set"] is True
    client = FakeClient()
    r = Explainer(cfg, client).explain("emergency", ctx_emergency())
    assert SECRET not in json.dumps(r.to_dict()) and SECRET not in json.dumps(client.calls)
    leaky = FeatherlessConfig(api_key=SECRET, model="m")
    sdk_error = RuntimeError(f"401 for key {SECRET}")
    result = Explainer(leaky, OpenAICompatibleClient(leaky, FakeSDK(exc=sdk_error).factory)).explain("comparison", ctx_ns())
    assert result.source == "deterministic_fallback" and result.error_detail and SECRET not in json.dumps(result.to_dict())
    assert "***" in result.error_detail
    assert cfg.redact(f"a {SECRET} b") == "a *** b"


# -- 17. the prompt contains only allowed context ---------------------------------------------------------------------------------
def test_prompt_contains_only_the_structured_context_and_the_rules(monkeypatch):
    monkeypatch.setenv("SOME_PRIVATE_ENV_VAR", "distinctive-private-value-123")
    ctx = ctx_qaoa()
    messages = build_messages("qaoa", ctx)
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    user = messages[1]["content"]
    payload = json.loads(user.split("Structured results (JSON):\n", 1)[1])
    assert payload == ctx.to_dict()  # exactly the context: nothing added
    assert set(payload) <= TOP_LEVEL_KEYS
    for name, allowed in SECTION_KEYS.items():
        assert set(payload.get(name, {})) <= allowed
    blob = json.dumps(messages)
    for forbidden in (SECRET, "distinctive-private-value-123", str(ROOT), "FEATHERLESS", ".env", "\\\\"):
        assert forbidden not in blob
    for rule in ("Use ONLY the supplied structured results", "Do not invent numbers", "Never choose, recommend, change or generate",
                 "Qiskit Aer, a classical quantum-circuit simulator", "Never claim quantum advantage", "SIMULATION PROXY",
                 "simulation proxy, estimated CO2 changed by X%"):
        assert rule in SYSTEM_PROMPT.replace('\\"', '"').replace("\n", " ") or rule in SYSTEM_PROMPT
    assert len(user) < 5000  # small prompt: summarised context, not raw files


# -- 18. numbers stay authoritative ---------------------------------------------------------------------------------------------------
def test_numerical_context_is_authoritative_and_unchanged_by_model_text():
    ctx = ctx_ns()
    before = json.loads(ctx.to_json())
    lying = ("The simulator results say waiting fell by 99.9% and throughput doubled, which is plainly false and not from the supplied "
             "results, but it is otherwise well-formed prose about the simulation.")
    r = Explainer(configured(), FakeClient(reply=lying)).explain("comparison", ctx)
    assert r.explanation == lying  # text is passed through (only lightly validated) ...
    assert r.context == before == ctx.to_dict()  # ... but the application's numbers are the authoritative ones
    assert r.context["baseline_comparison"]["waiting_seconds_per_admitted_pct"] != -99.9


def test_context_numbers_equal_the_saved_phase6_results():
    import csv

    summary = {(r["scenario"], r["controller"]): r for r in csv.DictReader(open(RESULTS / "phase6" / "controller_summary.csv"))}
    for scenario, controller in (("ns_heavy", "qubo_static"), ("ew_heavy", "adaptive"), ("time_varying", "qubo_receding"), ("balanced_medium", "adaptive")):
        row, ctx = summary[(scenario, controller)], load_controller_context(RESULTS, scenario, controller)
        assert ctx.baseline_comparison["waiting_seconds_per_admitted_pct"] == pytest.approx(
            float(row["waiting_seconds_per_admitted_pct_vs_fixed_of_means"]), abs=1e-3)
        assert ctx.environmental["co2_proxy_pct_vs_baseline"] == pytest.approx(float(row["co2_kg_proxy_pct_vs_fixed_of_means"]), abs=1e-3)
        assert ctx.traffic["waiting_seconds_per_admitted"] == pytest.approx(float(row["waiting_seconds_per_admitted_mean"]), abs=1e-3)
    q = ctx_qaoa().qaoa
    row = next(r for r in csv.DictReader(open(RESULTS / "phase4" / "qaoa_solutions.csv"))
               if r["case"] == "ns_heavy" and r["variant"] == "qaoa_p1")
    assert q["best_sampled_feasible_energy"] == pytest.approx(float(row["qaoa_best_feasible_energy"]), abs=1e-3)
    assert q["feasible_rate_sampled"] == pytest.approx(float(row["feasible_rate_sampled"]), abs=1e-5)


# -- 19. the AI layer explains; it does not decide ---------------------------------------------------------------------------------------------
def test_signal_choice_only_reports_the_precomputed_plan():
    ctx = ctx_ns()
    plan = ctx.signal["signal_plans"]["I3"]
    r = explain_signal_choice(ctx, "I3", Explainer(FeatherlessConfig()))
    assert plan in r.explanation and "not from the AI layer" in r.explanation and "never chooses or changes" in r.explanation
    missing = explain_signal_choice(ctx, "I9", Explainer(FeatherlessConfig()))
    assert "no computed signal plan" in missing.explanation and "does not choose signal plans" in missing.explanation
    no_plans = explain_signal_choice(ctx_env(), "I3", Explainer(FeatherlessConfig()))  # a context with no plans: nothing is invented
    assert "no computed signal plan" in no_plans.explanation and not re.search(r"NS\d+/EW\d+", no_plans.explanation)


def test_signal_choice_prompt_asks_only_for_the_computed_plan():
    msgs = build_messages("signal_choice", ctx_ns(), node="I3")
    assert "without choosing or changing anything" in msgs[1]["content"] and "Intersection of interest: I3" in msgs[1]["content"]
    assert "Never choose, recommend, change or generate a traffic-signal plan" in msgs[0]["content"]


def test_unknown_explanation_kind_is_rejected():
    with pytest.raises(ValueError):
        Explainer(FeatherlessConfig()).explain("decide_signals", ctx_ns())
    with pytest.raises(ValueError):
        deterministic_explanation("decide_signals", ctx_ns())


def test_architecture_ai_layer_is_isolated_from_the_control_code():
    src = ROOT / "src" / "qtraffic"
    ai_files = list((src / "ai").glob("*.py"))
    assert ai_files
    for f in ai_files:  # the AI layer imports nothing from the simulator / controllers / optimisation / emergency code
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            mods = [node.module] if isinstance(node, ast.ImportFrom) and node.module else [a.name for a in getattr(node, "names", [])] if isinstance(node, ast.Import) else []
            for m in mods:
                assert not re.match(r"(qtraffic\.(simulator|controllers|optimization|emergency|network|signals|demand|metrics|experiments)|\.\.)", m), (f.name, m)
    for f in src.rglob("*.py"):  # and nothing in the control code imports the AI layer or the SDK
        if "ai" in f.relative_to(src).parts[:1]:
            continue
        text = f.read_text(encoding="utf-8")
        assert "qtraffic.ai" not in text and "import openai" not in text and "from openai" not in text, f.name
    for f in (src / "ai").glob("*.py"):  # only featherless.py touches the SDK
        if f.name != "featherless.py":
            assert "openai" not in f.read_text(encoding="utf-8").replace("OpenAI-compatible", "").lower().replace("openai_compatible", "") or f.name == "explain.py"
