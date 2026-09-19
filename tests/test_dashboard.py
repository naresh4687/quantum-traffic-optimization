"""Non-visual tests of the dashboard logic (no browser, no Streamlit runtime, no network)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from qtraffic.ai import Explainer
from qtraffic.ai.config import FeatherlessConfig
from qtraffic.dashboard import charts, theme
from qtraffic.dashboard.network_view import build_network_figure
from qtraffic.dashboard.saved import SavedResults, load_saved_results
from qtraffic.dashboard.state import (
    AI_ACTIONS, CONTROLLERS, DEFAULT_CONFIG, PRIMARY_ROUTE, SCENARIOS, DashboardConfigError, RunConfig, available_controllers, badge_for,
    config_from_query, default_playhead, emergency_bounds, emergency_view, explain_action, kpis, live_context, live_emergency_context,
    network_view, normalize_config, parse_selected_node, run_dashboard, status_for, validate_config,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"


@pytest.fixture(scope="module")
def saved() -> SavedResults:
    return load_saved_results(RESULTS)


@pytest.fixture(scope="module")
def default_run(saved):
    return run_dashboard(DEFAULT_CONFIG, saved)


@pytest.fixture(scope="module")
def finished_run(saved):  # Fixed, balanced demand, corridor completes well before the end of the run
    return run_dashboard(RunConfig(scenario="balanced_medium", controller="fixed"), saved)


@pytest.fixture(scope="module")
def no_emergency_run(saved):
    return run_dashboard(RunConfig(scenario="ns_heavy", controller="adaptive", emergency_enabled=False), saved)


# ---------------------------------------------------------------------------------------------- saved results
def test_saved_results_load_real_files(saved):
    assert saved.has_phase6 and saved.has_qaoa
    assert saved.summary_row("ns_heavy", "adaptive") is not None
    assert saved.summary_row("ns_heavy", "qaoa_p1") is not None
    assert saved.qaoa_row("ns_heavy", "qaoa_p1") is not None


def test_missing_results_directory_degrades_without_raising(tmp_path):
    empty = load_saved_results(tmp_path / "does_not_exist")
    assert not empty.has_phase6 and not empty.has_qaoa
    assert empty.warnings  # the gaps are reported, not hidden
    assert empty.controller_summary == [] and empty.qaoa_runs == []
    assert empty.qaoa_plans("ns_heavy", 0) is None


def test_test_count_is_optional_and_read_when_present(tmp_path):
    assert load_saved_results(tmp_path).test_count is None
    (tmp_path / "dashboard").mkdir()
    (tmp_path / "dashboard" / "test_count.json").write_text(json.dumps({"tests_passed": 12}), encoding="utf-8")
    assert load_saved_results(tmp_path).test_count == {"tests_passed": 12}


def test_saved_qaoa_plans_are_valid_signal_plans(saved):
    plans = saved.qaoa_plans("ns_heavy", 0)
    assert plans is not None and set(plans) == {f"I{i}" for i in range(1, 7)}
    assert all(p.ns_green in (20, 30, 40) and p.ns_green + p.ew_green == 60 for p in plans.values())


# ---------------------------------------------------------------------------------------------- configuration
def test_scenarios_and_controllers_are_the_specified_sets():
    assert set(SCENARIOS) == {"balanced_medium", "ns_heavy", "ew_heavy", "time_varying", "downstream_congested"}
    assert set(CONTROLLERS) == {"fixed", "adaptive", "qubo_static", "qubo_receding", "qaoa_p1"}


def test_qaoa_is_only_offered_when_a_saved_plan_exists(saved):
    assert "qaoa_p1" in available_controllers("ns_heavy", 0, saved)
    assert "qaoa_p1" not in available_controllers("ns_heavy", 3, saved)
    assert "qaoa_p1" not in available_controllers("ns_heavy", 0, None)
    assert {"fixed", "adaptive", "qubo_static", "qubo_receding"} <= set(available_controllers("ns_heavy", 3, saved))


@pytest.mark.parametrize("bad", [
    dict(scenario="nope"), dict(controller="magic"), dict(seed=-1), dict(seed=1000), dict(seed=True), dict(cycles=5), dict(cycles=10_000),
    dict(route=("I1", "I6")), dict(route=("I1", "I9")), dict(emergency_start=1), dict(emergency_start=10_000),
    dict(controller="qaoa_p1", seed=3),
])
def test_invalid_configurations_are_rejected(bad, saved):
    with pytest.raises(DashboardConfigError):
        validate_config(RunConfig(**bad), saved)


def test_default_and_valid_configs_pass(saved):
    validate_config(DEFAULT_CONFIG, saved)
    validate_config(RunConfig(controller="qaoa_p1"), saved)
    validate_config(RunConfig(emergency_enabled=False, route=("garbage",)), saved)  # route ignored when the emergency is off


def test_run_dashboard_refuses_invalid_config(saved):
    with pytest.raises(DashboardConfigError):
        run_dashboard(RunConfig(controller="qaoa_p1", seed=5), saved)


def test_normalize_config_ignores_emergency_settings_when_off():
    a = RunConfig(emergency_enabled=False, route=("I4", "I5", "I6"), emergency_start=33)
    b = RunConfig(emergency_enabled=False)
    assert normalize_config(a) == normalize_config(b)
    assert normalize_config(RunConfig()) == normalize_config(RunConfig(emergency_start=RunConfig().resolved_start()))
    assert normalize_config(RunConfig(seed=1)) != normalize_config(RunConfig(seed=2))


def test_emergency_bounds_and_default_start():
    lo, hi, default = emergency_bounds("ns_heavy", 60)
    assert lo < default < hi
    assert DEFAULT_CONFIG.resolved_start() == default


def test_config_from_query_accepts_valid_and_notes_invalid(saved):
    cfg, playhead, notes = config_from_query({"scenario": "ew_heavy", "controller": "qubo_receding", "seed": "2", "cycles": "40",
                                              "emergency": "0", "cycle": "55"}, saved)
    assert (cfg.scenario, cfg.controller, cfg.seed, cfg.cycles, cfg.emergency_enabled, playhead) == ("ew_heavy", "qubo_receding", 2, 40, False, 55)
    assert notes == []
    cfg, _, notes = config_from_query({"scenario": "bogus", "seed": "x", "cycle": "y"}, saved)
    assert cfg == DEFAULT_CONFIG and len(notes) == 3
    cfg, _, notes = config_from_query({"controller": "qaoa_p1", "seed": "4"}, saved)  # parses but is not runnable
    assert cfg == DEFAULT_CONFIG and any("not runnable" in n for n in notes)
    cfg, _, _ = config_from_query({"route": "I1>I2>I3>I6", "emergency": "true"}, saved)
    assert cfg.route == PRIMARY_ROUTE and cfg.emergency_enabled


# ---------------------------------------------------------------------------------------------- live runs and KPIs
def test_live_run_is_deterministic(saved, default_run):
    again = run_dashboard(DEFAULT_CONFIG, saved)
    assert again.unified.traffic.waiting_vehicle_seconds == default_run.unified.traffic.waiting_vehicle_seconds
    assert again.emergency.vehicle.travel_time == default_run.emergency.vehicle.travel_time


def test_kpis_match_unified_metrics(default_run):
    k = kpis(default_run)
    t, e = default_run.unified.traffic, default_run.unified.environmental
    assert k["waiting_seconds_per_admitted"] == t.waiting_seconds_per_admitted
    assert k["throughput_vehicles_per_hour"] == t.throughput_vehicles_per_hour
    assert k["average_queue_vehicles"] == t.average_queue_vehicles and k["max_queue_vehicles"] == t.max_queue_vehicles
    assert k["co2_kg_proxy"] == e.co2_kg and k["fuel_liters_proxy"] == e.fuel_liters
    assert set(k["delta_vs_fixed_pct"]) >= {"waiting_seconds_per_admitted", "co2_kg_proxy"}


def test_fixed_controller_has_no_self_comparison(finished_run):
    assert finished_run.baseline_unified is None
    assert "delta_vs_fixed_pct" not in kpis(finished_run)


def test_co2_proxy_is_linear_in_waiting(default_run):
    e, t = default_run.unified.environmental, default_run.unified.traffic
    cfg = e.estimate.config
    fuel = t.waiting_vehicle_seconds / 3600.0 * cfg.idle_fuel_rate_lph
    assert math.isclose(e.fuel_liters, fuel, rel_tol=1e-9)
    assert math.isclose(e.co2_kg, fuel * cfg.emission_factor_kg_per_liter, rel_tol=1e-9)


def test_live_emergency_run_reproduces_saved_result(default_run):
    assert default_run.verification.ok is True
    assert "saved" in default_run.verification.text


def test_live_no_emergency_run_reproduces_saved_phase6(no_emergency_run):
    assert no_emergency_run.verification.ok is True
    assert no_emergency_run.emergency is None and no_emergency_run.emergency_off is None


def test_qaoa_live_run_uses_saved_plan(saved):
    run = run_dashboard(RunConfig(scenario="ns_heavy", controller="qaoa_p1", emergency_enabled=False), saved)
    plans = saved.qaoa_plans("ns_heavy", 0)
    assert all(rec.plans[n] == plans[n] for rec in run.history for n in plans)


def test_static_qubo_reports_exact_energy(saved):
    run = run_dashboard(RunConfig(controller="qubo_static", emergency_enabled=False, cycles=20), saved)
    assert run.qubo_energy is not None and math.isfinite(run.qubo_energy)


# ---------------------------------------------------------------------------------------------- network view
def test_network_view_transforms_simulator_state(default_run):
    idx = default_playhead(default_run)
    view = network_view(default_run, idx)
    rec = default_run.history[idx]
    assert view.cycle == rec.cycle and len(view.nodes) == 6 and len(view.roads) == len(default_run.net.approaches) == 24
    assert math.isclose(view.total_queue, sum(rec.queue_after_arrivals.values()))
    for n in view.nodes:
        assert n.plan == rec.plans[n.node].label and n.ns_green + n.ew_green == 60
        assert n.congestion in ("normal", "high", "severe")
    assert view.exit_stubs
    for r in view.roads:
        assert r.congestion == theme_class(r.ratio)


def theme_class(ratio: float) -> str:
    t = theme.CONGESTION_THRESHOLDS
    return "severe" if ratio >= t["severe"] else "high" if ratio >= t["high"] else "normal"


def test_network_view_marks_emergency_priority_only_while_the_override_acts(default_run):
    during = network_view(default_run, default_playhead(default_run))
    assert any(n.priority for n in during.nodes)
    assert not any(n.priority for n in network_view(default_run, 0).nodes)
    assert not any(n.priority for n in network_view(default_run, default_run.n_cycles - 1).nodes)


def test_network_view_without_emergency_has_no_priority(no_emergency_run):
    v = network_view(no_emergency_run, 10)
    assert not any(n.priority or n.overridden for n in v.nodes)
    assert all(n.plan == n.base_plan for n in v.nodes)


def test_network_view_clamps_the_index(default_run):
    assert network_view(default_run, -5).index == 0
    assert network_view(default_run, 10_000).index == default_run.n_cycles - 1


def test_network_figure_builds_for_every_state(default_run, no_emergency_run):
    for run in (default_run, no_emergency_run):
        for idx in (0, default_playhead(run), run.n_cycles - 1):
            fig = build_network_figure(network_view(run, idx), emergency_view(run, idx), "I2")
            assert len(fig.data) > 0


# ---------------------------------------------------------------------------------------------- emergency view
def test_emergency_view_phases_follow_the_simulated_vehicle(default_run):
    v = default_run.emergency.vehicle
    first = default_run.first_cycle
    assert emergency_view(default_run, 0).phase == "pending"
    active = emergency_view(default_run, v.entry_cycle - first + 1)
    assert active.phase == "active" and active.current in v.route and active.position is not None
    assert list(active.completed) + [active.current] + list(active.upcoming) == list(v.route)
    end = emergency_view(default_run, default_run.n_cycles - 1)
    assert end.phase == "complete" and end.completed == v.route and end.position is None


def test_emergency_view_reports_measured_metrics(default_run):
    ev = emergency_view(default_run, default_playhead(default_run))
    v, off = default_run.emergency.vehicle, default_run.emergency_off.vehicle
    assert ev.travel_time_s == v.travel_time and ev.travel_time_s_no_override == off.travel_time
    assert ev.stops == v.stops and ev.completion_cycle == v.completion_cycle
    assert v.travel_time < off.travel_time  # the corridor helped the EV in this configuration


def test_normal_controller_restored_only_after_completion(finished_run):
    v = finished_run.emergency.vehicle
    assert v.completed
    before = emergency_view(finished_run, v.entry_cycle - finished_run.first_cycle + 1)
    assert not before.restored
    after = emergency_view(finished_run, finished_run.n_cycles - 1)
    assert after.phase == "complete" and after.restored
    assert not any(n.priority for n in network_view(finished_run, finished_run.n_cycles - 1).nodes)


def test_emergency_view_off_when_disabled(no_emergency_run):
    assert emergency_view(no_emergency_run, 5).phase == "off"


def test_status_reflects_actual_state(default_run, no_emergency_run, finished_run):
    assert status_for(default_run, default_playhead(default_run), False) == ("EMERGENCY ACTIVE", "emergency")
    assert status_for(no_emergency_run, 5, False)[0] == "SYSTEM READY"
    assert status_for(no_emergency_run, 5, True)[0] == "SIMULATION RUNNING"
    assert status_for(finished_run, finished_run.n_cycles - 1, False)[0] == "SYSTEM READY"


# ---------------------------------------------------------------------------------------------- selection parsing
NODES = [f"I{i}" for i in range(1, 7)]


def test_parse_selected_node():
    ev = {"selection": {"points": [{"customdata": ["I4"]}]}}
    assert parse_selected_node(ev, NODES) == "I4"
    assert parse_selected_node({"selection": {"points": [{"customdata": "I2"}]}}, NODES) == "I2"
    assert parse_selected_node({"selection": {"points": []}}, NODES) is None
    assert parse_selected_node({"selection": {"points": [{"customdata": ["X9"]}]}}, NODES) is None
    assert parse_selected_node({}, NODES) is None
    assert parse_selected_node(None, NODES) is None


# ---------------------------------------------------------------------------------------------- AI layer (no key)
@pytest.fixture()
def keyless_explainer(monkeypatch):
    for var in ("FEATHERLESS_API_KEY", "FEATHERLESS_MODEL", "FEATHERLESS_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    return Explainer(FeatherlessConfig.from_env({}))


def test_live_contexts_are_valid_for_the_ai_layer(default_run, no_emergency_run):
    ctx = live_context(default_run)
    assert ctx.scenario == "ns_heavy" and ctx.controller == "adaptive" and ctx.baseline_controller == "fixed"
    assert ctx.traffic["waiting_seconds_per_admitted"] == round(default_run.unified.traffic.waiting_seconds_per_admitted, 4)
    assert "SIMULATION PROXY" in ctx.environmental["label"].upper()
    assert live_emergency_context(default_run) is not None
    assert live_emergency_context(no_emergency_run) is None


def test_ai_actions_fall_back_locally_without_a_key(default_run, saved, keyless_explainer):
    for action in AI_ACTIONS:
        out = explain_action(action, default_run, saved, keyless_explainer, RESULTS)
        assert out
        for title, result in out:
            assert result.source != "featherless" and result.explanation.strip()
            label, css, _ = badge_for(result)
            assert (label, css) == ("LOCAL FALLBACK", "local")
    assert keyless_explainer.requests_made == 0  # no API request without a key


def test_environment_explanation_states_it_is_a_proxy(default_run, saved, keyless_explainer):
    ((_, result),) = explain_action("environment", default_run, saved, keyless_explainer, RESULTS)
    assert "simulation proxy" in result.explanation.lower()


def test_emergency_action_needs_an_emergency(no_emergency_run, saved, keyless_explainer):
    with pytest.raises(DashboardConfigError):
        explain_action("emergency", no_emergency_run, saved, keyless_explainer, RESULTS)
    with pytest.raises(DashboardConfigError):
        explain_action("unknown", no_emergency_run, saved, keyless_explainer, RESULTS)


# ---------------------------------------------------------------------------------------------- charts and theme
def test_comparison_rows_come_from_saved_results(saved):
    rows = charts.comparison_rows(saved, "ns_heavy")
    assert [r["key"] for r in rows][:2] == ["fixed", "adaptive"]
    fixed = saved.summary_row("ns_heavy", "fixed")
    assert rows[0]["waiting"] == fixed["waiting_seconds_per_admitted_mean"]
    assert len(charts.comparison_figures(rows, "adaptive")) == 3
    assert charts.comparison_rows(SavedResults(), "ns_heavy") == []


def test_timeline_and_heatmap_build(default_run):
    idx = default_playhead(default_run)
    assert charts.queue_timeline(default_run, idx).data
    assert charts.signal_heatmap(default_run, idx).data


def test_sensitivity_and_feasibility_use_saved_values(saved):
    rows = charts.sensitivity_rows(saved, "ns_heavy", "adaptive")
    assert rows and all(r["co2"] is not None for r in rows)
    assert charts.sensitivity_figure(rows, "Adaptive").data
    row = saved.qaoa_row("ns_heavy", "qaoa_p1")
    assert charts.feasibility_figure(row).data


def test_theme_defines_css_variables_and_reduced_motion():
    sheet = theme.css()
    for name in ("--bg", "--surface", "--accent", "--positive", "--warning", "--emergency", "--text", "--muted"):
        assert name + ":" in sheet
    assert theme.PALETTE["bg"].lower() == "#080b10"
    assert "prefers-reduced-motion" in sheet
    assert "--accent_dim" not in sheet


def test_theme_config_matches_palette():
    text = (Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml").read_text(encoding="utf-8").lower()
    assert theme.PALETTE["bg"].lower() in text and theme.PALETTE["accent"].lower() in text


def test_no_secrets_or_paths_in_dashboard_sources():
    src = Path(__file__).resolve().parents[1] / "src" / "qtraffic" / "dashboard"
    for path in src.glob("*.py"):
        body = path.read_text(encoding="utf-8")
        assert "FEATHERLESS_API_KEY=" not in body and "sk-" not in body, path.name


# ---------------------------------------------------------------------------------------------- app-level regression (Streamlit AppTest)
def _app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=120)


def test_app_starts_without_exceptions_and_shows_the_default_demo():
    at = _app().run()
    assert not at.exception
    assert at.session_state["w_start"] == DEFAULT_CONFIG.resolved_start() == 40


def test_changing_the_cycle_count_keeps_the_chosen_emergency_start():
    """Regression: the start-cycle slider's bounds follow the cycle count, and Streamlit used to reset it to its minimum."""
    at = _app().run()
    at.slider(key="w_start").set_value(45).run()
    at.slider(key="w_cycles").set_value(40).run()
    assert not at.exception
    assert at.session_state["w_start"] == 45
    at.button(key="btn_run").click().run()
    assert not at.exception
    assert at.session_state["cfg"].cycles == 40 and at.session_state["cfg"].resolved_start() == 45


# ---------------------------------------------------------------------------------------------- documented demo preset
DEMO_QUERY = {"scenario": "ns_heavy", "controller": "adaptive", "seed": "0", "cycles": "60", "emergency": "1",
              "route": "I1,I2,I3,I6", "start": "30"}


def test_demo_preset_url_builds_the_documented_configuration(saved):
    cfg, playhead, notes = config_from_query(DEMO_QUERY, saved)
    assert notes == [] and playhead is None
    assert (cfg.scenario, cfg.controller, cfg.seed, cfg.cycles, cfg.emergency_enabled, cfg.route, cfg.emergency_start) == (
        "ns_heavy", "adaptive", 0, 60, True, ("I1", "I2", "I3", "I6"), 30)


def test_demo_preset_runs_cleanly_and_the_corridor_completes_and_restores(saved):
    cfg, _, _ = config_from_query(DEMO_QUERY, saved)
    run = run_dashboard(cfg, saved)
    v = run.emergency.vehicle
    assert v.completed and [c.node for c in v.crossings] == ["I1", "I2", "I3", "I6"] and v.entry_cycle == 30
    assert v.travel_time < run.emergency_off.vehicle.travel_time  # corridor helps the EV
    assert status_for(run, default_playhead(run), False)[0] == "EMERGENCY ACTIVE"
    last = run.n_cycles - 1
    assert emergency_view(run, last).restored and not any(n.priority for n in network_view(run, last).nodes)
    assert run.verification.ok is None  # honest: only start cycle 40 has a saved counterpart
