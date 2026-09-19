"""QUAN115 control-room dashboard.  Run:  streamlit run app.py

Thin entry point: layout and session state only. All traffic numbers come from the live simulator or the saved Phase 4-6
results through ``qtraffic.dashboard``; nothing is hard-coded. Works without an API key: the AI analyst falls back to
deterministic local analysis. Optional URL parameters: ?scenario=&controller=&seed=&cycles=&emergency=1&route=&start=&cycle=
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from qtraffic.ai import Explainer
from qtraffic.dashboard import components as ui
from qtraffic.dashboard.charts import queue_timeline, signal_heatmap
from qtraffic.dashboard.network_view import build_network_figure
from qtraffic.dashboard.saved import SavedResults, load_saved_results
from qtraffic.dashboard.state import (
    CONTROLLERS, DEFAULT_CONFIG, EMERGENCY_ROUTES, MAX_CYCLES, MAX_SEED, MIN_CYCLES, SCENARIO_INFO, SCENARIOS, DashboardConfigError,
    DashboardRun, RunConfig, available_controllers, config_from_query, default_playhead, emergency_bounds, emergency_view, explain_action,
    network_view, normalize_config, parse_selected_node, run_dashboard, status_for, validate_config,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SPEEDS = {"1×": 1.0, "2×": 0.5, "4×": 0.25}  # seconds per cycle while playing
ROUTE_LABELS = {" > ".join(v): v for v in EMERGENCY_ROUTES.values()}

st.set_page_config(page_title="QUAN115 · Traffic Control Center", layout="wide", initial_sidebar_state="expanded")
ss = st.session_state


# ------------------------------------------------------------------------------------------------- cached resources
@st.cache_data(show_spinner=False)
def load_saved(path: str) -> SavedResults:
    return load_saved_results(path)


@st.cache_resource(show_spinner="Running the simulator…", max_entries=24)
def cached_run(cfg: RunConfig, results_path: str) -> DashboardRun:
    return run_dashboard(cfg, load_saved(results_path))


@st.cache_resource(show_spinner=False)
def get_explainer() -> Explainer:
    return Explainer()  # configuration from the environment; no key needed


saved = load_saved(str(RESULTS_DIR))


# ------------------------------------------------------------------------------------------------- state
def apply_config(cfg: RunConfig, playhead: int | None = None) -> None:
    run = cached_run(cfg, str(RESULTS_DIR))
    ss.cfg, ss.run = cfg, run
    offset = default_playhead(run) if playhead is None else max(0, min(playhead - run.first_cycle, run.n_cycles - 1))
    ss.cycle = run.first_cycle + offset
    ss.playing, ss.ai_results, ss.selected_node, ss.net_pick = False, None, "I2", None


def set_widgets(cfg: RunConfig) -> None:
    ss.w_scenario, ss.w_controller, ss.w_seed, ss.w_cycles = cfg.scenario, cfg.controller, cfg.seed, cfg.cycles
    ss.w_emg, ss.w_route = cfg.emergency_enabled, " > ".join(cfg.route)
    ss.w_start = cfg.resolved_start()


def widget_config() -> RunConfig:
    return RunConfig(scenario=ss.w_scenario, controller=ss.w_controller, seed=int(ss.w_seed), cycles=int(ss.w_cycles),
                     emergency_enabled=bool(ss.w_emg), route=ROUTE_LABELS[ss.w_route], emergency_start=int(ss.w_start))


def cb_run() -> None:
    try:
        cfg = widget_config()
        validate_config(cfg, saved)
        apply_config(cfg)
        ss.config_error = None
    except DashboardConfigError as exc:
        ss.config_error = str(exc)


def cb_emergency() -> None:
    ss.w_emg = True
    cb_run()


def cb_reset() -> None:
    set_widgets(DEFAULT_CONFIG)
    apply_config(DEFAULT_CONFIG)
    ss.config_error = None


if "run" not in ss:
    cfg0, playhead0, notes = config_from_query(dict(st.query_params), saved)
    set_widgets(cfg0)
    apply_config(cfg0, playhead0)
    ss.query_notes, ss.config_error, ss.speed = notes, None, "1×"

run: DashboardRun = ss.run

# ------------------------------------------------------------------------------------------------- sidebar: control center
with st.sidebar:
    st.markdown('<div class="qt-label" style="margin-bottom:0.3rem">CONTROL CENTER</div>', unsafe_allow_html=True)
    ui.inject_css()
    st.selectbox("Scenario", SCENARIOS, key="w_scenario", format_func=lambda s: SCENARIO_INFO[s][0])
    st.caption(f"Demand: {SCENARIO_INFO[ss.w_scenario][1]} (fixed by the scenario)")
    st.number_input("Seed", 0, MAX_SEED, key="w_seed", step=1, help="Deterministic: the same seed always gives the same demand")
    options = available_controllers(ss.w_scenario, int(ss.w_seed), saved)
    if ss.w_controller not in options:
        ss.w_controller = DEFAULT_CONFIG.controller
    st.selectbox("Controller", options, key="w_controller", format_func=lambda k: CONTROLLERS[k].label)
    st.caption(CONTROLLERS[ss.w_controller].description)
    if "qaoa_p1" not in options:
        st.caption("QAOA p=1 plans are saved for the Phase 4 seed only, so it is unavailable for this seed.")
    st.slider("Simulation cycles", MIN_CYCLES, MAX_CYCLES, key="w_cycles", help="Each cycle is 60 s of simulated time")
    st.divider()
    st.markdown('<div class="qt-label" style="margin-bottom:0.3rem">EMERGENCY</div>', unsafe_allow_html=True)
    st.toggle("Enable emergency vehicle (EV1)", key="w_emg")
    st.selectbox("Route", list(ROUTE_LABELS), key="w_route", disabled=not ss.w_emg)
    lo, hi, default_start = emergency_bounds(ss.w_scenario, int(ss.w_cycles))
    if not lo <= int(ss.w_start) <= hi:
        ss.w_start = default_start
    st.slider("Start cycle", lo, hi, key="w_start", disabled=not ss.w_emg, help="Absolute simulator cycle at which EV1 enters")
    st.button("Run simulation", type="primary", width="stretch", on_click=cb_run, key="btn_run")
    c1, c2 = st.columns(2)
    c1.button("Run emergency", width="stretch", on_click=cb_emergency, key="btn_emg", help="Enable the emergency and run")
    c2.button("Reset", width="stretch", on_click=cb_reset, key="btn_reset", help="Back to the default demo")
    try:
        pending = normalize_config(widget_config()) != normalize_config(ss.cfg)
    except Exception:  # noqa: BLE001 - transient widget states while options change
        pending = False
    if ss.config_error:
        st.error(ss.config_error)
    elif pending:
        st.caption("Settings changed. Press Run simulation to apply.")
    for n in ss.query_notes:
        st.caption(n)
    st.divider()
    ui.experiment_info(run)
    for w in saved.warnings:
        st.caption(w)

# ------------------------------------------------------------------------------------------------- live region (auto-refreshing fragment)
interval = SPEEDS[ss.speed] if ss.playing else None


def live_region() -> None:
    run: DashboardRun = ss.run
    last = run.first_cycle + run.n_cycles - 1
    if ss.playing:
        if ss.cycle >= last:
            ss.playing = False
            st.rerun()
        else:
            ss.cycle += 1
    idx = int(ss.cycle) - run.first_cycle
    view, ev = network_view(run, idx), emergency_view(run, idx)
    ui.header(run, status_for(run, idx, ss.playing), saved)
    ui.kpi_strip(run)
    # playback controls
    b1, b2, b3, b4, b5 = st.columns([0.55, 0.55, 0.55, 1.5, 6], vertical_alignment="center")
    if b1.button("▶", key="btn_play", width="stretch", disabled=bool(ss.playing), help="Play the cycles"):
        ss.playing = True
        if ss.cycle >= last:
            ss.cycle = run.first_cycle
        st.rerun()
    if b2.button("⏸", key="btn_pause", width="stretch", disabled=not ss.playing, help="Pause"):
        ss.playing = False
        st.rerun()
    if b3.button("↺", key="btn_rewind", width="stretch", help="Rewind to the first cycle"):
        ss.playing, ss.cycle = False, run.first_cycle
        st.rerun()
    speed = b4.segmented_control("Speed", list(SPEEDS), default=ss.speed, key="speed_sel", label_visibility="collapsed")
    if speed and speed != ss.speed:
        ss.speed = speed
        st.rerun()
    b5.slider("Cycle", run.first_cycle, last, key="cycle", format="cycle %d", label_visibility="collapsed")
    idx = int(ss.cycle) - run.first_cycle

    left, right = st.columns([2.05, 1], gap="medium")
    with left:
        ui.section("Urban Traffic Network — schematic", f"cycle {view.cycle} · {view.total_queue:.0f} vehicles queued")
        event = st.plotly_chart(build_network_figure(view, ev, ss.selected_node), width="stretch", config=ui.PLOT_CONFIG, key="net",
                                on_select="rerun", selection_mode="points")
        pick = parse_selected_node(event, [n.node for n in view.nodes])
        if pick and pick != ss.net_pick:
            ss.net_pick, ss.selected_node = pick, pick
            st.rerun(scope="fragment")
        ui.network_legend()
        ui.section("Queue and signal timelines", "playhead = white line · shaded = emergency window · red squares = override")
        ui.plot(queue_timeline(run, idx), "tl_queue")
        ui.plot(signal_heatmap(run, idx), "tl_signal")
        st.caption("Signal-plan timeline: NS green per intersection per cycle. Dark = 20 s, grey = 30 s, cyan = 40 s (EW gets the remainder).")
    with right:
        ui.section("Emergency")
        ui.emergency_panel(run, ev, saved)
        ui.section("Intersection")
        st.segmented_control("Intersection", [n.node for n in view.nodes], key="selected_node", label_visibility="collapsed")
        ui.inspector(view.node(ss.selected_node or "I2"), run, ev)
        ui.signals_panel(view, ss.selected_node)


st.fragment(live_region, run_every=interval)()

# ------------------------------------------------------------------------------------------------- static sections
oc, ec = st.columns([1.15, 1], gap="medium")
with oc:
    ui.section("Quantum optimization", "QUBO · QAOA · saved Phase 4")
    ui.optimization_panel(run, saved)
with ec:
    ui.section("Environmental proxy", "simulation estimate")
    ui.environment_panel(run, saved)

ui.section("Controller comparison", f"scenario: {SCENARIO_INFO[run.config.scenario][0]}")
ui.comparison_section(run, saved)

ui.section("AI traffic analyst", "explains measured results · never controls")
ai_cfg = get_explainer().config
ai_reason = ai_cfg.unavailable_reason()
ai_status = (("FEATHERLESS READY", "ok", f"Featherless configured (model {ai_cfg.model})") if ai_reason is None else
             ("LOCAL FALLBACK", "local", "LOCAL ANALYSIS: Featherless is not configured" if ai_reason in ("missing_api_key", "missing_model")
              else f"LOCAL ANALYSIS: Featherless unavailable ({ai_reason})"))
action = ui.ai_panel(run, ai_status)
if action:
    with st.spinner("Analysing measured results…"):
        try:
            ss.ai_results = explain_action(action, run, saved, get_explainer(), RESULTS_DIR)
        except DashboardConfigError as exc:
            ss.ai_results = None
            st.info(str(exc))
if ss.ai_results:
    ui.ai_result(ss.ai_results)
else:
    st.caption("Choose an analysis above. With no Featherless key the analyst runs a deterministic local analysis of the same numbers.")

with st.expander("System architecture"):
    ui.architecture_panel()
