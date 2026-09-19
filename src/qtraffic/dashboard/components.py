"""Streamlit rendering components. They render view-models from ``state`` / ``charts``; they compute no traffic numbers."""

from __future__ import annotations

from html import escape

import streamlit as st

from ..optimization import VariableMap
from . import charts
from .saved import SavedResults
from .state import (
    AI_ACTION_LABELS, AI_ACTIONS, CONTROLLERS, DashboardRun, EmergencyView, NetworkView, NodeView, badge_for, kpis,
)
from .theme import css

PLOT_CONFIG = {"displayModeBar": False}


def h(x) -> str:
    return escape(str(x))


def inject_css() -> None:
    st.markdown(css(), unsafe_allow_html=True)


def section(title: str, right: str = "") -> None:
    st.markdown(f'<div class="qt-section"><span>{h(title)}</span><em>{h(right)}</em></div>', unsafe_allow_html=True)


def note(html: str, warn: bool = False) -> None:
    st.markdown(f'<div class="qt-note{" warn" if warn else ""}">{html}</div>', unsafe_allow_html=True)


def plot(fig, key: str, **kwargs):
    return st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, key=key, **kwargs)


# ------------------------------------------------------------------------------------------------- header
def header(run: DashboardRun, status: tuple[str, str], saved: SavedResults | None) -> None:
    n = len(run.net.nodes)
    chips = [f'<span class="qt-chip"><b>{n}</b> intersections</span>',
             f'<span class="qt-chip"><b>{VariableMap(run.net.nodes).n_variables}</b> QUBO variables</span>',
             '<span class="qt-chip">QAOA <b>p=1 / p=2</b></span>']
    if saved is not None and saved.test_count:
        chips.append(f'<span class="qt-chip"><b>{h(saved.test_count.get("tests_passed"))}</b> tests</span>')
    label, cls = status
    st.markdown(
        f'<div class="qt-top"><div class="qt-brand"><span class="qt-logo">QUAN115</span>'
        f'<span class="qt-title">Quantum-Enhanced Adaptive Urban Traffic Optimization</span></div>'
        f'<div class="qt-chips">{"".join(chips)}<span class="qt-status {cls}"><i></i>{h(label)}</span></div></div>'
        + ('<div class="qt-emergency-bar"></div>' if cls == "emergency" else ""), unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------------- KPI strip
def _delta(pct, lower_is_better: bool) -> str:
    if pct is None:
        return ""
    if abs(pct) < 0.05:
        return '<span class="qt-delta flat">= Fixed</span>'
    good = (pct < 0) == lower_is_better
    return f'<span class="qt-delta {"good" if good else "bad"}">{pct:+.1f}% vs Fixed</span>'


def kpi_card(label: str, value: str, unit: str, sub: str, delta_html: str = "") -> str:
    return (f'<div class="qt-card"><div class="qt-label">{label}</div><div class="qt-kpi">{value}<small>{h(unit)}</small></div>'
            f'<div class="qt-sub">{delta_html}{" · " if delta_html and sub else ""}{h(sub)}</div></div>')


def kpi_strip(run: DashboardRun) -> None:
    k = kpis(run)
    d = k.get("delta_vs_fixed_pct", {})
    cards = [
        kpi_card("Waiting", f"{k['waiting_seconds_per_admitted']:.1f}", "s / admitted vehicle", "", _delta(d.get("waiting_seconds_per_admitted"), True)),
        kpi_card("Throughput", f"{k['throughput_vehicles_per_hour']:,.0f}", "veh/h", "", _delta(d.get("throughput_vehicles_per_hour"), False)),
        kpi_card("Avg queue", f"{k['average_queue_vehicles']:.1f}", "vehicles", "per approach, time-avg", _delta(d.get("average_queue_vehicles"), True)),
        kpi_card("Max queue", f"{k['max_queue_vehicles']:.1f}", "vehicles", "peak, any stop line", _delta(d.get("max_queue_vehicles"), True)),
        kpi_card("CO₂ proxy", f"{k['co2_kg_proxy']:.1f}", "kg", "simulation estimate", _delta(d.get("co2_kg_proxy"), True)),
    ]
    st.markdown(f'<div class="qt-kpis">{"".join(cards)}</div>', unsafe_allow_html=True)
    base = "Fixed on the same demand" + (" and emergency" if run.config.emergency_enabled else "")
    st.caption(f"Full run of {run.n_cycles} cycles, measured from the live simulator. Percentages compare with {base}."
               if run.baseline_unified is not None else f"Full run of {run.n_cycles} cycles, measured from the live simulator (Fixed is the baseline).")


# ------------------------------------------------------------------------------------------------- signals / inspector
def _lamp(on: bool, name: str) -> str:
    return f'<span class="qt-lamp {"g" if on else "r"}"><i></i>{name} {"GREEN" if on else "RED"}</span>'


def signals_panel(view: NetworkView, selected: str | None) -> None:
    rows = []
    for n in view.nodes:
        ns_first = n.first_phase == "NS"
        tags = (' <span class="qt-tag em">override</span>' if n.overridden else "") + (' <span class="qt-tag em">priority</span>' if n.priority and not n.overridden else "")
        total = n.ns_green + n.ew_green
        rows.append(f'<div class="qt-sig{" sel" if n.node == selected else ""}"><div class="n">{n.node}</div>'
                    f'<div>{_lamp(ns_first, "NS")}{_lamp(not ns_first, "EW")}{tags}</div>'
                    f'<div class="qt-splitbar"><b class="ns" style="width:{100 * n.ns_green / total:.0f}%"></b><b class="ew" style="width:{100 * n.ew_green / total:.0f}%"></b></div></div>')
    st.markdown(f'<div class="qt-card"><div class="qt-label">Signals · cycle {view.cycle}</div>{"".join(rows)}'
                f'<div class="qt-sub">Lamp = phase that opens the 60 s cycle (NS first; the emergency phase first under priority). Bar = NS / EW green split.</div></div>',
                unsafe_allow_html=True)


def inspector(node: NodeView, run: DashboardRun, ev: EmergencyView) -> None:
    def kv(k, v):
        return f"<dt>{h(k)}</dt><dd>{v}</dd>"

    rows = [kv("Controller", h(run.spec.label)),
            kv("Current plan", f"NS {node.ns_green} s / EW {node.ew_green} s"),
            kv("Cycle opens with", h(node.first_phase)),
            kv("Queue NS / EW", f"{node.queue_ns:.1f} / {node.queue_ew:.1f}"),
            kv("Congestion", h(node.congestion)),
            kv("Queue pressure ΔP", f"{node.pressure_ns - node.pressure_ew:+.1f}" if node.pressure_ns is not None else "n/a")]
    if node.overridden:
        rows.append(kv("Base controller chose", h(node.base_plan)))
    if ev.phase != "off":
        rows.append(kv("Emergency", '<span style="color:var(--emergency)">ACTIVE</span>' if node.priority else "inactive"))
    st.markdown(f'<div class="qt-card{" emergency" if node.priority else ""}"><div class="qt-label">Intersection inspector</div>'
                f'<div class="qt-kpi" style="font-size:1.35rem">{h(node.node)}</div><dl class="qt-kv">{"".join(rows)}</dl>'
                f'<div class="qt-sub">ΔP = pressure(NS) − pressure(EW) from the Phase 2 definition, at the start of the cycle.</div></div>', unsafe_allow_html=True)


def network_legend() -> None:
    st.markdown('<div class="qt-legend"><span><i style="background:var(--accent)"></i>normal queue</span><span><i style="background:var(--warning)"></i>high (≥35% of road)</span>'
                '<span><i style="background:var(--emergency)"></i>severe (≥70%)</span><span>◆ emergency vehicle</span><span>○ red ring = emergency priority</span>'
                '<span>road width = queue / capacity</span></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------------- emergency
def emergency_panel(run: DashboardRun, ev: EmergencyView, saved: SavedResults | None) -> None:
    if ev.phase == "off":
        note("<b>No emergency in this run.</b> Enable it in the Control Center to simulate EV1 crossing the network with the green corridor.")
        rows = [r for r in (saved.emergency_summary if saved else []) if r["scenario"] == run.config.scenario and r["base_controller"] == run.config.controller]
        if rows:
            r = rows[0]
            st.markdown(f'<div class="qt-card"><div class="qt-label">Saved experiment · {r["seeds"]} seeds · {h(run.spec.label)} base</div>'
                        f'<dl class="qt-kv"><dt>Emergency travel time</dt><dd>{r["emergency_travel_time_s_A"]:.1f} s → {r["emergency_travel_time_s_B"]:.1f} s (corridor)</dd>'
                        f'<dt>Emergency delay</dt><dd>{r["emergency_delay_s_A"]:.1f} s → {r["emergency_delay_s_B"]:.1f} s</dd>'
                        f'<dt>Normal-traffic waiting</dt><dd>{r["waiting_pct_B_vs_A_of_means"]:+.2f}%</dd></dl></div>', unsafe_allow_html=True)
        return
    cls = "emergency" if ev.phase in ("active",) else ""
    steps = []
    for node in ev.route:
        state = "done" if node in ev.completed else "cur" if node == ev.current else "next" if node in ev.upcoming else ""
        steps.append(f'<span class="qt-step {state}">{h(node)}</span>')
    route_html = '<span class="qt-arrow">→</span>'.join(steps)
    if ev.phase == "active":
        title = '<span class="qt-status emergency"><i></i>EMERGENCY GREEN CORRIDOR</span>'
        line = f'{h(ev.vehicle_id)} at <b>{h(ev.current)}</b> · completed: {h(", ".join(ev.completed) or "none")} · next: {h(ev.upcoming[0] if ev.upcoming else "exit")}'
    elif ev.phase == "pending":
        title = '<span class="qt-status running"><i></i>EMERGENCY SCHEDULED</span>'
        line = f'{h(ev.vehicle_id)} enters the network at cycle {run.emergency.vehicle.entry_cycle}'
    else:
        title = '<span class="qt-status"><i></i>CORRIDOR COMPLETE</span>'
        line = ("<b>Normal controller restored</b> · override inactive, plans equal the base controller's" if ev.restored
                else "override releases the last intersection; the normal controller resumes next cycle")
    off = ev.travel_time_s_no_override
    tiles = [("Travel time", f"{ev.travel_time_s:.1f} s" if ev.travel_time_s is not None else "in progress", f"no override {off:.1f} s" if off is not None else ""),
             ("Delay", f"{ev.delay_s:.1f} s" if ev.delay_s is not None else "—", f"free-flow {ev.free_flow_time_s:.0f} s"),
             ("Signal stops", f"{ev.stops}", f"no override {ev.stops_no_override}" if ev.stops_no_override is not None else ""),
             ("Completion cycle", f"{ev.completion_cycle}" if ev.completion_cycle is not None else "—", ""),
             ("Priority intersections", f"{ev.prioritized_intersections}", "granted hop by hop")]
    cells = "".join(f'<div><div class="qt-label">{a}</div><div style="font-size:1.05rem;font-weight:600">{h(b)}</div><div class="qt-sub">{h(c)}</div></div>' for a, b, c in tiles)
    st.markdown(f'<div class="qt-card {cls}">{title}<div class="qt-route">{route_html}</div><div class="qt-sub" style="margin-bottom:0.6rem">{line}</div>'
                f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:0.7rem 1rem">{cells}</div></div>', unsafe_allow_html=True)
    if run.unified_off is not None:
        w_on, w_off = run.unified.traffic.waiting_vehicle_seconds, run.unified_off.traffic.waiting_vehicle_seconds
        note(f"<b>Cost to normal traffic (measured):</b> total waiting {100 * (w_on - w_off) / w_off:+.2f}% and maximum queue "
             f"{run.unified_off.traffic.max_queue_vehicles:.1f} → {run.unified.traffic.max_queue_vehicles:.1f} vehicles versus the same run without the override.", warn=w_on > w_off)


# ------------------------------------------------------------------------------------------------- optimization
def optimization_panel(run: DashboardRun, saved: SavedResults | None) -> None:
    n = len(run.net.nodes)
    variables, valid, full = VariableMap(run.net.nodes).n_variables, 3 ** n, 2 ** (3 * n)
    facts = [("QUBO variables", f"{variables}", "one-hot: 3 plans x 6 intersections"), ("Valid configurations", f"{valid:,}", "3^6 feasible assignments"),
             ("Full binary states", f"{full:,}", "2^18 bit strings")]
    st.markdown('<div class="qt-kpis">' + "".join(
        f'<div class="qt-card"><div class="qt-label">{a}</div><div class="qt-kpi" style="font-size:1.3rem">{b}</div><div class="qt-sub">{c}</div></div>'
        for a, b, c in facts) + "</div>", unsafe_allow_html=True)
    p = st.segmented_control("QAOA depth", ["p = 1", "p = 2"], default="p = 1", key="qaoa_p", label_visibility="collapsed") or "p = 1"
    variant = "qaoa_p1" if p == "p = 1" else "qaoa_p2"
    row = saved.qaoa_row(run.config.scenario, variant) if saved else None
    if row is None:
        note("No saved Phase 4 QAOA result is available for this scenario.", warn=True)
    else:
        m = [("Feasible sample rate", f"{100 * row['feasible_rate_sampled']:.2f}%", f"uniform sampling {100 * row['uniform_feasible_probability']:.2f}%"),
             ("Best sampled energy", f"{row['qaoa_best_feasible_energy']:,.1f}", "best feasible sample"),
             ("Exact reference", f"{row['exact_energy']:,.1f}", f"ground truth · {row['exact_n_optimal_assignments']} optimal assignment(s)"),
             ("Energy gap", f"{row['energy_gap']:,.1f}", f"approximation ratio {row['approximation_ratio']:.4f}"),
             ("Optimum probability", f"{row['optimum_probability_statevector']:.5f}", f"sampled {row['optimum_probability_sampled']:.5f} · uniform {row['uniform_optimum_probability']:.1e}"),
             ("Circuit", f"depth {row['logical_depth']}", f"{row['logical_gates']} gates · {row['basis_cx']} CX")]
        cells = "".join(f'<div><div class="qt-label">{h(a)}</div><div style="font-size:1.1rem;font-weight:600;font-variant-numeric:tabular-nums">{h(b)}</div><div class="qt-sub">{h(c)}</div></div>' for a, b, c in m)
        st.markdown(f'<div class="qt-card"><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:0.8rem 1.1rem">{cells}</div></div>', unsafe_allow_html=True)
        plot(charts.feasibility_figure(row), f"feas_{variant}")
        st.caption(f"Saved Phase 4 experiment, seed 0 ({'primary' if variant == 'qaoa_p1' else 'primary p=2, same deterministic start as p=1'}). "
                   f"Optimizer {'converged' if row['optimizer_converged'] else 'stopped at its iteration limit'} after {row['function_evaluations']} evaluations.")
    note("<b>QAOA experiments use Qiskit Aer classical simulation; no quantum hardware speedup is claimed.</b> Exact enumeration remains the ground truth.")
    if run.qubo_energy is not None:
        st.caption(f"Live QUBO Static plan: exact optimum, QUBO energy {run.qubo_energy:,.1f} (a surrogate objective, not a waiting time).")


# ------------------------------------------------------------------------------------------------- comparison
def comparison_section(run: DashboardRun, saved: SavedResults | None) -> None:
    rows = charts.comparison_rows(saved, run.config.scenario) if saved and saved.has_phase6 else []
    if not rows:
        note("Saved Phase 6 controller results are not available for this scenario.", warn=True)
        return
    waiting, thr, queues = charts.comparison_figures(rows, run.config.controller)
    for col, fig, key in zip(st.columns(3), (waiting, thr, queues), ("cmp_wait", "cmp_thr", "cmp_q")):
        with col:
            plot(fig, key)
    st.caption("Saved Phase 6 experiment: means over seeds 0–4 (QAOA p=1: seed 0 only, hatched) on identical demand; percentages are relative to Fixed. "
               "Measured values only, with no composite score or ranking.")


# ------------------------------------------------------------------------------------------------- environment
def environment_panel(run: DashboardRun, saved: SavedResults | None) -> None:
    k = kpis(run)
    tiles = [("Fuel proxy", f"{k['fuel_liters_proxy']:.1f}", " L", "waiting/idling only"), ("CO₂ proxy", f"{k['co2_kg_proxy']:.1f}", " kg", "simulation estimate"),
             ("Waiting", f"{k['waiting_vehicle_seconds']:,.0f}", " veh·s", "measured in the simulator")]
    st.markdown('<div class="qt-kpis">' + "".join(
        f'<div class="qt-card"><div class="qt-label">{a}</div><div class="qt-kpi" style="font-size:1.3rem">{b}<small>{u}</small></div><div class="qt-sub">{c}</div></div>'
        for a, b, u, c in tiles) + "</div>", unsafe_allow_html=True)
    note("<b>Environmental values are simulation proxies based on waiting time, not real-world measurements.</b> They are linear in waiting vehicle-seconds "
         "and ignore acceleration, speed, vehicle type and NOx/PM.", warn=True)
    rows = charts.sensitivity_rows(saved, run.config.scenario, run.config.controller) if saved else []
    if rows and rows[0]["co2_fixed"] is not None:
        plot(charts.sensitivity_figure(rows, run.spec.label), "sens")
        st.caption("Saved Phase 6 sensitivity (mean over seeds): absolute proxy values change with the idle-rate / fuel-type assumption, the relative comparison does not.")


# ------------------------------------------------------------------------------------------------- AI analyst
def ai_panel(run: DashboardRun, status: tuple[str, str, str]) -> str | None:
    """Renders the analyst; returns the clicked action key (or None). ``status`` = (badge, css class, detail)."""
    badge, cls, detail = status
    st.markdown(f'<div class="qt-ai"><span class="qt-label">AI traffic analyst</span> &nbsp; <span class="qt-badge {cls}">{h(badge)}</span>'
                f'<div class="qt-sub" style="margin-top:0.35rem">{h(detail)}. Explanations describe measured results; they never choose signals or change the simulation.</div></div>',
                unsafe_allow_html=True)
    clicked = None
    for col, action in zip(st.columns(4), AI_ACTIONS):
        disabled = action == "emergency" and not run.config.emergency_enabled
        if col.button(AI_ACTION_LABELS[action], key=f"ai_{action}", width="stretch", disabled=disabled,
                      help="Enable the emergency and run the simulation first" if disabled else None):
            clicked = action
    return clicked


def ai_result(results: list) -> None:
    for title, result in results:
        text, cls, detail = badge_for(result)
        st.markdown(f'<div class="qt-ai"><span class="qt-label">{h(title)}</span> &nbsp; <span class="qt-badge {cls}">{text}</span>'
                    f'<p>{h(result.explanation)}</p><div class="qt-sub">{"LOCAL ANALYSIS · " if cls == "local" else ""}{h(detail)}</div></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------------- architecture / provenance
def architecture_panel() -> None:
    boxes = [("Traffic demand", "seeded, deterministic scenarios"), ("Traffic network", "6 intersections, directed roads, queues"),
             ("Simulator", "60 s cycles, fluid queues, the source of truth"), ("Classical · QUBO · QAOA", "Fixed / Adaptive / exact QUBO / QAOA (Aer)"),
             ("Emergency override", "priority layer between controller and signals"), ("Metrics", "traffic · signal · emergency · environmental proxy"),
             ("Featherless AI", "explains results; never controls"), ("Dashboard", "presentation layer over the validated system")]
    html = '<div class="qt-arch">' + '<div class="link">▼</div>'.join(f'<div class="box"><b>{h(a)}</b><span>{h(b)}</span></div>' for a, b in boxes) + "</div>"
    st.markdown(html, unsafe_allow_html=True)
    st.caption("Schematic of the software architecture, not a data visualization.")


def experiment_info(run: DashboardRun, playhead_cycle: int | None = None) -> None:
    cfg, v = run.config, run.verification
    rows = [("Scenario", h(cfg.scenario)), ("Seed", str(cfg.seed)), ("Cycles", f"{run.first_cycle}–{run.first_cycle + run.n_cycles - 1}  ({run.n_cycles})"),
            ("Controller", h(run.spec.label)), ("Emergency", h(" > ".join(cfg.route) + f" @ cycle {cfg.resolved_start()}") if cfg.emergency_enabled else "off"),
            ("Source", h(run.source)), ("Generated", h(run.generated_at))]
    check = ""
    if v.ok is not None:
        color = "var(--positive)" if v.ok else "var(--warning)"
        check = f'<div class="qt-sub" style="color:{color}">{"✓" if v.ok else "!"} {h(v.text)}</div>'
    elif v.text:
        check = f'<div class="qt-sub">{h(v.text)}</div>'
    st.markdown('<div class="qt-card"><div class="qt-label">Experiment info</div><dl class="qt-kv">'
                + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in rows) + f"</dl>{check}"
                '<div class="qt-sub">Comparison, QAOA and sensitivity panels use the validated Phase 4–6 saved experiments.</div></div>', unsafe_allow_html=True)
