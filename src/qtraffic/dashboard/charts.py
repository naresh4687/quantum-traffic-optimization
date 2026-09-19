"""Plotly figures for the dashboard (timeline, signal-plan heatmap, controller comparison, proxy sensitivity, QAOA feasibility).

All data comes from the live run or from the saved Phase 4-6 results; nothing is filled in.
"""

from __future__ import annotations

import plotly.graph_objects as go

from .saved import SavedResults
from .state import CONTROLLERS, DashboardRun
from .theme import PALETTE, axis_style, plotly_layout, rgba

COMPARE_ORDER = ("fixed", "adaptive", "qubo_static", "qubo_receding", "qaoa_p1")


def queue_timeline(run: DashboardRun, idx: int, height: int = 200) -> go.Figure:
    """Total queued vehicles per cycle, the emergency window, the no-override run (dotted) and the playhead."""
    cycles = [r.cycle for r in run.history]
    total = [sum(r.queue_after_arrivals.values()) for r in run.history]
    fig = go.Figure()
    if run.emergency is not None and run.emergency.vehicle.entry_cycle is not None:
        v = run.emergency.vehicle
        end = v.completion_cycle if v.completion_cycle is not None else cycles[-1]
        fig.add_vrect(x0=v.entry_cycle - 0.5, x1=end + 0.5, fillcolor=rgba(PALETTE["emergency"], 0.12), line_width=0, layer="below")
    if run.emergency_off is not None:
        fig.add_trace(go.Scatter(x=cycles, y=[sum(r.queue_after_arrivals.values()) for r in run.emergency_off.history], mode="lines",
                                 line=dict(color=rgba(PALETTE["muted"], 0.8), width=1.5, dash="dot"), name="no override",
                                 hovertemplate="cycle %{x}<br>no override: %{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=cycles, y=total, mode="lines", line=dict(color=PALETTE["accent"], width=2), fill="tozeroy",
                             fillcolor=rgba(PALETTE["accent"], 0.08), name="displayed run", hovertemplate="cycle %{x}<br>queued: %{y:.0f}<extra></extra>"))
    fig.add_vline(x=run.cycle_at(idx), line=dict(color=PALETTE["text"], width=1))
    fig.update_layout(**plotly_layout(height=height, margin=dict(l=44, r=10, t=8, b=30), xaxis=axis_style(title="cycle"),
                                      yaxis=axis_style(title="vehicles queued"), hovermode="x unified"))
    return fig


def signal_heatmap(run: DashboardRun, idx: int, height: int = 210) -> go.Figure:
    """North-south green seconds per intersection per cycle (20 / 30 / 40 s); red markers = emergency override changed the plan."""
    nodes = list(run.net.nodes)
    cycles = [r.cycle for r in run.history]
    z = [[r.plans[n].ns_green for r in run.history] for n in nodes]
    scale = [[0.0, PALETTE["surface_2"]], [0.333, PALETTE["surface_2"]], [0.333, PALETTE["border_strong"]], [0.667, PALETTE["border_strong"]],
             [0.667, PALETTE["accent_dim"]], [1.0, PALETTE["accent_dim"]]]
    fig = go.Figure(go.Heatmap(z=z, x=cycles, y=nodes, zmin=20, zmax=40, colorscale=scale, xgap=1, ygap=2, showscale=False,
                               hovertemplate="%{y} cycle %{x}<br>NS green %{z} s<extra></extra>"))
    if run.emergency is not None:
        ox, oy = [], []
        for rec, entry in zip(run.history, run.emergency.log):
            for n in entry.changed_nodes:
                ox.append(rec.cycle), oy.append(n)
        if ox:
            fig.add_trace(go.Scatter(x=ox, y=oy, mode="markers", marker=dict(symbol="square", size=7, color=PALETTE["emergency"]),
                                     hovertemplate="%{y} cycle %{x}<br>emergency override<extra></extra>"))
    fig.add_vline(x=run.cycle_at(idx), line=dict(color=PALETTE["text"], width=1))
    fig.update_layout(**plotly_layout(height=height, margin=dict(l=32, r=10, t=8, b=30), xaxis=axis_style(title="cycle"),
                                      yaxis=axis_style(autorange="reversed", showgrid=False)))
    return fig


def comparison_rows(saved: SavedResults, scenario: str) -> list[dict]:
    """One row per controller from the saved Phase 6 summary (means over seeds; QAOA is the single saved seed)."""
    rows = []
    for key in COMPARE_ORDER:
        r = saved.summary_row(scenario, key)
        if r is not None:
            rows.append({"key": key, "label": CONTROLLERS[key].label + (f" (seed {int(str(r['seed_list']).split()[0])})" if key == "qaoa_p1" else ""),
                         "seeds": r["seeds"], "waiting": r["waiting_seconds_per_admitted_mean"], "throughput": r["throughput_vehicles_per_hour_mean"],
                         "avg_queue": r["average_queue_vehicles_mean"], "max_queue": r["max_queue_vehicles_mean"],
                         "waiting_pct": r["waiting_seconds_per_admitted_pct_vs_fixed_of_means"], "throughput_pct": r["throughput_vehicles_per_hour_pct_vs_fixed_of_means"]})
    return rows


def _tick(label: str) -> str:
    """Two-line tick label so five controller names stay horizontal and readable in a narrow chart."""
    if " (seed" in label:
        return label.replace(" (seed", "<br>(seed")
    return label.replace(" ", "<br>", 1) if label.startswith(("QUBO", "QAOA")) else label


def _bar(rows, key, selected, title, fmt, pct_key=None, height=230):
    labels = [_tick(r["label"]) for r in rows]
    colors = [PALETTE["accent"] if r["key"] == selected else PALETTE["border_strong"] for r in rows]
    text = []
    for r in rows:
        t = fmt.format(r[key])
        if pct_key and r["key"] != "fixed" and r.get(pct_key) is not None:
            t += f"<br>{r[pct_key]:+.1f}%"
        text.append(t)
    fig = go.Figure(go.Bar(x=labels, y=[r[key] for r in rows], marker=dict(color=colors, line=dict(color=PALETTE["border_strong"], width=1),
                                                                         pattern=dict(shape=["/" if r["key"] == "qaoa_p1" else "" for r in rows], fgcolor=PALETTE["bg"])),
                           text=text, textposition="outside", textfont=dict(size=10, color=PALETTE["text"]), cliponaxis=False,
                           hovertemplate="%{x}<br>%{y:.2f}<extra></extra>"))
    top = max(r[key] for r in rows) * 1.32
    fig.update_layout(**plotly_layout(height=height, margin=dict(l=8, r=8, t=44, b=8), title=dict(text=title, x=0.01, font=dict(size=11, color=PALETTE["muted"])),
                                      xaxis=axis_style(showgrid=False, tickangle=0, tickfont=dict(size=9, color=PALETTE["muted"])),
                                      yaxis=axis_style(range=[0, top], showticklabels=False)))
    return fig


def comparison_figures(rows: list[dict], selected: str) -> tuple[go.Figure, go.Figure, go.Figure]:
    """Waiting per admitted vehicle, throughput, and queues (average and maximum)."""
    waiting = _bar(rows, "waiting", selected, "WAITING - s per admitted vehicle<br>lower is better", "{:.1f}", "waiting_pct")
    thr = _bar(rows, "throughput", selected, "THROUGHPUT - vehicles per hour<br>higher is better", "{:,.0f}", "throughput_pct")
    labels = [_tick(r["label"]) for r in rows]
    q = go.Figure()
    q.add_trace(go.Bar(name="average", x=labels, y=[r["avg_queue"] for r in rows], marker=dict(color=PALETTE["accent_dim"]), text=[f"{r['avg_queue']:.1f}" for r in rows],
                       textposition="outside", textfont=dict(size=10, color=PALETTE["text"]), cliponaxis=False))
    q.add_trace(go.Bar(name="maximum", x=labels, y=[r["max_queue"] for r in rows], marker=dict(color=PALETTE["warning"]), text=[f"{r['max_queue']:.0f}" for r in rows],
                       textposition="outside", textfont=dict(size=10, color=PALETTE["text"]), cliponaxis=False))
    key = f"<span style='color:{PALETTE['accent_dim']}'>&#9632; average</span>  <span style='color:{PALETTE['warning']}'>&#9632; maximum</span>"
    q.update_layout(**plotly_layout(height=230, margin=dict(l=8, r=8, t=44, b=8), barmode="group",
                                    title=dict(text=f"QUEUES - vehicles<br>{key}", x=0.01, font=dict(size=11, color=PALETTE["muted"])),
                                    xaxis=axis_style(showgrid=False, tickangle=0, tickfont=dict(size=9, color=PALETTE["muted"])),
                                    yaxis=axis_style(range=[0, max(r["max_queue"] for r in rows) * 1.3], showticklabels=False)))
    return waiting, thr, q


def sensitivity_rows(saved: SavedResults, scenario: str, controller_key: str) -> list[dict]:
    """CO2 proxy of the controller and of Fixed under every saved coefficient set (Phase 6 sensitivity)."""
    name = CONTROLLERS[controller_key].summary_name
    out = []
    for r in saved.sensitivity:
        if r.get("kind") == "controller_comparison" and r["scenario"] == scenario and r["subject"] == name:
            fixed = next((f for f in saved.sensitivity if f.get("kind") == "controller_comparison" and f["scenario"] == scenario
                          and f["subject"] == "fixed" and f["parameter_set"] == r["parameter_set"]), None)
            out.append({"set": r["parameter_set"], "co2": r["co2_kg_proxy_mean"], "co2_fixed": fixed["co2_kg_proxy_mean"] if fixed else None,
                        "pct": r["co2_proxy_pct_vs_fixed"] if "co2_proxy_pct_vs_fixed" in r else r.get("co2_proxy_pct_vs_fixed")})
    return out


def sensitivity_figure(rows: list[dict], controller_label: str, height: int = 220) -> go.Figure:
    labels = [r["set"].replace("idle", "").replace("Lph_", " L/h<br>").replace("gasoline", "gas.").replace("diesel", "dsl.") for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Fixed", x=labels, y=[r["co2_fixed"] for r in rows], marker=dict(color=PALETTE["border_strong"])))
    fig.add_trace(go.Bar(name=controller_label, x=labels, y=[r["co2"] for r in rows], marker=dict(color=PALETTE["accent_dim"])))
    fig.update_layout(**plotly_layout(height=height, margin=dict(l=44, r=8, t=26, b=8), barmode="group", showlegend=True,
                                      legend=dict(orientation="h", y=1.16, x=1, xanchor="right", font=dict(size=10, color=PALETTE["muted"])),
                                      title=dict(text="CO2 proxy (kg) by coefficient set", x=0.01, font=dict(size=11, color=PALETTE["muted"])),
                                      xaxis=axis_style(showgrid=False, tickangle=0, tickfont=dict(size=9, color=PALETTE["muted"])), yaxis=axis_style(title="kg (proxy)")))
    return fig


def feasibility_figure(row: dict, height: int = 150) -> go.Figure:
    """QAOA feasible-sample probability against uniform sampling (saved Phase 4 numbers, percent)."""
    vals = [("uniform sampling", 100 * row["uniform_feasible_probability"]), ("QAOA sampled", 100 * row["feasible_rate_sampled"]),
            ("QAOA statevector", 100 * row["feasible_probability_statevector"])]
    fig = go.Figure(go.Bar(y=[v[0] for v in vals], x=[v[1] for v in vals], orientation="h", text=[f"{v[1]:.2f}%" for v in vals], textposition="outside",
                           marker=dict(color=[PALETTE["border_strong"], PALETTE["accent"], PALETTE["accent_dim"]]), cliponaxis=False,
                           textfont=dict(size=10, color=PALETTE["text"])))
    fig.update_layout(**plotly_layout(height=height, margin=dict(l=8, r=40, t=8, b=8), xaxis=axis_style(showticklabels=False, range=[0, max(v[1] for v in vals) * 1.3]),
                                      yaxis=axis_style(showgrid=False, autorange="reversed", tickfont=dict(size=10, color=PALETTE["muted"]))))
    return fig
