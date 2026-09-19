"""The centrepiece: a schematic Plotly view of the actual 6-intersection network (not a map).

Everything drawn comes from a ``NetworkView`` / ``EmergencyView`` (i.e. from the simulator's CycleRecord and the
emergency override log): road colour = queue / capacity, node ring = emergency priority, signal lamps = the phase
that opens the cycle, marker = the emergency vehicle's simulated position.
"""

from __future__ import annotations

import math

import plotly.graph_objects as go

from .state import EmergencyView, NetworkView
from .theme import CONGESTION_COLORS, PALETTE, axis_style, plotly_layout, rgba

SX, SY = 1.9, 1.35  # data-unit spacing between intersections (columns, rows)
LANE = 0.085  # perpendicular offset between the two directions of one road
TRIM = 0.27  # gap left at the ends of a road so it stops at the intersection ring
HEADING_ANGLE = {"N": 0, "E": 90, "S": 180, "W": 270}  # plotly arrow angle, clockwise from up
GRID_COLS, GRID_ROWS = 3, 2
STUB = 0.62  # length of entry / exit stubs outside the grid, as a fraction of the grid spacing


def xy(col: float, row: float) -> tuple[float, float]:
    return col * SX, -row * SY


def _outside(col: float, row: float) -> bool:
    return not (0 <= col <= GRID_COLS - 1 and 0 <= row <= GRID_ROWS - 1)


def _lane(p0, p1):
    """Offset a segment to the right-hand lane of its direction of travel and trim both ends."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    ox, oy = uy * LANE, -ux * LANE
    a = (p0[0] + ux * TRIM + ox, p0[1] + uy * TRIM + oy)
    b = (p1[0] - ux * TRIM + ox, p1[1] - uy * TRIM + oy)
    return a, b


def _road_geometry(road):
    (uc, ur), (dc, dr) = road.upstream, road.downstream
    if _outside(uc, ur):  # entry stub: shortened so the frame stays compact
        uc, ur = dc + (uc - dc) * STUB, dr + (ur - dr) * STUB
    return _lane(xy(uc, ur), xy(dc, dr))


def build_network_figure(view: NetworkView, ev: EmergencyView, selected: str | None = None, height: int = 500) -> go.Figure:
    fig = go.Figure()
    # ---- exit stubs (dim: vehicles leave the network here)
    for node, heading, (c0, r0), (c1, r1) in view.exit_stubs:
        end = (c0 + (c1 - c0) * STUB, r0 + (r1 - r0) * STUB)
        a, b = _lane(xy(c0, r0), xy(*end))
        fig.add_trace(go.Scatter(x=[a[0], b[0]], y=[a[1], b[1]], mode="lines", line=dict(color=rgba(PALETTE["muted"], 0.35), width=3),
                                 hoverinfo="skip", showlegend=False))
    # ---- roads coloured by congestion
    arrow_x, arrow_y, arrow_angle, arrow_color = [], [], [], []
    for road in view.roads:
        a, b = _road_geometry(road)
        color = CONGESTION_COLORS[road.congestion]
        width = 3.5 + 7.0 * min(road.ratio, 1.0)
        alpha = 0.5 if road.congestion == "normal" else 0.9
        hover = (f"<b>{road.approach}</b><br>queue {road.queue:.1f} of {road.capacity:.0f} vehicles ({100 * road.ratio:.0f}%)<br>"
                 f"{road.congestion} congestion" + ("<br>boundary entry" if road.external_entry else ""))
        fig.add_trace(go.Scatter(x=[a[0], b[0]], y=[a[1], b[1]], mode="lines", line=dict(color=rgba(color, alpha), width=width),
                                 hovertemplate=hover + "<extra></extra>", showlegend=False))
        arrow_x.append(a[0] + 0.62 * (b[0] - a[0]))
        arrow_y.append(a[1] + 0.62 * (b[1] - a[1]))
        arrow_angle.append(HEADING_ANGLE[road.heading])
        arrow_color.append(rgba(PALETTE["text"], 0.5))
    fig.add_trace(go.Scatter(x=arrow_x, y=arrow_y, mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(symbol="arrow", size=8, angle=arrow_angle, angleref="up", color=arrow_color)))
    # ---- emergency route overlay and vehicle
    if ev.route and ev.phase in ("active", "pending", "complete"):
        done = set(ev.completed)
        for i in range(len(ev.route) - 1):
            p, q = xy(*_node_xy(view, ev.route[i])), xy(*_node_xy(view, ev.route[i + 1]))
            finished = ev.route[i] in done
            fig.add_trace(go.Scatter(x=[p[0], q[0]], y=[p[1], q[1]], mode="lines", hoverinfo="skip", showlegend=False,
                                     line=dict(color=rgba(PALETTE["positive"] if finished else PALETTE["emergency"], 0.55), width=2.5,
                                               dash="solid" if finished else "dash")))
    # ---- intersections
    prio = [n for n in view.nodes if n.priority]
    if prio:
        px, py = zip(*(xy(n.col, n.row) for n in prio))
        fig.add_trace(go.Scatter(x=px, y=py, mode="markers", hoverinfo="skip", showlegend=False,
                                 marker=dict(size=76, color="rgba(0,0,0,0)", line=dict(color=PALETTE["emergency"], width=3))))
    xs, ys, texts, custom, lines, widths, hovers = [], [], [], [], [], [], []
    for n in view.nodes:
        x, y = xy(n.col, n.row)
        xs.append(x), ys.append(y), texts.append(f"<b>{n.node}</b>"), custom.append([n.node])
        is_sel = n.node == selected
        lines.append(PALETTE["accent"] if is_sel else PALETTE["emergency"] if n.priority else PALETTE["border_strong"])
        widths.append(3 if is_sel else 2)
        hovers.append(f"<b>{n.node}</b><br>plan {n.plan} (base {n.base_plan})<br>opens with {n.first_phase}<br>queue NS {n.queue_ns:.1f} / EW {n.queue_ew:.1f}"
                      + ("<br><b>EMERGENCY PRIORITY</b>" if n.priority else ""))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=texts, textposition="middle center", customdata=custom, hovertemplate="%{hovertext}<extra></extra>",
        hovertext=hovers, textfont=dict(family="Consolas, monospace", size=13, color=PALETTE["text"]),
        marker=dict(size=56, color=PALETTE["surface_2"], line=dict(color=lines, width=widths)),
        selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1)), showlegend=False, name="intersections"))
    # ---- per-intersection signal lamps and queue
    annotations = []
    for n in view.nodes:
        x, y = xy(n.col, n.row)
        ns_on = n.first_phase == "NS"
        lamp = lambda on: (f"<span style='color:{PALETTE['positive']}'>●</span>" if on else f"<span style='color:{PALETTE['emergency']}'>○</span>")
        qcol = CONGESTION_COLORS[n.congestion] if n.congestion != "normal" else PALETTE["muted"]
        text = (f"NS {lamp(ns_on)} {n.ns_green}s<br>EW {lamp(not ns_on)} {n.ew_green}s<br>"
                f"<span style='color:{qcol}'>Q {n.queue_total:.0f}</span>" + (f" <span style='color:{PALETTE['emergency']}'>PRIORITY</span>" if n.priority else ""))
        annotations.append(dict(x=x + 0.31, y=y - 0.17, text=text, showarrow=False, xanchor="left", yanchor="top", align="left",
                                font=dict(size=10, color=PALETTE["text"], family="Consolas, monospace"), bgcolor=rgba(PALETTE["bg"], 0.55)))
    if ev.position is not None:
        ex, ey = xy(*ev.position)
        fig.add_trace(go.Scatter(x=[ex], y=[ey], mode="markers+text", text=[f"<b>{ev.vehicle_id}</b>"], textposition="top center", showlegend=False,
                                 hovertemplate=f"<b>{ev.vehicle_id}</b><br>at {ev.current}<extra></extra>",
                                 textfont=dict(color=PALETTE["emergency"], size=11, family="Consolas, monospace"),
                                 marker=dict(symbol="diamond", size=17, color=PALETTE["emergency"], line=dict(color="#FFFFFF", width=1.5))))
    fig.update_layout(**plotly_layout(
        height=height, margin=dict(l=4, r=4, t=4, b=4), annotations=annotations, clickmode="event+select", dragmode=False,
        xaxis=dict(visible=False, range=[-1.4, 2 * SX + 1.7], fixedrange=True), yaxis=dict(visible=False, range=[-SY - 1.0, 1.0], fixedrange=True)))
    return fig


def _node_xy(view: NetworkView, name: str) -> tuple[float, float]:
    n = view.node(name)
    return n.col, n.row
