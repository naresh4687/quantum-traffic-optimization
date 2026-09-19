"""Design system for the control-room dashboard: one palette, exposed as CSS variables.

Every colour used by the CSS, the Plotly figures and the HTML components comes from ``PALETTE`` (single source),
so nothing is scattered through the code. Direction: dark, restrained, technical.
"""

from __future__ import annotations

PALETTE = {
    "bg": "#080B10",  # near-black charcoal
    "surface": "#0E131A",
    "surface_2": "#121820",
    "border": "#1E2836",  # subtle blue-gray
    "border_strong": "#2D3B4E",
    "accent": "#2BD4F0",  # electric cyan
    "accent_dim": "#1B8FA6",
    "positive": "#3DD68C",  # controlled green
    "warning": "#F5A524",  # amber
    "emergency": "#FF4D4F",  # red
    "text": "#E8EEF5",
    "muted": "#7F8FA4",  # muted blue-gray
    "faint": "#4B596B",
}

FONT_STACK = 'Inter, "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif'
MONO_STACK = '"JetBrains Mono", "Cascadia Mono", Consolas, "SFMono-Regular", monospace'

# congestion classes used by the network view (ratio = queue / road capacity); thresholds are documented constants
CONGESTION_THRESHOLDS = {"high": 0.35, "severe": 0.70}
CONGESTION_COLORS = {"normal": PALETTE["accent"], "high": PALETTE["warning"], "severe": PALETTE["emergency"]}


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def css() -> str:
    """The full stylesheet. Colours are CSS variables defined once in :root."""
    variables = "".join(f"--{k.replace('_', '-')}:{v};" for k, v in PALETTE.items())
    return f"""
<style>
:root {{{variables}--font:{FONT_STACK};--mono:{MONO_STACK};}}
html, body, [class*="css"], .stApp {{font-family:var(--font);color:var(--text);}}
.stApp {{background:var(--bg);}}
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{visibility:hidden;height:0;}}
header[data-testid="stHeader"] {{background:transparent;height:2.2rem;}}
.block-container {{padding:0.9rem 1.6rem 2.2rem 1.6rem;max-width:1680px;}}
[data-testid="stSidebar"] {{background:var(--surface);border-right:1px solid var(--border);}}
[data-testid="stSidebar"] .block-container {{padding:1.1rem 1rem;}}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p {{font-size:0.78rem;}}
h1,h2,h3,h4 {{font-family:var(--font);letter-spacing:-0.01em;}}
hr {{border-color:var(--border);margin:0.8rem 0;}}
div[data-testid="stVerticalBlock"] {{gap:0.7rem;}}
/* widgets */
.stButton > button, .stDownloadButton > button {{border-radius:5px;border:1px solid var(--border-strong);background:var(--surface-2);
  color:var(--text);font-size:0.78rem;font-weight:600;letter-spacing:0.02em;padding:0.32rem 0.7rem;min-height:2.1rem;transition:none;}}
.stButton > button:hover {{border-color:var(--accent);color:var(--accent);background:var(--surface-2);}}
.stButton > button[kind="primary"] {{background:var(--accent);border-color:var(--accent);color:#04121A;}}
.stButton > button[kind="primary"]:hover {{background:var(--accent);color:#04121A;filter:brightness(1.08);}}
.stButton > button:disabled {{opacity:0.4;}}
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"], .stNumberInput input {{background:var(--surface-2)!important;
  border-color:var(--border-strong)!important;border-radius:5px!important;font-size:0.82rem;}}
.stRadio label, .stToggle label, .stCheckbox label {{font-size:0.8rem;}}
[data-testid="stExpander"] {{border:1px solid var(--border);border-radius:6px;background:var(--surface);}}
[data-testid="stExpander"] summary {{font-size:0.78rem;letter-spacing:0.06em;text-transform:uppercase;color:var(--muted);}}
[data-testid="stPlotlyChart"] {{border:1px solid var(--border);border-radius:6px;background:var(--surface);padding:2px;}}
/* header */
.qt-top {{display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:0.55rem 0.2rem 0.85rem 0.2rem;
  border-bottom:1px solid var(--border);margin-bottom:0.2rem;flex-wrap:wrap;}}
.qt-brand {{display:flex;align-items:baseline;gap:0.9rem;flex-wrap:wrap;}}
.qt-logo {{font-family:var(--mono);font-weight:700;letter-spacing:0.22em;color:var(--accent);font-size:1.15rem;}}
.qt-title {{font-size:0.86rem;color:var(--muted);letter-spacing:0.02em;}}
.qt-chips {{display:flex;gap:0.45rem;flex-wrap:wrap;align-items:center;}}
.qt-chip {{font-family:var(--mono);font-size:0.66rem;letter-spacing:0.08em;text-transform:uppercase;color:var(--muted);
  border:1px solid var(--border);border-radius:4px;padding:0.22rem 0.5rem;background:var(--surface);}}
.qt-chip b {{color:var(--text);font-weight:600;}}
.qt-status {{display:inline-flex;align-items:center;gap:0.45rem;font-family:var(--mono);font-size:0.7rem;font-weight:600;letter-spacing:0.12em;
  padding:0.28rem 0.7rem;border-radius:4px;border:1px solid var(--border-strong);color:var(--positive);background:var(--surface);}}
.qt-status i {{width:7px;height:7px;border-radius:50%;background:currentColor;display:inline-block;}}
.qt-status.running {{color:var(--accent);}}
.qt-status.emergency {{color:var(--emergency);border-color:var(--emergency);background:rgba(255,77,79,0.08);}}
.qt-status.emergency i {{animation:qt-pulse 1.4s ease-in-out infinite;}}
@keyframes qt-pulse {{0%,100%{{opacity:1;}}50%{{opacity:0.25;}}}}
@media (prefers-reduced-motion: reduce) {{.qt-status.emergency i {{animation:none;}} * {{scroll-behavior:auto!important;}}}}
.qt-emergency-bar {{height:3px;background:var(--emergency);border-radius:2px;margin:0 0 0.4rem 0;}}
/* cards */
.qt-kpis {{display:grid;grid-template-columns:repeat(auto-fit,minmax(172px,1fr));gap:0.7rem;}}
[data-testid="stCaptionContainer"] {{margin-top:0.25rem;}}
.qt-card {{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:0.75rem 0.95rem;height:100%;}}
.qt-card.emergency {{border-color:var(--emergency);background:linear-gradient(180deg,rgba(255,77,79,0.07),var(--surface));}}
.qt-label {{font-size:0.64rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--muted);font-weight:600;}}
.qt-kpi {{font-size:1.75rem;font-weight:600;letter-spacing:-0.02em;line-height:1.15;margin-top:0.25rem;font-variant-numeric:tabular-nums;}}
.qt-kpi small {{font-size:0.72rem;font-weight:500;color:var(--muted);letter-spacing:0;margin-left:0.3rem;}}
.qt-sub {{font-size:0.72rem;color:var(--muted);margin-top:0.2rem;line-height:1.35;}}
.qt-delta {{font-family:var(--mono);font-size:0.7rem;font-weight:600;}}
.qt-delta.good {{color:var(--positive);}} .qt-delta.bad {{color:var(--warning);}} .qt-delta.flat {{color:var(--muted);}}
.qt-section {{display:flex;align-items:center;gap:0.55rem;margin:0.5rem 0 0.1rem 0;}}
.qt-section:before {{content:"";width:3px;height:0.95rem;background:var(--accent);border-radius:1px;}}
.qt-section span {{font-size:0.74rem;letter-spacing:0.14em;text-transform:uppercase;font-weight:600;color:var(--text);}}
.qt-section em {{font-style:normal;font-size:0.68rem;color:var(--muted);letter-spacing:0.04em;margin-left:auto;}}
.qt-note {{border:1px solid var(--border-strong);border-left:3px solid var(--accent);border-radius:4px;padding:0.5rem 0.75rem;font-size:0.74rem;
  color:var(--muted);background:var(--surface);line-height:1.45;}}
.qt-note.warn {{border-left-color:var(--warning);}}
.qt-note b {{color:var(--text);}}
/* signal rows */
.qt-sig {{display:grid;grid-template-columns:2.2rem 1fr;gap:0.1rem 0.5rem;align-items:center;font-size:0.76rem;padding:0.32rem 0;border-bottom:1px solid var(--border);}}
.qt-sig:last-child {{border-bottom:none;}}
.qt-sig .n {{font-family:var(--mono);font-weight:700;color:var(--text);}}
.qt-sig.sel .n {{color:var(--accent);}}
.qt-lamp {{display:inline-flex;align-items:center;gap:0.3rem;margin-right:0.9rem;font-family:var(--mono);font-size:0.68rem;letter-spacing:0.05em;color:var(--muted);}}
.qt-lamp i {{width:9px;height:9px;border-radius:50%;display:inline-block;background:var(--faint);}}
.qt-lamp.g i {{background:var(--positive);box-shadow:0 0 0 2px rgba(61,214,140,0.18);}} .qt-lamp.g {{color:var(--text);}}
.qt-lamp.r i {{background:var(--emergency);opacity:0.75;}}
.qt-splitbar {{display:flex;height:4px;border-radius:2px;overflow:hidden;background:var(--faint);margin-top:0.25rem;grid-column:2;}}
.qt-splitbar b {{display:block;height:100%;}} .qt-splitbar .ns {{background:var(--accent-dim);}} .qt-splitbar .ew {{background:var(--border-strong);}}
.qt-tag {{font-family:var(--mono);font-size:0.6rem;letter-spacing:0.08em;padding:0.08rem 0.35rem;border-radius:3px;border:1px solid var(--border-strong);color:var(--muted);text-transform:uppercase;}}
.qt-tag.em {{color:var(--emergency);border-color:var(--emergency);}}
/* route steps */
.qt-route {{display:flex;align-items:center;gap:0.35rem;flex-wrap:wrap;margin:0.4rem 0;}}
.qt-step {{font-family:var(--mono);font-size:0.78rem;font-weight:700;padding:0.25rem 0.6rem;border-radius:4px;border:1px solid var(--border-strong);color:var(--faint);}}
.qt-step.done {{color:var(--positive);border-color:rgba(61,214,140,0.45);}}
.qt-step.cur {{color:#fff;background:var(--emergency);border-color:var(--emergency);}}
.qt-step.next {{color:var(--text);border-color:var(--muted);}}
.qt-arrow {{color:var(--faint);font-size:0.8rem;}}
/* AI */
.qt-ai {{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:0.85rem 1rem;}}
.qt-badge {{font-family:var(--mono);font-size:0.62rem;font-weight:700;letter-spacing:0.12em;padding:0.18rem 0.5rem;border-radius:3px;border:1px solid;}}
.qt-badge.ok {{color:var(--accent);border-color:var(--accent);}} .qt-badge.local {{color:var(--muted);border-color:var(--border-strong);}}
.qt-ai p {{font-size:0.86rem;line-height:1.6;color:var(--text);margin:0.55rem 0 0.2rem 0;}}
/* architecture */
.qt-arch {{display:flex;flex-direction:column;align-items:center;gap:0;}}
.qt-arch .box {{width:min(420px,100%);text-align:center;border:1px solid var(--border-strong);border-radius:5px;background:var(--surface-2);padding:0.4rem 0.6rem;font-size:0.78rem;}}
.qt-arch .box b {{display:block;font-family:var(--mono);font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;color:var(--accent);}}
.qt-arch .box span {{color:var(--muted);font-size:0.68rem;}}
.qt-arch .link {{color:var(--faint);line-height:1;font-size:0.9rem;padding:0.1rem 0;}}
.qt-legend {{display:flex;gap:1rem;flex-wrap:wrap;font-size:0.68rem;color:var(--muted);padding:0.2rem 0.1rem;}}
.qt-legend i {{display:inline-block;width:18px;height:4px;border-radius:2px;margin-right:0.35rem;vertical-align:middle;}}
.qt-kv {{display:grid;grid-template-columns:auto 1fr;gap:0.25rem 0.9rem;font-size:0.76rem;}}
.qt-kv dt {{color:var(--muted);font-size:0.74rem;}} .qt-kv dd {{margin:0;text-align:right;font-family:var(--mono);color:var(--text);font-size:0.74rem;}}
@media (max-width: 1100px) {{.qt-kpi {{font-size:1.4rem;}} .block-container {{padding:0.6rem 0.8rem;}}}}
</style>
"""


def plotly_layout(**overrides) -> dict:
    """Shared dark Plotly layout (transparent paper so the CSS card shows through)."""
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, color=PALETTE["text"], size=11),
        margin=dict(l=8, r=8, t=8, b=8), showlegend=False, hoverlabel=dict(bgcolor=PALETTE["surface_2"], bordercolor=PALETTE["border_strong"],
                                                                          font=dict(family=FONT_STACK, size=11, color=PALETTE["text"])),
    )
    layout.update(overrides)
    return layout


def axis_style(**overrides) -> dict:
    style = dict(gridcolor=PALETTE["border"], zeroline=False, linecolor=PALETTE["border"], tickfont=dict(color=PALETTE["muted"], size=10),
                 title_font=dict(color=PALETTE["muted"], size=10))
    style.update(overrides)
    return style
