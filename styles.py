"""
styles.py — custom CSS injection + small reusable rendering helpers.

Base theme colors live in .streamlit/config.toml (Streamlit's official,
stable theming mechanism). This file layers finer custom-component
styling and animation on top, targeting Streamlit's more stable CSS
hooks plus this app's own custom markup. If a Streamlit upgrade ever
changes how something looks, this file — not config.toml — is the
first place to check.
"""

import streamlit as st
import plotly.graph_objects as go

RISK_COLORS = {"low": "#3FD6C0", "medium": "#F5A855", "high": "#F0615A"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@600;700&display=swap');

:root {
    --bg-base: #0E1620;
    --bg-surface: #16212C;
    --bg-raised: #1C2B39;
    --border-subtle: #263847;
    --text-primary: #EDF2F5;
    --text-secondary: #8FA1AF;
    --text-tertiary: #5C7080;
    --accent: #3FD6C0;
    --accent-dim: #2A9E8F;
    --risk-low: #3FD6C0;
    --risk-medium: #F5A855;
    --risk-high: #F0615A;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-base);
    color: var(--text-primary);
    font-family: 'Inter', -apple-system, sans-serif;
}

h1, h2, h3, .app-title { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.01em; }

[data-testid="stSidebar"] {
    background-color: var(--bg-surface);
    border-right: 1px solid var(--border-subtle);
}

[data-testid="stSidebar"] hr { border-color: var(--border-subtle); }

/* ---- status strip ---- */
.status-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 4px 0 22px 0; }
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--bg-surface); border: 1px solid var(--border-subtle);
    border-radius: 6px; padding: 5px 11px; font-size: 0.82rem; color: var(--text-secondary);
}
.status-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); flex-shrink: 0; }
.status-dot.off { background: var(--text-tertiary); }
.status-dot.warn { background: var(--risk-medium); }

/* ---- input sections: left accent border, not a floating card ---- */
.panel-label {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text-secondary);
    font-size: 0.8rem;
    margin: 2px 0 10px 0;
}

/* ---- buttons: subtle, functional transitions ---- */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
    border: 1px solid var(--border-subtle);
}
.stButton > button:hover { transform: translateY(-1px); filter: brightness(1.08); }
.stButton > button:active { transform: translateY(0); }
.stButton > button[kind="primary"] {
    background: var(--accent); color: #06201B; border: none;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 16px rgba(63, 214, 192, 0.28);
}

/* ---- the one orchestrated reveal moment ---- */
@keyframes riseIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-card {
    background: var(--bg-raised);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 26px 28px 10px 28px;
    animation: riseIn 0.45s ease-out;
    margin-top: 6px;
}

@keyframes pulseOnce {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.035); }
}
.risk-badge {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 7px 16px; border-radius: 999px; font-weight: 600; font-size: 0.9rem;
    margin: 10px 0 4px 0;
}
.risk-badge.low    { background: rgba(63,214,192,0.12);  color: var(--risk-low);    border: 1px solid rgba(63,214,192,0.35); }
.risk-badge.medium { background: rgba(245,168,85,0.12);  color: var(--risk-medium); border: 1px solid rgba(245,168,85,0.35); }
.risk-badge.high   { background: rgba(240,97,90,0.12);   color: var(--risk-high);   border: 1px solid rgba(240,97,90,0.35);
                      animation: pulseOnce 1.4s ease-in-out 2; }

.hero-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
    color: var(--text-primary);
}

.factor-list { margin: 10px 0 4px 0; padding: 0; list-style: none; }
.factor-list li {
    padding: 7px 0 7px 18px;
    border-bottom: 1px solid var(--border-subtle);
    color: var(--text-secondary);
    font-size: 0.88rem;
    position: relative;
}
.factor-list li:last-child { border-bottom: none; }
.factor-list li::before {
    content: '';
    position: absolute; left: 0; top: 14px;
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent);
}

.rec-list { margin: 10px 0 4px 0; padding: 0; list-style: none; }
.rec-list li {
    padding: 6px 0 6px 18px;
    color: var(--text-secondary);
    font-size: 0.88rem;
    position: relative;
}
.rec-list li::before {
    content: '→';
    position: absolute; left: 0; top: 5px;
    color: var(--accent-dim);
}

.empty-state {
    color: var(--text-tertiary);
    font-size: 0.95rem;
    padding: 44px 20px;
    text-align: center;
    border: 1px dashed var(--border-subtle);
    border-radius: 12px;
    margin-top: 6px;
}

.demo-note {
    color: var(--text-tertiary);
    font-size: 0.82rem;
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px solid var(--border-subtle);
}

/* ---- model details footer card ---- */
.model-detail-card {
    background: var(--bg-raised);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 4px 22px;
    margin: 10px 0 4px 0;
}
.model-detail-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 20px;
    padding: 11px 0;
    border-bottom: 1px solid var(--border-subtle);
    font-size: 0.88rem;
}
.model-detail-row:last-child { border-bottom: none; }
.model-detail-label { color: var(--text-secondary); flex-shrink: 0; }
.model-detail-value {
    color: var(--text-primary);
    font-weight: 600;
    text-align: right;
    word-break: break-word;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}

/* Streamlit tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: var(--bg-surface);
    border-radius: 8px 8px 0 0;
    color: var(--text-secondary);
}
.stTabs [aria-selected="true"] { color: var(--accent) !important; }

/* keep the main column from stretching edge-to-edge on wide screens */
.block-container { max-width: 1120px; padding-top: 2.2rem; }

@media (max-width: 640px) {
    .hero-number { font-size: 2.3rem; }
    .block-container { padding-left: 1rem; padding-right: 1rem; }
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def status_pill(text: str, state: str = "on") -> str:
    """state: 'on' | 'off' | 'warn'"""
    dot_class = "status-dot" if state == "on" else f"status-dot {state}"
    return f'<span class="status-pill"><span class="{dot_class}"></span>{text}</span>'


def risk_badge_html(level: str) -> str:
    label = {"low": "Low risk", "medium": "Medium risk", "high": "High risk"}[level]
    return f'<span class="risk-badge {level}">{label}</span>'


def build_gauge_figure(pct: float, level: str) -> go.Figure:
    color = RISK_COLORS[level]
    fig = go.Figure(
        go.Indicator(
            mode="gauge",
            value=pct,
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 0, "showticklabels": False},
                "bar": {"color": color, "thickness": 0.26},
                "bgcolor": "rgba(255,255,255,0.03)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35], "color": "rgba(63,214,192,0.07)"},
                    {"range": [35, 65], "color": "rgba(245,168,85,0.07)"},
                    {"range": [65, 100], "color": "rgba(240,97,90,0.07)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=170,
        margin=dict(l=16, r=16, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#EDF2F5", "family": "Inter"},
    )
    return fig