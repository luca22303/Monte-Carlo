"""Shared state, caching and chart helpers for the Streamlit app."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcengine.config import SimConfig  # noqa: E402
from mcengine.metrics import METRIC_LABELS, fan, fmt, real_contributions, summarize  # noqa: E402
from mcengine.portfolio.simulator import SimResult, simulate  # noqa: E402
from mcengine.scenarios import ScenarioSet, generate  # noqa: E402

# ----------------------------------------------------------------------------- palette
# Reference data-viz palette (validated categorical order; light and dark steps).
CAT_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CAT_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
SEQ = {"100": "#cde2fb", "250": "#86b6ef", "450": "#2a78d6", "600": "#184f95"}
SEQ_DARK = {"100": "#184f95", "250": "#256abf", "450": "#5598e7", "600": "#b7d3f6"}
DIVERGING = [[0.0, "#2a78d6"], [0.5, "#f0efec"], [1.0, "#e34948"]]
DIVERGING_DARK = [[0.0, "#3987e5"], [0.5, "#383835"], [1.0, "#e66767"]]


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def cat() -> list[str]:
    return CAT_DARK if is_dark() else CAT_LIGHT


def seq() -> dict[str, str]:
    return SEQ_DARK if is_dark() else SEQ


def ink() -> tuple[str, str]:
    """(secondary text, muted reference-line) colors for the current theme."""
    return ("#c3c2b7", "#8a8985") if is_dark() else ("#52514e", "#8a8985")


def diverging():
    return DIVERGING_DARK if is_dark() else DIVERGING


# ----------------------------------------------------------------------------- state
def get_cfg() -> SimConfig:
    if "cfg_json" not in st.session_state:
        st.session_state["cfg_json"] = SimConfig().model_dump_json()
    return SimConfig.model_validate_json(st.session_state["cfg_json"])


def set_cfg(cfg: SimConfig) -> None:
    st.session_state["cfg_json"] = cfg.model_dump_json()


def scenario_key(cfg: SimConfig) -> str:
    keep = cfg.model_dump(include={"assets", "market", "generator", "bootstrap_block_months",
                                   "bootstrap_mean_adjust", "n_paths", "seed"})
    keep["months"] = cfg.months
    return json.dumps(keep, sort_keys=True)


@st.cache_resource(max_entries=4, show_spinner="Generating market scenarios…")
def _scenarios(key: str, _cfg_json: str) -> ScenarioSet:
    # Only ``key`` (the scenario-relevant settings) is hashed; leading underscore = not hashed.
    return generate(SimConfig.model_validate_json(_cfg_json))


def scenarios(cfg: SimConfig) -> ScenarioSet:
    return _scenarios(scenario_key(cfg), cfg.model_dump_json())


@st.cache_resource(max_entries=12, show_spinner="Simulating portfolio…")
def _simulate(cfg_json: str) -> tuple[SimResult, dict[str, float]]:
    cfg = SimConfig.model_validate_json(cfg_json)
    res = simulate(cfg, scenarios(cfg))
    return res, summarize(res, cfg.goal_real)


def run(cfg: SimConfig) -> tuple[SimResult, dict[str, float]]:
    return _simulate(cfg.model_dump_json())


# ----------------------------------------------------------------------------- formatting
def eur(x: float) -> str:
    return f"€{x:,.0f}"


def metrics_table(rows: dict[str, dict[str, float]], keys: list[str] | None = None) -> pd.DataFrame:
    """Formatted table: rows = metrics, columns = variants."""
    keys = keys or [k for k in METRIC_LABELS if any(k in r for r in rows.values())]
    data = {name: [fmt(k, r.get(k, float("nan"))) for k in keys] for name, r in rows.items()}
    return pd.DataFrame(data, index=[METRIC_LABELS[k][0] for k in keys])


def kpi_row(s: dict[str, float], goal: float) -> None:
    c = st.columns(5)
    c[0].metric("Median wealth (today's €)", eur(s["real_wealth_p50"]),
                help="After liquidation and all taxes, deflated to today's purchasing power.")
    c[1].metric("Bad case (P5)", eur(s["real_wealth_p5"]), help="95 % of simulated futures end higher.")
    c[2].metric(f"P(≥ {eur(goal)})", fmt("p_goal", s["p_goal"]) if goal > 0 else "–")
    c[3].metric("Real return p.a. (median)", fmt("irr_real_p50", s["irr_real_p50"]),
                help="Money-weighted (IRR) real return of your savings after costs and taxes.")
    c[4].metric("Max drawdown (P95)", fmt("mdd_p95", s["mdd_p95"]),
                help="In 5 % of futures the portfolio at some point falls at least this far from a peak.")


# ----------------------------------------------------------------------------- charts
def _layout(fig: go.Figure, title: str, y_title: str = "", x_title: str = "", height: int = 420) -> go.Figure:
    sec, _ = ink()
    fig.update_layout(
        title={"text": title, "font": {"size": 15}},
        height=height,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "left", "x": 0},
        hoverlabel={"namelength": -1},
    )
    fig.update_xaxes(title=x_title, showgrid=False, zeroline=False, color=sec)
    fig.update_yaxes(title=y_title, gridwidth=1, zeroline=False, color=sec)
    return fig


def fan_chart(res: SimResult, goal: float = 0.0, title: str = "Wealth in today's €") -> go.Figure:
    f = fan(res, real=True)
    years = (np.arange(res.months) + 1) / 12
    contrib = np.median(np.cumsum(real_contributions(res), axis=1), axis=0)
    s = seq()
    sec, muted = ink()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=f[95], line={"width": 0}, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=years, y=f[5], fill="tonexty", fillcolor=s["100"], line={"width": 0},
                             name="5–95 % of futures", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=years, y=f[75], line={"width": 0}, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=years, y=f[25], fill="tonexty", fillcolor=s["250"], line={"width": 0},
                             name="25–75 %", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=years, y=f[50], line={"color": s["600"], "width": 2}, name="Median",
                             customdata=np.stack([f[5], f[25], f[75], f[95]], axis=1),
                             hovertemplate="Year %{x:.1f}<br>Median €%{y:,.0f}<br>P5 €%{customdata[0]:,.0f}"
                                           " · P25 €%{customdata[1]:,.0f}<br>P75 €%{customdata[2]:,.0f}"
                                           " · P95 €%{customdata[3]:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=years, y=contrib, line={"color": sec, "width": 2, "dash": "dash"},
                             name="Savings paid in",
                             hovertemplate="Savings paid in €%{y:,.0f}<extra></extra>"))
    if goal > 0:
        fig.add_hline(y=goal, line={"color": muted, "width": 1, "dash": "dot"},
                      annotation_text=f"Goal {eur(goal)}", annotation_position="top left",
                      annotation_font_color=sec)
    fig.update_layout(hovermode="x")
    return _layout(fig, title, "€ (today's purchasing power)", "Years from now")


def histogram(values: np.ndarray, title: str, x_title: str, markers: dict[str, float] | None = None,
              pct: bool = False, nbins: int = 60) -> go.Figure:
    sec, muted = ink()
    v = np.asarray(values, dtype=float)
    hi = np.percentile(v, 99.5)
    v = v[v <= hi]
    fig = go.Figure(go.Histogram(
        x=v, nbinsx=nbins, marker={"color": cat()[0], "line": {"width": 1, "color": "rgba(0,0,0,0)"}},
        hovertemplate=("%{x:.1%}" if pct else "€%{x:,.0f}") + "<br>%{y} paths<extra></extra>",
        name=x_title))
    for i, (label, x) in enumerate(sorted((markers or {}).items(), key=lambda kv: kv[1])):
        fig.add_vline(x=x, line={"color": muted, "width": 1, "dash": "dot"}, annotation_text=label,
                      annotation_font_color=sec, annotation_position="top left" if i == 0 else "top right")
    fig.update_layout(bargap=0.05, showlegend=False)
    if pct:
        fig.update_xaxes(tickformat=".0%")
    return _layout(fig, title, "Paths", x_title, height=340)


def box_by_variant(results: dict[str, SimResult], title: str) -> go.Figure:
    colors = cat()
    fig = go.Figure()
    for i, (label, res) in enumerate(results.items()):
        v = res.real_final_net()
        lo, hi = np.percentile(v, [1, 99])
        v = v[(v >= lo) & (v <= hi)]
        fig.add_trace(go.Box(x=v, name=label, marker_color=colors[i % len(colors)], boxpoints=False,
                             line={"width": 2}, orientation="h",
                             hovertemplate="€%{x:,.0f}<extra>" + label + "</extra>"))
    fig.update_layout(showlegend=False)
    fig = _layout(fig, title, "", "Net wealth after tax, today's € (1–99 % range)", height=80 + 60 * len(results))
    fig.update_yaxes(autorange="reversed")  # same order as the tables
    return fig
