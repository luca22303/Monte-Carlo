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
# Blue shades for individual simulated paths, light → dark. Dark mode drops the darkest steps,
# which would vanish against the background.
PATH_BLUES = ["#9ec5f4", "#78abec", "#5592e2", "#2a78d6", "#2266bc", "#1b56a1", "#154585", "#0f3569"]
PATH_BLUES_DARK = ["#d4e5fa", "#b2d0f6", "#8fbaf0", "#6ba3ea", "#4b8de0", "#3378cf", "#2865b4", "#20549a"]


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


def path_blues() -> list[str]:
    return PATH_BLUES_DARK if is_dark() else PATH_BLUES


def background() -> str:
    return "#0e1117" if is_dark() else "#ffffff"


# ----------------------------------------------------------------------------- state
def get_cfg() -> SimConfig:
    if "cfg_json" not in st.session_state:
        st.session_state["cfg_json"] = SimConfig().model_dump_json()
    return SimConfig.model_validate_json(st.session_state["cfg_json"])


def set_cfg(cfg: SimConfig) -> None:
    st.session_state["cfg_json"] = cfg.model_dump_json()


def apply_cfg(cfg: SimConfig, message: str) -> None:
    """Save ``cfg`` for every page and rerun, so the sidebar and all page defaults pick it up."""
    set_cfg(cfg)
    st.session_state["_flash"] = message
    st.rerun()


def show_flash() -> None:
    """Show the confirmation left by :func:`apply_cfg` before its rerun."""
    if msg := st.session_state.pop("_flash", None):
        st.success(msg)


def setup_summary(cfg: SimConfig) -> str:
    """One line with the settings every page runs on."""
    extras = []
    if cfg.glide.enabled:
        extras.append(f"glide → {cfg.glide.end_equity:.0%} over {cfg.glide.years}y")
    if cfg.reevaluation.enabled:
        extras.append(f"re-evaluation every {cfg.reevaluation.every_months}m")
    return (f"{cfg.plan.horizon_years} years · €{cfg.plan.monthly:,.0f}/month ({cfg.plan.growth}) · "
            f"{cfg.equity_share():.0%} equity · {cfg.strategy.label()} · "
            + "".join(f"{e} · " for e in extras)
            + f"{'German taxes' if cfg.tax.enabled else 'pre-tax'} · {cfg.n_paths:,} paths ({cfg.generator})")


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


def fan_chart(res: SimResult, goal: float = 0.0, title: str = "Wealth in today's €",
              n_lines: int = 250) -> go.Figure:
    """Individual simulated paths in shades of blue, with median, P5/P95 and savings on top."""
    w = res.real_wealth()
    f = fan(res, real=True)
    years = (np.arange(res.months) + 1) / 12
    contrib = np.median(np.cumsum(real_contributions(res), axis=1), axis=0)
    blues = path_blues()
    sec, muted = ink()
    halo = background()
    accent = "#e8f1fd" if is_dark() else "#0a2a52"
    fig = go.Figure()

    # A fixed random sample of paths; shades are dealt round-robin so neighbouring lines differ.
    rng = np.random.default_rng(0)
    idx = rng.choice(res.n_paths, size=min(n_lines, res.n_paths), replace=False)
    x_one = np.append(years, np.nan)
    for i, color in enumerate(blues):
        sel = idx[i::len(blues)]
        if sel.size == 0:
            continue
        y = np.concatenate([w[sel], np.full((sel.size, 1), np.nan)], axis=1).ravel()
        fig.add_trace(go.Scatter(x=np.tile(x_one, sel.size), y=y, mode="lines",
                                 line={"color": color, "width": 1}, opacity=0.75, connectgaps=False,
                                 hoverinfo="skip", legendgroup="paths", showlegend=i == len(blues) // 2,
                                 name=f"Individual futures ({idx.size:,} of {res.n_paths:,})"))

    def overlay(y, color, width, dash, name, **kw):
        # Background-coloured halo underneath keeps the line readable on top of the paths.
        fig.add_trace(go.Scatter(x=years, y=y, line={"color": halo, "width": width + 3}, hoverinfo="skip",
                                 showlegend=False, legendgroup=name))
        fig.add_trace(go.Scatter(x=years, y=y, line={"color": color, "width": width, "dash": dash}, name=name,
                                 legendgroup=name, **kw))

    overlay(f[5], accent, 1.5, "dash", "5 % / 95 % of futures", hoverinfo="skip")
    overlay(f[95], accent, 1.5, "dash", "5 % / 95 % of futures", hoverinfo="skip", showlegend=False)
    overlay(contrib, sec, 2, "dot", "Savings paid in", hovertemplate="Savings paid in €%{y:,.0f}<extra></extra>")
    overlay(f[50], accent, 3, "solid", "Median",
            customdata=np.stack([f[5], f[25], f[75], f[95]], axis=1),
            hovertemplate="Year %{x:.1f}<br>Median €%{y:,.0f}<br>P5 €%{customdata[0]:,.0f}"
                          " · P25 €%{customdata[1]:,.0f}<br>P75 €%{customdata[2]:,.0f}"
                          " · P95 €%{customdata[3]:,.0f}<extra></extra>")
    if goal > 0:
        fig.add_hline(y=goal, line={"color": muted, "width": 1, "dash": "dot"},
                      annotation_text=f"Goal {eur(goal)}", annotation_position="top left",
                      annotation_font_color=sec)
    # Frame the bulk of the paths; the few most extreme ones may leave the chart at the top.
    top = max(float(np.percentile(w, 99, axis=0).max()), goal) * 1.05
    fig.update_layout(hovermode="x")
    fig = _layout(fig, title, "€ (today's purchasing power)", "Years from now", height=480)
    fig.update_yaxes(range=[min(0.0, float(w.min())), top])
    return fig


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


def equity_share_chart(results: dict[str, SimResult], title: str) -> go.Figure:
    """Median equity share over time per variant (one line each, categorical colours in fixed order)."""
    colors = cat()
    fig = go.Figure()
    for i, (label, res) in enumerate(results.items()):
        if res.equity_share is None:
            continue
        years = (np.arange(res.months) + 1) / 12
        fig.add_trace(go.Scatter(x=years, y=np.median(res.equity_share, axis=0), mode="lines", name=label,
                                 line={"color": colors[i % len(colors)], "width": 2},
                                 hovertemplate=label + ": %{y:.0%} in year %{x:.1f}<extra></extra>"))
    fig = _layout(fig, title, "Equity share", "Years from now", height=360)
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    return fig


def equity_band_chart(res: SimResult, title: str = "Equity share over time") -> go.Figure:
    """Median and 10-90 % band of the equity share of one simulation."""
    s = seq()
    years = (np.arange(res.months) + 1) / 12
    q10, q50, q90 = np.percentile(res.equity_share, [10, 50, 90], axis=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=q90, line={"width": 0}, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=years, y=q10, fill="tonexty", fillcolor=s["100"], line={"width": 0},
                             name="10–90 % of futures", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=years, y=q50, line={"color": s["600"], "width": 2}, name="Median",
                             hovertemplate="Year %{x:.1f}: %{y:.0%} equity<extra></extra>"))
    fig = _layout(fig, title, "Equity share", "Years from now", height=320)
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    return fig
