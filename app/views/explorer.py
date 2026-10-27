import json

import numpy as np
import pandas as pd
import persist
import plotly.graph_objects as go
import streamlit as st
from common import _layout, apply_cfg, cat, get_cfg, ink, scenarios, setup_summary, show_flash

from mcengine.metrics import METRIC_LABELS
from mcengine.optimize import defensive_grid, explore_allocations

cfg = get_cfg()
st.title("Allocation explorer")
st.caption("Grid search over the defensive 40–50 % (bonds, linkers, gold, cash) at several equity levels. "
           "Each point is a full simulation with your savings plan, strategy, costs and taxes, all on the same "
           "scenarios. Look for a robust *region*, not a single 'optimal' point. The model has estimation "
           "error, and optimisers amplify it (for example by piling into whichever asset has the "
           "most generous assumption).")
st.info(f"**From your Setup:** {setup_summary(cfg)}. Current allocation: "
        + " · ".join(f"{a.key.replace('_', ' ')} {cfg.weights.get(a.key, 0):.0%}" for a in cfg.assets)
        + ". The grid varies only the allocation; it is marked ★ on the chart.")
show_flash()

def_assets = [a for a in cfg.assets if a.kind != "equity"]
# Defaults follow the setup: equity levels around the current share, caps at least the current weights.
level_opts = [round(0.30 + 0.05 * i, 2) for i in range(11)]
eq_now = min(level_opts, key=lambda v: abs(v - cfg.equity_share()))
default_levels = [v for v in level_opts if abs(v - eq_now) <= 0.051]
with st.form("explore"):
    c = st.columns(4)
    levels = c[0].multiselect("Equity levels", level_opts, default_levels, format_func="{:.0%}".format)
    step = c[1].selectbox("Grid step (share of sleeve)", [0.5, 0.25, 0.2, 0.125], index=1)
    paths = c[2].select_slider("Paths per point", [500, 1000, 2000, 5000], 1000 if cfg.n_paths >= 1000 else 500,
                               help="A subset of the scenarios used on the Simulation page.")
    risk_opts = ["mdd_p95", "real_wealth_p5", "p_real_loss", "cvar5_12m", "vol_annual"]
    risk = c[3].selectbox("Risk axis", risk_opts, format_func=lambda k: METRIC_LABELS[k][0])
    st.markdown("**Maximum allowed weight per asset** – a limit for the search, *not* an allocation. "
                "The search may still end up at 0 % for an asset.")
    cc = st.columns(len(def_assets))
    caps = {a.key: cc[i].number_input(
        a.name, 0.0, 1.0, max({"gold": 0.15, "cash": 0.15}.get(a.key, 1.0),
                              float(np.ceil(cfg.weights.get(a.key, 0) * 20) / 20)), 0.05)
        for i, a in enumerate(def_assets)}
    grid = defensive_grid(cfg, levels, step, caps)

    st.markdown("**Your own allocations to test** (in %, each row must add up to 100). They are simulated on "
                "the same scenarios and marked ◆ on the chart. Saved automatically.")
    keys = [a.key for a in cfg.assets]
    saved_custom = persist.load("explorer_custom.json")
    default_custom = pd.DataFrame(saved_custom) if saved_custom else pd.DataFrame(
        [{"Name": "My mix", **dict.fromkeys(keys, 0.0), "equity": 55.0, "gov_bonds": 35.0, "gold": 5.0,
          "cash": 5.0}])
    default_custom = default_custom.reindex(columns=["Name", *keys]).fillna({k: 0.0 for k in keys})
    custom_df = st.data_editor(
        default_custom, num_rows="dynamic", hide_index=True, width="stretch", key="custom_alloc",
        column_config={"Name": st.column_config.TextColumn("Name", required=True),
                       **{a.key: st.column_config.NumberColumn(a.name.split(" (")[0] + " %", min_value=0.0,
                                                               max_value=100.0, format="%.1f")
                          for a in cfg.assets}})
    # Stable label: a label that changes with the inputs would make Streamlit drop the click.
    c = st.columns([1, 4])
    go_btn = c[0].form_submit_button("Run simulations", type="primary")
    c[1].caption(f"{len(grid)} grid allocations (with the values shown before your last edit) plus your own rows.")


def parse_custom(table: pd.DataFrame) -> tuple[list[str], list[dict[str, float]], list[str]]:
    """Valid custom rows as (names, weight dicts); problems as messages."""
    names, weights, problems = [], [], []
    for i, row in table.iterrows():
        vals = {k: float(row.get(k) or 0.0) for k in keys}
        total = sum(vals.values())
        name = str(row.get("Name") or f"Custom {i + 1}").strip()
        if total <= 0:
            continue
        if abs(total - 100.0) > 0.5:
            problems.append(f"'{name}' adds up to {total:.1f} %, not 100 %; skipped.")
            continue
        names.append(name)
        weights.append({k: v / total for k, v in vals.items()})
    return names, weights, problems


def grid_scenarios(paths: int):
    # Same scenarios as the Simulation page (cached), cut to the first ``paths``.
    return scenarios(cfg.model_copy(update={"n_paths": max(paths, cfg.n_paths)})).subset(paths)


# Everything except the weights must match the setup the grid was run on.
signature = cfg.model_dump_json(exclude={"weights"})
custom_names, custom_weights, custom_problems = parse_custom(custom_df)
for msg in custom_problems:
    st.warning(msg)
if go_btn:
    persist.save("explorer_custom.json", json.loads(custom_df.to_json(orient="records")))
if go_btn and not grid:
    st.warning("Pick at least one equity level for the grid.")
if go_btn and grid:
    bar = st.progress(0.0, "Simulating…")
    sc_run = grid_scenarios(paths)
    st.session_state["explore_df"] = explore_allocations(
        cfg, grid, sc_run, risk_metric=risk, progress=lambda f, m: bar.progress(f, m))
    custom_res = explore_allocations(cfg, custom_weights, sc_run, risk_metric=risk) if custom_weights else None
    if custom_res is not None:
        custom_res.insert(0, "label", custom_names)
    st.session_state["explore_custom"] = custom_res
    st.session_state["explore_risk"] = risk
    st.session_state["explore_paths"] = paths
    st.session_state["explore_sig"] = signature
    bar.empty()

if "explore_df" in st.session_state and st.session_state.get("explore_sig") != signature:
    for k in ("explore_df", "explore_risk", "explore_paths", "explore_sig", "explore_custom"):
        st.session_state.pop(k, None)
    st.warning("Your Setup changed since the last run (savings, strategy, taxes, costs or engine), so the old "
               "results no longer apply. Run the grid again.")

df = st.session_state.get("explore_df")
if df is None:
    st.info("Pick settings and run the grid. 100 allocations × 1,000 paths take about a minute.")
    st.stop()

risk = st.session_state.get("explore_risk", "mdd_p95")


@st.cache_resource(max_entries=8, show_spinner="Simulating your current allocation…")
def _current(cfg_json: str, paths: int, risk: str):
    from mcengine.config import SimConfig

    c = SimConfig.model_validate_json(cfg_json)
    w = {a.key: c.weights.get(a.key, 0.0) for a in c.assets}
    sc = scenarios(c.model_copy(update={"n_paths": max(paths, c.n_paths)})).subset(paths)
    return explore_allocations(c, [w], sc, risk_metric=risk).iloc[0]


cur = _current(cfg.model_dump_json(), st.session_state.get("explore_paths", 1000), risk)
wcols = [c for c in df.columns if c.startswith("w_")]
eq_col = next(c for c in wcols if c == "w_equity") if "w_equity" in wcols else wcols[0]
colors = cat()
sec, _ = ink()
fig = go.Figure()
pct_risk = METRIC_LABELS[risk][1] == "pct"
for i, (lvl, g) in enumerate(df.groupby(eq_col)):
    hover = "<br>".join(f"{c[2:]} %{{customdata[{j}]:.0%}}" for j, c in enumerate(wcols))
    fig.add_trace(go.Scatter(
        x=g[risk], y=g["real_wealth_p50"], mode="markers", name=f"{lvl:.0%} equity",
        marker={"size": np.where(g["pareto"], 12, 8), "color": colors[i % 3],
                "line": {"width": np.where(g["pareto"], 2, 1), "color": sec}},
        customdata=g[wcols].to_numpy(),
        hovertemplate=hover + "<br>median €%{y:,.0f}<br>risk %{x" + (":.1%" if pct_risk else ":,.0f")
                      + "}<extra></extra>"))
hover = "<br>".join(f"{c[2:]} {cur[c]:.0%}" for c in wcols)
fig.add_trace(go.Scatter(
    x=[cur[risk]], y=[cur["real_wealth_p50"]], mode="markers", name="★ Your setup",
    marker={"symbol": "star", "size": 18, "color": sec, "line": {"width": 1, "color": sec}},
    hovertemplate="<b>Your setup</b><br>" + hover + "<br>median €%{y:,.0f}<br>risk %{x"
                  + (":.1%" if pct_risk else ":,.0f") + "}<extra></extra>"))
custom = st.session_state.get("explore_custom")
if custom is not None and len(custom):
    hover_c = "<br>".join(f"{c[2:]} %{{customdata[{j}]:.0%}}" for j, c in enumerate(wcols))
    fig.add_trace(go.Scatter(
        x=custom[risk], y=custom["real_wealth_p50"], mode="markers+text", name="◆ Your own allocations",
        text=custom["label"], textposition="top center", textfont={"color": sec},
        marker={"symbol": "diamond", "size": 15, "color": colors[3], "line": {"width": 1.5, "color": sec}},
        customdata=custom[wcols].to_numpy(),
        hovertemplate="<b>%{text}</b><br>" + hover_c + "<br>median €%{y:,.0f}<br>risk %{x"
                      + (":.1%" if pct_risk else ":,.0f") + "}<extra></extra>"))
fig = _layout(fig, "Median outcome vs. risk (outlined = efficient)", "Median net wealth, today's €",
              METRIC_LABELS[risk][0], height=480)
if pct_risk:
    fig.update_xaxes(tickformat=".0%")
st.plotly_chart(fig, width="stretch")
if len(df[eq_col].unique()) > 3:
    st.caption("Colours repeat beyond three equity levels; use the hover labels.")

HIGHER_IS_SAFER = {"real_wealth_p5", "real_wealth_cvar5", "irr_real_p5", "cvar5_12m"}


def safety(values):
    """Risk on a 'lower is better' scale."""
    v = np.asarray(values, dtype=float)
    return -v if risk in HIGHER_IS_SAFER else v


if custom is not None and len(custom):
    st.subheader("Your own allocations")
    grid_risk, grid_med = safety(df[risk]), df["real_wealth_p50"].to_numpy()
    rows = []
    for _, r in custom.iterrows():
        cr, cm = safety([r[risk]])[0], r["real_wealth_p50"]
        better = (grid_risk <= cr + 1e-12) & (grid_med >= cm - 1e-9) & ((grid_risk < cr - 1e-12) | (grid_med > cm))
        if better.any():
            best = df[better].sort_values("real_wealth_p50", ascending=False).iloc[0]
            mix = " / ".join(f"{best[c]:.0%}" for c in wcols)
            verdict = f"❌ beaten by {mix} ({best['real_wealth_p50'] - cm:+,.0f} € median, no more risk)"
        else:
            verdict = "✅ efficient"
        rows.append({"Name": r["label"], "Verdict": verdict, **{c[2:]: f"{r[c]:.0%}" for c in wcols},
                     "Median (today's €)": f"€{r['real_wealth_p50']:,.0f}", "Bad case P5": f"€{r['real_wealth_p5']:,.0f}",
                     METRIC_LABELS[risk][0]: (f"{r[risk]:.1%}" if pct_risk else f"€{r[risk]:,.0f}"),
                     "P(goal)": f"{r['p_goal']:.1%}", "Real return p.a.": f"{r['irr_real_p50']:.2%}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    cash_asset = next((a for a in cfg.assets if a.kind == "cash"), None)
    if cash_asset is not None:
        rate = (f"your Tagesgeld rate of {cash_asset.rate_today:.2%}"
                + (" (fixed)" if cash_asset.rate_fixed else ", then moving with market rates")
                if cash_asset.rate_today is not None else
                f"the market rate ({cfg.market.short_rate_start:.1%} today); set your own rate in Setup")
        st.caption(f"Cash earns {rate}. Mixes in the verdict are written as "
                   + " / ".join(c[2:] for c in wcols) + ". 'Beaten' means a grid mix has at least the same median "
                   "*and* at most the same risk; differences of a few hundred euros are within simulation noise.")

st.subheader("Efficient allocations")
show = df[df["pareto"]].sort_values(risk)
table = show[wcols + ["real_wealth_p50", "real_wealth_p5", "mdd_p95", "irr_real_p50", "p_goal"]].copy()
for c in wcols:
    table[c] = table[c].map("{:.0%}".format)
for c in ("real_wealth_p50", "real_wealth_p5"):
    table[c] = table[c].map("€{:,.0f}".format)
for c in ("mdd_p95", "irr_real_p50", "p_goal"):
    table[c] = table[c].map("{:.1%}".format)
table.columns = [c[2:] if c.startswith("w_") else METRIC_LABELS[c][0] for c in table.columns]
st.dataframe(table, hide_index=True, width="stretch")
zero = [c[2:] for c in wcols if (show[c] == 0).all()]
if zero:
    st.caption(f"{', '.join(zero)} at 0 % in every efficient allocation: on this risk axis, swapping it for another "
               "defensive asset never lowered risk enough to pay for its lower return. Cash, for example, earns less "
               "than bonds, is taxed every year, and does not rise in a crash the way government bonds do. Its "
               "real job is an emergency buffer, which the model does not value.")
with st.expander("All simulated allocations"):
    full = df[wcols + ["real_wealth_p50", "real_wealth_p5", risk, "pareto"]].sort_values(risk)
    st.dataframe(full, hide_index=True, width="stretch")

choices = show[wcols].assign(label=[" · ".join(f"{c[2:]} {r[c]:.0%}" for c in wcols) for _, r in show.iterrows()])
if custom is not None and len(custom):
    own = custom[wcols].assign(label=[f"{n} (yours): " + " · ".join(f"{c[2:]} {r[c]:.0%}" for c in wcols)
                                      for n, (_, r) in zip(custom["label"], custom.iterrows(), strict=True)])
    choices = pd.concat([own, choices], ignore_index=True)
pick = st.selectbox("Use one of these as the target allocation", range(len(choices)),
                    format_func=lambda i: choices.iloc[i]["label"])
if st.button("Apply to setup"):
    w = {c[2:]: float(choices.iloc[pick][c]) for c in wcols}
    apply_cfg(cfg.model_copy(update={"weights": w}), "Target allocation updated. Every page now uses it.")
