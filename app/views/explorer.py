import numpy as np
import plotly.graph_objects as go
import streamlit as st
from common import _layout, cat, get_cfg, ink, scenarios, set_cfg

from mcengine.metrics import METRIC_LABELS
from mcengine.optimize import defensive_grid, explore_allocations

cfg = get_cfg()
st.title("Allocation explorer")
st.caption("Grid search over the defensive 40–50 % (bonds, linkers, gold, cash) at several equity levels. "
           "Each point is a full simulation with your savings plan, strategy, costs and taxes, all on the same "
           "scenarios. Look for a robust *region*, not a single 'optimal' point. The model has estimation "
           "error, and optimisers amplify it (for example by piling into whichever asset has the "
           "most generous assumption).")

def_assets = [a for a in cfg.assets if a.kind != "equity"]
with st.form("explore"):
    c = st.columns(4)
    levels = c[0].multiselect("Equity levels", [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7], [0.5, 0.55, 0.6],
                              format_func="{:.0%}".format)
    step = c[1].selectbox("Grid step (share of sleeve)", [0.5, 0.25, 0.2, 0.125], index=1)
    paths = c[2].select_slider("Paths per point", [500, 1000, 2000, 5000], 1000)
    risk_opts = ["mdd_p95", "real_wealth_p5", "p_real_loss", "cvar5_12m", "vol_annual"]
    risk = c[3].selectbox("Risk axis", risk_opts, format_func=lambda k: METRIC_LABELS[k][0])
    st.markdown("**Caps** (max portfolio weight)")
    cc = st.columns(len(def_assets))
    caps = {a.key: cc[i].number_input(a.name, 0.0, 1.0, {"gold": 0.15, "cash": 0.15}.get(a.key, 1.0), 0.05,
                                      key=f"cap_{a.key}") for i, a in enumerate(def_assets)}
    grid = defensive_grid(cfg, levels, step, caps)
    go_btn = st.form_submit_button(f"Run {len(grid)} allocations", type="primary")

if go_btn and grid:
    bar = st.progress(0.0, "Simulating…")
    sc = scenarios(cfg.model_copy(update={"n_paths": max(paths, 1000)})).subset(paths)
    st.session_state["explore_df"] = explore_allocations(
        cfg, grid, sc, risk_metric=risk, progress=lambda f, m: bar.progress(f, m))
    st.session_state["explore_risk"] = risk
    bar.empty()

df = st.session_state.get("explore_df")
if df is None:
    st.info("Pick settings and run the grid. 100 allocations × 1,000 paths take about a minute.")
    st.stop()

risk = st.session_state.get("explore_risk", "mdd_p95")
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
fig = _layout(fig, "Median outcome vs. risk (outlined = efficient)", "Median net wealth, today's €",
              METRIC_LABELS[risk][0], height=480)
if pct_risk:
    fig.update_xaxes(tickformat=".0%")
st.plotly_chart(fig, use_container_width=True)
if len(df[eq_col].unique()) > 3:
    st.caption("Colours repeat beyond three equity levels; use the hover labels.")

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
st.dataframe(table, hide_index=True, use_container_width=True)

pick = st.selectbox("Use one of these as the target allocation", range(len(show)),
                    format_func=lambda i: " · ".join(f"{c[2:]} {show.iloc[i][c]:.0%}" for c in wcols))
if st.button("Apply to setup"):
    w = {c[2:]: float(show.iloc[pick][c]) for c in wcols}
    set_cfg(cfg.model_copy(update={"weights": w}))
    st.success("Target allocation updated.")
