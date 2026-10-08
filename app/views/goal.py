import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import _layout, cat, eur, get_cfg, ink, scenarios

from mcengine.optimize import required_savings
from mcengine.portfolio.simulator import simulate

cfg = get_cfg()
st.title("Goal planner")
st.caption("How much do you need to save each month to reach a goal with a chosen confidence? Runs on the "
           "first 2,000 scenarios of your current setup.")

c = st.columns(3)
goal = c[0].number_input("Goal (today's €)", 10_000.0, 50_000_000.0, max(cfg.goal_real, 10_000.0), 10_000.0)
confs = c[1].multiselect("Confidence levels", [0.5, 0.6, 0.7, 0.8, 0.9, 0.95], [0.5, 0.8, 0.9],
                         format_func="{:.0%}".format)
paths = c[2].select_slider("Paths", [1000, 2000, 5000], 2000)


@st.cache_resource(max_entries=8, show_spinner="Solving…")
def _solve(cfg_json: str, goal: float, confs: tuple[float, ...], paths: int):
    from mcengine.config import SimConfig

    c = SimConfig.model_validate_json(cfg_json)
    sc = scenarios(c.model_copy(update={"n_paths": max(paths, c.n_paths)})).subset(paths)
    sol = {conf: required_savings(c, goal, conf, sc) for conf in confs}
    grid = np.linspace(0.25, 2.0, 8) * c.plan.monthly
    probs = []
    for m in grid:
        cc = c.model_copy(update={"plan": c.plan.model_copy(update={"monthly": float(m)})})
        probs.append(float((simulate(cc, sc).real_final_net() >= goal).mean()))
    return sol, grid, np.array(probs)


if not confs:
    st.stop()
sol, grid, probs = _solve(cfg.model_dump_json(), goal, tuple(sorted(confs)), paths)

cols = st.columns(len(sol))
for col, (conf, s) in zip(cols, sol.items(), strict=True):
    col.metric(f"{conf:.0%} confidence", f"{eur(s['monthly'])}/month",
               help=f"With this savings rate, {conf:.0%} of simulated futures end at or above the goal.")
st.caption(f"Savings grow '{cfg.plan.growth}' over {cfg.plan.horizon_years} years; start capital "
           f"{eur(cfg.plan.start_capital)}.")

sec, muted = ink()
fig = go.Figure(go.Scatter(x=grid, y=probs, mode="lines+markers", line={"color": cat()[0], "width": 2},
                           marker={"size": 8}, name="P(goal)",
                           hovertemplate="€%{x:,.0f}/month → %{y:.0%}<extra></extra>"))
fig.add_vline(x=cfg.plan.monthly, line={"color": muted, "dash": "dot"}, annotation_text="current plan",
              annotation_font_color=sec)
fig = _layout(fig, "Probability of reaching the goal vs. monthly savings", "P(goal)", "Monthly savings (€)",
              height=380)
fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
st.plotly_chart(fig, width="stretch")
st.dataframe(pd.DataFrame({"Monthly savings": [eur(x) for x in grid], "P(goal)": [f"{p:.0%}" for p in probs]}),
             hide_index=True)
