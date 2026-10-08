import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import _layout, cat, diverging, eur, get_cfg, ink, run, scenarios

from mcengine.config import FACTORS
from mcengine.data.panel import annual_stats, historical_returns
from mcengine.metrics import summarize
from mcengine.portfolio.simulator import simulate
from mcengine.scenarios.historical import rolling_windows

cfg = get_cfg()
st.title("Assumptions & reality check")

mm = cfg.market
st.subheader("Regimes of the parametric model")
pi = mm.stationary_distribution()
p = np.asarray(mm.transition)
rows = []
for k, rg in enumerate(mm.regimes):
    rows.append({"Regime": rg.name, "Share of time": f"{pi[k]:.0%}",
                 "Avg. length": f"{1 / max(1 - p[k, k], 1e-9):.0f} months",
                 "Equity vol": f"{rg.vol['equity']:.0%}", "Equity drift (vs. calm)": f"{rg.drift['equity']:+.0%}",
                 "Yield drift p.a.": f"{rg.drift['nominal_yield']:+.1%}",
                 "Inflation drift p.a.": f"{rg.drift['inflation']:+.1%}", "Tail df": f"{rg.df:g}"})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
st.caption("Crisis regimes are short and violent; the inflation regime is the 2022 case where stocks and bonds "
           "fall together. Long-run averages are set separately, so regimes shape the risk without changing "
           "the expected return.")

st.subheader("Model vs. history (1973–2025, EUR)")


@st.cache_resource(max_entries=4, show_spinner="Comparing with history…")
def _stats(cfg_json: str):
    from mcengine.config import SimConfig

    c = SimConfig.model_validate_json(cfg_json)
    hist = historical_returns(c.assets, c.market)
    h = annual_stats(hist.returns)
    pc = c.model_copy(update={"generator": "parametric", "n_paths": min(c.n_paths, 3000)})
    m = annual_stats(scenarios(pc).returns.astype(float))
    return hist, h, m


hist, h, m = _stats(cfg.model_dump_json())
names = [a.name for a in cfg.assets]
tbl = pd.DataFrame({
    "Asset": names,
    "History: return p.a.": h["geo_return"].map("{:.1%}".format).to_numpy(),
    "Model: return p.a.": m["geo_return"].map("{:.1%}".format).to_numpy(),
    "History: volatility": h["volatility"].map("{:.1%}".format).to_numpy(),
    "Model: volatility": m["volatility"].map("{:.1%}".format).to_numpy(),
    "History: worst 12m": h["worst_12m"].map("{:.0%}".format).to_numpy(),
    "Data": [hist.notes[a.key] for a in cfg.assets],
})
st.dataframe(tbl, hide_index=True, use_container_width=True)
st.caption("History returns are higher than the model's on purpose. 1973–2025 started with 8 % bond yields, "
           "had falling rates for 40 years and a US-led equity boom. Today's starting yields are about 3 %. "
           "The model's volatilities should look like history's.")

c = st.columns(2)
sec, _ = ink()
for col, (title, mat) in zip(c, [
    ("Correlation – history", np.corrcoef(hist.returns.T)),
    ("Correlation – model", np.corrcoef(scenarios(cfg.model_copy(update={"generator": "parametric",
                                                                         "n_paths": min(cfg.n_paths, 3000)}))
                                       .returns.astype(float).reshape(-1, len(names)).T)),
], strict=True):
    short = [a.key for a in cfg.assets]
    fig = go.Figure(go.Heatmap(z=mat, x=short, y=short, zmin=-1, zmax=1, colorscale=diverging(),
                               text=np.round(mat, 2), texttemplate="%{text}", xgap=2, ygap=2,
                               hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>"))
    fig = _layout(fig, title, height=360)
    fig.update_yaxes(autorange="reversed")
    col.plotly_chart(fig, use_container_width=True)

st.subheader("Your plan on actual history")
st.caption("Your exact plan (savings, allocation, strategy, taxes) run on every historical window of your horizon "
           "that starts in a January. The windows overlap heavily, so this is a sanity check, not a forecast.")
try:
    @st.cache_resource(max_entries=4, show_spinner="Backtesting on history…")
    def _hist(cfg_json: str):
        from mcengine.config import SimConfig

        c = SimConfig.model_validate_json(cfg_json)
        sc, starts = rolling_windows(c)
        r = simulate(c, sc)
        return r, starts, summarize(r, c.goal_real)

    hr, starts, hs = _hist(cfg.model_dump_json())
    res, s = run(cfg)
    pcts = np.percentile(res.real_final_net(), [5, 50, 95])
    fig = go.Figure(go.Bar(x=[d.year for d in starts], y=hr.real_final_net(), marker={"color": cat()[0]},
                           name="Historical window", hovertemplate="start %{x}: €%{y:,.0f}<extra></extra>"))
    for label, v in zip(["MC P5", "MC median", "MC P95"], pcts, strict=True):
        fig.add_hline(y=v, line={"dash": "dot", "color": sec, "width": 1}, annotation_text=label,
                      annotation_font_color=sec)
    fig = _layout(fig, "Net wealth (today's €) by start year vs. Monte Carlo percentiles", "€", "Start year",
                  height=380)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"Historical windows: median **{eur(hs['real_wealth_p50'])}**, worst **{eur(hr.real_final_net().min())}**, "
                f"median real return **{hs['irr_real_p50']:.1%}** p.a. · Monte Carlo median "
                f"**{eur(s['real_wealth_p50'])}** ({s['irr_real_p50']:.1%} p.a.).")
except ValueError as e:
    st.info(f"Backtest not possible: {e}")

st.subheader("Risk factors")
st.caption("The parametric model simulates " + ", ".join(FACTORS) + " and derives asset returns from them: "
           "bonds from yield changes × duration, linkers from real-yield changes plus realised inflation, and "
           "cash from the short rate.")
