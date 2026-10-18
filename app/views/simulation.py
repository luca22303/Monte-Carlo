import numpy as np
import streamlit as st
from common import (
    equity_band_chart,
    eur,
    fan_chart,
    get_cfg,
    histogram,
    kpi_row,
    metrics_table,
    run,
    scenarios,
    setup_summary,
)

from mcengine.metrics import max_drawdown, real_contributions

cfg = get_cfg()
st.title("Simulation")
st.caption(setup_summary(cfg))

res, s = run(cfg)
kpi_row(s, cfg.goal_real)
n_lines = st.select_slider("Paths drawn", [50, 100, 250, 500, 1000], 250,
                           help="How many of the simulated futures are drawn as individual lines. "
                                "Median and percentiles always use all paths.")
st.plotly_chart(fan_chart(res, cfg.goal_real, n_lines=n_lines), width="stretch")

c = st.columns(2)
paid = float(np.median(real_contributions(res).sum(axis=1)))
markers = {"Savings paid": paid}
if cfg.goal_real > 0:
    markers["Goal"] = cfg.goal_real
c[0].plotly_chart(histogram(res.real_final_net(), "Where you end up (after tax, today's €)",
                            "Net wealth", markers), width="stretch")
c[1].plotly_chart(histogram(max_drawdown(res.nav.astype(float)), "Worst fall from a peak during the plan",
                            "Max drawdown", pct=True), width="stretch")

if res.equity_share is not None:
    st.plotly_chart(equity_band_chart(res, "Equity share over time (after drift, rebalancing and dynamic rules)"),
                    width="stretch")
if cfg.reevaluation.enabled and res.switches is not None:
    c = st.columns(3)
    c[0].metric("Paths that switched at least once", f"{(res.switches > 0).mean():.0%}")
    c[1].metric("Switches per path (avg.)", f"{res.switches.mean():.1f}")
    c[2].metric("Time in defensive mix (avg.)", f"{res.defensive_months.mean() / res.months:.0%}")

st.subheader("All metrics")
st.dataframe(metrics_table({"This plan": s}), width="stretch")
st.caption(
    f"Scenario source: {scenarios(cfg).source}. Monte Carlo standard error of the median: "
    f"{eur(s['se_median_wealth'])}. Differences between settings smaller than ~2× this are noise unless "
    "compared on the same scenarios (Strategies page)."
)
