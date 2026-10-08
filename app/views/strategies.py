import pandas as pd
import streamlit as st
from common import box_by_variant, eur, get_cfg, metrics_table, scenarios

from mcengine.compare import DEFAULT_STRATEGIES, compare_strategies, paired_difference

cfg = get_cfg()
st.title("Rebalancing strategies")
st.caption("All strategies run on the *same* market scenarios with the same savings outlay, so the "
           "differences come from the rebalancing rule alone (paired comparison).")


@st.cache_resource(max_entries=4, show_spinner="Running all strategies on shared scenarios…")
def _compare(cfg_json: str):
    from mcengine.config import SimConfig

    c = SimConfig.model_validate_json(cfg_json)
    return compare_strategies(c, DEFAULT_STRATEGIES, sc=scenarios(c))


df, results = _compare(cfg.model_dump_json())
st.plotly_chart(box_by_variant(results, "Net wealth at the end, by strategy"), use_container_width=True)

keys = ["real_wealth_p5", "real_wealth_p50", "real_wealth_p95", "real_wealth_cvar5", "p_goal", "irr_real_p50",
        "mdd_p95", "share_negative_years", "taxes_real", "fees_real", "turnover_real", "rebalances_p50"]
st.dataframe(metrics_table({k: v.to_dict() for k, v in df.iterrows()}, keys), use_container_width=True)

st.subheader("Paired difference vs. a baseline")
labels = list(results)
base = st.selectbox("Baseline", labels, index=1)
rows = []
for lab in labels:
    if lab == base:
        continue
    d = paired_difference(results[lab], results[base])
    rows.append({"Strategy": lab, "Mean difference": eur(d["mean"]), "± std. error": eur(d["se"]),
                 "Better on share of paths": f"{d['share_a_better']:.0%}",
                 "P5 of difference": eur(d["p5"]), "P95 of difference": eur(d["p95"])})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
st.caption("A mean difference of more than about 2–3 standard errors is a real effect, not luck of the draw.")
