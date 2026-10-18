import pandas as pd
import streamlit as st
from common import box_by_variant, equity_share_chart, eur, get_cfg, metrics_table, scenarios

from mcengine.compare import DEFAULT_STRATEGIES, paired_difference, run_variants

MINE = " (your setup)"

cfg = get_cfg()
st.title("Rebalancing strategies")
st.caption("All strategies run on the *same* market scenarios with the same savings outlay, so the "
           "differences come from the rebalancing rule alone (paired comparison). Savings, allocation, costs "
           f"and taxes come from your Setup; your own policy is marked *{MINE.strip()}*.")


@st.cache_resource(max_entries=4, show_spinner="Running all strategies on shared scenarios…")
def _compare(cfg_json: str):
    from mcengine.config import SimConfig

    c = SimConfig.model_validate_json(cfg_json)
    mine = c.strategy.label()
    strategies = DEFAULT_STRATEGIES + ([c.strategy] if mine not in {s.label() for s in DEFAULT_STRATEGIES} else [])
    variants = {s.label() + (MINE if s.label() == mine else ""): c.model_copy(update={"strategy": s})
                for s in strategies}
    return run_variants(variants, scenarios(c))


df, results = _compare(cfg.model_dump_json())
st.plotly_chart(box_by_variant(results, "Net wealth at the end, by strategy"), width="stretch")

keys = ["real_wealth_p5", "real_wealth_p50", "real_wealth_p95", "real_wealth_cvar5", "p_goal", "irr_real_p50",
        "mdd_p95", "share_negative_years", "taxes_real", "fees_real", "turnover_real", "rebalances_p50"]
st.dataframe(metrics_table({k: v.to_dict() for k, v in df.iterrows()}, keys), width="stretch")

st.subheader("Paired difference vs. a baseline")
labels = list(results)
base = st.selectbox("Baseline", labels, index=next(i for i, lab in enumerate(labels) if lab.endswith(MINE)))
rows = []
for lab in labels:
    if lab == base:
        continue
    d = paired_difference(results[lab], results[base])
    rows.append({"Strategy": lab, "Mean difference": eur(d["mean"]), "± std. error": eur(d["se"]),
                 "Better on share of paths": f"{d['share_a_better']:.0%}",
                 "P5 of difference": eur(d["p5"]), "P95 of difference": eur(d["p95"])})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption("A mean difference of more than about 2–3 standard errors is a real effect, not luck of the draw.")

st.subheader("Dynamic risk management")
st.caption("Your setup without dynamic rules, with a glide path, and with the 6- or 12-month re-evaluation switch, "
           "on the same scenarios. Glide-path and switch parameters come from Setup (defaults if switched off there).")


@st.cache_resource(max_entries=4, show_spinner="Comparing dynamic rules on shared scenarios…")
def _dynamic(cfg_json: str):
    from mcengine.config import GlidePath, Reevaluation, SimConfig

    c = SimConfig.model_validate_json(cfg_json)
    glide = c.glide.model_copy(update={"enabled": True})
    reev = c.reevaluation.model_copy(update={"enabled": True})
    off_g, off_r = GlidePath(), Reevaluation()
    variants = {
        "Static (no dynamic rules)": c.model_copy(update={"glide": off_g, "reevaluation": off_r}),
        f"Glide path → {glide.end_equity:.0%} over {glide.years}y": c.model_copy(
            update={"glide": glide, "reevaluation": off_r}),
        "Re-evaluation every 12 months": c.model_copy(
            update={"glide": off_g, "reevaluation": reev.model_copy(update={"every_months": 12})}),
        "Re-evaluation every 6 months": c.model_copy(
            update={"glide": off_g, "reevaluation": reev.model_copy(update={"every_months": 6})}),
        "Glide path + re-evaluation (12m)": c.model_copy(
            update={"glide": glide, "reevaluation": reev.model_copy(update={"every_months": 12})}),
    }
    return run_variants(variants, scenarios(c))


if cfg.goal_real <= 0:
    st.info("Set a wealth goal in Setup: the re-evaluation switch needs one.")
else:
    ddf, dres = _dynamic(cfg.model_dump_json())
    dkeys = ["real_wealth_p5", "real_wealth_p50", "real_wealth_p95", "p_goal", "irr_real_p50", "mdd_p95",
             "taxes_real", "turnover_real"]
    table = metrics_table({k: v.to_dict() for k, v in ddf.iterrows()}, dkeys)
    table.loc["Switches per path (avg.)"] = [f"{r.switches.mean():.1f}" for r in dres.values()]
    table.loc["Time in defensive mix (avg.)"] = [f"{r.defensive_months.mean() / r.months:.0%}" for r in dres.values()]
    st.dataframe(table, width="stretch")
    st.plotly_chart(equity_share_chart(dres, "Equity share over time (median of all paths)"), width="stretch")
    base_lab = next(iter(dres))
    rows = []
    for lab, r in list(dres.items())[1:]:
        d = paired_difference(r, dres[base_lab])
        rows.append({"Rule": lab, "Mean difference vs. static": eur(d["mean"]), "± std. error": eur(d["se"]),
                     "Better on share of paths": f"{d['share_a_better']:.0%}",
                     "P5 of difference": eur(d["p5"]), "P95 of difference": eur(d["p95"])})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("How to read it: dynamic rules trade upside for safety. They usually raise the bad case and lower "
               "the drawdown, but give up part of the good outcomes, so the *average* can fall even when the goal "
               "probability rises. Which trade-off you prefer is a personal choice, not a model result.")
