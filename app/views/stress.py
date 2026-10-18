import numpy as np
import pandas as pd
import streamlit as st
from common import box_by_variant, eur, get_cfg, metrics_table, scenarios

from mcengine.compare import paired_difference, run_variants
from mcengine.config import Behavior, SimConfig
from mcengine.portfolio.simulator import simulate
from mcengine.stress import CRISES, crisis_summary, insert_crisis, start_month

cfg = get_cfg()
st.title("Stress tests & behaviour")
st.caption("What happens to *your* plan (Setup: savings, allocation, strategy, taxes) if history repeats at a bad "
           "moment, or if you sell in a panic. Every comparison uses the same simulated futures, so the "
           "differences come only from the crisis or the behaviour.")

PATHS = 2000
KEYS = ["real_wealth_p5", "real_wealth_p50", "real_wealth_p95", "p_goal", "irr_real_p50", "mdd_p95", "taxes_real"]
WHEN = {"start": "In the first years", "middle": "In the middle of the plan",
        "end": "Right before the goal date", "year": "In a chosen year"}


def shared_scenarios(c: SimConfig):
    return scenarios(c.model_copy(update={"n_paths": max(PATHS, c.n_paths)})).subset(PATHS)


# ----------------------------------------------------------------------------- 1. stress tests
st.header("1 · Replay a historical crisis")
c = st.columns([2, 2, 1])
key = c[0].selectbox("Crisis", list(CRISES), format_func=lambda k: CRISES[k].name, index=2)
when = c[1].radio("When it hits", list(WHEN), format_func=WHEN.get, index=2)
year = c[2].number_input("Year", 1, cfg.plan.horizon_years, max(1, cfg.plan.horizon_years // 2),
                         disabled=when != "year")
recovery = st.checkbox("Also replay the 24 months after the crisis (historical recovery)", True,
                       help="Without it the crash is followed by ordinary simulated months, so the historical "
                            "rebound is missing. That is the worst case.")
rec_m = 24 if recovery else 0
st.markdown(f"**{CRISES[key].name}** – {CRISES[key].story}")
summ = crisis_summary(cfg, key, rec_m)
st.dataframe(summ.assign(**{col: summ[col].map("{:+.0%}".format) for col in summ.columns[1:]}),
             hide_index=True, width="stretch")
if when == "end" and recovery:
    st.caption("Right before the goal date the crash ends in your last month: there is no time left for the "
               "recovery, so the recovery months are not part of the plan.")


@st.cache_resource(max_entries=8, show_spinner="Running the plan with and without the crisis…")
def _stress(cfg_json: str, key: str, start: int, rec_m: int):
    c = SimConfig.model_validate_json(cfg_json)
    sc = shared_scenarios(c)
    base = simulate(c, sc, label="Normal futures")
    hit = simulate(c, insert_crisis(sc, c, key, start, rec_m), label=f"With {CRISES[key].name}")
    from mcengine.metrics import summarize

    return {base.label: base, hit.label: hit}, {base.label: summarize(base, c.goal_real),
                                                hit.label: summarize(hit, c.goal_real)}


start = start_month(cfg, key, year if when == "year" else when, rec_m)
results, summaries = _stress(cfg.model_dump_json(), key, start, rec_m)
st.caption(f"Crisis starts in month {start + 1} (year {start // 12 + 1}) of {cfg.months}.")
st.dataframe(metrics_table(summaries, KEYS), width="stretch")
labels = list(results)
d = paired_difference(results[labels[1]], results[labels[0]])
st.metric("Average effect on your final wealth (today's €)", f"{d['mean']:+,.0f} €",
          help=f"± {eur(d['se'])} standard error; worse on {1 - d['share_a_better']:.0%} of futures.")
st.plotly_chart(box_by_variant(results, "Net wealth at the end, with and without the crisis"), width="stretch")

with st.expander("Compare all crises at once (start / middle / right before the goal)"):
    if st.button("Run all crises"):
        rows = []
        bar = st.progress(0.0)
        sc = shared_scenarios(cfg).subset(1000)
        base = simulate(cfg, sc)
        for i, k in enumerate(CRISES):
            row = {"Crisis": CRISES[k].name}
            for w in ("start", "middle", "end"):
                res = simulate(cfg, insert_crisis(sc, cfg, k, start_month(cfg, k, w, rec_m), rec_m))
                row[WHEN[w]] = paired_difference(res, base)["mean"]
            rows.append(row)
            bar.progress((i + 1) / len(CRISES))
        bar.empty()
        tbl = pd.DataFrame(rows).set_index("Crisis")
        st.dataframe(tbl.map(lambda v: f"{v:+,.0f} €"), width="stretch")
        st.caption("Average change of final net wealth (today's €) versus normal futures, 1,000 paths. "
                   "The same crash costs far more close to the goal date, because more money is invested and "
                   "there is no time left to recover. That's the reason for a glide path.")

# ----------------------------------------------------------------------------- 2. behaviour
st.header("2 · What panic selling would cost")
st.markdown("Suppose that after a big fall you sell **all your stocks**, keep saving into cash only, and buy back "
            "after a while. This is a *what-if* to show the price of that reaction, **not** a strategy.")
c = st.columns(2)
dd = c[0].slider("Sell everything after the portfolio fell by", 0.15, 0.50, 0.30, 0.05, format="%.2f")
out_m = c[1].slider("Buy back after … months", 3, 36, 12, 3)


@st.cache_resource(max_entries=8, show_spinner="Simulating disciplined vs. panicking investor…")
def _panic(cfg_json: str, dd: float, out_m: int):
    c = SimConfig.model_validate_json(cfg_json)
    calm = c.model_copy(update={"behavior": Behavior()})
    panic = c.model_copy(update={"behavior": Behavior(panic=True, drawdown=dd, months_out=out_m)})
    return run_variants({"Stay invested": calm, f"Panic at −{dd:.0%}, back after {out_m}m": panic},
                        shared_scenarios(c))


ptable, pres = _panic(cfg.model_dump_json(), dd, out_m)
calm_r, panic_r = pres.values()
diff = panic_r.real_final_net() - calm_r.real_final_net()
hit = panic_r.panic_sales > 0
c = st.columns(3)
c[0].metric("Futures with a panic sale", f"{hit.mean():.0%}")
c[1].metric("Average cost when it happens", f"{diff[hit].mean():+,.0f} €" if hit.any() else "–",
            help="Mean change of final net wealth (today's €) on the futures where the sale was triggered.")
c[2].metric("Median cost when it happens", f"{np.median(diff[hit]):+,.0f} €" if hit.any() else "–")
st.dataframe(metrics_table({k: v.to_dict() for k, v in ptable.iterrows()}, KEYS), width="stretch")
st.caption("Selling after a fall locks in the loss, the sale is taxed, and the rebound that typically follows "
           "a crash is missed. Sometimes it helps (when the fall continues for long), but on average it costs. "
           "The drawdown metric looks better only because the portfolio sat in cash, and that is exactly the "
           "return you gave up.")
