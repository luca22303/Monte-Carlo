import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import _layout, cat, eur, get_cfg, ink, scenarios, set_cfg

from mcengine.compare import run_variants
from mcengine.etfs import (
    CANDIDATES_AS_OF,
    JUSTETF_URL,
    bank_fund_costs,
    cheapest_costs,
    checklist,
    class_costs,
    gold_is_taxable,
    load_candidates,
    missing_classes,
    savings_plan,
    with_costs,
)

cfg = get_cfg()
names = {a.key: a.name for a in cfg.assets}
short = {a.key: a.key.replace("_", " ") for a in cfg.assets}

st.title("ETFs & savings plan")
st.markdown(
    "Pick **one product per asset class**. ETFs that track the same index return almost exactly the same "
    "before costs, so products are chosen on **fixed criteria** (cost, size, accumulating, domicile, "
    "savings-plan availability), not on forecasts. Below you see what the cost differences add up to "
    "over your plan, and the exact monthly amount for each savings plan."
)
st.caption(f"The example shortlist was checked against justETF on {CANDIDATES_AS_OF}. Fees and fund sizes change, so "
           "open each product's justETF link and check it before buying. These are examples, not personal advice. "
           "Add your own products in the last table row.")

if "etf_df" not in st.session_state:
    st.session_state["etf_df"] = load_candidates()
base = st.session_state["etf_df"].copy()
base["ter_pct"] = base["ter"].astype(float) * 100
base["link"] = [JUSTETF_URL.format(i) if i else None for i in base["isin"]]

# ----------------------------------------------------------------------------- 1. choose
st.subheader("1 · Choose your products")
with st.form("etf_form"):
    edited = st.data_editor(
        base[["use", "asset_key", "name", "isin", "ter_pct", "fund_size_eur_m", "distribution", "replication",
              "domicile", "delivery_claim", "savings_plan", "share_pct", "link", "note"]],
        column_config={
            "use": st.column_config.CheckboxColumn("Use", help="Tick the product(s) you will buy"),
            "asset_key": st.column_config.SelectboxColumn("Asset class", options=list(names), required=True),
            "name": st.column_config.TextColumn("Product", width="large"),
            "isin": "ISIN",
            "ter_pct": st.column_config.NumberColumn("TER %", format="%.2f", min_value=0.0, max_value=5.0),
            "fund_size_eur_m": st.column_config.NumberColumn("Size €m", format="%.0f"),
            "distribution": st.column_config.SelectboxColumn("Payout", options=["Accumulating", "Distributing"]),
            "replication": st.column_config.SelectboxColumn("Replication", options=["Physical", "Synthetic"]),
            "domicile": "Domicile",
            "delivery_claim": st.column_config.CheckboxColumn("Gold: delivery claim"),
            "savings_plan": st.column_config.CheckboxColumn("Savings plan at my broker"),
            "share_pct": st.column_config.NumberColumn("Share in class %", min_value=0.0, max_value=100.0,
                                                       help="Only if you split one class across several products"),
            "link": st.column_config.LinkColumn("justETF", display_text="open"),
            "note": st.column_config.TextColumn("Note", width="medium"),
        },
        num_rows="dynamic", hide_index=True, width="stretch", disabled=["link"],
    )
    c = st.columns([1, 1, 3])
    saved = c[0].form_submit_button("Apply selection", type="primary")
    reset = c[1].form_submit_button("Reset to examples")

if reset:
    st.session_state["etf_df"] = load_candidates()
    st.session_state.pop("etf_costs_result", None)
    st.rerun()
if saved:
    df_new = edited.copy()
    df_new["ter"] = df_new["ter_pct"].fillna(0.0).astype(float) / 100
    df_new["isin"] = df_new["isin"].fillna("").astype(str)
    df_new["name"] = df_new["name"].fillna("(unnamed)")
    df_new["use"] = df_new["use"].fillna(False).astype(bool)
    df_new["savings_plan"] = df_new["savings_plan"].fillna(True).astype(bool)
    df_new["share_pct"] = df_new["share_pct"].fillna(100.0)
    df_new["note"] = df_new["note"].fillna("")
    df_new = df_new.dropna(subset=["asset_key"])
    st.session_state["etf_df"] = df_new.drop(columns=["ter_pct", "link"])
    st.session_state.pop("etf_costs_result", None)
    st.rerun()

df = st.session_state["etf_df"]
missing = missing_classes(df, cfg)
if missing:
    st.warning("No product selected for: " + ", ".join(names[k] for k in missing) + ". Tick **Use** for one product "
               "in each asset class that has a target weight.")

# ----------------------------------------------------------------------------- 2. checklist
st.subheader("2 · Checklist")
used = df[df["use"]]
if used.empty:
    st.info("Tick at least one product.")
    st.stop()
chk = checklist(used, cfg)
verdict = {"good": "✅ meets all criteria", "warn": "⚠️ usable, see notes", "bad": "❌ problem"}
chk.insert(0, "Asset class", chk.pop("asset_key").map(short))
chk["verdict"] = chk["verdict"].map(verdict)
st.dataframe(chk.rename(columns={"name": "Product", "verdict": "Verdict"}), hide_index=True, width="stretch")

# ----------------------------------------------------------------------------- 3. savings plan
st.subheader("3 · Your monthly savings plans")
c = st.columns(3)
minimum = c[0].number_input("Broker minimum per savings-plan execution (€)", 0.0, 500.0, 1.0, 1.0,
                            help="Many neobrokers: €1. Many banks: €25 or €50.")
plan = savings_plan(df, cfg, minimum)
plan_view = pd.DataFrame({
    "Asset class": plan["asset_key"].map(short),
    "Product": plan["name"],
    "ISIN": plan["isin"].replace("", "– (bank account)"),
    "Target weight": plan["weight"].map("{:.1%}".format),
    "Per month": plan["monthly"].map(eur),
    "Execute": [("monthly" if n <= 1 else f"every {n} months") for n in plan["every_months"]],
    "Amount per execution": plan["amount_per_execution"].map(eur),
})
if cfg.plan.start_capital > 0:
    plan_view["One-off start capital"] = plan["start_capital"].map(eur)
st.dataframe(plan_view, hide_index=True, width="stretch")
total = plan["monthly"].sum()
growth_note = {"inflation": " Raise the amounts each year with inflation, as your plan assumes.",
               "fixed": f" Raise the amounts by {cfg.plan.growth_rate:.1%} each year, as your plan assumes."}
st.caption(
    f"Total {eur(total)} per month of {eur(cfg.plan.monthly)} planned"
    + (" (some asset classes have no product selected)" if abs(total - cfg.plan.monthly) > 0.5 else "")
    + ". Cash goes to Tagesgeld by standing order." + growth_note.get(cfg.plan.growth, "")
)
st.markdown(
    "Set these up once as savings plans at your broker. Once a year, the **Annual review** tells you which plans to "
    "raise or lower for a few months, so the mix drifts back to target **without selling**."
)
st.download_button("Download savings plan (CSV)", plan.to_csv(index=False), "savings_plan.csv", "text/csv")

# ----------------------------------------------------------------------------- 4. cost impact
st.subheader("4 · What costs do to your result")
mine = class_costs(df, cfg)
cheap = cheapest_costs(df, cfg)
bank = bank_fund_costs(cfg)
w = cfg.weights
blended = {k: sum(w.get(a, 0) * v.get(a, 0) for a in w) for k, v in
           {"mine": mine, "cheap": cheap, "bank": bank}.items()}
c = st.columns(3)
c[0].metric("Your portfolio cost p.a.", f"{blended['mine']:.3%}")
c[1].metric("Cheapest listed", f"{blended['cheap']:.3%}")
c[2].metric("Typical bank funds", f"{blended['bank']:.3%}")
gold_tax = gold_is_taxable(df)
if gold_tax:
    st.info("Your gold product has no delivery claim, so its gains are taxed when you sell. The cost "
            "comparison models it that way.")

if st.button("Simulate the cost impact", type="primary"):
    sc = scenarios(cfg.model_copy(update={"n_paths": max(cfg.n_paths, 2000)})).subset(2000)
    variants = {
        "Your selection": with_costs(cfg, mine, gold_tax),
        "Cheapest listed": with_costs(cfg, cheap, gold_tax),
        "Typical bank funds": with_costs(cfg, bank, gold_tax),
    }
    bar = st.progress(0.0, "Simulating…")
    table, _ = run_variants(variants, sc, progress=lambda f, m: bar.progress(f, m))
    bar.empty()
    st.session_state["etf_costs_result"] = table

table = st.session_state.get("etf_costs_result")
if table is not None:
    base_med = table.loc["Your selection", "real_wealth_p50"]
    sec, _ = ink()
    fig = go.Figure(go.Bar(
        y=list(table.index), x=table["real_wealth_p50"], orientation="h", marker={"color": cat()[0]},
        text=[eur(v) for v in table["real_wealth_p50"]], textposition="outside", textfont={"color": sec},
        hovertemplate="%{y}: median €%{x:,.0f}<extra></extra>"))
    fig = _layout(fig, "Median net wealth after tax (today's €)", "", "€", height=260)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, table["real_wealth_p50"].max() * 1.15])
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame({
        "Median (today's €)": table["real_wealth_p50"].map(eur),
        "Bad case P5": table["real_wealth_p5"].map(eur),
        "Difference vs. your selection": (table["real_wealth_p50"] - base_med).map(lambda v: f"{v:+,.0f} €"),
        "Real return p.a.": table["irr_real_p50"].map("{:.2%}".format),
    }), width="stretch")
    st.caption("Same market scenarios for all three, so the differences are caused only by costs.")

if st.button("Use my selection's costs in all simulations"):
    set_cfg(with_costs(cfg, mine, gold_tax))
    st.success("Setup updated with your products' costs" + (" and gold taxation" if gold_tax is not None else "")
               + ". All pages now use them.")
