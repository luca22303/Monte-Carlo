import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import _layout, cat, eur, fan_chart, get_cfg, kpi_row

from mcengine.broker_import import parse_depot_csv, template_csv
from mcengine.etfs import load_candidates
from mcengine.review import run_review

cfg = get_cfg()
st.title("Annual review")
st.markdown(
    "Once a year (or when a weight leaves its band): enter what you hold today. The review applies your "
    "**fixed policy** to the **current state**. Do not change the return assumptions because markets "
    "just went up or down; that is how investors end up selling low and buying high."
)

assets = cfg.assets
names = {a.key: a.name for a in assets}
kinds = {a.key: a.kind for a in assets}


def current_holdings() -> dict:
    return st.session_state.get("review_holdings") or {
        "Asset": [a.name for a in assets], "key": [a.key for a in assets],
        "Market value €": [round(50_000 * cfg.weights.get(a.key, 0), 2) for a in assets],
        "Cost basis €": [round(45_000 * cfg.weights.get(a.key, 0), 2) for a in assets],
    }


with st.expander("📥 Import holdings from your broker (CSV)", expanded="review_holdings" not in st.session_state):
    st.markdown(
        "Export your **depot overview / positions** as CSV from your broker's website (often under *Depot → "
        "Export* or *Depotübersicht → CSV*) and upload it here. German and English exports are recognised "
        "automatically. No export available? Download the template, fill in one line per product, and upload it."
    )
    c = st.columns([3, 1])
    up = c[0].file_uploader("Depot export (.csv)", type=["csv", "txt"])
    c[1].download_button("Template (CSV)", template_csv(names), "depot_template.csv", "text/csv")
    if up is not None:
        cand = load_candidates()
        if "etf_df" in st.session_state:
            cand = pd.concat([st.session_state["etf_df"], cand])
        isin_map = {i: k for i, k in zip(cand["isin"], cand["asset_key"], strict=True) if i}
        try:
            imported = parse_depot_csv(up.getvalue(), isin_map, kinds)
        except ValueError as e:
            st.error(f"Could not read this file: {e}. If your broker's format is not recognised, use the template.")
        else:
            for w in imported.warnings:
                st.warning(w)
            st.caption("Recognised columns: " + ", ".join(f"{k} → *{v}*" for k, v in imported.columns.items()))
            pos = imported.positions.copy()
            pos["asset_key"] = pos["asset_key"].fillna("")
            edited = st.data_editor(
                pos, hide_index=True, width="stretch", key=f"pos_{up.name}_{up.size}",
                disabled=["isin", "name", "how"],
                column_config={
                    "isin": "ISIN", "name": st.column_config.TextColumn("Product", width="large"),
                    "value": st.column_config.NumberColumn("Market value €", format="%.2f"),
                    "basis": st.column_config.NumberColumn("Cost basis €", format="%.2f"),
                    "asset_key": st.column_config.SelectboxColumn(
                        "Asset class", options=[*names, "ignore"], help="'ignore' leaves a position out"),
                    "how": "Matched by",
                })
            unassigned = edited["asset_key"].isin(["", None]) | edited["asset_key"].isna()
            if unassigned.any():
                st.info(f"Choose an asset class (or 'ignore') for {int(unassigned.sum())} position(s).")
            if st.button("Use these holdings for the review", type="primary", disabled=bool(unassigned.any())):
                keep = edited[edited["asset_key"] != "ignore"]
                tot = keep.groupby("asset_key")[["value", "basis"]].sum()
                old = current_holdings()
                old_val = dict(zip(old["key"], old["Market value €"], strict=True))
                old_bas = dict(zip(old["key"], old["Cost basis €"], strict=True))
                # Classes not in the depot (typically Tagesgeld) keep what was entered by hand
                st.session_state["review_holdings"] = {
                    "Asset": [a.name for a in assets], "key": [a.key for a in assets],
                    "Market value €": [round(float(tot["value"].get(a.key, old_val.get(a.key, 0.0))), 2)
                                       for a in assets],
                    "Cost basis €": [round(float(tot["basis"].get(a.key, old_bas.get(a.key, 0.0))), 2)
                                     for a in assets],
                }
                st.session_state.pop("review_result", None)
                missing = [names[a.key] for a in assets if a.key not in tot.index and cfg.weights.get(a.key, 0) > 0]
                st.session_state["review_import_note"] = (
                    "Imported. Not in the export, so please check by hand: " + ", ".join(missing)
                    if missing else "Imported.")
                st.rerun()

if note := st.session_state.pop("review_import_note", None):
    st.success(note + " Check the table below, then **Run review**.")

default = current_holdings()
with st.form("review"):
    df = st.data_editor(pd.DataFrame(default), hide_index=True, disabled=["Asset", "key"],
                        column_config={"key": None}, width="stretch")
    c = st.columns(3)
    years = c[0].number_input("Years remaining", 1, 60, cfg.plan.horizon_years)
    allowance_left = c[1].number_input("Unused Sparerpauschbetrag this year (€)", 0.0, 2000.0, cfg.tax.allowance)
    sim = c[2].checkbox("Simulate forward from today", True)
    submitted = st.form_submit_button("Run review", type="primary")

if not submitted and "review_result" not in st.session_state:
    st.stop()
if submitted:
    st.session_state["review_holdings"] = df.to_dict(orient="list")
    values = dict(zip(df["key"], df["Market value €"], strict=True))
    basis = dict(zip(df["key"], df["Cost basis €"], strict=True))
    with st.spinner("Reviewing…"):
        st.session_state["review_result"] = run_review(cfg, values, basis, int(years), allowance_left, sim)

out, res = st.session_state["review_result"]

st.subheader(f"Portfolio {eur(out.total)}")
if out.band_breached and cfg.strategy.kind != "cashflow":
    st.warning("⚠ Outside the rebalancing band: trade back to target (see below).")
elif out.band_breached:
    st.warning("⚠ Outside the band. Your policy is cash-flow only, so steer savings and do not sell.")
else:
    st.success("✓ Within the band: no selling needed. Just steer the next savings as shown.")

keys = list(out.weights)
fig = go.Figure()
fig.add_trace(go.Bar(y=[names[k] for k in keys], x=[out.target[k] for k in keys], orientation="h", name="Target",
                     marker={"color": cat()[0]}, hovertemplate="target %{x:.1%}<extra></extra>"))
fig.add_trace(go.Bar(y=[names[k] for k in keys], x=[out.weights[k] for k in keys], orientation="h",
                     name="Current", marker={"color": cat()[1]}, hovertemplate="current %{x:.1%}<extra></extra>"))
fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
fig = _layout(fig, "Current vs. target weights", "", "Weight", height=320)
fig.update_xaxes(tickformat=".0%")
fig.update_yaxes(autorange="reversed")
st.plotly_chart(fig, width="stretch")

c = st.columns(2)
with c[0]:
    st.markdown("**Next month's savings: split**")
    st.dataframe(pd.DataFrame({"Asset": [names[k] for k in keys],
                               "Buy €": [eur(out.next_savings_split[k]) for k in keys]}),
                 hide_index=True, width="stretch")
    m = out.months_to_target_by_savings
    st.caption("Months of savings to reach target without selling: "
               + ("already on target" if m == 0 else "∞ (no savings)" if m == float("inf") else f"{m:.1f}"))
with c[1]:
    if out.trades:
        st.markdown("**Trades back to target** (+ buy / − sell)")
        st.dataframe(pd.DataFrame({"Asset": [names[k] for k in out.trades],
                                   "Trade €": [f"{v:+,.0f}" for v in out.trades.values()]}),
                     hide_index=True, width="stretch")
        st.caption(f"Estimated tax on the sales: {eur(out.trade_tax_estimate)} (average-cost basis, "
                   "gold assumed held > 1 year). If cash-flow steering gets back to target within a few "
                   "months, waiting is usually cheaper than paying this tax.")

if res is not None:
    st.subheader("Outlook from today")
    kpi_row(out.summary, cfg.goal_real)
    st.plotly_chart(fan_chart(res, cfg.goal_real, "Wealth from today (today's €)"), width="stretch")

st.download_button("Download review snapshot (JSON)", out.to_json(), f"review_{out.date}.json",
                   "application/json")
