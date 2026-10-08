import json

import pandas as pd
import streamlit as st
from common import get_cfg, set_cfg

from mcengine.config import CostSpec, SavingsPlan, SimConfig, StrategySpec, TaxDE
from mcengine.review import latest_market_state

cfg = get_cfg()
st.title("Setup")
st.caption("Everything below is saved for this browser session and used by every page. "
           "Amounts are in today's euros.")

with st.form("setup"):
    st.subheader("Savings plan")
    c = st.columns(4)
    monthly = c[0].number_input("Monthly savings (€)", 0.0, 100_000.0, cfg.plan.monthly, 50.0)
    start_capital = c[1].number_input("Start capital (€)", 0.0, 10_000_000.0, cfg.plan.start_capital, 1000.0)
    horizon = c[2].slider("Horizon (years)", 5, 50, cfg.plan.horizon_years)
    goal = c[3].number_input("Wealth goal (today's €)", 0.0, 50_000_000.0, cfg.goal_real, 10_000.0)
    growth_opts = {"inflation": "Grows with inflation (constant real savings)",
                   "fixed": "Fixed step-up per year", "none": "Constant in nominal €"}
    c = st.columns([2, 1])
    growth = c[0].radio("Savings growth", list(growth_opts), format_func=growth_opts.get,
                        index=list(growth_opts).index(cfg.plan.growth), horizontal=True)
    growth_rate = c[1].number_input("Step-up p.a. (if fixed)", 0.0, 0.2, cfg.plan.growth_rate, 0.005,
                                    format="%.3f")

    st.subheader("Allocation")
    eq_key = next(a.key for a in cfg.assets if a.kind == "equity")
    def_assets = [a for a in cfg.assets if a.kind != "equity"]
    eq = st.slider("Equity share", 0.30, 0.80, float(cfg.weights.get(eq_key, 0.6)), 0.05,
                   help="Your target band is 50–60 %. Every 10 points of equity add roughly 0.3–0.4 % "
                        "expected return and 5–7 points of bad-case drawdown.")
    st.markdown("**Defensive sleeve** – split of the remaining "
                f"{1 - eq:.0%} (normalised to 100 %)")
    sleeve = sum(cfg.weights.get(a.key, 0) for a in def_assets) or 1.0
    cols = st.columns(len(def_assets))
    raw = {}
    for col, a in zip(cols, def_assets, strict=True):
        raw[a.key] = col.number_input(a.name, 0.0, 100.0, round(100 * cfg.weights.get(a.key, 0) / sleeve, 1),
                                      5.0, key=f"w_{a.key}")

    st.subheader("Rebalancing strategy")
    kinds = {"hybrid": "Hybrid – steer savings, sell only outside wide bands (recommended)",
             "cashflow": "Cash-flow only – steer savings, never sell",
             "band": "Band – full rebalance when a weight leaves its band",
             "calendar": "Calendar – full rebalance every N months",
             "buy_and_hold": "Buy & hold – never rebalance"}
    c = st.columns([3, 1, 1, 1])
    kind = c[0].selectbox("Policy", list(kinds), format_func=kinds.get, index=list(kinds).index(cfg.strategy.kind))
    reb_m = c[1].number_input("Calendar: months", 1, 120, cfg.strategy.rebalance_months)
    abs_band = c[2].number_input("Band: absolute", 0.0, 0.5, cfg.strategy.abs_band, 0.01, format="%.2f")
    rel_band = c[3].number_input("Band: relative", 0.0, 2.0, cfg.strategy.rel_band, 0.05, format="%.2f",
                                 help="Band = max(absolute, relative × target weight).")

    st.subheader("Taxes (Germany) & costs")
    c = st.columns(4)
    tax_on = c[0].checkbox("Model German taxes", cfg.tax.enabled)
    joint = c[1].checkbox("Joint filing (€2,000 allowance)", cfg.tax.allowance >= 2000)
    church_opts = {0.0: "No church tax", 0.08: "Church tax 8 % (BY/BW)", 0.09: "Church tax 9 %"}
    church = c[2].selectbox("Kirchensteuer", list(church_opts), format_func=church_opts.get,
                            index=list(church_opts).index(cfg.tax.church_tax) if cfg.tax.church_tax in church_opts
                            else 0)
    trade_bps = c[3].number_input("Trading cost (bp per trade)", 0.0, 200.0, cfg.costs.trade_bps, 1.0)
    c = st.columns(3)
    bz_mode = c[0].radio("Basiszins (Vorabpauschale)", ["simulated", "fixed"],
                         index=0 if cfg.tax.basiszins is None else 1, horizontal=True,
                         help="Simulated: follows the simulated long yield each January.")
    bz_val = c[1].number_input("Fixed Basiszins", 0.0, 0.1, cfg.tax.basiszins or 0.0253, 0.001, format="%.4f")
    personal = c[2].number_input("Personal income tax rate (gold < 1 year)", 0.0, 0.5, cfg.tax.personal_rate,
                                 0.01, format="%.2f")
    ter_df = pd.DataFrame({"Asset": [a.name for a in cfg.assets], "TER % p.a.": [a.ter * 100 for a in cfg.assets]})
    with st.expander("Fund costs (TER) per asset"):
        ter_df = st.data_editor(ter_df, disabled=["Asset"], hide_index=True, use_container_width=True)

    st.subheader("Simulation engine")
    c = st.columns(4)
    gen = c[0].radio("Return generator", ["parametric", "bootstrap"], index=["parametric", "bootstrap"].index(
        cfg.generator), help="Parametric: regime-switching fat-tailed model anchored on today's yields. "
                             "Bootstrap: resampled 1973–2025 history (shape of risk from real markets).")
    n_paths = c[1].select_slider("Paths", [1000, 2000, 5000, 10000, 20000], cfg.n_paths)
    seed = c[2].number_input("Random seed", 0, 1_000_000, cfg.seed)
    block = c[3].number_input("Bootstrap block (months)", 1.0, 120.0, cfg.bootstrap_block_months, 1.0)
    mean_adj = st.checkbox("Bootstrap: shift historical means to today's assumptions", cfg.bootstrap_mean_adjust,
                           help="1973–2025 had much higher yields than today. Unadjusted bootstrap = "
                                "'history repeats', which is optimistic for bonds and cash.")

    with st.expander("Capital market assumptions (forward-looking, nominal EUR)"):
        mm = cfg.market
        st.caption("Change these only for a reason (new starting yields, a deliberate scenario) – "
                   "never because markets just went up or down.")
        c = st.columns(3)
        eq_ret = c[0].number_input("Equity return (geometric, gross)", -0.05, 0.15, mm.equity_return, 0.0025,
                                   format="%.4f")
        gold_ret = c[1].number_input("Gold return (geometric)", -0.05, 0.15, mm.gold_return, 0.0025, format="%.4f")
        cash_floor = c[2].number_input("Cash rate floor", -0.02, 0.05, mm.cash_floor, 0.0025, format="%.4f")
        st.markdown("Start → long-run mean")
        c = st.columns(4)
        y0 = c[0].number_input("Nominal govt yield, start", -0.02, 0.15, mm.nominal_yield_start, 0.001, format="%.4f")
        ym = c[0].number_input("… long-run", -0.02, 0.15, mm.nominal_yield_mean, 0.001, format="%.4f")
        r0 = c[1].number_input("Real yield (linkers), start", -0.03, 0.1, mm.real_yield_start, 0.001, format="%.4f")
        rm = c[1].number_input("… long-run ", -0.03, 0.1, mm.real_yield_mean, 0.001, format="%.4f")
        s0 = c[2].number_input("Short rate, start", -0.02, 0.15, mm.short_rate_start, 0.001, format="%.4f")
        sm = c[2].number_input("… long-run  ", -0.02, 0.15, mm.short_rate_mean, 0.001, format="%.4f")
        i0 = c[3].number_input("Inflation, start", -0.02, 0.2, mm.inflation_start, 0.001, format="%.4f")
        im = c[3].number_input("… long-run   ", -0.02, 0.1, mm.inflation_mean, 0.001, format="%.4f")
        try:
            ms = latest_market_state()
            st.info(f"Latest data ({ms['as_of']}): Bund 10y {ms['bund_10y']:.2%}, 3m money market "
                    f"{ms['money_market_3m']:.2%}, CPI inflation (12m) {ms['inflation_12m']:.2%}. "
                    "A EUR government index yields a little more than Bunds (spreads).")
        except FileNotFoundError:
            pass

    submitted = st.form_submit_button("Save settings", type="primary")

if submitted:
    try:
        total_raw = sum(raw.values())
        if total_raw <= 0:
            raise ValueError("defensive sleeve weights are all zero")
        weights = {eq_key: eq} | {k: (1 - eq) * v / total_raw for k, v in raw.items()}
        assets = [a.model_copy(update={"ter": float(t) / 100}) for a, t in
                  zip(cfg.assets, ter_df["TER % p.a."], strict=True)]
        market = cfg.market.model_copy(update=dict(
            equity_return=eq_ret, gold_return=gold_ret, cash_floor=cash_floor,
            nominal_yield_start=y0, nominal_yield_mean=ym, real_yield_start=r0, real_yield_mean=rm,
            short_rate_start=s0, short_rate_mean=sm, inflation_start=i0, inflation_mean=im))
        new = SimConfig(
            assets=assets, weights=weights, market=market,
            strategy=StrategySpec(kind=kind, rebalance_months=reb_m, abs_band=abs_band, rel_band=rel_band),
            plan=SavingsPlan(monthly=monthly, start_capital=start_capital, horizon_years=horizon,
                             growth=growth, growth_rate=growth_rate),
            costs=CostSpec(trade_bps=trade_bps),
            tax=TaxDE(enabled=tax_on, church_tax=church, allowance=2000.0 if joint else 1000.0,
                      basiszins=None if bz_mode == "simulated" else bz_val, personal_rate=personal,
                      gold_exemption_limit=cfg.tax.gold_exemption_limit),
            generator=gen, bootstrap_block_months=block, bootstrap_mean_adjust=mean_adj,
            n_paths=n_paths, seed=seed, goal_real=goal,
        )
        set_cfg(new)
        cfg = new
        st.success("Saved. Open **Simulation** to see the results.")
    except Exception as e:  # validation errors are shown to the user
        st.error(f"Invalid settings: {e}")

st.subheader("Current target allocation")
st.dataframe(pd.DataFrame({"Asset": [a.name for a in cfg.assets],
                           "Weight": [f"{cfg.weights.get(a.key, 0):.1%}" for a in cfg.assets],
                           "Tax class": [a.tax_class for a in cfg.assets],
                           "TER": [f"{a.ter:.2%}" for a in cfg.assets]}),
             hide_index=True, use_container_width=True)

st.subheader("Save / load configuration")
c = st.columns(2)
c[0].download_button("Download config (JSON)", cfg.model_dump_json(indent=2), "mc_config.json", "application/json")
up = c[1].file_uploader("Load config (JSON)", type="json")
if up is not None and st.session_state.get("_loaded_name") != up.name + str(up.size):
    try:
        set_cfg(SimConfig.model_validate(json.loads(up.read())))
        st.session_state["_loaded_name"] = up.name + str(up.size)
        st.rerun()
    except Exception as e:
        st.error(f"Could not load config: {e}")
