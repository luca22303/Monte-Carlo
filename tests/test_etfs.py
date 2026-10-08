import numpy as np
import pytest
from conftest import flat_scenarios

from mcengine.config import AssetSpec, CostSpec, SavingsPlan, SimConfig, StrategySpec, TaxDE
from mcengine.etfs import (
    checklist,
    class_costs,
    gold_is_taxable,
    load_candidates,
    missing_classes,
    savings_plan,
    with_costs,
)
from mcengine.portfolio.simulator import simulate


def test_candidates_cover_every_asset_class_with_one_default():
    cfg = SimConfig()
    df = load_candidates()
    assert set(df["asset_key"]) == set(cfg.asset_keys)
    assert df.groupby("asset_key")["use"].sum().eq(1).all()
    assert missing_classes(df, cfg) == []
    assert df.loc[df["isin"] != "", "isin"].str.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]").all()


def test_savings_plan_split_sums_to_monthly():
    cfg = SimConfig(plan=SavingsPlan(monthly=1000, start_capital=10_000))
    plan = savings_plan(load_candidates(), cfg)
    assert plan["monthly"].sum() == pytest.approx(1000)
    assert plan["start_capital"].sum() == pytest.approx(10_000)
    eq = plan[plan["asset_key"] == "equity"]
    assert eq["monthly"].iloc[0] == pytest.approx(600)


def test_broker_minimum_sets_interval():
    cfg = SimConfig(plan=SavingsPlan(monthly=200))
    plan = savings_plan(load_candidates(), cfg, broker_minimum=25)
    gold = plan[plan["asset_key"] == "gold"].iloc[0]  # 5 % of 200 = 10 €/month
    assert gold["every_months"] == 3 and gold["amount_per_execution"] == pytest.approx(30)


def test_split_inside_class_and_costs():
    cfg = SimConfig()
    df = load_candidates()
    eq = df["asset_key"] == "equity"
    df.loc[eq, "use"] = False
    df.loc[df["isin"].isin(["IE00BK5BQT80", "IE00B4L5Y983"]), "use"] = True  # 0.14 % and 0.20 %
    df.loc[df["isin"] == "IE00BK5BQT80", "share_pct"] = 70
    df.loc[df["isin"] == "IE00B4L5Y983", "share_pct"] = 30
    costs = class_costs(df, cfg)
    assert costs["equity"] == pytest.approx(0.7 * 0.0014 + 0.3 * 0.0020)
    plan = savings_plan(df, cfg)
    assert plan.loc[plan["asset_key"] == "equity", "monthly"].sum() == pytest.approx(600)


def test_checklist_flags_problems():
    cfg = SimConfig()
    df = load_candidates()
    chk = checklist(df, cfg).set_index("name")
    assert chk.loc["iShares Physical Gold ETC", "verdict"] == "warn"           # no delivery claim
    assert chk.loc["iShares Core Euro Government Bond UCITS ETF (Dist)", "verdict"] == "warn"  # distributing
    assert chk.loc["Vanguard FTSE All-World UCITS ETF (USD) Acc", "verdict"] == "good"
    df.loc[0, "savings_plan"] = False
    assert checklist(df.iloc[[0]], cfg)["verdict"].iloc[0] == "bad"


def test_gold_taxability_and_with_costs():
    cfg = SimConfig()
    df = load_candidates()
    assert gold_is_taxable(df) is False
    g = df["asset_key"] == "gold"
    df.loc[g, "use"] = df.loc[g, "isin"] == "IE00B4ND3602"
    assert gold_is_taxable(df) is True
    new = with_costs(cfg, class_costs(df, cfg), gold_taxable=True)
    gold = next(a for a in new.assets if a.key == "gold")
    assert gold.tax_class == "etc_taxable" and gold.ter == pytest.approx(0.0012)


def test_etc_taxable_gold_is_taxed_on_sale_even_after_one_year():
    gold = AssetSpec(key="gold", name="Gold ETC", kind="gold", ter=0.0, tax_class="etc_taxable")
    cfg = SimConfig(assets=[gold], weights={"gold": 1.0}, strategy=StrategySpec(kind="buy_and_hold"),
                    plan=SavingsPlan(monthly=0, start_capital=100_000, horizon_years=3, growth="none"),
                    costs=CostSpec(trade_bps=0), tax=TaxDE(), n_paths=100)
    res = simulate(cfg, flat_scenarios(cfg, 0.01, n_paths=2))
    gain = 100_000 * (1.01**36 - 1)
    np.testing.assert_allclose(res.taxes.sum(axis=1), (gain - 1000) * 0.26375, rtol=1e-6)
    assert res.taxes[:, :-1].sum() == 0  # no Vorabpauschale, nothing before the sale
