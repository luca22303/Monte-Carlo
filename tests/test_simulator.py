import numpy as np
import pytest
from conftest import flat_scenarios

from mcengine.config import StrategySpec
from mcengine.metrics import money_weighted_irr, summarize
from mcengine.portfolio.simulator import simulate


def annuity_due_fv(c: float, r: float, n: int) -> float:
    return c * ((1 + r) ** n - 1) / r * (1 + r)


@pytest.mark.parametrize("kind", ["buy_and_hold", "calendar", "band", "cashflow", "hybrid"])
def test_constant_returns_match_annuity(frictionless, kind):
    cfg = frictionless.model_copy(update={"strategy": StrategySpec(kind=kind)})
    r = 0.004
    res = simulate(cfg, flat_scenarios(cfg, r))
    expected = annuity_due_fv(1000, r, cfg.months)
    np.testing.assert_allclose(res.wealth[:, -1], expected, rtol=1e-9)
    np.testing.assert_allclose(res.final_net, expected, rtol=1e-9)
    np.testing.assert_allclose(res.nav[:, -1], (1 + r) ** cfg.months, rtol=1e-6)


def test_ter_reduces_return(frictionless):
    cfg = frictionless.model_copy(deep=True)
    for a in cfg.assets:
        a.ter = 0.012
    res = simulate(cfg, flat_scenarios(cfg, 0.004))
    np.testing.assert_allclose(res.wealth[:, -1], annuity_due_fv(1000, 0.003, cfg.months), rtol=1e-9)


def test_calendar_rebalance_restores_target(frictionless):
    cfg = frictionless.model_copy(update={"strategy": StrategySpec(kind="calendar", rebalance_months=1)})
    rets = np.array([0.02, -0.01, 0.0, 0.01, 0.001])
    res = simulate(cfg, flat_scenarios(cfg, rets))
    # Final month is not rebalanced: undo its drift and compare with target weights
    w_end = res.weights_end / (1 + rets)
    w_end /= w_end.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(w_end, np.broadcast_to(cfg.weight_vector(), w_end.shape), atol=1e-9)
    assert (res.rebalances == cfg.months - 1).all()


def test_cashflow_never_sells(frictionless):
    cfg = frictionless.model_copy(update={"strategy": StrategySpec(kind="cashflow")})
    res = simulate(cfg, flat_scenarios(cfg, np.array([0.03, -0.02, 0.0, 0.01, 0.0])))
    assert (res.turnover == 0).all() and (res.rebalances == 0).all()


def test_cashflow_tracks_target_better_than_buy_and_hold(frictionless):
    rets = np.array([0.012, 0.001, 0.001, 0.002, 0.001])
    out = {}
    for kind in ("buy_and_hold", "cashflow"):
        cfg = frictionless.model_copy(update={"strategy": StrategySpec(kind=kind)})
        out[kind] = np.abs(simulate(cfg, flat_scenarios(cfg, rets)).weights_end - cfg.weight_vector()).sum()
    assert out["cashflow"] < out["buy_and_hold"]


def test_band_triggers_only_outside_band(frictionless):
    cfg = frictionless.model_copy(update={"strategy": StrategySpec(kind="band", abs_band=0.05, rel_band=0.0)})
    calm = simulate(cfg, flat_scenarios(cfg, 0.003))
    assert (calm.rebalances == 0).all()
    wild = simulate(cfg, flat_scenarios(cfg, np.array([0.03, -0.01, -0.01, 0.0, 0.0])))
    assert (wild.rebalances > 0).all()


def test_inflation_indexed_savings(frictionless):
    plan = frictionless.plan.model_copy(update={"growth": "inflation"})
    cfg = frictionless.model_copy(update={"plan": plan})
    infl = 0.002
    res = simulate(cfg, flat_scenarios(cfg, 0.0, inflation=infl))
    np.testing.assert_allclose(res.contributions[0, 12], 1000 * (1 + infl) ** 12, rtol=1e-6)
    s = summarize(res)
    assert s["real_contributions"] == pytest.approx(1000 * cfg.months, rel=1e-6)


def test_irr_constant_path():
    t, r = 240, 0.005
    flows = np.full((3, t), 100.0)
    terminal = np.full(3, annuity_due_fv(100.0, r, t))
    np.testing.assert_allclose(money_weighted_irr(flows, terminal), (1 + r) ** 12 - 1, atol=1e-9)


def test_irr_negative_return():
    t, r = 120, -0.002
    flows = np.full((2, t), 50.0)
    terminal = np.full(2, annuity_due_fv(50.0, r, t))
    np.testing.assert_allclose(money_weighted_irr(flows, terminal), (1 + r) ** 12 - 1, atol=1e-9)
