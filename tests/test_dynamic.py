import numpy as np
import pytest
from conftest import flat_scenarios

from mcengine.config import Behavior, GlidePath, Reevaluation, SavingsPlan, SimConfig, StrategySpec, TaxDE
from mcengine.portfolio.dynamic import GoalEstimator, glide_equity, panic_weights, scale_equity
from mcengine.portfolio.simulator import simulate
from mcengine.scenarios import generate

IS_EQ = np.array([True, False, False, False, False])
W = np.array([0.6, 0.2, 0.1, 0.05, 0.05])


def test_scale_equity_keeps_defensive_mix():
    out = scale_equity(W, IS_EQ, 0.4)
    np.testing.assert_allclose(out, [0.4, 0.3, 0.15, 0.075, 0.075])
    many = scale_equity(np.tile(W, (3, 1)), IS_EQ, np.array([0.6, 0.5, 0.0]))
    np.testing.assert_allclose(many.sum(axis=1), 1.0)
    np.testing.assert_allclose(many[:, 0], [0.6, 0.5, 0.0])


def test_panic_weights_move_equity_to_cash():
    np.testing.assert_allclose(panic_weights(W, IS_EQ, 4), [0, 0.2, 0.1, 0.05, 0.65])


def test_glide_equity_schedule():
    glide = GlidePath(enabled=True, years=10, end_equity=0.4)
    cfg = SimConfig(plan=SavingsPlan(horizon_years=30), glide=glide)
    t_total = 360
    assert glide_equity(cfg, 0, t_total) == pytest.approx(0.6)
    assert glide_equity(cfg, 239, t_total) == pytest.approx(0.6)
    assert glide_equity(cfg, 359, t_total) == pytest.approx(0.4)
    assert 0.4 < glide_equity(cfg, 300, t_total) < 0.6


def test_glide_path_reaches_end_allocation(frictionless):
    cfg = frictionless.model_copy(update={
        "strategy": StrategySpec(kind="calendar", rebalance_months=1),
        "glide": GlidePath(enabled=True, years=5, end_equity=0.3)})
    res = simulate(cfg, flat_scenarios(cfg, 0.003))
    assert res.equity_share[0, 0] == pytest.approx(0.6, abs=1e-6)
    assert res.equity_share[0, -2] == pytest.approx(0.3, abs=0.01)


def test_goal_estimator_close_to_simulation():
    cfg = SimConfig(n_paths=3000, tax=TaxDE(enabled=False), plan=SavingsPlan(horizon_years=20),
                    goal_real=300_000)
    sc = generate(cfg)
    sim = (simulate(cfg, sc).real_final_net() >= cfg.goal_real).mean()
    est = GoalEstimator(sc).probability(np.array([0.0]), np.array([1000.0]), cfg.months, cfg.weight_vector(),
                                        cfg.goal_real)[0]
    assert abs(est - sim) < 0.10
    p = GoalEstimator(sc).probability(np.array([0.0, 50_000, 200_000]), np.full(3, 1000.0), 120,
                                      cfg.weight_vector(), cfg.goal_real)
    assert (np.diff(p) > 0).all()


def test_reevaluation_switches_to_defensive_when_goal_is_safe(frictionless):
    base = frictionless.model_copy(update={"goal_real": 50_000, "strategy": StrategySpec(kind="cashflow")})
    sc = flat_scenarios(base, 0.004)
    off = simulate(base, sc)
    on_cfg = base.model_copy(update={"reevaluation": Reevaluation(enabled=True, every_months=6,
                                                                  defensive_equity=0.3)})
    on = simulate(on_cfg, sc)
    assert off.switches.sum() == 0 and (on.switches >= 1).all() and (on.defensive_months > 0).all()
    assert on.equity_share[0, -1] == pytest.approx(0.3, abs=0.02)
    # no goal -> switch disabled
    assert simulate(on_cfg.model_copy(update={"goal_real": 0.0}), sc).switches.sum() == 0


def test_reevaluation_on_random_scenarios_raises_goal_probability():
    cfg = SimConfig(n_paths=1500, plan=SavingsPlan(horizon_years=20), goal_real=300_000)
    sc = generate(cfg)
    base = simulate(cfg, sc).real_final_net()
    dyn = simulate(cfg.model_copy(update={"reevaluation": Reevaluation(enabled=True)}), sc).real_final_net()
    assert (dyn >= cfg.goal_real).mean() >= (base >= cfg.goal_real).mean() - 0.01
    assert np.percentile(dyn, 95) < np.percentile(base, 95)  # upside is given up


def _crash_and_recover(cfg, n_paths=2):
    sc = flat_scenarios(cfg, 0.0, n_paths=n_paths)
    r = sc.returns
    r[:, :, 0] = 0.005
    r[:, 24:30, 0] = -0.15  # ~ -62 % equity crash over 6 months
    r[:, 30:42, 0] = 0.08   # sharp rebound
    return sc


def test_panic_sells_and_buys_back(frictionless):
    cfg = frictionless.model_copy(update={"behavior": Behavior(panic=True, drawdown=0.25, months_out=12)})
    sc = _crash_and_recover(cfg)
    calm = simulate(frictionless, sc)
    res = simulate(cfg, sc)
    assert (res.panic_sales >= 1).all()
    assert (res.months_out >= 12).all()
    out_month = int(np.argmax(res.equity_share[0] < 0.01))
    assert 24 <= out_month < 30 and res.equity_share[0, out_month + 13] > 0.5  # bought back
    assert (res.final_net < calm.final_net).all()  # missed the rebound
