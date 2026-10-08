import numpy as np

from mcengine.config import MarketModel, SimConfig, default_regimes
from mcengine.scenarios import generate


def _single_regime_normal() -> MarketModel:
    rg = default_regimes()[0]
    rg.df = 1000.0
    return MarketModel(regimes=[rg], transition=[[1.0]])


def test_seed_reproducible():
    cfg = SimConfig(n_paths=200)
    a, b = generate(cfg), generate(cfg)
    np.testing.assert_array_equal(a.returns, b.returns)
    c = generate(cfg.model_copy(update={"seed": 7}))
    assert not np.array_equal(a.returns, c.returns)


def test_long_run_means_match_assumptions():
    cfg = SimConfig(n_paths=4000)
    sc = generate(cfg)
    lr = np.log1p(sc.returns.astype(float)).mean(axis=(0, 1)) * 12
    geo = np.expm1(lr)
    assert abs(geo[0] - cfg.market.equity_return) < 0.004
    assert abs(geo[3] - cfg.market.gold_return) < 0.004
    infl = np.expm1(np.log1p(sc.inflation).mean() * 12)
    assert abs(infl - cfg.market.inflation_mean) < 0.003
    # Linkers earn the real yield plus inflation
    assert abs(geo[2] - (cfg.market.real_yield_mean + cfg.market.inflation_mean)) < 0.006


def test_regime_frequencies_match_stationary_distribution():
    cfg = SimConfig(n_paths=3000)
    sc = generate(cfg)
    freq = np.bincount(sc.regimes.ravel(), minlength=3) / sc.regimes.size
    np.testing.assert_allclose(freq, cfg.market.stationary_distribution(), atol=0.01)


def test_single_regime_reproduces_vol_and_correlation():
    mm = _single_regime_normal()
    cfg = SimConfig(n_paths=4000, market=mm)
    sc = generate(cfg)
    r = np.log1p(sc.returns.astype(float))
    vol = r[..., 0].std() * np.sqrt(12)
    assert abs(vol - mm.regimes[0].vol["equity"]) < 0.003
    eq_gold = np.corrcoef(r[..., 0].ravel(), r[..., 3].ravel())[0, 1]
    assert abs(eq_gold - mm.regimes[0].corr[0][1]) < 0.02


def test_crisis_regime_has_fat_left_tail():
    cfg = SimConfig(n_paths=3000)
    sc = generate(cfg)
    eq = sc.returns[..., 0].astype(float).ravel()
    z = (eq - eq.mean()) / eq.std()
    assert (z < -3).mean() > 3 * 0.00135  # far more 3-sigma drops than a normal distribution
