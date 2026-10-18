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


def _no_uncertainty(cfg: SimConfig) -> SimConfig:
    mm = cfg.market.model_copy(update={"equity_return_uncertainty": 0.0, "gold_return_uncertainty": 0.0,
                                       "inflation_mean_uncertainty": 0.0})
    return cfg.model_copy(update={"market": mm})


def test_parameter_uncertainty_only_shifts_drift():
    cfg = SimConfig(n_paths=300)
    with_u, without = generate(cfg), generate(_no_uncertainty(cfg))
    d = np.log1p(with_u.returns[..., 0].astype(float)) - np.log1p(without.returns[..., 0].astype(float))
    # same shocks: the difference is a constant per-path drift, not noise
    assert np.allclose(d, d[:, :1], atol=2e-5)
    sh = cfg.market.parameter_shifts(300, cfg.seed)
    np.testing.assert_allclose(d[:, 0] * 12, sh["equity"] + sh["inflation"], atol=3e-4)


def test_parameter_uncertainty_widens_outcomes():
    cfg = SimConfig(n_paths=3000)
    path_mean = lambda sc: np.log1p(sc.returns[..., 0].astype(float)).mean(axis=1) * 12  # noqa: E731
    spread_u = path_mean(generate(cfg)).std()
    spread_0 = path_mean(generate(_no_uncertainty(cfg))).std()
    expected = np.sqrt(spread_0**2 + 0.015**2 + 0.005**2)
    assert spread_u > spread_0 and abs(spread_u - expected) < 0.004


def test_bootstrap_parameter_uncertainty():
    cfg = SimConfig(n_paths=500, generator="bootstrap")
    a, b = generate(cfg), generate(_no_uncertainty(cfg))
    d = np.log1p(a.returns[..., 0].astype(float)) - np.log1p(b.returns[..., 0].astype(float))
    assert np.allclose(d, d[:, :1], atol=2e-5) and d[:, 0].std() * 12 > 0.01
