import numpy as np
import pytest

from mcengine.config import SimConfig
from mcengine.data.panel import historical_returns, load_panel
from mcengine.scenarios import generate
from mcengine.scenarios.bootstrap import stationary_bootstrap_indices
from mcengine.scenarios.historical import rolling_windows


def test_panel_is_contiguous_and_complete():
    df = load_panel()
    assert not df.isna().any().any()
    assert (df.index.to_series().diff().dropna().dt.days.between(28, 31)).all()
    assert df.index[0].year <= 1973 and len(df) > 600


def test_historical_returns_plausible():
    h = historical_returns(SimConfig().assets)
    assert np.isfinite(h.returns).all()
    geo = np.expm1(np.log1p(h.returns).mean(axis=0) * 12)
    assert 0.06 < geo[0] < 0.13        # equity in EUR
    assert 0.03 < geo[1] < 0.08        # Bunds
    assert 0.0 < geo[4] < 0.06         # cash


def test_bootstrap_block_structure():
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(500, 2000, 240, 24.0, rng)
    cont = (np.diff(idx, axis=1) % 500 == 1).mean()
    assert cont == pytest.approx(1 - 1 / 24, abs=0.01)


def test_bootstrap_mean_adjust_hits_targets():
    cfg = SimConfig(n_paths=3000, generator="bootstrap")
    sc = generate(cfg)
    geo = np.expm1(np.log1p(sc.returns.astype(float)).mean(axis=(0, 1)) * 12)
    assert abs(geo[0] - cfg.market.equity_return) < 0.01
    raw = generate(cfg.model_copy(update={"bootstrap_mean_adjust": False}))
    raw_geo = np.expm1(np.log1p(raw.returns.astype(float)).mean(axis=(0, 1)) * 12)
    assert raw_geo[0] > geo[0]  # history (1973-2025) beat today's forward-looking assumption


def test_rolling_windows():
    cfg = SimConfig()
    sc, starts = rolling_windows(cfg)
    assert sc.n_months == cfg.months and len(starts) == sc.n_paths >= 20
    assert all(d.month == 1 for d in starts)
