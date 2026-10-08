import numpy as np
from scipy import stats

from mcengine.metrics import calendar_year_returns, cvar, longest_underwater, max_drawdown, var


def test_var_cvar_normal():
    x = np.random.default_rng(0).standard_normal(2_000_000)
    assert abs(var(x, 0.05) - stats.norm.ppf(0.05)) < 0.01
    expected_es = -stats.norm.pdf(stats.norm.ppf(0.05)) / 0.05
    assert abs(cvar(x, 0.05) - expected_es) < 0.01


def test_max_drawdown_and_underwater():
    idx = np.array([[1.1, 1.21, 0.605, 0.9, 1.21, 1.3]])
    np.testing.assert_allclose(max_drawdown(idx), [0.5])
    assert longest_underwater(idx)[0] == 2


def test_calendar_year_returns():
    idx = np.cumprod(np.full((1, 24), 1.01), axis=1)
    np.testing.assert_allclose(calendar_year_returns(idx), 1.01**12 - 1)
