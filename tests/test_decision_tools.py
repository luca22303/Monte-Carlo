import numpy as np
import pytest

from mcengine.compare import compare_strategies, paired_difference
from mcengine.config import SavingsPlan, SimConfig
from mcengine.optimize import defensive_grid, explore_allocations, pareto_mask, required_savings
from mcengine.scenarios import generate


@pytest.fixture(scope="module")
def small():
    cfg = SimConfig(n_paths=400, plan=SavingsPlan(horizon_years=15))
    return cfg, generate(cfg)


def test_compare_uses_common_scenarios(small):
    cfg, sc = small
    df, res = compare_strategies(cfg, sc=sc)
    assert len(df) == 5
    # identical cash outlay for every strategy
    paid = {k: r.contributions.sum() for k, r in res.items()}
    assert np.allclose(list(paid.values()), next(iter(paid.values())))
    a, b = list(res.values())[:2]
    d = paired_difference(a, b)
    assert np.isfinite(d["mean"]) and d["se"] >= 0


def test_defensive_grid_shapes():
    cfg = SimConfig()
    grid = defensive_grid(cfg, equity_levels=(0.6,), step=0.5)
    assert len(grid) == 10  # 2 units over 4 defensive assets: C(5, 3)
    for w in grid:
        assert sum(w.values()) == pytest.approx(1.0) and w["equity"] == 0.6
    capped = defensive_grid(cfg, step=0.25, max_weight={"gold": 0.1})
    assert all(w["gold"] <= 0.1 + 1e-9 for w in capped)


def test_pareto_mask():
    risk = np.array([1.0, 2.0, 3.0, 2.5])
    reward = np.array([1.0, 3.0, 2.0, 4.0])
    assert pareto_mask(risk, reward).tolist() == [True, True, False, True]


def test_explore_allocations(small):
    cfg, sc = small
    grid = defensive_grid(cfg, equity_levels=(0.6,), step=0.5)
    df = explore_allocations(cfg, grid, sc.subset(150))
    assert len(df) == len(grid) and df["pareto"].any()


def test_required_savings_hits_goal(small):
    cfg, sc = small
    out = required_savings(cfg, goal_real=200_000, confidence=0.8, sc=sc)
    assert out["quantile_wealth"] == pytest.approx(200_000, rel=0.01)
    assert 200 < out["monthly"] < 2000
