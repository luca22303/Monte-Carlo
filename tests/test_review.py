import numpy as np
import pytest

from mcengine.config import SavingsPlan, SimConfig, StrategySpec
from mcengine.review import months_to_target, run_review, save_snapshot


def test_months_to_target():
    w = np.array([0.6, 0.4])
    assert months_to_target(np.array([60.0, 40.0]), w, 10) == 0
    # equity at 70 of 100: need total 116.67 -> 16.67 of new money
    assert months_to_target(np.array([70.0, 30.0]), w, 1.0) == pytest.approx(70 / 0.6 - 100)


def test_review_flags_breach_and_estimates_tax(tmp_path):
    cfg = SimConfig(n_paths=200, plan=SavingsPlan(monthly=500, horizon_years=20),
                    strategy=StrategySpec(kind="hybrid", abs_band=0.05, rel_band=0.25))
    values = {"equity": 80_000, "gov_bonds": 10_000, "linkers": 5_000, "gold": 3_000, "cash": 2_000}
    basis = {"equity": 50_000, "gov_bonds": 10_000, "linkers": 5_000, "gold": 2_000, "cash": 2_000}
    out, res = run_review(cfg, values, basis, remaining_years=10, allowance_left=0)
    assert out.band_breached
    assert out.trades["equity"] == pytest.approx(-20_000)
    # selling 20k of equity with 37.5 % gain share, 30 % exempt
    assert out.trade_tax_estimate == pytest.approx(20_000 * 0.375 * 0.7 * 0.26375)
    assert sum(out.next_savings_split.values()) == pytest.approx(500)
    assert out.next_savings_split["equity"] == 0
    assert res is not None and res.months == 120
    assert res.wealth[:, 0].mean() > 100_000 * 0.9
    p = save_snapshot(out, cfg, tmp_path)
    assert p.exists()


def test_review_within_band():
    cfg = SimConfig(n_paths=100)
    values = {k: 100_000 * w for k, w in cfg.weights.items()}
    out, _ = run_review(cfg, values, values, remaining_years=5, simulate_forward=False)
    assert not out.band_breached and out.trades == {} and out.months_to_target_by_savings == 0
