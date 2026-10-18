import numpy as np
import pytest

from mcengine.compare import paired_difference
from mcengine.config import SavingsPlan, SimConfig
from mcengine.portfolio.simulator import simulate
from mcengine.scenarios import generate
from mcengine.stress import CRISES, crisis_summary, crisis_window, insert_crisis, start_month


@pytest.fixture(scope="module")
def setup():
    cfg = SimConfig(n_paths=300, plan=SavingsPlan(horizon_years=20))
    return cfg, generate(cfg)


def test_all_crises_have_data(setup):
    cfg, _ = setup
    for key, c in CRISES.items():
        win = crisis_window(cfg, key)
        months = (np.datetime64(c.end, "M") - np.datetime64(c.start, "M")).astype(int) + 1
        assert len(win["returns"]) == months
        assert np.isfinite(win["returns"]).all()


def test_gfc_equity_fell_bonds_rose(setup):
    cfg, _ = setup
    s = crisis_summary(cfg, "gfc").set_index("")["Crisis"]
    assert s.iloc[0] < -0.35 and s.iloc[1] > 0.05  # equity down, Bunds up
    s22 = crisis_summary(cfg, "inflation_2022", 24).set_index("")["Crisis"]
    assert s22.iloc[0] < 0 and s22.iloc[1] < -0.10  # 2022: both down


def test_insert_overwrites_exact_months(setup):
    cfg, sc = setup
    start = start_month(cfg, "gfc", 5)
    assert start == 48
    st = insert_crisis(sc, cfg, "gfc", start)
    n = len(crisis_window(cfg, "gfc")["returns"])
    win = crisis_window(cfg, "gfc")["returns"]
    expected = np.broadcast_to(win, (sc.n_paths, n, 5))
    np.testing.assert_allclose(st.returns[:, start:start + n], expected, atol=1e-6)
    np.testing.assert_array_equal(st.returns[:, :start], sc.returns[:, :start])
    np.testing.assert_array_equal(st.returns[:, start + n:], sc.returns[:, start + n:])
    assert start_month(cfg, "gfc", "end") == cfg.months - n
    with_rec = len(crisis_window(cfg, "gfc", 24)["returns"])
    # 'end': the crash itself ends with the plan, recovery months fall after it
    assert with_rec == n + 24 and start_month(cfg, "gfc", "end", 24) == cfg.months - n
    st_rec = insert_crisis(sc, cfg, "gfc", cfg.months - n, 24)
    np.testing.assert_allclose(st_rec.returns[:, -n:], expected, atol=1e-6)


def test_crash_before_goal_hurts_more_than_early(setup):
    cfg, sc = setup
    base = simulate(cfg, sc)
    early = simulate(cfg, insert_crisis(sc, cfg, "dotcom", start_month(cfg, "dotcom", "start")))
    late = simulate(cfg, insert_crisis(sc, cfg, "dotcom", start_month(cfg, "dotcom", "end")))
    assert paired_difference(late, base)["mean"] < paired_difference(early, base)["mean"] < 0


def test_crisis_beyond_plan_raises(setup):
    cfg, sc = setup
    with pytest.raises(ValueError):
        insert_crisis(sc, cfg, "gfc", cfg.months + 5)
