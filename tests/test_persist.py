"""Auto-save: what the user enters survives a new session (= page reload / app restart)."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from mcengine.config import SavingsPlan, SimConfig  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))
import persist  # noqa: E402

SMALL = SimConfig(n_paths=1000, plan=SavingsPlan(horizon_years=20, monthly=777))


def page(name: str, timeout: int = 300) -> AppTest:
    return AppTest.from_file(str(APP / "views" / f"{name}.py"), default_timeout=timeout)


def test_setup_is_saved_and_reloaded():
    persist.save("config.json", json.loads(SMALL.model_dump_json()))
    at = page("setup")
    at.run()  # fresh session: nothing in session_state, must come from disk
    assert not at.exception, at.exception
    assert SimConfig.model_validate_json(at.session_state["cfg_json"]).plan.monthly == 777
    at.button[0].click().run()  # "Apply to all pages" writes the file again
    assert persist.load("config.json")["plan"]["monthly"] == 777


def test_corrupt_saved_setup_falls_back_to_defaults():
    persist.data_dir().mkdir(parents=True, exist_ok=True)
    (persist.data_dir() / "config.json").write_text('{"plan": {"monthly": -5}}')
    at = page("setup")
    at.run()
    assert not at.exception, at.exception
    loaded = SimConfig.model_validate_json(at.session_state["cfg_json"])
    assert loaded.plan.monthly == SimConfig().plan.monthly
    assert any("could not be read" in w.value for w in at.warning)


def test_etf_selection_is_saved_and_reloaded():
    persist.save("config.json", json.loads(SMALL.model_dump_json()))
    at = page("etfs")
    at.run()
    df = at.session_state["etf_df"]
    df.loc[df["isin"] == "IE00BK5BQT80", "use"] = False
    df.loc[df["isin"] == "IE00BJ0KDQ92", "use"] = True
    persist.save("etf_selection.json", json.loads(df.to_json(orient="records")))
    again = page("etfs")
    again.run()
    assert not again.exception, again.exception
    loaded = again.session_state["etf_df"]
    used = set(loaded.loc[loaded["use"].astype(bool), "isin"])
    assert "IE00BJ0KDQ92" in used and "IE00BK5BQT80" not in used


def test_review_holdings_and_history_are_saved():
    persist.save("config.json", json.loads(SMALL.model_dump_json()))
    at = page("review")
    at.run()
    run_btn = next(b for b in at.button if b.label == "Run review")
    run_btn.click().run()
    assert not at.exception, at.exception
    holdings = persist.load("review_holdings.json")
    assert holdings and "Market value €" in holdings
    assert len(persist.list_reviews()) == 1
    again = page("review")
    again.run()  # new session: holdings come back from disk, history is listed
    assert not again.exception, again.exception
    assert again.session_state["review_holdings"] == holdings
    assert any("Review history" in e.label for e in again.expander)


def test_reset_keeps_history_unless_asked():
    persist.save("config.json", {"x": 1})
    persist.save("etf_selection.json", [])
    persist.save_review_snapshot({"review": {}})
    persist.reset(keep_reviews=True)
    assert persist.saved_files() == [f"reviews/{persist.list_reviews()[0].name}"]
    persist.reset(keep_reviews=False)
    assert persist.saved_files() == []
