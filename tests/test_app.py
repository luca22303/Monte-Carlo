"""Smoke tests: every Streamlit page renders without exceptions (small config)."""

import sys
from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from mcengine.config import SavingsPlan, SimConfig  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

SMALL = SimConfig(n_paths=1000, plan=SavingsPlan(horizon_years=25)).model_dump_json()


@pytest.mark.parametrize("page", ["setup", "simulation", "strategies", "explorer", "goal", "review",
                                  "assumptions"])
def test_page_renders(page):
    at = AppTest.from_file(str(APP / "views" / f"{page}.py"), default_timeout=240)
    at.session_state["cfg_json"] = SMALL
    at.run()
    assert not at.exception, at.exception


def test_review_runs():
    at = AppTest.from_file(str(APP / "views" / "review.py"), default_timeout=240)
    at.session_state["cfg_json"] = SMALL
    at.run()
    at.button[0].click().run()  # form submit
    assert not at.exception, at.exception
    assert any("Portfolio" in h.value for h in at.subheader)


def test_setup_save_roundtrip():
    at = AppTest.from_file(str(APP / "views" / "setup.py"), default_timeout=120)
    at.session_state["cfg_json"] = SMALL
    at.run()
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert at.success, "expected a success message after saving"
    SimConfig.model_validate_json(at.session_state["cfg_json"])
