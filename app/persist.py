"""Automatic saving of what the user enters, in a local folder next to the app.

Files live in ``my_data/`` in the repository folder (git-ignored, never uploaded). Set the
environment variable ``MC_DATA_DIR`` to use another folder; the tests do this.

* ``config.json``          the Setup (every page runs on it)
* ``etf_selection.json``   the ETF table on "ETFs & savings plan"
* ``review_holdings.json`` the holdings table of the Annual review
* ``explorer_custom.json`` your own allocations in the Allocation explorer
* ``reviews/review_<date>.json``  one snapshot per day a review is run (history)

Writes are atomic (temporary file, then rename), so a crash never leaves a half-written file.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ("config.json", "etf_selection.json", "review_holdings.json", "explorer_custom.json")


def data_dir() -> Path:
    return Path(os.environ.get("MC_DATA_DIR") or ROOT / "my_data")


def load(name: str):
    """Parsed JSON content of ``name``, or None if missing or unreadable."""
    path = data_dir() / name
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save(name: str, obj) -> None:
    path = data_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def save_review_snapshot(payload: dict) -> Path:
    name = f"reviews/review_{date.today().isoformat()}.json"
    save(name, payload)
    return data_dir() / name


def list_reviews() -> list[Path]:
    folder = data_dir() / "reviews"
    return sorted(folder.glob("review_*.json"), reverse=True) if folder.exists() else []


def saved_files() -> list[str]:
    return [f for f in FILES if (data_dir() / f).exists()] + [f"reviews/{p.name}" for p in list_reviews()]


def reset(keep_reviews: bool = True) -> None:
    """Delete the saved settings (and, unless ``keep_reviews``, the review history)."""
    for f in FILES:
        (data_dir() / f).unlink(missing_ok=True)
    if not keep_reviews:
        shutil.rmtree(data_dir() / "reviews", ignore_errors=True)
