"""Stress tests: replay a real historical crisis inside every simulated future.

The months of the chosen crisis (monthly EUR returns of every asset, inflation and the long
yield, taken from the 1973-2025 panel without any mean adjustment) overwrite the same months
of every path, starting at a chosen point of the plan. Everything before and after stays as
simulated. Comparing against the unchanged scenarios ("paired") shows what that crisis at
that moment does to your plan. Being hit just before the goal date is the dangerous case
for a saver.

Paths continue after the inserted window as simulated, with no memory of the crash. Without
the historical recovery months a crash early in the plan therefore looks permanent. Replaying
the following ``recovery_months`` of history too (24 by default in the app) gives the more
realistic picture; "crisis only" is the worst case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SimConfig
from .data.panel import historical_returns
from .scenarios.base import ScenarioSet


@dataclass(frozen=True)
class Crisis:
    key: str
    name: str
    start: str  # YYYY-MM, first month of the window
    end: str    # YYYY-MM, last month (inclusive)
    story: str


CRISES: dict[str, Crisis] = {c.key: c for c in [
    Crisis("oil_1973", "Oil crisis & stagflation 1973–74", "1973-01", "1974-12",
           "Oil price shock, double-digit inflation, rising rates: stocks and bonds fell together "
           "in real terms."),
    Crisis("dotcom", "Dot-com crash 2000–03", "2000-04", "2003-03",
           "Three years of falling stock prices after the tech bubble; government bonds rallied."),
    Crisis("gfc", "Global financial crisis 2007–09", "2007-11", "2009-02",
           "Banking crisis: stocks roughly halved within 16 months; Bunds and gold held up."),
    Crisis("covid", "Covid crash 2020", "2020-02", "2020-03",
           "The fastest crash on record, about one third in a few weeks, followed by a quick recovery."),
    Crisis("inflation_2022", "Inflation shock 2022", "2022-01", "2022-12",
           "Inflation and rates jumped: stocks and bonds fell at the same time, the worst year for a "
           "60/40 in decades."),
    Crisis("lost_decade", "Lost decade 2000–09", "2000-01", "2009-12",
           "Two crashes in ten years: stocks ended the decade roughly where they started, in EUR "
           "even below."),
]}


def crisis_window(cfg: SimConfig, key: str, recovery_months: int = 0) -> dict:
    """Historical monthly data of a crisis window (plus the following ``recovery_months``)."""
    c = CRISES[key]
    hist = historical_returns(cfg.assets, cfg.market)
    end = pd.Timestamp(c.end) + pd.DateOffset(months=recovery_months)
    sel = (hist.dates >= pd.Timestamp(c.start)) & (hist.dates <= end)
    if not sel.any():
        raise ValueError(f"no data for {c.name}")
    return {"returns": hist.returns[sel], "inflation": hist.inflation[sel],
            "long_yield": hist.long_yield[sel], "dates": hist.dates[sel], "asset_keys": hist.asset_keys}


def crisis_summary(cfg: SimConfig, key: str, recovery_months: int = 0) -> pd.DataFrame:
    """Cumulative change of each asset, the target portfolio and inflation over the crisis
    (and, with ``recovery_months``, including the months after it)."""

    def changes(rec: int) -> list[float]:
        win = crisis_window(cfg, key, rec)
        r = win["returns"]
        port = np.prod(1 + r @ cfg.weight_vector()) - 1  # rebalanced monthly to target
        return [*(np.prod(1 + r, axis=0) - 1), port, np.prod(1 + win["inflation"]) - 1]

    names = [a.name for a in cfg.assets] + ["Your target portfolio (rebalanced monthly)", "Inflation"]
    out = pd.DataFrame({"": names, "Crisis": changes(0)})
    if recovery_months:
        out[f"Incl. {recovery_months} months after"] = changes(recovery_months)
    return out


def start_month(cfg: SimConfig, key: str, when: str | int, recovery_months: int = 0) -> int:
    """Month index where the crisis starts.

    'start' = first month, 'middle' = centred (crisis plus recovery), 'end' = the *crash itself*
    ends in the last month of the plan (no time left to recover; the recovery months fall after
    the plan), or a year number (1 = first year).
    """
    n_crisis = len(crisis_window(cfg, key)["returns"])
    n_total = len(crisis_window(cfg, key, recovery_months)["returns"])
    t = cfg.months
    match when:
        case "start":
            return 0
        case "middle":
            return max((t - n_total) // 2, 0)
        case "end":
            return max(t - n_crisis, 0)
        case _:
            return int(np.clip((int(when) - 1) * 12, 0, max(t - 1, 0)))


def insert_crisis(sc: ScenarioSet, cfg: SimConfig, key: str, start: int,
                  recovery_months: int = 0) -> ScenarioSet:
    """Copy of ``sc`` with the crisis months (and optionally the historical recovery after it)
    written into every path from month ``start`` on."""
    win = crisis_window(cfg, key, recovery_months)
    if win["asset_keys"] != sc.asset_keys:
        raise ValueError("scenario assets do not match the configuration")
    n = min(len(win["returns"]), sc.n_months - start)
    if n <= 0:
        raise ValueError("crisis starts after the end of the plan")
    returns = sc.returns.copy()
    inflation = sc.inflation.copy()
    long_yield = sc.long_yield.copy()
    returns[:, start:start + n, :] = win["returns"][:n].astype(returns.dtype)
    inflation[:, start:start + n] = win["inflation"][:n]
    long_yield[:, start:start + n] = win["long_yield"][:n]
    return ScenarioSet(sc.asset_keys, returns, inflation, long_yield, sc.regimes, sc.regime_names,
                       f"{sc.source} + {CRISES[key].name} from month {start + 1}")
