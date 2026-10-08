"""Historical rolling-window backtests: every path is an actual stretch of 1973-today history.

There are few windows (for example 24 overlapping 30-year windows starting each January)
and they overlap heavily, so treat the result as a reality check on the Monte Carlo
distribution, not as a distribution in its own right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import SimConfig
from ..data.panel import historical_returns
from .base import ScenarioSet


def rolling_windows(cfg: SimConfig, step_months: int = 12) -> tuple[ScenarioSet, list[pd.Timestamp]]:
    hist = historical_returns(cfg.assets, cfg.market)
    t = cfg.months
    first_jan = int(np.argmax(hist.dates.month == 1))
    starts = list(range(first_jan, len(hist.dates) - t + 1, step_months))
    if not starts:
        raise ValueError(f"history ({len(hist.dates)} months) is shorter than the horizon ({t} months)")
    idx = np.array([np.arange(s, s + t) for s in starts])
    sc = ScenarioSet(
        asset_keys=hist.asset_keys,
        returns=hist.returns[idx],
        inflation=hist.inflation[idx],
        long_yield=hist.long_yield[idx],
        source=f"historical rolling windows ({len(starts)} × {cfg.plan.horizon_years}y)",
    )
    return sc, [hist.dates[s] for s in starts]
