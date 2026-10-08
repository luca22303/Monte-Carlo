"""Load the historical panel and map it onto the configured asset universe (EUR, monthly)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from ..config import AssetSpec, MarketModel
from .build_dataset import PANEL_PATH

EXPECTED_INFLATION_MONTHS = 36  # trailing window used as the inflation expectation for synthetic linkers


@lru_cache(maxsize=1)
def load_panel() -> pd.DataFrame:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(f"{PANEL_PATH} missing - run `python -m mcengine.data.build_dataset`")
    return pd.read_csv(PANEL_PATH, index_col=0, parse_dates=True)


@dataclass
class HistoricalReturns:
    dates: pd.DatetimeIndex     # [T] month of each return
    returns: np.ndarray         # [T, A] nominal EUR simple returns, gross of fund costs
    inflation: np.ndarray       # [T]
    long_yield: np.ndarray      # [T] yield at the start of each month
    asset_keys: list[str]
    notes: dict[str, str]


def historical_returns(assets: list[AssetSpec], market: MarketModel | None = None) -> HistoricalReturns:
    """Monthly EUR asset returns from the panel.

    * Equity: US total market in USD translated to EUR (proxy for global equity).
    * Nominal bonds: German 10y yield turned into a constant-duration index return.
    * Linkers: synthetic. Real yield = 10y yield - trailing 3y inflation (an ex-ante proxy),
      return = real carry + realised inflation - duration * change in real yield.
    * Gold: USD gold price translated to EUR.
    * Cash: German 3-month rate / 12, floored at ``market.cash_floor``.
    """
    market = market or MarketModel()
    df = load_panel()
    fx = df["eur_per_usd"]
    fx_ret = fx / fx.shift(1)
    y = df["y10"]
    infl = df["cpi"].pct_change()
    n = EXPECTED_INFLATION_MONTHS
    infl_exp = (df["cpi"] / df["cpi"].shift(n)) ** (12 / n) - 1
    # before 3y of history exist fall back to trailing 12m inflation (first year: back-filled)
    infl_exp = infl_exp.fillna(df["cpi"] / df["cpi"].shift(12) - 1).bfill()
    ry = y - infl_exp

    cols, notes = {}, {}
    for a in assets:
        match a.kind:
            case "equity":
                cols[a.key] = (1 + df["equity_usd"]) * fx_ret - 1
                notes[a.key] = "US total market (Fama/French) in EUR – proxy for global equity"
            case "nominal_bond":
                dy = y.diff()
                cols[a.key] = y.shift(1) / 12 - a.duration * dy + 0.5 * a.convexity * dy**2
                notes[a.key] = f"German 10y yield, constant duration {a.duration:g}"
            case "linker":
                dr = ry.diff()
                cols[a.key] = ry.shift(1) / 12 + infl - a.duration * dr + 0.5 * a.convexity * dr**2
                notes[a.key] = "Synthetic: 10y yield − trailing 3y inflation as real yield"
            case "gold":
                cols[a.key] = df["gold_usd"].pct_change().add(1) * fx_ret - 1
                notes[a.key] = "Gold USD price in EUR"
            case "cash":
                cols[a.key] = np.maximum(df["y3m"].shift(1), market.cash_floor) / 12
                notes[a.key] = "German 3m money market rate"
    out = pd.DataFrame(cols).iloc[1:]
    return HistoricalReturns(
        dates=out.index,
        returns=out.to_numpy(dtype=float),
        inflation=infl.iloc[1:].to_numpy(dtype=float),
        long_yield=y.shift(1).iloc[1:].to_numpy(dtype=float),
        asset_keys=[a.key for a in assets],
        notes=notes,
    )


def annual_stats(returns: np.ndarray, inflation: np.ndarray | None = None) -> pd.DataFrame:
    """Geometric return, volatility and worst 12m return of monthly series [T, A] or [P, T, A]."""
    r = returns.reshape(-1, returns.shape[-1]) if returns.ndim == 3 else returns
    lr = np.log1p(r)
    stats = {
        "geo_return": np.expm1(lr.mean(axis=0) * 12),
        "volatility": lr.std(axis=0) * np.sqrt(12),
    }
    if returns.ndim == 2:
        roll = np.exp(pd.DataFrame(lr).rolling(12).sum().to_numpy()) - 1
        stats["worst_12m"] = np.nanmin(roll, axis=0)
    return pd.DataFrame(stats)
