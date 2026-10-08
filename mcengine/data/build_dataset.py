"""Build the monthly historical panel used by the bootstrap generator and the backtests.

Run ``python -m mcengine.data.build_dataset`` to refresh ``mcengine/data/panel_monthly.csv``.

The panel stores raw market series. Asset returns are derived from them at run time with
the durations in the config (see :mod:`mcengine.data.panel`). Columns:

* ``equity_usd``: US total stock market return in USD (Fama/French Mkt-RF + RF). This
  stands in for global equity: the US is about 65-70 % of MSCI World today, but a US-only
  history is more favourable than a true world index would have been.
* ``eur_per_usd``: monthly average. DEM/USD / 1.95583 before 1999, 1 / (USD/EUR) after.
* ``gold_usd``: monthly gold price in USD.
* ``y10``: German 10y government bond yield (decimal p.a., OECD MEI via FRED).
* ``y3m``: German 3-month interbank rate (decimal p.a.).
* ``cpi``: German CPI (OECD MEI), extended with the HICP Germany index chained at the overlap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from . import sources

PANEL_PATH = Path(__file__).with_name("panel_monthly.csv")
DEM_PER_EUR = 1.95583
START = "1972-12-01"  # first month is used only as the base for returns from 1973-01


def build() -> pd.DataFrame:
    fr = sources.french_market()
    dem_usd = sources.fred("EXGEUS")          # DEM per USD
    usd_eur = sources.fred("EXUSEU")          # USD per EUR
    y10 = sources.fred("IRLTLT01DEM156N") / 100.0
    y3m = sources.fred("IR3TIB01DEM156N") / 100.0
    cpi_oecd = sources.fred("DEUCPIALLMINMEI")
    hicp = sources.ecb_hicp_de()
    gold = sources.gold_usd()

    eur_usd = pd.concat([(dem_usd / DEM_PER_EUR).loc[:"1998-12-01"], 1.0 / usd_eur.loc["1999-01-01":]])

    # Chain HICP onto the OECD CPI from the last OECD month onwards.
    last = cpi_oecd.index.max()
    ext = hicp.loc[last:] / hicp.loc[last] * cpi_oecd.loc[last]
    cpi = pd.concat([cpi_oecd, ext.iloc[1:]])

    panel = pd.DataFrame({
        "equity_usd": fr["us_equity_tr"],
        "eur_per_usd": eur_usd,
        "gold_usd": gold,
        "y10": y10,
        "y3m": y3m,
        "cpi": cpi,
    }).loc[START:]
    panel = panel.dropna()
    # keep a contiguous monthly range
    full = pd.date_range(panel.index.min(), panel.index.max(), freq="MS")
    panel = panel.reindex(full)
    first_gap = panel.index[panel.isna().any(axis=1)]
    if len(first_gap):
        panel = panel.loc[: first_gap[0] - pd.offsets.MonthBegin(1)]
    panel.index.name = "month"
    return panel


def main() -> int:
    panel = build()
    panel.to_csv(PANEL_PATH, float_format="%.8g")
    print(f"wrote {PANEL_PATH} ({len(panel)} months, {panel.index[0]:%Y-%m} .. {panel.index[-1]:%Y-%m})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
