"""Fetchers for the public data behind the historical panel. All of them return monthly ``pd.Series``
indexed by month-start timestamps.

Sources:
* Kenneth R. French Data Library: US total market excess return and 1-month T-bill (USD).
* FRED (St. Louis Fed / OECD MEI): DEM/USD and USD/EUR exchange rates, German 10y
  government bond yield, German 3-month interbank rate, German CPI.
* ECB Data Portal: HICP Germany, used to extend the CPI past the end of the OECD series.
* datahub.io "gold-prices": monthly gold price in USD.
"""

from __future__ import annotations

import io
import time
import urllib.request
import zipfile

import pandas as pd

UA = {"User-Agent": "curl/8.5.0"}  # FRED drops connections from unknown agents
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
ECB_HICP_DE = "https://data-api.ecb.europa.eu/service/data/ICP/M.DE.N.000000.4.INX?format=csvdata"
GOLD = "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv"


def _get(url: str, timeout: int = 60, retries: int = 4) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed https URLs
                return r.read()
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("unreachable")


def _month_index(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(idx)).to_period("M").to_timestamp()


def fred(series_id: str) -> pd.Series:
    df = pd.read_csv(io.BytesIO(_get(FRED.format(series_id))))
    df.columns = ["date", "value"]
    s = pd.to_numeric(df["value"], errors="coerce")
    s.index = _month_index(df["date"])
    return s.dropna().rename(series_id)


def french_market() -> pd.DataFrame:
    """US total market return and T-bill return (monthly, decimals, USD)."""
    z = zipfile.ZipFile(io.BytesIO(_get(FRENCH)))
    lines = z.read(z.namelist()[0]).decode("latin1").splitlines()
    rows = []
    started = False
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if parts and len(parts[0]) == 6 and parts[0].isdigit():
            started = True
            rows.append((parts[0], float(parts[1]), float(parts[4])))
        elif started:
            break  # monthly block ends where the annual block begins
    df = pd.DataFrame(rows, columns=["ym", "mkt_rf", "rf"])
    df.index = pd.to_datetime(df["ym"], format="%Y%m")
    out = pd.DataFrame(index=df.index)
    out["us_equity_tr"] = (df["mkt_rf"] + df["rf"]) / 100.0
    out["us_tbill"] = df["rf"] / 100.0
    return out


def ecb_hicp_de() -> pd.Series:
    df = pd.read_csv(io.BytesIO(_get(ECB_HICP_DE)))
    s = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    s.index = _month_index(df["TIME_PERIOD"])
    return s.dropna().sort_index().rename("hicp_de")


def gold_usd() -> pd.Series:
    df = pd.read_csv(io.BytesIO(_get(GOLD)))
    s = pd.to_numeric(df["Price"], errors="coerce")
    s.index = _month_index(df["Date"])
    return s.dropna().rename("gold_usd")
