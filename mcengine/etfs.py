"""ETF selection: product checklist, cost impact and the monthly savings-plan split.

Products are chosen with fixed criteria, not forecasts: all ETFs on the same index return
almost the same, before costs. The simulation's job here is to show what the cost
differences add up to over the plan.

``etf_candidates.csv`` ships an example shortlist. The data was checked against justETF on
``CANDIDATES_AS_OF``. Fees and fund sizes change, so re-check every product before buying.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .config import SimConfig

CANDIDATES_PATH = Path(__file__).parent / "data" / "etf_candidates.csv"
CANDIDATES_AS_OF = "2026-10-08"
JUSTETF_URL = "https://www.justetf.com/en/etf-profile.html?isin={}"

GOOD, WARN, BAD, NA = "good", "warn", "bad", "na"
ICON = {GOOD: "✅", WARN: "⚠️", BAD: "❌", NA: "–"}

# (good up to, acceptable up to) TER per asset class
TER_LIMITS = {"equity": (0.0020, 0.0040), "nominal_bond": (0.0015, 0.0030), "linker": (0.0015, 0.0030),
              "gold": (0.0015, 0.0040), "cash": (0.0020, 0.0040)}

# Typical costs of actively managed bank funds, for comparison
BANK_FUND_TER = {"equity": 0.015, "nominal_bond": 0.008, "linker": 0.008, "gold": 0.004, "cash": 0.0}


def load_candidates() -> pd.DataFrame:
    df = pd.read_csv(CANDIDATES_PATH, dtype={"isin": str})
    df["isin"] = df["isin"].fillna("")
    df["use"] = df["use"].fillna(0).astype(bool)
    df["delivery_claim"] = df["delivery_claim"].map({1.0: True, 0.0: False})
    df["share_pct"] = 100.0
    df["savings_plan"] = True
    df["note"] = df["note"].fillna("")
    return df


def _kind(cfg: SimConfig) -> dict[str, str]:
    return {a.key: a.kind for a in cfg.assets}


def check_row(row: pd.Series, kind: str) -> dict[str, tuple[str, str]]:
    """Criterion -> (status, explanation) for one product."""
    out: dict[str, tuple[str, str]] = {}
    is_account = not row.get("isin")
    good, ok = TER_LIMITS.get(kind, (0.002, 0.004))
    ter = float(row.get("ter") or 0.0)
    out["Cost"] = ((GOOD, f"{ter:.2%} p.a. – low") if ter <= good else
                   (WARN, f"{ter:.2%} p.a. – acceptable, cheaper options exist") if ter <= ok else
                   (BAD, f"{ter:.2%} p.a. – expensive for this asset class"))
    if is_account:
        out.update({k: (NA, "bank account") for k in ("Size", "Payout", "Replication", "Domicile")})
    else:
        size = row.get("fund_size_eur_m")
        if size is None or (isinstance(size, float) and math.isnan(size)):
            out["Size"] = (WARN, "unknown fund size")
        else:
            size = float(size)
            out["Size"] = ((GOOD, f"€{size:,.0f}m") if size >= 1000 else
                           (WARN, f"€{size:,.0f}m – smaller fund") if size >= 100 else
                           (BAD, f"€{size:,.0f}m – risk of closure"))
        dist = str(row.get("distribution") or "")
        out["Payout"] = ((GOOD, "accumulating") if dist.lower().startswith("acc") else
                         (WARN, "distributing – payouts taxed yearly, reinvest by hand"))
        repl = str(row.get("replication") or "")
        out["Replication"] = ((GOOD, "physical") if repl.lower().startswith("phys") else
                              (GOOD if kind == "cash" else WARN, f"{repl.lower() or 'unknown'} (swap-based)"))
        dom = str(row.get("domicile") or "")
        if kind == "equity":
            out["Domicile"] = ((GOOD, "Ireland – lowest US dividend withholding tax") if dom == "Ireland" else
                               (WARN, f"{dom} – Irish funds lose less to US withholding tax"))
        elif kind == "gold":
            claim = row.get("delivery_claim")
            out["Domicile"] = ((GOOD, "delivery claim – tax-free after 1 year") if claim is True else
                               (WARN, "no delivery claim – gains taxed at sale"))
        else:
            known = dom in ("Ireland", "Luxembourg", "Germany")
            out["Domicile"] = (GOOD, dom) if known else (WARN, dom or "?")
    out["Savings plan"] = ((GOOD, "available at your broker") if bool(row.get("savings_plan", True)) else
                           (BAD, "not available as savings plan at your broker"))
    return out


def checklist(df: pd.DataFrame, cfg: SimConfig) -> pd.DataFrame:
    kinds = _kind(cfg)
    rows = []
    for _, r in df.iterrows():
        res = check_row(r, kinds.get(r["asset_key"], "equity"))
        statuses = [s for s, _ in res.values()]
        verdict = BAD if BAD in statuses else WARN if WARN in statuses else GOOD
        rows.append({"asset_key": r["asset_key"], "name": r["name"], "verdict": verdict,
                     **{k: f"{ICON[s]} {txt}" for k, (s, txt) in res.items()}})
    return pd.DataFrame(rows)


def normalised_selection(df: pd.DataFrame) -> pd.DataFrame:
    """Used products with their share inside the asset class rescaled to sum to 1."""
    sel = df[df["use"]].copy()
    sel["share"] = sel["share_pct"].clip(lower=0).astype(float)
    totals = sel.groupby("asset_key")["share"].transform("sum")
    sel["share"] = (sel["share"] / totals.where(totals > 0)).fillna(1.0 / sel.groupby("asset_key")["share"]
                                                                   .transform("count"))
    return sel


def missing_classes(df: pd.DataFrame, cfg: SimConfig) -> list[str]:
    used = set(df.loc[df["use"], "asset_key"])
    return [k for k, w in cfg.weights.items() if w > 0 and k not in used]


def class_costs(df: pd.DataFrame, cfg: SimConfig) -> dict[str, float]:
    """Share-weighted TER per asset class of the selection; unselected classes keep the config value."""
    costs = {a.key: a.ter for a in cfg.assets}
    sel = normalised_selection(df)
    for key, g in sel.groupby("asset_key"):
        costs[key] = float((g["ter"].astype(float) * g["share"]).sum())
    return costs


def cheapest_costs(df: pd.DataFrame, cfg: SimConfig) -> dict[str, float]:
    costs = {a.key: a.ter for a in cfg.assets}
    for key, g in df.groupby("asset_key"):
        costs[key] = float(g["ter"].astype(float).min())
    return costs


def bank_fund_costs(cfg: SimConfig) -> dict[str, float]:
    return {a.key: BANK_FUND_TER.get(a.kind, a.ter) for a in cfg.assets}


def with_costs(cfg: SimConfig, costs: dict[str, float], gold_taxable: bool | None = None) -> SimConfig:
    """Config with the given TER per asset (and optionally gold taxed as a plain ETC)."""
    assets = []
    for a in cfg.assets:
        upd: dict = {"ter": float(costs.get(a.key, a.ter))}
        if a.kind == "gold" and gold_taxable is not None:
            upd["tax_class"] = "etc_taxable" if gold_taxable else "gold_etc"
        assets.append(a.model_copy(update=upd))
    return SimConfig.model_validate({**cfg.model_dump(), "assets": [a.model_dump() for a in assets]})


def gold_is_taxable(df: pd.DataFrame) -> bool | None:
    """True if most of the selected gold has no delivery claim; None if no gold is selected."""
    sel = normalised_selection(df)
    g = sel[sel["asset_key"] == "gold"]
    if g.empty:
        return None
    taxable_share = float(g.loc[g["delivery_claim"] != True, "share"].sum())  # noqa: E712 - NaN-safe
    return taxable_share > 0.5


def savings_plan(df: pd.DataFrame, cfg: SimConfig, broker_minimum: float = 1.0) -> pd.DataFrame:
    """Monthly amount per product, with a suggested interval where the amount is below the broker minimum."""
    sel = normalised_selection(df)
    rows = []
    for _, r in sel.iterrows():
        w = cfg.weights.get(r["asset_key"], 0.0)
        monthly = cfg.plan.monthly * w * r["share"]
        every = max(1, math.ceil(broker_minimum / monthly)) if monthly > 0 else 0
        rows.append({
            "asset_key": r["asset_key"], "name": r["name"], "isin": r["isin"],
            "weight": w * r["share"], "monthly": monthly,
            "start_capital": cfg.plan.start_capital * w * r["share"],
            "every_months": every, "amount_per_execution": monthly * every,
        })
    return pd.DataFrame(rows)
