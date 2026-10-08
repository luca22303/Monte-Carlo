"""Annual review: update the *state*, apply the fixed *policy*.

Once a year, enter the current holdings and cost basis. The review reports:

* drift from the target weights and whether the rebalancing band is breached
* how to split the next savings (cash-flow steering), and how many months of savings it
  takes to get back to target without selling
* if the band is breached, the trades to target and an estimate of the tax they trigger
* a fresh simulation from today's portfolio, giving an updated goal probability

The return assumptions are deliberately *not* changed in reaction to recent performance.
Only observable state changes: holdings, cost basis and, optionally, today's starting
yields and inflation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from .config import FUND_CLASSES, PARTIAL_EXEMPTION, SimConfig
from .metrics import summarize
from .portfolio.simulator import SimResult, simulate
from .portfolio.strategies import drift_breach, fill_underweights, trades_to_target
from .scenarios import generate


@dataclass
class ReviewResult:
    date: str
    total: float
    weights: dict[str, float]
    target: dict[str, float]
    drift: dict[str, float]
    band_breached: bool
    next_savings_split: dict[str, float]
    months_to_target_by_savings: float
    trades: dict[str, float] = field(default_factory=dict)     # + buy / - sell, only if breached
    trade_tax_estimate: float = 0.0
    summary: dict[str, float] = field(default_factory=dict)
    remaining_years: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=float)


def months_to_target(values: np.ndarray, w: np.ndarray, monthly: float) -> float:
    """Months of savings needed to restore target weights without selling anything."""
    total = values.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        need = np.where(w > 0, values / w, np.where(values > 0, np.inf, 0.0))
    extra = max(float(need.max()) - total, 0.0)
    if extra == 0:
        return 0.0
    return float("inf") if monthly <= 0 else extra / monthly


def estimate_sale_tax(cfg: SimConfig, sells: np.ndarray, values: np.ndarray, basis: np.ndarray,
                      allowance_left: float) -> float:
    """Approximate tax on selling ``sells`` (average-cost basis, gold assumed held > 1 year)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        gain_share = np.where(values > 0, (values - basis) / values, 0.0)
    taxable = 0.0
    for a, s, g in zip(cfg.assets, sells, gain_share, strict=True):
        if a.tax_class in FUND_CLASSES:
            taxable += s * g * (1.0 - PARTIAL_EXEMPTION[a.tax_class])
    return max(taxable - allowance_left, 0.0) * cfg.tax.flat_rate


def run_review(
    cfg: SimConfig,
    values: dict[str, float],
    cost_basis: dict[str, float],
    remaining_years: int,
    allowance_left: float | None = None,
    simulate_forward: bool = True,
) -> tuple[ReviewResult, SimResult | None]:
    keys = cfg.asset_keys
    v = np.array([float(values.get(k, 0.0)) for k in keys])
    b = np.array([float(cost_basis.get(k, values.get(k, 0.0))) for k in keys])
    w = cfg.weight_vector()
    total = float(v.sum())
    if total <= 0:
        raise ValueError("portfolio is empty")
    cur = v / total
    spec = cfg.strategy
    abs_band, rel_band = (spec.abs_band, spec.rel_band) if spec.kind in ("band", "hybrid") else (0.05, 0.25)
    breached = bool(drift_breach(v[None], w, abs_band, rel_band)[0])

    split = fill_underweights(v[None], np.array([cfg.plan.monthly]), w)[0]
    trades: dict[str, float] = {}
    tax_est = 0.0
    if breached and spec.kind != "cashflow":
        sells, buys = trades_to_target(v[None], w)
        trades = {k: float(bu - se) for k, se, bu in zip(keys, sells[0], buys[0], strict=True)}
        left = cfg.tax.allowance if allowance_left is None else allowance_left
        tax_est = estimate_sale_tax(cfg, sells[0], v, b, left) if cfg.tax.enabled else 0.0

    res = None
    summary: dict[str, float] = {}
    if simulate_forward and remaining_years > 0:
        plan = cfg.plan.model_copy(update={"horizon_years": remaining_years, "start_capital": 0.0})
        fwd = cfg.model_copy(update={"plan": plan})
        res = simulate(fwd, generate(fwd), initial_values=v, initial_basis=b, label="From today")
        summary = summarize(res, cfg.goal_real)

    out = ReviewResult(
        date=date.today().isoformat(),
        total=total,
        weights=dict(zip(keys, cur.round(6).tolist(), strict=True)),
        target=dict(zip(keys, w.tolist(), strict=True)),
        drift=dict(zip(keys, (cur - w).round(6).tolist(), strict=True)),
        band_breached=breached,
        next_savings_split=dict(zip(keys, split.round(2).tolist(), strict=True)),
        months_to_target_by_savings=months_to_target(v, w, cfg.plan.monthly),
        trades=trades,
        trade_tax_estimate=float(tax_est),
        summary=summary,
        remaining_years=remaining_years,
    )
    return out, res


def save_snapshot(result: ReviewResult, cfg: SimConfig, directory: str | Path = "reviews") -> Path:
    """Persist the review and the config it used (one JSON file per review date)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"review_{result.date}.json"
    payload = {"review": json.loads(result.to_json()), "config": json.loads(cfg.model_dump_json())}
    path.write_text(json.dumps(payload, indent=2))
    return path


def latest_market_state() -> dict[str, float]:
    """Latest observed German rates and inflation from the bundled panel (suggested start values)."""
    from .data.panel import load_panel

    df = load_panel()
    last = df.iloc[-1]
    infl = float(df["cpi"].iloc[-1] / df["cpi"].iloc[-13] - 1)
    return {
        "as_of": df.index[-1].strftime("%Y-%m"),
        "bund_10y": float(last["y10"]),
        "money_market_3m": float(last["y3m"]),
        "inflation_12m": infl,
    }
