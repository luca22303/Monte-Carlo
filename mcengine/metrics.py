"""Outcome and risk metrics for savings-plan simulations.

For an accumulating investor the useful questions are about reaching a goal and about how
bumpy the ride is, not withdrawal solvency. All "real" figures are in today's EUR (deflated
by the simulated CPI).
"""

from __future__ import annotations

import numpy as np

from .portfolio.simulator import SimResult

PCTS = (5, 25, 50, 75, 95)


# ----------------------------------------------------------------------------- primitives
def cvar(x: np.ndarray, alpha: float = 0.05) -> float:
    """Mean of the worst ``alpha`` share of outcomes (lower tail, Expected Shortfall)."""
    x = np.sort(np.asarray(x, dtype=float).ravel())
    n = max(1, int(np.floor(alpha * x.size)))
    return float(x[:n].mean())


def var(x: np.ndarray, alpha: float = 0.05) -> float:
    return float(np.quantile(np.asarray(x, dtype=float).ravel(), alpha))


def max_drawdown(index: np.ndarray) -> np.ndarray:
    """[P] maximum peak-to-trough decline of each path of an index [P, T] (positive number)."""
    start = np.ones((index.shape[0], 1))
    idx = np.concatenate([start, index], axis=1)
    peak = np.maximum.accumulate(idx, axis=1)
    return (1.0 - idx / peak).max(axis=1)


def drawdown_series(index: np.ndarray) -> np.ndarray:
    idx = np.concatenate([np.ones((index.shape[0], 1)), index], axis=1)
    return 1.0 - idx / np.maximum.accumulate(idx, axis=1)


def longest_underwater(index: np.ndarray) -> np.ndarray:
    """[P] longest stretch (months) the index spends below its previous peak."""
    under = drawdown_series(index)[:, 1:] > 1e-12
    run = np.zeros(index.shape[0], dtype=np.int64)
    best = np.zeros_like(run)
    for t in range(under.shape[1]):
        run = np.where(under[:, t], run + 1, 0)
        np.maximum(best, run, out=best)
    return best


def ulcer_index(index: np.ndarray) -> np.ndarray:
    dd = drawdown_series(index)
    return np.sqrt((dd**2).mean(axis=1))


def calendar_year_returns(index: np.ndarray) -> np.ndarray:
    """[P, Y] returns per complete 12-month block of an index that starts at 1."""
    t = index.shape[1] - index.shape[1] % 12
    ends = index[:, 11:t:12]
    starts = np.concatenate([np.ones((index.shape[0], 1)), ends[:, :-1]], axis=1)
    return ends / starts - 1.0


def rolling_returns(index: np.ndarray, months: int = 12) -> np.ndarray:
    idx = np.concatenate([np.ones((index.shape[0], 1)), index], axis=1)
    return idx[:, months:] / idx[:, :-months] - 1.0


def money_weighted_irr(flows: np.ndarray, terminal: np.ndarray, iters: int = 30) -> np.ndarray:
    """Annualised IRR per path for monthly outflows ``flows`` [P, T] paid at the start of each
    month and a ``terminal`` [P] value received at the end of month T.

    Solves sum_t flows_t * (1+r)^(T-t) = terminal with safeguarded Newton steps (the future
    value is increasing and convex in r for non-negative flows, so Newton from above converges).
    """
    p, t = flows.shape
    flows = flows.astype(float)
    expo = (t - np.arange(t)).astype(float)
    lo, hi = np.full(p, -0.08), np.full(p, 0.08)
    r = np.full(p, 0.01)
    for _ in range(iters):
        g = np.power(1.0 + r[:, None], expo - 1.0)
        fv = (flows * g).sum(axis=1) * (1.0 + r)
        d = (flows * g * expo).sum(axis=1)
        f = fv - terminal
        hi = np.where(f > 0, r, hi)
        lo = np.where(f > 0, lo, r)
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.where(d > 0, r - f / d, 0.5 * (lo + hi))
        bad = ~np.isfinite(step) | (step <= lo) | (step >= hi)
        r_new = np.where(bad, 0.5 * (lo + hi), step)
        if np.max(np.abs(r_new - r)) < 1e-10:
            r = r_new
            break
        r = r_new
    return np.power(1.0 + r, 12.0) - 1.0


def batch_se(x: np.ndarray, stat, n_batches: int = 20) -> float:
    """Monte Carlo standard error of ``stat`` via batch means."""
    x = np.asarray(x)
    n = (x.shape[0] // n_batches) * n_batches
    if n < n_batches * 5:
        return float("nan")
    vals = np.array([stat(b) for b in np.split(x[:n], n_batches)])
    return float(vals.std(ddof=1) / np.sqrt(n_batches))


# ----------------------------------------------------------------------------- results
def real_contributions(res: SimResult) -> np.ndarray:
    """[P, T] contributions deflated to today's EUR at their payment date (month start)."""
    cpi_start = np.concatenate([np.ones((res.n_paths, 1)), res.cpi[:, :-1]], axis=1)
    return res.contributions / cpi_start


def fan(res: SimResult, real: bool = True, pcts=PCTS) -> dict[int, np.ndarray]:
    """Percentile trajectories of wealth over time (for fan charts)."""
    w = res.real_wealth() if real else res.wealth
    q = np.percentile(w, pcts, axis=0)
    return {p: q[i] for i, p in enumerate(pcts)}


def summarize(res: SimResult, goal_real: float = 0.0) -> dict[str, float]:
    """Headline metrics of one simulation, as a flat dict (ready for a table)."""
    real_final = res.real_final_net()
    contrib_real = real_contributions(res)
    paid_real = contrib_real.sum(axis=1)
    nav_real = res.nav / res.cpi
    years = res.months / 12.0

    irr_real = money_weighted_irr(contrib_real, real_final)
    irr_nom = money_weighted_irr(res.contributions.astype(float), res.final_net)
    twr_real = np.power(nav_real[:, -1], 1.0 / years) - 1.0
    mdd = max_drawdown(res.nav.astype(float))
    mdd_real = max_drawdown(nav_real)
    yr = calendar_year_returns(res.nav.astype(float))
    roll12 = rolling_returns(res.nav.astype(float), 12)
    uw = longest_underwater(res.nav.astype(float))
    cpi_end = res.cpi[:, -1]

    out: dict[str, float] = {}
    for p in PCTS:
        out[f"real_wealth_p{p}"] = float(np.percentile(real_final, p))
    out["real_wealth_mean"] = float(real_final.mean())
    out["real_wealth_cvar5"] = cvar(real_final, 0.05)
    out["real_contributions"] = float(np.median(paid_real))
    out["multiple_median"] = float(np.median(real_final / np.maximum(paid_real, 1e-9)))
    out["p_goal"] = float((real_final >= goal_real).mean()) if goal_real > 0 else float("nan")
    out["p_real_loss"] = float((real_final < paid_real).mean())
    for p in (5, 50, 95):
        out[f"irr_real_p{p}"] = float(np.percentile(irr_real, p))
    out["irr_nominal_p50"] = float(np.median(irr_nom))
    out["twr_real_p50"] = float(np.median(twr_real))
    out["mdd_p50"] = float(np.median(mdd))
    out["mdd_p95"] = float(np.percentile(mdd, 95))
    out["mdd_real_p95"] = float(np.percentile(mdd_real, 95))
    out["p_mdd_over_20"] = float((mdd > 0.20).mean())
    out["underwater_months_p50"] = float(np.median(uw))
    out["underwater_months_p95"] = float(np.percentile(uw, 95))
    out["ulcer_p50"] = float(np.median(ulcer_index(res.nav.astype(float))))
    out["share_negative_years"] = float((yr < 0).mean()) if yr.size else float("nan")
    out["worst_year_p50"] = float(np.median(yr.min(axis=1))) if yr.size else float("nan")
    out["cvar5_12m"] = cvar(roll12, 0.05) if roll12.size else float("nan")
    out["vol_annual"] = float(np.median(np.std(np.diff(np.log(res.nav.astype(float)), axis=1), axis=1))
                              * np.sqrt(12))
    out["taxes_real"] = float(np.median((res.taxes / res.cpi).sum(axis=1)))
    out["fees_real"] = float(np.median(res.fees / cpi_end))
    out["turnover_real"] = float(np.median(res.turnover / cpi_end))
    out["rebalances_p50"] = float(np.median(res.rebalances))
    out["inflation_p50"] = float(np.median(np.power(cpi_end, 1.0 / years) - 1.0))
    out["se_median_wealth"] = batch_se(real_final, np.median)
    if goal_real > 0:
        out["se_p_goal"] = batch_se(real_final, lambda b: (b >= goal_real).mean())
    return out


METRIC_LABELS: dict[str, tuple[str, str]] = {
    # key: (label, format) - format: "eur", "pct", "num", "x", "months"
    "real_wealth_p5": ("Real wealth, bad case (P5)", "eur"),
    "real_wealth_p25": ("Real wealth, P25", "eur"),
    "real_wealth_p50": ("Real wealth, median", "eur"),
    "real_wealth_p75": ("Real wealth, P75", "eur"),
    "real_wealth_p95": ("Real wealth, good case (P95)", "eur"),
    "real_wealth_cvar5": ("Real wealth, avg of worst 5% (CVaR)", "eur"),
    "real_contributions": ("Savings paid (real)", "eur"),
    "multiple_median": ("Wealth / savings, median", "x"),
    "p_goal": ("P(reach goal)", "pct"),
    "p_real_loss": ("P(real loss vs. savings)", "pct"),
    "irr_real_p5": ("Real return p.a. (MWR), P5", "pct"),
    "irr_real_p50": ("Real return p.a. (MWR), median", "pct"),
    "irr_real_p95": ("Real return p.a. (MWR), P95", "pct"),
    "irr_nominal_p50": ("Nominal return p.a. after tax, median", "pct"),
    "twr_real_p50": ("Real TWR p.a., median", "pct"),
    "vol_annual": ("Volatility p.a.", "pct"),
    "mdd_p50": ("Max drawdown, median", "pct"),
    "mdd_p95": ("Max drawdown, bad case (P95)", "pct"),
    "mdd_real_p95": ("Max real drawdown, P95", "pct"),
    "p_mdd_over_20": ("P(drawdown > 20%)", "pct"),
    "underwater_months_p50": ("Longest underwater, median", "months"),
    "underwater_months_p95": ("Longest underwater, P95", "months"),
    "ulcer_p50": ("Ulcer index, median", "pct"),
    "share_negative_years": ("Share of negative years", "pct"),
    "worst_year_p50": ("Worst calendar year, median", "pct"),
    "cvar5_12m": ("12m return, avg of worst 5% (CVaR)", "pct"),
    "taxes_real": ("Taxes paid (real)", "eur"),
    "fees_real": ("Trading costs (real)", "eur"),
    "turnover_real": ("Rebalancing sales (real)", "eur"),
    "rebalances_p50": ("Rebalancing events", "num"),
    "inflation_p50": ("Inflation p.a., median", "pct"),
    "se_median_wealth": ("MC std. error of median", "eur"),
    "se_p_goal": ("MC std. error of P(goal)", "pct"),
}


def fmt(key: str, value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    kind = METRIC_LABELS.get(key, (key, "num"))[1]
    match kind:
        case "eur":
            return f"€{value:,.0f}"
        case "pct":
            return f"{value:.1%}"
        case "x":
            return f"{value:.2f}×"
        case "months":
            return f"{value:.0f} mo"
        case _:
            return f"{value:,.1f}"
