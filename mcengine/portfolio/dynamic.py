"""Time- and state-dependent target allocations.

* **Glide path:** over the last ``years`` of the plan the equity share moves linearly from the
  base allocation to ``end_equity``, so a crash just before the goal date hurts less.
* **Re-evaluation switch** (the "re-run the Monte Carlo every 6-12 months" idea): every
  ``every_months`` the probability of reaching the goal is re-estimated from the current
  state of each path, assuming the base plan is kept. If it is at least ``high``, the path
  switches to a defensive mix (``defensive_equity``) and locks in its progress. If it later
  falls below ``low``, it switches back to the base mix; it never takes more risk than the
  base plan. The gap between the thresholds stops it from flip-flopping. Switches are
  executed as trades, so they cost tax and trading costs.
* **Panic selling** (a behaviour what-if, not a recommendation): when the portfolio falls
  ``drawdown`` below its peak, all equity is sold into cash, new savings stop buying
  equity, and the investor buys back in after ``months_out`` months.

The goal probability inside the switch uses a fast closed-form approximation (lognormal
future value of current wealth plus remaining savings, pre-tax). It runs for every path at
every review date, where a nested Monte Carlo would be far too slow.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ..config import SimConfig
from ..scenarios.base import ScenarioSet


def scale_equity(w: np.ndarray, is_eq: np.ndarray, equity) -> np.ndarray:
    """Re-weight ``w`` (shape [A] or [P, A]) to the equity share ``equity`` (scalar or [P]),
    scaling equity and non-equity assets proportionally."""
    w = np.asarray(w, dtype=float)
    eq = np.asarray(equity, dtype=float)
    if w.ndim == 2 and eq.ndim == 1:
        eq = eq[:, None]
    w_eq, w_def = w * is_eq, w * ~is_eq
    s_eq = w_eq.sum(axis=-1, keepdims=True)
    s_def = w_def.sum(axis=-1, keepdims=True)
    n_eq, n_def = max(int(is_eq.sum()), 1), max(int((~is_eq).sum()), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        eq_part = np.where(s_eq > 0, w_eq / s_eq, is_eq / n_eq) * eq
        def_part = np.where(s_def > 0, w_def / s_def, ~is_eq / n_def) * (1.0 - eq)
    return eq_part + def_part


def glide_equity(cfg: SimConfig, t: int, t_total: int) -> float:
    """Equity share of the (glide-path adjusted) base plan in month ``t``."""
    base = cfg.equity_share()
    g = cfg.glide
    if not g.enabled:
        return base
    start = max(t_total - 12 * g.years, 0)
    if t < start:
        return base
    frac = (t - start + 1) / max(t_total - start, 1)
    return base + (g.end_equity - base) * min(frac, 1.0)


def panic_weights(w: np.ndarray, is_eq: np.ndarray, safe_idx: int) -> np.ndarray:
    """``w`` with all equity moved into the safe asset."""
    out = np.array(w, dtype=float, copy=True)
    eq = (out * is_eq).sum(axis=-1)
    out[..., is_eq] = 0.0
    out[..., safe_idx] += eq
    return out


class GoalEstimator:
    """Closed-form P(real wealth at the horizon >= goal) per path, used for the re-evaluation switch.

    Monthly real return moments per asset come from the scenario set itself (all paths and
    months), i.e. from the same ex-ante distribution the investor believes in, not from any
    path's own future.
    """

    def __init__(self, sc: ScenarioSet, max_rows: int = 200_000):
        r = sc.returns.reshape(-1, sc.returns.shape[-1]).astype(float)
        infl = sc.inflation.reshape(-1)
        if r.shape[0] > max_rows:
            idx = np.random.default_rng(0).choice(r.shape[0], max_rows, replace=False)
            r, infl = r[idx], infl[idx]
        real = (1.0 + r) / (1.0 + infl)[:, None] - 1.0
        self.mean = real.mean(axis=0)
        self.cov = np.cov(real, rowvar=False)

    def probability(self, wealth_real: np.ndarray, saving_real: np.ndarray, months_left: int,
                    w: np.ndarray, goal: float) -> np.ndarray:
        if goal <= 0:
            return np.ones_like(wealth_real)
        n = max(int(months_left), 0)
        g = float(w @ self.mean)
        var = float(w @ self.cov @ w)
        growth = (1.0 + g) ** n
        lump = wealth_real * growth
        annuity = saving_real * (((growth - 1.0) / g * (1.0 + g)) if abs(g) > 1e-12 else n)
        v_lump = n * np.log1p(var / (1.0 + g) ** 2)
        sd = lump * np.sqrt(np.expm1(v_lump)) + annuity * np.sqrt(np.expm1(v_lump / 3.0))
        mean = np.maximum(lump + annuity, 1e-9)
        s2 = np.log1p((sd / mean) ** 2)
        z = (np.log(goal) - (np.log(mean) - s2 / 2.0)) / np.sqrt(np.maximum(s2, 1e-12))
        return 1.0 - norm.cdf(z)
