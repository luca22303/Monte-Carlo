"""Regime-switching, fat-tailed, yield-anchored scenario generator.

Six monthly risk factors (``config.FACTORS``) are driven by a Markov chain over market
regimes (calm / deflationary crisis / inflation shock by default). Within a regime the
shocks are multivariate Student-t with regime-specific volatilities and correlations, so
the model has fat tails, volatility clustering (regimes persist) and correlations that
change in a crisis, including the 2022 case where stocks and bonds fall together.

Asset returns are built from the factors:

* equity, gold: log-return = drift + vol * shock
* nominal bonds: carry - duration * dy + 1/2 convexity * dy^2  (yield follows a mean-reverting AR(1))
* linkers:       real carry + realised inflation - duration * dr + 1/2 convexity * dr^2
* cash:          short rate / 12, floored at ``cash_floor``

Because bond returns come from yield paths, today's starting yields anchor future bond
returns, the way they do in practice.
"""

from __future__ import annotations

import numpy as np

from ..config import FACTORS, MarketModel, SimConfig
from .base import ScenarioSet

EQ, GOLD, Y, R, S, INFL = range(len(FACTORS))


def _nearest_corr(m: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone and rescale to unit diagonal."""
    vals, vecs = np.linalg.eigh((m + m.T) / 2)
    vals = np.clip(vals, 1e-6, None)
    out = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(out))
    return out / np.outer(d, d)


def regime_parameters(mm: MarketModel) -> dict[str, np.ndarray]:
    """Monthly regime parameters with drifts re-centred on the long-run assumptions."""
    k = len(mm.regimes)
    pi = mm.stationary_distribution()
    vol = np.array([[rg.vol[f] for f in FACTORS] for rg in mm.regimes])  # [K, F] annual
    drift = np.array([[rg.drift[f] for f in FACTORS] for rg in mm.regimes])  # [K, F] annual
    drift = drift - pi @ drift  # stationary-weighted mean drift is zero for every factor
    # Equity/gold: add the long-run log growth so the stationary mean equals ln(1+g).
    drift[:, EQ] += np.log1p(mm.equity_return)
    drift[:, GOLD] += np.log1p(mm.gold_return)
    chol = np.empty((k, len(FACTORS), len(FACTORS)))
    for j, rg in enumerate(mm.regimes):
        chol[j] = np.linalg.cholesky(_nearest_corr(np.asarray(rg.corr, dtype=float)))
    return {
        "vol_m": vol / np.sqrt(12.0),
        "drift_m": drift / 12.0,
        "chol": chol,
        "df": np.array([rg.df for rg in mm.regimes], dtype=float),
        "trans_cum": np.cumsum(np.asarray(mm.transition, dtype=float), axis=1),
        "pi": pi,
    }


def simulate_factors(mm: MarketModel, n_paths: int, n_months: int, rng: np.random.Generator) -> dict:
    """Simulate regimes, rate levels and equity/gold log returns.

    Returns arrays of shape [P, T] (rates are levels at the *end* of each month; the
    start values are returned separately).
    """
    prm = regime_parameters(mm)
    k = len(mm.regimes)
    p, t = n_paths, n_months

    if mm.start_regime == "calm":
        reg = np.zeros(p, dtype=np.int64)
    else:
        reg = np.searchsorted(np.cumsum(prm["pi"]), rng.random(p), side="right").clip(0, k - 1)

    start = np.array([mm.nominal_yield_start, mm.real_yield_start, mm.short_rate_start, mm.inflation_start])
    mean = np.array([mm.nominal_yield_mean, mm.real_yield_mean, mm.short_rate_mean, mm.inflation_mean])
    kappa_m = np.array([mm.nominal_yield_kappa, mm.real_yield_kappa, mm.short_rate_kappa,
                        mm.inflation_kappa]) / 12.0
    rates = np.tile(start, (p, 1))  # [P, 4]: y, r, s, i

    regimes = np.empty((p, t), dtype=np.int8)
    eq_lr = np.empty((p, t))
    gold_lr = np.empty((p, t))
    rate_paths = np.empty((p, t, 4))

    for m in range(t):
        if m > 0:
            u = rng.random(p)
            reg = (u[:, None] > prm["trans_cum"][reg]).sum(axis=1).clip(0, k - 1)
        regimes[:, m] = reg

        z = rng.standard_normal((p, len(FACTORS)))
        x = np.empty_like(z)
        for j in range(k):
            mask = reg == j
            if not mask.any():
                continue
            nu = prm["df"][j]
            xj = z[mask] @ prm["chol"][j].T
            if nu < 200:  # multivariate Student-t with unit variance, shared mixing variable
                w = rng.chisquare(nu, size=mask.sum()) / nu
                xj *= np.sqrt((nu - 2.0) / nu) / np.sqrt(w)[:, None]
            x[mask] = xj

        drift = prm["drift_m"][reg]  # [P, F]
        vol = prm["vol_m"][reg]
        shock = drift + vol * x
        eq_lr[:, m] = shock[:, EQ]
        gold_lr[:, m] = shock[:, GOLD]
        rates = rates + kappa_m * (mean - rates) + shock[:, [Y, R, S, INFL]]
        rate_paths[:, m] = rates

    return {
        "regimes": regimes,
        "equity_lr": eq_lr,
        "gold_lr": gold_lr,
        "rates": rate_paths,
        "rates_start": start,
    }


def generate_parametric(cfg: SimConfig) -> ScenarioSet:
    mm = cfg.market
    rng = np.random.default_rng(cfg.seed)
    p, t = cfg.n_paths, cfg.months
    f = simulate_factors(mm, p, t, rng)

    rates_end = f["rates"]  # [P, T, 4]
    rates_beg = np.concatenate([np.broadcast_to(f["rates_start"], (p, 1, 4)), rates_end[:, :-1]], axis=1)
    y0, r0, s0, i_end = rates_beg[..., 0], rates_beg[..., 1], rates_beg[..., 2], rates_end[..., 3]
    dy = rates_end[..., 0] - y0
    dr = rates_end[..., 1] - r0
    # Annual inflation rate i -> monthly CPI change. Deflation is allowed; floor at -50 % p.a.
    infl_m = np.power(1.0 + np.maximum(i_end, -0.5), 1.0 / 12.0) - 1.0

    out = np.empty((p, t, len(cfg.assets)), dtype=np.float64)
    for a, spec in enumerate(cfg.assets):
        match spec.kind:
            case "equity":
                out[..., a] = np.expm1(f["equity_lr"])
            case "gold":
                out[..., a] = np.expm1(f["gold_lr"])
            case "nominal_bond":
                out[..., a] = y0 / 12.0 - spec.duration * dy + 0.5 * spec.convexity * dy**2
            case "linker":
                out[..., a] = r0 / 12.0 + infl_m - spec.duration * dr + 0.5 * spec.convexity * dr**2
            case "cash":
                out[..., a] = np.maximum(s0, mm.cash_floor) / 12.0
    np.maximum(out, -0.99, out=out)

    return ScenarioSet(
        asset_keys=cfg.asset_keys,
        returns=out.astype(np.float32),
        inflation=infl_m,
        long_yield=y0,
        regimes=f["regimes"],
        regime_names=[r.name for r in mm.regimes],
        source="parametric",
    )
