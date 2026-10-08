"""Stationary block bootstrap (Politis & Romano, 1994) of historical monthly EUR returns.

Resampling whole blocks of history keeps the fat tails, volatility clustering and
cross-asset correlation structure of real markets without any parametric assumptions.
Blocks have geometric lengths with mean ``bootstrap_block_months`` and wrap around circularly.

Mean adjustment (on by default): 1973-2025 had much higher bond yields (8 % in the 1970s/80s)
and inflation than today. Each series is shifted in log space so its average matches the
forward-looking parametric model. Today's yields then anchor the expected returns, while
history supplies the shape of the risk.
"""

from __future__ import annotations

import numpy as np

from ..config import SimConfig
from ..data.panel import historical_returns
from .base import ScenarioSet


def stationary_bootstrap_indices(n_obs: int, n_paths: int, n_months: int, mean_block: float,
                                 rng: np.random.Generator) -> np.ndarray:
    idx = np.empty((n_paths, n_months), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n_obs, n_paths)
    p_new = 1.0 / mean_block
    for t in range(1, n_months):
        new = rng.random(n_paths) < p_new
        idx[:, t] = np.where(new, rng.integers(0, n_obs, n_paths), (idx[:, t - 1] + 1) % n_obs)
    return idx


def forward_targets(cfg: SimConfig, n_paths: int = 1000) -> dict[str, np.ndarray | float]:
    """Mean monthly log returns, log inflation and long yield implied by the parametric model."""
    from .parametric import generate_parametric

    probe = generate_parametric(cfg.model_copy(update={"n_paths": n_paths, "seed": cfg.seed + 1}))
    return {
        "log_returns": np.log1p(probe.returns.astype(float)).mean(axis=(0, 1)),
        "log_inflation": float(np.log1p(probe.inflation).mean()),
        "long_yield": float(probe.long_yield.mean()),
    }


def generate_bootstrap(cfg: SimConfig) -> ScenarioSet:
    hist = historical_returns(cfg.assets, cfg.market)
    rng = np.random.default_rng(cfg.seed)
    idx = stationary_bootstrap_indices(len(hist.dates), cfg.n_paths, cfg.months,
                                       cfg.bootstrap_block_months, rng)

    lr = np.log1p(hist.returns)
    li = np.log1p(hist.inflation)
    ly = hist.long_yield.copy()
    if cfg.bootstrap_mean_adjust:
        tgt = forward_targets(cfg)
        lr = lr - lr.mean(axis=0) + tgt["log_returns"]
        li = li - li.mean() + tgt["log_inflation"]
        ly = ly - ly.mean() + tgt["long_yield"]

    returns = np.expm1(lr[idx]).astype(np.float32)
    adjusted = ", mean-adjusted" if cfg.bootstrap_mean_adjust else ""
    return ScenarioSet(
        asset_keys=hist.asset_keys,
        returns=np.maximum(returns, -0.99),
        inflation=np.expm1(li[idx]),
        long_yield=ly[idx],
        source=f"bootstrap ({hist.dates[0]:%Y-%m}–{hist.dates[-1]:%Y-%m}, "
               f"blocks ~{cfg.bootstrap_block_months:g}m{adjusted})",
    )
