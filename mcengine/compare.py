"""Run several variants on the *same* scenarios (common random numbers).

Every variant sees exactly the same market paths, so the differences between variants are
caused by the decisions (rebalancing rule, allocation, costs) and not by sampling noise.
This is a paired comparison: per-path differences are far more precise than the gap
between two independent runs.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .config import SimConfig, StrategySpec
from .metrics import summarize
from .portfolio.simulator import SimResult, simulate
from .scenarios import ScenarioSet, generate

DEFAULT_STRATEGIES: list[StrategySpec] = [
    StrategySpec(kind="buy_and_hold"),
    StrategySpec(kind="calendar", rebalance_months=12),
    StrategySpec(kind="band", abs_band=0.05, rel_band=0.25),
    StrategySpec(kind="cashflow"),
    StrategySpec(kind="hybrid", abs_band=0.10, rel_band=0.50),
]


def run_variants(
    variants: dict[str, SimConfig],
    sc: ScenarioSet | None = None,
    goal_real: float | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, SimResult]]:
    """Simulate each config on shared scenarios; returns (metrics table, results by label)."""
    if not variants:
        raise ValueError("no variants")
    first = next(iter(variants.values()))
    sc = sc if sc is not None else generate(first)
    goal = first.goal_real if goal_real is None else goal_real
    rows, results = {}, {}
    for i, (label, cfg) in enumerate(variants.items()):
        if progress:
            progress(i / len(variants), label)
        res = simulate(cfg, sc, label=label)
        results[label] = res
        rows[label] = summarize(res, goal)
    if progress:
        progress(1.0, "done")
    return pd.DataFrame(rows).T, results


def compare_strategies(cfg: SimConfig, strategies: list[StrategySpec] | None = None,
                       sc: ScenarioSet | None = None, **kw) -> tuple[pd.DataFrame, dict[str, SimResult]]:
    strategies = strategies or DEFAULT_STRATEGIES
    variants = {s.label(): cfg.model_copy(update={"strategy": s}) for s in strategies}
    return run_variants(variants, sc, **kw)


def paired_difference(a: SimResult, b: SimResult) -> dict[str, float]:
    """Per-path real terminal wealth difference a - b, with its MC standard error."""
    d = a.real_final_net() - b.real_final_net()
    return {
        "mean": float(d.mean()),
        "se": float(d.std(ddof=1) / np.sqrt(d.size)),
        "share_a_better": float((d > 0).mean()),
        "p5": float(np.percentile(d, 5)),
        "p95": float(np.percentile(d, 95)),
    }
