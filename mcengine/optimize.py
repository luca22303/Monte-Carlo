"""Allocation explorer and required-savings solver.

The explorer is an exhaustive grid search over the defensive sleeve (bonds, linkers, gold,
cash) at a few equity levels. Every allocation is simulated with taxes, costs and the
chosen rebalancing rule on the same scenarios, so the frontier reflects real after-tax
outcomes rather than a mean-variance abstraction. It deliberately does not "optimise" to
a single point: with estimation error, a robust region of good allocations beats a
spuriously precise optimum.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from .config import SimConfig
from .metrics import summarize
from .portfolio.simulator import simulate
from .scenarios import ScenarioSet, generate


def defensive_grid(
    cfg: SimConfig,
    equity_levels: Iterable[float] = (0.50, 0.55, 0.60),
    step: float = 0.25,
    max_weight: dict[str, float] | None = None,
) -> list[dict[str, float]]:
    """All allocations with the given equity shares whose defensive sleeve is a multiple of
    ``step`` (as a share of the sleeve). ``max_weight`` caps total portfolio weights per asset."""
    eq_keys = [a.key for a in cfg.assets if a.kind == "equity"]
    def_keys = [a.key for a in cfg.assets if a.kind != "equity"]
    n = round(1 / step)
    max_weight = max_weight or {}
    out = []
    for eq in equity_levels:
        for combo in itertools.product(range(n + 1), repeat=len(def_keys)):
            if sum(combo) != n:
                continue
            w = {k: eq / len(eq_keys) for k in eq_keys}
            w.update({k: (1 - eq) * c / n for k, c in zip(def_keys, combo, strict=True)})
            if any(w[k] > max_weight.get(k, 1.0) + 1e-9 for k in w):
                continue
            out.append({k: round(v, 6) for k, v in w.items()})
    return out


def pareto_mask(risk: np.ndarray, reward: np.ndarray) -> np.ndarray:
    """True for points that no other point beats on both lower risk and higher reward."""
    idx = np.argsort(risk, kind="stable")
    mask = np.zeros(risk.size, dtype=bool)
    best = -np.inf
    for i in idx:
        if reward[i] > best + 1e-12:
            mask[i] = True
            best = reward[i]
    return mask


def explore_allocations(
    cfg: SimConfig,
    allocations: list[dict[str, float]],
    sc: ScenarioSet | None = None,
    risk_metric: str = "mdd_p95",
    reward_metric: str = "real_wealth_p50",
    progress: Callable[[float, str], None] | None = None,
) -> pd.DataFrame:
    sc = sc if sc is not None else generate(cfg)
    rows = []
    for i, w in enumerate(allocations):
        if progress:
            progress(i / len(allocations), f"{i + 1}/{len(allocations)}")
        c = cfg.model_copy(update={"weights": w})
        s = summarize(simulate(c, sc), cfg.goal_real)
        rows.append({**{f"w_{k}": v for k, v in w.items()}, **s})
    if progress:
        progress(1.0, "done")
    df = pd.DataFrame(rows)
    risk = df[risk_metric].to_numpy()
    if risk_metric in ("real_wealth_p5", "real_wealth_cvar5", "irr_real_p5", "cvar5_12m"):
        risk = -risk  # metrics where higher is safer
    df["pareto"] = pareto_mask(risk, df[reward_metric].to_numpy())
    return df


def required_savings(
    cfg: SimConfig,
    goal_real: float,
    confidence: float = 0.8,
    sc: ScenarioSet | None = None,
    tol: float = 0.005,
    max_iter: int = 12,
) -> dict[str, float]:
    """Monthly savings (today's EUR) needed so that P(real net wealth >= goal) = ``confidence``.

    Wealth is close to linear in the savings rate, so a secant iteration on the
    (1 - confidence) quantile converges in a few steps.
    """
    sc = sc if sc is not None else generate(cfg)
    q = 100 * (1 - confidence)

    def quantile(monthly: float) -> float:
        c = cfg.model_copy(update={"plan": cfg.plan.model_copy(update={"monthly": monthly})})
        return float(np.percentile(simulate(c, sc).real_final_net(), q))

    s0 = max(cfg.plan.monthly, 50.0)
    q0 = quantile(s0)
    if q0 <= 0:
        raise ValueError("savings plan produces no wealth")
    s1 = max(s0 * goal_real / q0, 1.0)
    q1 = quantile(s1)
    for _ in range(max_iter):
        if abs(q1 - goal_real) <= tol * goal_real or q1 == q0:
            break
        s0, s1 = s1, max(s1 + (goal_real - q1) * (s1 - s0) / (q1 - q0), 0.0)
        q0, q1 = q1, quantile(s1)
    return {"monthly": s1, "quantile_wealth": q1, "confidence": confidence, "goal": goal_real}
