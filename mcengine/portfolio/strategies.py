"""Rebalancing policies.

Each policy answers two questions every month:

1. How is this month's net savings split across assets?  (``allocate``)
2. Should the portfolio be traded back to target at the end of the month?  (``rebalance``)

All functions are vectorised over paths. ``h`` is the [P, A] market value of holdings.
"""

from __future__ import annotations

import numpy as np

from ..config import StrategySpec


def drift_breach(h: np.ndarray, w: np.ndarray, abs_band: float, rel_band: float) -> np.ndarray:
    """[P] True where any weight is outside ``max(abs_band, rel_band * target)``."""
    total = h.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cur = np.where(total > 0, h / total, w)
    tol = np.maximum(abs_band, rel_band * w)
    return (np.abs(cur - w) > tol + 1e-12).any(axis=1)


def fill_underweights(h: np.ndarray, invest: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Allocate ``invest`` [P] to close the gaps to target, without selling.

    If the money covers all gaps, the gaps are closed and the rest is split at target
    weights. Otherwise the money is split in proportion to the gaps.
    """
    total_after = h.sum(axis=1) + invest
    gap = np.maximum(w * total_after[:, None] - h, 0.0)
    gap_sum = gap.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        partial = np.where(gap_sum[:, None] > 0, gap * (invest / gap_sum)[:, None], w * invest[:, None])
    full = gap + w * np.maximum(invest - gap_sum, 0.0)[:, None]
    return np.where((gap_sum <= invest)[:, None], full, partial)


def allocate(spec: StrategySpec, h: np.ndarray, invest: np.ndarray, w: np.ndarray) -> np.ndarray:
    """[P, A] purchase amounts for a non-negative ``invest`` [P]."""
    if spec.kind in ("cashflow", "hybrid"):
        return fill_underweights(h, invest, w)
    return w * invest[:, None]


def rebalance_mask(spec: StrategySpec, h: np.ndarray, w: np.ndarray, month_index: int) -> np.ndarray | None:
    """[P] paths that trade back to target at the end of ``month_index`` (None = no trading)."""
    match spec.kind:
        case "calendar":
            if (month_index + 1) % spec.rebalance_months:
                return None
            return np.ones(h.shape[0], dtype=bool)
        case "band" | "hybrid":
            mask = drift_breach(h, w, spec.abs_band, spec.rel_band)
            return mask if mask.any() else None
        case _:
            return None


def trades_to_target(h: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sell and buy amounts [P, A] that restore target weights."""
    diff = w * h.sum(axis=1, keepdims=True) - h
    return np.maximum(-diff, 0.0), np.maximum(diff, 0.0)
