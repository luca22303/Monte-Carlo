"""Scenario container shared by all return generators."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ScenarioSet:
    """Simulated market paths, monthly steps.

    Attributes
    ----------
    returns:    [P, T, A] nominal simple EUR returns per month, *gross* of fund costs.
    inflation:  [P, T] monthly simple CPI inflation.
    long_yield: [P, T] annual nominal long yield at the start of each month (drives the
                Basiszins for the German Vorabpauschale).
    regimes:    [P, T] regime index per month (parametric generator only).
    """

    asset_keys: list[str]
    returns: np.ndarray
    inflation: np.ndarray
    long_yield: np.ndarray
    regimes: np.ndarray | None = None
    regime_names: list[str] = field(default_factory=list)
    source: str = ""

    def __post_init__(self) -> None:
        p, t, a = self.returns.shape
        if a != len(self.asset_keys):
            raise ValueError("returns last axis must match asset_keys")
        for name in ("inflation", "long_yield"):
            if getattr(self, name).shape != (p, t):
                raise ValueError(f"{name} must have shape {(p, t)}")
        if not np.isfinite(self.returns).all():
            raise ValueError("non-finite returns in scenario set")
        if (self.returns <= -1.0).any():
            raise ValueError("returns <= -100% in scenario set")

    @property
    def n_paths(self) -> int:
        return self.returns.shape[0]

    @property
    def n_months(self) -> int:
        return self.returns.shape[1]

    def cpi(self) -> np.ndarray:
        """[P, T] price index at the *end* of each month (1.0 = today)."""
        return np.cumprod(1.0 + self.inflation, axis=1)

    def subset(self, n_paths: int) -> ScenarioSet:
        """First ``n_paths`` paths (keeps common random numbers when comparing)."""
        regimes = None if self.regimes is None else self.regimes[:n_paths]
        return ScenarioSet(self.asset_keys, self.returns[:n_paths], self.inflation[:n_paths],
                           self.long_yield[:n_paths], regimes, self.regime_names, self.source)
