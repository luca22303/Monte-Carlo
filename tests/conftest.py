import numpy as np
import pytest

from mcengine.config import CostSpec, SavingsPlan, SimConfig, TaxDE
from mcengine.scenarios.base import ScenarioSet


def flat_scenarios(cfg: SimConfig, monthly_returns, inflation: float = 0.0, long_yield: float = 0.0,
                   n_paths: int = 4) -> ScenarioSet:
    """Deterministic scenarios: constant monthly return per asset (scalar or [A])."""
    t, a = cfg.months, len(cfg.assets)
    r = np.broadcast_to(np.asarray(monthly_returns, dtype=float), (a,))
    return ScenarioSet(
        asset_keys=cfg.asset_keys,
        returns=np.broadcast_to(r, (n_paths, t, a)).astype(np.float64).copy(),
        inflation=np.full((n_paths, t), inflation),
        long_yield=np.full((n_paths, t), long_yield),
    )


@pytest.fixture
def frictionless() -> SimConfig:
    """Five-asset default universe without costs or taxes, 10 years, 1000/month."""
    cfg = SimConfig(
        plan=SavingsPlan(monthly=1000, horizon_years=10, growth="none"),
        costs=CostSpec(trade_bps=0),
        tax=TaxDE(enabled=False),
        n_paths=100,
    )
    for a in cfg.assets:
        a.ter = 0.0
    return cfg
