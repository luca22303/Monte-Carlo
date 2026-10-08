"""Configuration models for the engine.

All rates are annual decimals (0.03 == 3 % p.a.) unless a field name says otherwise.
Everything a simulation depends on lives in one ``SimConfig`` so that a run is fully
reproducible from its JSON (``SimConfig.model_dump_json()``).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

# Risk factors driving the parametric generator (order matters: correlation matrices use it).
FACTORS = ("equity", "gold", "nominal_yield", "real_yield", "short_rate", "inflation")
N_FACTORS = len(FACTORS)

AssetKind = Literal["equity", "nominal_bond", "linker", "gold", "cash"]
TaxClass = Literal["equity_fund", "mixed_fund", "bond_fund", "gold_etc", "interest"]

# Teilfreistellung (InvStG §20) per tax class.
PARTIAL_EXEMPTION: dict[str, float] = {
    "equity_fund": 0.30,
    "mixed_fund": 0.15,
    "bond_fund": 0.0,
    "gold_etc": 0.0,
    "interest": 0.0,
}
FUND_CLASSES = ("equity_fund", "mixed_fund", "bond_fund")


class AssetSpec(BaseModel):
    key: str
    name: str
    kind: AssetKind
    ter: float = Field(0.002, ge=0, le=0.05, description="Annual running cost (TER / custody)")
    tax_class: TaxClass
    duration: float = Field(0.0, ge=0, description="Modified duration (bonds/linkers)")
    convexity: float = Field(0.0, ge=0)


def default_assets() -> list[AssetSpec]:
    return [
        AssetSpec(key="equity", name="Global equity (MSCI ACWI/World, EUR unhedged)", kind="equity",
                  ter=0.0020, tax_class="equity_fund"),
        AssetSpec(key="gov_bonds", name="EUR government bonds (all maturities)", kind="nominal_bond",
                  ter=0.0015, tax_class="bond_fund", duration=7.0, convexity=70.0),
        AssetSpec(key="linkers", name="EUR inflation-linked govt bonds", kind="linker",
                  ter=0.0020, tax_class="bond_fund", duration=7.5, convexity=80.0),
        AssetSpec(key="gold", name="Physical gold ETC (EUR)", kind="gold",
                  ter=0.0015, tax_class="gold_etc"),
        AssetSpec(key="cash", name="Cash / Tagesgeld", kind="cash",
                  ter=0.0, tax_class="interest"),
    ]


class RegimeSpec(BaseModel):
    """One state of the Markov regime-switching model.

    ``vol`` and ``drift`` are annualised, keyed by factor name. Equity/gold vol is return
    volatility; for yields/inflation it is the volatility of *changes* in the rate
    (0.007 == 70 bp p.a.). ``drift`` for equity/gold is an annual log-return offset and
    for rates an annual change in the rate; drifts are re-centred so the long-run averages
    in :class:`MarketModel` are preserved (regimes shape the distribution, not the mean).
    """

    name: str
    vol: dict[str, float]
    drift: dict[str, float]
    corr: list[list[float]]
    df: float = Field(8.0, gt=2.0, description="Student-t degrees of freedom (fat tails); >=100 ~ normal")

    @field_validator("vol", "drift")
    @classmethod
    def _factors_complete(cls, v: dict[str, float]) -> dict[str, float]:
        missing = set(FACTORS) - set(v)
        if missing:
            raise ValueError(f"missing factors: {sorted(missing)}")
        return v

    @field_validator("corr")
    @classmethod
    def _corr_shape(cls, v: list[list[float]]) -> list[list[float]]:
        m = np.asarray(v, dtype=float)
        if m.shape != (N_FACTORS, N_FACTORS):
            raise ValueError(f"corr must be {N_FACTORS}x{N_FACTORS}")
        if not np.allclose(m, m.T, atol=1e-9) or not np.allclose(np.diag(m), 1.0):
            raise ValueError("corr must be symmetric with unit diagonal")
        return v


def _corr(pairs: dict[tuple[str, str], float]) -> list[list[float]]:
    m = np.eye(N_FACTORS)
    for (a, b), c in pairs.items():
        i, j = FACTORS.index(a), FACTORS.index(b)
        m[i, j] = m[j, i] = c
    return m.tolist()


def default_regimes() -> list[RegimeSpec]:
    calm = RegimeSpec(
        name="Calm / expansion",
        vol={"equity": 0.11, "gold": 0.13, "nominal_yield": 0.005, "real_yield": 0.005,
             "short_rate": 0.004, "inflation": 0.006},
        drift={"equity": 0.0, "gold": 0.0, "nominal_yield": 0.0, "real_yield": 0.0,
               "short_rate": 0.0, "inflation": 0.0},
        corr=_corr({
            ("equity", "gold"): 0.05, ("equity", "nominal_yield"): 0.20, ("equity", "real_yield"): 0.15,
            ("equity", "short_rate"): 0.05, ("equity", "inflation"): 0.0,
            ("gold", "nominal_yield"): -0.10, ("gold", "real_yield"): -0.30, ("gold", "short_rate"): -0.05,
            ("gold", "inflation"): 0.10,
            ("nominal_yield", "real_yield"): 0.75, ("nominal_yield", "short_rate"): 0.50,
            ("nominal_yield", "inflation"): 0.20,
            ("real_yield", "short_rate"): 0.40, ("real_yield", "inflation"): -0.10,
            ("short_rate", "inflation"): 0.20,
        }),
        df=8.0,
    )
    crisis = RegimeSpec(
        name="Deflationary crisis (2008-type)",
        vol={"equity": 0.25, "gold": 0.20, "nominal_yield": 0.009, "real_yield": 0.010,
             "short_rate": 0.010, "inflation": 0.010},
        drift={"equity": -0.30, "gold": 0.05, "nominal_yield": -0.010, "real_yield": 0.0,
               "short_rate": -0.015, "inflation": -0.015},
        corr=_corr({
            ("equity", "gold"): -0.10, ("equity", "nominal_yield"): 0.50, ("equity", "real_yield"): 0.30,
            ("equity", "short_rate"): 0.30, ("equity", "inflation"): 0.30,
            ("gold", "nominal_yield"): -0.30, ("gold", "real_yield"): -0.40, ("gold", "short_rate"): -0.10,
            ("gold", "inflation"): 0.0,
            ("nominal_yield", "real_yield"): 0.70, ("nominal_yield", "short_rate"): 0.60,
            ("nominal_yield", "inflation"): 0.40,
            ("real_yield", "short_rate"): 0.30, ("real_yield", "inflation"): -0.20,
            ("short_rate", "inflation"): 0.30,
        }),
        df=5.0,
    )
    inflation = RegimeSpec(
        name="Inflation shock (1970s / 2022-type)",
        vol={"equity": 0.18, "gold": 0.18, "nominal_yield": 0.010, "real_yield": 0.008,
             "short_rate": 0.008, "inflation": 0.015},
        drift={"equity": -0.08, "gold": 0.10, "nominal_yield": 0.012, "real_yield": 0.008,
               "short_rate": 0.018, "inflation": 0.035},
        corr=_corr({
            ("equity", "gold"): 0.10, ("equity", "nominal_yield"): -0.50, ("equity", "real_yield"): -0.40,
            ("equity", "short_rate"): -0.30, ("equity", "inflation"): -0.40,
            ("gold", "nominal_yield"): 0.0, ("gold", "real_yield"): -0.20, ("gold", "short_rate"): 0.0,
            ("gold", "inflation"): 0.30,
            ("nominal_yield", "real_yield"): 0.80, ("nominal_yield", "short_rate"): 0.60,
            ("nominal_yield", "inflation"): 0.50,
            ("real_yield", "short_rate"): 0.50, ("real_yield", "inflation"): 0.0,
            ("short_rate", "inflation"): 0.40,
        }),
        df=6.0,
    )
    return [calm, crisis, inflation]


class MarketModel(BaseModel):
    """Capital-market assumptions for the parametric generator.

    Expected equity/gold returns are long-run *geometric* (compound) nominal EUR returns,
    gross of fund costs. Bond returns are not set directly: they follow from the yield
    paths, so starting yields matter exactly as they do in reality.
    """

    equity_return: float = Field(0.060, description="Geometric nominal EUR return p.a., gross of TER")
    gold_return: float = Field(0.035, description="Geometric nominal EUR return p.a.")

    nominal_yield_start: float = 0.030
    nominal_yield_mean: float = 0.029
    nominal_yield_kappa: float = Field(0.15, ge=0, description="Mean reversion speed p.a.")

    real_yield_start: float = 0.008
    real_yield_mean: float = 0.008
    real_yield_kappa: float = Field(0.15, ge=0)

    short_rate_start: float = 0.022
    short_rate_mean: float = 0.020
    short_rate_kappa: float = Field(0.25, ge=0)
    cash_floor: float = Field(0.0, description="Lower bound of the cash rate (Tagesgeld rarely < 0)")

    inflation_start: float = 0.021
    inflation_mean: float = 0.020
    inflation_kappa: float = Field(0.50, ge=0)

    regimes: list[RegimeSpec] = Field(default_factory=default_regimes)
    transition: list[list[float]] = Field(
        default_factory=lambda: [
            [0.980, 0.012, 0.008],
            [0.150, 0.850, 0.000],
            [0.070, 0.000, 0.930],
        ],
        description="Monthly regime transition matrix (rows sum to 1)",
    )
    start_regime: Literal["stationary", "calm"] = "stationary"

    @model_validator(mode="after")
    def _check_transition(self) -> MarketModel:
        p = np.asarray(self.transition, dtype=float)
        k = len(self.regimes)
        if p.shape != (k, k):
            raise ValueError(f"transition must be {k}x{k}")
        if (p < 0).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("transition rows must be probabilities summing to 1")
        return self

    def stationary_distribution(self) -> np.ndarray:
        p = np.asarray(self.transition, dtype=float)
        vals, vecs = np.linalg.eig(p.T)
        v = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
        return v / v.sum()


StrategyKind = Literal["buy_and_hold", "calendar", "band", "cashflow", "hybrid"]


class StrategySpec(BaseModel):
    """Rebalancing policy. Target weights live in :class:`SimConfig`.

    * ``buy_and_hold`` - savings invested at target weights, never rebalanced.
    * ``calendar``     - full rebalance to target every ``rebalance_months``.
    * ``band``         - full rebalance whenever a weight leaves its tolerance band
                         ``max(abs_band, rel_band * target)``.
    * ``cashflow``     - savings are routed to underweight assets; never sells.
    * ``hybrid``       - cashflow steering plus a band rebalance with (wider) bands.
    """

    kind: StrategyKind = "hybrid"
    rebalance_months: int = Field(12, ge=1, le=120)
    abs_band: float = Field(0.10, ge=0, le=1)
    rel_band: float = Field(0.50, ge=0, le=5)

    def label(self) -> str:
        match self.kind:
            case "calendar":
                return f"Calendar ({self.rebalance_months}m)"
            case "band" | "hybrid":
                return f"{self.kind.capitalize()} ({self.abs_band:.0%} / {self.rel_band:.0%} bands)"
            case "cashflow":
                return "Cash-flow only"
            case _:
                return "Buy & hold"


class SavingsPlan(BaseModel):
    monthly: float = Field(1000.0, ge=0, description="Monthly savings in today's EUR")
    start_capital: float = Field(0.0, ge=0)
    horizon_years: int = Field(30, ge=1, le=60)
    growth: Literal["none", "fixed", "inflation"] = Field(
        "inflation", description="How the monthly amount grows: constant nominal, fixed % p.a., or with CPI")
    growth_rate: float = Field(0.02, description="Annual step-up for growth='fixed'")


class CostSpec(BaseModel):
    trade_bps: float = Field(5.0, ge=0, le=200, description="Spread + commission per trade, basis points")


class TaxDE(BaseModel):
    """German flat-tax regime for private investors (Abgeltungsteuer)."""

    enabled: bool = True
    church_tax: float = Field(0.0, description="Kirchensteuer rate: 0, 0.08 (BY/BW) or 0.09")
    allowance: float = Field(1000.0, ge=0, description="Sparerpauschbetrag (2000 for joint filers)")
    basiszins: float | None = Field(
        None, description="Fixed Basiszins for the Vorabpauschale; None = follow the simulated long yield")
    personal_rate: float = Field(0.35, ge=0, le=0.5,
                                 description="Marginal income tax for short-term gold gains (§23 EStG)")
    gold_exemption_limit: float = Field(1000.0, ge=0, description="§23 Freigrenze per year")

    @property
    def flat_rate(self) -> float:
        """Abgeltungsteuer + Soli (+ Kirchensteuer) effective rate."""
        k = self.church_tax
        est = 1.0 / (4.0 + k)  # §32d(1) EStG with church tax deductibility
        return est * (1.0 + 0.055 + k)


Generator = Literal["parametric", "bootstrap"]


class SimConfig(BaseModel):
    assets: list[AssetSpec] = Field(default_factory=default_assets)
    weights: dict[str, float] = Field(
        default_factory=lambda: {"equity": 0.60, "gov_bonds": 0.20, "linkers": 0.10, "gold": 0.05,
                                 "cash": 0.05})
    strategy: StrategySpec = Field(default_factory=StrategySpec)
    plan: SavingsPlan = Field(default_factory=SavingsPlan)
    market: MarketModel = Field(default_factory=MarketModel)
    costs: CostSpec = Field(default_factory=CostSpec)
    tax: TaxDE = Field(default_factory=TaxDE)

    generator: Generator = "parametric"
    bootstrap_block_months: float = Field(24.0, ge=1, description="Mean block length (stationary bootstrap)")
    bootstrap_mean_adjust: bool = Field(
        True, description="Shift historical returns so their means match the forward-looking CMAs")

    n_paths: int = Field(5000, ge=100, le=100_000)
    seed: int = 42
    goal_real: float = Field(500_000.0, ge=0, description="Wealth goal in today's EUR")

    @model_validator(mode="after")
    def _check_weights(self) -> SimConfig:
        keys = [a.key for a in self.assets]
        if len(set(keys)) != len(keys):
            raise ValueError("asset keys must be unique")
        unknown = set(self.weights) - set(keys)
        if unknown:
            raise ValueError(f"weights for unknown assets: {sorted(unknown)}")
        w = self.weight_vector()
        if (w < -1e-12).any():
            raise ValueError("weights must be non-negative")
        if not np.isclose(w.sum(), 1.0, atol=1e-6):
            raise ValueError(f"weights must sum to 1 (got {w.sum():.4f})")
        return self

    @property
    def asset_keys(self) -> list[str]:
        return [a.key for a in self.assets]

    @property
    def months(self) -> int:
        return self.plan.horizon_years * 12

    def weight_vector(self) -> np.ndarray:
        return np.array([self.weights.get(a.key, 0.0) for a in self.assets], dtype=float)

    def equity_share(self) -> float:
        return float(sum(self.weights.get(a.key, 0.0) for a in self.assets if a.kind == "equity"))
