"""Monthly portfolio simulation of a savings plan over a ``ScenarioSet``.

Each month runs these steps in order:

1. January only: pay last year's tax bill out of this month's savings budget. Money is
   sold pro rata if the budget is too small.
2. Invest the savings according to the strategy, paying trading costs.
3. Apply the market returns, net of fund costs (TER).
4. Rebalance if the strategy says so. Sales are FIFO and their gains are booked to the tax ledger.
5. December only: book the Vorabpauschale and interest, then settle the tax year. In the
   final month, liquidate everything and tax the result.

The investor's out-of-pocket budget is the savings plan. Taxes are paid from that budget,
so every strategy run on the same scenarios has the same cash outlay, which makes
comparisons fair.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import SimConfig
from ..scenarios.base import ScenarioSet
from .book import Book
from .strategies import allocate, rebalance_mask, trades_to_target
from .tax_de import TaxLedger


@dataclass
class SimResult:
    wealth: np.ndarray          # [P, T] nominal market value at month end (before latent tax)
    cpi: np.ndarray             # [P, T] price index at month end (1 = today)
    contributions: np.ndarray   # [P, T] out-of-pocket savings (start capital in month 0)
    nav: np.ndarray             # [P, T] time-weighted (unitised) portfolio index, starts at 1
    taxes: np.ndarray           # [P, T] taxes paid
    final_net: np.ndarray       # [P] nominal wealth after liquidation and final tax
    fees: np.ndarray            # [P] cumulative trading costs
    turnover: np.ndarray        # [P] cumulative gross sales for rebalancing
    rebalances: np.ndarray      # [P] number of rebalancing trades
    vap_total: np.ndarray       # [P] cumulative Vorabpauschale booked (pre-exemption)
    weights_end: np.ndarray     # [P, A] final weights
    unrealised_end: np.ndarray  # [P, A] latent gains before liquidation
    asset_keys: list[str] = field(default_factory=list)
    label: str = ""

    @property
    def n_paths(self) -> int:
        return self.wealth.shape[0]

    @property
    def months(self) -> int:
        return self.wealth.shape[1]

    def real_wealth(self) -> np.ndarray:
        return self.wealth / self.cpi

    def real_final_net(self) -> np.ndarray:
        return self.final_net / self.cpi[:, -1]


def contribution_schedule(cfg: SimConfig, cpi_start: np.ndarray, t: int) -> np.ndarray:
    """Nominal savings in month ``t`` for each path. ``cpi_start`` is the CPI level at month start."""
    plan = cfg.plan
    yr = t // 12
    match plan.growth:
        case "inflation":
            amt = plan.monthly * cpi_start
        case "fixed":
            amt = np.full_like(cpi_start, plan.monthly * (1.0 + plan.growth_rate) ** yr)
        case _:
            amt = np.full_like(cpi_start, plan.monthly)
    if t == 0:
        amt = amt + plan.start_capital
    return amt


def simulate(
    cfg: SimConfig,
    sc: ScenarioSet,
    initial_values: np.ndarray | None = None,
    initial_basis: np.ndarray | None = None,
    label: str | None = None,
) -> SimResult:
    """Run the savings plan and strategy in ``cfg`` over the scenarios ``sc``.

    ``initial_values`` / ``initial_basis`` ([A] arrays) seed existing holdings (annual review).
    """
    if sc.asset_keys != cfg.asset_keys:
        raise ValueError("scenario assets do not match config assets")
    p, t_total = sc.n_paths, min(sc.n_months, cfg.months)
    n_assets = len(cfg.assets)
    years = -(-t_total // 12)
    w = cfg.weight_vector()
    spec = cfg.strategy
    ter_m = np.array([a.ter for a in cfg.assets]) / 12.0
    tax_on = cfg.tax.enabled

    book = Book(p, n_assets, years, cfg.costs.trade_bps / 1e4)
    ledger = TaxLedger(p, cfg.assets, cfg.tax)
    if initial_values is not None:
        basis = initial_values if initial_basis is None else initial_basis
        book.seed_positions(np.broadcast_to(initial_values, (p, n_assets)).astype(float),
                            np.broadcast_to(basis, (p, n_assets)).astype(float))

    cpi = sc.cpi()[:, :t_total]
    wealth = np.empty((p, t_total))
    nav = np.empty((p, t_total), dtype=np.float32)
    contrib = np.empty((p, t_total), dtype=np.float32)
    taxes = np.zeros((p, t_total), dtype=np.float32)
    fees = np.zeros(p)
    turnover = np.zeros(p)
    n_reb = np.zeros(p, dtype=np.int32)
    final_net = np.zeros(p)

    tax_due = np.zeros(p)
    price_soy = book.price.copy()
    basiszins = np.zeros(p)
    nav_level = np.ones(p)
    w_prev = book.total()

    for t in range(t_total):
        yr, m = divmod(t, 12)
        bucket = yr + 1
        if m == 0:
            price_soy = book.price.copy()
            book.prorated[:] = 0.0
            basiszins = (np.full(p, cfg.tax.basiszins) if cfg.tax.basiszins is not None
                         else np.maximum(sc.long_yield[:, t], 0.0))

        cpi_start = cpi[:, t - 1] if t > 0 else np.ones(p)
        c = contribution_schedule(cfg, cpi_start, t)
        contrib[:, t] = c
        invest = c - tax_due
        taxes[:, t] = tax_due
        tax_due = np.zeros(p)

        h = book.values()
        pos = np.maximum(invest, 0.0)
        if (pos > 0).any():
            fees += book.buy(allocate(spec, h, pos, w), bucket, m)
        neg = np.maximum(-invest, 0.0)
        if (neg > 0).any():  # tax bill larger than the month's savings: sell pro rata
            tot = h.sum(axis=1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                share = np.where(tot > 0, h / tot, 0.0)
            sale = book.sell(share * neg[:, None], bucket)
            fees += sale.fees
            if tax_on:
                ledger.add_sale(sale)
        v_pre = w_prev + invest

        book.apply_returns(sc.returns[:, t, :], ter_m)

        if t < t_total - 1:
            h = book.values()
            mask = rebalance_mask(spec, h, w, t)
            if mask is not None:
                idx = np.flatnonzero(mask)
                sells, buys = trades_to_target(h[idx], w)
                sale = book.sell(sells, bucket, idx)
                cash = sale.proceeds.sum(axis=1) - sale.fees
                buy_tot = buys.sum(axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    scale = np.where(buy_tot > 0, cash / buy_tot, 0.0)
                fees[idx] += sale.fees + book.buy_at(idx, buys * scale[:, None], bucket, m)
                turnover[idx] += sale.proceeds.sum(axis=1)
                n_reb[idx] += 1
                if tax_on:
                    ledger.add_sale(sale, idx)

        w_now = book.total()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(v_pre > 1e-9, w_now / v_pre, 1.0)
        nav_level *= ratio
        nav[:, t] = nav_level
        wealth[:, t] = w_now
        w_prev = w_now

        last = t == t_total - 1
        if m == 11 or last:
            if last:
                unreal = book.unrealised_gain()
                weights_end = book.values() / np.maximum(w_now, 1e-12)[:, None]
                if tax_on:
                    ledger.add_sale(book.sell(book.values(), bucket))  # liquidation (fees ignored)
                    final_tax = ledger.settle()
                else:
                    final_tax = np.zeros(p)
                final_net = w_now - final_tax
                taxes[:, t] += final_tax
            elif tax_on:
                ledger.year_end(book, price_soy, basiszins, bucket)
                tax_due = ledger.settle()

    return SimResult(
        wealth=wealth, cpi=cpi, contributions=contrib, nav=nav, taxes=taxes, final_net=final_net,
        fees=fees, turnover=turnover, rebalances=n_reb, vap_total=ledger.vap_total,
        weights_end=weights_end, unrealised_end=unreal, asset_keys=cfg.asset_keys,
        label=label or spec.label(),
    )
