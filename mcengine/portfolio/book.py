"""Vectorised position book with FIFO tax lots.

Lots are bucketed by purchase year: bucket 0 holds pre-existing positions (annual review
mode), bucket ``y + 1`` holds everything bought in simulation year ``y``. Lots in one
asset share a price index, so lot value = units * price. Sales consume buckets oldest
first (German law requires FIFO per custody account). Within one year's bucket the cost
basis is averaged; this is the only simplification versus true per-trade FIFO.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SaleResult:
    proceeds: np.ndarray     # [P, A] gross market value sold
    gain: np.ndarray         # [P, A] realised gain (proceeds - cost - Vorabpauschalen already taxed)
    short_gain: np.ndarray   # [P, A] part of ``gain`` from lots held < ~1 year (gold §23 rule)
    fees: np.ndarray         # [P] trading costs


class Book:
    def __init__(self, n_paths: int, n_assets: int, n_years: int, trade_cost: float):
        b = n_years + 1
        self.units = np.zeros((n_paths, n_assets, b))
        self.cost = np.zeros((n_paths, n_assets, b))
        self.vap = np.zeros((n_paths, n_assets, b))  # Vorabpauschalen taxed, deductible on sale
        self.units_tot = np.zeros((n_paths, n_assets))
        self.price = np.ones((n_paths, n_assets))
        self.prorated = np.zeros((n_paths, n_assets))  # VAP-weighted units bought this year
        self.tc = trade_cost

    # ------------------------------------------------------------------ state
    def values(self) -> np.ndarray:
        return self.units_tot * self.price

    def total(self) -> np.ndarray:
        return self.values().sum(axis=1)

    def unrealised_gain(self) -> np.ndarray:
        """[P, A] market value minus remaining cost basis (incl. taxed Vorabpauschalen)."""
        return self.values() - self.cost.sum(axis=2) - self.vap.sum(axis=2)

    def seed_positions(self, values: np.ndarray, cost_basis: np.ndarray) -> None:
        """Pre-existing holdings (review mode): bucket 0, given market value and cost basis."""
        self.units[:, :, 0] += values / self.price
        self.cost[:, :, 0] += cost_basis
        self.units_tot += values / self.price

    # ------------------------------------------------------------------ trades
    def buy(self, amount: np.ndarray, bucket: int, month: int) -> np.ndarray:
        """Invest ``amount`` [P, A] (gross, fees deducted) in ``bucket``; returns fees [P]."""
        fees = amount * self.tc
        u = (amount - fees) / self.price
        self.units[:, :, bucket] += u
        self.cost[:, :, bucket] += amount  # acquisition costs are part of the tax basis
        self.units_tot += u
        self.prorated += u * (12 - month) / 12.0
        return fees.sum(axis=1)

    def sell(self, amount: np.ndarray, bucket: int, idx: np.ndarray | None = None) -> SaleResult:
        """Sell market value ``amount`` FIFO. ``bucket`` is the current year's bucket.

        With ``idx`` (path indices), ``amount`` is [len(idx), A] and only those paths trade;
        the result arrays are then also restricted to ``idx``.
        """
        sl = slice(None) if idx is None else idx
        units, cost, vap_acc, price = self.units[sl], self.cost[sl], self.vap[sl], self.price[sl]
        units_sell = np.minimum(amount / price, self.units_tot[sl])
        prev = np.cumsum(units, axis=2) - units
        take = np.clip(units_sell[:, :, None] - prev, 0.0, units)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(units > 0, take / units, 0.0)
        basis = frac * cost
        vap = frac * vap_acc
        proceeds_lot = take * price[:, :, None]
        gain_lot = proceeds_lot - basis - vap
        short_from = max(bucket - 1, 1)  # current and previous year's purchases count as short-term
        short_gain = gain_lot[:, :, short_from:].sum(axis=2)

        new_units = units - take
        self.units[sl] = new_units
        self.cost[sl] = cost - basis
        self.vap[sl] = vap_acc - vap
        self.units_tot[sl] = new_units.sum(axis=2)
        self.prorated[sl] = self.prorated[sl] * (1.0 - frac[:, :, bucket])

        proceeds = proceeds_lot.sum(axis=2)
        fees = (proceeds * self.tc).sum(axis=1)
        return SaleResult(proceeds, gain_lot.sum(axis=2), short_gain, fees)

    def buy_at(self, idx: np.ndarray, amount: np.ndarray, bucket: int, month: int) -> np.ndarray:
        """``buy`` restricted to paths ``idx`` (``amount`` is [len(idx), A]); returns fees [len(idx)]."""
        fees = amount * self.tc
        u = (amount - fees) / self.price[idx]
        self.units[idx, :, bucket] += u
        self.cost[idx, :, bucket] += amount
        self.units_tot[idx] += u
        self.prorated[idx] += u * (12 - month) / 12.0
        return fees.sum(axis=1)

    # ------------------------------------------------------------------ returns
    def apply_returns(self, gross: np.ndarray, ter_monthly: np.ndarray) -> None:
        self.price *= 1.0 + gross - ter_monthly
