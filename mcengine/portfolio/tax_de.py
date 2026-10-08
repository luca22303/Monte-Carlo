"""German taxation of a private investor's fund portfolio (Abgeltungsteuer regime).

Implemented rules:

* Flat 25 % + 5.5 % Soli (+ optional Kirchensteuer, §32d EStG) on capital income.
* Teilfreistellung (InvStG §20): 30 % of fund gains and losses from equity funds are exempt,
  15 % for mixed funds, 0 % for bond/money-market funds.
* Vorabpauschale (InvStG §18) for accumulating funds: per unit,
  ``min(price_start * Basiszins * 0.7, max(0, price_end - price_start))``, reduced by 1/12 for
  each full month before purchase in the year of purchase. Taxed in January of the following
  year, then deducted from the gain when the fund is sold.
* Sparerpauschbetrag (EUR 1,000 single / 2,000 joint) and a loss carry-forward (Verlustvortrag).
* Interest (cash / Tagesgeld) is taxed every year as it accrues.
* Physical gold ETCs with delivery claims (e.g. Xetra-Gold, EUWAX Gold II) are treated as
  private sales under §23 EStG: tax-free after one year, otherwise taxed at the personal
  rate, with a EUR 1,000 Freigrenze. This follows BFH VIII R 4/15 and VIII R 19/22. Check
  it for your specific product.

Out of scope: distributing funds (the model assumes accumulating ETFs), the Guenstigerpruefung,
the separate stock-loss pot (not relevant for ETFs) and inheritance.
"""

from __future__ import annotations

import numpy as np

from ..config import FUND_CLASSES, PARTIAL_EXEMPTION, AssetSpec, TaxDE
from .book import Book, SaleResult

VAP_FACTOR = 0.7


class TaxLedger:
    def __init__(self, n_paths: int, assets: list[AssetSpec], tax: TaxDE):
        self.tax = tax
        self.rate = tax.flat_rate
        cls = [a.tax_class for a in assets]
        self.exempt = np.array([PARTIAL_EXEMPTION[c] for c in cls])
        self.is_fund = np.array([c in FUND_CLASSES for c in cls])
        self.is_gold = np.array([c == "gold_etc" for c in cls])
        self.is_interest = np.array([c == "interest" for c in cls])
        self.base = np.zeros(n_paths)        # Abgeltungsteuer base of the running year
        self.gold_short = np.zeros(n_paths)  # §23 gains of the running year
        self.loss_pot = np.zeros(n_paths)
        self.vap_total = np.zeros(n_paths)   # diagnostics: cumulative Vorabpauschale (pre-exemption)

    def add_sale(self, sale: SaleResult, idx: np.ndarray | None = None) -> None:
        """Book realised gains (``idx`` = path indices when the sale covered a subset)."""
        sl = slice(None) if idx is None else idx
        flat = np.where(self.is_gold, 0.0, sale.gain * (1.0 - self.exempt))
        self.base[sl] += flat.sum(axis=1)
        self.gold_short[sl] += (sale.short_gain * self.is_gold).sum(axis=1)

    def year_end(self, book: Book, price_soy: np.ndarray, basiszins: np.ndarray, bucket: int) -> None:
        """Book the Vorabpauschale and interest income for the year ending now."""
        if self.is_fund.any():
            cap = price_soy * (np.maximum(basiszins, 0.0) * VAP_FACTOR)[:, None]
            per_unit = np.clip(book.price - price_soy, 0.0, cap) * self.is_fund  # [P, A]
            units_eff = book.units.copy()
            units_eff[:, :, bucket] = book.prorated
            vap_lot = units_eff * per_unit[:, :, None]
            book.vap += vap_lot
            vap = vap_lot.sum(axis=2)
            self.vap_total += vap.sum(axis=1)
            self.base += (vap * (1.0 - self.exempt)).sum(axis=1)
        if self.is_interest.any():
            value_lot = book.units * book.price[:, :, None]
            gain_lot = (value_lot - book.cost - book.vap) * self.is_interest[None, :, None]
            self.base += gain_lot.sum(axis=(1, 2))
            mask = self.is_interest[None, :, None]
            book.cost = np.where(mask, value_lot, book.cost)
            book.vap = np.where(mask, 0.0, book.vap)

    def settle(self) -> np.ndarray:
        """Close the tax year; returns the tax owed per path and resets the year's counters."""
        net = self.base - self.loss_pot
        self.loss_pot = np.maximum(-net, 0.0)
        taxable = np.maximum(net - self.tax.allowance, 0.0)
        tax = taxable * self.rate
        g = self.gold_short
        tax += np.where(g >= self.tax.gold_exemption_limit, g * self.tax.personal_rate, 0.0)
        self.base[:] = 0.0
        self.gold_short[:] = 0.0
        return tax
