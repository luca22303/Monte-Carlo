import numpy as np
import pytest
from conftest import flat_scenarios

from mcengine.config import AssetSpec, CostSpec, SavingsPlan, SimConfig, StrategySpec, TaxDE
from mcengine.portfolio.book import Book
from mcengine.portfolio.simulator import simulate
from mcengine.portfolio.tax_de import TaxLedger

RATE = 0.26375
EQ = AssetSpec(key="equity", name="Equity ETF", kind="equity", ter=0.0, tax_class="equity_fund")
BOND = AssetSpec(key="bonds", name="Bond ETF", kind="nominal_bond", ter=0.0, tax_class="bond_fund")
GOLD = AssetSpec(key="gold", name="Gold ETC", kind="gold", ter=0.0, tax_class="gold_etc")
CASH = AssetSpec(key="cash", name="Tagesgeld", kind="cash", ter=0.0, tax_class="interest")


def test_flat_rates():
    assert TaxDE().flat_rate == pytest.approx(RATE)
    assert TaxDE(church_tax=0.09).flat_rate == pytest.approx(0.27995, abs=1e-5)
    assert TaxDE(church_tax=0.08).flat_rate == pytest.approx(0.27819, abs=1e-5)


def _book_with_purchase(month: int, end_price: float):
    book = Book(1, 1, 2, 0.0)
    book.buy(np.array([[1000.0]]), bucket=1, month=month)
    soy = np.ones((1, 1))
    book.price[:] = end_price
    ledger = TaxLedger(1, [EQ], TaxDE())
    ledger.year_end(book, soy, np.array([0.0255]), bucket=1)
    return book, ledger


def test_vorabpauschale_prorated_in_purchase_year():
    # Bought in March: Jan + Feb (2 full months) reduce the VAP by 2/12.
    book, ledger = _book_with_purchase(month=2, end_price=1.10)
    vap = 1000 * 1.0 * 0.0255 * 0.7 * 10 / 12
    assert book.vap.sum() == pytest.approx(vap)
    assert ledger.base[0] == pytest.approx(vap * 0.7)  # 30 % Teilfreistellung


def test_vorabpauschale_capped_by_gain_and_zero_on_loss():
    book, _ = _book_with_purchase(month=0, end_price=1.01)
    assert book.vap.sum() == pytest.approx(1000 * 0.01)
    book, ledger = _book_with_purchase(month=0, end_price=0.95)
    assert book.vap.sum() == 0.0 and ledger.base[0] == 0.0


def test_fifo_sale():
    book = Book(1, 1, 3, 0.0)
    book.buy(np.array([[100.0]]), bucket=1, month=0)   # 100 units @ 1
    book.price[:] = 2.0
    book.buy(np.array([[200.0]]), bucket=2, month=0)   # 100 units @ 2
    book.price[:] = 3.0
    sale = book.sell(np.array([[450.0]]), bucket=2)   # 150 units: all of lot 1, half of lot 2
    assert sale.gain[0, 0] == pytest.approx((300 - 100) + (150 - 100))
    assert book.units_tot[0, 0] == pytest.approx(50)
    assert book.cost.sum() == pytest.approx(100)


def test_vorabpauschale_reduces_later_gain():
    book, ledger = _book_with_purchase(month=0, end_price=1.10)
    vap = book.vap.sum()
    sale = book.sell(np.array([[1100.0]]), bucket=2)
    assert sale.gain[0, 0] == pytest.approx(100 - vap)


def test_allowance_and_loss_carryforward():
    ledger = TaxLedger(1, [EQ], TaxDE())
    ledger.base[:] = 800
    assert ledger.settle()[0] == 0
    ledger.base[:] = 1500
    assert ledger.settle()[0] == pytest.approx(500 * RATE)
    ledger.base[:] = -300
    assert ledger.settle()[0] == 0 and ledger.loss_pot[0] == 300
    ledger.base[:] = 1500
    assert ledger.settle()[0] == pytest.approx(200 * RATE)


def _single(asset: AssetSpec, years: int = 1, capital: float = 100_000.0, **tax) -> SimConfig:
    return SimConfig(
        assets=[asset], weights={asset.key: 1.0}, strategy=StrategySpec(kind="buy_and_hold"),
        plan=SavingsPlan(monthly=0, start_capital=capital, horizon_years=years, growth="none"),
        costs=CostSpec(trade_bps=0), tax=TaxDE(**tax), n_paths=100,
    )


def test_liquidation_tax_equity_fund():
    cfg = _single(EQ, basiszins=0.0)
    res = simulate(cfg, flat_scenarios(cfg, 0.01, n_paths=2))
    gain = 100_000 * (1.01**12 - 1)
    tax = (0.7 * gain - 1000) * RATE
    np.testing.assert_allclose(res.final_net, 100_000 + gain - tax, rtol=1e-9)


def test_vorabpauschale_paid_each_january():
    cfg = _single(BOND, years=2, basiszins=0.03, allowance=0.0)
    res = simulate(cfg, flat_scenarios(cfg, 0.005, n_paths=2))
    vap = 100_000 * 0.03 * 0.7  # < realised gain of the year, so not capped
    assert res.taxes[0, 12] == pytest.approx(vap * RATE, rel=1e-9)


def test_gold_tax_free_after_one_year():
    cfg = _single(GOLD, years=3)
    res = simulate(cfg, flat_scenarios(cfg, 0.01, n_paths=2))
    assert res.taxes.sum() == 0
    np.testing.assert_allclose(res.final_net, res.wealth[:, -1])


def test_gold_short_term_taxed_at_personal_rate():
    cfg = _single(GOLD, years=1)
    res = simulate(cfg, flat_scenarios(cfg, 0.01, n_paths=2))
    gain = 100_000 * (1.01**12 - 1)
    np.testing.assert_allclose(res.taxes[:, -1], gain * 0.35, rtol=1e-6)


def test_interest_taxed_annually():
    cfg = _single(CASH, years=2)
    res = simulate(cfg, flat_scenarios(cfg, 0.002, n_paths=2))
    interest = 100_000 * (1.002**12 - 1)
    assert res.taxes[0, 12] == pytest.approx((interest - 1000) * RATE, rel=1e-9)


def test_taxes_reduce_wealth_vs_untaxed():
    base = SimConfig(n_paths=200, plan=SavingsPlan(horizon_years=10))
    from mcengine.scenarios import generate

    sc = generate(base)
    taxed = simulate(base, sc)
    untaxed = simulate(base.model_copy(update={"tax": TaxDE(enabled=False)}), sc)
    assert (taxed.final_net <= untaxed.final_net + 1e-6).mean() > 0.99
