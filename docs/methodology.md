# Methodology

## 1. Objective

The engine is for **accumulation**: a monthly savings plan into a ~60/40 portfolio for 10–40
years, taxed in Germany. The questions it answers are different from those of a retirement
(decumulation) planner:

| Metric | Why |
|---|---|
| Net wealth in today's € after liquidation and tax (P5…P95, CVaR) | What the money will actually buy |
| P(real net wealth ≥ goal) | Goal-based planning |
| P(real net wealth < real savings paid in) | "Did I lose purchasing power?" |
| Money-weighted real return (IRR) | Return on *your* cash flows, timing included |
| Time-weighted return, volatility, max drawdown, time under water, Ulcer index, share of negative years, CVaR of 12-month returns | How bumpy the ride is, which decides whether people stay invested |
| Taxes, trading costs, turnover | What the rebalancing rule costs |

Every estimate comes with a Monte Carlo standard error (batch means). When comparing
variants, use the paired difference on common scenarios, not two separate runs.

## 2. Scenario generators

All generators produce monthly nominal EUR asset returns gross of fund costs, monthly CPI
inflation, and the long yield (used for the Basiszins).

### 2.1 Parametric: regime-switching, fat-tailed, yield-anchored

Six risk factors: equity log-return, gold log-return, nominal yield, real yield, short rate
and inflation.

* **Regimes.** A Markov chain over *calm* (≈ 84 % of the time, average spell ≈ 4 years),
  *deflationary crisis* (≈ 7 %, ≈ 7 months; equities fall, yields fall, bonds hedge) and
  *inflation shock* (≈ 10 %, ≈ 14 months; inflation and yields rise, so stocks **and** bonds
  fall, as in the 1970s and 2022).
* **Shocks** are multivariate Student-t with a shared mixing variable, which gives tail
  dependence. Volatilities, correlations and tail thickness (df 8 / 5 / 6) depend on the regime.
* **Drift re-centring.** Regime drifts are re-centred so the stationary average equals the
  long-run assumption. Regimes shape the distribution but do not change the expected return.
* **Rates** follow mean-reverting AR(1) processes from today's starting values toward
  long-run means.
* **Asset mapping:**
  * equity and gold: `exp(log-return) − 1`
  * nominal bonds: `y/12 − D·Δy + ½·C·Δy²`
  * linkers: `r/12 + π − D·Δr + ½·C·Δr²`
  * cash: `max(short rate, floor)/12`

  Expected bond returns therefore follow from today's yields, as they do in practice.

Default long-run assumptions (nominal EUR, editable) are deliberately moderate:

| | Default |
|---|---|
| Global equity | 6.0 % geometric (≈ 4 % real) |
| Gold | 3.0 % (≈ 1 % real) |
| EUR govt yield | 3.0 % start, mean 2.9 % |
| Real yield | 0.8 % |
| Short rate | 2.2 % → 2.0 % |
| Inflation | 2.1 % → 2.0 % |

### 2.2 Stationary bootstrap of history

Monthly EUR series from 1973-01 to 2025-12, built by `python -m mcengine.data.build_dataset`:

| Asset | Source |
|---|---|
| Equity | Fama/French US total market (USD) translated to EUR (DEM before 1999). This is a US proxy for world equity; the US is ~65–70 % of MSCI World today. |
| Govt bonds | German 10y yield (OECD via FRED) turned into a constant-duration return |
| Linkers | **Synthetic.** Real yield = 10y yield − trailing 3y inflation. Real EUR linkers only exist from 1998. |
| Gold | USD price (datahub) translated to EUR |
| Cash | German 3m money-market rate |
| Inflation | German CPI (OECD), extended with HICP Germany (ECB) |

The stationary bootstrap (Politis & Romano) uses geometrically distributed blocks with a
24-month mean. It keeps fat tails, volatility clustering and cross-correlations without a
parametric model.

**Mean adjustment** (default on) shifts each series in log space so its mean equals the
parametric model's forward-looking mean. Without it, the bootstrap assumes the 1973–2025
bond bull market and equity boom repeat (≈ 5 % real for a 60/40), which is optimistic
given 3 % starting yields.

### 2.3 Historical rolling windows

Your exact plan is run on every horizon-length window of history that starts in January. With
30-year windows there are only 24, all overlapping and all containing both 2000–03 and 2008.
This is a sanity check, not a distribution.

### 2.4 Calibration against history

| | History 1973–2025 | Parametric model |
|---|---|---|
| Equity volatility (monthly, annualised) | 17.3 % | ≈ 15–16 % |
| Govt bond volatility | 4.9 % | 4.3 % |
| Gold volatility | 15.7 % | 14.1 % |
| Govt bond max drawdown | 20 % (2022) | P50 13 %, P95 24 % over 30y |
| 60/40 (monthly rebalanced) max drawdown | 35 % over 53y | P50 25 %, P95 ≈ 44 % over 30y |

The model's tail drawdowns are somewhat harsher than 1973–2025 history. That is deliberate:
53 years are one sample path, and 1929-type outcomes are not in it.

## 3. Portfolio simulation

The simulation steps monthly. Taxes are paid out of the savings budget of the following
January (sold pro rata if the budget is too small), so every strategy has the same cash
outlay.

**Lots.** Purchases are bucketed by calendar year. Sales consume buckets FIFO, as German law
requires. Within one year's bucket the cost basis is averaged, which is the only deviation
from exact per-trade FIFO.

**Rebalancing policies:**
* **Buy & hold:** savings are invested at target weights and nothing is ever sold.
* **Calendar:** full rebalance every *N* months.
* **Band:** full rebalance when any weight leaves `max(abs, rel × target)`.
* **Cash-flow:** savings go to the underweight assets, closing gaps first. Never sells.
* **Hybrid:** cash-flow steering plus a band rebalance with wide bands (10 % / 50 %) as an
  emergency brake.

### German taxation (`portfolio/tax_de.py`)

* **Flat rate.** Abgeltungsteuer 25 % + Soli 5.5 % = 26.375 %; with Kirchensteuer 27.82 % (8 %)
  or 27.99 % (9 %) per §32d EStG.
* **Teilfreistellung.** 30 % for equity funds, 15 % for mixed funds, 0 % for bond and
  money-market funds. It applies to gains, losses and Vorabpauschalen.
* **Vorabpauschale** (accumulating ETFs). Per unit:
  `min(price₁ₛₜ ⋅ Basiszins ⋅ 0.7, max(0, price_end − price₁ₛₜ))`, reduced by 1/12 per full
  month before purchase in the purchase year. It is taxed in January and deducted from the
  gain on sale.
* **Basiszins.** By default the simulated long yield each January, floored at 0. A fixed
  value is also possible.
* **Allowances and losses.** Sparerpauschbetrag €1,000 / €2,000 (not inflation-indexed, so
  its real value erodes) and a loss carry-forward.
* **Interest.** Cash / Tagesgeld interest is taxed every year.
* **Gold.** Physical gold ETCs with a delivery claim are treated as private sales (§23 EStG):
  tax-free after 1 year, otherwise the personal income-tax rate with a €1,000 Freigrenze.
  This is **an assumption to verify for your product**. The engine conservatively treats
  purchases from the current and previous calendar year as short-term.
* **End of plan.** Everything is liquidated and taxed; results are net of that tax.

Not modelled: distributing funds, Günstigerprüfung, church-tax details beyond the rate, and
inheritance or gifting.

## 4. Decision tools

* **Strategy comparison** runs all rules on the same scenarios and reports the paired
  difference with its standard error.
* **Allocation explorer** is an exhaustive grid over the defensive sleeve at given equity
  levels, with a full after-tax simulation per point. It reports a Pareto frontier for a
  chosen risk metric, not a single "optimal" weight vector.
* **Required savings** uses a secant search on the (1 − confidence) quantile of real net
  wealth. Wealth is nearly linear in the savings rate, so it converges in 3–5 simulations.
* **Annual review** updates the *state* (holdings, cost basis, optionally today's starting
  yields and inflation) and applies the *fixed policy*. It does not re-estimate expected
  returns from recent performance.

## 5. Review of the original research notes (Gemini)

The three supplied documents were a reasonable textbook introduction but needed correcting
in several places:

| Claim in the notes | Assessment | What the engine does |
|---|---|---|
| Plan "success rate / solvency" (target 80–90 %), Guyton-Klinger guardrails, sequence-of-returns risk with withdrawals | Decumulation concepts. For a monthly saver, sequence risk runs the other way: late crashes hurt most, early crashes are buying opportunities. | Goal probability, real-loss probability, money-weighted real return and drawdown metrics |
| Sobol quasi-Monte Carlo gives O(N⁻¹) convergence | Holds for smooth, low-dimensional integrands. Here there are 360 months × 6 factors, regime switches and path-dependent taxes, so the advantage largely disappears. | Seeded PRNG, 5–20k paths, reported standard errors, common random numbers for comparisons |
| GARCH(1,1) for volatility clustering | Valid, but regime switching captures clustering *and* correlation breaks (2008 vs 2022) more transparently | 3-regime Markov switching with Student-t shocks |
| Copulas, VAR macro scenario generators, CUDA | Overkill for one investor's 5-asset portfolio | Not used; 10k paths × 30y run in seconds in NumPy |
| "Re-run the simulation and follow its recommendation" (dynamic re-simulation) | Risk of moving goalposts and pro-cyclical changes | Annual review updates state only; the policy is fixed in advance |
| CPPI as a dynamic strategy | Path-dependent; risks being locked into cash after a crash; conflicts with a fixed 60/40 | Not included; cash-flow and band rebalancing instead |
| Scenario A: "after a 30 % equity rally, de-risk because success probability rose" | Contradicts the stated wish for high return; it is a goal-locking decision, not a free lunch | Can be explored with the goal planner (lower equity once the goal is nearly certain) |
| Missing entirely: taxes, costs, EUR inflation, FX, tax-aware rebalancing | For a German investor these are among the largest controllable levers | Full German tax module, TER and trading costs, CPI deflation, cash-flow rebalancing |
| Expectation of "low risk + constant + high ROI" | These trade off; a 60/40 has ~28 % negative years and a 1-in-20 chance of a 40 %+ fall over 30 years | Shown transparently: drawdown, underwater time, negative years, CVaR |

## 6. Limitations

* **Model risk.** Results depend on the capital market assumptions, especially the equity
  return and starting yields. Run both generators, try a few equity-return assumptions, and
  check the history page.
* **Equity history.** The historical equity series is US-only (in EUR), which flatters
  history versus a global index.
* **Linkers.** Synthetic before 1998, and modelled with constant duration.
* **Wages.** Savings are assumed to continue through crises (job loss is not modelled).
* **Tax rules** reflect the law as of 2025/26 and are held constant over the horizon.
* **Execution.** Monthly time steps: intra-month moves, distribution dates and settlement
  are ignored.
