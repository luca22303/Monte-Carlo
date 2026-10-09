# Monte Carlo wealth engine

A Monte Carlo engine and Streamlit dashboard for **long-term, low-risk wealth building with a
monthly savings plan** in a ~60/40 portfolio, for an investor **taxed in Germany (EUR)**.

It answers practical questions:

* Where will my savings plan likely end up, in today's purchasing power, after all taxes and costs?
* How likely am I to reach my goal, and how much must I save to reach it with 80 % or 90 % confidence?
* Which rebalancing rule gives the best risk/return after German taxes?
* How should I split the 40 % defensive part between bonds, inflation-linked bonds, gold and cash?
* At my annual review, should I sell anything, or just steer my next savings?

> Educational planning tool, not investment advice. The output is only as good as the assumptions.
> They are all visible and editable.

## Quick start

Needs Python 3.11 or newer (`python3 --version`). On a Mac, get it from
[python.org](https://www.python.org/downloads/) or with `brew install python@3.12`.

```bash
git clone https://github.com/luca22303/Monte-Carlo.git
cd Monte-Carlo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The dashboard opens at http://localhost:8501. On a Mac you can afterwards simply **double-click
`start_dashboard.command`** in Finder: it updates the code, installs new libraries and starts the dashboard. Next time, only run `cd Monte-Carlo`,
`source .venv/bin/activate` and `streamlit run app/streamlit_app.py`.

The historical panel (1973–2025) is bundled, so no network access is needed after installing.

Development:

```bash
pip install -e ".[dev]"
pytest
python -m mcengine.data.build_dataset
```

`pytest` runs 55 tests. The last command refreshes the historical data panel.

**Hosting on Streamlit Community Cloud:** at share.streamlit.io choose this repository, the
branch and main file `app/streamlit_app.py`. Dependencies come from `requirements.txt`.

## What it does

| Page | Purpose |
|---|---|
| **Setup** | Savings plan, goal, allocation, rebalancing rule, taxes, costs, engine and market assumptions. Config import/export as JSON. |
| **Simulation** | Fan chart of real wealth, terminal distribution, drawdown distribution, all metrics. |
| **Strategies** | Buy & hold, calendar, band, cash-flow and hybrid rebalancing, run on *identical* scenarios, with paired differences and standard errors. |
| **Allocation explorer** | Grid over the defensive sleeve at 50/55/60 % equity, full after-tax simulation per point, Pareto frontier, one-click apply. |
| **ETFs & savings plan** | Checklist for products (cost, size, accumulating, domicile, gold delivery claim, savings-plan availability) with a verified example shortlist, the exact monthly savings-plan amount per ETF, and what the cost differences mean in € over your plan. |
| **Goal planner** | Required monthly savings for a goal at chosen confidence levels; P(goal) vs. savings rate. |
| **Annual review** | Import your broker's depot CSV (German or English exports; template for brokers without export) or enter today's holdings and cost basis. Get drift, band status, how to split the next savings, months to target without selling, trades plus estimated tax, and an updated outlook. |
| **Assumptions & history** | Regime model, model vs. 1973–2025 history (returns, volatilities, correlations), your exact plan backtested on every historical window. |

## Engine in one paragraph

Two independent return generators feed one simulator:

1. **Parametric.** A 3-regime Markov-switching model (calm / deflationary crisis / inflation
   shock) with fat-tailed multivariate Student-t shocks on six risk factors: equity, gold,
   nominal yield, real yield, short rate and inflation. Bond and linker returns come from
   the simulated yield paths, so today's starting yields anchor expected bond returns.
2. **Bootstrap.** A stationary block bootstrap of monthly 1973–2025 EUR history (Fama/French,
   OECD, ECB), optionally mean-shifted to today's forward-looking assumptions.

The simulator runs the savings plan month by month:
- FIFO tax lots
- German taxation: Abgeltungsteuer + Soli (+ KiSt), Teilfreistellung, Vorabpauschale,
  Sparerpauschbetrag, loss carry-forward, annual taxation of interest, and gold ETCs
  tax-free after 1 year
- fund costs and trading costs
- the chosen rebalancing policy

Results are reported in today's euros after final liquidation. See
[`docs/methodology.md`](docs/methodology.md) for details, calibration and limitations.

## What the default run says

Default run: €1,000/month growing with inflation, 30 years, 60/40 (60 equity / 20 govt / 10
linkers / 5 gold / 5 cash), hybrid rebalancing, taxes on, 5,000 paths.

| | Forward-looking model | Your plan on actual 1973–2025 history |
|---|---|---|
| Median net wealth (today's €) | ≈ €520k | ≈ €895k |
| Bad case (P5) | ≈ €315k | ≈ €700k |
| Median real return p.a. (money-weighted, after tax) | ≈ 2.4 % | ≈ 5.5 % |
| Max drawdown, P95 | ≈ 42 % | ≈ 37 % |

Savings paid in are €360k in today's money. The gap between the two columns is the most
important number in the tool. The last 50 years were exceptional: bond yields started at
8 % and fell for 40 years, and US equities boomed. Today's starting yields are about 3 %.
Plan with the left column and treat the right one as upside.

Robust findings (stable across seeds and both generators):

* **Cash-flow steering beats calendar rebalancing after tax.** You buy the underweight asset
  with new savings instead of selling the overweight one. On identical scenarios it ends
  ≈ €14k higher on average (standard error ≈ €0.6k). It is better on ~65 % of paths, has a
  better bad case (P5) and a smaller P95 drawdown. The bootstrap generator confirms it
  (+€22k). The gain comes from never realising gains early: tax is deferred and keeps
  compounding, and winners run a little longer between corrections.
* **The engine can't promise "low risk, constant and high ROI" together.** A 60/40 portfolio
  has negative calendar years ~28 % of the time and a ~1-in-20 chance of a 40 %+ fall at
  some point in 30 years. What you control is the savings rate, costs, taxes, the
  rebalancing rule and staying invested.
* **Optimisers amplify assumptions.** Without caps, the explorer piles into gold because it
  is uncorrelated, tax-free after a year and has an assumed return close to bonds. The
  explorer caps gold and cash at 15 % by default.

## Project layout

```
mcengine/
  config.py              pydantic models for every assumption (one JSON = one reproducible run)
  scenarios/             parametric (regime-switching t), bootstrap, historical windows
  portfolio/             book (FIFO lots), tax_de (German tax), strategies, simulator
  metrics.py             goal probability, real IRR, drawdowns, CVaR, MC standard errors
  compare.py             common-random-number comparisons
  optimize.py            allocation grid + Pareto frontier, required-savings solver
  review.py              annual review
  data/                  data fetchers, panel builder, bundled panel_monthly.csv
app/                     Streamlit dashboard (streamlit_app.py + views/)
tests/                   unit, statistical and app smoke tests
docs/methodology.md      model details, calibration, critique of the original research notes
```
