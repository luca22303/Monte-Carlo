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

**Mac: app icon without a terminal window.** Run once in Terminal (inside the Monte-Carlo folder):

```bash
bash tools/macos/install.sh --autostart
```

* **App icon:** creates *Monte Carlo* in `~/Applications`, which you can drag to the Dock. A click checks for
  updates, starts the dashboard in the background with no terminal window, and opens it in your
  browser. If it is already running, it just opens the browser.
* **Autostart:** `--autostart` also starts the dashboard at every login, so a plain browser bookmark to
  `http://localhost:8501` always works. Leave the flag out for the app only.
* **Stop:** "⏻ Quit dashboard" in the sidebar, or `bash tools/macos/stop.sh`.
* **Remove:** `bash tools/macos/uninstall.sh`.
* **Log:** `logs/dashboard.log`.
* **Local only:** the dashboard listens only on your own computer (`localhost`), not on your network.

**Your inputs are saved automatically** in `my_data/` inside this folder: the Setup, the ETF
selection, the review holdings and a dated history of every annual review. They are loaded
again on the next start. The folder stays on your computer and is git-ignored. Reset it on
the Setup page under "Saved data". On Streamlit Community Cloud the disk is temporary, so use
"Download config" there.

Development:

```bash
pip install -e ".[dev]"
pytest
python -m mcengine.data.build_dataset
```

`pytest` runs the test suite. The last command refreshes the historical data panel.

**Hosting on Streamlit Community Cloud:** at share.streamlit.io choose this repository, the
branch and main file `app/streamlit_app.py`. Dependencies come from `requirements.txt`.

## What it does

| Page | Purpose |
|---|---|
| **Setup** | Savings plan, goal, allocation, rebalancing rule, optional glide path and 6/12-month re-evaluation switch, taxes, costs, engine and market assumptions (including how uncertain they are). Config import/export as JSON. |
| **Simulation** | Fan chart of real wealth, terminal distribution, drawdown distribution, equity share over time, all metrics. |
| **Strategies** | Buy & hold, calendar, band, cash-flow and hybrid rebalancing, plus static vs. glide path vs. re-evaluation switch (6/12 months), all on *identical* scenarios, with paired differences and standard errors. |
| **Allocation explorer** | Grid over the defensive sleeve at 50/55/60 % equity, full after-tax simulation per point, Pareto frontier, one-click apply. |
| **ETFs & savings plan** | Checklist for products (cost, size, accumulating, domicile, gold delivery claim, savings-plan availability) with a verified example shortlist, the exact monthly savings-plan amount per ETF, and what the cost differences mean in € over your plan. |
| **Goal planner** | Required monthly savings for a goal at chosen confidence levels; P(goal) vs. savings rate. |
| **Stress & behaviour** | Replay a real crisis (1973–74, dot-com, 2008, Covid, 2022, lost decade) at the start, middle, right before the goal or in any year, optionally with its historical recovery; compare all crises at once. Plus the cost of panic selling after a fall. |
| **Annual review** | Import your broker's depot CSV (German or English exports; template for brokers without export) or enter today's holdings and cost basis. Get drift, band status, how to split the next savings, months to target without selling, trades plus estimated tax, and an updated outlook. |
| **Assumptions & history** | Regime model, model vs. 1973–2025 history (returns, volatilities, correlations), your exact plan backtested on every historical window. |

## Engine in one paragraph

Two independent return generators feed one simulator:

1. **Parametric.** A 4-regime Markov-switching model (calm / deflationary crisis / inflation
   shock / post-crisis recovery) with fat-tailed multivariate Student-t shocks on six risk
   factors: equity, gold, nominal yield, real yield, short rate and inflation. Bond and linker
   returns come from the simulated yield paths, so today's starting yields anchor expected
   bond returns. Each path also draws its own long-run equity return, gold return and
   inflation level, because nobody knows those numbers exactly (parameter uncertainty).
2. **Bootstrap.** A stationary block bootstrap of monthly 1973–2025 EUR history (Fama/French,
   OECD, ECB), optionally mean-shifted to today's forward-looking assumptions.

The simulator runs the savings plan month by month:
- FIFO tax lots
- German taxation: Abgeltungsteuer + Soli (+ KiSt), Teilfreistellung, Vorabpauschale,
  Sparerpauschbetrag, loss carry-forward, annual taxation of interest, and gold ETCs
  tax-free after 1 year
- fund costs and trading costs
- the chosen rebalancing policy, optionally with a glide path, the re-evaluation switch, or
  (as a what-if) panic selling

Results are reported in today's euros after final liquidation. See
[`docs/methodology.md`](docs/methodology.md) for details, calibration and limitations.

## What the default run says

Default run: €1,000/month growing with inflation, 30 years, 60/40 (60 equity / 20 govt / 10
linkers / 5 gold / 5 cash), hybrid rebalancing, taxes on, 5,000 paths.

| | Forward-looking model | Your plan on actual 1973–2025 history |
|---|---|---|
| Median net wealth (today's €) | ≈ €515k | ≈ €895k |
| Bad case (P5) | ≈ €295k | ≈ €700k |
| Good case (P95) | ≈ €1.15m | ≈ €1.0m |
| Median real return p.a. (money-weighted, after tax) | ≈ 2.3 % | ≈ 5.5 % |
| Max drawdown, P95 | ≈ 45 % | ≈ 37 % |

Savings paid in are €360k in today's money. The gap between the two columns is the most
important number in the tool. The last 50 years were exceptional: bond yields started at
8 % and fell for 40 years, and US equities boomed. Today's starting yields are about 3 %.
Plan with the left column and treat the right one as upside.

Robust findings (stable across seeds and both generators):

* **Cash-flow steering beats calendar rebalancing after tax.** You buy the underweight asset
  with new savings instead of selling the overweight one. On identical scenarios it ends
  ≈ €32k higher on average (standard error ≈ €2k) and is better on ~64 % of paths. The
  bootstrap generator confirms it (+€29k). The gain comes from never realising gains early:
  tax is deferred and keeps compounding, and winners run a little longer between corrections.
* **The engine can't promise "low risk, constant and high ROI" together.** A 60/40 portfolio
  has negative calendar years ~30 % of the time and a ~1-in-20 chance of a 45 %+ fall at
  some point in 30 years. What you control is the savings rate, costs, taxes, the
  rebalancing rule and staying invested.
* **Optimisers amplify assumptions.** Without caps, the explorer piles into gold because it
  is uncorrelated, tax-free after a year and has an assumed return close to bonds. The
  explorer caps gold and cash at 15 % by default.
* **Timing of a crash matters more than its size.** A 2008-style crash in the first years
  barely changes the outcome; the same crash right before the goal date costs ≈ €70k and
  drops P(goal) from ~52 % to ~37 %. That is the case for a glide path.
* **Dynamic de-risking trades upside for safety.** The re-evaluation switch raises the goal
  probability by 1–2 points and the glide path improves the bad case, but both give up a
  lot of upside, so the average outcome falls. Which trade-off you prefer is a personal choice.
* **Panic selling costs money on average.** Selling all stocks after a 30 % fall and buying
  back a year later happens in about 1 in 4 futures and costs ≈ €5k on average when it does
  (more with the historical-data generator), because the rebound is missed and the sale is taxed.

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
