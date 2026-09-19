# Quantitative Multi-Asset Financial Intelligence & Backtesting Platform

Research platform for Gold (`GC=F`), Bitcoin (`BTC-USD`) and NVIDIA (`NVDA`): financial data engineering,
quantitative indicators, risk modelling, realistic backtesting and interactive visualisation in one Streamlit app.

## Run it
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
An internet connection is needed the first time (Yahoo Finance via `yfinance`). Real downloads are cached in `data/`
and used as a fallback if a later download fails. **Synthetic data is only used if you tick "Demo mode"**, and the
dashboard then shows a red banner so fake prices can never be mistaken for real ones.

## Deployment
The repository now includes a Vercel-compatible Flask entrypoint in `api/index.py`. `app.py` remains available for
the full local Streamlit dashboard; Vercel serves the browser dashboard and JSON backtest endpoint from the Flask
function. Deploy from the repository root:

```bash
npm install -g vercel
vercel login
vercel
```

The deployed page is available at `/`, and the JSON endpoint is `/api/backtest`. The function reads the repository's
bundled CSV data and may refresh it during local development; Vercel's filesystem is ephemeral, so durable runtime
data should be moved to object storage before production use.
For the full Streamlit experience, continue to use `streamlit run app.py` or Streamlit Community Cloud.

## Features
| Area | What it does |
|---|---|
| Data engineering | yfinance download, split/dividend-adjusted prices, cleaning (dedupe, sort, drop bad rows), CSV cache, per-asset calendars, common-calendar alignment for cross-asset work |
| Indicators | SMA, EMA, daily / cumulative / rolling returns, rolling & annualised volatility, Sharpe, drawdown, max drawdown |
| Risk | CAGR, Sharpe (with risk-free rate), max drawdown, historical VaR & CVaR, bootstrap Monte Carlo fan chart |
| Correlation | Correlation heatmap and rolling correlation for any pair |
| Backtesting | SMA crossover, EMA trend, momentum, mean reversion; initial capital; all-in / fixed-fraction / volatility-targeted sizing; transaction costs; trade log with entry/exit prices; portfolio value, cash and exposure history |
| Benchmark | Every result shown next to Buy & Hold (return, CAGR, Sharpe, volatility, drawdown, Calmar), plus an all-strategies comparison table |
| Robustness | Parameter heatmap (Sharpe), transaction-cost sensitivity, and a true train/test test: parameters are optimised on the first 70% and judged on the last 30% |
| Regimes | Bull / Bear and High / Low volatility; strategy vs Buy & Hold in each regime |
| Dashboard | Price + SMA/EMA, returns, volatility, drawdowns, buy/sell markers, equity curves, heatmaps, plain-English summary, CSV export |

## Financial integrity
* **Timing:** a signal computed from the close of bar *t* is delayed by one bar (or two, in conservative mode) before it earns
  any return, so a strategy can never profit from the bar that generated its signal. Tests cover this.
* **Costs:** charged on every change in position (per unit of equity traded); Buy & Hold pays its entry cost too.
* **Over-fitting:** parameter selection happens only on the training window; the test window is scored separately.
* **Regimes** are attributed with a one-bar lag, so labels never use same-day information.
* **Calendars:** each asset is analysed on its own calendar (Bitcoin annualised with 365 days, others 252). Cross-asset
  statistics use only days when all assets traded.

## Limitations
Daily bars only; long/flat single-asset strategies; no slippage, spread, tax or partial fills; Yahoo Finance is not an
institutional data feed; the Monte Carlo resamples returns independently (no volatility clustering); regime rules are simple.
Past performance does not guarantee future results, and nothing here is investment advice.

## Tests
```bash
pytest -q
```

## Future scope
Portfolio optimisation, GARCH / regime-switching models, machine-learning regime detection, paper trading, real-time
data and an AI research assistant.
