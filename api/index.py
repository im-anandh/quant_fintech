"""Vercel entrypoint for the quantitative backtesting dashboard."""
from __future__ import annotations

import html
from datetime import date, timedelta

from flask import Flask, jsonify, request

from src.backtester import backtest, buy_and_hold
from src.data import ASSETS, TRADING_DAYS, DataUnavailable, load_assets
from src.strategies import STRATEGIES

app = Flask(__name__)


def _request_values():
    asset = request.args.get("asset", "NVDA")
    strategy = request.args.get("strategy", STRATEGIES[0])
    if asset not in ASSETS:
        raise ValueError(f"Unknown asset. Choose one of: {', '.join(ASSETS)}")
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy. Choose one of: {', '.join(STRATEGIES)}")
    end = request.args.get("end", date.today().isoformat())
    start = request.args.get("start", (date.today() - timedelta(days=365 * 5)).isoformat())
    capital = float(request.args.get("capital", "100000"))
    cost = float(request.args.get("cost", "10"))
    if capital <= 0 or cost < 0:
        raise ValueError("Capital must be positive and transaction cost cannot be negative.")
    return asset, strategy, start, end, capital, cost


def _run_backtest():
    asset, strategy, start, end, capital, cost = _request_values()
    ticker = ASSETS[asset]
    data, sources = load_assets(
        [ticker], start, end,
        cache_dir="data",
        allow_synthetic=False,
    )
    close = data[ticker]["Close"].dropna()
    annualization = TRADING_DAYS.get(ticker, 252)
    result = backtest(
        close, strategy=strategy, initial_capital=capital,
        transaction_cost_bps=cost, annualization=annualization,
    )
    _, benchmark = buy_and_hold(
        close, initial_capital=capital, annualization=annualization,
        transaction_cost_bps=cost,
    )
    return asset, strategy, start, end, sources[ticker], result["metrics"], benchmark


def _clean(value):
    if isinstance(value, float) and value != value:
        return None
    return value


def _payload():
    asset, strategy, start, end, source, metrics, benchmark = _run_backtest()
    return {
        "asset": asset, "strategy": strategy, "start": start, "end": end,
        "source": source, "metrics": {k: _clean(v) for k, v in metrics.items()},
        "benchmark": {k: _clean(v) for k, v in benchmark.items()},
    }


@app.get("/api/backtest")
def api_backtest():
    try:
        return jsonify(_payload())
    except (DataUnavailable, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Backtest request failed")
        return jsonify({"error": f"Backtest failed: {exc}"}), 500


@app.get("/")
def dashboard():
    try:
        payload = _payload()
        error = ""
    except (DataUnavailable, ValueError) as exc:
        payload = None
        error = str(exc)
    if error:
        body = f'<p class="error">{html.escape(error)}</p>'
        asset = request.args.get("asset", "NVDA")
        strategy = request.args.get("strategy", STRATEGIES[0])
    else:
        asset, strategy = payload["asset"], payload["strategy"]
        metrics = payload["metrics"]
        benchmark = payload["benchmark"]
        body = "<div class=\"grid\">" + "".join(
            f'<div class="metric"><span>{html.escape(label)}</span><strong>{_format_metric(label, metrics.get(label))}</strong></div>'
            for label in ["Total Return", "CAGR", "Sharpe", "Max Drawdown", "Final Value"]
        ) + "</div>"
        body += f'<p class="muted">Compared with buy &amp; hold: {_format_metric("Total Return", benchmark.get("Total Return"))}. Data source: {html.escape(payload["source"])}.</p>'
    options = "".join(f'<option {"selected" if name == asset else ""}>{html.escape(name)}</option>' for name in ASSETS)
    strategy_options = "".join(f'<option {"selected" if name == strategy else ""}>{html.escape(name)}</option>' for name in STRATEGIES)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuantFin Backtester</title><style>
:root{{--ink:#17211b;--paper:#f4f1e8;--accent:#d85c35;--line:#d8d2c4}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px Georgia,serif}}main{{max-width:920px;margin:auto;padding:48px 22px}}h1{{font-size:clamp(2.2rem,6vw,5rem);line-height:.95;max-width:720px;margin:0 0 16px}}p{{line-height:1.5}}.kicker{{color:var(--accent);font:700 12px Arial,sans-serif;letter-spacing:2px;text-transform:uppercase}}form{{display:flex;flex-wrap:wrap;gap:12px;align-items:end;border-block:1px solid var(--line);padding:20px 0;margin:30px 0}}label{{display:grid;gap:6px;font:700 12px Arial,sans-serif;text-transform:uppercase}}select,input,button{{font:16px Georgia,serif;padding:11px;border:1px solid #aaa;background:#fff}}button{{background:var(--accent);border-color:var(--accent);color:#fff;cursor:pointer}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;background:var(--line);border:1px solid var(--line)}}.metric{{background:var(--paper);padding:20px;min-height:110px}}.metric span{{display:block;font:12px Arial,sans-serif;text-transform:uppercase;color:#667066}}.metric strong{{display:block;font-size:1.7rem;margin-top:18px}}.muted{{color:#667066;font:14px Arial,sans-serif}}.error{{border-left:4px solid var(--accent);padding:14px;background:#fff}}a{{color:var(--accent)}}
</style></head><body><main><div class="kicker">QuantFin / Vercel</div><h1>Research the market with a colder eye.</h1><p>Run the tested backtesting engine against live Yahoo Finance data.</p>
<form method="get"><label>Asset<select name="asset">{options}</select></label><label>Strategy<select name="strategy">{strategy_options}</select></label><label>Capital<input name="capital" type="number" min="1" step="1000" value="100000"></label><button type="submit">Run backtest</button></form>{body}<p class="muted"><a href="/api/backtest?asset={html.escape(asset)}&amp;strategy={html.escape(strategy)}">View JSON API response</a></p></main></body></html>"""


def _format_metric(label, value):
    if value is None:
        return "n/a"
    if label == "Final Value":
        return f"${value:,.0f}"
    if label == "Sharpe":
        return f"{value:.2f}"
    return f"{value:.1%}"
