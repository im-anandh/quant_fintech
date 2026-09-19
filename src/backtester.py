"""Portfolio-level backtesting engine (long/flat, single asset, daily bars).

Timing convention
-----------------
A signal computed from the close of bar t is delayed by ``execution_lag`` bars
before it earns any return:

* lag = 1 (default): the trade is executed at the close of the signal bar
  (market-on-close style), and the position earns the *next* bar's return.
* lag = 2: the trade is executed one full bar later (more conservative).

Either way the position that earns bar t's return was decided using data
strictly before bar t's return, so there is no look-ahead in the returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .strategies import signals

TRADE_COLUMNS = ["Entry Date", "Exit Date", "Entry Price", "Exit Price",
                 "Gross Return", "Net Return", "P&L ($)", "Bars Held", "Open"]


def performance_metrics(equity, annualization=252, risk_free_rate=0.0, trades=None, position=None):
    equity = equity.dropna()
    r = equity.pct_change().dropna()
    years = len(r) / annualization
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else np.nan
    vol = r.std() * np.sqrt(annualization)
    ex = r - ((1 + risk_free_rate) ** (1 / annualization) - 1)
    sharpe = ex.mean() / ex.std() * np.sqrt(annualization) if len(ex) > 1 and ex.std() > 0 else np.nan
    max_dd = (equity / equity.cummax() - 1).min()
    calmar = cagr / abs(max_dd) if max_dd < 0 and not np.isnan(cagr) else np.nan
    n_trades = len(trades) if trades is not None else np.nan
    win_rate = float((trades["Net Return"] > 0).mean()) if trades is not None and len(trades) else np.nan
    exposure = float((position > 0).mean()) if position is not None else np.nan
    return {
        "Final Value": float(equity.iloc[-1]), "Total Return": float(total), "CAGR": float(cagr),
        "Volatility": float(vol), "Sharpe": float(sharpe), "Max Drawdown": float(max_dd),
        "Calmar": float(calmar), "Trades": n_trades, "Win Rate": win_rate, "Time in Market": exposure,
    }


def _trade_log(close, pos, equity):
    held = pos > 0
    prev = held.shift(1, fill_value=False)
    starts = np.flatnonzero((held & ~prev).to_numpy())
    ends = np.flatnonzero((~held & prev).to_numpy())
    rows = []
    for i in starts:
        later = ends[ends > i]
        entry_bar = i - 1  # the position starts earning from this bar's close
        if len(later):
            j = later[0]
            exit_bar, eq_exit, is_open = j - 1, equity.iloc[j], False  # equity[j] includes exit cost
        else:
            exit_bar, eq_exit, is_open = len(close) - 1, equity.iloc[-1], True
        eq_entry = equity.iloc[entry_bar]
        rows.append({
            "Entry Date": close.index[entry_bar], "Exit Date": close.index[exit_bar],
            "Entry Price": float(close.iloc[entry_bar]), "Exit Price": float(close.iloc[exit_bar]),
            "Gross Return": float(close.iloc[exit_bar] / close.iloc[entry_bar] - 1),
            "Net Return": float(eq_exit / eq_entry - 1), "P&L ($)": float(eq_exit - eq_entry),
            "Bars Held": int(exit_bar - entry_bar), "Open": is_open,
        })
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def backtest(close, strategy="SMA Crossover", params=None, initial_capital=100_000,
             position_sizing="all-in", fraction=0.25, transaction_cost_bps=10,
             annualization=252, risk_free_rate=0.0, execution_lag=1,
             target_vol=0.15, vol_window=20):
    close = close.dropna()
    raw = signals(close, strategy, params)
    pos = raw.shift(execution_lag).fillna(0.0)
    asset_ret = close.pct_change().fillna(0.0)

    if position_sizing == "fixed fraction":
        pos = pos * fraction
    elif position_sizing == "volatility-targeted":
        # volatility is lagged one bar so sizing never uses the return it earns
        vol = asset_ret.rolling(vol_window).std().shift(1) * np.sqrt(annualization)
        pos = (pos * (target_vol / vol)).clip(0, 1).fillna(0.0)
    elif position_sizing != "all-in":
        raise ValueError(f"Unknown position sizing: {position_sizing}")

    traded = pos.diff().abs().fillna(pos.abs())           # fraction of equity traded each bar
    costs = traded * transaction_cost_bps / 10_000
    strat_ret = pos * asset_ret - costs
    equity = initial_capital * (1 + strat_ret).cumprod()
    equity.name = "Strategy"

    trades = _trade_log(close, pos, equity)
    portfolio = pd.DataFrame({
        "Position (% of equity)": pos, "Invested Value": equity * pos,
        "Cash": equity * (1 - pos), "Portfolio Value": equity,
    })
    return {"equity": equity, "position": pos, "signal": raw, "portfolio": portfolio,
            "trades": trades,
            "metrics": performance_metrics(equity, annualization, risk_free_rate, trades, pos)}


def buy_and_hold(close, initial_capital=100_000, annualization=252, risk_free_rate=0.0,
                 transaction_cost_bps=0):
    close = close.dropna()
    ret = close.pct_change().fillna(0.0)
    if len(ret) > 1:  # pay the entry cost once, like the strategy does
        ret.iloc[1] = (1 + ret.iloc[1]) * (1 - transaction_cost_bps / 10_000) - 1
    equity = initial_capital * (1 + ret).cumprod()
    equity.name = "Buy & Hold"
    m = performance_metrics(equity, annualization, risk_free_rate)
    m.update({"Trades": 1, "Time in Market": 1.0})
    return equity, m
