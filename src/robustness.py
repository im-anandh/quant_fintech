import itertools

import numpy as np
import pandas as pd

from .backtester import backtest, buy_and_hold, performance_metrics


def parameter_sweep(close, strategy, grid, metric="Sharpe", **kwargs):
    """Backtest every parameter combination and record one metric."""
    keys, rows = list(grid), []
    for vals in itertools.product(*[grid[k] for k in keys]):
        p = dict(zip(keys, vals))
        if "fast" in p and "slow" in p and p["fast"] >= p["slow"]:
            continue
        m = backtest(close, strategy, p, **kwargs)["metrics"]
        rows.append({**p, metric: m.get(metric, np.nan)})
    return pd.DataFrame(rows)


def cost_sweep(close, strategy, params, costs=(0, 5, 10, 25, 50, 100), **kwargs):
    """How quickly do transaction costs (bps) eat the strategy's edge?"""
    rows = []
    for c in costs:
        m = backtest(close, strategy, params, transaction_cost_bps=c, **kwargs)["metrics"]
        rows.append({"Cost (bps)": c, "Total Return": m["Total Return"],
                     "Sharpe": m["Sharpe"], "Trades": m["Trades"]})
    return pd.DataFrame(rows)


def train_test_split(close, ratio=0.7):
    n = int(len(close) * ratio)
    return close.iloc[:n], close.iloc[n:]


def optimise_train_test(close, strategy, grid, ratio=0.7, metric="Sharpe", **kwargs):
    """Pick the best parameters on the training window only, then judge them out-of-sample.

    The test window is evaluated by running the strategy over the full history (so the
    indicators are already warmed up) and scoring only the bars after the split.
    """
    close = close.dropna()
    split = int(len(close) * ratio)
    split_date = close.index[split]
    sweep = parameter_sweep(close.iloc[:split], strategy, grid, metric, **kwargs).dropna(subset=[metric])
    if sweep.empty:
        raise ValueError("No valid parameter combination on the training window")
    best = sweep.sort_values(metric, ascending=False).iloc[0]
    params = {k: type(grid[k][0])(best[k]) for k in grid if k in sweep.columns}

    ann = kwargs.get("annualization", 252)
    rf = kwargs.get("risk_free_rate", 0.0)
    train_res = backtest(close.iloc[:split], strategy, params, **kwargs)
    full = backtest(close, strategy, params, **kwargs)
    test_trades = full["trades"][full["trades"]["Entry Date"] >= split_date]
    test_metrics = performance_metrics(full["equity"].loc[split_date:], ann, rf,
                                       test_trades, full["position"].loc[split_date:])
    _, bh_metrics = buy_and_hold(close.loc[split_date:], annualization=ann, risk_free_rate=rf)
    return {"params": params, "split_date": split_date, "sweep": sweep,
            "train_metrics": train_res["metrics"], "test_metrics": test_metrics,
            "benchmark_test_metrics": bh_metrics}
