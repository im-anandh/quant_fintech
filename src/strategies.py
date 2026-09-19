"""Trading signals. A signal of 1 means 'want to be long', 0 means 'flat'.

Signals are computed from information available at the close of bar t only; the
backtester is responsible for delaying them before they earn any return.
"""
import numpy as np
import pandas as pd

STRATEGIES = ["SMA Crossover", "EMA Trend", "Momentum", "Mean Reversion"]

DEFAULT_PARAMS = {
    "SMA Crossover": {"fast": 20, "slow": 50},
    "EMA Trend": {"period": 50},
    "Momentum": {"lookback": 20},
    "Mean Reversion": {"window": 20, "entry_z": 1.0, "exit_z": 0.0},
}

# Grids used by the robustness tab (keys match DEFAULT_PARAMS).
PARAM_GRIDS = {
    "SMA Crossover": {"fast": [5, 10, 20, 30, 50], "slow": [50, 100, 150, 200]},
    "EMA Trend": {"period": [10, 20, 50, 100, 150, 200]},
    "Momentum": {"lookback": [5, 10, 20, 40, 60, 120]},
    "Mean Reversion": {"window": [10, 20, 30, 50], "entry_z": [0.5, 1.0, 1.5, 2.0]},
}


def _canonical(name: str) -> str:
    for s in STRATEGIES:
        if s.lower() == name.lower():
            return s
    raise ValueError(f"Unknown strategy: {name}")


def _mean_reversion(close, window, entry_z, exit_z):
    """Enter long when z-score < -entry_z; stay long until z-score >= exit_z."""
    z = ((close - close.rolling(window).mean()) / close.rolling(window).std()).to_numpy()
    out = np.zeros(len(z))
    in_trade = False
    for i, zi in enumerate(z):
        if np.isnan(zi):
            in_trade = False
        elif not in_trade and zi < -entry_z:
            in_trade = True
        elif in_trade and zi >= exit_z:
            in_trade = False
        out[i] = 1.0 if in_trade else 0.0
    return pd.Series(out, index=close.index)


def signals(close, name, params=None):
    name = _canonical(name)
    p = {**DEFAULT_PARAMS[name], **(params or {})}
    if name == "SMA Crossover":
        fast, slow = close.rolling(p["fast"]).mean(), close.rolling(p["slow"]).mean()
        return (fast > slow).astype(float)
    if name == "EMA Trend":
        return (close > close.ewm(span=p["period"], adjust=False).mean()).astype(float)
    if name == "Momentum":
        return (close.pct_change(p["lookback"]) > 0).astype(float)
    return _mean_reversion(close, p["window"], p["entry_z"], p["exit_z"])
