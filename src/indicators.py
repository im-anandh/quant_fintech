"""Quantitative indicators. Every function expects a clean (NaN-free) price series."""
import numpy as np
import pandas as pd


def sma(s, period): return s.rolling(period).mean()
def ema(s, period): return s.ewm(span=period, adjust=False).mean()
def daily_returns(s): return s.pct_change()
def cumulative_returns(s): return (1 + s.pct_change().fillna(0)).cumprod() - 1
def rolling_returns(s, window=20): return s.pct_change(window)


def volatility(s, window=20, annualization=252):
    return s.pct_change().rolling(window).std() * np.sqrt(annualization)


def sharpe(s, risk_free_rate=0.0, annualization=252):
    r = s.pct_change().dropna()
    rf_daily = (1 + risk_free_rate) ** (1 / annualization) - 1
    ex = r - rf_daily
    return np.nan if ex.std() == 0 or len(ex) < 2 else ex.mean() / ex.std() * np.sqrt(annualization)


def drawdown(s):
    wealth = s / s.dropna().iloc[0]
    return wealth / wealth.cummax() - 1


def max_drawdown(s): return float(drawdown(s).min())


def indicator_frame(close, annualization=252):
    return pd.DataFrame({
        "Price": close,
        "SMA20": sma(close, 20),
        "SMA50": sma(close, 50),
        "EMA20": ema(close, 20),
        "Daily Return": daily_returns(close),
        "Cumulative Return": cumulative_returns(close),
        "Volatility20": volatility(close, 20, annualization),
        "Rolling Return20": rolling_returns(close, 20),
        "Drawdown": drawdown(close),
    })
