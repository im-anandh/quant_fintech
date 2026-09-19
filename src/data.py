"""Data engineering: download, cache, clean and align market data.

Real data is always preferred. Synthetic data is only produced when the caller
explicitly opts in (demo mode) and every result is labelled with its source so
the UI can never present fake prices as real ones.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ASSETS = {"Gold": "GC=F", "Bitcoin": "BTC-USD", "NVIDIA": "NVDA"}
# Bitcoin trades every day of the year; gold futures and equities ~252 days.
TRADING_DAYS = {"BTC-USD": 365, "GC=F": 252, "NVDA": 252}
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


class DataUnavailable(RuntimeError):
    """Raised when real market data cannot be obtained and demo mode is off."""


def _synthetic(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Deterministic random-walk prices. FOR DEMOS ONLY - never real data."""
    idx = pd.date_range(start, end, freq="D" if ticker == "BTC-USD" else "B")
    rng = np.random.default_rng(sum(map(ord, ticker)))
    drift = {"BTC-USD": 0.0007, "GC=F": 0.00025, "NVDA": 0.0008}.get(ticker, 0.0003)
    vol = {"BTC-USD": 0.025, "GC=F": 0.009, "NVDA": 0.022}.get(ticker, 0.012)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, len(idx))))
    open_ = close * np.exp(rng.normal(0, 0.003, len(idx)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, len(idx))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, len(idx))))
    volume = rng.integers(100_000, 5_000_000, len(idx)).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise an OHLCV frame: tz-naive sorted unique index, numeric, Close > 0."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[[c for c in OHLCV if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Close"])
    return df[df["Close"] > 0]


def _download(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    # yfinance treats `end` as exclusive, so add a day to include it.
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    # auto_adjust=True -> Close is adjusted for splits AND dividends.
    df = yf.download(ticker, start=start, end=end_excl, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = clean_prices(df)
    if df.empty:
        raise ValueError("empty download")
    return df


def download_asset(ticker, start, end, cache_dir="data", allow_synthetic=False):
    """Return (dataframe, source) where source is 'live', 'cache' or 'synthetic'."""
    csv = Path(cache_dir) / f"{ticker.replace('=', '_').replace('-', '_')}.csv"
    live_error = None
    try:
        df = _download(ticker, start, end)
        csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv)  # only real data is ever written to the cache
        return df, "live"
    except Exception as exc:  # network down, rate limit, yfinance missing ...
        live_error = exc

    if csv.exists():
        cached = clean_prices(pd.read_csv(csv, index_col=0, parse_dates=True)).loc[start:end]
        if not cached.empty:
            return cached, "cache"

    if allow_synthetic:
        return clean_prices(_synthetic(ticker, start, end)), "synthetic"
    raise DataUnavailable(
        f"Could not download {ticker} ({live_error}) and no cached copy exists. "
        "Check your internet connection, or enable demo mode to use synthetic data."
    )


def load_assets(tickers, start, end, cache_dir="data", allow_synthetic=False):
    """Return ({ticker: OHLCV frame}, {ticker: source})."""
    data, sources = {}, {}
    for t in tickers:
        data[t], sources[t] = download_asset(t, start, end, cache_dir, allow_synthetic)
    return data, sources


def align_close(data: dict) -> pd.DataFrame:
    """Close prices on the dates where ALL assets traded (inner join).

    Needed for cross-asset work (correlation, rebased comparison): Bitcoin trades
    7 days a week while gold and equities trade on weekdays only, and mixing
    calendars creates NaNs that silently corrupt rolling calculations.
    """
    closes = [v["Close"].rename(k) for k, v in data.items()]
    return pd.concat(closes, axis=1, join="inner").sort_index()
