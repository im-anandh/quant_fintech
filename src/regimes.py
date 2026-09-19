import numpy as np
import pandas as pd


def classify(close, annualization=252, sma_window=200, vol_window=20, vol_lookback=252):
    """Label each day Bull/Bear (price vs SMA200) and High/Low volatility (vs rolling median).

    Days without enough history are left as NaN instead of being mislabelled.
    """
    close = close.dropna()
    sma = close.rolling(sma_window).mean()
    vol = close.pct_change().rolling(vol_window).std() * np.sqrt(annualization)
    vol_median = vol.rolling(vol_lookback, min_periods=60).median()
    trend = pd.Series(np.where(close >= sma, "Bull", "Bear"), index=close.index).where(sma.notna())
    volat = pd.Series(np.where(vol >= vol_median, "High Volatility", "Low Volatility"),
                      index=close.index).where(vol_median.notna())
    return pd.DataFrame({"Trend": trend, "Volatility": volat})


def strategy_by_regime(strategy_equity, benchmark_equity, regimes, annualization=252):
    """Compare strategy vs buy-and-hold inside each regime.

    Regimes are shifted one bar: a return is attributed to the regime that was
    known at the *previous* close, so the label never uses same-day information.
    """
    regimes = regimes.shift(1)
    s, b = strategy_equity.pct_change(), benchmark_equity.pct_change()
    rows = []
    for col in regimes.columns:
        for state in sorted(regimes[col].dropna().unique()):
            mask = (regimes[col] == state).reindex(s.index, fill_value=False)
            sr, br = s[mask].dropna(), b[mask].dropna()
            if sr.empty:
                continue
            sharpe = sr.mean() / sr.std() * np.sqrt(annualization) if len(sr) > 1 and sr.std() > 0 else np.nan
            rows.append({"Regime": f"{col}: {state}", "Days": len(sr),
                         "Strategy Return": (1 + sr).prod() - 1,
                         "Buy & Hold Return": (1 + br).prod() - 1,
                         "Strategy Sharpe": sharpe})
    return pd.DataFrame(rows)
