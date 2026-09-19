"""Cross-asset correlation on a common trading calendar (see data.align_close)."""
import pandas as pd


def correlation_matrix(aligned_closes: pd.DataFrame) -> pd.DataFrame:
    return aligned_closes.pct_change().dropna().corr()


def rolling_correlation(aligned_closes: pd.DataFrame, a: str, b: str, window: int = 60) -> pd.Series:
    r = aligned_closes.pct_change().dropna()
    return r[a].rolling(window).corr(r[b])
