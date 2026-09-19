import numpy as np
import pandas as pd
from src.strategies import signals, STRATEGIES


def _walk(n=600, seed=1):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, n))),
                     index=pd.date_range("2020-01-01", periods=n, freq="B"))


def test_mean_reversion_actually_trades():
    s = signals(_walk(), "Mean Reversion", {})
    assert s.sum() > 0 and s.nunique() == 2          # regression: it used to be all zeros


def test_mean_reversion_enters_on_dip_and_exits_at_mean():
    prices = pd.Series([100.0] * 25 + [80.0, 85.0, 90.0, 100.0, 110.0, 100.0])
    s = signals(prices, "Mean Reversion", {"window": 20, "entry_z": 1.0, "exit_z": 0.0})
    assert s.iloc[25] == 1 and s.iloc[-1] == 0


def test_all_strategies_return_binary_series():
    for name in STRATEGIES:
        s = signals(_walk(), name, {})
        assert set(s.unique()) <= {0.0, 1.0}
