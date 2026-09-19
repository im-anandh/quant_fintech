import numpy as np
import pandas as pd
from src.regimes import classify, strategy_by_regime
from src.risk import historical_var, cvar, monte_carlo_paths
from src.backtester import backtest, buy_and_hold


def _walk(n=900, seed=3):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n))),
                     index=pd.date_range("2020-01-01", periods=n, freq="B"))


def test_regimes_are_unlabelled_until_history_exists():
    reg = classify(_walk())
    assert reg["Trend"].iloc[:199].isna().all()          # used to be mislabelled "Bear"
    assert set(reg["Trend"].dropna().unique()) <= {"Bull", "Bear"}


def test_strategy_by_regime_table():
    p = _walk()
    res = backtest(p, "EMA Trend")
    bh, _ = buy_and_hold(p)
    table = strategy_by_regime(res["equity"], bh, classify(p))
    assert {"Regime", "Days", "Strategy Return", "Buy & Hold Return"} <= set(table.columns)
    assert len(table) >= 2


def test_var_and_cvar_ordering():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.02, 5000))
    assert cvar(r) < historical_var(r) < 0


def test_monte_carlo_shape_and_start():
    paths = monte_carlo_paths(_walk(), horizon=50, n_sims=200)
    assert paths.shape == (200, 50) and (paths > 0).all()
