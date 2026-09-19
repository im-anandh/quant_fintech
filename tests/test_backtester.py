import numpy as np
import pandas as pd
from src.backtester import backtest, buy_and_hold
from src.data import align_close


def _walk(n=800, seed=2, freq="B"):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n))),
                     index=pd.date_range("2020-01-01", periods=n, freq=freq))


def test_no_lookahead():
    s = pd.Series([100, 100, 110, 110], index=pd.date_range("2024-01-01", periods=4))
    r = backtest(s, "Momentum", {"lookback": 1}, initial_capital=1000)
    assert list(r["position"]) == [0, 0, 0, 1]


def test_signal_cannot_capture_the_bar_it_was_computed_on():
    # the big +10% jump on bar 2 is what triggers the momentum signal; the strategy must miss it
    s = pd.Series([100, 100, 110, 110, 110], index=pd.date_range("2024-01-01", periods=5))
    r = backtest(s, "Momentum", {"lookback": 1}, initial_capital=1000, transaction_cost_bps=0)
    assert r["equity"].iloc[-1] == 1000


def test_costs_reduce_returns():
    p = _walk()
    free = backtest(p, "EMA Trend", transaction_cost_bps=0)["metrics"]["Total Return"]
    costly = backtest(p, "EMA Trend", transaction_cost_bps=100)["metrics"]["Total Return"]
    assert costly < free


def test_every_strategy_trades_and_reports_metrics():
    p = _walk()
    for name in ["SMA Crossover", "EMA Trend", "Momentum", "Mean Reversion"]:
        m = backtest(p, name)["metrics"]
        assert m["Trades"] > 0, name
        assert not np.isnan(m["Sharpe"]), name


def test_trade_log_matches_equity():
    p = _walk()
    r = backtest(p, "SMA Crossover", transaction_cost_bps=0, initial_capital=1000)
    growth = (1 + r["trades"]["Net Return"]).prod()          # all-in, no costs: trades compound to the equity
    assert abs(growth - r["equity"].iloc[-1] / 1000) < 1e-9


def test_weekend_gaps_do_not_kill_signals():
    # regression: gold/equities on a 7-day calendar had NaN weekends, so SMA crossover never fired
    p = _walk(freq="B")
    assert backtest(p, "SMA Crossover")["metrics"]["Trades"] > 0


def test_volatility_targeting_has_no_nans():
    r = backtest(_walk(), "Momentum", position_sizing="volatility-targeted")
    assert r["equity"].notna().all() and r["position"].between(0, 1).all()


def test_buy_and_hold_equals_price_ratio_without_costs():
    p = _walk()
    eq, _ = buy_and_hold(p, 1000)
    assert abs(eq.iloc[-1] - 1000 * p.iloc[-1] / p.iloc[0]) < 1e-6


def test_align_close_uses_common_dates_only():
    days = pd.date_range("2024-01-01", "2024-02-29", freq="D")       # crypto: every day
    weekdays = pd.date_range("2024-01-01", "2024-02-29", freq="B")   # gold/equities: weekdays
    btc = pd.DataFrame({"Close": range(1, len(days) + 1)}, index=days, dtype=float)
    gold = pd.DataFrame({"Close": range(1, len(weekdays) + 1)}, index=weekdays, dtype=float)
    a = align_close({"BTC": btc, "Gold": gold})
    assert a.notna().all().all() and list(a.index) == list(weekdays)
