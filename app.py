"""Quantitative Multi-Asset Financial Intelligence & Backtesting Platform (Streamlit UI)."""
import numpy as np
import pandas as pd
import streamlit as st

from src.backtester import backtest, buy_and_hold
from src.correlation import correlation_matrix, rolling_correlation
from src.data import ASSETS, TRADING_DAYS, DataUnavailable, align_close, load_assets
from src.indicators import drawdown, indicator_frame
from src.plots import fan_chart, heatmap, line, param_heatmap, signal_chart
from src.regimes import classify, strategy_by_regime
from src.risk import monte_carlo_paths, risk_summary
from src.robustness import cost_sweep, optimise_train_test, parameter_sweep
from src.strategies import DEFAULT_PARAMS, PARAM_GRIDS, STRATEGIES

st.set_page_config(page_title="QuantFin Intelligence", layout="wide")
st.title("Quantitative Multi-Asset Financial Intelligence & Backtesting Platform")
st.warning(
    "Research only: historical performance is not a guarantee of future returns. "
    "Backtests are simulations, not investment advice."
)


# --------------------------------------------------------------------------- data
@st.cache_data(ttl=3600, show_spinner="Loading market data...")
def get_data(tickers, start, end, allow_synthetic):
    return load_assets(list(tickers), start, end, allow_synthetic=allow_synthetic)


def strategy_params_ui(strategy):
    """Sidebar widgets for the selected strategy's parameters."""
    d = DEFAULT_PARAMS[strategy]
    if strategy == "SMA Crossover":
        fast = st.sidebar.slider("Fast SMA (days)", 5, 100, d["fast"])
        slow = st.sidebar.slider("Slow SMA (days)", 20, 300, d["slow"])
        if fast >= slow:
            st.sidebar.warning("Fast SMA should be shorter than slow SMA.")
        return {"fast": fast, "slow": slow}
    if strategy == "EMA Trend":
        return {"period": st.sidebar.slider("EMA period (days)", 5, 300, d["period"])}
    if strategy == "Momentum":
        return {"lookback": st.sidebar.slider("Lookback (days)", 2, 250, d["lookback"])}
    return {
        "window": st.sidebar.slider("Z-score window (days)", 5, 100, d["window"]),
        "entry_z": st.sidebar.slider("Entry z-score (buy below -z)", 0.5, 3.0, d["entry_z"], 0.1),
        "exit_z": st.sidebar.slider("Exit z-score", -1.0, 1.0, d["exit_z"], 0.1),
    }


with st.sidebar:
    st.header("Data")
    names = st.multiselect("Assets", list(ASSETS), default=list(ASSETS))
    start = st.date_input("Start", pd.Timestamp("2020-01-01").date())
    end = st.date_input("End", pd.Timestamp.today().date())
    demo = st.checkbox("Demo mode (synthetic data if download fails)", value=False)

    st.header("Backtest settings")
    strategy = st.selectbox("Strategy", STRATEGIES)
    capital = st.number_input("Initial capital ($)", 1_000, 10_000_000, 100_000, 1_000)
    cost = st.number_input("Transaction cost (bps per trade)", 0.0, 500.0, 10.0, 1.0)
    rf = st.number_input("Risk-free rate (annual)", 0.0, 1.0, 0.04, 0.005)
    sizing = st.selectbox("Position sizing", ["all-in", "fixed fraction", "volatility-targeted"])
    fraction = st.slider("Fraction of equity", 0.1, 1.0, 0.5, 0.05) if sizing == "fixed fraction" else 0.5
    target_vol = st.slider("Target annual volatility", 0.05, 0.60, 0.15, 0.01) if sizing == "volatility-targeted" else 0.15
    lag = st.radio("Execution delay", [1, 2], format_func=lambda x: "Next bar (standard)" if x == 1 else "Two bars (conservative)")
    st.header("Strategy parameters")
    params = strategy_params_ui(strategy)

if not names:
    st.info("Select at least one asset in the sidebar.")
    st.stop()

try:
    data, sources = get_data(tuple(ASSETS[n] for n in names), str(start), str(end), demo)
except DataUnavailable as exc:
    st.error(str(exc))
    st.stop()

if "synthetic" in sources.values():
    st.error("DEMO MODE: at least one asset is using SYNTHETIC random data, not real prices. "
             "Do not draw financial conclusions from these results.")
elif "cache" in sources.values():
    st.info("Live download failed for some assets - using the last cached real data.")

# Each asset keeps its own calendar (needed for correct rolling windows and annualisation);
# `aligned` is the common-calendar view used only for cross-asset comparisons.
series = {n: data[ASSETS[n]]["Close"].dropna() for n in names}
ann = {n: TRADING_DAYS.get(ASSETS[n], 252) for n in names}
aligned = align_close({n: data[ASSETS[n]] for n in names})

bt_kwargs = dict(initial_capital=capital, position_sizing=sizing, fraction=fraction,
                 transaction_cost_bps=cost, risk_free_rate=rf, execution_lag=lag, target_vol=target_vol)


def pct_table(df):
    """Show metric tables with readable rounding."""
    return df.round(4)


tabs = st.tabs(["Overview", "Indicators", "Risk & Monte Carlo", "Correlation",
                "Backtest", "Robustness", "Regimes"])

# ----------------------------------------------------------------------- overview
with tabs[0]:
    st.subheader("Performance rebased to 100 (common trading days)")
    if aligned.empty:
        st.warning("The selected assets share no trading days in this range.")
    else:
        st.plotly_chart(line(aligned / aligned.iloc[0] * 100, "Growth of 100"))
    latest = pd.DataFrame({n: {"Last Close": s.iloc[-1], "First Date": s.index[0].date(),
                               "Last Date": s.index[-1].date(), "Data Source": sources[ASSETS[n]]}
                           for n, s in series.items()}).T
    st.dataframe(latest)

# --------------------------------------------------------------------- indicators
with tabs[1]:
    a = st.selectbox("Asset", names, key="ind_asset")
    frame = indicator_frame(series[a], ann[a])
    st.plotly_chart(line(frame[["Price", "SMA20", "SMA50", "EMA20"]], f"{a}: price with SMA / EMA"))
    c1, c2 = st.columns(2)
    c1.plotly_chart(line(frame[["Volatility20"]], "20-day annualised volatility"))
    c2.plotly_chart(line(frame[["Rolling Return20"]], "20-day rolling return"))
    c3, c4 = st.columns(2)
    c3.plotly_chart(line(frame[["Cumulative Return"]], "Cumulative return"))
    c4.plotly_chart(line(frame[["Drawdown"]], "Drawdown from peak"))

# ------------------------------------------------------------------- risk & MC
with tabs[2]:
    st.subheader("Risk summary")
    st.dataframe(pct_table(pd.DataFrame({n: risk_summary(series[n], rf, ann[n]) for n in names}).T))
    st.caption("VaR / CVaR are historical 1-day figures: the loss level exceeded on the worst 5% of days, "
               "and the average loss on those days.")
    st.subheader("Monte Carlo simulation (bootstrapped returns)")
    m1, m2, m3 = st.columns(3)
    mc_asset = m1.selectbox("Asset", names, key="mc_asset")
    horizon = m2.slider("Horizon (trading days)", 20, 504, 252)
    n_sims = m3.select_slider("Simulations", [200, 500, 1000, 2000], value=1000)
    paths = monte_carlo_paths(series[mc_asset], horizon, n_sims)
    st.plotly_chart(fan_chart(paths, f"{mc_asset}: {n_sims} simulated paths, {horizon} days ahead"))
    final = paths[:, -1] / series[mc_asset].iloc[-1] - 1
    s1, s2, s3 = st.columns(3)
    s1.metric("Median outcome", f"{np.median(final):.1%}")
    s2.metric("5th percentile outcome", f"{np.percentile(final, 5):.1%}")
    s3.metric("Probability of loss", f"{(final < 0).mean():.0%}")
    st.caption("Resamples historical daily returns independently, so it ignores volatility clustering. "
               "It illustrates the range of outcomes; it is not a forecast.")

# ------------------------------------------------------------------- correlation
with tabs[3]:
    if len(names) < 2:
        st.info("Select at least two assets to see correlations.")
    else:
        st.plotly_chart(heatmap(correlation_matrix(aligned), "Daily-return correlation"))
        p1, p2, p3 = st.columns(3)
        x = p1.selectbox("Asset A", names, key="corr_a")
        y = p2.selectbox("Asset B", names, index=1, key="corr_b")
        w = p3.slider("Rolling window (days)", 20, 250, 60)
        if x == y:
            st.info("Pick two different assets.")
        else:
            st.plotly_chart(line(rolling_correlation(aligned, x, y, w).to_frame(f"{x} vs {y}"),
                                 f"{w}-day rolling correlation"))
    st.caption("Computed on days when all selected assets traded, so weekend crypto moves are excluded.")

# ---------------------------------------------------------------------- backtest
with tabs[4]:
    a = st.selectbox("Asset to backtest", names, key="bt_asset")
    res = backtest(series[a], strategy, params, annualization=ann[a], **bt_kwargs)
    bh_eq, bh_m = buy_and_hold(series[a], capital, ann[a], rf, cost)
    m = res["metrics"]

    st.plotly_chart(signal_chart(series[a], res["trades"], f"{a}: {strategy} buy / sell signals"))
    st.plotly_chart(line(pd.concat([res["equity"], bh_eq], axis=1), "Equity curve: strategy vs buy & hold"))
    st.plotly_chart(line(pd.concat([drawdown(res["equity"]).rename("Strategy"),
                                    drawdown(bh_eq).rename("Buy & Hold")], axis=1), "Drawdown comparison"))

    st.subheader("Strategy vs benchmark")
    st.dataframe(pct_table(pd.DataFrame([m, bh_m], index=["Strategy", "Buy & Hold"])))

    beat = m["Total Return"] - bh_m["Total Return"]
    st.info(
        f"**Summary:** {strategy} on {a} returned {m['Total Return']:.1%} vs {bh_m['Total Return']:.1%} for "
        f"buy & hold ({'ahead' if beat > 0 else 'behind'} by {abs(beat):.1%}). Max drawdown was "
        f"{m['Max Drawdown']:.1%} vs {bh_m['Max Drawdown']:.1%}, using {int(m['Trades'])} trades and "
        f"{m['Time in Market']:.0%} time in the market. Transaction costs of {cost:.0f} bps are included."
    )

    st.subheader("All strategies (default parameters)")
    rows = {}
    for s in STRATEGIES:
        rows[s] = backtest(series[a], s, None, annualization=ann[a], **bt_kwargs)["metrics"]
    rows["Buy & Hold"] = bh_m
    cols = ["Total Return", "CAGR", "Sharpe", "Volatility", "Max Drawdown", "Trades", "Win Rate"]
    st.dataframe(pct_table(pd.DataFrame(rows).T[cols]))

    with st.expander("Trade log"):
        st.dataframe(res["trades"])
    st.download_button("Download portfolio history (CSV)",
                       pd.concat([res["portfolio"], bh_eq], axis=1).to_csv().encode(),
                       "backtest_results.csv", "text/csv")

# -------------------------------------------------------------------- robustness
with tabs[5]:
    a = st.selectbox("Asset", names, key="rob_asset")
    grid = PARAM_GRIDS[strategy]
    core = dict(bt_kwargs, annualization=ann[a])
    sweep_kwargs = {k: v for k, v in core.items() if k != "transaction_cost_bps"}

    st.subheader(f"Parameter sensitivity: {strategy} (Sharpe ratio)")
    sweep = parameter_sweep(series[a], strategy, grid, "Sharpe", transaction_cost_bps=cost, **sweep_kwargs)
    keys = list(grid)
    if len(keys) == 2:
        st.plotly_chart(param_heatmap(sweep, keys[1], keys[0], "Sharpe", "Sharpe by parameter pair"))
    else:
        st.plotly_chart(line(sweep.set_index(keys[0])[["Sharpe"]], "Sharpe by parameter value"))
    st.caption("A robust strategy shows a broad plateau of good values. One isolated bright cell is usually over-fitting.")

    st.subheader("Transaction cost sensitivity")
    st.plotly_chart(line(cost_sweep(series[a], strategy, params, **sweep_kwargs).set_index("Cost (bps)")[["Total Return"]],
                         "Total return vs transaction cost (bps)"))

    st.subheader("Out-of-sample test (optimise on first 70%, evaluate on last 30%)")
    try:
        oos = optimise_train_test(series[a], strategy, grid, 0.7, "Sharpe", transaction_cost_bps=cost, **sweep_kwargs)
        st.write(f"Best parameters on training data: **{oos['params']}** - test period starts {oos['split_date'].date()}")
        st.dataframe(pct_table(pd.DataFrame(
            [oos["train_metrics"], oos["test_metrics"], oos["benchmark_test_metrics"]],
            index=["Strategy - train (in-sample)", "Strategy - test (out-of-sample)", "Buy & Hold - test"])))
    except ValueError as exc:
        st.warning(str(exc))

# ----------------------------------------------------------------------- regimes
with tabs[6]:
    a = st.selectbox("Asset", names, key="reg_asset")
    reg = classify(series[a], ann[a])
    res_r = backtest(series[a], strategy, params, annualization=ann[a], **bt_kwargs)
    bh_r, _ = buy_and_hold(series[a], capital, ann[a], rf, cost)
    st.subheader(f"{strategy} vs buy & hold by market regime")
    table = strategy_by_regime(res_r["equity"], bh_r, reg, ann[a])
    if table.empty:
        st.info("Not enough history to classify regimes - widen the date range (200+ trading days needed).")
    else:
        st.dataframe(pct_table(table.set_index("Regime")))
    st.caption("Bull/Bear: price above/below its 200-day average. High/Low volatility: 20-day volatility above/below "
               "its trailing 1-year median. Regimes are known at the previous close, so labels never use same-day data.")
    st.dataframe(reg.dropna().tail(10))
