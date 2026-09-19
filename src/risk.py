import numpy as np
import pandas as pd
from .indicators import max_drawdown, sharpe


def historical_var(returns, level=0.95):
    """1-period Value at Risk as a (negative) return: the loss exceeded (1-level) of the time."""
    return float(np.percentile(pd.Series(returns).dropna(), (1 - level) * 100))


def cvar(returns, level=0.95):
    """Expected shortfall: average return on the days worse than the VaR threshold."""
    r = pd.Series(returns).dropna()
    return float(r[r <= historical_var(r, level)].mean())


def risk_summary(close, risk_free_rate=0.0, annualization=252, var_level=0.95):
    close = close.dropna()
    r = close.pct_change().dropna()
    years = len(r) / annualization
    total = close.iloc[-1] / close.iloc[0] - 1
    pct = int(var_level * 100)
    return {
        "Total Return": float(total),
        "CAGR": float((1 + total) ** (1 / years) - 1) if years > 0 else np.nan,
        "Annualized Volatility": float(r.std() * np.sqrt(annualization)),
        "Sharpe": float(sharpe(close, risk_free_rate, annualization)),
        "Max Drawdown": float(max_drawdown(close)),
        f"1-day VaR {pct}%": historical_var(r, var_level),
        f"1-day CVaR {pct}%": cvar(r, var_level),
    }


def monte_carlo_paths(close, horizon=252, n_sims=1000, seed=42):
    """Bootstrap Monte Carlo: resample historical daily returns with replacement.

    Returns an array (n_sims, horizon) of simulated prices starting from the last close.
    This keeps the empirical return distribution (fat tails) but assumes returns are
    i.i.d., so it ignores volatility clustering - a known limitation.
    """
    r = close.dropna().pct_change().dropna().to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.choice(r, size=(n_sims, horizon), replace=True)
    return float(close.dropna().iloc[-1]) * np.cumprod(1 + draws, axis=1)
