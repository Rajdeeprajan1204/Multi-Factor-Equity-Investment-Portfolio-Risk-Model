"""Performance and signal diagnostics."""

import numpy as np
import pandas as pd


def annualized_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio assuming zero risk-free rate."""
    clean = returns.dropna()
    if clean.std(ddof=1) == 0:
        return np.nan
    return float(np.sqrt(periods_per_year) * clean.mean() / clean.std(ddof=1))


def rank_information_coefficient(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.Series:
    """Compute daily cross-sectional rank IC between signal and next-period returns."""
    aligned_signal, aligned_returns = signal.align(forward_returns, join="inner", axis=0)
    values = []
    dates = []
    for date in aligned_signal.index:
        x = aligned_signal.loc[date]
        y = aligned_returns.loc[date]
        pair = pd.concat([x, y], axis=1).dropna()
        if len(pair) >= 3:
            values.append(pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank()))
            dates.append(date)
    return pd.Series(values, index=dates, name="rank_ic")


def factor_rank_autocorrelation(signal: pd.DataFrame) -> pd.Series:
    """Measure stability of cross-sectional signal ranks through time."""
    ranks = signal.rank(axis=1, pct=True)
    return ranks.T.corr().shift(1).apply(lambda row: row.iloc[-1] if len(row) else np.nan)
