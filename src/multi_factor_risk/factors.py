"""Simple, explainable cross-sectional equity signals."""

import numpy as np
import pandas as pd


def momentum(prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Trailing return signal used as a medium-term momentum hypothesis."""
    return prices.pct_change(window)


def reversal(prices: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Short-horizon reversal signal: negative trailing return."""
    return -prices.pct_change(window)


def overnight_return(open_prices: pd.DataFrame, close_prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate open_t / close_(t-1) - 1 for each asset."""
    return open_prices.div(close_prices.shift(1)) - 1.0


def trailing_overnight_signal(
    open_prices: pd.DataFrame,
    close_prices: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """Aggregate overnight returns over a trailing window."""
    return overnight_return(open_prices, close_prices).rolling(window).sum()


def cross_sectional_zscore(signal: pd.DataFrame) -> pd.DataFrame:
    """Standardize a signal across assets on each date."""
    mean = signal.mean(axis=1)
    std = signal.std(axis=1, ddof=1).replace(0, np.nan)
    return signal.sub(mean, axis=0).div(std, axis=0)


def sector_neutral_zscore(
    signal: pd.DataFrame,
    sectors: pd.Series,
) -> pd.DataFrame:
    """Demean a cross-section within sectors, then standardize it.

    ``sectors`` should be indexed by asset and align with signal columns.
    """
    sectors = sectors.reindex(signal.columns)
    neutral = signal.copy()
    for sector in sectors.dropna().unique():
        cols = sectors[sectors == sector].index
        sector_mean = signal[cols].mean(axis=1)
        neutral[cols] = signal[cols].sub(sector_mean, axis=0)
    return cross_sectional_zscore(neutral)
