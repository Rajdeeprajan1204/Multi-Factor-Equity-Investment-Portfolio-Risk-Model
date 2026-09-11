"""Data preparation utilities kept independent of the risk engine."""

import pandas as pd


def validate_price_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a date-by-asset price matrix."""
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices index must be a DatetimeIndex")
    if prices.empty:
        raise ValueError("prices cannot be empty")
    clean = prices.sort_index().astype(float)
    clean = clean.loc[:, clean.notna().any(axis=0)]
    if (clean <= 0).any().any():
        raise ValueError("prices must be strictly positive")
    return clean


def returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert prices into simple daily returns."""
    return validate_price_matrix(prices).pct_change().dropna(how="all")
