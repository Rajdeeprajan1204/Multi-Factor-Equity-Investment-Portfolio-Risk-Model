"""Portfolio-level risk and attribution calculations."""

import numpy as np
import pandas as pd


def portfolio_variance(weights: pd.Series, covariance: pd.DataFrame) -> float:
    """Return portfolio variance w' Sigma w."""
    w = weights.reindex(covariance.index).astype(float)
    return float(w.values @ covariance.values @ w.values)


def portfolio_volatility(weights: pd.Series, covariance: pd.DataFrame) -> float:
    """Return portfolio volatility from a covariance matrix."""
    variance = portfolio_variance(weights, covariance)
    return float(np.sqrt(max(variance, 0.0)))


def risk_contribution(weights: pd.Series, covariance: pd.DataFrame) -> pd.DataFrame:
    """Return marginal and component risk contribution by asset.

    Component contributions to variance sum to total portfolio variance.
    """
    w = weights.reindex(covariance.index).astype(float)
    sigma_w = covariance.values @ w.values
    variance = float(w.values @ sigma_w)
    if variance <= 0:
        raise ValueError("portfolio variance must be positive")

    component = w.values * sigma_w
    result = pd.DataFrame(
        {
            "weight": w.values,
            "marginal_variance_contribution": sigma_w,
            "component_variance_contribution": component,
            "component_risk_pct": component / variance,
        },
        index=covariance.index,
    )
    return result.sort_values("component_risk_pct", ascending=False)


def systematic_idiosyncratic_risk(
    weights: pd.Series,
    exposures: pd.DataFrame,
    factor_covariance: pd.DataFrame,
    idiosyncratic_variance: pd.Series,
) -> pd.Series:
    """Split portfolio variance into systematic and idiosyncratic components."""
    w = weights.reindex(exposures.index).astype(float)
    b = exposures.reindex(w.index)
    omega = factor_covariance.reindex(index=b.columns, columns=b.columns)
    d = idiosyncratic_variance.reindex(w.index)

    factor_exposure = b.T.values @ w.values
    systematic = float(factor_exposure @ omega.values @ factor_exposure)
    idiosyncratic = float(np.sum((w.values ** 2) * d.values))

    return pd.Series(
        {
            "systematic_variance": systematic,
            "idiosyncratic_variance": idiosyncratic,
            "total_variance": systematic + idiosyncratic,
        }
    )
