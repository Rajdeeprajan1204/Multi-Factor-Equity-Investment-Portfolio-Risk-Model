import numpy as np
import pandas as pd

from multi_factor_risk.portfolio import (
    portfolio_variance,
    portfolio_volatility,
    risk_contribution,
)
from multi_factor_risk.risk_model import StatisticalRiskModel


def test_portfolio_variance_and_volatility():
    covariance = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["A", "B"], columns=["A", "B"],
    )
    weights = pd.Series([0.5, 0.5], index=["A", "B"])
    assert np.isclose(portfolio_variance(weights, covariance), 0.0375)
    assert np.isclose(portfolio_volatility(weights, covariance), np.sqrt(0.0375))


def test_risk_contribution_sums_to_one():
    covariance = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["A", "B"], columns=["A", "B"],
    )
    weights = pd.Series([0.5, 0.5], index=["A", "B"])
    result = risk_contribution(weights, covariance)
    assert np.isclose(result["component_risk_pct"].sum(), 1.0)


def test_pca_model_reconstructs_covariance_structure():
    rng = np.random.default_rng(7)
    factor = rng.normal(size=200)
    returns = pd.DataFrame({
        "A": factor + rng.normal(scale=0.1, size=200),
        "B": 0.7 * factor + rng.normal(scale=0.1, size=200),
        "C": -0.4 * factor + rng.normal(scale=0.1, size=200),
    })
    result = StatisticalRiskModel(n_factors=1).fit(returns)
    assert result.covariance.shape == (3, 3)
    assert result.exposures.shape == (3, 1)
    assert np.isclose(result.explained_variance_ratio.sum(), result.explained_variance_ratio.iloc[0])
