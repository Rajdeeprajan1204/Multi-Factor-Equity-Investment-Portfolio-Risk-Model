"""Statistical factor risk model.

The implementation follows the standard decomposition:
    Sigma = B @ Omega @ B.T + D
where B contains asset exposures, Omega is factor covariance and D is
idiosyncratic variance.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass
class RiskModelResult:
    """Estimated components of the statistical risk model."""

    exposures: pd.DataFrame
    factor_returns: pd.DataFrame
    factor_covariance: pd.DataFrame
    idiosyncratic_variance: pd.Series
    covariance: pd.DataFrame
    explained_variance_ratio: pd.Series


class StatisticalRiskModel:
    """Estimate a PCA-based statistical equity risk model.

    Parameters
    ----------
    n_factors : int
        Number of latent factors retained.
    annualization_factor : int
        Number of return observations per year.
    """

    def __init__(self, n_factors: int = 5, annualization_factor: int = 252):
        if n_factors < 1:
            raise ValueError("n_factors must be positive")
        if annualization_factor < 1:
            raise ValueError("annualization_factor must be positive")
        self.n_factors = n_factors
        self.annualization_factor = annualization_factor
        self.result_: RiskModelResult | None = None

    def fit(self, returns: pd.DataFrame) -> RiskModelResult:
        """Fit the model to a date-by-asset return matrix."""
        clean = returns.dropna(axis=0, how="any").copy()
        if clean.empty:
            raise ValueError("returns contains no complete observations")
        if clean.shape[1] < self.n_factors:
            raise ValueError("n_factors cannot exceed the number of assets")

        pca = PCA(n_components=self.n_factors)
        factor_values = pca.fit_transform(clean.values)

        # PCA loadings are the asset exposures to the statistical factors.
        loadings = pd.DataFrame(
            pca.components_.T,
            index=clean.columns,
            columns=[f"Factor_{i + 1}" for i in range(self.n_factors)],
        )
        factor_returns = pd.DataFrame(
            factor_values,
            index=clean.index,
            columns=loadings.columns,
        )

        residuals = clean - factor_returns @ loadings.T
        idio_var = residuals.var(axis=0, ddof=1)
        factor_cov = factor_returns.cov()

        covariance = (
            loadings @ factor_cov @ loadings.T
            + np.diag(idio_var.values)
        )
        covariance = pd.DataFrame(
            covariance,
            index=clean.columns,
            columns=clean.columns,
        )

        self.result_ = RiskModelResult(
            exposures=loadings,
            factor_returns=factor_returns,
            factor_covariance=factor_cov,
            idiosyncratic_variance=idio_var,
            covariance=covariance,
            explained_variance_ratio=pd.Series(
                pca.explained_variance_ratio_, index=loadings.columns
            ),
        )
        return self.result_

    def fit_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Fit the model and return the annualized covariance matrix."""
        result = self.fit(returns)
        return result.covariance * self.annualization_factor
