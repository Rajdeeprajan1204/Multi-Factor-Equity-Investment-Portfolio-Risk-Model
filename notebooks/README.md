# Research notebooks

Notebooks are intentionally kept as the exploratory layer of the project.

Recommended sequence:

1. **01_data_quality.ipynb** — inspect market data, missingness and return distributions.
2. **02_factor_research.ipynb** — construct momentum, reversal and overnight signals.
3. **03_statistical_risk_model.ipynb** — fit PCA, inspect explained variance and residual risk.
4. **04_portfolio_risk_attribution.ipynb** — calculate portfolio volatility and position/factor risk contributions.
5. **05_validation.ipynb** — test sensitivity to estimation windows, factor counts and stress periods.

Core calculations belong in `src/multi_factor_risk/`; notebooks should call those functions rather than contain the canonical implementation.
