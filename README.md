# Multi-Factor Equity Risk Model

An institutional-style equity risk framework for decomposing portfolio risk into **systematic factor risk and idiosyncratic risk**, estimating a statistically robust covariance matrix, and translating the model into portfolio-level risk attribution.

> **Why this project:** I built this to bridge fundamental finance and quantitative implementation — the same intersection that matters when investment teams need portfolio analytics, risk infrastructure, and reliable market data rather than isolated research notebooks.

## What the model answers

Given a cross-section of equities and a portfolio, the framework answers:

- How much risk is explained by common systematic factors?
- Which latent factors drive the portfolio's volatility?
- How much risk is stock-specific and therefore not diversified away?
- Which holdings contribute most to total portfolio risk?
- How do momentum, reversal and overnight signals behave when evaluated with realistic implementation constraints?

## Model architecture

```text
Market Data
    │
    ├── prices / returns / liquidity inputs
    │
    ▼
Signal Research ───────► factor signals
    │
    │
    ▼
Statistical Risk Model
    │
    ├── PCA factor extraction
    ├── factor covariance Ω
    ├── asset-factor exposures B
    └── idiosyncratic variance D
    │
    ▼
Portfolio Risk Engine
    │
    ├── Σ = BΩB' + D
    ├── portfolio volatility
    ├── systematic / idiosyncratic split
    └── marginal & component risk contribution
    │
    ▼
Portfolio Analytics
    ├── factor exposure
    ├── risk concentration
    └── signal-aware portfolio construction
```

## Core methodology

For asset returns \(R\), the model uses a statistical factor representation:

\[
R_t = B F_t + \epsilon_t
\]

where:

- **B** = asset exposures to the latent factors
- **F** = factor returns
- **ε** = idiosyncratic return

The covariance matrix is then estimated as:

\[
\Sigma = B\Omega B^T + D
\]

where **Ω** is the factor covariance matrix and **D** is the diagonal idiosyncratic variance matrix.

For portfolio weights \(w\):

\[
\sigma_p^2 = w^T\Sigma w
\]

The implementation also decomposes portfolio volatility into systematic and idiosyncratic components and calculates component risk contributions, making the model interpretable at the portfolio level rather than stopping at asset-level regressions.

## Why PCA?

The original project used PCA as a statistical risk model. This refactor keeps that idea but makes the modelling choice explicit.

PCA is useful when the objective is to identify **common sources of variation** without assuming in advance that every relevant risk driver has a clean economic label. The trade-off is equally important: PCA factors are statistical, not inherently interpretable economic factors. The README and methodology therefore distinguish between **latent risk factors** and **investment signals**.

The number of components is configurable. The default is deliberately small enough to capture the dominant common variation without turning the model into a near-full-rank covariance estimator.

## Signal research

The repository also preserves the original project's alpha-research intuition, but separates it from the risk model.

### 1. 12-month momentum

A trailing 252-trading-day return signal captures medium-term price persistence. The signal is cross-sectionally standardized before portfolio use.

### 2. 5-day reversal

The negative of the trailing 5-day return is used as a simple short-horizon reversal signal. This is intentionally presented as a **research hypothesis**, not as a guaranteed source of alpha.

### 3. Overnight return

The overnight return is calculated as:

\[
R_{ON,t} = \frac{Open_t}{Close_{t-1}} - 1
\]

The implementation can aggregate the signal over a configurable trailing window.

### 4. Signal smoothing

Moving-average smoothing is treated as a turnover/stability decision rather than an automatic improvement. Smoothing can reduce noise but may also reduce responsiveness and alter the timing of a signal.

## Portfolio risk attribution

For portfolio covariance \(\Sigma\), marginal contribution to variance is:

\[
MCR_i = (\Sigma w)_i
\]

and component contribution is:

\[
CCR_i = \frac{w_i(\Sigma w)_i}{w^T\Sigma w}
\]

The contributions sum to 100% of portfolio variance. This is more useful for an institutional risk workflow than simply reporting portfolio volatility because it identifies **where the risk actually comes from**.

## Project structure

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── src/
│   └── multi_factor_risk/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── factors.py
│       ├── risk_model.py
│       ├── portfolio.py
│       ├── analytics.py
│       └── pipeline.py
│
├── notebooks/
│   └── README.md
│
├── docs/
│   └── methodology.md
│
└── tests/
    └── test_risk_model.py
```

The key design decision is that **notebooks are the research interface; Python modules contain the model logic**. This makes the analysis easier to test, reuse and audit.

## Running the project

```bash
python -m pip install -r requirements.txt
```

For public market data, the data layer can use Yahoo Finance through `yfinance`. For a controlled research environment, the same model accepts pre-cleaned price DataFrames, keeping the risk engine independent of the data vendor.

Example:

```python
from multi_factor_risk.pipeline import run_risk_analysis

result = run_risk_analysis(
    prices=prices,
    weights=weights,
    n_factors=5,
)

print(result.portfolio_volatility)
print(result.risk_contribution)
```

## Data engineering principles

This is deliberately a **data-aware** project rather than a data-vendor-specific one.

- Raw market data is not committed to the repository.
- Data ingestion and model estimation are separate layers.
- Missing values are handled explicitly before estimation.
- Asset ordering is preserved between returns, exposures and portfolio weights.
- Model parameters are configurable rather than embedded throughout notebooks.
- The risk engine works on clean tabular inputs, allowing the upstream data source to change without rewriting the model.

## Validation and limitations

A credible risk model is defined as much by its limitations as by its equations.

This implementation does **not** claim to be a production institutional risk system. Important limitations include:

- PCA factors are sample-dependent and may not have stable economic interpretations.
- Covariance estimates are sensitive to the estimation window.
- Historical relationships can break during structural market regimes.
- Public market data can contain survivorship, corporate-action and missing-data issues.
- Signal backtests require careful treatment of look-ahead bias, trading costs, shorting constraints and liquidity.
- A statistical covariance model should be complemented by stress testing and scenario analysis in a production setting.

These limitations are intentional discussion points: a risk model should expose assumptions instead of hiding them behind a single volatility number.

## Finance concepts demonstrated

**Portfolio construction:** diversification, concentration, long/short weights, risk budgets.

**Risk modelling:** covariance estimation, PCA, systematic risk, idiosyncratic risk, factor exposure.

**Portfolio analytics:** volatility, marginal risk, component risk contribution and factor-level attribution.

**Quantitative research:** momentum, reversal, overnight returns, cross-sectional normalization and signal smoothing.

**Data discipline:** reproducible inputs, explicit transformations, validation and separation of ingestion from modelling.

## What I would extend next

1. Exponentially weighted covariance estimation.
2. Shrinkage of the factor and idiosyncratic covariance estimates.
3. Economic factor overlays such as value, size, quality and low volatility.
4. Sector and country neutrality constraints.
5. Stress testing against historical and hypothetical factor shocks.
6. Transaction-cost and liquidity-aware portfolio optimization.
7. Automated data-quality checks and model monitoring.

## Author

**Rajdeep Rajan** — engineering background with a finance-first focus across investment banking, capital markets and quantitative investment analysis.

The objective is not to present a black-box quant strategy. It is to demonstrate that an investment professional can understand the financial problem, formulate the statistical model, implement the data pipeline and explain the resulting portfolio risk in business terms.
