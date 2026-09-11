# Methodology

## 1. Return matrix

The model starts from a date-by-asset matrix of simple returns. The data layer validates dates, positive prices and asset alignment before estimation.

## 2. Statistical factors

PCA is applied to the cross-section of asset returns. Each retained component represents a latent common source of return variation. The number of factors is configurable.

The PCA loadings form the exposure matrix **B** and the transformed observations form the factor-return matrix **F**.

## 3. Specific risk

After reconstructing returns through the retained factors, the residual is:

`epsilon = R - F B'`

The diagonal of the residual covariance matrix provides the idiosyncratic variance vector **D**.

## 4. Covariance

The model combines common-factor and specific risk:

`Sigma = B Omega B' + D`

where **Omega** is the covariance matrix of factor returns.

This structure is preferable to treating every pairwise asset covariance independently because it makes the major common drivers explicit and allows risk to be attributed back to the factor layer.

## 5. Portfolio risk

For weights `w`:

`portfolio variance = w' Sigma w`

`portfolio volatility = sqrt(w' Sigma w)`

Risk contribution is calculated from `Sigma w`, allowing each position's contribution to portfolio variance to be inspected.

## 6. Signal research

The signal layer is intentionally separate from the risk model. Momentum, short-term reversal and overnight returns are research hypotheses. Their historical performance does not alter the definition of the statistical covariance model.

This distinction matters: **alpha answers what may generate return; the risk model answers what may generate co-movement and loss.**

## 7. Model risk

Key areas for further validation are estimation-window sensitivity, number of PCA components, regime stability, missing observations, corporate actions, survivorship bias, look-ahead bias and transaction costs.
