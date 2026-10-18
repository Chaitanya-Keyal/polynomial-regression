# polynomial-regression

CS F320 (Foundations of Data Science), Assignment 1.

| Problem | Topic |
|---------|-------|
| P1 | Polynomial regression: closed form vs. Batch / Stochastic / Mini-batch GD (QR-reparametrised) |
| P2 | Ridge regression as a MAP estimate: closed form vs. GD, λ sweep |
| P3 | Lasso via subgradient descent in a Legendre basis, compared with ridge |
| P4 | Bayesian linear regression: MAP/posterior-mean equivalence, predictive uncertainty vs. N |

## Data

Put the provided CSV files (columns `x,y`) in `data/`:

```
poly_train.csv  poly_test.csv
p4_train_N5.csv  p4_train_N10.csv  p4_train_N20.csv  p4_train_N50.csv
```

The data is fixed. Nothing in this repo generates or modifies it.

## Setup

```bash
uv sync
```
