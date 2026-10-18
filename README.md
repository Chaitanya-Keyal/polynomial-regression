# Polynomial Regression: Closed Form, Gradient Descent, Ridge, Lasso and Bayesian

Polynomial regression on noisy samples of sin(2πx), built from scratch in NumPy: least squares,
the gradient descent family, ridge, lasso and Bayesian linear regression.

| Problem | What it does |
|---------|--------------|
| P1 | Degree 1-9 fits by least squares; Batch, Stochastic and Mini-batch GD for d = 9 on the QR-reparametrised problem; learning-rate study |
| P2 | Ridge as the MAP estimate: closed form vs. GD in the monomial basis, λ sweep, train/test RMSE |
| P3 | Lasso by subgradient descent in a Legendre basis: λ sweep, coefficient path, sparsity, comparison with ridge |
| P4 | Bayesian linear regression: posterior predictive, MAP = posterior mean, uncertainty vs. N |

## 1. Layout

```
assignment.pdf               the handout
data/                        the six provided CSVs (columns x,y), used as given
pyproject.toml, uv.lock      dependencies; .python-version pins Python 3.12
```

## 2. Environment

```bash
uv sync                      # numpy and matplotlib: enough to run everything
uv sync --group dev          # adds pytest and ruff
uv sync --group notebook     # adds Jupyter, for the demo notebook
```

`pyproject.toml` declares the dependencies and `uv.lock` pins exact versions: Python 3.12,
NumPy 2.5.4, Matplotlib 3.11.2; for tests and lint pytest 9.1.1 and ruff 0.17.1.

Lint, formatting and tests:

```bash
uv run ruff format . && uv run ruff check . && uv run python -m pytest tests
```

## 3. Data

The six files the assignment provides go in `data/` unchanged:

```
poly_train.csv               N = 10, x evenly spaced in [0, 1]     P1, P2, P3
poly_test.csv                N = 100, x uniform in [0, 1]          P2, P3
p4_train_N{5,10,20,50}.csv   x evenly spaced, noise variance 0.01  P4
```

The data is fixed: nothing in this repo generates or modifies it. sin(2πx) appears only in plots.
