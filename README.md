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
src/common/                  data loading, bases (monomial, Legendre), solvers, metrics, plot style
src/p1.py ... src/p4.py      one module per problem: compute() for the numbers, figures() for the plots
src/run.py                   runs the problems and writes results/ and figures/
src/report_numbers.py        results/*.json -> results/numbers.tex, the numbers the report quotes
results/                     p1.json ... p4.json: every reported number (written by src.run)
figures/                     every figure as PDF (for the report) and PNG (written by src.run)
tests/                       hand-worked checks of the solvers and of each problem's maths
report/report.tex            the write-up; numbers and figures come in from results/ and figures/
notebooks/demo.ipynb         the demo: each problem's figures and key numbers, by calling src/
docs/notes/                  study notes: the maths, the code, the demo, likely viva questions
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

## 4. Running

```bash
uv run python -m src.run              # all four problems, about 10 s
uv run python -m src.run p2 p3        # some of them
```

Each problem writes `results/<p>.json` and its figures; `results/numbers.tex` is rebuilt from
whatever JSON is present. The shuffling in P1 uses seed 42, and PDFs carry no timestamp, so a
rerun on the same data reproduces every file byte for byte.

## 5. Report and demo

```bash
cd report && latexmk -pdf report.tex                   # report.pdf
uv run --group notebook jupyter lab notebooks/demo.ipynb
```

The report reads every number through `\res{key}` from `results/numbers.tex`, so rerunning
`src.run` and rebuilding updates it; a number that has not been computed yet prints as **??**.
The notebook has no logic of its own: it calls the same `compute()` and `figures()` as `src.run`.
