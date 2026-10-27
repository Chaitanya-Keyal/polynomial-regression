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
results/                     p1.json ... p4.json (every reported number) and numbers.tex
figures/                     every figure, as figures/<name>.pdf
tests/                       hand-worked checks of the solvers and of each problem's maths
report/                      report.tex, the write-up, and report.pdf built from it
notebooks/demo.ipynb         the demo: each problem's figures and key numbers, by calling src/
docs/notes/                  study notes: the maths, the code, the demo, likely viva questions
README.txt                   the submission's readme: group, versions, how to run, open choices
pyproject.toml, uv.lock      dependencies; .python-version pins Python 3.12
```

`results/`, `figures/` and `report/report.pdf` are committed outputs of `python -m src.run` and
latexmk.

## 2. Environment

```bash
uv sync                      # numpy and matplotlib, plus pytest and ruff (the dev group)
uv sync --group notebook     # adds JupyterLab, for the demo notebook
```

`uv.lock` pins every version. Tests, then formatting and lint:

```bash
uv run pytest
uv run ruff format . && uv run ruff check .
```

## 3. Data

`data/` holds the six files the assignment provides, unchanged:

```
poly_train.csv               N = 10, x evenly spaced in [0, 1]     P1, P2, P3
poly_test.csv                N = 100, x uniform in [0, 1]          P2, P3
p4_train_N{5,10,20,50}.csv   x evenly spaced, noise variance 0.01  P4
```

The data is fixed: nothing in this repo generates or modifies it. sin(2πx) appears only in plots.

## 4. Running

```bash
uv run python -m src.run              # all four problems, under 10 s
uv run python -m src.run p2 p3        # some of them
uv run python -m src.run --out DIR    # write results/ and figures/ under DIR instead
```

Each problem writes `results/<p>.json` and its figures; `results/numbers.tex` is rebuilt from
whatever JSON is present. The shuffling in P1 uses seed 42, and PDFs carry no timestamp, so a
rerun on the same machine reproduces every file byte for byte.

## 5. Report and demo

```bash
cd report && latexmk -pdf report.tex                   # report.pdf
uv run --group notebook jupyter lab notebooks/demo.ipynb
```

The report reads every number through `\res{key}` from `results/numbers.tex`, so rerunning
`src.run` and rebuilding updates it. The notebook has no logic of its own: it calls the same
`compute()` and `figures()` as `src.run`, and is committed without outputs to be run live.

## 6. Submission

```bash
git archive --format=zip -o ../submission/2023B4A70727H_A1.zip HEAD -- README.txt \
  pyproject.toml uv.lock .python-version src tests notebooks data results figures report
```

The zip holds the committed files, flat at its root, dated by the commit, so the same commit
always gives the same bytes. It leaves out this README, the handout and `docs/notes/`.
