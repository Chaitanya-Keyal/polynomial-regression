CS F320 Foundations of Data Science, Assignment 1: Polynomial, Ridge, Lasso and Bayesian Regression

Group
  Chaitanya Keyal   2023B4A70727H
  Abhishek Verma    2023B4A71256H
  Palash Jain       2023A2B51271H

Report
  report/report.pdf holds every deliverable of P1 to P4: the figures, the numbers and the
  discussions. report/report.tex is its source and reads every number from results/numbers.tex.
  Repository: https://github.com/Chaitanya-Keyal/polynomial-regression

Versions used
  Python 3.12.10, numpy 2.5.4, matplotlib 3.11.2. The code imports only numpy, matplotlib and the
  Python standard library. It also runs on Python 3.10 with numpy 1.26 and matplotlib 3.8; there
  only the values set by rounding at the floating-point floor come out different, which is why
  the report prints those to one significant figure.

How to run
  From this folder, with uv (uv.lock pins every version):

    uv sync
    uv run python -m src.run          all four problems, under 10 s
    uv run python -m src.run p2 p3    some of them
    uv run pytest                     29 tests of the solvers and of each problem's maths

  or, with any Python 3.10+ that has numpy and matplotlib: python -m src.run

  Each problem writes results/<p>.json (every number it reports) and its figures to figures/, and
  results/numbers.tex is rebuilt from the JSON; cd report && latexmk -pdf report.tex then rebuilds
  the report. --out DIR writes results/ and figures/ under DIR instead. The P1 shuffling uses seed
  42, so a rerun on the same machine gives the same files byte for byte.

What is where
  data/                    the six provided CSVs, unchanged
  src/p1.py ... src/p4.py  one module per problem: compute() for the numbers, figures() for plots
  src/common/              loading, the bases (monomial, Legendre), solvers, metrics, plot style
  src/run.py               the runner above; src/report_numbers.py writes results/numbers.tex
  results/, figures/       the outputs the report uses
  tests/                   checks on the provided data, e.g. that an exact Lasso solver gives w*
  notebooks/demo.ipynb     the demo: calls src/ and shows each problem's figures and numbers

Choices the handout leaves open
  - No explicit matrix inverse anywhere: least squares by np.linalg.lstsq (SVD), ridge and the
    Bayesian posterior by np.linalg.solve, w = R^-1 alpha by a solve with R.
  - P1: a fresh np.random.default_rng(42) for each GD run; SGD and Mini-batch shuffle every epoch
    with rng.permutation, and Mini-batch takes consecutive slices of that order (3, 3, 3, 1).
    The loss curves and the verification share one 2000-epoch run at eta = 0.1.
  - P2: GD stops at the first iteration whose relative error is below 1e-6. On this test set the
    points within 1.05x of the minimum are not one interval: the report gives the contiguous run
    around lambda* as the interval and names the isolated point lambda = 1e-4.
  - P3: the basis is P_j(2x - 1), j = 0..9 (np.polynomial.legendre.legvander). Ridge in the same
    basis is solved in closed form on the same grid.
  - P4: S_N is never formed; m_N and phi^T S_N phi come from solves with the posterior precision
    alpha I + beta Phi^T Phi. Mean and largest interval widths are over 200 points in [0, 1].
