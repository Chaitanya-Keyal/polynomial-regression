# The assignment: what is asked and where it lives

## In short

Four problems on one theme: fit a polynomial to noisy samples of sin(2πx), first by least
squares and gradient descent (P1), then with an L2 penalty read as a Gaussian prior (P2), an L1
penalty (P3), and finally the full Bayesian posterior (P4). Every deliverable is a figure, a few
numbers, or a short discussion; all of them go in the report, and `python -m src.run` produces
every figure and number from the six provided CSVs. Due 23:55 on 27 October; a 20-minute demo and
viva per group on 28 or 29 October.

## Rules that apply everywhere

- **The data is fixed.** Load the six CSVs as given; never generate or modify data. sin(2πx) may
  be plotted, never fitted.
- **No explicit inverse.** Least squares via `np.linalg.lstsq` (SVD), ridge via
  `np.linalg.solve`, the Bayesian posterior via `np.linalg.solve` with its precision matrix. See
  [numerics.md](numerics.md).
- **λ is relative to the unnormalised SSE** ½‖Φw − t‖², not the mean. So the batch gradient is
  the full sum Φᵀ(Φw − t), not divided by N.
- **RMSE** is √(mean squared error) on the given set.
- **State the seed.** P1 shuffles with `np.random.default_rng(42)`, a fresh generator per run.

## The data

| File | N | x | Used in |
|---|---|---|---|
| `poly_train.csv` | 10 | evenly spaced in [0, 1] | P1, P2, P3 |
| `poly_test.csv` | 100 | uniform random in [0, 1] | P2, P3 |
| `p4_train_N{5,10,20,50}.csv` | 5, 10, 20, 50 | evenly spaced in [0, 1] | P4 |

Noise variance is 0.01 (standard deviation 0.1) in every file. That 0.1 is the reference line on
the RMSE plots: on fresh data, no model can do better than about 0.1 on average.

## Deliverables and where they are

Figures are `figures/<name>.pdf`; numbers are keys in `results/<p>.json`. In the report, problem
Pn is section n.

| Item | Code | Figure / numbers |
|---|---|---|
| P1.1 fits d = 1..9 | `p1.compute` (`least_squares`) | `p1_fits`, `closed` |
| P1.2 Batch, SGD, Mini-batch | `p1.gd`, `p1.batches` | |
| P1.3 loss curves, η = 0.1 | `p1.compute` (`runs`) | `p1_loss`, `verification.*.per_epoch_factor`, `sgd_batch_rel_gap` |
| P1.4 learning-rate study | `p1.compute` (`lr`) | `p1_lr`, `E_0`, `E_100`, `w_100_too_large`, `w_100_over_w_closed` |
| P1.5 verification | `p1.compute` | `verification` |
| P1.6 discussion | | |
| P2.1 closed form over the grid | `p2.compute` (`ridge`) | `lams`, `train_rmse`, `test_rmse` |
| P2.2 GD at λ = 10⁻² | `p2.ridge_gd` | `p2_gd`, `gd` |
| P2.3 RMSE sweep, λ*, 1.05× set | `p2.within_band` | `p2_rmse`, `panels.best`, `flat` |
| P2.4 three fits, ‖w‖ | `p2.compute` (`panels`) | `p2_fits`, `panels` |
| P3.1 convergence, optimality, w* | `p3.subgradient_descent`, `p3.optimality` | `p3_convergence`, `check` |
| P3.2 RMSE sweep | `p3.compute` | `p3_rmse`, `sweep` |
| P3.3 coefficient path, ≈ 0 counts | `p3.near_zero` | `p3_path`, `sweep.*.zero_idx` |
| P3.4 λ* and its zero indices | `p3.compute` | `table.lasso`, `zero_idx_star` |
| P3.5 Lasso vs. Ridge | `p3.compute` (`table`) | `table.lasso`, `table.ridge` |
| P3.6 fitted curves | | `p3_fits` |
| P3.7 discussion | | |
| P4.1 predictive plot | `p4.posterior`, `p4.predictive` | `p4_predictive` |
| P4.2 MAP = posterior mean | `p4.compute` | `m_N`, `w_map`, `w_map_minus_m_N` |
| P4.3 width vs. N | `p4.compute` | `p4_width`, `width` |
| P4.4 discussion | | |

## What is where

- `data/`: the six CSVs, unchanged.
- `uv run python -m src.run` writes `results/<p>.json`, `results/numbers.tex` (the same numbers as
  LaTeX macros) and `figures/<name>.pdf`; all three are committed.
- `report/report.tex` reads every number through `\res{key}` and every figure from `figures/`;
  `cd report && latexmk -pdf report.tex` builds the committed `report/report.pdf`.
- `notebooks/demo.ipynb` calls the same `compute()` and `figures()`; it is committed without
  outputs and run live in the demo.
- A quick check that the files and the basis are loaded correctly: P3 at λ = 0.1 lands next to
  the handout's reference (E = 0.2636 vs. 0.2625, ‖w − w*‖ = 0.0076; `results/p3.json`, `check`).

## Still open

- The submission format and channel (which files, naming, zip or not, page limit).
