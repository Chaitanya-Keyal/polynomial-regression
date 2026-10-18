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
  be plotted, never fitted. (Our smoke tests used stand-in files that live outside the repo.)
- **No explicit inverse.** Least squares via `np.linalg.lstsq` (SVD), ridge via
  `np.linalg.solve`, the Bayesian posterior via a Cholesky factor. See [numerics.md](numerics.md).
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

Several quantities depend only on the x values, not on the y values, so they were already
checked against the handout before the data was released: cond(Φ) ≈ 1.5 × 10⁷ for the degree-9
monomial basis, cond(ΦᵀΦ) ≈ 2.3 × 10¹⁴, cond(ΦᵀΦ) ≈ 6 × 10² for the Legendre basis, the step-size
limits in P2 and P3, and every P4 interval width.

## Deliverables and where they are

Figures are `figures/<name>.pdf` (and `.png`); numbers are keys in `results/<p>.json`.

| Item | Code | Figure / numbers | Report |
|---|---|---|---|
| P1.1 fits d = 1..9 | `p1.compute` (`least_squares`) | `p1_fits`, `closed` | Fig. 1, Table 1 |
| P1.2 Batch, SGD, Mini-batch | `p1.gd`, `p1.batches` | | §1 Method |
| P1.3 loss curves, η = 0.1 | `p1.compute` (`runs`) | `p1_loss`, `verification.*.per_epoch_factor` | Fig. 2 |
| P1.4 learning-rate study | `p1.compute` (`lr`) | `p1_lr`, `lr`, `divergent_w_over_w_closed` | Fig. 3, Table 2 |
| P1.5 verification | `p1.compute` | `verification` | Table 3 |
| P1.6 discussion | | | §1 Discussion |
| P2.1 closed form over the grid | `p2.compute` (`ridge`) | `train_rmse`, `test_rmse`, `w_norm` | §2 |
| P2.2 GD at λ = 10⁻² | `p2.ridge_gd` | `p2_gd`, `gd` | Fig. 4 |
| P2.3 RMSE sweep, λ*, 1.05× interval | `p2.flat_interval` | `p2_rmse`, `lam_star`, `flat` | Fig. 5 |
| P2.4 three fits, ‖w‖ | `p2.compute` (`panels`) | `p2_fits`, `panels` | Fig. 6, Table 4 |
| P3.1 convergence, optimality, w* | `p3.subgradient_descent`, `p3.optimality`, `p3.lasso_cd` | `p3_convergence`, `check` | Fig. 7 |
| P3.2 RMSE sweep | `p3.compute` | `p3_rmse`, `sweep` | Fig. 8 |
| P3.3 coefficient path, ≈ 0 counts | `p3.near_zero` | `p3_path`, `sweep.*.n_zero` | Fig. 9, Table 5 |
| P3.4 λ* and its zero indices | | `lam_star`, `zero_idx_star` | §3 |
| P3.5 Lasso vs. Ridge | `p3.compute` (`table`) | `table` | Table 6 |
| P3.6 fitted curves | | `p3_fits` | Fig. 10 |
| P3.7 discussion | | | §3 Discussion |
| P4.1 predictive plot | `p4.posterior`, `p4.predictive` | `p4_predictive` | Fig. 11 |
| P4.2 MAP = posterior mean | `p4.compute` | `w_map_minus_m_N` | §4 Equivalence |
| P4.3 width vs. N | `p4.compute` | `p4_width`, `width` | Fig. 12, Table 7 |
| P4.4 discussion | | | §4 Discussion |

## When the CSVs arrive

1. Copy the six files into `data/` unchanged and commit them.
2. `uv run python -m src.run`. First check: P3 at λ = 0.1 should give E close to the reference
   0.2625 and w close to the reference w* (`results/p3.json`, `check`). If it does, the files and
   the basis are loaded correctly.
3. Pick the P1 loss-plot range (`LOSS_PLOT_EPOCHS` in `src/p1.py`) so all three curves are shown
   flat, and rerun.
4. Rebuild the report (`cd report && latexmk -pdf report.tex`), read every red `\data{...}`
   sentence against the real figures, fix what is wrong, and remove the markers. `grep -n '\\data{'
   report/report.tex` lists them; `pdftotext report/report.pdf - | grep -c '??'` must print 0.
5. Execute the notebook in place:
   `uv run --group notebook jupyter nbconvert --to notebook --execute --inplace notebooks/demo.ipynb`.
6. Commit `results/`, `figures/`, `report/report.pdf` and the executed notebook.

## Still open

- The submission format and channel (which files, naming, zip or not, page limit).
