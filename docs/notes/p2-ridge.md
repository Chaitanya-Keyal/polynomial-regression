# P2: ridge regression as a MAP estimate

## In short

Put a zero-mean Gaussian prior on the weights and the most probable weights given the data (the
MAP estimate) minimise the squared error plus λ/2 ‖w‖², with λ = α/β: ridge regression. It has a
closed form, (ΦᵀΦ + λI) w = Φᵀt, which we solve for 49 values of λ in the degree-9 monomial
basis of P1. The penalty also adds λ to every eigenvalue of ΦᵀΦ, which turns a hopeless
condition number (≈ 2.3 × 10¹⁴) into a manageable one (≈ 2.0 × 10³ at λ = 10⁻²), so plain GD in
the original basis now works (28 227 iterations). Small λ overfits, large λ underfits, and
λ* = 1.78 × 10⁻⁷ is the grid value with the lowest test RMSE (0.115). The minimum is flat, not a
sharp U.

## MAP ⇒ ridge

- Bayes: p(w | t) ∝ p(t | w) p(w), likelihood ∝ exp(−β/2 ‖Φw − t‖²), prior ∝ exp(−α/2 ‖w‖²).
- −log posterior = β/2 ‖Φw − t‖² + α/2 ‖w‖² + const. Divide by β (it does not move the minimum):
  Ẽ(w) = ½‖Φw − t‖² + λ/2 ‖w‖², λ = α/β.
- α is how strongly we believe the weights are small (prior precision); β how much we trust each
  data point (noise precision). Only their ratio matters for the point estimate.
- ∇Ẽ = Φᵀ(Φw − t) + λw = 0 gives (ΦᵀΦ + λI) w = Φᵀt. For λ > 0 the matrix is symmetric positive
  definite (every eigenvalue is at least λ), so the solution exists, is unique, and `solve`
  always works (`src/common/linalg.py`, `ridge`).
- Note the penalty includes w₀, the intercept, exactly as the handout's formula does.

## GD at λ = 10⁻²

`src/p2.py`, `ridge_gd`: from w = 0, w ← w − η (Φᵀ(Φw − t) + λw) with η = 0.04, stopping as soon
as ‖w − w_closed‖ / ‖w_closed‖ < 10⁻⁶. It records Ẽ and that relative error at every iteration
(`p2_gd`). It stops after 28 227 iterations, with Ẽ equal to the closed form's to 12 digits.

- **Stability**: the Hessian is H = ΦᵀΦ + λI, so η must be below 2/λ_max(H) ≈ 0.098. η = 0.04 is
  safe; η = 0.1 would diverge.
- **Speed**: the slowest direction is H's smallest eigenvalue, about λ = 0.01 (ΦᵀΦ contributes
  ~10⁻¹³). It shrinks by 1 − ηλ = 0.9996 per step, so 10⁻⁶ takes at most
  ln(10⁶) / (ηλ) ≈ 35 000 steps; 28 227 in practice, because the error does not start entirely in
  that direction.
- **Conditioning**: κ(ΦᵀΦ) ≈ 2.3 × 10¹⁴ and κ(ΦᵀΦ + λI) ≈ 2.0 × 10³. The one-sentence answer for
  the report: the penalty lifts every eigenvalue by λ, so the smallest goes from almost 0 to about
  λ and the condition number drops by eleven orders of magnitude.
- Ẽ decreases at every step (GD on a convex quadratic with η < 2/λ_max), which the test checks.

## Choosing λ

- Train and test RMSE for every λ on `np.logspace(-10, 2, 49)` (`p2_rmse`). Train RMSE rises with
  λ, always: more penalty can only fit the training data worse.
- λ* = the grid value with the lowest test RMSE: 1.78 × 10⁻⁷, test RMSE 0.1146.
  `within_band(test)` returns its index, the contiguous run of grid points around it with test
  RMSE ≤ 1.05× the minimum (`WITHIN`), and any qualifying grid points outside that run.
- Test RMSE is not U-shaped. Around λ* it has a flat valley, and the 1.05× run is
  λ ∈ [3.16 × 10⁻⁸, 1.78 × 10⁻⁶] (8 grid points). Past it, test RMSE rises to a shallow peak of
  1.066× the minimum at λ = 10⁻⁵, then dips again, so the isolated grid point λ = 10⁻⁴ (1.046×)
  meets the 1.05× rule too (`flat`: `lam_lo`, `lam_hi`, `n_run`, `others`, `peak_lam`,
  `peak_ratio`). Overall test RMSE stays within 1.07× of its minimum from about 3 × 10⁻⁸ to
  2 × 10⁻⁴, almost four decades: a flat minimum.
- Below the interval the penalty is too weak and the polynomial chases the noise: at λ = 10⁻¹⁰
  train RMSE is 0.016, test 0.161, ‖w‖ ≈ 3979. Above it, test RMSE stays near 0.12 up to about
  10⁻⁴; then the weights are shrunk so hard that the fit flattens, and train and test RMSE rise
  together (both 0.266 at λ = 10⁻², about 0.7 at λ = 100).
- The 0.1 line is the noise standard deviation: on new data no model can beat it on average. Train
  RMSE is below it for every λ up to about 5.6 × 10⁻⁴ (ten weights for ten points always absorb
  some noise), so it is the gap between train and test, not the train RMSE alone, that shows
  overfitting.
- Honest caveat for the viva: λ* is picked on the test set, as the handout asks, which makes the
  test RMSE at λ* optimistic; in practice λ is picked on a validation set or by cross-validation.

## Weight norms

`p2_fits` and `panels`: ‖w‖ = 3979 at λ = 10⁻¹⁰ (overfit), 124 at λ*, 5.1 at λ = 10⁻² (underfit).
The overfit polynomial needs large, cancelling coefficients to wiggle through the noise; the
penalty makes those expensive.

## Demo

1. `uv run python -m src.run p2`; open `p2_gd.pdf`, `p2_rmse.pdf`, `p2_fits.pdf`.
2. Point at the iteration count on the right panel of `p2_gd`, and on `p2_rmse` at λ*, the flat
   valley around it and the second shallow dip at 10⁻⁴ (`flat` in `results/p2.json`).
3. Show `ridge` in `src/common/linalg.py` (a solve, no inverse) and the update line in `ridge_gd`.

## Likely viva questions

- *Derive ridge from MAP.* As above; λ = α/β.
- *Why is (ΦᵀΦ + λI) always invertible?* All its eigenvalues are ≥ λ > 0.
- *Why does GD work here but not on plain least squares in this basis?* κ drops from ~10¹⁴ to
  ~10³, and the rate is set by 1 − ηλ_min.
- *How was η chosen / what is the largest safe η?* 2/λ_max(ΦᵀΦ + λI) ≈ 0.098.
- *Why 28 227 iterations?* ln(10⁶)/(ηλ) ≈ 35 000 is the worst case.
- *Sharp or flat minimum?* Flat: test RMSE is within 1.07× of its minimum over almost four
  decades of λ.
- *Why is the 1.05× set not one interval?* Test RMSE rises to a shallow peak (1.066× at
  λ = 10⁻⁵) and dips again, so λ = 10⁻⁴ (1.046×) qualifies too. Report both pieces: the run
  [3.16 × 10⁻⁸, 1.78 × 10⁻⁶] around λ* and the single point 10⁻⁴. The bump is small (0.122
  against 0.115) and comes from this one test set: it says the minimum is flat, not that 10⁻⁵ is
  a bad choice.
- *Why does ‖w‖ shrink as λ grows?* That is exactly what the penalty charges for.
- *Is choosing λ on the test set a problem?* Yes in practice (optimistic estimate); the handout
  asks for it.

## Likely code changes

- GD at another λ or η: `GD_LAM`, `GD_ETA` in `src/p2.py`. A larger λ converges faster (better
  conditioned: 2 963 iterations at λ = 0.1, 286 at λ = 1). η above 0.098 diverges, and
  `ridge_gd` has no iteration cap: it runs until w overflows to nan (about 19 000 iterations at
  η = 0.1).
- A tighter or looser stopping rule: `GD_TOL`.
- A different λ grid: `LAMS`; a different band, say 1.1×: `WITHIN`; other overfit and underfit
  panels: `PANELS`.
- Do not penalise w₀: do it locally in `src/p2.py`, not in the shared `ridge` (P3 and P4 use it
  too) or `ridge_objective`. With `pen = np.r_[0, np.ones(D)]`, solve
  `np.linalg.solve(Phi.T @ Phi + lam * np.diag(pen), Phi.T @ t)` in place of `ridge`, use
  `lam * pen * w` in the `ridge_gd` update, and `0.5 * lam * pen @ w**2` as the penalty in Ẽ.
