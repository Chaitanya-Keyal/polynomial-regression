# P2: ridge regression as a MAP estimate

## In short

Put a zero-mean Gaussian prior on the weights and the most probable weights given the data (the
MAP estimate) minimise the squared error plus λ/2 ‖w‖², with λ = α/β: ridge regression. It has a
closed form, (ΦᵀΦ + λI) w = Φᵀt, which we solve for 49 values of λ in the degree-9 monomial
basis of P1. The penalty also adds λ to every eigenvalue of ΦᵀΦ, which turns a hopeless
condition number (≈ 10¹⁴) into a manageable one (≈ 2 × 10³ at λ = 10⁻²), so plain GD in the
original basis now works (about 30 000 iterations). Small λ overfits, large λ underfits, and λ*
is the grid value with the lowest test RMSE.

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
(Figure 4).

- **Stability**: the Hessian is H = ΦᵀΦ + λI, so η must be below 2/λ_max(H) ≈ 0.098. η = 0.04 is
  safe; η = 0.1 would diverge.
- **Speed**: the slowest direction is H's smallest eigenvalue, about λ = 0.01 (ΦᵀΦ contributes
  ~10⁻¹³). It shrinks by 1 − ηλ = 0.9996 per step, so 10⁻⁶ takes at most
  ln(10⁶) / (ηλ) ≈ 35 000 steps; fewer in practice because the error does not start entirely in
  that direction.
- **Conditioning**: κ(ΦᵀΦ) ≈ 2.3 × 10¹⁴ and κ(ΦᵀΦ + λI) ≈ 2.0 × 10³. The one-sentence answer for
  the report: the penalty lifts every eigenvalue by λ, so the smallest goes from almost 0 to about
  λ and the condition number drops by eleven orders of magnitude.
- Ẽ decreases at every step (GD on a convex quadratic with η < 2/λ_max), which the test checks.

## Choosing λ

- Train and test RMSE for every λ on `np.logspace(-10, 2, 49)` (Figure 5). Train RMSE rises with
  λ, always: more penalty can only fit the training data worse. Test RMSE is U-shaped.
- λ* = the grid value with the lowest test RMSE. `flat_interval` returns the contiguous run of
  grid points around λ* whose test RMSE is within 1.05× the minimum, and flags it if some other
  grid point also qualifies outside that run.
- Below the interval: the penalty is too weak, the polynomial chases the noise; train RMSE drops
  under the noise level 0.1, test RMSE climbs. Above it: the weights are shrunk so hard the fit
  flattens; train and test RMSE rise together.
- The 0.1 line is the noise standard deviation: on new data no model can beat it on average, so a
  train RMSE well below 0.1 is a sign of fitting noise.
- Honest caveat for the viva: picking λ on the test set (as the handout asks) makes the test RMSE
  at λ* optimistic; in practice λ is picked on a validation set or by cross-validation.

## Weight norms

Figure 6 and Table 4: ‖w‖ falls by orders of magnitude from λ = 10⁻¹⁰ to λ = 10⁻². The overfit
polynomial needs large, cancelling coefficients to wiggle through the noise; the penalty makes
those expensive.

## Demo

1. `uv run python -m src.run p2`; open `p2_gd.png`, `p2_rmse.png`, `p2_fits.png`.
2. Point at the iteration count on the right panel of `p2_gd`, and at λ* and the shaded band on
   `p2_rmse`.
3. Show `ridge` in `src/common/linalg.py` (a solve, no inverse) and the update line in `ridge_gd`.

## Likely viva questions

- *Derive ridge from MAP.* As above; λ = α/β.
- *Why is (ΦᵀΦ + λI) always invertible?* All its eigenvalues are ≥ λ > 0.
- *Why does GD work here but not on plain least squares in this basis?* κ drops from ~10¹⁴ to
  ~10³, and the rate is set by 1 − ηλ_min.
- *How was η chosen / what is the largest safe η?* 2/λ_max(ΦᵀΦ + λI) ≈ 0.098.
- *Why roughly 30 000 iterations?* ln(10⁶)/(ηλ) ≈ 35 000 is the worst case.
- *Sharp or flat minimum?* Read Figure 5: how many decades the 1.05× band spans.
- *Why does ‖w‖ shrink as λ grows?* That is exactly what the penalty charges for.
- *Is choosing λ on the test set a problem?* Yes in practice (optimistic estimate); the handout
  asks for it.

## Likely code changes

- GD at another λ or η: `GD_LAM`, `GD_ETA` in `src/p2.py`. A larger λ converges faster (better
  conditioned); η above 0.098 diverges (`ridge_gd` then raises after `GD_MAX_ITERS`).
- A tighter or looser stopping rule: `GD_TOL`.
- A different λ grid: `LAMS`.
- Do not penalise w₀: replace `lam * np.eye(...)` in `ridge` by a diagonal with a 0 in the first
  entry (and the same in the GD update).
