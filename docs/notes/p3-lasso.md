# P3: L1-regularised regression via subgradient descent

## In short

The Lasso adds λ‖w‖₁ to the squared error. |w| has a corner at 0, so there is no closed form and
no gradient there; subgradient descent uses sign(w) instead (with sign(0) = 0). It is not a
descent method, so E jitters and we also track the best E seen. With a constant step, coefficients
that should be zero chatter around 0 with amplitude about ηλ, so "≈ 0" means |w_j| < 2ηλ. The
model is the same degree-9 polynomial, written in Legendre polynomials because that basis is far
better conditioned and L1 adds no curvature to help. L1 sets coefficients exactly to zero; L2
only shrinks them.

## The basis

- u = 2x − 1 maps [0, 1] to [−1, 1]; Φ_nj = P_j(u_n) with P₀ = 1, P₁ = u and
  (j + 1) P_{j+1} = (2j + 1) u P_j − j P_{j−1}. `src/common/bases.py`, `legendre`, builds it
  from that recurrence; `tests/test_common.py` checks it against `legvander`.
- Same model class: P_j has degree j, so the ten P_j span the same polynomials as 1, x, ..., x⁹,
  and the unregularised fit is the same function.
- Conditioning: κ(ΦᵀΦ) ≈ 6 × 10² instead of 2 × 10¹⁴, because Legendre polynomials are orthogonal
  on [−1, 1] and their columns are nearly uncorrelated. Ridge's penalty fixed conditioning in P2;
  the L1 term adds no curvature (|w| is flat away from 0), so without a better basis subgradient
  descent would crawl for small λ.
- The penalty is on *these* coefficients: "w₆ = 0" means the P₆ component is zero, not that x⁶
  is missing.
- Symmetry: sin(2πx) = sin(π(u + 1)) = −sin(πu), an odd function of u. Odd P_j (P₁, P₃, P₅, ...)
  are odd, even ones are even, so only odd coefficients carry signal; even coefficients fit noise
  only and are the first the Lasso drops. The reference w* fits this: w₀ = w₆ = w₈ = 0, w₂ and w₄
  small, and w₁ ≈ −0.93, w₃ ≈ 1.12 carry the shape.

## Subgradients and the update

- Subgradient of |w|: −1 for w < 0, +1 for w > 0, any value in [−1, 1] at w = 0 (every line through
  the origin with slope in [−1, 1] stays below |w|). We use sign(0) = 0 (`np.sign`).
- `src/p3.py`, `subgradient_descent`: w ← w − η (Φᵀ(Φw − t) + λ sign(w)), η = 0.05, from w = 0,
  20 000 iterations, recording E every step. Stable because η < 2/λ_max(ΦᵀΦ) ≈ 0.161.
- **Not a descent method**: near the optimum a coefficient at 0 is pushed by +ηλ or −ηλ each step
  as its sign flips, so E rises and falls. The best-so-far E (`np.minimum.accumulate`) is plotted,
  but every reported number uses the final iterate, as the handout says.
- Chatter of size ηλ is the price of a constant step. A diminishing step (η_k → 0 with
  Σ η_k = ∞) would converge exactly, at the cost of speed.

## Optimality conditions (deliverable 1b)

With r = t − Φw and Φ_j the j-th column, w is optimal if and only if, for every j,

- Φ_jᵀ r = λ sign(w_j) when w_j ≠ 0, and
- |Φ_jᵀ r| ≤ λ when w_j = 0.

A coefficient can be exactly zero only when its feature's correlation with the residual is at most
λ. `optimality` reports the worst violation on the non-≈0 coefficients and the largest |Φ_jᵀ r|/λ
on the ≈0 ones (must be ≤ 1).

## The cross-check

`lasso_cd` solves the same problem exactly by cyclic coordinate descent: for each j in turn,
ρ = Φ_jᵀ(r + Φ_j w_j), w_j = S(ρ, λ) / ‖Φ_j‖², where S(z, λ) = sign(z) max(|z| − λ, 0) is
soft-thresholding (it sets w_j to exactly 0 when |ρ| ≤ λ). This is how the handout's w* was made.
It confirms our subgradient answer is within the ηλ chatter of the exact optimum, and that our
data and basis match the handout's (E_CD ≈ 0.2625).

## L1 vs. L2

- **Why L1 zeroes**: its subgradient at 0 is a whole interval, so w_j = 0 is optimal for any
  correlation up to λ: a finite threshold. Push λ up and coefficients cross it one by one
  (coefficient path, Figure 9).
- **Why L2 does not**: the gradient of λ/2 w_j² is λ w_j, which vanishes at 0. The pull toward zero
  weakens as the coefficient shrinks, so it never quite gets there unless Φ_jᵀ r = 0 exactly.
- **Geometry**: minimise SSE subject to ‖w‖ ≤ c. The ℓ1 ball is a diamond with corners on the axes;
  the elliptical SSE contours usually touch it first at a corner, where some coordinates are 0.
  The ℓ2 ball is round, so the first touch is generically off the axes.
- Past λ*, the penalty removes coefficients that carry signal: train RMSE keeps rising, test RMSE
  turns up, bias takes over from variance.

## Demo

1. `uv run python -m src.run p3` (about 4 s); open `p3_convergence.png` (the right panel shows
   the jitter), `p3_path.png`, `p3_rmse.png`, `p3_fits.png`.
2. In `results/p3.json`, `check`: E vs. the reference 0.2625, ‖w − w*‖, and the two optimality
   numbers.
3. Show the update line in `subgradient_descent` and `near_zero`.

## Likely viva questions

- *Why no closed form?* |w| is not differentiable at 0; the optimality condition is a set
  inclusion, not an equation.
- *What is a subgradient?* Any slope of a line through the point that stays below the function.
- *Why does E go up?* A subgradient step need not decrease E; the sign term overshoots near 0.
- *Why 2ηλ?* The chatter amplitude is about ηλ; twice that is a safe "zero" band.
- *Why Legendre?* Same model, κ ≈ 6 × 10² instead of 10¹⁴, and L1 brings no curvature of its own.
- *Why are the even coefficients the zero ones?* The target is odd in u.
- *How would you get exact zeros?* Coordinate descent / soft-thresholding (as in `lasso_cd`), or
  a proximal gradient step (ISTA): w ← S(w − η Φᵀ(Φw − t), ηλ).

## Likely code changes

- η, iterations, grid: `ETA`, `ITERS`, `LAMS` in `src/p3.py`. η above 0.161 diverges.
- Diminishing step: replace `eta` by `eta / np.sqrt(k)` inside the loop.
- Proximal step instead of the subgradient (exact zeros): `w = soft(w - eta * grad, eta * lam)`
  with `soft(z, a) = np.sign(z) * np.maximum(np.abs(z) - a, 0)`.
- Monomial basis instead: `monomial` for `legendre` in `compute`; at small λ the fit stops far
  from the optimum after 20 000 iterations.
