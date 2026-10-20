# P3: L1-regularised regression via subgradient descent

## In short

The Lasso adds λ‖w‖₁ to the squared error. |w| has a corner at 0, so there is no closed form and
no gradient there; subgradient descent uses sign(w) instead (with sign(0) = 0). It is not a
descent method, so E jitters and we also track the best E seen. With a constant step, coefficients
that should be zero chatter around 0 with amplitude about ηλ, so "≈ 0" means |w_j| < 2ηλ. The
model is the same degree-9 polynomial, written in Legendre polynomials because that basis is far
better conditioned and L1 adds no curvature to help. L1 sets coefficients exactly to zero; L2
only shrinks them. At λ* = 0.1 the Lasso has w₀, w₆, w₈ and w₉ ≈ 0: exactly the zeros of the
handout's reference w*.

## The basis

- u = 2x − 1 maps [0, 1] to [−1, 1]; Φ_nj = P_j(u_n) with P₀ = 1, P₁ = u and
  (j + 1) P_{j+1} = (2j + 1) u P_j − j P_{j−1}. `legendre` in `src/common/bases.py` is the
  handout's one-liner, `np.polynomial.legendre.legvander(2 * x - 1, d)`, which evaluates exactly
  these polynomials; `tests/test_common.py` checks P₂ by hand.
- Same model class: P_j has degree j, so the ten P_j span the same polynomials as 1, x, ..., x⁹,
  and the unregularised fit is the same function.
- Conditioning: κ(ΦᵀΦ) ≈ 6.1 × 10² instead of 2.3 × 10¹⁴, because Legendre polynomials are
  orthogonal on [−1, 1] and their columns are nearly uncorrelated. Ridge's penalty fixed
  conditioning in P2; the L1 term adds no curvature (|w| is flat away from 0), so without a better
  basis subgradient descent would crawl for small λ.
- The penalty is on *these* coefficients: "w₆ = 0" means the P₆ component is zero, not that x⁶
  is missing.
- Symmetry: sin(2πx) = sin(π(u + 1)) = −sin(πu) is odd in u, and P_j is odd for odd j, even for
  even j. So the target's own Legendre coefficients are about
  (0, −0.955, 0, 1.158, 0, −0.219, 0, 0.0166, 0, −0.0007): every even one is exactly 0, and the
  high odd ones w₇ and w₉ are tiny. Only w₁, w₃ and w₅ carry real signal; the rest of the fit is
  mostly noise. That does not make the even coefficients the first to go (see L1 vs. L2).

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

## The check at λ = 0.1

The handout's E* = 0.2625 and w* come from cyclic coordinate descent: for each j in turn,
ρ = Φ_jᵀ(r + Φ_j w_j), w_j = S(ρ, λ) / ‖Φ_j‖², where S(z, λ) = sign(z) max(|z| − λ, 0) is
soft-thresholding (it sets w_j to exactly 0 when |ρ| ≤ λ). `tests/test_p3.py` keeps that solver,
`lasso_cd`, as an exact reference: subgradient descent must land within 5ηλ of it with the same
≈ 0 set, and the optimality conditions must hold exactly at its solution. On the data
(`results/p3.json`, `check`):

| `check` at λ = 0.1 | value |
|---|---|
| `E_final` (handout: E* = 0.2625) | 0.2636 |
| `w_minus_w_ref`, ‖w − w*‖ | 0.0076 |
| `optimality.max_active_violation` (6 active coefficients) | 0.0144 |
| `optimality.max_zero_ratio` (4 ≈ 0 coefficients, must be ≤ 1) | 0.683 |

The gaps are what the ηλ = 0.005 chatter leaves: E sits 0.001 above E*, and the active conditions
hold to about 0.01 rather than exactly. The ≈ 0 set, {0, 6, 8, 9}, is exactly the zeros of w*,
which also confirms that the files and the basis match the handout's.

## L1 vs. L2

- **Why L1 zeroes**: its subgradient at 0 is a whole interval, so w_j = 0 is optimal for any
  correlation up to λ: a finite threshold. Push λ up and coefficients cross it one by one
  (`p3_path`, hollow markers ≈ 0): w₉ first (λ = 0.01), then w₀ and w₆ (0.0158), w₈ (0.0398), w₄
  and w₇ (0.158), w₂ (0.251) and w₅ (0.398); w₁ and w₃ never do on the grid. The order follows
  each column's correlation with the residual, noise included, not parity alone: the noise in ten
  points keeps w₂ and w₄ alive longer than w₇.
- **Why L2 does not**: the gradient of λ/2 w_j² is λ w_j, which vanishes at 0. The pull toward zero
  weakens as the coefficient shrinks, so it never quite gets there unless Φ_jᵀ r = 0 exactly.
- **Geometry**: minimise SSE subject to ‖w‖ ≤ c. The ℓ1 ball is a diamond with corners on the axes;
  the elliptical SSE contours usually touch it first at a corner, where some coordinates are 0.
  The ℓ2 ball is round, so the first touch is generically off the axes.

Each at its best λ (`table`, `p3_fits`), the two have similar norms, but the Lasso puts its weight
on fewer coefficients and fits the test set better:

| | best λ | test RMSE | ‖w‖ | coefficients below 0.01 in size |
|---|---|---|---|---|
| Lasso | 0.1 | 0.0985 | 1.467 | 4 |
| Ridge (Legendre) | 0.158 | 0.142 | 1.441 | 1 |

- The Lasso's 0.0985 is just under the noise line, but it does not beat the noise: this test
  set's own noise RMSE (of t − sin(2πx)) is about 0.091, and an RMSE over 100 points varies by
  about 0.007 (0.1/√200) from one test set to another.
- Past λ*, the penalty removes w₄, w₇ and w₂, then w₅, which carries signal, and shrinks w₁ and
  w₃: train and test RMSE rise together (0.075 and 0.0985 at λ*, both about 0.35 at λ = 1), and
  bias takes over from variance.

## Demo

1. `uv run python -m src.run p3` (about 4 s); open `p3_convergence.pdf` (the right panel is
   E − E*, the gap to the handout's 0.2625, where the rises show), `p3_path.pdf`, `p3_rmse.pdf`,
   `p3_fits.pdf`.
2. In `results/p3.json`, `check`: `E_final` vs. the reference 0.2625, `w_minus_w_ref`, and the two
   `optimality` numbers.
3. Show the update line in `subgradient_descent` and `near_zero`.

## Likely viva questions

- *Why no closed form?* |w| is not differentiable at 0; the optimality condition is a set
  inclusion, not an equation.
- *What is a subgradient?* Any slope of a line through the point that stays below the function.
- *Why does E go up?* A subgradient step need not decrease E; the sign term overshoots near 0.
- *Why 2ηλ?* The chatter amplitude is about ηλ; twice that is a safe "zero" band.
- *Why Legendre?* Same model, κ ≈ 6 × 10² instead of 10¹⁴, and L1 brings no curvature of its own.
- *Which coefficients go first, and why?* w₉, then w₀ and w₆, w₈, w₄ and w₇, w₂, w₅; w₁ and w₃
  never. The target is odd in u, so its even coefficients are 0 and w₇, w₉ tiny, but with ten
  noisy points the noise decides part of the order: w₂ and w₄ outlast w₇.
- *Test RMSE 0.0985 is below the noise level: is the Lasso beating the noise?* No. The 0.1 line is
  the average over fresh data; on this test set even sin(2πx) itself scores 0.091.
- *How would you get exact zeros?* Coordinate descent / soft-thresholding (`lasso_cd` in
  `tests/test_p3.py`), or a proximal gradient step (ISTA): w ← S(w − η Φᵀ(Φw − t), ηλ).

## Likely code changes

- η, iterations, grid: `ETA`, `ITERS`, `LAMS` in `src/p3.py`. η above 0.161 diverges.
- Diminishing step: in `subgradient_descent`, loop over `k in range(1, ITERS + 1)` and step with
  `ETA / np.sqrt(k)` instead of `ETA`.
- Proximal step instead of the subgradient (exact zeros, and E falls every step): in
  `subgradient_descent`, `z = w - ETA * Phi.T @ (Phi @ w - t)`, then
  `w = np.sign(z) * np.maximum(np.abs(z) - ETA * lam, 0)`.
- Monomial basis instead: `monomial` for `legendre` in `compute` and `fits_figure`; at small λ the
  fit stops far from the optimum after 20 000 iterations, and w* no longer applies (it is in
  Legendre coefficients).
