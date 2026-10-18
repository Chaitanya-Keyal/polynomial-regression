# P4: Bayesian regression

## In short

Keep the whole posterior over the weights instead of one best guess. With a Gaussian prior and
Gaussian noise the posterior is Gaussian: mean m_N, covariance S_N. Its mean is exactly the ridge
(MAP) solution with λ = α/β, which we confirm numerically to about 10⁻¹². Predictions get a mean
and a variance s²(x) = 1/β + φ(x)ᵀ S_N φ(x): observation noise plus uncertainty about the
weights. More data shrinks the second term roughly like 1/N; the first never moves, so the 95%
band narrows toward a floor of 2 × 1.96 × 0.1 = 0.392 and stops there.

## Setup

Cubic basis φ(x) = [1, x, x², x³], prior precision α = 0.05, noise precision β = 1/0.01 = 100,
fixed for every N. So λ = α/β = 5 × 10⁻⁴.

## The posterior

- log p(w | t) = −β/2 ‖Φw − t‖² − α/2 ‖w‖² + const
  = −½ wᵀ(αI + βΦᵀΦ) w + β wᵀ Φᵀ t + const, a quadratic in w, so the posterior is Gaussian with
  precision S_N⁻¹ = αI + βΦᵀΦ and mean m_N = β S_N Φᵀ t (complete the square).
- **MAP = mean**: m_N = β(αI + βΦᵀΦ)⁻¹Φᵀt = (ΦᵀΦ + (α/β) I)⁻¹Φᵀt, which is the ridge solution
  with λ = α/β. A Gaussian peaks at its mean, so the mode (MAP) and the mean are the same point.
  The numerical difference (~10⁻¹²) is rounding from two different solves.
- **Predictive**: a new target t = wᵀφ(x) + noise. Its mean is φ(x)ᵀ m_N; its variance is the
  noise variance 1/β plus the variance of wᵀφ(x) under the posterior, φ(x)ᵀ S_N φ(x).

## The code

`src/p4.py`:

```python
L = np.linalg.cholesky(alpha * np.eye(4) + beta * Phi.T @ Phi)    # S_N^-1 = L L^T
m = np.linalg.solve(L.T, np.linalg.solve(L, beta * Phi.T @ t))     # m_N, two solves
param_var = (np.linalg.solve(L, Phi_new.T) ** 2).sum(axis=0)       # phi^T S_N phi = |L^-1 phi|^2
```

S_N is never formed. `tests/test_p4.py` checks all three against the textbook formulas computed
with explicit inverses (fine in a test).

## Why more data narrows the band, down to a floor

- Every point adds β φ(x_n) φ(x_n)ᵀ to the precision S_N⁻¹. For evenly spaced x,
  ΦᵀΦ ≈ N × (a fixed matrix), so S_N ≈ (βN × fixed)⁻¹ and the parameter term falls roughly
  like 1/N.
- The band is widest at the ends of [0, 1]: a cubic is least pinned down where there is data on
  one side only.
- The noise term 1/β is the scatter of a new observation around the true curve. No amount of data
  about w reduces it, so the width tends to 2 × 1.96 × β^(−1/2) = 0.392, never below.
- **s²(x) does not depend on the t values at all**, only on where the inputs are. So the width
  plot (Figure 12) is fixed by the x positions in the files: if they are evenly spaced, as the
  handout says, it was already final before the data arrived.

## Demo

1. `uv run python -m src.run p4`; open `p4_predictive.png` (the dashed MAP curve sits on the mean)
   and `p4_width.png` (four curves stepping down to the floor).
2. In `results/p4.json`: `w_map_minus_m_N` and the `width` table (the parameter share falls as N
   grows).

## Likely viva questions

- *Why is the posterior Gaussian?* Gaussian prior × Gaussian likelihood is the exponential of a
  quadratic in w (conjugacy).
- *Why does MAP equal the posterior mean?* Same formula after dividing by β; a Gaussian's mode is
  its mean.
- *What are the two terms of s²(x)?* Observation noise 1/β (irreducible) and parameter uncertainty
  φᵀS_Nφ (reducible with data).
- *Why is the band wider at the edges?* Fewer neighbouring points constrain the cubic there.
- *What happens as N → ∞?* The parameter term → 0; the width → 0.392.
- *Effect of a larger α?* A tighter prior: smaller S_N (narrower band) but m_N pulled toward 0
  (more bias, the fit flattens). A larger β trusts each point more: narrower band, weaker
  effective prior (λ = α/β falls).
- *Does the band depend on the observed y?* No, only on the x positions (for fixed α, β).

## Likely code changes

- α, β or the degree: `ALPHA`, `BETA`, `D` in `src/p4.py` (everything else follows, including
  λ = α/β and the floor).
- A 99% band: change `Z` to 2.576.
- Plot the parameter term alone: `predictive` already returns it as its third value.
