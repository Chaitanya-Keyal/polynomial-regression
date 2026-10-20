# P4: Bayesian regression

## In short

Keep the whole posterior over the weights instead of one best guess. With a Gaussian prior and
Gaussian noise the posterior is Gaussian: mean m_N, covariance S_N. Its mean is exactly the ridge
(MAP) solution with λ = α/β, which we confirm numerically to about 10⁻¹². Predictions get a mean
and a variance s²(x) = 1/β + φ(x)ᵀ S_N φ(x): observation noise plus uncertainty about the
weights. More data shrinks the second term, roughly like 1/N in the middle of [0, 1] and more
slowly at the ends; the first never moves, so the 95% band narrows toward a floor of
2 × 1.96 × 0.1 = 0.392 and stops there.

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

`src/p4.py`: `posterior` returns m_N and the precision A = S_N⁻¹; `predictive` returns m(x) and
s²(x). Both work through solves with A:

```python
A = ALPHA * np.eye(Phi.shape[1]) + BETA * Phi.T @ Phi  # S_N^-1
m = np.linalg.solve(A, BETA * Phi.T @ t)  # m_N = beta S_N Phi^T t
var = 1 / BETA + (G.T * np.linalg.solve(A, G.T)).sum(axis=0)  # G = monomial(x, D)
```

S_N is never formed: `solve(A, G.T)` gives S_N φ(x) for every x at once, and summing its product
with φ(x) gives φ(x)ᵀ S_N φ(x). `tests/test_p4.py` checks A m_N = βΦᵀt, the predictive mean and
variance, and that m_N equals `ridge(Phi, t, ALPHA / BETA)`.

## Why more data narrows the band, down to a floor

- Every point adds β φ(x_n) φ(x_n)ᵀ to the precision S_N⁻¹. For evenly spaced x,
  ΦᵀΦ ≈ N × (a fixed matrix), so S_N ≈ (βN × fixed)⁻¹ and the parameter term falls roughly
  like 1/N: by 10.7× at x = 0.5 from N = 5 to N = 50.
- At the ends it falls only 3.5× over the same range. x = 0 and 1 are data points in every file,
  and at a data point the parameter term is always below the noise variance 1/β = 0.01; with
  N = 5 it is already just under that cap (0.0093 and 0.0096). The band stays widest at the ends
  for every N: a cubic is least pinned down where there is data on one side only.
- The noise term 1/β is the scatter of a new observation around the true curve. No amount of data
  about w reduces it, so the width tends to 2 × 1.96 × β^(−1/2) = 0.392, never below.
- **s²(x) does not depend on the t values at all**, only on where the inputs are. So the width
  plot (`p4_width`) is fixed by the x positions in the files.

| N | mean width | max width | parameter share of s² |
|---|---|---|---|
| 5 | 0.495 | 0.549 | 37.2% |
| 10 | 0.449 | 0.524 | 23.5% |
| 20 | 0.424 | 0.486 | 14.3% |
| 50 | 0.406 | 0.442 | 6.8% |

(`width`; the share is averaged over x.) The floor is 0.392.

## Demo

1. `uv run python -m src.run p4`; open `p4_predictive.pdf` (the dashed MAP curve sits on the mean)
   and `p4_width.pdf` (four curves stepping down to the floor).
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
- *sin(2πx) leaves the 95% band in places: is that wrong?* No. The band covers observation noise
  and parameter uncertainty within the cubic model, not the model's misfit: a cubic cannot follow
  sin(2πx) exactly. On the N = 10 plot the truth leaves the band around its peak and trough
  (x ≈ 0.23 to 0.40 and 0.63 to 0.74).
- *Effect of a larger α?* A tighter prior: smaller S_N (narrower band) but m_N pulled toward 0
  (more bias, the fit flattens). A larger β trusts each point more: narrower band, weaker
  effective prior (λ = α/β falls).
- *Does the band depend on the observed y?* No, only on the x positions (for fixed α, β).

## Likely code changes

- α, β or the degree: `ALPHA`, `BETA`, `D` in `src/p4.py` (everything else follows, including
  λ = α/β and the floor).
- A 99% band: change `Z` to 2.576.
- Plot the parameter term alone: `var - 1 / BETA`, with `var` from `predictive`.
