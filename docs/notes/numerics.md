# Numerics: conditioning, solving instead of inverting, and the floating-point floor

## In short

The condition number κ of a matrix says how much it can amplify small relative errors, and, for
gradient descent, how many steps it takes. The degree-9 monomial basis on [0, 1] is terribly
conditioned (κ(Φ) ≈ 1.5 × 10⁷), so the assignment avoids every operation that makes that worse:
no normal equations for least squares, no explicit inverses, and GD only on a better-conditioned
problem (QR in P1, the ridge penalty in P2, the Legendre basis in P3). Floating point carries
about 16 significant digits, so a loss like ½‖Q α − t‖² cannot reach 0: P1's Batch curve flattens
near 10⁻³⁰, and SGD and Mini-batch, which make more rounded updates per epoch, near 10⁻²⁹.

## Condition number

- κ(A) = σ_max / σ_min, the ratio of the largest to the smallest singular value
  (`np.linalg.cond`). For a symmetric positive definite matrix it is λ_max / λ_min.
- **Accuracy.** Solving A x = b in floating point can lose up to log₁₀ κ digits: with κ = 10⁷ and
  16 digits to start with, about 9 are left.
- **GD speed.** On a quadratic with Hessian H, GD is stable only for η < 2 / λ_max(H), and the
  slowest direction shrinks by (1 − η λ_min) per step. With the largest safe η that is about
  (1 − 2/κ) per step, so the step count grows like κ · ln(1/ε).

## Why the monomial basis is so bad

The columns 1, x, x², ..., x⁹ evaluated at ten points in [0, 1] are nearly parallel: x⁸ and x⁹
look almost the same on [0, 1]. So ΦᵀΦ has eigenvalues from about 20 down to about 10⁻¹³:

| d = 9, x = 10 evenly spaced points | value |
|---|---|
| κ(Φ) | ≈ 1.5 × 10⁷ |
| κ(ΦᵀΦ) = κ(Φ)² | ≈ 2.3 × 10¹⁴ |
| λ_max(ΦᵀΦ), so the GD step limit 2/λ_max | ≈ 20.4, η < 0.098 |
| κ(ΦᵀΦ + 10⁻² I) (P2) | ≈ 2.0 × 10³ |
| κ(ΦᵀΦ) in the Legendre basis (P3) | ≈ 6.1 × 10², η < 0.161 |

Plain GD on the monomial least-squares problem would shrink its slowest direction by about
1 − 10⁻¹⁴ per step: some 10¹⁴ steps per digit of accuracy. That is the "astronomically slowly"
in the handout.

## Least squares without the normal equations

Setting the gradient of ½‖Φw − t‖² to zero gives ΦᵀΦ w = Φᵀt, but forming ΦᵀΦ squares κ, from
10⁷ to 10¹⁴, which leaves about 2 correct digits. `np.linalg.lstsq` instead works through the SVD
Φ = U Σ Vᵀ and computes w = V Σ⁻¹ Uᵀ t, with errors governed by κ(Φ), not κ(Φ)²
(`src/common/linalg.py`, `least_squares`).

## Solve, never invert

`np.linalg.inv(A) @ b` is slower and less accurate than `np.linalg.solve(A, b)`: the inverse
needs three times the work of one LU factorisation, and multiplying by it adds its own rounding.
Every formula with an inverse in the handout is computed with solves:

- ridge (P2, P3, P4): `solve(ΦᵀΦ + λI, Φᵀt)` (`linalg.ridge`). For λ > 0 the matrix is symmetric
  positive definite, so the solve always succeeds.
- the QR back-map (P1): w = R⁻¹ α as `solve(R, α)`.
- the Bayesian posterior (P4): with the precision A = S_N⁻¹ = αI + βΦᵀΦ, m_N = β S_N Φᵀ t is
  `solve(A, β Φᵀ t)` and φᵀ S_N φ is φᵀ `solve(A, φ)` (`p4.posterior`, `p4.predictive`). S_N
  itself is never formed.

## QR: moving the bad conditioning out of the optimisation

Φ = Q R with Q (N × 10) having orthonormal columns and R (10 × 10) upper triangular. Substituting
α = R w turns ½‖Φw − t‖² into ½‖Qα − t‖², whose Hessian is QᵀQ = I: κ = 1, so GD converges at
the same rate in every direction. Same problem, same minimiser, just new coordinates. The bad
conditioning has not gone away; it moved into R (κ(R) = κ(Φ)), and it shows up once, at the end,
in w = R⁻¹ α. GD reaches the QR solution R⁻¹Qᵀt to within about 10⁻¹⁰, but that solution and
lstsq's SVD one differ by 4.1 × 10⁻⁶ (relative 3.7 × 10⁻¹¹): all of P1's verification error is
this gap between two closed forms, well within the κ(Φ) ε ≈ 3 × 10⁻⁹ relative error that
rounding allows (ε ≈ 2.2 × 10⁻¹⁶, below). Being rounding, the gap's digits are this machine's:
another BLAS changes them, which is why the report prints such values to one significant figure.

## The floating-point floor

Doubles carry about 16 significant digits (machine epsilon ≈ 2.2 × 10⁻¹⁶; one rounding is off by
at most u ≈ 1.1 × 10⁻¹⁶, the unit roundoff). Each update of α leaves a few u of rounding in it,
so the residual Qα − t (|t| ≈ 1) keeps a few × 10⁻¹⁶ in each of its ten entries, and E = ½‖r‖²
stops falling there. In P1, Batch (one update per epoch) flattens near 10⁻³⁰ (7.0 × 10⁻³¹ after
2000 epochs on this machine); SGD and Mini-batch (ten and four updates per epoch) carry more
rounding and flatten near 10⁻²⁹ (1.4 and 1.6 × 10⁻²⁹).

Two runs that are mathematically identical (SGD and Batch in P1) agree only up to rounding: their
E values differ by around 10⁻¹⁶ √E, so the *relative* difference grows as E shrinks, and at the
floor they differ by the gap between the two floors. That is why `tests/test_p1.py` (rtol 10⁻⁸)
and `sgd_batch_rel_gap` in `results/p1.json` (6.8 × 10⁻¹¹) compare the two only while
E > 10⁻¹⁰.
