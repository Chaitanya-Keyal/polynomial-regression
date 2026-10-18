# Numerics: conditioning, solving instead of inverting, and the floating-point floor

## In short

The condition number κ of a matrix says how much it can amplify small relative errors, and, for
gradient descent, how many steps it takes. The degree-9 monomial basis on [0, 1] is terribly
conditioned (κ(Φ) ≈ 1.5 × 10⁷), so the assignment avoids every operation that makes that worse:
no normal equations for least squares, no explicit inverses, and GD only on a better-conditioned
problem (QR in P1, the ridge penalty in P2, the Legendre basis in P3). Floating point carries
about 16 significant digits, so a loss like ½‖Q α − t‖² cannot get below roughly 10⁻³¹: every
"converged" curve flattens there.

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
- the Bayesian posterior (P4): factor S_N⁻¹ = L Lᵀ (Cholesky, `np.linalg.cholesky`); then
  m_N = β S_N Φᵀ t is two triangular solves, and φᵀ S_N φ = φᵀ (L Lᵀ)⁻¹ φ = ‖L⁻¹ φ‖², one more
  solve. S_N itself is never formed.

## QR: moving the bad conditioning out of the optimisation

Φ = Q R with Q (N × 10) having orthonormal columns and R (10 × 10) upper triangular. Substituting
α = R w turns ½‖Φw − t‖² into ½‖Qα − t‖², whose Hessian is QᵀQ = I: κ = 1, so GD converges at
the same rate in every direction. Same problem, same minimiser, just new coordinates. The bad
conditioning has not gone away; it moved into R (κ(R) = κ(Φ)), and it shows up once, at the end,
when w = R⁻¹ α turns tiny rounding in α into the ~10⁻¹⁰ relative error of P1's verification
table.

## The floating-point floor

Doubles carry about 16 significant digits (machine epsilon ≈ 2.2 × 10⁻¹⁶). Computing the
residual Qα − t with |t| ≈ 1 leaves about 10⁻¹⁶ of rounding in each entry, so
E = ½‖r‖² bottoms out near 10 × (10⁻¹⁶)² / 2 ≈ 10⁻³¹. That is where the P1 loss curves flatten.
Near the floor, two runs that are mathematically identical (SGD and Batch in P1) differ in their
last digits, and the *relative* difference of their E values grows as E shrinks; the absolute
difference stays around 10⁻¹⁶ √E. That is why `tests/test_p1.py` checks |E_SGD − E_Batch| ≤
10⁻¹⁴ √E rather than a relative tolerance.
