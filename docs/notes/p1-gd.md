# P1: closed form vs. the gradient descent family

## In short

Fit polynomials of degree 1 to 9 by least squares, then for d = 9 reach the same answer with
Batch, Stochastic and Mini-batch GD. GD runs on the QR-reparametrised problem
E(α) = ½‖Qα − t‖², where Q is 10 × 10 and orthogonal. That one fact explains everything in the
plots: the Hessian is the identity, so Batch GD shrinks the error by exactly (1 − η) per epoch;
the rows of Q are orthonormal, so SGD does exactly what Batch does; Mini-batch averages, so it
moves a third as fast; η must lie in (0, 2), with η = 1 converging in a single step.

## The model and the closed form

- Model y(x, w) = Σ_j w_j x^j, design matrix Φ (N × (d + 1)) with rows [1, x_n, ..., x_n^d].
- E(w) = ½‖Φw − t‖², gradient Φᵀ(Φw − t), minimum where ΦᵀΦ w = Φᵀt.
- `src/p1.py`, `compute`: `least_squares(monomial(x, d), t)` for d = 1..9, through `lstsq`
  (never the normal equations; see [numerics.md](numerics.md)).
- With N = 10 and d = 9 there are 10 weights for 10 points, so the fit interpolates the data
  exactly (E ≈ 0). The weights are huge (‖w‖ ~ 10⁵) and cancel each other: a degree-9 polynomial
  forced through noisy points swings hard between them.
- The training error can only fall as d grows: each model contains the previous one (set the new
  weight to 0).

## GD on the QR problem: the code

`gd(Q, t, eta, epochs, variant, rng)` is one loop for all three variants, starting at α = 0 and
recording E after every epoch (`losses[0]` is the start):

```python
if variant == "batch":
    alpha = alpha - eta * Q.T @ (Q @ alpha - t)            # full sum, not / N
else:
    for idx in batches(rng.permutation(len(t)), size):     # size 1 (SGD) or 3 (Mini-batch)
        Qb = Q[idx]
        alpha = alpha - eta * Qb.T @ (Qb @ alpha - t[idx]) / len(idx)   # averaged
```

- `batches(order, 3)` cuts a shuffled order into [3, 3, 3, 1]. With size 1 it gives ten singletons
  (SGD), and dividing by `len(idx) = 1` changes nothing, so SGD is exactly the handout's update.
- Epoch = one pass over the 10 points: 1 update (Batch), 10 (SGD), 4 (Mini-batch).
- `run_variant` gives each run its own `np.random.default_rng(42)`, so the order the variants run
  in does not change their shuffles.
- After GD, `w = np.linalg.solve(R, alpha)` maps back to the polynomial weights.

## Why SGD equals Batch, and Mini-batch is slower

Write r = Qα − t (one residual per point) and q_nᵀ for row n of Q. With N = d + 1 = 10 the thin
Q is square, and a square matrix with orthonormal columns also has orthonormal rows:
QQᵀ = I, so q_mᵀ q_n = 0 for m ≠ n and 1 for m = n.

- **Batch**: α ← α − η Qᵀ r, so r ← r − η QQᵀ r = (1 − η) r. Every residual shrinks by (1 − η),
  E by (1 − η)² = 0.81 at η = 0.1.
- **SGD**, step on point n: α ← α − η r_n q_n, so r_m ← r_m − η r_n q_mᵀ q_n. That is
  r_n ← (1 − η) r_n and **no other residual changes**. After a pass over all ten points, in any
  order, each residual has been multiplied by (1 − η) exactly once: the same as one Batch step.
  So the curves coincide epoch for epoch, and the shuffle (and seed) cannot matter.
- **Mini-batch**, batch B: α ← α − (η/|B|) Σ_{n∈B} r_n q_n, so each r_n in B shrinks by
  (1 − η/|B|). Nine points sit in batches of 3 and shrink by (1 − η/3) per epoch; the singleton
  shrinks by (1 − η). E falls by about (1 − η/3)² ≈ 0.934 per epoch instead of 0.81: roughly three
  times as many epochs. (Which point is the singleton changes with the shuffle, so Mini-batch is
  the only curve that depends slightly on the seed.)

This is special to N = d + 1. With N > d + 1, Q is tall, QQᵀ is a projection rather than I, SGD
steps interfere with each other, and since single-point gradients do not vanish at the least
squares solution, constant-step SGD would hover around the minimum instead of settling.

## Learning rates

The Hessian of E(α) is QᵀQ = I, so α_k − α* = (1 − η)^k (α_0 − α*) and E_k = (1 − η)^{2k} E_0,
with E_0 = ½‖t‖² (the start is α = 0).

| η | factor (1 − η) | behaviour |
|---|---|---|
| 0.01 | 0.99 | converges, but E only falls by 0.99² = 0.98 per epoch |
| 0.5 | 0.5 | E falls by 4 per epoch, at the floor in about 50 epochs |
| 1 | 0 | exact in one epoch: this is Newton's method, since the Hessian is I |
| 2 | −1 | α alternates 0, 2α*, 0, 2α*, ...; E stays at E_0 forever |
| 2.1 | −1.1 | the error flips sign and grows 1.1× per epoch: diverges |

At η = 2.1, after 100 epochs α_100 = (1 − (−1.1)^100) α* = (1 − 1.1^100) α* ≈ −13780 α*, and
because w = R⁻¹α, w_100 ≈ −13780 w_closed exactly (`divergent_w_over_w_closed`), while
E_100 = 1.1^200 E_0 ≈ 1.9 × 10⁸ E_0.

## The verification numbers

After 2000 epochs at η = 0.1, ‖w_GD − w_closed‖ is around 10⁻⁵ but ‖w_closed‖ is around 10⁵, so
the relative error is about 10⁻¹⁰. All three variants show nearly the same error: it is not
optimisation error (E is at the floor) but rounding magnified by κ(R) = κ(Φ) ≈ 10⁷ when mapping
α back to w, plus lstsq's own rounding.

## Demo

1. `uv run python -m src.run p1`, then open `figures/p1_fits.png`, `p1_loss.png`, `p1_lr.png`.
2. In the notebook, the "Try it" cell: set `eta = 1.0` (E hits the floor after one epoch), then
   `2.0` (E never moves), then `2.1` (E grows).
3. Show `gd` and `batches` in `src/p1.py`, and the `/ len(idx)` that makes Mini-batch slower.

## Likely viva questions

- *Why not solve the normal equations?* Forming ΦᵀΦ squares κ(Φ) ≈ 1.5 × 10⁷ to 2.3 × 10¹⁴; lstsq
  uses the SVD of Φ directly.
- *Why QR?* The Hessian becomes I (κ = 1), so GD converges at one rate in every direction. GD
  on the original weights would need about 10¹⁴ steps per digit.
- *Why is the batch gradient not divided by N?* The handout's convention (λ and E are on the
  unnormalised SSE). It also makes one Batch step equal one pass of SGD here.
- *What is an epoch?* One pass over all N points: 1, 10 and 4 updates.
- *Why does the SGD curve sit exactly on the Batch curve?* Orthonormal rows (above).
- *Would that still hold with more points than weights?* No: rows of a tall Q are not orthogonal.
- *Best η here?* η = 1: one step, Newton's method.
- *What happens at η = 2?* Pure oscillation between 0 and 2α*; E is constant.
- *Why are the d = 9 weights so large?* Interpolating noise with nearly parallel monomial columns
  needs large coefficients that cancel.
- *Which seed?* 42, a fresh `default_rng(42)` per run.

## Likely code changes

- Different η, epoch count or batch size: `ETA`, `EPOCHS`, `BATCH` at the top of `src/p1.py`.
  `batches` handles any remainder (size 4 gives 4, 4, 2).
- Divide the batch gradient by N: Batch then moves 10× slower per epoch than SGD.
- GD in the original basis: call `gd(Phi, t, 0.09, ...)` instead of `gd(Q, ...)`; η must stay
  below 2/λ_max ≈ 0.098, and the loss barely moves.
- A different seed: `SEED`. Batch and SGD do not change at all; Mini-batch changes slightly.
