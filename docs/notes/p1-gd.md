# P1: closed form vs. the gradient descent family

## In short

Fit polynomials of degree 1 to 9 by least squares, then for d = 9 reach the same answer with
Batch, Stochastic and Mini-batch GD. GD runs on the QR-reparametrised problem
E(α) = ½‖Qα − t‖², where Q is 10 × 10 and orthogonal. That one fact explains everything in the
plots: the Hessian is the identity, so Batch GD shrinks the error by exactly (1 − η) per epoch;
the rows of Q are orthonormal, so SGD does exactly what Batch does; Mini-batch averages over
batches of 3, so it needs about 2.6 times as many epochs; η must lie in (0, 2), with η = 1
converging in a single step.

## The model and the closed form

- Model y(x, w) = Σ_j w_j x^j, design matrix Φ (N × (d + 1)) with rows [1, x_n, ..., x_n^d].
- E(w) = ½‖Φw − t‖², gradient Φᵀ(Φw − t), minimum where ΦᵀΦ w = Φᵀt.
- `src/p1.py`, `compute`: `least_squares(monomial(x, d), t)` for d = 1..9, through `lstsq`
  (never the normal equations; see [numerics.md](numerics.md)).
- With N = 10 and d = 9 there are 10 weights for 10 points, so the fit interpolates the data
  (E = 6.6 × 10⁻²³). The weights are huge (‖w‖ = 1.09 × 10⁵) and cancel each other: a degree-9
  polynomial forced through noisy points swings hard between them.
- The training error can only fall as d grows: each model contains the previous one (set the new
  weight to 0). Here it falls in pairs, train RMSE 0.527 and 0.524 (d = 1, 2), 0.130 and 0.127
  (3, 4), 0.058 and 0.057 (5, 6): sin(2πx) is odd about x = 0.5 and the x are symmetric about it,
  so each even degree adds a direction that is even about x = 0.5, orthogonal on these points to
  the odd signal, and it can only fit noise.

## GD on the QR problem: the code

`gd(Q, t, eta, epochs, variant)` is one loop for all three variants, starting at α = 0 and
recording E after every epoch (`losses[0]` is the start):

```python
if variant == "batch":
    alpha = alpha - eta * Q.T @ (Q @ alpha - t)  # full sum, not / N
else:  # batches of 1 (SGD) or BATCH = 3 (Mini-batch)
    for idx in batches(rng.permutation(len(t)), 1 if variant == "sgd" else BATCH):
        Qb = Q[idx]
        alpha = alpha - eta * Qb.T @ (Qb @ alpha - t[idx]) / len(idx)  # averaged
```

- `batches(order, 3)` cuts a shuffled order into [3, 3, 3, 1]. With size 1 it gives ten singletons
  (SGD), and dividing by `len(idx) = 1` changes nothing, so SGD is exactly the handout's update.
- Epoch = one pass over the 10 points: 1 update (Batch), 10 (SGD), 4 (Mini-batch).
- `gd` creates its own `np.random.default_rng(SEED)` (SEED = 42), so every run gets the same
  fresh generator, whatever order the variants run in.
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
  (1 − η/|B|). Each epoch nine points sit in batches of 3 and shrink by (1 − η/3); the singleton
  shrinks by (1 − η). Each point is the singleton about one epoch in ten, so E falls by about
  ((1 − η/3)^0.9 (1 − η)^0.1)² ≈ 0.921 per epoch (measured 0.924, `per_epoch_factor`) instead
  of 0.81: 353 epochs to reach E < 10⁻¹² against 136, about 2.6 times as many. (Which point is
  the singleton changes with the shuffle, so Mini-batch is the only curve that depends slightly
  on the seed.)

On `p1_loss`, which shows all `EPOCHS` = 2000 epochs, SGD equals Batch epoch by epoch until both
reach the rounding floor (largest relative gap 6.8 × 10⁻¹¹ while E > 10⁻¹⁰,
`sgd_batch_rel_gap`). From about epoch 330 Batch is flat near 10⁻³⁰ and SGD near 10⁻²⁹: ten
rounded updates per epoch leave more rounding than one ([numerics.md](numerics.md)). Mini-batch
flattens near 10⁻²⁹ too, by about epoch 840.

This is special to N = d + 1. With N > d + 1, Q is tall, QQᵀ is a projection rather than I, SGD
steps interfere with each other, and since single-point gradients do not vanish at the least
squares solution, constant-step SGD would hover around the minimum instead of settling.

## Learning rates

The Hessian of E(α) is QᵀQ = I, so α_k − α* = (1 − η)^k (α_0 − α*) and, since Q is square and
so E(α*) = 0, E_k = (1 − η)^{2k} E_0, with E_0 = ½‖t‖² = 2.38 (the start is α = 0). The study (`LR_CASES`, `p1_lr`) runs Batch GD at
η = 0.01, 0.5 and 2.1 for `LR_EPOCHS` = 100 epochs; η = 1 and η = 2 follow from the formula and
are checked in `tests/test_p1.py`.

| η | factor (1 − η) | E after 100 epochs | behaviour |
|---|---|---|---|
| 0.01 | 0.99 | 0.319 | converges, but E only falls by 0.99² = 0.98 per epoch |
| 0.5 | 0.5 | 5.1 × 10⁻³² | E falls 4× per epoch, at the floor in about 50 epochs |
| 1 | 0 | not run | exact in one epoch: this is Newton's method, since the Hessian is I |
| 2 | −1 | not run | α alternates 0, 2α*, 0, 2α*, ...; E stays at E_0 forever |
| 2.1 | −1.1 | 4.5 × 10⁸ | the error flips sign and grows 1.1× per epoch: diverges |

At η = 2.1, α_100 = (1 − (−1.1)^100) α* = (1 − 1.1^100) α*, and because w = R⁻¹α the weights
after 100 epochs (`w_100_too_large`) are the same multiple of the closed form,
w_100 ≈ −13 780 w_closed (`w_100_over_w_closed` = −13 779.6), while
E_100 = 1.1^200 E_0 ≈ 1.9 × 10⁸ E_0.

## The verification numbers

After 2000 epochs at η = 0.1 all three variants give ‖w_GD − w_closed‖ ≈ 4.1 × 10⁻⁶, and with
‖w_closed‖ = 1.09 × 10⁵ a relative error of 3.7 × 10⁻¹¹ (`verification`). That is not
optimisation error: GD reaches the QR solution R⁻¹Qᵀt to within about 10⁻¹⁰. The 4.1 × 10⁻⁶ is
the gap between the two closed forms themselves, QR and lstsq's SVD, well within the
κ(Φ) ε ≈ 1.5 × 10⁷ × 2.2 × 10⁻¹⁶ ≈ 3 × 10⁻⁹ relative error that rounding allows
([numerics.md](numerics.md)). That is also why the three variants agree to about four digits.
These values are this machine's: rounding sets them, so another BLAS, or building Φ with `x**j`
instead of `np.vander` (6.2 × 10⁻⁶), changes even their first digit. That is why the report
prints them to one significant figure.

The final E in the report's verification table is E(α) in the QR coordinates, 10⁻³⁰ to 10⁻²⁹.
Recomputed as ½‖Φw − t‖² from the recovered w it is 2 to 4 × 10⁻²² for every variant, like the
closed form's 6.6 × 10⁻²³: with weights up to 7 × 10⁴, each entry of Φw is a sum of large terms
that cancel, so it carries rounding of about 10⁻¹¹.

## Demo

1. `uv run python -m src.run p1 --out /tmp/demo`, then open, in `/tmp/demo/figures/`,
   `p1_fits.pdf`, `p1_loss.pdf`, `p1_lr.pdf`.
2. Show `gd` and `batches` in `src/p1.py`, and the `/ len(idx)` that makes Mini-batch slower.

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
- *Why do the curves flatten at different heights?* Rounding: Batch near 10⁻³⁰, SGD and
  Mini-batch, with more rounded updates per epoch, near 10⁻²⁹.
- *Best η here?* η = 1: one step, Newton's method.
- *What happens at η = 2?* Pure oscillation between 0 and 2α*; E is constant.
- *Why is ‖w_GD − w_closed‖ not smaller?* GD matches the QR closed form to about 10⁻¹⁰; the
  4.1 × 10⁻⁶ is the QR vs. SVD gap, rounding amplified by κ(Φ), and tiny next to ‖w‖.
- *Why is GD's final E (10⁻³⁰ to 10⁻²⁹) so far below the closed form's (7 × 10⁻²³)?* Different
  coordinates: GD's E is ½‖Qα − t‖². Computed as ½‖Φw − t‖² from its w, GD's is 2 to 4 × 10⁻²²
  too; Φw sums terms up to 7 × 10⁴ that cancel, which costs rounding.
- *Why are the d = 9 weights so large?* Interpolating noise with nearly parallel monomial columns
  needs large coefficients that cancel.
- *Which seed?* 42, a fresh `default_rng(42)` per run.

## Likely code changes

- Different η, epoch count or batch size: `ETA`, `EPOCHS`, `BATCH` at the top of `src/p1.py`.
  `batches` handles any remainder (size 4 gives 4, 4, 2). Keep `EPOCHS` ≥ 100: the per-epoch
  factor is measured over the first 100 epochs. A variant that never gets below 10⁻¹² (Mini-batch
  at η = 0.01, say) reports `epochs_to_1e-12` as null, which the notebook and a rebuilt report
  print as None.
- The learning-rate study: `LR_CASES` (label to η) and `LR_EPOCHS`. To show η = 2 as well, add an
  entry such as `"oscillating": 2.0` and a fourth colour in `lr_figure` (it zips the cases with
  three colours, strictly).
- Divide the batch gradient by N (the batch line in `gd`): Batch then moves 10× slower per epoch
  than SGD.
- GD in the original basis: `gd(Phi, t, 0.09, EPOCHS, "batch")` returns w itself (no
  `solve(R, ...)`). η must stay below 2/λ_max ≈ 0.098, and E stalls (about 0.25 after 2000
  epochs) with w nowhere near w_closed.
- A different seed: `SEED`. Batch and SGD do not change beyond rounding; Mini-batch changes
  slightly.
