# Study notes

The ideas behind each problem, how the code implements them, and what to show and say in the
demo and viva. Not part of the submission. Every member may be asked about any problem, so read
them in this order:

- [assignment.md](assignment.md): what is asked, the rules that apply everywhere, and where each
  deliverable lives in the repo
- [numerics.md](numerics.md): conditioning, why no matrix is ever inverted, QR, and the
  floating-point floor; P1 to P4 all lean on it
- [p1-gd.md](p1-gd.md): least squares, Batch / Stochastic / Mini-batch GD on the QR problem,
  learning rates, and why SGD matches Batch exactly here
- [p2-ridge.md](p2-ridge.md): ridge as the MAP estimate, GD on it, choosing λ
- [p3-lasso.md](p3-lasso.md): the Lasso, subgradients, why L1 gives zeros and L2 does not
- [p4-bayes.md](p4-bayes.md): the Bayesian posterior and predictive distribution, and why more
  data narrows the band only down to the noise floor

Every problem file opens with an In short, then follows the same order: the maths, the code, the
demo, likely viva questions with answers, and the code changes most likely to be asked for (with
where to make them).
