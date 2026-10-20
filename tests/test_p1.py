"""P1's GD family on the assignment's design (N = 10 evenly spaced x, d = 9).

Q is then square with orthonormal rows, which pins down every behaviour below exactly. The
targets are an arbitrary fixed vector: none of this depends on the data.
"""

import numpy as np
import pytest

from src import p1
from src.common.bases import monomial
from src.common.linalg import least_squares
from tests import T, X

PHI = monomial(X, p1.D)
Q, R = np.linalg.qr(PHI)
ALPHA_STAR = Q.T @ T  # the minimiser, since Q^T Q = I


@pytest.mark.parametrize("variant", p1.NAMES)
def test_gd_recovers_the_closed_form(variant):
    alpha, losses = p1.gd(Q, T, p1.ETA, p1.EPOCHS, variant)
    w_closed = least_squares(PHI, T)
    assert np.linalg.norm(np.linalg.solve(R, alpha) - w_closed) < 1e-6 * np.linalg.norm(w_closed)
    assert losses[-1] < 1e-20


def test_sgd_matches_batch_every_epoch():
    # orthonormal rows: a step on point n changes only point n's residual, by (1 - eta)
    _, batch = p1.gd(Q, T, p1.ETA, 300, "batch")
    _, sgd = p1.gd(Q, T, p1.ETA, 300, "sgd")
    np.testing.assert_allclose(batch[1:101] / batch[:100], (1 - p1.ETA) ** 2, rtol=1e-9)
    above_floor = batch > 1e-10
    np.testing.assert_allclose(sgd[above_floor], batch[above_floor], rtol=1e-8)


def test_minibatch_averages_over_3_3_3_1():
    # each point's residual shrinks by (1 - eta / |B|) for the batch it falls in that epoch
    rng = np.random.default_rng(p1.SEED)
    scale = np.ones(10)
    for _ in range(2):
        for b in p1.batches(rng.permutation(10), p1.BATCH):
            scale[b] *= 1 - p1.ETA / len(b)
    _, losses = p1.gd(Q, T, p1.ETA, 2, "minibatch")
    assert [len(b) for b in p1.batches(np.arange(10), p1.BATCH)] == [3, 3, 3, 1]
    assert losses[2] == pytest.approx(0.5 * np.sum((scale * T) ** 2))  # residuals start at -T


def test_learning_rate_regimes():
    # alpha_k - alpha* = (1 - eta)^k (alpha_0 - alpha*) with alpha_0 = 0
    _, losses = p1.gd(Q, T, 1.0, 1, "batch")
    assert losses[1] < 1e-25  # eta = 1: one step
    alpha, losses = p1.gd(Q, T, 2.0, 1, "batch")
    np.testing.assert_allclose(alpha, 2 * ALPHA_STAR, atol=1e-12)  # eta = 2: 0, 2 alpha*, 0, ...
    np.testing.assert_allclose(losses, 0.5 * T @ T, rtol=1e-12)
    alpha, losses = p1.gd(Q, T, 2.1, 100, "batch")
    np.testing.assert_allclose(alpha, (1 - (-1.1) ** 100) * ALPHA_STAR, rtol=1e-9)
    np.testing.assert_allclose(losses[100] / losses[0], 1.1**200, rtol=1e-9)
