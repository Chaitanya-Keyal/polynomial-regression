"""P1's GD family on the assignment's design (N = 10 evenly spaced x, d = 9).

Q is then square with orthonormal rows, which pins down every behaviour below exactly. The
targets are an arbitrary fixed vector: none of this depends on the data.
"""

import numpy as np
import pytest

from src import p1
from src.common.bases import monomial
from src.common.linalg import least_squares

X = np.linspace(0, 1, 10)
T = np.array([0.3, -1.2, 0.8, 0.1, -0.5, 1.1, -0.9, 0.4, 0.0, -0.2])
PHI = monomial(X, p1.D)
Q, R = np.linalg.qr(PHI)
ALPHA_STAR = Q.T @ T  # the minimiser, since Q^T Q = I


def test_batches_are_3_3_3_1():
    sizes = [len(b) for b in p1.batches(np.arange(10), p1.BATCH)]
    assert sizes == [3, 3, 3, 1]


def test_q_is_orthogonal():
    np.testing.assert_allclose(Q.T @ Q, np.eye(10), atol=1e-12)
    np.testing.assert_allclose(Q @ Q.T, np.eye(10), atol=1e-12)


@pytest.mark.parametrize("variant", p1.VARIANTS)
def test_gd_recovers_the_closed_form(variant):
    alpha, losses = p1.run_variant(Q, T, p1.ETA, p1.EPOCHS, variant)
    w = np.linalg.solve(R, alpha)
    w_closed = least_squares(PHI, T)
    assert np.linalg.norm(w - w_closed) / np.linalg.norm(w_closed) < 1e-6
    assert losses[-1] < 1e-20


def test_sgd_matches_batch_every_epoch():
    # orthonormal rows: a step on point n changes only point n's residual, by (1 - eta)
    # (up to rounding: each residual carries ~1e-16 of noise, so |dE| ~ 1e-16 sqrt(E))
    _, batch = p1.gd(Q, T, p1.ETA, p1.EPOCHS, "batch")
    _, sgd = p1.run_variant(Q, T, p1.ETA, p1.EPOCHS, "sgd")
    assert np.all(np.abs(sgd - batch) <= 1e-14 * np.sqrt(batch) + 1e-30)


def test_batch_shrinks_by_one_minus_eta_squared():
    _, losses = p1.gd(Q, T, p1.ETA, 50, "batch")
    np.testing.assert_allclose(losses[1:] / losses[:-1], (1 - p1.ETA) ** 2, rtol=1e-9)


def test_minibatch_is_slower():
    # averaging divides the step by 3 for every point in a full batch
    _, batch = p1.gd(Q, T, p1.ETA, 100, "batch")
    _, mini = p1.run_variant(Q, T, p1.ETA, 100, "minibatch")
    assert mini[100] > 1e3 * batch[100]
    assert (mini[100] / mini[0]) ** (1 / 100) <= (1 - p1.ETA / p1.BATCH) ** 2 + 1e-12


def test_eta_one_converges_in_one_epoch():
    _, losses = p1.gd(Q, T, 1.0, 1, "batch")
    assert losses[1] < 1e-25


def test_eta_two_oscillates():
    # alpha_k - alpha* = (-1)^k (0 - alpha*): alpha alternates 0, 2 alpha*, and E never moves
    alpha_1, _ = p1.gd(Q, T, 2.0, 1, "batch")
    alpha_2, losses = p1.gd(Q, T, 2.0, 2, "batch")
    np.testing.assert_allclose(alpha_1, 2 * ALPHA_STAR, atol=1e-12)
    np.testing.assert_allclose(alpha_2, 0, atol=1e-12)
    np.testing.assert_allclose(losses, 0.5 * T @ T, rtol=1e-12)


def test_eta_above_two_diverges_geometrically():
    alpha, losses = p1.gd(Q, T, 2.1, 100, "batch")
    np.testing.assert_allclose(alpha, (1 - (-1.1) ** 100) * ALPHA_STAR, rtol=1e-9)
    np.testing.assert_allclose(losses[100] / losses[0], 1.1**200, rtol=1e-9)
