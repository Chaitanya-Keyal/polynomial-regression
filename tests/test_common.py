"""The shared solvers against answers worked out by hand, and the provided files as loaded.

python -m pytest
"""

import numpy as np
import pytest

from src.common import bases, data, linalg, metrics


def test_provided_files():
    x, t = data.load(data.TRAIN)
    np.testing.assert_allclose(x, np.linspace(0, 1, 10), atol=1e-10)  # evenly spaced, N = 10
    assert t.shape == (10,)
    assert data.load(data.TEST)[0].shape == (100,)
    for n in data.P4_SIZES:
        np.testing.assert_allclose(data.load(f"p4_train_N{n}.csv")[0], np.linspace(0, 1, n))


def test_legendre_by_hand():
    # u = 2x - 1 = -1, 0, 1; P_2(u) = (3u^2 - 1)/2 = 1, -1/2, 1
    np.testing.assert_allclose(
        bases.legendre(np.array([0, 0.5, 1]), 2), [[1, -1, 1], [1, 0, -0.5], [1, 1, 1]]
    )


def test_least_squares_exact_line():
    # t = 1 + 2x passes through every point, so the fit recovers it exactly
    Phi = bases.monomial(np.array([0.0, 1, 2]), 1)
    np.testing.assert_allclose(linalg.least_squares(Phi, np.array([1.0, 3, 5])), [1, 2])


def test_ridge_by_hand():
    # Phi = I: (1 + lam) w = t
    np.testing.assert_allclose(linalg.ridge(np.eye(2), np.array([2.0, 4]), 1.0), [1, 2])


def test_objectives_by_hand():
    Phi, t, w = np.eye(2), np.array([1.0, 1]), np.array([2.0, -1])
    # residual (1, -2): SSE = 5/2; ||w||^2 = 5; ||w||_1 = 3
    assert metrics.sse(Phi, w, t) == pytest.approx(2.5)
    assert metrics.ridge_objective(Phi, w, t, 0.4) == pytest.approx(2.5 + 0.2 * 5)
    assert metrics.lasso_objective(Phi, w, t, 0.4) == pytest.approx(2.5 + 0.4 * 3)
    assert metrics.rmse([1, 2, 3, 4], [1, 2, 3, 0]) == pytest.approx(2.0)
