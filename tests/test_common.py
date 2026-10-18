"""Shared code against answers worked out by hand and numpy's own implementations.

python -m pytest tests
"""

import numpy as np
import pytest

from src.common import bases, data, linalg, metrics


@pytest.mark.parametrize("d", [0, 1, 2, 9])
@pytest.mark.parametrize("n", [10, 37])
def test_legendre_matches_legvander(d, n):
    x = np.linspace(0, 1, n)
    expected = np.polynomial.legendre.legvander(2 * x - 1, d)
    np.testing.assert_allclose(bases.legendre(x, d), expected, rtol=0, atol=1e-13)


def test_legendre_by_hand():
    # u = 2x - 1 = -1, 0, 1; P_2(u) = (3u^2 - 1)/2 = 1, -1/2, 1
    np.testing.assert_allclose(
        bases.legendre([0, 0.5, 1], 2), [[1, -1, 1], [1, 0, -0.5], [1, 1, 1]]
    )


def test_monomial_by_hand():
    np.testing.assert_array_equal(bases.monomial([0, 2, 3], 2), [[1, 0, 0], [1, 2, 4], [1, 3, 9]])


def test_least_squares_exact_line():
    # t = 1 + 2x passes through every point, so the fit recovers it exactly
    Phi = bases.monomial([0, 1, 2], 1)
    np.testing.assert_allclose(linalg.least_squares(Phi, np.array([1.0, 3, 5])), [1, 2])


def test_least_squares_overdetermined():
    # the best constant fit is the mean
    Phi = np.ones((4, 1))
    np.testing.assert_allclose(linalg.least_squares(Phi, np.array([1.0, 2, 3, 6])), [3])


def test_ridge_by_hand():
    # Phi = I: (1 + lam) w = t
    np.testing.assert_allclose(linalg.ridge(np.eye(2), np.array([2.0, 4]), 1.0), [1, 2])


def test_ridge_matches_stacked_least_squares():
    # (Phi^T Phi + lam I) w = Phi^T t are the normal equations of [Phi; sqrt(lam) I] w = [t; 0]
    Phi = bases.monomial(np.linspace(0, 1, 6), 3)
    t = np.array([0.0, 1, 0, -1, 0, 1])
    lam = 0.3
    stacked = np.vstack([Phi, np.sqrt(lam) * np.eye(4)])
    expected = linalg.least_squares(stacked, np.concatenate([t, np.zeros(4)]))
    np.testing.assert_allclose(linalg.ridge(Phi, t, lam), expected, atol=1e-12)


def test_objectives_by_hand():
    Phi, t, w = np.eye(2), np.array([1.0, 1]), np.array([2.0, -1])
    # residual (1, -2): SSE = 5/2; ||w||^2 = 5; ||w||_1 = 3
    assert metrics.sse(Phi, w, t) == pytest.approx(2.5)
    assert metrics.ridge_objective(Phi, w, t, 0.4) == pytest.approx(2.5 + 0.2 * 5)
    assert metrics.lasso_objective(Phi, w, t, 0.4) == pytest.approx(2.5 + 0.4 * 3)


def test_rmse_by_hand():
    assert metrics.rmse([1, 2, 3, 4], [1, 2, 3, 0]) == pytest.approx(2.0)


def test_load_xy(tmp_path):
    path = tmp_path / "f.csv"
    path.write_text("x,y\n0.0,1.5\n0.5,-2.0\n")
    x, t = data.load_xy(path)
    np.testing.assert_array_equal(x, [0, 0.5])
    np.testing.assert_array_equal(t, [1.5, -2])


def test_load_xy_rejects_other_columns(tmp_path):
    path = tmp_path / "f.csv"
    path.write_text("y,x\n1,0\n")
    with pytest.raises(ValueError, match="x,y"):
        data.load_xy(path)
