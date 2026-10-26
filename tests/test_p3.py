"""P3's subgradient descent against an exact Lasso solver, on the provided training set."""

import numpy as np
import pytest

from src import p3
from src.common.bases import legendre
from src.common.metrics import lasso_objective
from tests import T, X

PHI = legendre(X, p3.D)


def lasso_cd(lam, sweeps=2000):
    """The exact Lasso by cyclic coordinate descent with soft-thresholding."""
    w, col_sq = np.zeros(PHI.shape[1]), (PHI**2).sum(axis=0)
    for _ in range(sweeps):
        for j in range(len(w)):
            rho = PHI[:, j] @ (T - PHI @ w) + col_sq[j] * w[j]
            w[j] = np.sign(rho) * max(abs(rho) - lam, 0) / col_sq[j]
    return w


@pytest.mark.parametrize("lam", [0.01, 0.1])
def test_subgradient_lands_within_the_chatter_of_the_exact_solution(lam):
    w_exact = lasso_cd(lam)
    w, _ = p3.subgradient_descent(PHI, T, lam)
    assert np.abs(w - w_exact).max() < 5 * p3.ETA * lam
    np.testing.assert_array_equal(p3.near_zero(w, lam), p3.near_zero(w_exact, lam))


def test_exact_solution_is_the_handout_reference():
    # w* and E* in the handout are this solution rounded to three and four decimals
    w = lasso_cd(p3.CHECK_LAM)
    np.testing.assert_array_equal(np.round(w, 3), p3.W_REF)
    assert lasso_objective(PHI, w, T, p3.CHECK_LAM) == pytest.approx(p3.E_REF, abs=5e-5)


def test_optimality_conditions_hold_exactly_at_the_exact_solution():
    w = lasso_cd(p3.CHECK_LAM)
    report = p3.optimality(PHI, T, w, p3.CHECK_LAM)
    assert report["max_active_violation"] < 1e-9
    assert report["max_zero_ratio"] <= 1
