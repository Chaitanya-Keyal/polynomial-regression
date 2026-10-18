"""P3's subgradient descent against an exact Lasso solver, on the assignment's design and an
arbitrary fixed target."""

import numpy as np
import pytest

from src import p3
from src.common.bases import legendre
from src.common.metrics import lasso_objective

X = np.linspace(0, 1, 10)
T = np.array([0.3, -1.2, 0.8, 0.1, -0.5, 1.1, -0.9, 0.4, 0.0, -0.2])
PHI = legendre(X, p3.D)


def kkt(w, lam):
    """Optimality residuals of an exact solution: on the support and off it."""
    c = PHI.T @ (T - PHI @ w)
    on = w != 0
    return np.abs(c - lam * np.sign(w))[on], np.abs(c[~on]) / lam


@pytest.mark.parametrize("lam", [1e-3, 0.1, 1.0])
def test_coordinate_descent_is_exact(lam):
    w = p3.lasso_cd(PHI, T, lam)
    support_violation, off_ratio = kkt(w, lam)
    assert support_violation.max(initial=0) < 1e-9
    assert off_ratio.max(initial=0) <= 1 + 1e-9


def test_large_lambda_zeroes_everything():
    # w = 0 is optimal once lam >= max_j |Phi_j^T t|
    lam = np.abs(PHI.T @ T).max() * 1.01
    np.testing.assert_array_equal(p3.lasso_cd(PHI, T, lam), 0)
    w, _ = p3.subgradient_descent(PHI, T, lam, iters=2000)
    assert p3.near_zero(w, lam).all()


@pytest.mark.parametrize("lam", [0.01, 0.1])
def test_subgradient_lands_near_the_exact_solution(lam):
    w_cd = p3.lasso_cd(PHI, T, lam)
    w, objs = p3.subgradient_descent(PHI, T, lam)
    # constant steps leave chatter of order eta * lam around the optimum
    assert np.abs(w - w_cd).max() < 5 * p3.ETA * lam
    assert objs.min() - lasso_objective(PHI, w_cd, T, lam) < 1e-3
    # the ≈ 0 rule flags the same coefficients on both (at lam = 0.1 the exact solution has a
    # zero and a nonzero -0.0009, both under the 2 eta lam = 0.01 cutoff)
    np.testing.assert_array_equal(p3.near_zero(w, lam), p3.near_zero(w_cd, lam))


def test_optimality_report():
    lam = 0.1
    w, _ = p3.subgradient_descent(PHI, T, lam)
    report = p3.optimality(PHI, T, w, lam)
    assert report["n_active"] + report["n_zero"] == p3.D + 1
    assert report["max_zero_ratio"] <= 1
    assert report["max_active_violation"] < 0.5 * lam


def test_eta_is_stable():
    assert np.linalg.eigvalsh(PHI.T @ PHI)[-1] * p3.ETA < 2


def test_near_zero_rule():
    # 2 eta lam = 0.01 at lam = 0.1
    np.testing.assert_array_equal(
        p3.near_zero(np.array([0.0, 0.009, -0.009, 0.011]), 0.1), [True, True, True, False]
    )
