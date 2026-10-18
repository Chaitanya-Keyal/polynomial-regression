"""P2's ridge GD and λ selection, on the assignment's design and an arbitrary fixed target."""

import numpy as np

from src import p2
from src.common.bases import monomial
from src.common.linalg import ridge

X = np.linspace(0, 1, 10)
T = np.array([0.3, -1.2, 0.8, 0.1, -0.5, 1.1, -0.9, 0.4, 0.0, -0.2])
PHI = monomial(X, p2.D)


def test_gd_reaches_the_closed_form():
    w_closed = ridge(PHI, T, p2.GD_LAM)
    w, objs, errs = p2.ridge_gd(PHI, T, p2.GD_LAM, p2.GD_ETA, w_closed)
    assert errs[-1] < p2.GD_TOL <= errs[-2]
    assert np.linalg.norm(w - w_closed) / np.linalg.norm(w_closed) < p2.GD_TOL
    # eta is below 2 / lambda_max, so every step lowers the objective (up to its rounding)
    assert np.all(np.diff(objs) <= 1e-14)


def test_penalty_improves_conditioning():
    PtP = PHI.T @ PHI
    before = np.linalg.cond(PtP)
    after = np.linalg.cond(PtP + p2.GD_LAM * np.eye(p2.D + 1))
    assert before > 1e13
    assert after < 1e4
    # lambda lifts every eigenvalue by lam, so the smallest is at least lam
    assert np.linalg.eigvalsh(PtP + p2.GD_LAM * np.eye(p2.D + 1))[0] >= p2.GD_LAM * (1 - 1e-9)


def test_eta_is_stable():
    lam_max = np.linalg.eigvalsh(PHI.T @ PHI + p2.GD_LAM * np.eye(p2.D + 1))[-1]
    assert lam_max * p2.GD_ETA < 2


def test_flat_interval_by_hand():
    values = np.array([3, 2, 1.04, 1, 1.05, 2, 1.01])
    best, lo, hi, contiguous = p2.flat_interval(values)
    assert (best, lo, hi) == (3, 2, 4)
    # index 6 also qualifies but is cut off from the run around the minimum
    assert not contiguous


def test_flat_interval_at_the_edge():
    best, lo, hi, contiguous = p2.flat_interval(np.array([1.0, 1.02, 2]))
    assert (best, lo, hi, contiguous) == (0, 0, 1, True)


def test_large_lambda_shrinks_weights():
    norms = [np.linalg.norm(ridge(PHI, T, lam)) for lam in p2.LAMS]
    assert norms[-1] < 0.1 * norms[0]
