"""P2's ridge GD on the provided training set, and the 1.05x band by hand."""

import numpy as np

from src import p2
from src.common.bases import monomial
from src.common.linalg import ridge
from tests import T, X

PHI = monomial(X, p2.D)


def test_gd_stops_within_tolerance_of_the_closed_form():
    w_closed = ridge(PHI, T, p2.GD_LAM)
    objs, errs = p2.ridge_gd(PHI, T, p2.GD_LAM, p2.GD_ETA, w_closed)
    assert errs[-1] < p2.GD_TOL <= errs[-2]
    # eta is below 2 / lambda_max, so every step lowers the objective (up to its rounding)
    assert np.all(np.diff(objs) <= 1e-14)


def test_within_band_by_hand():
    best, lo, hi, others = p2.within_band(np.array([3, 2, 1.04, 1, 1.05, 2, 1.01]))
    assert (best, lo, hi, others) == (3, 2, 4, [6])
